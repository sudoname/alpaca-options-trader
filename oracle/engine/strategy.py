"""
Oracle Engine — Upgrade A: Strategy protocol + OracleStrategyAdapter.

This is the "adapter over existing" heart of the engine. The ``Strategy``
protocol is the surface the bus dispatches into. ``OracleStrategyAdapter``
implements it by calling the EXISTING pure decision functions
(``oracle_intelligence_reports.compute_oracle_explain`` -> agents -> voting ->
probability) and emitting a ``TRADE_INTENT`` event. It NEVER touches a broker,
a clock's wall time, or the network — a broker adapter (``sim_broker`` in the
backtest, the live client in production) consumes the intent.

Because the decision math is the existing production code, a backtest and live
share one strategy body; they differ only by feed + execution adapter + clock.

Dependency injection (so the adapter is offline-testable and reuses production
logic without importing the heavy ``smart_trader`` method):

  * ``explain_fn``     : (ticker, ctx, prior) -> compute_oracle_explain dict.
                         Defaults to the real ``compute_oracle_explain``.
  * ``ctx_builder``    : (symbol, event, clock) -> evidence ctx dict for the
                         agents. Injected by the driver; a minimal default reads
                         the event's own OHLC/quote fields.
  * ``contract_resolver`` : (symbol, direction, event, clock) ->
                         (contract_symbol, bid, ask, market_dict) or None to
                         abstain. Injected (in the backtest, from the dataset's
                         option chain).
  * ``emit``           : the sink for produced events (normally ``bus.publish``).

Decision rule (unchanged philosophy): direction is an OUTPUT of the probability,
never a trigger. The adapter abstains unless a directional probability clears
``min_directional_p`` AND exceeds ``p_no_trade``. It may always abstain.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import (
    Any,
    Callable,
    Dict,
    List,
    Mapping,
    Optional,
    Protocol,
    runtime_checkable,
)

from oracle.engine.events import (
    BAR,
    OPTION_QUOTE,
    QUOTE,
    SESSION_CLOSE,
    SESSION_OPEN,
    TIMER,
    Event,
    SignalEvent,
    TradeIntentEvent,
)


# --------------------------------------------------------------------------- #
# Strategy protocol
# --------------------------------------------------------------------------- #
@runtime_checkable
class Strategy(Protocol):
    def on_bar(self, event: Event) -> None: ...
    def on_quote(self, event: Event) -> None: ...
    def on_option_quote(self, event: Event) -> None: ...
    def on_news(self, event: Event) -> None: ...
    def on_timer(self, event: Event) -> None: ...
    def on_fill(self, event: Event) -> None: ...
    def on_position_update(self, event: Event) -> None: ...
    def on_session_open(self, event: Event) -> None: ...
    def on_session_close(self, event: Event) -> None: ...


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class StrategyConfig:
    prior: float = 0.5
    min_directional_p: float = 0.55   # p_call/p_put must clear this to trade
    qty: int = 1
    decide_on: str = BAR              # which event type triggers a decision
    one_intent_per_symbol: bool = True
    emit_signal: bool = True          # also emit a SIGNAL event (telemetry)

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


def _default_ctx_builder(symbol: str, event: Event,
                         clock: Any) -> Dict[str, Any]:
    """Minimal evidence ctx from an event's own fields. Real backtests inject a
    richer builder (regime/features/catalyst); the agents fail-open on sparse
    ctx, so this keeps the adapter runnable + deterministic offline."""
    ctx: Dict[str, Any] = {}
    c = getattr(event, "c", None)
    o = getattr(event, "o", None)
    if c is not None and o is not None and o:
        ctx["trend"] = 1.0 if c > o else (-1.0 if c < o else 0.0)
        ctx["momentum"] = round((c - o) / o, 6)
    return ctx


# --------------------------------------------------------------------------- #
# Oracle adapter
# --------------------------------------------------------------------------- #
class OracleStrategyAdapter:
    """Implements ``Strategy`` by calling the existing Oracle decision pipeline
    and emitting ``TRADE_INTENT``. Broker-agnostic + deterministic."""

    def __init__(
        self,
        emit: Callable[[Event], None],
        *,
        config: Optional[StrategyConfig] = None,
        explain_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        ctx_builder: Optional[Callable[[str, Event, Any], Dict[str, Any]]] = None,
        contract_resolver: Optional[Callable[..., Optional[tuple]]] = None,
        clock: Any = None,
        use_kernel: bool = False,
        decision_config: Any = None,
        contracts_provider: Optional[Callable[..., Any]] = None,
        entry_limit_config: Optional[Mapping[str, Any]] = None,
    ) -> None:
        self.emit = emit
        self.config = config or StrategyConfig()
        self._explain_fn = explain_fn or _resolve_explain_fn()
        self._ctx_builder = ctx_builder or _default_ctx_builder
        self._contract_resolver = contract_resolver
        self.clock = clock
        # Opt-in shared decision kernel (Phase 2). Default OFF: the legacy
        # explain_fn path below is byte-for-byte unchanged when use_kernel is
        # False. When True, decisions come from oracle.decision.kernel.decide
        # over a Snapshot built by oracle.decision.snapshot_builder — the SAME
        # policy the scheduler shadow path and the replay consume.
        self._use_kernel = bool(use_kernel)
        self._decision_config = decision_config
        self._contracts_provider = contracts_provider
        # Opt-in EV-derived buy-limit entry variant (Phase 2). None = market
        # entry baseline (unchanged). A mapping turns kernel-mode entries into
        # limit orders capped by oracle.execution.limit_policy; keys:
        #   ev_floor (0.0), expiry_s (30.0), max_quote_age_s, max_spread_pct,
        #   penny_quoted (True), tick, cost_model,
        #   fallback: "abstain" (default; no cap -> no order, reason logged)
        #             or "market" (no cap -> baseline market entry).
        self._entry_limit_config = (dict(entry_limit_config)
                                    if entry_limit_config is not None else None)
        self.decisions: List[Any] = []    # kernel-mode telemetry (Decision objs)
        self._open_symbols: set = set()   # symbols with a live intent/position
        self.signals: List[dict] = []     # telemetry log (decisions made)
        self._corr = 0

    # -- Strategy protocol ------------------------------------------------ #
    def on_bar(self, event: Event) -> None:
        if self.config.decide_on == BAR:
            self._decide(event)

    def on_quote(self, event: Event) -> None:
        if self.config.decide_on == QUOTE:
            self._decide(event)

    def on_option_quote(self, event: Event) -> None:
        if self.config.decide_on == OPTION_QUOTE:
            self._decide(event)

    def on_news(self, event: Event) -> None:
        pass

    def on_timer(self, event: Event) -> None:
        if self.config.decide_on == TIMER:
            self._decide(event)

    def on_fill(self, event: Event) -> None:
        pass  # position accounting is the broker's job; adapter only reacts

    def on_position_update(self, event: Event) -> None:
        # Position fully closed -> allow a new intent on that symbol.
        if self.config.one_intent_per_symbol and getattr(event, "qty", 0) == 0:
            self._open_symbols.discard(event.symbol)

    def on_session_open(self, event: Event) -> None:
        pass

    def on_session_close(self, event: Event) -> None:
        self._open_symbols.clear()

    # -- routing helper (used by the bus subscription) -------------------- #
    def handle(self, event: Event) -> None:
        et = event.event_type
        if et == BAR:
            self.on_bar(event)
        elif et == QUOTE:
            self.on_quote(event)
        elif et == OPTION_QUOTE:
            self.on_option_quote(event)
        elif et == TIMER:
            self.on_timer(event)
        elif et == "NEWS":
            self.on_news(event)
        elif et == "ORDER_FILLED" or et == "ORDER_PARTIAL_FILL":
            self.on_fill(event)
        elif et == "POSITION_UPDATED":
            self.on_position_update(event)
        elif et == SESSION_OPEN:
            self.on_session_open(event)
        elif et == SESSION_CLOSE:
            self.on_session_close(event)

    # -- decision --------------------------------------------------------- #
    def _decide(self, event: Event) -> None:
        symbol = event.symbol
        if not symbol:
            return
        if self.config.one_intent_per_symbol and symbol in self._open_symbols:
            return
        if self._use_kernel:
            self._decide_kernel(event)
            return

        ctx = self._ctx_builder(symbol, event, self.clock) or {}
        try:
            explain = self._explain_fn(symbol, ctx, self.config.prior)
        except Exception:
            explain = None
        prob = (explain or {}).get("probability") or {}
        p_call = float(prob.get("p_call", 0.0) or 0.0)
        p_put = float(prob.get("p_put", 0.0) or 0.0)
        p_nt = float(prob.get("p_no_trade", 1.0) or 0.0)

        direction, p_dir = ("CALL", p_call) if p_call >= p_put else ("PUT", p_put)
        trade = (p_dir >= self.config.min_directional_p and p_dir > p_nt)

        if self.config.emit_signal:
            self.emit(SignalEvent(
                symbol=symbol, event_timestamp=event.event_timestamp,
                received_timestamp=event.received_timestamp,
                source="strategy", strategy_mode=event.strategy_mode,
                direction=(direction if trade else "NO_TRADE"),
                p_call=p_call, p_put=p_put, p_no_trade=p_nt))

        self.signals.append({
            "symbol": symbol, "direction": direction if trade else "NO_TRADE",
            "p_call": p_call, "p_put": p_put, "p_no_trade": p_nt,
            "ts": event.event_timestamp})

        if not trade:
            return

        resolved = self._resolve_contract(symbol, direction, event)
        if not resolved:
            return
        contract, bid, ask, market = resolved

        self._corr += 1
        self._open_symbols.add(symbol)
        self.emit(TradeIntentEvent(
            symbol=symbol, contract=contract, direction=direction,
            side="buy", qty=int(self.config.qty), order_type="market",
            bid=bid, ask=ask, market=dict(market or {}),
            event_timestamp=event.event_timestamp,
            received_timestamp=event.received_timestamp,
            source="strategy", strategy_mode=event.strategy_mode,
            correlation_id=f"corr-{self._corr:06d}"))

    def _resolve_contract(self, symbol: str, direction: str,
                          event: Event) -> Optional[tuple]:
        if self._contract_resolver is not None:
            try:
                return self._contract_resolver(symbol, direction, event,
                                               self.clock)
            except Exception:
                return None
        # Default (self-test): treat the event's own quote as the contract quote.
        bid = getattr(event, "bid", None)
        ask = getattr(event, "ask", None)
        if bid is None or ask is None:
            return None
        return (f"{symbol}-{direction}", bid, ask, {})

    # -- kernel mode (opt-in, Phase 2 shared policy) ----------------------- #
    def _gather_contracts(self, symbol: str, event: Event) -> tuple:
        """Candidate contracts for the kernel. Prefers the injected
        ``contracts_provider``; falls back to synthesizing one candidate per
        direction from ``contract_resolver``. Fail-open to ()."""
        if self._contracts_provider is not None:
            try:
                return tuple(self._contracts_provider(symbol, event,
                                                      self.clock) or ())
            except Exception:
                return ()
        out = []
        for d in ("CALL", "PUT"):
            resolved = self._resolve_contract(symbol, d, event)
            if resolved:
                contract, bid, ask, _market = resolved
                out.append({"symbol": contract, "type": d.lower(),
                            "bid": bid, "ask": ask})
        return tuple(out)

    def _decide_kernel(self, event: Event) -> None:
        """Shared-policy decision: Snapshot -> kernel.decide -> intent.

        The emitted TRADE_INTENT fields (contract, qty, side, order_type) come
        from ``snapshot_builder.intended_order`` — the same projection the
        scheduler shadow path records — so both paths stay provably identical.
        Fail-open: any import/build failure means abstain, never raise.
        """
        symbol = event.symbol
        try:
            from oracle.decision.kernel import decide
            from oracle.decision.snapshot_builder import (
                intended_order,
                snapshot_from_event,
            )
        except Exception:
            return

        ctx = self._ctx_builder(symbol, event, self.clock) or {}
        contracts = self._gather_contracts(symbol, event)
        snapshot = snapshot_from_event(event, ctx=ctx, contracts=contracts)
        decision = decide(snapshot, config=self._decision_config)
        self.decisions.append(decision)

        actionable = decision.is_actionable() and int(decision.size or 0) > 0
        direction = (decision.direction or "NO_TRADE").upper()
        if self.config.emit_signal:
            self.emit(SignalEvent(
                symbol=symbol, event_timestamp=event.event_timestamp,
                received_timestamp=event.received_timestamp,
                source="strategy", strategy_mode=event.strategy_mode,
                direction=(direction if actionable else "NO_TRADE"),
                p_call=float(decision.p_call or 0.0),
                p_put=float(decision.p_put or 0.0),
                p_no_trade=float(decision.p_no_trade or 0.0)))
        self.signals.append({
            "symbol": symbol,
            "direction": direction if actionable else "NO_TRADE",
            "p_call": decision.p_call, "p_put": decision.p_put,
            "p_no_trade": decision.p_no_trade,
            "action": decision.action, "vetoes": list(decision.vetoes),
            "fingerprint": decision.fingerprint(),
            "ts": event.event_timestamp})

        if not actionable:
            return
        order = intended_order(decision)
        if not order:
            return
        sc = decision.selected_contract or {}
        quote = snapshot.quote or {}
        bid = sc.get("bid", quote.get("bid"))
        ask = sc.get("ask", quote.get("ask"))

        if self._entry_limit_config is not None:
            order = self._apply_entry_limit(order, sc, quote, bid, ask)
            if order is None:
                return  # no valid cap; reason logged in signals entry

        self._corr += 1
        self._open_symbols.add(symbol)
        self.emit(TradeIntentEvent(
            symbol=symbol, contract=order["symbol"], direction=direction,
            side=order["side"], qty=int(order["qty"]),
            order_type=order["order_type"], limit_price=order["limit_price"],
            bid=bid, ask=ask, market={},
            event_timestamp=event.event_timestamp,
            received_timestamp=event.received_timestamp,
            source="strategy", strategy_mode=event.strategy_mode,
            correlation_id=f"corr-{self._corr:06d}"))

    # -- EV-capped limit entry (opt-in variant, Phase 2) -------------------- #
    def _apply_entry_limit(self, order: dict, sc: Mapping[str, Any],
                           quote: Mapping[str, Any], bid, ask) -> Optional[dict]:
        """Rewrite the intended order as an EV-capped buy limit, or abstain.

        Returns the (possibly rewritten) order dict, or None when no valid cap
        can be formed and the fallback is 'abstain'. The full ``EntryLimit``
        (cap, reason codes, net_ev_at_market baseline vs net_ev_at_limit) is
        logged on the current signals entry. Fail-open: any policy failure
        leaves the baseline market order unchanged."""
        cfg = self._entry_limit_config or {}
        try:
            from oracle.execution.limit_policy import (compute_entry_limit,
                                                       gross_ev_from_contract)
            age = None
            if isinstance(quote, Mapping):
                age = quote.get("quote_age_s", quote.get("age_s"))
            el = compute_entry_limit(
                gross_ev=gross_ev_from_contract(sc),
                qty=int(order.get("qty") or 0), bid=bid, ask=ask,
                quote_age_s=age,
                cost_model=cfg.get("cost_model"),
                ev_floor=cfg.get("ev_floor", 0.0),
                tick=cfg.get("tick"),
                penny_quoted=bool(cfg.get("penny_quoted", True)),
                max_quote_age_s=cfg.get("max_quote_age_s"),
                max_spread_pct=cfg.get("max_spread_pct"),
                expiry_s=cfg.get("expiry_s", 30.0))
        except Exception:
            return order  # fail-open: baseline market entry unchanged
        if self.signals:
            self.signals[-1]["entry_limit"] = el.to_dict()
        if el.is_ok:
            capped = dict(order)
            capped["order_type"] = "limit"
            capped["limit_price"] = el.limit_price
            capped["expiry_s"] = el.expiry_s
            return capped
        if str(cfg.get("fallback", "abstain")).lower() == "market":
            return order
        return None


def _resolve_explain_fn() -> Callable[..., Mapping[str, Any]]:
    """Bind the real ``compute_oracle_explain`` lazily so importing the engine
    never drags in the heavy report module unless a decision is actually made."""

    def _fn(ticker: str, ctx: Mapping[str, Any], prior: float):
        from oracle_intelligence_reports import compute_oracle_explain
        return compute_oracle_explain(ticker, dict(ctx or {}), prior=prior)

    return _fn


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic — uses a stubbed explain_fn, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime

    from oracle.engine.events import BarEvent

    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)
    emitted: List[Event] = []

    # Stub explain_fn: bullish ctx -> strong p_call; bearish -> strong p_put;
    # flat -> no trade. Deterministic, no network.
    def stub_explain(ticker, ctx, prior):
        trend = (ctx or {}).get("trend", 0.0)
        if trend > 0:
            prob = {"p_call": 0.7, "p_put": 0.2, "p_no_trade": 0.1}
        elif trend < 0:
            prob = {"p_call": 0.2, "p_put": 0.7, "p_no_trade": 0.1}
        else:
            prob = {"p_call": 0.3, "p_put": 0.3, "p_no_trade": 0.4}
        return {"ticker": ticker, "probability": prob, "verdict": "OK"}

    # Injected contract_resolver: in a real backtest the driver supplies the
    # option quote from the chain; bars themselves carry no option quote.
    def resolver(symbol, direction, event, clock):
        return (f"{symbol}-{direction}", 1.00, 1.05, {"ask_size": 100})

    adapter = OracleStrategyAdapter(
        emit=emitted.append, config=StrategyConfig(min_directional_p=0.55),
        explain_fn=stub_explain, contract_resolver=resolver)

    # Bullish bar -> a CALL TRADE_INTENT (contract priced by the resolver).
    up = BarEvent(symbol="AAPL", event_timestamp=t0, o=100.0, c=101.0)
    adapter.handle(up)
    intents = [e for e in emitted if e.event_type == "TRADE_INTENT"]
    if len(intents) != 1 or intents[0].direction != "CALL":
        print("FAIL: bullish should emit one CALL intent", intents); ok = False
    if intents and intents[0].correlation_id != "corr-000001":
        print("FAIL: correlation id", intents[0].correlation_id); ok = False

    # one_intent_per_symbol: a second bullish bar on AAPL must NOT re-enter.
    adapter.handle(BarEvent(symbol="AAPL", event_timestamp=t0, o=100.0, c=102.0))
    if len([e for e in emitted if e.event_type == "TRADE_INTENT"]) != 1:
        print("FAIL: duplicate intent not suppressed"); ok = False

    # Flat bar on a fresh symbol -> abstain (SIGNAL NO_TRADE, no intent).
    emitted.clear()
    adapter.handle(BarEvent(symbol="MSFT", event_timestamp=t0, o=100.0, c=100.0))
    if any(e.event_type == "TRADE_INTENT" for e in emitted):
        print("FAIL: flat should abstain"); ok = False
    sigs = [e for e in emitted if e.event_type == "SIGNAL"]
    if not sigs or sigs[0].direction != "NO_TRADE":
        print("FAIL: flat should signal NO_TRADE", sigs); ok = False

    # Bearish bar -> PUT intent.
    emitted.clear()
    adapter.handle(BarEvent(symbol="TSLA", event_timestamp=t0, o=100.0, c=98.0))
    puts = [e for e in emitted if e.event_type == "TRADE_INTENT"]
    if len(puts) != 1 or puts[0].direction != "PUT":
        print("FAIL: bearish should emit one PUT intent", puts); ok = False

    # Missing quote -> no intent (cannot price the contract), fail-open.
    emitted.clear()
    adapter2 = OracleStrategyAdapter(emit=emitted.append, explain_fn=stub_explain)
    adapter2.handle(BarEvent(symbol="NVDA", event_timestamp=t0, o=1.0, c=2.0))
    if any(e.event_type == "TRADE_INTENT" for e in emitted):
        print("FAIL: no-quote must not intent"); ok = False

    # Position closed -> symbol re-armed for a new intent.
    from oracle.engine.events import PositionEvent
    adapter.on_position_update(PositionEvent(symbol="AAPL", qty=0))
    if "AAPL" in adapter._open_symbols:
        print("FAIL: closed position should re-arm symbol"); ok = False

    # Protocol conformance.
    if not isinstance(adapter, Strategy):
        print("FAIL: adapter does not satisfy Strategy protocol"); ok = False

    print("engine.strategy self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
