"""
Oracle Engine — Upgrade A: the deterministic backtest driver.

This is the seam that ties the engine together for offline research. It wires

    feed events  ->  EventBus (+ SimulationClock)  ->  OracleStrategyAdapter
                 ->  TRADE_INTENT  ->  SimBroker  ->  ORDER_* / POSITION_UPDATED

into one synchronous, single-threaded run. Because the bus is timestamp-ordered
and the broker is seeded, the same ``(feed, config, seed)`` always produces a
byte-identical order log and realized P&L — the Upgrade-A acceptance invariant.

The driver is an ADAPTER over the existing decision code; it does not reimplement
strategy logic. The Oracle decision math is injected as ``explain_fn`` (the real
``compute_oracle_explain`` or a cached/mocked stand-in). The option quote for a
directional decision is injected as ``contract_resolver`` (in a real backtest,
sourced from the dataset's option chain via a point-in-time ``MarketView``).

LLM determinism: a replay must be reproducible, so a backtest MUST NOT make live
LLM calls. When ``forbid_live_llm`` is set (the default) the driver refuses to
bind the real ``compute_oracle_explain`` implicitly — the caller must inject an
``explain_fn`` (a cached/mocked provider). This makes non-determinism impossible
by construction rather than by convention.

The driver owns no network, no creds, no wall-clock. ``run(events)`` is pure over
its inputs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Callable, Dict, List, Mapping, Optional

from oracle.engine.bus import EventBus
from oracle.engine.clock import SimulationClock
from oracle.engine.events import (
    BAR,
    OPTION_QUOTE,
    ORDER_FILLED,
    ORDER_PARTIAL_FILL,
    POSITION_UPDATED,
    QUOTE,
    SESSION_CLOSE,
    SESSION_OPEN,
    TIMER,
    TRADE_INTENT,
    Event,
)
from oracle.engine.sim_broker import SimBroker
from oracle.engine.strategy import OracleStrategyAdapter, StrategyConfig


@dataclass(frozen=True)
class DriverConfig:
    """Deterministic run parameters. ``seed`` drives the broker's fill sim."""

    seed: int = 0
    strategy: StrategyConfig = field(default_factory=StrategyConfig)
    forbid_live_llm: bool = True
    start: Optional[datetime] = None
    max_events: int = 1_000_000


@dataclass(frozen=True)
class BacktestResult:
    """Everything a caller needs to compare / persist a run. Every field is
    plain data so two results compare with ``==`` for the determinism check."""

    realized_pnl: float
    order_log: List[dict]
    signals: List[dict]
    open_positions: Dict[str, float]
    orders: int
    dispatched: int
    errors: List[tuple]

    def to_dict(self) -> dict:
        return {
            "realized_pnl": self.realized_pnl,
            "orders": self.orders,
            "open_positions": dict(self.open_positions),
            "dispatched": self.dispatched,
            "signals": len(self.signals),
            "errors": list(self.errors),
        }


class BacktestDriver:
    """Deterministic engine driver: feed -> strategy -> broker -> outcome."""

    def __init__(
        self,
        *,
        config: Optional[DriverConfig] = None,
        explain_fn: Optional[Callable[..., Mapping[str, Any]]] = None,
        ctx_builder: Optional[Callable[[str, Event, Any], Dict[str, Any]]] = None,
        contract_resolver: Optional[Callable[..., Optional[tuple]]] = None,
        fill_simulator: Any = None,
    ) -> None:
        self.config = config or DriverConfig()
        if self.config.forbid_live_llm and explain_fn is None:
            raise ValueError(
                "BacktestDriver forbids live LLM calls: inject a cached/mocked "
                "explain_fn (set DriverConfig.forbid_live_llm=False only for an "
                "explicitly offline compute_oracle_explain).")
        self._explain_fn = explain_fn
        self._ctx_builder = ctx_builder
        self._contract_resolver = contract_resolver
        self._fill_simulator = fill_simulator

    def run(self, events) -> BacktestResult:
        """Drain ``events`` through the wired engine and return the outcome.

        Rebuilds the bus/clock/adapter/broker every call so a driver instance is
        reusable and each run is independent (no cross-run state bleed)."""
        clock = SimulationClock(self.config.start)
        bus = EventBus(clock)

        adapter = OracleStrategyAdapter(
            emit=bus.publish,
            config=self.config.strategy,
            explain_fn=self._explain_fn,
            ctx_builder=self._ctx_builder,
            contract_resolver=self._contract_resolver,
            clock=clock,
        )
        broker = SimBroker(
            emit=bus.publish,
            fill_simulator=self._fill_simulator,
            seed=self.config.seed,
        )

        # Feed / lifecycle events -> strategy.
        for et in (BAR, QUOTE, OPTION_QUOTE, TIMER, SESSION_OPEN, SESSION_CLOSE):
            bus.subscribe(et, adapter.handle)
        # Broker feedback -> strategy (re-arm on flat / react to fills).
        for et in (ORDER_FILLED, ORDER_PARTIAL_FILL, POSITION_UPDATED):
            bus.subscribe(et, adapter.handle)
        # Strategy intents -> broker.
        bus.subscribe(TRADE_INTENT, broker.handle)

        bus.publish_all(events)
        dispatched = bus.run(max_events=self.config.max_events)

        summary = broker.summary()
        return BacktestResult(
            realized_pnl=summary["realized_pnl"],
            order_log=list(broker.order_log),
            signals=list(adapter.signals),
            open_positions=dict(summary["open_positions"]),
            orders=summary["orders"],
            dispatched=dispatched,
            errors=list(bus.errors),
        )


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic — no network / creds / LLM / file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from oracle.engine.events import BarEvent

    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)
    t1 = datetime(2026, 8, 11, 13, 31, 0)
    t2 = datetime(2026, 8, 11, 13, 32, 0)

    # Cached/mocked Oracle head: trend sign -> directional probability.
    def stub_explain(ticker, ctx, prior):
        trend = (ctx or {}).get("trend", 0.0)
        if trend > 0:
            prob = {"p_call": 0.7, "p_put": 0.2, "p_no_trade": 0.1}
        elif trend < 0:
            prob = {"p_call": 0.2, "p_put": 0.7, "p_no_trade": 0.1}
        else:
            prob = {"p_call": 0.3, "p_put": 0.3, "p_no_trade": 0.4}
        return {"ticker": ticker, "probability": prob, "verdict": "OK"}

    # Deterministic option quote per (symbol, direction, event) — priced off the
    # bar close so the CLOSE bar quotes higher than the OPEN bar (a winner).
    def resolver(symbol, direction, event, clock):
        c = getattr(event, "c", 100.0) or 100.0
        px = round(c / 100.0, 4)
        return (f"{symbol}-{direction}", px, round(px + 0.02, 4),
                {"ask_size": 100, "bid_size": 100})

    def make_feed():
        # Bar 1: bullish -> opens a CALL. Bar 2: exit signal via a sell intent is
        # not modeled here; instead a second symbol keeps the run non-trivial.
        return [
            BarEvent(symbol="AAPL", event_timestamp=t0, o=100.0, c=101.0),
            BarEvent(symbol="MSFT", event_timestamp=t1, o=100.0, c=100.0),  # flat
            BarEvent(symbol="TSLA", event_timestamp=t2, o=100.0, c=98.0),   # PUT
        ]

    def run_once(seed: int) -> BacktestResult:
        drv = BacktestDriver(
            config=DriverConfig(seed=seed,
                                strategy=StrategyConfig(min_directional_p=0.55)),
            explain_fn=stub_explain, contract_resolver=resolver)
        return drv.run(make_feed())

    r1 = run_once(seed=1)

    # Two intents opened (AAPL CALL, TSLA PUT); MSFT flat abstains.
    if r1.orders != 2:
        print("FAIL: expected 2 orders", r1.order_log); ok = False
    dirs = sorted(s["direction"] for s in r1.signals)
    if dirs != ["CALL", "NO_TRADE", "PUT"]:
        print("FAIL: signal directions", dirs); ok = False
    # Both positions opened (buys), so the book is NOT flat at end.
    if set(r1.open_positions) != {"AAPL-CALL", "TSLA-PUT"}:
        print("FAIL: open positions", r1.open_positions); ok = False

    # Determinism: same seed -> identical order log + realized P&L.
    r2 = run_once(seed=1)
    if r1.order_log != r2.order_log:
        print("FAIL: non-deterministic order log"); ok = False
    if r1.realized_pnl != r2.realized_pnl:
        print("FAIL: non-deterministic pnl", r1.realized_pnl, r2.realized_pnl)
        ok = False
    if r1.to_dict() != r2.to_dict():
        print("FAIL: non-deterministic result dict"); ok = False

    # No handler errors during a clean run.
    if r1.errors:
        print("FAIL: unexpected handler errors", r1.errors); ok = False

    # forbid_live_llm guard: default config with no explain_fn must raise.
    raised = False
    try:
        BacktestDriver(config=DriverConfig())
    except ValueError:
        raised = True
    if not raised:
        print("FAIL: forbid_live_llm did not block implicit explain_fn"); ok = False

    # Opt-out path: forbid_live_llm=False allows construction without explain_fn.
    try:
        BacktestDriver(config=DriverConfig(forbid_live_llm=False),
                       contract_resolver=resolver)
    except Exception as exc:  # pragma: no cover - should not raise
        print("FAIL: opt-out construction raised", exc); ok = False

    print("engine.backtest_driver self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
