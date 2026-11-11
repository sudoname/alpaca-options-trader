"""
Oracle Decision Kernel — shared Snapshot construction (Phase 2, Upgrade 1B).

The kernel (``oracle.decision.kernel.decide``) is already one decision path,
but each caller used to hand-roll its own ``Snapshot.make`` call — so the
scheduler shadow path and the replay path could quietly feed the kernel
*differently shaped* inputs and still both claim to use "the same policy".
This module closes that gap: BOTH paths build their Snapshot here, through one
``_build`` funnel, so identical facts produce identical (``==``) Snapshots and
therefore identical Decisions (Decision equality is by fingerprint).

Two entry points, one funnel:

  * ``snapshot_from_candidate(candidate, timestamp=...)`` — from the live
    scheduler's candidate dict (``run_alpaca_intraday._evaluate`` shape:
    ``{"sym", "direction", "opt", "score", "underlying_price"}``).
  * ``snapshot_from_event(event, ...)`` — from an engine event
    (``oracle.engine.events``; symbol / event_timestamp / strategy_mode /
    bid / ask are read off the event itself).

Reserved ctx keys (the transport trick): an engine ``ctx_builder`` only gets to
return one dict, but a Snapshot separates prices / momentum / volatility /
regime / news / quote into their own fields. So both builders POP these
reserved, underscore-prefixed keys OUT of ctx and into the snapshot fields:

    _prices, _momentum, _volatility, _market_regime, _news, _quote

Precedence per field: explicit kwarg > reserved ctx key > inferred from the
source object (candidate opt / event quote). The reserved keys never remain in
``Snapshot.ctx`` — ctx stays byte-comparable across paths.

``intended_order(decision)`` is the canonical order-intent projection both
paths share: the integration test asserts the replay engine's TRADE_INTENT and
the scheduler shadow's would-be order agree field-for-field.

Pure + fail-open: no I/O, never raises; junk inputs degrade to an empty-ish
Snapshot / ``None`` order exactly like ``Snapshot.make`` itself.
"""

from __future__ import annotations

from typing import Any, Dict, Mapping, Optional, Sequence

from oracle.decision.schema import Decision, Snapshot, _to_float, _to_str

# Reserved ctx keys popped into Snapshot fields (see module docstring).
CTX_PRICES = "_prices"
CTX_MOMENTUM = "_momentum"
CTX_VOLATILITY = "_volatility"
CTX_MARKET_REGIME = "_market_regime"
CTX_NEWS = "_news"
CTX_QUOTE = "_quote"
RESERVED_CTX_KEYS = (CTX_PRICES, CTX_MOMENTUM, CTX_VOLATILITY,
                     CTX_MARKET_REGIME, CTX_NEWS, CTX_QUOTE)


def _quote_dict(bid: Any, ask: Any) -> Optional[Dict[str, Any]]:
    b, a = _to_float(bid), _to_float(ask)
    if b is None and a is None:
        return None
    return {"bid": b, "ask": a}


def _build(symbol: Any, timestamp: Any, *, strategy_mode: Any,
           ctx: Optional[Mapping[str, Any]],
           prices: Sequence[float], momentum: Any, volatility: Any,
           market_regime: Any, news: Sequence[Any],
           contracts: Sequence[Mapping[str, Any]],
           quote: Optional[Mapping[str, Any]]) -> Snapshot:
    """Single funnel: pop reserved ctx keys into fields, then Snapshot.make.

    Explicit kwargs win over reserved ctx keys; reserved keys never survive
    into Snapshot.ctx.
    """
    c = dict(ctx) if isinstance(ctx, Mapping) else {}
    r_prices = c.pop(CTX_PRICES, None)
    r_mom = c.pop(CTX_MOMENTUM, None)
    r_vol = c.pop(CTX_VOLATILITY, None)
    r_regime = c.pop(CTX_MARKET_REGIME, None)
    r_news = c.pop(CTX_NEWS, None)
    r_quote = c.pop(CTX_QUOTE, None)

    if not prices and r_prices is not None:
        prices = r_prices
    if momentum is None:
        momentum = r_mom
    if volatility is None:
        volatility = r_vol
    if market_regime is None:
        market_regime = r_regime
    if not news and r_news is not None:
        news = r_news
    if quote is None and isinstance(r_quote, Mapping):
        quote = r_quote

    return Snapshot.make(
        symbol, timestamp, strategy_mode=(strategy_mode or "intraday"),
        ctx=c, prices=prices or (), momentum=momentum, volatility=volatility,
        market_regime=market_regime, news=news or (),
        contracts=contracts or (), quote=quote)


# --------------------------------------------------------------------------- #
# Builders
# --------------------------------------------------------------------------- #
def snapshot_from_candidate(candidate: Optional[Mapping[str, Any]], *,
                            timestamp: Any,
                            ctx: Optional[Mapping[str, Any]] = None,
                            prices: Sequence[float] = (),
                            momentum: Any = None, volatility: Any = None,
                            market_regime: Any = None,
                            news: Sequence[Any] = (),
                            contracts: Optional[Sequence[Mapping[str, Any]]] = None,
                            quote: Optional[Mapping[str, Any]] = None,
                            strategy_mode: str = "intraday") -> Snapshot:
    """Snapshot from a scheduler candidate (``_evaluate`` dict).

    ``contracts`` defaults to the candidate's winning ``opt`` as a one-element
    list (the live path has already run ``select_best_option``); pass the full
    scored list to let the kernel re-select. ``underlying_price`` folds into
    ctx (if not already there) so the head sees it on both paths.
    """
    cand = candidate if isinstance(candidate, Mapping) else {}
    opt = cand.get("opt")
    opt = opt if isinstance(opt, Mapping) else {}

    c = dict(ctx) if isinstance(ctx, Mapping) else {}
    up = _to_float(cand.get("underlying_price"))
    if up is not None and "underlying_price" not in c:
        c["underlying_price"] = up

    if contracts is None:
        contracts = (opt,) if opt else ()
    if quote is None and CTX_QUOTE not in c:
        quote = _quote_dict(opt.get("bid"), opt.get("ask"))

    return _build(cand.get("sym") or cand.get("symbol"), timestamp,
                  strategy_mode=strategy_mode, ctx=c, prices=prices,
                  momentum=momentum, volatility=volatility,
                  market_regime=market_regime, news=news,
                  contracts=contracts, quote=quote)


def snapshot_from_event(event: Any, *,
                        ctx: Optional[Mapping[str, Any]] = None,
                        prices: Sequence[float] = (),
                        momentum: Any = None, volatility: Any = None,
                        market_regime: Any = None,
                        news: Sequence[Any] = (),
                        contracts: Sequence[Mapping[str, Any]] = (),
                        quote: Optional[Mapping[str, Any]] = None,
                        strategy_mode: Optional[str] = None) -> Snapshot:
    """Snapshot from an engine event (bar / quote / timer).

    Symbol, timestamp and strategy_mode come off the event; a quote on the
    event itself (QuoteEvent / OptionQuoteEvent) is used only when neither an
    explicit ``quote`` nor a ``_quote`` ctx key is supplied.
    """
    symbol = getattr(event, "symbol", None)
    ts = getattr(event, "event_timestamp", None)
    try:
        ts = ts.isoformat() if hasattr(ts, "isoformat") else ts
    except Exception:
        ts = _to_str(ts)
    mode = strategy_mode or getattr(event, "strategy_mode", None) or "intraday"

    c = dict(ctx) if isinstance(ctx, Mapping) else {}
    if quote is None and CTX_QUOTE not in c:
        quote = _quote_dict(getattr(event, "bid", None),
                            getattr(event, "ask", None))

    return _build(symbol, ts, strategy_mode=mode, ctx=c, prices=prices,
                  momentum=momentum, volatility=volatility,
                  market_regime=market_regime, news=news,
                  contracts=contracts, quote=quote)


# --------------------------------------------------------------------------- #
# Intended order — the shared execution-facing projection of a Decision
# --------------------------------------------------------------------------- #
def intended_order(decision: Optional[Decision], *,
                   order_type: str = "market",
                   limit_price: Any = None) -> Optional[Dict[str, Any]]:
    """Canonical would-be order for an actionable Decision, else ``None``.

    Both paths derive their order fields from this one projection: the replay
    adapter's TRADE_INTENT and the scheduler shadow's recorded intent must
    match it field-for-field. Long-only single leg (side fixed to "buy"),
    quantity is the kernel's ``decision.size`` — never a config default.
    """
    try:
        if decision is None or not decision.is_actionable():
            return None
        if int(decision.size) <= 0:
            return None
        sc = decision.selected_contract or {}
        sym = _to_str(sc.get("symbol"))
        if not sym:
            return None
        return {
            "symbol": sym,
            "side": "buy",
            "qty": int(decision.size),
            "order_type": _to_str(order_type) or "market",
            "limit_price": _to_float(limit_price),
            "direction": decision.direction,
            "underlying": decision.symbol,
            "timestamp": decision.timestamp,
        }
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic — no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime

    from oracle.decision.kernel import decide
    from oracle.engine.events import BarEvent

    ok = True
    t0 = datetime(2024, 1, 2, 16, 0, 0)
    contracts = [
        {"symbol": "AAA_C1", "type": "call", "iv": 0.30, "dte": 5,
         "score": 8.0, "confidence": 2, "expected_value": 15.0,
         "probability_of_profit": 0.55, "ev_per_dollar_risk": 0.12,
         "bid": 1.00, "ask": 1.05},
        {"symbol": "AAA_C2", "type": "call", "iv": 0.30, "dte": 5,
         "score": 6.0, "confidence": 2},
        {"symbol": "AAA_P1", "type": "put", "iv": 0.30, "dte": 5, "score": 9.0},
    ]
    facts = {"expected_move_pct": 3.0, "underlying_price": 498.0}
    prices = [100, 101, 102, 104, 106]

    # Path A: scheduler candidate shape.
    cand = {"sym": "AAA", "direction": "call", "opt": contracts[0],
            "score": 8.0, "underlying_price": 498.0}
    snap_a = snapshot_from_candidate(
        cand, timestamp=t0.isoformat(), ctx={"expected_move_pct": 3.0},
        prices=prices, momentum=0.05, volatility=0.2,
        market_regime="trending", contracts=contracts)

    # Path B: engine event shape; series carried via reserved ctx keys (the
    # way an engine ctx_builder transports them).
    ev = BarEvent(symbol="AAA", event_timestamp=t0, o=100.0, c=106.0,
                  strategy_mode="intraday")
    snap_b = snapshot_from_event(
        ev, ctx={**facts, CTX_PRICES: prices, CTX_MOMENTUM: 0.05,
                 CTX_VOLATILITY: 0.2, CTX_MARKET_REGIME: "trending",
                 CTX_QUOTE: {"bid": 1.00, "ask": 1.05}},
        contracts=contracts)

    if snap_a != snap_b:
        print("FAIL: candidate- and event-sourced snapshots differ")
        print("  A:", snap_a)
        print("  B:", snap_b)
        ok = False
    for key in RESERVED_CTX_KEYS:
        if key in snap_b.ctx:
            print("FAIL: reserved ctx key leaked into Snapshot.ctx:", key)
            ok = False

    # Same Snapshot -> same Decision (the shared-policy invariant).
    d_a, d_b = decide(snap_a), decide(snap_b)
    if d_a != d_b or d_a.fingerprint() != d_b.fingerprint():
        print("FAIL: decisions differ across sources"); ok = False
    if not d_a.is_actionable() or (d_a.selected_contract or {}).get("symbol") != "AAA_C1":
        print("FAIL: expected actionable AAA_C1 decision", d_a.action); ok = False

    # intended_order projection.
    o = intended_order(d_a)
    if not o or o["symbol"] != "AAA_C1" or o["qty"] != d_a.size or \
            o["side"] != "buy" or o["order_type"] != "market" or \
            o["underlying"] != "AAA":
        print("FAIL: intended_order shape", o); ok = False
    if intended_order(d_a) != intended_order(d_b):
        print("FAIL: intended_order parity"); ok = False

    # Explicit kwarg beats reserved ctx key.
    s_kw = snapshot_from_event(ev, ctx={CTX_MOMENTUM: 9.9}, momentum=0.05)
    if s_kw.momentum != 0.05:
        print("FAIL: kwarg should beat reserved ctx key", s_kw.momentum); ok = False

    # Candidate quote inference from the winning opt (bid/ask on opt).
    s_q = snapshot_from_candidate(cand, timestamp="t")
    if s_q.quote != {"bid": 1.00, "ask": 1.05}:
        print("FAIL: quote inference from opt", s_q.quote); ok = False
    if s_q.contracts != (dict(contracts[0]),):
        print("FAIL: default contracts should be the winning opt"); ok = False
    if s_q.ctx.get("underlying_price") != 498.0:
        print("FAIL: underlying_price should fold into ctx", s_q.ctx); ok = False

    # Non-actionable / junk -> None order; builders never raise on junk.
    from oracle.decision.schema import make_no_trade
    if intended_order(make_no_trade("AAA", "t", "intraday")) is not None:
        print("FAIL: no-trade must not produce an order"); ok = False
    if intended_order(None) is not None:
        print("FAIL: None decision must not produce an order"); ok = False
    try:
        snapshot_from_candidate(None, timestamp=None)
        snapshot_from_candidate({"sym": None, "opt": 42}, timestamp="t")
        snapshot_from_event(object())
    except Exception as exc:  # pragma: no cover
        print("FAIL: builders raised on junk:", exc); ok = False

    print("decision.snapshot_builder self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
