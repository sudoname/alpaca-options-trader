"""
Oracle Decision Kernel — decide().

    decide(snapshot, portfolio_state, strategy_state, config) -> Decision

ONE decision path. The live scan, a paper scan and a historical replay all build
a ``Snapshot`` and call this function; there is no ``backtest_decide`` /
``paper_decide`` / ``live_decide``. The kernel is a pure function of its inputs
and performs NO I/O and submits NO orders — execution is handled separately by
the ``ExecutionClient`` adapters (Upgrade 1C). That separation is what makes
``decide(frozen_inputs)`` reproducible and testable, and what lets the parity
test assert ``decision_live == decision_backtest``.

Pipeline (unchanged Oracle methodology; each stage reuses the production code):
  1. DIRECTION  — ``oracle.decision.direction.compute_direction`` (faithful
     extraction of the ``determine_option_strategy`` tally). Direction is an
     OUTPUT of the evidence, never an input.
  2. HEAD       — ``oracle_intelligence_reports.compute_oracle_explain`` on the
     snapshot ctx -> p_call / p_put / p_no_trade (the 14-agent Bayesian head).
  3. MOVE       — implied move from the contract IV*sqrt(dte/365); expected move
     from ctx; move_edge = expected - implied.
  4. CONTRACT   — pick the best already-scored candidate matching the direction.
  5. EV / PoP   — theoretical EV + probability-of-profit from the entry stamp /
     contract (executable EV is layered on in Upgrade 2).
  6. CONVICTION — folded conviction score from the direction stage.
  7. VETOES     — advisory, VETO-ONLY gates (Oracle no-trade ceiling, EV floor).
     They can only demote ENTER->SKIP; they never flip direction, size up, or
     override the hard risk engine (which stays fail-closed in execution).
  8. SIZE       — faithful ``_confidence_to_quantity`` mapping.

Fail-open: any stage that errors degrades to a safe default (abstain / no veto
data), never raises. Deterministic in (snapshot, portfolio, strategy, config).
"""

from __future__ import annotations

import math
from typing import Any, Dict, Optional, Tuple

from oracle.decision import direction as _direction
from oracle.decision.schema import (
    ACTION_ENTER,
    ACTION_SKIP,
    Decision,
    DecisionConfig,
    PortfolioState,
    Snapshot,
    StrategyState,
    _to_float,
    make_no_trade,
)

# Faithful sizing defaults (mirror smart_trader).
_CONF_HIGH = 2
_CONF_VERY_HIGH = 4


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _head_probabilities(snapshot: Snapshot,
                        cfg: DecisionConfig) -> Dict[str, Optional[float]]:
    """Run the Oracle head on the snapshot ctx. Fail-open to empty probs. The
    head is pure given ctx (no network) so this stays deterministic."""
    try:
        import oracle_intelligence_reports as oir
        coherence = cfg.get_bool("USE_EXPECTED_MOVE_COHERENCE", False)
        emp = snapshot.ctx.get("expected_move_pct") if coherence else None
        em_ref = cfg.get_float("ORACLE_EM_REF_PCT", 0.0) if coherence else None
        head = oir.compute_oracle_explain(
            snapshot.symbol, ctx=dict(snapshot.ctx),
            expected_move_pct=emp,
            em_ref_pct=(em_ref if (coherence and em_ref) else None))
        prob = (head or {}).get("probability", {}) or {}
        return {
            "p_call": _to_float(prob.get("p_call")),
            "p_put": _to_float(prob.get("p_put")),
            "p_no_trade": _to_float(prob.get("p_no_trade")),
        }
    except Exception:
        return {"p_call": None, "p_put": None, "p_no_trade": None}


def _select_contract(snapshot: Snapshot,
                     direction: str) -> Optional[Dict[str, Any]]:
    """Choose the best already-scored candidate matching the direction. The
    heavy live scoring already ran in ``select_best_option``; the kernel only
    resolves the winner deterministically (max score, then max confidence)."""
    cands = [c for c in (snapshot.contracts or ())
             if str(c.get("type", direction)).lower() == direction]
    if not cands:
        cands = list(snapshot.contracts or ())
    if not cands:
        return None

    def _key(c: Dict[str, Any]) -> Tuple[float, float, str]:
        return (_to_float(c.get("score")) or -1e18,
                _to_float(c.get("confidence")) or -1e18,
                str(c.get("symbol") or ""))

    return max(cands, key=_key)


def _implied_move_pct(contract: Optional[Dict[str, Any]]) -> Optional[float]:
    """Closed-form implied move IV*sqrt(dte/365)*100 (matches smart_trader ~2713)."""
    if not contract:
        return None
    iv = _to_float(contract.get("iv"))
    dte = contract.get("dte")
    if dte is None:
        dte = contract.get("days_to_expiration")
    dte = _to_float(dte)
    if iv is None or dte is None:
        return None
    if iv > 3.0:            # looks like a percent, not a decimal
        iv /= 100.0
    em = iv * math.sqrt(max(int(dte), 1) / 365.0) * 100.0
    return round(em, 6) if em > 0.0 else None


def _ev_and_pop(snapshot: Snapshot, contract: Optional[Dict[str, Any]],
                cfg: DecisionConfig) -> Tuple[Optional[float], Optional[float],
                                              Optional[float]]:
    """Return (theoretical_ev, pop, ev_per_dollar_risk). Prefers values already
    on the contract / ctx; else computes the entry stamp when the inputs exist.
    Fail-open to (None, None, None)."""
    if contract:
        ev = _to_float(contract.get("expected_value"))
        pop = _to_float(contract.get("probability_of_profit")
                        or contract.get("pop"))
        evpdr = _to_float(contract.get("ev_per_dollar_risk"))
        if ev is not None or pop is not None:
            return ev, pop, evpdr

    ctx = snapshot.ctx or {}
    dyn = ctx.get("dynamic_levels")
    entry = _to_float(ctx.get("entry_price"))
    qty = ctx.get("order_quantity") or 1
    quote = snapshot.quote or {}
    bid = _to_float(quote.get("bid"))
    ask = _to_float(quote.get("ask"))
    if contract and isinstance(dyn, dict) and entry is not None and ask is not None:
        try:  # pragma: no cover - depends on optional ev module + inputs
            from entry_ev_stamp import compute_entry_stamp
            stamp = compute_entry_stamp(contract, dyn, entry, int(qty),
                                        bid=bid, ask=ask)
            if stamp:
                return (_to_float(stamp.get("expected_value")),
                        _to_float(stamp.get("probability_of_profit")),
                        _to_float(stamp.get("ev_per_dollar_risk")))
        except Exception:
            pass
    return None, None, None


def _confidence_to_quantity(strength: Any, cfg: DecisionConfig) -> int:
    """Byte-faithful copy of smart_trader._confidence_to_quantity."""
    try:
        s = int(strength)
    except (TypeError, ValueError):
        return 1
    if cfg.get_bool("USE_NORMALIZED_CONFIDENCE", False) and s <= 0:
        return 0
    if s >= cfg.get_int("CONF_VERY_HIGH_SIGNALS", _CONF_VERY_HIGH):
        return 3
    if s >= cfg.get_int("CONF_HIGH_SIGNALS", _CONF_HIGH):
        return 2
    return 1


def _conviction_score(dr: "_direction.DirectionResult") -> Optional[float]:
    conv = dr.conviction
    if isinstance(conv, dict):
        return _to_float(conv.get("score") or conv.get("conviction"))
    return _to_float(conv)


# --------------------------------------------------------------------------- #
# decide()
# --------------------------------------------------------------------------- #
def decide(snapshot: Snapshot,
           portfolio_state: Optional[PortfolioState] = None,
           strategy_state: Optional[StrategyState] = None,
           config: Optional[DecisionConfig] = None) -> Decision:
    """Produce one immutable Decision from frozen inputs. Never raises."""
    cfg = config or DecisionConfig.make({})
    portfolio_state = portfolio_state or PortfolioState.make()
    strategy_state = strategy_state or StrategyState.make()
    cver = cfg.config_version

    try:
        dr = _direction.compute_direction(snapshot, cfg)
    except Exception:
        return make_no_trade(snapshot.symbol, snapshot.timestamp,
                             snapshot.strategy_mode,
                             reasons=("direction_error",), config_version=cver)

    probs = _head_probabilities(snapshot, cfg)

    # Weak-signal abstain: the tally chose to SKIP -> NO-TRADE.
    if dr.direction == "skip":
        return make_no_trade(
            snapshot.symbol, snapshot.timestamp, snapshot.strategy_mode,
            reasons=(dr.skip_reason or "weak_signal",),
            p_no_trade=probs.get("p_no_trade"), config_version=cver)

    direction = dr.direction  # 'call' | 'put'
    contract = _select_contract(snapshot, direction)
    implied_move = _implied_move_pct(contract)
    expected_move = _to_float((snapshot.ctx or {}).get("expected_move_pct"))
    move_edge = (round(expected_move - implied_move, 6)
                 if (expected_move is not None and implied_move is not None)
                 else None)

    theoretical_ev, pop, evpdr = _ev_and_pop(snapshot, contract, cfg)
    conviction = _conviction_score(dr)
    invalidation = _to_float((snapshot.ctx or {}).get("invalidation_pct"))
    size = _confidence_to_quantity(dr.confidence, cfg)

    reasons: Tuple[str, ...] = (f"tally bull={dr.bullish_signals} "
                                f"bear={dr.bearish_signals} "
                                f"conf={dr.confidence}",)

    # ---- advisory VETO-ONLY gates (never flip direction / override risk) ----
    vetoes = []

    # Oracle no-trade ceiling (mirror oracle_trade_gate; opt-in).
    if cfg.get_bool("USE_ORACLE_TRADE_GATE", False):
        p_nt = probs.get("p_no_trade")
        ceiling = cfg.get_float("ORACLE_MAX_NO_TRADE", 0.85)
        if p_nt is not None and p_nt > ceiling:
            vetoes.append(f"oracle_gate: p_no_trade {p_nt:.2f} > {ceiling:.2f}")

    # EV floor (mirror the EV entry gate; opt-in).
    if cfg.get_bool("USE_EV_ENTRY_GATE", False) and evpdr is not None:
        floor = cfg.get_float("MIN_EV_PER_DOLLAR_RISK", 0.0)
        if evpdr < floor:
            vetoes.append(f"ev_gate: ev/$risk {evpdr:+.4f} < {floor:+.4f}")

    # Zero-size (normalized confidence collapsed) is a legitimate abstain.
    if size <= 0:
        return make_no_trade(
            snapshot.symbol, snapshot.timestamp, snapshot.strategy_mode,
            reasons=reasons + ("zero_size",),
            p_no_trade=probs.get("p_no_trade"), config_version=cver)

    action = ACTION_SKIP if vetoes else ACTION_ENTER

    return Decision(
        timestamp=snapshot.timestamp,
        symbol=snapshot.symbol,
        strategy_mode=snapshot.strategy_mode,
        action=action,
        direction=direction,
        p_call=probs.get("p_call"),
        p_put=probs.get("p_put"),
        p_no_trade=probs.get("p_no_trade"),
        expected_move=expected_move,
        implied_move=implied_move,
        move_edge=move_edge,
        selected_contract=contract,
        theoretical_ev=theoretical_ev,
        pop=pop,
        conviction=conviction,
        size=(0 if vetoes else size),
        invalidation=invalidation,
        vetoes=tuple(vetoes),
        reasons=reasons,
        model_version=snapshot.model_version,
        config_version=cver,
    )


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    up = Snapshot.make(
        "AAA", "2024-01-02T16:00:00", strategy_mode="intraday",
        prices=[100, 101, 102, 104, 106], momentum=0.05, volatility=0.2,
        market_regime="trending",
        ctx={"expected_move_pct": 3.0},
        contracts=[
            {"symbol": "AAA_C1", "type": "call", "iv": 0.30, "dte": 5,
             "score": 8.0, "confidence": 2, "expected_value": 15.0,
             "probability_of_profit": 0.55, "ev_per_dollar_risk": 0.12},
            {"symbol": "AAA_C2", "type": "call", "iv": 0.30, "dte": 5,
             "score": 6.0, "confidence": 2},
            {"symbol": "AAA_P1", "type": "put", "iv": 0.30, "dte": 5,
             "score": 9.0},
        ])

    d = decide(up)
    if not d.is_actionable() or d.direction != "call":
        print("FAIL: up should ENTER call", d.action, d.direction); ok = False
    # Best matching-direction contract chosen (C1 over C2; the higher-scored
    # PUT must NOT win because direction filters first).
    if not d.selected_contract or d.selected_contract.get("symbol") != "AAA_C1":
        print("FAIL: contract selection", d.selected_contract); ok = False
    if d.theoretical_ev != 15.0 or d.pop != 0.55:
        print("FAIL: ev/pop passthrough", d.theoretical_ev, d.pop); ok = False
    if d.implied_move is None or d.expected_move != 3.0 or d.move_edge is None:
        print("FAIL: move fields", d.implied_move, d.expected_move,
              d.move_edge); ok = False
    if d.size < 1:
        print("FAIL: size", d.size); ok = False
    if not d.config_version.startswith("cfg-"):
        print("FAIL: config_version stamp", d.config_version); ok = False

    # Determinism / parity-in-miniature: same inputs -> equal Decision.
    if decide(up) != decide(up):
        print("FAIL: non-deterministic decide"); ok = False

    # Oracle gate veto demotes ENTER -> SKIP without flipping direction. Force a
    # ceiling of 0 so any positive p_no_trade trips it; if the head produced no
    # probability the gate is inert (acceptable — fail-open).
    cfg_gate = DecisionConfig.make({"USE_ORACLE_TRADE_GATE": True,
                                    "ORACLE_MAX_NO_TRADE": 0.0})
    dg = decide(up, config=cfg_gate)
    if dg.p_no_trade is not None:
        if dg.action != ACTION_SKIP or dg.direction != "call" or dg.size != 0:
            print("FAIL: oracle-gate veto", dg.action, dg.size); ok = False
        if not any("oracle_gate" in v for v in dg.vetoes):
            print("FAIL: oracle-gate veto reason", dg.vetoes); ok = False

    # EV floor veto (opt-in): floor above the contract's ev/$risk -> SKIP.
    cfg_ev = DecisionConfig.make({"USE_EV_ENTRY_GATE": True,
                                  "MIN_EV_PER_DOLLAR_RISK": 0.50})
    de = decide(up, config=cfg_ev)
    if de.action != ACTION_SKIP or not any("ev_gate" in v for v in de.vetoes):
        print("FAIL: ev-gate veto", de.action, de.vetoes); ok = False

    # Weak-signal skip -> NO-TRADE (the system may always abstain).
    flat = Snapshot.make("DDD", "t", prices=[100, 100, 100, 100, 100],
                         momentum=0.0, volatility=0.1)
    cfg_skip = DecisionConfig.make({"USE_SKIP_ON_WEAK_SIGNAL": True,
                                    "MIN_DIRECTION_SIGNALS": 2})
    ds = decide(flat, config=cfg_skip)
    if ds.action != "no_trade" or ds.direction is not None:
        print("FAIL: weak-signal no-trade", ds.action); ok = False

    # Normalized confidence collapse -> zero size -> NO-TRADE.
    cfg_norm = DecisionConfig.make({"USE_NORMALIZED_CONFIDENCE": True})
    tie = Snapshot.make("EEE", "t", prices=[100, 100, 100, 100, 100],
                        momentum=0.0, volatility=0.1)
    # Flat tally -> confidence 0 under normalized -> size 0 -> no_trade.
    dn = decide(tie, config=cfg_norm)
    if dn.action != "no_trade":
        print("FAIL: normalized zero-size no-trade", dn.action); ok = False

    # Junk never raises.
    try:
        decide(Snapshot.make("X", "t", prices=["a", None]))
    except Exception as exc:  # pragma: no cover
        print("FAIL: raised on junk", exc); ok = False

    print("decision.kernel self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
