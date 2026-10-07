"""Profitability gate validator for live option entries.

This module codifies the rule set that was identified from the realized trade
ledger: a candidate trade is only allowed when the directional conviction,
probability-of-profit, EV, and Oracle alignment all clear the same minimum bars.
It is designed for a dry-run validator and a live trade gate; the validator is a
pure function that returns pass/fail reasons instead of executing anything.
"""

from __future__ import annotations

from typing import Any, Dict, Iterable, List, Optional

DEFAULT_RULES = {
    "min_signal_strength": 4,
    "min_pop": 0.58,
    "require_positive_ev": True,
    "min_ev_per_dollar_risk": 0.008,
    "max_p_no_trade": 0.25,
    "min_directional_agreement": 0.55,
    "require_oracle_agreement": True,
    # PoP calibration (default OFF -> stamped PoP used verbatim, legacy behavior).
    # The realized ledger shows the model overstates PoP by ~28pp (stamped 66.5%
    # -> actual 38.1%). When enabled, the stamped PoP is corrected DOWN before the
    # min_pop bar so the gate compares against a measured win-rate estimate.
    # Accepted forms:
    #   None                         -> no correction
    #   {"offset_pp": 28}            -> corrected = pop - 0.28
    #   {"0.6": 0.35, "0.7": 0.42}   -> piecewise floor-bucket lookup (keys=raw 0.1 buckets)
    #   callable(pop) -> corrected   -> arbitrary curve
    "pop_calibration": None,
    # Session-bias penalty (default OFF -> legacy behavior). Measured from
    # live episodes (session_signal_eval.py): the signed underlying drift
    # over the REST OF THE SESSION after a signal fires. When a side's drift
    # is significantly negative (|t| >= min_abs_t and bp < 0), the
    # ev_per_dollar_risk bar is RAISED by the expected premium drag
    #   drag = |drift_bp| / 1e4 * leverage
    # where ``leverage`` approximates option gearing (delta * S / premium;
    # ~10-25x for the near-dated contracts traded here -- an estimate, not a
    # measurement). Veto-only: the rule can only raise the bar, never lower
    # it, and never flips a block into a pass. Fail-open on any bad config.
    # Form: {"call_bp": -12.6, "call_t": -4.0, "put_bp": 0.6, "put_t": 0.2,
    #        "leverage": 20.0, "min_abs_t": 2.0}
    "session_bias": None,
}


def _session_bias_drag(side: Optional[str], sb: Any) -> float:
    """Premium-drag (per $ risk) implied by measured session drift. Pure,
    fail-open: returns 0.0 (no penalty) unless the side's drift is negative
    AND statistically significant under the supplied rule dict."""
    try:
        if not side or not isinstance(sb, dict):
            return 0.0
        bp = _num(sb.get(f"{side}_bp"), None)
        t = _num(sb.get(f"{side}_t"), None)
        if bp is None or t is None or bp >= 0:
            return 0.0
        min_abs_t = _num(sb.get("min_abs_t"), 2.0) or 2.0
        if abs(t) < min_abs_t:
            return 0.0
        leverage = _num(sb.get("leverage"), 20.0) or 20.0
        return abs(bp) / 1e4 * leverage
    except Exception:
        return 0.0


def _apply_pop_calibration(pop: Optional[float], cal: Any) -> Optional[float]:
    """Map a stamped PoP through an optional calibration curve.

    Pure and fail-open: any missing/invalid calibration returns ``pop`` unchanged
    so the gate degrades to legacy behavior rather than blocking on a bad config.
    """
    if pop is None or cal is None:
        return pop
    try:
        if callable(cal):
            out = _num(cal(pop), pop)
        elif isinstance(cal, dict) and "offset_pp" in cal:
            off = _num(cal.get("offset_pp"), 0.0) or 0.0
            out = pop - (off / 100.0)
        elif isinstance(cal, dict):
            # Floor-bucket lookup on 0.1-wide raw-PoP buckets.
            bucket = int(float(pop) * 10) / 10.0
            hit = cal.get(f"{bucket:.1f}", cal.get(bucket))
            out = _num(hit, pop)
        else:
            out = pop
    except Exception:
        return pop
    if out is None:
        return pop
    return max(0.0, min(1.0, out))


def _num(value: Any, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value: Any, default: Optional[int] = None) -> Optional[int]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _norm_side(side: Any) -> Optional[str]:
    s = str(side or "").strip().lower()
    if s in {"call", "c", "buy_call", "long_call", "up", "bull", "bullish"}:
        return "call"
    if s in {"put", "p", "buy_put", "long_put", "down", "bear", "bearish"}:
        return "put"
    return None


def _option_dict(candidate: Dict[str, Any]) -> Dict[str, Any]:
    val = candidate.get("option") if isinstance(candidate, dict) else {}
    return val if isinstance(val, dict) else {}


def _entry_stamp(candidate: Dict[str, Any]) -> Dict[str, Any]:
    val = candidate.get("entry_stamp") if isinstance(candidate, dict) else {}
    return val if isinstance(val, dict) else {}


def _oracle_prob(candidate: Dict[str, Any]) -> Dict[str, Any]:
    data = candidate.get("oracle_probability") if isinstance(candidate, dict) else {}
    if isinstance(data, dict):
        return data
    data = candidate.get("oracle") if isinstance(candidate, dict) else {}
    if isinstance(data, dict):
        return data.get("probability", {}) if isinstance(data.get("probability"), dict) else {}
    return {}


def _robinhood_book(candidate: Dict[str, Any]) -> Dict[str, Any]:
    val = candidate.get("robinhood_book") if isinstance(candidate, dict) else {}
    return val if isinstance(val, dict) else {}


def _agreement_for_side(prob: Dict[str, Any], intended_side: Optional[str]) -> Optional[float]:
    if not intended_side:
        return None
    pc = _num(prob.get("p_call"), 0.0) or 0.0
    pp = _num(prob.get("p_put"), 0.0) or 0.0
    denom = pc + pp
    if denom <= 0:
        return None
    if intended_side == "call":
        return pc / denom
    if intended_side == "put":
        return pp / denom
    return None


def validate_trade(candidate: Dict[str, Any], rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Validate a candidate trade against the profitability rule set.

    Returns a dict shaped like:
        {
            "pass": True/False,
            "reasons": ["..."],
            "summary": { ... },
        }
    """
    if not isinstance(candidate, dict):
        return {"pass": False, "reasons": ["candidate is not a dict"], "summary": {}}

    cfg = {**DEFAULT_RULES, **(rules or {})}
    reasons: List[str] = []
    option = _option_dict(candidate)
    entry_stamp = _entry_stamp(candidate)
    oracle_prob = _oracle_prob(candidate)
    robinhood_book = _robinhood_book(candidate)

    signal_strength = None
    if "confidence" in option:
        signal_strength = _int(option.get("confidence"), None)
    elif "signal_strength" in candidate:
        signal_strength = _int(candidate.get("signal_strength"), None)
    if signal_strength is not None and signal_strength < cfg["min_signal_strength"]:
        reasons.append(
            f"signal_strength {signal_strength} < {cfg['min_signal_strength']}"
        )

    pop_raw = _num(entry_stamp.get("probability_of_profit"), None)
    pop = _apply_pop_calibration(pop_raw, cfg.get("pop_calibration"))
    if pop is not None and pop < cfg["min_pop"]:
        if pop_raw is not None and pop != pop_raw:
            reasons.append(
                f"probability_of_profit {pop:.3f} (calibrated from {pop_raw:.3f}) "
                f"< {cfg['min_pop']:.3f}"
            )
        else:
            reasons.append(f"probability_of_profit {pop:.3f} < {cfg['min_pop']:.3f}")

    ev = _num(entry_stamp.get("expected_value"), None)
    if cfg.get("require_positive_ev") and ev is not None and ev <= 0:
        reasons.append(f"expected_value {ev:.2f} <= 0")

    ev_per_dollar_risk = _num(entry_stamp.get("ev_per_dollar_risk"), None)
    if ev_per_dollar_risk is not None and ev_per_dollar_risk < cfg["min_ev_per_dollar_risk"]:
        reasons.append(
            f"ev_per_dollar_risk {ev_per_dollar_risk:.4f} < {cfg['min_ev_per_dollar_risk']:.4f}"
        )

    p_no_trade = _num(oracle_prob.get("p_no_trade"), None)
    if p_no_trade is not None and p_no_trade > cfg["max_p_no_trade"]:
        reasons.append(
            f"p_no_trade {p_no_trade:.3f} > {cfg['max_p_no_trade']:.3f}"
        )

    intended_side = _norm_side(
        option.get("type")
        or candidate.get("direction")
        or candidate.get("side")
        or candidate.get("intended_side")
    )
    agreement = _agreement_for_side(oracle_prob, intended_side)
    if cfg.get("require_oracle_agreement") and agreement is not None:
        if agreement < cfg["min_directional_agreement"]:
            reasons.append(
                f"oracle_agreement {agreement:.3f} < {cfg['min_directional_agreement']:.3f}"
            )

    if "orderbook_imbalance" in robinhood_book:
        ob_imb = _num(robinhood_book.get("orderbook_imbalance"), None)
        if ob_imb is not None and abs(ob_imb) < 0.05:
            reasons.append(f"orderbook_imbalance {ob_imb:.3f} too weak for conviction")

    # Session-bias penalty (veto-only): a significantly negative measured
    # rest-of-session drift for this side raises the ev_per_dollar_risk bar
    # by the implied premium drag. Default OFF (session_bias=None -> drag 0).
    session_drag = _session_bias_drag(intended_side, cfg.get("session_bias"))
    if session_drag > 0 and ev_per_dollar_risk is not None:
        raised_bar = cfg["min_ev_per_dollar_risk"] + session_drag
        if ev_per_dollar_risk < raised_bar:
            reasons.append(
                f"session_bias: ev_per_dollar_risk {ev_per_dollar_risk:.4f} < "
                f"{raised_bar:.4f} (base {cfg['min_ev_per_dollar_risk']:.4f} + "
                f"measured {intended_side} rest-of-session drag {session_drag:.4f})"
            )

    summary = {
        "signal_strength": signal_strength,
        "probability_of_profit": pop,
        "probability_of_profit_raw": pop_raw,
        "expected_value": ev,
        "ev_per_dollar_risk": ev_per_dollar_risk,
        "p_no_trade": p_no_trade,
        "oracle_agreement": agreement,
        "orderbook_imbalance": _num(robinhood_book.get("orderbook_imbalance"), None),
        "session_bias_side": intended_side,
        "session_bias_drag": session_drag,
    }
    return {"pass": not reasons, "reasons": reasons, "summary": summary}


def dry_run_scan(candidates: Iterable[Dict[str, Any]], rules: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """Evaluate a batch of candidate trades and count approvals / blocks."""
    out = []
    approved = 0
    blocked = 0
    for candidate in candidates or []:
        decision = validate_trade(candidate, rules)
        out.append({"candidate": candidate, **decision})
        if decision["pass"]:
            approved += 1
        else:
            blocked += 1
    return {
        "approved": approved,
        "blocked": blocked,
        "total": len(out),
        "results": out,
    }


if __name__ == "__main__":
    sample_ok = {
        "symbol": "AAPL",
        "entry_price": 1.25,
        "qty": 2,
        "option": {"type": "call", "confidence": 5, "delta": 0.55},
        "entry_stamp": {
            "expected_value": 2.5,
            "probability_of_profit": 0.68,
            "ev_per_dollar_risk": 0.02,
        },
        "oracle_probability": {"p_call": 0.60, "p_put": 0.20, "p_no_trade": 0.20},
        "robinhood_book": {"orderbook_imbalance": 0.18},
    }
    sample_bad = {
        "symbol": "AAPL",
        "entry_price": 1.25,
        "qty": 2,
        "option": {"type": "call", "confidence": 2, "delta": 0.30},
        "entry_stamp": {
            "expected_value": -0.5,
            "probability_of_profit": 0.45,
            "ev_per_dollar_risk": -0.004,
        },
        "oracle_probability": {"p_call": 0.20, "p_put": 0.30, "p_no_trade": 0.50},
        "robinhood_book": {"orderbook_imbalance": 0.03},
    }

    ok = validate_trade(sample_ok)
    bad = validate_trade(sample_bad)
    print("OK:", ok)
    print("BAD:", bad)
    assert ok["pass"] is True
    assert bad["pass"] is False

    # PoP calibration: default OFF must not change the legacy verdict.
    assert validate_trade(sample_ok)["summary"]["probability_of_profit"] == 0.68

    # With the measured ~28pp offset, sample_ok's stamped 0.68 -> 0.40 which now
    # fails the 0.58 bar. This is the whole point: the model overstates edge.
    cal_off = validate_trade(sample_ok, {"pop_calibration": {"offset_pp": 28}})
    assert abs(cal_off["summary"]["probability_of_profit"] - 0.40) < 1e-9
    assert cal_off["summary"]["probability_of_profit_raw"] == 0.68
    assert cal_off["pass"] is False
    assert any("calibrated from 0.680" in r for r in cal_off["reasons"])

    # Bucket-curve form (floor buckets) and callable form both supported.
    cal_curve = validate_trade(
        sample_ok, {"pop_calibration": {"0.6": 0.35, "0.7": 0.42}})
    assert abs(cal_curve["summary"]["probability_of_profit"] - 0.35) < 1e-9
    cal_call = validate_trade(
        sample_ok, {"pop_calibration": lambda p: p - 0.30})
    assert abs(cal_call["summary"]["probability_of_profit"] - 0.38) < 1e-9

    # Fail-open: a broken calibration leaves the stamped PoP untouched.
    def _boom(_):
        raise RuntimeError("bad curve")
    cal_bad = validate_trade(sample_ok, {"pop_calibration": _boom})
    assert cal_bad["summary"]["probability_of_profit"] == 0.68

    # Session bias: default OFF must not change the legacy verdict or drag.
    assert validate_trade(sample_ok)["summary"]["session_bias_drag"] == 0.0
    assert validate_trade(sample_ok)["pass"] is True

    # Live measured rule (session_signal_eval_live.json): CALL rest-of-day
    # drift -12.55bp (t=-4.05) -> drag 12.55/1e4*20 = 0.0251, raising the
    # CALL bar to 0.008+0.0251=0.0331 which blocks sample_ok (ev/$ = 0.02).
    live_rule = {"session_bias": {"call_bp": -12.55, "call_t": -4.05,
                                  "put_bp": 0.60, "put_t": 0.17,
                                  "leverage": 20.0, "min_abs_t": 2.0}}
    sb_call = validate_trade(sample_ok, live_rule)
    assert abs(sb_call["summary"]["session_bias_drag"] - 0.0251) < 1e-9
    assert sb_call["pass"] is False
    assert any(r.startswith("session_bias:") for r in sb_call["reasons"])

    # PUT side has an insignificant positive drift -> no penalty.
    sample_put = {**sample_ok, "option": {**sample_ok["option"], "type": "put"},
                  "oracle_probability": {"p_call": 0.20, "p_put": 0.60,
                                         "p_no_trade": 0.20}}
    sb_put = validate_trade(sample_put, live_rule)
    assert sb_put["summary"]["session_bias_drag"] == 0.0
    assert sb_put["pass"] is True

    # High-EV CALL clears the raised bar.
    sample_hi = {**sample_ok,
                 "entry_stamp": {**sample_ok["entry_stamp"],
                                 "ev_per_dollar_risk": 0.05}}
    assert validate_trade(sample_hi, live_rule)["pass"] is True

    # Fail-open: malformed session_bias config -> no penalty, legacy verdict.
    for junk in ("garbage", {"call_bp": "x", "call_t": None}, 42, [1, 2]):
        sb_junk = validate_trade(sample_ok, {"session_bias": junk})
        assert sb_junk["summary"]["session_bias_drag"] == 0.0
        assert sb_junk["pass"] is True

    print("profitability validator self-test: PASS")
