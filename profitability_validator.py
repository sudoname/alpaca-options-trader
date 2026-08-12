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
}


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

    pop = _num(entry_stamp.get("probability_of_profit"), None)
    if pop is not None and pop < cfg["min_pop"]:
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

    summary = {
        "signal_strength": signal_strength,
        "probability_of_profit": pop,
        "expected_value": ev,
        "ev_per_dollar_risk": ev_per_dollar_risk,
        "p_no_trade": p_no_trade,
        "oracle_agreement": agreement,
        "orderbook_imbalance": _num(robinhood_book.get("orderbook_imbalance"), None),
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
    print("profitability validator self-test: PASS")
