"""
Phase F (Oracle 2.1) — live Oracle trade gate (veto-only, pure).

Turns the previously shadow-only Oracle direction head into a *veto*: given the
enriched head probabilities and the trade the bot already intends to place, it
can only say "skip this" — it never sizes, never flips direction, never invents
a trade, and never overrides the EV / risk gates that still run after it.

Three veto rules, each fail-open to allow:

    no-trade mass   block when p_no_trade >= ORACLE_MAX_NO_TRADE (the head is
                    overwhelmingly "sit out").
    weak signal     block when max(p_call, p_put) < ORACLE_MIN_DIRECTIONAL
                    (no directional conviction either way).
    disagreement    block when the share of directional mass on the intended
                    side < ORACLE_MIN_AGREEMENT (the head leans the other way).

Anything malformed / missing -> allow. This is intentionally additive: it can
only ever turn allow -> skip.
"""

from typing import Optional

# Defaults (overridable via config). Chosen so a middling head passes: only a
# strongly-neutral or strongly-contrary head vetoes.
ORACLE_MAX_NO_TRADE_DEFAULT = 0.85
ORACLE_MIN_DIRECTIONAL_DEFAULT = 0.15
ORACLE_MIN_AGREEMENT_DEFAULT = 0.50


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _cfg(config, key: str, default: float) -> float:
    """Read a numeric threshold from a dict or an object, fail-open to default."""
    try:
        if config is None:
            return default
        if isinstance(config, dict):
            val = config.get(key)
        else:
            val = getattr(config, key, None)
            if val is None and hasattr(config, "get"):
                try:
                    val = config.get(key)
                except Exception:
                    val = None
        f = _to_float(val)
        return f if f is not None else default
    except Exception:
        return default


def _norm_side(side) -> Optional[str]:
    s = str(side or "").strip().lower()
    if s in ("call", "c", "buy_call", "long_call", "up", "bull", "bullish"):
        return "call"
    if s in ("put", "p", "buy_put", "long_put", "down", "bear", "bearish"):
        return "put"
    return None


def evaluate_oracle_gate(oracle_payload, config=None) -> dict:
    """Veto-only decision. ``oracle_payload`` carries the head probabilities
    (either at the top level or under ``probability``) and the intended side
    (``intended_side`` / ``direction`` / ``side``). Returns
    ``{allow, reason, direction}`` where ``direction`` is the head's favoured
    side. Never raises — any problem yields allow.
    """
    try:
        payload = oracle_payload if isinstance(oracle_payload, dict) else {}
        prob = payload.get("probability")
        if not isinstance(prob, dict):
            prob = payload
        p_call = _to_float(prob.get("p_call"))
        p_put = _to_float(prob.get("p_put"))
        p_no_trade = _to_float(prob.get("p_no_trade"))

        # No usable head -> fail-open allow.
        if p_call is None and p_put is None and p_no_trade is None:
            return {"allow": True, "reason": "no oracle head", "direction": None}

        pc = p_call or 0.0
        pp = p_put or 0.0
        head_dir = "call" if pc > pp else ("put" if pp > pc else None)

        max_no_trade = _cfg(config, "ORACLE_MAX_NO_TRADE",
                            ORACLE_MAX_NO_TRADE_DEFAULT)
        min_directional = _cfg(config, "ORACLE_MIN_DIRECTIONAL",
                              ORACLE_MIN_DIRECTIONAL_DEFAULT)
        min_agreement = _cfg(config, "ORACLE_MIN_AGREEMENT",
                            ORACLE_MIN_AGREEMENT_DEFAULT)

        # Rule 1 — overwhelming no-trade mass.
        if p_no_trade is not None and p_no_trade >= max_no_trade:
            return {"allow": False, "direction": head_dir,
                    "reason": (f"oracle no-trade {p_no_trade:.0%} "
                               f">= {max_no_trade:.0%}")}

        # Rule 2 — no directional conviction either way.
        directional = max(pc, pp)
        if directional < min_directional:
            return {"allow": False, "direction": head_dir,
                    "reason": (f"oracle directional {directional:.0%} "
                               f"< {min_directional:.0%}")}

        # Rule 3 — head disagrees with the intended side.
        intended = _norm_side(payload.get("intended_side")
                              or payload.get("direction")
                              or payload.get("side"))
        if intended is not None:
            denom = pc + pp
            if denom > 0.0:
                agree = (pc if intended == "call" else pp) / denom
                if agree < min_agreement:
                    return {"allow": False, "direction": head_dir,
                            "reason": (f"oracle {head_dir or 'neutral'} vs "
                                       f"intended {intended}: agreement "
                                       f"{agree:.0%} < {min_agreement:.0%}")}

        return {"allow": True, "direction": head_dir, "reason": "oracle allows"}
    except Exception:  # pragma: no cover - fail-open
        return {"allow": True, "reason": "oracle gate error (fail-open)",
                "direction": None}


# --------------------------------------------------------------------------- #
# Self-test (offline)
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    # Empty / malformed -> allow.
    assert evaluate_oracle_gate({})["allow"] is True
    assert evaluate_oracle_gate(None)["allow"] is True
    assert evaluate_oracle_gate({"probability": {}})["allow"] is True

    # Strong aligned call head -> allow.
    strong_call = {"probability": {"p_call": 0.7, "p_put": 0.1,
                                  "p_no_trade": 0.2},
                   "intended_side": "call"}
    r = evaluate_oracle_gate(strong_call)
    assert r["allow"] is True and r["direction"] == "call"

    # Rule 1 — no-trade too high.
    r = evaluate_oracle_gate({"probability": {"p_call": 0.05, "p_put": 0.05,
                                             "p_no_trade": 0.9},
                              "intended_side": "call"})
    assert r["allow"] is False and "no-trade" in r["reason"]

    # Rule 2 — no directional conviction (both sides tiny).
    r = evaluate_oracle_gate({"probability": {"p_call": 0.1, "p_put": 0.08,
                                             "p_no_trade": 0.82},
                              "intended_side": "call"})
    assert r["allow"] is False and "directional" in r["reason"]

    # Rule 3 — head leans put but we intend a call -> disagreement veto.
    r = evaluate_oracle_gate({"probability": {"p_call": 0.15, "p_put": 0.65,
                                             "p_no_trade": 0.2},
                              "intended_side": "call"})
    assert r["allow"] is False and "agreement" in r["reason"]
    assert r["direction"] == "put"

    # Same head, but we intended the put it favours -> allow.
    r = evaluate_oracle_gate({"probability": {"p_call": 0.15, "p_put": 0.65,
                                             "p_no_trade": 0.2},
                              "intended_side": "put"})
    assert r["allow"] is True

    # No intended side supplied -> rules 1&2 still apply, rule 3 skipped.
    r = evaluate_oracle_gate({"probability": {"p_call": 0.15, "p_put": 0.65,
                                             "p_no_trade": 0.2}})
    assert r["allow"] is True and r["direction"] == "put"

    # Config override tightens the no-trade ceiling to veto a mild head.
    mild = {"probability": {"p_call": 0.35, "p_put": 0.15, "p_no_trade": 0.5},
            "intended_side": "call"}
    assert evaluate_oracle_gate(mild)["allow"] is True
    assert evaluate_oracle_gate(
        mild, {"ORACLE_MAX_NO_TRADE": 0.4})["allow"] is False

    # Top-level probabilities (no "probability" wrapper) also work.
    r = evaluate_oracle_gate({"p_call": 0.6, "p_put": 0.2, "p_no_trade": 0.2,
                              "side": "call"})
    assert r["allow"] is True
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("oracle_trade_gate self-test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
