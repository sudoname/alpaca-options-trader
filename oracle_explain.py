"""
Oracle 3.0 — Explainability Engine (PURE, attribution-only).

Turns a slate of :class:`oracle_agents.AgentVote` plus the voting weights and the
final probability into a human-readable, fully-attributable explanation::

    explain(votes, weights=None, probability=None, regime=None) -> {
        agent_contributions: {name: share},   # >= 0, sums to 1.0
        top_reasons: [str],                    # from the biggest contributors
        regime: <label or None>,
        probability: <dict or None>,           # echo of p_call/p_put/p_no_trade
        summary_str: str,                      # one-line plain-English summary
    }

Each agent's *contribution* is its weighted, confidence-scaled directional pull
``weight * confidence * |bullish - bearish|``, normalized so the shares sum to
1.0. A purely neutral / zero-confidence agent contributes ~0. When nothing pulls
directionally we fall back to weighted confidence, then to uniform, so the shares
always sum to 1.0 and the explanation is never empty.

This module is ANALYTICS / SHADOW ONLY: it never opens, sizes, prices, blocks or
alters a trade, and every public function fails open (never raises).
"""

from typing import Dict, List, Optional

WEIGHT_DEFAULT = 1.0
TOP_REASONS = 4


def _to_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        return default if f != f else f  # NaN guard
    except (TypeError, ValueError):
        return default


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _weight_for(name, weights) -> float:
    if isinstance(weights, dict) and name in weights:
        w = _to_float(weights[name], WEIGHT_DEFAULT)
        return w if w >= 0.0 else 0.0
    return WEIGHT_DEFAULT


def _vote_fields(v):
    name = getattr(v, "name", None) or (v.get("name") if isinstance(v, dict)
                                        else None)
    if name is None:
        return None
    g = (lambda k: getattr(v, k, None)) if not isinstance(v, dict) else v.get
    bull = _clamp01(_to_float(g("bullish_score")))
    bear = _clamp01(_to_float(g("bearish_score")))
    conf = _clamp01(_to_float(g("confidence")))
    reasons = g("reasons") or []
    if not isinstance(reasons, list):
        reasons = [str(reasons)]
    return name, bull, bear, conf, list(reasons)


def _normalize(raw: Dict[str, float]) -> Dict[str, float]:
    total = sum(raw.values())
    if total <= 0.0:
        n = len(raw)
        if n == 0:
            return {}
        share = 1.0 / n
        return {k: round(share, 6) for k in raw}
    return {k: round(v / total, 6) for k, v in raw.items()}


def explain(votes: Optional[List], weights: Optional[Dict] = None,
            probability: Optional[dict] = None,
            regime=None) -> dict:
    """Attribute the decision to its agents. Never raises."""
    try:
        parsed = []
        for v in (votes or []):
            pf = _vote_fields(v)
            if pf is not None:
                parsed.append(pf)

        if not parsed:
            return {"agent_contributions": {}, "top_reasons": [],
                    "regime": _regime_label(regime), "probability": probability,
                    "summary_str": "No agent evidence available."}

        # Primary attribution: weighted directional pull.
        directional: Dict[str, float] = {}
        confidence_only: Dict[str, float] = {}
        reasons_by_agent: Dict[str, List[str]] = {}
        signed: Dict[str, float] = {}
        for name, bull, bear, conf, reasons in parsed:
            w = _weight_for(name, weights)
            directional[name] = w * conf * abs(bull - bear)
            confidence_only[name] = w * conf
            reasons_by_agent[name] = reasons
            signed[name] = bull - bear

        if sum(directional.values()) > 0.0:
            contributions = _normalize(directional)
        elif sum(confidence_only.values()) > 0.0:
            contributions = _normalize(confidence_only)
        else:
            contributions = _normalize({k: 1.0 for k in directional})

        # Top reasons come from the biggest contributors that actually pulled.
        ordered = sorted(contributions.items(), key=lambda kv: kv[1],
                         reverse=True)
        top_reasons: List[str] = []
        for name, share in ordered:
            if share <= 0.0:
                continue
            for r in reasons_by_agent.get(name, []):
                top_reasons.append(f"{name}: {r}")
                if len(top_reasons) >= TOP_REASONS:
                    break
            if len(top_reasons) >= TOP_REASONS:
                break

        summary = _summarize(ordered, signed, probability,
                             _regime_label(regime))
        return {"agent_contributions": contributions,
                "top_reasons": top_reasons,
                "regime": _regime_label(regime),
                "probability": probability,
                "summary_str": summary}
    except Exception:  # pragma: no cover - fail-open
        return {"agent_contributions": {}, "top_reasons": [],
                "regime": _regime_label(regime), "probability": probability,
                "summary_str": "Explanation unavailable (error)."}


def _regime_label(regime):
    if isinstance(regime, dict):
        return regime.get("label")
    return regime


def _summarize(ordered, signed, probability, regime_label) -> str:
    parts = []
    if regime_label:
        parts.append(f"Regime {regime_label}")
    if isinstance(probability, dict):
        pc = probability.get("p_call")
        pp = probability.get("p_put")
        pn = probability.get("p_no_trade")
        if pc is not None and pp is not None and pn is not None:
            parts.append(f"P(call)={pc:.0%} P(put)={pp:.0%} "
                         f"P(no-trade)={pn:.0%}")
    if ordered:
        lead_name, lead_share = ordered[0]
        lean = signed.get(lead_name, 0.0)
        direction = ("bullish" if lean > 0 else
                     "bearish" if lean < 0 else "neutral")
        parts.append(f"led by {lead_name} ({lead_share:.0%}, {direction})")
    return "; ".join(parts) if parts else "No directional consensus."


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    class _V:
        def __init__(self, name, bull, bear, conf, reasons):
            self.name = name
            self.bullish_score = bull
            self.bearish_score = bear
            self.confidence = conf
            self.reasons = reasons

    votes = [
        _V("trend", 0.8, 0.0, 0.9, ["up trend, momentum +0.080"]),
        _V("news", 0.6, 0.0, 0.7, ["news/sentiment +0.50"]),
        _V("liquidity", 0.0, 0.0, 0.8, ["spread 1.0%"]),
        _V("volatility", 0.0, 0.0, 0.4, ["calm vol 0.18"]),
    ]
    prob = {"p_call": 0.71, "p_put": 0.12, "p_no_trade": 0.17}
    out = explain(votes, weights=None, probability=prob,
                  regime={"label": "TRENDING_BULL"})

    contrib = out["agent_contributions"]
    if abs(sum(contrib.values()) - 1.0) > 1e-6:
        print("FAIL: contributions not normalized", contrib); ok = False
    # Trend (strong + confident) should out-contribute news.
    if not (contrib.get("trend", 0) > contrib.get("news", 0)):
        print("FAIL: trend should lead", contrib); ok = False
    # Purely neutral agents contribute ~0 in directional attribution.
    if contrib.get("liquidity", 0) > contrib.get("trend", 0):
        print("FAIL: neutral agent over-contributes", contrib); ok = False
    if not out["top_reasons"]:
        print("FAIL: no top reasons"); ok = False
    if out["regime"] != "TRENDING_BULL":
        print("FAIL: regime passthrough", out["regime"]); ok = False
    if "P(call)" not in out["summary_str"]:
        print("FAIL: summary missing probability", out["summary_str"]); ok = False

    # All-neutral slate -> uniform-ish, still sums to 1.0.
    neutral = [_V("a", 0.0, 0.0, 0.0, []), _V("b", 0.0, 0.0, 0.0, [])]
    no = explain(neutral)
    if abs(sum(no["agent_contributions"].values()) - 1.0) > 1e-6:
        print("FAIL: neutral contributions not normalized", no); ok = False

    # Empty.
    eo = explain([])
    if eo["agent_contributions"] != {} or eo["top_reasons"] != []:
        print("FAIL: empty not handled", eo); ok = False

    # Weights change attribution.
    w = explain(votes, weights={"news": 5.0, "trend": 0.25})
    if not (w["agent_contributions"]["news"]
            > w["agent_contributions"]["trend"]):
        print("FAIL: weights should reweight", w["agent_contributions"]); ok = False

    # Dict-form votes accepted.
    dv = [{"name": "trend", "bullish_score": 0.7, "bearish_score": 0.0,
           "confidence": 0.8, "reasons": ["r"]}]
    if not explain(dv)["agent_contributions"]:
        print("FAIL: dict votes"); ok = False

    # Determinism.
    if explain(votes, probability=prob) != explain(votes, probability=prob):
        print("FAIL: non-deterministic"); ok = False

    # Garbage never raises.
    for junk in (None, 42, "x", [None, 42], {"weird": object()}):
        try:
            explain(junk)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("oracle_explain self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
