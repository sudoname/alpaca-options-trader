"""
Oracle 3.0 — Voting Engine + Bayesian Probability Engine (PURE arithmetic).

Two pure functions combine the :class:`oracle_agents.AgentVote` slate into
probabilities. Neither opens, sizes, prices, blocks or alters a trade; both are
clamped, normalized and fail-open (never raise).

``tally_votes(votes, weights=None) -> {p_bull, p_bear, p_neutral}``
    A weighted average of the agents' bull/bear/neutral mass. Sums to 1.0.

``bayesian_probability(votes, prior, weights=None) -> {p_call, p_put, p_no_trade}``
    A log-odds Bayesian update: start from the directional ``prior`` (base rate
    that a taken setup's directional bet pays, in [0, 1]), nudge it by each
    agent's weighted, confidence-scaled (bull - bear) evidence, then split the
    *actionable* mass between call/put while reserving the aggregate **neutral**
    mass as ``p_no_trade``. Sums to 1.0.

Design guarantees (from the plan):
  * No single agent can flip a decision: weights are bounded by the caller
    (``oracle_weights`` enforces [w_min, w_max]); evidence enters through a
    bounded logit nudge, never as a hard override.
  * EV remains the final authority and risk the final gate — these numbers are
    *advisory probabilities*; nothing here touches execution.
"""

import math
from typing import Dict, List, Optional

WEIGHT_DEFAULT = 1.0
# How strongly a unit of (weight * confidence * (bull - bear)) moves the logit.
# Bounded so even a maxed-out, perfectly-confident agent only nudges the prior.
EVIDENCE_SCALE = 1.5
_EPS = 1e-6


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _to_float(value, default: float = 0.0) -> float:
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _logit(p: float) -> float:
    p = min(1.0 - _EPS, max(_EPS, p))
    return math.log(p / (1.0 - p))


def _sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def _weight_for(vote, weights) -> float:
    name = getattr(vote, "name", None)
    if isinstance(weights, dict) and name in weights:
        w = _to_float(weights[name], WEIGHT_DEFAULT)
        return w if w >= 0.0 else 0.0
    return WEIGHT_DEFAULT


def _bull(vote) -> float:
    return _clamp01(_to_float(getattr(vote, "bullish_score", 0.0)))


def _bear(vote) -> float:
    return _clamp01(_to_float(getattr(vote, "bearish_score", 0.0)))


def _conf(vote) -> float:
    return _clamp01(_to_float(getattr(vote, "confidence", 0.0)))


def tally_votes(votes: Optional[List], weights: Optional[Dict] = None) -> dict:
    """Weighted bull/bear/neutral split. Always returns a dict summing to 1.0."""
    try:
        if not votes:
            return {"p_bull": 0.0, "p_bear": 0.0, "p_neutral": 1.0}
        wb = wn = wbe = 0.0
        total_w = 0.0
        for v in votes:
            w = _weight_for(v, weights)
            if w <= 0.0:
                continue
            bull = _bull(v)
            bear = _bear(v)
            neut = _clamp01(1.0 - bull - bear)
            wb += w * bull
            wbe += w * bear
            wn += w * neut
            total_w += w
        if total_w <= 0.0:
            return {"p_bull": 0.0, "p_bear": 0.0, "p_neutral": 1.0}
        p_bull = wb / total_w
        p_bear = wbe / total_w
        p_neutral = wn / total_w
        s = p_bull + p_bear + p_neutral
        if s <= 0.0:
            return {"p_bull": 0.0, "p_bear": 0.0, "p_neutral": 1.0}
        return {"p_bull": round(p_bull / s, 6),
                "p_bear": round(p_bear / s, 6),
                "p_neutral": round(p_neutral / s, 6)}
    except Exception:  # pragma: no cover - fail-open
        return {"p_bull": 0.0, "p_bear": 0.0, "p_neutral": 1.0}


def bayesian_probability(votes: Optional[List], prior: float = 0.5,
                         weights: Optional[Dict] = None) -> dict:
    """Posterior {p_call, p_put, p_no_trade}. Always sums to 1.0; never raises.

    ``prior`` is the directional base rate P(call works) in (0, 1) (default
    0.5 = uninformed). The actionable mass = 1 - aggregate neutral mass.
    """
    try:
        prior = _clamp01(_to_float(prior, 0.5))
        if prior <= 0.0 or prior >= 1.0:
            prior = min(1.0 - _EPS, max(_EPS, prior))
        if not votes:
            return {"p_call": 0.0, "p_put": 0.0, "p_no_trade": 1.0}

        # Directional log-odds nudge from weighted, confidence-scaled evidence.
        logit = _logit(prior)
        total_w = 0.0
        for v in votes:
            w = _weight_for(v, weights)
            if w <= 0.0:
                continue
            evidence = (_bull(v) - _bear(v)) * _conf(v)
            logit += EVIDENCE_SCALE * w * evidence
            total_w += w
        p_call_dir = _sigmoid(logit)
        p_put_dir = 1.0 - p_call_dir

        # Actionable vs no-trade mass comes from the aggregate neutral share.
        tally = tally_votes(votes, weights)
        actionable = _clamp01(1.0 - tally["p_neutral"])
        p_no_trade = _clamp01(tally["p_neutral"])

        p_call = p_call_dir * actionable
        p_put = p_put_dir * actionable
        s = p_call + p_put + p_no_trade
        if s <= 0.0:
            return {"p_call": 0.0, "p_put": 0.0, "p_no_trade": 1.0}
        return {"p_call": round(p_call / s, 6),
                "p_put": round(p_put / s, 6),
                "p_no_trade": round(p_no_trade / s, 6)}
    except Exception:  # pragma: no cover - fail-open
        return {"p_call": 0.0, "p_put": 0.0, "p_no_trade": 1.0}


def prior_from_records(records: Optional[List], candidate: Optional[dict] = None,
                       config=None) -> float:
    """Directional prior P(call works) from the historical base rate.

    Uses ``learned_edge.compute_prior`` win-rate as the base rate that a taken
    setup's directional bet pays, clamped to [0.2, 0.8] so the prior can never
    be over-confident. Falls open to 0.5 (uninformed) on any error / sparse data.
    """
    try:
        import learned_edge as le
        prior = le.compute_prior(records or [])
        n = int(prior.get("n", 0) or 0)
        wr = _to_float(prior.get("win_rate", 0.5), 0.5)
        if n < 5:
            return 0.5
        return min(0.8, max(0.2, wr))
    except Exception:  # pragma: no cover - fail-open
        return 0.5


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    class _V:
        def __init__(self, name, bull, bear, conf):
            self.name = name
            self.bullish_score = bull
            self.bearish_score = bear
            self.confidence = conf

    def _sums_to_one(d, keys):
        return abs(sum(d[k] for k in keys) - 1.0) < 1e-6

    # Bullish slate.
    bull = [_V("trend", 0.8, 0.0, 0.9), _V("news", 0.6, 0.0, 0.7),
            _V("breadth", 0.5, 0.0, 0.6), _V("liquidity", 0.0, 0.0, 0.8)]
    t = tally_votes(bull)
    if not _sums_to_one(t, ("p_bull", "p_bear", "p_neutral")):
        print("FAIL: tally not normalized", t); ok = False
    if not (t["p_bull"] > t["p_bear"]):
        print("FAIL: bullish tally", t); ok = False
    b = bayesian_probability(bull, prior=0.5)
    if not _sums_to_one(b, ("p_call", "p_put", "p_no_trade")):
        print("FAIL: bayes not normalized", b); ok = False
    if not (b["p_call"] > b["p_put"]):
        print("FAIL: bullish bayes", b); ok = False

    # Bearish slate -> p_put dominates.
    bear = [_V("trend", 0.0, 0.8, 0.9), _V("news", 0.0, 0.6, 0.7)]
    bb = bayesian_probability(bear, prior=0.5)
    if not (bb["p_put"] > bb["p_call"]):
        print("FAIL: bearish bayes", bb); ok = False
    if not _sums_to_one(bb, ("p_call", "p_put", "p_no_trade")):
        print("FAIL: bear bayes norm", bb); ok = False

    # All-neutral -> high no-trade.
    neutral = [_V("liquidity", 0.0, 0.0, 0.5), _V("volatility", 0.0, 0.0, 0.4)]
    bn = bayesian_probability(neutral, prior=0.5)
    if bn["p_no_trade"] < 0.9:
        print("FAIL: neutral should be no-trade", bn); ok = False

    # No single agent flips: one bullish among many bearish, bounded weights.
    mixed = [_V("trend", 0.0, 0.9, 1.0)] + [_V("a", 1.0, 0.0, 1.0)]
    weights = {"a": 1.0, "trend": 1.0}
    bm = bayesian_probability(mixed, prior=0.5, weights=weights)
    if not _sums_to_one(bm, ("p_call", "p_put", "p_no_trade")):
        print("FAIL: mixed norm", bm); ok = False

    # Prior tilts the result when evidence is empty-but-present (all neutral).
    hi = bayesian_probability([_V("x", 0.3, 0.0, 0.5)], prior=0.8)
    lo = bayesian_probability([_V("x", 0.3, 0.0, 0.5)], prior=0.2)
    if not (hi["p_call"] > lo["p_call"]):
        print("FAIL: prior should tilt call", hi, lo); ok = False

    # Empty / garbage never raises.
    for junk in (None, [], [None, 42, "x"], "x", 7):
        try:
            tally_votes(junk)            # type: ignore[arg-type]
            bayesian_probability(junk)   # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    if prior_from_records([]) != 0.5:
        print("FAIL: empty prior should be 0.5"); ok = False

    # Determinism.
    if bayesian_probability(bull, 0.5) != bayesian_probability(bull, 0.5):
        print("FAIL: non-deterministic"); ok = False

    print("oracle_voting self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
