"""
P13B — Oracle Score v2 blending (pure arithmetic, no I/O, never raises).

Oracle Score v1 (``spread_builder.compute_oracle_score``) blends five first-
principles sub-scores:

    0.25 vol_edge + 0.20 liquidity + 0.25 risk_reward + 0.15 cost + 0.15 trend_align

Oracle Score v2 adds a sixth input — the LEARNED EDGE (the Bayesian-smoothed
historical edge of the trade's setup, from ``learned_edge``) — and re-weights:

    0.20 vol_edge + 0.15 liquidity + 0.15 risk_reward + 0.10 cost
    + 0.10 trend_align + 0.30 learned_edge

This module is intentionally trivial and dependency-free: it only does the
weighted blend. The learned-edge value is computed by the CALLER and passed in,
so v2 stays pure arithmetic and trivially testable.

IMPORTANT: v2 changes NOTHING by default. ``ORACLE_SCORE_VERSION`` defaults to
"v1"; the live ranker keeps calling ``compute_oracle_score`` with no version arg.
v2 is exercised only by offline reports, the shadow replay, and tests.
"""

from typing import Dict, Optional

from config_loader import ConfigLoader

V1 = "v1"
V2 = "v2"
VALID_VERSIONS = (V1, V2)

# Sub-score weights. v1 mirrors spread_builder.compute_oracle_score exactly.
V1_WEIGHTS = {
    "vol_edge": 0.25, "liquidity": 0.20, "risk_reward": 0.25,
    "cost": 0.15, "trend_align": 0.15,
}
V2_WEIGHTS = {
    "vol_edge": 0.20, "liquidity": 0.15, "risk_reward": 0.15,
    "cost": 0.10, "trend_align": 0.10, "learned_edge": 0.30,
}


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def blend_v2(subscores: Dict[str, float], learned_edge_score: float) -> float:
    """Blend the five v1 sub-scores plus the learned edge into a 0-100 score.

    ``subscores`` carries vol_edge/liquidity/risk_reward/cost/trend_align in
    [0, 1] (missing keys treated as 0.0). ``learned_edge_score`` is the learned
    edge in [0, 1]; a neutral setup is 0.5. Result is clamped to [0, 100] and
    rounded to 1 dp, matching the v1 output shape. Never raises.
    """
    try:
        le = _clamp01(float(learned_edge_score))
        blended = (
            V2_WEIGHTS["vol_edge"] * float(subscores.get("vol_edge", 0.0))
            + V2_WEIGHTS["liquidity"] * float(subscores.get("liquidity", 0.0))
            + V2_WEIGHTS["risk_reward"] * float(subscores.get("risk_reward", 0.0))
            + V2_WEIGHTS["cost"] * float(subscores.get("cost", 0.0))
            + V2_WEIGHTS["trend_align"] * float(subscores.get("trend_align", 0.0))
            + V2_WEIGHTS["learned_edge"] * le
        )
        return round(_clamp01(blended) * 100.0, 1)
    except Exception:  # pragma: no cover - defensive; callers also fail open
        return 0.0


def blend_v1(subscores: Dict[str, float]) -> float:
    """Reference v1 blend (for the offline comparison report). Identical math to
    ``spread_builder.compute_oracle_score``'s blend; never raises."""
    try:
        blended = (
            V1_WEIGHTS["vol_edge"] * float(subscores.get("vol_edge", 0.0))
            + V1_WEIGHTS["liquidity"] * float(subscores.get("liquidity", 0.0))
            + V1_WEIGHTS["risk_reward"] * float(subscores.get("risk_reward", 0.0))
            + V1_WEIGHTS["cost"] * float(subscores.get("cost", 0.0))
            + V1_WEIGHTS["trend_align"] * float(subscores.get("trend_align", 0.0))
        )
        return round(_clamp01(blended) * 100.0, 1)
    except Exception:  # pragma: no cover
        return 0.0


def score_version_from_env(loader: Optional[ConfigLoader] = None,
                           path: str = ".env") -> str:
    """Read ORACLE_SCORE_VERSION (default 'v1'). Anything unrecognized -> v1.

    Fail-open: any config error returns v1 so live behavior never breaks.
    """
    try:
        cfg = loader if loader is not None else ConfigLoader(path=path)
        v = str(cfg.get_str("ORACLE_SCORE_VERSION", V1)).strip().lower()
        return v if v in VALID_VERSIONS else V1
    except Exception:
        return V1


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Weights sum to 1.0.
    if round(sum(V1_WEIGHTS.values()), 6) != 1.0:
        print("FAIL: v1 weights do not sum to 1.0", V1_WEIGHTS); ok = False
    if round(sum(V2_WEIGHTS.values()), 6) != 1.0:
        print("FAIL: v2 weights do not sum to 1.0", V2_WEIGHTS); ok = False

    subs = {"vol_edge": 1.0, "liquidity": 1.0, "risk_reward": 1.0,
            "cost": 1.0, "trend_align": 1.0}
    # All sub-scores perfect, neutral edge 0.5 -> 0.70*100 + 0.30*0.5*100 = 85.0.
    got = blend_v2(subs, 0.5)
    if got != 85.0:
        print("FAIL: blend_v2 neutral edge", got); ok = False
    # Perfect sub-scores + perfect edge -> 100.
    if blend_v2(subs, 1.0) != 100.0:
        print("FAIL: blend_v2 perfect", blend_v2(subs, 1.0)); ok = False
    # v1 reference of perfect sub-scores -> 100.
    if blend_v1(subs) != 100.0:
        print("FAIL: blend_v1 perfect", blend_v1(subs)); ok = False

    # Manual arithmetic check with partial sub-scores.
    subs2 = {"vol_edge": 0.8, "liquidity": 0.6, "risk_reward": 0.5,
             "cost": 0.4, "trend_align": 0.2}
    manual = (0.20*0.8 + 0.15*0.6 + 0.15*0.5 + 0.10*0.4 + 0.10*0.2
              + 0.30*0.7) * 100.0
    if blend_v2(subs2, 0.7) != round(manual, 1):
        print("FAIL: blend_v2 manual", blend_v2(subs2, 0.7), round(manual, 1))
        ok = False

    # Clamping + fail-open on garbage.
    if blend_v2(subs, 5.0) != 100.0:
        print("FAIL: edge clamp high", blend_v2(subs, 5.0)); ok = False
    if blend_v2(subs, -5.0) != 70.0:    # edge clamps to 0 -> 0.70*100
        print("FAIL: edge clamp low", blend_v2(subs, -5.0)); ok = False
    if blend_v2({}, None) != 0.0:       # type: ignore[arg-type]
        print("FAIL: garbage should fail open to 0.0"); ok = False

    # Version parsing fail-open via a stub loader.
    class _Loader:
        def __init__(self, v):
            self._v = v

        def get_str(self, name, default=""):
            return self._v

    if score_version_from_env(_Loader("v2")) != V2:
        print("FAIL: parse v2"); ok = False
    if score_version_from_env(_Loader("V1")) != V1:
        print("FAIL: parse v1 upper"); ok = False
    if score_version_from_env(_Loader("garbage")) != V1:
        print("FAIL: unknown -> v1"); ok = False

    print("oracle_score_v2 self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
