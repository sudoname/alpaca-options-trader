"""
RL Environment definitions for the options-trading Q-learning layer.

Defines the action space, state featurization/discretization, valid-action
masking, and the reward function shared by the agent, the advisory wrapper,
and the offline trainer.

The state is intentionally discretized into a small number of buckets so a
tabular Q-table stays meaningful with very little data.
"""

from typing import Dict, List, Optional


# ---------------------------------------------------------------------------
# Action space
# ---------------------------------------------------------------------------
SKIP = "SKIP"
CALL = "CALL"
PUT = "PUT"
ACTIONS: List[str] = [SKIP, CALL, PUT]


# ---------------------------------------------------------------------------
# Discretization bins
#
# Each entry maps a feature name to an ordered list of (label, upper_bound)
# edges. The first edge whose upper_bound the value is < gets the label.
# The final label is used for everything at/above the last numeric edge.
# ---------------------------------------------------------------------------
BINS = {
    # SPY (or underlying) % change on the day
    "change": [
        ("dn_strong", -0.3),
        ("dn_mild", -0.1),
        ("flat", 0.1),
        ("up_mild", 0.3),
        ("up_strong", float("inf")),
    ],
    # Absolute VIX level
    "vix_level": [
        ("calm", 15.0),
        ("normal", 20.0),
        ("elevated", 25.0),
        ("high", 30.0),
        ("extreme", float("inf")),
    ],
    # VIX % change
    "vix_change": [
        ("falling", -5.0),
        ("steady", 5.0),
        ("rising", float("inf")),
    ],
    # Overnight gap %
    "gap": [
        ("gap_dn", -0.3),
        ("gap_flat", 0.3),
        ("gap_up", float("inf")),
    ],
    # Position within the intraday range [0..1]
    "intraday_position": [
        ("near_low", 0.34),
        ("mid", 0.67),
        ("near_high", float("inf")),
    ],
    # Rule-based confidence %
    "confidence": [
        ("lo", 60.0),
        ("med", 75.0),
        ("hi", float("inf")),
    ],
}


def _bucket(name: str, value) -> str:
    """Return the discrete bucket label for a numeric feature value."""
    edges = BINS.get(name)
    if edges is None:
        return str(value)
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "na"
    for label, upper in edges:
        if v < upper:
            return label
    return edges[-1][0]


def extract_features(
    analysis: Dict,
    pdt_remaining: Optional[int] = None,
    day_of_week: Optional[int] = None,
    strat_name: str = "generic",
) -> Dict[str, str]:
    """
    Build a discretized feature dict from a strategy's `analysis` output.

    `analysis` is the dict returned by the strategies' analyze_* methods. It is
    accessed defensively so partial dicts (or smart_trader's different schema)
    still produce a usable state.
    """
    # Underlying change: strategies use 'spy_change'/'change'; smart_trader uses 'momentum'.
    change = analysis.get("spy_change")
    if change is None:
        change = analysis.get("change")
    if change is None:
        # momentum is a fraction (~ -0.05..0.05); express as a % to match bins
        mom = analysis.get("momentum")
        change = (mom * 100.0) if mom is not None else 0.0

    intraday = analysis.get("intraday_position")
    if intraday is None:
        intraday = 0.5  # neutral when unknown

    features = {
        "strat": strat_name,
        "change": _bucket("change", change),
        "vix_level": _bucket("vix_level", analysis.get("vix_level", 15.0)),
        "vix_change": _bucket("vix_change", analysis.get("vix_change", 0.0)),
        "gap": _bucket("gap", analysis.get("gap", 0.0)),
        "intraday_position": _bucket("intraday_position", intraday),
        "confidence": _bucket("confidence", analysis.get("confidence", 0.0)),
    }

    if pdt_remaining is not None:
        try:
            features["pdt"] = str(max(0, min(3, int(pdt_remaining))))
        except (TypeError, ValueError):
            features["pdt"] = "na"
    else:
        features["pdt"] = "na"

    if day_of_week is not None:
        try:
            features["dow"] = str(int(day_of_week))
        except (TypeError, ValueError):
            features["dow"] = "na"
    else:
        features["dow"] = "na"

    return features


# Stable ordering so the same features always produce the same key.
_KEY_ORDER = [
    "strat",
    "change",
    "vix_level",
    "vix_change",
    "gap",
    "intraday_position",
    "confidence",
    "pdt",
    "dow",
]


def state_key(features: Dict[str, str]) -> str:
    """Deterministic string key for a discretized state."""
    return "|".join(f"{k}={features.get(k, 'na')}" for k in _KEY_ORDER)


def valid_actions(analysis: Dict, allow_override: bool = False) -> List[str]:
    """
    Which actions the agent may consider for a given context.

    In shadow mode we only compare SKIP vs the rule's chosen direction, since we
    can only ever observe the outcome of the direction the rules actually took.
    With allow_override=True (future active mode) the full action set is exposed.
    """
    if allow_override:
        return list(ACTIONS)

    direction = (analysis.get("direction") or "").upper()
    if direction in (CALL, PUT):
        return [SKIP, direction]
    return [SKIP]


def compute_reward(
    pnl_pct: Optional[float],
    action: str,
    pdt_remaining_before: Optional[int] = None,
    took_day_trade: bool = False,
) -> float:
    """
    Reward for a completed decision.

    - TRADE actions (CALL/PUT): scaled realized P/L, reward = pnl_pct / 100.
    - SKIP: baseline 0.0 (no counterfactual P/L is observable).
    - Shaping: extra penalty if a losing trade consumed the last day-trade slot,
      to discourage burning scarce PDT budget on poor setups.
    """
    if action == SKIP or pnl_pct is None:
        return 0.0

    reward = float(pnl_pct) / 100.0

    if took_day_trade and pdt_remaining_before is not None:
        try:
            if int(pdt_remaining_before) <= 1 and reward < 0:
                reward -= 0.10  # penalize wasting the last day trade on a loss
        except (TypeError, ValueError):
            pass

    return reward
