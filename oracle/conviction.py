"""
Oracle 2.1 — Conviction engine (ANALYTICS ONLY, pure, no I/O).

*Conviction* is a single normalized [0, 1] answer to "how strongly does the WHOLE
evidence slate agree with the trade direction we already chose?" It FOLDS the
orphaned per-slice readings (the rule engine's own confidence, the machine
thesis, the agent consensus, a catalyst, a candlestick, the regime, and the
chase/pullback diagnostics) into one number a sizer can consume.

Direction is NOT decided here — it is an INPUT (an OUTPUT of the first-principles
rule engine). This module only measures agreement with that given direction. It
is a FEATURE, never a standalone trigger:

  * SHADOW (default): stamped into ``features_json.evidence`` as ``conviction``
    for leaderboard analysis. Fed the FULL evidence slate.
  * ACTIVE (flag-gated upstream, default OFF): the caller may size the position
    off ``contracts`` instead of the crude 3-tier signal-count step function.
    Fed the SUBSET available before the order (rule confidence + chase/pullback).
    It never flips direction and never opens/prices a trade.

Fold (each present component -> an "aligned score" in [0, 1], 0.5 == neutral):
  * base               magnitude   — the rule engine's own confidence, normalized
  * thesis_conviction  magnitude   — the machine thesis conviction
  * agent_consensus    directional — mean signed agent contribution in [-1, 1]
  * catalyst           directional — severity, signed by its direction_hint
  * candlestick        directional — confidence, signed by its bias
  * regime_confidence  magnitude   — how cleanly the regime is classified
  blend = weighted mean over the PRESENT components (weights renormalize).
  Then two multiplicative modifiers from the same-horizon move diagnostics:
  * extended (aligned chase)  -> blend *= (1 - EXTENSION_PENALTY)   [discount]
  * opportunity (pullback)    -> blend *= (1 + REPRICING_BONUS)     [premium]
  conviction = clamp01(blend).

When NO component is present the fold is undefined -> conviction None,
contracts None; the caller must fall back to its legacy sizing (fail-open).

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass, field
from typing import Optional

# Blend weights per component (only the PRESENT ones are used; they renormalize
# so a sparse slate is still a valid weighted mean).
W_BASE = 0.35
W_THESIS = 0.20
W_AGENT = 0.20
W_CATALYST = 0.10
W_CANDLE = 0.10
W_REGIME = 0.05

# Multiplicative modifiers from the chase / pullback diagnostics.
EXTENSION_PENALTY = 0.15   # an aligned chase discounts conviction 15%
REPRICING_BONUS = 0.10     # a healthy pullback lifts conviction 10%

# Conviction -> contract-count ladder (inclusive lower bounds).
VERY_HIGH_AT = 0.75        # >= -> 3 contracts
HIGH_AT = 0.50             # >= -> 2 contracts
# below HIGH_AT and at/above SKIP_BELOW -> 1 contract; below SKIP_BELOW -> skip.
SKIP_BELOW = 0.0           # default 0.0 -> never skip (floor stays at 1)

TIER_VERY_HIGH = "very_high"
TIER_HIGH = "high"
TIER_REGULAR = "regular"
TIER_SKIP = "skip"


def _to_float(value) -> Optional[float]:
    """Coerce to float; bools and junk -> None (never raises)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def _dir_sign(direction) -> Optional[int]:
    """+1 for a bullish trade, -1 for bearish, None if unknown."""
    if not isinstance(direction, str):
        return None
    d = direction.strip().lower()
    if d in ("call", "up", "bull", "bullish", "long"):
        return 1
    if d in ("put", "down", "bear", "bearish", "short"):
        return -1
    return None


def _bias_sign(bias) -> Optional[int]:
    """+1 bullish bias, -1 bearish, None otherwise."""
    if not isinstance(bias, str):
        return None
    b = bias.strip().lower()
    if b in ("bull", "bullish", "up"):
        return 1
    if b in ("bear", "bearish", "down"):
        return -1
    return None


@dataclass
class ConvictionStamp:
    """One conviction reading. All fields analytics-only."""

    conviction: Optional[float]        # [0,1] folded agreement, None if no inputs
    tier: Optional[str]                # very_high / high / regular / skip / None
    contracts: Optional[int]           # suggested contract count, None if no inputs
    direction: Optional[str]           # trade direction as given (an INPUT)
    components_used: list = field(default_factory=list)  # which slices folded in
    reason: str = ""

    def to_dict(self) -> dict:
        return {
            "conviction": self.conviction,
            "tier": self.tier,
            "contracts": self.contracts,
            "direction": self.direction,
            "components_used": list(self.components_used),
            "reason": self.reason,
        }


def _none_stamp(direction, reason: str) -> dict:
    return ConvictionStamp(
        conviction=None, tier=None, contracts=None,
        direction=(direction if isinstance(direction, str) else None),
        components_used=[], reason=reason,
    ).to_dict()


def _tier_and_contracts(conviction: float,
                        skip_below: float = SKIP_BELOW) -> tuple:
    """Map a [0,1] conviction to (tier, contracts) via the fixed ladder."""
    if conviction < skip_below:
        return TIER_SKIP, 0
    if conviction >= VERY_HIGH_AT:
        return TIER_VERY_HIGH, 3
    if conviction >= HIGH_AT:
        return TIER_HIGH, 2
    return TIER_REGULAR, 1


def compute_conviction(components: Optional[dict], direction=None,
                       extension_penalty: float = EXTENSION_PENALTY,
                       repricing_bonus: float = REPRICING_BONUS,
                       skip_below: float = SKIP_BELOW) -> dict:
    """Fold a ``components`` dict into a conviction stamp. Never raises.

    ``components`` keys (all optional; missing -> that slice is skipped):
      * ``base``               magnitude [0,1] — rule-engine confidence, normalized
      * ``thesis_conviction``  magnitude [0,1]
      * ``agent_consensus``    directional [-1,1] — signed toward bullish
      * ``catalyst_severity``  magnitude [0,1]  (needs ``catalyst_dir`` to count)
      * ``catalyst_dir``       'call'/'put'/None — direction_hint of the catalyst
      * ``candlestick_confidence`` magnitude [0,1] (needs ``candlestick_bias``)
      * ``candlestick_bias``   'bullish'/'bearish'/None
      * ``regime_confidence``  magnitude [0,1]
      * ``extended``           bool — an aligned chase (discount)
      * ``opportunity``        bool — a healthy pullback (premium)

    ``direction`` is the already-chosen trade side (an INPUT). Directional
    components need it to know alignment; without it they are skipped. Returns a
    None-conviction stamp when no component contributes (fail-open).
    """
    if not isinstance(components, dict):
        return _none_stamp(direction, "no components")

    sign = _dir_sign(direction)

    # (weight, aligned_score in [0,1], label) for each present component.
    parts = []

    base = _to_float(components.get("base"))
    if base is not None:
        parts.append((W_BASE, _clamp01(base), "base"))

    thesis = _to_float(components.get("thesis_conviction"))
    if thesis is not None:
        parts.append((W_THESIS, _clamp01(thesis), "thesis"))

    # Directional: agent consensus signed toward the trade direction.
    consensus = _to_float(components.get("agent_consensus"))
    if consensus is not None and sign is not None:
        aligned = _clamp01((max(-1.0, min(1.0, consensus)) * sign + 1.0) / 2.0)
        parts.append((W_AGENT, aligned, "agent_consensus"))

    # Directional: catalyst severity signed by its direction_hint.
    cat_sev = _to_float(components.get("catalyst_severity"))
    cat_sign = _dir_sign(components.get("catalyst_dir"))
    if cat_sev is not None and cat_sign is not None and sign is not None:
        signed = (cat_sev if cat_sign == sign else -cat_sev)
        parts.append((W_CATALYST, _clamp01(0.5 + signed / 2.0), "catalyst"))

    # Directional: candlestick confidence signed by its bias.
    cs_conf = _to_float(components.get("candlestick_confidence"))
    cs_sign = _bias_sign(components.get("candlestick_bias"))
    if cs_conf is not None and cs_sign is not None and sign is not None:
        signed = (cs_conf if cs_sign == sign else -cs_conf)
        parts.append((W_CANDLE, _clamp01(0.5 + signed / 2.0), "candlestick"))

    regime = _to_float(components.get("regime_confidence"))
    if regime is not None:
        parts.append((W_REGIME, _clamp01(regime), "regime"))

    if not parts:
        return _none_stamp(direction, "no usable components")

    wsum = sum(w for w, _, _ in parts)
    blend = sum(w * s for w, s, _ in parts) / wsum
    used = [label for _, _, label in parts]

    # Multiplicative modifiers from the same-horizon move diagnostics.
    mods = []
    if bool(components.get("extended")):
        blend *= (1.0 - extension_penalty)
        mods.append(f"-{extension_penalty:.0%} chase")
    if bool(components.get("opportunity")):
        blend *= (1.0 + repricing_bonus)
        mods.append(f"+{repricing_bonus:.0%} pullback")

    conviction = round(_clamp01(blend), 6)
    tier, contracts = _tier_and_contracts(conviction, skip_below=skip_below)

    reason = f"folded {len(parts)} component(s): {'+'.join(used)}"
    if mods:
        reason += " | " + ", ".join(mods)
    reason += f" -> {conviction} ({tier})"

    return ConvictionStamp(
        conviction=conviction,
        tier=tier,
        contracts=contracts,
        direction=(direction if isinstance(direction, str) else None),
        components_used=used,
        reason=reason,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # A strong, one-sided bullish slate for a call -> high conviction, 3 lots.
    strong = compute_conviction({
        "base": 1.0, "thesis_conviction": 0.9, "agent_consensus": 0.8,
        "catalyst_severity": 0.8, "catalyst_dir": "call",
        "candlestick_confidence": 0.7, "candlestick_bias": "bullish",
        "regime_confidence": 0.8,
    }, direction="call")
    if strong["conviction"] is None or strong["conviction"] < VERY_HIGH_AT:
        print("FAIL: strong slate should be very-high", strong); ok = False
    if strong["contracts"] != 3 or strong["tier"] != TIER_VERY_HIGH:
        print("FAIL: strong slate -> 3 contracts", strong); ok = False

    # The SAME slate for a PUT (everything now opposed) -> low conviction.
    opposed = compute_conviction({
        "base": 1.0, "thesis_conviction": 0.9, "agent_consensus": 0.8,
        "catalyst_severity": 0.8, "catalyst_dir": "call",
        "candlestick_confidence": 0.7, "candlestick_bias": "bullish",
        "regime_confidence": 0.8,
    }, direction="put")
    if opposed["conviction"] is None or opposed["conviction"] >= strong["conviction"]:
        print("FAIL: opposed direction must lower conviction", opposed); ok = False

    # Direction agreement matters: a bullish consensus scores higher for a call
    # than the identical magnitude does for a put.
    if not (strong["conviction"] > opposed["conviction"]):
        print("FAIL: alignment ordering", strong, opposed); ok = False

    # Base-only fold: conviction == base, mapped straight onto the ladder.
    b = compute_conviction({"base": 0.5}, direction="call")
    if abs(b["conviction"] - 0.5) > 1e-9 or b["contracts"] != 2:
        print("FAIL: base-only fold", b); ok = False
    b2 = compute_conviction({"base": 0.25}, direction="call")
    if b2["contracts"] != 1 or b2["tier"] != TIER_REGULAR:
        print("FAIL: base-only regular tier", b2); ok = False

    # Chase discount lowers conviction; pullback premium raises it.
    plain = compute_conviction({"base": 0.8}, direction="call")
    chased = compute_conviction({"base": 0.8, "extended": True}, direction="call")
    lifted = compute_conviction({"base": 0.8, "opportunity": True}, direction="call")
    if not (chased["conviction"] < plain["conviction"] < lifted["conviction"]):
        print("FAIL: chase/pullback modifiers", chased, plain, lifted); ok = False
    if abs(chased["conviction"] - 0.8 * (1 - EXTENSION_PENALTY)) > 1e-6:
        print("FAIL: chase discount amount", chased); ok = False
    if abs(lifted["conviction"] - 0.8 * (1 + REPRICING_BONUS)) > 1e-6:
        print("FAIL: pullback premium amount", lifted); ok = False

    # Directional components are SKIPPED when direction is unknown (they can't
    # tell alignment), so an agent-only slate with no direction folds nothing.
    no_dir = compute_conviction({"agent_consensus": 0.9}, direction=None)
    if no_dir["conviction"] is not None or no_dir["contracts"] is not None:
        print("FAIL: no direction -> directional-only slate folds nothing", no_dir)
        ok = False

    # A magnitude-only slate still works without a direction.
    mag = compute_conviction({"base": 0.6}, direction=None)
    if mag["conviction"] is None or mag["contracts"] != 2:
        print("FAIL: magnitude-only without direction", mag); ok = False

    # Weight renormalization: base 1.0 + regime 0.0 -> weighted mean, not 0.5.
    renorm = compute_conviction({"base": 1.0, "regime_confidence": 0.0},
                                direction="call")
    expect = (W_BASE * 1.0 + W_REGIME * 0.0) / (W_BASE + W_REGIME)
    if abs(renorm["conviction"] - round(expect, 6)) > 1e-6:
        print("FAIL: weight renormalization", renorm, expect); ok = False

    # Skip floor: only fires when skip_below is raised above 0.
    weak = compute_conviction({"base": 0.05}, direction="call")
    if weak["tier"] == TIER_SKIP:
        print("FAIL: default floor should never skip", weak); ok = False
    weak_skip = compute_conviction({"base": 0.05}, direction="call",
                                   skip_below=0.1)
    if weak_skip["tier"] != TIER_SKIP or weak_skip["contracts"] != 0:
        print("FAIL: raised skip floor should skip", weak_skip); ok = False

    # Empty / no usable components -> None conviction (fail-open for the caller).
    empty = compute_conviction({}, direction="call")
    if empty["conviction"] is not None or empty["contracts"] is not None:
        print("FAIL: empty components -> None", empty); ok = False

    # Determinism + never-raise on junk.
    if compute_conviction({"base": 0.7}, "call") != \
            compute_conviction({"base": 0.7}, "call"):
        print("FAIL: non-deterministic"); ok = False
    for junk in (None, 42, "x", [], {"base": "bad"}, {"agent_consensus": "bad"},
                 {"weird": object()}):
        try:
            r = compute_conviction(junk, direction="call")  # type: ignore[arg-type]
            if "conviction" not in r or "contracts" not in r:
                print("FAIL: junk shape", junk, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("conviction self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
