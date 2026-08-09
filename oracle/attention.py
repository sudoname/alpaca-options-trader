"""
Oracle 2.1 — Attention engine (ANALYTICS ONLY, pure, no I/O).

*Attention* is how much the crowd is looking at a name right now, proxied by
relative volume (``volume_ratio`` = current / trailing-average volume, so 1.0 is
a normal day). This module RECORDS an attention level and — when a prior
reading is supplied — its *velocity* (is attention rising or fading). It is a
FEATURE, never a trade trigger: it enriches the shadow slate for later
leaderboard analysis and NEVER opens, sizes, prices, blocks, alters, or directs
a trade.

Two readings, both fail-open:
  * Level (default): a label from ``volume_ratio`` — low / normal / elevated /
    spike. Optionally corroborated by ``news_count`` (headline density is its
    own attention signal).
  * Velocity: current ``volume_ratio`` minus an optional ``prior_volume_ratio``
    baseline. Only computed when a prior is supplied (ctx is point-in-time and
    carries no history by default); otherwise velocity is None and
    ``accelerating`` is None. A positive velocity means attention is building.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

# Level labels.
LEVEL_LOW = "low"
LEVEL_NORMAL = "normal"
LEVEL_ELEVATED = "elevated"
LEVEL_SPIKE = "spike"

# volume_ratio thresholds (current / trailing-average volume; 1.0 == normal).
LOW_BELOW = 0.8
ELEVATED_AT = 1.5
SPIKE_AT = 3.0

# Headline density that on its own counts as elevated attention.
NEWS_COUNT_ELEVATED = 5.0


def _to_float(value) -> Optional[float]:
    """Coerce to float; bools and junk -> None (never raises)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _level_label(volume_ratio: Optional[float],
                 news_count: Optional[float]) -> Optional[str]:
    """Map a volume_ratio (+ optional news_count) to a level label. None if bare."""
    vr = volume_ratio
    if vr is None:
        # Fall back to headline density alone when volume is unavailable.
        if news_count is not None and news_count >= NEWS_COUNT_ELEVATED:
            return LEVEL_ELEVATED
        return None
    if vr < LOW_BELOW:
        label = LEVEL_LOW
    elif vr < ELEVATED_AT:
        label = LEVEL_NORMAL
    elif vr < SPIKE_AT:
        label = LEVEL_ELEVATED
    else:
        label = LEVEL_SPIKE
    # Heavy headline flow lifts a merely-normal tape to elevated.
    if label == LEVEL_NORMAL and news_count is not None \
            and news_count >= NEWS_COUNT_ELEVATED:
        label = LEVEL_ELEVATED
    return label


@dataclass
class AttentionStamp:
    """One attention reading. All fields analytics-only."""

    level: Optional[float]         # the raw volume_ratio used (None if absent)
    level_label: Optional[str]     # low / normal / elevated / spike / None
    velocity: Optional[float]      # current - prior volume_ratio (None if no prior)
    accelerating: Optional[bool]   # velocity > 0 (None when velocity is None)
    news_count: Optional[float]    # headline density corroborating the level

    def to_dict(self) -> dict:
        return {
            "level": self.level,
            "level_label": self.level_label,
            "velocity": self.velocity,
            "accelerating": self.accelerating,
            "news_count": self.news_count,
        }


def compute_attention(ctx: Optional[dict]) -> dict:
    """Return an attention stamp dict from a market ``ctx``. Never raises.

    Consumes (all optional): ``volume_ratio``, ``prior_volume_ratio``,
    ``news_count``. Fails open field-by-field to None.
    """
    if not isinstance(ctx, dict):
        ctx = {}

    vr = _to_float(ctx.get("volume_ratio"))
    prior = _to_float(ctx.get("prior_volume_ratio"))
    news_count = _to_float(ctx.get("news_count"))

    label = _level_label(vr, news_count)

    velocity: Optional[float] = None
    accelerating: Optional[bool] = None
    if vr is not None and prior is not None:
        velocity = round(vr - prior, 6)
        accelerating = velocity > 0.0

    return AttentionStamp(
        level=(round(vr, 6) if vr is not None else None),
        level_label=label,
        velocity=velocity,
        accelerating=accelerating,
        news_count=news_count,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Level labels across the volume_ratio ladder.
    cases = [
        (0.5, LEVEL_LOW),
        (1.0, LEVEL_NORMAL),
        (2.0, LEVEL_ELEVATED),
        (4.0, LEVEL_SPIKE),
    ]
    for vr, want in cases:
        a = compute_attention({"volume_ratio": vr})
        if a["level_label"] != want:
            print("FAIL: level label", vr, a["level_label"], "want", want)
            ok = False
        if abs((a["level"] or 0) - vr) > 1e-9:
            print("FAIL: level passthrough", vr, a); ok = False
        # No prior -> no velocity.
        if a["velocity"] is not None or a["accelerating"] is not None:
            print("FAIL: no prior -> no velocity", a); ok = False

    # Velocity: rising attention.
    up = compute_attention({"volume_ratio": 2.0, "prior_volume_ratio": 1.0})
    if abs(up["velocity"] - 1.0) > 1e-9 or up["accelerating"] is not True:
        print("FAIL: rising velocity", up); ok = False

    # Velocity: fading attention.
    down = compute_attention({"volume_ratio": 1.0, "prior_volume_ratio": 2.5})
    if not (down["velocity"] < 0) or down["accelerating"] is not False:
        print("FAIL: fading velocity", down); ok = False

    # Velocity monotonicity: a bigger jump -> bigger velocity.
    small = compute_attention({"volume_ratio": 1.5, "prior_volume_ratio": 1.0})
    big = compute_attention({"volume_ratio": 3.0, "prior_volume_ratio": 1.0})
    if not (big["velocity"] > small["velocity"]):
        print("FAIL: velocity monotonicity", small, big); ok = False

    # Headline density lifts a normal tape to elevated.
    lift = compute_attention({"volume_ratio": 1.0, "news_count": 8})
    if lift["level_label"] != LEVEL_ELEVATED:
        print("FAIL: news lifts normal -> elevated", lift); ok = False

    # News alone (no volume) can flag elevated attention.
    news_only = compute_attention({"news_count": 6})
    if news_only["level"] is not None:
        print("FAIL: no volume -> level None", news_only); ok = False
    if news_only["level_label"] != LEVEL_ELEVATED:
        print("FAIL: news-only elevated", news_only); ok = False

    # Bare ctx -> everything None, stable shape.
    bare = compute_attention({})
    for k in ("level", "level_label", "velocity", "accelerating"):
        if bare[k] is not None:
            print("FAIL: bare should be None", k, bare); ok = False

    # Determinism + never-raise on junk.
    if compute_attention({"volume_ratio": 2.0}) != \
            compute_attention({"volume_ratio": 2.0}):
        print("FAIL: non-deterministic"); ok = False
    for junk in (None, 42, "x", [], {"weird": object()},
                 {"volume_ratio": "bad"}, {"prior_volume_ratio": "bad"}):
        try:
            r = compute_attention(junk)  # type: ignore[arg-type]
            if "level_label" not in r or "velocity" not in r:
                print("FAIL: junk shape", junk, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("attention self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
