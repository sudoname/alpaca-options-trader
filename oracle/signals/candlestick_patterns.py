"""
Phase 11B-1 — Candlestick pattern detection (ANALYTICS ONLY, pure, no I/O).

Candlestick patterns are *market-behaviour features* only. They are NEVER a
trading strategy and NEVER trigger, block, size, or alter a real/paper trade.
They never change Oracle's first-principles signals (strategy, direction, EV,
PoP, volatility edge, advisory, gates, risk, approval). The ONLY thing a pattern
may ever do downstream is contribute a small, hard-capped *confidence
adjustment* — and even that boosted value is an analytics-only field that is
never fed back into advisory / EV / gate logic.

Design rules:
  * Pure functions. No network, no creds, no file writes (config read is
    fail-open via :class:`config_loader.ConfigLoader`).
  * Detection is opt-in / injected: it only runs when OHLCV candles are passed
    in by a caller. Nothing here reaches out for market data.
  * Prefer false negatives. When trend context is required but unclear, or when
    inputs are missing/malformed, detectors return ``None`` rather than guess.
  * Everything is offline-testable with synthetic OHLCV fixtures.

Candles are ordered oldest -> newest; the pattern is evaluated on the most
recent candle(s), with the candles before that window used for trend context.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

# --------------------------------------------------------------------------- #
# Constants
# --------------------------------------------------------------------------- #
# pattern_type
BULLISH_REVERSAL = "bullish_reversal"
BEARISH_REVERSAL = "bearish_reversal"
CONTINUATION = "continuation"
INDECISION = "indecision"

# bias
BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"

# strength
STRENGTH_WEAK = "weak"
STRENGTH_MEDIUM = "medium"
STRENGTH_STRONG = "strong"

# trend
TREND_UP = "up"
TREND_DOWN = "down"
TREND_FLAT = "flat"

# 16 pattern-name constants
HAMMER = "hammer"
INVERTED_HAMMER = "inverted_hammer"
HANGING_MAN = "hanging_man"
SHOOTING_STAR = "shooting_star"
DOJI = "doji"
SPINNING_TOP = "spinning_top"
BULLISH_ENGULFING = "bullish_engulfing"
BEARISH_ENGULFING = "bearish_engulfing"
PIERCING_LINE = "piercing_line"
DARK_CLOUD_COVER = "dark_cloud_cover"
MORNING_STAR = "morning_star"
EVENING_STAR = "evening_star"
THREE_WHITE_SOLDIERS = "three_white_soldiers"
THREE_BLACK_CROWS = "three_black_crows"
RISING_THREE_METHODS = "rising_three_methods"
FALLING_THREE_METHODS = "falling_three_methods"

# Candidate field names frozen onto a stamped candidate.
FIELD_PATTERN = "candlestick_pattern"
FIELD_BIAS = "candlestick_bias"
FIELD_STRENGTH = "candlestick_strength"
FIELD_CONFIDENCE = "candlestick_confidence"
FIELD_REASON = "candlestick_reason"
FIELD_REQUIRES_CONFIRMATION = "candlestick_requires_confirmation"
STAMP_FIELDS = (
    FIELD_PATTERN, FIELD_BIAS, FIELD_STRENGTH,
    FIELD_CONFIDENCE, FIELD_REASON, FIELD_REQUIRES_CONFIRMATION,
)

# Net-direction threshold (fraction of |first close|) for a "clear" trend.
TREND_THRESHOLD = 0.003

_STRENGTH_RANK = {STRENGTH_WEAK: 0, STRENGTH_MEDIUM: 1, STRENGTH_STRONG: 2}
# Reversals/continuations carry directional information; indecision does not.
_TYPE_RANK = {
    BULLISH_REVERSAL: 2, BEARISH_REVERSAL: 2, CONTINUATION: 2, INDECISION: 0,
}

# Advisory recommendations that count as "aligned" for context scoring.
_ALIGNED_ADVISORIES = ("ACCEPT", "STRONG_ACCEPT")


# --------------------------------------------------------------------------- #
# Data classes
# --------------------------------------------------------------------------- #
@dataclass
class PatternStamp:
    """One detected candlestick pattern. All fields are analytics-only."""

    pattern_name: str
    pattern_type: str
    bias: str
    strength: str
    confidence: float            # 0.0 - 1.0
    lookback_window: int
    reason: str
    requires_confirmation: bool

    def to_dict(self) -> dict:
        return {
            "pattern_name": self.pattern_name,
            "pattern_type": self.pattern_type,
            "bias": self.bias,
            "strength": self.strength,
            "confidence": self.confidence,
            "lookback_window": self.lookback_window,
            "reason": self.reason,
            "requires_confirmation": self.requires_confirmation,
        }


@dataclass
class CandlestickConfig:
    """Tunables (local ``.env`` only). Patterns never alter execution."""

    enabled: bool = True
    max_boost: float = 0.05
    require_trend_context: bool = True
    require_volume_confirmation: bool = False

    @staticmethod
    def from_env(path: str = ".env") -> "CandlestickConfig":
        """Read config with shell>.env>default precedence. Never raises."""
        try:
            from config_loader import ConfigLoader
            c = ConfigLoader(path=path)
            return CandlestickConfig(
                enabled=c.get_bool("ENABLE_CANDLESTICK_PATTERNS", True),
                max_boost=c.get_float("CANDLESTICK_MAX_BOOST", 0.05),
                require_trend_context=c.get_bool(
                    "CANDLESTICK_REQUIRE_TREND_CONTEXT", True),
                require_volume_confirmation=c.get_bool(
                    "CANDLESTICK_REQUIRE_VOLUME_CONFIRMATION", False),
            )
        except Exception:
            return CandlestickConfig()


# --------------------------------------------------------------------------- #
# Candle normalization + shape helpers
# --------------------------------------------------------------------------- #
Candle = Tuple[float, float, float, float, Optional[float]]


def _f(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm(candle) -> Optional[Candle]:
    """Coerce a candle (dict ``o/h/l/c/v`` or ``open/high/...`` or a Bar-like
    namedtuple/object) to ``(o, h, l, c, v)`` floats. ``None`` on any problem."""
    try:
        if candle is None:
            return None
        if isinstance(candle, dict):
            o = _f(candle.get("o", candle.get("open")))
            h = _f(candle.get("h", candle.get("high")))
            low = _f(candle.get("l", candle.get("low")))
            c = _f(candle.get("c", candle.get("close")))
            v = _f(candle.get("v", candle.get("volume")))
        elif isinstance(candle, (tuple, list)):
            if len(candle) < 4:
                return None
            o, h, low, c = (_f(candle[0]), _f(candle[1]),
                            _f(candle[2]), _f(candle[3]))
            v = _f(candle[4]) if len(candle) > 4 else None
        else:  # attribute-style (Bar namedtuple / object)
            o = _f(getattr(candle, "o", getattr(candle, "open", None)))
            h = _f(getattr(candle, "h", getattr(candle, "high", None)))
            low = _f(getattr(candle, "l", getattr(candle, "low", None)))
            c = _f(getattr(candle, "c", getattr(candle, "close", None)))
            v = _f(getattr(candle, "v", getattr(candle, "volume", None)))
        if None in (o, h, low, c):
            return None
        if h < low:
            return None
        return (o, h, low, c, v)
    except Exception:
        return None


def _normseq(candles) -> Optional[List[Candle]]:
    """Normalize a whole sequence; ``None`` if empty or any candle is bad."""
    if not candles:
        return None
    out: List[Candle] = []
    for c in candles:
        n = _norm(c)
        if n is None:
            return None
        out.append(n)
    return out


def _body(t: Candle) -> float:
    return abs(t[3] - t[0])


def _upper(t: Candle) -> float:
    return t[1] - max(t[0], t[3])


def _lower(t: Candle) -> float:
    return min(t[0], t[3]) - t[2]


def _rng(t: Candle) -> float:
    return t[1] - t[2]


def _is_bull(t: Candle) -> bool:
    return t[3] > t[0]


def _is_bear(t: Candle) -> bool:
    return t[3] < t[0]


def _mid(t: Candle) -> float:
    return (t[0] + t[3]) / 2.0


def _trend(seq: Sequence[Candle]) -> str:
    """Net close-to-close direction over the prior window."""
    closes = [t[3] for t in seq]
    if len(closes) < 2:
        return TREND_FLAT
    first, last = closes[0], closes[-1]
    if first == 0:
        return TREND_FLAT
    change = (last - first) / abs(first)
    if change > TREND_THRESHOLD:
        return TREND_UP
    if change < -TREND_THRESHOLD:
        return TREND_DOWN
    return TREND_FLAT


def _trend_ok(trend: str, expected: str, config: CandlestickConfig) -> bool:
    """True when the prior trend supports a reversal/continuation.

    When ``require_trend_context`` is on (default), the trend must match
    exactly. When off, only a clearly *contrary* trend blocks the pattern
    (flat is permitted) — still preferring false negatives over false signals.
    """
    if trend == expected:
        return True
    if config.require_trend_context:
        return False
    opposite = TREND_UP if expected == TREND_DOWN else TREND_DOWN
    return trend != opposite


def _volume_ok(window: Sequence[Candle], config: CandlestickConfig) -> bool:
    """When volume confirmation is required, the final candle's volume must
    exceed the mean of the earlier candles' volume. Missing volume -> False
    (false negative). When not required, always True."""
    if not config.require_volume_confirmation:
        return True
    vols = [t[4] for t in window]
    if any(v is None for v in vols) or len(vols) < 2:
        return False
    last = vols[-1]
    prior = vols[:-1]
    return last > (sum(prior) / len(prior))


def _stamp(name, ptype, bias, strength, confidence, lookback, reason,
           requires_confirmation) -> PatternStamp:
    return PatternStamp(
        pattern_name=name, pattern_type=ptype, bias=bias, strength=strength,
        confidence=round(float(confidence), 4), lookback_window=lookback,
        reason=reason, requires_confirmation=requires_confirmation)


# --------------------------------------------------------------------------- #
# Single-candle detectors
# --------------------------------------------------------------------------- #
def _has_hammer_shape(t: Candle) -> bool:
    """Small body near the top, long lower shadow, little upper shadow."""
    body, rng = _body(t), _rng(t)
    if rng <= 0 or body <= 0:
        return False
    return _lower(t) >= 2.0 * body and _upper(t) <= 0.30 * rng


def _has_star_shape(t: Candle) -> bool:
    """Small body near the bottom, long upper shadow, little lower shadow."""
    body, rng = _body(t), _rng(t)
    if rng <= 0 or body <= 0:
        return False
    return _upper(t) >= 2.0 * body and _lower(t) <= 0.30 * rng


def detect_hammer(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 2:
        return None
    t, prior = s[-1], s[:-1]
    if not _has_hammer_shape(t):
        return None
    if not _trend_ok(_trend(prior), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-1:], config):
        return None
    return _stamp(HAMMER, BULLISH_REVERSAL, BULLISH, STRENGTH_MEDIUM, 0.55, 1,
                  "Long lower shadow after a downtrend (hammer).", True)


def detect_inverted_hammer(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 2:
        return None
    t, prior = s[-1], s[:-1]
    if not _has_star_shape(t):
        return None
    if not _trend_ok(_trend(prior), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-1:], config):
        return None
    return _stamp(INVERTED_HAMMER, BULLISH_REVERSAL, BULLISH, STRENGTH_MEDIUM,
                  0.50, 1,
                  "Long upper shadow after a downtrend (inverted hammer).",
                  True)


def detect_hanging_man(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 2:
        return None
    t, prior = s[-1], s[:-1]
    if not _has_hammer_shape(t):
        return None
    if not _trend_ok(_trend(prior), TREND_UP, config):
        return None
    if not _volume_ok(s[-1:], config):
        return None
    return _stamp(HANGING_MAN, BEARISH_REVERSAL, BEARISH, STRENGTH_MEDIUM,
                  0.50, 1,
                  "Long lower shadow after an uptrend (hanging man).", True)


def detect_shooting_star(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 2:
        return None
    t, prior = s[-1], s[:-1]
    if not _has_star_shape(t):
        return None
    if not _trend_ok(_trend(prior), TREND_UP, config):
        return None
    if not _volume_ok(s[-1:], config):
        return None
    return _stamp(SHOOTING_STAR, BEARISH_REVERSAL, BEARISH, STRENGTH_MEDIUM,
                  0.55, 1,
                  "Long upper shadow after an uptrend (shooting star).", True)


def detect_doji(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s:
        return None
    t = s[-1]
    rng = _rng(t)
    if rng <= 0:
        return None
    if _body(t) > 0.10 * rng:
        return None
    return _stamp(DOJI, INDECISION, NEUTRAL, STRENGTH_WEAK, 0.30, 1,
                  "Open and close nearly equal (doji): indecision.", True)


def detect_spinning_top(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s:
        return None
    t = s[-1]
    body, rng = _body(t), _rng(t)
    if rng <= 0 or body <= 0:
        return None
    if body > 0.30 * rng or body <= 0.10 * rng:
        return None  # >0.10 rng excludes doji; <=0.30 keeps the body small
    if _upper(t) < body or _lower(t) < body:
        return None
    return _stamp(SPINNING_TOP, INDECISION, NEUTRAL, STRENGTH_WEAK, 0.35, 1,
                  "Small body with shadows on both sides (spinning top).", True)


# --------------------------------------------------------------------------- #
# Two-candle detectors
# --------------------------------------------------------------------------- #
def detect_bullish_engulfing(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 3:
        return None
    prev, cur, prior = s[-2], s[-1], s[:-2]
    if not (_is_bear(prev) and _is_bull(cur)):
        return None
    if not (cur[0] <= prev[3] and cur[3] >= prev[0]):
        return None
    if _body(cur) <= _body(prev):
        return None
    if not _trend_ok(_trend(prior + [prev]), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-2:], config):
        return None
    return _stamp(BULLISH_ENGULFING, BULLISH_REVERSAL, BULLISH, STRENGTH_STRONG,
                  0.70, 2,
                  "Bullish candle engulfs the prior bearish body.", False)


def detect_bearish_engulfing(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 3:
        return None
    prev, cur, prior = s[-2], s[-1], s[:-2]
    if not (_is_bull(prev) and _is_bear(cur)):
        return None
    if not (cur[0] >= prev[3] and cur[3] <= prev[0]):
        return None
    if _body(cur) <= _body(prev):
        return None
    if not _trend_ok(_trend(prior + [prev]), TREND_UP, config):
        return None
    if not _volume_ok(s[-2:], config):
        return None
    return _stamp(BEARISH_ENGULFING, BEARISH_REVERSAL, BEARISH, STRENGTH_STRONG,
                  0.70, 2,
                  "Bearish candle engulfs the prior bullish body.", False)


def detect_piercing_line(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 3:
        return None
    prev, cur, prior = s[-2], s[-1], s[:-2]
    if not (_is_bear(prev) and _is_bull(cur)):
        return None
    # Opens below the prior close, closes back into the upper half of the
    # prior (bearish) body but not above its open.
    if not (cur[0] < prev[3] and cur[3] > _mid(prev) and cur[3] < prev[0]):
        return None
    if not _trend_ok(_trend(prior + [prev]), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-2:], config):
        return None
    return _stamp(PIERCING_LINE, BULLISH_REVERSAL, BULLISH, STRENGTH_MEDIUM,
                  0.60, 2,
                  "Close pierces above the midpoint of the prior bearish body.",
                  True)


def detect_dark_cloud_cover(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 3:
        return None
    prev, cur, prior = s[-2], s[-1], s[:-2]
    if not (_is_bull(prev) and _is_bear(cur)):
        return None
    if not (cur[0] > prev[3] and cur[3] < _mid(prev) and cur[3] > prev[0]):
        return None
    if not _trend_ok(_trend(prior + [prev]), TREND_UP, config):
        return None
    if not _volume_ok(s[-2:], config):
        return None
    return _stamp(DARK_CLOUD_COVER, BEARISH_REVERSAL, BEARISH, STRENGTH_MEDIUM,
                  0.60, 2,
                  "Close drops below the midpoint of the prior bullish body.",
                  True)


# --------------------------------------------------------------------------- #
# Three-candle detectors
# --------------------------------------------------------------------------- #
def detect_morning_star(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 4:
        return None
    c1, c2, c3, prior = s[-3], s[-2], s[-1], s[:-3]
    if not (_is_bear(c1) and _is_bull(c3)):
        return None
    if _body(c2) >= _body(c1) or _body(c2) >= _body(c3):
        return None  # middle is a small star
    if c3[3] <= _mid(c1):
        return None  # third closes well into the first body
    if not _trend_ok(_trend(prior + [c1]), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-3:], config):
        return None
    return _stamp(MORNING_STAR, BULLISH_REVERSAL, BULLISH, STRENGTH_STRONG,
                  0.78, 3,
                  "Down candle, small star, then a strong up close (morning "
                  "star).", False)


def detect_evening_star(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 4:
        return None
    c1, c2, c3, prior = s[-3], s[-2], s[-1], s[:-3]
    if not (_is_bull(c1) and _is_bear(c3)):
        return None
    if _body(c2) >= _body(c1) or _body(c2) >= _body(c3):
        return None
    if c3[3] >= _mid(c1):
        return None
    if not _trend_ok(_trend(prior + [c1]), TREND_UP, config):
        return None
    if not _volume_ok(s[-3:], config):
        return None
    return _stamp(EVENING_STAR, BEARISH_REVERSAL, BEARISH, STRENGTH_STRONG,
                  0.78, 3,
                  "Up candle, small star, then a strong down close (evening "
                  "star).", False)


def detect_three_white_soldiers(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 4:
        return None
    c1, c2, c3, prior = s[-3], s[-2], s[-1], s[:-3]
    if not (_is_bull(c1) and _is_bull(c2) and _is_bull(c3)):
        return None
    if not (c2[3] > c1[3] and c3[3] > c2[3]):
        return None  # each closes higher
    # Each opens within the prior real body (no big gaps).
    if not (c1[0] < c2[0] <= c1[3] and c2[0] < c3[0] <= c2[3]):
        return None
    if min(_body(c1), _body(c2), _body(c3)) <= 0:
        return None
    if not _trend_ok(_trend(prior + [c1]), TREND_DOWN, config):
        return None
    if not _volume_ok(s[-3:], config):
        return None
    return _stamp(THREE_WHITE_SOLDIERS, BULLISH_REVERSAL, BULLISH,
                  STRENGTH_STRONG, 0.82, 3,
                  "Three rising bullish candles (three white soldiers).", False)


def detect_three_black_crows(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 4:
        return None
    c1, c2, c3, prior = s[-3], s[-2], s[-1], s[:-3]
    if not (_is_bear(c1) and _is_bear(c2) and _is_bear(c3)):
        return None
    if not (c2[3] < c1[3] and c3[3] < c2[3]):
        return None
    if not (c1[3] <= c2[0] < c1[0] and c2[3] <= c3[0] < c2[0]):
        return None
    if min(_body(c1), _body(c2), _body(c3)) <= 0:
        return None
    if not _trend_ok(_trend(prior + [c1]), TREND_UP, config):
        return None
    if not _volume_ok(s[-3:], config):
        return None
    return _stamp(THREE_BLACK_CROWS, BEARISH_REVERSAL, BEARISH, STRENGTH_STRONG,
                  0.82, 3,
                  "Three falling bearish candles (three black crows).", False)


# --------------------------------------------------------------------------- #
# Five-candle continuation detectors
# --------------------------------------------------------------------------- #
def detect_rising_three_methods(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 5:
        return None
    c1, mids, c5 = s[-5], s[-5 + 1:-1], s[-1]
    prior = s[:-5]
    if not (_is_bull(c1) and _is_bull(c5)):
        return None
    if c5[3] <= c1[3]:
        return None  # closes above the first close
    # The three middle candles are small and hold inside the first range.
    for m in mids:
        if _body(m) >= _body(c1):
            return None
        if m[1] > c1[1] or m[2] < c1[2]:
            return None
    if not _trend_ok(_trend(prior + [c1]) if prior else TREND_UP,
                     TREND_UP, config):
        return None
    if not _volume_ok(s[-5:], config):
        return None
    return _stamp(RISING_THREE_METHODS, CONTINUATION, BULLISH, STRENGTH_STRONG,
                  0.72, 5,
                  "Bullish pause then continuation (rising three methods).",
                  False)


def detect_falling_three_methods(seq, config) -> Optional[PatternStamp]:
    s = _normseq(seq)
    if not s or len(s) < 5:
        return None
    c1, mids, c5 = s[-5], s[-5 + 1:-1], s[-1]
    prior = s[:-5]
    if not (_is_bear(c1) and _is_bear(c5)):
        return None
    if c5[3] >= c1[3]:
        return None
    for m in mids:
        if _body(m) >= _body(c1):
            return None
        if m[1] > c1[1] or m[2] < c1[2]:
            return None
    if not _trend_ok(_trend(prior + [c1]) if prior else TREND_DOWN,
                     TREND_DOWN, config):
        return None
    if not _volume_ok(s[-5:], config):
        return None
    return _stamp(FALLING_THREE_METHODS, CONTINUATION, BEARISH, STRENGTH_STRONG,
                  0.72, 5,
                  "Bearish pause then continuation (falling three methods).",
                  False)


# Registry in a stable, deterministic order (governs detect_primary tie-breaks).
_DETECTORS = (
    detect_three_white_soldiers,
    detect_three_black_crows,
    detect_morning_star,
    detect_evening_star,
    detect_bullish_engulfing,
    detect_bearish_engulfing,
    detect_rising_three_methods,
    detect_falling_three_methods,
    detect_piercing_line,
    detect_dark_cloud_cover,
    detect_shooting_star,
    detect_hammer,
    detect_hanging_man,
    detect_inverted_hammer,
    detect_spinning_top,
    detect_doji,
)


# --------------------------------------------------------------------------- #
# Public detection API
# --------------------------------------------------------------------------- #
def detect_patterns(candles, config: Optional[CandlestickConfig] = None
                    ) -> List[PatternStamp]:
    """Every pattern that matches the most recent window. Never raises."""
    cfg = config or CandlestickConfig()
    if not cfg.enabled:
        return []
    out: List[PatternStamp] = []
    for det in _DETECTORS:
        try:
            stamp = det(candles, cfg)
        except Exception:
            stamp = None
        if stamp is not None:
            out.append(stamp)
    return out


def _rank_key(stamp: PatternStamp):
    return (
        stamp.confidence,
        _STRENGTH_RANK.get(stamp.strength, 0),
        _TYPE_RANK.get(stamp.pattern_type, 0),
    )


def detect_primary(candles, config: Optional[CandlestickConfig] = None
                   ) -> Optional[PatternStamp]:
    """Best single pattern by (confidence, strength, type). Reversals and
    continuations outrank indecision. ``None`` when disabled or no match."""
    matches = detect_patterns(candles, config)
    if not matches:
        return None
    best = matches[0]
    best_key = _rank_key(best)
    for stamp in matches[1:]:
        key = _rank_key(stamp)
        if key > best_key:
            best, best_key = stamp, key
    return best


# --------------------------------------------------------------------------- #
# Candidate stamping (frozen at candidate time)
# --------------------------------------------------------------------------- #
def stamp_candlestick_patterns(candidate: dict, candles,
                               config: Optional[CandlestickConfig] = None
                               ) -> dict:
    """Mutate + return ``candidate`` with the 6 frozen candlestick fields.

    All six are ``None`` when detection is disabled or nothing matched. Never
    raises — on any failure the fields are still present (set to ``None``)."""
    if candidate is None:
        candidate = {}
    fields = {f: None for f in STAMP_FIELDS}
    try:
        cfg = config or CandlestickConfig.from_env()
        if cfg.enabled:
            stamp = detect_primary(candles, cfg)
            if stamp is not None:
                fields = {
                    FIELD_PATTERN: stamp.pattern_name,
                    FIELD_BIAS: stamp.bias,
                    FIELD_STRENGTH: stamp.strength,
                    FIELD_CONFIDENCE: stamp.confidence,
                    FIELD_REASON: stamp.reason,
                    FIELD_REQUIRES_CONFIRMATION: stamp.requires_confirmation,
                }
    except Exception:
        fields = {f: None for f in STAMP_FIELDS}
    candidate.update(fields)
    return candidate


# --------------------------------------------------------------------------- #
# Context alignment + capped confidence boost (ANALYTICS ONLY)
# --------------------------------------------------------------------------- #
def evaluate_pattern_context(
    stamp: Optional[PatternStamp], *,
    trend: Optional[str] = None,
    support_resistance: Optional[str] = None,
    volume_confirms: Optional[bool] = None,
    volatility_regime: Optional[str] = None,
    pop: Optional[float] = None,
    ev: Optional[float] = None,
    advisory: Optional[str] = None,
    triple_gap: Optional[float] = None,
    config: Optional[CandlestickConfig] = None,
) -> dict:
    """How well the surrounding context agrees with the pattern's bias.

    ``context_alignment_score`` = aligned / considered over the NON-None
    signals. Neutral-bias patterns score 0 (no directional conviction). Never
    raises."""
    result = {"context_alignment_score": 0.0, "aligned": 0, "considered": 0}
    try:
        if stamp is None or stamp.bias == NEUTRAL:
            return result
        bullish = stamp.bias == BULLISH
        aligned = 0
        considered = 0

        def _vote(value_present: bool, is_aligned: bool):
            nonlocal aligned, considered
            if value_present:
                considered += 1
                if is_aligned:
                    aligned += 1

        _vote(trend is not None,
              trend == (TREND_UP if bullish else TREND_DOWN))
        _vote(support_resistance is not None,
              support_resistance == ("support" if bullish else "resistance"))
        _vote(volume_confirms is not None, bool(volume_confirms))
        _vote(ev is not None, (ev or 0) > 0)
        _vote(pop is not None, (pop or 0) >= 0.5)
        _vote(advisory is not None,
              str(advisory).upper() in _ALIGNED_ADVISORIES)
        _vote(triple_gap is not None, (triple_gap or 0) >= 50)
        # volatility_regime is observed but not directional; it is recorded as
        # "considered" only when it explicitly confirms (e.g. "elevated").
        _vote(volatility_regime is not None,
              str(volatility_regime).lower() in ("elevated", "high", "rising"))

        score = round(aligned / considered, 4) if considered else 0.0
        return {"context_alignment_score": score,
                "aligned": aligned, "considered": considered}
    except Exception:
        return result


def apply_candlestick_boost(base_confidence: float,
                            stamp: Optional[PatternStamp],
                            alignment_score: float,
                            config: Optional[CandlestickConfig] = None) -> dict:
    """Apply a small, hard-capped, ONLY-POSITIVE confidence boost.

    ``boost = min(max_boost * alignment_score * stamp.confidence, max_boost)``,
    floored at 0; neutral-bias or missing stamp -> 0. ``final = min(base +
    boost, 1.0)``. The returned ``final_confidence`` is analytics-only and is
    NEVER fed back into advisory / EV / gate logic. Never raises."""
    cfg = config or CandlestickConfig()
    base = _f(base_confidence) or 0.0
    out = {"base_confidence": round(base, 4), "boost_applied": 0.0,
           "final_confidence": round(base, 4), "capped": False}
    try:
        if stamp is None or stamp.bias == NEUTRAL:
            return out
        align = _f(alignment_score) or 0.0
        if align <= 0:
            return out
        raw = cfg.max_boost * align * stamp.confidence
        boost = max(0.0, min(raw, cfg.max_boost))
        final = min(base + boost, 1.0)
        out.update({
            "boost_applied": round(boost, 6),
            "final_confidence": round(final, 6),
            "capped": raw > cfg.max_boost,
        })
        return out
    except Exception:
        return out


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = CandlestickConfig()

    # A hammer after a clean downtrend.
    downtrend = [(0, 0, 0, c, 100) for c in (110, 108, 106, 104)]
    downtrend = [(c + 1, c + 1.2, c - 1, c, 100) for c in (110, 108, 106, 104)]
    hammer_candle = (100.0, 100.5, 95.0, 100.2, 120)  # long lower shadow
    stamp = detect_primary(downtrend + [hammer_candle], cfg)
    if stamp is None or stamp.pattern_name != HAMMER:
        print("FAIL: hammer not detected:", stamp); ok = False

    # Neutral doji -> no boost regardless of alignment.
    doji = (100.0, 101.0, 99.0, 100.02, 100)
    doji_stamp = detect_primary([doji], cfg)
    if doji_stamp is None or doji_stamp.bias != NEUTRAL:
        print("FAIL: doji not neutral:", doji_stamp); ok = False
    b = apply_candlestick_boost(0.70, doji_stamp, 1.0, cfg)
    if b["boost_applied"] != 0.0:
        print("FAIL: neutral boost should be 0:", b); ok = False

    # Boost never exceeds max_boost.
    if stamp is not None:
        ctx = evaluate_pattern_context(stamp, trend=TREND_UP, ev=10.0,
                                       pop=0.8, advisory="ACCEPT",
                                       triple_gap=80)
        boosted = apply_candlestick_boost(0.70, stamp,
                                          ctx["context_alignment_score"], cfg)
        if boosted["boost_applied"] > cfg.max_boost + 1e-9:
            print("FAIL: boost exceeded cap:", boosted); ok = False
        if boosted["final_confidence"] < 0.70:
            print("FAIL: boost must never reduce:", boosted); ok = False

    # Missing data -> None, never raises.
    if detect_primary([], cfg) is not None:
        print("FAIL: empty candles should be None"); ok = False
    if detect_primary([{"bad": 1}], cfg) is not None:
        print("FAIL: malformed candle should be None"); ok = False

    # Disabled config -> no detection.
    if detect_patterns(downtrend + [hammer_candle],
                       CandlestickConfig(enabled=False)):
        print("FAIL: disabled config should detect nothing"); ok = False

    print("candlestick_patterns self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
