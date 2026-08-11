"""
Oracle Decision Kernel — direction tally (faithful extraction).

``compute_direction`` is a *pure, line-for-line* extraction of the rule tally in
``smart_trader.determine_option_strategy`` (smart_trader.py:1533). It exists so
the live path and any replay/backtest compute direction with IDENTICAL code
instead of the three divergent reimplementations that exist today.

Faithfulness contract (verified against the live source):
  * Insufficient data (<5 closes) -> 'call' (the live default), confidence 0.
  * Momentum votes: <-strong +2 bear / <-moderate +1 bear / >strong +2 bull /
    >moderate +1 bull   (SIGNAL_MOMENTUM_MODERATE=0.01, _STRONG=0.03).
  * Trend votes on 3-bar and 5-bar returns vs SIGNAL_SHORT_TREND=0.02 /
    SIGNAL_MEDIUM_TREND=0.03.
  * volatility>0.4 adds +1 to the leading side; regime=='volatile' with any
    bearish adds +1 bear.
  * News votes are an INPUT (snapshot.news already fetched) folded via
    ``news.news_direction_vote`` when available; fail-open to (0,0).
  * Decision: 'put' iff bear>bull and bear>=2, else 'call'.
  * Optional weak-signal SKIP (USE_SKIP_ON_WEAK_SIGNAL / MIN_DIRECTION_SIGNALS).
  * Confidence: 0 on skip; normalized (USE_NORMALIZED_CONFIDENCE) via the same
    margin*agreement formula; else the winning side's raw count.
  * Extension guard / repricing tilt / conviction fold are ALWAYS diagnostic and
    only ADJUST confidence when their flags are on — direction is NEVER flipped.

Direction is an OUTPUT of the evidence, never an input. Pure compute, fail-open,
deterministic in (snapshot, config).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from oracle.decision.schema import DecisionConfig, Snapshot, _to_float


# Faithful defaults (mirror smart_trader.py _f2/_i2/_flag defaults).
_D = {
    "SIGNAL_MOMENTUM_MODERATE": 0.01,
    "SIGNAL_MOMENTUM_STRONG": 0.03,
    "SIGNAL_SHORT_TREND": 0.02,
    "SIGNAL_MEDIUM_TREND": 0.03,
    "MIN_DIRECTION_SIGNALS": 2,
    "CONF_VERY_HIGH_SIGNALS": 4,
    "EXTENSION_CHASE_RATIO": 2.0,
    "EXTENSION_CONV_PENALTY": 1,
    "REPRICING_CONV_BONUS": 1,
    "CONVICTION_EXTENSION_PENALTY": 0.15,
    "CONVICTION_REPRICING_BONUS": 0.10,
    "CONVICTION_SKIP_BELOW": 0.0,
}


@dataclass(frozen=True)
class DirectionResult:
    """Immutable output of the tally. ``direction`` is 'call' | 'put' | 'skip'.
    ``confidence`` is the integer signal-strength on the SAME scale the live
    ``_confidence_to_quantity`` consumes."""

    direction: str
    bullish_signals: int
    bearish_signals: int
    total_signals: int
    signal_margin: int
    confidence: int
    skip_reason: Optional[str] = None
    short_trend: Optional[float] = None
    medium_trend: Optional[float] = None
    insufficient_data: bool = False
    extension: Optional[Dict[str, Any]] = None
    repricing: Optional[Dict[str, Any]] = None
    conviction: Optional[Dict[str, Any]] = None
    notes: Dict[str, Any] = field(default_factory=dict)


def _normalized_confidence(bullish: int, bearish: int, strategy: str) -> int:
    """Byte-faithful copy of smart_trader._normalized_confidence."""
    total = bullish + bearish
    if total <= 0:
        return 0
    margin = abs(bullish - bearish)
    if margin <= 0:
        return 0
    agreement = max(bullish, bearish) / total
    raw = bearish if strategy == "put" else bullish
    eff = int(margin * agreement + 0.5)
    return max(0, min(eff, raw))


def _news_votes(snapshot: Snapshot) -> tuple:
    """Fold recent headlines into (bull, bear) votes. Prefers precomputed votes
    stashed in ctx (deterministic replay); else calls the live vote function
    fail-open. Empty/absent news -> (0, 0)."""
    ctx = snapshot.ctx or {}
    nb = ctx.get("news_bull")
    nr = ctx.get("news_bear")
    if nb is not None or nr is not None:
        try:
            return int(nb or 0), int(nr or 0)
        except (TypeError, ValueError):
            return 0, 0
    if not snapshot.news:
        return 0, 0
    try:  # pragma: no cover - depends on optional news module
        from news import NewsConfig, news_direction_vote
        news = snapshot.news[0] if len(snapshot.news) == 1 else \
            (snapshot.news if isinstance(snapshot.news, dict) else snapshot.news[0])
        bull, bear = news_direction_vote(news, NewsConfig.from_env())
        return int(bull or 0), int(bear or 0)
    except Exception:
        return 0, 0


def compute_direction(snapshot: Snapshot,
                      config: Optional[DecisionConfig] = None) -> DirectionResult:
    """Apply the live rule tally to a frozen Snapshot. Never raises."""
    cfg = config or DecisionConfig.make({})

    def gf(name: str) -> float:
        return cfg.get_float(name, _D[name])

    def gi(name: str) -> int:
        return cfg.get_int(name, int(_D[name]))

    prices = list(snapshot.prices or ())
    if len(prices) < 5:
        # Live returns 'call' on insufficient data (smart_trader.py:1546).
        return DirectionResult(
            direction="call", bullish_signals=0, bearish_signals=0,
            total_signals=0, signal_margin=0, confidence=0,
            insufficient_data=True, notes={"reason": "insufficient_data"})

    momentum = _to_float(snapshot.momentum) or 0.0
    volatility = _to_float(snapshot.volatility) or 0.0
    market_regime = snapshot.market_regime

    short_trend = (prices[-1] - prices[-3]) / prices[-3] if prices[-3] else 0.0
    medium_trend = (prices[-1] - prices[-5]) / prices[-5] if prices[-5] else 0.0

    bearish = 0
    bullish = 0

    mom_mod = gf("SIGNAL_MOMENTUM_MODERATE")
    mom_strong = gf("SIGNAL_MOMENTUM_STRONG")
    if momentum < -mom_strong:
        bearish += 2
    elif momentum < -mom_mod:
        bearish += 1
    elif momentum > mom_strong:
        bullish += 2
    elif momentum > mom_mod:
        bullish += 1

    sst = gf("SIGNAL_SHORT_TREND")
    if short_trend < -sst:
        bearish += 1
    elif short_trend > sst:
        bullish += 1

    smt = gf("SIGNAL_MEDIUM_TREND")
    if medium_trend < -smt:
        bearish += 1
    elif medium_trend > smt:
        bullish += 1

    if volatility > 0.4 and bearish > bullish:
        bearish += 1
    elif volatility > 0.4 and bullish > bearish:
        bullish += 1

    if market_regime == "volatile" and bearish > 0:
        bearish += 1

    nb, nr = _news_votes(snapshot)
    bullish += nb
    bearish += nr

    total_signals = bullish + bearish
    signal_margin = abs(bullish - bearish)

    if bearish > bullish and bearish >= 2:
        strategy = "put"
    else:
        strategy = "call"

    skip_reason = None
    if cfg.get_bool("USE_SKIP_ON_WEAK_SIGNAL", False):
        min_sig = gi("MIN_DIRECTION_SIGNALS")
        if total_signals < min_sig:
            skip_reason = f"below_min_signals ({total_signals} < {min_sig})"
            strategy = "skip"
        elif bullish == bearish:
            skip_reason = f"flat_signal (bull {bullish} == bear {bearish})"
            strategy = "skip"

    if strategy == "skip":
        confidence = 0
    elif cfg.get_bool("USE_NORMALIZED_CONFIDENCE", False):
        confidence = _normalized_confidence(bullish, bearish, strategy)
    else:
        confidence = bearish if strategy == "put" else bullish

    extension = None
    repricing = None
    if strategy in ("put", "call"):
        try:
            em3 = volatility * math.sqrt(3.0 / 252.0) * 100.0
            recent = short_trend * 100.0
            from oracle.extension import detect_extension
            from oracle.repricing import detect_repricing
            extension = detect_extension(
                recent, em3, strategy, chase_ratio=gf("EXTENSION_CHASE_RATIO"))
            repricing = detect_repricing(recent, em3, strategy)
            if cfg.get_bool("ENABLE_EXTENSION_GUARD", False) and \
                    extension.get("extended"):
                confidence = max(0, confidence - gi("EXTENSION_CONV_PENALTY"))
            if cfg.get_bool("ENABLE_REPRICING_TILT", False) and \
                    repricing.get("opportunity"):
                confidence = confidence + gi("REPRICING_CONV_BONUS")
        except Exception:
            extension = None
            repricing = None

    conviction = None
    if strategy in ("put", "call"):
        try:
            from oracle.conviction import compute_conviction
            denom = float(gi("CONF_VERY_HIGH_SIGNALS")) or 4.0
            base = max(0.0, min(1.0, float(confidence) / denom))
            comp = {
                "base": base,
                "extended": (extension or {}).get("extended"),
                "opportunity": (repricing or {}).get("opportunity"),
            }
            conviction = compute_conviction(
                comp, direction=strategy,
                extension_penalty=gf("CONVICTION_EXTENSION_PENALTY"),
                repricing_bonus=gf("CONVICTION_REPRICING_BONUS"),
                skip_below=gf("CONVICTION_SKIP_BELOW"))
        except Exception:
            conviction = None

    return DirectionResult(
        direction=strategy,
        bullish_signals=bullish,
        bearish_signals=bearish,
        total_signals=total_signals,
        signal_margin=signal_margin,
        confidence=int(confidence),
        skip_reason=skip_reason,
        short_trend=round(short_trend, 6),
        medium_trend=round(medium_trend, 6),
        extension=extension,
        repricing=repricing,
        conviction=conviction,
        notes={"news_bull": nb, "news_bear": nr},
    )


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Strong up move: rising closes, positive momentum -> CALL with confidence.
    up = Snapshot.make("AAA", "t", prices=[100, 101, 102, 104, 106],
                       momentum=0.05, volatility=0.2, market_regime="trending")
    r = compute_direction(up)
    if r.direction != "call":
        print("FAIL: up should be call", r.direction); ok = False
    if r.bullish_signals <= r.bearish_signals:
        print("FAIL: up bull<=bear", r); ok = False

    # Strong down move: falling closes, negative momentum -> PUT (bear>=2).
    dn = Snapshot.make("BBB", "t", prices=[110, 108, 106, 103, 100],
                       momentum=-0.05, volatility=0.2, market_regime="trending")
    r = compute_direction(dn)
    if r.direction != "put":
        print("FAIL: down should be put", r.direction, r); ok = False
    if r.bearish_signals < 2:
        print("FAIL: down bearish<2", r); ok = False

    # Insufficient data -> live default 'call', flagged.
    r = compute_direction(Snapshot.make("CCC", "t", prices=[1, 2, 3]))
    if r.direction != "call" or not r.insufficient_data:
        print("FAIL: insufficient-data default", r); ok = False

    # Flat market defaults to CALL (bear never reaches the put threshold).
    flat = Snapshot.make("DDD", "t", prices=[100, 100, 100, 100, 100],
                         momentum=0.0, volatility=0.1, market_regime="range")
    r = compute_direction(flat)
    if r.direction != "call":
        print("FAIL: flat should default call", r); ok = False

    # Weak-signal SKIP is opt-in: flat tally with the flag on -> skip.
    cfg_skip = DecisionConfig.make({"USE_SKIP_ON_WEAK_SIGNAL": True,
                                    "MIN_DIRECTION_SIGNALS": 2})
    r = compute_direction(flat, cfg_skip)
    if r.direction != "skip" or r.confidence != 0:
        print("FAIL: weak-signal skip", r); ok = False
    if not r.skip_reason:
        print("FAIL: skip_reason missing", r); ok = False

    # Determinism: identical inputs -> identical result.
    if compute_direction(up) != compute_direction(up):
        print("FAIL: non-deterministic"); ok = False

    # Normalized confidence can only lower size vs raw winning count.
    cfg_norm = DecisionConfig.make({"USE_NORMALIZED_CONFIDENCE": True})
    strong = Snapshot.make("EEE", "t", prices=[100, 101, 103, 106, 110],
                           momentum=0.05, volatility=0.2,
                           market_regime="trending")
    raw = compute_direction(strong).confidence
    norm = compute_direction(strong, cfg_norm).confidence
    if norm > raw:
        print("FAIL: normalized > raw", norm, raw); ok = False

    # Junk snapshot never raises.
    try:
        compute_direction(Snapshot.make("X", "t", prices=["a", None]))
    except Exception as exc:  # pragma: no cover
        print("FAIL: raised on junk", exc); ok = False

    print("decision.direction self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
