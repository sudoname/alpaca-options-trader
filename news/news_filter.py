"""
News-based trade decision helpers.

Three pure (no-network) integration points used by smart_trader:

    1. news_direction_vote(news, config)   -> (bull_votes, bear_votes)
       Direction tilt: feeds determine_option_strategy's bull/bear tally.

    2. news_score_multiplier(news, strategy, config) -> float
       Ranking nudge: scales select_best_option's score in
       [1 - rank_weight, 1 + rank_weight] based on agreement with `strategy`.

    3. adjust_trade_by_news(trade_candidate, news, config) -> decision dict
       Gate/size: mirrors adjust_trade_risk_by_sentiment. Can block or shrink a
       trade when news strongly OPPOSES the position direction.

All helpers FAIL OPEN: when news is unavailable/empty/neutral they return the
no-effect value (vote (0,0), multiplier 1.0, allowed-unchanged decision).
"""

import logging
from typing import Optional, Tuple

logger = logging.getLogger(__name__)

# Confidence (0-100) below which an opposing-news gate is allowed to block.
_LOW_CONFIDENCE = 60.0

# Score magnitude below which news is treated as no directional signal.
_MEANINGFUL = 0.15


def _usable(news: Optional[dict]) -> bool:
    return bool(
        news
        and news.get("status") == "available"
        and news.get("score") is not None
    )


def _signed_score(news: Optional[dict]) -> float:
    try:
        return max(-1.0, min(1.0, float(news.get("score"))))
    except (TypeError, ValueError, AttributeError):
        return 0.0


def _direction_sign(strategy: Optional[str]) -> int:
    """+1 for a bullish position (call), -1 for bearish (put), 0 otherwise."""
    s = (strategy or "").lower()
    if s == "call":
        return 1
    if s == "put":
        return -1
    return 0


def news_direction_vote(news: Optional[dict], config) -> Tuple[int, int]:
    """Return (bull_votes, bear_votes) to add to a direction tally."""
    if not _usable(news):
        return (0, 0)
    score = _signed_score(news)
    if abs(score) < _MEANINGFUL:
        return (0, 0)
    votes = max(0, int(getattr(config, "direction_votes", 1)))
    if votes == 0:
        return (0, 0)
    return (votes, 0) if score > 0 else (0, votes)


def news_score_multiplier(news: Optional[dict], strategy: Optional[str],
                          config) -> float:
    """Return a ranking multiplier in [1 - rank_weight, 1 + rank_weight].

    Boosts when news agrees with `strategy` ('call' likes positive news, 'put'
    likes negative news), trims when it disagrees, 1.0 when neutral/unavailable.
    """
    if not _usable(news):
        return 1.0
    sign = _direction_sign(strategy)
    if sign == 0:
        return 1.0
    score = _signed_score(news)
    if abs(score) < _MEANINGFUL:
        return 1.0
    weight = max(0.0, min(1.0, float(getattr(config, "rank_weight", 0.15))))
    # agreement in [-1, 1]: positive when news points the same way as the trade.
    agreement = sign * score
    return round(1.0 + weight * agreement, 4)


def _decision(allowed, original_size, adjusted_size, reason,
              label=None, score=None, size_multiplier=1.0):
    return {
        "allowed": allowed,
        "original_size": original_size,
        "adjusted_size": adjusted_size,
        "size_multiplier": round(size_multiplier, 3),
        "reason": reason,
        "label": label,
        "score": score,
    }


def _scale(size, multiplier):
    """Scale a contract count by a multiplier, keeping int-ness and a 1 floor."""
    try:
        if isinstance(size, int):
            scaled = int(round(size * multiplier))
            return max(1, scaled) if size >= 1 and multiplier > 0 else scaled
        return round(size * multiplier, 4)
    except (TypeError, ValueError):
        return size


def adjust_trade_by_news(trade_candidate: dict, news: Optional[dict],
                         config) -> dict:
    """Gate/size a trade by news that OPPOSES its direction.

    Args:
        trade_candidate: dict with ``size``; optional ``confidence`` (0-100) and
            ``direction`` ("CALL"/"PUT" or "call"/"put").
        news: payload from NewsService.get_news().
        config: NewsConfig (uses ``block_threshold``).

    Returns a decision dict mirroring adjust_trade_risk_by_sentiment. Fail-open:
    agreeing / neutral / unavailable news yields an allowed, unchanged decision.
    """
    original_size = (trade_candidate or {}).get("size", 1)
    confidence = (trade_candidate or {}).get("confidence")
    direction = (trade_candidate or {}).get("direction")

    if not _usable(news):
        return _decision(True, original_size, original_size,
                         "News unavailable; no adjustment applied")

    score = _signed_score(news)
    label = news.get("label") or "Unknown"
    sign = _direction_sign(direction)
    if sign == 0 or abs(score) < _MEANINGFUL:
        return _decision(True, original_size, original_size,
                         f"News neutral ({label}, score {score}); no change",
                         label, score)

    agreement = sign * score  # >0 agrees with the trade, <0 opposes it
    if agreement >= 0:
        return _decision(True, original_size, original_size,
                         f"News agrees ({label}, score {score}); normal sizing",
                         label, score, size_multiplier=1.0)

    # News OPPOSES the trade. Magnitude of opposition = -agreement in (0, 1].
    opposition = -agreement
    block_threshold = float(getattr(config, "block_threshold", 0.6))

    low_conf = confidence is not None and confidence < _LOW_CONFIDENCE
    if opposition >= block_threshold and low_conf:
        return _decision(
            False, original_size, 0,
            (f"News strongly opposes trade ({label}, score {score}); blocking — "
             f"confidence {confidence:.0f}% < {_LOW_CONFIDENCE:.0f}% floor"),
            label, score, size_multiplier=0.0,
        )

    # Mild opposition (or high-confidence): trim size by 25%.
    mult = 0.75
    return _decision(
        True, original_size, _scale(original_size, mult),
        (f"News opposes trade ({label}, score {score}); "
         f"reducing position size by 25%"),
        label, score, size_multiplier=mult,
    )


def summarize_for_log(news: Optional[dict]) -> str:
    """One-line human summary of a news payload for logging."""
    if not news:
        return "news: none"
    status = news.get("status")
    score = news.get("score")
    label = news.get("label", "Unknown")
    count = news.get("count", 0)
    symbol = news.get("symbol", "?")
    cache_note = ""
    if news.get("from_stale_cache"):
        cache_note = " (STALE cache)"
    elif news.get("from_cache"):
        cache_note = " (cached)"
    if status != "available" or score is None:
        return f"news[{symbol}]: {status}{cache_note}"
    return (f"news[{symbol}]: {score} {label} "
            f"({count} articles, {news.get('bullish', 0)}+/{news.get('bearish', 0)}-)"
            f"{cache_note}")
