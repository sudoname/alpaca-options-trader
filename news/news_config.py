"""
Configuration for the per-ticker News signal module.

Follows the project's existing convention (mirrors ``sentiment/sentiment_config``):
read from environment variables with sensible defaults. No external config object
or framework is used.
"""

import os
from dataclasses import dataclass


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# News score is a signed sentiment in [-1, +1]. Bands map |score| magnitude to a
# directional label. The thresholds are deliberately conservative so weak/mixed
# coverage stays "Neutral" and has no effect on trading.
def classify_news(score) -> str:
    """Map a signed news score in [-1, +1] to a directional label.

    Returns "Unknown" when score is None / not a number.
    """
    if score is None:
        return "Unknown"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    s = max(-1.0, min(1.0, s))
    if s >= 0.5:
        return "Very Bullish"
    if s >= 0.15:
        return "Bullish"
    if s <= -0.5:
        return "Very Bearish"
    if s <= -0.15:
        return "Bearish"
    return "Neutral"


@dataclass
class NewsConfig:
    """Resolved news configuration.

    Use ``NewsConfig.from_env()`` to build one from environment variables.
    """

    enabled: bool = True
    lookback_hours: int = 24
    cache_minutes: int = 15
    max_articles: int = 50

    # Ranking nudge: select_best_option's score is multiplied by a factor in
    # ``[1 - rank_weight, 1 + rank_weight]`` depending on news agreement.
    rank_weight: float = 0.15

    # Direction tilt: how many bull/bear votes a meaningful news score adds to
    # determine_option_strategy's tally.
    direction_votes: int = 1

    # Gate: only block a trade when news opposes the position with at least this
    # magnitude (|score| >= block_threshold) AND confidence is low.
    block_threshold: float = 0.6

    # Alpaca news endpoint (Benzinga-backed). Same host as market data.
    news_url: str = "https://data.alpaca.markets"
    news_timeout: int = 10

    cache_file: str = "news_cache.json"

    @classmethod
    def from_env(cls) -> "NewsConfig":
        return cls(
            enabled=_get_bool("NEWS_ENABLED", True),
            lookback_hours=_get_int("NEWS_LOOKBACK_HOURS", 24),
            cache_minutes=_get_int("NEWS_CACHE_MINUTES", 15),
            max_articles=_get_int("NEWS_MAX_ARTICLES", 50),
            rank_weight=_get_float("NEWS_RANK_WEIGHT", 0.15),
            direction_votes=_get_int("NEWS_DIRECTION_VOTES", 1),
            block_threshold=_get_float("NEWS_BLOCK_THRESHOLD", 0.6),
            news_url=os.getenv("NEWS_URL", "https://data.alpaca.markets"),
            news_timeout=_get_int("NEWS_TIMEOUT", 10),
            cache_file=os.getenv("NEWS_CACHE_FILE", "news_cache.json"),
        )
