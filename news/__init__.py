"""
Per-ticker News signal module.

Public API:

    from news import (
        NewsConfig,
        NewsService,
        news_direction_vote,
        news_score_multiplier,
        adjust_trade_by_news,
        summarize_for_log,
    )

News is fetched per symbol from Alpaca's news endpoint (Benzinga-backed), scored
with a deterministic recency-weighted keyword scorer, and cached per symbol.
Every entry point is designed to FAIL OPEN: a news failure must never crash or
block the trading bot — it simply has no effect on the decision.
"""

from .news_config import NewsConfig, classify_news
from .alpaca_news import fetch_alpaca_news
from .news_score import score_articles
from .news_cache import NewsCache
from .news_service import NewsService
from .news_filter import (
    news_direction_vote,
    news_score_multiplier,
    adjust_trade_by_news,
    summarize_for_log,
)

__all__ = [
    "NewsConfig",
    "classify_news",
    "fetch_alpaca_news",
    "score_articles",
    "NewsCache",
    "NewsService",
    "news_direction_vote",
    "news_score_multiplier",
    "adjust_trade_by_news",
    "summarize_for_log",
]
