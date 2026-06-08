"""
News service — orchestrates the Alpaca news fetch, the keyword scorer, and the
per-symbol cache.

Single entry point the rest of the bot uses:

    from news import NewsService, NewsConfig
    service = NewsService(NewsConfig.from_env(), data_url, headers)
    news = service.get_news("SPY")
    # -> {"symbol": "SPY", "score": 0.42, "label": "Bullish", "count": 12,
    #     "bullish": 7, "bearish": 1, "status": "available", "timestamp": ...}

Cache policy mirrors SentimentService:
    * Fresh cache (within TTL) -> returned immediately.
    * Otherwise refetch + rescore. On success, cache and return.
    * If refresh yields no usable payload but a stale cache exists, serve stale.
The service NEVER raises — on total failure it returns a payload with
``status == "error"`` so callers can no-op safely (fail-open).
"""

import logging
from datetime import datetime, timezone
from typing import Dict, Optional

from .alpaca_news import fetch_alpaca_news
from .news_cache import NewsCache
from .news_config import NewsConfig
from .news_score import score_articles

logger = logging.getLogger(__name__)


class NewsService:
    def __init__(self,
                 config: Optional[NewsConfig] = None,
                 data_url: Optional[str] = None,
                 headers: Optional[Dict[str, str]] = None,
                 cache: Optional[NewsCache] = None):
        self.config = config or NewsConfig.from_env()
        # data_url is accepted for symmetry with sentiment's provider; the news
        # host actually lives on the config (config.news_url). When a data_url is
        # supplied we honor it so paper/live hosts can be overridden.
        if data_url:
            self.config.news_url = data_url
        self.headers = headers or {}
        self.cache = cache or NewsCache(
            cache_file=self.config.cache_file,
            ttl_minutes=self.config.cache_minutes,
        )

    def _compute(self, symbol: str) -> dict:
        """Fetch + score (no cache)."""
        articles = fetch_alpaca_news(symbol, self.config, self.headers)
        scored = score_articles(articles, self.config)
        return {
            "symbol": symbol,
            "score": scored["score"],
            "label": scored["label"],
            "count": scored["count"],
            "bullish": scored["bullish"],
            "bearish": scored["bearish"],
            "status": "available" if scored["count"] > 0 else "empty",
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_news(self, symbol: str, force_refresh: bool = False) -> dict:
        """Return the news payload for ``symbol``, using the cache when possible."""
        if not self.config.enabled:
            return {
                "symbol": symbol, "score": 0.0, "label": "Unknown", "count": 0,
                "bullish": 0, "bearish": 0, "status": "disabled",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        if not force_refresh:
            fresh = self.cache.get_fresh(symbol)
            if fresh is not None:
                fresh = dict(fresh)
                fresh["from_cache"] = True
                return fresh

        try:
            payload = self._compute(symbol)
        except Exception as exc:  # belt-and-suspenders; sub-calls already guard
            logger.warning("News compute crashed for %s: %s", symbol, exc)
            payload = None

        # "available" means we actually got coverage. "empty" (no articles) is a
        # valid, cacheable result that has no trading effect.
        if payload is not None and payload.get("status") in ("available", "empty"):
            self.cache.set(symbol, payload)
            payload["from_cache"] = False
            return payload

        # Refresh failed — fall back to stale cache if we have one.
        stale = self.cache.get_stale(symbol)
        if stale is not None:
            logger.warning("News refresh failed for %s; serving stale cache", symbol)
            stale = dict(stale)
            stale["from_cache"] = True
            stale["from_stale_cache"] = True
            return stale

        # Nothing usable anywhere — error payload (fail-open at the caller).
        return {
            "symbol": symbol, "score": 0.0, "label": "Unknown", "count": 0,
            "bullish": 0, "bearish": 0, "status": "error",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "from_cache": False,
        }
