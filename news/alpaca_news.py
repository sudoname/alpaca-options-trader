"""
Alpaca News API fetcher (Benzinga-backed).

Fetches recent headlines for a single symbol from Alpaca's news endpoint using
the same auth headers smart_trader already builds. Mirrors the fail-open idiom in
``sentiment/alpaca_provider``: any RequestException / non-200 / bad JSON yields an
empty list so a news failure never blocks or crashes a trade.

Endpoint:
    GET {news_url}/v1beta1/news?symbols=SYM&start=<ISO>&limit=<N>&sort=desc
"""

import logging
from datetime import datetime, timedelta, timezone
from typing import Dict, List

import requests

logger = logging.getLogger(__name__)


def fetch_alpaca_news(symbol: str, config, headers: Dict[str, str]) -> List[dict]:
    """Return a list of article dicts for ``symbol`` (most recent first).

    Each article is the raw Alpaca payload, including at least ``headline``,
    ``summary``, and ``created_at`` (ISO 8601). Returns ``[]`` on any failure.
    """
    if not symbol:
        return []

    news_url = (getattr(config, "news_url", None)
                or "https://data.alpaca.markets").rstrip("/")
    start = (datetime.now(timezone.utc)
             - timedelta(hours=getattr(config, "lookback_hours", 24)))

    try:
        response = requests.get(
            f"{news_url}/v1beta1/news",
            headers=headers or {},
            params={
                "symbols": symbol,
                "start": start.isoformat(),
                "limit": min(50, getattr(config, "max_articles", 50)),
                "sort": "desc",
            },
            timeout=getattr(config, "news_timeout", 10),
        )
    except requests.exceptions.RequestException as exc:
        logger.warning("Alpaca news request failed for %s: %s", symbol, exc)
        return []
    except Exception as exc:
        logger.warning("Alpaca news unexpected error for %s: %s", symbol, exc)
        return []

    if getattr(response, "status_code", None) != 200:
        logger.warning("Alpaca news HTTP %s for %s",
                       getattr(response, "status_code", "?"), symbol)
        return []

    try:
        data = response.json()
    except Exception as exc:
        logger.warning("Alpaca news invalid JSON for %s: %s", symbol, exc)
        return []

    articles = data.get("news", []) if isinstance(data, dict) else []
    if not isinstance(articles, list):
        return []

    max_articles = getattr(config, "max_articles", 50)
    if max_articles and len(articles) > max_articles:
        articles = articles[:max_articles]
    return articles
