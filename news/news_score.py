"""
Deterministic, recency-weighted keyword scorer for news articles.

Turns a list of Alpaca news articles into a single signed score in [-1, +1]:
    +1  strongly bullish coverage
     0  neutral / no signal / no coverage
    -1  strongly bearish coverage

This is intentionally a simple, offline, unit-testable heuristic (no LLM/NLP
dependency). It can be swapped out behind this module's ``score_articles`` API
later without touching any caller.
"""

from datetime import datetime, timezone
from typing import List

from .news_config import classify_news

# Finance-oriented keyword lexicons. Matched case-insensitively as substrings of
# the combined headline + summary text.
_BULLISH = [
    "beats", "beat estimates", "tops estimates", "upgrade", "upgraded",
    "outperform", "buy rating", "price target raised", "raises guidance",
    "raised guidance", "record high", "record revenue", "surge", "surges",
    "soar", "soars", "rally", "rallies", "jumps", "gains", "strong demand",
    "better than expected", "approval", "approved", "wins", "awarded",
    "breakthrough", "expansion", "dividend increase", "buyback", "bullish",
]
_BEARISH = [
    "miss", "misses", "missed estimates", "downgrade", "downgraded",
    "underperform", "sell rating", "price target cut", "cuts guidance",
    "cut guidance", "lowered guidance", "plunge", "plunges", "tumble",
    "tumbles", "slump", "slumps", "falls", "drops", "weak demand",
    "worse than expected", "probe", "investigation", "lawsuit", "recall",
    "layoffs", "bankruptcy", "fraud", "warning", "halts", "bearish",
]


def _parse_ts(value):
    """Parse an ISO 8601 timestamp to an aware UTC datetime, or None."""
    if not value:
        return None
    try:
        text = str(value).replace("Z", "+00:00")
        dt = datetime.fromisoformat(text)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt.astimezone(timezone.utc)
    except (TypeError, ValueError):
        return None


def _recency_weight(created_at, lookback_hours: int) -> float:
    """Linear decay from 1.0 (now) to ~0.25 at the lookback edge.

    Articles outside the window (or with no/unparseable timestamp) still get a
    small floor weight so they are not silently dropped.
    """
    dt = _parse_ts(created_at)
    if dt is None or lookback_hours <= 0:
        return 0.5
    age_hours = (datetime.now(timezone.utc) - dt).total_seconds() / 3600.0
    if age_hours <= 0:
        return 1.0
    frac = max(0.0, min(1.0, age_hours / float(lookback_hours)))
    return 1.0 - 0.75 * frac


def _article_polarity(article) -> int:
    """Net keyword polarity for one article: positive, negative, or 0."""
    text = " ".join(
        str((article or {}).get(k, "") or "")
        for k in ("headline", "summary")
    ).lower()
    if not text.strip():
        return 0
    pos = sum(1 for kw in _BULLISH if kw in text)
    neg = sum(1 for kw in _BEARISH if kw in text)
    if pos > neg:
        return 1
    if neg > pos:
        return -1
    return 0


def score_articles(articles: List[dict], config) -> dict:
    """Aggregate articles into a signed news payload.

    Returns:
        {"score": float in [-1, 1], "label": str, "count": int,
         "bullish": int, "bearish": int}
    """
    lookback = getattr(config, "lookback_hours", 24)
    articles = articles or []

    weighted_sum = 0.0
    weight_total = 0.0
    bullish = 0
    bearish = 0

    for article in articles:
        polarity = _article_polarity(article)
        weight = _recency_weight((article or {}).get("created_at"), lookback)
        weight_total += weight
        weighted_sum += polarity * weight
        if polarity > 0:
            bullish += 1
        elif polarity < 0:
            bearish += 1

    if weight_total <= 0:
        score = 0.0
    else:
        score = max(-1.0, min(1.0, weighted_sum / weight_total))

    return {
        "score": round(score, 4),
        "label": classify_news(score),
        "count": len(articles),
        "bullish": bullish,
        "bearish": bearish,
    }
