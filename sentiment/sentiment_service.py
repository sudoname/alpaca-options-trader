"""
Sentiment service — orchestrates the CNN scraper, the custom score, and caching.

This is the single entry point the rest of the bot should use:

    from sentiment import SentimentService, SchwabMarketDataProvider

    service = SentimentService(SchwabMarketDataProvider(client))
    sentiment = service.get_sentiment()
    # -> {"cnn_score": {...}, "custom_score": {...},
    #     "primary_score": {...}, "primary_source": "blend", "timestamp": ...}

By default the primary score is a weighted BLEND of the custom score and CNN
(SENTIMENT_PRIMARY_SOURCE=blend, custom weighted by SENTIMENT_BLEND_CUSTOM_WEIGHT,
default 0.5). Set SENTIMENT_PRIMARY_SOURCE=custom or =cnn to use a single source.
Blend degrades to whichever single source is available. The service never raises
— on total failure it returns a payload with ``primary_score.status == "error"``
so callers can no-op safely.
"""

import logging
from datetime import datetime, timezone
from typing import Optional

from .cnn_fear_greed import fetch_cnn_fear_greed
from .custom_fear_greed import (
    MarketDataProvider,
    SchwabMarketDataProvider,
    compute_custom_fear_greed,
)
from .sentiment_cache import SentimentCache
from .sentiment_config import SentimentConfig, classify_score

logger = logging.getLogger(__name__)


def _is_available(score: Optional[dict]) -> bool:
    return bool(
        score
        and score.get("status") == "available"
        and score.get("score") is not None
    )


def _blend_scores(cnn: Optional[dict], custom: Optional[dict], w_custom: float):
    """Weighted blend of the custom and CNN scores into a 'blend' primary.

    When both sources are available, return a weighted-average payload (custom
    weighted by ``w_custom``, CNN by ``1 - w_custom``). When only one is
    available, return that source unchanged so the blend degrades gracefully.

    Returns ``(primary_payload, primary_source)`` or ``(None, None)`` if neither
    source is usable.
    """
    cnn_ok = _is_available(cnn)
    custom_ok = _is_available(custom)

    if cnn_ok and custom_ok:
        w_custom = max(0.0, min(1.0, w_custom))
        w_cnn = 1.0 - w_custom
        cs = float(custom["score"])
        xs = float(cnn["score"])
        blended = cs * w_custom + xs * w_cnn
        return {
            "source": "blend",
            "status": "available",
            "score": round(blended, 2),
            "classification": classify_score(blended),
            "components": [
                {"name": "custom", "score": cs, "weight": round(w_custom, 3)},
                {"name": "cnn", "score": xs, "weight": round(w_cnn, 3)},
            ],
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }, "blend"

    if custom_ok:
        return custom, "custom"
    if cnn_ok:
        return cnn, "cnn"
    return None, None


def _disabled_payload(reason: str) -> dict:
    return {
        "cnn_score": None,
        "custom_score": None,
        "primary_score": {
            "source": "none",
            "status": "disabled",
            "score": None,
            "classification": "Unknown",
            "reason": reason,
        },
        "primary_source": None,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


class SentimentService:
    def __init__(self,
                 provider: Optional[MarketDataProvider] = None,
                 config: Optional[SentimentConfig] = None,
                 cache: Optional[SentimentCache] = None):
        self.config = config or SentimentConfig.from_env()
        self.provider = provider
        self.cache = cache or SentimentCache(
            cache_file=self.config.cache_file,
            ttl_minutes=self.config.cache_minutes,
        )

    def _compute(self) -> dict:
        """Fetch both sources and assemble the combined payload (no cache)."""
        cnn = None
        if self.config.use_cnn:
            cnn = fetch_cnn_fear_greed(self.config)

        custom = None
        if self.config.use_custom:
            if self.provider is not None:
                custom = compute_custom_fear_greed(self.provider, self.config)
            else:
                logger.warning(
                    "Custom sentiment enabled but no market data provider supplied"
                )
                custom = {
                    "source": "custom",
                    "status": "error",
                    "score": None,
                    "classification": "Unknown",
                    "error": "no market data provider",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                }

        # Select the primary score. "blend" combines both sources; "custom"/"cnn"
        # pick that source and fall back to the other if it is missing/errored.
        primary_source = self.config.primary_source
        primary = None
        if primary_source == "blend":
            primary, primary_source = _blend_scores(
                cnn, custom, self.config.blend_custom_weight
            )
        elif primary_source == "cnn":
            primary = cnn if _is_available(cnn) else None
            if primary is None and _is_available(custom):
                primary = custom
                primary_source = "custom"
        else:  # default: custom primary
            primary = custom if _is_available(custom) else None
            if primary is None and _is_available(cnn):
                primary = cnn
                primary_source = "cnn"

        if primary is None:
            primary = {
                "source": "none",
                "status": "error",
                "score": None,
                "classification": "Unknown",
                "error": "no sentiment source available",
            }
            primary_source = None

        return {
            "cnn_score": cnn,
            "custom_score": custom,
            "primary_score": primary,
            "primary_source": primary_source,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_sentiment(self, force_refresh: bool = False) -> dict:
        """Return the combined sentiment payload, using the cache when possible.

        Cache policy:
            * Fresh cache (within TTL)  -> returned immediately.
            * Otherwise recompute. On success, cache and return.
            * If recompute yields no usable primary score but a stale cache
              exists, return the stale cache (flagged ``from_stale_cache``).
        """
        if not self.config.enabled:
            return _disabled_payload("SENTIMENT_ENABLED is false")

        if not force_refresh:
            fresh = self.cache.get_fresh()
            if fresh is not None:
                fresh = dict(fresh)
                fresh["from_cache"] = True
                return fresh

        try:
            payload = self._compute()
        except Exception as exc:  # belt-and-suspenders; sub-calls already guard
            logger.warning("Sentiment compute crashed: %s", exc)
            payload = None

        primary_ok = bool(
            payload
            and payload.get("primary_score")
            and payload["primary_score"].get("status") == "available"
        )

        if primary_ok:
            self.cache.set(payload)
            payload["from_cache"] = False
            return payload

        # Refresh failed — fall back to stale cache if we have one.
        stale = self.cache.get_stale()
        if stale is not None:
            logger.warning(
                "Sentiment refresh failed; serving stale cache as fallback"
            )
            stale = dict(stale)
            stale["from_cache"] = True
            stale["from_stale_cache"] = True
            return stale

        # Nothing usable anywhere — return whatever we computed (error payload).
        if payload is None:
            payload = {
                "cnn_score": None,
                "custom_score": None,
                "primary_score": {
                    "source": "none",
                    "status": "error",
                    "score": None,
                    "classification": "Unknown",
                    "error": "sentiment unavailable",
                },
                "primary_source": None,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
        payload["from_cache"] = False
        return payload
