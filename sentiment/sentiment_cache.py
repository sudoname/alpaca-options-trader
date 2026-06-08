"""
File-backed sentiment cache.

Sentiment data changes slowly (CNN updates a few times a day; the custom score
is built from daily candles), so we cache the last computed result and reuse it
within a TTL window. This avoids hammering the CNN endpoint and re-fetching
price history on every trade decision.

Behavior:
    * ``get_fresh()``  -> cached payload if within TTL, else None.
    * ``get_stale()``  -> the last cached payload regardless of age, or None.
    * ``set(payload)`` -> persist payload with the current timestamp.

The cache never raises: a corrupt/missing file simply behaves as a miss.
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class SentimentCache:
    def __init__(self, cache_file: str = "sentiment_cache.json",
                 ttl_minutes: int = 15):
        self.cache_file = cache_file
        self.ttl_seconds = max(0, ttl_minutes) * 60

    def _read(self) -> Optional[dict]:
        if not os.path.exists(self.cache_file):
            return None
        try:
            with open(self.cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict) and "cached_at" in data and "payload" in data:
                return data
        except Exception as exc:
            logger.warning("Sentiment cache read failed: %s", exc)
        return None

    def _age_seconds(self, entry: dict) -> Optional[float]:
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - cached_at).total_seconds()
        except Exception:
            return None

    def get_fresh(self) -> Optional[dict]:
        """Return the cached payload if it is within the TTL window, else None."""
        entry = self._read()
        if not entry:
            return None
        age = self._age_seconds(entry)
        if age is None:
            return None
        # A non-positive TTL means "never fresh" deterministically: without this
        # guard a same-instant set()+get_fresh() can read age==0 and 0<=0 would
        # report a hit, making the ttl=0 contract timing-dependent.
        if self.ttl_seconds <= 0:
            return None
        if age <= self.ttl_seconds:
            logger.debug("Sentiment cache hit (age %.0fs / ttl %ds)",
                         age, self.ttl_seconds)
            return entry["payload"]
        logger.debug("Sentiment cache stale (age %.0fs > ttl %ds)",
                     age, self.ttl_seconds)
        return None

    def get_stale(self) -> Optional[dict]:
        """Return the last cached payload regardless of age (or None)."""
        entry = self._read()
        if not entry:
            return None
        return entry.get("payload")

    def set(self, payload: dict) -> None:
        """Persist ``payload`` stamped with the current UTC time."""
        entry = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as fh:
                json.dump(entry, fh, indent=2)
        except Exception as exc:
            logger.warning("Sentiment cache write failed: %s", exc)
