"""
File-backed, per-symbol news cache.

Mirrors ``sentiment/sentiment_cache`` but keyed by symbol, since news is
ticker-specific. Headlines change slowly relative to a scan loop, so we reuse the
last computed payload within a TTL window to keep Alpaca news API calls bounded.

Behavior (per symbol):
    * ``get_fresh(symbol)``         -> cached payload if within TTL, else None.
    * ``get_stale(symbol)``         -> last cached payload regardless of age.
    * ``set(symbol, payload)``      -> persist payload with the current timestamp.

The cache never raises: a corrupt/missing file behaves as a miss.
JSON shape: {symbol: {"cached_at": ISO, "payload": {...}}}
"""

import json
import logging
import os
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)


class NewsCache:
    def __init__(self, cache_file: str = "news_cache.json",
                 ttl_minutes: int = 15):
        self.cache_file = cache_file
        self.ttl_seconds = max(0, ttl_minutes) * 60

    def _read_all(self) -> dict:
        if not os.path.exists(self.cache_file):
            return {}
        try:
            with open(self.cache_file, "r", encoding="utf-8") as fh:
                data = json.load(fh)
            if isinstance(data, dict):
                return data
        except Exception as exc:
            logger.warning("News cache read failed: %s", exc)
        return {}

    def _entry(self, symbol: str) -> Optional[dict]:
        entry = self._read_all().get(symbol)
        if isinstance(entry, dict) and "cached_at" in entry and "payload" in entry:
            return entry
        return None

    def _age_seconds(self, entry: dict) -> Optional[float]:
        try:
            cached_at = datetime.fromisoformat(entry["cached_at"])
            if cached_at.tzinfo is None:
                cached_at = cached_at.replace(tzinfo=timezone.utc)
            return (datetime.now(timezone.utc) - cached_at).total_seconds()
        except Exception:
            return None

    def get_fresh(self, symbol: str) -> Optional[dict]:
        """Return the cached payload for ``symbol`` if within TTL, else None."""
        entry = self._entry(symbol)
        if not entry:
            return None
        age = self._age_seconds(entry)
        if age is None:
            return None
        if age <= self.ttl_seconds:
            logger.debug("News cache hit for %s (age %.0fs / ttl %ds)",
                         symbol, age, self.ttl_seconds)
            return entry["payload"]
        return None

    def get_stale(self, symbol: str) -> Optional[dict]:
        """Return the last cached payload for ``symbol`` regardless of age."""
        entry = self._entry(symbol)
        if not entry:
            return None
        return entry.get("payload")

    def set(self, symbol: str, payload: dict) -> None:
        """Persist ``payload`` for ``symbol`` stamped with the current UTC time."""
        data = self._read_all()
        data[symbol] = {
            "cached_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
        }
        try:
            with open(self.cache_file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2)
        except Exception as exc:
            logger.warning("News cache write failed: %s", exc)
