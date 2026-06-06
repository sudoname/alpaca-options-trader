"""
Alpaca-backed market data provider for the custom Fear & Greed score.

``smart_trader.py`` talks to Alpaca's REST API (not Schwab), so it cannot use
``SchwabMarketDataProvider``. This provider fetches daily close series from
Alpaca's market-data bars endpoint, letting smart_trader compute the full custom
score instead of falling back to CNN only.

Notes / limitations:
    * Uses the free **IEX** feed by default (override with ``feed=``). IEX
      historical depth can be shallower than SIP, so the 125-day momentum
      component may be unavailable on the free tier — in which case it is simply
      reported as unavailable (never faked).
    * Alpaca's stock feed has no index data, so the VIX index symbol ``$VIX.X``
      is mapped to the **VIXY** ETF (a VIX short-term futures proxy). It tracks
      volatility directionally, which is what the percentile score needs.
    * Every fetch is wrapped: a failed/empty request yields an empty series, so
      that component becomes unavailable rather than crashing the score.
"""

import logging
from datetime import datetime, timedelta
from typing import Dict, List

import requests

from .custom_fear_greed import MarketDataProvider

logger = logging.getLogger(__name__)


class AlpacaMarketDataProvider(MarketDataProvider):
    """MarketDataProvider backed by Alpaca's market-data REST API.

    Args:
        data_url: Alpaca data base URL, e.g. ``https://data.alpaca.markets``.
        headers: auth headers (``APCA-API-KEY-ID`` / ``APCA-API-SECRET-KEY``).
        feed: data feed, ``iex`` (free) or ``sip`` (paid). Default ``iex``.
        timeout: per-request timeout in seconds.
    """

    # Non-stock symbols Alpaca's stock feed can't serve, mapped to ETF proxies.
    _SYMBOL_MAP = {"$VIX.X": "VIXY"}

    def __init__(self, data_url: str, headers: Dict[str, str],
                 feed: str = "iex", timeout: int = 10):
        self.data_url = (data_url or "https://data.alpaca.markets").rstrip("/")
        self.headers = headers or {}
        self.feed = feed
        self.timeout = timeout

    def get_close_series(self, symbol: str, days: int) -> List[float]:
        mapped = self._SYMBOL_MAP.get(symbol, symbol)
        # Pad the window so weekends/holidays still yield ~``days`` trading days.
        lookback = max(int(days * 1.6) + 10, days + 10)
        end_time = datetime.now()
        start_time = end_time - timedelta(days=lookback)

        try:
            response = requests.get(
                f"{self.data_url}/v2/stocks/{mapped}/bars",
                headers=self.headers,
                params={
                    "timeframe": "1Day",
                    "start": start_time.strftime("%Y-%m-%d"),
                    "end": end_time.strftime("%Y-%m-%d"),
                    "limit": 10000,
                    "feed": self.feed,
                },
                timeout=self.timeout,
            )
        except requests.exceptions.RequestException as exc:
            logger.warning("Alpaca bars request failed for %s: %s", mapped, exc)
            return []
        except Exception as exc:
            logger.warning("Alpaca bars unexpected error for %s: %s", mapped, exc)
            return []

        if getattr(response, "status_code", None) != 200:
            logger.warning("Alpaca bars HTTP %s for %s",
                           getattr(response, "status_code", "?"), mapped)
            return []

        try:
            data = response.json()
        except Exception as exc:
            logger.warning("Alpaca bars invalid JSON for %s: %s", mapped, exc)
            return []

        bars = data.get("bars", []) if isinstance(data, dict) else []
        closes = []
        for bar in bars:
            close = bar.get("c")
            if close is not None:
                closes.append(float(close))

        if days and len(closes) > days:
            closes = closes[-days:]
        return closes
