"""Kalshi series -> Oracle underlying mapping (Oracle 4.0.2 P1-6).

Deterministic, explicit table -- no fuzzy matching.  A Kalshi market
ticker looks like ``SERIES-EXPIRY-STRIKE`` (e.g. ``INXD-26NOV21-B6050``,
``KXNASDAQ100U-26NOV21-T24800``); the series is the part before the
first ``-``.

Two layers:
  * ``INDEX_BY_SERIES``: Kalshi series -> cash index symbol (SPX/NDX);
  * ``PROXY_BY_INDEX``: cash index -> tradable Oracle underlying proxy
    (SPY/QQQ) so Kalshi index probabilities can be joined onto the
    equity-ETF bars/decisions the rest of the pipeline uses.

Unknown series map to nothing (empty result), never to a guess.
"""
from __future__ import annotations

from typing import Dict, FrozenSet, Optional

#: Kalshi series prefixes for the two index families we research.
#: Historical (INX*/NASDAQ100*) and current (KX-prefixed) spellings.
INDEX_BY_SERIES: Dict[str, str] = {
    # S&P 500
    "INX": "SPX",
    "INXD": "SPX",
    "INXU": "SPX",
    "INXZ": "SPX",
    "KXINX": "SPX",
    "KXINXD": "SPX",
    "KXINXU": "SPX",
    "KXINXY": "SPX",
    # Nasdaq-100
    "NASDAQ100": "NDX",
    "NASDAQ100D": "NDX",
    "NASDAQ100U": "NDX",
    "KXNASDAQ100": "NDX",
    "KXNASDAQ100D": "NDX",
    "KXNASDAQ100U": "NDX",
    "KXNASDAQ100Y": "NDX",
}

#: cash index -> tradable ETF proxy used as the Oracle underlying key.
PROXY_BY_INDEX: Dict[str, str] = {
    "SPX": "SPY",
    "NDX": "QQQ",
}

#: approximate index/ETF level ratios for strike translation.  Kalshi
#: strikes are on the cash index; Oracle spots are ETF prices.  The ratio
#: is re-estimated per join when both levels are known; these are only
#: documented fallbacks and deliberately conservative (None disables).
DEFAULT_SERIES_FOR_UNDERLYING: Dict[str, tuple] = {
    "SPY": ("KXINXU", "KXINX", "INXU", "INX", "INXD", "KXINXD"),
    "QQQ": ("KXNASDAQ100U", "KXNASDAQ100", "NASDAQ100U", "NASDAQ100",
            "NASDAQ100D", "KXNASDAQ100D"),
}


def series_of(ticker: str) -> str:
    """Extract the series prefix of a Kalshi market/event ticker."""
    t = str(ticker or "").strip().upper()
    return t.split("-", 1)[0] if t else ""


def index_for_ticker(ticker: str) -> Optional[str]:
    """Cash index symbol (SPX/NDX) for a Kalshi ticker, or None."""
    return INDEX_BY_SERIES.get(series_of(ticker))


def underlying_for_ticker(ticker: str) -> Optional[str]:
    """Oracle underlying proxy (SPY/QQQ) for a Kalshi ticker, or None."""
    idx = index_for_ticker(ticker)
    return PROXY_BY_INDEX.get(idx) if idx else None


def series_for_underlying(underlying: str) -> FrozenSet[str]:
    """All Kalshi series whose markets map onto ``underlying``."""
    u = str(underlying or "").strip().upper()
    out = set()
    for series, idx in INDEX_BY_SERIES.items():
        if PROXY_BY_INDEX.get(idx) == u or idx == u:
            out.add(series)
    return frozenset(out)
