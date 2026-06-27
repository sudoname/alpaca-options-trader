"""
Static ticker -> GICS-sector map for portfolio analytics (offline, no network).

Covers the underlyings currently held plus the scheduler base symbols. Broad
market / index / bond ETFs are bucketed as ``Index ETF`` / ``Fixed Income ETF``
(they are diversification anchors, not single-sector exposure). Anything unknown
resolves to ``UNKNOWN`` so callers never KeyError.

This is reference data only — it never trades, prices or touches the network.
"""

from typing import Dict

UNKNOWN = "Unknown"

# GICS-style sectors. ETFs get pseudo-sectors so they aggregate sensibly.
SECTORS: Dict[str, str] = {
    # Information Technology
    "AAPL": "Information Technology", "ADBE": "Information Technology",
    "CRM": "Information Technology", "CSCO": "Information Technology",
    "ESTC": "Information Technology", "IONQ": "Information Technology",
    "NVDA": "Information Technology", "OKTA": "Information Technology",
    "ORCL": "Information Technology", "PLTR": "Information Technology",
    "SAP": "Information Technology", "ZM": "Information Technology",
    "ZS": "Information Technology",
    # Communication Services
    "DIS": "Communication Services", "NFLX": "Communication Services",
    "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "META": "Communication Services", "T": "Communication Services",
    # Consumer Discretionary
    "AMZN": "Consumer Discretionary", "F": "Consumer Discretionary",
    "GM": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "NKE": "Consumer Discretionary", "RIVN": "Consumer Discretionary",
    "TSLA": "Consumer Discretionary",
    # Consumer Staples
    "MO": "Consumer Staples", "PG": "Consumer Staples",
    "TGT": "Consumer Staples", "WMT": "Consumer Staples",
    "KO": "Consumer Staples", "COST": "Consumer Staples",
    # Health Care
    "ABT": "Health Care", "CVS": "Health Care", "GSK": "Health Care",
    "JNJ": "Health Care", "MDT": "Health Care", "MRK": "Health Care",
    "PFE": "Health Care", "SNY": "Health Care", "UNH": "Health Care",
    # Financials
    "BAC": "Financials", "C": "Financials", "SCHW": "Financials",
    "SOFI": "Financials", "V": "Financials", "WFC": "Financials",
    "MA": "Financials", "JPM": "Financials", "GS": "Financials",
    # Industrials
    "BA": "Industrials", "CARR": "Industrials", "LYFT": "Industrials",
    "MMM": "Industrials", "RTX": "Industrials", "UBER": "Industrials",
    "GE": "Industrials",
    # Energy
    "COP": "Energy", "CVX": "Energy", "ENB": "Energy", "XOM": "Energy",
    # Utilities
    "SO": "Utilities", "NEE": "Utilities", "DUK": "Utilities",
    # Real Estate
    "SPG": "Real Estate",
    # Index / broad-market ETFs
    "SPY": "Index ETF", "QQQ": "Index ETF", "DIA": "Index ETF",
    "IWM": "Index ETF",
    # Sector / fixed-income ETFs
    "XLE": "Energy", "TLT": "Fixed Income ETF",
}


def sector_of(ticker: str) -> str:
    """GICS sector for ``ticker`` (case-insensitive); UNKNOWN if unmapped."""
    if not isinstance(ticker, str):
        return UNKNOWN
    return SECTORS.get(ticker.strip().upper(), UNKNOWN)


def coverage(tickers) -> dict:
    """{'mapped', 'unmapped', 'unknown_tickers'} for a ticker iterable."""
    unknown = sorted({t for t in tickers if sector_of(t) == UNKNOWN})
    total = len({t for t in tickers})
    return {
        "mapped": total - len(unknown),
        "unmapped": len(unknown),
        "unknown_tickers": unknown,
    }


def _self_test() -> int:
    ok = True
    if sector_of("aapl") != "Information Technology":
        print("FAIL: case-insensitive lookup"); ok = False
    if sector_of("ZZZZ") != UNKNOWN:
        print("FAIL: unknown should map to UNKNOWN"); ok = False
    if sector_of(None) != UNKNOWN:
        print("FAIL: None should map to UNKNOWN"); ok = False
    # Every currently-held underlying must be mapped.
    held = ["AAPL", "ABT", "ADBE", "AMZN", "BA", "BAC", "C", "CARR", "COP",
            "CRM", "CSCO", "CVS", "CVX", "DIA", "DIS", "ENB", "ESTC", "F",
            "GM", "GSK", "HD", "IONQ", "IWM", "JNJ", "LYFT", "MDT", "MMM",
            "MO", "MRK", "NFLX", "NKE", "NVDA", "OKTA", "ORCL", "PFE", "PG",
            "PLTR", "RIVN", "RTX", "SAP", "SCHW", "SNY", "SO", "SOFI", "SPG",
            "TGT", "TLT", "UBER", "V", "WFC", "WMT", "XLE", "XOM", "ZM", "ZS"]
    cov = coverage(held)
    if cov["unmapped"] != 0:
        print("FAIL: unmapped held tickers", cov["unknown_tickers"]); ok = False
    print("sector_map self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
