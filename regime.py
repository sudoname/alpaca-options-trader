"""
Point-in-time market regime label.

`detect_regime(market_view, symbol="SPY")` returns the SAME volatile/trending/
ranging classification smart_trader uses (see smart_trader.get_market_regime),
but it is sourced entirely through a `MarketView`, so there is no `datetime.now()`
inside the feature path and a backtest at a fixed `as_of` is deterministic.

Reproduced rules (kept identical so the label means the same thing live and in
backtest):
  * realized_vol = stdev(daily returns) * sqrt(252), capped to [0.10, 0.80];
  * momentum is the smoothed short-window move from the last 5 closes;
  * regime = "volatile" if vol > 0.30, else "trending" if |momentum| > 0.05,
    else "ranging".

This module only LABELS the market. It does not pick trades or place orders.
"""

import math
from typing import Dict, List, Optional

# Thresholds mirror smart_trader.get_market_regime / calculate_volatility.
VOL_FLOOR = 0.10
VOL_CAP = 0.80
VOLATILE_VOL = 0.30
TRENDING_MOMENTUM = 0.05
DEFAULT_VOL_LOOKBACK = 10
TREND_FLAT_BAND = 0.01


def _realized_vol(closes: List[float]) -> float:
    """Annualized stdev of daily returns, capped — matches calculate_volatility."""
    if len(closes) < 2:
        return 0.20
    returns = [(closes[i] - closes[i - 1]) / closes[i - 1]
               for i in range(1, len(closes)) if closes[i - 1]]
    if not returns:
        return 0.20
    mean_r = sum(returns) / len(returns)
    var = sum((r - mean_r) ** 2 for r in returns) / len(returns)
    vol = math.sqrt(var) * math.sqrt(252)
    return min(max(vol, VOL_FLOOR), VOL_CAP)


def _momentum(closes: List[float]) -> float:
    """Smoothed short-window momentum — matches calculate_momentum."""
    if len(closes) < 3:
        return 0.0
    recent = (closes[-1] - closes[-3]) / closes[-3] if closes[-3] else 0.0
    if len(closes) >= 5:
        recent_avg = sum(closes[-3:]) / 3
        older_avg = sum(closes[-5:-2]) / 3
        trend = (recent_avg - older_avg) / older_avg if older_avg else 0.0
        return (recent + trend) / 2
    return recent


def detect_regime(market_view, symbol: str = "SPY",
                  vol_lookback: int = DEFAULT_VOL_LOOKBACK) -> Dict:
    """Classify the regime knowable at `market_view.as_of`.

    Returns {regime, trend, realized_vol, momentum, as_of, n_bars}. Falls back
    to a neutral "ranging"/"flat" label when there is too little known data,
    rather than guessing.
    """
    bars = market_view.daily_bars(symbol, vol_lookback)
    closes = [b.c for b in bars]
    as_of = market_view.as_of.isoformat()

    if len(closes) < 2:
        return {
            "regime": "ranging",
            "trend": "flat",
            "realized_vol": 0.20,
            "momentum": 0.0,
            "as_of": as_of,
            "n_bars": len(closes),
        }

    vol = _realized_vol(closes)
    mom = _momentum(closes)

    if vol > VOLATILE_VOL:
        regime = "volatile"
    elif abs(mom) > TRENDING_MOMENTUM:
        regime = "trending"
    else:
        regime = "ranging"

    if mom > TREND_FLAT_BAND:
        trend = "up"
    elif mom < -TREND_FLAT_BAND:
        trend = "down"
    else:
        trend = "flat"

    return {
        "regime": regime,
        "trend": trend,
        "realized_vol": round(vol, 4),
        "momentum": round(mom, 4),
        "as_of": as_of,
        "n_bars": len(closes),
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime
    from market_view import HistoricalMarketView, make_bar

    ok = True

    # Strong, steady uptrend with small daily steps -> trending/up, low vol.
    up = [make_bar(f"2026-01-{i+1:02d}", 100 + i, 100 + i + 0.5, 99.5 + i, 100.6 + i)
          for i in range(12)]
    mv_up = HistoricalMarketView(datetime(2026, 1, 31, 16, 0), daily={"SPY": up})
    r_up = detect_regime(mv_up, "SPY")
    if r_up["trend"] != "up":
        print("FAIL: steady rise should be trend=up", r_up); ok = False
    if r_up["momentum"] <= 0:
        print("FAIL: rising series should have positive momentum", r_up); ok = False

    # Large alternating swings -> high realized vol -> volatile.
    swing = []
    base = 100.0
    for i in range(12):
        c = base * (1.08 if i % 2 == 0 else 0.93)
        swing.append(make_bar(f"2026-02-{i+1:02d}", base, max(base, c), min(base, c), c))
        base = c
    mv_sw = HistoricalMarketView(datetime(2026, 2, 28, 16, 0), daily={"SPY": swing})
    r_sw = detect_regime(mv_sw, "SPY")
    if r_sw["regime"] != "volatile":
        print("FAIL: big swings should be volatile", r_sw); ok = False

    # Flat, tiny noise -> ranging/flat, vol at the floor.
    flat = [make_bar(f"2026-03-{i+1:02d}", 50.0, 50.1, 49.9, 50.0 + (0.01 if i % 2 else -0.01))
            for i in range(12)]
    mv_flat = HistoricalMarketView(datetime(2026, 3, 31, 16, 0), daily={"SPY": flat})
    r_flat = detect_regime(mv_flat, "SPY")
    if r_flat["regime"] != "ranging":
        print("FAIL: flat tape should be ranging", r_flat); ok = False
    if r_flat["trend"] != "flat":
        print("FAIL: flat tape should be trend=flat", r_flat); ok = False

    # Determinism: same as_of + data -> identical label.
    if detect_regime(mv_up, "SPY") != r_up:
        print("FAIL: non-deterministic regime"); ok = False

    # Point-in-time: an earlier as_of must not see later bars (fewer n_bars).
    mv_early = HistoricalMarketView(datetime(2026, 1, 6, 16, 0), daily={"SPY": up})
    r_early = detect_regime(mv_early, "SPY")
    if r_early["n_bars"] >= r_up["n_bars"]:
        print("FAIL: earlier as_of should know fewer bars", r_early, r_up); ok = False

    # Insufficient data -> neutral fallback.
    mv_empty = HistoricalMarketView(datetime(2026, 1, 1, 16, 0), daily={"SPY": []})
    r_empty = detect_regime(mv_empty, "SPY")
    if r_empty["regime"] != "ranging" or r_empty["n_bars"] != 0:
        print("FAIL: empty data should fall back to ranging", r_empty); ok = False

    print("regime self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
