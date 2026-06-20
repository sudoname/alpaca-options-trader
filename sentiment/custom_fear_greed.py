"""
Custom Fear & Greed score.

A self-contained, dependency-light reimplementation of CNN's index using market
data we can fetch ourselves (via Schwab price history). It is the PRIMARY source
for this bot: it keeps working even when the CNN endpoint is unavailable.

Seven components (CNN-style):
    1. Market momentum      — SPY vs its 125-day moving average (above = greed)
    2. Market volatility    — SPY realized-volatility percentile (high vol = fear, inverse)
    3. Put/call ratio       — high ratio = fear (inverse)            [provider-supplied]
    4. Junk bond demand     — HYG vs LQD relative strength (strong HYG = greed)
    5. Safe-haven demand    — SPY-minus-bond trailing return spread (stocks winning = greed)
    6. Market breadth       — advancers vs decliners                [provider-supplied]
    7. New highs / new lows — 52-week highs vs lows                 [provider-supplied]

Components 2 and 5 score *stationary* inputs (trailing return volatility and a
trailing return spread) rather than absolute price levels. A percentile of a
trending level mostly captures the secular trend, not current sentiment: VIXY
(the only VIX proxy on a stock feed) drifts to new lows via contango decay, and
an SPY/bond price ratio inherits the long bull/bond-bear trend — both would pin
the score near "extreme greed" regardless of risk. Returns are mean-reverting,
which matches CNN's own methodology.

Each component is scored 0-100 via rolling percentile. Components whose data is
unavailable are EXCLUDED from the average and reported in
``unavailable_components`` — they are never faked. The final score is the simple
average of the available component scores.
"""

import logging
import math
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .sentiment_config import SentimentConfig, classify_score

logger = logging.getLogger(__name__)


def percentile_score(series: Sequence[float], current_value: float,
                     inverse: bool = False) -> Optional[float]:
    """Score ``current_value`` against a historical ``series`` as a 0-100 percentile.

    The percentile is the percentage of historical values strictly below the
    current value. When ``inverse=True`` (used for metrics where high = fear,
    e.g. VIX and put/call ratio) the result is flipped (100 - percentile).

    Returns None when the series has no usable data.
    """
    if current_value is None:
        return None
    clean = [float(v) for v in series if v is not None]
    if not clean:
        return None
    below = sum(1 for v in clean if v < current_value)
    pct = (below / len(clean)) * 100.0
    if inverse:
        pct = 100.0 - pct
    return max(0.0, min(100.0, round(pct, 2)))


def _moving_average(series: Sequence[float], window: int) -> Optional[float]:
    clean = [float(v) for v in series if v is not None]
    if len(clean) < window:
        return None
    return sum(clean[-window:]) / window


def _relative_strength_series(numer: Sequence[float],
                              denom: Sequence[float]) -> List[float]:
    """Element-wise ratio of two aligned close series (numer/denom).

    Aligns from the most-recent end so series of slightly different lengths
    still line up correctly.
    """
    n = min(len(numer), len(denom))
    if n == 0:
        return []
    a = list(numer)[-n:]
    b = list(denom)[-n:]
    out = []
    for x, y in zip(a, b):
        if x is None or y is None or y == 0:
            continue
        out.append(float(x) / float(y))
    return out


def _daily_returns(series: Sequence[float]) -> List[float]:
    """Simple day-over-day returns from a close series (oldest -> newest)."""
    clean = [float(v) for v in series if v is not None]
    out = []
    for i in range(1, len(clean)):
        prev = clean[i - 1]
        if prev:
            out.append(clean[i] / prev - 1.0)
    return out


def _rolling_volatility(returns: Sequence[float], window: int) -> List[float]:
    """Rolling sample standard deviation of ``returns`` (one value per window end).

    Not annualized: the percentile is scale-invariant, so annualizing would not
    change the score. Returns an empty list when there is too little history.
    """
    clean = [float(r) for r in returns if r is not None]
    out: List[float] = []
    if window < 2 or len(clean) < window:
        return out
    for i in range(window, len(clean) + 1):
        w = clean[i - window:i]
        mean = sum(w) / len(w)
        var = sum((r - mean) ** 2 for r in w) / (len(w) - 1)
        out.append(math.sqrt(var))
    return out


def _rolling_return(series: Sequence[float], window: int) -> List[float]:
    """Rolling window-over-window return: ``series[i] / series[i-window] - 1``."""
    clean = [float(v) for v in series if v is not None]
    out: List[float] = []
    for i in range(window, len(clean)):
        prev = clean[i - window]
        if prev:
            out.append(clean[i] / prev - 1.0)
    return out


class MarketDataProvider:
    """Interface for the market data the custom score consumes.

    The default implementations of the "hard to source" feeds return None,
    which causes those components to be reported as unavailable (never faked).
    Concrete providers (e.g. Schwab) override ``get_close_series`` and may
    override the optional methods if a real data source becomes available.
    """

    def get_close_series(self, symbol: str, days: int) -> List[float]:
        """Return a list of daily closing prices (oldest -> newest)."""
        raise NotImplementedError

    # --- Optional feeds. Default None => component marked unavailable. ---
    def get_put_call_ratio_series(self, days: int):
        """Return (history_list, current_value) or None if unavailable."""
        return None

    def get_market_breadth(self):
        """Return (advancers, decliners) or None if unavailable."""
        return None

    def get_new_highs_lows(self):
        """Return (new_highs, new_lows) or None if unavailable."""
        return None


class SchwabMarketDataProvider(MarketDataProvider):
    """MarketDataProvider backed by a raw schwab-py client.

    Uses ``client.get_price_history(...)`` exactly as the backtests do. All calls
    are wrapped so a single failed symbol yields an empty series (that component
    becomes unavailable) rather than crashing the whole score.
    """

    def __init__(self, client):
        self.client = client

    def get_close_series(self, symbol: str, days: int) -> List[float]:
        try:
            PH = self.client.Client.PriceHistory
            end = datetime.now()
            response = self.client.get_price_history(
                symbol,
                period_type=PH.PeriodType.YEAR,
                period=PH.Period.ONE_YEAR,
                frequency_type=PH.FrequencyType.DAILY,
                frequency=PH.Frequency.DAILY,
            )
            payload = response.json()
            candles = payload.get("candles", []) if isinstance(payload, dict) else []
            closes = [c.get("close") for c in candles if c.get("close") is not None]
            if days and len(closes) > days:
                closes = closes[-days:]
            return closes
        except Exception as exc:
            logger.warning("Price history fetch failed for %s: %s", symbol, exc)
            return []


def _component(name: str, score: Optional[float], detail: str) -> dict:
    return {
        "name": name,
        "available": score is not None,
        "score": score,
        "detail": detail,
    }


def compute_custom_fear_greed(provider: MarketDataProvider,
                              config: Optional[SentimentConfig] = None) -> dict:
    """Compute the custom Fear & Greed score from market data.

    Always returns a normalized dict and never raises. If too few components are
    available (< ``config.min_components``) the status is "error" but partial
    component detail is still included for transparency.

    Returns:
        {"source": "custom",
         "status": "available" | "error",
         "score": float | None,
         "classification": str,
         "components": [ {name, available, score, detail}, ... ],
         "unavailable_components": [name, ...],
         "available_count": int,
         "timestamp": iso8601}
    """
    config = config or SentimentConfig.from_env()
    days = config.history_days
    components: List[dict] = []

    # --- 1. Market momentum: SPY vs N-day moving average ---
    try:
        spy = provider.get_close_series(config.spy_symbol, days)
        if spy and len(spy) >= config.momentum_window:
            ma = _moving_average(spy, config.momentum_window)
            # Score by percentile of the SPY/MA ratio over history.
            ratios = []
            for i in range(config.momentum_window, len(spy)):
                window_ma = sum(spy[i - config.momentum_window:i]) / config.momentum_window
                if window_ma:
                    ratios.append(spy[i] / window_ma)
            current_ratio = spy[-1] / ma if ma else None
            score = percentile_score(ratios, current_ratio) if current_ratio else None
            detail = (f"SPY {spy[-1]:.2f} vs {config.momentum_window}d MA "
                      f"{ma:.2f}" if ma else "insufficient data")
        else:
            score, detail = None, "insufficient SPY history"
        components.append(_component("market_momentum", score, detail))
    except Exception as exc:
        logger.warning("momentum component failed: %s", exc)
        components.append(_component("market_momentum", None, f"error: {exc}"))

    # --- 2. Market volatility: SPY realized-vol percentile (inverse: high = fear) ---
    # Scores SPY's trailing return volatility, not a VIX-level proxy. The only
    # index proxy on a stock feed is VIXY, whose contango decay drives its price
    # to new lows continuously -- a level-percentile of that pins the score near
    # "extreme greed" regardless of real risk. Realized vol is mean-reverting.
    try:
        spy_vol = provider.get_close_series(config.spy_symbol, days)
        vols = _rolling_volatility(_daily_returns(spy_vol), config.volatility_window)
        if len(vols) >= 2:
            score = percentile_score(vols[:-1], vols[-1], inverse=True)
            ann = vols[-1] * math.sqrt(252) * 100.0
            detail = (f"SPY {config.volatility_window}d realized vol "
                      f"{ann:.1f}% ann. (inverse percentile)")
        else:
            score, detail = None, "insufficient SPY history for realized vol"
        components.append(_component("market_volatility", score, detail))
    except Exception as exc:
        logger.warning("volatility component failed: %s", exc)
        components.append(_component("market_volatility", None, f"error: {exc}"))

    # --- 3. Put/call ratio (inverse). Provider-supplied; often unavailable. ---
    try:
        pc = provider.get_put_call_ratio_series(days)
        if pc:
            history, current = pc
            score = percentile_score(history, current, inverse=True)
            detail = f"put/call {current:.2f} (inverse percentile)"
        else:
            score, detail = None, "no put/call data source"
        components.append(_component("put_call_ratio", score, detail))
    except Exception as exc:
        logger.warning("put/call component failed: %s", exc)
        components.append(_component("put_call_ratio", None, f"error: {exc}"))

    # --- 4. Junk bond demand: HYG vs LQD relative strength (strong HYG = greed) ---
    try:
        hyg = provider.get_close_series(config.junk_bond_symbol, days)
        lqd = provider.get_close_series(config.investment_grade_symbol, days)
        rs = _relative_strength_series(hyg, lqd)
        if len(rs) >= 2:
            score = percentile_score(rs[:-1], rs[-1])
            detail = f"HYG/LQD strength ratio {rs[-1]:.4f}"
        else:
            score, detail = None, "insufficient HYG/LQD history"
        components.append(_component("junk_bond_demand", score, detail))
    except Exception as exc:
        logger.warning("junk bond component failed: %s", exc)
        components.append(_component("junk_bond_demand", None, f"error: {exc}"))

    # --- 5. Safe-haven demand: SPY-minus-bond trailing return spread ---
    # Stocks outperforming bonds recently = greed. Scores the difference in
    # trailing returns (CNN's method), not the absolute SPY/bond price-ratio
    # level: a level-percentile over a long window mostly captures the secular
    # bull/bond-bear trend rather than current risk appetite.
    try:
        spy = provider.get_close_series(config.spy_symbol, days)
        tlt = provider.get_close_series(config.long_treasury_symbol, days)
        ief = provider.get_close_series(config.mid_treasury_symbol, days)
        bond = tlt if tlt else ief
        win = config.return_spread_window
        spy_ret = _rolling_return(spy, win)
        bond_ret = _rolling_return(bond, win)
        n = min(len(spy_ret), len(bond_ret))
        if n >= 2:
            spread = [a - b for a, b in zip(spy_ret[-n:], bond_ret[-n:])]
            score = percentile_score(spread[:-1], spread[-1])
            label = "TLT" if tlt else ("IEF" if ief else "bond")
            detail = (f"SPY-{label} {win}d return spread "
                      f"{spread[-1] * 100:+.2f}pp")
        else:
            score, detail = None, "insufficient SPY/bond history"
        components.append(_component("safe_haven_demand", score, detail))
    except Exception as exc:
        logger.warning("safe-haven component failed: %s", exc)
        components.append(_component("safe_haven_demand", None, f"error: {exc}"))

    # --- 6. Market breadth (provider-supplied; often unavailable) ---
    try:
        breadth = provider.get_market_breadth()
        if breadth:
            adv, dec = breadth
            total = (adv or 0) + (dec or 0)
            score = round((adv / total) * 100, 2) if total else None
            detail = f"advancers {adv} / decliners {dec}"
        else:
            score, detail = None, "no breadth data source"
        components.append(_component("market_breadth", score, detail))
    except Exception as exc:
        logger.warning("breadth component failed: %s", exc)
        components.append(_component("market_breadth", None, f"error: {exc}"))

    # --- 7. New highs / new lows (provider-supplied; often unavailable) ---
    try:
        hl = provider.get_new_highs_lows()
        if hl:
            highs, lows = hl
            total = (highs or 0) + (lows or 0)
            score = round((highs / total) * 100, 2) if total else None
            detail = f"new highs {highs} / new lows {lows}"
        else:
            score, detail = None, "no highs/lows data source"
        components.append(_component("new_highs_lows", score, detail))
    except Exception as exc:
        logger.warning("highs/lows component failed: %s", exc)
        components.append(_component("new_highs_lows", None, f"error: {exc}"))

    # --- Aggregate available components only ---
    available = [c for c in components if c["available"]]
    unavailable = [c["name"] for c in components if not c["available"]]

    timestamp = datetime.now(timezone.utc).isoformat()

    if len(available) < config.min_components:
        logger.warning(
            "Custom F&G: only %d/%d required components available (%s)",
            len(available), config.min_components, ", ".join(unavailable),
        )
        return {
            "source": "custom",
            "status": "error",
            "score": None,
            "classification": "Unknown",
            "components": components,
            "unavailable_components": unavailable,
            "available_count": len(available),
            "error": (f"only {len(available)} components available, "
                      f"need {config.min_components}"),
            "timestamp": timestamp,
        }

    score = round(sum(c["score"] for c in available) / len(available), 2)
    classification = classify_score(score)
    logger.info(
        "Custom F&G: %.1f (%s) from %d components; unavailable: %s",
        score, classification, len(available),
        ", ".join(unavailable) if unavailable else "none",
    )
    return {
        "source": "custom",
        "status": "available",
        "score": score,
        "classification": classification,
        "components": components,
        "unavailable_components": unavailable,
        "available_count": len(available),
        "timestamp": timestamp,
    }
