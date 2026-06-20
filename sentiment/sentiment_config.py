"""
Configuration for the Market Fear & Greed sentiment module.

Follows the project's existing convention: read from environment variables
(via python-dotenv / os.getenv) with sensible defaults. No external config
object or framework is used.
"""

import os
from dataclasses import dataclass, field
from typing import List


def _get_bool(name: str, default: bool) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _get_int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _get_float(name: str, default: float) -> float:
    raw = os.getenv(name)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


# Classification thresholds (score 0-100). Ordered low -> high.
# 0-25 Extreme Fear | 26-45 Fear | 46-55 Neutral | 56-75 Greed | 76-100 Extreme Greed
CLASSIFICATION_BANDS = [
    (0, 25, "Extreme Fear"),
    (26, 45, "Fear"),
    (46, 55, "Neutral"),
    (56, 75, "Greed"),
    (76, 100, "Extreme Greed"),
]


def classify_score(score) -> str:
    """Map a 0-100 score to a CNN-style classification label.

    Returns "Unknown" when score is None or out of range.
    """
    if score is None:
        return "Unknown"
    try:
        s = float(score)
    except (TypeError, ValueError):
        return "Unknown"
    s = max(0.0, min(100.0, s))
    for low, high, label in CLASSIFICATION_BANDS:
        if low <= s <= high:
            return label
    return "Unknown"


@dataclass
class SentimentConfig:
    """Resolved sentiment configuration.

    Use ``SentimentConfig.from_env()`` to build one from environment variables.
    """

    enabled: bool = True
    cache_minutes: int = 15
    use_cnn: bool = True
    use_custom: bool = True
    primary_source: str = "blend"  # "blend", "custom", or "cnn"
    # Weight given to the custom score when primary_source == "blend".
    # CNN gets (1 - blend_custom_weight). Clamped to [0, 1].
    blend_custom_weight: float = 0.5
    min_components: int = 3

    # CNN scraper settings
    cnn_url: str = "https://production.dataviz.cnn.io/index/fearandgreed/graphdata"
    cnn_timeout: int = 10

    # Custom score: how many calendar days of daily history to request.
    history_days: int = 400

    # Symbols used by the custom score components.
    spy_symbol: str = "SPY"
    vix_symbol: str = "$VIX.X"
    junk_bond_symbol: str = "HYG"
    investment_grade_symbol: str = "LQD"
    long_treasury_symbol: str = "TLT"
    mid_treasury_symbol: str = "IEF"

    # Momentum lookback (trading days) for SPY vs moving average.
    momentum_window: int = 125

    # Realized-volatility window (trading days) for the volatility component.
    # The component scores SPY's trailing return volatility (stationary,
    # mean-reverting) instead of a VIX-level proxy whose secular drift would
    # bias the percentile.
    volatility_window: int = 20

    # Trailing-return window (trading days) for the safe-haven component, which
    # scores the SPY-minus-bond return spread (CNN's method) rather than an
    # absolute price-ratio level.
    return_spread_window: int = 20

    cache_file: str = "sentiment_cache.json"

    @classmethod
    def from_env(cls) -> "SentimentConfig":
        return cls(
            enabled=_get_bool("SENTIMENT_ENABLED", True),
            cache_minutes=_get_int("SENTIMENT_CACHE_MINUTES", 15),
            use_cnn=_get_bool("SENTIMENT_USE_CNN", True),
            use_custom=_get_bool("SENTIMENT_USE_CUSTOM", True),
            primary_source=os.getenv("SENTIMENT_PRIMARY_SOURCE", "blend").strip().lower(),
            blend_custom_weight=_get_float("SENTIMENT_BLEND_CUSTOM_WEIGHT", 0.5),
            min_components=_get_int("SENTIMENT_MIN_COMPONENTS", 3),
            cnn_url=os.getenv(
                "SENTIMENT_CNN_URL",
                "https://production.dataviz.cnn.io/index/fearandgreed/graphdata",
            ),
            cnn_timeout=_get_int("SENTIMENT_CNN_TIMEOUT", 10),
            history_days=_get_int("SENTIMENT_HISTORY_DAYS", 400),
            momentum_window=_get_int("SENTIMENT_MOMENTUM_WINDOW", 125),
            volatility_window=_get_int("SENTIMENT_VOLATILITY_WINDOW", 20),
            return_spread_window=_get_int("SENTIMENT_RETURN_SPREAD_WINDOW", 20),
            cache_file=os.getenv("SENTIMENT_CACHE_FILE", "sentiment_cache.json"),
        )
