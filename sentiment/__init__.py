"""
Market Fear & Greed sentiment module.

Public API:

    from sentiment import (
        SentimentService,
        SchwabMarketDataProvider,
        adjust_trade_risk_by_sentiment,
        summarize_for_log,
        SentimentConfig,
    )

The custom score (built from market data we fetch ourselves) is the PRIMARY
signal and keeps working even when the unofficial CNN endpoint is unavailable.
Every entry point is designed to fail open: a sentiment failure must never crash
or block the trading bot.
"""

from .sentiment_config import SentimentConfig, classify_score
from .cnn_fear_greed import fetch_cnn_fear_greed
from .custom_fear_greed import (
    MarketDataProvider,
    SchwabMarketDataProvider,
    compute_custom_fear_greed,
    percentile_score,
)
from .alpaca_provider import AlpacaMarketDataProvider
from .sentiment_cache import SentimentCache
from .sentiment_service import SentimentService
from .sentiment_filter import adjust_trade_risk_by_sentiment, summarize_for_log

__all__ = [
    "SentimentConfig",
    "classify_score",
    "fetch_cnn_fear_greed",
    "MarketDataProvider",
    "SchwabMarketDataProvider",
    "AlpacaMarketDataProvider",
    "compute_custom_fear_greed",
    "percentile_score",
    "SentimentCache",
    "SentimentService",
    "adjust_trade_risk_by_sentiment",
    "summarize_for_log",
]
