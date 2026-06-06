"""
Unit tests for the Market Fear & Greed sentiment module.

Run with:
    python -m unittest test_sentiment -v
    python test_sentiment.py

All external calls (CNN HTTP, Schwab price history) are MOCKED — these tests do
NOT touch the internet or any broker API.
"""

import os
import tempfile
import unittest
from unittest import mock

import requests

from sentiment.sentiment_config import SentimentConfig, classify_score
from sentiment.cnn_fear_greed import fetch_cnn_fear_greed
from sentiment.custom_fear_greed import (
    MarketDataProvider,
    compute_custom_fear_greed,
    percentile_score,
)
from sentiment.sentiment_cache import SentimentCache
from sentiment.sentiment_service import SentimentService
from sentiment.sentiment_filter import adjust_trade_risk_by_sentiment
from sentiment.alpaca_provider import AlpacaMarketDataProvider


# --------------------------------------------------------------------------- #
# Test helpers
# --------------------------------------------------------------------------- #
class FakeResponse:
    def __init__(self, status_code=200, json_data=None, raise_json=False):
        self.status_code = status_code
        self._json_data = json_data
        self._raise_json = raise_json

    def json(self):
        if self._raise_json:
            raise ValueError("not json")
        return self._json_data


class FakeSession:
    """A requests.Session-like object whose .get returns a preset response."""

    def __init__(self, response=None, exc=None):
        self._response = response
        self._exc = exc

    def get(self, *args, **kwargs):
        if self._exc is not None:
            raise self._exc
        return self._response


class GreedyProvider(MarketDataProvider):
    """Provider whose series imply a strongly bullish (greed) market.

    SPY rises steadily (price well above its MA), VIX trends down, HYG strong vs
    LQD, SPY strong vs bonds. Optional feeds remain unavailable (default None).
    """

    def get_close_series(self, symbol, days):
        n = 200
        if symbol == "$VIX.X":
            # Falling VIX, current near the low end.
            return [30.0 - (i * 0.1) for i in range(n)]
        if symbol in ("LQD", "IEF", "TLT"):
            # Flat-ish bonds.
            return [100.0 + (i * 0.001) for i in range(n)]
        # SPY / HYG strongly rising.
        return [100.0 + i for i in range(n)]


class FearfulProvider(MarketDataProvider):
    def get_close_series(self, symbol, days):
        n = 200
        if symbol == "$VIX.X":
            # Rising VIX, current near the high end => fear.
            return [10.0 + (i * 0.1) for i in range(n)]
        if symbol in ("LQD", "IEF", "TLT"):
            return [100.0 + i for i in range(n)]  # bonds outperforming
        # SPY / HYG falling.
        return [300.0 - i for i in range(n)]


class EmptyProvider(MarketDataProvider):
    def get_close_series(self, symbol, days):
        return []


# --------------------------------------------------------------------------- #
# percentile_score
# --------------------------------------------------------------------------- #
class TestPercentileScore(unittest.TestCase):
    def test_basic_percentile(self):
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        # 5 values below 6 => 50%.
        self.assertEqual(percentile_score(series, 6), 50.0)

    def test_inverse(self):
        series = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        self.assertEqual(percentile_score(series, 6, inverse=True), 50.0)
        # Top of range inverse => low score (fearful for VIX).
        self.assertEqual(percentile_score(series, 11, inverse=True), 0.0)

    def test_empty_series_returns_none(self):
        self.assertIsNone(percentile_score([], 5))

    def test_none_current_returns_none(self):
        self.assertIsNone(percentile_score([1, 2, 3], None))

    def test_clamped_0_100(self):
        self.assertEqual(percentile_score([1, 2, 3], 100), 100.0)
        self.assertEqual(percentile_score([1, 2, 3], -100), 0.0)


# --------------------------------------------------------------------------- #
# classify_score boundaries
# --------------------------------------------------------------------------- #
class TestClassification(unittest.TestCase):
    def test_boundaries(self):
        self.assertEqual(classify_score(0), "Extreme Fear")
        self.assertEqual(classify_score(25), "Extreme Fear")
        self.assertEqual(classify_score(26), "Fear")
        self.assertEqual(classify_score(45), "Fear")
        self.assertEqual(classify_score(46), "Neutral")
        self.assertEqual(classify_score(55), "Neutral")
        self.assertEqual(classify_score(56), "Greed")
        self.assertEqual(classify_score(75), "Greed")
        self.assertEqual(classify_score(76), "Extreme Greed")
        self.assertEqual(classify_score(100), "Extreme Greed")

    def test_none_and_invalid(self):
        self.assertEqual(classify_score(None), "Unknown")
        self.assertEqual(classify_score("abc"), "Unknown")


# --------------------------------------------------------------------------- #
# CNN scraper (mocked)
# --------------------------------------------------------------------------- #
class TestCnnScraper(unittest.TestCase):
    def setUp(self):
        self.config = SentimentConfig()

    def test_success_headline(self):
        payload = {"fear_and_greed": {"score": 61.2, "rating": "greed",
                                      "timestamp": "2025-01-01"}}
        session = FakeSession(FakeResponse(200, payload))
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["score"], 61.2)
        self.assertEqual(result["classification"], "Greed")
        self.assertEqual(result["source"], "cnn_unofficial")

    def test_success_historical_fallback(self):
        payload = {"fear_and_greed_historical": {
            "data": [{"y": 20.0, "rating": "extreme fear"}]}}
        session = FakeSession(FakeResponse(200, payload))
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["score"], 20.0)

    def test_http_error(self):
        session = FakeSession(FakeResponse(503, {}))
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["score"])
        self.assertIn("503", result["error"])

    def test_timeout(self):
        import requests
        session = FakeSession(exc=requests.exceptions.Timeout())
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "error")
        self.assertIn("timeout", result["error"].lower())

    def test_invalid_json(self):
        session = FakeSession(FakeResponse(200, raise_json=True))
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "error")

    def test_missing_score(self):
        session = FakeSession(FakeResponse(200, {"something_else": 1}))
        result = fetch_cnn_fear_greed(self.config, session=session)
        self.assertEqual(result["status"], "error")


# --------------------------------------------------------------------------- #
# Custom score
# --------------------------------------------------------------------------- #
class TestCustomScore(unittest.TestCase):
    def setUp(self):
        self.config = SentimentConfig(min_components=3)

    def test_greedy_market_high_score(self):
        result = compute_custom_fear_greed(GreedyProvider(), self.config)
        self.assertEqual(result["status"], "available")
        self.assertGreater(result["score"], 55)
        # The 3 provider-supplied feeds are unavailable, not faked.
        self.assertIn("put_call_ratio", result["unavailable_components"])
        self.assertIn("market_breadth", result["unavailable_components"])
        self.assertIn("new_highs_lows", result["unavailable_components"])

    def test_fearful_market_low_score(self):
        result = compute_custom_fear_greed(FearfulProvider(), self.config)
        self.assertEqual(result["status"], "available")
        self.assertLess(result["score"], 45)

    def test_unavailable_components_excluded(self):
        result = compute_custom_fear_greed(GreedyProvider(), self.config)
        available = [c for c in result["components"] if c["available"]]
        # Score is the mean of ONLY available components.
        expected = round(sum(c["score"] for c in available) / len(available), 2)
        self.assertEqual(result["score"], expected)
        self.assertEqual(result["available_count"], len(available))

    def test_insufficient_components_errors(self):
        result = compute_custom_fear_greed(EmptyProvider(), self.config)
        self.assertEqual(result["status"], "error")
        self.assertIsNone(result["score"])

    def test_optional_feed_used_when_available(self):
        class WithExtras(GreedyProvider):
            def get_put_call_ratio_series(self, days):
                return ([0.8, 0.9, 1.0, 1.1], 0.85)  # low ratio => greed

            def get_market_breadth(self):
                return (400, 100)

            def get_new_highs_lows(self):
                return (150, 50)

        result = compute_custom_fear_greed(WithExtras(), self.config)
        self.assertEqual(result["status"], "available")
        self.assertEqual(result["unavailable_components"], [])


# --------------------------------------------------------------------------- #
# Alpaca market data provider (mocked HTTP)
# --------------------------------------------------------------------------- #
class TestAlpacaProvider(unittest.TestCase):
    def _provider(self):
        return AlpacaMarketDataProvider(
            "https://data.alpaca.markets",
            {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"},
        )

    def test_parses_close_series(self):
        bars = {"bars": [{"c": 100.0}, {"c": 101.5}, {"c": 99.0}]}
        with mock.patch("sentiment.alpaca_provider.requests.get",
                        return_value=FakeResponse(200, bars)) as getter:
            closes = self._provider().get_close_series("SPY", 10)
        self.assertEqual(closes, [100.0, 101.5, 99.0])
        # SPY passed through unchanged in the URL.
        self.assertIn("/stocks/SPY/bars", getter.call_args[0][0])

    def test_vix_mapped_to_vixy(self):
        with mock.patch("sentiment.alpaca_provider.requests.get",
                        return_value=FakeResponse(200, {"bars": []})) as getter:
            self._provider().get_close_series("$VIX.X", 10)
        self.assertIn("/stocks/VIXY/bars", getter.call_args[0][0])

    def test_truncates_to_days(self):
        bars = {"bars": [{"c": float(i)} for i in range(50)]}
        with mock.patch("sentiment.alpaca_provider.requests.get",
                        return_value=FakeResponse(200, bars)):
            closes = self._provider().get_close_series("SPY", 10)
        self.assertEqual(len(closes), 10)
        self.assertEqual(closes[-1], 49.0)

    def test_http_error_returns_empty(self):
        with mock.patch("sentiment.alpaca_provider.requests.get",
                        return_value=FakeResponse(403, {})):
            self.assertEqual(self._provider().get_close_series("SPY", 10), [])

    def test_request_exception_returns_empty(self):
        with mock.patch("sentiment.alpaca_provider.requests.get",
                        side_effect=requests.exceptions.ConnectionError()):
            self.assertEqual(self._provider().get_close_series("SPY", 10), [])


# --------------------------------------------------------------------------- #
# Cache
# --------------------------------------------------------------------------- #
class TestCache(unittest.TestCase):
    def setUp(self):
        fd, self.path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.path)  # start with no file

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_miss_when_empty(self):
        cache = SentimentCache(self.path, ttl_minutes=15)
        self.assertIsNone(cache.get_fresh())
        self.assertIsNone(cache.get_stale())

    def test_set_and_get_fresh(self):
        cache = SentimentCache(self.path, ttl_minutes=15)
        cache.set({"primary_score": {"score": 50}})
        fresh = cache.get_fresh()
        self.assertIsNotNone(fresh)
        self.assertEqual(fresh["primary_score"]["score"], 50)

    def test_stale_when_ttl_zero(self):
        cache = SentimentCache(self.path, ttl_minutes=0)
        cache.set({"x": 1})
        # TTL of 0 => immediately not fresh, but still retrievable as stale.
        self.assertIsNone(cache.get_fresh())
        self.assertEqual(cache.get_stale(), {"x": 1})

    def test_corrupt_file_is_miss(self):
        with open(self.path, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        cache = SentimentCache(self.path, ttl_minutes=15)
        self.assertIsNone(cache.get_fresh())


# --------------------------------------------------------------------------- #
# Service orchestration (mocked CNN + provider)
# --------------------------------------------------------------------------- #
class TestService(unittest.TestCase):
    def setUp(self):
        fd, self.cache_path = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        os.remove(self.cache_path)

    def tearDown(self):
        if os.path.exists(self.cache_path):
            os.remove(self.cache_path)

    def _config(self, **over):
        base = dict(enabled=True, use_cnn=False, use_custom=True,
                    primary_source="custom", min_components=3,
                    cache_minutes=15, cache_file=self.cache_path)
        base.update(over)
        return SentimentConfig(**base)

    def test_custom_primary(self):
        svc = SentimentService(provider=GreedyProvider(),
                               config=self._config(),
                               cache=SentimentCache(self.cache_path, 15))
        result = svc.get_sentiment()
        self.assertEqual(result["primary_source"], "custom")
        self.assertEqual(result["primary_score"]["status"], "available")

    def test_disabled_returns_disabled(self):
        svc = SentimentService(provider=GreedyProvider(),
                               config=self._config(enabled=False),
                               cache=SentimentCache(self.cache_path, 15))
        result = svc.get_sentiment()
        self.assertEqual(result["primary_score"]["status"], "disabled")

    def test_fallback_to_cnn_when_custom_fails(self):
        config = self._config(use_cnn=True)
        svc = SentimentService(provider=EmptyProvider(), config=config,
                               cache=SentimentCache(self.cache_path, 15))
        cnn_ok = {"source": "cnn_unofficial", "status": "available",
                  "score": 70.0, "classification": "Greed", "timestamp": "t"}
        with mock.patch("sentiment.sentiment_service.fetch_cnn_fear_greed",
                        return_value=cnn_ok):
            result = svc.get_sentiment()
        self.assertEqual(result["primary_source"], "cnn")
        self.assertEqual(result["primary_score"]["score"], 70.0)

    def test_stale_cache_fallback_on_refresh_failure(self):
        cache = SentimentCache(self.cache_path, ttl_minutes=0)
        cache.set({"primary_score": {"status": "available", "score": 42.0,
                                     "classification": "Fear"},
                   "primary_source": "custom"})
        # Provider now fails (empty) and CNN disabled => refresh yields no
        # usable primary; service should return the stale cache.
        svc = SentimentService(provider=EmptyProvider(),
                               config=self._config(cache_minutes=0),
                               cache=cache)
        result = svc.get_sentiment()
        self.assertTrue(result.get("from_stale_cache"))
        self.assertEqual(result["primary_score"]["score"], 42.0)

    def test_fresh_cache_short_circuits(self):
        cache = SentimentCache(self.cache_path, ttl_minutes=15)
        cache.set({"primary_score": {"status": "available", "score": 99.0}})
        svc = SentimentService(provider=GreedyProvider(),
                               config=self._config(), cache=cache)
        result = svc.get_sentiment()
        self.assertTrue(result.get("from_cache"))
        self.assertEqual(result["primary_score"]["score"], 99.0)


# --------------------------------------------------------------------------- #
# Sentiment filter
# --------------------------------------------------------------------------- #
def _sentiment(score, classification):
    return {"primary_score": {"status": "available", "score": score,
                              "classification": classification},
            "primary_source": "custom"}


class TestSentimentFilter(unittest.TestCase):
    def test_neutral_unchanged(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 75, "direction": "CALL"},
            _sentiment(50, "Neutral"))
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 100)

    def test_fear_reduces_size(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 75, "direction": "CALL"},
            _sentiment(38, "Fear"))
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 75)
        self.assertEqual(d["size_multiplier"], 0.75)

    def test_extreme_fear_blocks_low_confidence(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 60, "direction": "CALL"},
            _sentiment(10, "Extreme Fear"))
        self.assertFalse(d["allowed"])
        self.assertEqual(d["adjusted_size"], 0)

    def test_extreme_fear_allows_high_confidence_reduced(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 90, "direction": "CALL"},
            _sentiment(10, "Extreme Fear"))
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 50)

    def test_extreme_greed_trims(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 75, "direction": "CALL"},
            _sentiment(90, "Extreme Greed"))
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 75)

    def test_unavailable_sentiment_passthrough(self):
        d = adjust_trade_risk_by_sentiment(
            {"size": 100, "confidence": 75},
            {"primary_score": {"status": "error", "score": None}})
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 100)

    def test_none_sentiment_passthrough(self):
        d = adjust_trade_risk_by_sentiment({"size": 5}, None)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 5)

    def test_size_never_rounds_below_one_when_allowed(self):
        # 1 contract * 0.75 would round to 1, never 0, for an allowed trade.
        d = adjust_trade_risk_by_sentiment(
            {"size": 1, "confidence": 75, "direction": "CALL"},
            _sentiment(38, "Fear"))
        self.assertEqual(d["adjusted_size"], 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
