"""
Unit tests for the per-ticker News signal module.

Run with:
    python -m unittest test_news -v
    python -m pytest test_news.py -q

All external calls (Alpaca news HTTP) are MOCKED — these tests do NOT touch the
internet or any broker API.
"""

import os
import tempfile
import unittest
from datetime import datetime, timedelta, timezone
from unittest import mock

import requests

from news.news_config import NewsConfig, classify_news
from news.alpaca_news import fetch_alpaca_news
from news.news_score import score_articles
from news.news_cache import NewsCache
from news.news_service import NewsService
from news.news_filter import (
    news_direction_vote,
    news_score_multiplier,
    adjust_trade_by_news,
    summarize_for_log,
)


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


def _now_iso(hours_ago=0):
    return (datetime.now(timezone.utc) - timedelta(hours=hours_ago)).isoformat()


def _article(headline="", summary="", hours_ago=1):
    return {"headline": headline, "summary": summary,
            "created_at": _now_iso(hours_ago)}


def _payload(score, status="available", label=None, count=5):
    return {"symbol": "SPY", "score": score, "label": label or "Bullish",
            "count": count, "bullish": 3, "bearish": 1, "status": status,
            "timestamp": _now_iso()}


# --------------------------------------------------------------------------- #
# classify_news
# --------------------------------------------------------------------------- #
class TestClassify(unittest.TestCase):
    def test_bands(self):
        self.assertEqual(classify_news(0.8), "Very Bullish")
        self.assertEqual(classify_news(0.3), "Bullish")
        self.assertEqual(classify_news(0.0), "Neutral")
        self.assertEqual(classify_news(-0.3), "Bearish")
        self.assertEqual(classify_news(-0.8), "Very Bearish")

    def test_none_and_bad(self):
        self.assertEqual(classify_news(None), "Unknown")
        self.assertEqual(classify_news("abc"), "Unknown")

    def test_clamped(self):
        self.assertEqual(classify_news(5.0), "Very Bullish")
        self.assertEqual(classify_news(-5.0), "Very Bearish")


# --------------------------------------------------------------------------- #
# score_articles
# --------------------------------------------------------------------------- #
class TestScoreArticles(unittest.TestCase):
    def setUp(self):
        self.cfg = NewsConfig(lookback_hours=24)

    def test_bullish(self):
        arts = [_article("Company beats estimates, raises guidance"),
                _article("Analyst upgrade as shares surge to record high")]
        out = score_articles(arts, self.cfg)
        self.assertGreater(out["score"], 0)
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["bullish"], 2)
        self.assertEqual(out["bearish"], 0)

    def test_bearish(self):
        arts = [_article("Company misses estimates, cuts guidance"),
                _article("Downgrade as shares plunge amid probe")]
        out = score_articles(arts, self.cfg)
        self.assertLess(out["score"], 0)
        self.assertEqual(out["bearish"], 2)

    def test_empty(self):
        out = score_articles([], self.cfg)
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(out["label"], "Neutral")
        self.assertEqual(out["count"], 0)

    def test_neutral_no_keywords(self):
        arts = [_article("Company announces annual shareholder meeting date")]
        out = score_articles(arts, self.cfg)
        self.assertEqual(out["score"], 0.0)
        self.assertEqual(out["label"], "Neutral")

    def test_recency_weighting(self):
        # A fresh bullish article should outweigh an old bearish one.
        arts = [_article("shares surge on upgrade", hours_ago=0),
                _article("shares plunge on downgrade", hours_ago=23)]
        out = score_articles(arts, self.cfg)
        self.assertGreater(out["score"], 0)


# --------------------------------------------------------------------------- #
# news_direction_vote
# --------------------------------------------------------------------------- #
class TestDirectionVote(unittest.TestCase):
    def setUp(self):
        self.cfg = NewsConfig(direction_votes=1)

    def test_bullish_votes(self):
        self.assertEqual(news_direction_vote(_payload(0.5), self.cfg), (1, 0))

    def test_bearish_votes(self):
        self.assertEqual(news_direction_vote(_payload(-0.5), self.cfg), (0, 1))

    def test_neutral_no_votes(self):
        self.assertEqual(news_direction_vote(_payload(0.05), self.cfg), (0, 0))

    def test_unavailable_no_votes(self):
        self.assertEqual(
            news_direction_vote(_payload(0.5, status="error"), self.cfg), (0, 0))
        self.assertEqual(news_direction_vote(None, self.cfg), (0, 0))

    def test_custom_vote_count(self):
        cfg = NewsConfig(direction_votes=2)
        self.assertEqual(news_direction_vote(_payload(0.5), cfg), (2, 0))


# --------------------------------------------------------------------------- #
# news_score_multiplier
# --------------------------------------------------------------------------- #
class TestScoreMultiplier(unittest.TestCase):
    def setUp(self):
        self.cfg = NewsConfig(rank_weight=0.15)

    def test_agree_call_boost(self):
        m = news_score_multiplier(_payload(1.0), "call", self.cfg)
        self.assertAlmostEqual(m, 1.15, places=3)

    def test_agree_put_boost(self):
        m = news_score_multiplier(_payload(-1.0), "put", self.cfg)
        self.assertAlmostEqual(m, 1.15, places=3)

    def test_disagree_trim(self):
        m = news_score_multiplier(_payload(-1.0), "call", self.cfg)
        self.assertAlmostEqual(m, 0.85, places=3)

    def test_neutral_unchanged(self):
        self.assertEqual(news_score_multiplier(_payload(0.05), "call", self.cfg), 1.0)

    def test_unavailable_unchanged(self):
        self.assertEqual(
            news_score_multiplier(_payload(1.0, status="error"), "call", self.cfg), 1.0)
        self.assertEqual(news_score_multiplier(None, "call", self.cfg), 1.0)

    def test_within_bounds(self):
        m = news_score_multiplier(_payload(0.5), "call", self.cfg)
        self.assertTrue(0.85 <= m <= 1.15)


# --------------------------------------------------------------------------- #
# adjust_trade_by_news
# --------------------------------------------------------------------------- #
class TestAdjustTrade(unittest.TestCase):
    def setUp(self):
        self.cfg = NewsConfig(block_threshold=0.6)

    def test_strong_oppose_low_conf_blocks(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 50, "direction": "CALL"},
            _payload(-0.8), self.cfg)
        self.assertFalse(d["allowed"])
        self.assertEqual(d["adjusted_size"], 0)

    def test_strong_oppose_high_conf_trims(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 90, "direction": "CALL"},
            _payload(-0.8), self.cfg)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 3)  # 4 * 0.75

    def test_mild_oppose_trims(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 50, "direction": "CALL"},
            _payload(-0.3), self.cfg)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 3)

    def test_agree_unchanged(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 50, "direction": "CALL"},
            _payload(0.8), self.cfg)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 4)
        self.assertEqual(d["size_multiplier"], 1.0)

    def test_neutral_unchanged(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 50, "direction": "CALL"},
            _payload(0.05), self.cfg)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 4)

    def test_unavailable_fail_open(self):
        d = adjust_trade_by_news(
            {"size": 4, "confidence": 50, "direction": "CALL"},
            _payload(-0.8, status="error"), self.cfg)
        self.assertTrue(d["allowed"])
        self.assertEqual(d["adjusted_size"], 4)


# --------------------------------------------------------------------------- #
# NewsCache (per-symbol)
# --------------------------------------------------------------------------- #
class TestNewsCache(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        self.tmp.close()
        self.path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def test_fresh_hit(self):
        cache = NewsCache(self.path, ttl_minutes=60)
        cache.set("SPY", _payload(0.5))
        got = cache.get_fresh("SPY")
        self.assertIsNotNone(got)
        self.assertEqual(got["symbol"], "SPY")

    def test_per_symbol_isolation(self):
        cache = NewsCache(self.path, ttl_minutes=60)
        cache.set("SPY", _payload(0.5))
        cache.set("QQQ", _payload(-0.5, label="Bearish"))
        self.assertEqual(cache.get_fresh("SPY")["score"], 0.5)
        self.assertEqual(cache.get_fresh("QQQ")["score"], -0.5)
        self.assertIsNone(cache.get_fresh("IWM"))

    def test_stale_fallback(self):
        # An entry older than the TTL is not fresh, but get_stale still returns
        # it. Write an explicitly-old cached_at to avoid any ttl=0 timing race.
        import json
        old = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        with open(self.path, "w", encoding="utf-8") as fh:
            json.dump({"SPY": {"cached_at": old, "payload": _payload(0.5)}}, fh)
        cache = NewsCache(self.path, ttl_minutes=15)
        self.assertIsNone(cache.get_fresh("SPY"))
        self.assertIsNotNone(cache.get_stale("SPY"))

    def test_missing_symbol(self):
        cache = NewsCache(self.path, ttl_minutes=60)
        self.assertIsNone(cache.get_fresh("NONE"))
        self.assertIsNone(cache.get_stale("NONE"))


# --------------------------------------------------------------------------- #
# fetch_alpaca_news (HTTP mocked)
# --------------------------------------------------------------------------- #
class TestFetch(unittest.TestCase):
    def setUp(self):
        self.cfg = NewsConfig()
        self.headers = {"APCA-API-KEY-ID": "k", "APCA-API-SECRET-KEY": "s"}

    def test_happy_path(self):
        body = {"news": [_article("beats estimates"), _article("upgrade")]}
        with mock.patch("news.alpaca_news.requests.get",
                        return_value=FakeResponse(200, body)):
            arts = fetch_alpaca_news("SPY", self.cfg, self.headers)
        self.assertEqual(len(arts), 2)

    def test_non_200(self):
        with mock.patch("news.alpaca_news.requests.get",
                        return_value=FakeResponse(403, {})):
            self.assertEqual(fetch_alpaca_news("SPY", self.cfg, self.headers), [])

    def test_request_exception(self):
        with mock.patch("news.alpaca_news.requests.get",
                        side_effect=requests.exceptions.RequestException("boom")):
            self.assertEqual(fetch_alpaca_news("SPY", self.cfg, self.headers), [])

    def test_bad_json(self):
        with mock.patch("news.alpaca_news.requests.get",
                        return_value=FakeResponse(200, None, raise_json=True)):
            self.assertEqual(fetch_alpaca_news("SPY", self.cfg, self.headers), [])

    def test_empty_symbol(self):
        self.assertEqual(fetch_alpaca_news("", self.cfg, self.headers), [])

    def test_respects_max_articles(self):
        body = {"news": [_article("upgrade") for _ in range(10)]}
        cfg = NewsConfig(max_articles=3)
        with mock.patch("news.alpaca_news.requests.get",
                        return_value=FakeResponse(200, body)):
            arts = fetch_alpaca_news("SPY", cfg, self.headers)
        self.assertEqual(len(arts), 3)


# --------------------------------------------------------------------------- #
# NewsService (fetch mocked at the module boundary)
# --------------------------------------------------------------------------- #
class TestNewsService(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.NamedTemporaryFile(
            suffix=".json", delete=False, mode="w")
        self.tmp.close()
        self.path = self.tmp.name

    def tearDown(self):
        if os.path.exists(self.path):
            os.remove(self.path)

    def _service(self, **cfg_kw):
        cfg = NewsConfig(cache_file=self.path, cache_minutes=60, **cfg_kw)
        return NewsService(cfg, headers={"APCA-API-KEY-ID": "k"})

    def test_available_and_cached(self):
        svc = self._service()
        body = [_article("beats estimates, upgrade, surge")]
        with mock.patch("news.news_service.fetch_alpaca_news", return_value=body) as f:
            first = svc.get_news("SPY")
            self.assertEqual(first["status"], "available")
            self.assertFalse(first["from_cache"])
            # Second call should hit the cache (no second fetch).
            second = svc.get_news("SPY")
            self.assertTrue(second["from_cache"])
            self.assertEqual(f.call_count, 1)

    def test_empty_is_cacheable_noeffect(self):
        svc = self._service()
        with mock.patch("news.news_service.fetch_alpaca_news", return_value=[]):
            out = svc.get_news("SPY")
        self.assertEqual(out["status"], "empty")
        self.assertEqual(out["score"], 0.0)

    def test_disabled(self):
        svc = self._service(enabled=False)
        out = svc.get_news("SPY")
        self.assertEqual(out["status"], "disabled")

    def test_never_raises(self):
        svc = self._service()
        with mock.patch("news.news_service.fetch_alpaca_news",
                        side_effect=RuntimeError("boom")):
            out = svc.get_news("SPY")
        self.assertEqual(out["status"], "error")
        self.assertEqual(out["score"], 0.0)


# --------------------------------------------------------------------------- #
# summarize_for_log
# --------------------------------------------------------------------------- #
class TestSummarize(unittest.TestCase):
    def test_available(self):
        s = summarize_for_log(_payload(0.5))
        self.assertIn("SPY", s)
        self.assertIn("0.5", s)

    def test_none(self):
        self.assertEqual(summarize_for_log(None), "news: none")

    def test_error(self):
        s = summarize_for_log(_payload(0.0, status="error"))
        self.assertIn("error", s)


if __name__ == "__main__":
    unittest.main(verbosity=2)
