"""
Offline tests for sentiment_report.compute_sentiment_report (the read-only
adapter behind the dashboard's /api/sentiment endpoint).

No creds, no network: every test injects a ``service_factory`` that returns
either ``None``, a raising stub, or a fake service whose ``get_sentiment()``
yields a canned payload. The contract pinned here:

  1. FAIL-OPEN. No service / raising factory / error payload -> verdict
     INSUFFICIENT_DATA (never raises), so the widget degrades cleanly.
  2. PASSTHROUGH. An available payload -> verdict OK with score, classification,
     blended source, sub-scores, and the per-component breakdown surfaced.
  3. READ-ONLY BY CONSTRUCTION. The module imports no execution path and issues
     no non-GET HTTP -- asserted by source grep.
  4. ``_self_test()`` returns 0.
"""

import inspect
import unittest

import sentiment_report as sr
from sentiment_report import compute_sentiment_report


class _FakeService:
    def __init__(self, payload):
        self._payload = payload

    def get_sentiment(self):
        return self._payload


_AVAILABLE = {
    "cnn_score": {"status": "available", "score": 62.0},
    "custom_score": {
        "source": "custom", "status": "available", "score": 48.0,
        "available_count": 4, "unavailable_components": ["put_call_ratio"],
        "components": [
            {"name": "market_momentum", "available": True, "score": 55.0,
             "detail": "SPY vs MA"},
            {"name": "market_volatility", "available": True, "score": 41.0,
             "detail": "VIX inverse pct"},
        ],
    },
    "primary_score": {"source": "blend", "status": "available", "score": 55.0,
                      "classification": "Neutral"},
    "primary_source": "blend",
    "timestamp": "2026-01-01T00:00:00+00:00",
    "from_cache": False,
}


class TestFailOpen(unittest.TestCase):
    def test_none_service_is_insufficient(self):
        r = compute_sentiment_report(service_factory=lambda: None)
        self.assertEqual(r["verdict"], "INSUFFICIENT_DATA")

    def test_raising_factory_is_insufficient(self):
        def boom():
            raise RuntimeError("network down")
        self.assertEqual(
            compute_sentiment_report(service_factory=boom)["verdict"],
            "INSUFFICIENT_DATA")

    def test_error_payload_is_insufficient(self):
        payload = {"primary_score": {"status": "error", "score": None,
                                     "classification": "Unknown"},
                   "primary_source": None}
        r = compute_sentiment_report(service_factory=lambda: _FakeService(payload))
        self.assertEqual(r["verdict"], "INSUFFICIENT_DATA")

    def test_always_carries_components_list(self):
        r = compute_sentiment_report(service_factory=lambda: None)
        self.assertIsInstance(r.get("components"), list)


class TestPassthrough(unittest.TestCase):
    def setUp(self):
        self.r = compute_sentiment_report(
            service_factory=lambda: _FakeService(_AVAILABLE))

    def test_verdict_ok(self):
        self.assertEqual(self.r["verdict"], "OK")

    def test_score_and_classification(self):
        self.assertEqual(self.r["score"], 55.0)
        self.assertEqual(self.r["classification"], "Neutral")

    def test_source_and_subscores(self):
        self.assertEqual(self.r["source"], "blend")
        self.assertEqual(self.r["cnn_score"], 62.0)
        self.assertEqual(self.r["custom_score"], 48.0)

    def test_components_surfaced(self):
        names = [c["name"] for c in self.r["components"]]
        self.assertIn("market_momentum", names)
        self.assertIn("market_volatility", names)

    def test_unavailable_components_surfaced(self):
        self.assertIn("put_call_ratio", self.r["unavailable_components"])


class TestReadOnlyByConstruction(unittest.TestCase):
    def test_no_execution_or_write_symbols_in_source(self):
        src = inspect.getsource(sr)
        forbidden = ("place_option_order(", "submit_order(", "execute_trade(",
                     "open_position(", "close_position(", "record_outcome(",
                     "requests.post(", "requests.put(", "requests.delete(",
                     "requests.patch(")
        for token in forbidden:
            self.assertNotIn(token, src, msg=f"forbidden token {token!r} present")


class TestSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(sr._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
