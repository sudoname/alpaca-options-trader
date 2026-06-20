"""
Offline tests for Oracle 3.0 — the Market Regime Agent (oracle_regime).

No creds, no network, no broker. Every test injects a ``regime.detect_regime``
dict, so nothing is read from disk or the market. Covers:
  1. Each of the 8 labels fires on the canonical inputs.
  2. Confidence is always in [0, 1].
  3. NO single axis triggers a dramatic label (PANIC_SELLING / NEWS_DRIVEN need
     two corroborating axes).
  4. Garbage / empty inputs fail open to a neutral RANGE_BOUND, never raise.

oracle_regime is ANALYTICS / SHADOW ONLY: classify_regime never opens, sizes,
prices, blocks or alters any trade and never raises.
"""

import unittest

import oracle_regime as orr
from oracle_regime import (
    classify_regime, REGIME_LABELS,
    TRENDING_BULL, TRENDING_BEAR, RANGE_BOUND, HIGH_VOLATILITY,
    LOW_VOLATILITY, NEWS_DRIVEN, BREAKOUT, PANIC_SELLING,
)


def _raw(regime, trend, rvol, mom):
    return {"regime": regime, "trend": trend, "realized_vol": rvol,
            "momentum": mom}


class TestLabels(unittest.TestCase):
    def test_trending_bull(self):
        est = classify_regime(regime_raw=_raw("trending", "up", 0.20, 0.06))
        self.assertEqual(est["label"], TRENDING_BULL)

    def test_trending_bear(self):
        est = classify_regime(regime_raw=_raw("trending", "down", 0.20, -0.06))
        self.assertEqual(est["label"], TRENDING_BEAR)

    def test_range_bound(self):
        est = classify_regime(regime_raw=_raw("ranging", "flat", 0.20, 0.0))
        self.assertEqual(est["label"], RANGE_BOUND)

    def test_low_volatility(self):
        est = classify_regime(regime_raw=_raw("ranging", "flat", 0.10, 0.0))
        self.assertEqual(est["label"], LOW_VOLATILITY)

    def test_high_volatility(self):
        est = classify_regime(regime_raw=_raw("volatile", "flat", 0.40, 0.0))
        self.assertEqual(est["label"], HIGH_VOLATILITY)

    def test_breakout(self):
        est = classify_regime(regime_raw=_raw("trending", "up", 0.25, 0.14))
        self.assertEqual(est["label"], BREAKOUT)

    def test_panic_selling(self):
        est = classify_regime(regime_raw=_raw("volatile", "down", 0.60, -0.12))
        self.assertEqual(est["label"], PANIC_SELLING)

    def test_news_driven(self):
        est = classify_regime(regime_raw=_raw("volatile", "flat", 0.35, 0.0),
                              news_score=-0.6)
        self.assertEqual(est["label"], NEWS_DRIVEN)


class TestConfidence(unittest.TestCase):
    def test_confidence_in_unit_interval(self):
        for raw in (_raw("trending", "up", 0.2, 0.06),
                    _raw("volatile", "down", 0.6, -0.12),
                    _raw("ranging", "flat", 0.2, 0.0)):
            est = classify_regime(regime_raw=raw)
            self.assertGreaterEqual(est["confidence"], 0.0)
            self.assertLessEqual(est["confidence"], 1.0)

    def test_stronger_thrust_higher_confidence(self):
        weak = classify_regime(regime_raw=_raw("trending", "up", 0.2, 0.055))
        strong = classify_regime(regime_raw=_raw("trending", "up", 0.2, 0.09))
        self.assertGreaterEqual(strong["confidence"], weak["confidence"])


class TestNoSingleAxis(unittest.TestCase):
    def test_lone_vol_spike_not_panic(self):
        # Extreme vol but no directional move must not be PANIC_SELLING.
        est = classify_regime(regime_raw=_raw("volatile", "flat", 0.70, 0.0))
        self.assertNotEqual(est["label"], PANIC_SELLING)

    def test_lone_news_not_news_driven(self):
        # Strong news but calm vol must not be NEWS_DRIVEN.
        est = classify_regime(regime_raw=_raw("ranging", "flat", 0.12, 0.0),
                              news_score=-0.9)
        self.assertNotEqual(est["label"], NEWS_DRIVEN)


class TestFailOpen(unittest.TestCase):
    def test_garbage_never_raises(self):
        for junk in (None, 42, "x", [], {"weird": object()}):
            est = classify_regime(regime_raw=junk)  # type: ignore[arg-type]
            self.assertIn(est["label"], REGIME_LABELS)

    def test_empty_is_neutral(self):
        est = classify_regime(regime_raw=None)
        self.assertEqual(est["label"], RANGE_BOUND)
        self.assertEqual(est["confidence"], 0.0)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(orr._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
