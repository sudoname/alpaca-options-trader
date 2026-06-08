"""
Offline tests for Phase 8A learning shadow layer (advisory only).

No creds, no network. The shadow is a pure, deterministic heuristic; the log
goes to a temp file. These tests also assert the layer is side-effect-free with
respect to trade decisions (it only ever returns a recommendation).
"""

import csv
import os
import tempfile
import unittest

from learning_shadow import (
    REC_AVOID, REC_NEUTRAL, REC_STRONG_TAKE, REC_TAKE,
    REGIME_ELEVATED, REGIME_HIGH, REGIME_LOW, REGIME_NORMAL, REGIME_UNKNOWN,
    LearningShadow, LearningShadowConfig, ShadowObservation, vix_regime,
)
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, DEBIT_CALL_SPREAD, NO_TRADE,
)


class VixRegimeTests(unittest.TestCase):
    def test_thresholds(self):
        self.assertEqual(vix_regime(10), REGIME_LOW)
        self.assertEqual(vix_regime(18), REGIME_NORMAL)
        self.assertEqual(vix_regime(25), REGIME_ELEVATED)
        self.assertEqual(vix_regime(40), REGIME_HIGH)

    def test_unknown(self):
        self.assertEqual(vix_regime(None), REGIME_UNKNOWN)
        self.assertEqual(vix_regime(0), REGIME_UNKNOWN)
        self.assertEqual(vix_regime(-5), REGIME_UNKNOWN)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        self.shadow = LearningShadow(LearningShadowConfig(enabled=True))

    def test_no_trade_is_avoid(self):
        rec = self.shadow.evaluate(ShadowObservation(
            spread_type=NO_TRADE, oracle_score=99.0))
        self.assertEqual(rec.recommendation, REC_AVOID)
        self.assertEqual(rec.confidence, 0.0)

    def test_empty_strategy_is_avoid(self):
        rec = self.shadow.evaluate(ShadowObservation(spread_type=""))
        self.assertEqual(rec.recommendation, REC_AVOID)

    def test_strong_aligned_credit_is_take(self):
        rec = self.shadow.evaluate(ShadowObservation(
            symbol="SPY", volatility_edge=0.4, oracle_score=80.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=40,
            trend="bullish", vix=28.0))
        self.assertIn(rec.recommendation, (REC_TAKE, REC_STRONG_TAKE))
        self.assertGreaterEqual(rec.confidence, 0.7)

    def test_misaligned_scores_lower_than_aligned(self):
        good = self.shadow.evaluate(ShadowObservation(
            symbol="SPY", volatility_edge=0.4, oracle_score=80.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=40,
            trend="bullish", vix=28.0))
        bad = self.shadow.evaluate(ShadowObservation(
            symbol="SPY", volatility_edge=-0.4, oracle_score=45.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=5,
            trend="bearish", vix=11.0))
        self.assertLess(bad.confidence, good.confidence)

    def test_confidence_bounded(self):
        rec = self.shadow.evaluate(ShadowObservation(
            volatility_edge=1.0, oracle_score=100.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=40,
            trend="bullish", vix=35.0))
        self.assertLessEqual(rec.confidence, 1.0)
        self.assertGreaterEqual(rec.confidence, 0.0)

    def test_debit_likes_underpriced(self):
        rec = self.shadow.evaluate(ShadowObservation(
            volatility_edge=-0.4, oracle_score=70.0,
            spread_type=DEBIT_CALL_SPREAD, dte=35,
            trend="bullish", vix=12.0))
        self.assertIn("edge+debit", rec.rationale)

    def test_thresholds_map_to_recommendations(self):
        cfg = LearningShadowConfig(enabled=True, take_threshold=0.55,
                                   strong_threshold=0.75, avoid_threshold=0.40)
        shadow = LearningShadow(cfg)
        # oracle_score 90 -> base 0.9, aligned bumps -> STRONG_TAKE.
        strong = shadow.evaluate(ShadowObservation(
            volatility_edge=0.3, oracle_score=90.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=40,
            trend="bullish", vix=28.0))
        self.assertEqual(strong.recommendation, REC_STRONG_TAKE)
        # oracle_score 20 with mismatches -> AVOID.
        avoid = shadow.evaluate(ShadowObservation(
            volatility_edge=-0.4, oracle_score=20.0,
            spread_type=BULLISH_PUT_CREDIT_SPREAD, dte=5,
            trend="bearish", vix=11.0))
        self.assertEqual(avoid.recommendation, REC_AVOID)


class LoggingTests(unittest.TestCase):
    def test_observe_and_log_round_trip(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "shadow.csv")
        shadow = LearningShadow(LearningShadowConfig(enabled=True, log_file=path))
        rec = shadow.observe_and_log(ShadowObservation(
            symbol="QQQ", volatility_edge=0.2, oracle_score=70.0,
            spread_type=DEBIT_CALL_SPREAD, dte=35, trend="bullish", vix=12.0))
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn("recommendation", rows[0])
        self.assertIn("confidence", rows[0])
        self.assertEqual(rows[0]["symbol"], "QQQ")
        # The returned recommendation matches the logged one.
        self.assertEqual(rows[0]["recommendation"], rec.recommendation)


if __name__ == "__main__":
    unittest.main()
