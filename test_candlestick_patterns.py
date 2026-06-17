"""
Offline tests for Phase 11B-1 — Candlestick pattern detection (pure, no I/O).

Synthetic OHLCV fixtures exercise all 16 detectors plus:
  - candle normalization (dict o/h/l/c/v + open/high/..., tuple, Bar-like obj)
  - trend-context gating (require_trend_context blocks unclear/contrary trends)
  - evaluate_pattern_context alignment math + neutral -> 0
  - apply_candlestick_boost: cap never exceeded, neutral -> 0, never reduces
  - stamp_candlestick_patterns sets / omits the 6 frozen fields
  - CandlestickConfig.from_env fail-open + _self_test() == 0
"""

import unittest
from collections import namedtuple

from oracle.signals import candlestick_patterns as cp
from oracle.signals.candlestick_patterns import (
    CandlestickConfig, PatternStamp,
    detect_patterns, detect_primary,
    evaluate_pattern_context, apply_candlestick_boost,
    stamp_candlestick_patterns,
)

CFG = CandlestickConfig()  # defaults: require_trend_context=True


def candle(o, h, l, c, v=100):
    return {"o": o, "h": h, "l": l, "c": c, "v": v}


def down_prior():
    """Four candles in a clear downtrend (closes 110 -> 104)."""
    return [candle(c + 0.3, c + 0.6, c - 0.6, c) for c in (110, 108, 106, 104)]


def up_prior():
    """Four candles in a clear uptrend (closes 90 -> 96)."""
    return [candle(c - 0.3, c + 0.6, c - 0.6, c) for c in (90, 92, 94, 96)]


def flat_prior():
    return [candle(100, 100.6, 99.4, 100) for _ in range(4)]


def names(seq, cfg=CFG):
    return {p.pattern_name for p in detect_patterns(seq, cfg)}


def find(seq, name, cfg=CFG):
    for p in detect_patterns(seq, cfg):
        if p.pattern_name == name:
            return p
    return None


# --------------------------------------------------------------------------- #
# Normalization
# --------------------------------------------------------------------------- #
class TestNormalization(unittest.TestCase):
    def test_dict_short_keys(self):
        self.assertEqual(cp._norm({"o": 1, "h": 2, "l": 0.5, "c": 1.5, "v": 9}),
                         (1.0, 2.0, 0.5, 1.5, 9.0))

    def test_dict_long_keys(self):
        self.assertEqual(
            cp._norm({"open": 1, "high": 2, "low": 0.5, "close": 1.5}),
            (1.0, 2.0, 0.5, 1.5, None))

    def test_tuple(self):
        self.assertEqual(cp._norm((1, 2, 0.5, 1.5, 9)), (1.0, 2.0, 0.5, 1.5, 9.0))

    def test_bar_namedtuple(self):
        Bar = namedtuple("Bar", "o h l c v")
        self.assertEqual(cp._norm(Bar(1, 2, 0.5, 1.5, 9)),
                         (1.0, 2.0, 0.5, 1.5, 9.0))

    def test_bad_candles_are_none(self):
        self.assertIsNone(cp._norm({"bad": 1}))
        self.assertIsNone(cp._norm(None))
        self.assertIsNone(cp._norm((1, 2)))
        # high < low is rejected
        self.assertIsNone(cp._norm({"o": 1, "h": 0, "l": 5, "c": 1}))

    def test_normseq_empty_or_bad(self):
        self.assertIsNone(cp._normseq([]))
        self.assertIsNone(cp._normseq([candle(1, 2, 0.5, 1.5), {"bad": 1}]))


# --------------------------------------------------------------------------- #
# Single-candle patterns
# --------------------------------------------------------------------------- #
class TestSinglePatterns(unittest.TestCase):
    def test_hammer(self):
        seq = down_prior() + [candle(100, 100.7, 98.5, 100.6)]
        p = find(seq, cp.HAMMER)
        self.assertIsNotNone(p)
        self.assertEqual(p.pattern_type, cp.BULLISH_REVERSAL)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertEqual(p.strength, cp.STRENGTH_MEDIUM)
        self.assertTrue(p.requires_confirmation)

    def test_inverted_hammer(self):
        seq = down_prior() + [candle(100, 103, 99.8, 100.5)]
        p = find(seq, cp.INVERTED_HAMMER)
        self.assertIsNotNone(p)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertEqual(p.pattern_type, cp.BULLISH_REVERSAL)

    def test_hanging_man(self):
        seq = up_prior() + [candle(100, 100.7, 98.5, 100.6)]
        p = find(seq, cp.HANGING_MAN)
        self.assertIsNotNone(p)
        self.assertEqual(p.bias, cp.BEARISH)
        self.assertEqual(p.pattern_type, cp.BEARISH_REVERSAL)

    def test_shooting_star(self):
        seq = up_prior() + [candle(100, 103, 99.8, 100.5)]
        p = find(seq, cp.SHOOTING_STAR)
        self.assertIsNotNone(p)
        self.assertEqual(p.bias, cp.BEARISH)
        self.assertEqual(p.pattern_type, cp.BEARISH_REVERSAL)

    def test_doji(self):
        p = detect_primary([candle(100, 101, 99, 100.02)], CFG)
        self.assertIsNotNone(p)
        self.assertEqual(p.pattern_name, cp.DOJI)
        self.assertEqual(p.bias, cp.NEUTRAL)
        self.assertEqual(p.pattern_type, cp.INDECISION)

    def test_spinning_top(self):
        p = detect_primary([candle(100, 102, 98, 100.5)], CFG)
        self.assertIsNotNone(p)
        self.assertEqual(p.pattern_name, cp.SPINNING_TOP)
        self.assertEqual(p.bias, cp.NEUTRAL)


# --------------------------------------------------------------------------- #
# Two-candle patterns
# --------------------------------------------------------------------------- #
class TestTwoCandlePatterns(unittest.TestCase):
    def test_bullish_engulfing(self):
        seq = down_prior() + [candle(100, 100.5, 98, 98.5),
                              candle(98, 102, 97.5, 101.5)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.BULLISH_ENGULFING)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertEqual(p.strength, cp.STRENGTH_STRONG)
        self.assertFalse(p.requires_confirmation)

    def test_bearish_engulfing(self):
        seq = up_prior() + [candle(100, 102, 99.5, 101.5),
                            candle(102, 102.5, 98, 100)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.BEARISH_ENGULFING)
        self.assertEqual(p.bias, cp.BEARISH)
        self.assertFalse(p.requires_confirmation)

    def test_piercing_line(self):
        seq = down_prior() + [candle(100, 100.5, 95, 96),
                              candle(95, 99.5, 94.5, 99)]
        p = find(seq, cp.PIERCING_LINE)
        self.assertIsNotNone(p)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertTrue(p.requires_confirmation)

    def test_dark_cloud_cover(self):
        seq = up_prior() + [candle(96, 101, 95.5, 100),
                            candle(101, 101.5, 96.5, 97)]
        p = find(seq, cp.DARK_CLOUD_COVER)
        self.assertIsNotNone(p)
        self.assertEqual(p.bias, cp.BEARISH)
        self.assertTrue(p.requires_confirmation)


# --------------------------------------------------------------------------- #
# Three-candle patterns
# --------------------------------------------------------------------------- #
class TestThreeCandlePatterns(unittest.TestCase):
    def test_morning_star(self):
        seq = down_prior() + [candle(105, 105.5, 99, 100),
                              candle(99.5, 100, 98.5, 99.2),
                              candle(100, 104, 99.8, 103.5)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.MORNING_STAR)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertEqual(p.strength, cp.STRENGTH_STRONG)

    def test_evening_star(self):
        seq = up_prior() + [candle(100, 105.5, 99.5, 105),
                            candle(105.5, 106, 105, 105.3),
                            candle(105, 105.5, 101, 101.5)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.EVENING_STAR)
        self.assertEqual(p.bias, cp.BEARISH)

    def test_three_white_soldiers(self):
        seq = down_prior() + [candle(100, 101.5, 99.5, 101),
                              candle(100.5, 102.5, 100.3, 102),
                              candle(101.5, 103.5, 101.3, 103)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.THREE_WHITE_SOLDIERS)
        self.assertEqual(p.bias, cp.BULLISH)
        self.assertEqual(p.strength, cp.STRENGTH_STRONG)

    def test_three_black_crows(self):
        seq = up_prior() + [candle(105, 105.5, 103.5, 104),
                            candle(104.5, 104.7, 102.5, 103),
                            candle(103.5, 103.7, 101.5, 102)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.THREE_BLACK_CROWS)
        self.assertEqual(p.bias, cp.BEARISH)


# --------------------------------------------------------------------------- #
# Five-candle continuation patterns
# --------------------------------------------------------------------------- #
class TestFiveCandlePatterns(unittest.TestCase):
    def test_rising_three_methods(self):
        seq = up_prior() + [candle(100, 106, 99.5, 105),
                            candle(104, 104.5, 103, 103.5),
                            candle(103.5, 104, 102.5, 103),
                            candle(103, 103.5, 102, 102.5),
                            candle(103, 107, 102.5, 106.5)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.RISING_THREE_METHODS)
        self.assertEqual(p.pattern_type, cp.CONTINUATION)
        self.assertEqual(p.bias, cp.BULLISH)

    def test_falling_three_methods(self):
        seq = down_prior() + [candle(105, 105.5, 99, 100),
                              candle(101, 101.5, 100.5, 101.2),
                              candle(101.2, 101.7, 100.8, 101),
                              candle(101, 101.3, 100.3, 100.7),
                              candle(101, 101.5, 97, 98)]
        p = detect_primary(seq, CFG)
        self.assertEqual(p.pattern_name, cp.FALLING_THREE_METHODS)
        self.assertEqual(p.pattern_type, cp.CONTINUATION)
        self.assertEqual(p.bias, cp.BEARISH)


# --------------------------------------------------------------------------- #
# Trend-context gating
# --------------------------------------------------------------------------- #
class TestTrendGating(unittest.TestCase):
    def test_flat_trend_blocks_reversal_by_default(self):
        seq = flat_prior() + [candle(100, 100.7, 98.5, 100.6)]
        self.assertNotIn(cp.HAMMER, names(seq))
        self.assertNotIn(cp.HANGING_MAN, names(seq))

    def test_wrong_trend_blocks_hammer(self):
        # Hammer shape after an uptrend reads as a hanging man, never a hammer.
        seq = up_prior() + [candle(100, 100.7, 98.5, 100.6)]
        self.assertNotIn(cp.HAMMER, names(seq))
        self.assertIn(cp.HANGING_MAN, names(seq))

    def test_lenient_allows_flat_when_not_required(self):
        cfg = CandlestickConfig(require_trend_context=False)
        seq = flat_prior() + [candle(100, 100.7, 98.5, 100.6)]
        self.assertIn(cp.HAMMER, names(seq, cfg))


# --------------------------------------------------------------------------- #
# Context alignment
# --------------------------------------------------------------------------- #
class TestEvaluateContext(unittest.TestCase):
    def _bull(self):
        return PatternStamp(cp.BULLISH_ENGULFING, cp.BULLISH_REVERSAL,
                            cp.BULLISH, cp.STRENGTH_STRONG, 0.70, 2, "", False)

    def test_fully_aligned(self):
        r = evaluate_pattern_context(
            self._bull(), trend=cp.TREND_UP, support_resistance="support",
            volume_confirms=True, volatility_regime="elevated", pop=0.8,
            ev=10.0, advisory="ACCEPT", triple_gap=80)
        self.assertEqual(r["considered"], 8)
        self.assertEqual(r["aligned"], 8)
        self.assertEqual(r["context_alignment_score"], 1.0)

    def test_partial_alignment(self):
        r = evaluate_pattern_context(
            self._bull(), trend=cp.TREND_DOWN, ev=-5.0, pop=0.4)
        # 3 signals considered, none aligned for a bullish bias.
        self.assertEqual(r["considered"], 3)
        self.assertEqual(r["aligned"], 0)
        self.assertEqual(r["context_alignment_score"], 0.0)

    def test_neutral_bias_scores_zero(self):
        doji = PatternStamp(cp.DOJI, cp.INDECISION, cp.NEUTRAL,
                            cp.STRENGTH_WEAK, 0.30, 1, "", True)
        r = evaluate_pattern_context(doji, trend=cp.TREND_UP, ev=10.0)
        self.assertEqual(r["context_alignment_score"], 0.0)
        self.assertEqual(r["considered"], 0)

    def test_none_stamp_safe(self):
        r = evaluate_pattern_context(None, trend=cp.TREND_UP)
        self.assertEqual(r["context_alignment_score"], 0.0)


# --------------------------------------------------------------------------- #
# Capped confidence boost
# --------------------------------------------------------------------------- #
class TestApplyBoost(unittest.TestCase):
    def _bull(self, conf=0.70):
        return PatternStamp(cp.BULLISH_ENGULFING, cp.BULLISH_REVERSAL,
                            cp.BULLISH, cp.STRENGTH_STRONG, conf, 2, "", False)

    def test_boost_never_exceeds_cap(self):
        r = apply_candlestick_boost(0.70, self._bull(1.0), 1.0, CFG)
        self.assertLessEqual(r["boost_applied"], CFG.max_boost + 1e-9)

    def test_boost_never_reduces(self):
        r = apply_candlestick_boost(0.70, self._bull(), 1.0, CFG)
        self.assertGreaterEqual(r["final_confidence"], 0.70)

    def test_neutral_no_boost(self):
        doji = PatternStamp(cp.DOJI, cp.INDECISION, cp.NEUTRAL,
                            cp.STRENGTH_WEAK, 0.30, 1, "", True)
        r = apply_candlestick_boost(0.70, doji, 1.0, CFG)
        self.assertEqual(r["boost_applied"], 0.0)
        self.assertEqual(r["final_confidence"], 0.70)

    def test_zero_alignment_no_boost(self):
        r = apply_candlestick_boost(0.70, self._bull(), 0.0, CFG)
        self.assertEqual(r["boost_applied"], 0.0)

    def test_final_capped_at_one(self):
        r = apply_candlestick_boost(0.99, self._bull(1.0), 1.0, CFG)
        self.assertLessEqual(r["final_confidence"], 1.0)

    def test_none_stamp_safe(self):
        r = apply_candlestick_boost(0.70, None, 1.0, CFG)
        self.assertEqual(r["boost_applied"], 0.0)


# --------------------------------------------------------------------------- #
# Candidate stamping
# --------------------------------------------------------------------------- #
class TestStampCandidate(unittest.TestCase):
    def test_sets_fields_on_match(self):
        seq = down_prior() + [candle(100, 100.5, 98, 98.5),
                              candle(98, 102, 97.5, 101.5)]
        cand = stamp_candlestick_patterns({}, seq, CFG)
        self.assertEqual(cand[cp.FIELD_PATTERN], cp.BULLISH_ENGULFING)
        self.assertEqual(cand[cp.FIELD_BIAS], cp.BULLISH)
        self.assertFalse(cand[cp.FIELD_REQUIRES_CONFIRMATION])
        self.assertIsInstance(cand[cp.FIELD_CONFIDENCE], float)

    def test_none_when_no_pattern(self):
        # A plain mid-body candle is neither a single nor multi-candle pattern.
        cand = stamp_candlestick_patterns({}, [candle(100, 102, 99, 101.5)], CFG)
        for f in cp.STAMP_FIELDS:
            self.assertIn(f, cand)
            self.assertIsNone(cand[f])

    def test_none_when_disabled(self):
        seq = down_prior() + [candle(100, 100.5, 98, 98.5),
                              candle(98, 102, 97.5, 101.5)]
        cand = stamp_candlestick_patterns(
            {}, seq, CandlestickConfig(enabled=False))
        self.assertIsNone(cand[cp.FIELD_PATTERN])

    def test_empty_candles_safe(self):
        cand = stamp_candlestick_patterns({}, [], CFG)
        self.assertIsNone(cand[cp.FIELD_PATTERN])


# --------------------------------------------------------------------------- #
# Config + self-test
# --------------------------------------------------------------------------- #
class TestConfigAndSelfTest(unittest.TestCase):
    def test_from_env_fail_open(self):
        cfg = CandlestickConfig.from_env(path="/nonexistent/.env")
        self.assertIsInstance(cfg, CandlestickConfig)
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_boost, 0.05)

    def test_disabled_detects_nothing(self):
        seq = down_prior() + [candle(100, 100.7, 98.5, 100.6)]
        self.assertEqual(detect_patterns(seq, CandlestickConfig(enabled=False)),
                         [])

    def test_self_test_passes(self):
        self.assertEqual(cp._self_test(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
