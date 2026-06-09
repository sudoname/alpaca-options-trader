"""
Unit tests for the barrier touch-probability engine (barrier_engine.py).

Run with:
    python -m unittest test_barrier -v
    python test_barrier.py

Pure math — these tests do NOT touch the internet, any broker API, or disk.
"""

import math
import unittest

import barrier_engine as be


class TestNormCdf(unittest.TestCase):
    def test_center_and_tails(self):
        self.assertAlmostEqual(be.norm_cdf(0.0), 0.5, places=9)
        self.assertGreater(be.norm_cdf(5.0), 0.999)
        self.assertLess(be.norm_cdf(-5.0), 0.001)

    def test_symmetry(self):
        for x in (0.3, 1.0, 2.5):
            self.assertAlmostEqual(be.norm_cdf(x) + be.norm_cdf(-x), 1.0, places=9)


class TestTouchProbability(unittest.TestCase):
    def test_bounds_and_degenerate(self):
        # Bad inputs fail safe to 0.
        self.assertEqual(be.touch_probability(0, 90, 0.2, 0.0, 30), 0.0)
        self.assertEqual(be.touch_probability(100, 0, 0.2, 0.0, 30), 0.0)
        self.assertEqual(be.touch_probability(100, 90, 0.0, 0.0, 30), 0.0)
        self.assertEqual(be.touch_probability(100, 90, 0.2, 0.0, 0), 0.0)

    def test_monotonic_in_time(self):
        p7 = be.touch_probability(100, 90, 0.2, 0.0, 7)
        p30 = be.touch_probability(100, 90, 0.2, 0.0, 30)
        p90 = be.touch_probability(100, 90, 0.2, 0.0, 90)
        self.assertTrue(0.0 <= p7 <= p30 <= p90 <= 1.0)

    def test_closer_barrier_more_likely(self):
        near = be.touch_probability(100, 98, 0.2, 0.0, 30)
        far = be.touch_probability(100, 80, 0.2, 0.0, 30)
        self.assertGreater(near, far)

    def test_higher_vol_more_likely(self):
        lo = be.touch_probability(100, 90, 0.10, 0.0, 30)
        hi = be.touch_probability(100, 90, 0.40, 0.0, 30)
        self.assertGreater(hi, lo)

    def test_drift_lower_barrier(self):
        base = be.touch_probability(100, 90, 0.2, 0.0, 30)
        down = be.touch_probability(100, 90, 0.2, -0.10, 30)
        up = be.touch_probability(100, 90, 0.2, +0.10, 30)
        # Downward drift makes hitting a LOWER barrier more likely; up less.
        self.assertGreater(down, base)
        self.assertGreater(base, up)

    def test_drift_upper_barrier(self):
        base = be.touch_probability(100, 110, 0.2, 0.0, 30)
        up = be.touch_probability(100, 110, 0.2, +0.10, 30)
        down = be.touch_probability(100, 110, 0.2, -0.10, 30)
        # Upward drift makes hitting an UPPER barrier more likely; down less.
        self.assertGreater(up, base)
        self.assertGreater(base, down)


class TestProbCloseBeyond(unittest.TestCase):
    def test_close_le_touch(self):
        # A terminal close beyond the barrier is never more likely than touching.
        for days in (7, 30, 90):
            c = be.prob_close_beyond(100, 90, 0.2, 0.0, days)
            t = be.touch_probability(100, 90, 0.2, 0.0, days)
            self.assertLessEqual(c, t + 1e-9)

    def test_direction(self):
        lower = be.prob_close_beyond(100, 90, 0.2, 0.0, 30)   # P(S_T<=90)
        upper = be.prob_close_beyond(100, 110, 0.2, 0.0, 30)  # P(S_T>=110)
        self.assertTrue(0.0 < lower < 0.5)
        self.assertTrue(0.0 < upper < 0.5)


class TestSignalToDrift(unittest.TestCase):
    def test_call_put_sign(self):
        self.assertGreater(be.signal_to_drift('call', 4, 0.0, 0.2), 0)
        self.assertLess(be.signal_to_drift('put', 4, 0.0, 0.2), 0)

    def test_conviction_scale(self):
        self.assertAlmostEqual(be.signal_to_drift('call', 4, 0, 0.2), 0.2, places=9)
        self.assertAlmostEqual(be.signal_to_drift('call', 2, 0, 0.2), 0.1, places=9)
        # Strength caps at 4 -> 1 sigma max.
        self.assertAlmostEqual(be.signal_to_drift('call', 10, 0, 0.2), 0.2, places=9)

    def test_skip_momentum_lean(self):
        self.assertLess(be.signal_to_drift('skip', 0, -0.02, 0.2), 0)
        self.assertGreater(be.signal_to_drift('skip', 0, 0.02, 0.2), 0)
        self.assertEqual(be.signal_to_drift('skip', 0, 0.0, 0.2), 0.0)

    def test_failopen(self):
        self.assertEqual(be.signal_to_drift('call', 4, 0, 0.0), 0.0)
        self.assertEqual(be.signal_to_drift(None, None, None, 0.2), 0.0)
        self.assertEqual(be.signal_to_drift('call', 'bad', 0, 0.2), 0.0)


class TestClassify(unittest.TestCase):
    def test_buckets(self):
        self.assertEqual(be.classify_probability(0.05), "near-impossible")
        self.assertEqual(be.classify_probability(0.20), "unlikely")
        self.assertEqual(be.classify_probability(0.40), "possible but less likely than not")
        self.assertEqual(be.classify_probability(0.50), "roughly a coin-flip")
        self.assertEqual(be.classify_probability(0.65), "more likely than not")
        self.assertEqual(be.classify_probability(0.80), "probable")
        self.assertEqual(be.classify_probability(0.95), "very likely")


class TestAnalyze(unittest.TestCase):
    def test_rows_shape_and_order(self):
        rows = be.analyze(100, 90, 0.2, -0.05, [30, 7, 7])
        self.assertEqual([r['days'] for r in rows], [7, 30])  # sorted, deduped
        for r in rows:
            self.assertEqual(set(r),
                             {'days', 'p_touch_driftless', 'p_touch_drift', 'p_close_drift'})

    def test_drift_shifts_lower_barrier(self):
        rows = be.analyze(100, 90, 0.2, -0.10, [30])
        self.assertGreater(rows[0]['p_touch_drift'], rows[0]['p_touch_driftless'])


class TestFormatReport(unittest.TestCase):
    def test_contains_key_sections(self):
        rep = be.format_report("SPY", 732.0, 710.0, 0.14, 30,
                               strategy='put', strength=1, momentum=-0.013)
        self.assertIn("Barrier Analysis", rep)
        self.assertIn("SPY", rep)
        self.assertIn("Verdict", rep)
        self.assertIn("710", rep)
        self.assertIn("P(touch)", rep)
        # Requested horizon is marked.
        self.assertIn("←", rep)

    def test_neutral_when_no_signal(self):
        rep = be.format_report("AAPL", 200.0, 180.0, 0.25, 14,
                               strategy=None, strength=None, momentum=None)
        self.assertIn("neutral baseline", rep)

    def test_upper_target_direction(self):
        rep = be.format_report("QQQ", 500.0, 520.0, 0.20, 30,
                               strategy='call', strength=3, momentum=0.02)
        self.assertIn("above", rep)


if __name__ == "__main__":
    unittest.main(verbosity=2)
