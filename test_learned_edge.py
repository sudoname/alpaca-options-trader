"""
Offline tests for Phase 13A — the Learned Edge Engine.

No creds, no network, no broker. Every test injects records, so nothing is read
from disk or the broker. Covers:
  1. Small cohorts regress toward the global prior; large cohorts trust the data.
  2. The learned_edge_score is pulled toward neutral 0.5 when confidence is low.
  3. The Beta-Binomial CI is ordered within [0, 1].
  4. Hierarchical backoff drops the least-important dimension (pattern first)
     until a cohort is large enough.
  5. Empty records -> a neutral, zero-confidence estimate.
  6. Never raises on garbage candidates.

learned_edge is ANALYTICS / SHADOW only: estimate_edge never opens, sizes,
prices, blocks or alters any trade and never raises.
"""

import unittest

import learned_edge as le
from learned_edge import LearnedEdgeConfig, estimate_edge, compute_prior


CFG = LearnedEdgeConfig(prior_strength_k=20.0, min_samples_full=30,
                        backoff_min_samples=8)


def _winners(n, regime="trending", direction="up", vol=0.20, win_every=5,
             prefix="w", **kw):
    # win_every controls the loss cadence: i % win_every != 0 wins.
    return [le._rec(regime, direction, vol, i % win_every != 0,
                    rid=f"{prefix}{i}", **kw) for i in range(n)]


class TestPriorVsSample(unittest.TestCase):
    def test_large_cohort_tracks_its_own_win_rate(self):
        # 50 records at 80% WR in trending/up/normal, plus mixed noise.
        big = _winners(50)                                   # 80% WR
        noise = [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"n{i}")
                 for i in range(10)]
        cand = {"regime": "trending", "trend": "up", "realized_vol": 0.20,
                "signal_strength": 2, "dte": 30, "entry_delta": 0.4}
        est = estimate_edge(cand, CFG, big + noise)
        self.assertGreaterEqual(est["sample_size"], 8)
        self.assertGreater(est["win_rate"], 0.65)
        self.assertGreater(est["learned_edge_score"], 0.5)

    def test_small_cohort_regresses_to_prior(self):
        # A modest 10-record cohort (>= backoff_min_samples, so it is used
        # directly without backing off to the global) that wins 100% must NOT
        # look like a sure thing: with min_samples_full=30 its confidence is
        # ~0.33 and the smoothed win rate is pulled toward the ~50% prior. A
        # large ~50% population in a DIFFERENT regime supplies that prior.
        prior_pop = [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"p{i}")
                     for i in range(40)]
        small = [le._rec("volatile", "flat", 0.60, True, rid=f"s{i}")
                 for i in range(10)]              # 100% WR, only 10 samples
        cand = {"regime": "volatile", "trend": "flat", "realized_vol": 0.60}
        est = estimate_edge(cand, CFG, small + prior_pop)
        self.assertEqual(est["sample_size"], 10)
        self.assertLess(est["confidence_score"], 0.5)
        # Smoothed win rate is pulled well below the raw 100%.
        self.assertLess(est["win_rate"], 0.9)


class TestNeutralPull(unittest.TestCase):
    def test_low_confidence_edge_near_neutral(self):
        prior_pop = [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"p{i}")
                     for i in range(40)]
        small = [le._rec("volatile", "flat", 0.60, True, ev_risk=0.5,
                         rid=f"s{i}") for i in range(10)]
        cand = {"regime": "volatile", "trend": "flat", "realized_vol": 0.60}
        est = estimate_edge(cand, CFG, small + prior_pop)
        # Despite a 100% sample win rate and strong EV, the ~0.33 confidence
        # pulls the edge toward neutral, so it stays modest.
        self.assertLessEqual(est["learned_edge_score"], 0.85)

    def test_confidence_scales_with_sample(self):
        small = estimate_edge(
            {"regime": "trending", "trend": "up", "realized_vol": 0.20},
            CFG, _winners(8))
        large = estimate_edge(
            {"regime": "trending", "trend": "up", "realized_vol": 0.20},
            CFG, _winners(40))
        self.assertLess(small["confidence_score"], large["confidence_score"])
        self.assertLessEqual(large["confidence_score"], 1.0)


class TestConfidenceInterval(unittest.TestCase):
    def test_ci_ordered_within_unit_interval(self):
        est = estimate_edge(
            {"regime": "trending", "trend": "up", "realized_vol": 0.20},
            CFG, _winners(50))
        self.assertTrue(0.0 <= est["ci_low"] <= est["ci_high"] <= 1.0)


class TestBackoff(unittest.TestCase):
    def test_unseen_pattern_drops_to_coarser_cohort(self):
        records = _winners(50)  # none carry a pattern
        cand = {"regime": "trending", "trend": "up", "realized_vol": 0.20,
                "signal_strength": 2, "dte": 30, "entry_delta": 0.4,
                "candlestick_pattern": "never_seen"}
        est = estimate_edge(cand, CFG, records)
        # Backoff recovers the big cohort and the pattern dim is gone.
        self.assertGreaterEqual(est["sample_size"], 8)
        self.assertNotIn("pattern", est["matched_dims"] or [])

    def test_backoff_drops_dims_in_order(self):
        # Records share regime/vol/direction but differ on delta/dte, so the
        # full key is sparse and backoff must drop the finer dims first.
        records = []
        for i in range(40):
            records.append(le._rec("trending", "up", 0.20, i % 5 != 0,
                                   dte=10 + i, delta=0.1 + 0.01 * i,
                                   rid=f"v{i}"))
        cand = {"regime": "trending", "trend": "up", "realized_vol": 0.20,
                "signal_strength": 2, "dte": 999, "entry_delta": 0.99}
        est = estimate_edge(cand, CFG, records)
        self.assertGreaterEqual(est["sample_size"], 8)
        # delta_bucket / dte_bucket should have been dropped (mismatched).
        self.assertNotIn("delta_bucket", est["matched_dims"] or [])

    def test_global_fallback_when_no_match(self):
        records = [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"g{i}")
                   for i in range(20)]
        cand = {"regime": "trending", "trend": "up", "realized_vol": 0.55,
                "signal_strength": 3, "dte": 5, "entry_delta": 0.6}
        est = estimate_edge(cand, CFG, records)
        # Falls all the way to the global prior (matched_dims empty).
        self.assertEqual(est["matched_dims"], [])
        self.assertEqual(est["sample_size"], len(records))


class TestEmptyAndPrior(unittest.TestCase):
    def test_empty_records_neutral(self):
        est = estimate_edge(
            {"regime": "trending", "trend": "up"}, CFG, [])
        self.assertEqual(est["learned_edge_score"], 0.5)
        self.assertEqual(est["confidence_score"], 0.0)
        self.assertEqual(est["sample_size"], 0)

    def test_prior_win_rate(self):
        # 6 wins out of 10 -> prior win rate 0.6.
        recs = [le._rec("trending", "up", 0.20, i < 6, rid=f"p{i}")
                for i in range(10)]
        prior = compute_prior(recs)
        self.assertEqual(prior["n"], 10)
        self.assertAlmostEqual(prior["win_rate"], 0.6, places=6)


class TestNeverRaises(unittest.TestCase):
    def test_garbage_candidates(self):
        records = _winners(20)
        for junk in (None, 42, "x", [], {"weird": object()}):
            est = estimate_edge(junk, CFG, records)  # type: ignore[arg-type]
            self.assertIn("learned_edge_score", est)


class TestEdgeIndex(unittest.TestCase):
    def test_index_has_readable_keys(self):
        idx = le.build_edge_index(_winners(20))
        self.assertTrue(idx)
        self.assertTrue(any("regime=" in v["key_str"] for v in idx.values()))


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(le._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
