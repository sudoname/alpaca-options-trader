"""
Offline tests for Phase 13E — the learned-edge calibration reports.

No creds, no network, no broker. All records are injected. Covers, for each of
the three compute/format/generate trios:
  1. compute never raises on empty input and yields INSUFFICIENT_DATA.
  2. format always ends with ANALYTICS_FOOTER.
  3. With enough evidence the verdict is OK and the strong cohort out-ranks the
     weak one / a best system is chosen.
  4. generate_*_text returns a string (disk fail-open with injected records).

learned_edge_reports is STRICTLY analytics: it only reads and reports, never
opens, sizes, prices, blocks or alters any trade.
"""

import unittest

import learned_edge as le
import learned_edge_reports as ler
from ev_attribution import ANALYTICS_FOOTER
from learned_edge_reports import (
    compute_learned_edge_report, format_learned_edge_report,
    generate_learned_edge_report_text,
    compute_oracle_score_comparison, format_oracle_score_comparison,
    generate_oracle_score_comparison_text,
    compute_learned_edge_leaderboard, format_learned_edge_leaderboard,
    generate_learned_edge_leaderboard_text,
    VERDICT_OK, VERDICT_INSUFFICIENT,
)

CFG = le.LearnedEdgeConfig(backoff_min_samples=8, min_samples_full=20)


def _records():
    return (
        [le._rec("trending", "up", 0.20, i % 5 != 0, ev_risk=0.25, rid=f"w{i}")
         for i in range(20)]                      # strong, 80% WR
        + [le._rec("ranging", "down", 0.10, i % 3 == 0, ev_risk=0.02,
                   rid=f"l{i}") for i in range(12)]   # weak
    )


class TestLearnedEdgeReport(unittest.TestCase):
    def test_ok_verdict_and_confident_keys(self):
        rep = compute_learned_edge_report(records=_records(), config=CFG)
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertGreaterEqual(rep["num_confident_keys"], 2)

    def test_ordering_strong_above_weak(self):
        rep = compute_learned_edge_report(records=_records(), config=CFG)
        if rep["top_setups"] and rep["bottom_setups"]:
            self.assertGreaterEqual(
                rep["top_setups"][0]["learned_edge_score"],
                rep["bottom_setups"][0]["learned_edge_score"])

    def test_format_has_footer(self):
        txt = format_learned_edge_report(
            compute_learned_edge_report(records=_records(), config=CFG))
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn("Learned Edge Report", txt)

    def test_empty_insufficient(self):
        rep = compute_learned_edge_report(records=[], config=CFG)
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)
        txt = format_learned_edge_report(rep)
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestOracleScoreComparison(unittest.TestCase):
    def test_ok_verdict_and_best_system(self):
        cmp = compute_oracle_score_comparison(records=_records(), config=CFG)
        self.assertEqual(cmp["verdict"], VERDICT_OK)
        self.assertIn(cmp["best_system"],
                      ("oracle", "best_ev", "learned"))

    def test_format_has_footer(self):
        txt = format_oracle_score_comparison(
            compute_oracle_score_comparison(records=_records(), config=CFG))
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn("Oracle Score Comparison", txt)

    def test_empty_insufficient(self):
        cmp = compute_oracle_score_comparison(records=[], config=CFG)
        self.assertEqual(cmp["verdict"], VERDICT_INSUFFICIENT)
        txt = format_oracle_score_comparison(cmp)
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestLeaderboard(unittest.TestCase):
    def test_leaderboard_populated(self):
        lb = compute_learned_edge_leaderboard(records=_records(), config=CFG)
        self.assertTrue(lb["leaderboard"])
        self.assertEqual(lb["verdict"], VERDICT_OK)

    def test_format_has_footer(self):
        txt = format_learned_edge_leaderboard(
            compute_learned_edge_leaderboard(records=_records(), config=CFG))
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn("Leaderboard", txt)

    def test_empty_insufficient(self):
        lb = compute_learned_edge_leaderboard(records=[], config=CFG)
        self.assertEqual(lb["verdict"], VERDICT_INSUFFICIENT)
        txt = format_learned_edge_leaderboard(lb)
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestGenerateText(unittest.TestCase):
    def test_generators_return_strings(self):
        for gen in (generate_learned_edge_report_text,
                    generate_oracle_score_comparison_text,
                    generate_learned_edge_leaderboard_text):
            txt = gen(config=CFG, records=_records())
            self.assertIsInstance(txt, str)
            self.assertIn(ANALYTICS_FOOTER, txt)

    def test_generators_fail_open_on_empty(self):
        for gen in (generate_learned_edge_report_text,
                    generate_oracle_score_comparison_text,
                    generate_learned_edge_leaderboard_text):
            txt = gen(config=CFG, records=[])
            self.assertIn(ANALYTICS_FOOTER, txt)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(ler._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
