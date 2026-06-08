"""
Offline tests for Phase 8A Oracle training-dataset builder.

No creds, no network. All CSV output goes to per-test temp files.
"""

import csv
import os
import tempfile
import unittest

from oracle_dataset_builder import OracleDatasetBuilder, OracleDatasetConfig


def _builder():
    d = tempfile.mkdtemp()
    cfg = OracleDatasetConfig(enabled=True,
                              dataset_file=os.path.join(d, "ds.csv"))
    return OracleDatasetBuilder(cfg), cfg


class LogTests(unittest.TestCase):
    def test_log_creates_pending_row(self):
        b, _ = _builder()
        rid = b.log(features={"hv20": 0.2, "trend": "bullish"},
                    predictions={"oracle_score": 72.0, "volatility_edge": 0.3},
                    symbol="SPY")
        self.assertTrue(rid)
        s = b.stats()
        self.assertEqual(s["total_rows"], 1)
        self.assertEqual(s["pending"], 1)
        self.assertEqual(s["with_outcome"], 0)

    def test_prefixes_and_header(self):
        b, cfg = _builder()
        b.log(features={"hv20": 0.2}, predictions={"oracle_score": 50.0},
              outcome={"pnl_pct": 0.1}, symbol="SPY")
        with open(cfg.dataset_file, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        for col in ("row_id", "timestamp", "symbol",
                    "feat_hv20", "pred_oracle_score", "out_pnl_pct"):
            self.assertIn(col, header)

    def test_nested_dict_flattened_one_level(self):
        b, cfg = _builder()
        b.log(features={"greeks": {"delta": 0.5, "theta": -0.1}}, symbol="SPY")
        with open(cfg.dataset_file, newline="", encoding="utf-8") as f:
            header = next(csv.reader(f))
        self.assertIn("feat_greeks_delta", header)
        self.assertIn("feat_greeks_theta", header)


class OutcomeTests(unittest.TestCase):
    def test_update_outcome_hit(self):
        b, _ = _builder()
        rid = b.log(features={"hv20": 0.2}, symbol="SPY")
        self.assertTrue(b.update_outcome(rid, {"pnl_pct": 0.5, "outcome": "win"}))
        s = b.stats()
        self.assertEqual(s["with_outcome"], 1)
        self.assertEqual(s["pending"], 0)

    def test_update_outcome_miss(self):
        b, _ = _builder()
        b.log(features={"hv20": 0.2}, symbol="SPY")
        self.assertFalse(b.update_outcome("nonexistent", {"pnl_pct": 1.0}))


class StatsTests(unittest.TestCase):
    def test_win_rate_and_mean_pnl(self):
        b, _ = _builder()
        b.log(predictions={"oracle_score": 70}, outcome={"pnl_pct": 0.5},
              symbol="SPY")
        b.log(predictions={"oracle_score": 30}, outcome={"pnl_pct": -0.3},
              symbol="QQQ")
        s = b.stats()
        self.assertEqual(s["total_rows"], 2)
        self.assertEqual(s["with_outcome"], 2)
        self.assertEqual(s["n_with_pnl"], 2)
        self.assertAlmostEqual(s["win_rate"], 0.5, places=6)
        self.assertAlmostEqual(s["mean_pnl"], 0.1, places=6)

    def test_evolving_columns_survive_rewrite(self):
        b, cfg = _builder()
        b.log(features={"hv20": 0.2}, symbol="SPY")
        # New feature key triggers header growth + full rewrite.
        b.log(features={"hv20": 0.3, "vix_regime": "high"}, symbol="QQQ")
        with open(cfg.dataset_file, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 2)
        self.assertIn("feat_vix_regime", rows[0])

    def test_empty_dataset_stats(self):
        b, _ = _builder()
        s = b.stats()
        self.assertEqual(s["total_rows"], 0)
        self.assertEqual(s["win_rate"], 0.0)
        self.assertEqual(s["completion_rate"], 0.0)


if __name__ == "__main__":
    unittest.main()
