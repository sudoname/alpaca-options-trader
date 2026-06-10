"""
Offline tests for Phase 10G-B — PoP Calibration.

No creds, no network, no broker. Covers:
  - record filtering (needs PnL + entry PoP stamp)
  - PoP bucket assignment (90-100 ... <50) and per-bucket stats
  - calibration_error = actual_win_rate - predicted_avg_pop
  - WELL_CALIBRATED / OVERCONFIDENT / UNDERCONFIDENT / INSUFFICIENT_DATA
  - boundary tolerance and minimum sample size
  - empty and malformed datasets (never raises)
  - Telegram output
  - no execution path touched (static guards)
"""

import os
import unittest

import pop_calibration as pc
from ev_attribution import ANALYTICS_FOOTER
from pop_calibration import (
    VERDICT_WELL_CALIBRATED, VERDICT_OVERCONFIDENT, VERDICT_UNDERCONFIDENT,
    VERDICT_INSUFFICIENT,
    compute_pop_calibration, format_pop_calibration, load_pop_records,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def rec(i, pnl, pop, max_loss=400.0):
    return {"id": f"p{i}", "pnl": pnl, "probability_of_profit": pop,
            "max_loss": max_loss}


def cohort(pop, wins, losses, start=0):
    """`wins` winners and `losses` losers all promised the same PoP."""
    rows = [rec(start + j, 50.0, pop) for j in range(wins)]
    rows += [rec(start + wins + j, -50.0, pop) for j in range(losses)]
    return rows


# --------------------------------------------------------------------------- #
# Record loading + buckets
# --------------------------------------------------------------------------- #
class TestRecordsAndBuckets(unittest.TestCase):
    def test_load_filters_unstamped_rows(self):
        rows = [rec(1, 10.0, 0.7), {"pnl": 5.0}, {"probability_of_profit": 0.8},
                "junk", None, rec(2, -5.0, 0.6)]
        loaded = load_pop_records(records=rows)
        self.assertEqual([r["id"] for r in loaded], ["p1", "p2"])

    def test_bucket_assignment(self):
        cases = {0.95: "PoP 90-100%", 0.90: "PoP 90-100%",
                 0.85: "PoP 80-90%", 0.75: "PoP 70-80%",
                 0.65: "PoP 60-70%", 0.55: "PoP 50-60%",
                 0.50: "PoP 50-60%", 0.45: "PoP <50%", 0.10: "PoP <50%"}
        for pop, label in cases.items():
            report = compute_pop_calibration(records=[rec(1, 10.0, pop)])
            self.assertEqual(report["buckets"][label]["trades"], 1,
                             (pop, label))

    def test_bucket_stats_fields(self):
        # 70-80% bucket: 3 wins / 1 loss, predicted 0.75.
        rows = cohort(0.75, wins=3, losses=1)
        report = compute_pop_calibration(records=rows)
        b = report["buckets"]["PoP 70-80%"]
        self.assertEqual(b["trades"], 4)
        self.assertAlmostEqual(b["predicted_avg_pop"], 0.75)
        self.assertAlmostEqual(b["actual_win_rate"], 0.75)
        self.assertAlmostEqual(b["calibration_error"], 0.0)
        self.assertAlmostEqual(b["profit_factor"], 3.0)  # 150 / 50
        self.assertAlmostEqual(b["avg_pnl"], 25.0)

    def test_empty_bucket_has_no_stats(self):
        report = compute_pop_calibration(records=cohort(0.75, 2, 2))
        b = report["buckets"]["PoP 90-100%"]
        self.assertEqual(b["trades"], 0)
        self.assertIsNone(b["predicted_avg_pop"])
        self.assertIsNone(b["calibration_error"])


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
class TestVerdicts(unittest.TestCase):
    def test_well_calibrated(self):
        # promised 70%, delivered 70% over 20 trades.
        report = compute_pop_calibration(records=cohort(0.70, 14, 6))
        self.assertEqual(report["verdict"], VERDICT_WELL_CALIBRATED)
        self.assertAlmostEqual(report["overall"]["calibration_error"], 0.0)

    def test_overconfident(self):
        # promised 80%, delivered 50%.
        report = compute_pop_calibration(records=cohort(0.80, 10, 10))
        self.assertEqual(report["verdict"], VERDICT_OVERCONFIDENT)
        self.assertAlmostEqual(report["overall"]["calibration_error"], -0.30)

    def test_underconfident(self):
        # promised 50%, delivered 80%.
        report = compute_pop_calibration(records=cohort(0.50, 16, 4))
        self.assertEqual(report["verdict"], VERDICT_UNDERCONFIDENT)
        self.assertAlmostEqual(report["overall"]["calibration_error"], 0.30)

    def test_tolerance_boundaries(self):
        # exactly -5pp -> OVERCONFIDENT; -4.5pp -> WELL_CALIBRATED.
        at_edge = compute_pop_calibration(records=cohort(0.75, 14, 6))
        self.assertAlmostEqual(at_edge["overall"]["calibration_error"], -0.05)
        self.assertEqual(at_edge["verdict"], VERDICT_OVERCONFIDENT)
        inside = compute_pop_calibration(records=cohort(0.745, 14, 6))
        self.assertEqual(inside["verdict"], VERDICT_WELL_CALIBRATED)

    def test_insufficient_below_min_trades(self):
        report = compute_pop_calibration(records=cohort(0.70, 4, 5))
        self.assertEqual(report["sample_size"], 9)
        self.assertEqual(report["verdict"], VERDICT_INSUFFICIENT)

    def test_empty_and_malformed(self):
        empty = compute_pop_calibration(records=[])
        self.assertEqual(empty["sample_size"], 0)
        self.assertEqual(empty["verdict"], VERDICT_INSUFFICIENT)
        messy = compute_pop_calibration(
            records=["junk", 7, None, {}, {"probability_of_profit": "x",
                                           "pnl": 1.0}, rec(1, 10.0, 0.7)])
        self.assertEqual(messy["sample_size"], 1)

    def test_missing_files_fail_open(self):
        from oracle_analytics import AnalyticsConfig
        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/pc_t.json",
            spread_positions_file="/nonexistent/pc_p.json",
            expected_move_file="/nonexistent/pc_e.csv",
            training_dataset_file="/nonexistent/pc_d.csv")
        report = compute_pop_calibration(config=cfg,
                                         attribution_path="/nonexistent/a.json")
        self.assertEqual(report["sample_size"], 0)


# --------------------------------------------------------------------------- #
# Telegram output
# --------------------------------------------------------------------------- #
class TestOutput(unittest.TestCase):
    def test_report_layout(self):
        rows = cohort(0.70, 14, 6) + cohort(0.95, 2, 2, start=100)
        text = format_pop_calibration(compute_pop_calibration(records=rows))
        self.assertIn("PoP Calibration", text)
        self.assertIn("`PoP 90-100%`:", text)
        self.assertIn("`PoP 70-80%`: `20` trades, predicted `70%` -> "
                      "actual `70%` (+0.0pp)", text)
        self.assertIn("`PoP <50%`: no trades", text)
        self.assertIn("*Overall:*", text)
        self.assertIn("*Verdict:*", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_empty_report(self):
        text = format_pop_calibration(compute_pop_calibration(records=[]))
        self.assertIn("No closed trades carrying an entry PoP stamp yet.",
                      text)
        self.assertIn(VERDICT_INSUFFICIENT, text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_telegram_bot_wires_the_command(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("POP_CALIBRATION", src)
        self.assertIn("def pop_calibration", src)


# --------------------------------------------------------------------------- #
# No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def test_module_never_imports_live_trader_or_network(self):
        with open(os.path.join(HERE, "pop_calibration.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
