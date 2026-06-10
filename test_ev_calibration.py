"""
Offline tests for Phase 10G-C — EV Calibration.

No creds, no network, no broker. Covers:
  - record filtering (needs PnL + entry EV stamp)
  - OLS regression of realized PnL on expected EV (exact synthetic fit)
  - EV bucket assignment (<0, 0-10, 10-20, 20-50, 50+) and per-bucket stats
  - within_tolerance (max($10, 50% of expected))
  - EV_CALIBRATED / EV_RANKS_BUT_MISPRICES / EV_NOT_PREDICTIVE /
    INSUFFICIENT_DATA verdicts
  - empty and malformed datasets (never raises)
  - Telegram output
  - no execution path touched (static guards)
"""

import os
import unittest

from ev_attribution import ANALYTICS_FOOTER, VERDICT_YES, VERDICT_NO
from ev_calibration import (
    VERDICT_EV_CALIBRATED, VERDICT_EV_RANKS, VERDICT_EV_NOT_PREDICTIVE,
    VERDICT_INSUFFICIENT,
    compute_ev_calibration, format_ev_calibration, load_ev_records,
    within_tolerance,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def rec(i, ev, pnl, max_loss=400.0):
    return {"id": f"e{i}", "expected_value": ev, "pnl": pnl,
            "max_loss": max_loss}


def cohort(ev, pnls, start=0):
    return [rec(start + j, ev, p) for j, p in enumerate(pnls)]


# Promised EV ~ delivered PnL: low bucket small profits, high bucket big.
# ev=5: PF 60/45=1.33, avg +2.50.  ev=30: PF 200/10=20, avg +31.67.
# Overall expected 17.50 vs realized 17.08 -> within tol. Slope > 0.
CALIBRATED = (cohort(5.0, [20, -15, 20, -15, 20, -15]) +
              cohort(30.0, [40, 40, 40, 40, 40, -10], start=100))

# Higher EV still earns more (ranking holds) but realized is far below
# promised: expected 27.50 vs realized 5.50, err -22 > tol 13.75.
RANKS_ONLY = (cohort(5.0, [5, 5, 5, -3, -3, -3]) +
              cohort(50.0, [20, 20, 20, 20, 20, -40], start=100))

# Inverted: low EV prints, high EV bleeds. Slope < 0, buckets rank NO.
INVERTED = (cohort(5.0, [40, 40, 40, 40, 40, -10]) +
            cohort(30.0, [-40, -40, -40, -40, -40, 10], start=100))


# --------------------------------------------------------------------------- #
# Record loading + regression
# --------------------------------------------------------------------------- #
class TestRecordsAndRegression(unittest.TestCase):
    def test_load_filters_unstamped_rows(self):
        rows = [rec(1, 10.0, 5.0), {"pnl": 5.0}, {"expected_value": 8.0},
                "junk", None, rec(2, -3.0, -5.0)]
        loaded = load_ev_records(records=rows)
        self.assertEqual([r["id"] for r in loaded], ["e1", "e2"])

    def test_regression_exact_fit(self):
        # pnl = 2 * ev exactly -> alpha 0, beta 2, r^2 1.
        rows = [rec(i, ev, 2.0 * ev) for i, ev in enumerate([5, 10, 20, 30])]
        reg = compute_ev_calibration(records=rows)["regression"]
        self.assertAlmostEqual(reg["alpha"], 0.0)
        self.assertAlmostEqual(reg["beta"], 2.0)
        self.assertAlmostEqual(reg["r_squared"], 1.0)
        self.assertEqual(reg["sample_size"], 4)

    def test_regression_slope_sign(self):
        up = compute_ev_calibration(records=CALIBRATED)
        self.assertTrue(up["slope_positive"])
        down = compute_ev_calibration(records=INVERTED)
        self.assertFalse(down["slope_positive"])
        self.assertLess(down["regression"]["beta"], 0)


# --------------------------------------------------------------------------- #
# Buckets + tolerance
# --------------------------------------------------------------------------- #
class TestBucketsAndTolerance(unittest.TestCase):
    def test_bucket_assignment(self):
        cases = {-5.0: "EV < 0", 0.0: "EV 0-10", 5.0: "EV 0-10",
                 10.0: "EV 10-20", 20.0: "EV 20-50", 50.0: "EV 50+",
                 120.0: "EV 50+"}
        for ev, label in cases.items():
            report = compute_ev_calibration(records=[rec(1, ev, 1.0)])
            self.assertEqual(report["buckets"][label]["trades"], 1,
                             (ev, label))

    def test_bucket_stats_fields(self):
        report = compute_ev_calibration(records=CALIBRATED)
        b = report["buckets"]["EV 0-10"]
        self.assertEqual(b["trades"], 6)
        self.assertAlmostEqual(b["avg_expected_ev"], 5.0)
        self.assertAlmostEqual(b["avg_realized_pnl"], 2.5)
        self.assertAlmostEqual(b["calibration_error"], -2.5)
        self.assertAlmostEqual(b["profit_factor"], 1.33)  # 60/45 rounded
        hi = report["buckets"]["EV 20-50"]
        self.assertEqual(hi["trades"], 6)
        self.assertAlmostEqual(hi["profit_factor"], 20.0)
        empty = report["buckets"]["EV < 0"]
        self.assertEqual(empty["trades"], 0)
        self.assertIsNone(empty["calibration_error"])

    def test_within_tolerance(self):
        self.assertTrue(within_tolerance(20.0, 25.0))    # tol 10, diff 5
        self.assertFalse(within_tolerance(20.0, 35.0))   # diff 15 > 10
        self.assertTrue(within_tolerance(100.0, 140.0))  # tol 50, diff 40
        self.assertFalse(within_tolerance(100.0, 160.0))
        self.assertIsNone(within_tolerance(None, 5.0))
        self.assertIsNone(within_tolerance(5.0, None))


# --------------------------------------------------------------------------- #
# Verdicts
# --------------------------------------------------------------------------- #
class TestVerdicts(unittest.TestCase):
    def test_ev_calibrated(self):
        report = compute_ev_calibration(records=CALIBRATED)
        self.assertEqual(report["sample_size"], 12)
        self.assertEqual(report["ranking"]["verdict"], VERDICT_YES)
        self.assertTrue(report["magnitude_calibrated"])
        self.assertEqual(report["verdict"], VERDICT_EV_CALIBRATED)

    def test_ev_ranks_but_misprices(self):
        report = compute_ev_calibration(records=RANKS_ONLY)
        self.assertAlmostEqual(report["overall"]["avg_expected_ev"], 27.5)
        self.assertAlmostEqual(report["overall"]["avg_realized_pnl"], 5.5)
        self.assertFalse(report["magnitude_calibrated"])
        self.assertEqual(report["verdict"], VERDICT_EV_RANKS)

    def test_ev_not_predictive(self):
        report = compute_ev_calibration(records=INVERTED)
        self.assertEqual(report["ranking"]["verdict"], VERDICT_NO)
        self.assertEqual(report["verdict"], VERDICT_EV_NOT_PREDICTIVE)

    def test_insufficient_below_min_trades(self):
        report = compute_ev_calibration(records=CALIBRATED[:9])
        self.assertEqual(report["sample_size"], 9)
        self.assertEqual(report["verdict"], VERDICT_INSUFFICIENT)

    def test_empty_and_malformed(self):
        empty = compute_ev_calibration(records=[])
        self.assertEqual(empty["sample_size"], 0)
        self.assertEqual(empty["verdict"], VERDICT_INSUFFICIENT)
        messy = compute_ev_calibration(
            records=["junk", 7, None, {}, {"expected_value": "x",
                                           "pnl": 1.0}, rec(1, 5.0, 1.0)])
        self.assertEqual(messy["sample_size"], 1)

    def test_missing_files_fail_open(self):
        from oracle_analytics import AnalyticsConfig
        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/ec_t.json",
            spread_positions_file="/nonexistent/ec_p.json",
            expected_move_file="/nonexistent/ec_e.csv",
            training_dataset_file="/nonexistent/ec_d.csv")
        report = compute_ev_calibration(config=cfg,
                                        attribution_path="/nonexistent/a.json")
        self.assertEqual(report["sample_size"], 0)


# --------------------------------------------------------------------------- #
# Telegram output
# --------------------------------------------------------------------------- #
class TestOutput(unittest.TestCase):
    def test_report_layout(self):
        text = format_ev_calibration(
            compute_ev_calibration(records=CALIBRATED))
        self.assertIn("EV Calibration", text)
        self.assertIn("*Regression:*", text)
        self.assertIn("R²", text)
        self.assertIn("`EV 0-10`: `6` trades, expected `+$5.00` -> "
                      "realized `+$2.50` (err `-$2.50`), PF `1.33`", text)
        self.assertIn("`EV < 0`: no trades", text)
        self.assertIn("*Ranking:* higher EV buckets outperform: `YES`", text)
        self.assertIn("*Magnitude:*", text)
        self.assertIn(f"*Verdict:* `{VERDICT_EV_CALIBRATED}`", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_empty_report(self):
        text = format_ev_calibration(compute_ev_calibration(records=[]))
        self.assertIn("No closed trades carrying an entry EV stamp yet.",
                      text)
        self.assertIn(VERDICT_INSUFFICIENT, text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_telegram_bot_wires_the_command(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("EV_CALIBRATION", src)
        self.assertIn("def ev_calibration", src)


# --------------------------------------------------------------------------- #
# No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def test_module_never_imports_live_trader_or_network(self):
        with open(os.path.join(HERE, "ev_calibration.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
