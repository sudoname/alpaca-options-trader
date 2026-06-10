"""
Offline tests for Phase 10G-A — Vol Forecast Scorecard.

No creds, no network, no broker. Covers:
  - realized vol from irregular snapshots (exact value, guards)
  - comparison-row construction across horizons and both CSV shapes
  - forecast vs IV MAE / RMSE / improvement calculations
  - Mincer-Zarnowitz regression (exact fit, degenerate inputs)
  - FORECAST_BEATS_IV / IV_BEATS_FORECAST / INCONCLUSIVE verdicts
  - empty and malformed datasets (never raises)
  - confidence tiers and Telegram output
  - no execution path touched (static guards)
"""

import math
import os
import unittest
from datetime import datetime, timedelta

import vol_forecast_scorecard as vfs
from ev_attribution import ANALYTICS_FOOTER
from vol_forecast_scorecard import (
    VERDICT_FORECAST_BEATS_IV, VERDICT_IV_BEATS_FORECAST,
    VERDICT_INCONCLUSIVE,
    build_rows, compute_scorecard, format_scorecard, linear_regression,
    realized_vol, scorecard_confidence,
)

HERE = os.path.dirname(os.path.abspath(__file__))
BASE = datetime(2026, 1, 5, 15, 0, 0)

# Daily log return that makes annualized realized vol EXACTLY 0.20 for any
# window of daily snapshots: sqrt(n*r^2 / (n/365.25)) = r*sqrt(365.25).
DAILY_R = 0.20 / math.sqrt(365.25)


def em_row(day, symbol="SPY", forecast=0.20, iv=0.30, price=None):
    if price is None:
        price = 100.0 * math.exp(DAILY_R * day)
    return {"timestamp": (BASE + timedelta(days=day)).isoformat(),
            "symbol": symbol, "in_price": f"{price:.10f}",
            "forecast_vol": f"{forecast}", "implied_vol": f"{iv}"}


def daily_points(n_days, start_price=100.0, r=DAILY_R):
    return [(BASE + timedelta(days=d), start_price * math.exp(r * d))
            for d in range(n_days + 1)]


# --------------------------------------------------------------------------- #
# Realized vol
# --------------------------------------------------------------------------- #
class TestRealizedVol(unittest.TestCase):
    def test_constant_daily_return_gives_exact_annualized_vol(self):
        points = daily_points(7)
        rv = realized_vol(points, BASE, 100.0, 7)
        self.assertAlmostEqual(rv, 0.20, places=10)

    def test_too_few_returns_is_none(self):
        # Only one snapshot inside the horizon -> fewer than MIN_RETURNS.
        points = daily_points(1)
        self.assertIsNone(realized_vol(points, BASE, 100.0, 7))

    def test_insufficient_coverage_is_none(self):
        # 10 days of data cover only 1/3 of a 30d horizon (< MIN_COVERAGE).
        points = daily_points(10)
        self.assertIsNone(realized_vol(points, BASE, 100.0, 30))

    def test_bad_prices_fail_open(self):
        self.assertIsNone(realized_vol([], BASE, 100.0, 7))
        self.assertIsNone(realized_vol(daily_points(7), BASE, 0.0, 7))
        self.assertIsNone(realized_vol(daily_points(7), BASE, None, 7))


# --------------------------------------------------------------------------- #
# Row construction
# --------------------------------------------------------------------------- #
class TestBuildRows(unittest.TestCase):
    def test_rows_built_per_horizon_with_correct_errors(self):
        em = [em_row(d) for d in range(10)]
        rows = build_rows(em_rows=em, dataset_rows=[])
        self.assertGreater(len(rows), 0)
        horizons = {r["horizon"] for r in rows}
        # daily snapshots give 1 return inside the 1d horizon (< MIN_RETURNS)
        # and never cover 60% of 30d -> only 3d and 7d resolve.
        self.assertEqual(horizons, {"3d", "7d"})
        for r in rows:
            self.assertEqual(r["symbol"], "SPY")
            self.assertAlmostEqual(r["realized_vol"], 0.20, places=6)
            self.assertAlmostEqual(r["forecast_error"], 0.0, places=6)
            self.assertAlmostEqual(r["iv_error"], 0.10, places=6)
            self.assertAlmostEqual(r["abs_iv_error"], 0.10, places=6)
            self.assertAlmostEqual(r["sq_iv_error"], 0.01, places=6)

    def test_snapshots_without_forecast_or_iv_are_price_points_only(self):
        em = [em_row(0)] + [
            {"timestamp": (BASE + timedelta(days=d)).isoformat(),
             "symbol": "SPY",
             "in_price": f"{100.0 * math.exp(DAILY_R * d):.10f}"}
            for d in range(1, 10)]
        rows = build_rows(em_rows=em, dataset_rows=[])
        # only the day-0 snapshot anchors comparisons, later ones lack vols
        self.assertEqual({r["date"] for r in rows},
                         {BASE.strftime("%Y-%m-%d")})
        self.assertGreater(len(rows), 0)

    def test_training_dataset_shape_is_recognized(self):
        ds = [{"timestamp": (BASE + timedelta(days=d)).isoformat(),
               "symbol": "QQQ",
               "feat_price": f"{100.0 * math.exp(DAILY_R * d):.10f}",
               "pred_forecast_vol": "0.20", "pred_implied_vol": "0.30"}
              for d in range(10)]
        rows = build_rows(em_rows=[], dataset_rows=ds)
        self.assertGreater(len(rows), 0)
        self.assertEqual({r["symbol"] for r in rows}, {"QQQ"})

    def test_vix_fallback_for_missing_iv(self):
        em = [dict(em_row(d), implied_vol="", in_vix="30") for d in range(10)]
        rows = build_rows(em_rows=em, dataset_rows=[])
        self.assertGreater(len(rows), 0)
        self.assertAlmostEqual(rows[0]["market_iv"], 0.30, places=6)

    def test_symbols_do_not_cross_contaminate(self):
        em = ([em_row(d, symbol="SPY") for d in range(10)]
              + [em_row(0, symbol="IWM")])  # IWM has no later points
        rows = build_rows(em_rows=em, dataset_rows=[])
        self.assertEqual({r["symbol"] for r in rows}, {"SPY"})

    def test_malformed_rows_fail_open(self):
        em = ["junk", 42, None, {}, {"timestamp": "not a date",
                                     "symbol": "SPY", "in_price": "x"}]
        self.assertEqual(build_rows(em_rows=em, dataset_rows=["junk"]), [])

    def test_missing_files_fail_open(self):
        from oracle_analytics import AnalyticsConfig
        cfg = AnalyticsConfig(
            expected_move_file="/nonexistent/vfs_e.csv",
            training_dataset_file="/nonexistent/vfs_d.csv")
        self.assertEqual(build_rows(config=cfg), [])


# --------------------------------------------------------------------------- #
# Regression
# --------------------------------------------------------------------------- #
class TestLinearRegression(unittest.TestCase):
    def test_exact_fit(self):
        xs = [1.0, 2.0, 3.0, 4.0]
        ys = [2.0 + 3.0 * x for x in xs]
        fit = linear_regression(xs, ys)
        self.assertAlmostEqual(fit["alpha"], 2.0, places=6)
        self.assertAlmostEqual(fit["beta"], 3.0, places=6)
        self.assertAlmostEqual(fit["r_squared"], 1.0, places=6)
        self.assertEqual(fit["n"], 4)

    def test_too_few_points(self):
        fit = linear_regression([1.0], [2.0])
        self.assertEqual(fit["n"], 1)
        self.assertIsNone(fit["beta"])

    def test_zero_x_variance(self):
        fit = linear_regression([2.0, 2.0, 2.0], [1.0, 2.0, 3.0])
        self.assertIsNone(fit["beta"])

    def test_none_values_dropped(self):
        fit = linear_regression([1.0, None, 2.0, "x"],
                                [1.0, 5.0, 2.0, 9.0])
        self.assertEqual(fit["n"], 2)
        self.assertAlmostEqual(fit["beta"], 1.0, places=6)


# --------------------------------------------------------------------------- #
# Scorecard metrics + verdicts
# --------------------------------------------------------------------------- #
def err_row(f_err, iv_err, horizon="3d"):
    return {"horizon": horizon,
            "abs_forecast_error": abs(f_err), "abs_iv_error": abs(iv_err),
            "sq_forecast_error": f_err ** 2, "sq_iv_error": iv_err ** 2}


class TestScorecard(unittest.TestCase):
    def test_forecast_beats_iv(self):
        card = compute_scorecard(
            rows=[err_row(0.02, 0.10), err_row(-0.02, -0.10)])
        self.assertAlmostEqual(card["forecast_mae"], 0.02, places=6)
        self.assertAlmostEqual(card["iv_mae"], 0.10, places=6)
        self.assertAlmostEqual(card["forecast_rmse"], 0.02, places=6)
        self.assertAlmostEqual(card["iv_rmse"], 0.10, places=6)
        self.assertAlmostEqual(card["forecast_vs_iv_improvement"], 0.80,
                               places=4)
        self.assertEqual(card["verdict"], VERDICT_FORECAST_BEATS_IV)

    def test_iv_beats_forecast(self):
        card = compute_scorecard(
            rows=[err_row(0.10, 0.02), err_row(-0.10, -0.02)])
        self.assertEqual(card["verdict"], VERDICT_IV_BEATS_FORECAST)
        self.assertLess(card["forecast_vs_iv_improvement"], 0)

    def test_split_decision_is_inconclusive(self):
        # forecast wins MAE (0.20 < 0.21) but loses RMSE (0.283 > 0.21).
        card = compute_scorecard(rows=[err_row(0.0, 0.21),
                                       err_row(0.4, 0.21)])
        self.assertLess(card["forecast_mae"], card["iv_mae"])
        self.assertGreater(card["forecast_rmse"], card["iv_rmse"])
        self.assertEqual(card["verdict"], VERDICT_INCONCLUSIVE)

    def test_end_to_end_from_snapshots(self):
        em = [em_row(d, forecast=0.20, iv=0.30) for d in range(10)]
        card = compute_scorecard(em_rows=em, dataset_rows=[])
        self.assertGreater(card["rows"], 0)
        self.assertEqual(card["verdict"], VERDICT_FORECAST_BEATS_IV)
        self.assertAlmostEqual(card["forecast_mae"], 0.0, places=5)
        self.assertAlmostEqual(card["iv_mae"], 0.10, places=5)
        self.assertGreater(card["by_horizon"]["3d"]["rows"], 0)
        self.assertEqual(card["by_horizon"]["30d"]["rows"], 0)

    def test_mz_regression_included(self):
        # vary the forecast so the regression has x-variance
        rows = [dict(err_row(0.0, 0.1), forecast_vol=0.10 + 0.01 * i,
                     market_iv=0.30, realized_vol=0.10 + 0.01 * i)
                for i in range(5)]
        card = compute_scorecard(rows=rows)
        self.assertAlmostEqual(card["mz_forecast"]["beta"], 1.0, places=4)
        self.assertAlmostEqual(card["mz_forecast"]["alpha"], 0.0, places=4)
        self.assertIsNone(card["mz_iv"]["beta"])  # constant IV

    def test_empty_dataset(self):
        card = compute_scorecard(em_rows=[], dataset_rows=[])
        self.assertEqual(card["rows"], 0)
        self.assertEqual(card["verdict"], VERDICT_INCONCLUSIVE)
        self.assertIsNone(card["forecast_mae"])
        text = format_scorecard(card)
        self.assertIn("No resolvable forecast/realized-vol pairs yet.", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_confidence_tiers(self):
        self.assertEqual(scorecard_confidence(0), "Low")
        self.assertEqual(scorecard_confidence(99), "Low")
        self.assertEqual(scorecard_confidence(100), "Medium")
        self.assertEqual(scorecard_confidence(1000), "Medium")
        self.assertEqual(scorecard_confidence(1001), "High")


# --------------------------------------------------------------------------- #
# Telegram output
# --------------------------------------------------------------------------- #
class TestOutput(unittest.TestCase):
    def test_report_layout(self):
        em = [em_row(d) for d in range(10)]
        text = format_scorecard(compute_scorecard(em_rows=em,
                                                  dataset_rows=[]))
        self.assertIn("Vol Forecast Scorecard", text)
        self.assertIn("MAE — forecast", text)
        self.assertIn("RMSE — forecast", text)
        self.assertIn("Forecast improvement over IV", text)
        self.assertIn("MZ forecast vol", text)
        self.assertIn("MZ market IV", text)
        self.assertIn("*By horizon (MAE forecast | IV):*", text)
        self.assertIn(f"*Verdict:* `{VERDICT_FORECAST_BEATS_IV}`", text)
        self.assertIn("Confidence: *Low*", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_telegram_bot_wires_the_command(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("VOL_FORECAST_SCORECARD", src)
        self.assertIn("def vol_forecast_scorecard", src)


# --------------------------------------------------------------------------- #
# No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def test_module_never_imports_live_trader_or_network(self):
        with open(os.path.join(HERE, "vol_forecast_scorecard.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
