"""
Offline tests for Phase 8C — threshold recommendation engine.

No creds, no network, no broker calls. Advisory analytics only: nothing here
opens, gates, or closes any position. Covers threshold calculations, empty and
partial datasets, confidence levels, data coverage, and the two Telegram
command handlers.

Run:  python -X utf8 -m unittest test_threshold_engine -v
"""

import unittest
from unittest import mock

import threshold_engine as te
from oracle_analytics import AnalyticsConfig


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _trades():
    """Winners cluster at high score/edge; the single loser sits low."""
    return [
        {"symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 82, "volatility_edge": 0.03,
         "pnl": 90.0, "dte": 40, "iv_rank": 55},
        {"symbol": "META", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 45, "volatility_edge": 0.005, "pnl": -100.0,
         "dte": 10, "iv_rank": 20},
    ]


def _empty_cfg():
    return AnalyticsConfig(
        spread_trades_file="/nope/threshold_trades.json",
        expected_move_file="/nope/threshold_em.csv",
        training_dataset_file="/nope/threshold_ds.csv")


# --------------------------------------------------------------------------- #
# Profit factor
# --------------------------------------------------------------------------- #
class TestProfitFactor(unittest.TestCase):
    def test_no_trades_is_none(self):
        self.assertIsNone(te._profit_factor([]))

    def test_no_losses_is_inf(self):
        self.assertEqual(te._profit_factor([{"pnl": 10}, {"pnl": 5}]),
                         float("inf"))

    def test_ratio(self):
        self.assertEqual(te._profit_factor([{"pnl": 30}, {"pnl": -10}]), 3.0)

    def test_only_losses_is_zero(self):
        self.assertEqual(te._profit_factor([{"pnl": -10}, {"pnl": -5}]), 0.0)


# --------------------------------------------------------------------------- #
# Confidence levels (Req 8)
# --------------------------------------------------------------------------- #
class TestConfidence(unittest.TestCase):
    def test_low(self):
        self.assertEqual(te.compute_confidence(0), "Low")
        self.assertEqual(te.compute_confidence(49), "Low")

    def test_medium(self):
        self.assertEqual(te.compute_confidence(50), "Medium")
        self.assertEqual(te.compute_confidence(200), "Medium")

    def test_high(self):
        self.assertEqual(te.compute_confidence(201), "High")
        self.assertEqual(te.compute_confidence(1000), "High")


# --------------------------------------------------------------------------- #
# Oracle score thresholds (Req 2)
# --------------------------------------------------------------------------- #
class TestOracleScoreThresholds(unittest.TestCase):
    def test_empty(self):
        out = te.analyze_oracle_score_thresholds(_empty_cfg())
        self.assertIsNone(out["recommended_min_oracle_score"])
        self.assertEqual(len(out["rows"]), len(te.ORACLE_SCORE_THRESHOLDS))
        self.assertTrue(all(r["trades"] == 0 for r in out["rows"]))

    def test_rows_have_all_metrics(self):
        out = te.analyze_oracle_score_thresholds(_empty_cfg(), trades=_trades())
        for r in out["rows"]:
            for key in ("threshold", "trades", "win_rate", "pnl", "avg_pnl",
                        "profit_factor"):
                self.assertIn(key, r)

    def test_counts_and_recommendation(self):
        out = te.analyze_oracle_score_thresholds(_empty_cfg(), trades=_trades())
        row40 = next(r for r in out["rows"] if r["threshold"] == 40.0)
        self.assertEqual(row40["trades"], 3)
        self.assertEqual(row40["profit_factor"], 2.1)   # 210 / 100
        row80 = next(r for r in out["rows"] if r["threshold"] == 80.0)
        self.assertEqual(row80["trades"], 2)
        self.assertEqual(row80["profit_factor"], float("inf"))
        # least-restrictive winning cut (loser scored 45) -> >= 50
        self.assertEqual(out["recommended_min_oracle_score"], 50.0)

    def test_high_cut_drops_all(self):
        out = te.analyze_oracle_score_thresholds(_empty_cfg(), trades=_trades())
        row90 = next(r for r in out["rows"] if r["threshold"] == 90.0)
        self.assertEqual(row90["trades"], 0)


# --------------------------------------------------------------------------- #
# Vol edge thresholds (Req 3)
# --------------------------------------------------------------------------- #
class TestVolEdgeThresholds(unittest.TestCase):
    def test_empty(self):
        out = te.analyze_vol_edge_thresholds(_empty_cfg())
        self.assertIsNone(out["recommended_min_volatility_edge"])
        self.assertEqual(len(out["rows"]), len(te.VOL_EDGE_THRESHOLDS))

    def test_recommendation(self):
        out = te.analyze_vol_edge_thresholds(_empty_cfg(), trades=_trades())
        row0 = next(r for r in out["rows"] if r["threshold"] == 0.0)
        self.assertEqual(row0["trades"], 3)
        # loser edge 0.5% -> any cut >= 1% keeps only winners; least is 1%.
        self.assertEqual(out["recommended_min_volatility_edge"], 0.01)
        row4 = next(r for r in out["rows"] if r["threshold"] == 0.04)
        self.assertEqual(row4["trades"], 0)


# --------------------------------------------------------------------------- #
# DTE & IV-rank buckets (Req 4 / Req 5)
# --------------------------------------------------------------------------- #
class TestBuckets(unittest.TestCase):
    def test_dte_empty(self):
        out = te.analyze_dte_buckets(_empty_cfg())
        self.assertIsNone(out["recommended_dte_range"])
        self.assertEqual([r["label"] for r in out["rows"]],
                         ["0-14", "15-30", "31-60", "60+"])

    def test_dte_recommendation(self):
        out = te.analyze_dte_buckets(_empty_cfg(), trades=_trades())
        # both winners (35, 40 DTE) land in 31-60 -> best PnL bucket.
        self.assertEqual(out["recommended_dte_range"], "31-60")

    def test_iv_empty(self):
        out = te.analyze_iv_rank_buckets(_empty_cfg())
        self.assertIsNone(out["recommended_iv_rank_range"])
        self.assertEqual([r["label"] for r in out["rows"]],
                         ["0-25", "25-50", "50-75", "75-100"])

    def test_iv_recommendation(self):
        out = te.analyze_iv_rank_buckets(_empty_cfg(), trades=_trades())
        # winners IV 60/55 -> 50-75 bucket.
        self.assertEqual(out["recommended_iv_rank_range"], "50-75")


# --------------------------------------------------------------------------- #
# Strategy performance (Req 6)
# --------------------------------------------------------------------------- #
class TestStrategyPerformance(unittest.TestCase):
    def test_empty_lists_canonical_with_no_best(self):
        out = te.analyze_strategy_performance(_empty_cfg())
        self.assertIsNone(out["best_strategy"])
        self.assertIsNone(out["worst_strategy"])
        labels = [r["label"] for r in out["rows"]]
        for s in te.CANONICAL_STRATEGIES:
            self.assertIn(s, labels)

    def test_best_and_worst(self):
        out = te.analyze_strategy_performance(_empty_cfg(), trades=_trades())
        self.assertEqual(out["best_strategy"], "bullish_put_credit_spread")
        self.assertEqual(out["worst_strategy"], "iron_condor")

    def test_partial_unknown_strategy_included(self):
        trades = [{"strategy": "mystery_spread", "status": "closed", "pnl": 5.0}]
        out = te.analyze_strategy_performance(_empty_cfg(), trades=trades)
        self.assertIn("mystery_spread", [r["label"] for r in out["rows"]])
        self.assertEqual(out["best_strategy"], "mystery_spread")


# --------------------------------------------------------------------------- #
# Data coverage (Req 7 / DATA_COVERAGE)
# --------------------------------------------------------------------------- #
class TestDataCoverage(unittest.TestCase):
    def test_empty(self):
        c = te.compute_data_coverage(_empty_cfg())
        self.assertEqual(c["trades_analyzed"], 0)
        self.assertEqual(c["symbols_analyzed"], 0)
        for h in ("1d", "3d", "7d", "30d"):
            self.assertEqual(c["prediction_coverage"][h], 0.0)

    def test_coverage_decreases_with_horizon(self):
        em = [
            {"timestamp": "2025-01-01T00:00:00", "symbol": "SPY",
             "in_price": "500", "expected_move_1d": "5", "expected_move_30d": "27"},
            {"timestamp": "2025-01-02T00:00:00", "symbol": "SPY",
             "in_price": "503", "expected_move_1d": "5", "expected_move_30d": "27"},
        ]
        c = te.compute_data_coverage(_empty_cfg(), trades=_trades(), em_rows=em)
        # row0 has a >=1d-later observation; row1 does not -> 1/2 = 0.5
        self.assertEqual(c["prediction_coverage"]["1d"], 0.5)
        # nothing is 30 days later -> 0 coverage
        self.assertEqual(c["prediction_coverage"]["30d"], 0.0)
        self.assertEqual(c["trades_analyzed"], 3)
        # symbols counted across trades (SPY/QQQ/META) + em (SPY) = 3
        self.assertEqual(c["symbols_analyzed"], 3)


# --------------------------------------------------------------------------- #
# Aggregate recommendations + confidence
# --------------------------------------------------------------------------- #
class TestComputeRecommendations(unittest.TestCase):
    def test_empty_is_low_confidence(self):
        r = te.compute_recommendations(_empty_cfg())
        self.assertEqual(r["n_trades"], 0)
        self.assertEqual(r["confidence"], "Low")
        self.assertIsNone(r["recommended_min_oracle_score"])
        self.assertIsNone(r["best_strategy"])

    def test_full_bundle(self):
        r = te.compute_recommendations(_empty_cfg(), trades=_trades())
        self.assertEqual(r["n_trades"], 3)
        self.assertEqual(r["confidence"], "Low")
        self.assertEqual(r["recommended_min_oracle_score"], 50.0)
        self.assertEqual(r["recommended_min_volatility_edge"], 0.01)
        self.assertEqual(r["recommended_dte_range"], "31-60")
        self.assertEqual(r["recommended_iv_rank_range"], "50-75")
        self.assertEqual(r["best_strategy"], "bullish_put_credit_spread")
        self.assertEqual(r["worst_strategy"], "iron_condor")


# --------------------------------------------------------------------------- #
# Telegram command handlers (output only; never trade)
# --------------------------------------------------------------------------- #
class TestTelegramCommands(unittest.TestCase):
    def _bot(self):
        from telegram_bot import TelegramTradingBot
        return TelegramTradingBot()

    def test_threshold_recommendations_empty_message(self):
        import threshold_engine
        bot = self._bot()
        with mock.patch.object(
                threshold_engine, "compute_recommendations",
                return_value={"n_trades": 0}):
            msg = bot.threshold_recommendations(chat_id="x")
        self.assertIn("No closed paper spreads", msg)
        self.assertIn("Advisory", msg)

    def test_threshold_recommendations_populated(self):
        import threshold_engine
        bot = self._bot()
        rec = threshold_engine.compute_recommendations(_empty_cfg(),
                                                       trades=_trades())
        with mock.patch.object(
                threshold_engine, "compute_recommendations", return_value=rec):
            msg = bot.threshold_recommendations(chat_id="x")
        self.assertIn("Threshold Recommendations", msg)
        self.assertIn("Oracle Score", msg)
        self.assertIn(">= 50", msg)
        self.assertIn("Bullish Put Credit Spread", msg)
        self.assertIn("Iron Condor", msg)
        self.assertIn("Low", msg)            # confidence
        self.assertIn("Advisory only", msg)

    def test_data_coverage_message(self):
        import threshold_engine
        bot = self._bot()
        cov = {"trades_analyzed": 143, "symbols_analyzed": 18,
               "prediction_coverage": {"1d": 0.92, "3d": 0.80, "7d": 0.74,
                                       "30d": 0.41}}
        with mock.patch.object(
                threshold_engine, "compute_data_coverage", return_value=cov):
            msg = bot.data_coverage(chat_id="x")
        self.assertIn("Trades analyzed", msg)
        self.assertIn("143", msg)
        self.assertIn("18", msg)
        self.assertIn("92%", msg)
        self.assertIn("41%", msg)


if __name__ == "__main__":
    unittest.main()
