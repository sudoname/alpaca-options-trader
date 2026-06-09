"""Tests for advisory_gate.py (Phase 9A — advisory trade gate).

Advisory / read-only: these tests assert the recommendation logic, threshold
checks, historical stats, log line, formatting and the Telegram command — and
that NO execution / order / gating functions are referenced.
"""

import unittest
from unittest import mock

import advisory_gate as ag
from oracle_analytics import AnalyticsConfig


def _empty_cfg():
    return AnalyticsConfig(spread_trades_file="/nonexistent/ag_t.json",
                           spread_positions_file="/nonexistent/ag_p.json",
                           expected_move_file="/nonexistent/ag_e.csv",
                           training_dataset_file="/nonexistent/ag_d.csv")


def _trades():
    return [
        {"id": "1", "symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "dte": 35, "iv_rank": 60},
        {"id": "2", "symbol": "QQQ", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 82, "volatility_edge": 0.03,
         "pnl": 90.0, "dte": 40, "iv_rank": 55},
        {"id": "3", "symbol": "META", "strategy": "iron_condor",
         "status": "closed", "oracle_score": 45, "volatility_edge": 0.005,
         "pnl": -100.0, "dte": 10, "iv_rank": 20},
    ]


class TestEmpty(unittest.TestCase):
    def test_empty_is_neutral_low(self):
        res = ag.evaluate_setup(oracle_score=85, volatility_edge=0.04, dte=40,
                                iv_rank=60, strategy="bullish_put_credit_spread",
                                config=_empty_cfg())
        self.assertEqual(res["recommendation"], ag.NEUTRAL)
        self.assertEqual(res["confidence"], "LOW")

    def test_output_schema(self):
        res = ag.evaluate_setup(config=_empty_cfg())
        for key in ("recommendation", "confidence", "checks",
                    "historical_win_rate", "historical_profit_factor"):
            self.assertIn(key, res)
        self.assertEqual(set(res["checks"]),
                         {"oracle_score", "vol_edge", "dte", "iv_rank", "strategy"})
        for v in res["checks"].values():
            self.assertIsInstance(v, bool)

    def test_recommendation_is_valid_label(self):
        res = ag.evaluate_setup(config=_empty_cfg())
        self.assertIn(res["recommendation"],
                      {ag.STRONG_ACCEPT, ag.ACCEPT, ag.NEUTRAL,
                       ag.WEAK_SETUP, ag.REJECT_CANDIDATE})


class TestChecks(unittest.TestCase):
    def test_check_min_unknown_threshold_passes(self):
        self.assertTrue(ag._check_min(5, None))

    def test_check_min_missing_value_fails(self):
        self.assertFalse(ag._check_min(None, 60))

    def test_check_min_boundary_inclusive(self):
        self.assertTrue(ag._check_min(60, 60))
        self.assertFalse(ag._check_min(59.9, 60))

    def test_check_bucket_membership(self):
        self.assertTrue(ag._check_bucket(40, "31-60", ag._DTE_PRED))
        self.assertFalse(ag._check_bucket(5, "31-60", ag._DTE_PRED))

    def test_check_bucket_unknown_label_passes(self):
        self.assertTrue(ag._check_bucket(5, None, ag._DTE_PRED))

    def test_check_strategy_worst_fails(self):
        rec = {"worst_strategy": "iron_condor", "best_strategy": "bullish_put_credit_spread"}
        self.assertFalse(ag._check_strategy("iron_condor", rec))
        self.assertTrue(ag._check_strategy("bullish_put_credit_spread", rec))
        self.assertTrue(ag._check_strategy(None, rec))


class TestRecommendationLogic(unittest.TestCase):
    def setUp(self):
        self.cfg = _empty_cfg()
        self.trades = _trades()

    def test_strong_setup_accepts(self):
        res = ag.evaluate_setup(oracle_score=88, volatility_edge=0.04, dte=38,
                                iv_rank=60, strategy="bullish_put_credit_spread",
                                config=self.cfg, trades=self.trades)
        self.assertIn(res["recommendation"], (ag.STRONG_ACCEPT, ag.ACCEPT))

    def test_weak_setup_checks_fail(self):
        res = ag.evaluate_setup(oracle_score=30, volatility_edge=0.0, dte=5,
                                iv_rank=10, strategy="iron_condor",
                                config=self.cfg, trades=self.trades)
        self.assertFalse(res["checks"]["oracle_score"])
        self.assertFalse(res["checks"]["vol_edge"])
        self.assertIn(res["recommendation"],
                      (ag.WEAK_SETUP, ag.REJECT_CANDIDATE, ag.NEUTRAL))

    def test_historical_stats_for_strategy(self):
        res = ag.evaluate_setup(oracle_score=88, volatility_edge=0.04, dte=38,
                                iv_rank=60, strategy="bullish_put_credit_spread",
                                config=self.cfg, trades=self.trades)
        # both bullish_put trades won -> win rate 1.0, PF inf (no losses)
        self.assertEqual(res["historical_win_rate"], 1.0)
        self.assertEqual(res["historical_profit_factor"], float("inf"))

    def test_classify_thresholds(self):
        rec = {"worst_strategy": None}
        hist = {"trades": 0, "win_rate": 0.0, "profit_factor": None}
        self.assertEqual(ag._classify(5, "x", rec, hist, 10), ag.STRONG_ACCEPT)
        self.assertEqual(ag._classify(4, "x", rec, hist, 10), ag.ACCEPT)
        self.assertEqual(ag._classify(3, "x", rec, hist, 10), ag.NEUTRAL)
        self.assertEqual(ag._classify(2, "x", rec, hist, 10), ag.WEAK_SETUP)
        self.assertEqual(ag._classify(1, "x", rec, hist, 10), ag.REJECT_CANDIDATE)

    def test_classify_no_data_is_neutral(self):
        rec = {"worst_strategy": None}
        hist = {"trades": 0, "win_rate": 0.0, "profit_factor": None}
        self.assertEqual(ag._classify(5, "x", rec, hist, 0), ag.NEUTRAL)

    def test_classify_historically_poor_rejects(self):
        rec = {"worst_strategy": None}
        # PF < 1.0 with enough samples -> REJECT_CANDIDATE despite passing checks
        hist = {"trades": 10, "win_rate": 0.3, "profit_factor": 0.4}
        self.assertEqual(ag._classify(5, "x", rec, hist, 10), ag.REJECT_CANDIDATE)


class TestLogLine(unittest.TestCase):
    def test_log_line_shape(self):
        res = ag.evaluate_setup(config=_empty_cfg())
        line = ag.log_advisory_gate("SPY", "iron_condor", 70, 0.02, res)
        self.assertTrue(line.startswith("[ADVISORY_GATE] "))
        for token in ("symbol=SPY", "strategy=iron_condor", "oracle_score=70",
                      "volatility_edge=0.02", "recommendation=",
                      "historical_win_rate=", "historical_profit_factor="):
            self.assertIn(token, line)


class TestFormatting(unittest.TestCase):
    def test_format_contains_sections(self):
        res = ag.evaluate_setup(config=_empty_cfg())
        features = {"oracle_score": 70, "volatility_edge": 0.02, "dte": 30,
                    "iv_rank": 50, "strategy": "iron_condor"}
        txt = ag.format_advisory_check("SPY", features, res)
        self.assertIn("Advisory Check — SPY", txt)
        self.assertIn("Recommendation:", txt)
        self.assertIn("Confidence:", txt)
        self.assertIn("Historical win rate:", txt)
        self.assertIn("Historical profit factor:", txt)
        self.assertIn(ag.ADVISORY_FOOTER, txt)

    def test_pf_str(self):
        self.assertEqual(ag._pf_str(None), "n/a")
        self.assertEqual(ag._pf_str(float("inf")), "∞")
        self.assertEqual(ag._pf_str(1.5), "1.50")


class TestSymbolFeatures(unittest.TestCase):
    def test_gather_from_trades(self):
        feats = ag.gather_symbol_features(
            "SPY", config=_empty_cfg(), em_rows=[], dataset_rows=[],
            trades=_trades(), positions=[])
        self.assertEqual(feats["symbol"], "SPY")
        self.assertEqual(feats["strategy"], "bullish_put_credit_spread")
        self.assertEqual(feats["dte"], 35)

    def test_gather_empty_symbol(self):
        feats = ag.gather_symbol_features("", config=_empty_cfg())
        self.assertIsNone(feats["strategy"])


class TestTelegramCommand(unittest.TestCase):
    def test_returns_text(self):
        with mock.patch.object(ag, "advisory_check_for_symbol",
                               return_value=({"oracle_score": 70,
                                              "volatility_edge": 0.02, "dte": 30,
                                              "iv_rank": 50, "strategy": "x"},
                                             ag.evaluate_setup(config=_empty_cfg()))):
            txt = ag.generate_advisory_check_text("SPY")
        self.assertIn("Advisory Check — SPY", txt)

    def test_empty_symbol_usage(self):
        self.assertIn("Usage", ag.generate_advisory_check_text(""))

    def test_command_fails_soft_via_bot(self):
        import telegram_bot
        bot = telegram_bot.TelegramTradingBot()
        with mock.patch.object(ag, "generate_advisory_check_text",
                               side_effect=RuntimeError("boom")):
            out = bot.advisory_check("SPY")
        self.assertIn("Could not run the advisory check", out)


class TestNoTrading(unittest.TestCase):
    def test_no_execution_imports_or_calls(self):
        with open("advisory_gate.py", "r", encoding="utf-8") as fh:
            src = fh.read()
        forbidden = ("import requests", "place_order", "submit_order",
                     "create_order", "import smart_trader", "from smart_trader",
                     "import spread_paper_trader", "from spread_paper_trader")
        for token in forbidden:
            self.assertNotIn(token, src, f"advisory_gate must not reference {token!r}")


if __name__ == "__main__":
    unittest.main()
