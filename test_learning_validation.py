"""Tests for learning_validation.py (Phase 9B — learning validation).

Advisory / read-only: these tests assert the Oracle-vs-RL decision derivation,
performance metrics, CSV dataset, formatting and the Telegram commands — and
that NO execution / order / gating functions are referenced.
"""

import os
import tempfile
import unittest
from unittest import mock

import learning_validation as lv
import oracle_analytics as oa
from oracle_analytics import AnalyticsConfig


def _empty_cfg():
    return AnalyticsConfig(spread_trades_file="/nonexistent/lv_t.json",
                           expected_move_file="/nonexistent/lv_e.csv",
                           training_dataset_file="/nonexistent/lv_d.csv")


def _trades():
    return [
        {"id": "a1", "symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "rl_recommendation": "TAKE"},
        {"id": "a2", "symbol": "QQQ", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 72, "volatility_edge": 0.01,
         "pnl": -80.0, "rl_recommendation": "SKIP"},
        {"id": "a3", "symbol": "META", "strategy": "iron_condor",
         "status": "closed", "oracle_score": 45, "volatility_edge": -0.01,
         "pnl": 40.0, "rl_recommendation": "TAKE"},
    ]


class TestDecisionDerivation(unittest.TestCase):
    def test_normalize_take_skip(self):
        self.assertEqual(lv._normalize_decision("TAKE"), lv.TAKE)
        self.assertEqual(lv._normalize_decision("call"), lv.TAKE)
        self.assertEqual(lv._normalize_decision("SKIP"), lv.SKIP)
        self.assertEqual(lv._normalize_decision("reject"), lv.SKIP)

    def test_normalize_unknown_is_none(self):
        self.assertIsNone(lv._normalize_decision(""))
        self.assertIsNone(lv._normalize_decision(None))
        self.assertIsNone(lv._normalize_decision("maybe"))

    def test_oracle_decision_threshold(self):
        self.assertEqual(lv._oracle_decision({"oracle_score": 85}, 60.0), lv.TAKE)
        self.assertEqual(lv._oracle_decision({"oracle_score": 45}, 60.0), lv.SKIP)

    def test_oracle_decision_missing_score_defaults_take(self):
        self.assertEqual(lv._oracle_decision({}, 60.0), lv.TAKE)

    def test_rl_decision_from_field(self):
        self.assertEqual(lv._rl_decision({"rl_recommendation": "SKIP"}, None), lv.SKIP)
        self.assertEqual(lv._rl_decision({"rl_action": "PUT"}, None), lv.TAKE)

    def test_rl_decision_from_lookup(self):
        self.assertEqual(lv._rl_decision({"id": "x"}, {"x": "TAKE"}), lv.TAKE)

    def test_rl_decision_unknown(self):
        self.assertIsNone(lv._rl_decision({"id": "x"}, None))


class TestRecords(unittest.TestCase):
    def test_record_count_and_fields(self):
        recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                           min_oracle_score=60.0)
        self.assertEqual(len(recs), 3)
        for key in ("trade_id", "symbol", "date", "strategy",
                    "oracle_decision", "rl_decision", "oracle_score",
                    "volatility_edge", "pnl", "win_loss", "actual_outcome"):
            self.assertIn(key, recs[0])

    def test_decisions_and_outcomes(self):
        recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                           min_oracle_score=60.0)
        d = {r["trade_id"]: r for r in recs}
        self.assertEqual((d["a1"]["oracle_decision"], d["a1"]["rl_decision"],
                          d["a1"]["win_loss"]), (lv.TAKE, lv.TAKE, "win"))
        self.assertEqual((d["a2"]["oracle_decision"], d["a2"]["rl_decision"],
                          d["a2"]["win_loss"]), (lv.TAKE, lv.SKIP, "loss"))
        self.assertEqual((d["a3"]["oracle_decision"], d["a3"]["rl_decision"],
                          d["a3"]["win_loss"]), (lv.SKIP, lv.TAKE, "win"))

    def test_empty_records(self):
        recs = lv.build_validation_records(_empty_cfg())
        self.assertEqual(recs, [])


class TestMetrics(unittest.TestCase):
    def setUp(self):
        self.recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                                min_oracle_score=60.0)
        self.m = lv.compute_rl_performance(records=self.recs)

    def test_win_rates(self):
        # Oracle TAKEs a1(win),a2(loss) -> 0.5 ; RL TAKEs a1(win),a3(win) -> 1.0
        self.assertAlmostEqual(self.m["oracle_win_rate"], 0.5)
        self.assertAlmostEqual(self.m["rl_win_rate"], 1.0)

    def test_profit_factors(self):
        self.assertEqual(self.m["oracle_profit_factor"], 1.5)  # 120 / 80
        self.assertEqual(self.m["rl_profit_factor"], float("inf"))  # no losses

    def test_agreement(self):
        self.assertAlmostEqual(self.m["agreement_rate"], 1 / 3)
        self.assertAlmostEqual(self.m["disagreement_rate"], 2 / 3)

    def test_sample_and_coverage(self):
        self.assertEqual(self.m["sample_size"], 3)
        self.assertEqual(self.m["rl_coverage"], 3)
        self.assertEqual(self.m["oracle_take_count"], 2)
        self.assertEqual(self.m["rl_take_count"], 2)

    def test_empty_metrics_safe(self):
        m = lv.compute_rl_performance(_empty_cfg())
        self.assertEqual(m["sample_size"], 0)
        self.assertEqual(m["agreement_rate"], 0.0)
        self.assertIsNone(m["oracle_profit_factor"])

    def test_unknown_rl_excluded_from_coverage(self):
        trades = [dict(_trades()[0]), dict(_trades()[1])]
        del trades[0]["rl_recommendation"]  # unknown RL for a1
        recs = lv.build_validation_records(_empty_cfg(), trades=trades,
                                           min_oracle_score=60.0)
        m = lv.compute_rl_performance(records=recs)
        self.assertEqual(m["rl_coverage"], 1)


class TestCSV(unittest.TestCase):
    def test_csv_header_and_rows(self):
        recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                           min_oracle_score=60.0)
        d = tempfile.mkdtemp()
        path = os.path.join(d, "lv.csv")
        self.assertTrue(lv.write_validation_csv(recs, path))
        back = oa.read_csv_rows(path)
        self.assertEqual(len(back), 3)
        self.assertEqual(list(back[0].keys()), lv.VALIDATION_CSV_FIELDS)

    def test_csv_empty(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "lv_empty.csv")
        self.assertTrue(lv.write_validation_csv([], path))
        self.assertEqual(oa.read_csv_rows(path), [])


class TestFormatting(unittest.TestCase):
    def test_rl_performance_text(self):
        recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                           min_oracle_score=60.0)
        m = lv.compute_rl_performance(records=recs)
        txt = lv.format_rl_performance(m)
        self.assertIn("RL Performance", txt)
        self.assertIn("Oracle win rate", txt)
        self.assertIn("RL win rate", txt)
        self.assertIn("Agreement rate", txt)
        self.assertIn("Sample size", txt)

    def test_rl_performance_empty(self):
        self.assertIn("No completed trades",
                      lv.format_rl_performance(lv.compute_rl_performance(_empty_cfg())))

    def test_validation_stats_text(self):
        recs = lv.build_validation_records(_empty_cfg(), trades=_trades(),
                                           min_oracle_score=60.0)
        m = lv.compute_rl_performance(records=recs)
        txt = lv.format_validation_stats(recs, m)
        self.assertIn("Validation Stats", txt)
        self.assertIn("Completed trades", txt)


class TestTelegramCommands(unittest.TestCase):
    def test_rl_performance_command(self):
        with mock.patch.object(lv, "build_validation_records",
                               return_value=lv.build_validation_records(
                                   _empty_cfg(), trades=_trades(),
                                   min_oracle_score=60.0)):
            txt = lv.generate_rl_performance_text()
        self.assertIn("RL Performance", txt)

    def test_validation_stats_command_writes_csv(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "lv_cmd.csv")
        with mock.patch.object(lv, "build_validation_records",
                               return_value=lv.build_validation_records(
                                   _empty_cfg(), trades=_trades(),
                                   min_oracle_score=60.0)):
            txt = lv.generate_validation_stats_text(config=_empty_cfg(),
                                                    write_csv=True, csv_path=path)
        self.assertIn("Validation Stats", txt)
        self.assertTrue(os.path.exists(path))

    def test_commands_fail_soft_via_bot(self):
        import telegram_bot
        bot = telegram_bot.TelegramTradingBot()
        with mock.patch.object(lv, "generate_rl_performance_text",
                               side_effect=RuntimeError("boom")):
            self.assertIn("Could not build RL performance", bot.rl_performance())
        with mock.patch.object(lv, "generate_validation_stats_text",
                               side_effect=RuntimeError("boom")):
            self.assertIn("Could not build validation stats", bot.validation_stats())


class TestNoTrading(unittest.TestCase):
    def test_no_execution_imports_or_calls(self):
        with open("learning_validation.py", "r", encoding="utf-8") as fh:
            src = fh.read()
        forbidden = ("import requests", "place_order", "submit_order",
                     "create_order", "import smart_trader", "from smart_trader",
                     "import spread_paper_trader", "from spread_paper_trader")
        for token in forbidden:
            self.assertNotIn(token, src,
                             f"learning_validation must not reference {token!r}")


if __name__ == "__main__":
    unittest.main()
