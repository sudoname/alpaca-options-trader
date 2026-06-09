"""
Offline tests for Phase 8D — daily Oracle report.

No creds, no network, no broker calls. Analytics only: nothing here opens,
gates, or closes any position. Covers empty + sample reports, the Telegram
command handler, the once-per-day scheduling predicate (including restart
de-duplication), the state file round-trip, and a guard that no execution
functions are referenced.

Run:  python -X utf8 -m unittest test_oracle_daily_report -v
"""

import os
import tempfile
import unittest
from datetime import datetime
from unittest import mock

import oracle_daily_report as odr
from oracle_analytics import AnalyticsConfig


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _trades():
    return [
        {"symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 45, "volatility_edge": 0.005, "pnl": -50.0,
         "dte": 12, "iv_rank": 20},
    ]


def _opens():
    return [{"symbol": "AAPL", "strategy": "debit_call_spread",
             "status": "open", "pnl": 25.0}]


def _empty_cfg():
    return AnalyticsConfig(
        spread_trades_file="/nope/dr_trades.json",
        spread_positions_file="/nope/dr_pos.json",
        expected_move_file="/nope/dr_em.csv",
        training_dataset_file="/nope/dr_ds.csv")


# --------------------------------------------------------------------------- #
# Empty data
# --------------------------------------------------------------------------- #
class TestEmptyReport(unittest.TestCase):
    def test_build_empty_is_safe(self):
        rep = odr.build_daily_report(_empty_cfg(), now=datetime(2025, 1, 2))
        self.assertEqual(rep["date"], "2025-01-02")
        self.assertEqual(rep["n_trades"], 0)
        self.assertEqual(rep["account"]["total_trades"], 0)
        self.assertEqual(rep["account"]["open_positions"], 0)
        self.assertEqual(rep["confidence"], "Low")
        self.assertIsNone(rep["best_strategy"])

    def test_format_empty_has_footer(self):
        rep = odr.build_daily_report(_empty_cfg(), now=datetime(2025, 1, 2))
        txt = odr.format_daily_report(rep)
        self.assertIn("Oracle Daily Report", txt)
        self.assertIn(odr.ANALYTICS_FOOTER, txt)
        self.assertIn("none", txt)  # no open positions


# --------------------------------------------------------------------------- #
# Sample data
# --------------------------------------------------------------------------- #
class TestSampleReport(unittest.TestCase):
    def test_build_with_sample(self):
        rep = odr.build_daily_report(_empty_cfg(), now=datetime(2025, 1, 2),
                                     trades=_trades(), positions=_opens())
        self.assertEqual(rep["account"]["total_trades"], 2)
        self.assertEqual(rep["account"]["open_positions"], 1)
        self.assertAlmostEqual(rep["account"]["closed_pnl"], 70.0, places=2)
        self.assertAlmostEqual(rep["account"]["win_rate"], 0.5, places=6)
        self.assertEqual(rep["best_strategy"], "bullish_put_credit_spread")

    def test_format_with_sample(self):
        rep = odr.build_daily_report(_empty_cfg(), now=datetime(2025, 1, 2),
                                     trades=_trades(), positions=_opens())
        txt = odr.format_daily_report(rep)
        self.assertIn("AAPL", txt)
        self.assertIn("Bullish Put Credit Spread", txt)
        self.assertIn("Threshold Recommendations", txt)
        self.assertIn(odr.ANALYTICS_FOOTER, txt)

    def test_generate_text_one_call(self):
        txt = odr.generate_daily_report_text(_empty_cfg(),
                                             now=datetime(2025, 1, 2),
                                             trades=_trades())
        self.assertIn("Oracle Daily Report", txt)
        self.assertIn(odr.ANALYTICS_FOOTER, txt)


# --------------------------------------------------------------------------- #
# Scheduling predicate + state file
# --------------------------------------------------------------------------- #
class TestScheduling(unittest.TestCase):
    def test_sends_after_time_when_not_sent(self):
        now = datetime(2025, 1, 2, 16, 20)
        self.assertTrue(odr.should_send_daily_report(now, 16, 15, None))

    def test_not_before_time(self):
        now = datetime(2025, 1, 2, 16, 10)
        self.assertFalse(odr.should_send_daily_report(now, 16, 15, None))

    def test_sends_once_per_day(self):
        now = datetime(2025, 1, 2, 16, 20)
        # already sent today -> must not send again
        self.assertFalse(odr.should_send_daily_report(now, 16, 15, "2025-01-02"))

    def test_no_duplicate_after_restart(self):
        # Simulate: send once, persist date, "restart" (fresh read), re-check.
        d = tempfile.mkdtemp()
        state = os.path.join(d, "state.json")
        now = datetime(2025, 1, 2, 16, 20)

        self.assertIsNone(odr.read_last_sent_date(state))
        self.assertTrue(odr.should_send_daily_report(
            now, 16, 15, odr.read_last_sent_date(state)))
        odr.write_last_sent_date(now.strftime("%Y-%m-%d"), state)

        # after restart the persisted date suppresses a duplicate send
        self.assertEqual(odr.read_last_sent_date(state), "2025-01-02")
        self.assertFalse(odr.should_send_daily_report(
            now, 16, 15, odr.read_last_sent_date(state)))

    def test_new_day_sends_again(self):
        next_day = datetime(2025, 1, 3, 16, 20)
        self.assertTrue(odr.should_send_daily_report(
            next_day, 16, 15, "2025-01-02"))

    def test_state_missing_is_none(self):
        self.assertIsNone(odr.read_last_sent_date("/nope/missing_state.json"))


# --------------------------------------------------------------------------- #
# Telegram command handler
# --------------------------------------------------------------------------- #
class TestTelegramCommand(unittest.TestCase):
    def _bot(self):
        from telegram_bot import TelegramTradingBot
        return TelegramTradingBot()

    def test_command_returns_report(self):
        bot = self._bot()
        with mock.patch.object(odr, "generate_daily_report_text",
                               return_value="📊 Oracle Daily Report — X"):
            msg = bot.oracle_daily_report(chat_id="x")
        self.assertIn("Oracle Daily Report", msg)

    def test_command_fails_soft(self):
        bot = self._bot()
        with mock.patch.object(odr, "generate_daily_report_text",
                               side_effect=RuntimeError("boom")):
            msg = bot.oracle_daily_report(chat_id="x")
        self.assertIn("Could not build the daily report", msg)

    def test_watcher_disabled_returns_immediately(self):
        bot = self._bot()
        bot.daily_oracle_report_enabled = False
        # Should return without entering the (infinite) loop.
        self.assertIsNone(bot.daily_oracle_report_watch())


# --------------------------------------------------------------------------- #
# Guard: analytics only — no execution functions referenced
# --------------------------------------------------------------------------- #
class TestNoTrading(unittest.TestCase):
    def test_no_execution_imports_or_calls(self):
        path = os.path.join(os.path.dirname(os.path.abspath(odr.__file__)),
                            "oracle_daily_report.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for forbidden in ("import requests", "place_order", "submit_order",
                          "create_order", "import smart_trader",
                          "from smart_trader", "import spread_paper_trader",
                          "from spread_paper_trader"):
            self.assertNotIn(forbidden, src,
                             f"daily report must not reference {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
