"""Tests for advisory_attribution.py (Phase 9C — advisory attribution).

Advisory / additive: these tests assert the entry-time snapshot, the close-time
outcome append (no recomputation of advisory fields), the per-category
ADVISORY_PERFORMANCE metrics, formatting, the Telegram command — and that NO
execution / order / gating functions are referenced.
"""

import os
import tempfile
import unittest
from unittest import mock

import advisory_attribution as aa
import advisory_gate as ag
from oracle_analytics import AnalyticsConfig


def _empty_cfg():
    return AnalyticsConfig(spread_trades_file="/nonexistent/aa_t.json",
                           spread_positions_file="/nonexistent/aa_p.json",
                           expected_move_file="/nonexistent/aa_e.csv",
                           training_dataset_file="/nonexistent/aa_d.csv")


def _store():
    return os.path.join(tempfile.mkdtemp(), "advisory_attribution.json")


def _open_pos(tid="t1", symbol="SPY", **kw):
    pos = {"id": tid, "symbol": symbol,
           "strategy": "bullish_put_credit_spread", "oracle_score": 85,
           "volatility_edge": 0.035, "dte": 35, "iv_rank": 60,
           "timestamp": "2026-01-02T10:00:00"}
    pos.update(kw)
    return pos


class TestOpenSnapshot(unittest.TestCase):
    def test_build_open_snapshot_fields(self):
        snap = aa.build_open_snapshot(_open_pos(), config=_empty_cfg())
        for key in aa.OPEN_FIELDS:
            self.assertIn(key, snap)
        self.assertEqual(snap["trade_id"], "t1")
        self.assertEqual(snap["symbol"], "SPY")
        self.assertEqual(snap["date_opened"], "2026-01-02")
        self.assertEqual(snap["oracle_score"], 85)
        self.assertIsNotNone(snap["advisory_recommendation"])
        self.assertIn(snap["advisory_confidence"], {"LOW", "MEDIUM", "HIGH"})

    def test_open_snapshot_has_no_outcome_yet(self):
        snap = aa.build_open_snapshot(_open_pos(), config=_empty_cfg())
        for key in ("pnl", "pnl_percent", "win_loss", "date_closed", "exit_reason"):
            self.assertIsNone(snap[key])

    def test_threshold_checks_captured(self):
        snap = aa.build_open_snapshot(_open_pos(), config=_empty_cfg())
        self.assertEqual(set(snap["threshold_checks"]),
                         {"oracle_score", "vol_edge", "dte", "iv_rank", "strategy"})


class TestRecordOpenClose(unittest.TestCase):
    def test_record_open_persists_and_upserts(self):
        store = _store()
        s1 = aa.record_open(_open_pos(), config=_empty_cfg(), path=store)
        self.assertIsNotNone(s1)
        self.assertEqual(len(aa.load_snapshots(store)), 1)
        # re-open same id overwrites (no duplicate row)
        aa.record_open(_open_pos(oracle_score=90), config=_empty_cfg(), path=store)
        rows = aa.load_snapshots(store)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["oracle_score"], 90)

    def test_record_open_without_id_is_none(self):
        store = _store()
        self.assertIsNone(aa.record_open({"symbol": "SPY"}, config=_empty_cfg(),
                                         path=store))

    def test_record_close_appends_outcome_only(self):
        store = _store()
        opened = aa.record_open(_open_pos(), config=_empty_cfg(), path=store)
        rec_at_open = opened["advisory_recommendation"]
        conf_at_open = opened["advisory_confidence"]
        closed = aa.record_close({"id": "t1", "pnl": 120.0, "pnl_percent": 40.0,
                                  "exit_reason": "take_profit",
                                  "closed_at": "2026-01-05T15:00:00"}, path=store)
        self.assertEqual(closed["win_loss"], "win")
        self.assertEqual(closed["pnl"], 120.0)
        self.assertEqual(closed["pnl_percent"], 40.0)
        self.assertEqual(closed["exit_reason"], "take_profit")
        self.assertEqual(closed["date_closed"], "2026-01-05")
        # advisory fields untouched (no hindsight recomputation)
        self.assertEqual(closed["advisory_recommendation"], rec_at_open)
        self.assertEqual(closed["advisory_confidence"], conf_at_open)

    def test_record_close_loss(self):
        store = _store()
        aa.record_open(_open_pos(), config=_empty_cfg(), path=store)
        closed = aa.record_close({"id": "t1", "pnl": -50.0}, path=store)
        self.assertEqual(closed["win_loss"], "loss")

    def test_record_close_without_open_is_noop(self):
        store = _store()
        self.assertIsNone(aa.record_close({"id": "ghost", "pnl": 10.0}, path=store))


class TestPerformanceMetrics(unittest.TestCase):
    def setUp(self):
        self.snaps = [
            {"trade_id": "a", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": 100.0},
            {"trade_id": "b", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": 50.0},
            {"trade_id": "c", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": -40.0},
            {"trade_id": "d", "advisory_recommendation": ag.WEAK_SETUP, "pnl": -60.0},
            {"trade_id": "e", "advisory_recommendation": ag.WEAK_SETUP, "pnl": 20.0},
            {"trade_id": "f", "advisory_recommendation": ag.NEUTRAL, "pnl": None},  # open
        ]
        self.m = aa.compute_advisory_performance(snapshots=self.snaps)

    def test_strong_accept_category(self):
        sa = self.m["categories"][ag.STRONG_ACCEPT]
        self.assertEqual(sa["trades"], 3)
        self.assertAlmostEqual(sa["win_rate"], 2 / 3)
        self.assertEqual(sa["total_pnl"], 110.0)
        self.assertEqual(sa["avg_pnl"], round(110.0 / 3, 2))
        self.assertAlmostEqual(sa["profit_factor"], 150.0 / 40.0)

    def test_weak_setup_category(self):
        ws = self.m["categories"][ag.WEAK_SETUP]
        self.assertEqual(ws["trades"], 2)
        self.assertAlmostEqual(ws["win_rate"], 0.5)
        self.assertEqual(ws["total_pnl"], -40.0)
        self.assertAlmostEqual(ws["profit_factor"], round(20.0 / 60.0, 2), places=2)

    def test_open_trade_excluded_from_sample(self):
        self.assertEqual(self.m["sample_size"], 5)

    def test_all_categories_zero_filled(self):
        for label in aa.CATEGORIES:
            self.assertIn(label, self.m["categories"])
        self.assertEqual(self.m["categories"][ag.ACCEPT]["trades"], 0)
        self.assertEqual(self.m["categories"][ag.ACCEPT]["avg_pnl"], 0.0)

    def test_confidence_label(self):
        self.assertIn(self.m["confidence"], {"LOW", "MEDIUM", "HIGH"})

    def test_uncategorized_bucket(self):
        snaps = [{"trade_id": "x", "advisory_recommendation": "WeirdLabel",
                  "pnl": 10.0}]
        m = aa.compute_advisory_performance(snapshots=snaps)
        self.assertEqual(m["uncategorized"]["trades"], 1)
        self.assertEqual(m["sample_size"], 1)

    def test_empty_metrics_safe(self):
        m = aa.compute_advisory_performance(snapshots=[])
        self.assertEqual(m["sample_size"], 0)
        self.assertEqual(m["confidence"], "LOW")
        for label in aa.CATEGORIES:
            self.assertEqual(m["categories"][label]["trades"], 0)


class TestFormatting(unittest.TestCase):
    def test_format_contains_seen_categories(self):
        snaps = [
            {"trade_id": "a", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": 100.0},
            {"trade_id": "d", "advisory_recommendation": ag.WEAK_SETUP, "pnl": -60.0},
        ]
        txt = aa.format_advisory_performance(
            aa.compute_advisory_performance(snapshots=snaps))
        self.assertIn("Advisory Performance", txt)
        self.assertIn("*%s*" % ag.STRONG_ACCEPT, txt)
        self.assertIn("*%s*" % ag.WEAK_SETUP, txt)
        self.assertIn("Win Rate", txt)
        self.assertIn("Profit Factor", txt)
        self.assertIn("Confidence", txt)
        # zero-trade categories are hidden
        self.assertNotIn("*%s*" % ag.ACCEPT, txt)

    def test_format_empty(self):
        txt = aa.format_advisory_performance(
            aa.compute_advisory_performance(snapshots=[]))
        self.assertIn("No completed trades", txt)

    def test_pf_str(self):
        self.assertEqual(aa._pf_str(None), "n/a")
        self.assertEqual(aa._pf_str(float("inf")), "∞")
        self.assertEqual(aa._pf_str(2.4), "2.40")

    def test_pnl_str(self):
        self.assertEqual(aa._pnl_str(None), "n/a")
        self.assertEqual(aa._pnl_str(81), "+$81")
        self.assertEqual(aa._pnl_str(-540), "-$540")
        # values past +/-1000 use a thousands separator (regression: the
        # "%+,.0f" %% form raised ValueError on this branch)
        self.assertEqual(aa._pnl_str(1820), "+$1,820")
        self.assertEqual(aa._pnl_str(-3420.0), "-$3,420")

    def test_format_large_pnl_does_not_raise(self):
        snaps = [{"trade_id": str(i), "advisory_recommendation": ag.STRONG_ACCEPT,
                  "pnl": 500.0} for i in range(5)]  # total 2500 -> >= 1000
        txt = aa.format_advisory_performance(
            aa.compute_advisory_performance(snapshots=snaps))
        self.assertIn("+$2,500", txt)


class TestTelegramCommand(unittest.TestCase):
    def test_generate_text(self):
        snaps = [{"trade_id": "a", "advisory_recommendation": ag.STRONG_ACCEPT,
                  "pnl": 100.0}]
        with mock.patch.object(aa, "load_snapshots", return_value=snaps):
            txt = aa.generate_advisory_performance_text()
        self.assertIn("Advisory Performance", txt)

    def test_command_fails_soft_via_bot(self):
        import telegram_bot
        bot = telegram_bot.TelegramTradingBot()
        with mock.patch.object(aa, "generate_advisory_performance_text",
                               side_effect=RuntimeError("boom")):
            self.assertIn("Could not build advisory performance",
                          bot.advisory_performance())


class TestNoTrading(unittest.TestCase):
    def test_no_execution_imports_or_calls(self):
        with open("advisory_attribution.py", "r", encoding="utf-8") as fh:
            src = fh.read()
        forbidden = ("import requests", "place_order", "submit_order",
                     "create_order", "import smart_trader", "from smart_trader",
                     "import spread_paper_trader", "from spread_paper_trader")
        for token in forbidden:
            self.assertNotIn(token, src,
                             f"advisory_attribution must not reference {token!r}")


if __name__ == "__main__":
    unittest.main()
