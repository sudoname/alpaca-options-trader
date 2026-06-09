"""
Offline tests for Phase 8B — Oracle analytics & validation layer.

No creds, no network, no broker calls. Every reader is exercised with empty,
partial and malformed inputs (analytics must fail open), and the five compute_*
functions are checked against synthetic data. Also covers the new analytics
fields captured on CLOSED simulated spread trades by spread_paper_trader.

Run:  python -X utf8 -m unittest test_oracle_analytics -v
"""

import csv
import json
import os
import tempfile
import unittest
from collections import OrderedDict

import oracle_analytics as oa
from oracle_analytics import (
    AnalyticsConfig,
    compute_learning_performance,
    compute_oracle_stats,
    compute_prediction_accuracy,
    compute_spread_performance,
    compute_vol_edge_leaderboard,
)


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _closed_trades():
    """Three closed simulated spreads: 2 wins / 1 loss, total PnL = +80."""
    return [
        {"symbol": "SPY", "strategy": "bull_put_credit_spread", "status": "closed",
         "oracle_score": 85, "volatility_edge": 0.035, "pnl": 120.0,
         "pnl_percent": 25, "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "bull_put_credit_spread", "status": "closed",
         "oracle_score": 72, "volatility_edge": 0.015, "pnl": -80.0,
         "pnl_percent": -20, "dte": 10, "iv_rank": 30},
        {"symbol": "META", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 55, "volatility_edge": -0.01, "pnl": 40.0,
         "pnl_percent": 8, "dte": 70, "iv_rank": 80},
    ]


def _tmp_config(d):
    return AnalyticsConfig(
        spread_trades_file=os.path.join(d, "trades.json"),
        spread_positions_file=os.path.join(d, "pos.json"),
        expected_move_file=os.path.join(d, "em.csv"),
        training_dataset_file=os.path.join(d, "ds.csv"),
        trade_history_file=os.path.join(d, "hist.json"),
    )


def _write_em_csv(path, rows):
    cols = ["timestamp", "symbol", "in_price", "expected_move_1d",
            "expected_move_3d", "expected_move_7d", "expected_move_30d",
            "market_expected_move", "volatility_edge", "in_dollars"]
    with open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            w.writerow({c: r.get(c, "") for c in cols})


# --------------------------------------------------------------------------- #
# Robust readers / coercion (fail open)
# --------------------------------------------------------------------------- #
class TestReaders(unittest.TestCase):
    def test_missing_files_are_empty(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        self.assertEqual(oa.read_csv_rows(cfg.expected_move_file), [])
        self.assertIsNone(oa.read_json(cfg.spread_trades_file))
        self.assertEqual(oa.load_closed_spread_trades(cfg), [])
        self.assertEqual(oa.load_open_spread_positions(cfg), [])

    def test_malformed_json_treated_as_empty(self):
        d = tempfile.mkdtemp()
        cfg = _tmp_config(d)
        with open(cfg.spread_trades_file, "w", encoding="utf-8") as fh:
            fh.write("{not valid json")
        self.assertIsNone(oa.read_json(cfg.spread_trades_file))
        self.assertEqual(oa.load_closed_spread_trades(cfg), [])

    def test_malformed_csv_treated_as_empty(self):
        d = tempfile.mkdtemp()
        cfg = _tmp_config(d)
        with open(cfg.expected_move_file, "w", encoding="utf-8") as fh:
            fh.write("garbage,,,\n\x00\x01")
        # never raises; returns a list (possibly with junk we then ignore)
        self.assertIsInstance(oa.read_csv_rows(cfg.expected_move_file), list)
        self.assertEqual(compute_vol_edge_leaderboard(cfg), [])

    def test_json_dict_not_list_is_empty(self):
        d = tempfile.mkdtemp()
        cfg = _tmp_config(d)
        with open(cfg.spread_trades_file, "w", encoding="utf-8") as fh:
            json.dump({"not": "a list"}, fh)
        self.assertEqual(oa.load_closed_spread_trades(cfg), [])

    def test_to_float_variants(self):
        self.assertEqual(oa._to_float("3.5"), 3.5)
        self.assertEqual(oa._to_float(2), 2.0)
        self.assertIsNone(oa._to_float("n/a"))
        self.assertIsNone(oa._to_float(""))
        self.assertIsNone(oa._to_float(None))
        self.assertIsNone(oa._to_float(True))  # bool is not a number here

    def test_get_case_insensitive(self):
        row = {"PnL": "12.5", "Oracle_Score": "80"}
        self.assertEqual(oa._get(row, "pnl"), "12.5")
        self.assertEqual(oa._get(row, "oracle_score"), "80")
        self.assertIsNone(oa._get(row, "missing"))


# --------------------------------------------------------------------------- #
# compute_oracle_stats
# --------------------------------------------------------------------------- #
class TestOracleStats(unittest.TestCase):
    def test_empty_is_safe(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        s = compute_oracle_stats(cfg)
        self.assertEqual(s["trades"], 0)
        self.assertEqual(s["total_pnl"], 0.0)
        self.assertEqual(s["win_rate"], 0.0)
        self.assertIsNone(s["avg_oracle_score"])
        self.assertIsNone(s["avg_volatility_edge"])
        self.assertEqual(s["open_positions"], 0)
        # expected_move_error has all horizons present (None when no data)
        for h in ("1d", "3d", "7d", "30d"):
            self.assertIn(h, s["expected_move_error"])

    def test_totals_and_win_rate(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        s = compute_oracle_stats(cfg, trades=_closed_trades())
        self.assertEqual(s["trades"], 3)
        self.assertAlmostEqual(s["total_pnl"], 80.0, places=2)
        self.assertAlmostEqual(s["win_rate"], 2 / 3, places=6)
        self.assertAlmostEqual(s["avg_oracle_score"], (85 + 72 + 55) / 3, places=4)
        self.assertAlmostEqual(s["avg_volatility_edge"],
                               (0.035 + 0.015 - 0.01) / 3, places=6)

    def test_open_pnl_summary(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        opens = [{"status": "open", "pnl": 15.0},
                 {"status": "open", "pnl": -5.0}]
        s = compute_oracle_stats(cfg, trades=[], positions=opens)
        self.assertEqual(s["open_positions"], 2)
        self.assertAlmostEqual(s["open_pnl"], 10.0, places=2)


# --------------------------------------------------------------------------- #
# compute_vol_edge_leaderboard
# --------------------------------------------------------------------------- #
class TestVolEdgeLeaderboard(unittest.TestCase):
    def test_empty(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        self.assertEqual(compute_vol_edge_leaderboard(cfg), [])

    def test_sorted_desc_with_latest_per_symbol(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        em = [
            {"timestamp": "2025-01-01T16:00:00+00:00", "symbol": "SPY",
             "in_price": "500", "expected_move_30d": "27",
             "market_expected_move": "30", "volatility_edge": "0.12"},
            # newer SPY row should win
            {"timestamp": "2025-01-02T16:00:00+00:00", "symbol": "SPY",
             "in_price": "503", "expected_move_30d": "27",
             "market_expected_move": "30", "volatility_edge": "0.10"},
            {"timestamp": "2025-01-01T16:00:00+00:00", "symbol": "NVDA",
             "in_price": "100", "expected_move_30d": "10",
             "market_expected_move": "11", "volatility_edge": "0.30"},
        ]
        board = compute_vol_edge_leaderboard(cfg, em_rows=em)
        self.assertEqual([e["symbol"] for e in board], ["NVDA", "SPY"])
        spy = next(e for e in board if e["symbol"] == "SPY")
        self.assertAlmostEqual(spy["volatility_edge"], 0.10, places=6)  # latest

    def test_oracle_score_joined_from_dataset(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        em = [{"timestamp": "2025-01-01T16:00:00+00:00", "symbol": "NVDA",
               "in_price": "100", "expected_move_30d": "10",
               "market_expected_move": "11", "volatility_edge": "0.30"}]
        ds = [{"symbol": "NVDA", "pred_oracle_score": "88"}]
        board = compute_vol_edge_leaderboard(cfg, em_rows=em, dataset_rows=ds)
        self.assertEqual(board[0]["oracle_score"], 88.0)

    def test_top_n_limit(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        em = [{"timestamp": "2025-01-01T00:00:00", "symbol": f"S{i}",
               "in_price": "10", "volatility_edge": str(i / 100.0)}
              for i in range(1, 8)]
        board = compute_vol_edge_leaderboard(cfg, em_rows=em, top_n=3)
        self.assertEqual(len(board), 3)


# --------------------------------------------------------------------------- #
# compute_spread_performance
# --------------------------------------------------------------------------- #
class TestSpreadPerformance(unittest.TestCase):
    def test_empty(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        self.assertEqual(compute_spread_performance(cfg), OrderedDict())

    def test_grouped_by_strategy(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        perf = compute_spread_performance(cfg, trades=_closed_trades())
        self.assertIn("bull_put_credit_spread", perf)
        self.assertEqual(perf["bull_put_credit_spread"]["trades"], 2)
        self.assertAlmostEqual(perf["bull_put_credit_spread"]["pnl"], 40.0, places=2)
        self.assertAlmostEqual(perf["bull_put_credit_spread"]["win_rate"], 0.5,
                               places=6)
        self.assertEqual(perf["iron_condor"]["trades"], 1)

    def test_sorted_by_pnl_desc(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        perf = compute_spread_performance(cfg, trades=_closed_trades())
        pnls = [a["pnl"] for a in perf.values()]
        self.assertEqual(pnls, sorted(pnls, reverse=True))


# --------------------------------------------------------------------------- #
# compute_learning_performance (bucketing)
# --------------------------------------------------------------------------- #
class TestLearningPerformance(unittest.TestCase):
    def test_empty(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        lp = compute_learning_performance(cfg)
        self.assertEqual(lp["n_trades"], 0)
        for key in ("by_oracle_score", "by_vol_edge", "by_dte", "by_iv_rank"):
            self.assertIn(key, lp)

    def test_buckets_assigned_correctly(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        lp = compute_learning_performance(cfg, trades=_closed_trades())
        self.assertEqual(lp["n_trades"], 3)
        # oracle 85 -> 80-100, 72 -> 60-79, 55 -> 40-59
        self.assertEqual(lp["by_oracle_score"]["80-100"]["trades"], 1)
        self.assertEqual(lp["by_oracle_score"]["60-79"]["trades"], 1)
        self.assertEqual(lp["by_oracle_score"]["40-59"]["trades"], 1)
        # edge 0.035 -> 3%+, 0.015 -> 1%-2%, -0.01 -> <0%
        self.assertEqual(lp["by_vol_edge"]["3%+"]["trades"], 1)
        self.assertEqual(lp["by_vol_edge"]["1%-2%"]["trades"], 1)
        self.assertEqual(lp["by_vol_edge"]["<0%"]["trades"], 1)
        # dte 35 -> 31-60, 10 -> 0-14, 70 -> 60+
        self.assertEqual(lp["by_dte"]["31-60"]["trades"], 1)
        self.assertEqual(lp["by_dte"]["0-14"]["trades"], 1)
        self.assertEqual(lp["by_dte"]["60+"]["trades"], 1)
        # iv 60 -> 50-75, 30 -> 25-50, 80 -> 75-100
        self.assertEqual(lp["by_iv_rank"]["50-75"]["trades"], 1)
        self.assertEqual(lp["by_iv_rank"]["25-50"]["trades"], 1)
        self.assertEqual(lp["by_iv_rank"]["75-100"]["trades"], 1)

    def test_none_values_skipped(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        trades = [{"strategy": "x", "status": "closed", "pnl": 5.0}]  # no buckets
        lp = compute_learning_performance(cfg, trades=trades)
        self.assertEqual(lp["n_trades"], 1)
        total_bucketed = sum(a["trades"] for a in lp["by_oracle_score"].values())
        self.assertEqual(total_bucketed, 0)


# --------------------------------------------------------------------------- #
# compute_prediction_accuracy (self-join)
# --------------------------------------------------------------------------- #
class TestPredictionAccuracy(unittest.TestCase):
    def test_empty(self):
        cfg = _tmp_config(tempfile.mkdtemp())
        acc = compute_prediction_accuracy(cfg)
        self.assertEqual(acc["n_rows"], 0)
        for h in ("1d", "3d", "7d", "30d"):
            self.assertEqual(acc["horizons"][h]["n"], 0)
            self.assertIsNone(acc["horizons"][h]["mae"])

    def test_one_matched_pair_1d(self):
        d = tempfile.mkdtemp()
        cfg = _tmp_config(d)
        em = [
            {"timestamp": "2025-01-01T16:00:00+00:00", "symbol": "SPY",
             "in_price": "500", "expected_move_1d": "5", "in_dollars": "True"},
            {"timestamp": "2025-01-02T16:00:00+00:00", "symbol": "SPY",
             "in_price": "503", "expected_move_1d": "5", "in_dollars": "True"},
        ]
        _write_em_csv(cfg.expected_move_file, em)
        acc = compute_prediction_accuracy(cfg)
        # realized = |503-500| = 3; predicted = 5; err = 2
        self.assertEqual(acc["horizons"]["1d"]["n"], 1)
        self.assertAlmostEqual(acc["horizons"]["1d"]["mae"], 2.0, places=6)

    def test_fractional_predicted_move_converted(self):
        d = tempfile.mkdtemp()
        cfg = _tmp_config(d)
        # in_dollars false -> expected_move treated as a fraction of price
        em = [
            {"timestamp": "2025-01-01T00:00:00", "symbol": "AAA",
             "in_price": "100", "expected_move_1d": "0.05", "in_dollars": "False"},
            {"timestamp": "2025-01-02T00:00:00", "symbol": "AAA",
             "in_price": "104", "expected_move_1d": "0.05", "in_dollars": "False"},
        ]
        _write_em_csv(cfg.expected_move_file, em)
        acc = compute_prediction_accuracy(cfg)
        # predicted = 0.05*100 = 5; realized = 4; err = 1
        self.assertAlmostEqual(acc["horizons"]["1d"]["mae"], 1.0, places=6)


# --------------------------------------------------------------------------- #
# spread_paper_trader analytics-field capture (Phase 8B, non-breaking)
# --------------------------------------------------------------------------- #
class TestSpreadPaperAnalyticsFields(unittest.TestCase):
    def setUp(self):
        from spread_paper_trader import SpreadPaperConfig, SpreadPaperTrader
        self.tmp = tempfile.mkdtemp()
        cfg = SpreadPaperConfig(
            enabled=True, min_oracle_score=70.0,
            positions_file=os.path.join(self.tmp, "positions.json"),
            trades_file=os.path.join(self.tmp, "trades.json"))
        self.trader = SpreadPaperTrader(cfg)

    def _proposal(self):
        from spread_builder import (BULLISH_PUT_CREDIT_SPREAD, SpreadLeg,
                                     SpreadProposal)
        legs = [SpreadLeg("sell", "put", 100, bid=1.95, ask=2.05),
                SpreadLeg("buy", "put", 95, bid=1.20, ask=1.30)]
        return SpreadProposal(
            strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol="SPY", legs=legs,
            net_credit_or_debit=0.75, max_profit=75.0, max_loss=425.0,
            breakeven=99.25, width=5.0, oracle_score=80.0)

    def test_open_captures_context_fields(self):
        ctx = {"volatility_edge": 0.035, "expected_move": 12.0,
               "market_expected_move": 10.0, "dte": 35, "iv_rank": 60,
               "entry_underlying_price": 500.0}
        pos = self.trader.open_position(self._proposal(), context=ctx)["position"]
        self.assertAlmostEqual(pos["volatility_edge"], 0.035, places=6)
        self.assertAlmostEqual(pos["expected_move"], 12.0, places=6)
        self.assertAlmostEqual(pos["market_expected_move"], 10.0, places=6)
        self.assertAlmostEqual(pos["dte"], 35.0, places=6)
        self.assertAlmostEqual(pos["iv_rank"], 60.0, places=6)
        self.assertAlmostEqual(pos["entry_underlying_price"], 500.0, places=6)
        self.assertIsNone(pos["actual_move"])

    def test_open_without_context_stores_none(self):
        pos = self.trader.open_position(self._proposal())["position"]
        for f in ("volatility_edge", "expected_move", "market_expected_move",
                  "dte", "iv_rank", "entry_underlying_price", "actual_move"):
            self.assertIsNone(pos[f])

    def test_close_finalizes_full_analytics_schema(self):
        from spread_paper_trader import ANALYTICS_FIELDS
        ctx = {"volatility_edge": 0.035, "dte": 35, "iv_rank": 60,
               "entry_underlying_price": 500.0}
        pos = self.trader.open_position(self._proposal(), context=ctx)["position"]
        closed = self.trader.close_position(
            pos["id"], exit_reason="take_profit",
            context={"exit_underlying_price": 507.0})
        # every analytics field exists on the closed record
        for field in ANALYTICS_FIELDS:
            self.assertIn(field, closed)
        # actual_move derived from exit - entry underlying price
        self.assertAlmostEqual(closed["actual_move"], 7.0, places=4)
        # date stamped as YYYY-MM-DD from closed_at
        self.assertRegex(closed["date"], r"^\d{4}-\d{2}-\d{2}$")
        self.assertEqual(closed["exit_reason"], "take_profit")

    def test_close_actual_move_explicit_context(self):
        pos = self.trader.open_position(self._proposal())["position"]
        closed = self.trader.close_position(
            pos["id"], context={"actual_move": -3.5})
        self.assertAlmostEqual(closed["actual_move"], -3.5, places=4)

    def test_closed_trade_is_analytics_consumable(self):
        # End-to-end: close a paper trade, then read it back via the analytics
        # loader and confirm the stats layer can consume it.
        ctx = {"volatility_edge": 0.02, "dte": 20, "iv_rank": 40,
               "entry_underlying_price": 500.0}
        pos = self.trader.open_position(self._proposal(), context=ctx)["position"]
        self.trader.close_position(pos["id"], exit_reason="take_profit")
        cfg = AnalyticsConfig(
            spread_trades_file=self.trader.config.trades_file,
            spread_positions_file=self.trader.config.positions_file)
        loaded = oa.load_closed_spread_trades(cfg)
        self.assertEqual(len(loaded), 1)
        stats = compute_oracle_stats(cfg)
        self.assertEqual(stats["trades"], 1)


if __name__ == "__main__":
    unittest.main()
