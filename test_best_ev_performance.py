"""
Offline tests for Phase 10F — Best EV Performance + Oracle Proof Report.

No creds, no network, no broker. Covers:
  - Best-EV runner trade identification (the EV belief stamp)
  - headline performance stats, averages and strategy breakdowns
  - best/worst strategy by profit factor
  - EV / EV-Risk / recommendation breakdown tables
  - Oracle proof report evidence + null-anchored PREDICTIVE /
    PROMISING_BUT_INCONCLUSIVE / NOT_PREDICTIVE_YET / INSUFFICIENT_DATA
    conclusions and confidence tiers (Phase 10G-D)
  - empty datasets and malformed rows (never raises)
  - Telegram output
  - no execution path touched (static guards)
"""

import os
import unittest

import advisory_gate as ag
import best_ev_performance as bep
import vol_forecast_scorecard as vfs
from best_ev_performance import (
    CONCLUSION_PREDICTIVE, CONCLUSION_PROMISING, CONCLUSION_NOT_PREDICTIVE,
    CONCLUSION_INSUFFICIENT, PROOF_QUESTION,
    is_best_ev_trade, load_best_ev_trades, compute_best_ev_performance,
    format_best_ev_performance, compute_proof_report, format_proof_report,
)
from ev_attribution import (
    ANALYTICS_FOOTER, VERDICT_YES, VERDICT_NO, VERDICT_INCONCLUSIVE,
)

HERE = os.path.dirname(os.path.abspath(__file__))

# Disk-free neutral vol scorecard injected so proof-report tests never read
# the real expected_move_history.csv.
NEUTRAL_SCORECARD = {"verdict": vfs.VERDICT_INCONCLUSIVE, "rows": 0}


def trade(i, pnl, ev=10.0, ratio=None, oracle=None, edge=None, adv=None,
          rec=None, strategy="bullish_put_credit_spread", max_loss=400.0,
          pop=None):
    """Synthetic closed Best-EV runner trade (carries the EV stamp)."""
    return {"id": f"b{i}", "pnl": pnl, "expected_value": ev,
            "ev_per_dollar_risk": ratio, "oracle_score": oracle,
            "volatility_edge": edge, "advisory_recommendation": adv,
            "ev_recommendation": rec, "strategy": strategy,
            "max_loss": max_loss, "probability_of_profit": pop}


def quad(rising=True, pop=None):
    """Four records correlating ALL four evidence dimensions.

    Low group (Oracle 0-39 / Edge <0% / EV<0 / WEAK_SETUP) and high group
    (Oracle 80-100 / Edge 3%+ / EV 50+ / STRONG_ACCEPT). rising=True gives
    the high group PF 2.0 and the low group PF 0.5; rising=False inverts.
    """
    lo_pnls, hi_pnls = ([50.0, -100.0], [200.0, -100.0])
    if not rising:
        lo_pnls, hi_pnls = hi_pnls, lo_pnls
    rows = []
    for j, pnl in enumerate(lo_pnls):
        rows.append(trade(f"lo{j}", pnl, ev=-5.0, oracle=35, edge=-0.004,
                          adv=ag.WEAK_SETUP, pop=pop))
    for j, pnl in enumerate(hi_pnls):
        rows.append(trade(f"hi{j}", pnl, ev=60.0, oracle=85, edge=0.034,
                          adv=ag.STRONG_ACCEPT, pop=pop))
    return rows


# --------------------------------------------------------------------------- #
# Best-EV trade identification
# --------------------------------------------------------------------------- #
class TestBestEvTradeFilter(unittest.TestCase):
    def test_ev_stamp_plus_pnl_is_best_ev(self):
        self.assertTrue(is_best_ev_trade({"expected_value": 12.0,
                                          "pnl": -4.0}))

    def test_plain_paper_trade_without_stamp_is_not(self):
        self.assertFalse(is_best_ev_trade({"pnl": 10.0}))

    def test_open_position_with_stamp_is_not(self):
        self.assertFalse(is_best_ev_trade({"expected_value": 12.0,
                                           "pnl": None}))

    def test_junk_rows_are_not(self):
        for row in (None, "x", 42, [], {},
                    {"expected_value": "junk", "pnl": 1.0}):
            self.assertFalse(is_best_ev_trade(row), row)

    def test_load_filters_mixed_trades_list(self):
        rows = [trade(1, 10.0), {"id": "plain", "pnl": 5.0}, "garbage",
                trade(2, -5.0)]
        loaded = load_best_ev_trades(trades=rows)
        self.assertEqual([r["id"] for r in loaded], ["b1", "b2"])

    def test_missing_trades_file_fails_open(self):
        from oracle_analytics import AnalyticsConfig
        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/bep_t.json",
            spread_positions_file="/nonexistent/bep_p.json",
            expected_move_file="/nonexistent/bep_e.csv",
            training_dataset_file="/nonexistent/bep_d.csv")
        self.assertEqual(load_best_ev_trades(config=cfg), [])


# --------------------------------------------------------------------------- #
# Best EV performance
# --------------------------------------------------------------------------- #
class TestBestEvPerformance(unittest.TestCase):
    def _trades(self):
        return [
            trade(1, 200.0, ev=60.0, ratio=0.25, oracle=85, edge=0.034,
                  rec=ag.STRONG_ACCEPT,
                  strategy="bullish_put_credit_spread"),
            trade(2, -100.0, ev=55.0, ratio=0.22, oracle=82, edge=0.030,
                  rec=ag.STRONG_ACCEPT,
                  strategy="bullish_put_credit_spread"),
            trade(3, 50.0, ev=-5.0, ratio=-0.02, oracle=35, edge=-0.004,
                  rec=ag.WEAK_SETUP,
                  strategy="bear_call_credit_spread"),
            trade(4, -100.0, ev=-10.0, ratio=-0.04, oracle=38, edge=-0.002,
                  rec=ag.WEAK_SETUP,
                  strategy="bear_call_credit_spread"),
        ]

    def test_headline_stats(self):
        perf = compute_best_ev_performance(trades=self._trades())
        m = perf["overall"]
        self.assertEqual(perf["sample_size"], 4)
        self.assertEqual(perf["confidence"], "Low")
        self.assertEqual(m["trades"], 4)
        self.assertEqual(m["wins"], 2)
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["total_pnl"], 50.0)
        self.assertAlmostEqual(m["profit_factor"], 1.25)  # 250 / 200

    def test_entry_belief_averages(self):
        perf = compute_best_ev_performance(trades=self._trades())
        self.assertAlmostEqual(perf["avg_expected_value"], 25.0)
        self.assertAlmostEqual(perf["avg_ev_per_risk"], 0.1025)
        self.assertAlmostEqual(perf["avg_oracle_score"], 60.0)
        self.assertAlmostEqual(perf["avg_volatility_edge"], 0.0145)

    def test_best_and_worst_strategy_by_profit_factor(self):
        # bull put: PF 200/100 = 2.0; bear call: PF 50/100 = 0.5.
        perf = compute_best_ev_performance(trades=self._trades())
        self.assertEqual(perf["best_strategy"], "bullish_put_credit_spread")
        self.assertEqual(perf["worst_strategy"], "bear_call_credit_spread")
        self.assertEqual(set(perf["by_strategy"]),
                         {"bullish_put_credit_spread",
                          "bear_call_credit_spread"})

    def test_breakdown_tables(self):
        perf = compute_best_ev_performance(trades=self._trades())
        self.assertEqual(perf["ev_buckets"]["EV 50+"]["trades"], 2)
        self.assertEqual(perf["ev_buckets"]["EV < 0"]["trades"], 2)
        self.assertEqual(perf["ev_risk_buckets"]["EV/Risk 0.20+"]["trades"], 2)
        self.assertEqual(
            perf["by_recommendation"][ag.STRONG_ACCEPT]["trades"], 2)
        self.assertEqual(perf["by_recommendation"][ag.NEUTRAL]["trades"], 0)

    def test_ev_predictiveness_rising_is_yes(self):
        perf = compute_best_ev_performance(trades=quad(rising=True))
        self.assertEqual(perf["ev_predictiveness"]["verdict"], VERDICT_YES)

    def test_empty_performance(self):
        perf = compute_best_ev_performance(trades=[])
        self.assertEqual(perf["sample_size"], 0)
        self.assertIsNone(perf["avg_expected_value"])
        self.assertIsNone(perf["best_strategy"])
        self.assertEqual(perf["ev_predictiveness"]["verdict"],
                         VERDICT_INCONCLUSIVE)

    def test_malformed_rows_never_raise(self):
        perf = compute_best_ev_performance(
            trades=["junk", 7, None, {}, {"expected_value": "x", "pnl": 1.0},
                    trade(1, 10.0)])
        self.assertEqual(perf["sample_size"], 1)
        self.assertIn(ANALYTICS_FOOTER, format_best_ev_performance(perf))


# --------------------------------------------------------------------------- #
# BEST_EV_PERFORMANCE Telegram output
# --------------------------------------------------------------------------- #
class TestBestEvPerformanceOutput(unittest.TestCase):
    def test_report_layout(self):
        perf = compute_best_ev_performance(trades=quad(rising=True))
        text = format_best_ev_performance(perf)
        self.assertIn("Best EV Paper Performance", text)
        self.assertIn("Trades: `4`", text)
        self.assertIn("Win Rate: `50%`", text)
        self.assertIn("Total PnL:", text)
        self.assertIn("Profit Factor:", text)
        self.assertIn("Avg EV:", text)
        self.assertIn("Avg EV/Risk:", text)
        self.assertIn("Best Strategy:", text)
        self.assertIn("Worst Strategy:", text)
        self.assertIn("Higher EV buckets outperform lower EV buckets: "
                      f"*{VERDICT_YES}*", text)
        self.assertIn("Confidence: *Low*", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_empty_report(self):
        text = format_best_ev_performance(
            compute_best_ev_performance(trades=[]))
        self.assertIn("No closed Best-EV paper trades yet.", text)
        self.assertIn("BEST_EV_PAPER_RUN", text)
        self.assertIn(ANALYTICS_FOOTER, text)


# --------------------------------------------------------------------------- #
# Oracle proof report
# --------------------------------------------------------------------------- #
class TestProofReport(unittest.TestCase):
    def test_rising_low_sample_is_promising_but_inconclusive(self):
        # 4 records: all four separation dims YES, but the null-anchored
        # checks have no data (no PoP stamps, EV n<10, neutral scorecard)
        # and confidence is Low -> cannot claim PREDICTIVE.
        report = compute_proof_report(records=quad(rising=True),
                                      best_ev_trades=quad(rising=True),
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["conclusion"], CONCLUSION_PROMISING)
        self.assertEqual(report["supportive"], 4)
        self.assertEqual(report["opposing"], 0)
        for dim in ("oracle_score", "volatility_edge", "expected_value",
                    "advisory_recommendation"):
            self.assertEqual(report["evidence"][dim]["verdict"], VERDICT_YES,
                             dim)
        for check in report["null_checks"].values():
            self.assertEqual(check["vote"], 0, check)

    def test_all_dimensions_inverted_is_not_predictive(self):
        report = compute_proof_report(records=quad(rising=False),
                                      best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["conclusion"], CONCLUSION_NOT_PREDICTIVE)
        self.assertEqual(report["supportive"], 0)
        self.assertEqual(report["opposing"], 4)
        for dim in ("oracle_score", "volatility_edge", "expected_value",
                    "advisory_recommendation"):
            self.assertEqual(report["evidence"][dim]["verdict"], VERDICT_NO,
                             dim)

    def test_inverted_at_scale_adds_ev_calibration_opposition(self):
        # 60 inverted records: EV calibration itself flips to
        # EV_NOT_PREDICTIVE (negative slope + inverted buckets) -> 5 opposing.
        report = compute_proof_report(records=quad(rising=False) * 15,
                                      best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["conclusion"], CONCLUSION_NOT_PREDICTIVE)
        self.assertEqual(report["opposing"], 5)
        self.assertEqual(report["null_checks"]["ev_calibration"]["vote"], -1)

    def test_null_anchored_predictive_needs_excess_win_rate(self):
        # 60 rising records that ALSO beat the null lines: win rate 50% vs
        # promised PoP 40% (+10pp excess), EV ranks (slope>0, buckets YES),
        # and the vol forecast beats IV -> 7 supportive, 0 opposing,
        # Medium confidence -> PREDICTIVE.
        records = quad(rising=True, pop=0.40) * 15
        report = compute_proof_report(
            records=records, best_ev_trades=[],
            scorecard={"verdict": vfs.VERDICT_FORECAST_BEATS_IV,
                       "rows": 500})
        self.assertEqual(report["conclusion"], CONCLUSION_PREDICTIVE)
        self.assertEqual(report["supportive"], 7)
        self.assertEqual(report["opposing"], 0)
        checks = report["null_checks"]
        self.assertEqual(checks["vol_forecast"]["vote"], 1)
        self.assertEqual(checks["pop_excess"]["vote"], 1)
        self.assertAlmostEqual(checks["pop_excess"]["excess_win_rate"], 0.10)
        self.assertEqual(checks["ev_calibration"]["vote"], 1)

    def test_winning_only_at_promised_pop_rate_is_not_supportive(self):
        # The null line: actual win rate == predicted PoP -> zero evidence.
        records = quad(rising=True, pop=0.50) * 15
        report = compute_proof_report(records=records, best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["null_checks"]["pop_excess"]["vote"], 0)
        self.assertNotEqual(report["conclusion"], CONCLUSION_PREDICTIVE)

    def test_losing_to_promised_pop_opposes(self):
        records = quad(rising=True, pop=0.70) * 15
        report = compute_proof_report(records=records, best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["null_checks"]["pop_excess"]["vote"], -1)

    def test_no_evidence_is_insufficient_data(self):
        report = compute_proof_report(records=[], best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["conclusion"], CONCLUSION_INSUFFICIENT)
        self.assertEqual(report["supportive"], 0)
        self.assertEqual(report["opposing"], 0)

    def test_confidence_tiers(self):
        rows = quad(rising=True)
        small = compute_proof_report(records=rows, best_ev_trades=[],
                                     scorecard=NEUTRAL_SCORECARD)
        medium = compute_proof_report(records=rows * 15, best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        large = compute_proof_report(records=rows * 65, best_ev_trades=[],
                                     scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(small["confidence"], "Low")
        self.assertEqual(medium["confidence"], "Medium")
        self.assertEqual(large["confidence"], "High")

    def test_embeds_best_ev_performance(self):
        report = compute_proof_report(records=quad(rising=True),
                                      best_ev_trades=quad(rising=True),
                                      scorecard=NEUTRAL_SCORECARD)
        self.assertEqual(report["best_ev"]["sample_size"], 4)
        self.assertEqual(report["question"], PROOF_QUESTION)


# --------------------------------------------------------------------------- #
# ORACLE_PROOF_REPORT Telegram output
# --------------------------------------------------------------------------- #
class TestProofReportOutput(unittest.TestCase):
    def test_report_layout(self):
        report = compute_proof_report(records=quad(rising=True),
                                      best_ev_trades=quad(rising=True),
                                      scorecard=NEUTRAL_SCORECARD)
        text = format_proof_report(report)
        self.assertIn("Oracle Proof Report", text)
        self.assertIn(PROOF_QUESTION, text)
        self.assertIn("*Oracle Score:*", text)
        self.assertIn("*Volatility Edge:*", text)
        self.assertIn("*Expected Value:*", text)
        self.assertIn("*Advisory Recommendation:*", text)
        self.assertIn("*Null-anchored checks:*", text)
        self.assertIn("*Vol forecast vs IV:*", text)
        self.assertIn("*Excess win rate vs PoP:*", text)
        self.assertIn("*EV calibration:*", text)
        self.assertIn("*Best-EV paper trades:* `4` trades", text)
        self.assertIn(f"*Overall conclusion:* {CONCLUSION_PROMISING} "
                      "(4 supportive / 0 opposing)", text)
        self.assertIn("Confidence: *Low*", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_predictive_layout_shows_excess_win_rate(self):
        report = compute_proof_report(
            records=quad(rising=True, pop=0.40) * 15, best_ev_trades=[],
            scorecard={"verdict": vfs.VERDICT_FORECAST_BEATS_IV,
                       "rows": 500})
        text = format_proof_report(report)
        self.assertIn("`+10.0pp`", text)
        self.assertIn("actual `50%` vs promised `40%`", text)
        self.assertIn(vfs.VERDICT_FORECAST_BEATS_IV, text)
        self.assertIn(f"*Overall conclusion:* {CONCLUSION_PREDICTIVE} "
                      "(7 supportive / 0 opposing)", text)

    def test_insufficient_evidence_lines(self):
        # One occupied bucket per dimension -> "insufficient data" lines.
        report = compute_proof_report(records=[trade(1, 10.0, ev=60.0,
                                                     oracle=85, edge=0.034,
                                                     adv=ag.STRONG_ACCEPT)],
                                      best_ev_trades=[],
                                      scorecard=NEUTRAL_SCORECARD)
        text = format_proof_report(report)
        self.assertIn("insufficient data", text)
        self.assertIn("*Best-EV paper trades:* none closed yet", text)
        self.assertIn(CONCLUSION_PROMISING, text)

    def test_empty_report(self):
        text = format_proof_report(
            compute_proof_report(records=[], best_ev_trades=[],
                                 scorecard=NEUTRAL_SCORECARD))
        self.assertIn("No closed paper spread trades yet", text)
        self.assertIn(CONCLUSION_INSUFFICIENT, text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_telegram_bot_wires_the_commands(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("BEST_EV_PERFORMANCE", src)
        self.assertIn("ORACLE_PROOF_REPORT", src)
        self.assertIn("def best_ev_performance", src)
        self.assertIn("def oracle_proof_report", src)


# --------------------------------------------------------------------------- #
# No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_live_modules_do_not_consume_the_report(self):
        for name in ("run_alpaca_intraday.py", "smart_trader.py"):
            src = self._read(name)
            self.assertNotIn("best_ev_performance", src,
                             f"{name} must not import best_ev_performance")

    def test_module_never_imports_live_trader_or_network(self):
        src = self._read("best_ev_performance.py")
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
