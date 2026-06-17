"""
Offline tests for Phase 11A-4 — Calibration Reports (pure, no I/O, no network).

Records are fed in-memory via the ``records=`` argument (folded candidate
dicts), so nothing touches the JSONL store. Covers:
  - Triple Gap bucket assignment + per-bucket performance + predictive verdict
  - PoP calibration buckets (predicted vs actual win rate + error)
  - EV calibration buckets (expected EV vs realized PnL + error)
  - Signal separation verdict + best/weakest signal
  - INSUFFICIENT_DATA paths (empty store)
  - ANALYTICS_FOOTER present in every formatter
"""

import os
import tempfile
import unittest

import calibration_reports as c
import candidate_resolution as cr
from ev_attribution import ANALYTICS_FOOTER


def resolved(tg=None, pnl=0.0, *, pop=None, ev=None, ev_risk=None,
             oracle=None, edge=None, advisory=None, move=None,
             max_loss=440.0, selected=False,
             status=cr.RESOLUTION_EXPIRY):
    """A folded, resolved candidate dict carrying the outcome PnL."""
    return {
        "resolution_status": status,
        "triple_gap_score": tg,
        "hypothetical_hold_to_expiry_pnl": pnl,
        "actual_paper_pnl": None,
        "probability_of_profit": pop,
        "expected_value": ev,
        "ev_per_dollar_risk": ev_risk,
        "oracle_score": oracle,
        "volatility_edge": edge,
        "advisory_recommendation": advisory,
        "actual_move": move,
        "max_loss": max_loss,
        "selected_for_paper_trade": selected,
    }


def predictive_set():
    """5 low-Triple-Gap losers + 5 high-Triple-Gap winners (10 resolved)."""
    losers = [resolved(tg=50.0, pnl=-100.0, move=-0.02) for _ in range(5)]
    winners = [resolved(tg=95.0, pnl=100.0, move=0.02, selected=True)
               for _ in range(5)]
    return losers + winners


class TestTripleGapReport(unittest.TestCase):
    def test_empty_is_insufficient(self):
        rep = c.compute_triple_gap_report(records=[])
        self.assertEqual(rep["candidates"], 0)
        self.assertEqual(rep["verdict"], c.VERDICT_INSUFFICIENT)

    def test_unresolved_only_is_insufficient(self):
        recs = [resolved(tg=95.0, pnl=100.0, status=cr.RESOLUTION_UNRESOLVED)]
        rep = c.compute_triple_gap_report(records=recs)
        self.assertEqual(rep["resolved"], 0)
        self.assertEqual(rep["verdict"], c.VERDICT_INSUFFICIENT)

    def test_bucket_assignment_and_predictive_verdict(self):
        rep = c.compute_triple_gap_report(records=predictive_set())
        self.assertEqual(rep["candidates"], 10)
        self.assertEqual(rep["scored"], 10)
        self.assertEqual(rep["resolved"], 10)
        low = rep["buckets"]["TG <60"]
        high = rep["buckets"]["TG 90-100"]
        self.assertEqual(low["candidates"], 5)
        self.assertEqual(low["resolved"], 5)
        self.assertEqual(low["win_rate"], 0.0)
        self.assertEqual(high["candidates"], 5)
        self.assertEqual(high["win_rate"], 1.0)
        self.assertEqual(high["selected"], 5)
        self.assertEqual(low["selected"], 0)
        self.assertGreater(rep["separation_score"], 0)
        self.assertEqual(rep["verdict"], c.VERDICT_PREDICTIVE)


class TestPopCalibration(unittest.TestCase):
    def test_bucket_predicted_vs_actual(self):
        # 4 trades at PoP 0.75 -> bucket "PoP 70-80%"; 3 wins, 1 loss.
        trades = [resolved(pop=0.75, pnl=10.0) for _ in range(3)]
        trades.append(resolved(pop=0.75, pnl=-10.0))
        rep = c.compute_pop_calibration(records=trades)
        self.assertEqual(rep["sample_size"], 4)
        b = rep["buckets"]["PoP 70-80%"]
        self.assertEqual(b["trades"], 4)
        self.assertEqual(b["predicted_avg_pop"], 0.75)
        self.assertEqual(b["actual_win_rate"], 0.75)
        self.assertEqual(b["calibration_error"], 0.0)

    def test_only_resolved_count(self):
        trades = [resolved(pop=0.6, pnl=5.0),
                  resolved(pop=0.6, pnl=5.0,
                           status=cr.RESOLUTION_UNRESOLVED)]
        rep = c.compute_pop_calibration(records=trades)
        self.assertEqual(rep["sample_size"], 1)


class TestEvCalibration(unittest.TestCase):
    def test_expected_vs_realized(self):
        # 3 trades at EV 25 -> bucket "EV 20-50"; realized 30 each.
        trades = [resolved(ev=25.0, pnl=30.0) for _ in range(3)]
        rep = c.compute_ev_calibration(records=trades)
        self.assertEqual(rep["sample_size"], 3)
        b = rep["buckets"]["EV 20-50"]
        self.assertEqual(b["trades"], 3)
        self.assertEqual(b["avg_expected_ev"], 25.0)
        self.assertEqual(b["avg_realized_pnl"], 30.0)
        self.assertEqual(b["calibration_error"], 5.0)


class TestSignalSeparation(unittest.TestCase):
    def test_insufficient_when_below_min(self):
        rep = c.compute_signal_separation(records=[resolved(tg=95.0,
                                                            pnl=100.0)])
        self.assertEqual(rep["overall_verdict"], c.VERDICT_INSUFFICIENT)

    def test_predictive_triple_gap_is_best(self):
        rep = c.compute_signal_separation(records=predictive_set())
        self.assertEqual(rep["sample_size"], 10)
        self.assertEqual(rep["best_predictive_signal"], "Triple Gap")
        self.assertGreater(rep["separation_score_by_signal"]["Triple Gap"], 0)
        self.assertEqual(rep["overall_verdict"], c.VERDICT_PREDICTIVE)


class TestFormatters(unittest.TestCase):
    def test_triple_gap_formatter_has_footer(self):
        empty = c.format_triple_gap_report(
            c.compute_triple_gap_report(records=[]))
        self.assertIn(ANALYTICS_FOOTER, empty)
        self.assertIn(c.VERDICT_INSUFFICIENT, empty)
        full = c.format_triple_gap_report(
            c.compute_triple_gap_report(records=predictive_set()))
        self.assertIn(ANALYTICS_FOOTER, full)
        self.assertIn(c.VERDICT_PREDICTIVE, full)

    def test_signal_separation_formatter_has_footer(self):
        empty = c.format_signal_separation(
            c.compute_signal_separation(records=[]))
        self.assertIn(ANALYTICS_FOOTER, empty)
        self.assertIn(c.VERDICT_INSUFFICIENT, empty)
        full = c.format_signal_separation(
            c.compute_signal_separation(records=predictive_set()))
        self.assertIn(ANALYTICS_FOOTER, full)


class TestTopLevelEntriesFailOpen(unittest.TestCase):
    def test_generators_on_missing_store(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "nope.jsonl")
            tg_text = c.generate_triple_gap_report_text(jsonl_path=missing)
            sep_text = c.generate_signal_separation_text(jsonl_path=missing)
        for text in (tg_text, sep_text):
            self.assertIn(ANALYTICS_FOOTER, text)
            self.assertIn(c.VERDICT_INSUFFICIENT, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
