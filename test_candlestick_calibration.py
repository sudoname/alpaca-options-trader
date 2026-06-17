"""
Offline tests for Phase 11B-3 — Candlestick calibration (pure, no I/O).

Records are fed in-memory via ``records=`` (folded candidate dicts), so nothing
touches the JSONL store. Covers:
  - per-pattern stats (occurrences / resolved / win_rate / profit_factor)
  - ev_impact Positive / Negative / Neutral
  - LOW_SAMPLE_WARNING below MIN_PATTERN_SAMPLE_SIZE
  - leaderboards (occurrences / win_rate / profit_factor)
  - neutral-bias patterns skipped
  - overall tagged-vs-untagged "did patterns help" verdict
  - daily summary helper
  - empty store -> clean report; ANALYTICS_FOOTER present in formatters
"""

import unittest

import candidate_resolution as cr
import candlestick_calibration as cc
from ev_attribution import ANALYTICS_FOOTER


def rec(pattern, bias, pnl, *, move=0.0, selected=False, pop=0.6, ev=20.0,
        tg=70.0, oracle=80.0, edge=0.03, advisory="ACCEPT", conf=0.7,
        status=cr.RESOLUTION_EXPIRY):
    """A folded, resolved candidate carrying a candlestick stamp + outcome."""
    return {
        "resolution_status": status,
        "candlestick_pattern": pattern,
        "candlestick_bias": bias,
        "candlestick_confidence": conf,
        "hypothetical_hold_to_expiry_pnl": pnl,
        "actual_paper_pnl": None,
        "actual_move": move,
        "max_loss": 440.0,
        "probability_of_profit": pop,
        "expected_value": ev,
        "triple_gap_score": tg,
        "oracle_score": oracle,
        "volatility_edge": edge,
        "advisory_recommendation": advisory,
        "selected_for_paper_trade": selected,
    }


def winners(pattern, bias, n, **kw):
    return [rec(pattern, bias, 100.0, move=0.02, selected=True, **kw)
            for _ in range(n)]


def losers(pattern, bias, n, **kw):
    return [rec(pattern, bias, -80.0, move=-0.02, **kw) for _ in range(n)]


class TestPerPatternStats(unittest.TestCase):
    def test_positive_ev_impact(self):
        recs = winners("bullish_engulfing", "bullish", 3) + \
            losers("bullish_engulfing", "bullish", 1)
        rep = cc.compute_candlestick_calibration(records=recs)
        p = rep["patterns"]["bullish_engulfing"]
        self.assertEqual(p["occurrences"], 4)
        self.assertEqual(p["resolved_trades"], 4)
        self.assertEqual(p["win_rate"], 0.75)
        self.assertEqual(p["ev_impact"], cc.EV_IMPACT_POSITIVE)
        self.assertEqual(p["options_outcomes"]["selected"], 3)

    def test_negative_ev_impact(self):
        recs = losers("shooting_star", "bearish", 4)
        rep = cc.compute_candlestick_calibration(records=recs)
        p = rep["patterns"]["shooting_star"]
        self.assertEqual(p["win_rate"], 0.0)
        self.assertEqual(p["ev_impact"], cc.EV_IMPACT_NEGATIVE)

    def test_excursion_proxies(self):
        recs = winners("morning_star", "bullish", 2) + \
            losers("morning_star", "bullish", 2)
        rep = cc.compute_candlestick_calibration(records=recs)
        p = rep["patterns"]["morning_star"]
        self.assertEqual(p["average_max_favorable_excursion"], 0.02)
        self.assertEqual(p["average_max_adverse_excursion"], -0.02)

    def test_low_sample_warning(self):
        recs = winners("hammer", "bullish", 3)
        rep = cc.compute_candlestick_calibration(records=recs)
        p = rep["patterns"]["hammer"]
        self.assertTrue(p["low_sample"])
        self.assertEqual(p["warning"], cc.LOW_SAMPLE_WARNING)

    def test_advisory_distribution(self):
        recs = (winners("piercing_line", "bullish", 2, advisory="ACCEPT")
                + losers("piercing_line", "bullish", 1, advisory="NEUTRAL"))
        rep = cc.compute_candlestick_calibration(records=recs)
        dist = rep["patterns"]["piercing_line"]["advisory_distribution"]
        self.assertEqual(dist["ACCEPT"], 2)
        self.assertEqual(dist["NEUTRAL"], 1)


class TestNeutralSkipped(unittest.TestCase):
    def test_neutral_bias_excluded(self):
        recs = winners("doji", "neutral", 5) + winners("hammer", "bullish", 2)
        rep = cc.compute_candlestick_calibration(records=recs)
        self.assertNotIn("doji", rep["patterns"])
        self.assertIn("hammer", rep["patterns"])


class TestLeaderboards(unittest.TestCase):
    def test_ranked_by_win_rate(self):
        recs = winners("bullish_engulfing", "bullish", 4)
        recs += losers("shooting_star", "bearish", 3)
        rep = cc.compute_candlestick_calibration(records=recs)
        wr = rep["leaderboards"]["by_win_rate"]
        self.assertEqual(wr[0]["pattern_name"], "bullish_engulfing")
        occ = rep["leaderboards"]["by_occurrences"]
        self.assertEqual(occ[0]["pattern_name"], "bullish_engulfing")


class TestOverall(unittest.TestCase):
    def test_improved_ev_true(self):
        recs = winners("bullish_engulfing", "bullish", 4)
        # Untagged resolved losers (no candlestick pattern).
        for _ in range(3):
            r = rec(None, None, -50.0)
            r["candlestick_pattern"] = None
            recs.append(r)
        rep = cc.compute_candlestick_calibration(records=recs)
        overall = rep["overall"]
        self.assertEqual(overall["tagged"]["trades"], 4)
        self.assertEqual(overall["untagged"]["trades"], 3)
        self.assertTrue(overall["improved_ev"])

    def test_improved_ev_none_without_both_groups(self):
        recs = winners("hammer", "bullish", 3)
        rep = cc.compute_candlestick_calibration(records=recs)
        self.assertIsNone(rep["overall"]["improved_ev"])


class TestDailySummary(unittest.TestCase):
    def test_compact_summary(self):
        recs = winners("bullish_engulfing", "bullish", 4)
        s = cc.compute_daily_candlestick_summary(records=recs)
        self.assertEqual(s["patterns_detected"], 1)
        self.assertEqual(s["sample_size"], 4)
        self.assertEqual(s["top_patterns"][0]["pattern_name"],
                         "bullish_engulfing")

    def test_summary_fail_open(self):
        # A bad records type must not raise; helper returns a dict.
        s = cc.compute_daily_candlestick_summary(records=[None, 5, "x"])
        self.assertIsInstance(s, dict)


class TestFormatters(unittest.TestCase):
    def test_empty_report_is_clean(self):
        text = cc.generate_candlestick_report_text(records=[])
        self.assertIn(cc.NO_DATA_MSG, text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_full_report_has_footer_and_sections(self):
        recs = winners("bullish_engulfing", "bullish", 4) + \
            losers("shooting_star", "bearish", 2)
        text = cc.generate_candlestick_report_text(records=recs)
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn("bullish_engulfing", text)
        self.assertIn(cc.EV_IMPACT_POSITIVE, text)


if __name__ == "__main__":
    unittest.main(verbosity=2)
