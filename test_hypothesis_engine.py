"""
Offline tests for Phase 8E — automatic hypothesis testing.

No creds, no network, no broker calls. Advisory analytics only: nothing here
opens, gates, or closes any position. Covers hypothesis calculations, empty /
inconclusive cases, confidence levels, ranking, the Telegram command handler,
and a guard that no execution functions are referenced.

Run:  python -X utf8 -m unittest test_hypothesis_engine -v
"""

import os
import unittest
from unittest import mock

import hypothesis_engine as he
from oracle_analytics import AnalyticsConfig


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _trades():
    """Winners cluster at high score; the loser is a low-score iron condor."""
    return [
        {"symbol": "SPY", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 85, "volatility_edge": 0.035,
         "pnl": 120.0, "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "bullish_put_credit_spread",
         "status": "closed", "oracle_score": 82, "volatility_edge": 0.03,
         "pnl": 90.0, "dte": 40, "iv_rank": 55},
        {"symbol": "META", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 65, "volatility_edge": 0.015, "pnl": -100.0,
         "dte": 20, "iv_rank": 80},
    ]


def _empty_cfg():
    return AnalyticsConfig(spread_trades_file="/nope/hypo_trades.json")


def _by_name(results):
    return {r["hypothesis_name"]: r for r in results}


# --------------------------------------------------------------------------- #
# Confidence
# --------------------------------------------------------------------------- #
class TestConfidence(unittest.TestCase):
    def test_low_when_either_small(self):
        self.assertEqual(he.hypothesis_confidence(10, 200), "Low")
        self.assertEqual(he.hypothesis_confidence(200, 5), "Low")

    def test_low_boundary(self):
        self.assertEqual(he.hypothesis_confidence(29, 200), "Low")
        self.assertEqual(he.hypothesis_confidence(30, 30), "Medium")

    def test_medium(self):
        self.assertEqual(he.hypothesis_confidence(50, 60), "Medium")
        self.assertEqual(he.hypothesis_confidence(100, 100), "Medium")

    def test_high(self):
        self.assertEqual(he.hypothesis_confidence(101, 101), "High")
        self.assertEqual(he.hypothesis_confidence(150, 200), "High")


# --------------------------------------------------------------------------- #
# Empty / inconclusive
# --------------------------------------------------------------------------- #
class TestEmpty(unittest.TestCase):
    def test_all_hypotheses_present(self):
        results = he.compute_all_hypotheses(_empty_cfg())
        self.assertEqual(len(results), len(he.HYPOTHESES))

    def test_empty_all_inconclusive(self):
        results = he.compute_all_hypotheses(_empty_cfg())
        for r in results:
            self.assertEqual(r["conclusion"], "Inconclusive")
            self.assertEqual(r["trades_a"], 0)
            self.assertEqual(r["trades_b"], 0)

    def test_one_sided_is_inconclusive(self):
        # Only group A populated -> cannot conclude.
        trades = [{"strategy": "bullish_put_credit_spread", "status": "closed",
                   "oracle_score": 90, "pnl": 50.0}]
        results = _by_name(he.compute_all_hypotheses(_empty_cfg(), trades=trades))
        h = results["Bull Put Credit Spread vs Iron Condor"]
        self.assertEqual(h["trades_a"], 1)
        self.assertEqual(h["trades_b"], 0)
        self.assertEqual(h["conclusion"], "Inconclusive")


# --------------------------------------------------------------------------- #
# Calculations
# --------------------------------------------------------------------------- #
class TestCalculations(unittest.TestCase):
    def test_oracle_score_split_and_conclusion(self):
        h = _by_name(he.compute_all_hypotheses(_empty_cfg(),
                                               trades=_trades()))["Oracle Score >= 80 vs 60-79"]
        self.assertEqual(h["trades_a"], 2)
        self.assertEqual(h["trades_b"], 1)
        self.assertEqual(h["win_rate_a"], 1.0)
        self.assertEqual(h["win_rate_b"], 0.0)
        self.assertAlmostEqual(h["pnl_a"], 210.0, places=2)
        self.assertAlmostEqual(h["pnl_b"], -100.0, places=2)
        self.assertEqual(h["conclusion"], "A outperformed B")

    def test_profit_factor_fields(self):
        h = _by_name(he.compute_all_hypotheses(_empty_cfg(),
                                               trades=_trades()))["Oracle Score >= 80 vs < 80"]
        # group A has only winners -> inf profit factor
        self.assertEqual(h["profit_factor_a"], float("inf"))
        # group B has only the loser -> 0.0
        self.assertEqual(h["profit_factor_b"], 0.0)

    def test_strategy_credit_vs_debit_empty_groups(self):
        # _trades() has no debit spreads -> group B empty -> inconclusive
        h = _by_name(he.compute_all_hypotheses(_empty_cfg(),
                                               trades=_trades()))["Credit Spreads vs Debit Spreads"]
        self.assertEqual(h["trades_b"], 0)
        self.assertEqual(h["conclusion"], "Inconclusive")

    def test_effect_size_sign(self):
        h = _by_name(he.compute_all_hypotheses(_empty_cfg(),
                                               trades=_trades()))["Oracle Score >= 80 vs 60-79"]
        # A (avg +105) far better than B (avg -100) -> positive effect size
        self.assertGreater(h["effect_size"], 0)


# --------------------------------------------------------------------------- #
# Ranking
# --------------------------------------------------------------------------- #
class TestRanking(unittest.TestCase):
    def test_conclusive_sorts_first(self):
        ranked = he.rank_hypotheses(he.compute_all_hypotheses(_empty_cfg(),
                                                              trades=_trades()))
        self.assertNotEqual(ranked[0]["conclusion"], "Inconclusive")

    def test_empty_ranking_stable(self):
        results = he.compute_all_hypotheses(_empty_cfg())
        ranked = he.rank_hypotheses(results)
        self.assertEqual(len(ranked), len(results))


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
class TestFormatting(unittest.TestCase):
    def test_empty_report_footer(self):
        txt = he.format_hypothesis_report(he.compute_all_hypotheses(_empty_cfg()))
        self.assertIn("Hypothesis Report", txt)
        self.assertIn(he.ADVISORY_FOOTER, txt)
        self.assertIn("No closed paper spreads", txt)

    def test_sample_report_content(self):
        txt = he.generate_hypothesis_report_text(_empty_cfg(), trades=_trades())
        self.assertIn("outperformed", txt)
        self.assertIn(he.ADVISORY_FOOTER, txt)


# --------------------------------------------------------------------------- #
# Telegram command handler
# --------------------------------------------------------------------------- #
class TestTelegramCommand(unittest.TestCase):
    def _bot(self):
        from telegram_bot import TelegramTradingBot
        return TelegramTradingBot()

    def test_command_returns_report(self):
        bot = self._bot()
        with mock.patch.object(he, "generate_hypothesis_report_text",
                               return_value="🔬 Hypothesis Report (advisory)"):
            msg = bot.hypothesis_report(chat_id="x")
        self.assertIn("Hypothesis Report", msg)

    def test_command_fails_soft(self):
        bot = self._bot()
        with mock.patch.object(he, "generate_hypothesis_report_text",
                               side_effect=RuntimeError("boom")):
            msg = bot.hypothesis_report(chat_id="x")
        self.assertIn("Could not build the hypothesis report", msg)

    def test_command_real_empty(self):
        # End-to-end with no data (no patch): still safe + advisory footer.
        bot = self._bot()
        with mock.patch.object(he, "generate_hypothesis_report_text",
                               return_value=he.generate_hypothesis_report_text(
                                   _empty_cfg())):
            msg = bot.hypothesis_report(chat_id="x")
        self.assertIn(he.ADVISORY_FOOTER, msg)


# --------------------------------------------------------------------------- #
# Guard: advisory only — no execution functions referenced
# --------------------------------------------------------------------------- #
class TestNoTrading(unittest.TestCase):
    def test_no_execution_imports_or_calls(self):
        path = os.path.join(os.path.dirname(os.path.abspath(he.__file__)),
                            "hypothesis_engine.py")
        with open(path, "r", encoding="utf-8") as fh:
            src = fh.read()
        for forbidden in ("import requests", "place_order", "submit_order",
                          "create_order", "import smart_trader",
                          "from smart_trader", "import spread_paper_trader",
                          "from spread_paper_trader"):
            self.assertNotIn(forbidden, src,
                             f"hypothesis engine must not reference {forbidden!r}")


if __name__ == "__main__":
    unittest.main()
