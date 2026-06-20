"""
Offline tests for Oracle 3.0 — the Intelligence-layer reports
(oracle_intelligence_reports) and their Telegram wiring.

No creds, no network, no broker. Every test injects records / contexts. Covers,
for the eight report families:
  1. compute_* never raises on empty / garbage input and yields INSUFFICIENT_DATA.
  2. format_* / generate_*_text always end with ANALYTICS_FOOTER.
  3. With enough injected evidence the verdict is OK and the numbers are sane.
  4. The 8 Telegram commands dispatch to a string ending with the footer.

oracle_intelligence_reports is STRICTLY analytics: it only reads and reports,
never opens, sizes, prices, blocks or alters any trade.
"""

import unittest

import oracle_intelligence_reports as oir
from oracle_intelligence_reports import (
    ANALYTICS_FOOTER, VERDICT_OK, VERDICT_INSUFFICIENT,
    compute_oracle_regime_report, generate_oracle_regime_report_text,
    compute_oracle_explain, generate_oracle_explain_text,
    compute_oracle_agent_report, generate_oracle_agent_report_text,
    compute_oracle_probability_report, generate_oracle_probability_report_text,
    compute_oracle_feature_importance, generate_oracle_feature_importance_text,
    compute_oracle_weight_changes, generate_oracle_weight_changes_text,
    generate_oracle_hypothesis_report_text,
    compute_oracle_regime_performance, generate_oracle_regime_performance_text,
)


def _records(n=14):
    return oir._synthetic_records(n)


class TestRegimeReport(unittest.TestCase):
    def test_with_context_ok(self):
        rep = compute_oracle_regime_report(
            regime_raw={"regime": "trending", "trend": "up",
                        "realized_vol": 0.2, "momentum": 0.06})
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertEqual(rep["label"], "TRENDING_BULL")

    def test_no_context_insufficient(self):
        rep = compute_oracle_regime_report()
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)
        txt = generate_oracle_regime_report_text()
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestExplainReport(unittest.TestCase):
    def test_with_ctx_ok(self):
        rep = compute_oracle_explain(
            "SPY", ctx={"trend": "up", "momentum": 0.08, "news_score": 0.5})
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertIn("p_call", rep["probability"])
        self.assertGreater(rep["probability"]["p_call"],
                           rep["probability"]["p_put"])

    def test_no_ctx_insufficient(self):
        txt = generate_oracle_explain_text("SPY", ctx=None)
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestAgentReport(unittest.TestCase):
    def test_ok_and_ranked(self):
        rep = compute_oracle_agent_report(records=_records())
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertTrue(rep["agents"])
        scores = [a["hit_rate"] for a in rep["agents"]]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_empty_insufficient(self):
        rep = compute_oracle_agent_report(records=[])
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)
        txt = generate_oracle_agent_report_text(records=[])
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestProbabilityReport(unittest.TestCase):
    def test_ok_brier(self):
        rep = compute_oracle_probability_report(records=_records())
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertIsNotNone(rep["brier"])
        self.assertGreaterEqual(rep["brier"], 0.0)

    def test_empty_insufficient(self):
        txt = generate_oracle_probability_report_text(records=[])
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestFeatureImportance(unittest.TestCase):
    def test_ok_sorted(self):
        rep = compute_oracle_feature_importance(records=_records())
        self.assertEqual(rep["verdict"], VERDICT_OK)
        imp = [f["importance"] for f in rep["features"]]
        self.assertEqual(imp, sorted(imp, reverse=True))

    def test_empty_insufficient(self):
        txt = generate_oracle_feature_importance_text(records=[])
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestWeightChanges(unittest.TestCase):
    def test_with_history_ok(self):
        rep = compute_oracle_weight_changes(
            history=[{"weights": {"trend": 1.0, "news": 1.0}},
                     {"weights": {"trend": 1.5, "news": 0.8}}])
        self.assertEqual(rep["verdict"], VERDICT_OK)
        self.assertEqual(rep["snapshots"], 2)
        self.assertIsNotNone(rep["drift"])

    def test_empty_insufficient(self):
        txt = generate_oracle_weight_changes_text(history=[])
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestHypothesisReport(unittest.TestCase):
    def test_footer_present(self):
        txt = generate_oracle_hypothesis_report_text(trades=[])
        self.assertIn(ANALYTICS_FOOTER, txt)


class TestRegimePerformance(unittest.TestCase):
    def test_ok_grouped(self):
        rep = compute_oracle_regime_performance(records=_records())
        self.assertEqual(rep["verdict"], VERDICT_OK)
        labels = {r["regime"] for r in rep["regimes"]}
        self.assertIn("TRENDING_BULL", labels)

    def test_empty_insufficient(self):
        txt = generate_oracle_regime_performance_text(records=[])
        self.assertIn(ANALYTICS_FOOTER, txt)
        self.assertIn(VERDICT_INSUFFICIENT, txt)


class TestNeverRaises(unittest.TestCase):
    def test_garbage(self):
        for junk in (None, 42, "x", [None, 42], {"weird": object()}):
            compute_oracle_agent_report(records=junk)
            compute_oracle_probability_report(records=junk)
            compute_oracle_feature_importance(records=junk)
            compute_oracle_regime_performance(records=junk)


class TestTelegramDispatch(unittest.TestCase):
    """Smoke: the 8 commands route to a footer-terminated string on empty data."""

    def _bot(self):
        import telegram_bot as tb
        bot = tb.TelegramTradingBot.__new__(tb.TelegramTradingBot)
        bot.supported_tickers = set()      # only attr touched before our branch
        return bot

    def test_eight_commands_dispatch(self):
        try:
            bot = self._bot()
        except Exception as exc:  # pragma: no cover
            self.skipTest(f"telegram_bot unavailable: {exc}")
        commands = [
            "ORACLE_REGIME", "ORACLE_EXPLAIN SPY", "ORACLE_AGENT_REPORT",
            "ORACLE_PROBABILITY_REPORT", "ORACLE_FEATURE_IMPORTANCE",
            "ORACLE_WEIGHT_CHANGES", "ORACLE_HYPOTHESIS_REPORT",
            "ORACLE_REGIME_PERFORMANCE",
        ]
        for cmd in commands:
            out = bot.process_command(cmd, 0)
            self.assertIsInstance(out, str, cmd)
            self.assertIn(ANALYTICS_FOOTER, out, cmd)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(oir._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
