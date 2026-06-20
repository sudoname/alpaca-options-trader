"""
Offline tests for Oracle 3.0 — the Explainability Engine (oracle_explain).

No creds, no network, no broker. Every test injects vote stand-ins. Covers:
  1. agent_contributions are non-negative and sum to 1.0.
  2. A strong, confident agent out-contributes a weaker one; purely neutral
     agents contribute ~0 in directional attribution.
  3. Weights re-balance attribution.
  4. top_reasons / regime / probability are surfaced; summary mentions the prob.
  5. Empty / garbage never raises; results are deterministic; dict votes work.

oracle_explain is ANALYTICS / SHADOW ONLY: it attributes a decision and never
opens, sizes, prices, blocks or alters a trade.
"""

import unittest

import oracle_explain as oe
from oracle_explain import explain


class _V:
    def __init__(self, name, bull, bear, conf, reasons):
        self.name = name
        self.bullish_score = bull
        self.bearish_score = bear
        self.confidence = conf
        self.reasons = reasons


VOTES = [
    _V("trend", 0.8, 0.0, 0.9, ["up trend, momentum +0.080"]),
    _V("news", 0.6, 0.0, 0.7, ["news/sentiment +0.50"]),
    _V("liquidity", 0.0, 0.0, 0.8, ["spread 1.0%"]),
    _V("volatility", 0.0, 0.0, 0.4, ["calm vol 0.18"]),
]
PROB = {"p_call": 0.71, "p_put": 0.12, "p_no_trade": 0.17}


class TestContributions(unittest.TestCase):
    def test_sum_to_one(self):
        out = explain(VOTES, probability=PROB)
        c = out["agent_contributions"]
        self.assertLess(abs(sum(c.values()) - 1.0), 1e-6)
        for v in c.values():
            self.assertGreaterEqual(v, 0.0)

    def test_strong_agent_leads(self):
        c = explain(VOTES)["agent_contributions"]
        self.assertGreater(c["trend"], c["news"])

    def test_neutral_agent_negligible(self):
        c = explain(VOTES)["agent_contributions"]
        self.assertGreater(c["trend"], c["liquidity"])

    def test_weights_rebalance(self):
        c = explain(VOTES, weights={"news": 5.0, "trend": 0.25})[
            "agent_contributions"]
        self.assertGreater(c["news"], c["trend"])


class TestSurfacing(unittest.TestCase):
    def test_top_reasons(self):
        out = explain(VOTES, probability=PROB)
        self.assertTrue(out["top_reasons"])
        self.assertTrue(any("trend" in r for r in out["top_reasons"]))

    def test_regime_and_probability(self):
        out = explain(VOTES, probability=PROB, regime={"label": "TRENDING_BULL"})
        self.assertEqual(out["regime"], "TRENDING_BULL")
        self.assertEqual(out["probability"], PROB)
        self.assertIn("P(call)", out["summary_str"])


class TestEdgeCases(unittest.TestCase):
    def test_empty(self):
        out = explain([])
        self.assertEqual(out["agent_contributions"], {})
        self.assertEqual(out["top_reasons"], [])

    def test_all_neutral_still_normalized(self):
        out = explain([_V("a", 0.0, 0.0, 0.0, []), _V("b", 0.0, 0.0, 0.0, [])])
        self.assertLess(abs(sum(out["agent_contributions"].values()) - 1.0),
                        1e-6)

    def test_dict_votes(self):
        dv = [{"name": "trend", "bullish_score": 0.7, "bearish_score": 0.0,
               "confidence": 0.8, "reasons": ["r"]}]
        self.assertTrue(explain(dv)["agent_contributions"])

    def test_deterministic(self):
        self.assertEqual(explain(VOTES, probability=PROB),
                         explain(VOTES, probability=PROB))

    def test_garbage_never_raises(self):
        for junk in (None, 42, "x", [None, 42], {"weird": object()}):
            explain(junk)  # type: ignore[arg-type]


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(oe._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
