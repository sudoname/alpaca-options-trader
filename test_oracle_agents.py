"""
Offline tests for Oracle 3.0 — the 10 Evidence Agents (oracle_agents).

No creds, no network, no broker. Every test injects a ``ctx`` dict. Covers:
  1. run_agents always returns one vote per agent with the bull+bear<=1 invariant.
  2. Directional agents lean the right way on bullish / bearish context.
  3. The candlestick agent is confidence-capped (a lone pattern can't dominate).
  4. A single broken agent degrades to a neutral vote (fail-open).
  5. Empty / garbage ctx never raises.

oracle_agents is ANALYTICS / SHADOW ONLY: agents only emit evidence and never
open, size, price, block or alter a trade.
"""

import unittest

import oracle_agents as oa
from oracle_agents import run_agents, AGENTS, AgentVote, OracleAgentsConfig


CFG = OracleAgentsConfig()

BULL = {
    "trend": "up", "momentum": 0.08, "realized_vol": 0.18, "vix": 16.0,
    "volume_ratio": 1.6, "news_score": 0.5, "breadth": 0.4,
    "candlestick": {"pattern_name": "hammer", "bias": "bullish",
                    "confidence": 0.95, "requires_confirmation": True},
    "skew": 0.3, "iv_rank": 40.0, "rel_strength": 0.04, "rl_preference": 0.6,
    "spread_pct": 0.01, "open_interest": 5000, "option_volume": 2000,
}
BEAR = {"trend": "down", "momentum": -0.08, "news_score": -0.6,
        "breadth": -0.5, "rel_strength": -0.05, "rl_preference": -0.7}


class TestProtocol(unittest.TestCase):
    def test_one_vote_per_agent(self):
        votes = run_agents(BULL, CFG)
        self.assertEqual(len(votes), len(AGENTS))
        names = {v.name for v in votes}
        self.assertEqual(names, set(a.name for a in AGENTS))

    def test_invariants(self):
        for v in run_agents(BULL, CFG):
            self.assertIsInstance(v, AgentVote)
            self.assertGreaterEqual(v.bullish_score, 0.0)
            self.assertGreaterEqual(v.bearish_score, 0.0)
            self.assertLessEqual(v.bullish_score + v.bearish_score, 1.0 + 1e-9)
            self.assertGreaterEqual(v.confidence, 0.0)
            self.assertLessEqual(v.confidence, 1.0)
            self.assertAlmostEqual(
                v.neutral_score, 1.0 - v.bullish_score - v.bearish_score,
                places=6)


class TestDirection(unittest.TestCase):
    def test_bullish_context(self):
        by = {v.name: v for v in run_agents(BULL, CFG)}
        for name in ("trend", "news", "breadth", "relative_strength",
                     "rl_preference"):
            self.assertGreater(by[name].bullish_score, by[name].bearish_score,
                               f"{name} should be bullish")

    def test_bearish_context(self):
        by = {v.name: v for v in run_agents(BEAR, CFG)}
        for name in ("trend", "news", "breadth", "relative_strength",
                     "rl_preference"):
            self.assertGreater(by[name].bearish_score, by[name].bullish_score,
                               f"{name} should be bearish")


class TestCandlestickCap(unittest.TestCase):
    def test_candlestick_confidence_capped(self):
        by = {v.name: v for v in run_agents(BULL, CFG)}
        self.assertLessEqual(by["candlestick"].confidence,
                             CFG.candlestick_max_confidence + 1e-9)

    def test_candlestick_bias_respected(self):
        ctx = {"candlestick": {"pattern_name": "shooting_star",
                               "bias": "bearish", "confidence": 0.9}}
        by = {v.name: v for v in run_agents(ctx, CFG)}
        self.assertGreater(by["candlestick"].bearish_score,
                           by["candlestick"].bullish_score)


class TestFailOpen(unittest.TestCase):
    def test_one_broken_agent_is_neutral(self):
        class Boom:
            name = "boom"

            def evaluate(self, ctx, config):
                raise RuntimeError("kaboom")

        original = list(oa.AGENTS)
        try:
            oa.AGENTS.append(Boom())
            votes = {v.name: v for v in run_agents(BULL, CFG)}
            self.assertIn("boom", votes)
            self.assertEqual(votes["boom"].bullish_score, 0.0)
            self.assertEqual(votes["boom"].bearish_score, 0.0)
            # The rest still produced their votes.
            self.assertGreater(votes["trend"].bullish_score, 0.0)
        finally:
            oa.AGENTS[:] = original

    def test_empty_ctx_all_neutral(self):
        for v in run_agents({}, CFG):
            self.assertEqual(v.bullish_score, 0.0)
            self.assertEqual(v.bearish_score, 0.0)
            self.assertEqual(v.confidence, 0.0)

    def test_garbage_never_raises(self):
        for junk in (None, 42, "x", [], {"weird": object()}):
            run_agents(junk, CFG)  # type: ignore[arg-type]


class TestDeterminism(unittest.TestCase):
    def test_repeated_identical(self):
        a = [v.to_dict() for v in run_agents(BULL, CFG)]
        b = [v.to_dict() for v in run_agents(BULL, CFG)]
        self.assertEqual(a, b)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(oa._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
