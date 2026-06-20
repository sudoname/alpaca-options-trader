"""
Offline tests for Oracle 3.0 — the Voting + Bayesian Probability engine
(oracle_voting).

No creds, no network, no broker. Every test injects simple vote stand-ins.
Covers:
  1. tally_votes -> {p_bull, p_bear, p_neutral} always sums to 1.0.
  2. bayesian_probability -> {p_call, p_put, p_no_trade} always sums to 1.0.
  3. Bullish slate -> p_call dominates; bearish -> p_put; all-neutral -> no-trade.
  4. The prior tilts the posterior; weights re-balance influence.
  5. No single agent flips a decision under bounded weights.
  6. Empty / garbage never raises; results are deterministic.

oracle_voting is PURE arithmetic: it never opens, sizes, prices, blocks or
alters a trade.
"""

import unittest

import oracle_voting as ov
from oracle_voting import tally_votes, bayesian_probability, prior_from_records


class _V:
    def __init__(self, name, bull, bear, conf):
        self.name = name
        self.bullish_score = bull
        self.bearish_score = bear
        self.confidence = conf


def _sums(d, keys):
    return abs(sum(d[k] for k in keys) - 1.0)


BULL = [_V("trend", 0.8, 0.0, 0.9), _V("news", 0.6, 0.0, 0.7),
        _V("breadth", 0.5, 0.0, 0.6), _V("liquidity", 0.0, 0.0, 0.8)]
BEAR = [_V("trend", 0.0, 0.8, 0.9), _V("news", 0.0, 0.6, 0.7)]
NEUTRAL = [_V("liquidity", 0.0, 0.0, 0.5), _V("volatility", 0.0, 0.0, 0.4)]


class TestTally(unittest.TestCase):
    def test_normalized(self):
        t = tally_votes(BULL)
        self.assertLess(_sums(t, ("p_bull", "p_bear", "p_neutral")), 1e-6)

    def test_bullish_leans_bull(self):
        t = tally_votes(BULL)
        self.assertGreater(t["p_bull"], t["p_bear"])

    def test_empty_is_all_neutral(self):
        t = tally_votes([])
        self.assertEqual(t["p_neutral"], 1.0)


class TestBayesian(unittest.TestCase):
    def test_normalized(self):
        b = bayesian_probability(BULL, prior=0.5)
        self.assertLess(_sums(b, ("p_call", "p_put", "p_no_trade")), 1e-6)

    def test_bullish_call(self):
        b = bayesian_probability(BULL, prior=0.5)
        self.assertGreater(b["p_call"], b["p_put"])

    def test_bearish_put(self):
        b = bayesian_probability(BEAR, prior=0.5)
        self.assertGreater(b["p_put"], b["p_call"])

    def test_neutral_no_trade(self):
        b = bayesian_probability(NEUTRAL, prior=0.5)
        self.assertGreater(b["p_no_trade"], 0.9)

    def test_prior_tilts(self):
        hi = bayesian_probability([_V("x", 0.3, 0.0, 0.5)], prior=0.8)
        lo = bayesian_probability([_V("x", 0.3, 0.0, 0.5)], prior=0.2)
        self.assertGreater(hi["p_call"], lo["p_call"])


class TestWeights(unittest.TestCase):
    def test_weights_rebalance(self):
        votes = [_V("a", 0.9, 0.0, 1.0), _V("b", 0.0, 0.9, 1.0)]
        a_heavy = bayesian_probability(votes, 0.5, {"a": 3.0, "b": 1.0})
        b_heavy = bayesian_probability(votes, 0.5, {"a": 1.0, "b": 3.0})
        self.assertGreater(a_heavy["p_call"], b_heavy["p_call"])

    def test_no_single_agent_flips_under_bounded_weights(self):
        # One strongly bearish agent against three bullish; with equal bounded
        # weights it cannot, by itself, flip the call to a put.
        votes = [_V("bear1", 0.0, 1.0, 1.0), _V("bull1", 0.9, 0.0, 0.9),
                 _V("bull2", 0.9, 0.0, 0.9), _V("bull3", 0.9, 0.0, 0.9)]
        b = bayesian_probability(votes, 0.5)
        self.assertGreater(b["p_call"], b["p_put"])


class TestPriorFromRecords(unittest.TestCase):
    def test_empty_is_uninformed(self):
        self.assertEqual(prior_from_records([]), 0.5)

    def test_clamped_range(self):
        recs = [{"pnl": 10.0} for _ in range(20)]      # all winners
        p = prior_from_records(recs)
        self.assertGreaterEqual(p, 0.2)
        self.assertLessEqual(p, 0.8)


class TestFailOpen(unittest.TestCase):
    def test_garbage_never_raises(self):
        for junk in (None, [], [None, 42, "x"], "x", 7):
            tally_votes(junk)            # type: ignore[arg-type]
            bayesian_probability(junk)   # type: ignore[arg-type]

    def test_deterministic(self):
        self.assertEqual(bayesian_probability(BULL, 0.5),
                         bayesian_probability(BULL, 0.5))


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(ov._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
