"""
Offline tests for Phase 12 — Strategy EV Matrix (EV-first analytics).

No creds, no network, no broker. Covers the required EV-first guarantees:
  1. Ranking core puts higher EV/Risk ahead of lower.
  2. SKIP wins when every real structure is non-positive.
  3. Both CALL and PUT single legs are evaluated BEFORE any choice is made
     (the inverse of the live direction-first ordering).
  4. An injected chain is scored deterministically (no network).
  5. The output-row field contract is stable.
  6. Advisory + RL annotations are read-only and fail open.
  7. Formatters never raise; the analytics footer is always present.

strategy_ev_matrix is ANALYTICS ONLY: it never opens, closes, sizes, prices,
blocks or triggers a trade, never mutates a Q-table, and never reaches the
network. The chain is injected, so the whole module is deterministic.
"""

import unittest

import strategy_ev_matrix as sem
from strategy_ev_matrix import (
    SKIP, LONG_CALL, LONG_PUT, ANALYTICS_FOOTER,
    BULLISH, BEARISH, NEUTRAL, NONE_BIAS,
    rank_candidates, build_ev_matrix, format_ev_matrix,
    generate_strategy_ev_matrix_text,
)
import spread_builder as sb

# Required output fields for every candidate row (the Task-4 contract).
_REQUIRED_FIELDS = {
    "symbol", "strategy", "direction_bias", "expected_value",
    "ev_per_dollar_risk", "probability_of_profit", "max_profit", "max_loss",
    "liquidity_score", "oracle_score", "volatility_edge",
    "advisory_recommendation", "rl_state_q", "final_rank",
}


def _synthetic_chain(spot=100.0):
    return sem._synthetic_chain(spot)


class TestRankingCore(unittest.TestCase):
    def test_higher_ev_risk_ranks_first(self):
        rows = [
            {"strategy": "low", "expected_value": 5.0, "ev_per_dollar_risk": 0.05},
            {"strategy": "high", "expected_value": 10.0, "ev_per_dollar_risk": 0.30},
            {"strategy": "mid", "expected_value": 8.0, "ev_per_dollar_risk": 0.15},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual([r["strategy"] for r in ranked], ["high", "mid", "low"])
        # final_rank is dense 1..N.
        self.assertEqual([r["final_rank"] for r in ranked], [1, 2, 3])

    def test_ev_tiebreak_when_ev_risk_equal(self):
        rows = [
            {"strategy": "a", "expected_value": 5.0, "ev_per_dollar_risk": 0.10},
            {"strategy": "b", "expected_value": 9.0, "ev_per_dollar_risk": 0.10},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["strategy"], "b")

    def test_unrankable_rows_sort_last_but_kept(self):
        rows = [
            {"strategy": "ok", "expected_value": 1.0, "ev_per_dollar_risk": 0.02},
            {"strategy": "bad", "expected_value": None, "ev_per_dollar_risk": None},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["strategy"], "ok")
        self.assertEqual(ranked[-1]["strategy"], "bad")
        self.assertEqual(len(ranked), 2)  # nothing dropped

    def test_skip_wins_when_all_negative(self):
        rows = [
            {"strategy": "a", "expected_value": -5.0, "ev_per_dollar_risk": -0.10},
            {"strategy": "b", "expected_value": -2.0, "ev_per_dollar_risk": -0.04},
            {"strategy": SKIP, "expected_value": 0.0, "ev_per_dollar_risk": 0.0},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["strategy"], SKIP)

    def test_positive_structure_beats_skip(self):
        rows = [
            {"strategy": "a", "expected_value": 4.0, "ev_per_dollar_risk": 0.08},
            {"strategy": SKIP, "expected_value": 0.0, "ev_per_dollar_risk": 0.0},
        ]
        ranked = rank_candidates(rows)
        self.assertEqual(ranked[0]["strategy"], "a")
        self.assertEqual(ranked[-1]["strategy"], SKIP)


class TestBuildMatrix(unittest.TestCase):
    def setUp(self):
        self.report = build_ev_matrix(
            "SPY", 100.0, 0.25, _synthetic_chain(100.0),
            days=30, volatility_edge=0.01, advisory=False)
        self.cands = self.report["candidates"]
        self.by_strat = {r["strategy"]: r for r in self.cands}

    def test_both_call_and_put_evaluated(self):
        # EV-first: a CALL AND a PUT structure must both be present before any
        # choice — the inverse of the live direction-first ordering.
        self.assertIn(LONG_CALL, self.by_strat)
        self.assertIn(LONG_PUT, self.by_strat)

    def test_skip_anchor_present(self):
        self.assertIn(SKIP, self.by_strat)
        skip = self.by_strat[SKIP]
        self.assertEqual(skip["expected_value"], 0.0)
        self.assertEqual(skip["ev_per_dollar_risk"], 0.0)
        self.assertEqual(skip["direction_bias"], NONE_BIAS)

    def test_all_eight_candidates_built(self):
        expected = {
            LONG_CALL, LONG_PUT, sb.DEBIT_CALL_SPREAD, sb.DEBIT_PUT_SPREAD,
            sb.BULLISH_PUT_CREDIT_SPREAD, sb.BEARISH_CALL_CREDIT_SPREAD,
            sb.IRON_CONDOR, SKIP,
        }
        self.assertEqual(set(self.by_strat), expected)

    def test_direction_bias_mapping(self):
        self.assertEqual(self.by_strat[LONG_CALL]["direction_bias"], BULLISH)
        self.assertEqual(self.by_strat[LONG_PUT]["direction_bias"], BEARISH)
        self.assertEqual(self.by_strat[sb.IRON_CONDOR]["direction_bias"], NEUTRAL)

    def test_ranks_dense_and_unique(self):
        ranks = sorted(r["final_rank"] for r in self.cands)
        self.assertEqual(ranks, list(range(1, len(self.cands) + 1)))

    def test_output_field_contract(self):
        for row in self.cands:
            self.assertTrue(_REQUIRED_FIELDS.issubset(set(row)),
                            f"missing fields in {row.get('strategy')}: "
                            f"{_REQUIRED_FIELDS - set(row)}")

    def test_best_and_recommend_skip_keys(self):
        self.assertIn("best", self.report)
        self.assertIn("best_strategy", self.report)
        self.assertIn("recommend_skip", self.report)
        self.assertEqual(self.report["best"], self.cands[0])


class TestFailOpen(unittest.TestCase):
    def test_missing_spot_vol_fails_open_to_skip(self):
        rep = build_ev_matrix("X", None, None, None)
        self.assertTrue(rep["candidates"])
        self.assertEqual(rep["best_strategy"], SKIP)
        self.assertTrue(rep["recommend_skip"])

    def test_empty_chain_still_returns_skip(self):
        rep = build_ev_matrix("X", 100.0, 0.25, [])
        self.assertIn(SKIP, {r["strategy"] for r in rep["candidates"]})

    def test_garbage_chain_does_not_raise(self):
        rep = build_ev_matrix("X", 100.0, 0.25,
                              [{"junk": 1}, None, "nonsense", 42])
        self.assertTrue(rep["candidates"])


class TestAnnotations(unittest.TestCase):
    def test_rl_annotation_read_only_and_fail_open(self):
        calls = {"n": 0}

        class _Agent:
            def get_q(self, state_key, action):
                calls["n"] += 1
                return 0.42  # constant, read-only

        rep = build_ev_matrix(
            "SPY", 100.0, 0.25, _synthetic_chain(100.0),
            advisory=False, rl_agent=_Agent(), rl_state_key="s|t")
        # Every row got annotated with the read-only Q value.
        self.assertTrue(calls["n"] >= len(rep["candidates"]))
        for row in rep["candidates"]:
            self.assertEqual(row["rl_state_q"], 0.42)

    def test_rl_annotation_swallows_agent_errors(self):
        class _Boom:
            def get_q(self, *a, **k):
                raise RuntimeError("boom")

        rep = build_ev_matrix(
            "SPY", 100.0, 0.25, _synthetic_chain(100.0),
            advisory=False, rl_agent=_Boom(), rl_state_key="s|t")
        for row in rep["candidates"]:
            self.assertIsNone(row["rl_state_q"])

    def test_advisory_fn_invoked_when_supplied(self):
        seen = {"n": 0}

        def _adv(**kwargs):
            seen["n"] += 1
            return {"recommendation": "ACCEPT"}

        rep = build_ev_matrix(
            "SPY", 100.0, 0.25, _synthetic_chain(100.0),
            advisory=True, advisory_fn=_adv)
        # SKIP rows are not sent to advisory; some real OK rows should be.
        recs = {r["advisory_recommendation"] for r in rep["candidates"]}
        # If any structure scored OK, advisory ran at least once.
        if seen["n"]:
            self.assertIn("ACCEPT", recs)


class TestFormatting(unittest.TestCase):
    def test_format_never_raises_and_has_footer(self):
        rep = build_ev_matrix("SPY", 100.0, 0.25, _synthetic_chain(100.0),
                              advisory=False)
        text = format_ev_matrix(rep)
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn("Strategy EV Matrix", text)

    def test_empty_candidates_message(self):
        text = format_ev_matrix({"symbol": "SPY", "candidates": []})
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_generate_text_no_provider_is_clean(self):
        text = generate_strategy_ev_matrix_text("SPY")  # no chain provider
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn("SPY", text)

    def test_generate_text_with_injected_provider(self):
        def _provider(symbol):
            return {"spot": 100.0, "sigma": 0.25,
                    "chain": _synthetic_chain(100.0), "days": 30}

        text = generate_strategy_ev_matrix_text("SPY", chain_provider=_provider,
                                                advisory=False)
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn("EV-first pick", text)

    def test_generate_text_provider_error_fails_open(self):
        def _boom(symbol):
            raise RuntimeError("network down")

        text = generate_strategy_ev_matrix_text("SPY", chain_provider=_boom)
        self.assertIn(ANALYTICS_FOOTER, text)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(sem._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
