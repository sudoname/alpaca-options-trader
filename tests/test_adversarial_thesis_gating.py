"""
Oracle 3.0 — Upgrade E acceptance tests: adversarial-thesis gating.

Upgrade E promotes the bounded skeptic in ``oracle.thesis_debate`` from shadow
to gate-feeding. When ``ADVERSARIAL_THESIS_GATING`` is ON (and
``ENABLE_ADVERSARIAL_THESIS`` is ON) the review's ``adjusted_probability``
replaces the raw Oracle-head probability fed to
``oracle_trade_gate.evaluate_oracle_gate`` — so a raised ``p_no_trade`` can turn
an ALLOW into a veto. It stays strictly veto-only: the review only ever RAISES
``p_no_trade`` (capped by ``THESIS_MAX_NO_TRADE_BOOST``) and never flips the
leading side, and the gate never sizes or invents a trade.

These pin the promotion invariants of the live wire in ``smart_trader.py``
WITHOUT instantiating the trader (no creds / no network). The wire is a pure
composition of three functions, all exercised directly here through the
``_feed_gate`` helper that mirrors the wire exactly:

    build_theses(ctx, tally, prob)              # no evidence threaded (== shadow)
      -> adversarial_review(theses, prob, tally=..., max_no_trade_boost=...)
      -> evaluate_oracle_gate({'probability': <raw|adjusted>, ...}, config)

Invariants:
  1. Flag OFF is byte-identical: the prob fed to the gate is the raw head and the
     verdict equals ``evaluate_oracle_gate`` on the raw head.
  2. Flag ON feeds the adjusted prob: a raised ``p_no_trade`` can flip ALLOW->BLOCK.
  3. Boost is capped at ``THESIS_MAX_NO_TRADE_BOOST`` (piling on flags saturates).
  4. Direction is preserved (leading side never flips) and the fed prob sums to 1.
  5. An already-vetoing head (overwhelming no-trade mass) stays a veto; gating can
     only add doubt, never rescue a trade.
  6. Determinism: identical inputs -> identical fed prob.
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.thesis_debate import (
    THESIS_MAX_NO_TRADE_BOOST_DEFAULT,
    adversarial_review,
    build_theses,
)
from oracle_trade_gate import evaluate_oracle_gate


def _feed_gate(prob, tally, ctx, *, intended_side, gating_on,
               enable_thesis=True, max_boost=THESIS_MAX_NO_TRADE_BOOST_DEFAULT,
               gate_config=None):
    """Mirror the smart_trader.py Upgrade-E wire EXACTLY.

    Returns ``(fed_prob, review_or_None, verdict)``. When gating is OFF the raw
    ``prob`` is fed straight through (byte-identical to pre-Upgrade-E).
    """
    fed = prob
    review = None
    if gating_on and enable_thesis:
        theses = build_theses(ctx, tally, prob)
        review = adversarial_review(theses, prob, tally=tally,
                                    max_no_trade_boost=max_boost)
        adj = review.adjusted_probability or {}
        if adj:
            fed = adj
    verdict = evaluate_oracle_gate(
        {"probability": fed, "intended_side": intended_side}, gate_config)
    return fed, review, verdict


# A +call head that comfortably passes the default gate, but where the vote
# tally leans bear -> the skeptic will flag a direction/tally mismatch.
_PROB_MISMATCH = {"p_call": 0.50, "p_put": 0.15, "p_no_trade": 0.35}
_TALLY_BEAR = {"p_bull": 0.20, "p_bear": 0.70, "p_neutral": 0.10}
_CTX = {"regime": "uptrend"}
# Tighten the no-trade ceiling so the raw head passes but the boosted one vetoes.
_TIGHT_CEILING = {"ORACLE_MAX_NO_TRADE": 0.40}


class TestOffIsByteIdentical(unittest.TestCase):
    def test_gating_off_feeds_raw_and_matches_baseline(self):
        fed, review, verdict = _feed_gate(
            _PROB_MISMATCH, _TALLY_BEAR, _CTX, intended_side="call",
            gating_on=False, gate_config=_TIGHT_CEILING)
        # Nothing computed, raw prob fed unchanged.
        self.assertIsNone(review)
        self.assertEqual(fed, _PROB_MISMATCH)
        # Verdict is exactly the legacy verdict on the raw head.
        baseline = evaluate_oracle_gate(
            {"probability": _PROB_MISMATCH, "intended_side": "call"},
            _TIGHT_CEILING)
        self.assertEqual(verdict, baseline)
        self.assertTrue(verdict["allow"])   # raw 0.35 < 0.40 ceiling -> allow

    def test_enable_thesis_off_also_byte_identical(self):
        # Gating flag ON but the thesis layer itself OFF -> still shadow/raw.
        fed, review, _ = _feed_gate(
            _PROB_MISMATCH, _TALLY_BEAR, _CTX, intended_side="call",
            gating_on=True, enable_thesis=False, gate_config=_TIGHT_CEILING)
        self.assertIsNone(review)
        self.assertEqual(fed, _PROB_MISMATCH)


class TestOnFeedsAdjusted(unittest.TestCase):
    def test_gating_flips_allow_to_block(self):
        raw = evaluate_oracle_gate(
            {"probability": _PROB_MISMATCH, "intended_side": "call"},
            _TIGHT_CEILING)
        self.assertTrue(raw["allow"])   # raw head is allowed

        fed, review, verdict = _feed_gate(
            _PROB_MISMATCH, _TALLY_BEAR, _CTX, intended_side="call",
            gating_on=True, gate_config=_TIGHT_CEILING)
        # The skeptic raised p_no_trade above the ceiling -> veto.
        self.assertIn("direction_tally_mismatch", review.flags)
        self.assertGreater(fed["p_no_trade"], _PROB_MISMATCH["p_no_trade"])
        self.assertFalse(verdict["allow"])
        self.assertIn("no-trade", verdict["reason"])


class TestBoostCapped(unittest.TestCase):
    def test_piling_flags_saturates_at_cap(self):
        # Downtrend + call lead (regime_conflict) + bear tally (mismatch) +
        # empty support (thin_support): raw penalties 0.04+0.05+0.03 = 0.12 > cap.
        ctx_counter = {"regime": "downtrend"}
        _, review, _ = _feed_gate(
            _PROB_MISMATCH, _TALLY_BEAR, ctx_counter, intended_side="call",
            gating_on=True, max_boost=0.10)
        self.assertGreaterEqual(len(review.flags), 2)
        self.assertLessEqual(review.no_trade_boost, 0.10 + 1e-9)
        self.assertAlmostEqual(review.no_trade_boost, 0.10, places=9)


class TestDirectionPreservedAndNormalized(unittest.TestCase):
    def test_leading_side_never_flips_and_sums_to_one(self):
        fed, review, _ = _feed_gate(
            _PROB_MISMATCH, _TALLY_BEAR, _CTX, intended_side="call",
            gating_on=True, gate_config=_TIGHT_CEILING)
        # Call led before; it must still lead after (doubt-only adjustment).
        self.assertGreater(_PROB_MISMATCH["p_call"], _PROB_MISMATCH["p_put"])
        self.assertGreater(fed["p_call"], fed["p_put"])
        self.assertAlmostEqual(
            fed["p_call"] + fed["p_put"] + fed["p_no_trade"], 1.0, places=6)


class TestVetoOnly(unittest.TestCase):
    def test_overwhelming_no_trade_stays_blocked(self):
        # A head that already vetoes on no-trade mass must stay vetoed; gating
        # only ADDS doubt, so it can never rescue the trade.
        sit_out = {"p_call": 0.05, "p_put": 0.05, "p_no_trade": 0.90}
        raw = evaluate_oracle_gate(
            {"probability": sit_out, "intended_side": "call"})
        self.assertFalse(raw["allow"])
        fed, _, verdict = _feed_gate(
            sit_out, _TALLY_BEAR, _CTX, intended_side="call", gating_on=True)
        self.assertFalse(verdict["allow"])
        self.assertGreaterEqual(fed["p_no_trade"], sit_out["p_no_trade"])

    def test_gating_never_turns_block_into_allow(self):
        # Sweep: for any head, the gated verdict is never MORE permissive than
        # the raw verdict (allow can become block, never the reverse).
        heads = [
            {"p_call": 0.50, "p_put": 0.15, "p_no_trade": 0.35},
            {"p_call": 0.60, "p_put": 0.20, "p_no_trade": 0.20},
            {"p_call": 0.15, "p_put": 0.65, "p_no_trade": 0.20},
            {"p_call": 0.10, "p_put": 0.08, "p_no_trade": 0.82},
        ]
        for h in heads:
            raw = evaluate_oracle_gate(
                {"probability": h, "intended_side": "call"}, _TIGHT_CEILING)
            _, _, gated = _feed_gate(
                h, _TALLY_BEAR, _CTX, intended_side="call",
                gating_on=True, gate_config=_TIGHT_CEILING)
            if not raw["allow"]:
                self.assertFalse(gated["allow"],
                                 f"gating rescued a blocked head: {h}")


class TestCleanHeadNotSpuriouslyBlocked(unittest.TestCase):
    def test_strong_aligned_head_still_allows_under_gating(self):
        # A strong, coherent call head: the small universal doubt tax never
        # crosses the default 0.85 ceiling, so gating leaves it tradable.
        strong = {"p_call": 0.70, "p_put": 0.10, "p_no_trade": 0.20}
        tally_bull = {"p_bull": 0.70, "p_bear": 0.15, "p_neutral": 0.15}
        fed, review, verdict = _feed_gate(
            strong, tally_bull, _CTX, intended_side="call", gating_on=True)
        self.assertNotIn("direction_tally_mismatch", review.flags)
        self.assertTrue(verdict["allow"])
        self.assertLess(fed["p_no_trade"], 0.85)


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_identical_fed_prob(self):
        a = _feed_gate(_PROB_MISMATCH, _TALLY_BEAR, _CTX,
                       intended_side="call", gating_on=True,
                       gate_config=_TIGHT_CEILING)
        b = _feed_gate(_PROB_MISMATCH, _TALLY_BEAR, _CTX,
                       intended_side="call", gating_on=True,
                       gate_config=_TIGHT_CEILING)
        self.assertEqual(a[0], b[0])
        self.assertEqual(a[2], b[2])


if __name__ == "__main__":
    unittest.main()
