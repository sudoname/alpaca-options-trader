"""
Upgrade 5 — Bull / Bear / No-Trade adversarial thesis tests.

These pin the ``oracle.thesis_debate`` + ``oracle.trade_memory`` contracts:

  1. All three theses execute over the SAME evidence set; confidences are seeded
     verbatim from the tally (p_bull/p_bear/p_neutral); direction is NEVER
     computed here.
  2. The adversarial review is a BOUNDED skeptic: it may only raise p_no_trade
     (within THESIS_MAX_NO_TRADE_BOOST), never flip the leading direction, and
     the result always renormalizes to sum 1.0.
  3. A quant-vs-tally direction mismatch is flagged and pushes mass to no-trade.
  4. An LLM provider may only add natural-language text/flags — it can never
     move the numbers.
  5. trade_memory records a reflection and retrieves it back by filter.

No creds, no network, no order placement.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.thesis_debate import (
    DIR_CALL,
    DIR_PUT,
    THESIS_MAX_NO_TRADE_BOOST_DEFAULT,
    ReviewResult,
    adversarial_review,
    build_theses,
)
from oracle.trade_memory import record_reflection, retrieve_lessons

CTX = {
    "regime": "uptrend",
    "technical_state": "above_20ma",
    "vol_state": "normal",
    "attention_state": "elevated",
    "fundamental_state": "ok",
    "expected_move_pct": 4.0,
}
EVIDENCE = [
    {"name": "trend", "stance": "bull", "text": "price > 20/50 MA"},
    {"name": "news", "stance": "bull", "text": "upgrade", "catalyst": True},
    {"name": "breadth", "stance": "bear", "text": "weak breadth"},
    {"name": "liquidity", "stance": "neutral", "text": "tight spread"},
]


def _sums_to_one(d):
    return abs(sum(d[k] for k in ("p_call", "p_put", "p_no_trade")) - 1.0) < 2e-6


class TestBuildTheses(unittest.TestCase):
    def setUp(self):
        self.tally = {"p_bull": 0.6, "p_bear": 0.25, "p_neutral": 0.15}
        self.theses = build_theses(CTX, self.tally, None, EVIDENCE)

    def test_three_theses_present(self):
        for key in ("bull", "bear", "no_trade"):
            self.assertIn(key, self.theses)

    def test_confidence_seeded_from_tally(self):
        self.assertAlmostEqual(self.theses["bull"]["confidence"], 0.6)
        self.assertAlmostEqual(self.theses["bear"]["confidence"], 0.25)
        self.assertAlmostEqual(self.theses["no_trade"]["confidence"], 0.15)

    def test_same_evidence_mirrored(self):
        # Bullish item supports bull, counters bear (and vice versa).
        self.assertIn("price > 20/50 MA", self.theses["bull"]["support"])
        self.assertIn("price > 20/50 MA", self.theses["bear"]["counter_evidence"])
        self.assertIn("weak breadth", self.theses["bear"]["support"])
        self.assertIn("weak breadth", self.theses["bull"]["counter_evidence"])

    def test_catalyst_surfaced(self):
        self.assertIn("upgrade", self.theses["bull"]["catalysts"])

    def test_regime_alignment(self):
        self.assertEqual(self.theses["bull"]["regime_alignment"], "aligned")
        self.assertEqual(self.theses["bear"]["regime_alignment"], "counter")

    def test_expected_move_from_dict(self):
        th = build_theses(CTX, self.tally, None, EVIDENCE,
                          expected_move={"sigma1_pct": 7.5})
        self.assertAlmostEqual(th["bull"]["expected_move"], 7.5)
        self.assertAlmostEqual(th["bull"]["invalidation"], 7.5)

    def test_never_invents_direction(self):
        self.assertEqual(self.theses["bull"]["direction"], DIR_CALL)
        self.assertEqual(self.theses["bear"]["direction"], DIR_PUT)
        self.assertIsNone(self.theses["no_trade"]["direction"])


class TestAdversarialReviewBounded(unittest.TestCase):
    def setUp(self):
        self.tally = {"p_bull": 0.6, "p_bear": 0.25, "p_neutral": 0.15}
        self.theses = build_theses(CTX, self.tally, None, EVIDENCE)
        self.prob = {"p_call": 0.6, "p_put": 0.2, "p_no_trade": 0.2}

    def test_clean_case_no_flags(self):
        r = adversarial_review(self.theses, self.prob, tally=self.tally,
                               evidence=EVIDENCE)
        self.assertIsInstance(r, ReviewResult)
        self.assertEqual(r.flags, [])
        self.assertTrue(_sums_to_one(r.adjusted_probability))
        self.assertGreater(r.adjusted_probability["p_call"],
                           r.adjusted_probability["p_put"])

    def test_review_never_flips_direction(self):
        tally_bear = {"p_bull": 0.2, "p_bear": 0.7, "p_neutral": 0.1}
        theses = build_theses(CTX, tally_bear, None, EVIDENCE)
        r = adversarial_review(theses, self.prob, tally=tally_bear,
                               evidence=EVIDENCE)
        self.assertIn("direction_tally_mismatch", r.flags)
        # No-trade rises, but call still leads (doubt-only adjustment).
        self.assertGreater(r.adjusted_probability["p_no_trade"],
                           self.prob["p_no_trade"])
        self.assertGreater(r.adjusted_probability["p_call"],
                           r.adjusted_probability["p_put"])
        self.assertTrue(_sums_to_one(r.adjusted_probability))

    def test_boost_capped(self):
        ctx_counter = dict(CTX, regime="downtrend")
        ev_stale = [{"name": "news", "stance": "bull", "text": "old upgrade",
                     "catalyst": True, "stale": True}]
        prob_ambig = {"p_call": 0.46, "p_put": 0.44, "p_no_trade": 0.10}
        tally_bear = {"p_bull": 0.2, "p_bear": 0.7, "p_neutral": 0.1}
        theses = build_theses(ctx_counter, tally_bear, None, ev_stale)
        r = adversarial_review(theses, prob_ambig, tally=tally_bear,
                               evidence=ev_stale, max_no_trade_boost=0.10)
        self.assertLessEqual(r.no_trade_boost,
                             THESIS_MAX_NO_TRADE_BOOST_DEFAULT + 1e-9)
        self.assertGreaterEqual(len(r.flags), 2)
        self.assertTrue(_sums_to_one(r.adjusted_probability))

    def test_stale_catalyst_flag_and_no_marker_leak(self):
        ev_stale = [{"name": "news", "stance": "bull", "text": "old upgrade",
                     "catalyst": True, "stale": True}]
        tally = {"p_bull": 0.6, "p_bear": 0.2, "p_neutral": 0.2}
        theses = build_theses(CTX, tally, None, ev_stale)
        r = adversarial_review(theses, {"p_call": 0.6, "p_put": 0.2,
                                        "p_no_trade": 0.2},
                               tally=tally, evidence=ev_stale)
        self.assertIn("stale_catalyst", r.flags)
        self.assertNotIn("_stale_support", theses.get("bull", {}))

    def test_deterministic(self):
        tally_bear = {"p_bull": 0.2, "p_bear": 0.7, "p_neutral": 0.1}
        theses = build_theses(CTX, tally_bear, None, EVIDENCE)
        a = adversarial_review(theses, self.prob, tally=tally_bear,
                               evidence=EVIDENCE).to_dict()
        b = adversarial_review(theses, self.prob, tally=tally_bear,
                               evidence=EVIDENCE).to_dict()
        self.assertEqual(a, b)

    def test_fail_open_on_junk(self):
        for junk in (None, 42, "x", [], {"weird": object()}):
            r = adversarial_review(junk, junk, junk)
            self.assertTrue(_sums_to_one(r.adjusted_probability))


class TestLLMTextOnly(unittest.TestCase):
    def test_llm_adds_text_not_numbers(self):
        tally = {"p_bull": 0.6, "p_bear": 0.25, "p_neutral": 0.15}
        prob = {"p_call": 0.6, "p_put": 0.2, "p_no_trade": 0.2}

        def fake_llm(_payload):
            return {"bull": {"support": ["llm: momentum intact"]},
                    "flags": ["llm_advisory_note"]}

        th = build_theses(CTX, tally, None, EVIDENCE, llm_provider=fake_llm)
        self.assertIn("llm: momentum intact", th["bull"]["support"])

        clean = adversarial_review(th, prob, tally=tally, evidence=EVIDENCE)
        with_llm = adversarial_review(th, prob, tally=tally, evidence=EVIDENCE,
                                      llm_provider=fake_llm)
        self.assertIn("llm_advisory_note", with_llm.flags)
        # Advisory flag carries zero penalty -> boost identical to clean.
        self.assertEqual(with_llm.no_trade_boost, clean.no_trade_boost)


class TestTradeMemory(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp(prefix="test_trade_memory_")
        self.path = os.path.join(self.tmp, "mem.jsonl")

    def tearDown(self):
        try:
            if os.path.exists(self.path):
                os.remove(self.path)
            os.rmdir(self.tmp)
        except Exception:
            pass

    def test_empty_ledger(self):
        self.assertEqual(retrieve_lessons({}, path=self.path), [])

    def test_record_and_retrieve(self):
        rec = record_reflection({
            "symbol": "AAPL", "sector": "tech", "regime": "uptrend",
            "catalyst": "earnings", "strategy_mode": "intraday",
            "failure_mode": "chased_extension", "lesson": "wait for reclaim",
            "confidence": 0.8,
        }, path=self.path, now=datetime(2024, 1, 1, tzinfo=timezone.utc))
        self.assertIsNotNone(rec)
        self.assertEqual(rec["confidence"], 0.8)

        got = retrieve_lessons({"symbol": "aapl"}, path=self.path)
        self.assertEqual(len(got), 1)
        self.assertEqual(got[0]["symbol"], "AAPL")

    def test_newest_first_and_limit(self):
        record_reflection({"symbol": "AAPL"}, path=self.path,
                          now=datetime(2024, 1, 1, tzinfo=timezone.utc))
        record_reflection({"symbol": "MSFT"}, path=self.path,
                          now=datetime(2024, 1, 2, tzinfo=timezone.utc))
        rows = retrieve_lessons({}, path=self.path)
        self.assertEqual(rows[0]["symbol"], "MSFT")
        self.assertEqual(len(retrieve_lessons({}, path=self.path, limit=1)), 1)

    def test_junk_record_returns_none(self):
        self.assertIsNone(record_reflection(None, path=self.path))
        self.assertIsNone(record_reflection(42, path=self.path))


def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print("tests.test_adversarial_thesis self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
