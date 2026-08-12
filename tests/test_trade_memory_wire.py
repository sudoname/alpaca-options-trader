"""
Oracle 3.0 — Upgrade F acceptance tests: semantic trade-memory wire.

Upgrade F turns on the postmortem loop and feeds it back as CONTEXT ONLY:

  RECORD side (smart_trader.record_trade_outcome, gated by
  ``ENABLE_SEMANTIC_TRADE_MEMORY``): on close, append one reflection row to the
  append-only ``oracle.trade_memory`` ledger.

  RETRIEVE side (smart_trader thesis path, additionally gated by the same flag):
  after ``adversarial_review`` runs, pull newest-first lessons for the current
  {symbol, sector, regime, catalyst, strategy_mode} and attach them to the
  review ``notes`` as natural-language context. It NEVER touches the numbers —
  ``adjusted_probability`` is identical whether or not lessons are attached, so
  the gate/sizing/risk are unaffected.

These pin the wire WITHOUT instantiating the trader (no creds / no network).
The retrieval wire is a pure composition mirrored exactly by ``_thesis_notes``:

    build_theses(ctx, tally, prob)
      -> adversarial_review(theses, prob, tally=...)
      -> retrieve_lessons({symbol, sector, regime, ...})  # flag ON only
      -> review.notes += 'lessons: ...'                   # context-only

Invariants:
  1. Reflection is persisted append-only and read back newest-first.
  2. Retrieval filters correctly (symbol / regime / compound no-match).
  3. Lessons NEVER alter the numbers: ``adjusted_probability`` (and every
     flag / boost) is byte-identical with memory ON vs OFF — only ``notes`` grow.
  4. Flag OFF is byte-identical: no ledger read, notes unchanged, no attachment.
  5. Empty / non-matching ledger -> nothing attached, review unchanged.
  6. Determinism: identical inputs + ledger -> identical attached notes.

No creds, no network. File writes go to a tmp path only.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.thesis_debate import adversarial_review, build_theses
from oracle.trade_memory import record_reflection, retrieve_lessons


# A +call head that leans against a bear tally -> the skeptic raises p_no_trade.
_PROB = {"p_call": 0.50, "p_put": 0.15, "p_no_trade": 0.35}
_TALLY = {"p_bull": 0.20, "p_bear": 0.70, "p_neutral": 0.10}
_CTX = {"regime": "uptrend", "sector": "tech",
        "catalyst": "earnings", "strategy_mode": "intraday"}
_SYMBOL = "AAPL"


def _thesis_notes(prob, tally, ctx, symbol, *, memory_on, path):
    """Mirror the smart_trader.py Upgrade-F retrieval wire EXACTLY.

    Returns ``(review, lessons)``. When ``memory_on`` is False the ledger is
    never read and the review is byte-identical to the pre-Upgrade-F shadow
    block (notes untouched, no attachment).
    """
    theses = build_theses(ctx, tally, prob)
    review = adversarial_review(theses, prob, tally=tally)
    lessons = []
    if memory_on:
        filters = {
            "symbol": symbol,
            "sector": (ctx or {}).get("sector"),
            "regime": (ctx or {}).get("regime"),
            "catalyst": (ctx or {}).get("catalyst"),
            "strategy_mode": (ctx or {}).get("strategy_mode"),
        }
        filters = {k: v for k, v in filters.items() if v}
        lessons = retrieve_lessons(filters, path=path, limit=3)
        if lessons:
            notes = [str(l.get("lesson")) for l in lessons if l.get("lesson")]
            if notes:
                review.notes = (
                    (review.notes + " | " if review.notes else "")
                    + "lessons: " + " | ".join(notes))
    return review, lessons


def _seed(path):
    """Two AAPL/tech/uptrend lessons (newest last) + one off-target row."""
    from datetime import datetime, timezone
    record_reflection(
        {"symbol": "AAPL", "sector": "tech", "regime": "uptrend",
         "catalyst": "earnings", "strategy_mode": "intraday",
         "failure_mode": "chased_extension", "lesson": "wait for the reclaim"},
        path=path, now=datetime(2024, 1, 1, tzinfo=timezone.utc))
    record_reflection(
        {"symbol": "AAPL", "sector": "tech", "regime": "uptrend",
         "catalyst": "earnings", "strategy_mode": "intraday",
         "failure_mode": "fought_tally", "lesson": "respect the bear tally"},
        path=path, now=datetime(2024, 1, 2, tzinfo=timezone.utc))
    # Off-target: different symbol/regime, must not match the AAPL filter.
    record_reflection(
        {"symbol": "MSFT", "sector": "tech", "regime": "downtrend",
         "lesson": "unrelated"},
        path=path, now=datetime(2024, 1, 3, tzinfo=timezone.utc))


class TestReflectionPersistedAppendOnly(unittest.TestCase):
    def test_records_append_and_read_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            # Append-only: three physical lines.
            with open(path, encoding="utf-8") as fh:
                self.assertEqual(sum(1 for ln in fh if ln.strip()), 3)
            rows = retrieve_lessons({"symbol": "AAPL"}, path=path)
            self.assertEqual([r["lesson"] for r in rows],
                             ["respect the bear tally", "wait for the reclaim"])


class TestRetrievalFilters(unittest.TestCase):
    def test_symbol_and_regime_and_compound_nomatch(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            self.assertEqual(len(retrieve_lessons({"symbol": "AAPL"}, path=path)), 2)
            self.assertEqual(len(retrieve_lessons({"regime": "downtrend"}, path=path)), 1)
            # Compound with no matching row.
            self.assertEqual(
                retrieve_lessons({"symbol": "AAPL", "regime": "downtrend"},
                                 path=path), [])


class TestLessonsNeverAlterNumbers(unittest.TestCase):
    def test_adjusted_probability_identical_on_vs_off(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            off, off_lessons = _thesis_notes(
                _PROB, _TALLY, _CTX, _SYMBOL, memory_on=False, path=path)
            on, on_lessons = _thesis_notes(
                _PROB, _TALLY, _CTX, _SYMBOL, memory_on=True, path=path)

            # Memory OFF touches nothing.
            self.assertEqual(off_lessons, [])
            # Memory ON attached the two AAPL lessons.
            self.assertEqual(len(on_lessons), 2)

            # The NUMBERS are byte-identical — the whole point of Upgrade F.
            self.assertEqual(on.adjusted_probability, off.adjusted_probability)
            self.assertEqual(on.flags, off.flags)
            self.assertEqual(on.no_trade_boost, off.no_trade_boost)
            self.assertEqual(on.invalidation, off.invalidation)

            # Only the human-readable notes grew, carrying the newest lesson first.
            self.assertNotEqual(on.notes, off.notes)
            self.assertIn("lessons:", on.notes)
            self.assertIn("respect the bear tally", on.notes)
            self.assertLess(on.notes.index("respect the bear tally"),
                            on.notes.index("wait for the reclaim"))
            self.assertNotIn("lessons:", off.notes)


class TestFlagOffByteIdentical(unittest.TestCase):
    def test_off_never_reads_ledger_and_leaves_review_untouched(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            # Baseline review with no memory layer at all.
            base_theses = build_theses(_CTX, _TALLY, _PROB)
            base = adversarial_review(base_theses, _PROB, tally=_TALLY)
            off, off_lessons = _thesis_notes(
                _PROB, _TALLY, _CTX, _SYMBOL, memory_on=False, path=path)
            self.assertEqual(off_lessons, [])
            self.assertEqual(off.to_dict(), base.to_dict())


class TestEmptyLedgerNoAttachment(unittest.TestCase):
    def test_missing_and_nonmatching_ledger_attach_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            missing = os.path.join(d, "nope.jsonl")
            base = adversarial_review(
                build_theses(_CTX, _TALLY, _PROB), _PROB, tally=_TALLY)
            rev, lessons = _thesis_notes(
                _PROB, _TALLY, _CTX, _SYMBOL, memory_on=True, path=missing)
            self.assertEqual(lessons, [])
            self.assertEqual(rev.notes, base.notes)

            # Ledger exists but nothing matches this symbol.
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            rev2, lessons2 = _thesis_notes(
                _PROB, _TALLY, _CTX, "TSLA", memory_on=True, path=path)
            self.assertEqual(lessons2, [])
            self.assertEqual(rev2.notes, base.notes)


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_and_ledger_identical_notes(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "mem.jsonl")
            _seed(path)
            a, _ = _thesis_notes(_PROB, _TALLY, _CTX, _SYMBOL,
                                 memory_on=True, path=path)
            b, _ = _thesis_notes(_PROB, _TALLY, _CTX, _SYMBOL,
                                 memory_on=True, path=path)
            self.assertEqual(a.notes, b.notes)
            self.assertEqual(a.adjusted_probability, b.adjusted_probability)


if __name__ == "__main__":
    unittest.main()
