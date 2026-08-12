"""
Oracle 3.0 — Phase-2 Upgrade H acceptance tests: the promotion audit ledger.

Upgrade H persists every ``run_promotion_check`` verdict to an append-only JSONL
history so the human Stage-4 promotion decision is TRACKED and reproducible over
time. ``oracle.promotion_audit`` is analytics-only: it appends to / reads a local
file and NEVER edits ``.env``, flips a flag, or touches the live path.

Invariants under test:
  1. Empty / missing ledger -> [] history and an empty summary (fail-open).
  2. record_promotion_check appends exactly ONE row per run, normalising each
     layer decision (promote/reasons/metrics) and stamping a summary + provenance.
  3. Junk decisions (None / {} / non-dict verdicts) -> None, no row written.
  4. History is returned newest-first and is filterable by layer, and by
     layer+promote.
  5. summarize_history folds to a per-layer latest-verdict view with a stable
     ``first_promoted_at`` (earliest promoting run), tolerant of junk rows.
  6. Determinism: identical rows -> identical summary.
  7. Runner wiring: a full ``main`` run appends exactly one row; ``--no-audit``
     suppresses the write; ``--daily-summary`` reads the history back and exits 0.

No creds, no network. All file writes go to a tmp path only.
"""

import os
import sys
import tempfile
import unittest
from datetime import datetime, timezone

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.promotion_audit import (
    format_daily_summary,
    load_promotion_history,
    record_promotion_check,
    summarize_history,
)
import run_promotion_check as rpc


def _dec(promote, reasons=(), metrics=None):
    return {"promote": promote, "reasons": list(reasons),
            "metrics": dict(metrics or {"n_evals": 40})}


def _ts(day):
    return datetime(2024, 1, day, tzinfo=timezone.utc)


class TestEmptyLedgerFailOpen(unittest.TestCase):
    def test_missing_file_is_empty(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self.assertEqual(load_promotion_history({}, path=path), [])
            self.assertEqual(summarize_history([])["n_runs"], 0)
            self.assertIn("no checks recorded", format_daily_summary([]))


class TestRecordAppendsOneRow(unittest.TestCase):
    def test_one_run_one_row_with_summary_and_provenance(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            rec = record_promotion_check(
                {"executable_ev": _dec(True, []),
                 "fill_model": _dec(False, ["insufficient_fills"])},
                thresholds={"min_evals": 30.0},
                sources={"lab_json": "study.json"},
                path=path, now=_ts(1))
            self.assertIsNotNone(rec)
            self.assertEqual(rec["n_layers"], 2)
            self.assertEqual(rec["n_promoted"], 1)
            self.assertEqual(rec["promoted"], ["executable_ev"])
            self.assertEqual(rec["thresholds"], {"min_evals": 30.0})
            self.assertEqual(rec["sources"], {"lab_json": "study.json"})
            # exactly one line in the ledger
            with open(path, "r", encoding="utf-8") as fh:
                self.assertEqual(sum(1 for ln in fh if ln.strip()), 1)


class TestJunkDecisionsFailOpen(unittest.TestCase):
    def test_none_empty_and_nondict_write_nothing(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self.assertIsNone(record_promotion_check(None, path=path))
            self.assertIsNone(record_promotion_check({}, path=path))
            self.assertIsNone(record_promotion_check({"x": 42}, path=path))
            self.assertFalse(os.path.exists(path))

    def test_history_never_raises_on_junk_filters(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            record_promotion_check({"executable_ev": _dec(False)}, path=path,
                                   now=_ts(1))
            for junk in (None, 42, "x", []):
                load_promotion_history(junk, path=path)  # must not raise


class TestHistoryOrderingAndFilters(unittest.TestCase):
    def _seed(self, path):
        record_promotion_check(
            {"executable_ev": _dec(False, ["insufficient_evals"]),
             "fill_model": _dec(False, ["insufficient_fills"])},
            path=path, now=_ts(1))
        record_promotion_check(
            {"executable_ev": _dec(True, []),
             "fill_model": _dec(False, ["fill_rate_bias_too_high"])},
            path=path, now=_ts(2))

    def test_newest_first(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._seed(path)
            hist = load_promotion_history({}, path=path)
            self.assertEqual(len(hist), 2)
            self.assertEqual(hist[0]["recorded_at"], _ts(2).isoformat())

    def test_layer_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._seed(path)
            self.assertEqual(
                len(load_promotion_history({"layer": "executable_ev"}, path=path)),
                2)
            self.assertEqual(
                load_promotion_history({"layer": "nope"}, path=path), [])

    def test_layer_plus_promote_filter(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._seed(path)
            hits = load_promotion_history(
                {"layer": "executable_ev", "promote": True}, path=path)
            self.assertEqual(len(hits), 1)
            self.assertEqual(hits[0]["recorded_at"], _ts(2).isoformat())
            self.assertEqual(
                load_promotion_history(
                    {"layer": "fill_model", "promote": True}, path=path), [])

    def test_limit(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            self._seed(path)
            self.assertEqual(len(load_promotion_history({}, path=path, limit=1)), 1)


class TestSummary(unittest.TestCase):
    def test_first_promoted_at_is_earliest_promoting_run(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "audit.jsonl")
            record_promotion_check({"executable_ev": _dec(False, ["x"])},
                                   path=path, now=_ts(1))
            record_promotion_check({"executable_ev": _dec(True, [])},
                                   path=path, now=_ts(2))
            record_promotion_check({"executable_ev": _dec(True, [])},
                                   path=path, now=_ts(3))
            summ = summarize_history(load_promotion_history({}, path=path))
            ev = summ["layers"]["executable_ev"]
            self.assertEqual(summ["n_runs"], 3)
            self.assertTrue(ev["latest_promote"])
            self.assertEqual(ev["runs"], 3)
            self.assertEqual(ev["first_promoted_at"], _ts(2).isoformat())

    def test_hold_layer_has_no_first_promoted(self):
        rows = [
            {"type": "promotion_check", "recorded_at": _ts(1).isoformat(),
             "layers": {"fill_model": _dec(False, ["insufficient_fills"])}},
        ]
        fm = summarize_history(rows)["layers"]["fill_model"]
        self.assertFalse(fm["latest_promote"])
        self.assertIsNone(fm["first_promoted_at"])

    def test_tolerates_junk_rows(self):
        self.assertEqual(
            summarize_history([{"bad": 1}, 7, None])["n_runs"], 0)

    def test_daily_summary_text_renders_verdicts(self):
        rows = [
            {"type": "promotion_check", "recorded_at": _ts(2).isoformat(),
             "layers": {"executable_ev": _dec(True, []),
                        "fill_model": _dec(False, ["insufficient_fills"])}},
        ]
        txt = format_daily_summary(rows)
        self.assertIn("PROMOTE", txt)
        self.assertIn("HOLD", txt)
        self.assertIn("ADVISORY ONLY", txt)


class TestDeterminism(unittest.TestCase):
    def test_identical_rows_identical_summary(self):
        rows = [
            {"type": "promotion_check", "recorded_at": _ts(1).isoformat(),
             "layers": {"executable_ev": _dec(True, [])}},
            {"type": "promotion_check", "recorded_at": _ts(2).isoformat(),
             "layers": {"executable_ev": _dec(False, ["insufficient_evals"])}},
        ]
        self.assertEqual(summarize_history(rows), summarize_history(list(rows)))


# --------------------------------------------------------------------------- #
# Runner wiring (offline; tmp files only)
# --------------------------------------------------------------------------- #
class TestRunnerAuditWiring(unittest.TestCase):
    def test_full_run_appends_one_row(self):
        with tempfile.TemporaryDirectory() as d:
            audit = os.path.join(d, "audit.jsonl")
            rc = rpc.main(["--layer", "executable_ev", "--audit-jsonl", audit])
            self.assertEqual(rc, 0)
            rows = load_promotion_history(path=audit)
            self.assertEqual(len(rows), 1)
            # No evidence -> HOLD is recorded, not PROMOTE.
            self.assertFalse(
                rows[0]["layers"]["executable_ev"]["promote"])

    def test_no_audit_suppresses_write(self):
        with tempfile.TemporaryDirectory() as d:
            audit = os.path.join(d, "audit.jsonl")
            rc = rpc.main(["--layer", "executable_ev", "--no-audit",
                           "--audit-jsonl", audit])
            self.assertEqual(rc, 0)
            self.assertFalse(os.path.exists(audit))

    def test_daily_summary_reads_back_and_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            audit = os.path.join(d, "audit.jsonl")
            rpc.main(["--layer", "executable_ev", "--audit-jsonl", audit])
            rc = rpc.main(["--daily-summary", "--audit-jsonl", audit])
            self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main()
