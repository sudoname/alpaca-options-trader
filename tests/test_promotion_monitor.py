"""
Oracle 3.0 — Phase-2 Upgrade I acceptance tests: the promotion regression monitor.

Upgrade I scans the append-only promotion-audit history (Upgrade H) and raises a
regression alert when a layer that previously cleared ALL gates is back to HOLD
(PROMOTION LOST) or a still-holding layer develops a NEW failing gate
(NEW FAILURE). ``oracle.promotion_monitor`` is pure analytics: it only READS
audit rows and NEVER edits ``.env``, flips a flag, or touches the live path.

Invariants under test:
  1. Empty / junk input -> a clean report, no regressions (fail-open, no raise).
  2. A steadily-promoting layer is NOT a regression.
  3. PROMOTION LOST fires when a promoted layer returns to HOLD, with the correct
     last_promoted_at / lost_at; a later re-PROMOTE clears it.
  4. NEW FAILURE fires only HOLD->HOLD when a new reason appears; the PROMOTE->
     HOLD transition is reported as LOST (not double-counted as a new failure).
  5. A never-promoted, non-worsening HOLD is not a regression.
  6. ``layers`` scopes both the summary and the regression list.
  7. Determinism: identical rows (any order) -> identical report.
  8. Runner wiring: --check-regressions exits 0; a recorded PROMOTE followed by a
     real HOLD run is detected via the runner.

No creds, no network. Runner-wiring file writes go to a tmp path only.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.promotion_monitor import (
    KIND_NEW_FAILURE,
    KIND_PROMOTION_LOST,
    detect_regressions,
    format_regression_alert,
)
import run_promotion_check as rpc


def _row(ts, layers):
    return {"type": "promotion_check", "recorded_at": ts, "layers": layers}


def _dec(promote, reasons=()):
    return {"promote": promote, "reasons": list(reasons), "metrics": {}}


def _kinds(report, kind):
    return [e for e in report["regressions"] if e["kind"] == kind]


class TestEmptyAndJunkFailOpen(unittest.TestCase):
    def test_empty(self):
        rep = detect_regressions([])
        self.assertEqual(rep["n_runs"], 0)
        self.assertEqual(rep["regressions"], [])
        self.assertIn("no regressions", format_regression_alert(rep))

    def test_junk_never_raises(self):
        for junk in (None, 42, "x", [{"bad": 1}, 7, None]):
            rep = detect_regressions(junk)
            self.assertEqual(rep["regressions"], [])

    def test_format_junk_report(self):
        self.assertIn("no regressions", format_regression_alert(None))
        self.assertIn("no regressions", format_regression_alert({}))


class TestNoRegressionOnSteadyPromote(unittest.TestCase):
    def test_steady(self):
        rep = detect_regressions([
            _row("2024-01-01", {"executable_ev": _dec(False, ["insufficient_evals"])}),
            _row("2024-01-02", {"executable_ev": _dec(True, [])}),
            _row("2024-01-03", {"executable_ev": _dec(True, [])}),
        ])
        self.assertEqual(rep["regressions"], [])
        ev = rep["layers"]["executable_ev"]
        self.assertTrue(ev["latest_promote"])
        self.assertEqual(ev["first_promoted_at"], "2024-01-02")


class TestPromotionLost(unittest.TestCase):
    def test_detects_lost_with_timestamps(self):
        rep = detect_regressions([
            _row("2024-01-01", {"fill_model": _dec(True, [])}),
            _row("2024-01-02", {"fill_model": _dec(True, [])}),
            _row("2024-01-03", {"fill_model": _dec(False, ["fill_rate_bias_too_high"])}),
        ])
        lost = _kinds(rep, KIND_PROMOTION_LOST)
        self.assertEqual(len(lost), 1)
        self.assertEqual(lost[0]["layer"], "fill_model")
        self.assertEqual(lost[0]["last_promoted_at"], "2024-01-02")
        self.assertEqual(lost[0]["lost_at"], "2024-01-03")
        self.assertEqual(lost[0]["latest_reasons"], ["fill_rate_bias_too_high"])
        self.assertIn("PROMOTION LOST", format_regression_alert(rep))

    def test_recovery_clears_regression(self):
        rep = detect_regressions([
            _row("2024-01-01", {"fill_model": _dec(True, [])}),
            _row("2024-01-02", {"fill_model": _dec(False, ["x"])}),
            _row("2024-01-03", {"fill_model": _dec(True, [])}),
        ])
        self.assertEqual(rep["regressions"], [])


class TestNewFailure(unittest.TestCase):
    def test_new_reason_hold_to_hold(self):
        rep = detect_regressions([
            _row("2024-01-01", {"adversarial_thesis": _dec(
                False, ["insufficient_oos_trades"])}),
            _row("2024-01-02", {"adversarial_thesis": _dec(
                False, ["insufficient_oos_trades", "oos_edge_not_captured"])}),
        ])
        nf = _kinds(rep, KIND_NEW_FAILURE)
        self.assertEqual(len(nf), 1)
        self.assertEqual(nf[0]["added_reasons"], ["oos_edge_not_captured"])
        # Never promoted -> not a promotion_lost.
        self.assertEqual(_kinds(rep, KIND_PROMOTION_LOST), [])

    def test_transition_is_lost_not_new_failure(self):
        rep = detect_regressions([
            _row("2024-01-01", {"executable_ev": _dec(True, [])}),
            _row("2024-01-02", {"executable_ev": _dec(False, ["low_reject_precision"])}),
        ])
        self.assertEqual(_kinds(rep, KIND_NEW_FAILURE), [])
        self.assertEqual(len(_kinds(rep, KIND_PROMOTION_LOST)), 1)

    def test_stable_hold_is_not_a_regression(self):
        rep = detect_regressions([
            _row("2024-01-01", {"fill_model": _dec(False, ["insufficient_fills"])}),
            _row("2024-01-02", {"fill_model": _dec(False, ["insufficient_fills"])}),
        ])
        self.assertEqual(rep["regressions"], [])


class TestLayersFilter(unittest.TestCase):
    def test_scopes_summary_and_regressions(self):
        rep = detect_regressions([
            _row("2024-01-01", {"fill_model": _dec(True, []),
                                "executable_ev": _dec(True, [])}),
            _row("2024-01-02", {"fill_model": _dec(False, ["x"]),
                                "executable_ev": _dec(False, ["y"])}),
        ], layers=["fill_model"])
        self.assertEqual(set(rep["layers"]), {"fill_model"})
        self.assertTrue(all(e["layer"] == "fill_model" for e in rep["regressions"]))


class TestDeterminism(unittest.TestCase):
    def test_order_independent(self):
        rows = [
            _row("2024-01-01", {"fill_model": _dec(True, [])}),
            _row("2024-01-02", {"fill_model": _dec(False, ["a", "b"])}),
        ]
        self.assertEqual(detect_regressions(rows),
                         detect_regressions(list(reversed(rows))))


# --------------------------------------------------------------------------- #
# Runner wiring (offline; tmp files only)
# --------------------------------------------------------------------------- #
class TestRunnerRegressionWiring(unittest.TestCase):
    def test_check_regressions_empty_exits_zero(self):
        with tempfile.TemporaryDirectory() as d:
            audit = os.path.join(d, "audit.jsonl")
            self.assertEqual(
                rpc.main(["--check-regressions", "--audit-jsonl", audit]), 0)

    def test_seeded_promote_then_hold_is_detected(self):
        from datetime import datetime, timezone

        from oracle.promotion_audit import (
            load_promotion_history, record_promotion_check)
        with tempfile.TemporaryDirectory() as d:
            audit = os.path.join(d, "audit.jsonl")
            record_promotion_check(
                {"executable_ev": _dec(True, [])}, path=audit,
                now=datetime(2024, 1, 1, tzinfo=timezone.utc))
            # A real no-evidence run appends a HOLD for executable_ev.
            self.assertEqual(
                rpc.main(["--layer", "executable_ev", "--audit-jsonl", audit]), 0)
            rep = detect_regressions(load_promotion_history(path=audit))
            lost = _kinds(rep, KIND_PROMOTION_LOST)
            self.assertEqual(len(lost), 1)
            self.assertEqual(lost[0]["layer"], "executable_ev")
            self.assertEqual(
                rpc.main(["--check-regressions", "--audit-jsonl", audit]), 0)


if __name__ == "__main__":
    unittest.main()
