"""
Oracle 3.0 — Phase-2 Upgrade G acceptance tests: automated promotion gates.

Upgrade G codifies the manual Stage-4 promotion decision into a PURE,
deterministic gate so flipping a shadow layer to a live veto is auditable and
reproducible. ``oracle.promotion.evaluate_promotion(layer, calibration_stats,
lab_result, thresholds)`` returns ``{promote, reasons, metrics}`` and can only
ever WITHHOLD a promotion — it never edits ``.env``, flips a flag, or touches
the live path. ``run_promotion_check`` is the offline front end that assembles
the two evidence blocks from a ledger + a study JSON and prints the report.

Invariants:
  1. Fail-closed / fail-open: missing or malformed evidence (None / junk / empty)
     -> ``promote=False`` with a reason. Never a spurious PROMOTE.
  2. OOS collapse always withholds, regardless of sample size / capture.
  3. Insufficient sample (evals / sessions / OOS trades) withholds.
  4. "Margin inside the CI" (OOS edge lower-bound <= min_margin) withholds.
  5. Layer-specific gates: executable_ev needs correct would-be rejects;
     fill_model needs a well-calibrated fill rate + slippage.
  6. All gates pass -> ``promote=True`` with empty reasons.
  7. Thresholds can only TIGHTEN (a stricter .env value flips PROMOTE->HOLD).
  8. Determinism: identical inputs -> identical decision.
  9. Runner wiring: n_sessions is injected from distinct resolved dates; the
     walk_forward block is unwrapped from a study JSON; a full evidence set
     promotes end-to-end and tightened thresholds hold.

No creds, no network. File writes (runner tests) go to a tmp path only.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.promotion import (
    LAYERS,
    evaluate_promotion,
    format_promotion_report,
)
import run_promotion_check as rpc


# A full, passing calibration block (covers every layer's requirements).
def _cal(**over):
    base = {"n_resolved": 40, "n_filled": 40, "n_sessions": 8,
            "n_would_reject": 6, "reject_precision": 0.83,
            "fill_rate_bias": 0.05, "mean_slippage_error": 0.03}
    base.update(over)
    return base


# A robust out-of-sample walk-forward result.
def _lab(**over):
    base = {"oos_collapse": False, "oos_expectancy": 12.0,
            "oos_capture_ratio": 0.8, "oos_ci_low": 4.0,
            "oos_metrics": {"trade_count": 30, "expectancy": 12.0}}
    base.update(over)
    return base


class TestAllGatesPassPromotes(unittest.TestCase):
    def test_each_layer_promotes_on_full_evidence(self):
        for layer in LAYERS:
            dec = evaluate_promotion(layer, _cal(), _lab())
            self.assertTrue(dec["promote"], f"{layer}: {dec['reasons']}")
            self.assertEqual(dec["reasons"], [])


class TestFailOpenOnMissingInput(unittest.TestCase):
    def test_junk_and_empty_withhold(self):
        for bad in (None, 42, "x", [], {}):
            d1 = evaluate_promotion("executable_ev", bad, _lab())
            d2 = evaluate_promotion("executable_ev", _cal(), bad)
            self.assertFalse(d1["promote"])
            self.assertFalse(d2["promote"])
        # A non-dict stats/lab is a hard insufficient_input.
        self.assertEqual(
            evaluate_promotion("executable_ev", None, _lab())["reasons"],
            ["insufficient_input"])

    def test_unknown_layer_withholds(self):
        d = evaluate_promotion("nope", _cal(), _lab())
        self.assertFalse(d["promote"])
        self.assertEqual(d["reasons"], ["unknown_layer"])


class TestOosCollapseWithholds(unittest.TestCase):
    def test_collapse_blocks_even_with_perfect_stats(self):
        d = evaluate_promotion("adversarial_thesis", _cal(), _lab(oos_collapse=True))
        self.assertFalse(d["promote"])
        self.assertIn("oos_collapse", d["reasons"])


class TestInsufficientSample(unittest.TestCase):
    def test_too_few_evals(self):
        d = evaluate_promotion("executable_ev", _cal(n_resolved=3), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("insufficient_evals", d["reasons"])

    def test_too_few_sessions(self):
        d = evaluate_promotion("executable_ev", _cal(n_sessions=1), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("insufficient_sessions", d["reasons"])

    def test_missing_sessions_field_withholds(self):
        cal = _cal()
        cal.pop("n_sessions")
        d = evaluate_promotion("executable_ev", cal, _lab())
        self.assertFalse(d["promote"])
        self.assertIn("insufficient_sessions", d["reasons"])

    def test_too_few_oos_trades(self):
        d = evaluate_promotion(
            "adversarial_thesis", _cal(), _lab(oos_metrics={"trade_count": 3}))
        self.assertFalse(d["promote"])
        self.assertIn("insufficient_oos_trades", d["reasons"])


class TestMarginInsideCI(unittest.TestCase):
    def test_lower_bound_at_or_below_margin_withholds(self):
        # Explicit CI lower bound below the 0.0 margin -> not distinguishable.
        d = evaluate_promotion("adversarial_thesis", _cal(), _lab(oos_ci_low=-1.0))
        self.assertFalse(d["promote"])
        self.assertIn("margin_inside_ci", d["reasons"])

    def test_falls_back_to_point_estimate_when_no_ci(self):
        # No oos_ci_low; a non-positive point expectancy is the lower bound.
        lab = _lab(oos_expectancy=0.0)
        lab.pop("oos_ci_low")
        d = evaluate_promotion("adversarial_thesis", _cal(), lab)
        self.assertFalse(d["promote"])
        self.assertIn("margin_inside_ci", d["reasons"])

    def test_custom_margin_threshold_raises_bar(self):
        # Edge is real (+4 lower bound) but a 5.0 required margin withholds.
        d = evaluate_promotion("adversarial_thesis", _cal(), _lab(),
                               thresholds={"min_margin": 5.0})
        self.assertFalse(d["promote"])
        self.assertIn("margin_inside_ci", d["reasons"])


class TestEdgeNotCaptured(unittest.TestCase):
    def test_low_capture_ratio_withholds(self):
        d = evaluate_promotion("adversarial_thesis", _cal(), _lab(oos_capture_ratio=0.2))
        self.assertFalse(d["promote"])
        self.assertIn("oos_edge_not_captured", d["reasons"])


class TestExecutableEvLayer(unittest.TestCase):
    def test_no_would_reject_evidence_withholds(self):
        d = evaluate_promotion("executable_ev", _cal(n_would_reject=0), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("no_would_reject_evidence", d["reasons"])

    def test_low_reject_precision_withholds(self):
        d = evaluate_promotion("executable_ev", _cal(reject_precision=0.2), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("low_reject_precision", d["reasons"])


class TestFillModelLayer(unittest.TestCase):
    def test_biased_fill_rate_and_slippage_withhold(self):
        d = evaluate_promotion(
            "fill_model", _cal(fill_rate_bias=0.40, mean_slippage_error=0.50), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("fill_rate_bias_too_high", d["reasons"])
        self.assertIn("slippage_error_too_high", d["reasons"])

    def test_insufficient_fills_withholds(self):
        d = evaluate_promotion("fill_model", _cal(n_filled=2), _lab())
        self.assertFalse(d["promote"])
        self.assertIn("insufficient_fills", d["reasons"])


class TestThresholdsOnlyTighten(unittest.TestCase):
    def test_stricter_capture_flips_promote_to_hold(self):
        passing = evaluate_promotion("adversarial_thesis", _cal(), _lab())
        self.assertTrue(passing["promote"])
        strict = evaluate_promotion("adversarial_thesis", _cal(), _lab(),
                                    thresholds={"min_capture": 0.99})
        self.assertFalse(strict["promote"])
        self.assertIn("oos_edge_not_captured", strict["reasons"])


class TestDeterminism(unittest.TestCase):
    def test_identical_inputs_identical_decision(self):
        a = evaluate_promotion("executable_ev", _cal(), _lab())
        b = evaluate_promotion("executable_ev", _cal(), _lab())
        self.assertEqual(a, b)


class TestReportRenders(unittest.TestCase):
    def test_verdicts_render(self):
        promote = evaluate_promotion("executable_ev", _cal(), _lab())
        hold = evaluate_promotion("executable_ev", _cal(n_resolved=1), _lab())
        self.assertIn("PROMOTE", format_promotion_report(promote))
        self.assertIn("HOLD", format_promotion_report(hold))


# --------------------------------------------------------------------------- #
# Runner wiring (offline; tmp files only)
# --------------------------------------------------------------------------- #
class TestRunnerSessionCount(unittest.TestCase):
    def test_counts_distinct_resolved_dates_only(self):
        recs = [
            {"resolution_status": "resolved", "resolved_at": "2024-01-01T15:00:00"},
            {"resolution_status": "resolved", "resolved_at": "2024-01-01T16:00:00"},
            {"resolution_status": "resolved", "resolved_at": "2024-01-02T16:00:00"},
            {"resolution_status": "resolved", "recorded_at": "2024-01-03T16:00:00"},
            {"resolution_status": "pending", "resolved_at": "2024-01-09T16:00:00"},
        ]
        self.assertEqual(rpc._count_sessions(recs), 3)


class TestRunnerLabUnwrap(unittest.TestCase):
    def test_unwraps_walk_forward_and_fails_open(self):
        with tempfile.TemporaryDirectory() as d:
            wf = _lab()
            study = os.path.join(d, "study.json")
            with open(study, "w", encoding="utf-8") as fh:
                json.dump({"walk_forward": wf, "report": {}}, fh)
            self.assertEqual(rpc.load_lab_result(study), wf)
            # Missing file -> None.
            self.assertIsNone(rpc.load_lab_result(os.path.join(d, "nope.json")))
        self.assertIsNone(rpc.load_lab_result(None))


class TestRunnerThresholdMap(unittest.TestCase):
    def test_maps_promo_env_keys(self):
        with tempfile.TemporaryDirectory() as d:
            envp = os.path.join(d, ".env")
            with open(envp, "w", encoding="utf-8") as fh:
                fh.write("PROMO_MIN_EVALS=50\nPROMO_MIN_CAPTURE=0.9\nJUNK=x\n")
            th = rpc.resolve_thresholds(envp)
            self.assertEqual(th.get("min_evals"), 50.0)
            self.assertEqual(th.get("min_capture"), 0.9)
            self.assertNotIn("JUNK", th)


class TestRunnerEndToEnd(unittest.TestCase):
    def test_full_evidence_promotes_and_tightening_holds(self):
        rep = rpc.build_report(_cal(), _lab(), {}, layers=("executable_ev",))
        self.assertTrue(rep["decisions"]["executable_ev"]["promote"])
        self.assertIn("does NOT edit .env", rep["text"])
        rep2 = rpc.build_report(_cal(), _lab(), {"min_capture": 0.99},
                                layers=("executable_ev",))
        self.assertFalse(rep2["decisions"]["executable_ev"]["promote"])

    def test_no_evidence_holds_every_layer(self):
        rep = rpc.build_report(None, None, {})
        self.assertFalse(any(d.get("promote") for d in rep["decisions"].values()))


if __name__ == "__main__":
    unittest.main()
