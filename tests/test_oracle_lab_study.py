"""
Oracle Lab — Phase-2 study acceptance tests (offline; no creds / no network).

These pin the Upgrade-B invariants — the Lab now *runs* and produces the
out-of-sample evidence a promotion is supposed to depend on:

  1. IS-vs-OOS report: ``compute_lab_report`` always emits an ``is_vs_oos``
     section (the acceptance-critical headline) and ``format_lab_report`` renders
     the "In-sample vs Out-of-sample" label + a Verdict line.
  2. OOS-collapse detection: a rule that wins in-sample and reverses
     out-of-sample is flagged ``oos_collapse=True`` -> verdict OVERFIT.
  3. Robustness: a rule that survives OOS is NOT flagged collapse -> not OVERFIT.
  4. Methodology guard: every sweep candidate is stamped ``in_sample_only=True``
     (the sweep never certifies a winner on the full sample).
  5. Determinism: the same (dataset, config, grid, seed) yields a byte-identical
     JSON-safe study result.
  6. Fail-open: junk / empty datasets never raise and render INSUFFICIENT_DATA.

No creds, no network, no file writes.
"""

import json
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.lab.experiment import ExperimentConfig
from oracle.lab.reports import (
    VERDICT_INSUFFICIENT,
    VERDICT_OVERFIT,
    compute_lab_report,
    format_lab_report,
)
from oracle.lab.run_phase2_study import run_study


GRID = {"momentum_threshold_pct": [1.0, 2.0]}


def _cfg():
    return ExperimentConfig.make("study_test", seed=1,
                                 params={"notional": 1000.0}, symbols=["AAA"])


def _row(day, mom, ret, regime="trending"):
    return {
        "symbol": "AAA",
        "as_of": f"2024-01-{day:02d}T16:00:00",
        "mode": "intraday",
        "ctx": {"momentum_5d_pct": mom, "regime": regime},
        "label_return_pct": ret,
    }


def _robust_dataset():
    # Alternating momentum: the tuned rule keeps working out-of-sample.
    rows = []
    for d in range(1, 25):
        if d % 2 == 1:
            rows.append(_row(d, 3.0, 4.0))     # up  -> call wins
        else:
            rows.append(_row(d, -3.0, -3.0))   # down -> put wins
    return rows


def _overfit_dataset():
    # Same signal wins for the first half, reverses in the second half. A single
    # fold keeps in-sample entirely winners and OOS entirely losers -> collapse.
    rows = [_row(d, 3.0, 5.0) for d in range(1, 13)]      # IS winners
    rows += [_row(d, 3.0, -6.0) for d in range(13, 25)]   # OOS losers
    return rows


class TestReportHasISvsOOS(unittest.TestCase):
    def test_report_carries_is_vs_oos_section(self):
        out = run_study(_robust_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=3)
        self.assertIn("is_vs_oos", out["report"])
        ivo = out["report"]["is_vs_oos"]
        for k in ("is_expectancy", "oos_expectancy", "oos_capture_ratio",
                  "oos_collapse", "is_trade_count", "oos_trade_count"):
            self.assertIn(k, ivo)

    def test_formatted_text_has_headline_and_verdict(self):
        txt = run_study(_robust_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=3)["report_text"]
        self.assertIn("In-sample vs Out-of-sample", txt)
        self.assertIn("Verdict:", txt)


class TestOOSCollapse(unittest.TestCase):
    def test_overfit_flags_collapse_and_overfit_verdict(self):
        out = run_study(_overfit_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=1)
        self.assertIs(out["report"]["is_vs_oos"]["oos_collapse"], True)
        self.assertEqual(out["report"]["verdict"], VERDICT_OVERFIT)

    def test_robust_not_flagged_collapse(self):
        out = run_study(_robust_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=3)
        self.assertIs(out["report"]["is_vs_oos"]["oos_collapse"], False)
        self.assertNotEqual(out["report"]["verdict"], VERDICT_OVERFIT)


class TestSweepInSampleOnly(unittest.TestCase):
    def test_every_sweep_candidate_is_in_sample_only(self):
        out = run_study(_robust_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=3)
        results = out["sweep"]["results"]
        self.assertTrue(results)
        self.assertTrue(all(r["in_sample_only"] for r in results))


class TestDeterminism(unittest.TestCase):
    def test_same_inputs_identical_study(self):
        a = run_study(_robust_dataset(), base_cfg=_cfg(),
                      param_grid=GRID, n_folds=3)
        b = run_study(_robust_dataset(), base_cfg=_cfg(),
                      param_grid=GRID, n_folds=3)
        self.assertEqual(
            json.dumps(a, sort_keys=True, default=str),
            json.dumps(b, sort_keys=True, default=str))


class TestBreakdownsUnlock(unittest.TestCase):
    def test_pooled_oos_breakdown_by_direction_present(self):
        out = run_study(_robust_dataset(), base_cfg=_cfg(),
                        param_grid=GRID, n_folds=3)
        bd = out["report"]["breakdowns"]["oos"]
        self.assertIn("by_direction", bd)
        self.assertTrue(bd["by_direction"])


class TestFailOpen(unittest.TestCase):
    def test_empty_report_is_insufficient_and_renders(self):
        rep = compute_lab_report({})
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)
        self.assertIn(VERDICT_INSUFFICIENT, format_lab_report(rep))

    def test_junk_datasets_never_raise(self):
        for junk in (None, 42, "x", [], [{"bad": 1}]):
            run_study(junk, base_cfg=_cfg(), param_grid=GRID)


if __name__ == "__main__":
    unittest.main()
