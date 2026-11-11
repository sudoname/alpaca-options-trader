"""
Phase 3 regression tests: probability-of-profit recalibration with temporal
integrity (pop_recalibration.py).

Covers:
  * isotonic (PAV) correctness, tie pooling, monotonicity, clamping;
  * cohort -> pooled -> uncalibrated fallback with reported sample sizes;
  * exit-policy alignment and simulated-evidence gating (excluded + counted);
  * walk-forward: prior-sessions-only fits (no leakage), purge of outcomes
    that close after the next session starts, embargo widening, untouched
    final evaluation sessions;
  * calibration summary carries NO profitability fields; determinism; input
    dicts are not mutated.

Deterministic, offline, no credentials.
"""
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from pop_recalibration import (  # noqa: E402
    SOURCE_COHORT,
    SOURCE_POOLED,
    SOURCE_UNCALIBRATED,
    IsotonicCalibrator,
    calibrate_p,
    calibration_summary,
    final_eval_report,
    fit_pop_calibrators,
    recalibrate_walk_forward,
)


def _obs(t_decision, pnl, p=0.7, regime="trend", t_outcome=None, **kw):
    o = {
        "p_win": p,
        "net_pnl_pct": pnl,
        "regime": regime,
        "t_decision": t_decision,
        "t_outcome": t_outcome or t_decision.replace(":00:00", ":30:00"),
    }
    o.update(kw)
    return o


class IsotonicCalibratorTest(unittest.TestCase):
    def test_pav_hand_computed_blocks(self):
        # sorted: 0.2->0, 0.4->1, 0.6->0 (violation, merge to mean 0.5), 0.8->1
        iso = IsotonicCalibrator.fit(
            [(0.2, 0), (0.4, 1), (0.6, 0), (0.8, 1)])
        self.assertEqual(iso.n, 4)
        self.assertEqual(
            iso.blocks,
            ((0.2, 0.2, 0.0, 1), (0.4, 0.6, 0.5, 2), (0.8, 0.8, 1.0, 1)))
        self.assertEqual(iso.predict(0.1), 0.0)
        self.assertEqual(iso.predict(0.5), 0.5)
        self.assertEqual(iso.predict(0.7), 1.0)

    def test_tied_raw_probabilities_are_pooled(self):
        # 10 wins + 10 losses, all predicted 0.8 -> exactly 0.5, one block.
        pairs = [(0.8, 1)] * 10 + [(0.8, 0)] * 10
        iso = IsotonicCalibrator.fit(pairs)
        self.assertEqual(len(iso.blocks), 1)
        self.assertEqual(iso.predict(0.8), 0.5)

    def test_overconfidence_is_recentered(self):
        # Model says 0.9; reality is 18/30 = 0.6.
        pairs = [(0.9, 1)] * 18 + [(0.9, 0)] * 12
        iso = IsotonicCalibrator.fit(pairs)
        self.assertEqual(iso.predict(0.9), 0.6)

    def test_predictions_are_monotone_in_p(self):
        import random
        rng = random.Random(11)
        pairs = [(rng.random(), 1 if rng.random() < 0.5 else 0)
                 for _ in range(300)]
        iso = IsotonicCalibrator.fit(pairs)
        grid = [iso.predict(x / 100.0) for x in range(101)]
        for a, b in zip(grid, grid[1:]):
            self.assertLessEqual(a, b + 1e-12)

    def test_clamping_and_bad_inputs(self):
        iso = IsotonicCalibrator.fit([(0.3, 0), (0.7, 1)])
        self.assertEqual(iso.predict(-0.5), iso.predict(0.0))
        self.assertEqual(iso.predict(1.5), iso.predict(1.0))
        self.assertIsNone(iso.predict("junk"))
        self.assertIsNone(iso.predict(None))
        empty = IsotonicCalibrator.fit([])
        self.assertIsNone(empty.predict(0.5))


class FallbackChainTest(unittest.TestCase):
    def _train(self):
        rows = []
        # trend: 30 rows at p=0.7, 21 wins -> cohort calibrates 0.7 -> 0.7
        rows += [_obs("2026-01-01T10:00:00+00:00", 1.0, regime="trend")
                 for _ in range(21)]
        rows += [_obs("2026-01-01T10:00:00+00:00", -1.0, regime="trend")
                 for _ in range(9)]
        # chop: 30 rows at p=0.7, 9 wins -> cohort calibrates 0.7 -> 0.3
        rows += [_obs("2026-01-01T11:00:00+00:00", 1.0, regime="chop")
                 for _ in range(9)]
        rows += [_obs("2026-01-01T11:00:00+00:00", -1.0, regime="chop")
                 for _ in range(21)]
        return rows

    def test_cohort_source_with_sample_size(self):
        model = fit_pop_calibrators(self._train(),
                                    min_cohort_n=30, min_pooled_n=20)
        trend = calibrate_p(model, 0.7, "trend")
        chop = calibrate_p(model, 0.7, "chop")
        self.assertEqual(trend["source"], SOURCE_COHORT)
        self.assertEqual(trend["n_fit"], 30)
        self.assertEqual(trend["p_calibrated"], 0.7)
        self.assertEqual(chop["source"], SOURCE_COHORT)
        self.assertEqual(chop["p_calibrated"], 0.3)

    def test_small_cohort_falls_back_to_pooled(self):
        rows = self._train()[:35]  # trend 30 + 5 chop rows
        model = fit_pop_calibrators(rows, min_cohort_n=30, min_pooled_n=20)
        self.assertIsNone(model["cohorts"]["chop"]["calibrator"])
        self.assertEqual(model["cohorts"]["chop"]["n"], 5)
        c = calibrate_p(model, 0.7, "chop")
        self.assertEqual(c["source"], SOURCE_POOLED)
        self.assertEqual(c["n_fit"], 35)

    def test_unknown_cohort_falls_back_to_pooled(self):
        model = fit_pop_calibrators(self._train(),
                                    min_cohort_n=30, min_pooled_n=20)
        c = calibrate_p(model, 0.7, "never_seen")
        self.assertEqual(c["source"], SOURCE_POOLED)
        self.assertEqual(c["n_fit"], 60)

    def test_thin_pooled_evidence_passes_raw_through(self):
        model = fit_pop_calibrators(self._train()[:5],
                                    min_cohort_n=30, min_pooled_n=20)
        c = calibrate_p(model, 0.7, "trend")
        self.assertEqual(c["source"], SOURCE_UNCALIBRATED)
        self.assertEqual(c["p_calibrated"], 0.7)
        self.assertEqual(c["n_fit"], 5)

    def test_unparseable_prediction(self):
        model = fit_pop_calibrators(self._train())
        c = calibrate_p(model, "n/a", "trend")
        self.assertIsNone(c["p_calibrated"])
        self.assertEqual(c["source"], SOURCE_UNCALIBRATED)
        self.assertEqual(c["n_fit"], 0)

    def test_exit_policy_mismatch_is_excluded_and_counted(self):
        rows = [_obs("2026-01-01T10:00:00+00:00", 1.0,
                     exit_policy="baseline") for _ in range(10)]
        rows += [_obs("2026-01-01T10:00:00+00:00", -1.0,
                      exit_policy="tight_stop") for _ in range(4)]
        model = fit_pop_calibrators(rows, exit_policy="baseline",
                                    min_pooled_n=5)
        self.assertEqual(model["pooled"]["n"], 10)
        self.assertEqual(model["excluded"], {"exit_policy_mismatch": 4})

    def test_simulated_excluded_by_default_and_counted_when_included(self):
        rows = [_obs("2026-01-01T10:00:00+00:00", 1.0) for _ in range(6)]
        rows += [_obs("2026-01-01T10:00:00+00:00", -1.0, simulated=True)
                 for _ in range(3)]
        model = fit_pop_calibrators(rows, min_pooled_n=5)
        self.assertEqual(model["pooled"]["n"], 6)
        self.assertEqual(model["excluded"], {"simulated_excluded": 3})
        self.assertEqual(model["n_simulated_in_fit"], 0)

        model_sim = fit_pop_calibrators(rows, min_pooled_n=5,
                                        include_simulated=True)
        self.assertEqual(model_sim["pooled"]["n"], 9)
        self.assertEqual(model_sim["n_simulated_in_fit"], 3)
        self.assertEqual(model_sim["excluded"], {})

    def test_missing_prediction_or_outcome_excluded(self):
        rows = [
            {"p_win": None, "net_pnl_pct": 1.0,
             "t_decision": "2026-01-01T10:00:00+00:00"},
            {"p_win": 0.6, "net_pnl_pct": None,
             "t_decision": "2026-01-01T10:00:00+00:00"},
            "not even a dict",
        ]
        model = fit_pop_calibrators(rows)
        self.assertEqual(model["pooled"]["n"], 0)
        self.assertEqual(model["excluded"],
                         {"no_prediction": 1, "no_outcome": 1,
                          "not_a_dict": 1})


class WalkForwardTest(unittest.TestCase):
    def _sessions(self, days=4, per_day=6, win_rate=0.5):
        """per_day rows per UTC day, raw p=0.8, wins first (deterministic)."""
        rows = []
        wins = int(per_day * win_rate)
        for d in range(1, days + 1):
            for i in range(per_day):
                ts = f"2026-02-{d:02d}T{10 + i}:00:00+00:00"
                rows.append(_obs(ts, 1.0 if i < wins else -1.0, p=0.8))
        return rows

    def test_first_session_runs_uncalibrated(self):
        res = recalibrate_walk_forward(self._sessions(), min_cohort_n=99,
                                       min_pooled_n=5, final_eval_frac=0.0)
        s0 = res["sessions"][0]
        self.assertEqual(s0["train_n"], 0)
        self.assertEqual(s0["sources"], {SOURCE_UNCALIBRATED: 6})
        first_rows = [r for r in res["rows"] if r["session"] == s0["session"]]
        for r in first_rows:
            self.assertEqual(r["p_calibrated"], r["p_raw"])
            self.assertEqual(r["n_fit"], 0)

    def test_later_sessions_fit_on_prior_sessions_only(self):
        res = recalibrate_walk_forward(self._sessions(), min_cohort_n=99,
                                       min_pooled_n=5, final_eval_frac=0.0)
        # Session 2 fit on session 1 (6 rows, 3 wins at p=0.8) -> 0.5.
        s1 = res["sessions"][1]
        self.assertEqual(s1["train_n"], 6)
        self.assertEqual(s1["sources"], {SOURCE_POOLED: 6})
        day2 = [r for r in res["rows"] if r["session"] == s1["session"]]
        for r in day2:
            self.assertEqual(r["p_calibrated"], 0.5)
            self.assertEqual(r["n_fit"], 6)
        # Session 4 fit on sessions 1-3.
        self.assertEqual(res["sessions"][3]["train_n"], 18)

    def test_no_leakage_from_later_sessions(self):
        base = self._sessions()
        res_a = recalibrate_walk_forward(base, min_cohort_n=99,
                                         min_pooled_n=5, final_eval_frac=0.0)
        # Flip every outcome in the LAST session; earlier sessions' calibrated
        # outputs must be byte-identical.
        flipped = [dict(o) for o in base]
        for o in flipped:
            if o["t_decision"].startswith("2026-02-04"):
                o["net_pnl_pct"] = -o["net_pnl_pct"]
        res_b = recalibrate_walk_forward(flipped, min_cohort_n=99,
                                         min_pooled_n=5, final_eval_frac=0.0)
        key = lambda r: (r["t_decision"], r["p_raw"], r["p_calibrated"],
                         r["source"], r["n_fit"])
        early_a = [key(r) for r in res_a["rows"]
                   if not r["session"].endswith("-04")]
        early_b = [key(r) for r in res_b["rows"]
                   if not r["session"].endswith("-04")]
        self.assertEqual(early_a, early_b)

    def test_outcome_closing_after_next_session_start_is_purged(self):
        rows = self._sessions(days=2)
        # One session-1 win's outcome window closes AFTER session 2 begins.
        overlap = next(r for r in rows
                       if r["t_decision"].startswith("2026-02-01")
                       and r["net_pnl_pct"] > 0)
        overlap["t_outcome"] = "2026-02-02T10:30:00+00:00"
        res = recalibrate_walk_forward(rows, min_cohort_n=99,
                                       min_pooled_n=3, final_eval_frac=0.0)
        s1 = res["sessions"][1]
        self.assertEqual(s1["purged"], 1)
        self.assertEqual(s1["train_n"], 5)  # 6 - 1 purged
        # 2 wins / 5 rows at p=0.8 -> 0.4 (would be 0.5 without the purge).
        day2 = [r for r in res["rows"] if r["session"] == s1["session"]]
        self.assertTrue(all(r["p_calibrated"] == 0.4 for r in day2))

    def test_embargo_widens_the_exclusion(self):
        rows = self._sessions(days=2)
        near = next(r for r in rows
                    if r["t_decision"].startswith("2026-02-01"))
        near["t_outcome"] = "2026-02-02T09:45:00+00:00"  # 15 min pre-open
        res0 = recalibrate_walk_forward(rows, min_cohort_n=99,
                                        min_pooled_n=3, final_eval_frac=0.0,
                                        embargo_minutes=0.0)
        res60 = recalibrate_walk_forward(rows, min_cohort_n=99,
                                         min_pooled_n=3, final_eval_frac=0.0,
                                         embargo_minutes=60.0)
        self.assertEqual(res0["sessions"][1]["purged"], 0)
        self.assertEqual(res60["sessions"][1]["purged"], 1)
        self.assertEqual(res60["sessions"][1]["train_n"], 5)

    def test_final_eval_sessions_are_flagged_and_kept_out_of_summary(self):
        rows = self._sessions(days=5, per_day=4)
        res = recalibrate_walk_forward(rows, min_cohort_n=99,
                                       min_pooled_n=3, final_eval_frac=0.2)
        self.assertEqual(res["n_sessions"], 5)
        self.assertEqual(res["n_final_eval_sessions"], 1)
        self.assertTrue(res["final_eval_untouched"])
        final_rows = [r for r in res["rows"] if r["final_eval"]]
        self.assertEqual({r["session"] for r in final_rows}, {"2026-02-05"})
        self.assertEqual(len(final_rows), 4)
        # Development summary excludes the held-out rows.
        self.assertEqual(res["summary"]["n"], 16)
        rpt = final_eval_report(res["rows"])
        self.assertEqual(rpt["period"], "final_eval")
        self.assertEqual(rpt["n"], 4)
        self.assertEqual(rpt["n_sessions"], 1)

    def test_summary_carries_no_profitability_fields(self):
        res = recalibrate_walk_forward(self._sessions(), min_cohort_n=99,
                                       min_pooled_n=5, final_eval_frac=0.0)
        banned = ("pnl", "profit", "expectancy", "dollars", "return")
        for k in res["summary"]:
            for word in banned:
                self.assertNotIn(word, k.lower())

    def test_simulated_and_policy_gating_in_walk_forward(self):
        rows = self._sessions(days=2)
        rows.append(_obs("2026-02-01T16:00:00+00:00", 1.0, p=0.8,
                         simulated=True))
        rows.append(_obs("2026-02-01T17:00:00+00:00", 1.0, p=0.8,
                         exit_policy="other"))
        res = recalibrate_walk_forward(rows, min_cohort_n=99, min_pooled_n=3,
                                       final_eval_frac=0.0,
                                       exit_policy=None)
        # No policy filter: the 'other' row stays; simulated still excluded.
        self.assertEqual(res["excluded"].get("simulated_excluded"), 1)
        self.assertEqual(len(res["rows"]), 13)
        res2 = recalibrate_walk_forward(rows, min_cohort_n=99, min_pooled_n=3,
                                        final_eval_frac=0.0,
                                        exit_policy="baseline")
        # Named policy: everything without exit_policy == 'baseline' drops.
        # (Policy alignment is checked before the simulated flag, so the
        # simulated row is also counted as a policy mismatch: 12 + 1 + 1.)
        self.assertEqual(len(res2["rows"]), 0)
        self.assertEqual(res2["excluded"].get("exit_policy_mismatch"), 14)
        self.assertNotIn("simulated_excluded", res2["excluded"])

    def test_rows_without_timestamp_are_counted(self):
        rows = self._sessions(days=1)
        rows.append({"p_win": 0.6, "net_pnl_pct": 1.0, "t_decision": None,
                     "t_outcome": None})
        res = recalibrate_walk_forward(rows, min_cohort_n=99, min_pooled_n=3,
                                       final_eval_frac=0.0)
        self.assertEqual(res["excluded"].get("no_timestamp"), 1)
        self.assertEqual(len(res["rows"]), 6)

    def test_deterministic_and_input_not_mutated(self):
        rows = self._sessions()
        snapshot = [dict(o) for o in rows]
        res_a = recalibrate_walk_forward(rows, min_cohort_n=99,
                                         min_pooled_n=5, final_eval_frac=0.25)
        res_b = recalibrate_walk_forward(rows, min_cohort_n=99,
                                         min_pooled_n=5, final_eval_frac=0.25)
        self.assertEqual(res_a["rows"], res_b["rows"])
        self.assertEqual(res_a["sessions"], res_b["sessions"])
        self.assertEqual(rows, snapshot)  # caller dicts untouched
        for o in rows:
            self.assertNotIn("p_calibrated", o)


class CalibrationSummaryTest(unittest.TestCase):
    def test_brier_raw_vs_calibrated(self):
        # Raw p=0.9 on a 50% reality; calibrated to 0.5.
        rows = []
        for i in range(10):
            rows.append({"p_win": 0.9, "p_calibrated": 0.5,
                         "net_pnl_pct": 1.0 if i < 5 else -1.0,
                         "source": SOURCE_POOLED, "n_fit": 40})
        s = calibration_summary(rows)
        self.assertEqual(s["n"], 10)
        self.assertAlmostEqual(s["brier_raw"], 0.41, places=9)
        self.assertAlmostEqual(s["brier_calibrated"], 0.25, places=9)
        self.assertEqual(s["sources"], {SOURCE_POOLED: 10})
        self.assertEqual(s["avg_n_fit"], 40.0)
        self.assertEqual(len(s["reliability_raw"]), 5)

    def test_empty_rows(self):
        s = calibration_summary([])
        self.assertEqual(s["n"], 0)
        self.assertIsNone(s["brier_raw"])
        self.assertIsNone(s["avg_n_fit"])


if __name__ == "__main__":
    unittest.main()
