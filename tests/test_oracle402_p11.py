"""Tests for the Oracle 4.0.2 promotion gate (P11).

Builds synthetic backtest artifacts exercising every verdict branch and
the never-auto-promote invariant.
"""
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.ml import promotion402 as pm  # noqa: E402


def _artifact(*, exp=10.0, mean=100.0, ci_lo=5.0, ci_hi=300.0,
              pf=1.5, sharpe=1.4, net_15=50.0,
              n_sessions=80, trades_kept=300, n_executed=150,
              status="ok", with_paired=True):
    """One-challenger artifact with every knob settable per test."""
    variant = {
        "status": status,
        "accounting": {"n": n_executed},
        "by_friction_mult": {
            "1.0": {"expectancy_per_trade": exp, "profit_factor": pf,
                    "sharpe_daily_ann": sharpe, "net_pnl": exp * n_executed,
                    "n_executed": n_executed},
            "1.5": {"net_pnl": net_15},
            "2.0": {"net_pnl": net_15 - 100.0},
        },
        "stop_overshoot_stress": {"net_pnl": 0.0},
    }
    art = {
        "fingerprint": "f" * 64,
        "variants": {"baseline_frozen": dict(variant), "chall": variant},
        "paired_vs_baseline": {},
        "session_calendar": {"n_sessions": n_sessions},
        "trade_universe": {"trades_kept": trades_kept},
        "risk_metrics_limited": {"limited": True},
        "matched_cohorts": {"classical": {"paired": {"bootstrap": {
            "mean": 1.0, "ci_lo": 0.5, "ci_hi": 1.5, "n_sessions": 5}}}},
    }
    if with_paired:
        art["paired_vs_baseline"]["chall"] = {"bootstrap": {
            "mean": mean, "ci_lo": ci_lo, "ci_hi": ci_hi, "n_sessions": 7}}
    return art


class TestEvaluateVariant:
    def test_all_pass_is_eligible_pending_human_review(self):
        r = pm.evaluate_variant(_artifact(), "chall")
        assert r["verdict"] == "ELIGIBLE_PENDING_HUMAN_REVIEW"
        assert r["failed"] == []
        assert all(c["pass"] is True for c in r["criteria"])

    def test_confidently_harmful_is_do_not_promote(self):
        art = _artifact(mean=-50.0, ci_lo=-120.0, ci_hi=-1.0)
        r = pm.evaluate_variant(art, "chall")
        assert r["verdict"] == "DO_NOT_PROMOTE"
        assert "confidently WORSE" in r["why"]

    def test_do_not_promote_trumps_insufficient_evidence(self):
        art = _artifact(mean=-50.0, ci_lo=-120.0, ci_hi=0.0, n_sessions=3)
        r = pm.evaluate_variant(art, "chall")
        assert r["verdict"] == "DO_NOT_PROMOTE"

    def test_evidence_unmet_is_insufficient_evidence(self):
        for kw in ({"n_sessions": 59}, {"trades_kept": 199},
                   {"n_executed": 99}):
            r = pm.evaluate_variant(_artifact(**kw), "chall")
            assert r["verdict"] == "INSUFFICIENT_EVIDENCE", kw

    def test_evidence_trumps_performance_failures(self):
        # bad performance AND thin evidence -> INSUFFICIENT_EVIDENCE
        art = _artifact(exp=-5.0, pf=0.8, n_sessions=10)
        r = pm.evaluate_variant(art, "chall")
        assert r["verdict"] == "INSUFFICIENT_EVIDENCE"

    def test_performance_failure_is_no_validated_edge(self):
        cases = [{"exp": -1.0}, {"ci_lo": -1.0}, {"pf": 1.2},
                 {"sharpe": 0.5}, {"net_15": -10.0}]
        for kw in cases:
            r = pm.evaluate_variant(_artifact(**kw), "chall")
            assert r["verdict"] == "NO_VALIDATED_EDGE", kw
            assert r["failed"], kw

    def test_missing_metric_fails_its_criterion(self):
        # None profit factor (no losses recorded) must not pass C4.
        r = pm.evaluate_variant(_artifact(pf=None), "chall")
        assert r["verdict"] == "NO_VALIDATED_EDGE"
        assert "C4_profit_factor" in r["failed"]

    def test_unavailable_variant_not_scoreable(self):
        r = pm.evaluate_variant(_artifact(status="unavailable"), "chall")
        assert r["verdict"] == "NOT_SCOREABLE"

    def test_no_paired_entry_not_scoreable(self):
        r = pm.evaluate_variant(_artifact(with_paired=False), "chall")
        assert r["verdict"] == "NOT_SCOREABLE"
        assert "paired" in r["why"]

    def test_verdict_vocabulary_is_closed(self):
        # whatever the knobs, the verdict is from the frozen set
        for kw in ({}, {"mean": -1.0, "ci_hi": -0.1}, {"n_sessions": 1},
                   {"pf": 0.1}, {"status": "unavailable"}):
            r = pm.evaluate_variant(_artifact(**kw), "chall")
            assert r["verdict"] in pm.VERDICTS + ("NOT_SCOREABLE",)
            # the module can never emit a "PROMOTE"-style verdict
            assert "PROMOTE" not in r["verdict"] or \
                r["verdict"] == "DO_NOT_PROMOTE"


class TestRunPromotion:
    def test_end_to_end_report(self, tmp_path):
        art_path = str(tmp_path / "replay402.json")
        out_path = str(tmp_path / "promotion402.json")
        with open(art_path, "w", encoding="utf-8") as fh:
            json.dump(_artifact(), fh)
        report = pm.run_promotion(art_path, out_path=out_path)
        assert report["phase"] == 11
        assert report["inputs"]["backtest_fingerprint"] == "f" * 64
        assert "baseline_frozen" not in report["variants"]
        assert report["best_variant"] == "chall"
        assert report["overall_verdict"] == "ELIGIBLE_PENDING_HUMAN_REVIEW"
        assert report["never_auto_promote"] == pm.NEVER_AUTO_PROMOTE_NOTE
        assert report["criteria_spec"]["verdicts"] == list(pm.VERDICTS)
        assert report["matched_cohorts_summary"]["classical"]["mean"] == 1.0
        with open(out_path, "r", encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["report_fingerprint"] == report["report_fingerprint"]

    def test_nothing_scoreable_overall_insufficient(self, tmp_path):
        art_path = str(tmp_path / "replay402.json")
        with open(art_path, "w", encoding="utf-8") as fh:
            json.dump(_artifact(status="unavailable"), fh)
        report = pm.run_promotion(art_path)
        assert report["best_variant"] is None
        assert report["overall_verdict"] == "INSUFFICIENT_EVIDENCE"

    def test_best_variant_is_highest_paired_mean(self, tmp_path):
        art = _artifact()
        art["variants"]["weak"] = json.loads(
            json.dumps(art["variants"]["chall"]))
        art["paired_vs_baseline"]["weak"] = {"bootstrap": {
            "mean": 1.0, "ci_lo": 0.5, "ci_hi": 2.0, "n_sessions": 7}}
        art_path = str(tmp_path / "replay402.json")
        with open(art_path, "w", encoding="utf-8") as fh:
            json.dump(art, fh)
        report = pm.run_promotion(art_path)
        assert report["best_variant"] == "chall"


class TestCli:
    def test_main_smoke(self, tmp_path, capsys):
        art_path = str(tmp_path / "replay402.json")
        out_path = str(tmp_path / "promotion402.json")
        with open(art_path, "w", encoding="utf-8") as fh:
            json.dump(_artifact(pf=1.2), fh)
        rc = pm.main(["--artifact", art_path, "--out", out_path])
        assert rc == 0
        text = capsys.readouterr().out
        assert "NO_VALIDATED_EDGE" in text
        assert "best_variant=chall" in text
        assert "report_fingerprint=" in text
        assert os.path.exists(out_path)
