"""Phase 7c tests: pre-registered promotion gates and three-tier verdict."""
import json

import pytest

from oracle.ml import promotion as pm


# --------------------------------------------------------------------------- #
# fixture builders (synthetic replay-shaped dicts; no market data invented —
# these exercise gate plumbing only)
# --------------------------------------------------------------------------- #

def _replay(*, daily, uplift_lo, net15, net20, sharpe, pf, v_dd, b_dd,
            parity_ok=True, mark_derived=0):
    return {
        "fingerprint": "test-fp",
        "sessions_replayed": ["2026-11-04", "2026-11-05"],
        "parity": {"ok": parity_ok},
        "holdout_refusal": {"sessions": []},
        "trade_universe": {"outcome_status_counts": {"mark_derived": mark_derived}},
        "paired_vs_baseline": {
            "combined": {"bootstrap": {"ci_lo": uplift_lo,
                                       "ci_hi": uplift_lo + 100.0,
                                       "mean": uplift_lo + 50.0,
                                       "n_sessions": len(daily)}},
        },
        "variants": {
            "combined": {
                "by_friction_mult": {
                    "1.0": {
                        "pnl_by_exit_day": {f"2026-11-{4 + i:02d}": p
                                            for i, p in enumerate(daily)},
                        "net_pnl": sum(daily),
                        "n_executed": 40,
                        "sharpe_daily_ann": sharpe,
                        "profit_factor": pf,
                        "max_drawdown": v_dd,
                    },
                    "1.5": {"net_pnl": net15},
                    "2.0": {"net_pnl": net20},
                },
                "stop_overshoot_stress": {"net_pnl": net20 - 100.0},
            },
            "baseline_frozen": {
                "by_friction_mult": {"1.0": {"max_drawdown": b_dd}},
            },
        },
    }


def _forward(**over):
    fe = {
        "forward_sessions": 80,
        "resolved_opportunities": 300,
        "actual_fills": 250,
        "calibration_improved": True,
        "coverage_disparity": 0.02,
        "live_fill_based": True,
    }
    fe.update(over)
    return fe


def _good_replay(**over):
    kw = dict(daily=[100.0, 120.0, 90.0, 110.0], uplift_lo=10.0,
              net15=500.0, net20=200.0, sharpe=2.0, pf=1.8,
              v_dd=-0.02, b_dd=-0.05)
    kw.update(over)
    return _replay(**kw)


def _status(report, name):
    return next(x for x in report["gates"] if x["name"] == name)["status"]


# --------------------------------------------------------------------------- #
# verdict tiers
# --------------------------------------------------------------------------- #

def test_thin_evidence_is_insufficient_never_relaxed():
    rep = pm.evaluate_promotion(_good_replay())  # no forward evidence at all
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "never relaxed" in rep["verdict_reason"]
    # the three volume gates are INSUFFICIENT without forward evidence,
    # even though replay context numbers are shown
    for name in ("forward_sessions", "resolved_opportunities",
                 "actual_fills_effective_n", "calibration_improved",
                 "coverage_disparity", "evidence_basis"):
        assert _status(rep, name) == "INSUFFICIENT"
    # replay sessions list is reported as a count, not a raw list
    sess = next(x for x in rep["gates"] if x["name"] == "forward_sessions")
    assert sess["observed"] == 2
    # report metadata invariants
    assert rep["human_review_required"] is True
    assert rep["no_auto_promotion"] is True
    assert rep["gates_preregistered"] == pm.GATES_PREREGISTERED
    assert rep["replay_fingerprint"] == "test-fp"


def test_sufficient_evidence_with_failing_perf_is_do_not_promote():
    replay = _replay(daily=[-200.0, -150.0, -300.0, -50.0], uplift_lo=-80.0,
                     net15=-1000.0, net20=-1500.0, sharpe=0.5, pf=1.0,
                     v_dd=-0.30, b_dd=-0.05)
    rep = pm.evaluate_promotion(replay, forward_evidence=_forward())
    assert rep["n_insufficient"] == 0
    assert rep["n_fail"] > 0
    assert rep["verdict"] == "DO_NOT_PROMOTE"
    for name in ("expectancy_bootstrap_lb", "paired_uplift_bootstrap_lb",
                 "net_positive_at_1_5x_friction", "sharpe_daily_ann",
                 "profit_factor", "max_drawdown"):
        assert _status(rep, name) == "FAIL"


def test_all_pass_is_eligible_pending_human_review_only():
    rep = pm.evaluate_promotion(_good_replay(), forward_evidence=_forward())
    assert rep["n_fail"] == 0
    assert rep["n_insufficient"] == 0
    assert rep["verdict"] == "ELIGIBLE_PENDING_HUMAN_REVIEW"
    # there is no PROMOTE verdict and no side-effect fields
    assert "PROMOTE" not in rep["verdict"].replace("DO_NOT_PROMOTE", "")
    assert rep["human_review_required"] is True
    assert rep["no_auto_promotion"] is True
    assert "registry.approve_model" in rep["note"]


def test_non_live_evidence_basis_fails_even_when_perf_passes():
    rep = pm.evaluate_promotion(
        _good_replay(),
        forward_evidence=_forward(live_fill_based=False))
    assert _status(rep, "evidence_basis") == "FAIL"
    assert rep["verdict"] == "DO_NOT_PROMOTE"


def test_parity_violation_fails_mandatory_invariants():
    rep = pm.evaluate_promotion(_good_replay(parity_ok=False),
                                forward_evidence=_forward())
    assert _status(rep, "mandatory_invariants") == "FAIL"
    assert rep["verdict"] == "DO_NOT_PROMOTE"


def test_drawdown_worse_than_baseline_fails_despite_within_limit():
    rep = pm.evaluate_promotion(_good_replay(v_dd=-0.06, b_dd=-0.03),
                                forward_evidence=_forward())
    assert _status(rep, "max_drawdown") == "FAIL"


# --------------------------------------------------------------------------- #
# bootstrap helper
# --------------------------------------------------------------------------- #

def test_bootstrap_lb_thin_and_deterministic():
    assert pm._session_bootstrap_lb([], 100, 7) is None
    assert pm._session_bootstrap_lb([5.0], 100, 7) is None
    # constant series: every resampled mean equals the constant
    assert pm._session_bootstrap_lb([3.0, 3.0, 3.0], 200, 7) == pytest.approx(3.0)
    # seeded: same inputs -> same lower bound
    a = pm._session_bootstrap_lb([1.0, -2.0, 3.0, 0.5], 500, 7)
    b = pm._session_bootstrap_lb([1.0, -2.0, 3.0, 0.5], 500, 7)
    assert a == b


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_writes_report(tmp_path, capsys):
    rpath = tmp_path / "replay.json"
    rpath.write_text(json.dumps(_good_replay()), encoding="utf-8")
    out = tmp_path / "report.json"
    rc = pm.main(["--replay", str(rpath),
                  "--execution", str(tmp_path / "missing.json"),
                  "--out", str(out)])
    assert rc == 0
    rep = json.loads(out.read_text(encoding="utf-8"))
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert rep["execution_calibration"] is None
    printed = json.loads(capsys.readouterr().out)
    assert printed["verdict"] == "INSUFFICIENT_EVIDENCE"
