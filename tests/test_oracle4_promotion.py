"""Stage F tests: pre-registered promotion gates fed by ONE validated
immutable forward-evidence artifact (oracle.ml.evidence).

Replay numbers are development data and never satisfy forward gates; the
replay artifact contributes only invariants (accounting parity, demonstrated
holdout refusal) and context.
"""
import json
from datetime import date, timedelta

import pytest

from oracle.ml import evidence as ev
from oracle.ml import promotion as pm


# --------------------------------------------------------------------------- #
# fixture builders (synthetic, deterministic; no market data invented —
# these exercise gate plumbing and the evidence validator only)
# --------------------------------------------------------------------------- #

def _session_ids(n=80, start="2026-06-01"):
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _make_fe(*, n_sessions=80, daily_cycle=(60.0, 40.0, 55.0, 45.0),
             uplift=20.0,
             fill_cycle=((60.0, "live_order"), (-20.0, "live_order")),
             stress=None, calibration=None, coverage=None, **over):
    """A coherent forward-evidence artifact.  Defaults pass every gate."""
    sess = _session_ids(n_sessions)
    rows, fills = [], []
    for i, sid in enumerate(sess):
        ch = daily_cycle[i % len(daily_cycle)]
        rows.append({"session": sid,
                     "challenger_net_pnl": ch,
                     "baseline_net_pnl": ch - uplift,
                     "n_opportunities": 4,
                     "n_fills": len(fill_cycle)})
        for pnl, src in fill_cycle:
            fills.append({"session": sid, "source": src,
                          "net_pnl": pnl, "fees_known": True})
    doc = {
        "kind": ev.EVIDENCE_KIND,
        "schema_version": ev.SCHEMA_VERSION,
        "model_id": "oracle401-mlp",
        "model_hash": "modelhash-abc",
        "baseline_id": "baseline_frozen",
        "policy_id": "policy-v1",
        "dataset_fingerprint": "ds-fp-123",
        "code_git_sha": "deadbeef",
        "generated_utc": "2026-09-01T00:00:00Z",   # after the window closes
        "capital_base": 100000.0,
        "cost_assumptions": {"fee_per_contract": 0.65, "friction_mult": 1.0},
        "window": {"start": sess[0], "end": sess[-1]},
        "sessions": rows,
        "fills": fills,
        "live_fill_based": all(src == "live_order" for _, src in fill_cycle),
        "stress": stress if stress is not None else {
            "net_pnl_1_5x": 1200.0,
            "net_pnl_2_0x": 400.0,
            "stop_overshoot_net_pnl": 250.0,
        },
        "calibration": calibration if calibration is not None else {
            "challenger": {"brier": 0.20, "n": 160},
            "baseline": {"brier": 0.24, "n": 160},
        },
        "coverage": coverage if coverage is not None else {"groups": {
            "call": {"n_opportunities": 160, "n_predictions": 150},
            "put": {"n_opportunities": 160, "n_predictions": 152},
        }},
    }
    doc.update(over)
    return ev.sign(doc)


def _replay(parity_ok=True, holdout=True, mark_derived=40):
    doc = {
        "fingerprint": "test-replay-fp",
        "sessions_replayed": ["2026-11-04", "2026-11-05"],
        "parity": {"ok": parity_ok},
        "trade_universe": {"outcome_status_counts":
                           {"mark_derived": mark_derived}},
    }
    if holdout:
        doc["holdout_refusal"] = {"sessions": ["2026-11-12", "2026-11-13"],
                                  "decisions_excluded": 12}
    return doc


def _status(report, name):
    return next(x for x in report["gates"] if x["name"] == name)["status"]


FORWARD_GATES = (
    "forward_sessions", "resolved_opportunities", "actual_fills_effective_n",
    "expectancy_bootstrap_lb", "paired_uplift_bootstrap_lb",
    "net_positive_at_1_5x_friction", "stress_2x_and_overshoot_examined",
    "sharpe_daily_ann", "profit_factor", "max_drawdown",
    "calibration_improved", "coverage_disparity", "evidence_basis",
)
REPLAY_GATES = ("accounting_parity", "holdout_refusal_demonstrated")


# --------------------------------------------------------------------------- #
# verdict tiers
# --------------------------------------------------------------------------- #

def test_no_forward_evidence_is_insufficient_never_relaxed():
    rep = pm.evaluate_promotion(_replay())  # no forward evidence at all
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "never relaxed" in rep["verdict_reason"]
    # every forward gate is INSUFFICIENT; replay numbers never substitute
    for name in FORWARD_GATES:
        assert _status(rep, name) == "INSUFFICIENT"
    # replay-side invariants are still assessed
    assert _status(rep, "accounting_parity") == "PASS"
    assert _status(rep, "holdout_refusal_demonstrated") == "PASS"
    # replay session list reported as a count, for context only
    sess = next(x for x in rep["gates"] if x["name"] == "forward_sessions")
    assert sess["observed"] == 2
    assert "context" in sess["note"]
    # validation block is explicit about why
    val = rep["forward_evidence_validation"]
    assert val["present"] is False and val["valid"] is False
    assert any("no forward evidence artifact supplied" in e
               for e in val["errors"])
    # report metadata invariants
    assert rep["human_review_required"] is True
    assert rep["no_auto_promotion"] is True
    assert rep["gates_preregistered"] == pm.GATES_PREREGISTERED
    assert rep["replay_fingerprint"] == "test-replay-fp"
    assert rep["estimands"] is None
    assert isinstance(rep["fingerprint"], str) and len(rep["fingerprint"]) == 64


def test_sufficient_evidence_with_failing_perf_is_do_not_promote():
    fe = _make_fe(daily_cycle=(-60.0, -40.0, -55.0, -45.0), uplift=-20.0,
                  fill_cycle=((-60.0, "live_order"), (20.0, "live_order")),
                  stress={"net_pnl_1_5x": -500.0, "net_pnl_2_0x": -800.0,
                          "stop_overshoot_net_pnl": -900.0},
                  calibration={"challenger": {"brier": 0.26},
                               "baseline": {"brier": 0.24}})
    rep = pm.evaluate_promotion(_replay(), forward_evidence=fe)
    assert rep["forward_evidence_validation"]["valid"] is True
    assert rep["n_insufficient"] == 0
    assert rep["n_fail"] > 0
    assert rep["verdict"] == "DO_NOT_PROMOTE"
    for name in ("expectancy_bootstrap_lb", "paired_uplift_bootstrap_lb",
                 "net_positive_at_1_5x_friction", "sharpe_daily_ann",
                 "profit_factor", "max_drawdown", "calibration_improved"):
        assert _status(rep, name) == "FAIL"
    # both stresses were documented, so the examination gate still passes
    assert _status(rep, "stress_2x_and_overshoot_examined") == "PASS"


def test_all_pass_is_eligible_pending_human_review_only():
    rep = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    assert rep["forward_evidence_validation"]["valid"] is True
    assert rep["n_fail"] == 0
    assert rep["n_insufficient"] == 0
    assert rep["n_pass"] == len(FORWARD_GATES) + len(REPLAY_GATES)
    assert rep["verdict"] == "ELIGIBLE_PENDING_HUMAN_REVIEW"
    # there is no PROMOTE verdict and no side-effect fields
    assert rep["human_review_required"] is True
    assert rep["no_auto_promotion"] is True
    assert "registry.approve_model" in rep["note"]
    assert "verify_human_review" in rep["human_review_protocol"]


def test_positive_fixture_is_deterministic():
    a = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    b = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    # everything except the generation timestamp (and hence fingerprint)
    for k in ("verdict", "gates", "n_pass", "n_fail", "n_insufficient",
              "estimands"):
        assert a[k] == b[k]


# --------------------------------------------------------------------------- #
# estimands are registered and separated
# --------------------------------------------------------------------------- #

def test_estimands_per_opportunity_and_per_fill_are_separate():
    rep = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    est = rep["estimands"]
    assert est["per_session_mean"] == pytest.approx(50.0)
    # 80 sessions * 4 opportunities = 320; total pnl = 80 * 50 = 4000
    assert est["per_opportunity_expectancy"] == pytest.approx(4000.0 / 320.0)
    # 160 live fills: (80*60 - 80*20) / 160 = 20
    assert est["per_fill_expectancy"] == pytest.approx(20.0)
    assert "NOT interchangeable" in est["note"]
    assert est["session_ci_lb"] is not None and est["session_ci_lb"] > 0


def test_paired_uplift_reports_clustered_uncertainty():
    rep = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    obs = next(x for x in rep["gates"]
               if x["name"] == "paired_uplift_bootstrap_lb")["observed"]
    assert obs["n_sessions"] == 80
    assert obs["n_nonzero_delta_sessions"] == 80
    # constant |delta| => ESS equals n
    assert obs["effective_sample_size_abs_delta"] == pytest.approx(80.0)
    assert obs["ci_lo"] == pytest.approx(20.0)


def test_actual_fills_gate_reports_effective_sample_size():
    rep = pm.evaluate_promotion(_replay(), forward_evidence=_make_fe())
    obs = next(x for x in rep["gates"]
               if x["name"] == "actual_fills_effective_n")["observed"]
    assert obs["actual_fills"] == 160
    assert obs["sessions_with_live_fills"] == 80
    # 2 live fills per session, constant => ESS = number of sessions
    assert obs["effective_sample_size"] == pytest.approx(80.0)


# --------------------------------------------------------------------------- #
# replay-side invariants
# --------------------------------------------------------------------------- #

def test_parity_violation_fails_accounting_parity():
    rep = pm.evaluate_promotion(_replay(parity_ok=False),
                                forward_evidence=_make_fe())
    assert _status(rep, "accounting_parity") == "FAIL"
    assert rep["verdict"] == "DO_NOT_PROMOTE"


def test_missing_holdout_refusal_blocks_eligibility():
    rep = pm.evaluate_promotion(_replay(holdout=False),
                                forward_evidence=_make_fe())
    assert _status(rep, "holdout_refusal_demonstrated") == "INSUFFICIENT"
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_drawdown_worse_than_baseline_fails_despite_within_limit():
    # challenger loses 50/day; baseline loses 30/day (uplift = -20) =>
    # drawdowns both within the 20% limit but challenger is worse.
    fe = _make_fe(daily_cycle=(-60.0, -40.0, -55.0, -45.0), uplift=-20.0)
    rep = pm.evaluate_promotion(_replay(), forward_evidence=fe)
    gate = next(x for x in rep["gates"] if x["name"] == "max_drawdown")
    assert gate["status"] == "FAIL"
    assert abs(gate["observed"]["variant"]) <= 0.20
    assert "no_worse_than_baseline=False" in gate["note"]


# --------------------------------------------------------------------------- #
# bootstrap + drawdown helpers
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


def test_equity_max_drawdown():
    assert pm._equity_max_drawdown([], 1000.0) is None
    assert pm._equity_max_drawdown([1.0], 0.0) is None
    # monotone gains => zero drawdown
    assert pm._equity_max_drawdown([10.0, 10.0], 100.0) == 0.0
    # 100 -> 120 -> 90: trough 90 from peak 120 => -0.25
    assert pm._equity_max_drawdown([20.0, -30.0], 100.0) == pytest.approx(-0.25)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def test_cli_writes_report(tmp_path, capsys):
    rpath = tmp_path / "replay.json"
    rpath.write_text(json.dumps(_replay()), encoding="utf-8")
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
    assert printed["forward_evidence_valid"] is False


def test_cli_with_forward_evidence_and_expected(tmp_path, capsys):
    rpath = tmp_path / "replay.json"
    rpath.write_text(json.dumps(_replay()), encoding="utf-8")
    fpath = tmp_path / "fe.json"
    fpath.write_text(json.dumps(_make_fe()), encoding="utf-8")
    epath = tmp_path / "expected.json"
    epath.write_text(json.dumps({"model_hash": "modelhash-abc"}),
                     encoding="utf-8")
    out = tmp_path / "report.json"
    rc = pm.main(["--replay", str(rpath),
                  "--execution", str(tmp_path / "missing.json"),
                  "--forward-evidence", str(fpath),
                  "--expected", str(epath),
                  "--out", str(out)])
    assert rc == 0
    printed = json.loads(capsys.readouterr().out)
    assert printed["forward_evidence_valid"] is True
    assert printed["verdict"] == "ELIGIBLE_PENDING_HUMAN_REVIEW"
