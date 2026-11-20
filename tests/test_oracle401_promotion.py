"""Oracle 4.0.1 Stage F acceptance battery.

Negative tests required by the master prompt: mismatched model/hash/date,
forged summary-only evidence, mark/counterfactual mislabelled as fill,
missing holdout refusal, one stress missing, wrong comparator, malformed
types, NaN, inconsistent counts, stale reports — plus the prospective
holdout registry (refuse reuse after exposure) and explicit named human
review bound to a report fingerprint.  No paper-only result may qualify.
"""

from __future__ import annotations

from datetime import date, timedelta

import pytest

import oracle.ml.evidence as ev
import oracle.ml.promotion as pm


# --------------------------------------------------------------------------- #
# fixture builders (duplicated locally: tests/ has no package __init__)
# --------------------------------------------------------------------------- #

def _session_ids(n=80, start="2026-06-01"):
    d0 = date.fromisoformat(start)
    return [(d0 + timedelta(days=i)).isoformat() for i in range(n)]


def _make_fe(*, n_sessions=80, daily_cycle=(60.0, 40.0, 55.0, 45.0),
             uplift=20.0,
             fill_cycle=((60.0, "live_order"), (-20.0, "live_order")),
             stress=None, calibration=None, coverage=None, **over):
    sess = _session_ids(n_sessions)
    rows, fills = [], []
    for i, sid in enumerate(sess):
        ch = daily_cycle[i % len(daily_cycle)]
        rows.append({"session": sid, "challenger_net_pnl": ch,
                     "baseline_net_pnl": ch - uplift,
                     "n_opportunities": 4, "n_fills": len(fill_cycle)})
        for pnl, src in fill_cycle:
            fills.append({"session": sid, "source": src,
                          "net_pnl": pnl, "fees_known": True})
    doc = {"kind": ev.EVIDENCE_KIND, "schema_version": ev.SCHEMA_VERSION,
           "model_id": "oracle401-mlp", "model_hash": "modelhash-abc",
           "baseline_id": "baseline_frozen", "policy_id": "policy-v1",
           "dataset_fingerprint": "ds-fp-123", "code_git_sha": "deadbeef",
           "generated_utc": "2026-09-01T00:00:00Z",
           "capital_base": 100000.0,
           "cost_assumptions": {"fee_per_contract": 0.65, "friction_mult": 1.0},
           "window": {"start": sess[0], "end": sess[-1]},
           "sessions": rows, "fills": fills,
           "live_fill_based": all(src == "live_order" for _, src in fill_cycle),
           "stress": stress if stress is not None else {
               "net_pnl_1_5x": 1200.0, "net_pnl_2_0x": 400.0,
               "stop_overshoot_net_pnl": 250.0},
           "calibration": calibration if calibration is not None else {
               "challenger": {"brier": 0.20, "n": 160},
               "baseline": {"brier": 0.24, "n": 160}},
           "coverage": coverage if coverage is not None else {"groups": {
               "call": {"n_opportunities": 160, "n_predictions": 150},
               "put": {"n_opportunities": 160, "n_predictions": 152}}}}
    doc.update(over)
    return ev.sign(doc)


def _replay(parity_ok=True, holdout=True, mark_derived=40):
    doc = {"fingerprint": "test-replay-fp",
           "sessions_replayed": ["2026-11-04", "2026-11-05"],
           "parity": {"ok": parity_ok},
           "trade_universe": {"outcome_status_counts": {"mark_derived": mark_derived}}}
    if holdout:
        doc["holdout_refusal"] = {"sessions": ["2026-11-12", "2026-11-13"],
                                  "decisions_excluded": 12}
    return doc


def _gates(report):
    return {g["name"]: g for g in report["gates"]}


def _assert_err(errors, needle):
    assert any(needle in e for e in errors), f"{needle!r} not in {errors}"


# --------------------------------------------------------------------------- #
# validator: structural + binding negatives
# --------------------------------------------------------------------------- #

def test_valid_fixture_has_no_validation_errors():
    assert ev.validate_forward_evidence(_make_fe()) == []


def test_binding_mismatch_model_and_hash_rejected():
    fe = _make_fe()
    errs = ev.validate_forward_evidence(
        fe, expected={"model_hash": "other-hash", "model_id": "other-model"})
    _assert_err(errs, "binding mismatch: expected model_hash")
    _assert_err(errs, "binding mismatch: expected model_id")
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe,
                                expected={"model_hash": "other-hash"})
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert rep["forward_evidence_validation"]["valid"] is False


def test_binding_mismatch_window_dates_rejected():
    fe = _make_fe()
    errs = ev.validate_forward_evidence(
        fe, expected={"window_start": "2026-01-01", "window_end": "2026-01-31"})
    _assert_err(errs, "binding mismatch: expected window start")
    _assert_err(errs, "binding mismatch: expected window end")


def test_wrong_comparator_rejected():
    fe = _make_fe()  # carries baseline_id = baseline_frozen
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe,
                                baseline="spy_buy_hold")
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    _assert_err(rep["forward_evidence_validation"]["errors"],
                "binding mismatch: expected baseline_id")


def test_forged_summary_only_evidence_rejected():
    fe = _make_fe()
    del fe["sessions"]
    fe["forward_sessions"] = 80
    fe["resolved_opportunities"] = 320
    fe["actual_fills"] = 160
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe),
                "summary-only evidence rejected: per-session rows are required")
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe)
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_tampered_fingerprint_detected():
    fe = _make_fe()
    fe["model_id"] = "swapped-after-signing"  # keep stale fingerprint
    _assert_err(ev.validate_forward_evidence(fe),
                "fingerprint mismatch: evidence altered after signing or forged")


def test_mark_derived_mislabelled_as_actual_fill_rejected():
    fe = _make_fe()
    fe["fills"][0]["source"] = "mark_derived"
    fe["live_fill_based"] = True
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe), "live_fill_based overclaim")


def test_unknown_fill_source_rejected():
    fe = _make_fe()
    fe["fills"][0]["source"] = "guessed_pnl"
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe), "unknown fill source")


def test_broken_cohort_pairing_rejected():
    fe = _make_fe()
    del fe["sessions"][5]["baseline_net_pnl"]
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe), "cohort pairing broken")


def test_nan_pnl_rejected():
    fe = _make_fe()
    fe["sessions"][0]["challenger_net_pnl"] = float("nan")
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe),
                "challenger_net_pnl missing or non-finite")


def test_inconsistent_counts_rejected():
    fe = _make_fe()
    fe["actual_fills"] = 999  # top-level summary contradicts rows
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe),
                "inconsistent counts: actual_fills=999")

    fe2 = _make_fe()
    fe2["sessions"][0]["n_fills"] = 5  # rows contradict the fill list
    fe2 = ev.sign(fe2)
    _assert_err(ev.validate_forward_evidence(fe2),
                "inconsistent counts: sum(sessions[].n_fills)")


def test_stale_report_rejected():
    fe = _make_fe(generated_utc="2026-07-01T00:00:00Z")  # before window end
    _assert_err(ev.validate_forward_evidence(fe),
                "stale report: generated_utc precedes the evidence window end")


def test_out_of_order_sessions_rejected():
    fe = _make_fe()
    fe["sessions"][3], fe["sessions"][4] = fe["sessions"][4], fe["sessions"][3]
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe),
                "sessions out of chronological order")


def test_session_outside_window_rejected():
    fe = _make_fe()
    fe["sessions"][-1]["session"] = "2026-12-31"
    fe = ev.sign(fe)
    _assert_err(ev.validate_forward_evidence(fe), "outside declared window")


def test_missing_required_field_rejected():
    fe = _make_fe(model_hash="")
    _assert_err(ev.validate_forward_evidence(fe),
                "missing or empty required field: model_hash")


def test_bad_capital_base_rejected():
    fe = _make_fe(capital_base=0)
    _assert_err(ev.validate_forward_evidence(fe),
                "capital_base must be a finite positive number")


# --------------------------------------------------------------------------- #
# promotion-level: adequacy, basis and stress negatives
# --------------------------------------------------------------------------- #

def test_one_stress_missing_is_insufficient_not_pass():
    fe = _make_fe(stress={"net_pnl_1_5x": 1200.0, "net_pnl_2_0x": 400.0})
    assert ev.validate_forward_evidence(fe) == []  # structurally valid
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe)
    g = _gates(rep)["stress_2x_and_overshoot_examined"]
    assert g["status"] == "INSUFFICIENT"
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"


def test_paper_only_can_never_qualify_for_live_promotion():
    fe = _make_fe(fill_cycle=((60.0, "paper_order"), (-20.0, "paper_order")))
    assert ev.validate_forward_evidence(fe) == []  # honest paper evidence
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe)
    g = _gates(rep)
    assert g["evidence_basis"]["status"] == "FAIL"
    assert g["actual_fills_effective_n"]["status"] == "INSUFFICIENT"
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert rep["verdict"] != "ELIGIBLE_PENDING_HUMAN_REVIEW"


def test_invalid_evidence_reports_known_failures():
    fe = _make_fe()
    fe["model_id"] = "tampered"  # stale fingerprint
    rep = pm.evaluate_promotion(_replay(), None, forward_evidence=fe)
    val = rep["forward_evidence_validation"]
    assert val["present"] is True and val["valid"] is False
    assert any("fingerprint mismatch" in e for e in val["errors"])
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"


# --------------------------------------------------------------------------- #
# units: kish ESS and canonical signing
# --------------------------------------------------------------------------- #

def test_kish_ess_units():
    assert ev.kish_ess([]) == 0.0
    assert ev.kish_ess([2, 2]) == pytest.approx(2.0)
    assert ev.kish_ess([1, 0, 0]) == pytest.approx(1.0)
    assert ev.kish_ess([3, 1]) == pytest.approx(1.6)


def test_sign_and_fingerprint_deterministic():
    a = _make_fe()
    b = _make_fe()
    assert a["fingerprint"] == b["fingerprint"]
    assert ev.canonical_fingerprint(a) == a["fingerprint"]


# --------------------------------------------------------------------------- #
# prospective holdout registry
# --------------------------------------------------------------------------- #

def test_holdout_register_and_usable(tmp_path):
    p = tmp_path / "holdout.json"
    doc = ev.register_prospective_holdout(
        p, ["2026-12-02", "2026-12-01"], code_git_sha="deadbeef",
        artifact_hashes={"dataset": "ds-fp-123"}, registered_by="Yomi",
        development_end="2026-11-19")
    assert p.exists()
    assert doc["sessions"] == ["2026-12-01", "2026-12-02"]  # sorted, deduped
    assert ev.check_holdout_usable(ev.load_holdout(p)) == []


def test_holdout_rejects_development_sessions(tmp_path):
    with pytest.raises(ValueError, match="postdate development_end"):
        ev.register_prospective_holdout(
            tmp_path / "h.json", ["2026-11-01"], code_git_sha="x",
            artifact_hashes={}, registered_by="Yomi",
            development_end="2026-11-19")


def test_holdout_rejects_empty_and_anonymous(tmp_path):
    with pytest.raises(ValueError, match="at least one session"):
        ev.register_prospective_holdout(
            tmp_path / "h.json", [], code_git_sha="x",
            artifact_hashes={}, registered_by="Yomi",
            development_end="2026-11-19")
    with pytest.raises(ValueError, match="registered_by"):
        ev.register_prospective_holdout(
            tmp_path / "h.json", ["2026-12-01"], code_git_sha="x",
            artifact_hashes={}, registered_by="",
            development_end="2026-11-19")


def test_holdout_refuses_reuse_after_exposure(tmp_path):
    p = tmp_path / "holdout.json"
    ev.register_prospective_holdout(
        p, ["2026-12-01"], code_git_sha="x", artifact_hashes={},
        registered_by="Yomi", development_end="2026-11-19")
    ev.mark_holdout_exposed(p, reason="reviewed replay results")
    errs = ev.check_holdout_usable(ev.load_holdout(p))
    _assert_err(errs, "refuse reuse after exposure")


def test_holdout_tamper_detected(tmp_path):
    p = tmp_path / "holdout.json"
    ev.register_prospective_holdout(
        p, ["2026-12-01"], code_git_sha="x", artifact_hashes={},
        registered_by="Yomi", development_end="2026-11-19")
    reg = ev.load_holdout(p)
    reg["sessions"] = ["2026-12-01", "2026-12-05"]  # mutate after signing
    _assert_err(ev.check_holdout_usable(reg),
                "holdout registration fingerprint mismatch: tampered")


# --------------------------------------------------------------------------- #
# explicit named human review, bound to the report fingerprint
# --------------------------------------------------------------------------- #

def _eligible_report():
    return pm.evaluate_promotion(_replay(), None, forward_evidence=_make_fe())


def test_human_review_good_binding_passes():
    rep = _eligible_report()
    review = {"reviewer_name": "Ada Lovelace", "decision": "approved",
              "report_fingerprint": rep["fingerprint"],
              "report_schema_version": rep["schema_version"]}
    assert ev.verify_human_review(rep, review) == []


def test_bare_approved_by_string_is_not_sufficient():
    rep = _eligible_report()
    errs = ev.verify_human_review(
        rep, {"approved_by": "ops", "decision": "approved",
              "report_fingerprint": rep["fingerprint"]})
    _assert_err(errs, "a bare approved_by string is not sufficient")


def test_review_requires_name_decision_and_binding():
    rep = _eligible_report()
    _assert_err(ev.verify_human_review(rep, {}), "review missing reviewer_name")
    _assert_err(ev.verify_human_review(
        rep, {"reviewer_name": "Ada", "decision": "lgtm",
              "report_fingerprint": rep["fingerprint"]}),
        "review decision must be 'approved'")
    _assert_err(ev.verify_human_review(
        rep, {"reviewer_name": "Ada", "decision": "approved",
              "report_fingerprint": "f" * 64}),
        "report_fingerprint mismatch")
    _assert_err(ev.verify_human_review(
        rep, {"reviewer_name": "Ada", "decision": "approved",
              "report_fingerprint": rep["fingerprint"],
              "report_schema_version": 1}),
        "different report schema version")
    _assert_err(ev.verify_human_review(rep, "approved"),
                "review must be a JSON object")
    _assert_err(ev.verify_human_review({}, {"reviewer_name": "Ada"}),
                "promotion report missing a fingerprint")
