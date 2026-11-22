"""Oracle 4.0.2 P0-4: competing EV models + calibrated executable EV.

Acceptance covered here:
* extraction mirrors Stage-D gates (units, timestamps) and reconstructs
  GROSS EV for pre-v3 stamps (net + costs) -- never unit-coerced
* feature matrix is fixed-order with NaN for missing values
* models fit on train sessions only; leaked splits raise
* a planted entry-time signal is detected out-of-sample (LightGBM)
* pure noise yields NO_VALIDATED_EDGE (abstain) -- never a fabricated edge
* too little evidence yields INSUFFICIENT_EVIDENCE (abstain)
* executable EV = P(fill) * E[net|fill] with no-double-count cost handling
* P(fill) is the conservative shrunk prior, labelled (survivorship)
* missing optional dependencies produce explicit `unavailable` entries
* the study runner is deterministic and writes a reloadable report
"""
import json
import math
import random
from datetime import date, timedelta

import pytest

from oracle.ml import ev_models as em
from oracle.ml import pnl_calibration as pc
from oracle.ml.execution import shrunk_fill_probability

EXPIRY = "261219"  # 2026-12-19


def _occ(und, side, strike, exp=EXPIRY):
    return f"{und}{exp}{side}{int(strike * 1000):08d}"


def make_trades(n_sessions=24, per_session=25, signal=True, seed=3):
    """Synthetic resolved history with an optional planted entry signal."""
    rng = random.Random(seed)
    start = date(2026, 10, 1)
    trades = []
    for d in range(n_sessions):
        day = (start + timedelta(days=d)).isoformat()
        regime = "trend" if d % 2 == 0 else "chop"
        for j in range(per_session):
            und = "SPY" if j % 2 == 0 else "QQQ"
            side = "C" if j % 3 else "P"
            s = float(rng.randint(0, 5))
            if signal:
                pnl = 40.0 * s - 90.0 + rng.gauss(0.0, 25.0)
            else:
                pnl = rng.gauss(0.0, 50.0)
            trades.append({
                "symbol": _occ(und, side, 600 + j),
                "entry_time": f"{day}T15:{j:02d}:00Z",
                "pnl": round(pnl, 2),
                "probability_of_profit": 0.45 + 0.1 * rng.random(),
                "expected_value": 10.0,
                "regime_label": regime,
                "metrics": {
                    "ev_units": "usd_total",
                    "gross_expected_value": 15.0,
                    "round_trip_costs": 5.0,
                    "max_loss": 500.0,
                    "ev_per_dollar_risk": 0.02,
                    "signal_strength": s,
                    "entry_delta": 0.3 + 0.1 * rng.random(),
                    "take_profit_pct": 0.25,
                    "stop_loss_pct": 0.15,
                    "quantity": 1,
                },
            })
    return trades


def _write_history(tmp_path, trades, name="hist.json"):
    p = tmp_path / name
    p.write_text(json.dumps({"trades": trades}), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# extraction + features
# --------------------------------------------------------------------------- #

def test_extract_opportunities_gates_and_gross_reconstruction():
    good = make_trades(1, 1)[0]
    pre_v3 = make_trades(1, 1)[0]
    del pre_v3["metrics"]["gross_expected_value"]  # reconstruct = net + costs
    bad_units = make_trades(1, 1)[0]
    bad_units["metrics"]["ev_units"] = "usd_per_share"
    bad_ts = make_trades(1, 1)[0]
    bad_ts["entry_time"] = "not-a-time"
    missing = make_trades(1, 1)[0]
    del missing["pnl"]

    opps, rep = em.extract_opportunities(
        [good, pre_v3, bad_units, bad_ts, missing])
    assert rep == {"rows_scanned": 5, "kept": 2,
                   "dropped_missing_fields": 1,
                   "dropped_unit_mismatch": 1,
                   "dropped_bad_timestamp": 1}
    assert len(opps) == 2
    for o in opps:
        f = o["features"]
        assert f["gross_ev"] == pytest.approx(15.0)  # 10 net + 5 costs
        assert f["signal_strength"] is not None
        assert f["side_is_call"] in (0.0, 1.0)
        assert o["provenance"] == "paper_marks"
        assert o["dte"] is not None and o["dte"] > 0
        assert o["regime"] in ("trend", "chop")


def test_feature_matrix_order_and_nan_for_missing():
    opps, _ = em.extract_opportunities(make_trades(1, 2))
    del opps[0]["features"]["entry_delta"]
    opps[0]["features"]["max_loss"] = None
    opps[0]["features"]["signal_strength"] = True  # bool is not a number
    X = em.feature_matrix(opps)
    assert len(X) == 2 and all(len(r) == len(em.FEATURE_NAMES) for r in X)
    row = dict(zip(em.FEATURE_NAMES, X[0]))
    assert math.isnan(row["entry_delta"])
    assert math.isnan(row["max_loss"])
    assert math.isnan(row["signal_strength"])
    assert row["p_pred"] == opps[0]["p_pred"]
    assert not any(math.isnan(v) for v in X[1])


# --------------------------------------------------------------------------- #
# evaluation: leak guard + structure
# --------------------------------------------------------------------------- #

def test_evaluate_models_structure_and_leak_guard():
    opps, _ = em.extract_opportunities(make_trades())
    train, eval_, _meta = pc.chronological_split(opps, 0.25)
    m = em.evaluate_models(train, eval_)
    assert m["n_train"] == len(train) and m["n_eval"] == len(eval_)
    for name in ("stamped_pop", "base_rate", "logistic_recal", "cohort_pop"):
        assert m["pop_models"][name]["status"] == "ok"
        assert 0.0 <= m["pop_models"][name]["brier"] <= 1.0
    for name in ("stamped_ev", "cohort_profit"):
        assert m["profit_models"][name]["status"] == "ok"
    # stamped_ev is constant -> no rank power, never validated
    assert m["profit_models"]["stamped_ev"]["verdict"]["validated"] is False
    assert m["per_fill_note"] == em.SURVIVORSHIP_NOTE

    # leaked split: eval session not strictly after train raises
    with pytest.raises(ValueError):
        em.evaluate_models(train, train[-5:])


@pytest.mark.skipif(not em._HAS_LGBM, reason="lightgbm not installed")
def test_planted_signal_is_detected_out_of_sample():
    opps, _ = em.extract_opportunities(make_trades(signal=True))
    train, eval_, _ = pc.chronological_split(opps, 0.25)
    m = em.evaluate_models(train, eval_)
    prof = m["profit_models"]["lgbm_profit"]
    assert prof["status"] == "ok"
    assert prof["verdict"]["validated"] is True
    assert prof["spearman"] > 0.3
    assert prof["session_bootstrap"]["lower_bound"] > 0.0
    assert prof["ols_slope_realized_on_pred"] > 0.0
    # the learned PoP model beats the uninformative base rate
    assert (m["pop_models"]["lgbm_pop"]["brier"]
            < m["pop_models"]["base_rate"]["brier"])
    # quantile block reports pinball + interval coverage
    q = m["quantile_model"]
    assert q["status"] == "ok"
    assert set(q["pinball_usd"]) == {"q10", "q50", "q90"}
    assert 0.0 <= q["interval_coverage_q10_q90"] <= 1.0
    verdict = em._study_verdict(m)
    assert verdict["verdict"] == "VALIDATED_RANK_POWER"
    assert verdict["challenger_policy"] == "research_only"
    assert verdict["best_model"] in m["profit_models"]


def test_pure_noise_declares_no_validated_edge(tmp_path):
    hist = _write_history(tmp_path, make_trades(signal=False, seed=9))
    rep = em.run_ev_model_study(hist)
    assert rep["verdict"] == "NO_VALIDATED_EDGE"
    assert rep["challenger_policy"] == "abstain"
    assert rep["best_model"] is None
    exe = rep["executable_ev"]
    assert exe["e_net_model"] is None
    assert exe["eval_mean_executable_ev_usd"] is None
    assert "NOT composed" in exe["note"]


def test_tiny_history_is_insufficient_evidence(tmp_path):
    hist = _write_history(tmp_path, make_trades(n_sessions=3, per_session=5))
    rep = em.run_ev_model_study(hist)
    assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert rep["challenger_policy"] == "abstain"

    missing = em.run_ev_model_study(str(tmp_path / "nope.json"))
    assert missing["verdict"] == "INSUFFICIENT_EVIDENCE"
    assert "history unavailable" in missing["error"]


# --------------------------------------------------------------------------- #
# executable EV composition + fill probability honesty
# --------------------------------------------------------------------------- #

def test_executable_ev_from_model_product_and_validation():
    out = em.executable_ev_from_model(0.8, -12.5)
    assert out["executable_ev_usd"] == pytest.approx(0.8 * -12.5)
    assert out["cost_handling"] == em.COST_HANDLING_NOTE
    for bad_p in (-0.1, 1.1, float("nan"), float("inf"), True, None, "0.5"):
        with pytest.raises(ValueError):
            em.executable_ev_from_model(bad_p, 1.0)
    for bad_e in (float("nan"), float("inf"), True, None, "1"):
        with pytest.raises(ValueError):
            em.executable_ev_from_model(0.5, bad_e)


def test_fill_probability_is_labelled_conservative_prior():
    blk = em._fill_probability_block()
    assert blk["value"] == pytest.approx(shrunk_fill_probability(0, 0))
    assert blk["value"] == pytest.approx(0.80)
    assert blk["basis"] == "conservative_prior_shrunk"
    assert blk["n_nonfill_evidence"] == 0
    assert "survivorship" in blk["survivorship_note"]


# --------------------------------------------------------------------------- #
# optional-dependency honesty
# --------------------------------------------------------------------------- #

def test_missing_dependencies_yield_unavailable_entries(monkeypatch):
    monkeypatch.setattr(em, "_HAS_LGBM", False)
    monkeypatch.setattr(em, "_HAS_TORCH", False)
    opps, _ = em.extract_opportunities(make_trades())
    train, eval_, _ = pc.chronological_split(opps, 0.25)
    m = em.evaluate_models(train, eval_)
    assert m["pop_models"]["lgbm_pop"] == {
        "status": "unavailable", "reason": "LIGHTGBM_UNAVAILABLE"}
    for name in ("lgbm_profit", "lgbm_quantile_q50"):
        assert m["profit_models"][name] == {
            "status": "unavailable", "reason": "LIGHTGBM_UNAVAILABLE"}
    assert m["profit_models"]["neural_profit"] == {
        "status": "unavailable", "reason": "TORCH_UNAVAILABLE"}
    assert m["quantile_model"]["status"] == "unavailable"
    # pure-python models still score and a verdict is still computed
    assert m["pop_models"]["base_rate"]["status"] == "ok"
    assert m["profit_models"]["cohort_profit"]["status"] == "ok"
    assert em._study_verdict(m)["verdict"] in (
        "NO_VALIDATED_EDGE", "VALIDATED_RANK_POWER")


# --------------------------------------------------------------------------- #
# runner: determinism + artifact
# --------------------------------------------------------------------------- #

def test_study_runner_deterministic_and_writes_report(tmp_path):
    hist = _write_history(tmp_path, make_trades())
    out = tmp_path / "artifacts" / "ev_model_report.json"
    rep1 = em.run_ev_model_study(hist, out_path=str(out))
    rep2 = em.run_ev_model_study(hist)
    for r in (rep1, rep2):
        r.pop("generated_utc")
    assert rep1 == rep2

    with open(out, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert reloaded["inputs"]["history"]["sha256"]
    pre = reloaded["pre_registered"]
    assert pre["seed"] == em.SEED
    assert pre["min_eval_n"] == em.MIN_EVAL_N
    assert pre["fill_prior"] == pytest.approx(0.80)
    assert reloaded["executable_ev"]["formula"] == \
        "executable_EV = P(fill) * E[net_profit | fill]"
    assert reloaded["metrics"]["n_eval"] >= em.MIN_EVAL_N
    assert reloaded["split"] == rep1["split"]
