"""Acceptance tests for oracle/ml/backtest402.py (Oracle 4.0.2 P10).

Hermetic: reuses the replay test fixture (written through the production
``_write_store``) so every loader runs against the real on-disk schema.
GBM-dependent tests are skipped when lightgbm is not importable; they
assert MECHANICS (fold accounting, probability simplex, coverage), never
prediction quality -- LightGBM's min_data_in_leaf on a tiny fixture can
legitimately yield near-prior probabilities.
"""
from __future__ import annotations

import dataclasses
import json
import os
import sys

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.ml import backtest402 as bt  # noqa: E402
from oracle.ml import gbm as gbm_mod  # noqa: E402
from oracle.ml import replay as rp  # noqa: E402
from tests.test_oracle4_replay import (  # noqa: E402
    HOLDOUT,
    SESSIONS,
    _occ,
    _trade,
    _write_fixture,
    _write_valuation,
)

N_TRADES = 14 * len(SESSIONS)           # 70 (fixture arithmetic)
COHORT_NO_PRED = 14 * 2                 # first two sessions never predicted
COHORT_QUOTE_INELIGIBLE = 2 * 3         # i==1 (wide) + i==2 (no quote) x 3
COHORT_N = N_TRADES - COHORT_NO_PRED - COHORT_QUOTE_INELIGIBLE  # 36

ALL_VARIANTS = {
    "baseline_frozen", "baseline_quality", "valuation_ranked",
    "classical_gated", "neural_gated", "combined",
    "gbm_gated", "ensemble_gated", "gbm_combined",
}


def _selected_ids():
    return [f"d-{s}-0" for s in SESSIONS]


# --------------------------------------------------------------------------- #
# pure helpers (no model, no store)
# --------------------------------------------------------------------------- #

class TestEnsemblePreds:
    def test_intersection_only_and_renormalized_mean(self):
        classical = {"a": (0.2, 0.2, 0.6), "b": (0.5, 0.3, 0.2)}
        gbm = {"a": (0.4, 0.2, 0.4), "c": (1.0, 0.0, 0.0)}
        out = bt.ensemble_preds(classical, gbm)
        assert set(out) == {"a"}
        assert out["a"][0] == pytest.approx(0.3)
        assert out["a"][1] == pytest.approx(0.2)
        assert out["a"][2] == pytest.approx(0.5)
        assert sum(out["a"]) == pytest.approx(1.0)

    def test_zero_mass_dropped(self):
        out = bt.ensemble_preds({"a": (0.0, 0.0, 0.0)},
                                {"a": (0.0, 0.0, 0.0)})
        assert out == {}

    def test_empty_inputs(self):
        assert bt.ensemble_preds({}, {"a": (0.1, 0.2, 0.7)}) == {}


class TestGateTrades:
    def test_accounting_and_passthrough(self):
        t_pass = _trade("pass", SESSIONS[0])
        t_block = _trade("block", SESSIONS[0], minute=1)
        t_abstain = _trade("abstain", SESSIONS[0], minute=2)
        t_nopred = _trade("nopred", SESSIONS[0], minute=3)
        preds = {
            "pass": (0.1, 0.2, 0.7),      # call, up argmax -> pass
            "block": (0.7, 0.2, 0.1),     # call, down argmax -> gated_out
            "abstain": (0.32, 0.30, 0.38),  # edge below margin -> abstain
        }
        kept, acct = bt.gate_trades(
            [t_pass, t_block, t_abstain, t_nopred], preds)
        assert acct == {"gated_out": 1, "no_prediction": 1,
                        "gate_passed": 1, "abstained": 1}
        # no-prediction trades are KEPT (gate only acts where evidence is)
        assert [t.decision_id for t in kept] == ["pass", "nopred"]


class TestDteBucket:
    def _with_symbol(self, symbol):
        return dataclasses.replace(_trade("x", SESSIONS[0]), symbol=symbol)

    def test_buckets_from_occ_expiration(self):
        # entry date 2026-01-05
        assert bt._dte_bucket(
            self._with_symbol("SPY260105C00500000")) == "0dte"
        assert bt._dte_bucket(
            self._with_symbol("SPY260108C00500000")) == "1-7d"
        assert bt._dte_bucket(
            self._with_symbol("SPY260125C00500000")) == "8-30d"
        assert bt._dte_bucket(self._with_symbol(_occ("call"))) == ">30d"

    def test_unknown_when_unparseable(self):
        assert bt._dte_bucket(self._with_symbol(None)) == "unknown"
        assert bt._dte_bucket(self._with_symbol("NOTANOCC")) == "unknown"


class TestComputeBreakdowns:
    def test_hand_computed_single_trade(self):
        t = _trade("a", SESSIONS[0])        # call, premium 1000, +10% -> +100
        b = bt.compute_breakdowns([t], [SESSIONS[0]], one_way_frac=0.0)
        cell = b["by_underlying"]["SPY"]
        assert cell["n_executed"] == 1
        assert cell["net_pnl"] == pytest.approx(100.0)
        assert cell["win_rate"] == 1.0
        assert cell["profit_factor"] is None
        assert set(b["by_direction"]) == {"call"}
        assert set(b["by_dte_bucket"]) == {"unknown"}    # symbol=None
        assert set(b["by_exit_reason"]) == {"take_profit"}
        assert b["by_regime"] == {"status": "unavailable",
                                  "why": bt.REGIME_NOTE}
        assert b["note"] == bt.BREAKDOWN_NOTE

    def test_cell_keys_are_slim(self):
        t = _trade("a", SESSIONS[0])
        b = bt.compute_breakdowns([t], [SESSIONS[0]], one_way_frac=0.0)
        assert set(b["by_underlying"]["SPY"]) == {
            "n_executed", "net_pnl", "win_rate",
            "expectancy_per_trade", "profit_factor"}


class TestBuildVariants402:
    def test_gbm_unavailable_stubs(self):
        trades = [_trade("a", SESSIONS[0])]
        v = bt.build_variants_402(trades, None, {}, {}, "unavailable",
                                  {}, "unavailable")
        # frozen replay semantics: no valuation artifact -> the
        # valuation_ranked / combined keys are omitted entirely.
        assert set(v) == ALL_VARIANTS - {"valuation_ranked", "combined"}
        for name in ("gbm_gated", "ensemble_gated", "gbm_combined"):
            assert v[name] == {"trades": None,
                               "accounting": {"status": "unavailable"}}

    def test_gbm_ok_without_valuation(self):
        trades = [_trade("a", SESSIONS[0])]
        v = bt.build_variants_402(trades, None, {}, {}, "unavailable",
                                  {}, "ok")
        acct = v["gbm_gated"]["accounting"]
        assert acct["n"] == 1 and acct["no_prediction"] == 1
        assert v["ensemble_gated"]["accounting"]["ensemble_covered"] == 0
        assert v["ensemble_gated"]["accounting"]["members"] == [
            "multinomial_logistic", "lightgbm_direction"]
        assert v["gbm_combined"]["trades"] is None
        assert v["gbm_combined"]["accounting"]["why"] == \
            "no valuation selection artifact"

    def test_gbm_ok_with_valuation_intersection(self):
        t1 = _trade("a", SESSIONS[0])
        t2 = _trade("b", SESSIONS[0], minute=1)
        v = bt.build_variants_402([t1, t2], {"a"}, {}, {}, "unavailable",
                                  {"b": (0.1, 0.2, 0.7)}, "ok")
        # gbm gate keeps both (a=no_prediction passthrough, b=pass);
        # intersection with valuation {a} keeps only a.
        kept = [t.decision_id for t in v["gbm_combined"]["trades"]]
        assert kept == ["a"]
        acct = v["gbm_combined"]["accounting"]
        assert acct["n"] == 1
        assert acct["and_valuation_selected"] is True
        assert acct["note"] == rp.COUNTERFACTUAL_NOTE


# --------------------------------------------------------------------------- #
# end-to-end (classical only: deterministic, fast)
# --------------------------------------------------------------------------- #

@pytest.fixture()
def store_and_val(tmp_path):
    store = str(tmp_path / "ds.sqlite")
    _write_fixture(store)
    val = str(tmp_path / "val.json")
    _write_valuation(val, _selected_ids())
    return store, val


def _run(store, val, out=None, **kw):
    kw.setdefault("n_boot", 25)
    kw.setdefault("include_neural", False)
    kw.setdefault("include_gbm", False)
    return bt.run_backtest402(store, valuation_path=val, out_path=out,
                              holdout=tuple(HOLDOUT), **kw)


class TestRunBacktest402:
    def test_full_artifact_no_gbm_no_neural(self, store_and_val, tmp_path):
        store, val = store_and_val
        out = str(tmp_path / "replay402.json")
        art = _run(store, val, out=out)

        # parity: engine vs independent per-trade arithmetic
        p = art["parity"]
        assert p["ok"] is True
        assert p["n_trades_engine"] == N_TRADES
        assert p["max_abs_diff"] < 1e-6

        # all 9 variants present; disabled challengers are honest stubs
        assert set(art["variants"]) == ALL_VARIANTS
        for name in ("neural_gated", "gbm_gated", "ensemble_gated",
                     "gbm_combined"):
            assert art["variants"][name]["status"] == "unavailable"

        # friction sweep: monotone, equity_curve stripped, exit-days kept
        base = art["variants"]["baseline_frozen"]
        assert set(base["by_friction_mult"]) == {"1.0", "1.5", "2.0"}
        nets = [base["by_friction_mult"][k]["net_pnl"]
                for k in ("1.0", "1.5", "2.0")]
        assert nets[0] > nets[1] > nets[2]
        for cell in base["by_friction_mult"].values():
            assert "equity_curve" not in cell
            assert "pnl_by_exit_day" in cell

        # stop-overshoot stress always hurts (fixture has stop exits)
        stress = base["stop_overshoot_stress"]
        assert stress["extra_pts"] == pytest.approx(10.0)
        assert stress["delta_vs_unstressed"] < 0

        # paired bootstrap for every ok non-baseline variant
        assert set(art["paired_vs_baseline"]) == {
            "baseline_quality", "valuation_ranked", "classical_gated",
            "combined"}
        for entry in art["paired_vs_baseline"].values():
            bs = entry["bootstrap"]
            assert {"mean", "ci_lo", "ci_hi"} <= set(bs)
            assert entry["n_nonzero_delta_sessions"] >= 0

        # matched cohorts: classical ok with fixture arithmetic; gbm stub
        coh = art["matched_cohorts"]["classical"]
        assert coh["status"] == "ok"
        assert coh["n_cohort"] == COHORT_N
        assert coh["excluded_no_prediction"] == COHORT_NO_PRED
        assert coh["excluded_quote_ineligible"] == COHORT_QUOTE_INELIGIBLE
        assert sum(coh["challenger_gate"].values()) == COHORT_N
        assert coh["challenger_gate"]["gated_out"] == 0   # separable fixture
        slim = {"n_executed", "net_pnl", "profit_factor",
                "sharpe_daily_ann", "max_drawdown"}
        assert set(coh["baseline_on_cohort"]) == slim
        assert set(coh["challenger_on_cohort"]) == slim
        gbm_coh = art["matched_cohorts"]["gbm"]
        assert gbm_coh["status"] == "unavailable"
        assert gbm_coh["why"] == "disabled by --no-gbm"

        # honesty metadata
        assert art["holdout_refusal"]["decisions_excluded"] == 15
        assert art["holdout_refusal"]["sessions"] == list(HOLDOUT)
        mda = art["missing_data_accounting"]
        assert mda["classical_no_prediction"] == COHORT_NO_PRED
        assert mda["gbm_no_prediction"] == N_TRADES     # disabled -> no preds
        assert art["oof_gates"]["gbm"]["why"] == "disabled by --no-gbm"
        assert art["oof_gates"]["neural"]["why"] == "disabled by --no-neural"
        cal = art["session_calendar"]
        assert cal["n_sessions"] == len(SESSIONS)
        assert cal["no_trade_sessions"] == []
        rml = art["risk_metrics_limited"]
        assert rml["limited"] is True
        assert any("fewer than 30" in r for r in rml["reasons"])
        assert "entry-before-exit" in art["event_ordering_note"]
        assert art["breakdown_note"] == bt.BREAKDOWN_NOTE
        assert art["regime_note"] == bt.REGIME_NOTE

        # breakdowns per ok variant, fixture-shaped
        ok_names = {n for n, r in art["variants"].items()
                    if r["status"] == "ok"}
        assert set(art["breakdowns"]) == ok_names
        bd = art["breakdowns"]["baseline_frozen"]
        assert set(bd["by_underlying"]) == {"SPY"}
        assert set(bd["by_direction"]) == {"call", "put"}
        assert set(bd["by_dte_bucket"]) == {">30d"}     # exp 2026-02-20
        assert set(bd["by_exit_reason"]) == {"take_profit",
                                             "dynamic_stop_loss"}

        # artifact written and round-trips with its fingerprint
        with open(out, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["fingerprint"] == art["fingerprint"]
        assert on_disk["phase"] == 10

    def test_fingerprint_deterministic(self, store_and_val):
        store, val = store_and_val
        a = _run(store, val)
        b = _run(store, val)
        assert a["fingerprint"] == b["fingerprint"]

    def test_crosscheck_refuses_mismatched_valuation(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        val = str(tmp_path / "bad_val.json")
        with open(val, "w", encoding="utf-8") as fh:
            json.dump({"report_fingerprint": "valfp",
                       "dataset_hash": "WRONG-DATASET",
                       "result": {"scans": []}}, fh)
        with pytest.raises(ValueError, match="refusing to mix artifacts"):
            _run(store, val)

    def test_runs_without_valuation(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        art = _run(store, None)
        # frozen replay semantics: variants needing the valuation artifact
        # are omitted (not stubbed) when it is absent.
        assert "valuation_ranked" not in art["variants"]
        assert "combined" not in art["variants"]
        assert art["variants"]["gbm_combined"]["accounting"][
            "status"] == "unavailable"
        assert art["parity"]["ok"] is True


# --------------------------------------------------------------------------- #
# GBM challenger (mechanics only; skipped when lightgbm missing)
# --------------------------------------------------------------------------- #

needs_lgbm = pytest.mark.skipif(
    not gbm_mod.LGBM_AVAILABLE,
    reason=gbm_mod.LGBM_WHY or "lightgbm missing")


@needs_lgbm
class TestComputeOofGbm:
    def test_mechanics_and_coverage(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        preds, rep = bt.compute_oof_gbm(store, holdout=HOLDOUT)
        assert rep["status"] == "ok"
        assert rep["model"] == "lightgbm_direction"
        assert rep["missingness_mode"] == "native"
        assert rep["lgbm_version"] == gbm_mod.LGBM_VERSION
        assert rep["classes"] == ["down", "neutral", "up"]
        assert rep["folds_total"] == rep["folds_used"] + rep["folds_skipped"]
        assert rep["folds_used"] >= 1
        assert rep["predictions"] == len(preds) > 0
        for p in preds.values():
            assert all(0.0 <= x <= 1.0 for x in p)
            assert sum(p) == pytest.approx(1.0, abs=1e-6)
        # identical purged-fold construction => identical coverage
        classical, _ = rp.compute_oof_classical(store, holdout=HOLDOUT)
        assert set(preds) == set(classical)

    def test_unavailable_stub_shape(self, monkeypatch, tmp_path):
        monkeypatch.setattr(gbm_mod, "LGBM_AVAILABLE", False)
        monkeypatch.setattr(gbm_mod, "LGBM_WHY", "forced off for test")
        preds, rep = bt.compute_oof_gbm(str(tmp_path / "missing.sqlite"))
        assert preds == {}
        assert rep == {"model": "lightgbm_direction",
                       "status": "unavailable",
                       "why": "forced off for test"}


@needs_lgbm
class TestEndToEndWithGbm:
    def test_gbm_variants_and_cohort(self, store_and_val):
        store, val = store_and_val
        art = _run(store, val, include_gbm=True)
        assert art["oof_gates"]["gbm"]["status"] == "ok"

        g = art["variants"]["gbm_gated"]
        assert g["status"] == "ok"
        acct = g["accounting"]
        assert (acct["gate_passed"] + acct["abstained"]
                + acct["gated_out"] + acct["no_prediction"]) == N_TRADES
        assert acct["n"] == N_TRADES - acct["abstained"] - acct["gated_out"]

        e = art["variants"]["ensemble_gated"]
        assert e["status"] == "ok"
        assert e["accounting"]["members"] == [
            "multinomial_logistic", "lightgbm_direction"]
        assert e["accounting"]["ensemble_covered"] >= 0

        c = art["variants"]["gbm_combined"]
        assert c["status"] == "ok"
        assert c["accounting"]["note"] == rp.COUNTERFACTUAL_NOTE
        assert c["accounting"]["and_valuation_selected"] is True

        coh = art["matched_cohorts"]["gbm"]
        assert coh["status"] == "ok"
        assert coh["n_cohort"] == COHORT_N
        assert coh["excluded_no_prediction"] == COHORT_NO_PRED
        assert coh["excluded_quote_ineligible"] == COHORT_QUOTE_INELIGIBLE
        assert sum(coh["challenger_gate"].values()) == COHORT_N
        assert art["missing_data_accounting"]["gbm_no_prediction"] == \
            COHORT_NO_PRED


# --------------------------------------------------------------------------- #
# CLI smoke
# --------------------------------------------------------------------------- #

class TestCli:
    def test_main_smoke(self, tmp_path, capsys):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        out = str(tmp_path / "out.json")
        rc = bt.main([
            "--dataset", store, "--out", out,
            "--holdout", *HOLDOUT,
            "--n-boot", "10", "--no-neural", "--no-gbm"])
        assert rc == 0
        text = capsys.readouterr().out
        assert "parity ok=True" in text
        assert "cohort[classical]" in text
        assert "fingerprint=" in text
        assert f"written: {out}" in text
        assert os.path.exists(out)
