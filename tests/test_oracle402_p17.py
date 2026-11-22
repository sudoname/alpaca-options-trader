# -*- coding: utf-8 -*-
"""Oracle 4.0.2 P1-7: unified forecast provider interface.

Covers the non-neural providers added in 4.0.2 (classical logistic, LightGBM,
bootstrap baseline, ensemble) plus the shared registry verification chain,
kind routing and the versioned provider cache.  The neural provider keeps its
own suite in tests/test_oracle401_forecast.py.
"""

import json
import os

import numpy as np
import pytest
from datetime import datetime, timezone

from oracle.lab.purged_validation import Standardizer
from oracle.ml import forecast as fcmod
from oracle.ml import gbm as gbm_mod
from oracle.ml.baselines import MultinomialLogisticModel, save_classical
from oracle.ml.forecast import (
    BaselineForecastProvider,
    ClassicalForecastProvider,
    EnsembleForecastProvider,
    ForecastUnavailable,
    GBMForecastProvider,
    get_provider,
    provider_from_registry,
)
from oracle.ml.mlp import standardizer_state
from oracle.ml.registry import register_model, sha256_file

FEAT_NAMES = ["alpha", "beta", "gamma"]
FEATS = {"alpha": 0.8, "beta": -0.3, "gamma": 0.1}
TRAIN_SESSIONS = ["2025-12-01"]
VAL_SESSION = "2025-12-02"
# last trained session 2025-12-02 -> train_end 2025-12-03; 45d window ends
# 2026-01-17, so 2026-01-05 is inside it.
AS_OF = datetime(2026, 1, 5, 15, 30, tzinfo=timezone.utc)
STALE_AS_OF = datetime(2026, 3, 1, 15, 30, tzinfo=timezone.utc)
OVERLAP_AS_OF = datetime(2025, 12, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _fresh_cache():
    fcmod.clear_cache()
    yield
    fcmod.clear_cache()


def _train_classical(tmp_path, name="clf.json", temperature=1.0,
                     val_session=VAL_SESSION, feature_columns=None):
    """Train + persist a real multinomial-logistic artifact on synthetic data."""
    rng = np.random.default_rng(3)
    rows, labels = [], []
    for _ in range(120):
        a, b, g = (float(v) for v in rng.normal(size=3))
        rows.append({"alpha": a, "beta": b, "gamma": g})
        labels.append(0 if a < -0.4 else (2 if a > 0.4 else 1))
    std = Standardizer().fit(rows, FEAT_NAMES)
    xs = []
    for r in rows:
        z = std.transform(r)
        xs.append([0.0 if v is None else float(v) for v in z]
                  + [1.0 if v is None else 0.0 for v in z])
    model = MultinomialLogisticModel().fit(
        np.asarray(xs, dtype=np.float64), np.asarray(labels, dtype=np.int64))
    cols = (feature_columns if feature_columns is not None
            else FEAT_NAMES + [f"{n}_missing" for n in FEAT_NAMES])
    meta = {"standardizer": standardizer_state(std),
            "train_sessions": list(TRAIN_SESSIONS),
            "val_session": val_session,
            "horizon": "60m"}
    path = str(tmp_path / name)
    save_classical(path, model, cols, temperature=temperature, meta=meta)
    return path


def _registry(tmp_path):
    return str(tmp_path / "registry.json")


def _repin(regpath, model_id, new_sha):
    """Test-only registry edit: repin an entry's checksums (bypasses
    immutability on purpose to simulate a legitimate re-release)."""
    with open(regpath, encoding="utf-8") as f:
        doc = json.load(f)
    cks = doc["models"][model_id]["checksums"]
    for k in list(cks):
        cks[k] = new_sha
    with open(regpath, "w", encoding="utf-8") as f:
        json.dump(doc, f)


# --------------------------------------------------------------------------- #
# classical (multinomial logistic)
# --------------------------------------------------------------------------- #

class TestClassicalProvider:
    def test_round_trip_via_registry(self, tmp_path):
        art = _train_classical(tmp_path, temperature=1.25)
        regpath = _registry(tmp_path)
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art, manifest_path=art)
        p = provider_from_registry(regpath, "clf1")
        assert isinstance(p, ClassicalForecastProvider)
        assert p.model_kind == "classical"
        assert p.temperature == 1.25
        assert p.train_sessions == TRAIN_SESSIONS
        assert p.val_session == VAL_SESSION
        assert p.horizon == "60m"
        file_sha = sha256_file(art)
        assert p.checksums["manifest"] == file_sha
        assert p.checksums["artifact"] == file_sha

        fc = p.forecast(FEATS, as_of=AS_OF)
        s = fc.p_down + fc.p_neutral + fc.p_up
        assert abs(s - 1.0) < 1e-9
        for v in (fc.p_down, fc.p_neutral, fc.p_up):
            assert 0.0 < v < 1.0
        # direction-only: never fabricates quantiles/RV/CIF
        assert fc.quantiles == {}
        assert fc.rv_forecast is None
        assert fc.cif == {}
        assert fc.cif_bin_edges_min == []
        assert fc.meta["payload_kind"] == "multinomial_logistic"
        assert fc.n_features == 3 and fc.n_missing == 0
        assert len(fc.direction_logits) == 3
        d = fc.to_dict()
        assert d["available"] is True
        assert d["model_id"] == "clf1"

    def test_partial_missing_features_still_forecast(self, tmp_path):
        art = _train_classical(tmp_path)
        p = ClassicalForecastProvider.from_artifact(art)
        fc = p.forecast({"alpha": 0.5, "beta": None, "gamma": 0.1},
                        as_of=AS_OF)
        assert fc.n_missing == 1
        assert fc.missing_features == ["beta"]
        assert abs(fc.p_down + fc.p_neutral + fc.p_up - 1.0) < 1e-9

    def test_all_missing_refused(self, tmp_path):
        art = _train_classical(tmp_path)
        p = ClassicalForecastProvider.from_artifact(art)
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast({n: None for n in FEAT_NAMES}, as_of=AS_OF)
        assert ei.value.reason == "FEATURES_ALL_MISSING"

    def test_cutoff_gating(self, tmp_path):
        art = _train_classical(tmp_path)
        p = ClassicalForecastProvider.from_artifact(art)
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=OVERLAP_AS_OF)
        assert ei.value.reason == "TRAIN_OVERLAP"
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=STALE_AS_OF)
        assert ei.value.reason == "MODEL_STALE"
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=datetime(2026, 1, 5, 15, 30))  # naive
        assert ei.value.reason == "INFERENCE_ERROR"

    def test_temperature_flattens_probabilities(self, tmp_path):
        cold = ClassicalForecastProvider.from_artifact(
            _train_classical(tmp_path, name="t1.json", temperature=1.0))
        hot = ClassicalForecastProvider.from_artifact(
            _train_classical(tmp_path, name="t5.json", temperature=5.0))
        f1 = cold.forecast(FEATS, as_of=AS_OF)
        f5 = hot.forecast(FEATS, as_of=AS_OF)
        assert (max(f5.p_down, f5.p_neutral, f5.p_up)
                < max(f1.p_down, f1.p_neutral, f1.p_up))

    def test_val_session_list_normalized_to_last(self, tmp_path):
        art = _train_classical(tmp_path,
                               val_session=["2025-12-02", "2025-12-03"])
        p = ClassicalForecastProvider.from_artifact(art)
        assert p.val_session == "2025-12-03"

    def test_internal_checksum_tamper_refused(self, tmp_path):
        art = _train_classical(tmp_path)
        with open(art, encoding="utf-8") as f:
            doc = json.load(f)
        doc["checksum_sha256"] = "0" * 64  # corrupt BEFORE registering
        with open(art, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        regpath = _registry(tmp_path)
        register_model(regpath, "bad1", kind="classical",
                       artifact_path=art, manifest_path=art)
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "bad1")
        assert ei.value.reason == "MODEL_INTEGRITY"

    def test_post_registration_edit_refused(self, tmp_path):
        art = _train_classical(tmp_path)
        regpath = _registry(tmp_path)
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art, manifest_path=art)
        with open(art, encoding="utf-8") as f:
            doc = json.load(f)
        doc["payload"]["temperature"] = 9.0  # edit AFTER pinning
        with open(art, "w", encoding="utf-8") as f:
            json.dump(doc, f)
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "clf1")
        assert ei.value.reason == "REGISTRY_CHECKSUM_MISMATCH"

    def test_schema_invalid_without_missing_flag_columns(self, tmp_path):
        art = _train_classical(tmp_path, feature_columns=list(FEAT_NAMES))
        regpath = _registry(tmp_path)
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art, manifest_path=art)
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "clf1")
        assert ei.value.reason == "SCHEMA_INVALID"

    def test_manifest_not_pinned_refused(self, tmp_path):
        art = _train_classical(tmp_path)
        regpath = _registry(tmp_path)
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art)  # no manifest_path
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "clf1")
        assert ei.value.reason == "MANIFEST_NOT_PINNED"

    def test_unknown_payload_kind_refused(self, tmp_path):
        path = str(tmp_path / "weird.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"payload": {"kind": "mystery"}}, f)
        regpath = _registry(tmp_path)
        register_model(regpath, "w1", kind="classical",
                       artifact_path=path, manifest_path=path)
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "w1")
        assert ei.value.reason == "KIND_UNSUPPORTED"
        assert "mystery" in ei.value.detail


# --------------------------------------------------------------------------- #
# LightGBM
# --------------------------------------------------------------------------- #

class TestGBMProvider:
    @pytest.mark.skipif(not gbm_mod.LGBM_AVAILABLE,
                        reason=gbm_mod.LGBM_WHY or "lightgbm unavailable")
    def test_round_trip_via_registry(self, tmp_path):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(240, 3))
        y = np.where(x[:, 0] < -0.3, 0,
                     np.where(x[:, 0] > 0.3, 2, 1)).astype(int)
        booster, _ = gbm_mod.fit_direction_gbm(
            x[:200], y[:200], x[200:], y[200:])
        path = str(tmp_path / "gbm.json")
        gbm_mod.save_gbm_direction(
            path, booster, FEAT_NAMES, "native", gbm_mod.GBMConfig(),
            meta={"train_sessions": list(TRAIN_SESSIONS),
                  "val_session": VAL_SESSION, "horizon": "60m"})
        regpath = _registry(tmp_path)
        register_model(regpath, "g1", kind="classical",
                       artifact_path=path, manifest_path=path)
        p = provider_from_registry(regpath, "g1")
        assert isinstance(p, GBMForecastProvider)
        assert p.missingness_mode == "native"
        assert p.val_session == VAL_SESSION

        fc = p.forecast(FEATS, as_of=AS_OF)
        assert abs(fc.p_down + fc.p_neutral + fc.p_up - 1.0) < 1e-9
        assert fc.quantiles == {} and fc.rv_forecast is None
        assert fc.meta["payload_kind"] == "lightgbm_multiclass_direction"
        assert fc.meta["missingness_mode"] == "native"
        assert fc.n_features == 3

        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast({n: None for n in FEAT_NAMES}, as_of=AS_OF)
        assert ei.value.reason == "FEATURES_ALL_MISSING"
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast(FEATS, as_of=OVERLAP_AS_OF)
        assert ei.value.reason == "TRAIN_OVERLAP"

    def test_lightgbm_unavailable_refusal(self, tmp_path, monkeypatch):
        # from_artifact checks the runtime BEFORE touching the file, so a
        # stub artifact exercises the refusal without needing lightgbm.
        path = str(tmp_path / "fake_gbm.json")
        with open(path, "w", encoding="utf-8") as f:
            json.dump({"payload": {"kind": "lightgbm_multiclass_direction"}},
                      f)
        regpath = _registry(tmp_path)
        register_model(regpath, "g0", kind="classical",
                       artifact_path=path, manifest_path=path)
        monkeypatch.setattr(gbm_mod, "LGBM_AVAILABLE", False)
        monkeypatch.setattr(gbm_mod, "LGBM_WHY", "forced off for test")
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "g0")
        assert ei.value.reason == "LIGHTGBM_UNAVAILABLE"
        assert ei.value.detail == "forced off for test"

    def test_flags_mode_schema_validation(self):
        with pytest.raises(ForecastUnavailable) as ei:
            GBMForecastProvider(object(), ["a", "b", "c"], "flags",
                                {"train_sessions": TRAIN_SESSIONS},
                                model_id="x", checksums={})
        assert ei.value.reason == "SCHEMA_INVALID"
        with pytest.raises(ForecastUnavailable) as ei:
            GBMForecastProvider(object(), ["a"], "weird",
                                {"train_sessions": TRAIN_SESSIONS},
                                model_id="x", checksums={})
        assert ei.value.reason == "SCHEMA_INVALID"


# --------------------------------------------------------------------------- #
# bootstrap baseline
# --------------------------------------------------------------------------- #

class TestBaselineProvider:
    RETURNS = [0.01, -0.02, 0.0, 0.015, -0.005]
    SESSIONS = ["2025-12-01", "2025-12-02"]

    def test_laplace_direction_probs_and_heads(self):
        p = BaselineForecastProvider(self.RETURNS, self.SESSIONS)
        assert p.model_kind == "baseline"
        fc = p.forecast({}, as_of=AS_OF)
        # counts: down 2, neutral 1, up 2 -> Laplace [3, 2, 3] / 8
        assert fc.p_down == pytest.approx(3 / 8)
        assert fc.p_neutral == pytest.approx(2 / 8)
        assert fc.p_up == pytest.approx(3 / 8)
        assert set(fc.quantiles) == {"0.05", "0.1", "0.25", "0.5",
                                     "0.75", "0.9", "0.95"}
        assert fc.quantiles["0.5"] == pytest.approx(
            float(np.quantile(self.RETURNS, 0.5)))
        assert fc.rv_forecast is not None and fc.rv_forecast > 0.0
        assert fc.n_features == 0 and fc.n_missing == 0
        assert fc.meta["basis"] == "bootstrap_empirical_train_window"
        assert fc.meta["n_train_returns"] == 5

    def test_deterministic_checksums(self):
        a = BaselineForecastProvider(self.RETURNS, self.SESSIONS)
        b = BaselineForecastProvider(list(self.RETURNS), list(self.SESSIONS))
        assert a.checksums == b.checksums
        assert a.checksums["manifest"] == a.checksums["artifact"]
        c = BaselineForecastProvider(self.RETURNS + [0.03], self.SESSIONS)
        assert c.checksums != a.checksums

    def test_cutoff_gating(self):
        p = BaselineForecastProvider(self.RETURNS, self.SESSIONS)
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast({}, as_of=OVERLAP_AS_OF)
        assert ei.value.reason == "TRAIN_OVERLAP"
        with pytest.raises(ForecastUnavailable) as ei:
            p.forecast({}, as_of=STALE_AS_OF)
        assert ei.value.reason == "MODEL_STALE"

    def test_input_validation(self):
        with pytest.raises(ValueError):
            BaselineForecastProvider([], self.SESSIONS)
        with pytest.raises(ValueError):
            BaselineForecastProvider([float("nan"), float("inf")],
                                     self.SESSIONS)
        with pytest.raises(ValueError):
            BaselineForecastProvider(self.RETURNS, [])


# --------------------------------------------------------------------------- #
# ensemble
# --------------------------------------------------------------------------- #

class TestEnsembleProvider:
    def _two_members(self, tmp_path, regpath):
        a1 = _train_classical(tmp_path, name="m1.json", temperature=1.0)
        a2 = _train_classical(tmp_path, name="m2.json", temperature=2.0)
        register_model(regpath, "m1", kind="classical",
                       artifact_path=a1, manifest_path=a1)
        register_model(regpath, "m2", kind="classical",
                       artifact_path=a2, manifest_path=a2)
        return ["m1", "m2"]

    def _register_ensemble(self, tmp_path, regpath, members,
                           model_id="ens1", name="ens.json"):
        epath = str(tmp_path / name)
        with open(epath, "w", encoding="utf-8") as f:
            json.dump({"members": members, "horizon": "60m"}, f)
        register_model(regpath, model_id, kind="ensemble",
                       manifest_path=epath)
        return epath

    def test_mean_of_members_via_registry(self, tmp_path):
        regpath = _registry(tmp_path)
        members = self._two_members(tmp_path, regpath)
        self._register_ensemble(tmp_path, regpath, members)
        ens = provider_from_registry(regpath, "ens1")
        assert isinstance(ens, EnsembleForecastProvider)
        assert ens.model_kind == "ensemble"
        assert ens.horizon == "60m"

        f1 = provider_from_registry(regpath, "m1").forecast(FEATS,
                                                            as_of=AS_OF)
        f2 = provider_from_registry(regpath, "m2").forecast(FEATS,
                                                            as_of=AS_OF)
        fe = ens.forecast(FEATS, as_of=AS_OF)
        exp = np.mean([[f1.p_down, f1.p_neutral, f1.p_up],
                       [f2.p_down, f2.p_neutral, f2.p_up]], axis=0)
        exp = exp / exp.sum()
        assert fe.p_down == pytest.approx(float(exp[0]))
        assert fe.p_neutral == pytest.approx(float(exp[1]))
        assert fe.p_up == pytest.approx(float(exp[2]))
        assert fe.meta["member_model_ids"] == members
        assert set(fe.meta["prob_dispersion"]) == {"down", "neutral", "up"}
        assert fe.meta["uncertainty_max_std"] >= 0.0
        # direction-only members: heads stay empty, never fabricated
        assert fe.quantiles == {} and fe.rv_forecast is None

    def test_member_failure_propagates(self, tmp_path):
        regpath = _registry(tmp_path)
        members = self._two_members(tmp_path, regpath)
        self._register_ensemble(tmp_path, regpath, members)
        ens = provider_from_registry(regpath, "ens1")
        with pytest.raises(ForecastUnavailable) as ei:
            ens.forecast(FEATS, as_of=STALE_AS_OF)
        assert ei.value.reason == "MODEL_STALE"

    def test_membership_cycle_refused(self, tmp_path):
        regpath = _registry(tmp_path)
        self._register_ensemble(tmp_path, regpath, ["loop"],
                                model_id="loop", name="loop.json")
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "loop")
        assert ei.value.reason == "SCHEMA_INVALID"
        assert "cycle" in ei.value.detail

    def test_empty_members_refused(self, tmp_path):
        regpath = _registry(tmp_path)
        self._register_ensemble(tmp_path, regpath, [],
                                model_id="ens0", name="ens0.json")
        with pytest.raises(ForecastUnavailable) as ei:
            provider_from_registry(regpath, "ens0")
        assert ei.value.reason == "SCHEMA_INVALID"
        with pytest.raises(ForecastUnavailable) as ei:
            EnsembleForecastProvider([], [], model_id="e", checksums={})
        assert ei.value.reason == "SCHEMA_INVALID"

    def test_mixed_heads_never_partially_fabricated(self, tmp_path):
        base = BaselineForecastProvider(
            [0.01, -0.02, 0.0, 0.015, -0.005],
            ["2025-12-01", "2025-12-02"])
        clf = ClassicalForecastProvider.from_artifact(
            _train_classical(tmp_path))
        ens = EnsembleForecastProvider(
            [base, clf], ["baseline_bootstrap", "unregistered"],
            model_id="mix", checksums={"manifest": "test"})
        fc = ens.forecast(FEATS, as_of=AS_OF)
        # quantile ladders differ (baseline full vs classical empty) ->
        # omitted entirely; RV is the mean over members that report one.
        assert fc.quantiles == {}
        base_rv = base.forecast({}, as_of=AS_OF).rv_forecast
        assert fc.rv_forecast == pytest.approx(base_rv)


# --------------------------------------------------------------------------- #
# cache with versioned refresh
# --------------------------------------------------------------------------- #

class TestProviderCache:
    def test_cache_hit_and_versioned_refresh(self, tmp_path):
        art = _train_classical(tmp_path, temperature=1.0)
        regpath = _registry(tmp_path)
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art, manifest_path=art)
        p1 = get_provider(regpath, "clf1")
        p2 = get_provider(regpath, "clf1")
        assert p1 is p2
        assert p1.temperature == 1.0

        # legitimate re-release: new artifact at the same path + repinned
        # registry checksums -> next lookup must reload, not serve stale.
        _train_classical(tmp_path, name=os.path.basename(art),
                         temperature=3.0)
        _repin(regpath, "clf1", sha256_file(art))
        p3 = get_provider(regpath, "clf1")
        assert p3 is not p1
        assert p3.temperature == 3.0

    def test_failures_never_cached(self, tmp_path):
        regpath = _registry(tmp_path)
        art = _train_classical(tmp_path)
        with pytest.raises(ForecastUnavailable) as ei:
            get_provider(regpath, "clf1")  # registry missing
        assert ei.value.reason == "REGISTRY_MISSING"
        register_model(regpath, "clf1", kind="classical",
                       artifact_path=art, manifest_path=art)
        p = get_provider(regpath, "clf1")  # succeeds right after
        assert isinstance(p, ClassicalForecastProvider)
