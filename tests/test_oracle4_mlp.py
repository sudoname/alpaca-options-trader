"""Phase 4 tests: multitask MLP challenger (oracle/ml/mlp.py + train CLI).

Framework-free pieces (label assembly, purge gating, head masking) are
tested unconditionally.  Torch-dependent tests skip honestly when torch
is not installed (optional dependency) -- never faked.

All fixtures are synthetic; nothing here is evidence of profitability.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.lab import purged_validation as pv  # noqa: E402
from oracle.ml import mlp  # noqa: E402
from oracle.ml import train as tr  # noqa: E402
from oracle.ml.dataset import _write_store  # noqa: E402
from oracle.ml.schema import FeatureRow, LabelRow, LabelStatus  # noqa: E402

UTC = timezone.utc
SESSIONS = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]

needs_torch = pytest.mark.skipif(
    not mlp.TORCH_AVAILABLE,
    reason="torch not installed (optional dependency)")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _sample(did, session, minute=0, *, value=1.0, status="ok",
            end_min=60, features=None):
    """Build a pv.Sample; label ends end_min minutes after as_of."""
    as_of = _ts(session, minute)
    end = (as_of + timedelta(minutes=end_min)
           if end_min is not None else None)
    return pv.Sample(decision_id=did, session=session, as_of=as_of,
                     kind="candidate", features=features or {"x": 1.0},
                     label_value=value, label_status=status,
                     label_end=end, label_available_at=end)


def _cfg(**kw):
    base = dict(hidden=(16, 8, 8), dropout=0.0, lr=1e-2,
                weight_decay=0.0, batch_size=64, max_epochs=8, patience=3)
    base.update(kw)
    return mlp.MLPConfig(**base)


def _direction_head(x: np.ndarray) -> mlp.HeadData:
    n = len(x)
    return mlp.HeadData(
        direction_cls=[0 if float(v) < 0 else 2 for v in x[:, 0]],
        log_return=[None] * n, rv=[None] * n,
        hazard_time=[None] * n, hazard_cause=[None] * n)


def _feature_row(did, session, minute, features, underlying="SPY"):
    as_of = _ts(session, minute)
    return FeatureRow(
        decision_id=did, snapshot_id=f"s-{did}", as_of=as_of,
        available_at=as_of, underlying=underlying, symbol=None,
        kind="candidate", session=session, features=features, quality={})


def _write(path, feats, labels):
    _write_store(path, {}, feats, labels, [],
                 {"dataset_hash": "fixturehash", "schema_version": "1"})


def _multitask_store(path: str, n_per_session: int = 24) -> None:
    """Fixture store with all four heads per decision.

    * direction / log_return / future_rv on the 60m horizon
    * first_touch events ('event' horizon) with stop/other_exit/target
      causes; every 10th subject censored beyond coverage
    * every 5th future_rv label omitted entirely (masking path)
    * baseline_p_win missing on odd rows (missing-indicator path)
    """
    feats, labels = [], []
    causes = ["stop", "other_exit", "target"]
    for session in SESSIONS:
        day = int(session[-2:])
        for i in range(n_per_session):
            did = f"m-{session}-{i}"
            k = (2 * i + 5 * day + 1) % 13
            x = (k - 6.5) / 6.5  # never exactly zero
            as_of = _ts(session, i)
            feats.append(_feature_row(did, session, i, {
                "x": x,
                "ret_60m": 0.5 * x,
                "baseline_p_win": (None if i % 2 else 0.5 + 0.2 * x),
                "rv_session": 0.0001 + 0.00005 * (i % 7),
            }))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="direction",
                value=(1.0 if x > 0 else -1.0), status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="log_return",
                value=0.01 * x + 0.001 * ((i % 5) - 2),
                status=LabelStatus.OK, label_end=end, available_at=end,
                meta={}))
            if i % 5 != 0:
                labels.append(LabelRow(
                    decision_id=did, horizon="60m", label_type="future_rv",
                    value=0.0001 + 0.00002 * (i % 7), status=LabelStatus.OK,
                    label_end=end, available_at=end, meta={}))
            if i % 10 == 0:
                labels.append(LabelRow(
                    decision_id=did, horizon="event",
                    label_type="first_touch", value=None,
                    status=LabelStatus.CENSORED,
                    label_end=_ts(SESSIONS[-1], 300), available_at=None,
                    meta={"why": "beyond_coverage"}))
            else:
                minutes = [10.0, 45.0, 90.0, 200.0][i % 4]
                t_end = as_of + timedelta(minutes=minutes)
                labels.append(LabelRow(
                    decision_id=did, horizon="event",
                    label_type="first_touch", value=minutes,
                    status=LabelStatus.OK, label_end=t_end,
                    available_at=t_end, meta={"touch": causes[i % 3]}))
    _write(path, feats, labels)


# --------------------------------------------------------------------------- #
# framework-free unit tests (always run)
# --------------------------------------------------------------------------- #

class TestDirectionClass:
    def test_mapping(self):
        assert mlp.direction_class(-1.0) == 0
        assert mlp.direction_class(-0.5) == 0
        assert mlp.direction_class(0.0) == 1
        assert mlp.direction_class(1.0) == 2
        assert mlp.direction_class(2.0) == 2


class TestPurgeSafe:
    def test_truth_table(self):
        t = _ts(SESSIONS[1])
        before = _ts(SESSIONS[0])
        after = _ts(SESSIONS[2])
        # no test boundary => everything usable
        assert mlp._purge_safe(None, None, None) is True
        assert mlp._purge_safe(after, after, None) is True
        # unknown end => assume leak
        assert mlp._purge_safe(None, before, t) is False
        # end at/after boundary leaks
        assert mlp._purge_safe(after, before, t) is False
        assert mlp._purge_safe(t, before, t) is False
        # late availability leaks even if the window ended early
        assert mlp._purge_safe(before, after, t) is False
        assert mlp._purge_safe(before, t, t) is False
        # fully before the boundary is safe
        assert mlp._purge_safe(before, before, t) is True
        assert mlp._purge_safe(before, None, t) is True


class TestAssembleHeadData:
    def test_direction_from_own_label(self):
        rows = [_sample("a", SESSIONS[0], value=-1.0),
                _sample("b", SESSIONS[0], value=0.0),
                _sample("c", SESSIONS[0], value=1.0),
                _sample("d", SESSIONS[0], value=None, status="missing")]
        hd = mlp.assemble_head_data(rows, {}, {}, {}, {}, None)
        assert hd.direction_cls == [0, 1, 2, None]
        assert hd.log_return == [None] * 4
        assert hd.rv == [None] * 4
        assert hd.hazard_time == [None] * 4
        assert hd.counts() == {"direction": 3, "quantile": 0,
                               "rv": 0, "hazard": 0}

    def test_aux_joins_and_rv_positivity(self):
        rows = [_sample("a", SESSIONS[0]), _sample("b", SESSIONS[0])]
        aux_lr = {"a": _sample("a", SESSIONS[0], value=0.012)}
        aux_rv = {"a": _sample("a", SESSIONS[0], value=0.0),  # not > 0
                  "b": _sample("b", SESSIONS[0], value=0.004)}
        hd = mlp.assemble_head_data(rows, aux_lr, aux_rv, {}, {}, None)
        assert hd.log_return == [pytest.approx(0.012), None]
        assert hd.rv == [None, pytest.approx(0.004)]
        assert hd.counts()["quantile"] == 1
        assert hd.counts()["rv"] == 1

    def test_purge_gate_masks_overlapping_labels(self):
        t0 = datetime.fromisoformat(SESSIONS[1]).replace(tzinfo=UTC)
        safe = _sample("a", SESSIONS[0])                    # ends same day
        leaky = _sample("b", SESSIONS[0], end_min=24 * 60)  # crosses t0
        aux_lr = {"a": _sample("a", SESSIONS[0], value=0.01),
                  "b": _sample("b", SESSIONS[0], value=0.01,
                               end_min=24 * 60)}
        hd = mlp.assemble_head_data([safe, leaky], aux_lr, {}, {}, {}, t0)
        assert hd.direction_cls == [2, None]
        assert hd.log_return == [pytest.approx(0.01), None]

    def test_hazard_events_and_censoring(self):
        rows = [_sample(d, SESSIONS[0]) for d in ("a", "b", "c", "d")]
        ev = {
            "a": _sample("a", SESSIONS[0], value=45.0, end_min=45),
            "b": _sample("b", SESSIONS[0], value=45.0, end_min=45),
            "c": _sample("c", SESSIONS[0], value=None, status="censored",
                         end_min=300),
            "d": _sample("d", SESSIONS[0], value=-5.0, end_min=45),
        }
        causes = {"a": "stop", "b": "weird_cause", "d": "stop"}
        # default: censored excluded; unknown cause / non-positive masked
        hd = mlp.assemble_head_data(rows, {}, {}, ev, causes, None)
        assert hd.hazard_time == [45.0, None, None, None]
        assert hd.hazard_cause == ["stop", None, None, None]
        # with the flag, censored subjects contribute survival time only
        hd2 = mlp.assemble_head_data(rows, {}, {}, ev, causes, None,
                                     include_censored_hazard=True)
        assert hd2.hazard_time == [45.0, None, pytest.approx(300.0), None]
        assert hd2.hazard_cause == ["stop", None, None, None]


# --------------------------------------------------------------------------- #
# torch-dependent tests
# --------------------------------------------------------------------------- #

@needs_torch
class TestModelOutputs:
    def _fixture(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(40, 5)).astype(np.float32)
        mlp.torch.manual_seed(0)
        model = mlp.MultitaskMLP(5, _cfg())
        return model, x

    def test_shapes_and_coherence(self):
        model, x = self._fixture()
        p = mlp.predict_heads(model, x, device="cpu")
        n_q = len(_cfg().quantiles)
        assert p["direction_probs"].shape == (40, 3)
        assert np.allclose(p["direction_probs"].sum(axis=1), 1.0, atol=1e-5)
        assert p["quantiles"].shape == (40, n_q)
        assert np.all(np.diff(p["quantiles"], axis=1) >= -1e-7)
        assert p["rv_forecast"].shape == (40,)
        assert np.all(p["rv_forecast"] > 0)
        assert p["hazards"].shape == (40, mlp.N_HAZARD_BINS,
                                      mlp.N_HAZARD_CLASSES)
        assert np.allclose(p["hazards"].sum(axis=2), 1.0, atol=1e-5)
        assert np.allclose(p["p_up"], p["direction_probs"][:, 2])

    def test_cif_coherent(self):
        model, x = self._fixture()
        p = mlp.predict_heads(model, x)
        cif = mlp.cif_from_hazards(p["hazards"])
        assert cif.shape == (40, mlp.N_HAZARD_BINS,
                             mlp.N_HAZARD_CLASSES - 1)
        assert np.all(np.diff(cif, axis=1) >= -1e-9)
        assert np.all(cif.sum(axis=2) <= 1.0 + 1e-9)

    def test_temperature_flattens_direction_probs(self):
        model, x = self._fixture()
        p1 = mlp.predict_heads(model, x, temperature=1.0)
        p5 = mlp.predict_heads(model, x, temperature=5.0)
        assert (p5["direction_probs"].max(axis=1).mean()
                < p1["direction_probs"].max(axis=1).mean())


@needs_torch
class TestTrainingMechanics:
    def _xy(self, n=80, d=4, seed=1):
        rng = np.random.default_rng(seed)
        x = rng.normal(size=(n, d)).astype(np.float32)
        return x, _direction_head(x)

    def test_direction_only_training(self):
        x, hd = self._xy()
        xv, hv = self._xy(n=24, seed=2)
        _, hist = mlp.train_mlp(x, hd, xv, hv, cfg=_cfg(), seed=42,
                                device="cpu")
        assert hist["epochs_run"] >= 1
        assert hist["normalizers"]["direction"] is not None
        assert hist["normalizers"]["quantile"] is None
        assert hist["normalizers"]["rv"] is None
        assert hist["normalizers"]["hazard"] is None
        assert hist["val_loss_at_best"] is not None
        assert hist["train_val_gap"] is not None
        assert hist["head_label_counts_train"]["direction"] == 80

    def test_all_masked_heads_raise(self):
        x, _ = self._xy(n=10)
        hd = mlp.HeadData(direction_cls=[None] * 10,
                          log_return=[None] * 10, rv=[None] * 10,
                          hazard_time=[None] * 10, hazard_cause=[None] * 10)
        hv = mlp.HeadData(direction_cls=[None] * 2,
                          log_return=[None] * 2, rv=[None] * 2,
                          hazard_time=[None] * 2, hazard_cause=[None] * 2)
        with pytest.raises(ValueError, match="no head has any usable"):
            mlp.train_mlp(x, hd, x[:2], hv, cfg=_cfg())

    def test_empty_x_raises(self):
        hd = mlp.HeadData()
        with pytest.raises(ValueError, match="non-empty"):
            mlp.train_mlp(np.zeros((0, 3), dtype=np.float32), hd,
                          np.zeros((0, 3), dtype=np.float32), hd,
                          cfg=_cfg())

    def test_class_weights_hand_case(self):
        hd = mlp.HeadData(direction_cls=[0, 0, 0, 2],
                          log_return=[None] * 4, rv=[None] * 4,
                          hazard_time=[None] * 4, hazard_cause=[None] * 4)
        w = mlp._class_weights(hd, mlp.torch.device("cpu"))
        assert w is not None
        # counts [3, 0, 1]; absent class gets weight 0
        assert np.allclose(w.numpy(), [4.0 / 6.0, 0.0, 2.0])

    def test_early_stopping_lr_zero(self):
        # lr=0 => epoch 0 is the only improvement; patience exhausts.
        x, hd = self._xy()
        xv, hv = self._xy(n=24, seed=2)
        _, hist = mlp.train_mlp(
            x, hd, xv, hv, cfg=_cfg(lr=0.0, max_epochs=10, patience=3),
            seed=42)
        assert hist["epochs_run"] == 4  # best epoch + 3 bad epochs
        assert hist["best_epoch"] == 0
        assert hist["stopped_early"] is True

    def test_max_epochs_cap(self):
        x, hd = self._xy()
        xv, hv = self._xy(n=24, seed=2)
        _, hist = mlp.train_mlp(
            x, hd, xv, hv, cfg=_cfg(lr=0.0, max_epochs=3, patience=20),
            seed=42)
        assert hist["epochs_run"] == 3
        assert hist["stopped_early"] is False

    def test_cpu_determinism_same_seed(self):
        x, hd = self._xy()
        xv, hv = self._xy(n=24, seed=2)
        m1, _ = mlp.train_mlp(x, hd, xv, hv, cfg=_cfg(), seed=7)
        m2, _ = mlp.train_mlp(x, hd, xv, hv, cfg=_cfg(), seed=7)
        p1 = mlp.predict_heads(m1, xv)
        p2 = mlp.predict_heads(m2, xv)
        for key in ("direction_probs", "quantiles", "rv_forecast",
                    "hazards"):
            assert np.allclose(p1[key], p2[key], atol=0.0), key

    def test_learns_planted_direction_signal(self):
        # Mechanics check on synthetic data only -- no profitability claim.
        rng = np.random.default_rng(3)
        x = rng.normal(size=(400, 4)).astype(np.float32)
        hd = _direction_head(x)
        model, _ = mlp.train_mlp(
            x, hd, x[:80], _direction_head(x[:80]),
            cfg=_cfg(max_epochs=30, patience=30), seed=0)
        p = mlp.predict_heads(model, x)
        pred = p["direction_probs"].argmax(axis=1)
        truth = np.array(hd.direction_cls)
        assert (pred == truth).mean() > 0.8


@needs_torch
class TestTemperature:
    def test_degenerate_returns_one(self):
        assert mlp.fit_temperature(np.zeros((0, 3)), np.zeros((0,))) == 1.0
        lg = np.random.default_rng(0).normal(size=(10, 3))
        assert mlp.fit_temperature(lg, np.ones(10)) == 1.0

    def test_overconfident_logits_cooled(self):
        rng = np.random.default_rng(5)
        z = rng.normal(size=(600, 3))
        p = np.exp(z) / np.exp(z).sum(axis=1, keepdims=True)
        targets = np.array([rng.choice(3, p=row) for row in p])
        # logits 3x too sharp => fitted temperature should be near 3
        t = mlp.fit_temperature(3.0 * z, targets)
        assert 1.5 < t < 6.0


@needs_torch
class TestSaveLoad:
    def _saved(self, tmp_path):
        mlp.torch.manual_seed(0)
        model = mlp.MultitaskMLP(4, _cfg())
        std = pv.Standardizer().fit(
            [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}], ["a", "b"])
        stem = str(tmp_path / "mlp_test")
        manifest = mlp.save_model(
            model, stem,
            feature_columns=["a", "b", "a_missing", "b_missing"],
            standardizer_state=mlp.standardizer_state(std),
            temperature=1.25, seed=42, git_sha="deadbeef",
            dataset_hash="fixturehash", train_sessions=SESSIONS[:2],
            val_session=SESSIONS[2])
        return model, stem, manifest

    def test_roundtrip_parity_and_manifest(self, tmp_path):
        model, stem, manifest = self._saved(tmp_path)
        assert manifest["d_in"] == 4
        assert manifest["feature_columns"] == ["a", "b", "a_missing",
                                               "b_missing"]
        assert manifest["temperature"] == 1.25
        assert manifest["checksum_sha256"]
        assert manifest["torch_version"]
        assert manifest["standardizer"]["mean"]["a"] == pytest.approx(2.0)
        loaded, m2 = mlp.load_model(stem + "_manifest.json", device="cpu")
        x = np.random.default_rng(1).normal(size=(16, 4)).astype(np.float32)
        p1 = mlp.predict_heads(model, x)
        p2 = mlp.predict_heads(loaded, x)
        for key in ("direction_probs", "quantiles", "rv_forecast",
                    "hazards"):
            assert np.allclose(p1[key], p2[key], atol=1e-6), key
        assert m2["checksum_sha256"] == manifest["checksum_sha256"]

    def test_tampered_weights_rejected(self, tmp_path):
        _, stem, _ = self._saved(tmp_path)
        with open(stem + "_model.pt", "ab") as f:
            f.write(b"x")
        with pytest.raises(mlp.ModelIntegrityError, match="checksum"):
            mlp.load_model(stem + "_manifest.json")

    def test_missing_model_file_rejected(self, tmp_path):
        _, stem, _ = self._saved(tmp_path)
        os.remove(stem + "_model.pt")
        with pytest.raises(mlp.ModelIntegrityError, match="missing"):
            mlp.load_model(stem + "_manifest.json")

    def test_feature_schema_mismatch_rejected(self, tmp_path):
        _, stem, _ = self._saved(tmp_path)
        path = stem + "_manifest.json"
        with open(path, encoding="utf-8") as f:
            manifest = json.load(f)
        manifest["feature_columns"] = manifest["feature_columns"][:2]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(manifest, f)
        with pytest.raises(mlp.ModelIntegrityError, match="feature schema"):
            mlp.load_model(path)

    def test_absent_manifest_rejected(self, tmp_path):
        with pytest.raises(mlp.ModelIntegrityError, match="unreadable"):
            mlp.load_model(str(tmp_path / "nope_manifest.json"))


# --------------------------------------------------------------------------- #
# CLI integration
# --------------------------------------------------------------------------- #

class TestTrainCliMlp:
    def test_wrong_label_type_rejected(self):
        with pytest.raises(SystemExit):
            tr.main(["--model", "mlp", "--dataset", "unused.sqlite",
                     "--out", "x.json", "--label-type", "log_return"])

    @needs_torch
    def test_end_to_end(self, tmp_path):
        store = str(tmp_path / "store.sqlite")
        _multitask_store(store)
        out = str(tmp_path / "mlp.json")
        rc = tr.main(["--model", "mlp", "--dataset", store,
                      "--horizon", "60m", "--out", out,
                      "--n-boot", "20", "--device", "cpu"])
        assert rc == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        # holdout untouched
        assert r["sessions"]["holdout"] == SESSIONS[-2:]
        res = r["result"]
        assert res["device"] == "cpu"
        assert len(res["folds"]) == 4
        for fold in res["folds"]:
            assert fold.get("skipped") is False
            assert not set(fold["test_sessions"]) & set(SESSIONS[-2:])
            assert fold["history"]["epochs_run"] >= 1
            assert fold["temperature"] > 0
        heads = res["heads"]
        assert set(heads) == {"direction", "quantile", "rv", "hazard"}
        assert heads["direction"] is not None
        assert "comparators" in heads["direction"]
        assert "deltas_log_loss" in heads["direction"]
        assert heads["quantile"] is not None
        assert heads["rv"] is not None
        assert heads["hazard"] is not None
        assert heads["hazard"]["cif_coherent"] is True
        # final artifact saved + reload parity verified by the CLI itself
        art = res["artifact"]
        assert art["status"] == "saved"
        assert art["reload_parity"]["ok"] is True
        stem = out[:-5]
        assert os.path.exists(stem + "_model.pt")
        assert os.path.exists(stem + "_manifest.json")
