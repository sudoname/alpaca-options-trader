"""Oracle 4.0.1 Stage E tests — 3-class direction toolkit + LightGBM.

Covers (per the master prompt acceptance criteria):
  * multinomial logistic: determinism, simplex outputs, input validation;
  * temperature scaling: identity on single-class tails, never worse NLL;
  * hand-computed multiclass metrics (log loss, Brier, confusion, macro F1);
  * naive comparators (constant class frequency, trailing-sign momentum);
  * classical artifact save / reload / tamper detection;
  * actual LightGBM fit / predict / serialize on a fixture (skipped
    honestly when the pinned optional dep is absent), probs sum to 1,
    determinism, quantile non-crossing, native-vs-flags missingness,
    shuffled-label negative control;
  * train.py 3-class harness helpers: availability-purged fit/val split,
    pooled metrics, and the baseline_p_win semantic-mismatch report.

Research-only tests; nothing here touches the production trading path.
"""
from __future__ import annotations

import json
import math
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

import oracle.ml.baselines as bl
import oracle.ml.gbm as gbm
import oracle.ml.train as tr
from oracle.lab.purged_validation import Sample
from oracle.ml.mlp import direction_class


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

_T0 = datetime(2026, 1, 5, 15, 0, tzinfo=timezone.utc)


def _sample(did: str, session: str, feats: dict, label: float = 1.0,
            available_at=None, as_of=None) -> Sample:
    as_of = as_of or _T0
    return Sample(
        decision_id=did, session=session, as_of=as_of, kind="candidate",
        features=feats, label_value=label, label_status="ok",
        label_end=as_of + timedelta(minutes=60),
        label_available_at=available_at)


def _xy(n: int, seed: int = 0):
    """Separable 3-class fixture: class k lights up its own feature."""
    rng = np.random.default_rng(seed)
    y = np.asarray([i % 3 for i in range(n)], dtype=np.int64)
    x = np.column_stack([
        (y == 0) * 1.0 + rng.normal(0, 0.05, n),
        (y == 2) * 1.0 + rng.normal(0, 0.05, n),
    ])
    return x, y


# --------------------------------------------------------------------------- #
# canonical class order
# --------------------------------------------------------------------------- #

class TestCanonicalClasses:
    def test_direction_class_mapping(self):
        assert direction_class(-1.0) == 0
        assert direction_class(0.0) == 1
        assert direction_class(1.0) == 2

    def test_order_shared_everywhere(self):
        assert bl.DIRECTION_CLASSES == ("down", "neutral", "up")

    def test_train_direction_cls_uses_canonical_mapping(self):
        s = _sample("d", "2026-01-05", {}, label=-1.0)
        assert tr._direction_cls(s) == 0
        s = _sample("d", "2026-01-05", {}, label=0.0)
        assert tr._direction_cls(s) == 1


# --------------------------------------------------------------------------- #
# multinomial logistic
# --------------------------------------------------------------------------- #

class TestMultinomialLogistic:
    def test_fit_predict_separable_simplex(self):
        x, y = _xy(120)
        m = bl.MultinomialLogisticModel().fit(x, y)
        p = m.predict_proba(x)
        assert p.shape == (120, 3)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert (p.argmax(axis=1) == y).mean() == 1.0

    def test_deterministic_refit(self):
        x, y = _xy(90)
        w1 = bl.MultinomialLogisticModel().fit(x, y).w
        w2 = bl.MultinomialLogisticModel().fit(x, y).w
        assert np.array_equal(w1, w2)

    def test_label_out_of_range_rejected(self):
        x, _ = _xy(6)
        with pytest.raises(ValueError):
            bl.MultinomialLogisticModel().fit(x, [0, 1, 2, 3, 0, 1])
        with pytest.raises(ValueError):
            bl.MultinomialLogisticModel().fit(x, [0, 1, 2, -1, 0, 1])

    def test_sample_weight_validation(self):
        x, y = _xy(6)
        with pytest.raises(ValueError):
            bl.MultinomialLogisticModel().fit(x, y, sample_weight=[1.0])
        with pytest.raises(ValueError):
            bl.MultinomialLogisticModel().fit(
                x, y, sample_weight=[1, 1, 1, 1, 1, -1])

    def test_class_weight_shifts_minority_mass(self):
        # 2 classes present out of 3; upweighting the minority must raise
        # its predicted probability on its own rows
        x = np.asarray([[1.0], [1.1], [0.9], [1.05], [-1.0]])
        y = [2, 2, 2, 2, 0]
        plain = bl.MultinomialLogisticModel().fit(x, y).predict_proba(x)
        boosted = bl.MultinomialLogisticModel().fit(
            x, y, sample_weight=[1, 1, 1, 1, 4]).predict_proba(x)
        assert boosted[4, 0] > plain[4, 0]


# --------------------------------------------------------------------------- #
# temperature scaling
# --------------------------------------------------------------------------- #

class TestTemperature:
    def test_identity_when_single_class(self):
        logits = np.asarray([[2.0, 0.0, -2.0]] * 5)
        assert bl.fit_temperature_np(logits, [2] * 5) == 1.0

    def test_apply_rows_sum_to_one(self):
        logits = np.asarray([[3.0, 0.0, -3.0], [0.0, 0.0, 0.0]])
        for t in (0.1, 1.0, 10.0):
            p = bl.apply_temperature_np(logits, t)
            assert np.allclose(p.sum(axis=1), 1.0, atol=1e-12)

    def test_never_worse_than_identity_nll(self):
        # overconfident logits on noisy labels: grid includes T=1 exactly,
        # so the fitted temperature can never lose to the identity
        rng = np.random.default_rng(3)
        y = rng.integers(0, 3, 60)
        logits = np.zeros((60, 3))
        logits[np.arange(60), y] = 8.0          # overconfident
        flip = rng.random(60) < 0.3             # 30% wrong
        logits[flip] = np.roll(logits[flip], 1, axis=1)
        t = bl.fit_temperature_np(logits, y)
        nll_id = bl.multiclass_log_loss(y, bl.apply_temperature_np(logits, 1.0))
        nll_fit = bl.multiclass_log_loss(y, bl.apply_temperature_np(logits, t))
        assert nll_fit <= nll_id + 1e-12
        assert t > 1.0                          # softening, not sharpening


# --------------------------------------------------------------------------- #
# hand-computed metrics
# --------------------------------------------------------------------------- #

class TestHandComputedMetrics:
    def test_log_loss_hand(self):
        p = np.asarray([[0.5, 0.25, 0.25]])
        assert bl.multiclass_log_loss([0], p) == pytest.approx(
            -math.log(0.5), abs=1e-9)

    def test_brier_hand(self):
        p = np.asarray([[0.2, 0.2, 0.6]])
        # (0.2^2 + 0.2^2 + 0.4^2) = 0.24
        assert bl.multiclass_brier([2], p) == pytest.approx(0.24)

    def test_confusion_and_macro_f1_hand(self):
        y, pred = [0, 0, 1, 2], [0, 1, 1, 2]
        assert bl.confusion_matrix(y, pred) == [
            [1, 1, 0], [0, 1, 0], [0, 0, 1]]
        # f1(down)=2/3, f1(neutral)=2/3, f1(up)=1 -> macro 7/9
        assert bl.macro_f1(y, pred) == pytest.approx(7.0 / 9.0)

    def test_macro_f1_excludes_absent_classes(self):
        assert bl.macro_f1([0, 0], [0, 0]) == pytest.approx(1.0)

    def test_empty_inputs_are_none(self):
        assert bl.multiclass_log_loss([], np.zeros((0, 3))) is None
        assert bl.multiclass_brier([], np.zeros((0, 3))) is None
        assert bl.macro_f1([], []) is None

    def test_per_class_reliability_shape(self):
        y = [0, 1, 2, 2]
        p = np.asarray([[0.8, 0.1, 0.1], [0.1, 0.8, 0.1],
                        [0.1, 0.1, 0.8], [0.2, 0.2, 0.6]])
        out = bl.per_class_reliability(y, p)
        assert set(out) == set(bl.DIRECTION_CLASSES)
        for v in out.values():
            assert "bins" in v and "ece" in v


# --------------------------------------------------------------------------- #
# naive comparators
# --------------------------------------------------------------------------- #

class TestComparators:
    def test_constant_class_probs(self):
        p = bl.constant_class_probs([0, 0, 2, 2, 2, 1], n_test=4)
        assert p.shape == (4, 3)
        assert np.allclose(p[0], [2 / 6, 1 / 6, 3 / 6])
        # uniform fallback
        u = bl.constant_class_probs([], n_test=2)
        assert np.allclose(u, 1.0 / 3.0)

    def test_momentum_class_probs_masks_missing(self):
        train = [_sample(f"t{i}", "2026-01-05", {"ret_60m": v})
                 for i, v in enumerate([0.5, 0.4, -0.5, -0.4])]
        train_cls = [2, 2, 0, 0]
        test = [_sample("a", "2026-01-06", {"ret_60m": 0.3}),
                _sample("b", "2026-01-06", {"ret_60m": -0.3}),
                _sample("c", "2026-01-06", {})]          # feature missing
        probs, mask = bl.momentum_class_probs(train, train_cls, test)
        assert mask.tolist() == [1.0, 1.0, 0.0]
        assert probs[0].argmax() == 2                    # positive -> up bucket
        assert probs[1].argmax() == 0
        assert np.allclose(probs.sum(axis=1), 1.0)


# --------------------------------------------------------------------------- #
# classical artifact integrity
# --------------------------------------------------------------------------- #

class TestClassicalArtifact:
    def test_round_trip(self, tmp_path):
        x, y = _xy(60)
        m = bl.MultinomialLogisticModel().fit(x, y)
        path = str(tmp_path / "clf.json")
        digest = bl.save_classical(path, m, ["f0", "f1"], temperature=1.7,
                                   meta={"note": "fixture"})
        m2, cols, temp, meta = bl.load_classical(path)
        assert len(digest) == 64
        assert cols == ["f0", "f1"]
        assert temp == pytest.approx(1.7)
        assert meta == {"note": "fixture"}
        assert np.allclose(m2.predict_proba(x), m.predict_proba(x))

    def test_tamper_detected(self, tmp_path):
        x, y = _xy(30)
        m = bl.MultinomialLogisticModel().fit(x, y)
        path = str(tmp_path / "clf.json")
        bl.save_classical(path, m, ["f0", "f1"])
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["payload"]["weights"][0][0] += 1.0
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        with pytest.raises(bl.ClassicalIntegrityError):
            bl.load_classical(path)

    def test_malformed_rejected(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"nope": 1}, fh)
        with pytest.raises(bl.ClassicalIntegrityError):
            bl.load_classical(path)


# --------------------------------------------------------------------------- #
# LightGBM (actual training; honestly skipped when the pin is absent)
# --------------------------------------------------------------------------- #

needs_lgbm = pytest.mark.skipif(
    not gbm.LGBM_AVAILABLE,
    reason=f"lightgbm optional dep unavailable: {gbm.LGBM_WHY}")


class TestBuildMatrix:
    def test_native_nan_and_flags(self):
        s = [_sample("a", "2026-01-05", {"x": 1.5, "y": None}),
             _sample("b", "2026-01-05", {"x": float("nan"), "y": 2.0})]
        xn, cn = gbm.build_matrix(s, ["x", "y"], mode="native")
        assert cn == ["x", "y"]
        assert xn[0, 0] == 1.5 and math.isnan(xn[0, 1])
        assert math.isnan(xn[1, 0]) and xn[1, 1] == 2.0
        xf, cf = gbm.build_matrix(s, ["x", "y"], mode="flags")
        assert cf == ["x", "y", "x_missing", "y_missing"]
        assert xf[0].tolist() == [1.5, 0.0, 0.0, 1.0]
        assert xf[1].tolist() == [0.0, 2.0, 1.0, 0.0]

    def test_unknown_mode_rejected(self):
        with pytest.raises(ValueError):
            gbm.build_matrix([], ["x"], mode="zero")

    def test_class_weights_inverse_frequency(self):
        w = gbm._class_weights([0, 0, 0, 2])
        assert w.mean() == pytest.approx(1.0)
        assert w[3] == pytest.approx(3 * w[0])   # minority upweighted


@needs_lgbm
class TestGBMDirection:
    def test_fit_predict_simplex_and_separable(self):
        xt, yt = _xy(300, seed=1)
        xv, yv = _xy(99, seed=2)
        booster, info = gbm.fit_direction_gbm(xt, yt, xv, yv)
        p = gbm.predict_direction_proba(booster, xv)
        assert p.shape == (99, 3)
        assert np.allclose(p.sum(axis=1), 1.0, atol=1e-9)
        assert (p.argmax(axis=1) == yv).mean() > 0.95
        assert info["val_used"] is True
        assert info["class_weights_used"] is True
        assert info["n_train"] == 300 and info["n_val"] == 99

    def test_deterministic_refit(self):
        xt, yt = _xy(300, seed=1)
        xv, yv = _xy(99, seed=2)
        b1, _ = gbm.fit_direction_gbm(xt, yt, xv, yv)
        b2, _ = gbm.fit_direction_gbm(xt, yt, xv, yv)
        assert np.array_equal(gbm.predict_direction_proba(b1, xv),
                              gbm.predict_direction_proba(b2, xv))

    def test_tiny_or_single_class_val_skips_early_stopping(self):
        xt, yt = _xy(300, seed=1)
        booster, info = gbm.fit_direction_gbm(
            xt, yt, np.zeros((3, 2)), [2, 2, 2])
        assert info["val_used"] is False
        assert "single-class" in info["val_why_skipped"]
        assert booster is not None

    def test_save_reload_parity_and_tamper(self, tmp_path):
        xt, yt = _xy(300, seed=1)
        xv, yv = _xy(99, seed=2)
        booster, _ = gbm.fit_direction_gbm(xt, yt, xv, yv)
        path = str(tmp_path / "gbm.json")
        digest = gbm.save_gbm_direction(path, booster, ["f0", "f1"],
                                        "native", gbm.GBMConfig(),
                                        meta={"horizon": "60m"})
        assert len(digest) == 64
        b2, cols, mode, meta = gbm.load_gbm_direction(path)
        assert cols == ["f0", "f1"] and mode == "native"
        assert meta == {"horizon": "60m"}
        assert np.allclose(gbm.predict_direction_proba(b2, xv),
                           gbm.predict_direction_proba(booster, xv))
        # tamper with the model text -> integrity error
        with open(path, encoding="utf-8") as fh:
            doc = json.load(fh)
        doc["model_text"] += "\n"
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(doc, fh)
        with pytest.raises(gbm.GBMIntegrityError):
            gbm.load_gbm_direction(path)

    def test_malformed_artifact_rejected(self, tmp_path):
        path = str(tmp_path / "bad.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"kind": "lightgbm_multiclass_direction"}, fh)
        with pytest.raises(gbm.GBMIntegrityError):
            gbm.load_gbm_direction(path)

    def test_shuffled_label_negative_control(self):
        # destroying the feature-label link must destroy the skill
        xt, yt = _xy(300, seed=1)
        xv, yv = _xy(99, seed=2)
        rng = np.random.default_rng(7)
        yt_shuffled = yt.copy()
        rng.shuffle(yt_shuffled)
        booster, _ = gbm.fit_direction_gbm(xt, yt_shuffled, xv, yv)
        acc = (gbm.predict_direction_proba(booster, xv)
               .argmax(axis=1) == yv).mean()
        assert acc < 0.6                         # ~1/3 expected, 1.0 if leak


@needs_lgbm
class TestGBMQuantiles:
    def test_non_crossing_ladder(self):
        rng = np.random.default_rng(5)
        xt = rng.normal(0, 1, (300, 2))
        yt = xt[:, 0] + rng.normal(0, 0.3, 300)
        xv = rng.normal(0, 1, (60, 2))
        yv = xv[:, 0] + rng.normal(0, 0.3, 60)
        boosters, info = gbm.fit_quantile_gbms(xt, yt, xv, yv)
        assert set(boosters) == set(float(t) for t in bl.QUANTILE_LADDER)
        assert info["val_used"] is True
        q = gbm.predict_quantiles(boosters, xv)
        taus = sorted(q)
        for lo, hi in zip(taus, taus[1:]):
            assert (q[lo] <= q[hi] + 1e-12).all()


# --------------------------------------------------------------------------- #
# train.py 3-class harness helpers
# --------------------------------------------------------------------------- #

class TestTrainHelpers:
    def test_availability_split_purges_leaky_fit_rows(self):
        val_start = datetime(2026, 1, 7, tzinfo=timezone.utc)
        train = [
            _sample("keep", "2026-01-05", {},
                    available_at=val_start - timedelta(hours=3)),
            _sample("purge", "2026-01-06", {},
                    available_at=val_start),          # >= val start: purged
            _sample("unknown", "2026-01-06", {}, available_at=None),
            _sample("val1", "2026-01-07", {},
                    available_at=val_start + timedelta(hours=1)),
        ]
        fit, val, acct = tr._availability_split(train)
        assert [s.decision_id for s in fit] == ["keep"]
        assert [s.decision_id for s in val] == ["val1"]
        assert acct == {"fit_rows_kept": 1,
                        "fit_rows_purged_availability": 1,
                        "fit_rows_unknown_availability": 1}

    def test_mc_metrics_hand_and_empty(self):
        out = tr._mc_metrics([], np.zeros((0, 3)))
        assert out["n"] == 0 and out["log_loss"] is None
        p = np.asarray([[0.7, 0.2, 0.1], [0.1, 0.2, 0.7]])
        out = tr._mc_metrics([0, 2], p)
        assert out["accuracy"] == 1.0
        assert out["class_counts"] == {"down": 1, "neutral": 0, "up": 1}
        assert out["confusion_matrix"][0][0] == 1
        assert out["confusion_matrix"][2][2] == 1

    def test_baseline_p_win_reported_unavailable_for_3class(self):
        assert tr._BASELINE_PWIN_3CLASS["status"] == "unavailable"
        assert "PoP" in tr._BASELINE_PWIN_3CLASS["why"]

    def test_binary_vs_3class_note_present(self):
        assert "NOT comparable" in tr._BINARY_VS_3CLASS_NOTE
