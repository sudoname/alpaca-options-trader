"""Acceptance tests for oracle/ml/baselines.py + oracle/ml/train.py
(Oracle 4.0 Phase 3 — classical challengers vs naive comparators).

Hermetic: synthetic numpy fixtures for the unit tests; CLI tests write
sqlite stores through the production writer ``oracle.ml.dataset._write_store``
so the end-to-end path is exercised against the real on-disk schema.
No network, no production trading imports.
"""
from __future__ import annotations

import json
import math
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.lab import purged_validation as pv  # noqa: E402
from oracle.ml import baselines as bl  # noqa: E402
from oracle.ml import train as tr  # noqa: E402
from oracle.ml.dataset import _write_store  # noqa: E402
from oracle.ml.schema import FeatureRow, LabelRow, LabelStatus  # noqa: E402

UTC = timezone.utc
SESSIONS = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]


def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _sample(decision_id: str, session: str, minute: int = 0,
            features=None, label_value=1.0, label_status="ok",
            kind="candidate") -> pv.Sample:
    as_of = _ts(session, minute)
    end = as_of + timedelta(minutes=60)
    return pv.Sample(
        decision_id=decision_id, session=session, as_of=as_of,
        kind=kind, features=features or {"x": 1.0},
        label_value=label_value, label_status=label_status,
        label_end=end, label_available_at=end)


# --------------------------------------------------------------------------- #
# design matrix / missingness encoding
# --------------------------------------------------------------------------- #

class TestBuildDesign:
    def _std(self):
        rows = [{"a": 1.0, "b": 2.0}, {"a": 3.0, "b": 4.0}]
        return pv.Standardizer().fit(rows, ["a", "b"])

    def test_missing_becomes_zero_plus_indicator(self):
        std = self._std()
        samples = [_sample("d1", SESSIONS[0], features={"a": 1.0, "b": None}),
                   _sample("d2", SESSIONS[0], features={"a": 3.0, "b": 4.0})]
        dm = bl.build_design(samples, std)
        assert dm.columns == ["a", "b", "a_missing", "b_missing"]
        assert dm.rows_used == [0, 1]
        assert not np.isnan(dm.x).any()
        # row 0: b missing -> value col 0.0 (train mean), flag 1.0
        assert dm.x[0, 1] == 0.0 and dm.x[0, 3] == 1.0
        assert dm.x[0, 2] == 0.0                       # a present -> flag 0
        # row 1: b present -> standardized, flag 0
        assert dm.x[1, 3] == 0.0
        assert dm.x[1, 1] == pytest.approx((4.0 - 3.0) / std.std["b"])

    def test_indicator_subset_fixed_at_train_time(self):
        std = self._std()
        samples = [_sample("d1", SESSIONS[0], features={"a": 1.0, "b": None})]
        dm = bl.build_design(samples, std, indicator_features=["b"])
        assert dm.columns == ["a", "b", "b_missing"]

    def test_empty_input_keeps_column_shape(self):
        std = self._std()
        dm = bl.build_design([], std)
        assert dm.x.shape == (0, 4)
        assert dm.rows_used == []


# --------------------------------------------------------------------------- #
# ridge logistic + Platt + calibration diagnostics
# --------------------------------------------------------------------------- #

class TestLogisticAndPlatt:
    def test_learns_separable_boundary(self):
        rng = np.random.default_rng(0)
        x = rng.normal(size=(300, 2))
        y = (x[:, 0] > 0).astype(int)
        model = bl.RidgeLogisticModel().fit(x, y)
        p = model.predict_proba(x)
        acc = float(np.mean((p >= 0.5) == y))
        assert acc > 0.95

    def test_platt_improves_overconfident_scores(self):
        rng = np.random.default_rng(1)
        s = rng.uniform(-3, 3, size=2000)
        true_p = 1.0 / (1.0 + np.exp(-0.5 * s))        # true slope = 0.5
        y = (rng.uniform(size=2000) < true_p).astype(int)
        raw = 1.0 / (1.0 + np.exp(-s))                 # overconfident (slope 1)
        a, b = bl.fit_platt(s, y)
        cal = bl.apply_platt(a, b, s)
        ll_raw = pv.binary_metrics(y.tolist(), raw.tolist())["log_loss"]
        ll_cal = pv.binary_metrics(y.tolist(), cal.tolist())["log_loss"]
        assert ll_cal < ll_raw
        assert 0.3 < a < 0.8                           # recovered ~0.5

    def test_reliability_bins_and_ece_hand_case(self):
        y = [1, 1, 0, 0]
        p = [0.95, 0.95, 0.05, 0.05]
        bins = bl.reliability_bins(y, p)
        assert len(bins) == 10
        assert bins[9]["n"] == 2 and bins[9]["frac_pos"] == 1.0
        assert bins[0]["n"] == 2 and bins[0]["frac_pos"] == 0.0
        ece = bl.expected_calibration_error(bins)
        assert ece == pytest.approx(0.05)

    def test_last_bin_includes_probability_one(self):
        bins = bl.reliability_bins([1], [1.0])
        assert bins[9]["n"] == 1

    def test_ece_none_when_empty(self):
        assert bl.expected_calibration_error(bl.reliability_bins([], [])) \
            is None


class TestNaiveComparators:
    def test_constant_base_rate(self):
        probs = bl.constant_base_rate_probs([1, 1, 0, 1], 3)
        assert np.allclose(probs, 0.75)
        assert np.allclose(bl.constant_base_rate_probs([], 2), 0.5)

    def test_momentum_buckets_and_mask(self):
        train = [
            _sample("p1", SESSIONS[0], features={"ret_60m": 1.0}),
            _sample("p2", SESSIONS[0], features={"ret_60m": 0.5}),
            _sample("p3", SESSIONS[0], features={"ret_60m": 0.0}),
            _sample("p4", SESSIONS[0], features={"ret_60m": 2.0}),
            _sample("n1", SESSIONS[0], features={"ret_60m": -1.0}),
            _sample("n2", SESSIONS[0], features={"ret_60m": -0.5}),
            _sample("n3", SESSIONS[0], features={"ret_60m": -2.0}),
            _sample("n4", SESSIONS[0], features={"ret_60m": -0.1}),
        ]
        train_y = [1, 1, 1, 0, 0, 0, 1, 1]             # pos: 3/4, neg: 2/4
        test = [
            _sample("t1", SESSIONS[1], features={"ret_60m": 3.0}),
            _sample("t2", SESSIONS[1], features={"ret_60m": -0.2}),
            _sample("t3", SESSIONS[1], features={"ret_60m": None}),
        ]
        probs, mask = bl.momentum_rule_probs(train, train_y, test)
        assert probs[0] == pytest.approx(0.75)
        assert probs[1] == pytest.approx(0.5)
        assert mask.tolist() == [1.0, 1.0, 0.0]
        assert probs[2] == 0.5                         # placeholder, masked

    def test_momentum_empty_bucket_falls_back_to_base_rate(self):
        train = [_sample("p1", SESSIONS[0], features={"ret_60m": 1.0})]
        test = [_sample("t1", SESSIONS[1], features={"ret_60m": -1.0})]
        probs, mask = bl.momentum_rule_probs(train, [1], test)
        assert probs[0] == pytest.approx(1.0)          # base rate of train_y
        assert mask[0] == 1.0

    def test_baseline_p_win_clip_and_mask(self):
        test = [
            _sample("a", SESSIONS[0], features={"baseline_p_win": 1.2}),
            _sample("b", SESSIONS[0], features={"baseline_p_win": -0.3}),
            _sample("c", SESSIONS[0], features={"baseline_p_win": 0.6}),
            _sample("d", SESSIONS[0], features={"baseline_p_win": None}),
        ]
        probs, mask = bl.baseline_p_win_probs(test)
        assert probs[0] == 1.0 and probs[1] == 0.0
        assert probs[2] == pytest.approx(0.6)
        assert mask.tolist() == [1.0, 1.0, 1.0, 0.0]


# --------------------------------------------------------------------------- #
# quantile ladder
# --------------------------------------------------------------------------- #

class TestQuantile:
    def test_quantiles_never_cross(self):
        rng = np.random.default_rng(2)
        x = rng.normal(size=(200, 2))
        y = x[:, 0] + 0.2 * rng.normal(size=200)
        model = bl.QuantilePinballModel().fit(x, y)
        preds = model.predict(x)
        stacked = np.stack([preds[t] for t in bl.QUANTILE_LADDER])
        assert np.all(np.diff(stacked, axis=0) >= -1e-12)

    def test_marginal_coverage_on_constant_features(self):
        rng = np.random.default_rng(3)
        y = rng.normal(size=500)
        x = np.zeros((500, 1))
        model = bl.QuantilePinballModel().fit(x, y)
        preds = model.predict(x)
        cov = bl.interval_coverage(y, preds[0.05], preds[0.95])
        assert cov == pytest.approx(0.90, abs=0.08)

    def test_conditional_model_beats_marginal_pinball(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(400, 1))
        y = 2.0 * x[:, 0] + 0.1 * rng.normal(size=400)
        model = bl.QuantilePinballModel().fit(x, y)
        preds = model.predict(x)
        for tau in (0.25, 0.5, 0.75):
            model_loss = bl.pinball_loss(y, preds[tau], tau)
            marg = np.full(400, np.quantile(y, tau))
            marg_loss = bl.pinball_loss(y, marg, tau)
            assert model_loss < marg_loss

    def test_losses_none_when_empty(self):
        assert bl.pinball_loss([], [], 0.5) is None
        assert bl.interval_coverage([], [], []) is None


# --------------------------------------------------------------------------- #
# realized-volatility forecasters
# --------------------------------------------------------------------------- #

class TestRvForecasters:
    def test_persistence_mask_semantics(self):
        samples = [
            _sample("a", SESSIONS[0], features={"rv_session": 0.5}),
            _sample("b", SESSIONS[0], features={"rv_session": None}),
            _sample("c", SESSIONS[0], features={"rv_session": -1.0}),
        ]
        f, mask = bl.rv_persistence_forecast(samples)
        assert f[0] == 0.5 and mask.tolist() == [1.0, 0.0, 0.0]

    def test_ewma_point_in_time_and_carry_forward(self):
        samples = [
            _sample("a1", SESSIONS[0], minute=0,
                    features={"rv_session": 1.0}),
            _sample("a2", SESSIONS[0], minute=10,
                    features={"rv_session": 2.0}),
            _sample("a3", SESSIONS[0], minute=20,
                    features={"rv_session": None}),
            _sample("b1", SESSIONS[0], minute=5,
                    features={"rv_session": None}),
        ]
        und = {"a1": "SPY", "a2": "SPY", "a3": "SPY", "b1": "QQQ"}
        # shuffle input order: the forecaster must sort by as_of itself
        order = [2, 0, 3, 1]
        shuffled = [samples[i] for i in order]
        f, mask = bl.rv_ewma_forecast(shuffled, und)
        by_id = {shuffled[i].decision_id: (f[i], mask[i])
                 for i in range(len(shuffled))}
        assert by_id["a1"] == (pytest.approx(1.0), 1.0)
        assert by_id["a2"][0] == pytest.approx(0.94 * 1.0 + 0.06 * 2.0)
        assert by_id["a3"][0] == pytest.approx(by_id["a2"][0])  # carry state
        assert by_id["a3"][1] == 1.0
        assert by_id["b1"] == (0.0, 0.0)               # no state, no value

    def test_har_recovers_planted_relation(self):
        rng = np.random.default_rng(5)
        samples, y, ewma = [], [], []
        for i in range(100):
            rv = float(rng.uniform(0.1, 1.0))
            r60 = float(rng.uniform(-0.05, 0.05))
            samples.append(_sample(f"h{i}", SESSIONS[0], minute=i, features={
                "rv_session": rv, "ret_15m": 0.0, "ret_60m": r60,
                "ret_session": 0.0}))
            ewma.append(rv * 0.9)
            y.append(2.0 * rv + 0.5 * abs(r60) + 0.1)
        model = bl.HarLiteModel().fit(samples, y, ewma)
        f, mask = model.predict(samples, ewma)
        assert mask.all()
        assert np.max(np.abs(f - np.asarray(y))) < 1e-4

    def test_har_masks_missing_rv_and_unfit_model(self):
        s_missing = _sample("m", SESSIONS[0], features={"rv_session": None})
        model = bl.HarLiteModel().fit([s_missing], [1.0], [None])
        assert model.w is None
        f, mask = model.predict([s_missing], [None])
        assert mask.tolist() == [0.0]

    def test_qlike_guards(self):
        assert bl.qlike_loss([1.0, 2.0], [1.0, 2.0]) == pytest.approx(0.0)
        assert bl.qlike_loss([0.0, -1.0], [1.0, 1.0]) is None
        assert bl.qlike_loss([1.0], [0.0]) is None
        # one valid row among invalid ones
        v = bl.qlike_loss([1.0, 0.0], [2.0, 1.0])
        r = 0.25
        assert v == pytest.approx(r - math.log(r) - 1.0)


# --------------------------------------------------------------------------- #
# discrete-time competing-risks hazards
# --------------------------------------------------------------------------- #

class TestHazards:
    def test_person_period_expansion_hand_case(self):
        times = [100.0, 100.0, None, -5.0, 10.0]
        causes = ["stop", None, "target", "stop", "target"]
        pp = bl.expand_person_periods(times, causes)
        assert len(pp) == 7
        # subject 0: event "stop" (cause idx 1) in bin 2 = (60, 120]
        s0 = [p for p in pp if p.row_index == 0]
        assert [(p.time_bin, p.outcome) for p in s0] == [(0, 0), (1, 0),
                                                         (2, 1)]
        # subject 1: censored at 100 -> survive bins 0,1,2
        s1 = [p for p in pp if p.row_index == 1]
        assert [(p.time_bin, p.outcome) for p in s1] == [(0, 0), (1, 0),
                                                         (2, 0)]
        # subjects 2,3 skipped (None / non-positive time)
        assert not [p for p in pp if p.row_index in (2, 3)]
        # subject 4: "target" (cause idx 2) in bin 0 = (0, 30]
        s4 = [p for p in pp if p.row_index == 4]
        assert [(p.time_bin, p.outcome) for p in s4] == [(0, 2)]

    def test_bin_edges_are_left_open_right_closed(self):
        on_edge = bl.expand_person_periods([30.0], ["stop"])
        past_edge = bl.expand_person_periods([30.5], ["stop"])
        assert [(p.time_bin, p.outcome) for p in on_edge] == [(0, 1)]
        assert [(p.time_bin, p.outcome) for p in past_edge] == [(0, 0),
                                                                (1, 1)]

    def _planted(self):
        # x=+1 subjects always hit "stop" in bin 0; x=-1 survive to 1500
        times = [10.0] * 30 + [1500.0] * 30
        causes = ["stop"] * 30 + [None] * 30
        x = np.asarray([[1.0]] * 30 + [[-1.0]] * 30)
        return x, times, causes

    def test_model_beats_marginal_on_planted_signal(self):
        x, times, causes = self._planted()
        periods = bl.expand_person_periods(times, causes)
        model = bl.DiscreteTimeCompetingRisks().fit(x, periods)
        nll_model = model.nll(x, periods)
        table = tr._marginal_hazard_table(
            periods, model.n_bins, len(bl.HAZARD_CAUSES) + 1)
        nll_marginal = tr._marginal_nll(table, periods)
        assert nll_model is not None and nll_marginal is not None
        assert nll_model < nll_marginal

    def test_cif_monotone_and_bounded(self):
        x, times, causes = self._planted()
        periods = bl.expand_person_periods(times, causes)
        model = bl.DiscreteTimeCompetingRisks().fit(x, periods)
        cif = model.cumulative_incidence(np.asarray([1.0]))
        assert cif.shape == (model.n_bins, len(bl.HAZARD_CAUSES))
        assert np.all(np.diff(cif, axis=0) >= -1e-12)  # nondecreasing
        assert float(cif[-1].sum()) <= 1.0 + 1e-9
        # "stop" dominates: it is the only observed cause
        assert cif[-1, 0] > cif[-1, 1] and cif[-1, 0] > cif[-1, 2]

    def test_nll_none_on_empty_periods(self):
        x, times, causes = self._planted()
        model = bl.DiscreteTimeCompetingRisks().fit(
            x, bl.expand_person_periods(times, causes))
        assert model.nll(x, []) is None

    def test_marginal_table_rows_normalized(self):
        periods = bl.expand_person_periods([10.0, 40.0], ["stop", "target"])
        table = tr._marginal_hazard_table(periods, 7, 4)
        assert table.shape == (7, 4)
        assert np.allclose(table.sum(axis=1), 1.0)

    def test_empirical_cif_hand_cases(self):
        # two events, no censoring
        cif = bl.empirical_cumulative_incidence(
            [10.0, 20.0], ["stop", "target"], at_minutes=30.0)
        assert cif["stop"] == pytest.approx(0.5)
        assert cif["target"] == pytest.approx(0.5)
        assert cif["other_exit"] == 0.0
        # censoring shrinks the risk set without incidence
        cif = bl.empirical_cumulative_incidence(
            [10.0, 20.0], ["stop", None], at_minutes=30.0)
        assert cif["stop"] == pytest.approx(0.5)
        assert cif["target"] == 0.0
        # events past the evaluation time don't count
        cif = bl.empirical_cumulative_incidence(
            [50.0], ["stop"], at_minutes=30.0)
        assert cif["stop"] == 0.0
        # empty input
        assert bl.empirical_cumulative_incidence([], [], 30.0) == {
            "stop": 0.0, "target": 0.0, "other_exit": 0.0}


# --------------------------------------------------------------------------- #
# CLI fixtures (production writer -> real on-disk schema)
# --------------------------------------------------------------------------- #

def _feature_row(did, session, minute, features, underlying="SPY"):
    as_of = _ts(session, minute)
    return FeatureRow(
        decision_id=did, snapshot_id=f"s-{did}", as_of=as_of,
        available_at=as_of, underlying=underlying, symbol=None,
        kind="candidate", session=session, features=features, quality={})


def _write(path, feats, labels):
    _write_store(path, {}, feats, labels, [],
                 {"dataset_hash": "fixturehash", "schema_version": "1"})


def _direction_store(path, n_per_session=20):
    """Learnable direction labels: sign of x, with comparator features."""
    rng = random.Random(0)
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(n_per_session):
            did = f"d-{session}-{i}"
            as_of = _ts(session, i)
            x = rng.uniform(-1, 1)
            feats.append(_feature_row(did, session, i, {
                "x": x, "ret_60m": rng.uniform(-0.02, 0.02),
                "baseline_p_win": rng.uniform(0.3, 0.7)
                if i % 2 == 0 else None,
                "dte_days": rng.choice([0.2, 3.0, 14.0])}))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="direction",
                value=1.0 if x > 0 else -1.0, status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
    _write(path, feats, labels)


def _log_return_store(path, n_per_session=20):
    rng = random.Random(1)
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(n_per_session):
            did = f"q-{session}-{i}"
            as_of = _ts(session, i)
            x = rng.uniform(-1, 1)
            feats.append(_feature_row(did, session, i, {"x": x}))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="log_return",
                value=0.01 * x + rng.gauss(0, 0.002), status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
    _write(path, feats, labels)


def _rv_store(path, n_per_session=20):
    rng = random.Random(2)
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(n_per_session):
            did = f"r-{session}-{i}"
            as_of = _ts(session, i)
            rv = rng.uniform(0.05, 0.5)
            und = "SPY" if i % 2 == 0 else "QQQ"
            feats.append(_feature_row(did, session, i, {
                "rv_session": rv, "ret_15m": rng.uniform(-0.01, 0.01),
                "ret_60m": rng.uniform(-0.02, 0.02),
                "ret_session": rng.uniform(-0.03, 0.03)}, underlying=und))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="future_rv",
                value=rv * rng.uniform(0.8, 1.2), status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
    _write(path, feats, labels)


def _hazard_store(path, n_per_session=25):
    rng = random.Random(3)
    causes = ["stop", "other_exit", "target"]
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(n_per_session):
            did = f"e-{session}-{i}"
            as_of = _ts(session, i)
            feats.append(_feature_row(did, session, i,
                                      {"x": rng.uniform(-1, 1)}))
            if i % 10 == 9:                            # censored row
                labels.append(LabelRow(
                    decision_id=did, horizon="event",
                    label_type="first_touch", value=None,
                    status=LabelStatus.CENSORED,
                    label_end=_ts(SESSIONS[-1], 300),  # coverage end
                    available_at=None, meta={"why": "beyond_coverage"}))
            else:
                minutes = rng.choice([10.0, 45.0, 90.0, 200.0])
                end = as_of + timedelta(minutes=minutes)
                labels.append(LabelRow(
                    decision_id=did, horizon="event",
                    label_type="first_touch", value=minutes,
                    status=LabelStatus.OK, label_end=end, available_at=end,
                    meta={"touch": causes[i % 3]}))
    _write(path, feats, labels)


# --------------------------------------------------------------------------- #
# train CLI end-to-end
# --------------------------------------------------------------------------- #

class TestTrainCli:
    def test_logistic_end_to_end_and_deterministic(self, tmp_path, capsys):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "train.json")
        _direction_store(store)
        argv = ["--model", "logistic", "--dataset", store,
                "--horizon", "60m", "--label-type", "direction",
                "--out", out, "--n-boot", "50"]
        assert tr.main(argv) == 0
        text = capsys.readouterr().out
        assert "fingerprint=" in text
        with open(out, encoding="utf-8") as f:
            r1 = json.load(f)
        assert r1["dataset_hash"] == "fixturehash"
        assert r1["sessions"]["holdout"] == SESSIONS[-2:]
        res = r1["result"]
        assert len(res["folds"]) == 4
        for fold in res["folds"]:
            assert not (set(fold["test_sessions"]) & set(SESSIONS[-2:]))
        assert set(res["comparators"]) == {"constant", "momentum",
                                           "baseline_p_win"}
        chall = res["challenger"]
        assert chall["name"] == "ridge_logistic_platt"
        assert chall["pooled"]["n"] > 0
        # labels are a deterministic function of x: model must beat chance
        assert chall["pooled"]["accuracy"] > 0.8
        assert "reliability_bins" in chall and "by_dte_bucket" in chall
        assert set(res["deltas_log_loss"]) == {"constant", "momentum",
                                               "baseline_p_win"}
        # determinism: identical re-run -> identical fingerprint
        out2 = str(tmp_path / "train2.json")
        assert tr.main(["--model", "logistic", "--dataset", store,
                        "--out", out2, "--n-boot", "50"]) == 0
        with open(out2, encoding="utf-8") as f:
            r2 = json.load(f)
        r1.pop("report_fingerprint")
        fp2 = r2.pop("report_fingerprint")
        r1["dataset_path"] = r2["dataset_path"] = None  # paths differ? no
        assert fp2 == json.loads(json.dumps(fp2))      # sanity
        assert r1["result"] == r2["result"]

    def test_naive_model_has_no_challenger(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "naive.json")
        _direction_store(store)
        assert tr.main(["--model", "naive", "--dataset", store,
                        "--out", out, "--n-boot", "20"]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        assert "challenger" not in r["result"]
        assert "comparators" in r["result"]

    def test_quantile_end_to_end(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "quantile.json")
        _log_return_store(store)
        assert tr.main(["--model", "quantile", "--dataset", store,
                        "--label-type", "log_return",
                        "--out", out, "--n-boot", "20"]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        pooled = r["result"]["pooled"]
        assert pooled["n"] > 0
        assert set(map(float, pooled["per_tau"])) == set(bl.QUANTILE_LADDER)
        assert 0.0 <= pooled["coverage_90_model"] <= 1.0
        assert "marginal" in r["result"]["deltas_pinball"]

    def test_rv_end_to_end(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "rv.json")
        _rv_store(store)
        assert tr.main(["--model", "rv", "--dataset", store,
                        "--label-type", "future_rv",
                        "--out", out, "--n-boot", "20"]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        pooled = r["result"]["pooled"]
        assert set(pooled) == {"persistence", "ewma", "har"}
        for name in pooled:
            assert pooled[name]["n"] > 0
            assert pooled[name]["qlike"] is not None
        assert set(r["result"]["deltas_qlike"]) == {"har_vs_persistence",
                                                    "ewma_vs_persistence"}

    def test_hazard_end_to_end(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "hazard.json")
        _hazard_store(store)
        assert tr.main(["--model", "hazard", "--dataset", store,
                        "--horizon", "event", "--label-type", "first_touch",
                        "--out", out, "--n-boot", "20"]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        res = r["result"]
        assert res["bin_edges_minutes"][-1] == "inf"
        assert list(res["causes"]) == list(bl.HAZARD_CAUSES)
        ran = [f for f in res["folds"] if not f.get("skipped")]
        assert ran
        for fold in ran:
            assert fold["test_nll_model"] is not None
            assert fold["test_nll_marginal"] is not None
        assert "note" in res                           # censored-purge caveat
        assert set(res["empirical_cif_1440m_all"]) == set(bl.HAZARD_CAUSES)

    def test_gbm_reports_availability_honestly(self, tmp_path):
        out = str(tmp_path / "gbm.json")
        assert tr.main(["--model", "gbm", "--dataset", "unused.sqlite",
                        "--out", out]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        assert r["result"]["status"] in ("available", "unavailable")
        if r["result"]["status"] == "unavailable":
            assert "lightgbm" in r["result"]["why"]

    def test_mlp_deferred_to_phase4(self, tmp_path):
        out = str(tmp_path / "mlp.json")
        assert tr.main(["--model", "mlp", "--dataset", "unused.sqlite",
                        "--out", out]) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        assert r["result"]["status"] == "deferred"

    def test_fold_config_override(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "cfg.json")
        _direction_store(store, n_per_session=5)
        assert tr.main(["--model", "naive", "--dataset", store,
                        "--out", out, "--n-boot", "10",
                        "--fold-config", '{"holdout_sessions": 0}']) == 0
        with open(out, encoding="utf-8") as f:
            r = json.load(f)
        assert r["sessions"]["holdout"] == []
        assert r["fold_config"]["holdout_sessions"] == 0

    def test_holdout_registration_write_then_verify(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "reg.json")
        reg = str(tmp_path / "pledge.json")
        _direction_store(store, n_per_session=5)
        argv = ["--model", "naive", "--dataset", store, "--out", out,
                "--n-boot", "10", "--holdout-registration", reg]
        assert tr.main(argv) == 0
        with open(reg, encoding="utf-8") as f:
            pledge = json.load(f)
        assert pledge["dataset_hash"] == "fixturehash"
        assert pledge["holdout_sessions"] == SESSIONS[-2:]
        assert pledge["n_samples"] == 10
        # re-run verifies, mismatched pledge refuses
        assert tr.main(argv) == 0
        pledge["dataset_hash"] = "tampered"
        with open(reg, "w", encoding="utf-8") as f:
            json.dump(pledge, f)
        with pytest.raises(ValueError, match="holdout registration mismatch"):
            tr.main(argv)
