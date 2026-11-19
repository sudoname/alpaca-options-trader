"""Acceptance tests for oracle/lab/purged_validation.py (Oracle 4.0 Phase 2).

Hermetic: every store fixture is written through the production writer
``oracle.ml.dataset._write_store`` so the reader is exercised against the
real on-disk schema.  No network, no production imports in the harness.
"""
from __future__ import annotations

import json
import os
import random
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.lab import purged_validation as pv  # noqa: E402
from oracle.ml.dataset import _write_store  # noqa: E402
from oracle.ml.schema import FeatureRow, LabelRow, LabelStatus  # noqa: E402

UTC = timezone.utc
SESSIONS = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]


def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _sample(decision_id: str, session: str, minute: int = 0,
            features=None, label_value=1.0, label_status="ok",
            label_end="auto", label_available_at="auto") -> pv.Sample:
    as_of = _ts(session, minute)
    if label_end == "auto":
        label_end = as_of + timedelta(minutes=60)
    if label_available_at == "auto":
        label_available_at = label_end
    return pv.Sample(
        decision_id=decision_id, session=session, as_of=as_of,
        kind="candidate", features=features or {"x": 1.0},
        label_value=label_value, label_status=label_status,
        label_end=label_end, label_available_at=label_available_at)


def _synthetic_samples(n_per_session: int = 20, seed: int = 0):
    """Noise features + random binary direction labels across all sessions."""
    rng = random.Random(seed)
    out = []
    for session in SESSIONS:
        for i in range(n_per_session):
            out.append(_sample(
                f"d-{session}-{i}", session, minute=i,
                features={"x": rng.uniform(-1, 1),
                          "y": rng.uniform(-1, 1),
                          "dte_days": rng.choice([0.2, 3.0, 14.0, None])},
                label_value=rng.choice([-1.0, 1.0])))
    return out


# --------------------------------------------------------------------------- #
# fold construction
# --------------------------------------------------------------------------- #

class TestMakeFolds:
    def test_expanding_chronological_folds(self):
        samples = _synthetic_samples()
        folds, fold_sessions, holdout = pv.make_folds(
            samples, test_block_sessions=1, min_train_sessions=2,
            embargo_sessions=0, holdout_sessions=2)
        assert holdout == SESSIONS[-2:]
        assert fold_sessions == SESSIONS[:-2]
        # 6 fold sessions, min_train 2 -> tests on sessions idx 2..5 = 4 folds
        assert len(folds) == 4
        for f in folds:
            assert f.train_sessions == fold_sessions[:len(f.train_sessions)]
            assert max(f.train_sessions) < min(f.test_sessions)
            assert len(f.train_sessions) >= 2

    def test_holdout_never_in_any_fold(self):
        samples = _synthetic_samples()
        folds, _, holdout = pv.make_folds(samples, holdout_sessions=2)
        hold = set(holdout)
        for f in folds:
            assert not (set(f.train_sessions) & hold)
            assert not (set(f.test_sessions) & hold)
        by_id = {s.decision_id: s for s in samples}
        for f in folds:
            for did in f.train_ids + f.test_ids:
                assert by_id[did].session not in hold

    def test_group_overlap_raises(self):
        # same decision_id appears in a train session and a test session
        samples = [
            _sample("a", SESSIONS[0]), _sample("b", SESSIONS[0]),
            _sample("c", SESSIONS[1]),
            _sample("dup", SESSIONS[1]),
            _sample("dup", SESSIONS[2]),
        ]
        with pytest.raises(AssertionError, match="decision groups leak"):
            pv.make_folds(samples, min_train_sessions=2,
                          embargo_sessions=0, holdout_sessions=0)

    def test_no_holdout_requested(self):
        samples = _synthetic_samples()
        folds, fold_sessions, holdout = pv.make_folds(
            samples, holdout_sessions=0)
        assert holdout == []
        assert fold_sessions == SESSIONS
        assert len(folds) == 6  # tests on idx 2..7


class TestPurging:
    def _base(self):
        # sessions 0,1 train; session 2 test (min_train=2, block=1)
        return [
            _sample("t0", SESSIONS[0]),
            _sample("t1", SESSIONS[1]),
            _sample("te", SESSIONS[2]),
        ]

    def test_label_end_overlap_is_purged(self):
        test_start = datetime.fromisoformat(SESSIONS[2]).replace(tzinfo=UTC)
        bad = _sample("leaky", SESSIONS[1],
                      label_end=test_start + timedelta(hours=1))
        folds, _, _ = pv.make_folds(self._base() + [bad],
                                    embargo_sessions=0, holdout_sessions=0)
        f = folds[0]
        assert "leaky" not in f.train_ids
        assert f.purged == 1
        assert f.purge_detail["label_end_overlap"] == 1

    def test_availability_overlap_is_purged(self):
        test_start = datetime.fromisoformat(SESSIONS[2]).replace(tzinfo=UTC)
        # label measured before the window but only KNOWABLE after it starts
        bad = _sample("late", SESSIONS[1],
                      label_end=_ts(SESSIONS[1], 60),
                      label_available_at=test_start + timedelta(minutes=5))
        folds, _, _ = pv.make_folds(self._base() + [bad],
                                    embargo_sessions=0, holdout_sessions=0)
        f = folds[0]
        assert "late" not in f.train_ids
        assert f.purge_detail["availability_overlap"] == 1

    def test_unknown_end_embargoed_only_near_test(self):
        unk_near = _sample("unk_near", SESSIONS[1], label_status="missing",
                           label_value=None, label_end=None,
                           label_available_at=None)
        unk_far = _sample("unk_far", SESSIONS[0], label_status="missing",
                          label_value=None, label_end=None,
                          label_available_at=None)
        folds, _, _ = pv.make_folds(self._base() + [unk_near, unk_far],
                                    embargo_sessions=1, holdout_sessions=0)
        f = folds[0]
        assert "unk_near" not in f.train_ids       # inside 1-session embargo
        assert "unk_far" in f.train_ids            # outside embargo, kept
        assert f.embargoed == 1
        assert f.purge_detail["unknown_end_in_embargo"] == 1

    def test_known_end_not_embargoed(self):
        # a row in the embargo zone whose label END is known and before the
        # test window must be KEPT (embargo is only for unknown ends)
        ok_near = _sample("ok_near", SESSIONS[1])  # ends same session
        folds, _, _ = pv.make_folds(self._base() + [ok_near],
                                    embargo_sessions=1, holdout_sessions=0)
        assert "ok_near" in folds[0].train_ids
        assert folds[0].embargoed == 0


# --------------------------------------------------------------------------- #
# fold-train-only preprocessing
# --------------------------------------------------------------------------- #

class TestStandardizer:
    def test_mean_std_and_none_passthrough(self):
        rows = [{"a": 1.0, "b": 10.0}, {"a": 3.0, "b": 10.0},
                {"a": None, "b": 10.0}]
        std = pv.Standardizer().fit(rows)
        assert std.mean["a"] == pytest.approx(2.0)
        assert std.std["a"] == pytest.approx(2 ** 0.5)
        assert std.std["b"] == 1.0                 # zero variance -> 1
        out = std.transform({"a": 3.0, "b": None})
        assert out[0] == pytest.approx(1.0 / 2 ** 0.5)
        assert out[1] is None                      # None stays None
        assert std.transform({})[0] is None        # absent feature -> None

    def test_all_missing_feature(self):
        std = pv.Standardizer().fit([{"a": None}], ["a"])
        assert (std.mean["a"], std.std["a"]) == (0.0, 1.0)

    def test_stats_come_from_fit_rows_only(self):
        std = pv.Standardizer().fit([{"a": 0.0}, {"a": 2.0}], ["a"])
        # a wildly different "test" row must not move the stats
        before = (std.mean["a"], std.std["a"])
        std.transform({"a": 1e9})
        assert (std.mean["a"], std.std["a"]) == before


# --------------------------------------------------------------------------- #
# DTE stratification
# --------------------------------------------------------------------------- #

class TestDteBuckets:
    def test_bucket_edges(self):
        assert pv.dte_bucket(None) == "unknown"
        assert pv.dte_bucket(0.0) == "0dte"
        assert pv.dte_bucket(0.99) == "0dte"
        assert pv.dte_bucket(1.0) == "short"
        assert pv.dte_bucket(7.0) == "short"
        assert pv.dte_bucket(7.01) == "swing"

    def test_stratify(self):
        samples = [
            _sample("a", SESSIONS[0], features={"dte_days": 0.5}),
            _sample("b", SESSIONS[0], features={"dte_days": 3.0}),
            _sample("c", SESSIONS[0], features={"dte_days": 30.0}),
            _sample("d", SESSIONS[0], features={"x": 1.0}),
        ]
        strat = pv.stratify_by_dte(samples)
        assert {k: [s.decision_id for s in v] for k, v in strat.items()} == {
            "0dte": ["a"], "short": ["b"], "swing": ["c"], "unknown": ["d"]}


# --------------------------------------------------------------------------- #
# metrics, fingerprint, bootstrap
# --------------------------------------------------------------------------- #

class TestMetrics:
    def test_perfect_and_empty(self):
        m = pv.binary_metrics([1, 0, 1], [0.9, 0.1, 0.8])
        assert m["accuracy"] == 1.0
        assert m["log_loss"] < 0.25
        assert m["base_rate"] == pytest.approx(2 / 3)
        empty = pv.binary_metrics([], [])
        assert empty == {"n": 0, "accuracy": None, "log_loss": None,
                         "brier": None, "base_rate": None}

    def test_fingerprint_is_canonical(self):
        a = {"x": 1, "y": [1, 2], "z": {"b": 2, "a": 1}}
        b = {"z": {"a": 1, "b": 2}, "y": [1, 2], "x": 1}
        assert pv.metrics_fingerprint(a) == pv.metrics_fingerprint(b)
        assert pv.metrics_fingerprint(a) != pv.metrics_fingerprint({"x": 2})

    def test_bootstrap_deterministic_and_signed(self):
        deltas = {s: 0.1 + i * 0.01 for i, s in enumerate(SESSIONS)}
        r1 = pv.paired_session_bootstrap(deltas, n_boot=500, seed=42)
        r2 = pv.paired_session_bootstrap(deltas, n_boot=500, seed=42)
        assert r1 == r2
        assert r1["ci_lo"] > 0                      # all-positive deltas
        assert r1["mean"] == pytest.approx(sum(deltas.values()) / len(deltas))
        assert pv.paired_session_bootstrap({})["mean"] is None


# --------------------------------------------------------------------------- #
# negative controls
# --------------------------------------------------------------------------- #

class TestNegativeControls:
    def test_leak_detected_and_shuffle_collapses(self):
        samples = _synthetic_samples(n_per_session=20, seed=0)
        folds, _, _ = pv.make_folds(samples, holdout_sessions=2)
        feature_names = sorted({k for s in samples for k in s.features})
        report = pv.run_negative_controls(samples, folds, feature_names,
                                          seed=42)
        ff, sl = report["future_feature"], report["shuffled_labels"]
        assert ff["folds"] == len(folds)
        # leaky probe must look superhuman -> the harness WOULD catch a leak
        assert ff["mean_test_accuracy"] >= 0.95
        assert ff["leak_detected"] is True
        # shuffled labels: apparent skill dies on noise
        assert sl["collapsed_to_chance"] is True
        assert report["all_passed"] is True

    def test_controls_deterministic(self):
        samples = _synthetic_samples(n_per_session=10, seed=1)
        folds, _, _ = pv.make_folds(samples, holdout_sessions=0)
        names = sorted({k for s in samples for k in s.features})
        r1 = pv.run_negative_controls(samples, folds, names, seed=7)
        r2 = pv.run_negative_controls(samples, folds, names, seed=7)
        assert pv.metrics_fingerprint(r1) == pv.metrics_fingerprint(r2)


# --------------------------------------------------------------------------- #
# holdout pre-registration (write-once)
# --------------------------------------------------------------------------- #

class TestRegisterHoldout:
    def test_write_then_verify(self, tmp_path):
        path = str(tmp_path / "holdout.json")
        rec = pv.register_holdout(path, "hash123", SESSIONS[-2:], 40)
        assert os.path.exists(path)
        assert rec["holdout_sessions"] == sorted(SESSIONS[-2:])
        assert "registered_at_utc" in rec
        # identical re-registration verifies and returns the pledge
        again = pv.register_holdout(path, "hash123", SESSIONS[-2:], 40)
        assert again["registered_at_utc"] == rec["registered_at_utc"]

    def test_mismatch_refuses_overwrite(self, tmp_path):
        path = str(tmp_path / "holdout.json")
        pv.register_holdout(path, "hash123", SESSIONS[-2:], 40)
        with pytest.raises(ValueError, match="holdout registration mismatch"):
            pv.register_holdout(path, "hash456", SESSIONS[-2:], 40)
        with pytest.raises(ValueError, match="holdout_sessions"):
            pv.register_holdout(path, "hash123", SESSIONS[-3:-1], 40)
        with pytest.raises(ValueError, match="n_samples"):
            pv.register_holdout(path, "hash123", SESSIONS[-2:], 41)
        # original pledge is untouched
        with open(path, encoding="utf-8") as f:
            assert json.load(f)["dataset_hash"] == "hash123"


# --------------------------------------------------------------------------- #
# store round-trip + build_report + CLI (end-to-end on a real store file)
# --------------------------------------------------------------------------- #

def _write_fixture_store(path: str, n_per_session: int = 20):
    rng = random.Random(0)
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(n_per_session):
            did = f"d-{session}-{i}"
            as_of = _ts(session, i)
            feats.append(FeatureRow(
                decision_id=did, snapshot_id=f"s-{did}", as_of=as_of,
                available_at=as_of, underlying="SPY", symbol=None,
                kind="candidate", session=session,
                features={"x": rng.uniform(-1, 1), "y": rng.uniform(-1, 1),
                          "dte_days": rng.choice([0.2, 3.0, 14.0])},
                quality={}))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="direction",
                value=rng.choice([-1.0, 1.0]), status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
    _write_store(path, {}, feats, labels, [],
                 {"dataset_hash": "fixturehash", "schema_version": "1"})


class TestEndToEnd:
    def test_load_samples_round_trip(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture_store(store, n_per_session=5)
        samples = pv.load_samples(store, "60m", "direction")
        assert len(samples) == 5 * len(SESSIONS)
        assert all(s.label_status == "ok" for s in samples)
        assert all(s.label_end is not None for s in samples)
        assert all(s.as_of.tzinfo is not None for s in samples)
        assert pv.load_samples(store, "60m", "direction",
                               kinds=["shadow_candidate"]) == []
        assert pv.dataset_meta(store)["dataset_hash"] == "fixturehash"

    def test_build_report_reproducible_fingerprint(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture_store(store)
        r1 = pv.build_report(store, "60m", "direction", holdout_sessions=2)
        r2 = pv.build_report(store, "60m", "direction", holdout_sessions=2)
        assert r1["report_fingerprint"] == r2["report_fingerprint"]
        assert r1["dataset_hash"] == "fixturehash"
        assert len(r1["folds"]) == 4
        assert r1["sessions"]["holdout"] == SESSIONS[-2:]
        assert r1["samples"]["label_ok"] == r1["samples"]["total"]
        assert set(r1["samples"]["dte_buckets_ok"]) == {"0dte", "short",
                                                        "swing"}
        assert r1["negative_controls"]["all_passed"] is True

    def test_cli_smoke_with_registration(self, tmp_path, capsys):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "folds.json")
        reg = str(tmp_path / "holdout.json")
        _write_fixture_store(store)
        rc = pv.main(["--dataset", store, "--horizon", "60m",
                      "--label-type", "direction", "--holdout-sessions", "2",
                      "--out", out, "--register-holdout", reg])
        assert rc == 0
        text = capsys.readouterr().out
        assert "report_fingerprint=" in text
        assert "holdout_registered" in text
        assert "all_passed=True" in text
        with open(out, encoding="utf-8") as f:
            report = json.load(f)
        assert report["config"]["embargo_sessions"] == 1   # 60m default
        with open(reg, encoding="utf-8") as f:
            pledge = json.load(f)
        assert pledge["holdout_sessions"] == SESSIONS[-2:]
        assert pledge["n_samples"] == 40
        # re-running the CLI verifies (not overwrites) the pledge
        assert pv.main(["--dataset", store, "--out", out,
                        "--register-holdout", reg,
                        "--holdout-sessions", "2"]) == 0
