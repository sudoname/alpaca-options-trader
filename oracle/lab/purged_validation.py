"""Purged, session-based walk-forward validation for Oracle 4.0 datasets.

Research-only: consumes the sqlite store written by ``oracle.ml.dataset``.
No production module imports this file.

Guarantees enforced here (per the master prompt):
  * folds split on whole exchange sessions, chronologically;
  * all samples of one ``decision_id`` stay on one side of every split;
  * training rows are PURGED when their label interval (label_end or the
    label's availability timestamp) reaches into the test window;
  * an EMBARGO of N sessions before the test window drops training rows
    whose label end is unknown (missing/censored labels cannot prove they
    finished before the test window);
  * a final chronological holdout is carved off FIRST and never enters any
    fold; it is pre-registered to a JSON file that refuses silent changes;
  * preprocessing statistics (standardization) are fit on fold-train only;
  * negative controls (future-feature leak, shuffled labels) validate that
    the harness can detect leakage and that apparent skill dies on noise.

Usage:
    python -m oracle.lab.purged_validation \
        --dataset artifacts/oracle4/dataset_v1.sqlite \
        --horizon 60m --label-type direction \
        --holdout-sessions 2 --out artifacts/oracle4/folds_60m.json \
        --register-holdout artifacts/oracle4/holdout_registration.json
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from oracle.ml.schema import parse_utc

__all__ = [
    "Sample", "Fold", "load_samples", "make_folds", "Standardizer",
    "dte_bucket", "stratify_by_dte", "binary_metrics",
    "metrics_fingerprint", "paired_session_bootstrap",
    "run_negative_controls", "register_holdout", "main",
]

# Embargo (in whole sessions) implied by each label horizon.  The embargo
# protects rows whose label end is unknown; rows with a known label_end are
# purged exactly.
HORIZON_EMBARGO_SESSIONS = {"15m": 1, "60m": 1, "1s": 1, "5s": 5, "event": 1}


# --------------------------------------------------------------------------- #
# samples
# --------------------------------------------------------------------------- #

@dataclass
class Sample:
    decision_id: str
    session: str                      # YYYY-MM-DD (UTC) session key
    as_of: datetime
    kind: str
    features: Dict[str, Optional[float]]
    label_value: Optional[float]
    label_status: str                 # ok | censored | missing
    label_end: Optional[datetime]
    label_available_at: Optional[datetime]


def load_samples(dataset_path: str, horizon: str, label_type: str,
                 kinds: Optional[Sequence[str]] = None) -> List[Sample]:
    """Join features+labels for one (horizon, label_type) from the store."""
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    try:
        kinds_set = set(kinds) if kinds else None
        out: List[Sample] = []
        cur = conn.execute(
            """SELECT f.decision_id, f.session, f.as_of, f.kind,
                      f.features_json, l.value, l.status,
                      l.label_end, l.available_at
                 FROM features f JOIN labels l
                   ON l.decision_id = f.decision_id
                WHERE l.horizon = ? AND l.label_type = ?
                ORDER BY f.as_of, f.decision_id""",
            (horizon, label_type))
        for row in cur:
            if kinds_set is not None and row["kind"] not in kinds_set:
                continue
            as_of = parse_utc(row["as_of"])
            if as_of is None:
                continue
            out.append(Sample(
                decision_id=row["decision_id"],
                session=row["session"],
                as_of=as_of,
                kind=row["kind"],
                features=json.loads(row["features_json"]),
                label_value=row["value"],
                label_status=row["status"],
                label_end=parse_utc(row["label_end"]),
                label_available_at=parse_utc(row["available_at"]),
            ))
        return out
    finally:
        conn.close()


def dataset_meta(dataset_path: str) -> Dict[str, str]:
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        return dict(conn.execute("SELECT key, value FROM meta"))
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# folds
# --------------------------------------------------------------------------- #

@dataclass
class Fold:
    fold_id: int
    train_sessions: List[str]
    test_sessions: List[str]
    train_ids: List[str]
    test_ids: List[str]
    purged: int = 0                   # label interval reached test window
    embargoed: int = 0                # unknown label end inside embargo zone
    purge_detail: Dict[str, int] = field(default_factory=dict)


def _session_start(session: str) -> datetime:
    return datetime.fromisoformat(session).replace(tzinfo=timezone.utc)


def make_folds(samples: Sequence[Sample],
               test_block_sessions: int = 1,
               min_train_sessions: int = 2,
               embargo_sessions: int = 1,
               holdout_sessions: int = 0,
               ) -> Tuple[List[Fold], List[str], List[str]]:
    """Build expanding-window purged walk-forward folds.

    Returns ``(folds, fold_sessions, holdout_session_keys)``.  The last
    ``holdout_sessions`` sessions are removed before any fold is formed.
    """
    sessions = sorted({s.session for s in samples})
    if holdout_sessions > 0:
        holdout = sessions[-holdout_sessions:]
        sessions = sessions[:-holdout_sessions]
    else:
        holdout = []
    by_session: Dict[str, List[Sample]] = {}
    for s in samples:
        if s.session in holdout:
            continue
        by_session.setdefault(s.session, []).append(s)

    folds: List[Fold] = []
    fold_id = 0
    i = min_train_sessions
    while i + 1 <= len(sessions) - 0 and i < len(sessions):
        test_sess = sessions[i:i + test_block_sessions]
        if not test_sess:
            break
        train_sess = sessions[:i]
        test_start = _session_start(test_sess[0])
        embargo_cut = set(sessions[max(0, i - embargo_sessions):i])

        train_ids: List[str] = []
        purged = embargoed = 0
        detail = {"label_end_overlap": 0, "availability_overlap": 0,
                  "unknown_end_in_embargo": 0}
        for sess in train_sess:
            for s in by_session.get(sess, []):
                end_known = s.label_end is not None
                if end_known and s.label_end >= test_start:
                    purged += 1
                    detail["label_end_overlap"] += 1
                    continue
                if (s.label_available_at is not None
                        and s.label_available_at >= test_start):
                    purged += 1
                    detail["availability_overlap"] += 1
                    continue
                if not end_known and sess in embargo_cut:
                    embargoed += 1
                    detail["unknown_end_in_embargo"] += 1
                    continue
                train_ids.append(s.decision_id)

        test_ids = [s.decision_id
                    for sess in test_sess
                    for s in by_session.get(sess, [])]
        overlap = set(train_ids) & set(test_ids)
        if overlap:                   # grouped by decision_id — must not happen
            raise AssertionError(
                f"decision groups leak across fold {fold_id}: "
                f"{sorted(overlap)[:3]}")
        if train_ids and test_ids:
            folds.append(Fold(
                fold_id=fold_id, train_sessions=list(train_sess),
                test_sessions=list(test_sess), train_ids=train_ids,
                test_ids=test_ids, purged=purged, embargoed=embargoed,
                purge_detail=detail))
            fold_id += 1
        i += test_block_sessions
    return folds, sessions, holdout


# --------------------------------------------------------------------------- #
# fold-train-only preprocessing
# --------------------------------------------------------------------------- #

class Standardizer:
    """Per-feature mean/std fit on training rows only.  None stays None."""

    def __init__(self) -> None:
        self.mean: Dict[str, float] = {}
        self.std: Dict[str, float] = {}
        self.feature_names: List[str] = []

    def fit(self, rows: Sequence[Dict[str, Optional[float]]],
            feature_names: Optional[Sequence[str]] = None) -> "Standardizer":
        names = list(feature_names) if feature_names else sorted(
            {k for r in rows for k in r})
        self.feature_names = names
        for name in names:
            vals = [r[name] for r in rows
                    if r.get(name) is not None
                    and isinstance(r[name], (int, float))
                    and r[name] == r[name]]
            if not vals:
                self.mean[name] = 0.0
                self.std[name] = 1.0
                continue
            m = sum(vals) / len(vals)
            var = (sum((v - m) ** 2 for v in vals) / (len(vals) - 1)
                   if len(vals) > 1 else 0.0)
            self.mean[name] = m
            self.std[name] = math.sqrt(var) if var > 0 else 1.0
        return self

    def transform(self, row: Dict[str, Optional[float]]
                  ) -> List[Optional[float]]:
        out: List[Optional[float]] = []
        for name in self.feature_names:
            v = row.get(name)
            if v is None or not isinstance(v, (int, float)) or v != v:
                out.append(None)
            else:
                out.append((v - self.mean[name]) / self.std[name])
        return out


# --------------------------------------------------------------------------- #
# DTE stratification
# --------------------------------------------------------------------------- #

def dte_bucket(dte_days: Optional[float]) -> str:
    if dte_days is None:
        return "unknown"
    if dte_days < 1.0:
        return "0dte"
    if dte_days <= 7.0:
        return "short"
    return "swing"


def stratify_by_dte(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    out: Dict[str, List[Sample]] = {}
    for s in samples:
        out.setdefault(dte_bucket(s.features.get("dte_days")), []).append(s)
    return out


# --------------------------------------------------------------------------- #
# metrics + deterministic fingerprint
# --------------------------------------------------------------------------- #

def binary_metrics(y_true: Sequence[int], p_up: Sequence[float]
                   ) -> Dict[str, Optional[float]]:
    """Accuracy / log loss / Brier for binary (up vs not-up) predictions."""
    n = len(y_true)
    if n == 0:
        return {"n": 0, "accuracy": None, "log_loss": None,
                "brier": None, "base_rate": None}
    eps = 1e-12
    correct = ll = brier = 0.0
    for y, p in zip(y_true, p_up):
        p = min(max(p, eps), 1.0 - eps)
        pred = 1 if p >= 0.5 else 0
        correct += 1.0 if pred == y else 0.0
        ll += -(y * math.log(p) + (1 - y) * math.log(1.0 - p))
        brier += (p - y) ** 2
    return {"n": n,
            "accuracy": correct / n,
            "log_loss": ll / n,
            "brier": brier / n,
            "base_rate": sum(y_true) / n}


def metrics_fingerprint(obj: Any) -> str:
    """sha256 of canonical JSON — bit-for-bit parity checks."""
    blob = json.dumps(obj, sort_keys=True, separators=(",", ":"),
                      default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def paired_session_bootstrap(per_session_delta: Dict[str, float],
                             n_boot: int = 2000,
                             seed: int = 42) -> Dict[str, Optional[float]]:
    """Block bootstrap over sessions of a paired metric delta.

    Resamples whole sessions with replacement; returns mean and 95% CI.
    """
    sessions = sorted(per_session_delta)
    if not sessions:
        return {"n_sessions": 0, "mean": None, "ci_lo": None, "ci_hi": None}
    vals = [per_session_delta[s] for s in sessions]
    rng = random.Random(seed)
    means = []
    for _ in range(n_boot):
        draw = [vals[rng.randrange(len(vals))] for _ in vals]
        means.append(sum(draw) / len(draw))
    means.sort()
    lo = means[int(0.025 * (n_boot - 1))]
    hi = means[int(0.975 * (n_boot - 1))]
    return {"n_sessions": len(sessions),
            "mean": sum(vals) / len(vals), "ci_lo": lo, "ci_hi": hi}


# --------------------------------------------------------------------------- #
# tiny deterministic logistic learner (negative-control probe only)
# --------------------------------------------------------------------------- #

def _fit_logistic(x_rows: List[List[Optional[float]]], y: List[int],
                  iters: int = 300, lr: float = 0.5, l2: float = 1e-3
                  ) -> List[float]:
    """Plain-python ridge logistic (None -> 0 after standardization).

    Deterministic; used only to probe the harness in negative controls.
    Real challengers arrive in Phase 3.
    """
    if not x_rows:
        return []
    d = len(x_rows[0])
    xs = [[0.0 if v is None else v for v in r] + [1.0] for r in x_rows]
    w = [0.0] * (d + 1)
    n = len(xs)
    for _ in range(iters):
        grad = [0.0] * (d + 1)
        for xi, yi in zip(xs, y):
            z = sum(wj * xj for wj, xj in zip(w, xi))
            z = max(min(z, 35.0), -35.0)
            p = 1.0 / (1.0 + math.exp(-z))
            err = p - yi
            for j in range(d + 1):
                grad[j] += err * xi[j]
        for j in range(d + 1):
            w[j] -= lr * (grad[j] / n + l2 * w[j])
    return w


def _predict_logistic(w: List[float], x_rows: List[List[Optional[float]]]
                      ) -> List[float]:
    out = []
    for r in x_rows:
        xi = [0.0 if v is None else v for v in r] + [1.0]
        z = sum(wj * xj for wj, xj in zip(w, xi))
        z = max(min(z, 35.0), -35.0)
        out.append(1.0 / (1.0 + math.exp(-z)))
    return out


def run_negative_controls(samples: Sequence[Sample], folds: Sequence[Fold],
                          feature_names: Sequence[str],
                          seed: int = 42) -> Dict[str, Any]:
    """Future-feature and shuffled-label controls over the given folds.

    * ``future_feature``: injects the true label as a feature.  The probe
      must score near-perfectly — proving the harness WOULD expose a leak.
    * ``shuffled_labels``: permutes training labels.  Test accuracy must
      collapse towards the base rate — apparent skill on noise fails.
    Both verdicts are returned; callers treat a failed control as a broken
    evaluation setup, not as model information.
    """
    by_id = {s.decision_id: s for s in samples}

    def _xy(ids: Sequence[str], std: Standardizer, leak: bool,
            label_map: Optional[Dict[str, int]] = None):
        xs, ys = [], []
        for did in ids:
            s = by_id[did]
            if s.label_status != "ok" or s.label_value is None:
                continue
            y = 1 if s.label_value > 0 else 0
            if label_map is not None:
                y = label_map[did]
            x = std.transform(s.features)
            if leak:
                x = x + [float(1 if s.label_value > 0 else 0) * 4.0 - 2.0]
            xs.append(x)
            ys.append(y)
        return xs, ys

    rng = random.Random(seed)
    leak_metrics, shuf_metrics, true_rates = [], [], []
    for fold in folds:
        train_rows = [by_id[d].features for d in fold.train_ids
                      if by_id[d].label_status == "ok"]
        if not train_rows:
            continue
        std = Standardizer().fit(train_rows, feature_names)

        # -- future-feature leak probe
        xtr, ytr = _xy(fold.train_ids, std, leak=True)
        xte, yte = _xy(fold.test_ids, std, leak=True)
        if len(set(ytr)) == 2 and yte:
            w = _fit_logistic(xtr, ytr)
            leak_metrics.append(binary_metrics(yte, _predict_logistic(w, xte)))

        # -- shuffled-label probe
        ok_train = [d for d in fold.train_ids
                    if by_id[d].label_status == "ok"
                    and by_id[d].label_value is not None]
        ys = [1 if by_id[d].label_value > 0 else 0 for d in ok_train]
        shuffled = ys[:]
        rng.shuffle(shuffled)
        label_map = dict(zip(ok_train, shuffled))
        xtr, ytr = _xy(ok_train, std, leak=False, label_map=label_map)
        xte, yte = _xy(fold.test_ids, std, leak=False)
        if len(set(ytr)) == 2 and yte:
            w = _fit_logistic(xtr, ytr)
            shuf_metrics.append(binary_metrics(yte, _predict_logistic(w, xte)))
            true_rates.append(sum(yte) / len(yte))

    def _mean(ms, key):
        vals = [m[key] for m in ms if m[key] is not None]
        return sum(vals) / len(vals) if vals else None

    leak_acc = _mean(leak_metrics, "accuracy")
    shuf_acc = _mean(shuf_metrics, "accuracy")
    base = (sum(max(r, 1.0 - r) for r in true_rates) / len(true_rates)
            if true_rates else None)
    report = {
        "future_feature": {
            "folds": len(leak_metrics),
            "mean_test_accuracy": leak_acc,
            # the harness detects the leak iff the leaky probe looks superhuman
            "leak_detected": (leak_acc is not None and leak_acc >= 0.95),
        },
        "shuffled_labels": {
            "folds": len(shuf_metrics),
            "mean_test_accuracy": shuf_acc,
            "majority_base_rate": base,
            # skill must collapse: no better than majority + small margin
            "collapsed_to_chance": (
                shuf_acc is not None and base is not None
                and shuf_acc <= base + 0.05),
        },
    }
    report["all_passed"] = bool(
        report["future_feature"]["leak_detected"]
        and report["shuffled_labels"]["collapsed_to_chance"])
    return report


# --------------------------------------------------------------------------- #
# holdout pre-registration
# --------------------------------------------------------------------------- #

def register_holdout(path: str, dataset_hash: str,
                     holdout_sessions: Sequence[str],
                     n_samples: int) -> Dict[str, Any]:
    """Write-once holdout registration; verify (never overwrite) on re-run.

    Raises ``ValueError`` if an existing registration for the same dataset
    disagrees — the holdout must not silently change after being pledged.
    """
    record = {
        "dataset_hash": dataset_hash,
        "holdout_sessions": sorted(holdout_sessions),
        "n_samples": n_samples,
        "pledge": ("These sessions are reserved for one final evaluation "
                   "after all model and hyperparameter choices are frozen. "
                   "They must not be used for training, tuning, feature "
                   "selection, or early stopping."),
    }
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            existing = json.load(f)
        for key in ("dataset_hash", "holdout_sessions", "n_samples"):
            if existing.get(key) != record[key]:
                raise ValueError(
                    f"holdout registration mismatch on '{key}': "
                    f"registered={existing.get(key)!r} vs now={record[key]!r}"
                    " -- refusing to overwrite a pledged holdout")
        return existing
    record["registered_at_utc"] = datetime.now(timezone.utc).isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, indent=2, sort_keys=True)
    return record


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def build_report(dataset_path: str, horizon: str, label_type: str,
                 kinds: Optional[Sequence[str]] = None,
                 test_block_sessions: int = 1,
                 min_train_sessions: int = 2,
                 embargo_sessions: Optional[int] = None,
                 holdout_sessions: int = 2,
                 seed: int = 42,
                 run_controls: bool = True) -> Dict[str, Any]:
    samples = load_samples(dataset_path, horizon, label_type, kinds=kinds)
    meta = dataset_meta(dataset_path)
    embargo = (embargo_sessions if embargo_sessions is not None
               else HORIZON_EMBARGO_SESSIONS.get(horizon, 1))
    folds, fold_sessions, holdout = make_folds(
        samples, test_block_sessions=test_block_sessions,
        min_train_sessions=min_train_sessions,
        embargo_sessions=embargo, holdout_sessions=holdout_sessions)

    feature_names = sorted({k for s in samples for k in s.features})
    ok = [s for s in samples if s.label_status == "ok"]
    strat = {b: len(v) for b, v in stratify_by_dte(ok).items()}

    report: Dict[str, Any] = {
        "dataset_path": dataset_path,
        "dataset_hash": meta.get("dataset_hash"),
        "horizon": horizon,
        "label_type": label_type,
        "kinds": sorted(kinds) if kinds else None,
        "config": {
            "test_block_sessions": test_block_sessions,
            "min_train_sessions": min_train_sessions,
            "embargo_sessions": embargo,
            "holdout_sessions": holdout_sessions,
            "seed": seed,
        },
        "sessions": {"folds": fold_sessions, "holdout": holdout},
        "samples": {
            "total": len(samples),
            "label_ok": len(ok),
            "label_censored": sum(1 for s in samples
                                  if s.label_status == "censored"),
            "label_missing": sum(1 for s in samples
                                 if s.label_status == "missing"),
            "dte_buckets_ok": strat,
        },
        "folds": [
            {"fold_id": f.fold_id,
             "train_sessions": f.train_sessions,
             "test_sessions": f.test_sessions,
             "n_train": len(f.train_ids),
             "n_test": len(f.test_ids),
             "purged": f.purged,
             "embargoed": f.embargoed,
             "purge_detail": f.purge_detail}
            for f in folds
        ],
        "feature_names": feature_names,
    }
    if run_controls and folds:
        report["negative_controls"] = run_negative_controls(
            samples, folds, feature_names, seed=seed)
    report["report_fingerprint"] = metrics_fingerprint(
        {k: v for k, v in report.items() if k != "report_fingerprint"})
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Purged session walk-forward folds for an Oracle 4.0 "
                    "dataset store (research-only)")
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", default="60m",
                   choices=["15m", "60m", "1s", "5s", "event"])
    p.add_argument("--label-type", default="direction")
    p.add_argument("--kinds", nargs="*", default=None,
                   choices=["candidate", "shadow_candidate"])
    p.add_argument("--test-block", type=int, default=1)
    p.add_argument("--min-train", type=int, default=2)
    p.add_argument("--embargo-sessions", type=int, default=None,
                   help="default: implied by --horizon")
    p.add_argument("--holdout-sessions", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--no-controls", action="store_true")
    p.add_argument("--out", required=True)
    p.add_argument("--register-holdout", default=None,
                   help="write-once holdout registration JSON")
    args = p.parse_args(argv)

    report = build_report(
        args.dataset, args.horizon, args.label_type, kinds=args.kinds,
        test_block_sessions=args.test_block,
        min_train_sessions=args.min_train,
        embargo_sessions=args.embargo_sessions,
        holdout_sessions=args.holdout_sessions,
        seed=args.seed, run_controls=not args.no_controls)

    if args.register_holdout:
        n_hold = 0
        hold = set(report["sessions"]["holdout"])
        samples = load_samples(args.dataset, args.horizon, args.label_type,
                               kinds=args.kinds)
        n_hold = sum(1 for s in samples if s.session in hold)
        register_holdout(args.register_holdout,
                         report["dataset_hash"] or "",
                         report["sessions"]["holdout"], n_hold)
        print(f"holdout_registered sessions={sorted(hold)} n={n_hold}")

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(f"folds={len(report['folds'])} "
          f"sessions={len(report['sessions']['folds'])} "
          f"holdout={report['sessions']['holdout']}")
    for fr in report["folds"]:
        print(f"  fold {fr['fold_id']}: train={fr['n_train']} "
              f"test={fr['n_test']} purged={fr['purged']} "
              f"embargoed={fr['embargoed']} test_sessions={fr['test_sessions']}")
    nc = report.get("negative_controls")
    if nc:
        print(f"controls: leak_detected={nc['future_feature']['leak_detected']} "
              f"collapsed_to_chance="
              f"{nc['shuffled_labels']['collapsed_to_chance']} "
              f"all_passed={nc['all_passed']}")
    print(f"report_fingerprint={report['report_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
