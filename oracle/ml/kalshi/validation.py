"""Matched Kalshi ablation experiments A-E (Oracle 4.0.2 P1-6).

Does Kalshi information add predictive value over the existing feature
set?  Five experiments share *identical* samples, folds, learner, and
hyperparameters -- the ONLY difference is the feature dict per sample:

  * A ``base``          existing dataset features only (control);
  * B ``base+kalshi``   A plus PIT kalshi_* features;
  * C ``kalshi_only``   kalshi_* features alone;
  * D ``base+placebo``  A plus kalshi vectors permuted across samples
                        (seeded) -- any "improvement" here is leakage or
                        overfitting, not signal;
  * E ``base+lagged``   A plus kalshi features computed at
                        ``as_of - lag_min`` (staleness sensitivity).

Learner: deterministic closed-form ridge regression on standardized
features (pure python, no numpy), identical settings everywhere.
Evaluation: pooled OOS per purged/embargoed walk-forward fold from
``oracle.lab.purged_validation.make_folds``; paired per-session deltas
vs A with session block-bootstrap CIs.

Hard rules: settlement results are never inputs (the snapshot store has
no result column); labels come only from the Oracle dataset store; the
verdict is research evidence only -- never an automatic promotion.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from oracle.lab.purged_validation import (Fold, Sample, Standardizer,
                                          load_samples, make_folds,
                                          metrics_fingerprint,
                                          paired_session_bootstrap)

from .feature_builder import (DEFAULT_MAX_STALENESS_MIN, KALSHI_FEATURES,
                              kalshi_features_for_samples)

UTC = timezone.utc

EXPERIMENTS: Tuple[str, ...] = (
    "A_base", "B_base_plus_kalshi", "C_kalshi_only",
    "D_base_plus_placebo", "E_base_plus_lagged")

DEFAULT_RIDGE_LAMBDA = 1.0
DEFAULT_MIN_COVERAGE = 0.5
DEFAULT_MIN_SAMPLES = 40
DEFAULT_LAG_MIN = 60.0
DEFAULT_SEED = 7


# --------------------------------------------------------------------------- #
# deterministic ridge (pure python)
# --------------------------------------------------------------------------- #

def _solve(a: List[List[float]], b: List[float]) -> Optional[List[float]]:
    """Gaussian elimination with partial pivoting; None if singular."""
    n = len(b)
    m = [row[:] + [b[i]] for i, row in enumerate(a)]
    for col in range(n):
        piv = max(range(col, n), key=lambda r: abs(m[r][col]))
        if abs(m[piv][col]) < 1e-12:
            return None
        m[col], m[piv] = m[piv], m[col]
        inv = 1.0 / m[col][col]
        for r in range(n):
            if r == col:
                continue
            f = m[r][col] * inv
            if f == 0.0:
                continue
            for c in range(col, n + 1):
                m[r][c] -= f * m[col][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def _ridge_fit(x_rows: List[List[float]], y: List[float],
               lam: float) -> List[float]:
    """w = argmin ||Xw - y||^2 + lam||w||^2 (bias column unpenalized)."""
    d = len(x_rows[0])
    a = [[0.0] * d for _ in range(d)]
    b = [0.0] * d
    for row, yv in zip(x_rows, y):
        for i in range(d):
            ri = row[i]
            if ri == 0.0:
                continue
            b[i] += ri * yv
            ai = a[i]
            for j in range(d):
                ai[j] += ri * row[j]
    for i in range(d - 1):          # last column is the bias
        a[i][i] += lam
    a[d - 1][d - 1] += 1e-9
    w = _solve(a, b)
    return w if w is not None else [0.0] * d


def _predict(w: Sequence[float], row: Sequence[float]) -> float:
    return sum(wi * xi for wi, xi in zip(w, row))


def _vectorize(std: Standardizer, row: Dict[str, Optional[float]]
               ) -> List[float]:
    """Standardize then mean-impute (None -> 0.0 == train mean); +bias."""
    vec = [0.0 if v is None else v for v in std.transform(row)]
    vec.append(1.0)
    return vec


# --------------------------------------------------------------------------- #
# experiment evaluation
# --------------------------------------------------------------------------- #

def _eval_experiment(
    rows: Sequence[Dict[str, Optional[float]]],
    samples: Sequence[Sample],
    folds: Sequence[Fold],
    *,
    ridge_lambda: float,
) -> Dict:
    """Walk-forward OOS evaluation of one feature representation.

    Returns pooled metrics plus per-sample OOS squared errors and
    direction hits keyed by decision_id (for paired deltas).
    """
    by_id = {s.decision_id: i for i, s in enumerate(samples)}
    sq_err: Dict[str, float] = {}
    dir_hit: Dict[str, Optional[int]] = {}
    session_of: Dict[str, str] = {}
    n_folds_used = 0
    for fold in folds:
        train_idx = [by_id[i] for i in fold.train_ids if i in by_id]
        test_idx = [by_id[i] for i in fold.test_ids if i in by_id]
        if len(train_idx) < 5 or not test_idx:
            continue
        std = Standardizer().fit([rows[i] for i in train_idx])
        if not std.feature_names:
            continue
        x_tr = [_vectorize(std, rows[i]) for i in train_idx]
        y_tr = [float(samples[i].label_value) for i in train_idx]
        w = _ridge_fit(x_tr, y_tr, ridge_lambda)
        n_folds_used += 1
        for i in test_idx:
            s = samples[i]
            pred = _predict(w, _vectorize(std, rows[i]))
            y = float(s.label_value)
            sq_err[s.decision_id] = (pred - y) ** 2
            session_of[s.decision_id] = s.session
            if y != 0.0:
                dir_hit[s.decision_id] = (
                    1 if (pred > 0.0) == (y > 0.0) else 0)
            else:
                dir_hit[s.decision_id] = None
    n = len(sq_err)
    hits = [h for h in dir_hit.values() if h is not None]
    return {
        "n_oos": n,
        "n_folds_used": n_folds_used,
        "mse": (sum(sq_err.values()) / n) if n else None,
        "directional_accuracy": (sum(hits) / len(hits)) if hits else None,
        "n_directional": len(hits),
        "_sq_err": sq_err,
        "_dir_hit": dir_hit,
        "_session_of": session_of,
    }


def _paired_delta(base: Dict, other: Dict, *, n_boot: int, seed: int
                  ) -> Dict:
    """Session-level paired deltas (base - other; positive = other wins)."""
    common = sorted(set(base["_sq_err"]) & set(other["_sq_err"]))
    per_session_mse: Dict[str, List[float]] = {}
    per_session_dir: Dict[str, List[float]] = {}
    for did in common:
        sess = base["_session_of"][did]
        per_session_mse.setdefault(sess, []).append(
            base["_sq_err"][did] - other["_sq_err"][did])
        hb, ho = base["_dir_hit"].get(did), other["_dir_hit"].get(did)
        if hb is not None and ho is not None:
            per_session_dir.setdefault(sess, []).append(float(ho - hb))
    mse_delta = {s: sum(v) / len(v) for s, v in per_session_mse.items()}
    dir_delta = {s: sum(v) / len(v) for s, v in per_session_dir.items()}
    return {
        "n_paired": len(common),
        "mse_improvement": paired_session_bootstrap(
            mse_delta, n_boot=n_boot, seed=seed),
        "directional_improvement": paired_session_bootstrap(
            dir_delta, n_boot=n_boot, seed=seed),
    }


# --------------------------------------------------------------------------- #
# ablation driver
# --------------------------------------------------------------------------- #

def run_ablation_on_samples(
    samples: Sequence[Sample],
    kalshi_db: str,
    underlying: str,
    *,
    lag_min: float = DEFAULT_LAG_MIN,
    max_staleness_min: float = DEFAULT_MAX_STALENESS_MIN,
    index_spot_fn: Optional[Callable] = None,
    ridge_lambda: float = DEFAULT_RIDGE_LAMBDA,
    min_coverage: float = DEFAULT_MIN_COVERAGE,
    min_samples: int = DEFAULT_MIN_SAMPLES,
    seed: int = DEFAULT_SEED,
    n_boot: int = 2000,
    test_block_sessions: int = 1,
    min_train_sessions: int = 2,
    embargo_sessions: int = 1,
) -> Dict:
    """Run matched experiments A-E on pre-loaded samples.

    Returns the full ablation report dict (verdict, per-experiment
    metrics, paired deltas vs A, coverage).  Never raises on sparse
    data -- the verdict degrades to INSUFFICIENT_DATA instead.
    """
    usable = [s for s in samples
              if s.label_status == "ok" and s.label_value is not None
              and math.isfinite(float(s.label_value))]
    report: Dict = {
        "stage": "oracle4.0.2-kalshi-ablation",
        "generated_at_utc": datetime.now(UTC).strftime(
            "%Y-%m-%dT%H:%M:%SZ"),
        "underlying": str(underlying).upper(),
        "kalshi_db": kalshi_db,
        "params": {
            "lag_min": lag_min,
            "max_staleness_min": max_staleness_min,
            "ridge_lambda": ridge_lambda,
            "min_coverage": min_coverage,
            "min_samples": min_samples,
            "seed": seed,
            "n_boot": n_boot,
            "test_block_sessions": test_block_sessions,
            "min_train_sessions": min_train_sessions,
            "embargo_sessions": embargo_sessions,
        },
        "n_samples_total": len(samples),
        "n_samples_usable": len(usable),
    }
    if len(usable) < int(min_samples):
        report["verdict"] = "INSUFFICIENT_DATA"
        report["verdict_reason"] = (
            f"only {len(usable)} usable samples (< {min_samples})")
        return report

    # kalshi feature vectors: live, lagged, placebo (seeded permutation).
    kalshi_now = kalshi_features_for_samples(
        usable, kalshi_db, underlying,
        max_staleness_min=max_staleness_min, index_spot_fn=index_spot_fn)
    kalshi_lag = kalshi_features_for_samples(
        usable, kalshi_db, underlying,
        max_staleness_min=max_staleness_min, lag_min=lag_min,
        index_spot_fn=index_spot_fn)
    covered = sum(1 for f in kalshi_now
                  if f.get("kalshi_staleness_min") is not None)
    coverage = covered / len(usable)
    report["kalshi_coverage"] = coverage
    report["kalshi_feature_names"] = list(KALSHI_FEATURES)
    if coverage < float(min_coverage):
        report["verdict"] = "INSUFFICIENT_DATA"
        report["verdict_reason"] = (
            f"kalshi coverage {coverage:.3f} < {min_coverage} "
            f"({covered}/{len(usable)} samples with a fresh snapshot)")
        return report

    rng = random.Random(seed)
    perm = list(range(len(usable)))
    rng.shuffle(perm)
    kalshi_placebo = [kalshi_now[j] for j in perm]

    rows_by_exp: Dict[str, List[Dict[str, Optional[float]]]] = {
        "A_base": [dict(s.features) for s in usable],
        "B_base_plus_kalshi": [
            {**s.features, **kalshi_now[i]}
            for i, s in enumerate(usable)],
        "C_kalshi_only": [dict(kalshi_now[i])
                          for i in range(len(usable))],
        "D_base_plus_placebo": [
            {**s.features, **kalshi_placebo[i]}
            for i, s in enumerate(usable)],
        "E_base_plus_lagged": [
            {**s.features, **kalshi_lag[i]}
            for i, s in enumerate(usable)],
    }

    folds, fold_sessions, _ = make_folds(
        usable, test_block_sessions=test_block_sessions,
        min_train_sessions=min_train_sessions,
        embargo_sessions=embargo_sessions, holdout_sessions=0)
    report["n_folds"] = len(folds)
    report["n_sessions"] = len(fold_sessions)
    if not folds:
        report["verdict"] = "INSUFFICIENT_DATA"
        report["verdict_reason"] = "no walk-forward folds (too few sessions)"
        return report

    results: Dict[str, Dict] = {}
    for name in EXPERIMENTS:
        results[name] = _eval_experiment(
            rows_by_exp[name], usable, folds, ridge_lambda=ridge_lambda)
    base = results["A_base"]
    deltas = {name: _paired_delta(base, results[name],
                                  n_boot=n_boot, seed=seed)
              for name in EXPERIMENTS if name != "A_base"}

    report["experiments"] = {
        name: {k: v for k, v in res.items() if not k.startswith("_")}
        for name, res in results.items()}
    report["deltas_vs_A"] = deltas
    if base["n_oos"] == 0:
        report["verdict"] = "INSUFFICIENT_DATA"
        report["verdict_reason"] = "no OOS predictions in any fold"
        return report

    b_mse = deltas["B_base_plus_kalshi"]["mse_improvement"]
    d_mse = deltas["D_base_plus_placebo"]["mse_improvement"]
    b_sig = (b_mse["ci_lo"] is not None and b_mse["ci_lo"] > 0.0)
    d_clean = (d_mse["ci_lo"] is None or d_mse["ci_lo"] <= 0.0)
    if b_sig and d_clean:
        report["verdict"] = "SIGNAL_CANDIDATE"
        report["verdict_reason"] = (
            "B improves OOS MSE over A (95% CI low > 0) and the "
            "permuted placebo D does not")
    elif b_sig and not d_clean:
        report["verdict"] = "SUSPECT_LEAKAGE"
        report["verdict_reason"] = (
            "placebo D also 'improves' over A -- treat B's gain as "
            "overfitting/leakage, not signal")
    else:
        report["verdict"] = "NO_SIGNAL"
        report["verdict_reason"] = (
            "B does not significantly improve OOS MSE over A")
    report["fingerprint"] = metrics_fingerprint(
        {"experiments": report["experiments"],
         "deltas_vs_A": report["deltas_vs_A"]})
    return report


def run_ablation(
    dataset_path: str,
    kalshi_db: str,
    underlying: str,
    *,
    horizon: str = "15m",
    label_type: str = "log_return",
    kinds: Optional[Sequence[str]] = None,
    output_path: Optional[str] = None,
    **kwargs,
) -> Dict:
    """Load samples from an Oracle dataset store and run the ablation."""
    samples = load_samples(dataset_path, horizon, label_type, kinds=kinds)
    report = run_ablation_on_samples(samples, kalshi_db, underlying,
                                     **kwargs)
    report["dataset_path"] = dataset_path
    report["horizon"] = horizon
    report["label_type"] = label_type
    report["kinds"] = list(kinds) if kinds else None
    if output_path:
        d = os.path.dirname(os.path.abspath(output_path))
        if d:
            os.makedirs(d, exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2, sort_keys=True)
    return report


def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oracle.ml.kalshi.validation",
        description=("Matched Kalshi ablation A-E (research-only; "
                     "verdict is evidence, never a promotion)."))
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--kalshi-db", required=True)
    ap.add_argument("--underlying", required=True)
    ap.add_argument("--horizon", default="15m")
    ap.add_argument("--label-type", default="log_return")
    ap.add_argument("--kinds", nargs="*", default=None)
    ap.add_argument("--lag-min", type=float, default=DEFAULT_LAG_MIN)
    ap.add_argument("--output", default=os.path.join(
        "artifacts", "oracle402", "kalshi_ablation.json"))
    args = ap.parse_args(argv)
    report = run_ablation(
        args.dataset, args.kalshi_db, args.underlying,
        horizon=args.horizon, label_type=args.label_type,
        kinds=args.kinds, lag_min=args.lag_min, output_path=args.output)
    print(f"verdict={report['verdict']}")
    print(f"reason={report.get('verdict_reason')}")
    for name, res in (report.get("experiments") or {}).items():
        print(f"{name}: n_oos={res['n_oos']} mse={res['mse']} "
              f"dir_acc={res['directional_accuracy']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
