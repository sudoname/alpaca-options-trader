"""Oracle 4.0 Phases 3–4 — challenger training/evaluation CLI.

Research-only.  Nothing in the production trading path imports this.

Every challenger is evaluated through the Phase 2 purged session
walk-forward harness (``oracle.lab.purged_validation``): expanding-window
folds over whole sessions, purging + embargo, train-only standardization,
and a pre-registered holdout that is carved off and **never** evaluated
here.

Models:
  * logistic  — ridge logistic direction classifier with Platt scaling
                fit on a chronological tail of train; compared against
                the naive comparators with paired session bootstrap.
  * naive     — the comparators alone (constant base rate, momentum
                rule, production ``baseline_p_win``).
  * quantile  — pinball quantile ladder on log returns vs marginal
                train quantiles.
  * rv        — HAR-lite realized-vol forecast vs persistence and EWMA.
  * hazard    — discrete-time competing risks (stop/target/other_exit)
                vs a marginal per-bin hazard table.
  * gbm       — optional dependency; reports "unavailable" when
                lightgbm is not installed (never faked).
  * mlp       — Phase 4 multitask MLP (direction / quantile / RV /
                competing-risk hazard heads, masked losses, early
                stopping, temperature calibration on validation only,
                checksummed weight artifacts).  Requires torch; reports
                "unavailable" honestly otherwise.

Usage:
  python -m oracle.ml.train --model logistic --dataset artifacts/oracle4/
      dataset_v1.sqlite --horizon 60m --label-type direction
      --out artifacts/oracle4/train_logistic_60m.json --seed 42
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import subprocess
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.lab.purged_validation import (
    HORIZON_EMBARGO_SESSIONS,
    Fold,
    Sample,
    Standardizer,
    binary_metrics,
    dataset_meta,
    dte_bucket,
    load_samples,
    make_folds,
    metrics_fingerprint,
    paired_session_bootstrap,
    register_holdout,
)
from oracle.ml.baselines import (
    DIRECTION_CLASSES,
    HAZARD_BIN_EDGES_MIN,
    HAZARD_CAUSES,
    DiscreteTimeCompetingRisks,
    HarLiteModel,
    MultinomialLogisticModel,
    QUANTILE_LADDER,
    QuantilePinballModel,
    RidgeLogisticModel,
    apply_platt,
    apply_temperature_np,
    baseline_p_win_probs,
    build_design,
    confusion_matrix,
    constant_base_rate_probs,
    constant_class_probs,
    empirical_cumulative_incidence,
    expand_person_periods,
    expected_calibration_error,
    fit_platt,
    fit_temperature_np,
    interval_coverage,
    load_classical,
    macro_f1,
    momentum_class_probs,
    momentum_rule_probs,
    multiclass_brier,
    multiclass_log_loss,
    per_class_reliability,
    pinball_loss,
    qlike_loss,
    reliability_bins,
    rv_ewma_forecast,
    rv_persistence_forecast,
    save_classical,
)
from oracle.ml.mlp import direction_class, standardizer_state

MODELS = ("logistic", "naive", "quantile", "rv", "hazard", "gbm", "mlp")


# --------------------------------------------------------------------------- #
# small utilities
# --------------------------------------------------------------------------- #

def _py(obj: Any) -> Any:
    """Recursively convert numpy scalars/arrays so json.dumps succeeds."""
    if isinstance(obj, np.floating):
        return float(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.ndarray):
        return [_py(v) for v in obj.tolist()]
    if isinstance(obj, dict):
        return {str(k): _py(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_py(v) for v in obj]
    return obj


def _git_sha() -> Optional[str]:
    try:
        out = subprocess.run(["git", "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=10)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except Exception:
        return None


def _ok(s: Sample) -> bool:
    return s.label_status == "ok" and s.label_value is not None


def _status_counts(samples: Sequence[Sample]) -> Dict[str, int]:
    out = {"total": len(samples), "ok": 0, "censored": 0, "missing": 0}
    for s in samples:
        if s.label_status in out:
            out[s.label_status] += 1
    return out


def _feature_names(samples: Sequence[Sample]) -> List[str]:
    return sorted({k for s in samples for k in s.features})


def _per_session_log_loss(sessions: Sequence[str], y: Sequence[int],
                          p: Sequence[float],
                          mask: Optional[Sequence[float]] = None
                          ) -> Dict[str, float]:
    groups: Dict[str, Tuple[List[int], List[float]]] = {}
    for i, sess in enumerate(sessions):
        if mask is not None and not mask[i]:
            continue
        ys, ps = groups.setdefault(sess, ([], []))
        ys.append(y[i])
        ps.append(p[i])
    out = {}
    for sess, (ys, ps) in groups.items():
        m = binary_metrics(ys, ps)["log_loss"]
        if m is not None:
            out[sess] = m
    return out


def _delta_report(comp_by_sess: Dict[str, float],
                  chall_by_sess: Dict[str, float],
                  n_boot: int, seed: int) -> Dict[str, Any]:
    """Paired per-session delta = comparator loss − challenger loss.

    Positive mean = challenger better (lower loss).
    """
    common = sorted(set(comp_by_sess) & set(chall_by_sess))
    deltas = {s: comp_by_sess[s] - chall_by_sess[s] for s in common}
    boot = paired_session_bootstrap(deltas, n_boot=n_boot, seed=seed)
    return {"per_session_delta": deltas, "sessions": len(common),
            "bootstrap": boot}


def _stratified_binary(samples: Sequence[Sample], y: Sequence[int],
                       p: Sequence[float],
                       key_fn) -> Dict[str, Dict[str, Any]]:
    groups: Dict[str, Tuple[List[int], List[float]]] = {}
    for s, yi, pi in zip(samples, y, p):
        ys, ps = groups.setdefault(key_fn(s), ([], []))
        ys.append(yi)
        ps.append(pi)
    out = {}
    for k, (ys, ps) in sorted(groups.items()):
        m = binary_metrics(ys, ps)
        m["ece"] = expected_calibration_error(reliability_bins(ys, ps))
        out[k] = m
    return out


# --------------------------------------------------------------------------- #
# direction: logistic challenger + naive comparators
# --------------------------------------------------------------------------- #

def _direction_y(s: Sample) -> int:
    return 1 if s.label_value > 0 else 0        # up vs not-up (harness parity)


def _direction_cls(s: Sample) -> int:
    """True 3-class target: 0=down, 1=neutral, 2=up (Oracle 4.0.1)."""
    return direction_class(float(s.label_value))


_BINARY_VS_3CLASS_NOTE = (
    "binary (up-vs-not-up) and 3-class metrics are NOT comparable: the "
    "binary collapse merges neutral and bearish outcomes; compare models "
    "only within the same class definition")

_BASELINE_PWIN_3CLASS = {
    "status": "unavailable",
    "why": "baseline_p_win is the production probability that the OPTION "
           "TRADE profits (PoP), not P(underlying direction class); it is "
           "semantically mismatched as a 3-class directional comparator "
           "and is reported unavailable rather than coerced"}


def _mc_metrics(y: Sequence[int], probs: np.ndarray) -> Dict[str, Any]:
    """Pooled 3-class metrics (log loss, Brier, accuracy, macro F1, CM)."""
    y = list(y)
    if not y:
        return {"n": 0, "log_loss": None, "brier": None, "accuracy": None,
                "macro_f1": None, "confusion_matrix": None,
                "class_counts": {c: 0 for c in DIRECTION_CLASSES}}
    pred = [int(i) for i in np.asarray(probs).argmax(axis=1)]
    counts = {c: 0 for c in DIRECTION_CLASSES}
    for yi in y:
        counts[DIRECTION_CLASSES[yi]] += 1
    return {
        "n": len(y),
        "log_loss": multiclass_log_loss(y, probs),
        "brier": multiclass_brier(y, probs),
        "accuracy": float(np.mean([p == t for p, t in zip(pred, y)])),
        "macro_f1": macro_f1(y, pred),
        "confusion_matrix": confusion_matrix(y, pred),
        "class_counts": counts,
    }


def _per_session_mc_log_loss(sessions: Sequence[str], y: Sequence[int],
                             probs: np.ndarray,
                             mask: Optional[Sequence[float]] = None
                             ) -> Dict[str, float]:
    probs = np.asarray(probs, dtype=float)
    groups: Dict[str, Tuple[List[int], List[int]]] = {}
    for i, sess in enumerate(sessions):
        if mask is not None and not mask[i]:
            continue
        ys, idx = groups.setdefault(sess, ([], []))
        ys.append(y[i])
        idx.append(i)
    out: Dict[str, float] = {}
    for sess, (ys, idx) in groups.items():
        m = multiclass_log_loss(ys, probs[idx])
        if m is not None:
            out[sess] = m
    return out


def _run_direction_three_class(samples: Sequence[Sample],
                               folds: Sequence[Fold],
                               seed: int, n_boot: int,
                               challenger: str = "logistic"
                               ) -> Dict[str, Any]:
    """True 3-class direction evaluation (down / neutral / up).

    Fixes the Oracle 4.0 up-vs-not-up collapse (which merged neutral and
    bearish outcomes).  The multinomial logistic challenger is temperature
    calibrated on a chronological tail of train (the last train session)
    with **label-availability purging**: any fit row whose label became
    available at or after the calibration session start is excluded from
    the fit split (counted, never silently dropped), so the calibration
    tail can never leak into the fit.
    """
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    fold_rows: List[Dict[str, Any]] = []

    pool_y: List[int] = []
    pool_probs: List[List[float]] = []
    pool_sess: List[str] = []
    comp_pool: Dict[str, Dict[str, list]] = {
        c: {"p": [], "mask": []} for c in ("constant", "momentum")}
    purge_totals = {"fit_rows_kept": 0, "fit_rows_purged_availability": 0,
                    "fit_rows_unknown_availability": 0}

    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids if _ok(by_id[d])]
        y_train = [_direction_cls(s) for s in train]
        y_test = [_direction_cls(s) for s in test]
        if len(set(y_train)) < 2 or not test:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "degenerate train or empty test"})
            continue

        probs_test: Optional[np.ndarray] = None
        temperature = 1.0
        calib_used = False
        purge = {"fit_rows_kept": 0, "fit_rows_purged_availability": 0,
                 "fit_rows_unknown_availability": 0}
        if challenger == "logistic":
            # chronological calibration tail = last train session
            last_sess = max(s.session for s in train)
            calib_start = _session_start_utc(last_sess)
            calib = [s for s in train if s.session == last_sess]
            fitp: List[Sample] = []
            for s in train:
                if s.session == last_sess:
                    continue
                # label-availability purge between fit and calibration
                if s.label_available_at is None:
                    purge["fit_rows_unknown_availability"] += 1
                    continue
                if s.label_available_at >= calib_start:
                    purge["fit_rows_purged_availability"] += 1
                    continue
                fitp.append(s)
                purge["fit_rows_kept"] += 1
            y_fit = [_direction_cls(s) for s in fitp]
            calib_used = len(set(y_fit)) >= 2 and len(calib) >= 2
            if not calib_used:
                # fall back to fitting on all train, no calibration
                fitp, y_fit, calib = list(train), list(y_train), []
            std = Standardizer().fit([s.features for s in fitp], names)
            dm_fit = build_design(fitp, std, std.feature_names)
            model = MultinomialLogisticModel().fit(dm_fit.x, y_fit)
            dm_test = build_design(test, std, std.feature_names)
            logits_test = model.decision_scores(dm_test.x)
            if calib_used:
                dm_cal = build_design(calib, std, std.feature_names)
                temperature = fit_temperature_np(
                    model.decision_scores(dm_cal.x),
                    [_direction_cls(s) for s in calib])
            probs_test = apply_temperature_np(logits_test, temperature)
        for k in purge_totals:
            purge_totals[k] += purge[k]

        # comparators on this fold's test rows (identical masks/labels)
        c_const = constant_class_probs(y_train, len(test))
        c_mom, m_mom = momentum_class_probs(train, y_train, test)
        comp_pool["constant"]["p"].extend(c_const.tolist())
        comp_pool["constant"]["mask"].extend([1.0] * len(test))
        comp_pool["momentum"]["p"].extend(c_mom.tolist())
        comp_pool["momentum"]["mask"].extend(m_mom.tolist())

        pool_y.extend(y_test)
        pool_sess.extend([s.session for s in test])
        row: Dict[str, Any] = {
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "n_train_ok": len(train), "n_test_ok": len(test),
            "calibration_used": calib_used,
            "temperature": temperature,
            "purge_accounting": purge}
        if probs_test is not None:
            pool_probs.extend(probs_test.tolist())
            row["test_metrics"] = _mc_metrics(y_test, probs_test)
        fold_rows.append(row)

    report: Dict[str, Any] = {
        "classes": list(DIRECTION_CLASSES),
        "note": _BINARY_VS_3CLASS_NOTE,
        "folds": fold_rows,
        "purge_accounting": purge_totals,
    }

    comparators: Dict[str, Any] = {}
    comp_sess_loss: Dict[str, Dict[str, float]] = {}
    for cname, pool in comp_pool.items():
        mask = pool["mask"]
        ys = [y for y, m in zip(pool_y, mask) if m]
        ps = np.asarray([p for p, m in zip(pool["p"], mask) if m],
                        dtype=float)
        met = _mc_metrics(ys, ps if len(ys) else np.zeros((0, 3)))
        met["n_unavailable"] = int(len(mask) - sum(mask))
        comparators[cname] = met
        comp_sess_loss[cname] = _per_session_mc_log_loss(
            pool_sess, pool_y, np.asarray(pool["p"], dtype=float)
            if pool["p"] else np.zeros((0, 3)), mask)
    comparators["baseline_p_win"] = dict(_BASELINE_PWIN_3CLASS)
    report["comparators"] = comparators

    if challenger == "logistic" and pool_probs:
        probs = np.asarray(pool_probs, dtype=float)
        pooled = _mc_metrics(pool_y, probs)
        report["challenger"] = {
            "name": "multinomial_logistic_temperature",
            "pooled": pooled,
            "per_class_reliability": per_class_reliability(pool_y, probs),
        }
        deltas = {}
        for cname, pool in comp_pool.items():
            chall_by_sess = _per_session_mc_log_loss(
                pool_sess, pool_y, probs, pool["mask"])  # paired rows
            deltas[cname] = _delta_report(comp_sess_loss[cname],
                                          chall_by_sess, n_boot, seed)
        report["deltas_log_loss"] = deltas
    return report


def run_direction(samples: Sequence[Sample], folds: Sequence[Fold],
                  seed: int, n_boot: int, challenger: str = "logistic"
                  ) -> Dict[str, Any]:
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    fold_rows: List[Dict[str, Any]] = []

    pool_s: List[Sample] = []
    pool_y: List[int] = []
    pool_p: List[float] = []
    pool_sess: List[str] = []
    comp_pool: Dict[str, Dict[str, List[float]]] = {
        c: {"p": [], "mask": []}
        for c in ("constant", "momentum", "baseline_p_win")}

    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids if _ok(by_id[d])]
        y_train = [_direction_y(s) for s in train]
        y_test = [_direction_y(s) for s in test]
        if len(set(y_train)) < 2 or not test:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "degenerate train or empty test"})
            continue

        p_test: Optional[np.ndarray] = None
        platt_used = False
        if challenger == "logistic":
            # chronological calibration tail = last train session
            last_sess = max(s.session for s in train)
            fitp = [s for s in train if s.session != last_sess]
            calib = [s for s in train if s.session == last_sess]
            y_fit = [_direction_y(s) for s in fitp]
            y_cal = [_direction_y(s) for s in calib]
            platt_used = (len(set(y_fit)) == 2 and len(set(y_cal)) == 2)
            if not platt_used:
                fitp, y_fit, calib, y_cal = train, y_train, [], []
            std = Standardizer().fit([s.features for s in fitp], names)
            dm_fit = build_design(fitp, std, std.feature_names)
            model = RidgeLogisticModel().fit(dm_fit.x, y_fit)
            dm_test = build_design(test, std, std.feature_names)
            if platt_used:
                a, b = fit_platt(
                    model.decision_scores(
                        build_design(calib, std, std.feature_names).x),
                    y_cal)
                p_test = apply_platt(a, b, model.decision_scores(dm_test.x))
            else:
                p_test = model.predict_proba(dm_test.x)

        # comparators on this fold's test rows
        c_const = constant_base_rate_probs(y_train, len(test))
        c_mom, m_mom = momentum_rule_probs(train, y_train, test)
        c_base, m_base = baseline_p_win_probs(test)
        comp_pool["constant"]["p"].extend(c_const.tolist())
        comp_pool["constant"]["mask"].extend([1.0] * len(test))
        comp_pool["momentum"]["p"].extend(c_mom.tolist())
        comp_pool["momentum"]["mask"].extend(m_mom.tolist())
        comp_pool["baseline_p_win"]["p"].extend(c_base.tolist())
        comp_pool["baseline_p_win"]["mask"].extend(m_base.tolist())

        pool_s.extend(test)
        pool_y.extend(y_test)
        pool_sess.extend([s.session for s in test])
        if p_test is not None:
            pool_p.extend(p_test.tolist())

        row = {"fold_id": fold.fold_id, "skipped": False,
               "train_sessions": fold.train_sessions,
               "test_sessions": fold.test_sessions,
               "n_train_ok": len(train), "n_test_ok": len(test),
               "purged": fold.purged, "embargoed": fold.embargoed,
               "platt_used": platt_used}
        if p_test is not None:
            row["test_metrics"] = binary_metrics(y_test, p_test.tolist())
        fold_rows.append(row)

    report: Dict[str, Any] = {"folds": fold_rows}

    # pooled comparator metrics (masked rows only)
    comparators: Dict[str, Any] = {}
    comp_sess_loss: Dict[str, Dict[str, float]] = {}
    for cname, pool in comp_pool.items():
        mask = pool["mask"]
        ys = [y for y, m in zip(pool_y, mask) if m]
        ps = [p for p, m in zip(pool["p"], mask) if m]
        met = binary_metrics(ys, ps)
        met["ece"] = expected_calibration_error(reliability_bins(ys, ps))
        met["n_unavailable"] = int(len(mask) - sum(mask))
        comparators[cname] = met
        comp_sess_loss[cname] = _per_session_log_loss(
            pool_sess, pool_y, pool["p"], mask)
    report["comparators"] = comparators

    if challenger == "logistic" and pool_p:
        pooled = binary_metrics(pool_y, pool_p)
        bins = reliability_bins(pool_y, pool_p)
        pooled["ece"] = expected_calibration_error(bins)
        report["challenger"] = {
            "name": "ridge_logistic_platt",
            "pooled": pooled,
            "reliability_bins": bins,
            "by_dte_bucket": _stratified_binary(
                pool_s, pool_y, pool_p,
                lambda s: dte_bucket(s.features.get("dte_days"))),
            "by_kind": _stratified_binary(pool_s, pool_y, pool_p,
                                          lambda s: s.kind),
        }
        deltas = {}
        for cname, pool in comp_pool.items():
            chall_by_sess = _per_session_log_loss(
                pool_sess, pool_y, pool_p, pool["mask"])  # paired rows
            deltas[cname] = _delta_report(comp_sess_loss[cname],
                                          chall_by_sess, n_boot, seed)
        report["deltas_log_loss"] = deltas

    # Oracle 4.0.1: true 3-class evaluation alongside the legacy binary
    # track (kept for harness parity; never compared across definitions).
    report["binary_note"] = _BINARY_VS_3CLASS_NOTE
    report["three_class"] = _run_direction_three_class(
        samples, folds, seed, n_boot, challenger)
    return report


# --------------------------------------------------------------------------- #
# quantile ladder on log returns
# --------------------------------------------------------------------------- #

def run_quantile(samples: Sequence[Sample], folds: Sequence[Fold],
                 seed: int, n_boot: int) -> Dict[str, Any]:
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    fold_rows: List[Dict[str, Any]] = []

    pool_y: List[float] = []
    pool_sess: List[str] = []
    pool_pred: Dict[float, List[float]] = {t: [] for t in QUANTILE_LADDER}
    pool_marg: Dict[float, List[float]] = {t: [] for t in QUANTILE_LADDER}

    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids if _ok(by_id[d])]
        if len(train) < 10 or not test:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "insufficient train or empty test"})
            continue
        y_train = [float(s.label_value) for s in train]
        y_test = [float(s.label_value) for s in test]
        std = Standardizer().fit([s.features for s in train], names)
        model = QuantilePinballModel().fit(
            build_design(train, std, std.feature_names).x, y_train)
        preds = model.predict(build_design(test, std, std.feature_names).x)
        marg = {t: float(np.quantile(np.asarray(y_train), t))
                for t in QUANTILE_LADDER}

        pool_y.extend(y_test)
        pool_sess.extend([s.session for s in test])
        for t in QUANTILE_LADDER:
            pool_pred[t].extend(preds[t].tolist())
            pool_marg[t].extend([marg[t]] * len(test))

        fold_rows.append({
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "n_train_ok": len(train), "n_test_ok": len(test),
            "purged": fold.purged, "embargoed": fold.embargoed,
            "pinball_mean_model": float(np.mean([
                pinball_loss(y_test, preds[t], t) for t in QUANTILE_LADDER])),
            "pinball_mean_marginal": float(np.mean([
                pinball_loss(y_test, [marg[t]] * len(test), t)
                for t in QUANTILE_LADDER])),
        })

    report: Dict[str, Any] = {"folds": fold_rows,
                              "quantiles": list(QUANTILE_LADDER)}
    if not pool_y:
        report["pooled"] = None
        return report

    per_tau = {}
    for t in QUANTILE_LADDER:
        per_tau[t] = {
            "pinball_model": pinball_loss(pool_y, pool_pred[t], t),
            "pinball_marginal": pinball_loss(pool_y, pool_marg[t], t),
        }
    report["pooled"] = {
        "n": len(pool_y),
        "per_tau": per_tau,
        "coverage_90_model": interval_coverage(pool_y, pool_pred[0.05],
                                               pool_pred[0.95]),
        "coverage_90_marginal": interval_coverage(pool_y, pool_marg[0.05],
                                                  pool_marg[0.95]),
        "coverage_50_model": interval_coverage(pool_y, pool_pred[0.25],
                                               pool_pred[0.75]),
        "coverage_50_marginal": interval_coverage(pool_y, pool_marg[0.25],
                                                  pool_marg[0.75]),
    }

    # per-session mean pinball across the ladder (paired delta)
    def _sess_pinball(pred: Dict[float, List[float]]) -> Dict[str, float]:
        groups: Dict[str, List[int]] = {}
        for i, sess in enumerate(pool_sess):
            groups.setdefault(sess, []).append(i)
        out = {}
        for sess, idx in groups.items():
            losses = []
            for t in QUANTILE_LADDER:
                losses.append(pinball_loss([pool_y[i] for i in idx],
                                           [pred[t][i] for i in idx], t))
            vals = [v for v in losses if v is not None]
            if vals:
                out[sess] = float(np.mean(vals))
        return out

    report["deltas_pinball"] = {"marginal": _delta_report(
        _sess_pinball(pool_marg), _sess_pinball(pool_pred), n_boot, seed)}
    return report


# --------------------------------------------------------------------------- #
# realized-volatility forecasting
# --------------------------------------------------------------------------- #

def _underlying_by_id(dataset_path: str) -> Dict[str, str]:
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        return {r[0]: (r[1] or "?") for r in conn.execute(
            "SELECT decision_id, underlying FROM features")}
    finally:
        conn.close()


def run_rv(samples: Sequence[Sample], folds: Sequence[Fold],
           dataset_path: str, seed: int, n_boot: int) -> Dict[str, Any]:
    by_id = {s.decision_id: s for s in samples}
    und = _underlying_by_id(dataset_path)
    # Point-in-time EWMA over feature-side rv_session (uses only values
    # observed at or before each row) — safe to compute once globally.
    ewma_all, ewma_mask = rv_ewma_forecast(samples, und)
    idx_of = {s.decision_id: i for i, s in enumerate(samples)}
    fold_rows: List[Dict[str, Any]] = []

    pool_y: List[float] = []
    pool_sess: List[str] = []
    pools: Dict[str, Dict[str, List[float]]] = {
        n: {"f": [], "mask": []} for n in ("persistence", "ewma", "har")}

    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids if _ok(by_id[d])]
        if len(train) < 10 or not test:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "insufficient train or empty test"})
            continue
        y_train = [float(s.label_value) for s in train]
        y_test = [float(s.label_value) for s in test]

        def _ewma_for(rows: Sequence[Sample]) -> List[Optional[float]]:
            return [float(ewma_all[idx_of[s.decision_id]])
                    if ewma_mask[idx_of[s.decision_id]] else None
                    for s in rows]

        har = HarLiteModel().fit(train, y_train, _ewma_for(train))
        f_har, m_har = har.predict(test, _ewma_for(test))
        f_per, m_per = rv_persistence_forecast(test)
        f_ew = np.array([ewma_all[idx_of[s.decision_id]] for s in test])
        m_ew = np.array([ewma_mask[idx_of[s.decision_id]] for s in test])

        pool_y.extend(y_test)
        pool_sess.extend([s.session for s in test])
        for name, (f, m) in (("persistence", (f_per, m_per)),
                             ("ewma", (f_ew, m_ew)),
                             ("har", (f_har, m_har))):
            pools[name]["f"].extend(np.asarray(f).tolist())
            pools[name]["mask"].extend(np.asarray(m).tolist())

        fold_rows.append({
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "n_train_ok": len(train), "n_test_ok": len(test),
            "purged": fold.purged, "embargoed": fold.embargoed,
            "har_coefficients": (har.w.tolist()
                                 if har.w is not None else None),
        })

    report: Dict[str, Any] = {"folds": fold_rows}
    if not pool_y:
        report["pooled"] = None
        return report

    def _masked(name: str) -> Tuple[List[float], List[float], List[str]]:
        ys, fs, ss = [], [], []
        for y, f, m, sess in zip(pool_y, pools[name]["f"],
                                 pools[name]["mask"], pool_sess):
            if m:
                ys.append(y)
                fs.append(f)
                ss.append(sess)
        return ys, fs, ss

    pooled = {}
    sess_qlike: Dict[str, Dict[str, float]] = {}
    for name in pools:
        ys, fs, ss = _masked(name)
        rmse = (float(np.sqrt(np.mean((np.asarray(ys)
                                       - np.asarray(fs)) ** 2)))
                if ys else None)
        pooled[name] = {"n": len(ys), "rmse": rmse,
                        "qlike": qlike_loss(ys, fs),
                        "n_unavailable": int(len(pool_y) - len(ys))}
        groups: Dict[str, Tuple[List[float], List[float]]] = {}
        for y, f, sess in zip(ys, fs, ss):
            a, b = groups.setdefault(sess, ([], []))
            a.append(y)
            b.append(f)
        sess_qlike[name] = {
            sess: q for sess, (a, b) in groups.items()
            if (q := qlike_loss(a, b)) is not None}
    report["pooled"] = pooled
    report["deltas_qlike"] = {
        "har_vs_persistence": _delta_report(
            sess_qlike["persistence"], sess_qlike["har"], n_boot, seed),
        "ewma_vs_persistence": _delta_report(
            sess_qlike["persistence"], sess_qlike["ewma"], n_boot, seed),
    }
    return report


# --------------------------------------------------------------------------- #
# discrete-time competing-risks hazards
# --------------------------------------------------------------------------- #

def _first_touch_causes(dataset_path: str) -> Dict[str, Optional[str]]:
    conn = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    try:
        out: Dict[str, Optional[str]] = {}
        for did, meta in conn.execute(
                "SELECT decision_id, meta_json FROM labels "
                "WHERE horizon='event' AND label_type='first_touch'"):
            try:
                out[did] = (json.loads(meta) or {}).get("touch")
            except (TypeError, ValueError):
                out[did] = None
        return out
    finally:
        conn.close()


def _hazard_rows(rows: Sequence[Sample],
                 causes: Dict[str, Optional[str]]
                 ) -> Tuple[List[Sample], List[float],
                            List[Optional[str]]]:
    """Usable hazard subjects: resolved events with a known cause, plus
    right-censored rows (cause None, time = minutes to coverage end).
    MISSING (never-filled) rows are not at risk and are excluded."""
    subj, times, cs = [], [], []
    for s in rows:
        if s.label_status == "ok" and s.label_value is not None \
                and s.label_value > 0:
            c = causes.get(s.decision_id)
            if c in HAZARD_CAUSES:
                subj.append(s)
                times.append(float(s.label_value))
                cs.append(c)
        elif s.label_status == "censored" and s.label_end is not None:
            t = (s.label_end - s.as_of).total_seconds() / 60.0
            if t > 0:
                subj.append(s)
                times.append(t)
                cs.append(None)
    return subj, times, cs


def _marginal_hazard_table(periods, n_bins: int, n_classes: int,
                           eps: float = 1e-6) -> np.ndarray:
    counts = np.full((n_bins, n_classes), eps)
    for pp in periods:
        counts[pp.time_bin, pp.outcome] += 1.0
    return counts / counts.sum(axis=1, keepdims=True)


def _marginal_nll(table: np.ndarray, periods) -> Optional[float]:
    if not periods:
        return None
    return float(-np.mean([np.log(table[pp.time_bin, pp.outcome])
                           for pp in periods]))


def run_hazard(samples: Sequence[Sample], folds: Sequence[Fold],
               dataset_path: str, seed: int, n_boot: int) -> Dict[str, Any]:
    causes = _first_touch_causes(dataset_path)
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    n_bins = len(HAZARD_BIN_EDGES_MIN) - 1
    fold_rows: List[Dict[str, Any]] = []
    model_sess_nll: Dict[str, List[Tuple[float, int]]] = {}
    marg_sess_nll: Dict[str, List[Tuple[float, int]]] = {}
    cause_counts = {c: 0 for c in HAZARD_CAUSES}
    cens_count = 0
    cif_rows: List[np.ndarray] = []

    for fold in folds:
        tr_s, tr_t, tr_c = _hazard_rows(
            [by_id[d] for d in fold.train_ids], causes)
        te_s, te_t, te_c = _hazard_rows(
            [by_id[d] for d in fold.test_ids], causes)
        if len(tr_s) < 20 or not te_s:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "insufficient usable subjects"})
            continue
        std = Standardizer().fit([s.features for s in tr_s], names)
        x_tr = build_design(tr_s, std, std.feature_names).x
        x_te = build_design(te_s, std, std.feature_names).x
        pp_tr = expand_person_periods(tr_t, tr_c)
        pp_te = expand_person_periods(te_t, te_c)
        model = DiscreteTimeCompetingRisks().fit(x_tr, pp_tr)
        table = _marginal_hazard_table(pp_tr, n_bins,
                                       len(HAZARD_CAUSES) + 1)
        nll_model = model.nll(x_te, pp_te)
        nll_marg = _marginal_nll(table, pp_te)

        for s, c in zip(te_s, te_c):
            if c is None:
                cens_count += 1
            else:
                cause_counts[c] += 1
        for i in range(x_te.shape[0]):
            cif_rows.append(model.cumulative_incidence(x_te[i])[-1])

        # per-session paired NLL on test person-periods
        sess_pp: Dict[str, List[Any]] = {}
        for pp in pp_te:
            sess_pp.setdefault(te_s[pp.row_index].session, []).append(pp)
        for sess, pps in sess_pp.items():
            mv = model.nll(x_te, pps)
            gv = _marginal_nll(table, pps)
            if mv is not None and gv is not None:
                model_sess_nll.setdefault(sess, []).append((mv, len(pps)))
                marg_sess_nll.setdefault(sess, []).append((gv, len(pps)))

        fold_rows.append({
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "n_train_subjects": len(tr_s), "n_test_subjects": len(te_s),
            "n_train_periods": len(pp_tr), "n_test_periods": len(pp_te),
            "purged": fold.purged, "embargoed": fold.embargoed,
            "test_nll_model": nll_model, "test_nll_marginal": nll_marg,
        })

    def _weighted(d: Dict[str, List[Tuple[float, int]]]
                  ) -> Dict[str, float]:
        return {sess: sum(v * n for v, n in rows) / sum(n for _, n in rows)
                for sess, rows in d.items()}

    all_s, all_t, all_c = _hazard_rows(samples, causes)
    report: Dict[str, Any] = {
        "folds": fold_rows,
        "bin_edges_minutes": list(HAZARD_BIN_EDGES_MIN[:-1]) + ["inf"],
        "causes": list(HAZARD_CAUSES),
        "subjects": {"usable_total": len(all_s),
                     "test_events_by_cause": cause_counts,
                     "test_censored": cens_count},
        "note": ("Censored subjects carry label_end = dataset coverage end, "
                 "so the purge removes them from every training window "
                 "(their survival status is future information); training "
                 "therefore uses fully-resolved events only."),
        "empirical_cif_1440m_all": empirical_cumulative_incidence(
            all_t, all_c, 1440.0),
    }
    if cif_rows:
        mean_cif = np.mean(np.stack(cif_rows), axis=0)
        report["mean_model_cif_final_bin"] = {
            c: float(mean_cif[i]) for i, c in enumerate(HAZARD_CAUSES)}
    if model_sess_nll:
        report["deltas_nll"] = {"marginal": _delta_report(
            _weighted(marg_sess_nll), _weighted(model_sess_nll),
            n_boot, seed)}
    return report


# --------------------------------------------------------------------------- #
# optional models: LightGBM (actual training; honest unavailable path)
# --------------------------------------------------------------------------- #

def _session_start_utc(session: str) -> datetime:
    """Midnight UTC at the start of a YYYY-MM-DD session string."""
    return datetime.fromisoformat(session).replace(tzinfo=timezone.utc)


def _lgbm_ready() -> bool:
    """Call-time lightgbm check (tests may monkeypatch gbm.LGBM_AVAILABLE)."""
    import oracle.ml.gbm as gbm_mod
    return bool(gbm_mod.LGBM_AVAILABLE)


def _availability_split(train: Sequence[Sample]
                        ) -> Tuple[List[Sample], List[Sample],
                                   Dict[str, int]]:
    """Split train into (fit, val) at the last train session with
    label-availability purging of the fit side.

    Any fit-candidate row whose label became available at or after the
    validation session start is purged (counted, never silently dropped);
    rows with unknown availability are excluded as well.
    """
    last_sess = max(s.session for s in train)
    val_start = _session_start_utc(last_sess)
    val = [s for s in train if s.session == last_sess]
    fit: List[Sample] = []
    acct = {"fit_rows_kept": 0, "fit_rows_purged_availability": 0,
            "fit_rows_unknown_availability": 0}
    for s in train:
        if s.session == last_sess:
            continue
        if s.label_available_at is None:
            acct["fit_rows_unknown_availability"] += 1
            continue
        if s.label_available_at >= val_start:
            acct["fit_rows_purged_availability"] += 1
            continue
        fit.append(s)
        acct["fit_rows_kept"] += 1
    return fit, val, acct


def run_gbm(samples: Sequence[Sample], folds: Sequence[Fold],
            dataset_path: str, horizon: str, seed: int, n_boot: int,
            out_stem: str, fold_sessions: Sequence[str],
            holdout: Sequence[str]) -> Dict[str, Any]:
    """Actual LightGBM training through the purged walk-forward harness.

    * 3-class direction (multiclass softmax probabilities), early stopped
      on a chronological validation tail (last train session) with
      label-availability purging between fit and validation.
    * Both missingness treatments — native NaN handling vs explicit
      missingness flags — are trained on identical rows and compared on
      identical labels.
    * Class weights (inverse frequency) are applied to the TRAIN split
      only; evaluation is always on the unweighted true distribution.
    * Quantile regressors on log returns at the pre-registered ladder,
      against marginal train quantiles.
    * Net-P&L model: reported unavailable (the frozen ledger has no
      qualified monetary labels — fees unknown), never faked.
    * Final artifact: fit on all fold sessions, saved with checksum and
      reloaded for a prediction-parity check.
    """
    import oracle.ml.gbm as gbm

    if not gbm.LGBM_AVAILABLE:
        return {"status": "unavailable", "why": gbm.LGBM_WHY}

    cfg = gbm.GBMConfig()            # pre-registered budget (seed 1337)
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    modes = ("native", "flags")
    fold_rows: List[Dict[str, Any]] = []

    pool_y: List[int] = []
    pool_sess: List[str] = []
    pool_probs: Dict[str, List[List[float]]] = {m: [] for m in modes}
    comp_pool: Dict[str, Dict[str, list]] = {
        c: {"p": [], "mask": []} for c in ("constant", "momentum")}
    purge_totals = {"fit_rows_kept": 0, "fit_rows_purged_availability": 0,
                    "fit_rows_unknown_availability": 0}

    for fold in folds:
        train = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids if _ok(by_id[d])]
        y_train = [_direction_cls(s) for s in train]
        y_test = [_direction_cls(s) for s in test]
        if len(set(y_train)) < 2 or not test:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "degenerate train or empty test"})
            continue

        fitp, val, acct = _availability_split(train)
        y_fit = [_direction_cls(s) for s in fitp]
        if len(set(y_fit)) < 2:
            # fall back: fit on all train, no early-stopping tail
            fitp, y_fit, val = list(train), list(y_train), []
        for k in purge_totals:
            purge_totals[k] += acct[k]
        y_val = [_direction_cls(s) for s in val]

        row: Dict[str, Any] = {
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "n_train_ok": len(train), "n_test_ok": len(test),
            "purge_accounting": acct, "modes": {}}
        for mode in modes:
            x_fit, _ = gbm.build_matrix(fitp, names, mode)
            x_val, _ = gbm.build_matrix(val, names, mode)
            x_test, _ = gbm.build_matrix(test, names, mode)
            booster, info = gbm.fit_direction_gbm(
                x_fit, np.asarray(y_fit), x_val, np.asarray(y_val),
                cfg=cfg)
            probs = gbm.predict_direction_proba(booster, x_test)
            pool_probs[mode].extend(probs.tolist())
            row["modes"][mode] = {
                "fit_info": info,
                "test_metrics": _mc_metrics(y_test, probs)}

        # comparators on identical test rows / labels / masks
        c_const = constant_class_probs(y_train, len(test))
        c_mom, m_mom = momentum_class_probs(train, y_train, test)
        comp_pool["constant"]["p"].extend(c_const.tolist())
        comp_pool["constant"]["mask"].extend([1.0] * len(test))
        comp_pool["momentum"]["p"].extend(c_mom.tolist())
        comp_pool["momentum"]["mask"].extend(m_mom.tolist())
        pool_y.extend(y_test)
        pool_sess.extend([s.session for s in test])
        fold_rows.append(row)

    report: Dict[str, Any] = {
        "status": "trained",
        "lightgbm_version": gbm.LGBM_VERSION,
        "lightgbm_pinned": gbm.LGBM_PINNED,
        "version_note": gbm.LGBM_WHY,
        "config": {k: getattr(cfg, k) for k in (
            "learning_rate", "num_leaves", "max_depth",
            "min_data_in_leaf", "feature_fraction", "lambda_l2",
            "max_estimators", "early_stopping_rounds", "seed")},
        "classes": list(DIRECTION_CLASSES),
        "note": _BINARY_VS_3CLASS_NOTE,
        "folds": fold_rows,
        "purge_accounting": purge_totals,
    }

    # pooled comparators (masked rows only)
    comparators: Dict[str, Any] = {}
    comp_sess_loss: Dict[str, Dict[str, float]] = {}
    for cname, pool in comp_pool.items():
        mask = pool["mask"]
        ys = [y for y, m in zip(pool_y, mask) if m]
        ps = np.asarray([p for p, m in zip(pool["p"], mask) if m],
                        dtype=float)
        met = _mc_metrics(ys, ps if len(ys) else np.zeros((0, 3)))
        met["n_unavailable"] = int(len(mask) - sum(mask))
        comparators[cname] = met
        comp_sess_loss[cname] = _per_session_mc_log_loss(
            pool_sess, pool_y, np.asarray(pool["p"], dtype=float)
            if pool["p"] else np.zeros((0, 3)), mask)
    comparators["baseline_p_win"] = dict(_BASELINE_PWIN_3CLASS)
    report["comparators"] = comparators

    # pooled per-mode metrics + deltas (vs comparators and native-vs-flags)
    mode_sess_loss: Dict[str, Dict[str, float]] = {}
    if pool_y:
        report["direction"] = {}
        for mode in modes:
            probs = np.asarray(pool_probs[mode], dtype=float)
            mode_sess_loss[mode] = _per_session_mc_log_loss(
                pool_sess, pool_y, probs)
            report["direction"][mode] = {
                "pooled": _mc_metrics(pool_y, probs),
                "per_class_reliability": per_class_reliability(
                    pool_y, probs),
            }
        deltas: Dict[str, Any] = {}
        for mode in modes:
            for cname, pool in comp_pool.items():
                chall_by_sess = _per_session_mc_log_loss(
                    pool_sess, pool_y,
                    np.asarray(pool_probs[mode], dtype=float),
                    pool["mask"])
                deltas[f"{mode}_vs_{cname}"] = _delta_report(
                    comp_sess_loss[cname], chall_by_sess, n_boot, seed)
        # positive mean = flags better (delta = native loss − flags loss)
        deltas["native_vs_flags"] = _delta_report(
            mode_sess_loss["native"], mode_sess_loss["flags"],
            n_boot, seed)
        report["deltas_log_loss"] = deltas

    # ---------------- quantile regressors on log returns ---------------- #
    lr_samples = load_samples(dataset_path, horizon, "log_return")
    lr_by_id = {s.decision_id: s for s in lr_samples if _ok(s)}
    q_pool_y: List[float] = []
    q_pool_sess: List[str] = []
    q_pool_pred: Dict[float, List[float]] = {t: [] for t in QUANTILE_LADDER}
    q_pool_marg: Dict[float, List[float]] = {t: [] for t in QUANTILE_LADDER}
    q_folds = 0
    for fold in folds:
        train = [lr_by_id[d] for d in fold.train_ids if d in lr_by_id]
        test = [lr_by_id[d] for d in fold.test_ids if d in lr_by_id]
        if len(train) < 10 or not test:
            continue
        fitp, val, _ = _availability_split(train)
        if len(fitp) < 5:
            fitp, val = list(train), []
        y_fit = np.asarray([float(s.label_value) for s in fitp])
        y_val = np.asarray([float(s.label_value) for s in val])
        y_tst = [float(s.label_value) for s in test]
        x_fit, _ = gbm.build_matrix(fitp, names, "flags")
        x_val, _ = gbm.build_matrix(val, names, "flags")
        x_tst, _ = gbm.build_matrix(test, names, "flags")
        boosters, _info = gbm.fit_quantile_gbms(x_fit, y_fit, x_val, y_val,
                                                cfg=cfg)
        preds = gbm.predict_quantiles(boosters, x_tst)
        y_train_all = [float(s.label_value) for s in train]
        for tau in QUANTILE_LADDER:
            q_pool_pred[tau].extend(preds[tau].tolist())
            q_pool_marg[tau].extend(
                [float(np.quantile(y_train_all, tau))] * len(test))
        q_pool_y.extend(y_tst)
        q_pool_sess.extend([s.session for s in test])
        q_folds += 1
    if q_pool_y:
        pin_model = {str(t): pinball_loss(q_pool_y, q_pool_pred[t], t)
                     for t in QUANTILE_LADDER}
        pin_marg = {str(t): pinball_loss(q_pool_y, q_pool_marg[t], t)
                    for t in QUANTILE_LADDER}
        report["quantiles"] = {
            "folds_used": q_folds,
            "n": len(q_pool_y),
            "pinball_model": pin_model,
            "pinball_marginal": pin_marg,
            "coverage_90": interval_coverage(
                q_pool_y, q_pool_pred[0.05], q_pool_pred[0.95]),
            "coverage_50": interval_coverage(
                q_pool_y, q_pool_pred[0.25], q_pool_pred[0.75]),
        }
    else:
        report["quantiles"] = {"status": "unavailable",
                               "why": "no log_return labels joined the "
                                      "direction folds"}

    # ---------------- net-P&L model: honest unavailability -------------- #
    report["net_pnl_model"] = {
        "status": "unavailable",
        "why": "no qualified monetary labels: frozen ledger fees are "
               "unknown (None) and exit prices are sparse; a net-P&L "
               "objective would be trained on invented costs"}

    # ---------------- final artifact: save / reload / parity ------------ #
    artifact: Dict[str, Any] = {"status": "skipped",
                                "why": "no usable folds"}
    all_ok = [s for s in samples
              if _ok(s) and s.session in set(fold_sessions)]
    if all_ok and len({_direction_cls(s) for s in all_ok}) >= 2:
        fitp, val, acct = _availability_split(all_ok)
        y_fit = [_direction_cls(s) for s in fitp]
        if len(set(y_fit)) < 2:
            fitp, y_fit, val = list(all_ok), \
                [_direction_cls(s) for s in all_ok], []
        y_val = [_direction_cls(s) for s in val]
        x_fit, cols = gbm.build_matrix(fitp, names, "flags")
        x_val, _ = gbm.build_matrix(val, names, "flags")
        booster, info = gbm.fit_direction_gbm(
            x_fit, np.asarray(y_fit), x_val, np.asarray(y_val), cfg=cfg)
        path = f"{out_stem}_gbm.json"
        meta = {"git_sha": _git_sha(),
                "dataset_hash": dataset_meta(dataset_path).get(
                    "dataset_hash"),
                "horizon": horizon,
                "train_sessions": sorted({s.session for s in fitp}),
                "val_session": sorted({s.session for s in val}),
                "holdout_excluded": list(holdout)}
        digest = gbm.save_gbm_direction(path, booster, cols, "flags",
                                        cfg, meta=meta)
        re_booster, re_cols, re_mode, _re_meta = gbm.load_gbm_direction(
            path)
        probe = x_fit[: min(64, len(x_fit))]
        parity = float(np.max(np.abs(
            gbm.predict_direction_proba(booster, probe)
            - gbm.predict_direction_proba(re_booster, probe)))) \
            if len(probe) else 0.0
        artifact = {"status": "saved", "path": path,
                    "checksum_sha256": digest,
                    "missingness_mode": re_mode,
                    "n_features": len(re_cols),
                    "fit_info": info,
                    "purge_accounting": acct,
                    "reload_parity_max_abs_diff": parity}
    report["artifact"] = artifact
    return report


def run_classical_artifact(samples: Sequence[Sample], dataset_path: str,
                           horizon: str, out_stem: str,
                           fold_sessions: Sequence[str],
                           holdout: Sequence[str]) -> Dict[str, Any]:
    """Save a deployable multinomial-logistic artifact (P1-7).

    Fits on the purged fold sessions only (holdout excluded upstream),
    temperature-calibrates on the chronological validation tail, saves via
    :func:`save_classical` and proves reload parity, so the forecast
    provider loads exactly what was trained.
    """
    artifact: Dict[str, Any] = {"status": "skipped",
                                "why": "no usable folds"}
    all_ok = [s for s in samples
              if _ok(s) and s.session in set(fold_sessions)]
    if not all_ok or len({_direction_cls(s) for s in all_ok}) < 2:
        return artifact
    fitp, val, acct = _availability_split(all_ok)
    y_fit = [_direction_cls(s) for s in fitp]
    if len(set(y_fit)) < 2:
        fitp, y_fit, val = list(all_ok), \
            [_direction_cls(s) for s in all_ok], []
    y_val = [_direction_cls(s) for s in val]
    names = _feature_names(all_ok)
    std = Standardizer().fit([s.features for s in fitp], names)
    dm_fit = build_design(fitp, std, std.feature_names)
    model = MultinomialLogisticModel().fit(dm_fit.x, y_fit)
    temperature = 1.0
    if val:
        dm_val = build_design(val, std, std.feature_names)
        temperature = fit_temperature_np(
            model.decision_scores(dm_val.x), y_val)
    path = f"{out_stem}_classical.json"
    meta = {"git_sha": _git_sha(),
            "dataset_hash": dataset_meta(dataset_path).get("dataset_hash"),
            "horizon": horizon,
            "standardizer": standardizer_state(std),
            "train_sessions": sorted({s.session for s in fitp}),
            "val_session": sorted({s.session for s in val}),
            "holdout_excluded": list(holdout)}
    digest = save_classical(path, model, dm_fit.columns, temperature,
                            meta=meta)
    re_model, _re_cols, re_temp, _re_meta = load_classical(path)
    probe = dm_fit.x[: min(64, len(dm_fit.x))]
    parity = float(np.max(np.abs(
        model.decision_scores(probe)
        - re_model.decision_scores(probe)))) if len(probe) else 0.0
    return {"status": "saved", "path": path,
            "checksum_sha256": digest,
            "n_features": len(dm_fit.columns),
            "temperature": float(re_temp),
            "purge_accounting": acct,
            "reload_parity_max_abs_diff": parity}


def _torch_ready() -> bool:
    """Call-time torch check (tests may monkeypatch mlp.TORCH_AVAILABLE)."""
    import oracle.ml.mlp as mlp_mod
    return bool(mlp_mod.TORCH_AVAILABLE)


def run_mlp(samples: Sequence[Sample], folds: Sequence[Fold],
            dataset_path: str, horizon: str, seed: int, n_boot: int,
            device: str, out_stem: str, holdout: Sequence[str],
            fold_sessions: Sequence[str]) -> Dict[str, Any]:
    """Phase 4 multitask MLP evaluated through the purged harness.

    Anchor rows are direction Samples (one per decision); the quantile /
    RV / hazard heads join their targets by decision_id and mask what is
    missing (never imputed).  Each fold trains on its train sessions
    minus a chronological validation tail (the last train session),
    early-stops and temperature-calibrates on that tail only, then
    predicts the embargo-protected test block.  Head targets in train
    and validation are additionally purge-gated against the next split's
    session start so no label overlaps the evaluation window.
    """
    import oracle.ml.mlp as mlp_mod

    dev = mlp_mod.resolve_device(device)
    by_id = {s.decision_id: s for s in samples}
    names = _feature_names(samples)
    causes = _first_touch_causes(dataset_path)
    und = _underlying_by_id(dataset_path)
    aux_logret = {s.decision_id: s
                  for s in load_samples(dataset_path, horizon, "log_return")}
    rv_samples = load_samples(dataset_path, horizon, "future_rv")
    aux_rv = {s.decision_id: s for s in rv_samples}
    aux_event = {s.decision_id: s
                 for s in load_samples(dataset_path, "event", "first_touch")}
    # point-in-time EWMA comparator for the RV head (same as run_rv)
    ewma_f, ewma_m = rv_ewma_forecast(rv_samples, und)
    ewma_by_id = {s.decision_id: (float(ewma_f[i]), bool(ewma_m[i]))
                  for i, s in enumerate(rv_samples)}
    cfg = mlp_mod.MLPConfig()
    n_bins = len(HAZARD_BIN_EDGES_MIN) - 1
    n_classes = len(HAZARD_CAUSES) + 1

    fold_rows: List[Dict[str, Any]] = []
    # direction pools (binary up-vs-not-up for harness parity)
    dir_s: List[Sample] = []
    dir_y: List[int] = []
    dir_p: List[float] = []
    dir_sess: List[str] = []
    dir_cls: List[int] = []
    dir_cls_pred: List[int] = []
    comp_pool: Dict[str, Dict[str, List[float]]] = {
        c: {"p": [], "mask": []}
        for c in ("constant", "momentum", "baseline_p_win")}
    # quantile pools
    q_y: List[float] = []
    q_sess: List[str] = []
    q_pred: Dict[float, List[float]] = {t: [] for t in cfg.quantiles}
    q_marg: Dict[float, List[float]] = {t: [] for t in cfg.quantiles}
    # rv pools
    rv_y: List[float] = []
    rv_sess: List[str] = []
    rv_f_model: List[float] = []
    rv_f_ewma: List[float] = []
    rv_m_ewma: List[bool] = []
    # hazard pools (per-session weighted per-subject NLL, like run_hazard)
    hz_model_sess: Dict[str, List[Tuple[float, int]]] = {}
    hz_marg_sess: Dict[str, List[Tuple[float, int]]] = {}
    hz_event_counts = {c: 0 for c in HAZARD_CAUSES}
    hz_cens = 0
    cif_final_rows: List[np.ndarray] = []
    cif_coherent = True

    for fold in folds:
        train_rows = [by_id[d] for d in fold.train_ids]
        test_rows = [by_id[d] for d in fold.test_ids]
        val_sess = max(fold.train_sessions) if fold.train_sessions else None
        sub_train = [s for s in train_rows if s.session != val_sess]
        val_rows = [s for s in train_rows if s.session == val_sess]
        if not sub_train or not val_rows or not test_rows:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": "no sub-train/validation/test rows"})
            continue
        val_start = _session_start_utc(val_sess)
        test_start = _session_start_utc(min(fold.test_sessions))

        std = Standardizer().fit([s.features for s in sub_train], names)
        dm_tr = build_design(sub_train, std, std.feature_names)
        dm_va = build_design(val_rows, std, std.feature_names)
        dm_te = build_design(test_rows, std, std.feature_names)

        hd_tr = mlp_mod.assemble_head_data(
            sub_train, aux_logret, aux_rv, aux_event, causes, val_start)
        hd_va = mlp_mod.assemble_head_data(
            val_rows, aux_logret, aux_rv, aux_event, causes, test_start)
        hd_te = mlp_mod.assemble_head_data(
            test_rows, aux_logret, aux_rv, aux_event, causes, None,
            include_censored_hazard=True)

        try:
            model, history = mlp_mod.train_mlp(
                dm_tr.x, hd_tr, dm_va.x, hd_va,
                cfg=cfg, seed=seed, device=dev)
        except (RuntimeError, ValueError) as exc:
            fold_rows.append({"fold_id": fold.fold_id, "skipped": True,
                              "why": f"train_mlp failed: {exc}"})
            continue

        # temperature from validation direction labels only
        va_idx = [i for i, c in enumerate(hd_va.direction_cls)
                  if c is not None]
        temperature = 1.0
        if va_idx:
            va_pred = mlp_mod.predict_heads(model, dm_va.x[va_idx],
                                            device=dev)
            temperature = mlp_mod.fit_temperature(
                va_pred["direction_logits"],
                np.asarray([hd_va.direction_cls[i] for i in va_idx]))
        preds = mlp_mod.predict_heads(model, dm_te.x, device=dev,
                                      temperature=temperature)

        # ---- direction pooling
        te_dir_idx = [i for i, c in enumerate(hd_te.direction_cls)
                      if c is not None]
        dir_train = [s for s in train_rows if _ok(s)]
        if te_dir_idx and dir_train:
            d_test = [test_rows[i] for i in te_dir_idx]
            y_train_bin = [_direction_y(s) for s in dir_train]
            c_const = constant_base_rate_probs(y_train_bin, len(d_test))
            c_mom, m_mom = momentum_rule_probs(dir_train, y_train_bin,
                                               d_test)
            c_base, m_base = baseline_p_win_probs(d_test)
            comp_pool["constant"]["p"].extend(c_const.tolist())
            comp_pool["constant"]["mask"].extend([1.0] * len(d_test))
            comp_pool["momentum"]["p"].extend(c_mom.tolist())
            comp_pool["momentum"]["mask"].extend(m_mom.tolist())
            comp_pool["baseline_p_win"]["p"].extend(c_base.tolist())
            comp_pool["baseline_p_win"]["mask"].extend(m_base.tolist())
            dir_s.extend(d_test)
            dir_y.extend(_direction_y(s) for s in d_test)
            dir_sess.extend(s.session for s in d_test)
            dir_p.extend(float(preds["p_up"][i]) for i in te_dir_idx)
            dir_cls.extend(int(hd_te.direction_cls[i]) for i in te_dir_idx)
            dir_cls_pred.extend(
                int(np.argmax(preds["direction_probs"][i]))
                for i in te_dir_idx)

        # ---- quantile pooling (marginal fit on purge-safe train+val)
        tr_lr = [v for v in hd_tr.log_return + hd_va.log_return
                 if v is not None]
        te_q_idx = [i for i, v in enumerate(hd_te.log_return)
                    if v is not None]
        if tr_lr and te_q_idx:
            marg = {t: float(np.quantile(np.asarray(tr_lr), t))
                    for t in cfg.quantiles}
            for i in te_q_idx:
                q_y.append(float(hd_te.log_return[i]))
                q_sess.append(test_rows[i].session)
                for j, t in enumerate(cfg.quantiles):
                    q_pred[t].append(float(preds["quantiles"][i, j]))
                    q_marg[t].append(marg[t])

        # ---- rv pooling
        for i in [k for k, v in enumerate(hd_te.rv) if v is not None]:
            rv_y.append(float(hd_te.rv[i]))
            rv_sess.append(test_rows[i].session)
            rv_f_model.append(float(preds["rv_forecast"][i]))
            f, m = ewma_by_id.get(test_rows[i].decision_id, (0.0, False))
            rv_f_ewma.append(f)
            rv_m_ewma.append(m)

        # ---- hazard pooling (model vs train-marginal table, paired NLL)
        tr_tc = [(t, c) for t, c in zip(
                     hd_tr.hazard_time + hd_va.hazard_time,
                     hd_tr.hazard_cause + hd_va.hazard_cause)
                 if t is not None]
        te_hz_idx = [i for i, t in enumerate(hd_te.hazard_time)
                     if t is not None]
        hz_fold: Optional[Dict[str, Any]] = None
        if len(tr_tc) >= 20 and te_hz_idx:
            pp_tr = expand_person_periods([t for t, _ in tr_tc],
                                          [c for _, c in tr_tc])
            table = _marginal_hazard_table(pp_tr, n_bins, n_classes)
            te_t = [float(hd_te.hazard_time[i]) for i in te_hz_idx]
            te_c = [hd_te.hazard_cause[i] for i in te_hz_idx]
            pp_te = expand_person_periods(te_t, te_c)
            hz_sel = preds["hazards"][te_hz_idx]
            row_nll = mlp_mod.hazard_row_nll(hz_sel, pp_te)
            marg_by_row: Dict[int, Tuple[float, int]] = {}
            for pp in pp_te:
                s_nll, n_pp = marg_by_row.get(pp.row_index, (0.0, 0))
                marg_by_row[pp.row_index] = (
                    s_nll - float(np.log(table[pp.time_bin, pp.outcome])),
                    n_pp + 1)
            for ridx, (s_nll, n_pp) in row_nll.items():
                sess = test_rows[te_hz_idx[ridx]].session
                hz_model_sess.setdefault(sess, []).append(
                    (s_nll / n_pp, n_pp))
            for ridx, (s_nll, n_pp) in marg_by_row.items():
                sess = test_rows[te_hz_idx[ridx]].session
                hz_marg_sess.setdefault(sess, []).append(
                    (s_nll / n_pp, n_pp))
            for c in te_c:
                if c is None:
                    hz_cens += 1
                else:
                    hz_event_counts[c] += 1
            cif = mlp_mod.cif_from_hazards(hz_sel)
            cif_coherent = cif_coherent and bool(
                np.all(np.diff(cif, axis=1) >= -1e-9)
                and np.all(cif.sum(axis=2) <= 1.0 + 1e-9))
            cif_final_rows.append(cif[:, -1, :])
            hz_fold = {"n_train_periods": len(pp_tr),
                       "n_test_periods": len(pp_te),
                       "n_test_subjects": len(te_hz_idx)}

        row: Dict[str, Any] = {
            "fold_id": fold.fold_id, "skipped": False,
            "train_sessions": fold.train_sessions,
            "test_sessions": fold.test_sessions,
            "val_session": val_sess,
            "purged": fold.purged, "embargoed": fold.embargoed,
            "temperature": float(temperature),
            "head_counts_train": hd_tr.counts(),
            "head_counts_val": hd_va.counts(),
            "head_counts_test": hd_te.counts(),
            "history": {k: history.get(k) for k in (
                "epochs_run", "best_epoch", "stopped_early",
                "train_loss_at_best", "val_loss_at_best",
                "train_val_gap", "val_head_losses_at_best")},
        }
        if hz_fold is not None:
            row["hazard"] = hz_fold
        fold_rows.append(row)

    result: Dict[str, Any] = {
        "folds": fold_rows,
        "device": dev,
        "config": {"hidden": list(cfg.hidden), "dropout": cfg.dropout,
                   "lr": cfg.lr, "weight_decay": cfg.weight_decay,
                   "batch_size": cfg.batch_size,
                   "max_epochs": cfg.max_epochs, "patience": cfg.patience,
                   "grad_clip": cfg.grad_clip,
                   "quantiles": list(cfg.quantiles)},
        "note": "multitask heads share one trunk; each head is evaluated "
                "only on rows whose target exists (masked, never imputed); "
                "validation tail = last train session, used for early "
                "stopping and temperature calibration only",
        "heads": {},
    }

    # ---- direction head (pooled)
    if dir_p:
        pooled = binary_metrics(dir_y, dir_p)
        bins = reliability_bins(dir_y, dir_p)
        pooled["ece"] = expected_calibration_error(bins)
        comparators: Dict[str, Any] = {}
        deltas: Dict[str, Any] = {}
        for cname, pool in comp_pool.items():
            mask = pool["mask"]
            ys = [y for y, m in zip(dir_y, mask) if m]
            ps = [p for p, m in zip(pool["p"], mask) if m]
            met = binary_metrics(ys, ps)
            met["ece"] = expected_calibration_error(
                reliability_bins(ys, ps))
            met["n_unavailable"] = int(len(mask) - sum(mask))
            comparators[cname] = met
            comp_sess = _per_session_log_loss(dir_sess, dir_y,
                                              pool["p"], mask)
            chall_sess = _per_session_log_loss(dir_sess, dir_y, dir_p,
                                               mask)  # paired rows
            deltas[cname] = _delta_report(comp_sess, chall_sess,
                                          n_boot, seed)
        result["heads"]["direction"] = {
            "pooled": pooled,
            "accuracy_3class": float(np.mean(
                np.asarray(dir_cls_pred) == np.asarray(dir_cls))),
            "reliability_bins": bins,
            "by_dte_bucket": _stratified_binary(
                dir_s, dir_y, dir_p,
                lambda s: dte_bucket(s.features.get("dte_days"))),
            "by_kind": _stratified_binary(dir_s, dir_y, dir_p,
                                          lambda s: s.kind),
            "comparators": comparators,
            "deltas_log_loss": deltas,
        }
    else:
        result["heads"]["direction"] = None

    # ---- quantile head (pooled)
    if q_y:
        per_tau = {t: {"pinball_model": pinball_loss(q_y, q_pred[t], t),
                       "pinball_marginal": pinball_loss(q_y, q_marg[t], t)}
                   for t in cfg.quantiles}

        def _sess_pinball(pred: Dict[float, List[float]]
                          ) -> Dict[str, float]:
            groups: Dict[str, List[int]] = {}
            for i, sess in enumerate(q_sess):
                groups.setdefault(sess, []).append(i)
            out = {}
            for sess, idx in groups.items():
                losses = [pinball_loss([q_y[i] for i in idx],
                                       [pred[t][i] for i in idx], t)
                          for t in cfg.quantiles]
                vals = [v for v in losses if v is not None]
                if vals:
                    out[sess] = float(np.mean(vals))
            return out

        result["heads"]["quantile"] = {
            "n": len(q_y),
            "per_tau": per_tau,
            "coverage_90_model": interval_coverage(q_y, q_pred[0.05],
                                                   q_pred[0.95]),
            "coverage_90_marginal": interval_coverage(q_y, q_marg[0.05],
                                                      q_marg[0.95]),
            "coverage_50_model": interval_coverage(q_y, q_pred[0.25],
                                                   q_pred[0.75]),
            "coverage_50_marginal": interval_coverage(q_y, q_marg[0.25],
                                                      q_marg[0.75]),
            "deltas_pinball": {"marginal": _delta_report(
                _sess_pinball(q_marg), _sess_pinball(q_pred),
                n_boot, seed)},
        }
    else:
        result["heads"]["quantile"] = None

    # ---- rv head (pooled; paired vs point-in-time EWMA)
    if rv_y:
        pooled_rv = {"model": {
            "n": len(rv_y),
            "rmse": float(np.sqrt(np.mean(
                (np.asarray(rv_y) - np.asarray(rv_f_model)) ** 2))),
            "qlike": qlike_loss(rv_y, rv_f_model)}}
        ys_e = [y for y, m in zip(rv_y, rv_m_ewma) if m]
        fs_e = [f for f, m in zip(rv_f_ewma, rv_m_ewma) if m]
        pooled_rv["ewma"] = {
            "n": len(ys_e),
            "rmse": (float(np.sqrt(np.mean(
                (np.asarray(ys_e) - np.asarray(fs_e)) ** 2)))
                if ys_e else None),
            "qlike": qlike_loss(ys_e, fs_e),
            "n_unavailable": int(len(rv_y) - len(ys_e))}
        g_m: Dict[str, Tuple[List[float], List[float]]] = {}
        g_e: Dict[str, Tuple[List[float], List[float]]] = {}
        for y, fm, fe, m, sess in zip(rv_y, rv_f_model, rv_f_ewma,
                                      rv_m_ewma, rv_sess):
            if not m:
                continue        # pair only on EWMA-available rows
            a, b = g_m.setdefault(sess, ([], []))
            a.append(y)
            b.append(fm)
            a2, b2 = g_e.setdefault(sess, ([], []))
            a2.append(y)
            b2.append(fe)
        sess_m = {s: q for s, (a, b) in g_m.items()
                  if (q := qlike_loss(a, b)) is not None}
        sess_e = {s: q for s, (a, b) in g_e.items()
                  if (q := qlike_loss(a, b)) is not None}
        result["heads"]["rv"] = {
            "pooled": pooled_rv,
            "deltas_qlike": {"model_vs_ewma": _delta_report(
                sess_e, sess_m, n_boot, seed)},
        }
    else:
        result["heads"]["rv"] = None

    # ---- hazard head (pooled)
    if hz_model_sess:
        def _weighted(d: Dict[str, List[Tuple[float, int]]]
                      ) -> Dict[str, float]:
            return {sess: sum(v * n for v, n in rows)
                    / sum(n for _, n in rows)
                    for sess, rows in d.items()}

        tot_n = sum(n for rows in hz_model_sess.values() for _, n in rows)
        mean_cif = (np.vstack(cif_final_rows).mean(axis=0)
                    if cif_final_rows else None)
        result["heads"]["hazard"] = {
            "pooled_nll_model": (sum(
                v * n for rows in hz_model_sess.values()
                for v, n in rows) / tot_n) if tot_n else None,
            "pooled_nll_marginal": (sum(
                v * n for rows in hz_marg_sess.values()
                for v, n in rows) / tot_n) if tot_n else None,
            "deltas_nll": {"marginal": _delta_report(
                _weighted(hz_marg_sess), _weighted(hz_model_sess),
                n_boot, seed)},
            "cif_coherent": bool(cif_coherent),
            "mean_model_cif_final_bin": (
                {c: float(mean_cif[k])
                 for k, c in enumerate(HAZARD_CAUSES)}
                if mean_cif is not None else None),
            "test_events_by_cause": hz_event_counts,
            "test_censored": hz_cens,
        }
    else:
        result["heads"]["hazard"] = None

    # ---- final artifact: train on all fold sessions, purged vs holdout
    fold_set = set(fold_sessions)
    final_rows = [s for s in samples if s.session in fold_set]
    f_val_sess = max(fold_sessions) if fold_sessions else None
    f_sub = [s for s in final_rows if s.session != f_val_sess]
    f_val = [s for s in final_rows if s.session == f_val_sess]
    if len(fold_sessions) >= 2 and f_sub and f_val:
        f_val_start = _session_start_utc(f_val_sess)
        hold_start = _session_start_utc(min(holdout)) if holdout else None
        std_f = Standardizer().fit([s.features for s in f_sub], names)
        dm_sub = build_design(f_sub, std_f, std_f.feature_names)
        dm_val = build_design(f_val, std_f, std_f.feature_names)
        hd_sub = mlp_mod.assemble_head_data(
            f_sub, aux_logret, aux_rv, aux_event, causes, f_val_start)
        hd_val = mlp_mod.assemble_head_data(
            f_val, aux_logret, aux_rv, aux_event, causes, hold_start)
        try:
            model_f, hist_f = mlp_mod.train_mlp(
                dm_sub.x, hd_sub, dm_val.x, hd_val,
                cfg=cfg, seed=seed, device=dev)
            v_idx = [i for i, c in enumerate(hd_val.direction_cls)
                     if c is not None]
            temp_f = 1.0
            if v_idx:
                vp = mlp_mod.predict_heads(model_f, dm_val.x[v_idx],
                                           device=dev)
                temp_f = mlp_mod.fit_temperature(
                    vp["direction_logits"],
                    np.asarray([hd_val.direction_cls[i] for i in v_idx]))
            manifest = mlp_mod.save_model(
                model_f, out_stem,
                feature_columns=dm_sub.columns,
                standardizer_state=mlp_mod.standardizer_state(std_f),
                temperature=temp_f,
                seed=seed,
                git_sha=_git_sha(),
                dataset_hash=dataset_meta(dataset_path).get("dataset_hash"),
                train_sessions=list(fold_sessions),
                val_session=f_val_sess,
                extra={"horizon": horizon})
            loaded, _mf = mlp_mod.load_model(
                f"{out_stem}_manifest.json", device="cpu")
            probe = (dm_val.x if len(dm_val.x) else dm_sub.x)[:32]
            p1 = mlp_mod.predict_heads(model_f, probe, device="cpu",
                                       temperature=temp_f)
            p2 = mlp_mod.predict_heads(loaded, probe, device="cpu",
                                       temperature=temp_f)
            max_diff = max(float(np.max(np.abs(p1[k] - p2[k])))
                           for k in p1)
            result["artifact"] = {
                "status": "saved",
                "manifest": f"{out_stem}_manifest.json",
                "model_file": f"{out_stem}_model.pt",
                "checksum_sha256": manifest.get("checksum_sha256"),
                "temperature": float(temp_f),
                "val_session": f_val_sess,
                "history": {k: hist_f.get(k) for k in (
                    "epochs_run", "best_epoch", "stopped_early",
                    "train_loss_at_best", "val_loss_at_best",
                    "train_val_gap")},
                "reload_parity": {"max_abs_diff": max_diff,
                                  "ok": bool(max_diff < 1e-5)},
            }
        except (RuntimeError, ValueError) as exc:
            result["artifact"] = {"status": "skipped",
                                  "why": f"final fit failed: {exc}"}
    else:
        result["artifact"] = {"status": "skipped",
                              "why": "fewer than 2 fold sessions with rows"}
    return result


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Oracle 4.0 Phases 3-4 challenger training/evaluation "
                    "(research-only; evaluated through the purged "
                    "walk-forward harness)")
    p.add_argument("--model", required=True, choices=MODELS)
    p.add_argument("--dataset", required=True)
    p.add_argument("--horizon", default="60m",
                   choices=["15m", "60m", "1s", "5s", "event"])
    p.add_argument("--label-type", default="direction")
    p.add_argument("--kinds", nargs="*", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--device", default="auto",
                   help="torch device for --model mlp: auto|cpu|cuda "
                        "(recorded in the report)")
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--fold-config", default=None,
                   help="JSON overrides: test_block_sessions, "
                        "min_train_sessions, embargo_sessions, "
                        "holdout_sessions")
    p.add_argument("--holdout-registration", default=None,
                   help="verify/record the write-once holdout pledge")
    args = p.parse_args(argv)

    fold_cfg = {"test_block_sessions": 1, "min_train_sessions": 2,
                "embargo_sessions": HORIZON_EMBARGO_SESSIONS.get(
                    args.horizon, 1),
                "holdout_sessions": 2}
    if args.fold_config:
        fold_cfg.update(json.loads(args.fold_config))

    report: Dict[str, Any] = {
        "model": args.model,
        "horizon": args.horizon,
        "label_type": args.label_type,
        "kinds": args.kinds,
        "seed": args.seed,
        "device": args.device,
        "dataset_path": args.dataset,
        "git_sha": _git_sha(),
        "fold_config": fold_cfg,
    }

    if args.model == "mlp" and args.label_type != "direction":
        p.error("--model mlp anchors rows on direction labels; "
                "use --label-type direction")
    if args.model == "gbm" and args.label_type != "direction":
        p.error("--model gbm anchors rows on direction labels; "
                "use --label-type direction")

    if args.model == "gbm" and not _lgbm_ready():
        import oracle.ml.gbm as gbm_mod
        report["result"] = {
            "status": "unavailable",
            "why": gbm_mod.LGBM_WHY}
    elif args.model == "mlp" and not _torch_ready():
        report["result"] = {
            "status": "unavailable",
            "why": "torch not installed (optional dependency); "
                   "result reported honestly, never faked"}
    else:
        meta = dataset_meta(args.dataset)
        report["dataset_hash"] = meta.get("dataset_hash")
        samples = load_samples(args.dataset, args.horizon, args.label_type,
                               kinds=args.kinds)
        folds, fold_sessions, holdout = make_folds(samples, **fold_cfg)
        report["sessions"] = {"folds": fold_sessions, "holdout": holdout}
        report["samples"] = _status_counts(samples)
        if args.holdout_registration and holdout:
            n_hold = sum(1 for s in samples if s.session in set(holdout))
            register_holdout(args.holdout_registration,
                             meta.get("dataset_hash", ""), holdout, n_hold)
            report["holdout_registration"] = args.holdout_registration

        if args.model == "logistic":
            report["result"] = run_direction(samples, folds, args.seed,
                                             args.n_boot, "logistic")
            out_stem = (args.out[:-5] if args.out.endswith(".json")
                        else args.out)
            # top-level (not under "result"): the artifact block carries the
            # run-specific output path, while "result" must stay identical
            # across reruns (pinned by the CLI determinism test)
            report["artifact"] = run_classical_artifact(
                samples, args.dataset, args.horizon, out_stem,
                fold_sessions, holdout)
        elif args.model == "naive":
            report["result"] = run_direction(samples, folds, args.seed,
                                             args.n_boot, "naive")
        elif args.model == "quantile":
            report["result"] = run_quantile(samples, folds, args.seed,
                                            args.n_boot)
        elif args.model == "rv":
            report["result"] = run_rv(samples, folds, args.dataset,
                                      args.seed, args.n_boot)
        elif args.model == "hazard":
            report["result"] = run_hazard(samples, folds, args.dataset,
                                          args.seed, args.n_boot)
        elif args.model == "gbm":
            out_stem = (args.out[:-5] if args.out.endswith(".json")
                        else args.out)
            report["result"] = run_gbm(samples, folds, args.dataset,
                                       args.horizon, args.seed,
                                       args.n_boot, out_stem,
                                       fold_sessions, holdout)
        elif args.model == "mlp":
            out_stem = (args.out[:-5] if args.out.endswith(".json")
                        else args.out)
            report["result"] = run_mlp(samples, folds, args.dataset,
                                       args.horizon, args.seed,
                                       args.n_boot, args.device, out_stem,
                                       holdout, fold_sessions)

    clean = _py(report)
    clean["report_fingerprint"] = metrics_fingerprint(clean)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(clean, fh, indent=2, sort_keys=True)
    print(f"model={args.model} horizon={args.horizon} "
          f"label={args.label_type} -> {args.out}")
    print(f"fingerprint={clean['report_fingerprint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
