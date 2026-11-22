"""Oracle 4.0.2 P0-4 -- competing EV models and calibrated executable EV.

Extends the Stage D study (``oracle.ml.pnl_calibration``) into a competing
model suite, all fit on prior sessions only (chronological split with the
same hard leak guard) and scored strictly out-of-sample:

  A ``base_rate``        historical train win rate (PoP floor)
  B ``logistic_recal``   logistic recalibration of the stamped PoP
  C ``lgbm_pop``         gradient-boosted PoP classifier (LightGBM)
  D ``lgbm_profit``      gradient-boosted net-P&L regressor (USD)
  E ``lgbm_quantile``    net-P&L quantile regressors (q10 / q50 / q90)
  F ``neural_profit``    small MLP net-P&L regressor (torch, CPU)
  G ``cohort_pop`` / ``cohort_profit``
                         regime|side|dte hierarchical shrinkage models

Executable EV is composed as ``P(fill) * E[net profit | fill]`` with
explicit no-double-count cost handling: the profit models are trained on
*realized net* P&L, so execution costs are embedded in the label and are
NEVER subtracted again after the model.  (The structural per-quote layer,
``oracle.execution.executable_ev``, keeps its own single-charge accounting
and requires GROSS theoretical EV -- the two layers are never chained.)

P(fill) cannot be learned from the trade history because that source only
contains resolved fills (survivorship); it is taken from the conservative
shrunk prior in ``oracle.ml.execution`` and labelled as such.

Honesty rules: optional dependencies that are missing produce explicit
``unavailable`` model entries (never fabricated numbers); the verdict is
pre-registered and declares ``NO_VALIDATED_EDGE`` whenever no profit model
shows out-of-sample rank power (session-bootstrap 95% LB > 0 and positive
OLS slope of realized on predicted).

Research-only; no production module imports this file.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from . import pnl_calibration as pc
from .execution import (
    CONSERVATIVE_FILL_PRIOR,
    PRIOR_STRENGTH,
    shrunk_fill_probability,
)

try:  # optional: models C / D / E
    import numpy as _np
    import lightgbm as _lgbm
    _HAS_LGBM = True
except Exception:  # pragma: no cover - environment dependent
    _np = None
    _lgbm = None
    _HAS_LGBM = False

try:  # optional: model F
    import torch as _torch
    _HAS_TORCH = True
except Exception:  # pragma: no cover - environment dependent
    _torch = None
    _HAS_TORCH = False

__all__ = [
    "FEATURE_NAMES",
    "extract_opportunities",
    "feature_matrix",
    "cohort_profit",
    "evaluate_models",
    "executable_ev_from_model",
    "run_ev_model_study",
]

UTC = timezone.utc

SEED = 11                      # pre-registered model seed
QUANTILES = (0.10, 0.50, 0.90)
N_BUCKETS = 5
MIN_EVAL_N = pc.MIN_EVAL_N     # shared pre-registered evidence floor

# Entry-time features frozen before any outcome is known (entry_ev_stamp v3
# metrics + contract descriptors).  Missing values become NaN; the pure
# python models never read the matrix, LightGBM handles NaN natively and
# the neural model imputes with train medians.
FEATURE_NAMES = (
    "p_pred", "net_ev", "gross_ev", "round_trip_costs", "max_loss",
    "ev_per_dollar_risk", "signal_strength", "entry_delta",
    "take_profit_pct", "stop_loss_pct", "dte", "side_is_call",
)

COST_HANDLING_NOTE = (
    "profit models are trained on realized NET P&L: execution costs are "
    "embedded in the training label and are NOT subtracted again after "
    "the model (double-count guard). The structural per-quote layer "
    "(oracle.execution.executable_ev) keeps its own single-charge "
    "accounting over GROSS theoretical EV; the two layers are never "
    "chained on the same dollars.")

SURVIVORSHIP_NOTE = (
    "history rows are resolved paper fills only; unfilled opportunities "
    "are structurally absent, so P(fill) is NOT learnable from this "
    "source and realized expectancy per opportunity equals per fill "
    "(survivorship limitation, reported -- never hidden).")


# --------------------------------------------------------------------------- #
# opportunity extraction (superset of pnl_calibration's opportunities)
# --------------------------------------------------------------------------- #

def extract_opportunities(
    trades: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Resolved history trades -> model opportunities with entry features.

    Mirrors ``pnl_calibration.opportunities_from_history`` gates exactly
    (same-unit ``ev_units == "usd_total"`` enforcement, finite pnl / PoP /
    EV, parseable entry timestamp) and adds a ``features`` dict of frozen
    entry-time inputs for the learned models.
    """
    report = {
        "rows_scanned": 0,
        "kept": 0,
        "dropped_missing_fields": 0,
        "dropped_unit_mismatch": 0,
        "dropped_bad_timestamp": 0,
    }
    out: List[Dict[str, Any]] = []
    for t in trades:
        report["rows_scanned"] += 1
        pnl = pc._f(t.get("pnl"))
        p_pred = pc._f(t.get("probability_of_profit"))
        ev_pred = pc._f(t.get("expected_value"))
        if pnl is None or p_pred is None or ev_pred is None:
            report["dropped_missing_fields"] += 1
            continue
        metrics = t.get("metrics") or {}
        if metrics.get("ev_units") != "usd_total":
            report["dropped_unit_mismatch"] += 1
            continue
        entry = pc._parse_dt(t.get("entry_time"))
        if entry is None:
            report["dropped_bad_timestamp"] += 1
            continue
        underlying, expiry, side = pc._occ_parts(t.get("symbol"))
        dte = (expiry - entry.date()).days if expiry else None
        p_clamped = min(max(p_pred, 0.0), 1.0)
        rtc = pc._f(metrics.get("round_trip_costs"))
        gross = pc._f(metrics.get("gross_expected_value"))
        if gross is None and rtc is not None:
            # pre-v3 stamps: documented reconstruction (net + costs)
            gross = ev_pred + rtc
        features = {
            "p_pred": p_clamped,
            "net_ev": ev_pred,
            "gross_ev": gross,
            "round_trip_costs": rtc,
            "max_loss": pc._f(metrics.get("max_loss")),
            "ev_per_dollar_risk": pc._f(metrics.get("ev_per_dollar_risk")),
            "signal_strength": pc._f(metrics.get("signal_strength")),
            "entry_delta": pc._f(metrics.get("entry_delta")),
            "take_profit_pct": pc._f(metrics.get("take_profit_pct")),
            "stop_loss_pct": pc._f(metrics.get("stop_loss_pct")),
            "dte": float(dte) if dte is not None else None,
            "side_is_call": (1.0 if side == "call"
                             else 0.0 if side == "put" else None),
        }
        out.append({
            "entry_time": entry,
            "session": entry.date().isoformat(),
            "symbol": t.get("symbol"),
            "underlying": underlying,
            "side": side or "unknown",
            "dte": dte,
            "dte_bucket": pc._dte_bucket(dte),
            "regime": t.get("regime_label") or "unknown",
            "p_pred": p_clamped,
            "ev_pred": ev_pred,
            "realized": pnl,
            "win": pnl > 0.0,
            "provenance": "paper_marks",
            "features": features,
        })
        report["kept"] += 1
    out.sort(key=lambda o: (o["entry_time"], str(o["symbol"])))
    return out, report


def _fv(v: Any) -> float:
    if isinstance(v, bool) or not isinstance(v, (int, float)):
        return math.nan
    return float(v) if math.isfinite(v) else math.nan


def feature_matrix(opps: Sequence[Dict[str, Any]]) -> List[List[float]]:
    """Rows of floats in FEATURE_NAMES order; missing values become NaN."""
    rows: List[List[float]] = []
    for o in opps:
        f = o.get("features") or {}
        rows.append([_fv(f.get(name)) for name in FEATURE_NAMES])
    return rows


# --------------------------------------------------------------------------- #
# pure-python helpers (imputation / scaling for the neural model)
# --------------------------------------------------------------------------- #

def _col_medians(X: Sequence[Sequence[float]]) -> List[float]:
    meds: List[float] = []
    for j in range(len(FEATURE_NAMES)):
        vals = sorted(r[j] for r in X if not math.isnan(r[j]))
        if not vals:
            meds.append(0.0)
        else:
            m = len(vals) // 2
            meds.append(vals[m] if len(vals) % 2 else
                        0.5 * (vals[m - 1] + vals[m]))
    return meds


def _impute(X: Sequence[Sequence[float]],
            meds: Sequence[float]) -> List[List[float]]:
    return [[meds[j] if math.isnan(v) else v for j, v in enumerate(r)]
            for r in X]


def _col_stats(X: Sequence[Sequence[float]]
               ) -> Tuple[List[float], List[float]]:
    n = len(X)
    mus, sds = [], []
    for j in range(len(FEATURE_NAMES)):
        col = [r[j] for r in X]
        mu = sum(col) / n
        sd = math.sqrt(sum((v - mu) ** 2 for v in col) / n)
        mus.append(mu)
        sds.append(max(sd, 1e-9))
    return mus, sds


def _scale(X: Sequence[Sequence[float]], mus: Sequence[float],
           sds: Sequence[float]) -> List[List[float]]:
    return [[(v - mus[j]) / sds[j] for j, v in enumerate(r)] for r in X]


# --------------------------------------------------------------------------- #
# learned models (fit on train only); each returns (predict_fn | None, reason)
# --------------------------------------------------------------------------- #

def _lgbm_params(**extra: Any) -> Dict[str, Any]:
    p: Dict[str, Any] = dict(
        n_estimators=150, learning_rate=0.05, num_leaves=15,
        min_child_samples=10, subsample=1.0, colsample_bytree=1.0,
        random_state=SEED, n_jobs=1, deterministic=True,
        force_row_wise=True, verbosity=-1)
    p.update(extra)
    return p


def _fit_lgbm_pop(Xtr: Sequence[Sequence[float]], ytr: Sequence[bool]
                  ) -> Tuple[Optional[Callable], Optional[str]]:
    if not _HAS_LGBM:
        return None, "LIGHTGBM_UNAVAILABLE"
    if len(set(ytr)) < 2:
        return None, "DEGENERATE_LABELS"
    clf = _lgbm.LGBMClassifier(objective="binary", **_lgbm_params())
    clf.fit(_np.asarray(Xtr, dtype=float),
            _np.asarray([1 if y else 0 for y in ytr]))

    def predict(X: Sequence[Sequence[float]]) -> List[float]:
        pr = clf.predict_proba(_np.asarray(X, dtype=float))[:, 1]
        return [float(min(max(v, 0.0), 1.0)) for v in pr]
    return predict, None


def _fit_lgbm_regressor(Xtr: Sequence[Sequence[float]],
                        ytr: Sequence[float], **objective: Any
                        ) -> Tuple[Optional[Callable], Optional[str]]:
    if not _HAS_LGBM:
        return None, "LIGHTGBM_UNAVAILABLE"
    reg = _lgbm.LGBMRegressor(**_lgbm_params(**objective))
    reg.fit(_np.asarray(Xtr, dtype=float), _np.asarray(ytr, dtype=float))

    def predict(X: Sequence[Sequence[float]]) -> List[float]:
        return [float(v) for v in reg.predict(_np.asarray(X, dtype=float))]
    return predict, None


def _fit_neural_profit(Xtr: Sequence[Sequence[float]], ytr: Sequence[float]
                       ) -> Tuple[Optional[Callable], Optional[str]]:
    if not _HAS_TORCH:
        return None, "TORCH_UNAVAILABLE"
    meds = _col_medians(Xtr)
    Xi = _impute(Xtr, meds)
    mus, sds = _col_stats(Xi)
    Xs = _scale(Xi, mus, sds)
    n = len(ytr)
    ymu = sum(ytr) / n
    ysd = max(math.sqrt(sum((v - ymu) ** 2 for v in ytr) / n), 1e-9)
    ys = [(v - ymu) / ysd for v in ytr]

    _torch.manual_seed(SEED)
    dev = _torch.device("cpu")  # CPU for determinism
    net = _torch.nn.Sequential(
        _torch.nn.Linear(len(FEATURE_NAMES), 16), _torch.nn.ReLU(),
        _torch.nn.Linear(16, 8), _torch.nn.ReLU(),
        _torch.nn.Linear(8, 1)).to(dev)
    opt = _torch.optim.Adam(net.parameters(), lr=0.01)
    xt = _torch.tensor(Xs, dtype=_torch.float32, device=dev)
    yt = _torch.tensor(ys, dtype=_torch.float32, device=dev).view(-1, 1)
    loss_fn = _torch.nn.MSELoss()
    net.train()
    for _ in range(300):  # full-batch; tiny net, deterministic on CPU
        opt.zero_grad()
        loss = loss_fn(net(xt), yt)
        loss.backward()
        opt.step()
    net.eval()

    def predict(X: Sequence[Sequence[float]]) -> List[float]:
        Z = _scale(_impute(X, meds), mus, sds)
        with _torch.no_grad():
            out = net(_torch.tensor(Z, dtype=_torch.float32,
                                    device=dev)).view(-1).tolist()
        return [float(v) * ysd + ymu for v in out]
    return predict, None


def cohort_profit(train: Sequence[Dict[str, Any]],
                  k: float = pc.SHRINKAGE_K) -> Dict[str, Any]:
    """Mean realized net P&L per (regime, side, dte_bucket) cohort with
    hierarchical shrinkage toward the global train mean (model G)."""
    if not train:
        raise ValueError("cohort_profit needs a non-empty training set")
    g = sum(o["realized"] for o in train) / len(train)
    agg: Dict[str, List[float]] = {}
    for o in train:
        key = f"{o['regime']}|{o['side']}|{o['dte_bucket']}"
        c = agg.setdefault(key, [0.0, 0.0])
        c[0] += 1.0
        c[1] += o["realized"]
    means = {key: (c[1] + k * g) / (c[0] + k) for key, c in agg.items()}
    return {"global_mean": g, "k": k, "means": means,
            "cohort_n": {key: int(c[0]) for key, c in agg.items()}}


def _cohort_profit_predict(model: Dict[str, Any],
                           o: Dict[str, Any]) -> float:
    key = f"{o['regime']}|{o['side']}|{o['dte_bucket']}"
    return model["means"].get(key, model["global_mean"])


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _session_bootstrap_preds(
    preds: Sequence[float],
    eval_: Sequence[Dict[str, Any]],
    n_boot: int = pc.N_BOOT,
    seed: int = pc.BOOT_SEED,
) -> Optional[Dict[str, Any]]:
    """Session-block bootstrap CI for Spearman(preds, realized)."""
    by_session: Dict[str, List[int]] = {}
    for i, o in enumerate(eval_):
        by_session.setdefault(o["session"], []).append(i)
    sessions = sorted(by_session)
    if len(sessions) < 2:
        return None
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        idxs: List[int] = []
        for _ in sessions:
            idxs.extend(by_session[rng.choice(sessions)])
        rho = pc.spearman([preds[i] for i in idxs],
                          [eval_[i]["realized"] for i in idxs])
        if rho is not None:
            draws.append(rho)
    if len(draws) < n_boot // 2:
        return None
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"n_boot": len(draws), "seed": seed,
            "ci95": [lo, hi], "lower_bound": lo}


def _bucket_table(train_preds: Sequence[float],
                  eval_preds: Sequence[float],
                  realized: Sequence[float],
                  n_buckets: int = N_BUCKETS) -> Dict[str, Any]:
    """Quantile buckets from TRAIN predictions; eval realized per bucket,
    plus the fraction of adjacent populated buckets that are monotone."""
    svals = sorted(train_preds)
    edges = []
    for i in range(1, n_buckets):
        idx = min(len(svals) - 1,
                  max(0, int(round(i * len(svals) / n_buckets))))
        edges.append(svals[idx])

    def bucket_of(v: float) -> int:
        for i, e in enumerate(edges):
            if v < e:
                return i
        return len(edges)

    rows: List[Dict[str, Any]] = []
    for i in range(len(edges) + 1):
        vals = [r for p, r in zip(eval_preds, realized) if bucket_of(p) == i]
        preds_in = [p for p in eval_preds if bucket_of(p) == i]
        rows.append({
            "bucket": i,
            "n": len(vals),
            "mean_pred": (sum(preds_in) / len(preds_in)
                          if preds_in else None),
            "mean_realized": (sum(vals) / len(vals) if vals else None),
        })
    means = [r["mean_realized"] for r in rows if r["mean_realized"] is not None]
    pairs = max(len(means) - 1, 0)
    mono = (sum(1 for a, b in zip(means, means[1:]) if b >= a) / pairs
            if pairs else None)
    return {"edges_from_train": edges, "rows": rows,
            "monotonic_fraction": mono}


def _model_verdict(n_eval: int, boot: Optional[Dict[str, Any]],
                   slope: Optional[float]) -> Dict[str, Any]:
    """Pre-registered per-model rule (no tuning on the eval set)."""
    if n_eval < MIN_EVAL_N:
        return {"validated": False,
                "reason": f"only {n_eval} eval rows (< {MIN_EVAL_N})"}
    lb = boot["lower_bound"] if boot else None
    if lb is None or slope is None or lb <= 0.0 or slope <= 0.0:
        return {"validated": False,
                "reason": (f"spearman 95% LB={lb} / OLS slope={slope}: no "
                           "out-of-sample rank power")}
    return {"validated": True,
            "reason": (f"spearman 95% LB={lb:.4f} > 0 and OLS slope="
                       f"{slope:.4f} > 0 on forward sessions")}


def _profit_entry(train_preds: Sequence[float],
                  eval_preds: Sequence[float],
                  eval_: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    realized = [o["realized"] for o in eval_]
    rho = pc.spearman(list(eval_preds), realized)
    boot = _session_bootstrap_preds(eval_preds, eval_)
    slope, intercept = pc._ols(list(eval_preds), realized)
    mae = sum(abs(r - p) for r, p in zip(realized, eval_preds)) / len(realized)
    return {
        "status": "ok",
        "spearman": rho,
        "session_bootstrap": boot,
        "ols_slope_realized_on_pred": slope,
        "ols_intercept": intercept,
        "mae_usd": mae,
        "eval_mean_pred_usd": sum(eval_preds) / len(eval_preds),
        "ev_buckets": _bucket_table(train_preds, eval_preds, realized),
        "verdict": _model_verdict(len(eval_), boot, slope),
    }


def _pinball(preds: Sequence[float], realized: Sequence[float],
             alpha: float) -> float:
    total = 0.0
    for q, y in zip(preds, realized):
        d = y - q
        total += alpha * d if d >= 0 else (alpha - 1.0) * d
    return total / len(preds)


# --------------------------------------------------------------------------- #
# evaluation (fit on train only; score on forward sessions)
# --------------------------------------------------------------------------- #

def evaluate_models(
    train: Sequence[Dict[str, Any]],
    eval_: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit the competing model suite on ``train`` and score on ``eval_``.

    Every model parameter comes from training sessions only; the split is
    re-validated here so leaked inputs raise instead of scoring."""
    pc.validate_split(train, eval_)
    if not train or not eval_:
        raise ValueError("evaluate_models needs non-empty train and eval sets")

    ys_tr = [o["win"] for o in train]
    ys_ev = [o["win"] for o in eval_]
    rz_tr = [o["realized"] for o in train]
    Xtr = feature_matrix(train)
    Xev = feature_matrix(eval_)

    # ---- fits (train only)
    base_rate = sum(1 for y in ys_tr if y) / len(train)
    rec_a, rec_b = pc.fit_logistic(
        [pc._logit(o["p_pred"]) for o in train], ys_tr)
    cohorts = pc.cohort_rates(train)
    cprofit = cohort_profit(train)

    # ---- PoP models (A, B, C, G + stamped reference)
    pop_models: Dict[str, Any] = {}
    pop_preds: Dict[str, List[float]] = {
        "stamped_pop": [o["p_pred"] for o in eval_],
        "base_rate": [base_rate] * len(eval_),
        "logistic_recal": [
            pc._sigmoid(rec_a + rec_b * pc._logit(o["p_pred"]))
            for o in eval_],
        "cohort_pop": [pc._cohort_predict(cohorts, o) for o in eval_],
    }
    pop_fn, reason = _fit_lgbm_pop(Xtr, ys_tr)
    if pop_fn is not None:
        pop_preds["lgbm_pop"] = pop_fn(Xev)
    else:
        pop_models["lgbm_pop"] = {"status": "unavailable", "reason": reason}
    for name, ps in pop_preds.items():
        entry: Dict[str, Any] = {
            "status": "ok",
            "brier": pc._brier(ps, ys_ev),
            "log_loss": pc._log_loss(ps, ys_ev),
        }
        if len(set(ps)) > 1:
            ca, cb = pc.fit_logistic([pc._logit(p) for p in ps], ys_ev)
            entry["calibration_intercept"] = ca
            entry["calibration_slope"] = cb
        else:
            entry["calibration_intercept"] = None
            entry["calibration_slope"] = None
        pop_models[name] = entry

    # ---- profit models E[net P&L | fill] (D, E-q50, F, G + stamped ref)
    profit_models: Dict[str, Any] = {}
    pairs: Dict[str, Tuple[List[float], List[float]]] = {
        "stamped_ev": ([o["ev_pred"] for o in train],
                       [o["ev_pred"] for o in eval_]),
        "cohort_profit": ([_cohort_profit_predict(cprofit, o) for o in train],
                          [_cohort_profit_predict(cprofit, o) for o in eval_]),
    }
    pfn, reason = _fit_lgbm_regressor(Xtr, rz_tr, objective="regression")
    if pfn is not None:
        pairs["lgbm_profit"] = (pfn(Xtr), pfn(Xev))
    else:
        profit_models["lgbm_profit"] = {"status": "unavailable",
                                        "reason": reason}
    nfn, reason = _fit_neural_profit(Xtr, rz_tr)
    if nfn is not None:
        pairs["neural_profit"] = (nfn(Xtr), nfn(Xev))
    else:
        profit_models["neural_profit"] = {"status": "unavailable",
                                          "reason": reason}

    # ---- quantile model (E): q10/q50/q90; q50 also ranks as a profit model
    quantile_block: Dict[str, Any]
    qpreds: Dict[float, Tuple[List[float], List[float]]] = {}
    qreason: Optional[str] = None
    for a in QUANTILES:
        qfn, qreason = _fit_lgbm_regressor(Xtr, rz_tr,
                                           objective="quantile", alpha=a)
        if qfn is None:
            qpreds = {}
            break
        qpreds[a] = (qfn(Xtr), qfn(Xev))
    if qpreds:
        rz_ev = [o["realized"] for o in eval_]
        covered = sum(1 for lo, hi, y in zip(qpreds[0.10][1],
                                             qpreds[0.90][1], rz_ev)
                      if lo <= y <= hi)
        quantile_block = {
            "status": "ok",
            "pinball_usd": {f"q{int(a * 100)}": _pinball(qpreds[a][1],
                                                         rz_ev, a)
                            for a in QUANTILES},
            "interval_coverage_q10_q90": covered / len(rz_ev),
            "nominal_coverage": 0.80,
        }
        pairs["lgbm_quantile_q50"] = qpreds[0.50]
    else:
        quantile_block = {"status": "unavailable", "reason": qreason}
        profit_models["lgbm_quantile_q50"] = {"status": "unavailable",
                                              "reason": qreason}

    for name, (tp, ep) in pairs.items():
        profit_models[name] = _profit_entry(tp, ep, eval_)

    return {
        "n_train": len(train),
        "n_eval": len(eval_),
        "train_base_rate": base_rate,
        "eval_base_rate": sum(1 for y in ys_ev if y) / len(ys_ev),
        "pop_models": pop_models,
        "profit_models": profit_models,
        "quantile_model": quantile_block,
        "expectancy_per_opportunity_usd": pc._mean_ci(
            [o["realized"] for o in eval_]),
        "per_fill_note": SURVIVORSHIP_NOTE,
    }


# --------------------------------------------------------------------------- #
# executable EV composition (no double counting)
# --------------------------------------------------------------------------- #

def executable_ev_from_model(p_fill: float,
                             e_net_given_fill: float) -> Dict[str, Any]:
    """``executable_EV = P(fill) * E[net profit | fill]`` -- the net term is
    realized-net-calibrated, so no execution cost is subtracted again."""
    if (not isinstance(p_fill, (int, float)) or isinstance(p_fill, bool)
            or not math.isfinite(p_fill) or not 0.0 <= p_fill <= 1.0):
        raise ValueError("p_fill must be a finite probability in [0, 1]")
    if (not isinstance(e_net_given_fill, (int, float))
            or isinstance(e_net_given_fill, bool)
            or not math.isfinite(e_net_given_fill)):
        raise ValueError("e_net_given_fill must be a finite dollar amount")
    return {
        "p_fill": float(p_fill),
        "e_net_given_fill_usd": float(e_net_given_fill),
        "executable_ev_usd": float(p_fill) * float(e_net_given_fill),
        "cost_handling": COST_HANDLING_NOTE,
    }


def _fill_probability_block() -> Dict[str, Any]:
    """Conservative shrunk prior -- the only honest P(fill) available."""
    return {
        "value": shrunk_fill_probability(0, 0),
        "basis": "conservative_prior_shrunk",
        "prior": CONSERVATIVE_FILL_PRIOR,
        "prior_strength": PRIOR_STRENGTH,
        "n_nonfill_evidence": 0,
        "survivorship_note": SURVIVORSHIP_NOTE,
    }


def _study_verdict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-registered study verdict:
       * < MIN_EVAL_N eval rows -> INSUFFICIENT_EVIDENCE (abstain)
       * no profit model validated -> NO_VALIDATED_EDGE (abstain)
       * else VALIDATED_RANK_POWER naming the best validated model
         (highest eval Spearman; research-only, never auto-promoted)."""
    n_eval = metrics["n_eval"]
    if n_eval < MIN_EVAL_N:
        return {"verdict": "INSUFFICIENT_EVIDENCE",
                "challenger_policy": "abstain",
                "best_model": None,
                "reason": f"only {n_eval} eval opportunities "
                          f"(< {MIN_EVAL_N})"}
    validated = [
        (name, m) for name, m in metrics["profit_models"].items()
        if m.get("status") == "ok" and m["verdict"]["validated"]
        and m.get("spearman") is not None]
    if not validated:
        return {"verdict": "NO_VALIDATED_EDGE",
                "challenger_policy": "abstain",
                "best_model": None,
                "reason": ("no profit model shows out-of-sample rank power "
                           "(session-bootstrap 95% LB > 0 and positive OLS "
                           "slope required)")}
    best_name, best = max(validated, key=lambda kv: kv[1]["spearman"])
    return {"verdict": "VALIDATED_RANK_POWER",
            "challenger_policy": "research_only",
            "best_model": best_name,
            "reason": (f"{best_name}: spearman={best['spearman']:.4f}, "
                       f"95% LB={best['session_bootstrap']['lower_bound']:.4f}"
                       f" > 0, OLS slope="
                       f"{best['ols_slope_realized_on_pred']:.4f} > 0")}


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def run_ev_model_study(
    history_path: str,
    eval_frac: float = 0.25,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load history, split chronologically, evaluate the competing model
    suite, compose executable EV, publish the pre-registered verdict."""
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {"history": {"path": history_path,
                               "sha256": pc._sha256(history_path)}},
        "provenance": "paper_marks",
        "dependencies": {"lightgbm": _HAS_LGBM, "torch": _HAS_TORCH},
        "pre_registered": {
            "eval_frac": eval_frac,
            "seed": SEED,
            "quantiles": list(QUANTILES),
            "n_buckets": N_BUCKETS,
            "logistic_l2": pc.LOGISTIC_L2,
            "shrinkage_k": pc.SHRINKAGE_K,
            "n_boot": pc.N_BOOT,
            "boot_seed": pc.BOOT_SEED,
            "min_eval_n": MIN_EVAL_N,
            "fill_prior": CONSERVATIVE_FILL_PRIOR,
            "fill_prior_strength": PRIOR_STRENGTH,
        },
        "survivorship_note": SURVIVORSHIP_NOTE,
    }
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        trades = obj.get("trades", []) if isinstance(obj, dict) else obj
    except (OSError, ValueError) as exc:
        report["verdict"] = "INSUFFICIENT_EVIDENCE"
        report["challenger_policy"] = "abstain"
        report["error"] = f"history unavailable: {exc}"
        return report

    opps, extract_report = extract_opportunities(
        [t for t in trades if isinstance(t, dict)])
    report["extraction"] = extract_report
    report["coverage"] = (extract_report["kept"]
                          / extract_report["rows_scanned"]
                          if extract_report["rows_scanned"] else None)

    try:
        train, eval_, split_meta = pc.chronological_split(opps, eval_frac)
        report["split"] = split_meta
        metrics = evaluate_models(train, eval_)
        report["metrics"] = metrics
        verdict = _study_verdict(metrics)
        report.update(verdict)

        fill = _fill_probability_block()
        exe: Dict[str, Any] = {
            "formula": "executable_EV = P(fill) * E[net_profit | fill]",
            "cost_handling": COST_HANDLING_NOTE,
            "fill_probability": fill,
            "e_net_model": verdict["best_model"],
        }
        if verdict["best_model"] is not None:
            mean_e_net = metrics["profit_models"][
                verdict["best_model"]]["eval_mean_pred_usd"]
            composed = executable_ev_from_model(fill["value"], mean_e_net)
            exe["eval_mean_e_net_given_fill_usd"] = mean_e_net
            exe["eval_mean_executable_ev_usd"] = composed[
                "executable_ev_usd"]
        else:
            exe["eval_mean_e_net_given_fill_usd"] = None
            exe["eval_mean_executable_ev_usd"] = None
            exe["note"] = ("no validated E[net|fill] model: executable EV "
                           "is NOT composed from an unvalidated belief")
        report["executable_ev"] = exe
    except ValueError as exc:
        report["verdict"] = "INSUFFICIENT_EVIDENCE"
        report["challenger_policy"] = "abstain"
        report["best_model"] = None
        report["error"] = str(exc)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0.2 P0-4 competing EV model study "
                    "(research-only)")
    ap.add_argument("--history",
                    default="server_data/2026-11-19/trading_history.json")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--out",
                    default="artifacts/oracle402/ev_model_report.json")
    args = ap.parse_args(argv)
    rep = run_ev_model_study(args.history, args.eval_frac, out_path=args.out)
    summary: Dict[str, Any] = {k: rep.get(k) for k in
                               ("verdict", "challenger_policy", "best_model",
                                "reason", "split", "dependencies")}
    if "metrics" in rep:
        summary["profit_models"] = {
            name: {k: m.get(k) for k in ("status", "spearman", "mae_usd")}
            for name, m in rep["metrics"]["profit_models"].items()}
        summary["pop_models"] = {
            name: {k: m.get(k) for k in ("status", "brier")}
            for name, m in rep["metrics"]["pop_models"].items()}
    summary["executable_ev"] = rep.get("executable_ev")
    print(json.dumps(summary, indent=2, default=str))
    print(f"report written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
