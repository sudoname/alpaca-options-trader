"""LightGBM challengers for Oracle 4.0.1 (Stage E).

Research-only.  Nothing in the production trading path imports this.
LightGBM is an OPTIONAL, PINNED research dependency (4.7.0).  When it is
absent every entry point reports honestly-unavailable; nothing is faked.

Scope (per the Oracle 4.0.1 master prompt):
  * 3-class direction model (down / neutral / up — the canonical class
    order shared with ``oracle.ml.baselines.DIRECTION_CLASSES`` and
    ``oracle.ml.mlp.direction_class``), multiclass probabilities that
    sum to 1, early stopping on a chronologically PRIOR validation
    block, class weights applied on train only (validation and test are
    always scored on the unweighted true distribution).
  * Quantile regressors on log-return at the shared ladder with
    non-crossing rearrangement.
  * Two missingness treatments compared consistently: native NaN routing
    ("native") vs zero-impute + explicit indicator flags ("flags").
  * Deterministic budgeted config, locked here BEFORE any holdout is
    touched (single thread, fixed seed, deterministic=True).
  * Artifact save/reload via ``model_to_string`` + sha256 manifest with
    integrity verification on load.

A separate net-P&L model is only trained when the dataset actually
carries qualified monetary labels; otherwise it is reported unavailable
(the frozen ledger has no fee provenance, so this is expected).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.lab.purged_validation import Sample
from oracle.ml.baselines import QUANTILE_LADDER

__all__ = [
    "LGBM_AVAILABLE", "LGBM_WHY", "LGBM_VERSION", "LGBM_PINNED",
    "GBMConfig", "build_matrix",
    "fit_direction_gbm", "predict_direction_proba",
    "fit_quantile_gbms", "predict_quantiles",
    "GBMIntegrityError", "save_gbm_direction", "load_gbm_direction",
]

# --------------------------------------------------------------------------- #
# optional dependency (pinned)
# --------------------------------------------------------------------------- #

LGBM_PINNED = "4.7.0"

try:  # pragma: no cover - import guard exercised via flags
    import lightgbm as _lgb
    LGBM_AVAILABLE = True
    LGBM_VERSION: Optional[str] = getattr(_lgb, "__version__", None)
    LGBM_WHY = ""
    if LGBM_VERSION != LGBM_PINNED:
        LGBM_WHY = (f"lightgbm {LGBM_VERSION} != pinned {LGBM_PINNED}; "
                    "results remain comparable only under the pin")
except Exception as exc:  # ImportError or binary load failure
    _lgb = None  # type: ignore[assignment]
    LGBM_AVAILABLE = False
    LGBM_VERSION = None
    LGBM_WHY = f"lightgbm not importable: {exc}"


def _require_lgbm() -> None:
    if not LGBM_AVAILABLE:
        raise RuntimeError(f"lightgbm unavailable: {LGBM_WHY}")


# --------------------------------------------------------------------------- #
# pre-registered budgeted config (locked before any holdout evaluation)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class GBMConfig:
    """Deterministic small-data budget.  Frozen: do not tune on holdout."""
    learning_rate: float = 0.05
    num_leaves: int = 15
    max_depth: int = 4
    min_data_in_leaf: int = 20
    feature_fraction: float = 0.9
    lambda_l2: float = 1.0
    max_estimators: int = 400
    early_stopping_rounds: int = 30
    seed: int = 1337

    def base_params(self) -> Dict[str, Any]:
        return {
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "max_depth": self.max_depth,
            "min_data_in_leaf": self.min_data_in_leaf,
            "feature_fraction": self.feature_fraction,
            "bagging_fraction": 1.0,          # no row bagging: deterministic
            "bagging_freq": 0,
            "lambda_l2": self.lambda_l2,
            "deterministic": True,
            "force_row_wise": True,
            "num_threads": 1,
            "seed": self.seed,
            "feature_fraction_seed": self.seed,
            "verbosity": -1,
        }


# --------------------------------------------------------------------------- #
# design matrices: native-NaN vs explicit flags
# --------------------------------------------------------------------------- #

def build_matrix(samples: Sequence[Sample], feature_names: Sequence[str],
                 mode: str = "native") -> Tuple[np.ndarray, List[str]]:
    """Raw (unstandardized — trees are scale-invariant) feature matrix.

    mode="native": missing -> NaN, letting LightGBM route missing values
    natively at each split.
    mode="flags":  missing -> 0.0 plus a ``*_missing`` indicator column
    per feature (same encoding philosophy as ``baselines.build_design``).
    Both modes see the SAME rows so comparisons are matched.
    """
    if mode not in ("native", "flags"):
        raise ValueError(f"unknown missingness mode: {mode}")
    names = list(feature_names)
    rows: List[List[float]] = []
    for s in samples:
        vals: List[float] = []
        flags: List[float] = []
        for name in names:
            v = s.features.get(name)
            missing = (v is None or not isinstance(v, (int, float)) or v != v)
            if missing:
                vals.append(float("nan") if mode == "native" else 0.0)
                flags.append(1.0)
            else:
                vals.append(float(v))
                flags.append(0.0)
        rows.append(vals + flags if mode == "flags" else vals)
    cols = names + ([f"{n}_missing" for n in names] if mode == "flags"
                    else [])
    x = (np.asarray(rows, dtype=np.float64) if rows
         else np.zeros((0, len(cols))))
    return x, cols


# --------------------------------------------------------------------------- #
# 3-class direction model
# --------------------------------------------------------------------------- #

def _class_weights(y: Sequence[int], n_classes: int = 3) -> np.ndarray:
    """Inverse-frequency weights, normalized to mean 1.  TRAIN ONLY."""
    yv = np.asarray(y, dtype=np.int64)
    counts = np.bincount(yv, minlength=n_classes).astype(np.float64)
    inv = np.where(counts > 0, 1.0 / np.maximum(counts, 1.0), 0.0)
    w = inv[yv]
    mean = w.mean() if len(w) else 1.0
    return w / mean if mean > 0 else np.ones(len(yv))


def fit_direction_gbm(x_train: np.ndarray, y_train: Sequence[int],
                      x_val: np.ndarray, y_val: Sequence[int],
                      cfg: GBMConfig = GBMConfig(),
                      use_class_weights: bool = True,
                      n_classes: int = 3) -> Tuple[Any, Dict[str, Any]]:
    """Fit the multiclass model with early stopping on a PRIOR validation.

    Class weights (if enabled) are computed from and applied to TRAIN
    rows only; the validation metric is evaluated unweighted so early
    stopping reflects the true class distribution.
    Returns (booster, info).
    """
    _require_lgbm()
    yt = np.asarray(y_train, dtype=np.int64)
    yv = np.asarray(y_val, dtype=np.int64)
    params = dict(cfg.base_params())
    params.update({"objective": "multiclass", "num_class": n_classes,
                   "metric": "multi_logloss"})
    weights = _class_weights(yt, n_classes) if use_class_weights else None
    dtrain = _lgb.Dataset(x_train, label=yt, weight=weights,
                          free_raw_data=False)
    have_val = len(yv) >= 2 and len(set(yv.tolist())) >= 2
    callbacks = [_lgb.log_evaluation(period=0)]
    valid_sets = [dtrain]
    valid_names = ["train"]
    if have_val:
        dval = _lgb.Dataset(x_val, label=yv, reference=dtrain,
                            free_raw_data=False)      # unweighted
        valid_sets.append(dval)
        valid_names.append("val")
        callbacks.append(_lgb.early_stopping(cfg.early_stopping_rounds,
                                             verbose=False))
    booster = _lgb.train(params, dtrain,
                         num_boost_round=cfg.max_estimators,
                         valid_sets=valid_sets, valid_names=valid_names,
                         callbacks=callbacks)
    info = {
        "best_iteration": int(booster.best_iteration or
                              booster.current_iteration()),
        "early_stopped": bool(have_val and booster.best_iteration and
                              booster.best_iteration < cfg.max_estimators),
        "val_used": have_val,
        "val_why_skipped": (None if have_val else
                            "validation block too small or single-class; "
                            "trained to a fixed budget instead"),
        "class_weights_used": use_class_weights,
        "n_train": int(len(yt)), "n_val": int(len(yv)),
    }
    return booster, info


def predict_direction_proba(booster: Any, x: np.ndarray) -> np.ndarray:
    """(n, K) class probabilities at the best iteration; rows sum to 1."""
    _require_lgbm()
    p = booster.predict(x, num_iteration=booster.best_iteration or None)
    p = np.asarray(p, dtype=np.float64)
    if p.ndim == 1:                                   # defensive: n_classes=1
        p = p.reshape(len(p), -1)
    rs = p.sum(axis=1, keepdims=True)
    return p / np.where(rs > 0, rs, 1.0)


# --------------------------------------------------------------------------- #
# quantile regressors (log-return ladder)
# --------------------------------------------------------------------------- #

def fit_quantile_gbms(x_train: np.ndarray, y_train: Sequence[float],
                      x_val: np.ndarray, y_val: Sequence[float],
                      cfg: GBMConfig = GBMConfig(),
                      quantiles: Sequence[float] = QUANTILE_LADDER,
                      ) -> Tuple[Dict[float, Any], Dict[str, Any]]:
    """One quantile-objective booster per tau, early stopped on prior val."""
    _require_lgbm()
    yt = np.asarray(y_train, dtype=np.float64)
    yv = np.asarray(y_val, dtype=np.float64)
    have_val = len(yv) >= 2
    boosters: Dict[float, Any] = {}
    best_iters: Dict[str, int] = {}
    dtrain = _lgb.Dataset(x_train, label=yt, free_raw_data=False)
    for tau in quantiles:
        params = dict(cfg.base_params())
        params.update({"objective": "quantile", "alpha": float(tau),
                       "metric": "quantile"})
        callbacks = [_lgb.log_evaluation(period=0)]
        valid_sets = [dtrain]
        valid_names = ["train"]
        if have_val:
            dval = _lgb.Dataset(x_val, label=yv, reference=dtrain,
                                free_raw_data=False)
            valid_sets.append(dval)
            valid_names.append("val")
            callbacks.append(_lgb.early_stopping(cfg.early_stopping_rounds,
                                                 verbose=False))
        b = _lgb.train(params, dtrain, num_boost_round=cfg.max_estimators,
                       valid_sets=valid_sets, valid_names=valid_names,
                       callbacks=callbacks)
        boosters[float(tau)] = b
        best_iters[str(tau)] = int(b.best_iteration or b.current_iteration())
    info = {"val_used": have_val, "best_iterations": best_iters,
            "n_train": int(len(yt)), "n_val": int(len(yv))}
    return boosters, info


def predict_quantiles(boosters: Dict[float, Any], x: np.ndarray
                      ) -> Dict[float, np.ndarray]:
    """Predict the ladder, then rearrange (sort) per row: no crossing."""
    _require_lgbm()
    taus = sorted(boosters)
    raw = np.stack([
        np.asarray(boosters[t].predict(
            x, num_iteration=boosters[t].best_iteration or None),
            dtype=np.float64)
        for t in taus
    ])
    raw.sort(axis=0)
    return {t: raw[i] for i, t in enumerate(taus)}


# --------------------------------------------------------------------------- #
# artifact save / reload with integrity check
# --------------------------------------------------------------------------- #

class GBMIntegrityError(RuntimeError):
    """Raised when a GBM artifact fails its checksum or manifest on load."""


def save_gbm_direction(path: str, booster: Any,
                       feature_columns: Sequence[str],
                       missingness_mode: str,
                       cfg: GBMConfig,
                       meta: Optional[Dict[str, Any]] = None) -> str:
    """Persist booster text + manifest; sha256 covers the model text."""
    _require_lgbm()
    model_str = booster.model_to_string(
        num_iteration=booster.best_iteration or None)
    digest = hashlib.sha256(model_str.encode("utf-8")).hexdigest()
    doc = {
        "kind": "lightgbm_multiclass_direction",
        "lightgbm_version": LGBM_VERSION,
        "lightgbm_pinned": LGBM_PINNED,
        "model_text": model_str,
        "checksum_sha256": digest,
        "feature_columns": list(feature_columns),
        "missingness_mode": missingness_mode,
        "config": {k: getattr(cfg, k) for k in (
            "learning_rate", "num_leaves", "max_depth", "min_data_in_leaf",
            "feature_fraction", "lambda_l2", "max_estimators",
            "early_stopping_rounds", "seed")},
        "meta": meta or {},
    }
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(doc, fh, sort_keys=True, indent=1)
    return digest


def load_gbm_direction(path: str
                       ) -> Tuple[Any, List[str], str, Dict[str, Any]]:
    """Reload + verify.  Returns (booster, feature_columns, mode, meta)."""
    _require_lgbm()
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    model_str = doc.get("model_text")
    stored = doc.get("checksum_sha256")
    if not isinstance(model_str, str) or not stored:
        raise GBMIntegrityError(f"malformed gbm artifact: {path}")
    digest = hashlib.sha256(model_str.encode("utf-8")).hexdigest()
    if digest != stored:
        raise GBMIntegrityError(f"gbm artifact checksum mismatch: {path}")
    if doc.get("kind") != "lightgbm_multiclass_direction":
        raise GBMIntegrityError(f"unexpected artifact kind: {doc.get('kind')}")
    booster = _lgb.Booster(model_str=model_str)
    return (booster, list(doc.get("feature_columns", [])),
            str(doc.get("missingness_mode", "native")),
            dict(doc.get("meta", {})))
