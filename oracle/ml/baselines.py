"""Classical challenger models for Oracle 4.0 (Phase 3).

Research-only.  Nothing in the production trading path imports this.

Models (all deterministic, numpy-only — no sklearn/lightgbm deps):
  * RidgeLogisticModel     — direction classifier with calibrated
                             probabilities (optional Platt scaling fit on
                             a chronological calibration tail of train).
  * Naive comparators      — constant base rate, momentum rule, and the
                             production baseline's own ``baseline_p_win``.
  * QuantilePinballModel   — linear quantile regression at a ladder of
                             quantiles with non-crossing rearrangement.
  * RV forecasters         — persistence, per-underlying EWMA, HAR-lite
                             linear regression.
  * DiscreteTimeCompetingRisks — person-period hazards for
                             stop / target / other_exit with right
                             censoring and cumulative incidence.

Missingness policy: missing *labels* are never trained or scored on;
missing *features* are represented explicitly — standardized values with
None mean-imputed to 0 **plus** a missingness-indicator column per
feature that is ever missing.  Nothing is silently invented.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from oracle.lab.purged_validation import Sample, Standardizer

__all__ = [
    "DesignMatrix", "build_design",
    "RidgeLogisticModel", "fit_platt", "apply_platt",
    "reliability_bins", "expected_calibration_error",
    "constant_base_rate_probs", "momentum_rule_probs",
    "baseline_p_win_probs",
    "QuantilePinballModel", "pinball_loss", "interval_coverage",
    "rv_persistence_forecast", "rv_ewma_forecast", "HarLiteModel",
    "qlike_loss",
    "HAZARD_BIN_EDGES_MIN", "HAZARD_CAUSES",
    "expand_person_periods", "DiscreteTimeCompetingRisks",
    "empirical_cumulative_incidence",
    # 3-class direction (Oracle 4.0.1 Stage E)
    "DIRECTION_CLASSES", "MultinomialLogisticModel", "fit_temperature_np",
    "apply_temperature_np", "multiclass_log_loss", "multiclass_brier",
    "confusion_matrix", "macro_f1", "per_class_reliability",
    "constant_class_probs", "momentum_class_probs",
    "ClassicalIntegrityError", "save_classical", "load_classical",
]


# --------------------------------------------------------------------------- #
# design matrix with explicit missingness
# --------------------------------------------------------------------------- #

@dataclass
class DesignMatrix:
    x: np.ndarray                     # (n, d) float64, no NaNs
    columns: List[str]                # feature names incl. *_missing flags
    rows_used: List[int]              # indices into the input sample list


def build_design(samples: Sequence[Sample], std: Standardizer,
                 indicator_features: Optional[Sequence[str]] = None
                 ) -> DesignMatrix:
    """Standardize (train-fit ``std``) and encode missingness explicitly.

    None -> 0.0 (the train mean) AND a 1.0 in the matching ``*_missing``
    indicator column.  Indicator columns are emitted for every feature in
    ``indicator_features`` (fixed at train time so train/test agree).
    """
    names = std.feature_names
    ind = list(indicator_features) if indicator_features is not None else names
    ind_set = set(ind)
    rows: List[List[float]] = []
    for s in samples:
        z = std.transform(s.features)
        vals, flags = [], []
        for name, v in zip(names, z):
            if v is None:
                vals.append(0.0)
                if name in ind_set:
                    flags.append(1.0)
            else:
                vals.append(float(v))
                if name in ind_set:
                    flags.append(0.0)
        # keep indicator order aligned with `ind`
        flag_map = {}
        fi = 0
        for name in names:
            if name in ind_set:
                flag_map[name] = flags[fi]
                fi += 1
        rows.append(vals + [flag_map[n] for n in ind])
    cols = list(names) + [f"{n}_missing" for n in ind]
    x = (np.asarray(rows, dtype=np.float64) if rows
         else np.zeros((0, len(cols))))
    return DesignMatrix(x=x, columns=cols, rows_used=list(range(len(samples))))


# --------------------------------------------------------------------------- #
# ridge logistic challenger
# --------------------------------------------------------------------------- #

class RidgeLogisticModel:
    """Full-batch gradient-descent ridge logistic.  Deterministic."""

    def __init__(self, l2: float = 1e-3, lr: float = 0.5,
                 iters: int = 500) -> None:
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.w: Optional[np.ndarray] = None   # (d+1,) last entry = bias

    def fit(self, x: np.ndarray, y: Sequence[int]) -> "RidgeLogisticModel":
        n, d = x.shape
        xb = np.hstack([x, np.ones((n, 1))])
        yv = np.asarray(y, dtype=np.float64)
        w = np.zeros(d + 1)
        for _ in range(self.iters):
            z = np.clip(xb @ w, -35.0, 35.0)
            p = 1.0 / (1.0 + np.exp(-z))
            grad = xb.T @ (p - yv) / n + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None, "fit first"
        xb = np.hstack([x, np.ones((x.shape[0], 1))])
        z = np.clip(xb @ self.w, -35.0, 35.0)
        return 1.0 / (1.0 + np.exp(-z))

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None, "fit first"
        xb = np.hstack([x, np.ones((x.shape[0], 1))])
        return np.clip(xb @ self.w, -35.0, 35.0)


def fit_platt(scores: Sequence[float], y: Sequence[int],
              iters: int = 500, lr: float = 0.1) -> Tuple[float, float]:
    """1-d logistic recalibration p = sigma(a*score + b) on held-out data."""
    s = np.asarray(scores, dtype=np.float64)
    yv = np.asarray(y, dtype=np.float64)
    a, b = 1.0, 0.0
    n = max(len(s), 1)
    for _ in range(iters):
        z = np.clip(a * s + b, -35.0, 35.0)
        p = 1.0 / (1.0 + np.exp(-z))
        err = p - yv
        ga = float(err @ s) / n
        gb = float(err.sum()) / n
        a -= lr * ga
        b -= lr * gb
    return a, b


def apply_platt(a: float, b: float, scores: Sequence[float]) -> np.ndarray:
    z = np.clip(a * np.asarray(scores, dtype=np.float64) + b, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-z))


def reliability_bins(y_true: Sequence[int], p: Sequence[float],
                     n_bins: int = 10) -> List[Dict[str, Optional[float]]]:
    """Equal-width reliability bins: mean predicted vs observed frequency."""
    out = []
    yv = np.asarray(y_true, dtype=np.float64)
    pv = np.asarray(p, dtype=np.float64)
    for i in range(n_bins):
        lo, hi = i / n_bins, (i + 1) / n_bins
        mask = (pv >= lo) & (pv < hi) if i < n_bins - 1 else \
               (pv >= lo) & (pv <= hi)
        n = int(mask.sum())
        out.append({
            "bin_lo": lo, "bin_hi": hi, "n": n,
            "mean_p": float(pv[mask].mean()) if n else None,
            "frac_pos": float(yv[mask].mean()) if n else None,
        })
    return out


def expected_calibration_error(bins: Sequence[Dict[str, Any]]
                               ) -> Optional[float]:
    total = sum(b["n"] for b in bins)
    if total == 0:
        return None
    ece = 0.0
    for b in bins:
        if b["n"] and b["mean_p"] is not None and b["frac_pos"] is not None:
            ece += (b["n"] / total) * abs(b["mean_p"] - b["frac_pos"])
    return ece


# --------------------------------------------------------------------------- #
# naive comparators
# --------------------------------------------------------------------------- #

def constant_base_rate_probs(train_y: Sequence[int], n_test: int
                             ) -> np.ndarray:
    """Historical-frequency comparator: train base rate for every row."""
    base = (sum(train_y) / len(train_y)) if train_y else 0.5
    return np.full(n_test, base, dtype=np.float64)


def momentum_rule_probs(train_samples: Sequence[Sample],
                        train_y: Sequence[int],
                        test_samples: Sequence[Sample],
                        feature: str = "ret_60m",
                        ) -> Tuple[np.ndarray, np.ndarray]:
    """Sign-of-trailing-return comparator.

    P(up | sign bucket) is the train frequency within that bucket.  Rows
    where the feature is unavailable get prediction-unavailable (mask=0),
    never an invented value.
    Returns (probs, available_mask).
    """
    pos_n = pos_up = neg_n = neg_up = 0
    for s, y in zip(train_samples, train_y):
        v = s.features.get(feature)
        if v is None or not isinstance(v, (int, float)) or v != v:
            continue
        if v >= 0:
            pos_n += 1
            pos_up += y
        else:
            neg_n += 1
            neg_up += y
    base = (sum(train_y) / len(train_y)) if train_y else 0.5
    p_pos = pos_up / pos_n if pos_n else base
    p_neg = neg_up / neg_n if neg_n else base
    probs = np.full(len(test_samples), 0.5)
    mask = np.zeros(len(test_samples))
    for i, s in enumerate(test_samples):
        v = s.features.get(feature)
        if v is None or not isinstance(v, (int, float)) or v != v:
            continue
        probs[i] = p_pos if v >= 0 else p_neg
        mask[i] = 1.0
    return probs, mask


def baseline_p_win_probs(test_samples: Sequence[Sample]
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """The production baseline's own win probability, where it exists.

    No fitting: this is the incumbent's opinion recorded point-in-time.
    Returns (probs, available_mask); unavailable rows are masked out.
    """
    probs = np.full(len(test_samples), 0.5)
    mask = np.zeros(len(test_samples))
    for i, s in enumerate(test_samples):
        v = s.features.get("baseline_p_win")
        if v is None or not isinstance(v, (int, float)) or v != v:
            continue
        probs[i] = min(max(float(v), 0.0), 1.0)
        mask[i] = 1.0
    return probs, mask


# --------------------------------------------------------------------------- #
# quantile regression (pinball) with non-crossing rearrangement
# --------------------------------------------------------------------------- #

QUANTILE_LADDER = (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95)


class QuantilePinballModel:
    """Linear quantile regression per tau via subgradient descent.

    Predictions across the ladder are rearranged (sorted) per row so the
    reported quantiles never cross.
    """

    def __init__(self, quantiles: Sequence[float] = QUANTILE_LADDER,
                 l2: float = 1e-4, lr: float = 0.05,
                 iters: int = 400) -> None:
        self.quantiles = tuple(quantiles)
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.weights: Dict[float, np.ndarray] = {}

    def fit(self, x: np.ndarray, y: Sequence[float]
            ) -> "QuantilePinballModel":
        n, d = x.shape
        xb = np.hstack([x, np.ones((n, 1))])
        yv = np.asarray(y, dtype=np.float64)
        scale = float(np.std(yv)) or 1.0
        for tau in self.quantiles:
            w = np.zeros(d + 1)
            w[-1] = float(np.quantile(yv, tau))      # warm start at marginal
            for _ in range(self.iters):
                pred = xb @ w
                # subgradient of pinball: tau - 1{y < pred}
                g = np.where(yv < pred, 1.0 - tau, -tau)
                grad = xb.T @ g / n + self.l2 * w
                w -= self.lr * scale * grad
            self.weights[tau] = w
        return self

    def predict(self, x: np.ndarray) -> Dict[float, np.ndarray]:
        xb = np.hstack([x, np.ones((x.shape[0], 1))])
        raw = np.stack([xb @ self.weights[t] for t in self.quantiles])
        raw.sort(axis=0)                             # rearrangement: no crossing
        return {t: raw[i] for i, t in enumerate(self.quantiles)}


def pinball_loss(y: Sequence[float], pred: Sequence[float], tau: float
                 ) -> Optional[float]:
    if not len(y):
        return None
    yv = np.asarray(y, dtype=np.float64)
    pv = np.asarray(pred, dtype=np.float64)
    diff = yv - pv
    return float(np.mean(np.where(diff >= 0, tau * diff, (tau - 1) * diff)))


def interval_coverage(y: Sequence[float], lo: Sequence[float],
                      hi: Sequence[float]) -> Optional[float]:
    if not len(y):
        return None
    yv = np.asarray(y, dtype=np.float64)
    return float(np.mean((yv >= np.asarray(lo)) & (yv <= np.asarray(hi))))


# --------------------------------------------------------------------------- #
# realized-volatility forecasters
# --------------------------------------------------------------------------- #

def rv_persistence_forecast(samples: Sequence[Sample]
                            ) -> Tuple[np.ndarray, np.ndarray]:
    """forecast = current session RV (random-walk persistence).

    Returns (forecast, available_mask); rows without rv_session masked.
    """
    f = np.zeros(len(samples))
    mask = np.zeros(len(samples))
    for i, s in enumerate(samples):
        v = s.features.get("rv_session")
        if v is None or not isinstance(v, (int, float)) or v != v or v < 0:
            continue
        f[i] = float(v)
        mask[i] = 1.0
    return f, mask


def rv_ewma_forecast(samples: Sequence[Sample],
                     underlying_by_id: Dict[str, str],
                     lam: float = 0.94) -> Tuple[np.ndarray, np.ndarray]:
    """Per-underlying EWMA over chronologically prior rv_session values.

    Point-in-time safe: samples are processed in as_of order and the EWMA
    state for an underlying only ever contains values observed at or
    before the row being forecast.
    """
    order = sorted(range(len(samples)), key=lambda i: (samples[i].as_of,
                                                       samples[i].decision_id))
    state: Dict[str, float] = {}
    f = np.zeros(len(samples))
    mask = np.zeros(len(samples))
    for i in order:
        s = samples[i]
        und = underlying_by_id.get(s.decision_id, "?")
        v = s.features.get("rv_session")
        have = (v is not None and isinstance(v, (int, float))
                and v == v and v >= 0)
        prev = state.get(und)
        if have:
            cur = lam * prev + (1 - lam) * float(v) if prev is not None \
                else float(v)
            state[und] = cur
            f[i] = cur
            mask[i] = 1.0
        elif prev is not None:
            f[i] = prev
            mask[i] = 1.0
    return f, mask


class HarLiteModel:
    """HAR-style OLS on available volatility proxies.

    With only ~8 sessions of coverage the classical 1d/5d/22d HAR
    components are not estimable; this uses the point-in-time proxies
    [rv_session, ewma_rv, |ret_15m|, |ret_60m|, |ret_session|].
    Rows missing rv_session are unavailable (masked), not imputed.
    """

    INPUTS = ("rv_session", "ret_15m", "ret_60m", "ret_session")

    def __init__(self) -> None:
        self.w: Optional[np.ndarray] = None

    @staticmethod
    def _row(s: Sample, ewma: Optional[float]) -> Optional[List[float]]:
        rv = s.features.get("rv_session")
        if rv is None or not isinstance(rv, (int, float)) or rv != rv:
            return None
        vals = [float(rv), float(ewma) if ewma is not None else float(rv)]
        for k in ("ret_15m", "ret_60m", "ret_session"):
            v = s.features.get(k)
            vals.append(abs(float(v)) if isinstance(v, (int, float))
                        and v == v else 0.0)
        return vals + [1.0]

    def fit(self, samples: Sequence[Sample], y: Sequence[float],
            ewma: Sequence[Optional[float]]) -> "HarLiteModel":
        xs, ys = [], []
        for s, yi, e in zip(samples, y, ewma):
            r = self._row(s, e)
            if r is not None and yi is not None:
                xs.append(r)
                ys.append(float(yi))
        if not xs:
            self.w = None
            return self
        xm = np.asarray(xs)
        ym = np.asarray(ys)
        # ridge-stabilized least squares
        d = xm.shape[1]
        reg = 1e-8 * np.eye(d)
        self.w, *_ = np.linalg.lstsq(xm.T @ xm + reg, xm.T @ ym, rcond=None)
        return self

    def predict(self, samples: Sequence[Sample],
                ewma: Sequence[Optional[float]]
                ) -> Tuple[np.ndarray, np.ndarray]:
        f = np.zeros(len(samples))
        mask = np.zeros(len(samples))
        if self.w is None:
            return f, mask
        for i, (s, e) in enumerate(zip(samples, ewma)):
            r = self._row(s, e)
            if r is None:
                continue
            f[i] = max(float(np.dot(self.w, r)), 0.0)
            mask[i] = 1.0
        return f, mask


def qlike_loss(y: Sequence[float], forecast: Sequence[float]
               ) -> Optional[float]:
    """QLIKE on the variance scale; rows with non-positive y or forecast
    are unavailable for this loss and skipped (counted by caller)."""
    vals = []
    for yi, fi in zip(y, forecast):
        if yi is None or fi is None or yi <= 0 or fi <= 0:
            continue
        r = (yi * yi) / (fi * fi)
        vals.append(r - math.log(r) - 1.0)
    return (sum(vals) / len(vals)) if vals else None


# --------------------------------------------------------------------------- #
# discrete-time competing-risks hazards
# --------------------------------------------------------------------------- #

HAZARD_BIN_EDGES_MIN = (0.0, 30.0, 60.0, 120.0, 240.0, 480.0, 1440.0,
                        float("inf"))
HAZARD_CAUSES = ("stop", "target", "other_exit")


def _bin_of(minutes: float) -> int:
    for k in range(len(HAZARD_BIN_EDGES_MIN) - 1):
        if HAZARD_BIN_EDGES_MIN[k] < minutes <= HAZARD_BIN_EDGES_MIN[k + 1]:
            return k
    return 0


@dataclass
class PersonPeriod:
    row_index: int                    # which subject
    time_bin: int
    outcome: int                      # 0 = survive, 1..K = cause index + 1


def expand_person_periods(times: Sequence[Optional[float]],
                          causes: Sequence[Optional[str]],
                          ) -> List[PersonPeriod]:
    """Expand (time-to-event, cause) into discrete person-periods.

    cause None = right-censored at `time` (contributes survival periods
    through its censoring bin).  Events contribute survival periods for
    every earlier bin plus one event period in the bin containing the
    event time.
    """
    out: List[PersonPeriod] = []
    for i, (t, c) in enumerate(zip(times, causes)):
        if t is None or t <= 0:
            continue
        kb = _bin_of(float(t))
        for k in range(kb):
            out.append(PersonPeriod(i, k, 0))
        if c is None:                                 # censored in bin kb
            out.append(PersonPeriod(i, kb, 0))
        else:
            cause_idx = HAZARD_CAUSES.index(c) + 1
            out.append(PersonPeriod(i, kb, cause_idx))
    return out


class DiscreteTimeCompetingRisks:
    """Multinomial logistic over person-periods.

    Classes: survive + one per cause.  Inputs: subject features plus
    one-hot time-bin dummies.  Deterministic full-batch GD.
    """

    def __init__(self, l2: float = 1e-3, lr: float = 0.5,
                 iters: int = 400) -> None:
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.w: Optional[np.ndarray] = None          # (d_total, K+1)
        self.n_bins = len(HAZARD_BIN_EDGES_MIN) - 1
        self.n_classes = len(HAZARD_CAUSES) + 1

    def _design(self, x: np.ndarray, periods: Sequence[PersonPeriod]
                ) -> np.ndarray:
        rows = np.zeros((len(periods), x.shape[1] + self.n_bins + 1))
        for r, pp in enumerate(periods):
            rows[r, :x.shape[1]] = x[pp.row_index]
            rows[r, x.shape[1] + pp.time_bin] = 1.0
            rows[r, -1] = 1.0                        # bias
        return rows

    def fit(self, x: np.ndarray, periods: Sequence[PersonPeriod]
            ) -> "DiscreteTimeCompetingRisks":
        xb = self._design(x, periods)
        y = np.array([pp.outcome for pp in periods])
        n, d = xb.shape
        w = np.zeros((d, self.n_classes))
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), y] = 1.0
        for _ in range(self.iters):
            z = np.clip(xb @ w, -35.0, 35.0)
            z -= z.max(axis=1, keepdims=True)
            e = np.exp(z)
            p = e / e.sum(axis=1, keepdims=True)
            grad = xb.T @ (p - onehot) / n + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def hazards(self, x_row: np.ndarray) -> np.ndarray:
        """Per-bin class probabilities, shape (n_bins, n_classes)."""
        assert self.w is not None, "fit first"
        out = np.zeros((self.n_bins, self.n_classes))
        for k in range(self.n_bins):
            row = np.zeros(self.w.shape[0])
            row[:len(x_row)] = x_row
            row[len(x_row) + k] = 1.0
            row[-1] = 1.0
            z = np.clip(row @ self.w, -35.0, 35.0)
            z -= z.max()
            e = np.exp(z)
            out[k] = e / e.sum()
        return out

    def cumulative_incidence(self, x_row: np.ndarray) -> np.ndarray:
        """CIF per cause at the end of each bin, shape (n_bins, K)."""
        hz = self.hazards(x_row)
        surv = 1.0
        cif = np.zeros((self.n_bins, len(HAZARD_CAUSES)))
        acc = np.zeros(len(HAZARD_CAUSES))
        for k in range(self.n_bins):
            acc = acc + surv * hz[k, 1:]
            cif[k] = acc
            surv *= hz[k, 0]
        return cif

    def nll(self, x: np.ndarray, periods: Sequence[PersonPeriod]
            ) -> Optional[float]:
        if not periods:
            return None
        xb = self._design(x, periods)
        y = np.array([pp.outcome for pp in periods])
        z = np.clip(xb @ self.w, -35.0, 35.0)
        z -= z.max(axis=1, keepdims=True)
        e = np.exp(z)
        p = e / e.sum(axis=1, keepdims=True)
        eps = 1e-12
        return float(-np.mean(np.log(p[np.arange(len(y)), y] + eps)))


def empirical_cumulative_incidence(times: Sequence[Optional[float]],
                                   causes: Sequence[Optional[str]],
                                   at_minutes: float) -> Dict[str, float]:
    """Aalen–Johansen cumulative incidence at a fixed time (discrete)."""
    events = [(float(t), c) for t, c in zip(times, causes)
              if t is not None and t > 0]
    if not events:
        return {c: 0.0 for c in HAZARD_CAUSES}
    events.sort()
    n_risk = len(events)
    surv = 1.0
    cif = {c: 0.0 for c in HAZARD_CAUSES}
    for t, c in events:
        if t > at_minutes:
            break
        if c is not None:
            cif[c] += surv * (1.0 / n_risk)
            surv *= 1.0 - 1.0 / n_risk
        else:
            surv *= 1.0                              # censoring: leave risk set
        n_risk -= 1
        if n_risk <= 0:
            break
    return cif


# --------------------------------------------------------------------------- #
# 3-class direction: multinomial logistic + metrics (Oracle 4.0.1 Stage E)
# --------------------------------------------------------------------------- #

# Canonical class order everywhere: 0 = down, 1 = neutral, 2 = up
# (matches oracle.ml.mlp.direction_class and labels.py's {-1, 0, +1}).
DIRECTION_CLASSES = ("down", "neutral", "up")


def _softmax_rows(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35.0, 35.0)
    z = z - z.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


class MultinomialLogisticModel:
    """Deterministic full-batch GD ridge softmax over K classes.

    Same optimization recipe as ``DiscreteTimeCompetingRisks.fit`` but on
    plain (x, class) rows.  Optional per-row ``sample_weight`` applies at
    FIT time only (class rebalancing on train); evaluation is always on
    the unweighted true distribution.
    """

    def __init__(self, n_classes: int = 3, l2: float = 1e-3,
                 lr: float = 0.5, iters: int = 500) -> None:
        self.n_classes = n_classes
        self.l2 = l2
        self.lr = lr
        self.iters = iters
        self.w: Optional[np.ndarray] = None          # (d+1, K); last row bias

    def fit(self, x: np.ndarray, y: Sequence[int],
            sample_weight: Optional[Sequence[float]] = None
            ) -> "MultinomialLogisticModel":
        n, d = x.shape
        xb = np.hstack([x, np.ones((n, 1))])
        yv = np.asarray(y, dtype=np.int64)
        if yv.min() < 0 or yv.max() >= self.n_classes:
            raise ValueError("class label outside [0, n_classes)")
        if sample_weight is None:
            sw = np.ones(n, dtype=np.float64)
        else:
            sw = np.asarray(sample_weight, dtype=np.float64)
            if sw.shape != (n,) or (sw < 0).any():
                raise ValueError("sample_weight must be n non-negative floats")
        sw_sum = float(sw.sum()) or 1.0
        onehot = np.zeros((n, self.n_classes))
        onehot[np.arange(n), yv] = 1.0
        w = np.zeros((d + 1, self.n_classes))
        for _ in range(self.iters):
            p = _softmax_rows(xb @ w)
            grad = xb.T @ ((p - onehot) * sw[:, None]) / sw_sum + self.l2 * w
            w -= self.lr * grad
        self.w = w
        return self

    def decision_scores(self, x: np.ndarray) -> np.ndarray:
        assert self.w is not None, "fit first"
        xb = np.hstack([x, np.ones((x.shape[0], 1))])
        return np.clip(xb @ self.w, -35.0, 35.0)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        """(n, K) rows that sum to 1."""
        return _softmax_rows(self.decision_scores(x))


def apply_temperature_np(logits: np.ndarray, temperature: float
                         ) -> np.ndarray:
    """Softmax(logits / T) — T from a held-out calibration tail."""
    t = max(float(temperature), 1e-6)
    return _softmax_rows(np.asarray(logits, dtype=np.float64) / t)


def fit_temperature_np(logits: np.ndarray, y: Sequence[int]) -> float:
    """Deterministic temperature scaling: NLL grid search, no SGD.

    Grid is log-spaced over [0.05, 20].  Returns 1.0 (identity) when the
    calibration tail has fewer than 2 distinct classes — there is nothing
    to calibrate against, and inventing a temperature would be fiction.
    """
    yv = np.asarray(y, dtype=np.int64)
    if len(set(yv.tolist())) < 2:
        return 1.0
    lg = np.asarray(logits, dtype=np.float64)
    best_t, best_nll = 1.0, float("inf")
    for t in np.exp(np.linspace(math.log(0.05), math.log(20.0), 121)):
        p = apply_temperature_np(lg, float(t))
        nll = float(-np.mean(np.log(p[np.arange(len(yv)), yv] + 1e-12)))
        if nll < best_nll - 1e-12:
            best_nll, best_t = nll, float(t)
    return best_t


def multiclass_log_loss(y: Sequence[int], p: np.ndarray) -> Optional[float]:
    if not len(y):
        return None
    yv = np.asarray(y, dtype=np.int64)
    pv = np.asarray(p, dtype=np.float64)
    return float(-np.mean(np.log(pv[np.arange(len(yv)), yv] + 1e-12)))


def multiclass_brier(y: Sequence[int], p: np.ndarray) -> Optional[float]:
    """Mean squared distance to the one-hot target (sums over classes)."""
    if not len(y):
        return None
    yv = np.asarray(y, dtype=np.int64)
    pv = np.asarray(p, dtype=np.float64)
    onehot = np.zeros_like(pv)
    onehot[np.arange(len(yv)), yv] = 1.0
    return float(np.mean(np.sum((pv - onehot) ** 2, axis=1)))


def confusion_matrix(y: Sequence[int], pred: Sequence[int],
                     n_classes: int = 3) -> List[List[int]]:
    """rows = true class, cols = predicted class."""
    m = [[0] * n_classes for _ in range(n_classes)]
    for t, pr in zip(y, pred):
        m[int(t)][int(pr)] += 1
    return m


def macro_f1(y: Sequence[int], pred: Sequence[int],
             n_classes: int = 3) -> Optional[float]:
    """Unweighted mean of per-class F1.  Classes absent from BOTH truth
    and predictions are excluded (undefined), never counted as 0."""
    if not len(y):
        return None
    m = confusion_matrix(y, pred, n_classes)
    f1s = []
    for k in range(n_classes):
        tp = m[k][k]
        fp = sum(m[r][k] for r in range(n_classes)) - tp
        fn = sum(m[k][c] for c in range(n_classes)) - tp
        if tp + fp + fn == 0:
            continue
        f1s.append(2 * tp / (2 * tp + fp + fn))
    return (sum(f1s) / len(f1s)) if f1s else None


def per_class_reliability(y: Sequence[int], p: np.ndarray,
                          n_bins: int = 10) -> Dict[str, Any]:
    """One-vs-rest reliability bins + ECE per class (canonical order)."""
    out: Dict[str, Any] = {}
    yv = np.asarray(y, dtype=np.int64)
    pv = np.asarray(p, dtype=np.float64)
    for k, name in enumerate(DIRECTION_CLASSES):
        yk = (yv == k).astype(np.int64)
        bins = reliability_bins(yk.tolist(), pv[:, k].tolist(), n_bins)
        out[name] = {"bins": bins,
                     "ece": expected_calibration_error(bins)}
    return out


def constant_class_probs(train_cls: Sequence[int], n_test: int,
                         n_classes: int = 3) -> np.ndarray:
    """Historical per-class frequency comparator (uniform if no train)."""
    counts = np.zeros(n_classes)
    for c in train_cls:
        counts[int(c)] += 1.0
    if counts.sum() == 0:
        base = np.full(n_classes, 1.0 / n_classes)
    else:
        base = counts / counts.sum()
    return np.tile(base, (n_test, 1))


def momentum_class_probs(train_samples: Sequence[Sample],
                         train_cls: Sequence[int],
                         test_samples: Sequence[Sample],
                         feature: str = "ret_60m",
                         n_classes: int = 3,
                         ) -> Tuple[np.ndarray, np.ndarray]:
    """Sign-of-trailing-return 3-class comparator.

    P(class | sign bucket) is the train class frequency within that
    bucket.  Rows where the feature is unavailable are masked out
    (prediction unavailable), never invented.
    Returns (probs (n, K), available_mask (n,)).
    """
    pos = np.zeros(n_classes)
    neg = np.zeros(n_classes)
    for s, c in zip(train_samples, train_cls):
        v = s.features.get(feature)
        if v is None or not isinstance(v, (int, float)) or v != v:
            continue
        (pos if v >= 0 else neg)[int(c)] += 1.0
    base = constant_class_probs(train_cls, 1, n_classes)[0]
    p_pos = pos / pos.sum() if pos.sum() else base
    p_neg = neg / neg.sum() if neg.sum() else base
    probs = np.tile(base, (len(test_samples), 1))
    mask = np.zeros(len(test_samples))
    for i, s in enumerate(test_samples):
        v = s.features.get(feature)
        if v is None or not isinstance(v, (int, float)) or v != v:
            continue
        probs[i] = p_pos if v >= 0 else p_neg
        mask[i] = 1.0
    return probs, mask


# --------------------------------------------------------------------------- #
# classical artifact save / reload with integrity check
# --------------------------------------------------------------------------- #

class ClassicalIntegrityError(RuntimeError):
    """Raised when a classical artifact fails its checksum on reload."""


def _classical_payload_bytes(payload: Dict[str, Any]) -> bytes:
    return json.dumps(payload, sort_keys=True,
                      separators=(",", ":")).encode("utf-8")


def save_classical(path: str, model: MultinomialLogisticModel,
                   feature_columns: Sequence[str],
                   temperature: float = 1.0,
                   meta: Optional[Dict[str, Any]] = None) -> str:
    """Write a JSON artifact with sha256 over its payload; returns digest."""
    assert model.w is not None, "fit first"
    payload = {
        "kind": "multinomial_logistic",
        "n_classes": model.n_classes,
        "l2": model.l2, "lr": model.lr, "iters": model.iters,
        "weights": [[float(v) for v in row] for row in model.w],
        "feature_columns": list(feature_columns),
        "classes": list(DIRECTION_CLASSES[:model.n_classes]),
        "temperature": float(temperature),
        "meta": meta or {},
    }
    digest = hashlib.sha256(_classical_payload_bytes(payload)).hexdigest()
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"payload": payload, "checksum_sha256": digest}, fh,
                  sort_keys=True, indent=1)
    return digest


def load_classical(path: str
                   ) -> Tuple[MultinomialLogisticModel, List[str], float,
                              Dict[str, Any]]:
    """Reload and verify; raises ClassicalIntegrityError on any mismatch."""
    with open(path, "r", encoding="utf-8") as fh:
        doc = json.load(fh)
    payload = doc.get("payload")
    stored = doc.get("checksum_sha256")
    if not isinstance(payload, dict) or not stored:
        raise ClassicalIntegrityError(f"malformed classical artifact: {path}")
    digest = hashlib.sha256(_classical_payload_bytes(payload)).hexdigest()
    if digest != stored:
        raise ClassicalIntegrityError(
            f"classical artifact checksum mismatch: {path}")
    model = MultinomialLogisticModel(
        n_classes=int(payload["n_classes"]), l2=float(payload["l2"]),
        lr=float(payload["lr"]), iters=int(payload["iters"]))
    model.w = np.asarray(payload["weights"], dtype=np.float64)
    return (model, list(payload["feature_columns"]),
            float(payload.get("temperature", 1.0)),
            dict(payload.get("meta", {})))
