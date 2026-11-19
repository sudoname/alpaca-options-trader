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
