"""Oracle 4.0.1 Stage D -- EV / probability-of-profit calibration study.

Learns net P&L and probability of profit *conditional on actual execution*
from the legacy trade history, strictly out-of-sample:

  * opportunities are built from resolved history trades (paper marks
    provenance -- labelled as such, never claimed as broker fills);
  * models fit on prior sessions only (chronological split, enforced);
  * comparators: historical base rate, logistic recalibration of the
    stamped PoP, cohort empirical rates with hierarchical shrinkage,
    and EV-quantile buckets;
  * metrics: Brier, log loss, calibration slope/intercept, Spearman rank
    correlation with session-block bootstrap CI, signed dollar residuals,
    reliability bins, realized expectancy per opportunity and per fill;
  * verdict requires meaningful out-of-sample rank power; otherwise the
    challenger must abstain and the study is marked ``NO_VALIDATED_EDGE``.
  * no thresholds are optimized on the evaluation set.

Research-only; no production module imports this file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import random
from datetime import date, datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

__all__ = [
    "opportunities_from_history",
    "chronological_split",
    "validate_split",
    "fit_logistic",
    "cohort_rates",
    "ev_bucket_edges",
    "spearman",
    "evaluate",
    "run_pnl_calibration",
]

UTC = timezone.utc

P_CLAMP = 1e-4          # PoP clamp before logit
LOSS_CLAMP = 1e-6       # probability clamp inside log loss
LOGISTIC_L2 = 1e-4      # ridge toward (a=0, b=1) -- pre-registered
SHRINKAGE_K = 20.0      # hierarchical shrinkage pseudo-count -- pre-registered
N_BOOT = 500
BOOT_SEED = 7
MIN_EVAL_N = 50
DTE_BUCKETS = ((1, "0-1"), (7, "2-7"), (30, "8-30"))


# --------------------------------------------------------------------------- #
# parsing helpers
# --------------------------------------------------------------------------- #

def _parse_dt(raw: Any) -> Optional[datetime]:
    if not raw or not isinstance(raw, str):
        return None
    try:
        dt = datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _f(v: Any) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
            return None
        out = float(v)
    except (TypeError, ValueError):
        return None
    return out if math.isfinite(out) else None


def _occ_parts(symbol: Any) -> Tuple[Optional[str], Optional[date],
                                     Optional[str]]:
    """(underlying, expiry, side) from an OCC symbol; None where unparseable."""
    if not isinstance(symbol, str) or len(symbol) < 16:
        return None, None, None
    tail = symbol[-15:]           # yymmdd + C/P + 8-digit strike
    root = symbol[:-15]
    if not (tail[:6].isdigit() and tail[6] in "CP" and tail[7:].isdigit()):
        return None, None, None
    try:
        exp = date(2000 + int(tail[0:2]), int(tail[2:4]), int(tail[4:6]))
    except ValueError:
        return root or None, None, None
    side = "call" if tail[6] == "C" else "put"
    return (root or None), exp, side


def _dte_bucket(dte: Optional[int]) -> str:
    if dte is None:
        return "unknown"
    for limit, name in DTE_BUCKETS:
        if dte <= limit:
            return name
    return ">30"


# --------------------------------------------------------------------------- #
# opportunity extraction
# --------------------------------------------------------------------------- #

def opportunities_from_history(
    trades: Sequence[Dict[str, Any]],
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Turn resolved history trades into calibration opportunities.

    Same-unit enforcement: the stamped EV must declare
    ``metrics.ev_units == "usd_total"`` -- trades without it are counted
    and skipped, never silently converted.  Realized P&L here is the
    legacy mark-based ``pnl`` in USD (provenance ``paper_marks``).
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
        pnl = _f(t.get("pnl"))
        p_pred = _f(t.get("probability_of_profit"))
        ev_pred = _f(t.get("expected_value"))
        if pnl is None or p_pred is None or ev_pred is None:
            report["dropped_missing_fields"] += 1
            continue
        metrics = t.get("metrics") or {}
        if metrics.get("ev_units") != "usd_total":
            report["dropped_unit_mismatch"] += 1
            continue
        entry = _parse_dt(t.get("entry_time"))
        if entry is None:
            report["dropped_bad_timestamp"] += 1
            continue
        underlying, expiry, side = _occ_parts(t.get("symbol"))
        dte = (expiry - entry.date()).days if expiry else None
        out.append({
            "entry_time": entry,
            "session": entry.date().isoformat(),
            "symbol": t.get("symbol"),
            "underlying": underlying,
            "side": side or "unknown",
            "dte": dte,
            "dte_bucket": _dte_bucket(dte),
            "regime": t.get("regime_label") or "unknown",
            "p_pred": min(max(p_pred, 0.0), 1.0),
            "ev_pred": ev_pred,
            "realized": pnl,
            "win": pnl > 0.0,
            "provenance": "paper_marks",
        })
        report["kept"] += 1
    out.sort(key=lambda o: (o["entry_time"], str(o["symbol"])))
    return out, report


# --------------------------------------------------------------------------- #
# chronological split (fit on prior sessions only)
# --------------------------------------------------------------------------- #

def chronological_split(
    opps: Sequence[Dict[str, Any]],
    eval_frac: float = 0.25,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], Dict[str, Any]]:
    """Split by whole sessions: training sessions strictly precede every
    evaluation session.  Raises if the data cannot form a valid split."""
    if not 0.0 < eval_frac < 1.0:
        raise ValueError("eval_frac must be in (0, 1)")
    sessions = sorted({o["session"] for o in opps})
    if len(sessions) < 2:
        raise ValueError("need at least 2 sessions for a chronological split")
    n_eval = max(1, int(round(len(sessions) * eval_frac)))
    n_eval = min(n_eval, len(sessions) - 1)
    eval_sessions = set(sessions[-n_eval:])
    boundary = sessions[-n_eval]
    train = [o for o in opps if o["session"] not in eval_sessions]
    eval_ = [o for o in opps if o["session"] in eval_sessions]
    meta = {
        "n_sessions": len(sessions),
        "n_train_sessions": len(sessions) - n_eval,
        "n_eval_sessions": n_eval,
        "first_eval_session": boundary,
        "n_train": len(train),
        "n_eval": len(eval_),
    }
    validate_split(train, eval_)
    return train, eval_, meta


def validate_split(train: Sequence[Dict[str, Any]],
                   eval_: Sequence[Dict[str, Any]]) -> None:
    """Hard guard: no training sample may come from an evaluation session
    or later.  Raises ValueError (future data must never leak into fit)."""
    if not eval_:
        return
    first_eval = min(o["session"] for o in eval_)
    for o in train:
        if o["session"] >= first_eval:
            raise ValueError(
                f"training sample from session {o['session']} is not "
                f"strictly before first eval session {first_eval}")


# --------------------------------------------------------------------------- #
# models (fit on train only)
# --------------------------------------------------------------------------- #

def _logit(p: float) -> float:
    p = min(max(p, P_CLAMP), 1.0 - P_CLAMP)
    return math.log(p / (1.0 - p))


def _sigmoid(z: float) -> float:
    if z >= 0:
        return 1.0 / (1.0 + math.exp(-z))
    e = math.exp(z)
    return e / (1.0 + e)


def fit_logistic(xs: Sequence[float], ys: Sequence[bool],
                 l2: float = LOGISTIC_L2) -> Tuple[float, float]:
    """Deterministic 2-parameter logistic fit p = sigmoid(a + b*x) via
    Newton iterations with a small ridge toward (a=0, b=1).  Used both to
    recalibrate the stamped PoP (x = logit(p_pred)) and to measure
    calibration slope/intercept on held-out data."""
    if len(xs) != len(ys) or not xs:
        raise ValueError("fit_logistic needs equal-length non-empty inputs")
    a, b = 0.0, 1.0
    for _ in range(100):
        g_a = 2.0 * l2 * a
        g_b = 2.0 * l2 * (b - 1.0)
        h_aa = 2.0 * l2
        h_ab = 0.0
        h_bb = 2.0 * l2
        for x, y in zip(xs, ys):
            p = _sigmoid(a + b * x)
            w = p * (1.0 - p)
            r = p - (1.0 if y else 0.0)
            g_a += r
            g_b += r * x
            h_aa += w
            h_ab += w * x
            h_bb += w * x * x
        det = h_aa * h_bb - h_ab * h_ab
        if det <= 1e-12:
            break
        da = (h_bb * g_a - h_ab * g_b) / det
        db = (h_aa * g_b - h_ab * g_a) / det
        a -= da
        b -= db
        a = min(max(a, -50.0), 50.0)
        b = min(max(b, -50.0), 50.0)
        if abs(da) < 1e-10 and abs(db) < 1e-10:
            break
    return a, b


def cohort_rates(train: Sequence[Dict[str, Any]],
                 k: float = SHRINKAGE_K) -> Dict[str, Any]:
    """Empirical win rate per (regime, side, dte_bucket) cohort with
    hierarchical shrinkage toward the global rate:
    rate = (n*r + k*g) / (n + k)."""
    if not train:
        raise ValueError("cohort_rates needs a non-empty training set")
    g = sum(1 for o in train if o["win"]) / len(train)
    counts: Dict[str, List[int]] = {}
    for o in train:
        key = f"{o['regime']}|{o['side']}|{o['dte_bucket']}"
        c = counts.setdefault(key, [0, 0])
        c[0] += 1
        c[1] += 1 if o["win"] else 0
    rates = {key: (c[1] + k * g) / (c[0] + k) for key, c in counts.items()}
    return {"global_rate": g, "k": k, "rates": rates,
            "cohort_n": {key: c[0] for key, c in counts.items()}}


def _cohort_predict(model: Dict[str, Any], o: Dict[str, Any]) -> float:
    key = f"{o['regime']}|{o['side']}|{o['dte_bucket']}"
    return model["rates"].get(key, model["global_rate"])


def ev_bucket_edges(train: Sequence[Dict[str, Any]],
                    n_buckets: int = 5) -> List[float]:
    """Quantile edges of the stamped EV, from training data only."""
    if not train:
        raise ValueError("ev_bucket_edges needs a non-empty training set")
    evs = sorted(o["ev_pred"] for o in train)
    edges = []
    for i in range(1, n_buckets):
        idx = min(len(evs) - 1, max(0, int(round(i * len(evs) / n_buckets))))
        edges.append(evs[idx])
    return edges


def _bucket_of(ev: float, edges: Sequence[float]) -> int:
    for i, e in enumerate(edges):
        if ev < e:
            return i
    return len(edges)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #

def _rank(values: Sequence[float]) -> List[float]:
    """Average-tie ranks (1-based)."""
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2.0 + 1.0
        for idx in range(i, j + 1):
            ranks[order[idx]] = avg
        i = j + 1
    return ranks


def spearman(x: Sequence[float], y: Sequence[float]) -> Optional[float]:
    """Spearman rank correlation (Pearson on average-tie ranks)."""
    n = len(x)
    if n != len(y) or n < 3:
        return None
    rx, ry = _rank(x), _rank(y)
    mx = sum(rx) / n
    my = sum(ry) / n
    cov = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    vx = sum((a - mx) ** 2 for a in rx)
    vy = sum((b - my) ** 2 for b in ry)
    if vx <= 0 or vy <= 0:
        return None
    return cov / math.sqrt(vx * vy)


def _brier(ps: Sequence[float], ys: Sequence[bool]) -> float:
    return sum((p - (1.0 if y else 0.0)) ** 2
               for p, y in zip(ps, ys)) / len(ps)


def _log_loss(ps: Sequence[float], ys: Sequence[bool]) -> float:
    total = 0.0
    for p, y in zip(ps, ys):
        p = min(max(p, LOSS_CLAMP), 1.0 - LOSS_CLAMP)
        total += -math.log(p) if y else -math.log(1.0 - p)
    return total / len(ps)


def _ols(x: Sequence[float], y: Sequence[float]
         ) -> Tuple[Optional[float], Optional[float]]:
    """(slope, intercept) of y ~ a + b*x; None on degenerate x."""
    n = len(x)
    if n < 3:
        return None, None
    mx = sum(x) / n
    my = sum(y) / n
    sxx = sum((a - mx) ** 2 for a in x)
    if sxx <= 0:
        return None, None
    b = sum((a - mx) * (c - my) for a, c in zip(x, y)) / sxx
    return b, my - b * mx


def _session_bootstrap_spearman(
    eval_: Sequence[Dict[str, Any]],
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
) -> Optional[Dict[str, Any]]:
    """Session-block bootstrap CI for Spearman(ev_pred, realized)."""
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for o in eval_:
        by_session.setdefault(o["session"], []).append(o)
    sessions = sorted(by_session)
    if len(sessions) < 2:
        return None
    rng = random.Random(seed)
    draws: List[float] = []
    for _ in range(n_boot):
        sample: List[Dict[str, Any]] = []
        for _ in sessions:
            sample.extend(by_session[rng.choice(sessions)])
        rho = spearman([o["ev_pred"] for o in sample],
                       [o["realized"] for o in sample])
        if rho is not None:
            draws.append(rho)
    if len(draws) < n_boot // 2:
        return None
    draws.sort()
    lo = draws[int(0.025 * len(draws))]
    hi = draws[min(len(draws) - 1, int(0.975 * len(draws)))]
    return {"n_boot": len(draws), "seed": seed,
            "ci95": [lo, hi], "lower_bound": lo}


def _mean_ci(values: Sequence[float]) -> Dict[str, Any]:
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "ci95": None}
    mean = sum(values) / n
    if n > 1:
        sd = math.sqrt(sum((v - mean) ** 2 for v in values) / (n - 1))
        half = 1.96 * sd / math.sqrt(n)
        return {"n": n, "mean": mean, "ci95": [mean - half, mean + half]}
    return {"n": n, "mean": mean, "ci95": None}


# --------------------------------------------------------------------------- #
# evaluation (forward sessions only)
# --------------------------------------------------------------------------- #

def evaluate(
    train: Sequence[Dict[str, Any]],
    eval_: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    """Fit comparators on ``train`` and score them on ``eval_``.

    All model parameters (recalibration coefficients, cohort rates,
    bucket edges) come from training sessions only; nothing is tuned on
    the evaluation set."""
    validate_split(train, eval_)
    if not train or not eval_:
        raise ValueError("evaluate needs non-empty train and eval sets")

    # ---- fit on train only
    base_rate = sum(1 for o in train if o["win"]) / len(train)
    rec_a, rec_b = fit_logistic(
        [_logit(o["p_pred"]) for o in train],
        [o["win"] for o in train])
    cohorts = cohort_rates(train)
    edges = ev_bucket_edges(train)

    ys = [o["win"] for o in eval_]
    realized = [o["realized"] for o in eval_]
    ev_pred = [o["ev_pred"] for o in eval_]

    models: Dict[str, List[float]] = {
        "stamped_pop": [o["p_pred"] for o in eval_],
        "base_rate": [base_rate] * len(eval_),
        "logistic_recal": [
            _sigmoid(rec_a + rec_b * _logit(o["p_pred"])) for o in eval_],
        "cohort_shrunk": [_cohort_predict(cohorts, o) for o in eval_],
    }

    prob_metrics: Dict[str, Any] = {}
    for name, ps in models.items():
        entry: Dict[str, Any] = {
            "brier": _brier(ps, ys),
            "log_loss": _log_loss(ps, ys),
        }
        if len(set(ps)) > 1:
            ca, cb = fit_logistic([_logit(p) for p in ps], ys)
            entry["calibration_intercept"] = ca
            entry["calibration_slope"] = cb
        else:
            entry["calibration_intercept"] = None
            entry["calibration_slope"] = None
        prob_metrics[name] = entry

    # ---- EV rank power & residuals
    rho = spearman(ev_pred, realized)
    boot = _session_bootstrap_spearman(eval_)
    ev_slope, ev_intercept = _ols(ev_pred, realized)
    residuals = _mean_ci([r - e for r, e in zip(realized, ev_pred)])

    # ---- reliability bins (stamped PoP, 10 equal-width bins)
    bins: List[Dict[str, Any]] = []
    for i in range(10):
        lo, hi = i / 10.0, (i + 1) / 10.0
        members = [(p, y) for p, y in zip(models["stamped_pop"], ys)
                   if (lo <= p < hi) or (i == 9 and p == 1.0)]
        if members:
            bins.append({
                "bin": f"[{lo:.1f},{hi:.1f})",
                "n": len(members),
                "mean_pred": sum(p for p, _ in members) / len(members),
                "frac_win": sum(1 for _, y in members if y) / len(members),
            })

    # ---- EV buckets: train edges, eval realized
    bucket_rows: List[Dict[str, Any]] = []
    for i in range(len(edges) + 1):
        rows = [o for o in eval_ if _bucket_of(o["ev_pred"], edges) == i]
        bucket_rows.append({
            "bucket": i,
            "n": len(rows),
            "mean_ev_pred": (sum(o["ev_pred"] for o in rows) / len(rows)
                             if rows else None),
            "mean_realized": (sum(o["realized"] for o in rows) / len(rows)
                              if rows else None),
        })

    # Every history opportunity is a resolved paper fill, so per-fill and
    # per-opportunity expectancy coincide here; unresolved/unfilled
    # opportunities are structurally absent from this source (coverage
    # limitation, reported -- not hidden).
    expectancy = _mean_ci(realized)

    out = {
        "n_train": len(train),
        "n_eval": len(eval_),
        "train_base_rate": base_rate,
        "eval_base_rate": sum(1 for y in ys if y) / len(ys),
        "recalibration": {"intercept": rec_a, "slope": rec_b,
                          "l2": LOGISTIC_L2},
        "cohort_model": {"k": cohorts["k"],
                         "n_cohorts": len(cohorts["rates"]),
                         "global_rate": cohorts["global_rate"]},
        "prob_metrics": prob_metrics,
        "ev_rank": {
            "spearman": rho,
            "session_bootstrap": boot,
            "ols_slope_realized_on_ev": ev_slope,
            "ols_intercept": ev_intercept,
        },
        "signed_residual_usd": residuals,
        "reliability_bins": bins,
        "ev_buckets": {"edges_from_train": edges, "rows": bucket_rows},
        "expectancy_per_opportunity_usd": expectancy,
        "expectancy_per_fill_usd": expectancy,
        "per_fill_note": (
            "history rows are resolved paper fills; per-fill equals "
            "per-opportunity and unfilled opportunities are absent from "
            "this source (survivorship limitation)"),
    }
    return out


def _verdict(metrics: Dict[str, Any]) -> Dict[str, Any]:
    """Pre-registered verdict rule (no tuning on the eval set):
       * < MIN_EVAL_N eval rows        -> INSUFFICIENT_EVIDENCE
       * Spearman 95% LB <= 0, or OLS slope of realized-on-EV <= 0, or
         either unavailable            -> NO_VALIDATED_EDGE (abstain)
       * otherwise                     -> VALIDATED_RANK_POWER (research)
    """
    n_eval = metrics["n_eval"]
    if n_eval < MIN_EVAL_N:
        return {"verdict": "INSUFFICIENT_EVIDENCE",
                "challenger_policy": "abstain",
                "reason": f"only {n_eval} eval opportunities "
                          f"(< {MIN_EVAL_N})"}
    boot = metrics["ev_rank"]["session_bootstrap"]
    slope = metrics["ev_rank"]["ols_slope_realized_on_ev"]
    lb = boot["lower_bound"] if boot else None
    if lb is None or slope is None or lb <= 0.0 or slope <= 0.0:
        return {
            "verdict": "NO_VALIDATED_EDGE",
            "challenger_policy": "abstain",
            "reason": (f"spearman 95% LB={lb} / OLS slope={slope}: no "
                       "meaningful out-of-sample rank power established"),
        }
    return {
        "verdict": "VALIDATED_RANK_POWER",
        "challenger_policy": "research_only",
        "reason": (f"spearman 95% LB={lb:.4f} > 0 and OLS slope="
                   f"{slope:.4f} > 0 on forward sessions"),
    }


# --------------------------------------------------------------------------- #
# runner
# --------------------------------------------------------------------------- #

def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def run_pnl_calibration(
    history_path: str,
    eval_frac: float = 0.25,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Load history, split chronologically, evaluate, publish verdict."""
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "inputs": {"history": {"path": history_path,
                               "sha256": _sha256(history_path)}},
        "provenance": "paper_marks",
        "pre_registered": {
            "eval_frac": eval_frac,
            "logistic_l2": LOGISTIC_L2,
            "shrinkage_k": SHRINKAGE_K,
            "n_boot": N_BOOT,
            "boot_seed": BOOT_SEED,
            "min_eval_n": MIN_EVAL_N,
        },
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

    opps, extract_report = opportunities_from_history(
        [t for t in trades if isinstance(t, dict)])
    report["extraction"] = extract_report
    report["coverage"] = (extract_report["kept"]
                          / extract_report["rows_scanned"]
                          if extract_report["rows_scanned"] else None)

    try:
        train, eval_, split_meta = chronological_split(opps, eval_frac)
        report["split"] = split_meta
        metrics = evaluate(train, eval_)
        report["metrics"] = metrics
        report.update(_verdict(metrics))
    except ValueError as exc:
        report["verdict"] = "INSUFFICIENT_EVIDENCE"
        report["challenger_policy"] = "abstain"
        report["error"] = str(exc)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0.1 Stage D EV/PoP out-of-sample calibration")
    ap.add_argument("--history",
                    default="server_data/2026-11-19/trading_history.json")
    ap.add_argument("--eval-frac", type=float, default=0.25)
    ap.add_argument("--out",
                    default="artifacts/oracle401/pnl_calibration_report.json")
    args = ap.parse_args(argv)
    rep = run_pnl_calibration(args.history, args.eval_frac, out_path=args.out)
    summary = {k: rep.get(k) for k in
               ("verdict", "challenger_policy", "reason", "split")}
    if "metrics" in rep:
        summary["ev_rank"] = rep["metrics"]["ev_rank"]
        summary["prob_metrics"] = rep["metrics"]["prob_metrics"]
    print(json.dumps(summary, indent=2, default=str))
    print(f"report written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
