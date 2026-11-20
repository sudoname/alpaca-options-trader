"""Oracle 4.0 Phase 6a — execution calibration (research-only).

Calibrates the execution assumptions used by the valuation/replay stack
against every piece of *measured* execution evidence available:

  * ``execution_calibration.jsonl`` (bot's own execution log):
      - ``execution_estimate``    pre-trade model expectations
      - ``execution_realization`` post-trade resolutions (paper)
      - ``order_intent``          indicative kernel-shadow intents
        (never submitted to a broker; evidence of quotes only)
  * ``trading_history.json``      closed trades; the roll_take_profit /
      roll_failed_take_profit pair is the only genuine limit-order
      nonfill evidence in the system
  * dataset outcomes + research ledger: stop-loss overshoot
      distribution (realized gross return past the configured stop
      floor), per the Phase 1 study definition

Honesty rules (per the master prompt):
  * every estimate carries a provenance label (paper / indicative);
    paper fills are NEVER presented as proof of attainable live
    execution;
  * unknown coverage is reported as unknown — never as zero
    (e.g. ``actual_fill_delay`` is null on every record, so
    time-to-fill is an UNKNOWN_COVERAGE block, not "0s");
  * fill probabilities are shrunk toward a conservative prior when
    sample support is thin, with Jeffreys posterior intervals;
  * nothing here touches a broker or any production trading path.

CLI::

    python -m oracle.ml.execution \
        --execution-log server_data/2026-11-19/execution_calibration.jsonl \
        --history server_data/2026-11-19/trading_history.json \
        --dataset artifacts/oracle4/dataset_v1.sqlite \
        --ledger server_data/2026-11-19/research_ledger.db \
        --out artifacts/oracle4/execution_calibration.json
"""
from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Sequence, Tuple

from oracle.lab.purged_validation import (
    metrics_fingerprint,
    paired_session_bootstrap,
)

__all__ = [
    "load_execution_log",
    "link_pairs",
    "load_history_trades",
    "jeffreys_interval",
    "shrunk_fill_probability",
    "fill_probability_report",
    "slippage_report",
    "ev_calibration_report",
    "stop_overshoot_report",
    "quantiles",
    "spread_bucket",
    "run_execution_calibration",
    "main",
    "CONSERVATIVE_FILL_PRIOR",
    "PRIOR_STRENGTH",
]

# Conservative prior for fill probability shrinkage: deliberately below
# every observed paper fill rate so that thin groups are pulled DOWN,
# never up.  Documented in the artifact.
CONSERVATIVE_FILL_PRIOR = 0.80
PRIOR_STRENGTH = 10.0          # pseudo-observations behind the prior
MIN_GROUP = 20                 # groups thinner than this pool together

PROVENANCE_NOTE = (
    "All execution evidence in this artifact is from a PAPER account "
    "(simulated fills) or from indicative kernel-shadow intents that "
    "were never submitted to a broker. Paper fill rates and paper "
    "slippage are NOT proof of attainable live execution; they bound "
    "the model's own internal consistency only."
)


def _parse_utc(raw: Optional[str]) -> Optional[datetime]:
    if not raw:
        return None
    try:
        dt = datetime.fromisoformat(raw)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


def _session_key(dt: Optional[datetime]) -> Optional[str]:
    return dt.strftime("%Y-%m-%d") if dt else None


def _f(obj: Dict[str, Any], key: str) -> Optional[float]:
    v = obj.get(key)
    if v is None:
        return None
    try:
        v = float(v)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


# --------------------------------------------------------------------------- #
# loaders
# --------------------------------------------------------------------------- #

def load_execution_log(path: str) -> Tuple[Dict[str, List[Dict[str, Any]]],
                                           Dict[str, int]]:
    """Parse the execution-calibration jsonl into typed record lists.

    Keep-first dedup per ``(record_type, trade_id)``; bad JSON lines and
    rows without a trade_id are dropped and counted, never guessed.
    """
    report = {
        "lines_scanned": 0,
        "dropped_bad_json": 0,
        "dropped_missing_trade_id": 0,
        "dropped_duplicate": 0,
        "dropped_unknown_type": 0,
    }
    out: Dict[str, List[Dict[str, Any]]] = {
        "execution_estimate": [],
        "execution_realization": [],
        "order_intent": [],
    }
    seen: set = set()
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            report["lines_scanned"] += 1
            try:
                rec = json.loads(line)
            except ValueError:
                report["dropped_bad_json"] += 1
                continue
            if not isinstance(rec, dict):
                report["dropped_bad_json"] += 1
                continue
            rtype = rec.get("record_type")
            if rtype not in out:
                report["dropped_unknown_type"] += 1
                continue
            tid = rec.get("trade_id")
            if not tid:
                report["dropped_missing_trade_id"] += 1
                continue
            key = (rtype, tid)
            if key in seen:
                report["dropped_duplicate"] += 1
                continue
            seen.add(key)
            out[rtype].append(rec)
    for recs in out.values():
        recs.sort(key=lambda r: (r.get("recorded_at")
                                 or r.get("resolved_at") or "",
                                 r.get("trade_id") or ""))
    report.update({f"kept_{k}": len(v) for k, v in out.items()})
    return out, report


def link_pairs(estimates: Sequence[Dict[str, Any]],
               realizations: Sequence[Dict[str, Any]],
               ) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """Join estimate -> realization on trade_id (broker-id reconciliation).

    Returns merged pair dicts plus a report of unlinked counts on both
    sides — unlinked records are evidence gaps, not errors.
    """
    by_tid = {r["trade_id"]: r for r in realizations}
    pairs: List[Dict[str, Any]] = []
    linked_tids: set = set()
    for est in estimates:
        real = by_tid.get(est["trade_id"])
        if real is None:
            continue
        linked_tids.add(est["trade_id"])
        rec_at = _parse_utc(est.get("recorded_at"))
        pairs.append({
            "trade_id": est["trade_id"],
            "symbol": est.get("symbol"),
            "session": _session_key(rec_at),
            "qty": _f(est, "qty"),
            "mid_price": _f(est, "mid_price"),
            "spread_cost": _f(est, "spread_cost"),
            "expected_entry_price": _f(est, "expected_entry_price"),
            "expected_exit_price": _f(est, "expected_exit_price"),
            "fill_probability": _f(est, "fill_probability"),
            "expected_fill_delay": _f(est, "expected_fill_delay"),
            "executable_EV": _f(est, "executable_EV"),
            "theoretical_EV": _f(est, "theoretical_EV"),
            "filled": bool(real.get("filled")),
            "actual_entry_price": _f(real, "actual_entry_price"),
            "actual_exit_price": _f(real, "actual_exit_price"),
            "actual_fill_delay": _f(real, "actual_fill_delay"),
            "realized_EV": _f(real, "realized_EV"),
        })
    report = {
        "pairs_linked": len(pairs),
        "estimates_unlinked": len(estimates) - len(pairs),
        "realizations_unlinked": len(realizations) - len(linked_tids),
    }
    return pairs, report


def load_history_trades(path: str) -> Tuple[List[Dict[str, Any]],
                                            Dict[str, int]]:
    """Load closed trades from trading_history.json (validated, counted)."""
    with open(path, "r", encoding="utf-8") as fh:
        payload = json.load(fh)
    trades = payload.get("trades") if isinstance(payload, dict) else None
    if not isinstance(trades, list):
        return [], {"trades_kept": 0, "dropped_not_dict": 0}
    kept: List[Dict[str, Any]] = []
    dropped = 0
    for t in trades:
        if isinstance(t, dict):
            kept.append(t)
        else:
            dropped += 1
    return kept, {"trades_kept": len(kept), "dropped_not_dict": dropped}


# --------------------------------------------------------------------------- #
# statistics helpers
# --------------------------------------------------------------------------- #

def _betacf(a: float, b: float, x: float) -> float:
    """Continued fraction for the regularized incomplete beta (NR 6.4)."""
    max_it, eps, fpmin = 300, 3e-12, 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < fpmin:
        d = fpmin
    d = 1.0 / d
    h = d
    for m in range(1, max_it + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < fpmin:
            d = fpmin
        c = 1.0 + aa / c
        if abs(c) < fpmin:
            c = fpmin
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def _betainc(a: float, b: float, x: float) -> float:
    """Regularized incomplete beta I_x(a, b)."""
    if x <= 0.0:
        return 0.0
    if x >= 1.0:
        return 1.0
    ln_bt = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
             + a * math.log(x) + b * math.log(1.0 - x))
    bt = math.exp(ln_bt)
    if x < (a + 1.0) / (a + b + 2.0):
        return bt * _betacf(a, b, x) / a
    return 1.0 - bt * _betacf(b, a, 1.0 - x) / b


def _beta_quantile(a: float, b: float, q: float) -> float:
    """Inverse of I_x(a, b) by bisection (deterministic, no scipy)."""
    lo, hi = 0.0, 1.0
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if _betainc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


def jeffreys_interval(k: int, n: int,
                      level: float = 0.95) -> Tuple[float, float]:
    """Jeffreys Beta(0.5, 0.5) posterior interval for a binomial rate."""
    if n <= 0:
        return (0.0, 1.0)
    alpha = 1.0 - level
    a, b = k + 0.5, n - k + 0.5
    lo = 0.0 if k == 0 else _beta_quantile(a, b, alpha / 2.0)
    hi = 1.0 if k == n else _beta_quantile(a, b, 1.0 - alpha / 2.0)
    return (lo, hi)


def shrunk_fill_probability(k: int, n: int,
                            prior: float = CONSERVATIVE_FILL_PRIOR,
                            strength: float = PRIOR_STRENGTH) -> float:
    """Empirical-Bayes shrinkage toward the conservative prior."""
    return (k + strength * prior) / (n + strength)


def quantiles(values: Sequence[float],
              qs: Sequence[float] = (0.10, 0.25, 0.50, 0.75,
                                     0.90, 0.99)) -> Dict[str, float]:
    """Deterministic linear-interpolation quantiles (sorted copy)."""
    xs = sorted(float(v) for v in values)
    out: Dict[str, float] = {}
    n = len(xs)
    if n == 0:
        return out
    for q in qs:
        pos = q * (n - 1)
        lo = int(math.floor(pos))
        hi = min(lo + 1, n - 1)
        frac = pos - lo
        out[f"p{int(round(q * 100)):02d}"] = xs[lo] + frac * (xs[hi] - xs[lo])
    out["min"] = xs[0]
    out["max"] = xs[-1]
    out["mean"] = sum(xs) / n
    return out


def spread_bucket(spread_pct: Optional[float]) -> str:
    """Relative bid/ask spread bucket (fraction of mid notional)."""
    if spread_pct is None:
        return "unknown"
    if spread_pct < 0.02:
        return "tight(<2%)"
    if spread_pct < 0.05:
        return "normal(2-5%)"
    if spread_pct < 0.10:
        return "wide(5-10%)"
    return "very_wide(>=10%)"


def _pair_spread_pct(pair: Dict[str, Any]) -> Optional[float]:
    """spread_cost is dollars for the whole order: /(mid*100*qty)."""
    sc, mid, qty = (pair.get("spread_cost"), pair.get("mid_price"),
                    pair.get("qty"))
    if not sc or not mid or not qty or mid <= 0 or qty <= 0:
        return None
    return sc / (mid * 100.0 * qty)


# --------------------------------------------------------------------------- #
# report blocks
# --------------------------------------------------------------------------- #

def _binomial_block(k: int, n: int, provenance: str,
                    note: str) -> Dict[str, Any]:
    lo, hi = jeffreys_interval(k, n)
    return {
        "n": n,
        "filled": k,
        "raw_rate": (k / n) if n else None,
        "jeffreys_95ci": [lo, hi] if n else None,
        "shrunk_rate": shrunk_fill_probability(k, n) if n else None,
        "prior": CONSERVATIVE_FILL_PRIOR,
        "prior_strength": PRIOR_STRENGTH,
        "provenance": provenance,
        "note": note,
    }


def fill_probability_report(records: Dict[str, List[Dict[str, Any]]],
                            pairs: Sequence[Dict[str, Any]],
                            history: Sequence[Dict[str, Any]],
                            min_group: int = MIN_GROUP) -> Dict[str, Any]:
    """Fill-probability evidence, grouped where the sample supports it.

    Three evidence sources, each labeled:
      * resolved paper realizations (survivorship: only resolved rows
        exist, and every one is filled — an upper bound, not a rate);
      * limit take-profit rolls from trading history (the ONLY genuine
        nonfill evidence: roll_take_profit vs roll_failed_take_profit);
      * indicative kernel-shadow intents (quotes only; no outcome).
    """
    reals = records["execution_realization"]
    k_res = sum(1 for r in reals if r.get("filled"))

    by_bucket: Dict[str, List[int]] = {}
    for p in pairs:
        by_bucket.setdefault(spread_bucket(_pair_spread_pct(p)), []).append(
            1 if p["filled"] else 0)
    groups: Dict[str, Any] = {}
    pooled: List[int] = []
    for bucket in sorted(by_bucket):
        flags = by_bucket[bucket]
        if len(flags) < min_group:
            pooled.extend(flags)
            continue
        groups[bucket] = _binomial_block(
            sum(flags), len(flags), "paper_resolved",
            "linked estimate->realization pairs in this spread bucket")
    if pooled:
        groups["pooled(<min_group)"] = _binomial_block(
            sum(pooled), len(pooled), "paper_resolved",
            f"buckets with n<{min_group} pooled; shrunk toward prior")

    tp_filled = sum(1 for t in history
                    if t.get("outcome") == "roll_take_profit")
    tp_failed = sum(1 for t in history
                    if t.get("outcome") == "roll_failed_take_profit")

    intents = records["order_intent"]
    return {
        "resolved_paper_overall": _binomial_block(
            k_res, len(reals), "paper_resolved",
            "SURVIVORSHIP: the log only contains resolved realizations "
            "and every one is filled=true. This is an upper bound from "
            "paper marks, not a live fill rate."),
        "by_spread_bucket": groups,
        "limit_take_profit": _binomial_block(
            tp_filled, tp_filled + tp_failed, "paper_limit_orders",
            "roll_take_profit (limit TP filled) vs "
            "roll_failed_take_profit (limit TP expired unfilled). The "
            "only genuine nonfill evidence in the system."),
        "indicative_intents": {
            "n": len(intents),
            "submitted_to_broker": 0,
            "provenance": "indicative_kernel_shadow",
            "note": "order_intent records are kernel-shadow quotes that "
                    "were never submitted; they carry NO fill outcome "
                    "and contribute no fill-rate evidence.",
        },
        "min_group": min_group,
    }


def slippage_report(pairs: Sequence[Dict[str, Any]],
                    history: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Signed slippage / price-realization evidence (paper provenance)."""
    entry_vs_mid: List[float] = []        # (actual - mid) / mid, buy side
    entry_vs_expected: List[float] = []   # (actual - expected) / mid
    exit_vs_expected: List[float] = []    # (actual - expected) / mid
    for p in pairs:
        mid = p.get("mid_price")
        if not mid or mid <= 0:
            continue
        ae, ee = p.get("actual_entry_price"), p.get("expected_entry_price")
        ax, ex = p.get("actual_exit_price"), p.get("expected_exit_price")
        if ae is not None:
            entry_vs_mid.append((ae - mid) / mid)
            if ee is not None:
                entry_vs_expected.append((ae - ee) / mid)
        if ax is not None and ex is not None:
            exit_vs_expected.append((ax - ex) / mid)

    fills_rows = [t for t in history if t.get("pnl_source") == "fills"]
    return {
        "entry_vs_mid": {
            "n": len(entry_vs_mid),
            "quantiles": quantiles(entry_vs_mid),
            "provenance": "paper_resolved",
            "note": "signed (actual_entry - mid)/mid at decision quote; "
                    "positive = paid above mid (buy side). Paper fills.",
        },
        "entry_vs_expected": {
            "n": len(entry_vs_expected),
            "quantiles": quantiles(entry_vs_expected),
            "provenance": "paper_resolved",
            "note": "(actual_entry - model expected_entry)/mid; measures "
                    "the entry model's bias, not live slippage.",
        },
        "exit_vs_expected": {
            "n": len(exit_vs_expected),
            "quantiles": quantiles(exit_vs_expected),
            "provenance": "paper_resolved",
            "note": "(actual_exit - model expected_exit)/mid; the "
                    "expected exit is an EV-weighted model value, so "
                    "this is exit-path realization error, NOT slippage.",
        },
        "fills_sourced_closes": {
            "n": len(fills_rows),
            "provenance": "paper_broker_fills",
            "note": "closes booked from broker fill prices "
                    "(pnl_source='fills'); no mark-based counterpart is "
                    "stored per trade, so mark-vs-fill exit slippage is "
                    "UNKNOWN coverage, not zero.",
        },
        "fees": {
            "n_observed": 0,
            "provenance": "unknown",
            "note": "no realized fee field exists in any execution "
                    "record; fee realism is UNKNOWN coverage.",
        },
    }


def time_to_fill_report(pairs: Sequence[Dict[str, Any]]) -> Dict[str, Any]:
    """Time-to-fill: expected values exist, actuals are never recorded."""
    expected = [p["expected_fill_delay"] for p in pairs
                if p.get("expected_fill_delay") is not None]
    actual = [p["actual_fill_delay"] for p in pairs
              if p.get("actual_fill_delay") is not None]
    return {
        "status": "UNKNOWN_COVERAGE" if not actual else "OK",
        "n_expected": len(expected),
        "n_actual": len(actual),
        "expected_quantiles_s": quantiles(expected) if expected else {},
        "actual_quantiles_s": quantiles(actual) if actual else {},
        "note": "actual_fill_delay is null on every realization record; "
                "time-to-fill calibration is impossible with current "
                "logging. Reported as unknown coverage — NOT zero.",
    }


def ev_calibration_report(pairs: Sequence[Dict[str, Any]],
                          n_boot: int = 2000,
                          seed: int = 42) -> Dict[str, Any]:
    """Predicted (executable_EV) vs realized EV on linked paper pairs."""
    rows = [(p["executable_EV"], p["realized_EV"], p["session"])
            for p in pairs
            if p.get("executable_EV") is not None
            and p.get("realized_EV") is not None]
    n = len(rows)
    if n == 0:
        return {"n": 0, "status": "INSUFFICIENT_EVIDENCE"}
    pred = [r[0] for r in rows]
    real = [r[1] for r in rows]
    err = [re - pr for pr, re in zip(pred, real)]
    mean_pred = sum(pred) / n
    mean_real = sum(real) / n
    var_pred = sum((p - mean_pred) ** 2 for p in pred)
    cov = sum((p - mean_pred) * (r - mean_real)
              for p, r in zip(pred, real))
    slope = (cov / var_pred) if var_pred > 0 else None
    intercept = (mean_real - slope * mean_pred) if slope is not None else None
    var_real = sum((r - mean_real) ** 2 for r in real)
    corr = (cov / math.sqrt(var_pred * var_real)
            if var_pred > 0 and var_real > 0 else None)

    per_session: Dict[str, List[float]] = {}
    for (pr, re, sess), e in zip(rows, err):
        if sess:
            per_session.setdefault(sess, []).append(e)
    session_mean = {s: sum(v) / len(v) for s, v in per_session.items()}
    boot = paired_session_bootstrap(session_mean, n_boot=n_boot, seed=seed)
    return {
        "n": n,
        "n_sessions": len(session_mean),
        "mean_predicted_ev": mean_pred,
        "mean_realized_ev": mean_real,
        "mean_error_realized_minus_predicted": sum(err) / n,
        "mae": sum(abs(e) for e in err) / n,
        "ols_slope_realized_on_predicted": slope,
        "ols_intercept": intercept,
        "correlation": corr,
        "error_quantiles": quantiles(err),
        "session_bootstrap_error_ci": boot,
        "provenance": "paper_resolved",
        "note": "realized EV comes from paper marks; a slope far below "
                "1 means the executable-EV model is optimistic even "
                "against its own paper fills.",
    }


def stop_overshoot_report(dataset_path: str,
                          ledger_path: str) -> Dict[str, Any]:
    """Realized gross return past the configured stop floor (Phase 1 defn).

    overshoot_pts := (-stop_loss_pct*100) - gross_pnl_pct
    positive = realized worse than the configured floor.
    Consumable by the valuation exit simulator as quantiles.
    """
    led = sqlite3.connect(f"file:{ledger_path}?mode=ro", uri=True)
    led.row_factory = sqlite3.Row
    stop_by_decision: Dict[str, float] = {}
    try:
        cur = led.execute(
            "SELECT decision_id, prediction_json FROM snapshots "
            "WHERE prediction_json LIKE '%stop_loss_pct%'")
        for row in cur:
            try:
                pj = json.loads(row["prediction_json"] or "{}")
            except ValueError:
                continue
            sl = (pj.get("exit_policy") or {}).get("stop_loss_pct")
            if isinstance(sl, (int, float)) and math.isfinite(sl) and sl > 0:
                stop_by_decision.setdefault(row["decision_id"], float(sl))
    finally:
        led.close()

    ds = sqlite3.connect(f"file:{dataset_path}?mode=ro", uri=True)
    ds.row_factory = sqlite3.Row
    per_reason: Dict[str, List[float]] = {}
    missing_stop = 0
    try:
        cur = ds.execute(
            "SELECT decision_id, reason, gross_pnl_pct FROM outcomes "
            "WHERE status='mark_derived' AND gross_pnl_pct IS NOT NULL "
            "AND reason IN ('dynamic_stop_loss', 'dynamic_trailing_stop')")
        for row in cur:
            sl = stop_by_decision.get(row["decision_id"])
            if sl is None:
                missing_stop += 1
                continue
            overshoot = (-sl * 100.0) - float(row["gross_pnl_pct"])
            per_reason.setdefault(row["reason"], []).append(overshoot)
    finally:
        ds.close()

    out: Dict[str, Any] = {
        "definition": "overshoot_pts = (-stop_loss_pct*100) - "
                      "gross_pnl_pct; positive = realized worse than "
                      "the configured stop floor (points of premium)",
        "missing_stop_config": missing_stop,
        "provenance": "paper_marks",
        "note": "hard stops are evaluated on the scan cadence, so the "
                "mark is typically past the floor when the stop fires; "
                "challenger EV must apply these quantiles instead of "
                "assuming the floor is achievable.",
    }
    for reason in sorted(per_reason):
        vals = per_reason[reason]
        worse = sum(1 for v in vals if v > 0)
        out[reason] = {
            "n": len(vals),
            "frac_worse_than_floor": worse / len(vals),
            "overshoot_quantiles_pts": quantiles(vals),
        }
    return out


# --------------------------------------------------------------------------- #
# orchestration
# --------------------------------------------------------------------------- #

def run_execution_calibration(execution_log: str,
                              history_path: str,
                              dataset_path: Optional[str] = None,
                              ledger_path: Optional[str] = None,
                              min_group: int = MIN_GROUP,
                              n_boot: int = 2000,
                              seed: int = 42) -> Dict[str, Any]:
    """Build the full execution-calibration artifact (deterministic)."""
    records, load_report = load_execution_log(execution_log)
    pairs, link_report = link_pairs(records["execution_estimate"],
                                    records["execution_realization"])
    history, hist_report = load_history_trades(history_path)

    result: Dict[str, Any] = {
        "generated_for": "oracle4_phase6_execution_calibration",
        "provenance_note": PROVENANCE_NOTE,
        "inputs": {
            "execution_log": execution_log,
            "history": history_path,
            "dataset": dataset_path,
            "ledger": ledger_path,
        },
        "load_report": load_report,
        "link_report": link_report,
        "history_report": hist_report,
        "fill_probability": fill_probability_report(
            records, pairs, history, min_group=min_group),
        "slippage": slippage_report(pairs, history),
        "time_to_fill": time_to_fill_report(pairs),
        "ev_calibration": ev_calibration_report(
            pairs, n_boot=n_boot, seed=seed),
    }
    if dataset_path and ledger_path:
        result["stop_overshoot"] = stop_overshoot_report(
            dataset_path, ledger_path)
    else:
        result["stop_overshoot"] = {
            "status": "SKIPPED",
            "note": "dataset/ledger not supplied; overshoot calibration "
                    "unavailable in this run.",
        }
    result["fingerprint"] = metrics_fingerprint(result)
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Oracle 4.0 execution calibration (research-only; "
                    "no broker access, no production imports)")
    p.add_argument("--execution-log", required=True)
    p.add_argument("--history", required=True)
    p.add_argument("--dataset", default=None)
    p.add_argument("--ledger", default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--min-group", type=int, default=MIN_GROUP)
    p.add_argument("--n-boot", type=int, default=2000)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args(argv)

    result = run_execution_calibration(
        execution_log=args.execution_log,
        history_path=args.history,
        dataset_path=args.dataset,
        ledger_path=args.ledger,
        min_group=args.min_group,
        n_boot=args.n_boot,
        seed=args.seed,
    )
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=1, sort_keys=True, default=str)
    ev = result["ev_calibration"]
    print(f"execution calibration written to {args.out}")
    print(f"  linked pairs: {result['link_report']['pairs_linked']}  "
          f"ev slope: {ev.get('ols_slope_realized_on_predicted')}  "
          f"mean err: {ev.get('mean_error_realized_minus_predicted')}")
    print(f"  fingerprint: {result['fingerprint'][:16]}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
