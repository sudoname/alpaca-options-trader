"""
Oracle Execution — Upgrade 2D: execution calibration telemetry.

Upgrades 2A–2C produce, for every candidate/trade, a *prediction* about
execution: an expected entry/exit price, a fill probability, a fill delay, and
the theoretical/executable EV split. This module closes the loop — it records
those predictions to an append-only JSONL ledger and, once a trade actually
fills and closes, records what REALLY happened, so the fill/slippage/EV models
can be CALIBRATED from real fills (backtest -> paper -> live).

It answers the questions the spec asks of a mature execution stack:
  * Did executable_EV predict realized_EV?      -> model_capture_ratio
  * How much edge survives model -> reality?     -> realized_capture_ratio
  * Is the fill model's slippage biased?         -> mean signed slippage error
  * Is fill_probability well-calibrated?         -> predicted p_fill vs actual
    fill rate, bucketed
  * How often would −executable_EV have (correctly) rejected a loser?

Mirrors the append-only / fold-by-id / fail-open idiom of
``oracle_prob_recorder`` and reuses ``executable_ev.capture_ratios`` for the
degradation ladder. STRICTLY telemetry: it never opens, sizes, gates or alters
a trade, and every entry point is fail-open (returns None / an empty report
rather than raising).
"""

from __future__ import annotations

import json
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oracle.execution.executable_ev import capture_ratios

LOG_TAG = "[ORACLE_EXEC_CALIBRATION]"
JSONL_FILE_DEFAULT = "execution_calibration.jsonl"

RECORD_TYPE_ESTIMATE = "execution_estimate"
RECORD_TYPE_REALIZATION = "execution_realization"


# --------------------------------------------------------------------------- #
# Helpers (pure)  — local copies so this module has no hidden coupling
# --------------------------------------------------------------------------- #
def _now(now: Optional[datetime]) -> datetime:
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _to_float(v: Any) -> Optional[float]:
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _mean(xs: List[float]) -> Optional[float]:
    return round(sum(xs) / len(xs), 6) if xs else None


def _jsonl_path(jsonl_path: Optional[str] = None) -> str:
    if jsonl_path:
        return jsonl_path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(
            "EXECUTION_CALIBRATION_JSONL", JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


# --------------------------------------------------------------------------- #
# Record — estimate (decision time) and realization (close time)
# --------------------------------------------------------------------------- #
def record_execution_estimate(fields: dict, *, jsonl_path: Optional[str] = None,
                              now: Optional[datetime] = None) -> Optional[str]:
    """Append ONE frozen execution-estimate line at decision time. Accepts an
    ``ExecutableEV`` (or its ``.to_dict()``) plus a ``trade_id``/``symbol``.
    Returns the trade_id or None (fail-open)."""
    try:
        f = dict(fields or {})
        # allow passing an ExecutableEV dataclass directly
        if hasattr(fields, "to_dict") and not isinstance(fields, dict):
            f = fields.to_dict()  # type: ignore[assignment]
        symbol = str(f.get("symbol") or "").upper()
        tid = f.get("trade_id") or uuid.uuid4().hex[:12]
        rec = {
            "record_type": RECORD_TYPE_ESTIMATE,
            "trade_id": tid,
            "symbol": symbol,
            "recorded_at": _now(now).isoformat(),
            "strategy_mode": f.get("strategy_mode"),
            "qty": f.get("qty"),
            "mid_price": _to_float(f.get("mid_price")),
            "expected_entry_price": _to_float(f.get("expected_entry_price")),
            "expected_exit_price": _to_float(f.get("expected_exit_price")),
            "fill_probability": _to_float(f.get("fill_probability")),
            "expected_fill_delay": _to_float(f.get("expected_fill_delay")),
            "spread_cost": _to_float(f.get("spread_cost")),
            "slippage_estimate": _to_float(f.get("slippage_estimate")),
            "execution_cost": _to_float(f.get("execution_cost")),
            "theoretical_EV": _to_float(f.get("theoretical_EV")),
            "executable_EV": _to_float(f.get("executable_EV")),
            "execution_risk": _to_float(f.get("execution_risk")),
            "shadow_would_reject": bool(f.get("shadow_would_reject", False)),
            "resolution_status": None,
        }
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return tid
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} estimate ignored: {exc}")
        return None


def record_execution_realization(trade_id: str, fields: dict, *,
                                 jsonl_path: Optional[str] = None,
                                 now: Optional[datetime] = None
                                 ) -> Optional[str]:
    """Append ONE realization line once the trade has actually filled/closed.
    ``fields`` carries actual entry/exit fill prices, actual fill delay, whether
    it filled, and realized_EV (from ``executable_ev.compute_realized_ev``)."""
    try:
        if not trade_id:
            return None
        f = dict(fields or {})
        rec = {
            "record_type": RECORD_TYPE_REALIZATION,
            "trade_id": trade_id,
            "resolved_at": _now(now).isoformat(),
            "filled": bool(f.get("filled", True)),
            "actual_entry_price": _to_float(f.get("actual_entry_price")),
            "actual_exit_price": _to_float(f.get("actual_exit_price")),
            "actual_fill_delay": _to_float(f.get("actual_fill_delay")),
            "realized_EV": _to_float(f.get("realized_EV")),
            "resolution_status": "resolved",
        }
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return trade_id
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} realization ignored: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Load (fold by trade_id — last non-null wins)
# --------------------------------------------------------------------------- #
def load_records(jsonl_path: Optional[str] = None) -> List[dict]:
    path = _jsonl_path(jsonl_path)
    folded: Dict[str, dict] = {}
    order: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                tid = rec.get("trade_id")
                if not tid:
                    continue
                if tid not in folded:
                    folded[tid] = {}
                    order.append(tid)
                merged = folded[tid]
                for k, v in rec.items():
                    if v is not None or k not in merged:
                        merged[k] = v
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"{LOG_TAG} load ignored: {exc}")
        return []
    return [folded[t] for t in order]


# --------------------------------------------------------------------------- #
# Aggregate calibration (pure)
# --------------------------------------------------------------------------- #
def compute_calibration(records: List[dict]) -> dict:
    """Fold folded records into calibration statistics. Pure; never raises."""
    recs = records or []
    n_est = sum(1 for r in recs if r.get("theoretical_EV") is not None
                or r.get("executable_EV") is not None)
    resolved = [r for r in recs if r.get("resolution_status") == "resolved"]
    filled = [r for r in resolved if r.get("filled")]

    # EV degradation ladder (means over resolved+filled trades).
    exec_caps, real_caps, model_caps = [], [], []
    theo_vals, exec_vals, real_vals = [], [], []
    slip_errs, delay_errs = [], []
    for r in filled:
        theo = _to_float(r.get("theoretical_EV"))
        exe = _to_float(r.get("executable_EV"))
        rea = _to_float(r.get("realized_EV"))
        if theo is not None:
            theo_vals.append(theo)
        if exe is not None:
            exec_vals.append(exe)
        if rea is not None:
            real_vals.append(rea)
        caps = capture_ratios(theo, exe, rea)
        if caps["execution_capture_ratio"] is not None:
            exec_caps.append(caps["execution_capture_ratio"])
        if caps["realized_capture_ratio"] is not None:
            real_caps.append(caps["realized_capture_ratio"])
        if caps["model_capture_ratio"] is not None:
            model_caps.append(caps["model_capture_ratio"])
        # signed slippage error: actual entry - expected entry (positive = we
        # predicted a better price than we got, i.e. under-estimated slippage).
        exp_e = _to_float(r.get("expected_entry_price"))
        act_e = _to_float(r.get("actual_entry_price"))
        if exp_e is not None and act_e is not None:
            slip_errs.append(round(act_e - exp_e, 6))
        exp_d = _to_float(r.get("expected_fill_delay"))
        act_d = _to_float(r.get("actual_fill_delay"))
        if exp_d is not None and act_d is not None:
            delay_errs.append(round(act_d - exp_d, 6))

    # fill-probability calibration: mean predicted p_fill vs actual fill rate.
    pfills = [_to_float(r.get("fill_probability")) for r in resolved
              if _to_float(r.get("fill_probability")) is not None]
    actual_fill_rate = round(len(filled) / len(resolved), 6) if resolved else None
    predicted_fill_rate = _mean([p for p in pfills if p is not None])

    # would-be rejections: trades flagged −executable / shadow_would_reject and
    # how they turned out (a correct rejection is one whose realized_EV < 0).
    would_reject = [r for r in resolved
                    if r.get("shadow_would_reject")
                    or (_to_float(r.get("executable_EV")) is not None
                        and _to_float(r.get("executable_EV")) <= 0)]
    correct_rejects = sum(1 for r in would_reject
                          if _to_float(r.get("realized_EV")) is not None
                          and _to_float(r.get("realized_EV")) <= 0)

    return {
        "n_estimates": n_est,
        "n_resolved": len(resolved),
        "n_filled": len(filled),
        "mean_theoretical_EV": _mean(theo_vals),
        "mean_executable_EV": _mean(exec_vals),
        "mean_realized_EV": _mean(real_vals),
        "execution_capture_ratio": _mean(exec_caps),
        "realized_capture_ratio": _mean(real_caps),
        "model_capture_ratio": _mean(model_caps),
        "mean_slippage_error": _mean(slip_errs),
        "mean_delay_error": _mean(delay_errs),
        "predicted_fill_rate": predicted_fill_rate,
        "actual_fill_rate": actual_fill_rate,
        "fill_rate_bias": (round(predicted_fill_rate - actual_fill_rate, 6)
                           if predicted_fill_rate is not None
                           and actual_fill_rate is not None else None),
        "n_would_reject": len(would_reject),
        "n_correct_rejects": correct_rejects,
        "reject_precision": (round(correct_rejects / len(would_reject), 6)
                             if would_reject else None),
    }


def format_calibration_report(stats: dict) -> str:
    """Markdown execution-calibration report (pure formatting)."""
    s = stats or {}
    if not s.get("n_resolved"):
        return ("📊 *EXECUTION CALIBRATION*\n"
                f"Estimates recorded: {s.get('n_estimates', 0)}\n"
                "_No resolved trades yet — nothing to calibrate._")

    def _fmt(v, pct=False, sign=False):
        if v is None:
            return "n/a"
        if pct:
            return f"{v * 100:.1f}%"
        return f"{v:+.4f}" if sign else f"{v:.4f}"

    lines = [
        "📊 *EXECUTION CALIBRATION*",
        "",
        f"Estimates: {s.get('n_estimates', 0)}  |  "
        f"Resolved: {s.get('n_resolved', 0)}  |  Filled: {s.get('n_filled', 0)}",
        "",
        "*EV degradation ladder (means)*",
        f"  theoretical: {_fmt(s.get('mean_theoretical_EV'), sign=True)}",
        f"  executable : {_fmt(s.get('mean_executable_EV'), sign=True)}",
        f"  realized   : {_fmt(s.get('mean_realized_EV'), sign=True)}",
        "",
        f"execution capture (exec/theo): {_fmt(s.get('execution_capture_ratio'))}",
        f"realized  capture (real/theo): {_fmt(s.get('realized_capture_ratio'))}",
        f"model     capture (real/exec): {_fmt(s.get('model_capture_ratio'))}",
        "",
        "*Fill model calibration*",
        f"  slippage error (actual-expected): {_fmt(s.get('mean_slippage_error'), sign=True)}",
        f"  delay error (s): {_fmt(s.get('mean_delay_error'), sign=True)}",
        f"  predicted fill rate: {_fmt(s.get('predicted_fill_rate'), pct=True)}",
        f"  actual fill rate   : {_fmt(s.get('actual_fill_rate'), pct=True)}",
        f"  fill-rate bias     : {_fmt(s.get('fill_rate_bias'), sign=True)}",
        "",
        "*Would-be rejections (negative executable EV)*",
        f"  flagged: {s.get('n_would_reject', 0)}  |  "
        f"correct: {s.get('n_correct_rejects', 0)}  |  "
        f"precision: {_fmt(s.get('reject_precision'), pct=True)}",
        "",
        "_(Telemetry only — no orders placed or altered.)_",
    ]
    return "\n".join(lines)


def generate_calibration_report_text(jsonl_path: Optional[str] = None) -> str:
    """Convenience: load -> compute -> format. Fail-open to an empty report."""
    try:
        return format_calibration_report(
            compute_calibration(load_records(jsonl_path)))
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} report ignored: {exc}")
        return format_calibration_report({})


# --------------------------------------------------------------------------- #
# Self-test (offline; temp ledger; no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import os
    import tempfile

    ok = True
    tmp = tempfile.mkdtemp()
    path = os.path.join(tmp, "exec_cal.jsonl")

    # Trade A: model said +executable, it filled and realized a smaller win.
    ta = record_execution_estimate({
        "trade_id": "A", "symbol": "AAA", "qty": 1, "mid_price": 1.01,
        "expected_entry_price": 1.03, "expected_exit_price": 0.99,
        "fill_probability": 0.95, "expected_fill_delay": 1.0,
        "spread_cost": 3.0, "slippage_estimate": 0.01, "execution_cost": 3.04,
        "theoretical_EV": 15.0, "executable_EV": 11.0, "execution_risk": 0.27,
    }, jsonl_path=path)
    if ta != "A":
        print("FAIL: estimate id", ta); ok = False

    # Trade B: model said −executable (would reject); it indeed lost.
    record_execution_estimate({
        "trade_id": "B", "symbol": "BBB", "qty": 1, "mid_price": 1.20,
        "expected_entry_price": 1.35, "expected_exit_price": 1.05,
        "fill_probability": 0.70, "theoretical_EV": 3.0,
        "executable_EV": -6.0, "execution_risk": 3.0,
        "shadow_would_reject": True,
    }, jsonl_path=path)

    # Only estimates so far -> report says nothing resolved.
    mid = compute_calibration(load_records(path))
    if mid["n_estimates"] != 2 or mid["n_resolved"] != 0:
        print("FAIL: pre-resolution counts", mid); ok = False

    # Resolve A (filled, realized +9 -> slight under-estimate of edge).
    record_execution_realization("A", {
        "filled": True, "actual_entry_price": 1.05, "actual_exit_price": 1.20,
        "actual_fill_delay": 1.5, "realized_EV": 9.0}, jsonl_path=path)
    # Resolve B (filled, realized -5 -> a correct would-be rejection).
    record_execution_realization("B", {
        "filled": True, "actual_entry_price": 1.36, "actual_exit_price": 1.10,
        "actual_fill_delay": 4.0, "realized_EV": -5.0}, jsonl_path=path)

    stats = compute_calibration(load_records(path))
    if stats["n_resolved"] != 2 or stats["n_filled"] != 2:
        print("FAIL: resolved counts", stats); ok = False
    # capture ratios present and ordered sensibly for A's ladder.
    if stats["execution_capture_ratio"] is None or \
            stats["realized_capture_ratio"] is None or \
            stats["model_capture_ratio"] is None:
        print("FAIL: capture ratios missing", stats); ok = False
    # slippage error is positive (we under-estimated the entry price paid).
    if stats["mean_slippage_error"] is None or stats["mean_slippage_error"] <= 0:
        print("FAIL: slippage error should be positive", stats); ok = False
    # delay error positive (fills slower than modeled).
    if stats["mean_delay_error"] is None or stats["mean_delay_error"] <= 0:
        print("FAIL: delay error should be positive", stats); ok = False
    # both would-be rejections handled; B was a correct reject.
    if stats["n_would_reject"] < 1 or stats["n_correct_rejects"] < 1:
        print("FAIL: rejection accounting", stats); ok = False
    # fill-rate bias computable (predicted vs actual).
    if stats["actual_fill_rate"] != 1.0:
        print("FAIL: both filled -> actual fill rate 1.0", stats); ok = False

    # Report renders without error and includes the ladder.
    txt = format_calibration_report(stats)
    if "EV degradation ladder" not in txt or "EXECUTION CALIBRATION" not in txt:
        print("FAIL: report text", txt[:120]); ok = False

    # Fail-open: bad inputs / missing file never raise.
    if record_execution_realization("", {}, jsonl_path=path) is not None:
        print("FAIL: empty trade_id should be None"); ok = False
    if load_records(os.path.join(tmp, "nope.jsonl")) != []:
        print("FAIL: missing file should be []"); ok = False
    if "No resolved trades" not in format_calibration_report({"n_estimates": 0}):
        print("FAIL: empty report"); ok = False

    print("execution.calibration self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
