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
RECORD_TYPE_INTENT = "order_intent"
RECORD_TYPE_OUTCOME = "order_outcome"


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
            "net_ev_after_costs": _to_float(f.get("net_ev_after_costs")),
            "fees": _to_float(f.get("fees")),
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

    # Opportunity-level P&L: a resolved nonfill is a $0 outcome, not a skip.
    # (Filled-only means answer "how did fills do"; this answers "how did the
    # opportunities we tried to take do", which is what entry-EV calibrates.)
    opp_real = []
    for r in resolved:
        if r.get("filled"):
            v = _to_float(r.get("realized_EV"))
            if v is not None:
                opp_real.append(v)
        else:
            opp_real.append(0.0)   # nonfill = zero P&L at opportunity level

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
        "opportunity_mean_realized_EV": _mean(opp_real),
        "n_opportunities": len(resolved),
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
        f"  realized/opportunity (nonfills = $0 over "
        f"{s.get('n_opportunities', 0)}): "
        f"{_fmt(s.get('opportunity_mean_realized_EV'), sign=True)}",
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
# Order intent/outcome (Phase 2: pre-submission persistence + broker join)
#
# The estimate/realization records above calibrate the EV *model*. These two
# record types calibrate the *orders themselves*:
#   * ``order_intent``  — persisted BEFORE submission (or decision not to
#     submit): the observed quote + source/timestamps, the exact order we
#     meant to send (limit, qty, expiry), and the predicted fill probability.
#     No-trade candidates are intents with decision="no_trade" and a reason.
#   * ``order_outcome`` — the broker's terminal verdict, joined by the stable
#     ``client_order_id``: terminal state, requested vs filled qty, average
#     fill price, fees, exit price / realized P&L when reconciled fills exist,
#     and an explicit ``missing_evidence`` list for anything we could NOT
#     verify (never silently fabricated).
#
# ``fit_fill_and_slippage`` then fits fill-rate and slippage calibration from
# SUBMITTED intents with terminal outcomes — nonfills included — bucketed with
# a pooled fallback so small samples never masquerade as signal.
# --------------------------------------------------------------------------- #
def record_order_intent(fields: dict, *, jsonl_path: Optional[str] = None,
                        now: Optional[datetime] = None) -> Optional[str]:
    """Persist ONE order intent BEFORE submission. Returns the join key
    (client_order_id, else trade_id, else a fresh uuid) or None (fail-open).

    Pass ``decision="no_trade"`` (+ ``reason``) for candidates we chose not to
    submit — they are evidence too (the denominator of opportunity accounting).
    """
    try:
        f = dict(fields or {})
        coid = f.get("client_order_id")
        key = coid or f.get("trade_id") or uuid.uuid4().hex[:12]
        rec = {
            "record_type": RECORD_TYPE_INTENT,
            "client_order_id": coid,
            "trade_id": f.get("trade_id") or key,
            "recorded_at": _now(now).isoformat(),
            "symbol": str(f.get("symbol") or "").upper(),
            "decision": str(f.get("decision") or "submit"),
            "reason": f.get("reason"),
            "strategy_mode": f.get("strategy_mode"),
            # the exact order we intend to send
            "side": f.get("side"),
            "order_type": f.get("order_type"),
            "requested_qty": _to_float(f.get("requested_qty", f.get("qty"))),
            "limit_price": _to_float(f.get("limit_price")),
            "expiry_s": _to_float(f.get("expiry_s")),
            # the quote we saw when we decided (provenance included)
            "bid": _to_float(f.get("bid")),
            "ask": _to_float(f.get("ask")),
            "mid": _to_float(f.get("mid")),
            "quote_source": f.get("quote_source"),
            "quote_ts": f.get("quote_ts"),
            "quote_age_s": _to_float(f.get("quote_age_s")),
            # model predictions to calibrate against
            "fill_probability": _to_float(f.get("fill_probability")),
            "executable_EV": _to_float(f.get("executable_EV")),
            "net_ev_after_costs": _to_float(f.get("net_ev_after_costs")),
            "outcome_status": None,
        }
        if rec["mid"] is None and rec["bid"] is not None \
                and rec["ask"] is not None:
            rec["mid"] = round((rec["bid"] + rec["ask"]) / 2.0, 6)
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return key
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} intent ignored: {exc}")
        return None


#: evidence keys an outcome can be missing; used by tests and report text.
EVIDENCE_FILL_PRICE = "avg_fill_price"
EVIDENCE_ORDER_ID = "order_id"
EVIDENCE_FEES = "fees"
EVIDENCE_EXIT = "exit_fills"


def record_order_outcome(client_order_id: str, fields: dict, *,
                         jsonl_path: Optional[str] = None,
                         now: Optional[datetime] = None) -> Optional[str]:
    """Persist ONE terminal broker outcome, joined to its intent by
    ``client_order_id``. Marks missing evidence explicitly rather than
    guessing. Returns the client_order_id or None (fail-open)."""
    try:
        if not client_order_id:
            return None
        f = dict(fields or {})
        filled_qty = _to_float(f.get("filled_qty")) or 0.0
        avg_price = _to_float(f.get("avg_fill_price"))
        fees = _to_float(f.get("fees"))
        exit_price = _to_float(f.get("exit_price"))
        missing = list(f.get("missing_evidence") or [])
        if filled_qty > 0 and avg_price is None \
                and EVIDENCE_FILL_PRICE not in missing:
            missing.append(EVIDENCE_FILL_PRICE)
        if not f.get("order_id") and EVIDENCE_ORDER_ID not in missing:
            missing.append(EVIDENCE_ORDER_ID)
        if filled_qty > 0 and fees is None and EVIDENCE_FEES not in missing:
            missing.append(EVIDENCE_FEES)
        if filled_qty > 0 and exit_price is None \
                and _to_float(f.get("realized_pnl")) is None \
                and EVIDENCE_EXIT not in missing:
            missing.append(EVIDENCE_EXIT)
        rec = {
            "record_type": RECORD_TYPE_OUTCOME,
            "client_order_id": client_order_id,
            "order_id": f.get("order_id"),
            "resolved_at": _now(now).isoformat(),
            "terminal_state": f.get("terminal_state", f.get("state")),
            "requested_qty": _to_float(f.get("requested_qty", f.get("qty"))),
            "filled_qty": filled_qty,
            "avg_fill_price": avg_price,
            "fees": fees,
            "exit_price": exit_price,
            "realized_pnl": _to_float(f.get("realized_pnl")),
            "late_fill": bool(f.get("late_fill", False)),
            "missing_evidence": missing,
            "outcome_status": "terminal",
        }
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return client_order_id
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} outcome ignored: {exc}")
        return None


def record_order_outcome_from_lifecycle(order: dict, *,
                                        fees: Optional[float] = None,
                                        exit_price: Optional[float] = None,
                                        realized_pnl: Optional[float] = None,
                                        jsonl_path: Optional[str] = None,
                                        now: Optional[datetime] = None
                                        ) -> Optional[str]:
    """Adapter: persist a ``ManagedOrder.to_dict()`` (oracle.execution
    .lifecycle) as an order_outcome. Only terminal (or resolved-ambiguous)
    orders are recorded; a working/unknown order returns None untouched."""
    try:
        o = dict(order or {})
        state = str(o.get("state") or "")
        if state in ("working", "unknown", ""):
            return None
        events = o.get("events") or []
        late = any("late_fill" in str(e[1] if len(e) > 1 else e)
                   for e in events)
        return record_order_outcome(
            str(o.get("client_order_id") or ""), {
                "order_id": o.get("order_id"),
                "terminal_state": state,
                "requested_qty": o.get("qty"),
                "filled_qty": o.get("filled_qty"),
                "avg_fill_price": o.get("avg_fill_price"),
                "fees": fees,
                "exit_price": exit_price,
                "realized_pnl": realized_pnl,
                "late_fill": late,
            }, jsonl_path=jsonl_path, now=now)
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} lifecycle outcome ignored: {exc}")
        return None


def load_order_records(jsonl_path: Optional[str] = None) -> List[dict]:
    """Load + join intent/outcome rows by client_order_id (fallback trade_id).
    Last non-null wins per field, like ``load_records``. Fail-open to []."""
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
                if rec.get("record_type") not in (RECORD_TYPE_INTENT,
                                                  RECORD_TYPE_OUTCOME):
                    continue
                key = rec.get("client_order_id") or rec.get("trade_id")
                if not key:
                    continue
                if key not in folded:
                    folded[key] = {}
                    order.append(key)
                merged = folded[key]
                for k, v in rec.items():
                    if v is not None or k not in merged:
                        merged[k] = v
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"{LOG_TAG} order load ignored: {exc}")
        return []
    return [folded[k] for k in order]


#: fill-probability buckets (inclusive upper bound) and spread buckets ($).
_PFILL_BUCKETS = ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 1.0))
_SPREAD_BUCKETS = ((0.0, 0.02), (0.02, 0.05), (0.05, 0.10),
                   (0.10, float("inf")))


def fit_fill_and_slippage(records: List[dict], *, min_n: int = 5) -> dict:
    """Fit fill-rate and slippage calibration from joined order records.

    Inputs are ``load_order_records`` rows. Only SUBMITTED intents with a
    terminal outcome participate (decision == submit AND outcome_status ==
    terminal); nonfills count in every denominator. No-trade intents are
    reported as a count only. Buckets with n < min_n are flagged low_sample
    and the pooled row is the authoritative fallback. Pure; never raises.

    Slippage is per-share vs the intent-time mid, signed so positive = worse
    for us: buy -> fill − mid; sell -> mid − fill.
    """
    recs = records or []
    submitted = []
    n_no_trade = 0
    for r in recs:
        if str(r.get("decision") or "submit") == "no_trade":
            n_no_trade += 1
            continue
        if r.get("record_type") not in (RECORD_TYPE_INTENT,
                                        RECORD_TYPE_OUTCOME):
            continue
        if r.get("outcome_status") == "terminal":
            submitted.append(r)
    n_unresolved = sum(
        1 for r in recs
        if str(r.get("decision") or "submit") != "no_trade"
        and r.get("record_type") in (RECORD_TYPE_INTENT, RECORD_TYPE_OUTCOME)
        and r.get("outcome_status") != "terminal")

    def _row(rows: List[dict]) -> dict:
        n = len(rows)
        any_fill = [1.0 if (_to_float(r.get("filled_qty")) or 0.0) > 0
                    else 0.0 for r in rows]
        fracs = []
        for r in rows:
            req = _to_float(r.get("requested_qty"))
            fq = _to_float(r.get("filled_qty")) or 0.0
            if req and req > 0:
                fracs.append(min(fq / req, 1.0))
        preds = [p for p in (_to_float(r.get("fill_probability"))
                             for r in rows) if p is not None]
        slips = []
        for r in rows:
            fq = _to_float(r.get("filled_qty")) or 0.0
            px = _to_float(r.get("avg_fill_price"))
            mid = _to_float(r.get("mid"))
            if fq > 0 and px is not None and mid is not None:
                sgn = -1.0 if str(r.get("side") or "buy").lower() == "sell" \
                    else 1.0
                slips.append(round(sgn * (px - mid), 6))
        pred = _mean(preds)
        actual = _mean(any_fill)
        return {
            "n": n,
            "predicted_fill_rate": pred,
            "actual_fill_rate": actual,
            "fill_rate_bias": (round(pred - actual, 6)
                               if pred is not None and actual is not None
                               else None),
            "mean_fill_fraction": _mean(fracs),
            "mean_slippage": _mean(slips),
            "n_slippage": len(slips),
            "low_sample": n < min_n,
        }

    by_pfill = []
    for lo, hi in _PFILL_BUCKETS:
        rows = [r for r in submitted
                if (_to_float(r.get("fill_probability")) is not None
                    and lo <= _to_float(r.get("fill_probability")) <= hi
                    and (_to_float(r.get("fill_probability")) > lo
                         or lo == 0.0))]
        by_pfill.append({"bucket": f"({lo:.2f}, {hi:.2f}]", **_row(rows)})
    by_spread = []
    for lo, hi in _SPREAD_BUCKETS:
        rows = []
        for r in submitted:
            b, a = _to_float(r.get("bid")), _to_float(r.get("ask"))
            if b is None or a is None:
                continue
            sp = a - b
            if lo <= sp <= hi and (sp > lo or lo == 0.0):
                rows.append(r)
        label = f"({lo:.2f}, {'inf' if hi == float('inf') else f'{hi:.2f}'}]"
        by_spread.append({"bucket": label, **_row(rows)})

    n_missing = sum(1 for r in submitted if r.get("missing_evidence"))
    return {
        "n_intents": len([r for r in recs
                          if r.get("record_type") in (RECORD_TYPE_INTENT,
                                                      RECORD_TYPE_OUTCOME)]),
        "n_no_trade": n_no_trade,
        "n_submitted_terminal": len(submitted),
        "n_unresolved": n_unresolved,
        "n_with_missing_evidence": n_missing,
        "pooled": _row(submitted),
        "by_fill_probability": by_pfill,
        "by_spread": by_spread,
        "min_n": min_n,
    }


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

    # Trade C: estimate recorded, order never filled (resolved nonfill).
    record_execution_estimate({
        "trade_id": "C", "symbol": "CCC", "qty": 1, "mid_price": 0.50,
        "expected_entry_price": 0.48, "fill_probability": 0.30,
        "theoretical_EV": 4.0, "executable_EV": 1.0,
        "net_ev_after_costs": 3.3,
    }, jsonl_path=path)

    # Resolve A (filled, realized +9 -> slight under-estimate of edge).
    record_execution_realization("A", {
        "filled": True, "actual_entry_price": 1.05, "actual_exit_price": 1.20,
        "actual_fill_delay": 1.5, "realized_EV": 9.0}, jsonl_path=path)
    # Resolve B (filled, realized -5 -> a correct would-be rejection).
    record_execution_realization("B", {
        "filled": True, "actual_entry_price": 1.36, "actual_exit_price": 1.10,
        "actual_fill_delay": 4.0, "realized_EV": -5.0}, jsonl_path=path)
    # Resolve C (never filled): a $0 opportunity, not a skipped row.
    record_execution_realization("C", {"filled": False}, jsonl_path=path)

    stats = compute_calibration(load_records(path))
    if stats["n_resolved"] != 3 or stats["n_filled"] != 2:
        print("FAIL: resolved counts", stats); ok = False
    # Opportunity-level mean counts C's nonfill as $0: (9 - 5 + 0) / 3.
    if stats["n_opportunities"] != 3 or \
            stats["opportunity_mean_realized_EV"] is None or \
            abs(stats["opportunity_mean_realized_EV"] - (4.0 / 3.0)) > 1e-6:
        print("FAIL: opportunity-level EV", stats); ok = False
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
    # fill-rate bias computable (predicted vs actual): 2 fills / 3 resolved.
    if stats["actual_fill_rate"] is None or \
            abs(stats["actual_fill_rate"] - (2.0 / 3.0)) > 1e-6:
        print("FAIL: 2 of 3 filled -> actual fill rate 2/3", stats); ok = False

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

    # ---- order intent/outcome + fit (Phase 2) ---------------------------- #
    opath = os.path.join(tmp, "orders.jsonl")
    # Submitted limit buy: quote 1.00/1.02, p_fill 0.8, fills 1@1.02 -> slip
    # vs mid = +0.01.
    k1 = record_order_intent({
        "client_order_id": "c1", "symbol": "OPT", "side": "buy",
        "order_type": "limit", "requested_qty": 1, "limit_price": 1.02,
        "bid": 1.00, "ask": 1.02, "quote_source": "sim", "quote_ts": "t0",
        "quote_age_s": 0.5, "fill_probability": 0.8}, jsonl_path=opath)
    if k1 != "c1":
        print("FAIL: intent key", k1); ok = False
    # No-trade candidate persisted too.
    record_order_intent({
        "trade_id": "nt1", "symbol": "OPT", "decision": "no_trade",
        "reason": "stale_quote"}, jsonl_path=opath)
    # Nonfill: submitted, expired with zero fill.
    record_order_intent({
        "client_order_id": "c2", "symbol": "OPT", "side": "buy",
        "order_type": "limit", "requested_qty": 2, "limit_price": 0.99,
        "bid": 1.00, "ask": 1.02, "fill_probability": 0.8},
        jsonl_path=opath)
    # Outcomes via the lifecycle adapter (ManagedOrder.to_dict() shape).
    record_order_outcome_from_lifecycle({
        "client_order_id": "c1", "order_id": "o1", "state": "filled",
        "qty": 1, "filled_qty": 1.0, "avg_fill_price": 1.02,
        "events": []}, fees=0.02, exit_price=1.20, jsonl_path=opath)
    record_order_outcome_from_lifecycle({
        "client_order_id": "c2", "order_id": "o2", "state": "expired",
        "qty": 2, "filled_qty": 0.0, "avg_fill_price": None,
        "events": []}, jsonl_path=opath)
    # A working order must NOT be recorded as terminal.
    if record_order_outcome_from_lifecycle(
            {"client_order_id": "c3", "state": "working"},
            jsonl_path=opath) is not None:
        print("FAIL: working order recorded"); ok = False
    joined = load_order_records(opath)
    if len(joined) != 3:
        print("FAIL: joined rows", len(joined)); ok = False
    c1 = next(r for r in joined if r.get("client_order_id") == "c1")
    if c1.get("terminal_state") != "filled" or c1.get("bid") != 1.00 \
            or c1.get("missing_evidence"):
        print("FAIL: c1 join", c1); ok = False
    fit = fit_fill_and_slippage(joined, min_n=1)
    pooled = fit["pooled"]
    # 2 submitted terminal, 1 filled -> actual 0.5 vs predicted 0.8.
    if fit["n_submitted_terminal"] != 2 or fit["n_no_trade"] != 1 \
            or pooled["actual_fill_rate"] != 0.5 \
            or abs(pooled["fill_rate_bias"] - 0.3) > 1e-9 \
            or abs(pooled["mean_slippage"] - 0.01) > 1e-9:
        print("FAIL: fit", fit); ok = False

    print("execution.calibration self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
