"""Oracle 4.0.1 Stage D -- P&L reconciliation (research-only, read-only).

Builds a fills-first reconciliation of the research ledger event stream
against the legacy mark-based trade history and the fills-based realized
P&L log.  Rules (per the Oracle 4.0.1 master prompt, Stage D):

  * keyed by stable decision/order IDs and fill event keys;
  * duplicate closes are deduplicated (keep-first) and *counted*;
  * entry/exit prices are fills-weighted; matched quantity, contract
    multiplier and fees are applied exactly once;
  * fees come from broker-reported event fields only -- a missing fee is
    *unknown*, never inferred;
  * cash flows are reconciled against realized P&L; sale proceeds are
    never treated as profit;
  * unmatched positions and unallocated fees are kept separately;
  * paper vs live provenance is carried from the event ``source`` column.

No production module imports this file.  The ledger is opened read-only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .ingest import LedgerEvent, connect, load_events

__all__ = [
    "ReconcileConfig",
    "reconcile_positions",
    "match_history",
    "marks_vs_fills",
    "capture_groups",
    "run_reconciliation",
]

UTC = timezone.utc

# Legacy headline estimate explicitly invalidated by the master prompt.
INVALIDATED_ESTIMATES = [
    {
        "estimate": "legacy_mean_ev_usd",
        "value": 51.9,
        "status": "INVALIDATED",
        "reason": (
            "Mark-derived, in-sample, survivorship-biased mean EV from the "
            "Oracle 4.0 report. Must not be reused as a prediction of "
            "future expectancy (Oracle 4.0.1 Stage D)."
        ),
    }
]

FEES_POLICY = (
    "Fees are counted exactly once from broker-reported event fields. A "
    "missing fee is recorded as unknown and left unallocated; it is never "
    "inferred or substituted with an estimate."
)


class ReconcileConfig:
    """Pre-registered reconciliation constants."""

    def __init__(
        self,
        contract_multiplier: float = 100.0,
        marks_match_tolerance_s: float = 180.0,
        exact_delta_tolerance: float = 0.005,
        min_matched_for_ci: int = 10,
    ) -> None:
        if contract_multiplier <= 0:
            raise ValueError("contract_multiplier must be > 0")
        if marks_match_tolerance_s < 0:
            raise ValueError("marks_match_tolerance_s must be >= 0")
        self.contract_multiplier = float(contract_multiplier)
        self.marks_match_tolerance_s = float(marks_match_tolerance_s)
        self.exact_delta_tolerance = float(exact_delta_tolerance)
        self.min_matched_for_ci = int(min_matched_for_ci)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "contract_multiplier": self.contract_multiplier,
            "marks_match_tolerance_s": self.marks_match_tolerance_s,
            "exact_delta_tolerance": self.exact_delta_tolerance,
            "min_matched_for_ci": self.min_matched_for_ci,
        }


# --------------------------------------------------------------------------- #
# small stats helpers
# --------------------------------------------------------------------------- #

def _stats(values: Sequence[float], exact_tol: float = 0.005) -> Dict[str, Any]:
    """Mean / sd / normal-approx 95% CI / median / frac-exact of a sample."""
    n = len(values)
    if n == 0:
        return {"n": 0, "mean": None, "sd": None, "ci95": None,
                "median": None, "frac_exact": None}
    mean = sum(values) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in values) / (n - 1)
        sd = math.sqrt(var)
        half = 1.96 * sd / math.sqrt(n)
        ci: Optional[List[float]] = [mean - half, mean + half]
    else:
        sd = None
        ci = None
    ordered = sorted(values)
    mid = n // 2
    median = ordered[mid] if n % 2 else (ordered[mid - 1] + ordered[mid]) / 2.0
    frac_exact = sum(1 for v in values if abs(v) <= exact_tol) / n
    return {"n": n, "mean": mean, "sd": sd, "ci95": ci,
            "median": median, "frac_exact": frac_exact}


def _sha256(path: str) -> Optional[str]:
    try:
        h = hashlib.sha256()
        with open(path, "rb") as f:
            for chunk in iter(lambda: f.read(1 << 20), b""):
                h.update(chunk)
        return h.hexdigest()
    except OSError:
        return None


def _parse_naive_utc(raw: Any) -> Optional[datetime]:
    """Parse ISO timestamps; naive values are server-UTC by convention."""
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
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# position reconciliation from ledger events
# --------------------------------------------------------------------------- #

def _weighted_price(pairs: List[Tuple[float, float]]) -> Optional[float]:
    """Fills-weighted price from (qty, price) pairs."""
    total_q = sum(q for q, _ in pairs)
    if total_q <= 0:
        return None
    return sum(q * p for q, p in pairs) / total_q


def reconcile_positions(
    events_by_decision: Dict[str, List[LedgerEvent]],
    config: Optional[ReconcileConfig] = None,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, int]]:
    """Reconcile each decision's event stream into one position record.

    Events within a decision are assumed sorted by ``(event_at, event_key)``
    (``ingest.load_events`` guarantees this).  Duplicate outcome ("close")
    events are deduplicated keep-first and counted, never merged.
    """
    cfg = config or ReconcileConfig()
    report = {
        "decisions_total": 0,
        "resolved": 0,
        "open_or_unresolved": 0,
        "submitted_unfilled": 0,
        "exit_without_entry": 0,
        "multi_fill_entries": 0,
        "partial_entries": 0,
        "duplicate_outcomes_dropped": 0,
        "fills_missing_qty_or_price": 0,
        "exit_price_missing": 0,
        "fees_reported_events": 0,
        "fees_unknown_events": 0,
        "unallocated_fee_events": 0,
    }
    positions: Dict[str, Dict[str, Any]] = {}

    for decision_id, events in events_by_decision.items():
        report["decisions_total"] += 1
        submitted = [e for e in events if e.event_type == "submitted"]
        fills = [e for e in events if e.event_type == "fill"]
        exits = [e for e in events if e.event_type == "exit"]
        outcomes = [e for e in events if e.event_type == "outcome"]

        # ---- duplicate close guard: keep-first by (event_at, event_key)
        outcome = outcomes[0] if outcomes else None
        if len(outcomes) > 1:
            report["duplicate_outcomes_dropped"] += len(outcomes) - 1

        # ---- entry fills (weighted)
        entry_pairs: List[Tuple[float, float]] = []
        entry_fees_known = 0.0
        entry_fees_unknown = 0
        broker_order_ids: List[str] = []
        sources: List[str] = []
        for e in fills:
            if e.broker_order_id and e.broker_order_id not in broker_order_ids:
                broker_order_ids.append(e.broker_order_id)
            if e.source and e.source not in sources:
                sources.append(e.source)
            if e.qty is None or e.price is None or e.qty <= 0:
                report["fills_missing_qty_or_price"] += 1
                continue
            entry_pairs.append((e.qty, e.price))
            if e.fees is not None:
                entry_fees_known += e.fees
                report["fees_reported_events"] += 1
            else:
                entry_fees_unknown += 1
                report["fees_unknown_events"] += 1
        if len(entry_pairs) > 1:
            report["multi_fill_entries"] += 1

        matched_qty = sum(q for q, _ in entry_pairs)
        entry_price = _weighted_price(entry_pairs)

        # partial fill: filled less than submitted quantity
        submitted_qty = None
        for e in submitted:
            if e.qty is not None:
                submitted_qty = (submitted_qty or 0.0) + e.qty
        if (submitted_qty is not None and matched_qty > 0
                and matched_qty < submitted_qty):
            report["partial_entries"] += 1

        # ---- exit fills (weighted; real ledger often has price=None)
        exit_pairs: List[Tuple[float, float]] = []
        exit_fees_known = 0.0
        exit_fees_unknown = 0
        exit_reason = None
        for e in exits:
            if e.source and e.source not in sources:
                sources.append(e.source)
            if exit_reason is None:
                exit_reason = (e.payload or {}).get("reason")
            if e.price is None or e.qty is None or e.qty <= 0:
                report["exit_price_missing"] += 1
            else:
                exit_pairs.append((e.qty, e.price))
            if e.fees is not None:
                exit_fees_known += e.fees
                report["fees_reported_events"] += 1
            else:
                exit_fees_unknown += 1
                report["fees_unknown_events"] += 1
        exit_qty = sum(q for q, _ in exit_pairs)
        exit_price = _weighted_price(exit_pairs)

        # fees on events that are neither entry nor exit stay unallocated
        unallocated_fees = 0.0
        for e in submitted + ([outcome] if outcome else []):
            if e is not None and e.fees is not None:
                unallocated_fees += e.fees
                report["unallocated_fee_events"] += 1

        fees_fully_known = (entry_fees_unknown == 0 and exit_fees_unknown == 0
                            and (fills or exits))

        # ---- cash-flow identity: pnl = proceeds - cost_basis - fees.
        # Sale proceeds alone are never profit.
        mult = cfg.contract_multiplier
        cost_basis = (entry_price * matched_qty * mult
                      if entry_price is not None else None)
        proceeds = (exit_price * exit_qty * mult
                    if exit_price is not None else None)
        if cost_basis is not None and proceeds is not None:
            realized_cash_pnl: Optional[float] = (
                proceeds - cost_basis - entry_fees_known - exit_fees_known)
            cash_pnl_basis = ("fills" if fees_fully_known
                              else "fills_ex_unknown_fees")
        else:
            realized_cash_pnl = None
            cash_pnl_basis = "unavailable"

        # ---- ledger-claimed outcome (mark-derived unless stated otherwise)
        outcome_payload = (outcome.payload or {}) if outcome else {}
        ledger_gross_pnl_pct = _f(outcome_payload.get("gross_pnl_pct"))
        ledger_net_pnl_dollars = _f(outcome_payload.get("net_pnl_dollars"))

        # ---- status
        if fills and outcome is not None:
            status = "resolved"
            report["resolved"] += 1
        elif fills:
            status = "open_or_unresolved"
            report["open_or_unresolved"] += 1
        elif exits or outcome is not None:
            status = "exit_without_entry"
            report["exit_without_entry"] += 1
        else:
            status = "submitted_unfilled"
            report["submitted_unfilled"] += 1

        positions[decision_id] = {
            "decision_id": decision_id,
            "status": status,
            "broker_order_ids": broker_order_ids,
            "provenance_sources": sources,
            "n_entry_fills": len(entry_pairs),
            "matched_qty": matched_qty,
            "entry_price_weighted": entry_price,
            "submitted_qty": submitted_qty,
            "exit_qty": exit_qty,
            "exit_price_weighted": exit_price,
            "exit_reason": exit_reason,
            "contract_multiplier": mult,
            "multiplier_source": "config_default",
            "fees_entry_known": entry_fees_known,
            "fees_exit_known": exit_fees_known,
            "fees_unknown_events": entry_fees_unknown + exit_fees_unknown,
            "fees_unallocated": unallocated_fees,
            "fees_fully_known": bool(fees_fully_known),
            "cost_basis": cost_basis,
            "proceeds": proceeds,
            "realized_cash_pnl": realized_cash_pnl,
            "cash_pnl_basis": cash_pnl_basis,
            "ledger_gross_pnl_pct": ledger_gross_pnl_pct,
            "ledger_net_pnl_dollars": ledger_net_pnl_dollars,
            "duplicate_outcomes": max(0, len(outcomes) - 1),
        }

    return positions, report


# --------------------------------------------------------------------------- #
# history <-> ledger matching (order_id <-> broker_order_id)
# --------------------------------------------------------------------------- #

def match_history(
    history_trades: Iterable[Dict[str, Any]],
    positions: Dict[str, Dict[str, Any]],
    config: Optional[ReconcileConfig] = None,
) -> Dict[str, Any]:
    """Match legacy history trades to reconciled ledger positions by
    ``order_id`` <-> entry-fill ``broker_order_id``.  Reports coverage and
    pnl_percent deltas (history mark vs ledger-claimed gross_pnl_pct)."""
    cfg = config or ReconcileConfig()
    by_order: Dict[str, Dict[str, Any]] = {}
    for pos in positions.values():
        for oid in pos["broker_order_ids"]:
            by_order.setdefault(oid, pos)

    n_history = 0
    n_with_order_id = 0
    matched: List[Tuple[Dict[str, Any], Dict[str, Any]]] = []
    for trade in history_trades:
        n_history += 1
        oid = trade.get("order_id")
        if not oid:
            continue
        n_with_order_id += 1
        pos = by_order.get(oid)
        if pos is not None:
            matched.append((trade, pos))

    pct_deltas: List[float] = []
    for trade, pos in matched:
        hist_pct = _f(trade.get("pnl_percent"))
        ledger_pct = pos.get("ledger_gross_pnl_pct")
        if hist_pct is not None and ledger_pct is not None:
            pct_deltas.append(hist_pct - ledger_pct)

    resolved_positions = sum(
        1 for p in positions.values() if p["status"] == "resolved")
    return {
        "n_history_trades": n_history,
        "n_history_with_order_id": n_with_order_id,
        "n_matched": len(matched),
        "coverage_history": (len(matched) / n_with_order_id
                             if n_with_order_id else None),
        "coverage_ledger_resolved": (len(matched) / resolved_positions
                                     if resolved_positions else None),
        "pnl_pct_delta_history_minus_ledger": _stats(
            pct_deltas, cfg.exact_delta_tolerance),
    }


# --------------------------------------------------------------------------- #
# legacy marks vs fills-based realized log
# --------------------------------------------------------------------------- #

def marks_vs_fills(
    history_trades: Iterable[Dict[str, Any]],
    realized_log: Iterable[Dict[str, Any]],
    config: Optional[ReconcileConfig] = None,
) -> Dict[str, Any]:
    """Compare legacy mark-based trade P&L against fills-based realized
    amounts.  Match key: same symbol, nearest exit_time/timestamp within
    tolerance; each log row is consumed at most once (dedupes repeated
    closes). Delta = history.pnl - log.amount (positive = legacy marks
    were optimistic)."""
    cfg = config or ReconcileConfig()

    log_by_symbol: Dict[str, List[Tuple[datetime, float, bool]]] = {}
    n_log = 0
    log_bad_rows = 0
    for row in realized_log:
        n_log += 1
        sym = row.get("symbol")
        ts = _parse_naive_utc(row.get("timestamp"))
        amt = _f(row.get("amount"))
        if not sym or ts is None or amt is None:
            log_bad_rows += 1
            continue
        log_by_symbol.setdefault(sym, []).append((ts, amt, False))
    for rows in log_by_symbol.values():
        rows.sort(key=lambda r: r[0])

    n_history = 0
    n_history_eligible = 0
    deltas: List[float] = []
    trades = sorted(
        (t for t in history_trades if t.get("exit_time")),
        key=lambda t: str(t.get("exit_time")),
    )
    for trade in trades:
        n_history += 1
        sym = trade.get("symbol")
        ts = _parse_naive_utc(trade.get("exit_time"))
        pnl = _f(trade.get("pnl"))
        if not sym or ts is None or pnl is None:
            continue
        n_history_eligible += 1
        rows = log_by_symbol.get(sym)
        if not rows:
            continue
        best_i, best_dt = -1, None
        for i, (lts, _amt, used) in enumerate(rows):
            if used:
                continue
            gap = abs((lts - ts).total_seconds())
            if gap <= cfg.marks_match_tolerance_s and (
                    best_dt is None or gap < best_dt):
                best_i, best_dt = i, gap
        if best_i >= 0:
            lts, amt, _ = rows[best_i]
            rows[best_i] = (lts, amt, True)
            deltas.append(pnl - amt)

    n_log_matched = len(deltas)
    unmatched_log = sum(
        1 for rows in log_by_symbol.values() for r in rows if not r[2])
    stats = _stats(deltas, cfg.exact_delta_tolerance)
    if stats["n"] < cfg.min_matched_for_ci:
        stats["ci95"] = None
        stats["ci_note"] = (
            f"fewer than {cfg.min_matched_for_ci} matches; CI suppressed")
    return {
        "n_history_trades": n_history,
        "n_history_eligible": n_history_eligible,
        "n_realized_log_rows": n_log,
        "realized_log_bad_rows": log_bad_rows,
        "n_matched": n_log_matched,
        "coverage_history": (n_log_matched / n_history_eligible
                             if n_history_eligible else None),
        "unmatched_log_rows": unmatched_log,
        "match_tolerance_s": cfg.marks_match_tolerance_s,
        "delta_history_pnl_minus_fills_amount": stats,
    }


# --------------------------------------------------------------------------- #
# candidate capture groups (for future unbiased evaluation)
# --------------------------------------------------------------------------- #

def capture_groups(conn: Any) -> Dict[str, Any]:
    """Count snapshot decisions by kind / accepted / reject_code so the
    baseline, challenger (shadow) and confirmed-rejected groups are
    captured for future unbiased evaluation."""
    out: Dict[str, Any] = {"by_kind_accepted": {}, "rejected_by_code": {}}
    cur = conn.execute(
        "SELECT kind, accepted, COUNT(*) AS n FROM snapshots "
        "GROUP BY kind, accepted")
    for row in cur:
        key = f"{row['kind']}|accepted={int(bool(row['accepted']))}"
        out["by_kind_accepted"][key] = row["n"]
    cur = conn.execute(
        "SELECT reject_code, COUNT(*) AS n FROM snapshots "
        "WHERE accepted = 0 GROUP BY reject_code")
    for row in cur:
        out["rejected_by_code"][str(row["reject_code"])] = row["n"]
    return out


# --------------------------------------------------------------------------- #
# top-level runner
# --------------------------------------------------------------------------- #

def _load_history(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        trades = obj.get("trades", [])
    else:
        trades = obj
    return [t for t in trades if isinstance(t, dict)]


def _load_realized_log(path: str) -> List[Dict[str, Any]]:
    with open(path, "r", encoding="utf-8") as f:
        obj = json.load(f)
    if isinstance(obj, dict):
        obj = obj.get("entries", [])
    return [r for r in obj if isinstance(r, dict)]


def run_reconciliation(
    ledger_path: str,
    history_path: Optional[str] = None,
    realized_log_path: Optional[str] = None,
    config: Optional[ReconcileConfig] = None,
    out_path: Optional[str] = None,
) -> Dict[str, Any]:
    """End-to-end reconciliation; fails open to INSUFFICIENT_DATA."""
    cfg = config or ReconcileConfig()
    report: Dict[str, Any] = {
        "generated_utc": datetime.now(UTC).isoformat(),
        "config": cfg.to_dict(),
        "inputs": {
            "ledger": {"path": ledger_path, "sha256": _sha256(ledger_path)},
            "history": ({"path": history_path,
                         "sha256": _sha256(history_path)}
                        if history_path else None),
            "realized_log": ({"path": realized_log_path,
                              "sha256": _sha256(realized_log_path)}
                             if realized_log_path else None),
        },
        "fees_policy": FEES_POLICY,
        "invalidated_estimates": INVALIDATED_ESTIMATES,
        "notes": [
            "Ledger exit events carry no fill price in the current data; "
            "cash P&L from the ledger alone is therefore incomplete and "
            "marks-vs-fills uses the realized P&L log.",
            "All reconciled fills so far are paper provenance; no live "
            "fills exist. Paper results can never qualify a model for "
            "live promotion.",
        ],
    }

    # ---- ledger events -> positions
    try:
        conn = connect(ledger_path)
    except Exception as exc:  # noqa: BLE001 - fail open with reason
        report["verdict"] = "INSUFFICIENT_DATA"
        report["error"] = f"ledger unavailable: {exc}"
        return report
    try:
        events_by_decision, ingest_report = load_events(conn)
        positions, pos_report = reconcile_positions(events_by_decision, cfg)
        report["ingest"] = ingest_report
        report["positions"] = pos_report

        prov: Dict[str, int] = {}
        for p in positions.values():
            for s in p["provenance_sources"] or ["unknown"]:
                prov[s] = prov.get(s, 0) + 1
        report["provenance_breakdown"] = prov

        cash_known = [p["realized_cash_pnl"] for p in positions.values()
                      if p["realized_cash_pnl"] is not None]
        report["cash_pnl"] = {
            "n_with_full_cash_identity": len(cash_known),
            "n_resolved": pos_report["resolved"],
            "stats": _stats(cash_known, cfg.exact_delta_tolerance),
            "note": ("cash identity requires both entry and exit fill "
                     "prices; ledger exit prices are missing in current "
                     "data, so this is expected to be sparse"),
        }
        report["capture_groups"] = capture_groups(conn)
    finally:
        conn.close()

    # ---- legacy history & realized log comparisons
    history: List[Dict[str, Any]] = []
    if history_path and os.path.exists(history_path):
        try:
            history = _load_history(history_path)
        except (OSError, ValueError) as exc:
            report["history_error"] = str(exc)
    if history:
        report["history_match"] = match_history(history, positions, cfg)

    if realized_log_path and os.path.exists(realized_log_path) and history:
        try:
            realized = _load_realized_log(realized_log_path)
        except (OSError, ValueError) as exc:
            realized = []
            report["realized_log_error"] = str(exc)
        if realized:
            report["marks_vs_fills"] = marks_vs_fills(history, realized, cfg)

    has_events = report.get("ingest", {}).get("events_kept", 0) > 0
    report["verdict"] = "OK" if has_events else "INSUFFICIENT_DATA"

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, sort_keys=True, default=str)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0.1 Stage D ledger/history P&L reconciliation")
    ap.add_argument("--ledger", default="server_data/2026-11-19/research_ledger.db")
    ap.add_argument("--history", default="server_data/2026-11-19/trading_history.json")
    ap.add_argument("--realized-log",
                    default="server_data/2026-11-19/realized_pnl_log.json")
    ap.add_argument("--out",
                    default="artifacts/oracle401/reconciliation_report.json")
    args = ap.parse_args(argv)
    rep = run_reconciliation(
        args.ledger, args.history, args.realized_log, out_path=args.out)
    print(json.dumps(
        {k: rep.get(k) for k in
         ("verdict", "ingest", "positions", "provenance_breakdown",
          "history_match", "marks_vs_fills")},
        indent=2, default=str))
    print(f"report written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
