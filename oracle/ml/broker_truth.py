"""Read-only, exact-order broker activity audit for Oracle decisions.

Never join an exit by symbol or timestamp. An absent exit order ID, partial
fills, missing execution IDs, or missing order-linked fee coverage remain
unresolved. The legacy history is used only to identify mark-derived records.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from typing import Any, Dict, Iterable, List, Mapping, Optional

from .ingest import connect, load_events


def _number(value: Any, *, positive: bool = False) -> Optional[float]:
    try:
        result = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(result) or (positive and result <= 0):
        return None
    return result


def audit_activities(
    events_by_decision: Mapping[str, list],
    activities: Iterable[dict],
    history: Iterable[dict] = (),
) -> Dict[str, Any]:
    """Return separate after-fee, before-fee, mark, and unresolved cohorts.

    FILL activities need a unique broker execution ``id``; FEE activities
    need an ``order_id`` and signed ``net_amount`` (usually negative).
    Quantity and price are taken from execution rows, not ledger intent or
    optimistic marks. An explicit FEE row is required for *each* order to
    call its P&L after-fee, even if that fee is zero.
    """
    fills: Dict[str, List[dict]] = defaultdict(list)
    fees: Dict[str, List[float]] = defaultdict(list)
    seen_activities = set()
    activity_counts: Counter = Counter()
    unallocated_fees = 0.0
    for row in activities:
        typ = str(row.get("activity_type") or row.get("type") or "").upper()
        if typ not in ("FILL", "FEE"):
            continue
        aid = row.get("id")
        if not aid:
            activity_counts["missing_activity_id"] += 1
            continue
        if aid in seen_activities:
            activity_counts["duplicate_activity_id"] += 1
            continue
        seen_activities.add(aid)
        oid = row.get("order_id")
        if typ == "FEE":
            amount = _number(row.get("net_amount"))
            if amount is None or amount > 0:
                activity_counts["invalid_fee"] += 1
            elif not oid:
                unallocated_fees += -amount
                activity_counts["unallocated_fee"] += 1
            else:
                fees[oid].append(-amount)
            continue
        qty = _number(row.get("qty"), positive=True)
        price = _number(row.get("price"), positive=True)
        side = str(row.get("side") or "").lower()
        if not oid or qty is None or price is None or side not in ("buy", "sell"):
            activity_counts["invalid_fill"] += 1
            continue
        fills[oid].append({"qty": qty, "price": price, "side": side,
                           "symbol": row.get("symbol"), "execution_id": aid})

    marks: Dict[str, dict] = {}
    ambiguous_history_ids = set()
    for row in history:
        oid = row.get("order_id")
        if not oid:
            continue
        if oid in marks:
            ambiguous_history_ids.add(oid)
        else:
            marks[oid] = row

    identities: Dict[str, tuple] = {}
    owners: Dict[str, set] = defaultdict(set)
    for did, events in events_by_decision.items():
        entry = {e.broker_order_id for e in events
                 if e.event_type in ("submitted", "fill", "partial_fill")
                 and e.broker_order_id}
        exit_ids = {e.broker_order_id for e in events
                    if e.event_type == "exit" and e.broker_order_id}
        identities[did] = (entry, exit_ids)
        for oid in entry | exit_ids:
            owners[oid].add(did)

    decisions = []
    counts: Counter = Counter()
    trusted_pnl = 0.0
    for did in sorted(identities):
        entries, exits = identities[did]
        related = entries | exits
        reason = None
        entry_qty = exit_qty = None
        pnl_before_fees = pnl_after_fees = None
        fee_total = None
        if any(len(owners[oid]) != 1 for oid in related):
            reason = "order_id_shared_by_decisions"
        elif not entries or not exits or bool(entries & exits):
            reason = "missing_or_ambiguous_order_identity"
            if entries and not (entries & exits):
                observed_entry = [r for oid in entries for r in fills.get(oid, [])]
                entry_qty = sum(r["qty"] for r in observed_entry) if observed_entry else None
        else:
            a = [r for oid in entries for r in fills.get(oid, [])]
            b = [r for oid in exits for r in fills.get(oid, [])]
            symbols = {r["symbol"] for r in a + b if r["symbol"]}
            entry_qty = sum(r["qty"] for r in a) if a else None
            exit_qty = sum(r["qty"] for r in b) if b else 0.0
            if not a or not b:
                reason = "entry_or_exit_fills_missing"
            elif any(r["side"] != "buy" for r in a) or any(
                    r["side"] != "sell" for r in b) or len(symbols) != 1:
                reason = "side_or_symbol_mismatch"
            else:
                if not math.isclose(entry_qty, exit_qty, abs_tol=1e-8):
                    reason = "unmatched_position_quantity"
                else:
                    pnl_before_fees = round(100.0 * (
                        sum(r["qty"] * r["price"] for r in b) -
                        sum(r["qty"] * r["price"] for r in a)), 6)
                    if all(oid in fees for oid in related):
                        fee_total = round(sum(fee for oid in related for fee in fees[oid]), 6)
                        pnl_after_fees = round(pnl_before_fees - fee_total, 6)
                    else:
                        reason = "missing_order_linked_fee_coverage"

        if pnl_after_fees is not None:
            state = "fill_backed_after_order_fees"
            trusted_pnl += pnl_after_fees
        elif pnl_before_fees is not None:
            state = "fill_backed_fee_unknown"
        elif reason != "order_id_shared_by_decisions" and len(entries) == 1 and next(iter(entries)) in marks and next(
                iter(entries)) not in ambiguous_history_ids and _number(
                marks[next(iter(entries))].get("pnl")) is not None:
            state = "mark_derived"
        else:
            state = "unresolved"
        counts[state] += 1
        if reason:
            counts[reason] += 1
        decisions.append({"decision_id": did, "state": state, "reason": reason,
                          "entry_order_ids": sorted(entries),
                          "exit_order_ids": sorted(exits),
                          "entry_qty": entry_qty, "exit_qty": exit_qty,
                          "unmatched_qty": (round(entry_qty - (exit_qty or 0), 8)
                                            if entry_qty is not None and
                                            (exit_qty is None or entry_qty > exit_qty)
                                            else None),
                          "pnl_before_fees": pnl_before_fees,
                          "order_linked_fees": fee_total,
                          "pnl_after_order_fees": pnl_after_fees})

    return {"decisions": decisions, "counts": dict(counts),
            "activity_counts": dict(activity_counts),
            "unallocated_account_fees": round(unallocated_fees, 6),
            "sum_after_order_fees": round(trusted_pnl, 6),
            "account_fee_reconciliation_complete": activity_counts["unallocated_fee"] == 0
                and activity_counts["invalid_fee"] == 0,
            "note": "Only exact order/unique execution IDs count. Paper fills do "
                    "not establish live execution; unallocated fees remain outside "
                    "per-decision P&L."}


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ledger", required=True)
    parser.add_argument("--activities", required=True)
    parser.add_argument("--history", help="optional legacy mark-derived history")
    parser.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    conn = connect(args.ledger)
    try:
        events, _ = load_events(conn)
    finally:
        conn.close()
    with open(args.activities, newline="", encoding="utf-8") as fh:
        activities = list(csv.DictReader(fh))
    history = []
    if args.history:
        with open(args.history, encoding="utf-8") as fh:
            obj = json.load(fh)
            history = obj.get("trades", []) if isinstance(obj, dict) else obj
    result = audit_activities(events, activities, history)
    import os
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != "decisions"},
                     indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
