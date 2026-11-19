"""Research-ledger (sqlite) readers for the Oracle 4.0 dataset builder.

Reads the frozen research ledger produced by ``research_ledger.py``:
  * ``snapshots``  -- one row per decision (candidate / shadow_candidate)
  * ``events``     -- submitted / fill / exit / outcome / cancel / reject

Rules (per the master prompt):
  * every record carries event time and availability time
    (``event_at_utc`` / ``received_at_utc``);
  * duplicates are dropped deterministically (keep-first) and counted,
    never silently merged;
  * invalid timestamps make a row invalid -- it is dropped and counted,
    never guessed;
  * missing data stays ``None``.

No production module imports this file.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional, Tuple

from .schema import DecisionRecord, UnderlyingObs, parse_utc

__all__ = [
    "connect",
    "load_underlying_obs",
    "load_decisions",
    "load_events",
    "LedgerEvent",
]


# --------------------------------------------------------------------------- #
# connection
# --------------------------------------------------------------------------- #

def connect(path: str) -> sqlite3.Connection:
    """Open the ledger read-only (URI) so research can never mutate it."""
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _loads(raw: Optional[str]) -> Dict[str, Any]:
    if not raw:
        return {}
    try:
        obj = json.loads(raw)
    except (ValueError, TypeError):
        return {}
    return obj if isinstance(obj, dict) else {}


def _in_range(dt: datetime,
              start: Optional[datetime],
              end: Optional[datetime]) -> bool:
    if start is not None and dt < start:
        return False
    if end is not None and dt >= end:
        return False
    return True


# --------------------------------------------------------------------------- #
# underlying observations
# --------------------------------------------------------------------------- #

def load_underlying_obs(
    conn: sqlite3.Connection,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
) -> Tuple[Dict[str, List[UnderlyingObs]], Dict[str, int]]:
    """Extract the underlying price series from snapshot rows.

    Every snapshot row with a non-null ``underlying_price`` is one price
    observation.  Dedup key is ``(underlying, event_at)``: the first row by
    ``(received_at, snapshot_id)`` wins; later duplicates are counted.

    Returns ``(obs_by_underlying, report)`` where each list is sorted by
    ``event_at`` and report counts rows kept / dropped and why.
    """
    report = {
        "rows_scanned": 0,
        "obs_kept": 0,
        "dropped_null_price": 0,
        "dropped_bad_timestamp": 0,
        "dropped_duplicate": 0,
        "dropped_out_of_range": 0,
    }
    seen: set = set()
    out: Dict[str, List[UnderlyingObs]] = {}

    cur = conn.execute(
        """SELECT snapshot_id, underlying, underlying_price,
                  event_at_utc, received_at_utc
             FROM snapshots
            WHERE underlying IS NOT NULL
            ORDER BY received_at_utc, snapshot_id"""
    )
    for row in cur:
        report["rows_scanned"] += 1
        price = row["underlying_price"]
        if price is None:
            report["dropped_null_price"] += 1
            continue
        event_at = parse_utc(row["event_at_utc"])
        if event_at is None:
            report["dropped_bad_timestamp"] += 1
            continue
        available_at = parse_utc(row["received_at_utc"]) or event_at
        if available_at < event_at:
            # availability can never precede the event; treat as same instant
            available_at = event_at
        if not _in_range(event_at, start, end):
            report["dropped_out_of_range"] += 1
            continue
        key = (row["underlying"], event_at)
        if key in seen:
            report["dropped_duplicate"] += 1
            continue
        seen.add(key)
        out.setdefault(row["underlying"], []).append(
            UnderlyingObs(
                underlying=row["underlying"],
                event_at=event_at,
                available_at=available_at,
                price=float(price),
                source_snapshot_id=row["snapshot_id"],
            )
        )
        report["obs_kept"] += 1

    for series in out.values():
        series.sort(key=lambda o: o.event_at)
    return out, report


# --------------------------------------------------------------------------- #
# decisions
# --------------------------------------------------------------------------- #

def load_decisions(
    conn: sqlite3.Connection,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    kinds: Optional[List[str]] = None,
) -> Tuple[List[DecisionRecord], Dict[str, int]]:
    """Load decision snapshots as typed records.

    Dedup key is ``decision_id`` (keep-first by ``received_at, snapshot_id``);
    the frozen ledger has unique decision_ids, but the reader must not rely
    on that. Rows with unparseable event timestamps are dropped and counted.
    """
    report = {
        "rows_scanned": 0,
        "decisions_kept": 0,
        "dropped_bad_timestamp": 0,
        "dropped_duplicate_decision_id": 0,
        "dropped_out_of_range": 0,
        "dropped_kind_filtered": 0,
        "dropped_missing_underlying": 0,
    }
    kinds_set = set(kinds) if kinds else None
    seen: set = set()
    out: List[DecisionRecord] = []

    cur = conn.execute(
        """SELECT * FROM snapshots
            ORDER BY received_at_utc, snapshot_id"""
    )
    for row in cur:
        report["rows_scanned"] += 1
        if kinds_set is not None and row["kind"] not in kinds_set:
            report["dropped_kind_filtered"] += 1
            continue
        if not row["underlying"]:
            report["dropped_missing_underlying"] += 1
            continue
        as_of = parse_utc(row["event_at_utc"])
        if as_of is None:
            report["dropped_bad_timestamp"] += 1
            continue
        available_at = parse_utc(row["received_at_utc"]) or as_of
        if available_at < as_of:
            available_at = as_of
        if not _in_range(as_of, start, end):
            report["dropped_out_of_range"] += 1
            continue
        if row["decision_id"] in seen:
            report["dropped_duplicate_decision_id"] += 1
            continue
        seen.add(row["decision_id"])

        def _f(col: str) -> Optional[float]:
            v = row[col]
            return float(v) if v is not None else None

        out.append(
            DecisionRecord(
                decision_id=row["decision_id"],
                snapshot_id=row["snapshot_id"],
                kind=row["kind"],
                accepted=bool(row["accepted"]),
                reject_code=row["reject_code"],
                as_of=as_of,
                available_at=available_at,
                underlying=row["underlying"],
                symbol=row["symbol"],
                expiration=row["expiration"],
                strike=_f("strike"),
                option_type=row["option_type"],
                multiplier=_f("multiplier"),
                quote_bid=_f("quote_bid"),
                quote_ask=_f("quote_ask"),
                underlying_price=_f("underlying_price"),
                iv=_f("iv"),
                delta=_f("delta"),
                direction=row["direction"],
                score=_f("score"),
                p_win=_f("p_win"),
                ev=_f("ev"),
                signals=_loads(row["signals_json"]),
                prediction=_loads(row["prediction_json"]),
                sizing=_loads(row["sizing_json"]),
                regime=row["regime"],
                config_hash=row["config_hash"],
                code_rev=row["code_rev"],
                extra=_loads(row["extra_json"]),
            )
        )
        report["decisions_kept"] += 1

    out.sort(key=lambda d: (d.as_of, d.decision_id))
    return out, report


# --------------------------------------------------------------------------- #
# events
# --------------------------------------------------------------------------- #

class LedgerEvent:
    """One broker/outcome event, with parsed times and payload."""

    __slots__ = ("event_key", "decision_id", "event_type", "event_at",
                 "available_at", "broker_order_id", "qty", "price",
                 "fees", "status", "payload")

    def __init__(self, **kw: Any) -> None:
        for name in self.__slots__:
            setattr(self, name, kw.get(name))

    def __repr__(self) -> str:  # pragma: no cover - debug aid
        return (f"LedgerEvent({self.event_type} decision={self.decision_id} "
                f"at={self.event_at})")


def load_events(
    conn: sqlite3.Connection,
) -> Tuple[Dict[str, List[LedgerEvent]], Dict[str, int]]:
    """Load all events grouped by decision_id, each list sorted by time.

    Dedup key is ``event_key`` (the ledger's idempotency hash). Events with
    unparseable timestamps are dropped and counted. No date filtering: an
    in-range decision's outcome may land after the window and must be found.
    """
    report = {
        "rows_scanned": 0,
        "events_kept": 0,
        "dropped_bad_timestamp": 0,
        "dropped_duplicate_key": 0,
        "dropped_missing_decision_id": 0,
        "event_at_fallback_received": 0,
    }
    seen: set = set()
    out: Dict[str, List[LedgerEvent]] = {}

    cur = conn.execute(
        """SELECT event_key, decision_id, event_type, event_at_utc,
                  received_at_utc, broker_order_id, qty, price, fees,
                  status, payload_json
             FROM events
            ORDER BY received_at_utc, event_key"""
    )
    for row in cur:
        report["rows_scanned"] += 1
        if not row["decision_id"]:
            report["dropped_missing_decision_id"] += 1
            continue
        if row["event_key"] in seen:
            report["dropped_duplicate_key"] += 1
            continue
        event_at = parse_utc(row["event_at_utc"])
        available_at = parse_utc(row["received_at_utc"])
        if event_at is None:
            # event_at_utc is nullable ("if known"); received_at is NOT NULL.
            # Falling back preserves real evidence without inventing a time.
            if available_at is None:
                report["dropped_bad_timestamp"] += 1
                continue
            event_at = available_at
            report["event_at_fallback_received"] += 1
        if available_at is None or available_at < event_at:
            available_at = event_at
        seen.add(row["event_key"])
        out.setdefault(row["decision_id"], []).append(
            LedgerEvent(
                event_key=row["event_key"],
                decision_id=row["decision_id"],
                event_type=row["event_type"],
                event_at=event_at,
                available_at=available_at,
                broker_order_id=row["broker_order_id"],
                qty=float(row["qty"]) if row["qty"] is not None else None,
                price=float(row["price"]) if row["price"] is not None else None,
                fees=float(row["fees"]) if row["fees"] is not None else None,
                status=row["status"],
                payload=_loads(row["payload_json"]),
            )
        )
        report["events_kept"] += 1

    for evs in out.values():
        evs.sort(key=lambda e: (e.event_at, e.event_key))
    return out, report
