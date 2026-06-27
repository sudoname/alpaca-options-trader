"""
P0d — Backfill bridge: seed episodes.db from existing closed trades.

The learning leaderboards (Phase 1) would be EMPTY on day one because
``episodes.db`` has no rows yet. This one-shot bridge populates outcome rows
from the two histories that already exist so the Trading / Execution /
Confidence report sections have real data immediately, while the
agent/regime/pattern tables fill in as new *live* trades close.

Two sources (both READ-ONLY; this never touches the broker):

  1. Broker round-trips  (source of truth, single-leg calls/puts)
     FIFO-match buy/sell FILLs per OCC contract from a broker export's
     ``account_activities.json`` into closed round-trips with real net P&L
     (fees attributed from FEE activities by order_id). Evidence is thin:
     direction (call=>up / put=>down), option_type, dte_bucket, strategy.

  2. Advisory spreads    (``advisory_attribution.json``, 165 closed)
     bullish_put_credit_spreads with oracle_score and, where present,
     dte / iv_rank. P&L from the record.

Every inserted row is tagged ``mode="backfill"`` (broker) or
``mode="backfill_advisory"`` so live vs backfilled trades stay separable. The
backfill is idempotent: it first deletes existing rows with those two modes,
then re-inserts — it NEVER deletes or alters live rows.
"""

import json
import os
import re
import uuid
from collections import deque
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import feature_buckets
from episode_store import EpisodeStore
from evidence_context import iv_bucket

# Stable namespace so deterministic decision_ids survive re-runs.
_NS = uuid.UUID("a1b2c3d4-0000-4000-8000-000000000d04")

BACKFILL_MODES = ("backfill", "backfill_advisory")

_OCC_RE = re.compile(r"^([A-Z]+)(\d{6})([CP])(\d{8})$")


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def parse_occ(symbol: str) -> Optional[Tuple[str, str, str, float]]:
    """Parse an OCC option symbol -> (underlying, 'YYYY-MM-DD', 'C'|'P', strike).

    e.g. ``KO260717C00081000`` -> ("KO", "2026-07-17", "C", 81.0). None when
    the symbol is not a standard OCC contract.
    """
    if not isinstance(symbol, str):
        return None
    m = _OCC_RE.match(symbol.strip())
    if not m:
        return None
    root, ymd, cp, strike8 = m.groups()
    try:
        exp = f"20{ymd[0:2]}-{ymd[2:4]}-{ymd[4:6]}"
        strike = int(strike8) / 1000.0
        return root, exp, cp, strike
    except Exception:
        return None


def _f(value) -> Optional[float]:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).replace("Z", "+00:00")
    try:
        return datetime.fromisoformat(s)
    except ValueError:
        for fmt in ("%Y-%m-%dT%H:%M:%S.%f%z", "%Y-%m-%dT%H:%M:%S%z",
                    "%Y-%m-%d"):
            try:
                return datetime.strptime(str(value), fmt)
            except ValueError:
                continue
    return None


def _det_id(*parts) -> str:
    return str(uuid.uuid5(_NS, "|".join(str(p) for p in parts)))


def _purge_backfill(store: EpisodeStore) -> int:
    """Delete prior backfill rows (both modes) so re-runs stay idempotent.

    Only ever removes rows whose ``mode`` is a backfill mode — live rows are
    untouched.
    """
    q = ",".join("?" for _ in BACKFILL_MODES)
    cur = store.conn.execute(
        f"DELETE FROM episodes WHERE mode IN ({q})", BACKFILL_MODES)
    store.conn.commit()
    return cur.rowcount


# --------------------------------------------------------------------------- #
# Source 1: broker round-trips (FIFO lot matching)
# --------------------------------------------------------------------------- #
def _fee_by_order(activities: List[dict]) -> Dict[str, float]:
    """Total fee (positive dollars) per order_id from FEE activities."""
    fees: Dict[str, float] = {}
    for a in activities:
        if a.get("activity_type") != "FEE":
            continue
        oid = a.get("order_id")
        amt = _f(a.get("net_amount"))
        if oid and amt is not None:
            fees[oid] = fees.get(oid, 0.0) + abs(amt)
    return fees


def _qty_by_order(fills: List[dict]) -> Dict[str, float]:
    """Total filled qty per order_id (to pro-rate per-order fees)."""
    q: Dict[str, float] = {}
    for fl in fills:
        oid = fl.get("order_id")
        qf = _f(fl.get("qty")) or 0.0
        if oid:
            q[oid] = q.get(oid, 0.0) + qf
    return q


def match_round_trips(activities: List[dict]) -> List[dict]:
    """FIFO-match FILLs per OCC contract into closed round-trips.

    Returns a list of dicts: underlying, option_type ('call'/'put'), strike,
    expiry, qty, entry_price, exit_price, entry_time, exit_time, entry_order,
    exit_order, is_long. Only fully-closed matched lots are returned; still-open
    lots are ignored.
    """
    fills = [a for a in activities if a.get("activity_type") == "FILL"]
    fee_order = _fee_by_order(activities)
    qty_order = _qty_by_order(fills)

    by_symbol: Dict[str, List[dict]] = {}
    for fl in fills:
        by_symbol.setdefault(fl.get("symbol"), []).append(fl)

    trips: List[dict] = []
    for symbol, sfills in by_symbol.items():
        parsed = parse_occ(symbol)
        if parsed is None:
            continue
        underlying, expiry, cp, strike = parsed
        sfills.sort(key=lambda f: str(f.get("transaction_time") or ""))

        # Signed FIFO lot book. +1 = long lot (opened by buy), -1 = short.
        lots: deque = deque()
        for fl in sfills:
            side = str(fl.get("side") or "")
            sign = 1 if side == "buy" else -1   # sell / sell_short -> -1
            qty = _f(fl.get("qty")) or 0.0
            price = _f(fl.get("price"))
            t = _parse_ts(fl.get("transaction_time"))
            oid = fl.get("order_id")
            if qty <= 0 or price is None:
                continue

            # Same direction as the open book (or empty) -> opening lots.
            if not lots or lots[0]["sign"] == sign:
                lots.append({"sign": sign, "qty": qty, "price": price,
                             "time": t, "order": oid})
                continue

            # Opposite direction -> closes existing lots FIFO.
            remaining = qty
            while remaining > 1e-9 and lots and lots[0]["sign"] != sign:
                lot = lots[0]
                m = min(remaining, lot["qty"])
                is_long = lot["sign"] == 1
                entry_price = lot["price"]
                exit_price = price
                # Pro-rated fees from each leg's order.
                ef = fee_order.get(lot["order"], 0.0)
                xf = fee_order.get(oid, 0.0)
                eq = qty_order.get(lot["order"], 0.0) or 1.0
                xq = qty_order.get(oid, 0.0) or 1.0
                fee_share = ef * (m / eq) + xf * (m / xq)
                trips.append({
                    "underlying": underlying, "symbol": symbol,
                    "option_type": "call" if cp == "C" else "put",
                    "strike": strike, "expiry": expiry, "qty": m,
                    "entry_price": entry_price, "exit_price": exit_price,
                    "entry_time": lot["time"], "exit_time": t,
                    "entry_order": lot["order"], "exit_order": oid,
                    "is_long": is_long, "fees": fee_share,
                })
                lot["qty"] -= m
                remaining -= m
                if lot["qty"] <= 1e-9:
                    lots.popleft()
            # Any leftover opens a new lot in the new direction.
            if remaining > 1e-9:
                lots.append({"sign": sign, "qty": remaining, "price": price,
                             "time": t, "order": oid})
    return trips


def _trip_pnl(trip: dict) -> Tuple[float, float, float]:
    """(gross_pnl_pct, net_pnl_pct, net_pnl_dollars) for a round-trip."""
    entry = trip["entry_price"]
    exit_ = trip["exit_price"]
    qty = trip["qty"]
    long_ = trip["is_long"]
    gross_dollars = (exit_ - entry) * qty * 100.0
    if not long_:
        gross_dollars = -gross_dollars
    net_dollars = gross_dollars - trip.get("fees", 0.0)
    notional = entry * qty * 100.0
    gross_pct = (gross_dollars / notional * 100.0) if notional else 0.0
    net_pct = (net_dollars / notional * 100.0) if notional else 0.0
    return round(gross_pct, 4), round(net_pct, 4), round(net_dollars, 4)


def backfill_broker(store: EpisodeStore, activities: List[dict]) -> int:
    """Insert one episode per closed broker round-trip. Returns rows written."""
    n = 0
    for idx, trip in enumerate(match_round_trips(activities)):
        if trip["entry_time"] is None or trip["exit_time"] is None:
            continue
        gross_pct, net_pct, net_dollars = _trip_pnl(trip)
        entry_dt = trip["entry_time"]
        exit_dt = trip["exit_time"]
        hold_days = max(0, (exit_dt.date() - entry_dt.date()).days)
        exp_dt = _parse_ts(trip["expiry"])
        dte_entry = ((exp_dt.date() - entry_dt.date()).days
                     if exp_dt else None)
        is_call = trip["option_type"] == "call"
        strategy = ("long_" if trip["is_long"] else "short_") + trip["option_type"]
        direction = ("up" if is_call else "down") if trip["is_long"] else \
                    ("down" if is_call else "up")
        evidence = {
            "direction": direction,
            "option_type": trip["option_type"],
            "dte_bucket": feature_buckets.dte_bucket(dte_entry),
            "strategy": strategy,
            "source": "broker",
        }
        did = _det_id("broker", idx, trip["symbol"], trip["entry_order"],
                      trip["exit_order"], trip["qty"], entry_dt.isoformat())
        features = {
            "as_of": entry_dt.isoformat(),
            "source": "broker_roundtrip",
            "backfill": True,
            "underlying": trip["underlying"],
            "strike": trip["strike"],
            "expiry": trip["expiry"],
            "dte_entry": dte_entry,
            "evidence": evidence,
        }
        store.log_decision(
            symbol=trip["symbol"], underlying=trip["underlying"],
            strat=strategy, features=features, quote=None, modeled_cost=None,
            rule_action="CALL" if is_call else "PUT", rule_confidence=0.0,
            gate=None, chosen_action="CALL" if is_call else "PUT",
            qty=int(round(trip["qty"])), mode="backfill",
            as_of=entry_dt.isoformat(), decision_id=did,
        )
        store.record_outcome(
            did, fill_price=trip["entry_price"], exit_price=trip["exit_price"],
            gross_pnl_pct=gross_pct, net_pnl_pct=net_pct,
            net_pnl_dollars=net_dollars, hold_days=hold_days,
            outcome="win" if net_dollars > 0 else "loss",
            closed_at=exit_dt.isoformat(),
        )
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Source 2: advisory credit spreads
# --------------------------------------------------------------------------- #
def backfill_advisory(store: EpisodeStore, records: List[dict]) -> int:
    """Insert one episode per CLOSED advisory record. Returns rows written."""
    n = 0
    for rec in records:
        if not rec.get("date_closed"):
            continue
        opened = _parse_ts(rec.get("date_opened"))
        closed = _parse_ts(rec.get("date_closed"))
        hold_days = (max(0, (closed.date() - opened.date()).days)
                     if opened and closed else 0)
        pnl = _f(rec.get("pnl")) or 0.0
        pnl_pct = _f(rec.get("pnl_percent")) or 0.0
        wl = str(rec.get("win_loss") or "").lower()
        outcome = rec.get("exit_reason") or ("win" if wl == "win" else "loss")
        strategy = rec.get("strategy") or "spread"
        evidence = {
            "strategy": strategy,
            "oracle_score": _f(rec.get("oracle_score")),
            "iv_bucket": iv_bucket(rec.get("iv_rank")),
            "dte_bucket": feature_buckets.dte_bucket(rec.get("dte")),
            "advisory_recommendation": rec.get("advisory_recommendation"),
            "direction": "up" if "bullish" in strategy else
                         ("down" if "bearish" in strategy else None),
            "source": "advisory",
        }
        did = _det_id("advisory", rec.get("trade_id"))
        as_of = (opened.isoformat() if opened
                 else str(rec.get("date_opened") or ""))
        features = {
            "as_of": as_of, "source": "advisory_attribution",
            "backfill": True, "evidence": evidence,
        }
        store.log_decision(
            symbol=rec.get("symbol") or "UNKNOWN",
            underlying=rec.get("symbol") or "UNKNOWN",
            strat=strategy, features=features, quote=None, modeled_cost=None,
            rule_action="SPREAD", rule_confidence=_f(rec.get("oracle_score")) or 0.0,
            gate=None, chosen_action="SPREAD", qty=1, mode="backfill_advisory",
            as_of=as_of, decision_id=did,
        )
        store.record_outcome(
            did, gross_pnl_pct=pnl_pct, net_pnl_pct=pnl_pct,
            net_pnl_dollars=pnl, hold_days=hold_days, outcome=str(outcome),
            closed_at=(closed.isoformat() if closed
                       else str(rec.get("date_closed") or "")),
        )
        n += 1
    return n


# --------------------------------------------------------------------------- #
# Discovery + orchestration
# --------------------------------------------------------------------------- #
def _latest_export_activities() -> Optional[str]:
    """Path to the newest broker-export account_activities.json, if any."""
    base = "trade_export"
    if not os.path.isdir(base):
        return None
    candidates = []
    for name in os.listdir(base):
        p = os.path.join(base, name, "account_activities.json")
        if os.path.isfile(p):
            candidates.append(p)
    if not candidates:
        return None
    return max(candidates, key=os.path.getmtime)


def _load_json(path: Optional[str]):
    if not path or not os.path.isfile(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def run_backfill(db_path: str = "episodes.db",
                 activities_path: Optional[str] = None,
                 advisory_path: str = "advisory_attribution.json") -> dict:
    """Populate episodes.db from broker + advisory histories. Idempotent."""
    store = EpisodeStore(db_path)
    try:
        purged = _purge_backfill(store)

        if activities_path is None:
            activities_path = _latest_export_activities()
        activities = _load_json(activities_path) or []
        broker_n = backfill_broker(store, activities) if activities else 0

        advisory = _load_json(advisory_path) or []
        advisory_n = backfill_advisory(store, advisory) if advisory else 0

        stats = store.stats()
    finally:
        store.close()
    return {
        "purged": purged,
        "activities_path": activities_path,
        "broker_rows": broker_n,
        "advisory_rows": advisory_n,
        "db_stats": stats,
    }


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, in-memory store + synthetic fills)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # OCC parsing.
    assert parse_occ("KO260717C00081000") == ("KO", "2026-07-17", "C", 81.0)
    assert parse_occ("SPY260109P00475000") == ("SPY", "2026-01-09", "P", 475.0)
    if parse_occ("garbage") is not None:
        print("FAIL: bad OCC should be None"); ok = False

    # A long call round-trip: buy 2 @1.00, sell 2 @1.50 (+$100 gross, -fees).
    activities = [
        {"activity_type": "FILL", "symbol": "KO260717C00081000", "side": "buy",
         "qty": "2", "price": "1.00", "order_id": "o1",
         "transaction_time": "2026-06-01T15:00:00Z"},
        {"activity_type": "FILL", "symbol": "KO260717C00081000", "side": "sell",
         "qty": "2", "price": "1.50", "order_id": "o2",
         "transaction_time": "2026-06-03T15:00:00Z"},
        {"activity_type": "FEE", "order_id": "o1", "net_amount": "-0.04"},
        {"activity_type": "FEE", "order_id": "o2", "net_amount": "-0.04"},
    ]
    trips = match_round_trips(activities)
    if len(trips) != 1:
        print("FAIL: expected one round-trip", len(trips)); ok = False
    else:
        t = trips[0]
        if t["qty"] != 2 or not t["is_long"] or t["option_type"] != "call":
            print("FAIL: trip fields", t); ok = False
        gp, np_, nd = _trip_pnl(t)
        # gross = (1.50-1.00)*2*100 = 100; fees 0.08; net = 99.92
        if abs(gp - 50.0) > 1e-6 or abs(nd - 99.92) > 1e-6:
            print("FAIL: pnl math", gp, np_, nd); ok = False

    # End-to-end into an in-memory store.
    store = EpisodeStore(":memory:")
    try:
        bn = backfill_broker(store, activities)
        if bn != 1:
            print("FAIL: broker backfill rows", bn); ok = False
        adv = [{"trade_id": "t1", "symbol": "SPY",
                "date_opened": "2026-06-09", "date_closed": "2026-06-09",
                "strategy": "bullish_put_credit_spread", "oracle_score": 80.0,
                "pnl": 25.0, "pnl_percent": 5.88, "win_loss": "win",
                "exit_reason": "take_profit", "iv_rank": 45.0, "dte": 7}]
        an = backfill_advisory(store, adv)
        if an != 1:
            print("FAIL: advisory backfill rows", an); ok = False
        comp = store.completed()
        if len(comp) != 2:
            print("FAIL: expected 2 completed", len(comp)); ok = False
        # Evidence persisted under features_json.evidence.
        feats = json.loads(comp[0]["features_json"])
        if "evidence" not in feats:
            print("FAIL: evidence not persisted", feats); ok = False
        # Advisory evidence has iv_bucket from rank 45 -> medium.
        adv_row = [r for r in comp if r["mode"] == "backfill_advisory"][0]
        adv_ev = json.loads(adv_row["features_json"])["evidence"]
        if adv_ev["iv_bucket"] != "medium" or adv_ev["dte_bucket"] != "0-7":
            print("FAIL: advisory evidence buckets", adv_ev); ok = False
    finally:
        store.close()

    # Idempotency: purge removes only backfill rows.
    store2 = EpisodeStore(":memory:")
    try:
        backfill_broker(store2, activities)
        # Add a fake live row that must survive purge.
        store2.log_decision(
            symbol="LIVE", underlying="LIVE", strat="live", features={},
            quote=None, modeled_cost=None, rule_action="CALL",
            rule_confidence=1.0, gate=None, chosen_action="CALL", qty=1,
            mode="live")
        purged = _purge_backfill(store2)
        if purged != 1:
            print("FAIL: purge count", purged); ok = False
        if store2.stats()["total"] != 1:
            print("FAIL: live row should survive purge"); ok = False
    finally:
        store2.close()

    print("backfill_episodes self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--run" in sys.argv:
        result = run_backfill()
        print(json.dumps(result, indent=2, default=str))
        sys.exit(0)
    sys.exit(_self_test())
