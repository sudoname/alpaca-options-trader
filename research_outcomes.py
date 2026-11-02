"""
Outcome reconciliation for the research decision ledger.

Turns broker truth into append-only ledger events so every accepted decision
ends in exactly one of three classes:

  realized    — an ``outcome`` event backed by broker fills/exits
  simulated   — an ``outcome`` event produced by an EXPLICIT hypothetical
                policy (rejected/skipped candidates only), labeled as such
  unresolved  — accepted, but no terminal event yet (honest "don't know")

Design rules (mirrors the live hooks' guarantees):
  * Idempotent by construction: every write goes through
    ``ResearchLedger.record_event`` whose deterministic event_key +
    INSERT OR IGNORE makes re-runs and restarts no-ops. The live hook and the
    reconciler recording the same fill produce ONE row.
  * Never claims a fill from a price touch. Fills come only from the broker
    order record or the FILL activity stream.
  * Skipped candidates are resolved ONLY under the explicit
    ``underlying_counterfactual_v1`` policy and the result is stored with
    ``class='simulated'`` so it can never be confused with realized P/L.
  * Network is injected (``fetch_order`` / ``fetch_activities`` callables) so
    the whole module is testable offline; the CLI wires real Alpaca fetchers.
  * Dry-run by default from the CLI; ``--apply`` writes.

This module only READS broker state and WRITES research_ledger.db. It cannot
place, replace, or cancel orders — no order-submitting code path exists here.
"""

import os
from datetime import datetime, timezone
from typing import Callable, Dict, List, Optional

from research_ledger import ResearchLedger

SKIP_POLICY = "underlying_counterfactual_v1"

# Broker order states that terminate an order without (further) fills.
_TERMINAL_ORDER_STATES = ("canceled", "expired", "rejected", "done_for_day")


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _parse_iso(ts) -> Optional[datetime]:
    if not ts:
        return None
    try:
        s = str(ts).replace("Z", "+00:00")
        dt = datetime.fromisoformat(s)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return dt
    except ValueError:
        return None


def _f(x) -> Optional[float]:
    try:
        return None if x in (None, "") else float(x)
    except (TypeError, ValueError):
        return None


class OutcomeReconciler:
    """Reconcile ledger decisions against broker truth. Read-broker, write-ledger."""

    def __init__(
        self,
        ledger: ResearchLedger,
        fetch_order: Optional[Callable[[str], Optional[Dict]]] = None,
        fetch_activities: Optional[Callable[[str], List[Dict]]] = None,
        price_fn: Optional[Callable[[str], Optional[float]]] = None,
        apply: bool = False,
        now: Optional[datetime] = None,
    ):
        self.ledger = ledger
        self.fetch_order = fetch_order
        self.fetch_activities = fetch_activities
        self.price_fn = price_fn
        self.apply = bool(apply)
        self.now = now or _utcnow()
        self.planned: List[Dict] = []   # every action, written or not

    # ------------------------------------------------------------- plumbing
    def _emit(self, **event) -> bool:
        """Record one event (or plan it in dry-run). Returns True if written."""
        self.planned.append(dict(event))
        if not self.apply:
            return False
        _, inserted = self.ledger.record_event(source="reconciler", **event)
        return inserted

    def _open_submissions(self) -> Dict[str, List[Dict]]:
        """decision_id -> submitted events, for decisions without a terminal event."""
        open_ids = {r["decision_id"] for r in self.ledger.unresolved()}
        out: Dict[str, List[Dict]] = {}
        for did in sorted(open_ids):
            evs = self.ledger.get_decision(did)["events"]
            subs = [e for e in evs if e["event_type"] == "submitted"
                    and e.get("broker_order_id")]
            if subs:
                out[did] = subs
        return out

    # ------------------------------------------------- 1) order-status sync
    def reconcile_orders(self) -> Dict:
        """Fetch each open decision's order(s); append fill/cancel/reject events.

        Only the broker order record is trusted: ``filled_avg_price`` /
        ``filled_qty`` become fill events; terminal no-fill states become
        reject/cancel events. Nothing is inferred from quotes.
        """
        summary = {"checked": 0, "fills": 0, "cancels": 0, "rejects": 0,
                   "errors": 0, "skipped_no_fetcher": 0}
        if self.fetch_order is None:
            summary["skipped_no_fetcher"] = 1
            return summary
        for did, subs in self._open_submissions().items():
            for sub in subs:
                oid = sub["broker_order_id"]
                summary["checked"] += 1
                try:
                    order = self.fetch_order(oid)
                except Exception:
                    summary["errors"] += 1
                    continue
                if not isinstance(order, dict):
                    summary["errors"] += 1
                    continue
                status = str(order.get("status") or "").lower()
                fqty = _f(order.get("filled_qty")) or 0.0
                fprice = _f(order.get("filled_avg_price"))
                if fqty > 0 and fprice is not None:
                    partial = status == "partially_filled"
                    if self._emit(
                        decision_id=did,
                        event_type="partial_fill" if partial else "fill",
                        broker_order_id=oid, qty=fqty, price=fprice,
                        status=status,
                        event_at=order.get("filled_at") or order.get("updated_at"),
                        dedupe_tag=f"order_sync:{oid}",
                    ):
                        summary["fills"] += 1
                elif status in _TERMINAL_ORDER_STATES:
                    etype = "reject" if status == "rejected" else "cancel"
                    if self._emit(
                        decision_id=did, event_type=etype,
                        broker_order_id=oid, status=status,
                        event_at=order.get("canceled_at")
                        or order.get("failed_at") or order.get("updated_at"),
                        payload={"reason": f"broker_{status}_no_fill"},
                    ):
                        summary["cancels" if etype == "cancel" else "rejects"] += 1
        return summary

    # ----------------------------------------- 2) FILL activity-stream sync
    def reconcile_activities(self, after_iso: str) -> Dict:
        """Match broker FILL activities to decisions via their order ids.

        Each activity carries its own broker id which becomes the dedupe_tag,
        so two partial fills with identical qty/price still record as two
        events, and a re-run records nothing new.
        """
        summary = {"activities": 0, "matched": 0, "recorded": 0,
                   "skipped_no_fetcher": 0}
        if self.fetch_activities is None:
            summary["skipped_no_fetcher"] = 1
            return summary
        try:
            acts = self.fetch_activities(after_iso) or []
        except Exception:
            return summary
        summary["activities"] = len(acts)
        for act in acts:
            oid = act.get("order_id")
            if not oid:
                continue
            linked = self.ledger.events_for_order(str(oid))
            if not linked:
                continue
            did = linked[0]["decision_id"]
            summary["matched"] += 1
            qty = _f(act.get("qty"))
            price = _f(act.get("price"))
            side = str(act.get("side") or "")
            partial = str(act.get("type") or "") == "partial_fill"
            if self._emit(
                decision_id=did,
                event_type="partial_fill" if partial else "fill",
                broker_order_id=str(oid), qty=qty, price=price,
                status="filled",
                event_at=act.get("transaction_time"),
                payload={"side": side} if side else None,
                dedupe_tag=f"activity:{act.get('id')}",
            ):
                summary["recorded"] += 1
        return summary

    # -------------------------------------------------- 3) expiry resolution
    def reconcile_expired(self) -> Dict:
        """Mark accepted decisions whose contract expired with no exit.

        Calendar truth only: the contract's expiration (parsed from the OCC
        symbol at snapshot time) is strictly before today (UTC date). Records a
        terminal ``expiry`` event; the VALUE stays honest — the payload says
        the settlement value was not observed. No price is fabricated.
        """
        summary = {"checked": 0, "expired": 0}
        today = self.now.date()
        for snap in self.ledger.unresolved():
            summary["checked"] += 1
            exp = _parse_iso(snap.get("expiration"))
            if exp is None:
                continue
            if exp.date() >= today:
                continue
            if self._emit(
                decision_id=snap["decision_id"], event_type="expiry",
                event_at=snap.get("expiration"),
                payload={"class": "realized",
                         "note": "contract expired without recorded exit; "
                                 "settlement value not observed"},
            ):
                summary["expired"] += 1
        return summary

    # ------------------------------------- 4) skipped-candidate counterfactual
    def resolve_skipped(self, horizon_min: float = 390.0) -> Dict:
        """Attach an EXPLICITLY SIMULATED counterfactual to old skipped candidates.

        Policy ``underlying_counterfactual_v1``: the signed forward underlying
        move in the direction the bot declined (reuses the live project's
        skip_counterfactual math). Requires the decision-time underlying price
        recorded in the snapshot; rows without one stay unresolved. The result
        is an ``outcome`` event with class='simulated' — it is never a fill and
        never mixes with realized P/L.
        """
        summary = {"checked": 0, "resolved": 0, "no_entry_price": 0,
                   "not_due": 0, "no_price": 0, "skipped_no_fetcher": 0}
        if self.price_fn is None:
            summary["skipped_no_fetcher"] = 1
            return summary
        from skip_counterfactual import counterfactual_return

        already = set(self.ledger.decision_ids_with_event("outcome"))
        cache: Dict[str, Optional[float]] = {}
        for snap in self.ledger.snapshots(accepted=False, kind="candidate"):
            if snap["decision_id"] in already:
                continue
            summary["checked"] += 1
            ts = _parse_iso(snap.get("event_at_utc"))
            if ts is None or (self.now - ts).total_seconds() / 60.0 < horizon_min:
                summary["not_due"] += 1
                continue
            entry = _f(snap.get("underlying_price"))
            if entry is None or entry <= 0:
                summary["no_entry_price"] += 1
                continue
            und = snap.get("underlying")
            if und not in cache:
                try:
                    cache[und] = self.price_fn(und)
                except Exception:
                    cache[und] = None
            now_px = cache[und]
            cf = counterfactual_return(snap.get("direction"), entry, now_px)
            if cf is None:
                summary["no_price"] += 1
                continue
            if self._emit(
                decision_id=snap["decision_id"], event_type="outcome",
                price=now_px,
                event_at=self.now.isoformat(),
                payload={"class": "simulated", "policy": SKIP_POLICY,
                         "horizon_min": horizon_min,
                         "entry_underlying_price": entry,
                         "counterfactual_pnl_pct": cf},
            ):
                summary["resolved"] += 1
        return summary

    # --------------------------------------------------------- classification
    def classify(self) -> Dict:
        """Count decisions by outcome class: realized / simulated / unresolved."""
        import json as _json
        realized, simulated = set(), set()
        for row in self.ledger._rows(
                "SELECT decision_id, event_type, payload_json FROM events "
                "WHERE event_type IN ('outcome','expiry','assignment')"):
            cls = None
            try:
                cls = (_json.loads(row["payload_json"] or "{}") or {}).get("class")
            except ValueError:
                pass
            if cls == "simulated":
                simulated.add(row["decision_id"])
            else:
                realized.add(row["decision_id"])
        return {
            "realized": len(realized),
            "simulated": len(simulated),
            "unresolved": len(self.ledger.unresolved()),
        }

    def run_all(self, activities_after: Optional[str] = None,
                skip_horizon_min: float = 390.0) -> Dict:
        out = {"apply": self.apply}
        out["orders"] = self.reconcile_orders()
        if activities_after:
            out["activities"] = self.reconcile_activities(activities_after)
        out["expired"] = self.reconcile_expired()
        out["skipped"] = self.resolve_skipped(horizon_min=skip_horizon_min)
        out["classes"] = self.classify()
        return out


# --------------------------------------------------------------------------- #
# Real Alpaca fetchers (creds required) — read-only endpoints, never orders
# --------------------------------------------------------------------------- #
def _load_credentials():
    """Resolve (base_url, headers) from shell env / .env via ConfigLoader."""
    from config_loader import ConfigLoader
    env = ConfigLoader(path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env"))
    key = env.get_str("ALPACA_API_KEY")
    secret = env.get_str("ALPACA_SECRET_KEY")
    paper = env.get_bool("ALPACA_PAPER", True)
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    base_url = ("https://paper-api.alpaca.markets" if paper
                else "https://api.alpaca.markets")
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return base_url, headers


def build_alpaca_fetchers():
    """(fetch_order, fetch_activities) bound to GET-only Alpaca endpoints."""
    import requests
    base_url, headers = _load_credentials()

    def fetch_order(order_id: str) -> Optional[Dict]:
        r = requests.get(f"{base_url}/v2/orders/{order_id}",
                         headers=headers, timeout=30)
        return r.json() if r.status_code == 200 else None

    def fetch_activities(after_iso: str) -> List[Dict]:
        out, page_token = [], None
        while True:
            params = {"activity_types": "FILL", "after": after_iso,
                      "direction": "asc", "page_size": 100}
            if page_token:
                params["page_token"] = page_token
            r = requests.get(f"{base_url}/v2/account/activities",
                             headers=headers, params=params, timeout=30)
            if r.status_code != 200:
                raise RuntimeError(f"activities HTTP {r.status_code}")
            batch = r.json()
            if not batch:
                break
            out.extend(batch)
            if len(batch) < 100:
                break
            page_token = batch[-1].get("id")
            if not page_token:
                break
        return out

    return fetch_order, fetch_activities


def _stock_price_fn(symbol: str) -> Optional[float]:
    """Latest trade price for the skip counterfactual (data API, read-only)."""
    import requests
    from config_loader import ConfigLoader
    env = ConfigLoader(path=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), ".env"))
    headers = {"APCA-API-KEY-ID": env.get_str("ALPACA_API_KEY"),
               "APCA-API-SECRET-KEY": env.get_str("ALPACA_SECRET_KEY")}
    r = requests.get(
        f"https://data.alpaca.markets/v2/stocks/{symbol}/trades/latest",
        headers=headers, timeout=15)
    if r.status_code != 200:
        return None
    return _f(((r.json() or {}).get("trade") or {}).get("p"))


def main(argv=None) -> int:
    import argparse
    import json as _json

    ap = argparse.ArgumentParser(
        description="Reconcile research_ledger.db outcomes against the broker "
                    "(dry-run by default; --apply writes).")
    ap.add_argument("--db", default="research_ledger.db")
    ap.add_argument("--apply", action="store_true",
                    help="write events (default: dry-run, print plan only)")
    ap.add_argument("--activities-after", default=None,
                    help="ISO timestamp; also sync the FILL activity stream")
    ap.add_argument("--skip-horizon-min", type=float, default=390.0)
    ap.add_argument("--no-network", action="store_true",
                    help="expiry + classification only (no broker calls)")
    args = ap.parse_args(argv)

    ledger = ResearchLedger(args.db)
    fetch_order = fetch_activities = price_fn = None
    if not args.no_network:
        try:
            fetch_order, fetch_activities = build_alpaca_fetchers()
            price_fn = _stock_price_fn
        except Exception as e:
            print(f"[RECONCILE] broker fetchers unavailable ({e}); "
                  f"running expiry/classification only")
    rec = OutcomeReconciler(ledger, fetch_order=fetch_order,
                            fetch_activities=fetch_activities,
                            price_fn=price_fn, apply=args.apply)
    result = rec.run_all(activities_after=args.activities_after,
                         skip_horizon_min=args.skip_horizon_min)
    print(_json.dumps(result, indent=2, default=str))
    if not args.apply and rec.planned:
        print(f"[DRY-RUN] {len(rec.planned)} event(s) planned; "
              f"re-run with --apply to write")
    ledger.close()
    return 0


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — injected stub fetchers)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import timedelta

    ok = True
    led = ResearchLedger(":memory:")
    now = datetime(2026, 11, 20, 16, 0, tzinfo=timezone.utc)

    # Decision A: accepted + submitted, broker says fully filled then exited.
    did_a = led.record_snapshot(accepted=True, symbol="SPY261127C00500000",
                                direction="CALL",
                                event_at="2026-11-19T15:00:00+00:00")
    led.record_event(decision_id=did_a, event_type="submitted",
                     broker_order_id="ord-A", qty=2,
                     event_at="2026-11-19T15:00:01+00:00")

    # Decision B: accepted + submitted, broker canceled with no fill.
    did_b = led.record_snapshot(accepted=True, symbol="QQQ261127P00400000",
                                direction="PUT",
                                event_at="2026-11-19T15:10:00+00:00")
    led.record_event(decision_id=did_b, event_type="submitted",
                     broker_order_id="ord-B", qty=1,
                     event_at="2026-11-19T15:10:01+00:00")

    # Decision C: accepted, contract already expired, never exited.
    did_c = led.record_snapshot(accepted=True, symbol="IWM261113C00200000",
                                direction="CALL",
                                event_at="2026-11-10T15:00:00+00:00")
    led.record_event(decision_id=did_c, event_type="submitted",
                     broker_order_id="ord-C", qty=1)

    # Decision D: rejected (risk-blocked) 2 days ago, underlying 100 -> 94.
    did_d = led.record_snapshot(
        accepted=False, reject_code="risk_blocked:test",
        symbol="DIA261127C00100000", direction="CALL", underlying_price=100.0,
        event_at=(now - timedelta(days=2)).isoformat())

    # Decision E: rejected but too recent — must stay untouched.
    did_e = led.record_snapshot(
        accepted=False, reject_code="risk_blocked:test",
        symbol="DIA261127P00100000", direction="PUT", underlying_price=100.0,
        event_at=(now - timedelta(minutes=5)).isoformat())

    orders = {
        "ord-A": {"status": "filled", "filled_qty": "2",
                  "filled_avg_price": "1.25",
                  "filled_at": "2026-11-19T15:00:05Z"},
        "ord-B": {"status": "canceled", "filled_qty": "0",
                  "filled_avg_price": None,
                  "canceled_at": "2026-11-19T15:12:00Z"},
        "ord-C": {"status": "filled", "filled_qty": "1",
                  "filled_avg_price": "0.80",
                  "filled_at": "2026-11-10T15:00:05Z"},
    }
    activities = [
        # Two distinct partial fills on ord-A with identical qty/price.
        {"id": "act-1", "order_id": "ord-A", "qty": "1", "price": "1.25",
         "type": "partial_fill", "transaction_time": "2026-11-19T15:00:03Z"},
        {"id": "act-2", "order_id": "ord-A", "qty": "1", "price": "1.25",
         "type": "partial_fill", "transaction_time": "2026-11-19T15:00:04Z"},
        {"id": "act-X", "order_id": "ord-UNKNOWN", "qty": "5", "price": "9"},
    ]
    prices = {"DIA": 94.0}

    # ---- dry-run writes nothing -------------------------------------------
    dry = OutcomeReconciler(led, fetch_order=orders.get,
                            fetch_activities=lambda a: activities,
                            price_fn=prices.get, apply=False, now=now)
    dry.run_all(activities_after="2026-11-19T00:00:00Z")
    if led.stats()["events"] != 3:  # only the 3 submitted events pre-existing
        print("FAIL: dry-run must not write", led.stats()); ok = False
    if not dry.planned:
        print("FAIL: dry-run should plan actions"); ok = False

    # ---- apply -------------------------------------------------------------
    rec = OutcomeReconciler(led, fetch_order=orders.get,
                            fetch_activities=lambda a: activities,
                            price_fn=prices.get, apply=True, now=now)
    res = rec.run_all(activities_after="2026-11-19T00:00:00Z")

    # Order sync: A + C filled, B canceled.
    if res["orders"]["fills"] != 2 or res["orders"]["cancels"] != 1:
        print("FAIL: order sync counts", res["orders"]); ok = False
    ev_a = [e["event_type"] for e in led.get_decision(did_a)["events"]]
    if "fill" not in ev_a:
        print("FAIL: A should have a fill event", ev_a); ok = False
    ev_b = [e["event_type"] for e in led.get_decision(did_b)["events"]]
    if "cancel" not in ev_b or "fill" in ev_b:
        print("FAIL: B should cancel without fill", ev_b); ok = False

    # Activities: both partial fills recorded (distinct activity ids),
    # unknown order ignored.
    if res["activities"]["recorded"] != 2 or res["activities"]["matched"] != 2:
        print("FAIL: activity sync", res["activities"]); ok = False

    # Expiry: C's 2026-11-13 contract is before 2026-11-20 -> terminal expiry.
    if res["expired"]["expired"] != 1:
        print("FAIL: expiry count", res["expired"]); ok = False
    if did_c in {r["decision_id"] for r in led.unresolved()}:
        print("FAIL: expiry must resolve C"); ok = False

    # Skips: D resolved as SIMULATED (CALL 100->94 = -6%), E not due.
    if res["skipped"]["resolved"] != 1 or res["skipped"]["not_due"] != 1:
        print("FAIL: skip resolution", res["skipped"]); ok = False
    import json as _json
    d_events = led.get_decision(did_d)["events"]
    oc = [e for e in d_events if e["event_type"] == "outcome"]
    pl = _json.loads(oc[0]["payload_json"])
    if pl["class"] != "simulated" or pl["policy"] != SKIP_POLICY:
        print("FAIL: skip outcome must be labeled simulated", pl); ok = False
    if abs(pl["counterfactual_pnl_pct"] - (-6.0)) > 1e-9:
        print("FAIL: counterfactual math", pl); ok = False
    if led.get_decision(did_e)["events"]:
        print("FAIL: fresh skip must stay untouched"); ok = False

    # Classification: realized = {C via expiry} + A/B have no outcome yet ->
    # A and B are still unresolved (fill/cancel are not terminal; exit/outcome
    # comes from the live close hook or expiry).
    cls = res["classes"]
    if cls["simulated"] != 1 or cls["realized"] != 1 or cls["unresolved"] != 2:
        print("FAIL: classification", cls); ok = False

    # ---- idempotency: full re-run adds nothing -----------------------------
    before = led.stats()["events"]
    rec2 = OutcomeReconciler(led, fetch_order=orders.get,
                             fetch_activities=lambda a: activities,
                             price_fn=prices.get, apply=True, now=now)
    rec2.run_all(activities_after="2026-11-19T00:00:00Z")
    after = led.stats()["events"]
    if before != after:
        print(f"FAIL: re-run must be idempotent ({before} -> {after})"); ok = False

    # (restart recovery against a real on-disk DB is covered by the file-based
    # pytest case; the deterministic event_key makes any re-run a no-op.)

    led.close()
    print("research_outcomes self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv:
        sys.exit(_self_test())
    sys.exit(main())
