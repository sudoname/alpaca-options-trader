"""
Fills-aware reconcile for ``realized_pnl_log.json`` (operator tool).

Why this exists
---------------
The kill-switch reads today's realized P/L from ``realized_pnl_log.json`` via
``RealizedPnLTracker.get_today_realized()``. When that log drifts from reality
(e.g. an EOD/external close that wasn't recorded, or a manual mid-day "reset"
that blindly *wiped* the day's entries and silently dropped real closed losses),
the kill-switch sees the wrong number.

``RealizedPnLTracker.reset_today()`` is a blunt instrument: it deletes today's
rows, assuming a fresh day starts flat. That is only correct *before* any close.
Used mid-day it erases booked P/L from trades that really closed.

This tool replaces the blind wipe with a **fills-aware reconcile**: it re-derives
the target day's realized P/L from the broker's FILL activities (the source of
truth) and rewrites only that day's rows, preserving every other day untouched.

Realized math (per symbol that has a SELL on the target day):
    avg_buy = sum(buy_price * buy_qty) / sum(buy_qty)          # over the window
    realized = (sum(sell_price*qty over day) - avg_buy * sell_qty_day) * 100

The ``* 100`` is the option contract multiplier; activity prices are per-contract.

Safety
------
* Default mode is ``--dry-run``: prints the diff, writes nothing.
* ``--apply`` backs up the existing log to ``<log>.bak.<epoch>`` before writing.
* Pure functions (``realized_by_symbol``/``build_rows``) take no network and are
  unit-tested in ``test_reconcile_realized.py``.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta

from realized_pnl_tracker import RealizedPnLTracker

DEFAULT_LOG = "realized_pnl_log.json"
RECONCILE_SOURCE = "fills_reconcile"
OPTION_MULTIPLIER = 100.0

_SELL_SIDES = ("sell", "sell_to_close", "sell_to_open")
_BUY_SIDES = ("buy", "buy_to_open", "buy_to_close")


# --------------------------------------------------------------------------- #
# Pure computation (no network, no I/O) — unit-tested
# --------------------------------------------------------------------------- #
def realized_by_symbol(activities, day):
    """Realized dollar P/L per symbol for symbols with a SELL on ``day``.

    ``activities`` is a list of Alpaca FILL activity dicts (any window covering
    the relevant buys). ``day`` is an ISO date string ("YYYY-MM-DD").

    Returns ``(realized, last_sell_ts)`` where ``realized`` maps symbol -> dollar
    P/L (rounded to 2dp) and ``last_sell_ts`` maps symbol -> the newest sell
    ``transaction_time`` seen on ``day`` (for the log row's timestamp).

    Fail-open on a per-row basis: malformed rows are skipped, never fatal.
    """
    buy_qty = {}
    buy_cost = {}          # sum(price * qty) over all buys in the window
    sell_day_qty = {}
    sell_day_proceeds = {}  # sum(price * qty) for sells dated `day`
    last_sell_ts = {}

    for a in activities or []:
        try:
            sym = a.get("symbol")
            side = (a.get("side") or "").lower()
            qty = float(a.get("qty") or 0)
            price = float(a.get("price") or 0)
        except (TypeError, ValueError, AttributeError):
            continue
        if not sym or qty <= 0 or price <= 0:
            continue
        ts = str(a.get("transaction_time") or "")

        if side in _BUY_SIDES:
            buy_qty[sym] = buy_qty.get(sym, 0.0) + qty
            buy_cost[sym] = buy_cost.get(sym, 0.0) + price * qty
        elif side in _SELL_SIDES and ts.startswith(day):
            sell_day_qty[sym] = sell_day_qty.get(sym, 0.0) + qty
            sell_day_proceeds[sym] = sell_day_proceeds.get(sym, 0.0) + price * qty
            if ts > str(last_sell_ts.get(sym, "")):
                last_sell_ts[sym] = ts

    realized = {}
    for sym, sqty in sell_day_qty.items():
        if sqty <= 0:
            continue
        bq = buy_qty.get(sym, 0.0)
        if bq <= 0:
            # No buy in the window to cost against; can't derive P/L → skip.
            continue
        avg_buy = buy_cost[sym] / bq
        pnl = (sell_day_proceeds[sym] - avg_buy * sqty) * OPTION_MULTIPLIER
        realized[sym] = round(pnl, 2)
    return realized, last_sell_ts


def build_rows(realized, last_sell_ts, day):
    """Build ``realized_pnl_log`` rows for ``day`` from a realized map.

    Mirrors ``RealizedPnLTracker`` row shape and tags ``source`` so reconciled
    rows are distinguishable from live-path bookings.
    """
    rows = []
    for sym in sorted(realized):
        ts = last_sell_ts.get(sym) or f"{day}T00:00:00"
        rows.append({
            "date": day,
            "timestamp": ts,
            "amount": realized[sym],
            "symbol": sym,
            "source": RECONCILE_SOURCE,
        })
    return rows


def merge_log(existing, day, new_day_rows):
    """Return existing rows for other days + the rebuilt rows for ``day``."""
    preserved = [r for r in (existing or []) if str(r.get("date")) != day]
    return preserved + new_day_rows


# --------------------------------------------------------------------------- #
# Network (creds required) — not unit-tested
# --------------------------------------------------------------------------- #
def _load_credentials():
    """Resolve (base_url, headers) from shell env / .env via ConfigLoader."""
    from config_loader import ConfigLoader
    env = ConfigLoader(path=os.path.join(os.path.dirname(os.path.abspath(__file__)), ".env"))
    key = env.get_str("ALPACA_API_KEY")
    secret = env.get_str("ALPACA_SECRET_KEY")
    paper = env.get_bool("ALPACA_PAPER", True)
    if not key or not secret:
        raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    base_url = "https://paper-api.alpaca.markets" if paper else "https://api.alpaca.markets"
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    return base_url, headers


def _paginate_fills(base_url, headers, after_iso):
    """Pull all FILL activities at/after ``after_iso`` (ascending), paginated."""
    import requests
    out = []
    page_token = None
    while True:
        params = {"activity_types": "FILL", "after": after_iso,
                  "direction": "asc", "page_size": 100}
        if page_token:
            params["page_token"] = page_token
        r = requests.get(f"{base_url}/v2/account/activities",
                         headers=headers, params=params, timeout=30)
        if r.status_code != 200:
            raise RuntimeError(f"activities HTTP {r.status_code}: {r.text[:200]}")
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


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def reconcile(day, lookback_days, log_file, apply):
    """Reconcile ``log_file``'s rows for ``day`` from broker fills.

    Returns a result dict with the computed total, per-symbol breakdown, the
    pre-existing today-total, and whether changes were written.
    """
    base_url, headers = _load_credentials()
    after_iso = (datetime.fromisoformat(day) - timedelta(days=lookback_days)).date().isoformat()
    activities = _paginate_fills(base_url, headers, after_iso)

    realized, last_sell_ts = realized_by_symbol(activities, day)
    new_rows = build_rows(realized, last_sell_ts, day)
    new_total = round(sum(realized.values()), 2)

    tracker = RealizedPnLTracker(log_file)
    existing = tracker._load()
    prior_day_total = round(
        sum(float(r.get("amount", 0) or 0) for r in existing
            if str(r.get("date")) == day), 2)

    result = {
        "day": day, "fills_pulled": len(activities),
        "symbols": len(realized), "new_total": new_total,
        "prior_day_total": prior_day_total, "rows": new_rows,
        "applied": False, "backup": None,
    }

    if apply:
        backup = None
        if os.path.exists(log_file):
            backup = f"{log_file}.bak.{int(time.time())}"
            with open(log_file, "r") as f:
                _orig = f.read()
            with open(backup, "w") as f:
                f.write(_orig)
        merged = merge_log(existing, day, new_rows)
        with open(log_file, "w") as f:
            json.dump(merged, f, indent=2)
        result["applied"] = True
        result["backup"] = backup
        # Verify via the same reader the kill-switch uses (only meaningful when
        # `day` is today; otherwise get_today_realized reflects the live day).
        result["verify_today_total"] = round(RealizedPnLTracker(log_file).get_today_realized(), 2)

    return result


def _print_result(result):
    print(f"[RECONCILE] day={result['day']} fills={result['fills_pulled']} "
          f"symbols={result['symbols']}")
    for r in result["rows"]:
        print(f"  {r['symbol']:<24} {r['amount']:>12.2f}  {r['timestamp']}")
    print(f"[RECONCILE] prior day-total = {result['prior_day_total']:.2f}")
    print(f"[RECONCILE] fills  day-total = {result['new_total']:.2f}")
    if result["applied"]:
        print(f"[RECONCILE] APPLIED. backup -> {result['backup']}")
        if "verify_today_total" in result:
            print(f"[RECONCILE] verify get_today_realized() = {result['verify_today_total']:.2f}")
    else:
        print("[RECONCILE] DRY-RUN (no changes written). Re-run with --apply to commit.")


def main(argv=None):
    p = argparse.ArgumentParser(description="Fills-aware reconcile of realized_pnl_log.json")
    p.add_argument("--date", default=datetime.now().date().isoformat(),
                   help="Target day YYYY-MM-DD (default: today, server-local)")
    p.add_argument("--lookback-days", type=int, default=14,
                   help="How far back to pull buy fills for cost basis (default 14)")
    p.add_argument("--log-file", default=DEFAULT_LOG, help="Path to realized_pnl_log.json")
    g = p.add_mutually_exclusive_group()
    g.add_argument("--dry-run", action="store_true", default=True,
                   help="Show the diff without writing (default)")
    g.add_argument("--apply", action="store_true",
                   help="Write the reconciled rows (backs up the log first)")
    p.add_argument("--selftest", action="store_true", help="Run offline self-test and exit")
    args = p.parse_args(argv)

    if args.selftest:
        return _self_test()

    try:
        result = reconcile(args.date, args.lookback_days, args.log_file, apply=args.apply)
    except Exception as e:
        print(f"[RECONCILE] ERROR: {e}")
        return 1
    _print_result(result)
    return 0


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test():
    ok = True
    day = "2026-06-09"

    # Single symbol: buy 3 @ 24.9, sell 3 @ 22.15 on `day`.
    acts = [
        {"symbol": "UNH...C", "side": "buy", "qty": 3, "price": 24.9,
         "transaction_time": "2026-06-02T15:00:00Z"},
        {"symbol": "UNH...C", "side": "sell", "qty": 3, "price": 22.15,
         "transaction_time": f"{day}T19:45:23Z"},
    ]
    realized, ts = realized_by_symbol(acts, day)
    want = round((22.15 - 24.9) * 3 * 100, 2)
    if abs(realized.get("UNH...C", 0) - want) > 1e-6:
        print("FAIL: single-symbol realized", realized); ok = False
    if ts.get("UNH...C") != f"{day}T19:45:23Z":
        print("FAIL: last_sell_ts", ts); ok = False

    # Split fills: two 1-lot sells weighted-averaged vs avg buy.
    acts2 = [
        {"symbol": "HD...C", "side": "buy", "qty": 2, "price": 20.6,
         "transaction_time": "2026-06-01T15:00:00Z"},
        {"symbol": "HD...C", "side": "sell", "qty": 1, "price": 18.2,
         "transaction_time": f"{day}T19:45:23Z"},
        {"symbol": "HD...C", "side": "sell", "qty": 1, "price": 18.2,
         "transaction_time": f"{day}T19:45:24Z"},
    ]
    r2, _ = realized_by_symbol(acts2, day)
    want2 = round((18.2 * 2 - 20.6 * 2) * 100, 2)
    if abs(r2.get("HD...C", 0) - want2) > 1e-6:
        print("FAIL: split-fill realized", r2); ok = False

    # Sell on a different day must NOT count toward `day`.
    acts3 = [
        {"symbol": "X...C", "side": "buy", "qty": 1, "price": 10.0,
         "transaction_time": "2026-06-01T15:00:00Z"},
        {"symbol": "X...C", "side": "sell", "qty": 1, "price": 12.0,
         "transaction_time": "2026-06-08T19:00:00Z"},
    ]
    r3, _ = realized_by_symbol(acts3, day)
    if r3:
        print("FAIL: other-day sell leaked into day", r3); ok = False

    # No buy in window → can't cost; skip rather than guess.
    acts4 = [{"symbol": "Y...C", "side": "sell", "qty": 1, "price": 5.0,
              "transaction_time": f"{day}T19:00:00Z"}]
    r4, _ = realized_by_symbol(acts4, day)
    if r4:
        print("FAIL: sell with no buy should be skipped", r4); ok = False

    # build_rows shape + tag.
    rows = build_rows({"A": -10.0, "B": 5.0}, {"A": f"{day}T19:00:00Z"}, day)
    if [r["symbol"] for r in rows] != ["A", "B"]:
        print("FAIL: build_rows ordering", rows); ok = False
    if any(r["source"] != RECONCILE_SOURCE for r in rows):
        print("FAIL: build_rows source tag", rows); ok = False
    if rows[1]["timestamp"] != f"{day}T00:00:00":
        print("FAIL: build_rows default ts", rows); ok = False

    # merge_log preserves other days, replaces target day.
    existing = [
        {"date": "2026-06-08", "amount": -50.0, "symbol": "OLD"},
        {"date": day, "amount": 999.0, "symbol": "STALE"},
    ]
    merged = merge_log(existing, day, rows)
    if any(r.get("symbol") == "STALE" for r in merged):
        print("FAIL: stale target-day row not replaced", merged); ok = False
    if not any(r.get("symbol") == "OLD" for r in merged):
        print("FAIL: other-day row not preserved", merged); ok = False

    print("reconcile_realized self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
