#!/usr/bin/env python3
"""Pull all trading data from Alpaca + local state into a dated export folder.

Read-only. Writes CSV + JSON snapshots for offline analysis:
  - account_summary.csv/json     account snapshot (equity, cash, buying power...)
  - positions_open.csv           current open positions (with greeks where present)
  - orders_all.csv               every order ever (paginated, newest->oldest)
  - account_activities.csv       every fill/activity (FILL, DIV, etc., paginated)
  - portfolio_history.csv        daily equity / P&L curve (max window)
  - local_state/                 copies of local trade JSON/db for joining
"""
import os
import csv
import json
import shutil
import datetime as dt

import requests
from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("ALPACA_API_KEY")
SECRET = os.getenv("ALPACA_SECRET_KEY")
PAPER = os.getenv("ALPACA_PAPER", "true").lower() == "true"
BASE = "https://paper-api.alpaca.markets" if PAPER else "https://api.alpaca.markets"
HEADERS = {"APCA-API-KEY-ID": API_KEY, "APCA-API-SECRET-KEY": SECRET}

if not API_KEY or not SECRET:
    raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")

STAMP = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
OUTDIR = os.path.join("trade_export", f"export_{STAMP}")
os.makedirs(OUTDIR, exist_ok=True)


def get(path, params=None):
    r = requests.get(BASE + path, headers=HEADERS, params=params or {}, timeout=30)
    r.raise_for_status()
    return r.json()


def write_csv(name, rows):
    path = os.path.join(OUTDIR, name)
    if not rows:
        open(path, "w").close()
        print(f"  {name}: 0 rows")
        return
    keys = []
    for row in rows:
        for k in row:
            if k not in keys:
                keys.append(k)
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: row.get(k, "") for k in keys})
    print(f"  {name}: {len(rows)} rows")


def export_account():
    acct = get("/v2/account")
    with open(os.path.join(OUTDIR, "account_summary.json"), "w") as f:
        json.dump(acct, f, indent=2)
    write_csv("account_summary.csv", [acct])
    return acct


def export_positions():
    rows = get("/v2/positions")
    write_csv("positions_open.csv", rows)
    return rows


def export_orders():
    """Paginate all orders newest->oldest via submitted before cursor."""
    rows = []
    until = None
    seen = set()
    while True:
        params = {"status": "all", "direction": "desc", "limit": 500, "nested": "false"}
        if until:
            params["until"] = until
        batch = get("/v2/orders", params)
        if not batch:
            break
        new = [o for o in batch if o["id"] not in seen]
        if not new:
            break
        for o in new:
            seen.add(o["id"])
        rows.extend(new)
        until = batch[-1]["submitted_at"]
        if len(batch) < 500:
            break
    write_csv("orders_all.csv", rows)
    with open(os.path.join(OUTDIR, "orders_all.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def export_activities():
    """Paginate all account activities via page_token."""
    rows = []
    page_token = None
    while True:
        params = {"page_size": 100}
        if page_token:
            params["page_token"] = page_token
        batch = get("/v2/account/activities", params)
        if not batch:
            break
        rows.extend(batch)
        page_token = batch[-1].get("id")
        if len(batch) < 100:
            break
    write_csv("account_activities.csv", rows)
    with open(os.path.join(OUTDIR, "account_activities.json"), "w") as f:
        json.dump(rows, f, indent=2)
    return rows


def export_portfolio_history():
    try:
        hist = get("/v2/account/portfolio/history",
                   {"period": "all", "timeframe": "1D", "extended_hours": "false"})
    except requests.HTTPError:
        hist = get("/v2/account/portfolio/history",
                   {"period": "1A", "timeframe": "1D"})
    ts = hist.get("timestamp", [])
    eq = hist.get("equity", [])
    pl = hist.get("profit_loss", [])
    plp = hist.get("profit_loss_pct", [])
    rows = []
    for i, t in enumerate(ts):
        rows.append({
            "timestamp": dt.datetime.utcfromtimestamp(t).strftime("%Y-%m-%d %H:%M:%S"),
            "equity": eq[i] if i < len(eq) else "",
            "profit_loss": pl[i] if i < len(pl) else "",
            "profit_loss_pct": plp[i] if i < len(plp) else "",
        })
    write_csv("portfolio_history.csv", rows)
    return rows


def copy_local_state():
    dest = os.path.join(OUTDIR, "local_state")
    os.makedirs(dest, exist_ok=True)
    patterns = [
        "active_trades.json", "trading_history.json", "realized_pnl_log.json",
        "day_trades_log.json", "telegram_trades.json", "schwab_trades.json",
        "spy_qqq_hybrid_trades.json", "candidate_resolutions.json",
        "candidate_resolution.jsonl", "scheduler_status.json",
        "advisory_attribution.json", "episodes.db",
    ]
    copied = []
    for p in patterns:
        if os.path.exists(p):
            shutil.copy2(p, os.path.join(dest, p))
            copied.append(p)
    print(f"  local_state: copied {len(copied)} files -> {copied}")
    return copied


def main():
    print(f"Exporting Alpaca {'PAPER' if PAPER else 'LIVE'} data -> {OUTDIR}\n")
    acct = export_account()
    positions = export_positions()
    orders = export_orders()
    acts = export_activities()
    hist = export_portfolio_history()
    copy_local_state()

    fills = [a for a in acts if a.get("activity_type") == "FILL"]
    manifest = {
        "exported_at": dt.datetime.now().isoformat(),
        "mode": "paper" if PAPER else "live",
        "account_number": acct.get("account_number"),
        "equity": acct.get("equity"),
        "cash": acct.get("cash"),
        "buying_power": acct.get("buying_power"),
        "counts": {
            "open_positions": len(positions),
            "orders": len(orders),
            "activities": len(acts),
            "fills": len(fills),
            "portfolio_history_points": len(hist),
        },
    }
    with open(os.path.join(OUTDIR, "MANIFEST.json"), "w") as f:
        json.dump(manifest, f, indent=2)
    print("\n=== MANIFEST ===")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
