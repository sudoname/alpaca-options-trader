"""Read-only Alpaca FILL/FEE/option-terminal archive for forward reconciliation.

Poll the account activities GET endpoint with overlapping lookback and stable
activity IDs. Fees can arrive the following UTC day and may lack order IDs;
retain them rather than guessing which trade paid them. No order API is used.

    python broker_activity_capture.py --account paper --once
    python broker_activity_capture.py --account live --once --after 2026-11-20T00:00:00Z

The output SQLite + CSV are private account data. Keep them out of git.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sqlite3
import urllib.error
import urllib.parse
import urllib.request
from datetime import datetime, timedelta, timezone

from quote_capture import _load_env

TYPES = ("FILL", "FEE", "OPEXP", "OPASN")
CSV_FIELDS = ("id", "activity_type", "order_id", "symbol", "side", "qty",
              "price", "net_amount", "transaction_time", "created_at", "date", "type")
BASE_URLS = {"paper": "https://paper-api.alpaca.markets",
             "live": "https://api.alpaca.markets"}


def _utc(value):
    dt = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("--after must include a timezone")
    return dt.astimezone(timezone.utc).isoformat()


def _open_store(path, account):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    conn = sqlite3.connect(path)
    conn.execute("CREATE TABLE IF NOT EXISTS meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
    conn.execute("CREATE TABLE IF NOT EXISTS activities ("
                 "id TEXT PRIMARY KEY, activity_type TEXT NOT NULL, order_id TEXT, "
                 "received_at TEXT NOT NULL, payload_json TEXT NOT NULL)")
    scope = conn.execute("SELECT value FROM meta WHERE key='account_scope'").fetchone()
    if scope and scope[0] != account:
        conn.close()
        raise ValueError("account scope mismatch: keep paper and live archives separate")
    conn.execute("INSERT OR IGNORE INTO meta VALUES ('account_scope', ?)", (account,))
    conn.commit()
    return conn


def fetch_page_http(account, after, page_token=None, *, opener=None):
    """Single read-only GET page; credentials are never written or logged."""
    key = os.environ.get("ALPACA_API_KEY")
    secret = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise RuntimeError("Alpaca API credentials unavailable")
    params = {"activity_types": ",".join(TYPES), "after": after,
              "direction": "asc", "page_size": "100"}
    if page_token:
        params["page_token"] = page_token
    url = BASE_URLS[account] + "/v2/account/activities?" + urllib.parse.urlencode(params)
    req = urllib.request.Request(url, method="GET", headers={
        "APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret,
        "Accept": "application/json"})
    try:
        with (opener or urllib.request.urlopen)(req, timeout=20) as response:
            rows = json.load(response)
    except urllib.error.HTTPError as exc:
        raise RuntimeError(f"account activity GET failed: HTTP {exc.code}") from None
    if not isinstance(rows, list):
        raise ValueError("account activity response was not a list")
    return rows


def collect_once(conn, fetch_page, *, after, max_pages=100):
    """Store exact activity IDs once; fail closed on incomplete pagination."""
    after = _utc(after)
    token = None
    seen_tokens = set()
    stats = {"pages": 0, "seen": 0, "inserted": 0,
             "missing_id": 0, "fee_without_order_id": 0}
    for _ in range(max_pages):
        rows = fetch_page(after, token)
        if not isinstance(rows, list):
            raise ValueError("invalid activity page")
        stats["pages"] += 1
        for row in rows:
            if not isinstance(row, dict) or row.get("activity_type") not in TYPES:
                continue
            stats["seen"] += 1
            aid = row.get("id")
            if not aid:
                stats["missing_id"] += 1
                continue
            if row["activity_type"] == "FEE" and not row.get("order_id"):
                stats["fee_without_order_id"] += 1
            cur = conn.execute(
                "INSERT OR IGNORE INTO activities VALUES (?,?,?,?,?)",
                (str(aid), row["activity_type"], row.get("order_id"),
                 datetime.now(timezone.utc).isoformat(),
                 json.dumps(row, sort_keys=True, default=str)))
            stats["inserted"] += cur.rowcount
        conn.commit()
        if len(rows) < 100:
            stats["complete"] = True
            return stats
        last = rows[-1].get("id") if isinstance(rows[-1], dict) else None
        if not last or last == token or last in seen_tokens:
            raise RuntimeError("activity pagination stalled; archive is incomplete")
        seen_tokens.add(last)
        token = last
    raise RuntimeError("activity pagination page limit reached; archive is incomplete")


def export_csv(conn, path):
    """Stable CSV shape accepted by oracle.ml.broker_truth."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    temp = path + ".tmp"
    with open(temp, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=CSV_FIELDS, extrasaction="ignore")
        writer.writeheader()
        for (payload,) in conn.execute("SELECT payload_json FROM activities ORDER BY id"):
            writer.writerow(json.loads(payload))
    os.replace(temp, path)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--account", choices=tuple(BASE_URLS), required=True)
    ap.add_argument("--once", action="store_true", required=True)
    ap.add_argument("--after", help="UTC start; default seven-day overlap")
    ap.add_argument("--db", default="evidence_capture/broker_activities.sqlite")
    ap.add_argument("--csv", default="evidence_capture/account_activities.csv")
    args = ap.parse_args(argv)
    os.umask(0o077)  # account history and credentials must not become world-readable
    _load_env()
    after = args.after or (datetime.now(timezone.utc) - timedelta(days=7)).isoformat()
    conn = _open_store(args.db, args.account)
    try:
        stats = collect_once(conn, lambda stamp, token: fetch_page_http(
            args.account, stamp, token), after=after)
        export_csv(conn, args.csv)
    finally:
        conn.close()
    print(json.dumps({"account": args.account, "after": _utc(after), **stats},
                     sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
