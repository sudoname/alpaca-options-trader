"""Point-in-time Kalshi snapshot collector (Oracle 4.0.2 P1-6).

Fetches open markets for configured series and appends one timestamped
snapshot row per market into a sqlite store.

Rules:
  * research-path only; strictly read-only against Kalshi (public GETs);
  * **settlement results are never stored**: the snapshot schema has no
    ``result`` column, so settled outcomes structurally cannot leak into
    any feature built from this store;
  * point-in-time: ``observed_at`` is the collection wall-clock (UTC);
    a feature built for time t may only read rows with
    ``observed_at <= t`` (enforced by the feature builder's SQL);
  * keep-first primary key (ticker, observed_at) makes reruns idempotent;
  * prices arrive in cents (0-100) and are stored as probabilities
    (0.0-1.0); non-finite/out-of-range values become NULL, never
    repaired.

Usage:
    python -m oracle.ml.kalshi.collector \
        --series KXINXU KXNASDAQ100U \
        --db artifacts/oracle4/kalshi/kalshi.sqlite
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence

from .client import (DEFAULT_BASE_URL, DEFAULT_TIMEOUT, KalshiUnavailable,
                     get_markets)
from .market_mapping import series_of, underlying_for_ticker

UTC = timezone.utc

DEFAULT_DB = os.path.join("artifacts", "oracle4", "kalshi", "kalshi.sqlite")
SOURCE = "kalshi_public_v2"

#: snapshot schema -- deliberately NO settlement/result column.
_DDL = """
CREATE TABLE IF NOT EXISTS snapshots (
    ticker TEXT NOT NULL,
    observed_at TEXT NOT NULL,
    series TEXT NOT NULL,
    event_ticker TEXT,
    underlying TEXT,
    status TEXT,
    yes_bid REAL,
    yes_ask REAL,
    last_price REAL,
    volume REAL,
    open_interest REAL,
    floor_strike REAL,
    cap_strike REAL,
    strike_type TEXT,
    close_time TEXT,
    source TEXT NOT NULL,
    PRIMARY KEY (ticker, observed_at));
CREATE INDEX IF NOT EXISTS ix_snapshots_series_t
    ON snapshots(series, observed_at);
CREATE INDEX IF NOT EXISTS ix_snapshots_underlying_t
    ON snapshots(underlying, observed_at);
CREATE TABLE IF NOT EXISTS ingest_manifests (
    source TEXT NOT NULL,
    ingested_at TEXT NOT NULL,
    manifest_json TEXT NOT NULL,
    PRIMARY KEY (source, ingested_at));
"""


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _prob(v) -> Optional[float]:
    """Cents (0-100) -> probability (0-1); bad values -> None."""
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(f) or f < 0.0 or f > 100.0:
        return None
    return f / 100.0


def _opt_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _opt_str(v) -> Optional[str]:
    s = str(v).strip() if v is not None else ""
    return s or None


def collect_snapshots(
    series_tickers: Sequence[str],
    *,
    db_path: str = DEFAULT_DB,
    base_url: str = DEFAULT_BASE_URL,
    api_key: Optional[str] = None,
    status: Optional[str] = "open",
    now: Optional[datetime] = None,
    timeout: float = DEFAULT_TIMEOUT,
) -> Dict:
    """Fetch open markets for each series and append one snapshot each.

    Returns a summary dict: {series, markets_fetched, inserted,
    dropped_duplicate, skipped_no_quote, errors, observed_at}.
    Per-series fetch failures are recorded in ``errors`` (fail-open for
    the remaining series); a store error raises.
    """
    now = datetime.now(UTC) if now is None else (
        now if now.tzinfo else now.replace(tzinfo=UTC)).astimezone(UTC)
    observed_at = _iso_z(now)

    series = sorted({str(s).strip().upper() for s in series_tickers
                     if s and str(s).strip()})
    summary: Dict = {
        "series": len(series),
        "markets_fetched": 0,
        "inserted": 0,
        "dropped_duplicate": 0,
        "skipped_no_quote": 0,
        "errors": {},
        "observed_at": observed_at,
    }

    rows: List[tuple] = []
    for st in series:
        try:
            markets = get_markets(base_url=base_url, series_ticker=st,
                                  status=status, api_key=api_key,
                                  timeout=timeout)
        except KalshiUnavailable as exc:
            summary["errors"][st] = f"{exc.reason}: {exc.detail}"
            continue
        for m in markets:
            summary["markets_fetched"] += 1
            ticker = _opt_str(m.get("ticker"))
            if not ticker:
                continue
            yes_bid = _prob(m.get("yes_bid"))
            yes_ask = _prob(m.get("yes_ask"))
            last_price = _prob(m.get("last_price"))
            if yes_bid is None and yes_ask is None and last_price is None:
                summary["skipped_no_quote"] += 1
                continue
            rows.append((
                ticker.upper(), observed_at, series_of(ticker),
                _opt_str(m.get("event_ticker")),
                underlying_for_ticker(ticker),
                _opt_str(m.get("status")),
                yes_bid, yes_ask, last_price,
                _opt_float(m.get("volume")),
                _opt_float(m.get("open_interest")),
                _opt_float(m.get("floor_strike")),
                _opt_float(m.get("cap_strike")),
                _opt_str(m.get("strike_type")),
                _opt_str(m.get("close_time")),
                SOURCE,
            ))

    d = os.path.dirname(os.path.abspath(db_path))
    if d:
        os.makedirs(d, exist_ok=True)
    conn = sqlite3.connect(db_path)
    try:
        conn.executescript(_DDL)
        cur = conn.executemany(
            "INSERT OR IGNORE INTO snapshots VALUES "
            "(?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)", rows)
        summary["inserted"] = cur.rowcount if cur.rowcount >= 0 else 0
        summary["dropped_duplicate"] = len(rows) - summary["inserted"]
        if summary["inserted"] > 0:
            conn.execute(
                "INSERT OR REPLACE INTO ingest_manifests VALUES (?,?,?)",
                (SOURCE, observed_at, json.dumps({
                    "manifest_version": 1,
                    "stage": "oracle4.0.2-kalshi-collect",
                    "base_url": base_url,
                    "series": series,
                    "status_filter": status,
                    "observed_at": observed_at,
                    "settlement_note": ("snapshot schema stores NO "
                                        "settlement/result field; "
                                        "settled outcomes can never leak "
                                        "into features from this store"),
                    "counts": {k: summary[k] for k in (
                        "markets_fetched", "inserted", "dropped_duplicate",
                        "skipped_no_quote")},
                }, sort_keys=True)))
        conn.commit()
    finally:
        conn.close()
    return summary


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oracle.ml.kalshi.collector",
        description=("Collect point-in-time Kalshi market snapshots "
                     "(research-only, read-only, no settlement data)."))
    ap.add_argument("--series", nargs="+", required=True,
                    help="Kalshi series tickers, e.g. KXINXU KXNASDAQ100U")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--base-url", default=DEFAULT_BASE_URL)
    ap.add_argument("--status", default="open")
    args = ap.parse_args(argv)

    from config_loader import ConfigLoader
    api_key = ConfigLoader().get_str("KALSHI_API_KEY") or None

    summary = collect_snapshots(args.series, db_path=args.db,
                                base_url=args.base_url, api_key=api_key,
                                status=args.status)
    for k in ("series", "markets_fetched", "inserted", "dropped_duplicate",
              "skipped_no_quote", "observed_at"):
        print(f"{k}={summary[k]}")
    for st, err in summary["errors"].items():
        print(f"error[{st}]={err}")
    return 0 if not summary["errors"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
