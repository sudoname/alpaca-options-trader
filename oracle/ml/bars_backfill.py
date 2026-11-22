"""Multi-year, multi-symbol Alpaca bars backfill into the Stage-A BarStore.

Oracle 4.0.2 P1-5 (expand historical training data): downloads completed
intraday stock bars over an explicit historical window and upserts them
into the existing ``oracle.ml.bars.BarStore`` sqlite so the bars-based
dataset builder (``oracle.ml.bars_dataset``) has years of real underlying
history instead of a handful of live ledger sessions.

Rules (same contract as the live feeder ``bars_feed.py``):
  * research-path only -- nothing in the production trading path imports
    this module;
  * strictly read-only against the broker: market-data GETs only, no
    trading client is ever constructed;
  * **forming bars are never inserted**: ``BarStore.insert_bars`` is
    keep-first (INSERT OR IGNORE), so a partially formed close would be
    frozen forever.  A bar only exists here once ``ts_utc + resolution
    <= now``;
  * bars with a non-positive or non-finite close are dropped and counted,
    never repaired;
  * keep-first primary key (symbol, resolution_s, ts_utc) makes every run
    idempotent and restart-safe: re-running a window re-fetches but never
    rewrites, and overlap at chunk boundaries is deduplicated;
  * the backfill uses split-adjusted prices by default and a **distinct
    source tag** from the live raw feed, and (by default) a distinct
    resolution (5-minute vs the live feeder's 1-minute), so adjusted
    history and raw live bars are never mixed in one series.

Usage:
    python -m oracle.ml.bars_backfill \
        --symbols SPY QQQ IWM AAPL NVDA MSFT \
        --from 2023-01-01 --to 2026-01-01 \
        --db artifacts/oracle4/bars/bars.sqlite

Credentials come from .env / environment (ALPACA_API_KEY /
ALPACA_SECRET_KEY) via ``config_loader.ConfigLoader`` -- same pattern as
``fetch_underlying_bars.py``.  Never hard-code keys.
"""
from __future__ import annotations

import argparse
import math
import time
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from .bars import DEFAULT_DB, Bar, BarStore, is_rth, session_date_of

UTC = timezone.utc

#: supported bar resolutions (seconds) -> Alpaca timeframe strings.
#: Intraday only: daily bars have no well-defined RTH/session semantics
#: in this store and are rejected rather than guessed.
TIMEFRAME_BY_RESOLUTION: Dict[int, str] = {
    60: "1Min",
    300: "5Min",
    900: "15Min",
    3600: "1Hour",
}

DEFAULT_RESOLUTION_S = 300
DEFAULT_CHUNK_DAYS = 30
DEFAULT_BATCH_SYMBOLS = 20


def _http_get_json(url: str, headers: Dict[str, str], params: Dict,
                   timeout: float) -> Dict:
    """Thin HTTP wrapper; tests monkeypatch this symbol."""
    import requests

    resp = requests.get(url, headers=headers, params=params, timeout=timeout)
    resp.raise_for_status()
    return resp.json() or {}


def _parse_ts_utc(text) -> datetime:
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _opt_float(v) -> Optional[float]:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _as_utc(dt: datetime) -> datetime:
    if dt.tzinfo is None:
        return dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def backfill_bars(
    symbols: Sequence[str],
    *,
    headers: Dict[str, str],
    data_url: str,
    db_path: str,
    start: datetime,
    end: Optional[datetime] = None,
    resolution_s: int = DEFAULT_RESOLUTION_S,
    feed: str = "iex",
    adjustment: str = "split",
    chunk_days: int = DEFAULT_CHUNK_DAYS,
    batch_symbols: int = DEFAULT_BATCH_SYMBOLS,
    timeout: float = 30.0,
    page_limit: int = 10000,
    max_pages: int = 200,
    sleep_s: float = 0.0,
    source: Optional[str] = None,
    now: Optional[datetime] = None,
) -> Dict:
    """Fetch historical bars for ``symbols`` over [start, end) and upsert.

    Returns a summary dict:
      {symbols, batches, chunks, pages, fetched, inserted,
       dropped_duplicate_ts, skipped_forming, skipped_bad_close,
       source, resolution_s, window_start_utc, window_end_utc}
    Raises on fetch/store errors (a partial run is safe to re-run:
    keep-first insertion makes the backfill idempotent).
    """
    if resolution_s not in TIMEFRAME_BY_RESOLUTION:
        raise ValueError(
            f"unsupported resolution_s={resolution_s}; "
            f"supported: {sorted(TIMEFRAME_BY_RESOLUTION)}")
    timeframe = TIMEFRAME_BY_RESOLUTION[resolution_s]

    now = datetime.now(UTC) if now is None else _as_utc(now)
    start = _as_utc(start)
    end_eff = min(_as_utc(end), now) if end is not None else now
    if source is None:
        source = f"alpaca_{feed}_backfill_{adjustment}"

    syms = sorted({str(s).strip().upper() for s in symbols
                   if s and str(s).strip()})
    summary: Dict = {
        "symbols": len(syms),
        "batches": 0,
        "chunks": 0,
        "pages": 0,
        "fetched": 0,
        "inserted": 0,
        "dropped_duplicate_ts": 0,
        "skipped_forming": 0,
        "skipped_bad_close": 0,
        "source": source,
        "resolution_s": resolution_s,
        "window_start_utc": _iso_z(start),
        "window_end_utc": _iso_z(end_eff),
    }
    if not syms or start >= end_eff:
        return summary

    url = f"{data_url.rstrip('/')}/v2/stocks/bars"
    chunk = timedelta(days=int(chunk_days))

    with BarStore(db_path, readonly=False) as store:
        for b0 in range(0, len(syms), max(1, int(batch_symbols))):
            batch = syms[b0:b0 + max(1, int(batch_symbols))]
            summary["batches"] += 1
            cs = start
            while cs < end_eff:
                ce = min(cs + chunk, end_eff)
                summary["chunks"] += 1
                bars: List[Bar] = []
                base_params = {
                    "symbols": ",".join(batch),
                    "timeframe": timeframe,
                    "start": _iso_z(cs),
                    "end": _iso_z(ce),
                    "feed": feed,
                    "limit": page_limit,
                    "adjustment": adjustment,
                }
                page_token: Optional[str] = None
                for _ in range(max_pages):
                    params = dict(base_params)
                    if page_token:
                        params["page_token"] = page_token
                    payload = _http_get_json(url, headers, params,
                                             timeout) or {}
                    summary["pages"] += 1
                    for sym, raw_list in (payload.get("bars") or {}).items():
                        sym_u = str(sym).strip().upper()
                        for raw in raw_list or []:
                            summary["fetched"] += 1
                            try:
                                ts = _parse_ts_utc(raw.get("t"))
                                close = float(raw.get("c"))
                            except (TypeError, ValueError):
                                summary["skipped_bad_close"] += 1
                                continue
                            if not math.isfinite(close) or close <= 0:
                                summary["skipped_bad_close"] += 1
                                continue
                            # completed-bar rule: never insert a forming bar
                            if ts + timedelta(seconds=resolution_s) > now:
                                summary["skipped_forming"] += 1
                                continue
                            bars.append(Bar(
                                symbol=sym_u,
                                resolution_s=resolution_s,
                                ts_utc=ts,
                                open=_opt_float(raw.get("o")),
                                high=_opt_float(raw.get("h")),
                                low=_opt_float(raw.get("l")),
                                close=close,
                                volume=_opt_float(raw.get("v")),
                                session_date=session_date_of(ts),
                                rth=is_rth(ts),
                                source=source,
                            ))
                    page_token = payload.get("next_page_token")
                    if not page_token:
                        break
                counts = store.insert_bars(bars)
                summary["inserted"] += counts["inserted"]
                summary["dropped_duplicate_ts"] += \
                    counts["dropped_duplicate_ts"]
                cs = ce
                if sleep_s > 0:
                    time.sleep(sleep_s)
        if summary["inserted"] > 0:
            store.write_manifest(source, {
                "manifest_version": 1,
                "stage": "oracle4.0.2-backfill",
                "feed": feed,
                "adjustment": adjustment,
                "timeframe": timeframe,
                "resolution_s": resolution_s,
                "symbols": syms,
                "window_start_utc": _iso_z(start),
                "window_end_utc": _iso_z(end_eff),
                "bar_convention": ("timestamp = bar START, UTC; "
                                   f"resolution {resolution_s}s; "
                                   "completed bars only; "
                                   f"adjustment={adjustment}"),
                "counts": {k: summary[k] for k in (
                    "batches", "chunks", "pages", "fetched", "inserted",
                    "dropped_duplicate_ts", "skipped_forming",
                    "skipped_bad_close")},
            })
    return summary


def _parse_date(s: str) -> datetime:
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None \
        else dt.astimezone(UTC)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oracle.ml.bars_backfill",
        description=("Backfill historical Alpaca stock bars into the "
                     "Oracle BarStore (research-only, read-only vs broker)."))
    ap.add_argument("--symbols", nargs="+", required=True,
                    help="underlying symbols, e.g. SPY QQQ IWM AAPL")
    ap.add_argument("--from", dest="start", required=True,
                    help="UTC start date/time (inclusive), e.g. 2023-01-01")
    ap.add_argument("--to", dest="end", default=None,
                    help="UTC end date/time (exclusive; default: now)")
    ap.add_argument("--db", default=DEFAULT_DB,
                    help=f"BarStore sqlite path (default: {DEFAULT_DB})")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_S,
                    choices=sorted(TIMEFRAME_BY_RESOLUTION),
                    help="bar resolution seconds (default: 300 = 5Min)")
    ap.add_argument("--feed", default="iex", choices=["iex", "sip"])
    ap.add_argument("--adjustment", default="split",
                    choices=["raw", "split", "dividend", "all"])
    ap.add_argument("--chunk-days", type=int, default=DEFAULT_CHUNK_DAYS)
    ap.add_argument("--batch-symbols", type=int,
                    default=DEFAULT_BATCH_SYMBOLS)
    ap.add_argument("--sleep-s", type=float, default=0.25,
                    help="pause between chunks (rate-limit courtesy)")
    args = ap.parse_args(argv)

    from config_loader import ConfigLoader
    cfg = ConfigLoader()
    key = cfg.get_str("ALPACA_API_KEY")
    secret = cfg.get_str("ALPACA_SECRET_KEY")
    if not key or not secret:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")
        return 2
    headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    data_url = cfg.get_str("ALPACA_DATA_URL", "https://data.alpaca.markets")

    summary = backfill_bars(
        args.symbols,
        headers=headers,
        data_url=data_url,
        db_path=args.db,
        start=_parse_date(args.start),
        end=_parse_date(args.end) if args.end else None,
        resolution_s=args.resolution,
        feed=args.feed,
        adjustment=args.adjustment,
        chunk_days=args.chunk_days,
        batch_symbols=args.batch_symbols,
        sleep_s=args.sleep_s,
    )
    for k in ("symbols", "batches", "chunks", "pages", "fetched", "inserted",
              "dropped_duplicate_ts", "skipped_forming", "skipped_bad_close",
              "source", "resolution_s", "window_start_utc",
              "window_end_utc"):
        print(f"{k}={summary[k]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
