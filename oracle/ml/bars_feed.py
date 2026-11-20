"""Live 1-minute bars feeder for the Oracle 4.0.1 shadow (observe-only).

Keeps the Stage-A ``BarStore`` sqlite current during live scans so the
shadow's bars mode (``ORACLE4_BARS_DB``) can produce real, timestamped
feature rows instead of abstaining with ``HISTORY_UNTIMESTAMPED``.

Rules:
  * research-path only -- nothing in the production trading path imports
    this module; the scan-loop hook is env-gated and fail-open;
  * reuses the trader's already-resolved Alpaca data credentials
    (``headers`` / ``data_url``) -- no new credential plumbing;
  * **forming bars are never inserted**: ``BarStore.insert_bars`` is
    keep-first (INSERT OR IGNORE), so writing a partially formed bar
    would freeze a wrong close forever.  A bar only exists here once
    ``ts_utc + resolution <= now``;
  * bars with a non-positive or non-finite close are dropped and counted,
    never repaired;
  * the fetch window is a fixed lookback from ``now``; combined with the
    keep-first primary key this makes every feed idempotent and
    self-healing across restarts and missed scans.
"""
from __future__ import annotations

import math
from datetime import datetime, timedelta, timezone
from typing import Dict, Iterable, List, Optional, Sequence

from .bars import Bar, BarStore, is_rth, session_date_of

UTC = timezone.utc

RESOLUTION_S = 60
SOURCE = "alpaca_iex_live"
DEFAULT_LOOKBACK_MIN = 480.0


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


def feed_bars(
    symbols: Sequence[str],
    *,
    headers: Dict[str, str],
    data_url: str,
    db_path: str,
    lookback_min: float = DEFAULT_LOOKBACK_MIN,
    now: Optional[datetime] = None,
    feed: str = "iex",
    timeout: float = 20.0,
    page_limit: int = 10000,
    max_pages: int = 20,
) -> Dict:
    """Fetch recent 1-minute bars for ``symbols`` and upsert into BarStore.

    Returns a summary dict:
      {symbols, pages, fetched, inserted, dropped_duplicate_ts,
       skipped_forming, skipped_bad_close}
    Raises on fetch/store errors -- the scan-loop hook catches and logs.
    """
    if now is None:
        now = datetime.now(UTC)
    elif now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    else:
        now = now.astimezone(UTC)

    syms = sorted({str(s).strip().upper() for s in symbols
                   if s and str(s).strip()})
    summary: Dict = {
        "symbols": len(syms),
        "pages": 0,
        "fetched": 0,
        "inserted": 0,
        "dropped_duplicate_ts": 0,
        "skipped_forming": 0,
        "skipped_bad_close": 0,
    }
    if not syms:
        return summary

    start = now - timedelta(minutes=float(lookback_min))
    url = f"{data_url.rstrip('/')}/v2/stocks/bars"
    base_params = {
        "symbols": ",".join(syms),
        "timeframe": "1Min",
        "start": _iso_z(start),
        "end": _iso_z(now),
        "feed": feed,
        "limit": page_limit,
        "adjustment": "raw",
    }

    bars: List[Bar] = []
    page_token: Optional[str] = None
    for _ in range(max_pages):
        params = dict(base_params)
        if page_token:
            params["page_token"] = page_token
        payload = _http_get_json(url, headers, params, timeout) or {}
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
                # completed-bar rule: never insert a forming bar (keep-first
                # dedupe would freeze its partial close forever)
                if ts + timedelta(seconds=RESOLUTION_S) > now:
                    summary["skipped_forming"] += 1
                    continue
                bars.append(Bar(
                    symbol=sym_u,
                    resolution_s=RESOLUTION_S,
                    ts_utc=ts,
                    open=_opt_float(raw.get("o")),
                    high=_opt_float(raw.get("h")),
                    low=_opt_float(raw.get("l")),
                    close=close,
                    volume=_opt_float(raw.get("v")),
                    session_date=session_date_of(ts),
                    rth=is_rth(ts),
                    source=SOURCE,
                ))
        page_token = payload.get("next_page_token")
        if not page_token:
            break

    with BarStore(db_path, readonly=False) as store:
        counts = store.insert_bars(bars)
        summary["inserted"] = counts["inserted"]
        summary["dropped_duplicate_ts"] = counts["dropped_duplicate_ts"]
        if summary["inserted"] > 0:
            store.write_manifest(SOURCE, {
                "manifest_version": 1,
                "stage": "oracle4.0.1-live-feed",
                "feed": feed,
                "symbols": syms,
                "window_start_utc": _iso_z(start),
                "window_end_utc": _iso_z(now),
                "bar_convention": ("timestamp = bar START, UTC; "
                                   f"resolution {RESOLUTION_S}s; "
                                   "completed bars only"),
                "counts": {k: summary[k] for k in (
                    "pages", "fetched", "inserted", "dropped_duplicate_ts",
                    "skipped_forming", "skipped_bad_close")},
            })
    return summary
