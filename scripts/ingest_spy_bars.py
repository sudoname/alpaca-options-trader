#!/usr/bin/env python
"""Ingest user-supplied SPY 1-minute CSVs into the canonical bars cache.

Oracle 4.0.1 Stage A. Three verified source formats:

  frd_et       FirstRateData samples: header
               ``timestamp,open,high,low,close,volume``; timestamps are
               US/Eastern wall clock, bar-START convention; includes
               pre/post-market rows (flagged rth=0, never dropped).

  denver_rth   2008-2021 cleaned file: header
               ``date,open,high,low,close,volume,barCount,average``;
               timestamps verified America/Denver (sessions 07:30-13:59
               constant across DST == 09:30-15:59 ET); RTH only.

  utc_iminu    2024-2026 file: header
               ``iminu,idte,itick,pcls,phgh,plow,popn,pvol``; iminu is a
               compact UTC timestamp (yyyymmddHHMMSS, 133000 == 09:30 ET
               in winter); itick 1..390 RTH only.

Every ingest writes a manifest (source SHA-256, timezone mapping,
coverage, drop counts) into the store and as a sibling JSON.

Usage:
  python scripts/ingest_spy_bars.py --format frd_et --file SPY_1min_sample.csv
  python scripts/ingest_spy_bars.py --format denver_rth --file spy_1min_2008_2021_cleaned.csv
  python scripts/ingest_spy_bars.py --format utc_iminu --file spy_1min_..._rows.csv

Read-only everywhere else; only this CLI writes the store.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from datetime import datetime, timezone
from typing import Dict, Iterator, Optional
from zoneinfo import ZoneInfo

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.ml.bars import (  # noqa: E402
    Bar, BarStore, DEFAULT_DB, is_rth, session_date_of, sha256_file,
)

UTC = timezone.utc
ET = ZoneInfo("America/New_York")
DENVER = ZoneInfo("America/Denver")

FORMATS = ("frd_et", "denver_rth", "utc_iminu")


def _f(v: str) -> Optional[float]:
    try:
        x = float(v)
    except (TypeError, ValueError):
        return None
    return x


def _rows_frd_et(path: str, symbol: str, source: str,
                 counts: Dict[str, int]) -> Iterator[Bar]:
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                local = datetime.strptime(row["timestamp"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                counts["dropped_bad_timestamp"] += 1
                continue
            ts = local.replace(tzinfo=ET).astimezone(UTC)
            close = _f(row.get("close", ""))
            if close is None or close <= 0:
                counts["dropped_bad_price"] += 1
                continue
            yield Bar(symbol=symbol, resolution_s=60, ts_utc=ts,
                      open=_f(row.get("open", "")), high=_f(row.get("high", "")),
                      low=_f(row.get("low", "")), close=close,
                      volume=_f(row.get("volume", "")),
                      session_date=session_date_of(ts), rth=is_rth(ts),
                      source=source)


def _rows_denver(path: str, symbol: str, source: str,
                 counts: Dict[str, int]) -> Iterator[Bar]:
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                local = datetime.strptime(row["date"], "%Y-%m-%d %H:%M:%S")
            except (KeyError, ValueError):
                counts["dropped_bad_timestamp"] += 1
                continue
            ts = local.replace(tzinfo=DENVER).astimezone(UTC)
            close = _f(row.get("close", ""))
            if close is None or close <= 0:
                counts["dropped_bad_price"] += 1
                continue
            yield Bar(symbol=symbol, resolution_s=60, ts_utc=ts,
                      open=_f(row.get("open", "")), high=_f(row.get("high", "")),
                      low=_f(row.get("low", "")), close=close,
                      volume=_f(row.get("volume", "")),
                      session_date=session_date_of(ts), rth=is_rth(ts),
                      source=source)


def _rows_iminu(path: str, symbol: str, source: str,
                counts: Dict[str, int]) -> Iterator[Bar]:
    with open(path, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            raw = (row.get("iminu") or "").strip()
            try:
                ts = datetime.strptime(raw, "%Y%m%d%H%M%S").replace(tzinfo=UTC)
            except ValueError:
                counts["dropped_bad_timestamp"] += 1
                continue
            close = _f(row.get("pcls", ""))
            if close is None or close <= 0:
                counts["dropped_bad_price"] += 1
                continue
            yield Bar(symbol=symbol, resolution_s=60, ts_utc=ts,
                      open=_f(row.get("popn", "")), high=_f(row.get("phgh", "")),
                      low=_f(row.get("plow", "")), close=close,
                      volume=_f(row.get("pvol", "")),
                      session_date=session_date_of(ts), rth=is_rth(ts),
                      source=source)


READERS = {"frd_et": _rows_frd_et, "denver_rth": _rows_denver,
           "utc_iminu": _rows_iminu}
TZ_NOTE = {"frd_et": "America/New_York wall clock -> UTC",
           "denver_rth": "America/Denver wall clock (verified vs DST) -> UTC",
           "utc_iminu": "compact UTC yyyymmddHHMMSS, no conversion"}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--file", required=True)
    ap.add_argument("--format", required=True, choices=FORMATS)
    ap.add_argument("--symbol", default="SPY")
    ap.add_argument("--db", default=DEFAULT_DB)
    args = ap.parse_args()

    if not os.path.isfile(args.file):
        print(f"source not found: {args.file}", file=sys.stderr)
        return 1

    print(f"hashing {args.file} ...")
    digest = sha256_file(args.file)
    counts: Dict[str, int] = {"dropped_bad_timestamp": 0,
                              "dropped_bad_price": 0}
    source = os.path.basename(args.file)

    with BarStore(args.db, readonly=False) as store:
        print(f"ingesting ({args.format}) -> {args.db}")
        ins = store.insert_bars(
            READERS[args.format](args.file, args.symbol, source, counts))
        cov = store.coverage(args.symbol, 60)
        manifest = {
            "manifest_version": 1,
            "stage": "oracle4.0.1-A",
            "source_file": os.path.abspath(args.file),
            "source_sha256": digest,
            "source_format": args.format,
            "timezone_mapping": TZ_NOTE[args.format],
            "bar_convention": "timestamp = bar START, UTC; resolution 60s",
            "adjustment": "split-adjusted per vendor documentation"
            if args.format != "utc_iminu" else "as-provided",
            "symbol": args.symbol,
            "inserted": ins["inserted"],
            "dropped_duplicate_ts": ins["dropped_duplicate_ts"],
            **counts,
            "store_coverage_after": cov,
        }
        store.write_manifest(source, manifest)

    out_json = os.path.join(os.path.dirname(args.db) or ".",
                            f"manifest_{source}.json")
    with open(out_json, "w", encoding="utf-8") as fh:
        json.dump(manifest, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in manifest.items()
                      if k != "store_coverage_after"}, indent=2))
    print("coverage:", json.dumps(manifest["store_coverage_after"], indent=2))
    print(f"manifest written: {out_json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
