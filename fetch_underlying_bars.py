"""Read-only 5-minute underlying-bar fetcher for the VWAP/EMA/RSI study.

Downloads SIP 5-minute stock bars (split-adjusted, vendor per-bar ``vwap``
included) for the underlyings that appear in the research export, over an
explicit date range, and writes one JSONL row per bar to an isolated
research output file:

    {"symbol", "ts", "o", "h", "l", "c", "v", "vwap"}

``ts`` is the ISO-8601 UTC bar START timestamp (Alpaca labels bars by
start), which is exactly what vwap_trend_signal.signal_at expects. The
fetcher stores every bar the vendor returns (pre/post market included);
session slicing is the replay's job.

Strictly read-only: uses only StockHistoricalDataClient (market-data API).
No trading client is ever constructed; zero broker order submissions.

Usage:
    python fetch_underlying_bars.py --start 2026-11-03 --end 2026-11-13 \
        --decisions export_decisions.jsonl --out research_vwap/bars_5min.jsonl
    python fetch_underlying_bars.py --selftest     # hermetic, no network
Keys come from .env / environment (ALPACA_API_KEY / ALPACA_SECRET_KEY)
via config_loader.ConfigLoader.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

_BATCH = 50  # symbols per StockBarsRequest


def underlyings_from_export(path):
    """Distinct underlyings of decisions that actually submitted an order."""
    out = set()
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            if not any(e.get("event_type") == "submitted"
                       for e in r.get("events") or []):
                continue
            snaps = r.get("snapshots") or []
            u = (snaps[0].get("underlying") or "").strip().upper() \
                if snaps else ""
            if u:
                out.add(u)
    return sorted(out)


def _bar_row(symbol, bar):
    """Normalize one alpaca-py Bar into the JSONL row shape."""
    ts = getattr(bar, "timestamp", None)
    if ts is None:
        return None
    if ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    return {
        "symbol": symbol,
        "ts": ts.astimezone(timezone.utc).isoformat(),
        "o": float(bar.open), "h": float(bar.high), "l": float(bar.low),
        "c": float(bar.close), "v": float(bar.volume),
        "vwap": (float(bar.vwap) if getattr(bar, "vwap", None) is not None
                 else None),
    }


def fetch_bars(client, symbols, start, end, batch=_BATCH):
    """Yield normalized bar rows for ``symbols`` over [start, end)."""
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    from alpaca.data.enums import Adjustment

    for i in range(0, len(symbols), batch):
        chunk = symbols[i:i + batch]
        req = StockBarsRequest(
            symbol_or_symbols=chunk,
            timeframe=TimeFrame(5, TimeFrameUnit.Minute),
            start=start, end=end,
            feed="sip", adjustment=Adjustment.SPLIT,
        )
        resp = client.get_stock_bars(req)
        data = getattr(resp, "data", None) or {}
        for sym in chunk:
            for bar in data.get(sym) or []:
                row = _bar_row(sym, bar)
                if row is not None:
                    yield row


def write_jsonl(rows, out_path):
    os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
    n = 0
    seen = set()
    with open(out_path, "w", encoding="utf-8") as fh:
        for row in rows:
            key = (row["symbol"], row["ts"])
            if key in seen:  # vendor duplicates are rare but cheap to drop
                continue
            seen.add(key)
            fh.write(json.dumps(row) + "\n")
            n += 1
    return n


def _make_client():
    from alpaca.data.historical import StockHistoricalDataClient
    from config_loader import ConfigLoader
    cfg = ConfigLoader()
    key = cfg.get("ALPACA_API_KEY")
    secret = cfg.get("ALPACA_SECRET_KEY")
    if not key or not secret:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not configured")
    return StockHistoricalDataClient(key, secret)


def main(argv=None, client=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--start", default="2026-11-03")
    ap.add_argument("--end", default="2026-11-13")
    ap.add_argument("--decisions", default="export_decisions.jsonl")
    ap.add_argument("--symbols", default=None,
                    help="comma list; overrides --decisions discovery")
    ap.add_argument("--out", default="research_vwap/bars_5min.jsonl")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    if args.symbols:
        symbols = sorted({s.strip().upper()
                          for s in args.symbols.split(",") if s.strip()})
    else:
        symbols = underlyings_from_export(args.decisions)
    if not symbols:
        print("no symbols to fetch"); return 1

    start = datetime.fromisoformat(args.start).replace(tzinfo=timezone.utc)
    end = (datetime.fromisoformat(args.end).replace(tzinfo=timezone.utc)
           + timedelta(days=1))  # inclusive end date
    client = client or _make_client()
    n = write_jsonl(fetch_bars(client, symbols, start, end), args.out)
    print(f"wrote {n} bars for {len(symbols)} symbols -> {args.out}")
    return 0


# ----------------------------------------------------------------- selftest

class _FakeBar:
    def __init__(self, ts, o, h, l, c, v, vwap=None):
        self.timestamp, self.open, self.high, self.low = ts, o, h, l
        self.close, self.volume, self.vwap = c, v, vwap


class _FakeResp:
    def __init__(self, data):
        self.data = data


class _FakeClient:
    def __init__(self, data):
        self._data = data
        self.calls = []

    def get_stock_bars(self, req):
        self.calls.append(req)
        syms = req.symbol_or_symbols
        return _FakeResp({s: self._data.get(s, []) for s in syms})


def _selftest():
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + ": " + name)
        ok = ok and cond

    t0 = datetime(2026, 11, 4, 14, 30, tzinfo=timezone.utc)
    naive = datetime(2026, 11, 4, 14, 35)  # tz-naive vendor timestamp
    data = {
        "AAA": [_FakeBar(t0, 1, 2, 0.5, 1.5, 100, vwap=1.4),
                _FakeBar(t0, 1, 2, 0.5, 1.5, 100, vwap=1.4),  # duplicate ts
                _FakeBar(naive, 2, 3, 1.5, 2.5, 50)],          # no vwap
        "BBB": [_FakeBar(t0, 9, 9, 9, 9, 10, vwap=9.0)],
    }
    client = _FakeClient(data)
    rows = list(fetch_bars(client, ["AAA", "BBB"], t0,
                           t0 + timedelta(days=1), batch=1))
    check("batching splits symbols", len(client.calls) == 2)
    check("rows normalized", all(set(r) == {"symbol", "ts", "o", "h", "l",
                                            "c", "v", "vwap"} for r in rows))
    aaa = [r for r in rows if r["symbol"] == "AAA"]
    check("naive ts coerced to UTC",
          aaa[-1]["ts"] == "2026-11-04T14:35:00+00:00")
    check("missing vendor vwap -> None", aaa[-1]["vwap"] is None)

    with tempfile.TemporaryDirectory() as td:
        out = os.path.join(td, "sub", "bars.jsonl")
        n = write_jsonl(rows, out)
        check("duplicate (symbol, ts) dropped", n == 3)
        back = [json.loads(x) for x in open(out, encoding="utf-8")]
        check("round-trip", len(back) == n and back[0]["symbol"] == "AAA")

        dec = os.path.join(td, "dec.jsonl")
        with open(dec, "w", encoding="utf-8") as fh:
            fh.write(json.dumps({"events": [{"event_type": "submitted"}],
                                 "snapshots": [{"underlying": "nvda"}]}) + "\n")
            fh.write(json.dumps({"events": [],
                                 "snapshots": [{"underlying": "SKIP"}]}) + "\n")
            fh.write("not json\n")
        syms = underlyings_from_export(dec)
        check("export discovery: submitted-only, uppercased",
              syms == ["NVDA"])

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
