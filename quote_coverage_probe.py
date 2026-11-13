"""Quote-coverage probe: what option market data can we actually replay?

Phase 1 evidence tool (master prompt): before promising historical quote
backfill, probe a KNOWN contract + session against the Alpaca options data
API and report exactly what comes back, per endpoint and per feed.

What it checks (read-only, data API only, no orders):
  1. historical 1-min BARS for the contract over the session
  2. historical TRADES for the contract over the session
  3. latest QUOTE on the 'indicative' feed (free tier)
  4. latest QUOTE on the 'opra' feed (requires OPRA entitlement)
  5. SNAPSHOT (latest quote/trade + greeks) on the indicative feed

Known constraint it documents: alpaca-py 0.42.0 exposes OptionBarsRequest
and OptionTradesRequest but NO historical option *quotes* request — there
is no bid/ask time series endpoint in this SDK. Latest-quote endpoints
cannot be backfilled. So counterfactual fill simulation for non-selected
candidates (research_export groups: outcome_state=counterfactual_required)
cannot come from historical NBBO via this stack; it needs either trade/bar
mid-based approximation (label it as such) or FORWARD quote capture
(exit_replay.ticks_from_jsonl format).

Usage:
    python quote_coverage_probe.py --symbol SPG270617C00240000 \
        --session 2026-11-12
Keys come from .env / environment (ALPACA_API_KEY / ALPACA_SECRET_KEY).
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone


def _load_env(path=".env"):
    if not os.path.exists(path):
        return
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            k, _, v = line.partition("=")
            os.environ.setdefault(k.strip(), v.strip())


def probe(symbol, session, out=None):
    from alpaca.data.historical.option import OptionHistoricalDataClient
    from alpaca.data.requests import (
        OptionBarsRequest, OptionTradesRequest,
        OptionLatestQuoteRequest, OptionSnapshotRequest,
    )
    from alpaca.data.timeframe import TimeFrame
    from alpaca.data.enums import OptionsFeed

    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        return 2

    client = OptionHistoricalDataClient(api_key=key, secret_key=sec)
    day = datetime.strptime(session, "%Y-%m-%d").replace(tzinfo=timezone.utc)
    start, end = day, day + timedelta(days=1)
    report = {
        "probe_at_utc": datetime.now(timezone.utc).isoformat(),
        "symbol": symbol,
        "session": session,
        "sdk": "alpaca-py",
        "checks": {},
    }

    def _check(name, fn):
        try:
            report["checks"][name] = fn()
        except Exception as exc:  # evidence tool: record, never crash
            report["checks"][name] = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}

    def _bars():
        resp = client.get_option_bars(OptionBarsRequest(
            symbol_or_symbols=symbol, timeframe=TimeFrame.Minute,
            start=start, end=end))
        bars = resp.data.get(symbol, [])
        return {
            "ok": True, "n_bars": len(bars),
            "first_ts": bars[0].timestamp.isoformat() if bars else None,
            "last_ts": bars[-1].timestamp.isoformat() if bars else None,
            "sample": ({"o": bars[0].open, "h": bars[0].high,
                        "l": bars[0].low, "c": bars[0].close,
                        "v": bars[0].volume} if bars else None),
        }

    def _trades():
        resp = client.get_option_trades(OptionTradesRequest(
            symbol_or_symbols=symbol, start=start, end=end, limit=10000))
        trades = resp.data.get(symbol, [])
        return {
            "ok": True, "n_trades": len(trades),
            "first_ts": trades[0].timestamp.isoformat() if trades else None,
            "last_ts": trades[-1].timestamp.isoformat() if trades else None,
        }

    def _latest_quote(feed):
        resp = client.get_option_latest_quote(OptionLatestQuoteRequest(
            symbol_or_symbols=symbol, feed=feed))
        q = resp.get(symbol)
        return {
            "ok": True, "feed": feed.value,
            "bid": getattr(q, "bid_price", None),
            "ask": getattr(q, "ask_price", None),
            "bid_size": getattr(q, "bid_size", None),
            "ask_size": getattr(q, "ask_size", None),
            "ts": q.timestamp.isoformat() if getattr(q, "timestamp", None) else None,
        }

    def _snapshot():
        resp = client.get_option_snapshot(OptionSnapshotRequest(
            symbol_or_symbols=symbol, feed=OptionsFeed.INDICATIVE))
        s = resp.get(symbol)
        lq = getattr(s, "latest_quote", None)
        return {
            "ok": True,
            "latest_quote_ts": (lq.timestamp.isoformat()
                                if lq is not None and lq.timestamp else None),
            "has_greeks": getattr(s, "greeks", None) is not None,
            "implied_vol": getattr(s, "implied_volatility", None),
        }

    _check("historical_bars_1min", _bars)
    _check("historical_trades", _trades)
    _check("latest_quote_indicative",
           lambda: _latest_quote(OptionsFeed.INDICATIVE))
    _check("latest_quote_opra", lambda: _latest_quote(OptionsFeed.OPRA))
    _check("snapshot_indicative", _snapshot)

    # SDK-level fact, independent of credentials:
    import alpaca.data.requests as reqs
    report["historical_quotes_request_in_sdk"] = hasattr(
        reqs, "OptionQuotesRequest")

    c = report["checks"]
    report["verdict"] = {
        "historical_bars": bool(c.get("historical_bars_1min", {}).get("n_bars")),
        "historical_trades": bool(c.get("historical_trades", {}).get("n_trades")),
        "historical_nbbo_quotes": False if not report[
            "historical_quotes_request_in_sdk"] else None,
        "opra_entitled": bool(c.get("latest_quote_opra", {}).get("ok")),
        "backfill_promise": (
            "NO historical bid/ask series available via this SDK; "
            "counterfactual alternative fills require forward quote "
            "capture or labeled mid-of-trades approximation."),
    }

    text = json.dumps(report, indent=2, default=str)
    print(text)
    if out:
        with open(out, "w", encoding="utf-8") as fh:
            fh.write(text + "\n")
    return 0


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--symbol", required=True,
                    help="OCC option symbol, e.g. SPG270617C00240000")
    ap.add_argument("--session", required=True, help="YYYY-MM-DD (UTC day)")
    ap.add_argument("--out", default=None, help="write JSON report here")
    args = ap.parse_args(argv)
    _load_env()
    return probe(args.symbol, args.session, out=args.out)


if __name__ == "__main__":
    sys.exit(main())
