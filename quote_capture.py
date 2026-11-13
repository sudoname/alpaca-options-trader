"""Forward option-quote capture: build the intra-trade quote series the
ledger lacks.

Why this exists (probe evidence, 2026-11-13, quote_coverage_probe.py):
  * alpaca-py 0.42.0 has NO historical option-quotes request; latest-quote
    only. OPRA is not entitled ("OPRA agreement is not signed").
  * Historical bars/trades for contracts Oracle actually traded are ~1
    print per session (our own fill), so mid-of-trades replay is useless.
  * Therefore counterfactual fills for non-selected candidates
    (research_export: outcome_state=counterfactual_required) and exit
    replay (exit_replay.py) can only be supported going FORWARD, by
    polling the indicative latest-quote endpoint and appending ticks.

What it does (observe-only; data API only; never touches trading path):
  1. Discovers symbols to watch:
       - open positions from active_trades.json
       - recent candidate-group contracts (kind=group_candidate) from the
         research ledger — this is what makes alternatives replayable.
  2. Polls the latest indicative quote for all symbols in one batch.
  3. Appends one JSONL tick per symbol to <out-dir>/<SYMBOL>.jsonl in the
     exit_replay.ticks_from_jsonl format:
       {"ts", "bid", "ask", "bid_size", "ask_size", "quote_age_s",
        "provenance": "indicative_latest", "symbol", "captured_at_utc"}
     Duplicate quote timestamps are skipped (no fake ticks when the quote
     has not updated).

Usage:
    python quote_capture.py --once                  # one poll (cron/timer)
    python quote_capture.py --interval 60           # loop every 60s
    python quote_capture.py --selftest              # hermetic, no network
Keys come from .env / environment (ALPACA_API_KEY / ALPACA_SECRET_KEY).
"""
import argparse
import json
import os
import sys
import time
from datetime import datetime, timedelta, timezone

_PROVENANCE = "indicative_latest"
_BATCH = 100


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


def discover_symbols(active_trades_path="active_trades.json",
                     ledger=None, lookback_hours=6, limit=200):
    """Open positions + recently considered candidate-group contracts."""
    symbols = []
    seen = set()

    def _add(sym):
        if sym and sym not in seen:
            seen.add(sym)
            symbols.append(sym)

    if os.path.exists(active_trades_path):
        try:
            with open(active_trades_path, encoding="utf-8") as fh:
                for t in json.load(fh) or []:
                    if isinstance(t, dict):
                        _add(t.get("symbol"))
        except (ValueError, OSError):
            pass

    if ledger is not None:
        since = (datetime.now(timezone.utc)
                 - timedelta(hours=lookback_hours)).isoformat()
        try:
            for snap in ledger.snapshots(since=since, kind="group_candidate"):
                _add(snap.get("symbol"))
                if len(symbols) >= limit:
                    break
        except Exception:
            pass  # capture must not die on a ledger hiccup

    return symbols[:limit]


def _last_ts(path):
    """Quote timestamp of the final line of a tick file (dedupe guard)."""
    try:
        with open(path, "rb") as fh:
            fh.seek(0, os.SEEK_END)
            size = fh.tell()
            fh.seek(max(0, size - 4096))
            tail = fh.read().decode("utf-8", "replace").strip().splitlines()
        if tail:
            return json.loads(tail[-1]).get("ts")
    except (OSError, ValueError):
        pass
    return None


def capture_once(client, symbols, out_dir, now=None):
    """One poll: fetch latest quotes in batches, append fresh ticks.

    Returns {"polled": n, "written": n, "skipped_dup": n, "errors": [...]}.
    """
    from alpaca.data.requests import OptionLatestQuoteRequest
    from alpaca.data.enums import OptionsFeed

    os.makedirs(out_dir, exist_ok=True)
    now = now or datetime.now(timezone.utc)
    stats = {"polled": 0, "written": 0, "skipped_dup": 0, "errors": []}

    for i in range(0, len(symbols), _BATCH):
        chunk = symbols[i:i + _BATCH]
        try:
            resp = client.get_option_latest_quote(OptionLatestQuoteRequest(
                symbol_or_symbols=chunk, feed=OptionsFeed.INDICATIVE))
        except Exception as exc:
            stats["errors"].append(f"{type(exc).__name__}: {exc}")
            continue
        for sym in chunk:
            q = resp.get(sym) if hasattr(resp, "get") else None
            if q is None:
                continue
            stats["polled"] += 1
            qts = getattr(q, "timestamp", None)
            ts_iso = qts.isoformat() if qts is not None else now.isoformat()
            path = os.path.join(out_dir, f"{sym}.jsonl")
            if _last_ts(path) == ts_iso:
                stats["skipped_dup"] += 1
                continue
            age = None
            if qts is not None:
                try:
                    age = max(0.0, (now - qts).total_seconds())
                except TypeError:
                    age = None
            tick = {
                "ts": ts_iso,
                "bid": getattr(q, "bid_price", None),
                "ask": getattr(q, "ask_price", None),
                "bid_size": getattr(q, "bid_size", None),
                "ask_size": getattr(q, "ask_size", None),
                "quote_age_s": age,
                "provenance": _PROVENANCE,
                "symbol": sym,
                "captured_at_utc": now.isoformat(),
            }
            with open(path, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(tick, default=str) + "\n")
            stats["written"] += 1
    return stats


def _real_client():
    from alpaca.data.historical.option import OptionHistoricalDataClient
    key = os.environ.get("ALPACA_API_KEY")
    sec = os.environ.get("ALPACA_SECRET_KEY")
    if not key or not sec:
        raise SystemExit("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
    return OptionHistoricalDataClient(api_key=key, secret_key=sec)


def _open_ledger(db_path):
    if not os.path.exists(db_path):
        return None
    try:
        from research_ledger import ResearchLedger
        return ResearchLedger(db_path)
    except Exception:
        return None


def _selftest():
    """Hermetic: fake client, temp dir, round-trip via exit_replay loader."""
    import tempfile
    from exit_replay import ticks_from_jsonl

    class _Q:
        def __init__(self, bid, ask, ts):
            self.bid_price, self.ask_price = bid, ask
            self.bid_size, self.ask_size = 10, 20
            self.timestamp = ts

    class _FakeClient:
        def __init__(self):
            self.calls = 0

        def get_option_latest_quote(self, req):
            self.calls += 1
            ts = datetime(2026, 11, 13, 15, 30 + self.calls,
                          tzinfo=timezone.utc)
            return {s: _Q(1.00 + self.calls * 0.05, 1.10 + self.calls * 0.05,
                          ts)
                    for s in req.symbol_or_symbols}

    ok = True

    def check(name, cond):
        nonlocal ok
        ok = ok and bool(cond)
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")

    with tempfile.TemporaryDirectory() as tmp:
        at = os.path.join(tmp, "active_trades.json")
        with open(at, "w", encoding="utf-8") as fh:
            json.dump([{"symbol": "SPG270617C00240000"},
                       {"symbol": "RIVN270319P00015000"},
                       {"symbol": "SPG270617C00240000"}], fh)
        syms = discover_symbols(active_trades_path=at, ledger=None)
        check("discovery dedupes open positions", syms == [
            "SPG270617C00240000", "RIVN270319P00015000"])

        client = _FakeClient()
        out = os.path.join(tmp, "ticks")
        now = datetime(2026, 11, 13, 15, 32, tzinfo=timezone.utc)
        s1 = capture_once(client, syms, out, now=now)
        check("first poll writes one tick per symbol",
              s1["written"] == 2 and s1["skipped_dup"] == 0)

        # same quote ts again -> dedupe (fake bumps ts per call, so replay
        # the dedupe path by polling twice with a frozen client call count)
        client.calls -= 1
        s2 = capture_once(client, syms, out, now=now)
        check("unchanged quote ts is skipped, not duplicated",
              s2["written"] == 0 and s2["skipped_dup"] == 2)

        s3 = capture_once(client, syms, out, now=now)
        check("fresh quote ts appends", s3["written"] == 2)

        path = os.path.join(out, "SPG270617C00240000.jsonl")
        ticks = ticks_from_jsonl(path)
        check("exit_replay.ticks_from_jsonl loads capture output",
              len(ticks) == 2 and ticks[0].bid == 1.05
              and ticks[1].ask is not None)
        with open(path, encoding="utf-8") as fh:
            rec = json.loads(fh.readline())
        check("ticks carry provenance + quote_age_s",
              rec.get("provenance") == _PROVENANCE
              and rec.get("quote_age_s") is not None)

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--once", action="store_true", help="single poll")
    ap.add_argument("--interval", type=float, default=None,
                    help="loop every N seconds")
    ap.add_argument("--out-dir", default="quote_ticks")
    ap.add_argument("--active-trades", default="active_trades.json")
    ap.add_argument("--db", default="research_ledger.db")
    ap.add_argument("--lookback-hours", type=float, default=6.0,
                    help="group-candidate discovery window")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()
    if not args.once and args.interval is None:
        ap.error("choose --once, --interval, or --selftest")

    _load_env()
    client = _real_client()
    ledger = _open_ledger(args.db)
    while True:
        symbols = discover_symbols(active_trades_path=args.active_trades,
                                   ledger=ledger,
                                   lookback_hours=args.lookback_hours)
        stats = capture_once(client, symbols, args.out_dir)
        print(json.dumps({"at": datetime.now(timezone.utc).isoformat(),
                          "n_symbols": len(symbols), **stats}))
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    sys.exit(main())
