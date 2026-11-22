"""Hermetic tests for forward quote capture (quote_capture.py).

The module selftest uses a fake client + temp dir and round-trips its
output through exit_replay.ticks_from_jsonl; no network, no credentials.
"""
import json
import os
from datetime import datetime, timezone

import quote_capture


def test_selftest_passes(capsys):
    assert quote_capture._selftest() == 0
    out = capsys.readouterr().out
    assert "SELFTEST PASS" in out


def test_discover_symbols_missing_inputs(tmp_path):
    # no active_trades file, no ledger -> empty, never raises
    assert quote_capture.discover_symbols(
        active_trades_path=str(tmp_path / "nope.json"), ledger=None) == []


def test_discover_symbols_respects_limit(tmp_path):
    at = tmp_path / "active_trades.json"
    at.write_text(json.dumps([{"symbol": f"SYM{i}"} for i in range(10)]),
                  encoding="utf-8")
    syms = quote_capture.discover_symbols(active_trades_path=str(at),
                                          ledger=None, limit=3)
    assert syms == ["SYM0", "SYM1", "SYM2"]


def test_last_ts_reads_final_line(tmp_path):
    p = tmp_path / "T.jsonl"
    p.write_text('{"ts": "a"}\n{"ts": "b"}\n', encoding="utf-8")
    assert quote_capture._last_ts(str(p)) == "b"
    assert quote_capture._last_ts(str(tmp_path / "missing.jsonl")) is None


def test_observation_mode_preserves_poll_coverage_without_inventing_new_quotes(tmp_path):
    symbol = "SPY261120C00700000"
    quote_time = datetime(2026, 11, 20, 17, 30, tzinfo=timezone.utc)

    class Quote:
        timestamp = quote_time
        bid_price, ask_price = 1.0, 1.1
        bid_size, ask_size = 10, 10

    class Client:
        def get_option_latest_quote(self, request):
            return {symbol: Quote()}

    req = lambda symbols: symbols
    t1 = datetime(2026, 11, 20, 17, 30, 1, tzinfo=timezone.utc)
    t2 = datetime(2026, 11, 20, 17, 30, 6, tzinfo=timezone.utc)
    out = str(tmp_path)
    assert quote_capture.capture_once(Client(), [symbol], out, now=t1,
                                      observation_mode=True, request_factory=req)["written"] == 1
    assert quote_capture.capture_once(Client(), [symbol], out, now=t2,
                                      observation_mode=True, request_factory=req)["written"] == 1
    assert quote_capture.capture_once(Client(), [symbol], out, now=t2,
                                      observation_mode=True, request_factory=req)["skipped_dup"] == 1
    rows = [json.loads(x) for x in (tmp_path / (symbol + ".jsonl")).read_text().splitlines()]
    assert [r["ts"] for r in rows] == [t1.isoformat(), t2.isoformat()]
    assert {r["quote_ts_utc"] for r in rows} == {quote_time.isoformat()}
    assert [r["quote_age_s"] for r in rows] == [1, 6]


def test_future_dated_quote_does_not_leak_into_observation_time(tmp_path):
    symbol = "SPY261120C00700000"

    class Quote:
        timestamp = datetime(2026, 11, 20, 17, 31, tzinfo=timezone.utc)
        bid_price, ask_price = 1.0, 1.1
        bid_size, ask_size = 10, 10

    class Client:
        def get_option_latest_quote(self, request):
            return {symbol: Quote()}

    result = quote_capture.capture_once(
        Client(), [symbol], str(tmp_path),
        now=datetime(2026, 11, 20, 17, 30, tzinfo=timezone.utc),
        observation_mode=True, request_factory=lambda symbols: symbols)
    assert result["written"] == 0 and result["skipped_future"] == 1


def test_discovery_watches_open_paper_legs_and_recent_decisions(tmp_path):
    p = tmp_path / "positions.json"
    p.write_text(json.dumps([{"status": "open", "legs": [
        {"symbol": "SPY261120P00700000"},
        {"symbol": "SPY261120P00695000"}]},
        {"status": "closed", "legs": [{"symbol": "CLOSED261120C00001000"}]}]))

    class Ledger:
        def snapshots(self, since, kind):
            return [{"symbol": "SPY261120C00700000"}] if kind == "candidate" else []

    symbols = quote_capture.discover_symbols(active_trades_path=str(tmp_path / "none"),
                                             paper_positions_path=str(p), ledger=Ledger())
    assert symbols == ["SPY261120P00700000", "SPY261120P00695000",
                       "SPY261120C00700000"]


def test_market_window_uses_new_york_daylight_time():
    assert quote_capture._market_window(datetime(2026, 7, 6, 13, 30,
                                                 tzinfo=timezone.utc))
    assert quote_capture._market_window(datetime(2026, 11, 20, 14, 30,
                                                 tzinfo=timezone.utc))
    assert not quote_capture._market_window(datetime(2026, 11, 21, 14, 30,
                                                     tzinfo=timezone.utc))
