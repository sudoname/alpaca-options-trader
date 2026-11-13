"""Hermetic tests for forward quote capture (quote_capture.py).

The module selftest uses a fake client + temp dir and round-trips its
output through exit_replay.ticks_from_jsonl; no network, no credentials.
"""
import json
import os

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
