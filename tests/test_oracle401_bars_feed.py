"""Oracle 4.0.1 live bars feeder tests (hermetic -- HTTP layer mocked).

Covers: raw->Bar conversion (UTC parse, rth flag, ET session date, source
tag), the forming-bar exclusion rule (a bar whose end is after ``now`` is
never inserted; one ending exactly at ``now`` is kept), bad-close drops,
idempotent re-feeds under keep-first dedupe, pagination via
``next_page_token``, manifest writing, request-parameter hygiene, and an
end-to-end check that a fed store drives the shadow's bars mode.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import bars_feed as bf
from oracle.ml import shadow as shmod
from oracle.ml.bars import BarStore

UTC = timezone.utc

# 2026-01-05 is a Monday under EST: RTH = 14:30-21:00 UTC.
S1_OPEN = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _raw(ts: datetime, close: float, o=None, h=None, lo=None, v=100):
    return {"t": _iso(ts), "o": o if o is not None else close,
            "h": h if h is not None else close,
            "l": lo if lo is not None else close, "c": close, "v": v}


def _minute_raws(start: datetime, n: int, base: float = 100.0):
    return [_raw(start + timedelta(minutes=i), base + (i % 5) * 0.2)
            for i in range(n)]


@pytest.fixture()
def db_path(tmp_path):
    return str(tmp_path / "bars.sqlite")


def _patch_http(monkeypatch, pages):
    """Replace the HTTP layer with canned payload pages; record calls."""
    calls = []

    def fake(url, headers, params, timeout):
        calls.append({"url": url, "headers": headers,
                      "params": dict(params), "timeout": timeout})
        return pages[len(calls) - 1]

    monkeypatch.setattr(bf, "_http_get_json", fake)
    return calls


def _feed(db_path, pages, monkeypatch, symbols=("SPY",), **kw):
    calls = _patch_http(monkeypatch, pages)
    kw.setdefault("now", S1_OPEN + timedelta(hours=2))
    res = bf.feed_bars(list(symbols), headers={"k": "v"},
                       data_url="https://data.example.test",
                       db_path=db_path, **kw)
    return res, calls


def _load(db_path, sym="SPY", rth_only=True):
    with BarStore(db_path) as st:
        return st.load_bars(sym, 60, rth_only=rth_only)


def _manifests(db_path):
    con = sqlite3.connect(db_path)
    try:
        rows = con.execute(
            "SELECT source, manifest_json FROM ingest_manifests").fetchall()
    finally:
        con.close()
    return [(s, json.loads(m)) for s, m in rows]


# --------------------------------------------------------------------------- #
# conversion
# --------------------------------------------------------------------------- #

class TestConversion:
    def test_raw_to_bar_fields(self, db_path, monkeypatch):
        pre_market = S1_OPEN - timedelta(hours=1)        # 13:30 UTC = 08:30 ET
        pages = [{"bars": {"spy": [_raw(S1_OPEN, 100.5, o=100.0, h=100.9,
                                        lo=99.8, v=1234),
                                   _raw(pre_market, 99.5)]},
                  "next_page_token": None}]
        res, _ = _feed(db_path, pages, monkeypatch)
        assert res["fetched"] == 2 and res["inserted"] == 2
        got = _load(db_path, rth_only=False)
        assert [b.ts_utc for b in got] == [pre_market, S1_OPEN]
        rth_bar = got[1]
        assert rth_bar.symbol == "SPY"                   # upper-cased
        assert rth_bar.rth is True and got[0].rth is False
        assert rth_bar.session_date == "2026-01-05"
        assert rth_bar.source == bf.SOURCE == "alpaca_iex_live"
        assert (rth_bar.open, rth_bar.high, rth_bar.low, rth_bar.close,
                rth_bar.volume) == (100.0, 100.9, 99.8, 100.5, 1234.0)

    def test_bad_close_dropped_and_counted(self, db_path, monkeypatch):
        pages = [{"bars": {"SPY": [
            _raw(S1_OPEN, 100.0),
            _raw(S1_OPEN + timedelta(minutes=1), 0.0),       # non-positive
            _raw(S1_OPEN + timedelta(minutes=2), -1.0),      # negative
            {"t": _iso(S1_OPEN + timedelta(minutes=3)), "c": None},
            {"t": _iso(S1_OPEN + timedelta(minutes=4)), "c": "nan"},
        ]}}]
        res, _ = _feed(db_path, pages, monkeypatch)
        assert res["inserted"] == 1
        assert res["skipped_bad_close"] == 4
        got = _load(db_path)
        assert len(got) == 1 and got[0].close == 100.0

    def test_missing_ohlv_tolerated(self, db_path, monkeypatch):
        pages = [{"bars": {"SPY": [{"t": _iso(S1_OPEN), "c": 101.0}]}}]
        res, _ = _feed(db_path, pages, monkeypatch)
        assert res["inserted"] == 1
        b = _load(db_path)[0]
        assert b.close == 101.0
        assert b.open is None and b.high is None and b.low is None
        assert b.volume is None


# --------------------------------------------------------------------------- #
# forming-bar exclusion (critical under keep-first dedupe)
# --------------------------------------------------------------------------- #

class TestFormingBarExclusion:
    def test_forming_bar_skipped_completed_kept(self, db_path, monkeypatch):
        now = S1_OPEN + timedelta(minutes=10, seconds=30)
        pages = [{"bars": {"SPY": [
            _raw(S1_OPEN + timedelta(minutes=9), 100.0),   # ends 10:00 <= now
            _raw(S1_OPEN + timedelta(minutes=10), 55.5),   # ends 11:00 > now
        ]}}]
        res, _ = _feed(db_path, pages, monkeypatch, now=now)
        assert res["inserted"] == 1
        assert res["skipped_forming"] == 1
        got = _load(db_path)
        assert len(got) == 1
        assert got[0].ts_utc == S1_OPEN + timedelta(minutes=9)

    def test_bar_ending_exactly_at_now_is_kept(self, db_path, monkeypatch):
        now = S1_OPEN + timedelta(minutes=5)
        pages = [{"bars": {"SPY": [
            _raw(S1_OPEN + timedelta(minutes=4), 100.4),   # ends exactly now
        ]}}]
        res, _ = _feed(db_path, pages, monkeypatch, now=now)
        assert res["inserted"] == 1 and res["skipped_forming"] == 0

    def test_forming_close_never_frozen_by_dedupe(self, db_path, monkeypatch):
        """The partial close must not survive to block the real close later."""
        ts = S1_OPEN + timedelta(minutes=10)
        # scan 1: bar still forming (partial close 55.5) -> skipped
        pages1 = [{"bars": {"SPY": [_raw(ts, 55.5)]}}]
        res1, _ = _feed(db_path, pages1, monkeypatch,
                        now=ts + timedelta(seconds=30))
        assert res1["inserted"] == 0 and res1["skipped_forming"] == 1
        # scan 2: bar completed with the true close -> inserted
        pages2 = [{"bars": {"SPY": [_raw(ts, 100.9)]}}]
        res2, _ = _feed(db_path, pages2, monkeypatch,
                        now=ts + timedelta(minutes=2))
        assert res2["inserted"] == 1
        assert _load(db_path)[-1].close == 100.9


# --------------------------------------------------------------------------- #
# idempotency / keep-first
# --------------------------------------------------------------------------- #

class TestIdempotency:
    def test_refeed_inserts_nothing_and_keeps_first(self, db_path,
                                                    monkeypatch):
        raws = _minute_raws(S1_OPEN, 5)
        res1, _ = _feed(db_path, [{"bars": {"SPY": raws}}], monkeypatch)
        assert res1["inserted"] == 5
        first_closes = [b.close for b in _load(db_path)]
        # second feed: same timestamps, different closes
        raws2 = [dict(r, c=999.0) for r in raws]
        res2, _ = _feed(db_path, [{"bars": {"SPY": raws2}}], monkeypatch)
        assert res2["inserted"] == 0
        assert res2["dropped_duplicate_ts"] == 5
        assert [b.close for b in _load(db_path)] == first_closes


# --------------------------------------------------------------------------- #
# pagination + request params
# --------------------------------------------------------------------------- #

class TestPaginationAndParams:
    def test_pagination_follows_next_page_token(self, db_path, monkeypatch):
        p1 = {"bars": {"SPY": _minute_raws(S1_OPEN, 3)},
              "next_page_token": "tok123"}
        p2 = {"bars": {"SPY": _minute_raws(S1_OPEN + timedelta(minutes=3), 2)},
              "next_page_token": None}
        res, calls = _feed(db_path, [p1, p2], monkeypatch)
        assert res["pages"] == 2 and res["inserted"] == 5
        assert "page_token" not in calls[0]["params"]
        assert calls[1]["params"]["page_token"] == "tok123"

    def test_request_params_and_symbol_normalization(self, db_path,
                                                     monkeypatch):
        now = S1_OPEN + timedelta(hours=2)
        res, calls = _feed(db_path, [{"bars": {}}], monkeypatch,
                           symbols=[" spy ", "QQQ", "qqq", "", None],
                           now=now, lookback_min=60.0)
        assert res["symbols"] == 2 and res["inserted"] == 0
        p = calls[0]["params"]
        assert p["symbols"] == "QQQ,SPY"
        assert p["timeframe"] == "1Min"
        assert p["feed"] == "iex"
        assert p["adjustment"] == "raw"
        assert p["end"] == _iso(now)
        assert p["start"] == _iso(now - timedelta(minutes=60))
        assert calls[0]["url"] == \
            "https://data.example.test/v2/stocks/bars"
        assert calls[0]["headers"] == {"k": "v"}

    def test_no_symbols_no_http(self, db_path, monkeypatch):
        calls = _patch_http(monkeypatch, [])
        res = bf.feed_bars([], headers={}, data_url="https://x",
                           db_path=db_path)
        assert res["pages"] == 0 and res["fetched"] == 0
        assert calls == []

    def test_max_pages_bounds_runaway_pagination(self, db_path, monkeypatch):
        page = {"bars": {"SPY": []}, "next_page_token": "again"}
        res, calls = _feed(db_path, [page] * 10, monkeypatch, max_pages=3)
        assert res["pages"] == 3 and len(calls) == 3


# --------------------------------------------------------------------------- #
# manifest
# --------------------------------------------------------------------------- #

class TestManifest:
    def test_manifest_written_on_insert(self, db_path, monkeypatch):
        _feed(db_path, [{"bars": {"SPY": _minute_raws(S1_OPEN, 4)}}],
              monkeypatch)
        rows = _manifests(db_path)
        assert len(rows) == 1
        src, man = rows[0]
        assert src == "alpaca_iex_live"
        assert man["stage"] == "oracle4.0.1-live-feed"
        assert man["counts"]["inserted"] == 4
        assert man["symbols"] == ["SPY"]
        assert "bar START" in man["bar_convention"]

    def test_no_manifest_when_nothing_inserted(self, db_path, monkeypatch):
        raws = _minute_raws(S1_OPEN, 4)
        _feed(db_path, [{"bars": {"SPY": raws}}], monkeypatch)
        _feed(db_path, [{"bars": {"SPY": raws}}], monkeypatch)  # all dupes
        assert len(_manifests(db_path)) == 1


# --------------------------------------------------------------------------- #
# end-to-end: fed store drives shadow bars mode (no HISTORY_UNTIMESTAMPED)
# --------------------------------------------------------------------------- #

def _cand():
    return {"sym": "SPY", "score": 1.0, "underlying_price": 100.5,
            "opt": {"symbol": "SPY260105C00100000", "bid": 1.0, "ask": 1.1,
                    "dte": 1}}


class TestEndToEndShadow:
    def test_fed_store_enables_bars_mode(self, db_path, tmp_path,
                                         monkeypatch):
        now = S1_OPEN + timedelta(minutes=50)
        pages = [{"bars": {"SPY": _minute_raws(S1_OPEN, 41)}}]
        res, _ = _feed(db_path, pages, monkeypatch, now=now)
        assert res["inserted"] == 41

        calls = []

        def hist(u, days=90):
            calls.append(u)
            return []

        cfg = shmod.ShadowConfig(enabled=True, max_scenarios=64,
                                 bars_db=db_path)
        rec = shmod.shadow_scan(
            [_cand()], scan_id="feed-e2e", price_history_fn=hist,
            step_minutes=1.0, now=now, cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        meta = rec["scenarios"]["SPY"]
        assert rec["history_grid_assumed"] is False
        assert meta["history_source"] == "bars"
        assert meta["status"] == "ok"
        assert meta["n_history_points"] == 41
        assert calls == []       # untimestamped legacy path never consulted
