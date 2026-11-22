"""Oracle 4.0.2 P1-5 tests: bars backfill + bars-based dataset builder.

Covers:
  * ``oracle.ml.bars_backfill``: pagination, idempotency, forming-bar and
    bad-close skips, request parameters, manifest write, bad resolution;
  * ``oracle.ml.bars_dataset``: PIT obs (event_at = bar END), sampling,
    label signs per direction, session-label censoring, option-dependent
    labels MISSING (never guessed), counterfactual outcomes, determinism,
    skipped symbols, purged walk-forward integration.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import bars_backfill
from oracle.ml.bars import Bar, BarStore
from oracle.ml.bars_backfill import backfill_bars
from oracle.ml.bars_dataset import build_bars_dataset
from oracle.lab.purged_validation import load_samples, make_folds

UTC = timezone.utc

#: 5 consecutive non-DST sessions (EST): RTH = 14:30-21:00 UTC.
SESSIONS = ["2026-01-05", "2026-01-06", "2026-01-07",
            "2026-01-08", "2026-01-09"]
RES = 300
BARS_PER_SESSION = 78  # 14:30 .. 20:55 starts


def _parse_iso(text: str) -> datetime:
    dt = datetime.fromisoformat(str(text).replace("Z", "+00:00"))
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _make_store(db_path: str) -> None:
    """AAA rises +0.2/bar from 100; BBB falls -0.2/bar from 300."""
    bars = []
    i = 0
    for sd in SESSIONS:
        day = datetime.fromisoformat(sd).replace(tzinfo=UTC)
        open_utc = day.replace(hour=14, minute=30)
        for k in range(BARS_PER_SESSION):
            ts = open_utc + timedelta(seconds=k * RES)
            for sym, close in (("AAA", 100.0 + 0.2 * i),
                               ("BBB", 300.0 - 0.2 * i)):
                bars.append(Bar(
                    symbol=sym, resolution_s=RES, ts_utc=ts,
                    open=close, high=close, low=close, close=close,
                    volume=1000.0, session_date=sd, rth=True,
                    source="test_fixture"))
            i += 1
    with BarStore(db_path, readonly=False) as store:
        counts = store.insert_bars(bars)
        assert counts["inserted"] == len(bars)


@pytest.fixture(scope="module")
def built(tmp_path_factory):
    tmp = tmp_path_factory.mktemp("p15")
    db = str(tmp / "bars.sqlite")
    _make_store(db)
    out = str(tmp / "ds.sqlite")
    man = str(tmp / "ds_manifest.json")
    manifest = build_bars_dataset(db, out, man, ["AAA", "BBB", "ZZZ"])
    return {"bars_db": db, "out": out, "manifest_path": man,
            "manifest": manifest, "tmp": tmp}


def _q(path, sql, args=()):
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        return conn.execute(sql, args).fetchall()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# bars_dataset
# --------------------------------------------------------------------------- #

class TestBarsDataset:

    def test_counts_and_sampling(self, built):
        m = built["manifest"]
        c = m["counts"]
        # 10 samples/session (15:30..20:00 step 30) x 5 sessions x 2 symbols
        assert c["decisions"] == 100
        assert c["feature_rows"] == 100
        assert c["outcome_rows"] == 100
        assert c["sessions"] == 5
        assert c["underlyings"] == 2
        assert c["underlying_obs"] == 2 * 5 * BARS_PER_SESSION
        rep = m["sampling"]["report"]
        assert rep["sampled"] == 100
        assert rep["skipped_no_asof"] == 0
        assert rep["skipped_stale_asof"] == 0
        assert m["symbols"]["built"] == ["AAA", "BBB"]

    def test_skipped_symbol_reported(self, built):
        skipped = built["manifest"]["symbols"]["skipped"]
        assert "ZZZ" in skipped and skipped["ZZZ"]

    def test_obs_event_at_is_bar_end(self, built):
        rows = _q(built["out"],
                  "SELECT MIN(event_at) FROM underlying_obs "
                  "WHERE underlying='AAA'")
        first = _parse_iso(rows[0][0])
        # first bar starts 14:30, resolution 300s -> PIT availability 14:35
        assert first == datetime(2026, 1, 5, 14, 35, tzinfo=UTC)

    def test_decision_kind_and_session(self, built):
        rows = _q(built["out"],
                  "SELECT DISTINCT kind FROM features")
        assert [r[0] for r in rows] == ["bars_synthetic"]
        rows = _q(built["out"],
                  "SELECT DISTINCT session FROM features ORDER BY session")
        assert [r[0] for r in rows] == SESSIONS

    def test_minute_label_signs(self, built):
        for horizon in ("15m", "60m"):
            rows = _q(built["out"],
                      "SELECT decision_id, value, status FROM labels "
                      "WHERE horizon=? AND label_type='log_return'",
                      (horizon,))
            assert rows
            for did, value, status in rows:
                assert status == "ok", (did, horizon, status)
                if did.startswith("bars-AAA-"):
                    assert value > 0
                else:
                    assert value < 0
            rows = _q(built["out"],
                      "SELECT decision_id, value, status FROM labels "
                      "WHERE horizon=? AND label_type='direction'",
                      (horizon,))
            assert rows
            for did, value, status in rows:
                assert status == "ok"
                assert value == (1.0 if did.startswith("bars-AAA-")
                                 else -1.0)

    def test_session_label_censoring(self, built):
        # 5 sessions + session_ahead=4: every '5s' target is the coverage
        # session itself -> strictly-less rule censors all of them.
        rows = _q(built["out"],
                  "SELECT DISTINCT status FROM labels "
                  "WHERE horizon='5s' AND label_type='log_return'")
        assert [r[0] for r in rows] == ["censored"]
        # '1s' on the final (coverage) session is censored...
        rows = _q(built["out"],
                  "SELECT status FROM labels WHERE horizon='1s' "
                  "AND label_type='log_return' "
                  "AND decision_id LIKE 'bars-AAA-20260109%'")
        assert rows and all(r[0] == "censored" for r in rows)
        # ...but earlier sessions resolve
        rows = _q(built["out"],
                  "SELECT status FROM labels WHERE horizon='1s' "
                  "AND label_type='log_return' "
                  "AND decision_id LIKE 'bars-AAA-20260105%'")
        assert rows and all(r[0] == "ok" for r in rows)

    def test_option_dependent_labels_missing(self, built):
        rows = _q(built["out"],
                  "SELECT DISTINCT status FROM labels "
                  "WHERE label_type='iv_change'")
        assert [r[0] for r in rows] == ["missing"]
        rows = _q(built["out"],
                  "SELECT meta_json FROM labels "
                  "WHERE label_type='iv_change' LIMIT 1")
        assert "no_start_iv" in rows[0][0]
        rows = _q(built["out"],
                  "SELECT DISTINCT status FROM labels "
                  "WHERE label_type='first_touch'")
        assert [r[0] for r in rows] == ["missing"]
        rows = _q(built["out"],
                  "SELECT meta_json FROM labels "
                  "WHERE label_type='first_touch' LIMIT 1")
        assert "never_filled" in rows[0][0]

    def test_outcomes_all_counterfactual(self, built):
        rows = _q(built["out"], "SELECT DISTINCT status FROM outcomes")
        assert [r[0] for r in rows] == ["counterfactual"]

    def test_meta_and_manifest(self, built):
        rows = dict(_q(built["out"], "SELECT key, value FROM meta"))
        assert rows["builder"] == "bars_dataset"
        assert rows["dataset_hash"] == built["manifest"]["dataset_hash"]
        assert rows["bars_db_sha256"] == \
            built["manifest"]["input"]["bars_db_sha256"]
        with open(built["manifest_path"], encoding="utf-8") as f:
            on_disk = json.load(f)
        assert on_disk["dataset_hash"] == built["manifest"]["dataset_hash"]
        assert on_disk["coverage"]["outcome_status_counts"] == \
            {"counterfactual": 100}

    def test_deterministic_hash(self, built):
        out2 = str(built["tmp"] / "ds2.sqlite")
        man2 = str(built["tmp"] / "ds2_manifest.json")
        m2 = build_bars_dataset(built["bars_db"], out2, man2,
                                ["AAA", "BBB", "ZZZ"])
        assert m2["dataset_hash"] == built["manifest"]["dataset_hash"]

    def test_purged_validation_integration(self, built):
        samples = load_samples(built["out"], "15m", "log_return",
                               kinds=["bars_synthetic"])
        assert len(samples) == 100
        folds, fold_sessions, holdout = make_folds(
            samples, test_block_sessions=1, min_train_sessions=2,
            embargo_sessions=1, holdout_sessions=0)
        assert folds
        # embargo: no test session may also appear in its train sessions
        for fold in folds:
            train_sessions = set(fold.train_sessions)
            test_sessions = set(fold.test_sessions)
            assert not (train_sessions & test_sessions)
            assert fold.train_ids and fold.test_ids

    def test_window_filter(self, built):
        out3 = str(built["tmp"] / "ds3.sqlite")
        man3 = str(built["tmp"] / "ds3_manifest.json")
        m3 = build_bars_dataset(
            built["bars_db"], out3, man3, ["AAA"],
            start=datetime(2026, 1, 7, tzinfo=UTC),
            end=datetime(2026, 1, 8, tzinfo=UTC))
        assert m3["counts"]["decisions"] == 10  # one session, one symbol
        assert m3["counts"]["sessions"] == 1


# --------------------------------------------------------------------------- #
# bars_backfill
# --------------------------------------------------------------------------- #

def _bar_raw(ts: datetime, close: float):
    return {"t": ts.strftime("%Y-%m-%dT%H:%M:%SZ"), "o": close, "h": close,
            "l": close, "c": close, "v": 10}


NOW = datetime(2026, 2, 1, tzinfo=UTC)
START = datetime(2026, 1, 5, tzinfo=UTC)
END = datetime(2026, 1, 6, tzinfo=UTC)


class TestBarsBackfill:

    def _run(self, db, fake, monkeypatch, **kw):
        monkeypatch.setattr(bars_backfill, "_http_get_json", fake)
        kw.setdefault("headers", {"k": "v"})
        kw.setdefault("data_url", "https://data.example")
        kw.setdefault("db_path", db)
        kw.setdefault("start", START)
        kw.setdefault("end", END)
        kw.setdefault("now", NOW)
        return backfill_bars(["AAA", "BBB"], **kw)

    def test_insert_and_idempotent(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)

        def fake(url, headers, params, timeout):
            return {"bars": {
                "AAA": [_bar_raw(t0, 101.0),
                        _bar_raw(t0 + timedelta(seconds=300), 102.0)],
                "BBB": [_bar_raw(t0, 201.0)],
            }}

        s1 = self._run(db, fake, monkeypatch)
        assert s1["fetched"] == 3 and s1["inserted"] == 3
        assert s1["source"] == "alpaca_iex_backfill_split"
        s2 = self._run(db, fake, monkeypatch)
        assert s2["inserted"] == 0
        assert s2["dropped_duplicate_ts"] == 3
        # inserted bars are loadable from the store
        with BarStore(db) as store:
            loaded = store.load_bars("AAA", 300, rth_only=False)
        assert [b.close for b in loaded] == [101.0, 102.0]
        assert loaded[0].source == "alpaca_iex_backfill_split"

    def test_pagination(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        calls = []

        def fake(url, headers, params, timeout):
            calls.append(dict(params))
            if "page_token" not in params:
                return {"bars": {"AAA": [_bar_raw(t0, 101.0)]},
                        "next_page_token": "tok1"}
            assert params["page_token"] == "tok1"
            return {"bars": {"AAA": [
                _bar_raw(t0 + timedelta(seconds=300), 102.0)]}}

        s = self._run(db, fake, monkeypatch)
        assert s["pages"] == 2 and s["inserted"] == 2
        assert len(calls) == 2

    def test_skips_forming_and_bad_close(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        t_ok = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
        t_forming = NOW - timedelta(seconds=100)  # end > now

        def fake(url, headers, params, timeout):
            return {"bars": {"AAA": [
                _bar_raw(t_ok, 101.0),
                _bar_raw(t_forming, 102.0),
                _bar_raw(t_ok + timedelta(seconds=300), 0.0),     # close<=0
                {"t": (t_ok + timedelta(seconds=600)).strftime(
                    "%Y-%m-%dT%H:%M:%SZ"), "c": "oops"},          # unparsable
            ]}}

        s = self._run(db, fake, monkeypatch,
                      end=datetime(2026, 3, 1, tzinfo=UTC))
        assert s["inserted"] == 1
        assert s["skipped_forming"] == 1
        assert s["skipped_bad_close"] == 2

    def test_request_params_batching_chunking(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        calls = []

        def fake(url, headers, params, timeout):
            assert url == "https://data.example/v2/stocks/bars"
            calls.append(dict(params))
            return {"bars": {}}

        self._run(db, fake, monkeypatch,
                  end=datetime(2026, 1, 7, tzinfo=UTC),
                  chunk_days=1, batch_symbols=1)
        # 2 symbol batches x 2 one-day chunks
        assert len(calls) == 4
        assert {p["symbols"] for p in calls} == {"AAA", "BBB"}
        for p in calls:
            assert p["timeframe"] == "5Min"
            assert p["adjustment"] == "split"
            assert p["feed"] == "iex"
        starts = sorted({p["start"] for p in calls})
        assert starts == ["2026-01-05T00:00:00Z", "2026-01-06T00:00:00Z"]

    def test_manifest_written(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        t0 = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)

        def fake(url, headers, params, timeout):
            return {"bars": {"AAA": [_bar_raw(t0, 101.0)]}}

        self._run(db, fake, monkeypatch)
        rows = _q(db, "SELECT source, manifest_json FROM ingest_manifests")
        assert len(rows) == 1
        assert rows[0][0] == "alpaca_iex_backfill_split"
        man = json.loads(rows[0][1])
        assert man["stage"] == "oracle4.0.2-backfill"
        assert man["adjustment"] == "split"
        assert man["counts"]["inserted"] == 1

    def test_unsupported_resolution(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")

        def fake(url, headers, params, timeout):  # pragma: no cover
            raise AssertionError("must not be called")

        with pytest.raises(ValueError):
            self._run(db, fake, monkeypatch, resolution_s=86400)

    def test_no_symbols_noop(self, tmp_path, monkeypatch):
        db = str(tmp_path / "bf.sqlite")
        monkeypatch.setattr(
            bars_backfill, "_http_get_json",
            lambda *a, **k: pytest.fail("must not fetch"))
        s = backfill_bars([], headers={}, data_url="https://x",
                          db_path=db, start=START, end=END, now=NOW)
        assert s["symbols"] == 0 and s["fetched"] == 0
