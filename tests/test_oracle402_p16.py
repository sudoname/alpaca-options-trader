"""Oracle 4.0.2 P1-6 tests: Kalshi integration + matched ablation.

Covers:
  * client: pagination, auth header, typed refusals (no network);
  * market_mapping: deterministic series -> underlying lookups;
  * collector: cents->probability, keep-first idempotency, fail-open
    per-series errors, and the structural no-settlement-column guarantee;
  * feature_builder: point-in-time reads, staleness bound, CDF math;
  * validation: matched A-E ablation finds a planted Kalshi signal and
    degrades to INSUFFICIENT_DATA on sparse inputs.
"""
from __future__ import annotations

import math
import os
import random
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from oracle.lab.purged_validation import Sample
from oracle.ml.kalshi import client as kclient
from oracle.ml.kalshi import collector as kcollector
from oracle.ml.kalshi import market_mapping as kmap
from oracle.ml.kalshi.client import KalshiUnavailable, get_markets
from oracle.ml.kalshi.collector import collect_snapshots
from oracle.ml.kalshi.feature_builder import (KALSHI_FEATURES,
                                              build_kalshi_features)
from oracle.ml.kalshi.validation import (run_ablation_on_samples)

UTC = timezone.utc


def _iso(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


# --------------------------------------------------------------------------- #
# market_mapping
# --------------------------------------------------------------------------- #

class TestMarketMapping:
    def test_series_of(self):
        assert kmap.series_of("KXINXU-26NOV21-T6050") == "KXINXU"
        assert kmap.series_of("inxd-26nov21-b6050") == "INXD"
        assert kmap.series_of("") == ""
        assert kmap.series_of(None) == ""

    def test_index_and_underlying(self):
        assert kmap.index_for_ticker("KXINXU-26NOV21-T6050") == "SPX"
        assert kmap.underlying_for_ticker("KXINXU-26NOV21-T6050") == "SPY"
        assert kmap.underlying_for_ticker(
            "KXNASDAQ100U-26NOV21-T24800") == "QQQ"
        # unknown series: None, never a guess
        assert kmap.index_for_ticker("KXBTCD-26NOV21-T90000") is None
        assert kmap.underlying_for_ticker("KXBTCD-26NOV21-T90000") is None

    def test_series_for_underlying(self):
        spy = kmap.series_for_underlying("SPY")
        assert "KXINXU" in spy and "INX" in spy
        assert kmap.series_for_underlying("SPX") == spy
        assert kmap.series_for_underlying("TSLA") == frozenset()


# --------------------------------------------------------------------------- #
# client
# --------------------------------------------------------------------------- #

class TestClient:
    def test_pagination_and_auth(self, monkeypatch):
        calls = []

        def fake(url, headers, params, timeout):
            calls.append((url, dict(headers), dict(params)))
            if "cursor" not in params:
                return {"markets": [{"ticker": "A-1"}, {"ticker": "A-2"}],
                        "cursor": "c2"}
            assert params["cursor"] == "c2"
            return {"markets": [{"ticker": "A-3"}, "not-a-dict"],
                    "cursor": ""}

        monkeypatch.setattr(kclient, "_http_get_json", fake)
        out = get_markets(series_ticker="KXINXU", api_key="k123")
        assert [m["ticker"] for m in out] == ["A-1", "A-2", "A-3"]
        assert len(calls) == 2
        url, headers, params = calls[0]
        assert url.endswith("/markets")
        assert headers["Authorization"] == "Bearer k123"
        assert params["series_ticker"] == "KXINXU"
        assert params["status"] == "open"

    def test_fetch_failed_refusal(self, monkeypatch):
        def fake(url, headers, params, timeout):
            raise ValueError("boom")
        monkeypatch.setattr(kclient, "_http_get_json", fake)
        with pytest.raises(KalshiUnavailable) as ei:
            get_markets(series_ticker="KXINXU")
        assert ei.value.reason == "FETCH_FAILED"

    def test_bad_payload_refusals(self, monkeypatch):
        monkeypatch.setattr(kclient, "_http_get_json",
                            lambda *a, **k: {"nope": 1})
        with pytest.raises(KalshiUnavailable) as ei:
            get_markets(series_ticker="KXINXU")
        assert ei.value.reason == "BAD_PAYLOAD"

        monkeypatch.setattr(kclient, "_http_get_json",
                            lambda *a, **k: {"markets": "wrong"})
        with pytest.raises(KalshiUnavailable) as ei:
            get_markets(series_ticker="KXINXU")
        assert ei.value.reason == "BAD_PAYLOAD"


# --------------------------------------------------------------------------- #
# collector
# --------------------------------------------------------------------------- #

def _mkt(ticker, bid, ask, last=None, **kw):
    m = {"ticker": ticker, "event_ticker": kw.get("event_ticker"),
         "status": "active", "yes_bid": bid, "yes_ask": ask,
         "last_price": last, "volume": 10, "open_interest": 5,
         "floor_strike": kw.get("floor_strike"),
         "cap_strike": kw.get("cap_strike"),
         "strike_type": kw.get("strike_type", "greater"),
         "close_time": kw.get("close_time")}
    return m


class TestCollector:
    def test_insert_prob_conversion_and_idempotency(self, tmp_path,
                                                    monkeypatch):
        db = str(tmp_path / "k.sqlite")
        now = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        markets = [
            _mkt("KXINXU-26NOV20-T6000", 48, 52, 50, floor_strike=6000),
            _mkt("KXINXU-26NOV20-T6100", None, None, None,
                 floor_strike=6100),                      # no quote at all
            _mkt("KXINXU-26NOV20-T6200", 150, 40, None,
                 floor_strike=6200),                      # bad bid -> NULL
        ]
        monkeypatch.setattr(kcollector, "get_markets",
                            lambda **kw: list(markets))
        s1 = collect_snapshots(["KXINXU"], db_path=db, now=now)
        assert s1["inserted"] == 2 and s1["skipped_no_quote"] == 1
        assert not s1["errors"]
        # keep-first idempotency: identical rerun inserts nothing new
        s2 = collect_snapshots(["KXINXU"], db_path=db, now=now)
        assert s2["inserted"] == 0 and s2["dropped_duplicate"] == 2

        conn = sqlite3.connect(db)
        try:
            row = conn.execute(
                "SELECT yes_bid, yes_ask, last_price, underlying, series "
                "FROM snapshots WHERE ticker='KXINXU-26NOV20-T6000'"
            ).fetchone()
            assert row == (0.48, 0.52, 0.50, "SPY", "KXINXU")
            bad = conn.execute(
                "SELECT yes_bid, yes_ask FROM snapshots "
                "WHERE ticker='KXINXU-26NOV20-T6200'").fetchone()
            assert bad == (None, 0.40)    # out-of-range bid -> NULL
            n_manifest = conn.execute(
                "SELECT COUNT(*) FROM ingest_manifests").fetchone()[0]
            assert n_manifest == 1
        finally:
            conn.close()

    def test_no_settlement_column_structurally(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")
        monkeypatch.setattr(kcollector, "get_markets", lambda **kw: [])
        collect_snapshots(["KXINXU"], db_path=db)
        conn = sqlite3.connect(db)
        try:
            cols = {r[1].lower() for r in conn.execute(
                "PRAGMA table_info(snapshots)")}
        finally:
            conn.close()
        assert "result" not in cols
        assert "settlement" not in cols
        assert not any("settle" in c or "outcome" in c for c in cols)

    def test_per_series_fail_open(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")

        def fake(**kw):
            if kw.get("series_ticker") == "KXINXU":
                raise KalshiUnavailable("FETCH_FAILED", "down")
            return [_mkt("KXNASDAQ100U-26NOV20-T25000", 30, 34,
                         floor_strike=25000)]

        monkeypatch.setattr(kcollector, "get_markets", fake)
        s = collect_snapshots(["KXINXU", "KXNASDAQ100U"], db_path=db)
        assert s["inserted"] == 1
        assert "KXINXU" in s["errors"]
        assert "FETCH_FAILED" in s["errors"]["KXINXU"]


# --------------------------------------------------------------------------- #
# feature_builder
# --------------------------------------------------------------------------- #

def _seed_cdf_db(db, snap_t, monkeypatch, extra=None):
    """3-market SPX CDF: P(above 100)=.9, P(above 200)=.5, P(above 300)=.1."""
    close = _iso(snap_t + timedelta(hours=2))
    markets = [
        _mkt("KXINXU-X-T100", 88, 92, floor_strike=100.0,
             event_ticker="KXINXU-X", close_time=close),
        _mkt("KXINXU-X-T200", 48, 52, floor_strike=200.0,
             event_ticker="KXINXU-X", close_time=close),
        _mkt("KXINXU-X-T300", 8, 12, floor_strike=300.0,
             event_ticker="KXINXU-X", close_time=close),
    ] + (extra or [])
    monkeypatch.setattr(kcollector, "get_markets",
                        lambda **kw: list(markets))
    collect_snapshots(["KXINXU"], db_path=db, now=snap_t)


class TestFeatureBuilder:
    def test_cdf_math(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")
        snap_t = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        _seed_cdf_db(db, snap_t, monkeypatch)
        as_of = snap_t + timedelta(minutes=10)
        f = build_kalshi_features(db, "SPY", as_of, index_spot=150.0)
        assert set(f) == set(KALSHI_FEATURES)
        assert f["kalshi_n_markets"] == 3.0
        assert f["kalshi_p_above_spot"] == pytest.approx(0.7)
        # median=200; q25 strike=262.5; q75 strike=137.5
        assert f["kalshi_implied_dispersion"] == pytest.approx(
            (262.5 - 137.5) / 200.0)
        assert f["kalshi_book_spread_mean"] == pytest.approx(0.04)
        assert f["kalshi_staleness_min"] == pytest.approx(10.0)

    def test_point_in_time_and_staleness(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")
        snap_t = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        _seed_cdf_db(db, snap_t, monkeypatch)
        # as_of BEFORE the snapshot: must see nothing (PIT)
        before = build_kalshi_features(db, "SPY",
                                       snap_t - timedelta(minutes=1),
                                       index_spot=150.0)
        assert all(v is None for v in before.values())
        # as_of too far after: staleness bound refuses
        stale = build_kalshi_features(db, "SPY",
                                      snap_t + timedelta(minutes=120),
                                      index_spot=150.0,
                                      max_staleness_min=60.0)
        assert all(v is None for v in stale.values())

    def test_nearest_event_selection(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")
        snap_t = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        far = _mkt("KXINXU-Y-T999", 1, 3, floor_strike=999.0,
                   event_ticker="KXINXU-Y",
                   close_time=_iso(snap_t + timedelta(days=5)))
        _seed_cdf_db(db, snap_t, monkeypatch, extra=[far])
        f = build_kalshi_features(db, "SPY",
                                  snap_t + timedelta(minutes=5),
                                  index_spot=150.0)
        # only the 3 markets of the sooner-closing event are used
        assert f["kalshi_n_markets"] == 3.0
        assert f["kalshi_p_above_spot"] == pytest.approx(0.7)

    def test_fail_open_paths(self, tmp_path):
        missing = str(tmp_path / "nope.sqlite")
        as_of = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        f = build_kalshi_features(missing, "SPY", as_of, index_spot=150.0)
        assert all(v is None for v in f.values())
        assert not os.path.exists(missing)     # never creates the db
        f2 = build_kalshi_features(missing, "TSLA", as_of)
        assert all(v is None for v in f2.values())

    def test_no_spot_means_no_p_above(self, tmp_path, monkeypatch):
        db = str(tmp_path / "k.sqlite")
        snap_t = datetime(2026, 11, 20, 15, 0, tzinfo=UTC)
        _seed_cdf_db(db, snap_t, monkeypatch)
        f = build_kalshi_features(db, "SPY",
                                  snap_t + timedelta(minutes=5))
        assert f["kalshi_p_above_spot"] is None       # never guessed
        assert f["kalshi_n_markets"] == 3.0


# --------------------------------------------------------------------------- #
# validation (matched ablation A-E)
# --------------------------------------------------------------------------- #

def _build_ablation_fixture(tmp_path, monkeypatch, n_sessions=8,
                            per_session=12, seed=123):
    """Samples with noise-only base features + a Kalshi-encoded signal.

    For each sample, the snapshot's implied P(above 200) equals
    0.5 + 0.35*u and the label is 0.02*u + tiny noise, so experiment B
    (base+kalshi) must beat A (base) out of sample, while the permuted
    placebo D must not.
    """
    db = str(tmp_path / "kalshi.sqlite")
    rng = random.Random(seed)
    samples = []
    i = 0
    for d in range(n_sessions):
        session = f"2026-01-{5 + d:02d}"
        day = datetime(2026, 1, 5 + d, 15, 0, tzinfo=UTC)
        for j in range(per_session):
            as_of = day + timedelta(minutes=30 * j)
            u = rng.uniform(-1.0, 1.0)
            y = 0.02 * u + 0.002 * rng.gauss(0.0, 1.0)
            p = 0.5 + 0.35 * u
            close = _iso(as_of + timedelta(hours=2))
            markets = [
                _mkt("KXINXU-Z-T199", 100 * (p + 0.02), 100 * (p + 0.02),
                     floor_strike=199.0, event_ticker="KXINXU-Z",
                     close_time=close),
                _mkt("KXINXU-Z-T201", 100 * (p - 0.02), 100 * (p - 0.02),
                     floor_strike=201.0, event_ticker="KXINXU-Z",
                     close_time=close),
            ]
            monkeypatch.setattr(kcollector, "get_markets",
                                lambda **kw: list(markets))
            s = collect_snapshots(["KXINXU"], db_path=db, now=as_of)
            assert s["inserted"] == 2
            samples.append(Sample(
                decision_id=f"s{i:04d}", session=session, as_of=as_of,
                kind="bars_synthetic",
                features={"f0": rng.gauss(0.0, 1.0),
                          "f1": rng.gauss(0.0, 1.0)},
                label_value=y, label_status="ok",
                label_end=as_of + timedelta(minutes=15),
                label_available_at=as_of + timedelta(minutes=15)))
            i += 1
    return samples, db


class TestAblation:
    def test_planted_kalshi_signal_detected(self, tmp_path, monkeypatch):
        samples, db = _build_ablation_fixture(tmp_path, monkeypatch)
        report = run_ablation_on_samples(
            samples, db, "SPY",
            index_spot_fn=lambda s: 200.0,
            max_staleness_min=15.0, n_boot=300)
        assert report["kalshi_coverage"] == pytest.approx(1.0)
        exps = report["experiments"]
        assert set(exps) == {"A_base", "B_base_plus_kalshi",
                             "C_kalshi_only", "D_base_plus_placebo",
                             "E_base_plus_lagged"}
        a, b = exps["A_base"], exps["B_base_plus_kalshi"]
        assert a["n_oos"] == b["n_oos"] > 0      # matched OOS sets
        assert b["mse"] < a["mse"]               # kalshi helps
        delta_b = report["deltas_vs_A"]["B_base_plus_kalshi"]
        assert delta_b["mse_improvement"]["mean"] > 0.0
        delta_d = report["deltas_vs_A"]["D_base_plus_placebo"]
        # placebo must not show a significant gain
        assert (delta_d["mse_improvement"]["ci_lo"] is None
                or delta_d["mse_improvement"]["ci_lo"] <= 0.0)
        assert report["verdict"] == "SIGNAL_CANDIDATE"
        assert "fingerprint" in report

    def test_lagged_experiment_loses_coverage_or_signal(self, tmp_path,
                                                        monkeypatch):
        samples, db = _build_ablation_fixture(tmp_path, monkeypatch,
                                              n_sessions=6)
        # lag 60min with 15min staleness: lagged reads mostly miss
        report = run_ablation_on_samples(
            samples, db, "SPY", index_spot_fn=lambda s: 200.0,
            max_staleness_min=15.0, lag_min=60.0, n_boot=200)
        if report["verdict"] == "INSUFFICIENT_DATA":
            return
        b = report["experiments"]["B_base_plus_kalshi"]["mse"]
        e = report["experiments"]["E_base_plus_lagged"]["mse"]
        assert e >= b    # stale info can only be worse or equal here

    def test_insufficient_samples(self, tmp_path):
        report = run_ablation_on_samples(
            [], str(tmp_path / "k.sqlite"), "SPY")
        assert report["verdict"] == "INSUFFICIENT_DATA"
        assert "usable samples" in report["verdict_reason"]

    def test_insufficient_coverage(self, tmp_path, monkeypatch):
        samples, _ = _build_ablation_fixture(tmp_path, monkeypatch,
                                             n_sessions=4)
        missing_db = str(tmp_path / "empty.sqlite")   # never created
        report = run_ablation_on_samples(
            samples, missing_db, "SPY", index_spot_fn=lambda s: 200.0)
        assert report["verdict"] == "INSUFFICIENT_DATA"
        assert "coverage" in report["verdict_reason"]
        assert not os.path.exists(missing_db)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
