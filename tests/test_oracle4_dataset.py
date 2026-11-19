"""Phase 1 acceptance tests for the Oracle 4.0 feature/label store.

Covers (per the master prompt): timestamp edge cases, no future joins,
duplicate handling, censoring, missing outcomes preserved, reproducibility,
and the CLI + coverage report naming real gaps. All hermetic: a tiny ledger
is synthesized into a tmp sqlite file with the production DDL.
"""
from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import dataset as ds
from oracle.ml import ingest
from oracle.ml.features import FeatureConfig, ObsIndex, build_feature_row
from oracle.ml.labels import LabelConfig, build_labels, classify_outcome
from oracle.ml.schema import (
    LabelStatus,
    OutcomeStatus,
    UnderlyingObs,
    parse_utc,
    rows_hash,
)

UTC = timezone.utc
T0 = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)   # mid-session UTC


# --------------------------------------------------------------------------- #
# tmp ledger builder (production DDL)
# --------------------------------------------------------------------------- #

_SNAPSHOT_COLS = (
    "snapshot_id", "decision_id", "schema_version", "kind", "accepted",
    "reject_code", "event_at_utc", "received_at_utc", "symbol", "underlying",
    "expiration", "strike", "option_type", "multiplier", "quote_bid",
    "quote_ask", "quote_bid_size", "quote_ask_size", "quote_ts_utc",
    "quote_age_s", "underlying_price", "iv", "delta", "gamma", "theta",
    "vega", "direction", "score", "p_win", "ev", "signals_json", "regime",
    "cost_json", "sizing_json", "risk_json", "prediction_json",
    "provenance_json", "strategy_version", "config_hash", "code_rev",
    "extra_json",
)
_EVENT_COLS = (
    "event_key", "decision_id", "schema_version", "event_type",
    "event_at_utc", "received_at_utc", "broker_order_id", "qty", "price",
    "fees", "status", "source", "payload_json",
)


def _make_ledger(path):
    conn = sqlite3.connect(str(path))
    conn.execute(f"CREATE TABLE snapshots ({', '.join(_SNAPSHOT_COLS)})")
    conn.execute(f"CREATE TABLE events ({', '.join(_EVENT_COLS)})")
    return conn


def _snap(conn, **kw):
    row = {c: None for c in _SNAPSHOT_COLS}
    row.update(schema_version=1, kind="candidate", accepted=1)
    row.update(kw)
    if row["received_at_utc"] is None:
        row["received_at_utc"] = row["event_at_utc"]
    conn.execute(
        f"INSERT INTO snapshots VALUES ({','.join('?' * len(_SNAPSHOT_COLS))})",
        [row[c] for c in _SNAPSHOT_COLS])


def _event(conn, **kw):
    row = {c: None for c in _EVENT_COLS}
    row.update(schema_version=1)
    row.update(kw)
    if row["received_at_utc"] is None:
        row["received_at_utc"] = row["event_at_utc"]
    conn.execute(
        f"INSERT INTO events VALUES ({','.join('?' * len(_EVENT_COLS))})",
        [row[c] for c in _EVENT_COLS])


def _iso(dt):
    return dt.astimezone(UTC).isoformat()


def _obs_rows(conn, underlying, start, prices, step_min=5, sid_prefix="s"):
    """One snapshot row per price observation, step_min apart."""
    for i, p in enumerate(prices):
        t = start + timedelta(minutes=i * step_min)
        _snap(conn, snapshot_id=f"{sid_prefix}{underlying}{i}",
              decision_id=f"d{sid_prefix}{underlying}{i}",
              kind="shadow_candidate", accepted=0,
              event_at_utc=_iso(t), underlying=underlying,
              underlying_price=p)


# --------------------------------------------------------------------------- #
# schema / timestamp edge cases
# --------------------------------------------------------------------------- #

class TestTimestamps:
    def test_parse_utc_offset_and_z(self):
        a = parse_utc("2026-11-04T15:00:00+00:00")
        b = parse_utc("2026-11-04T15:00:00Z")
        c = parse_utc("2026-11-04T10:00:00-05:00")
        assert a == b == c == T0

    def test_parse_utc_rejects_naive_and_garbage(self):
        assert parse_utc("2026-11-04T15:00:00") is None   # naive: data defect
        assert parse_utc("not-a-time") is None
        assert parse_utc(None) is None
        assert parse_utc("") is None


# --------------------------------------------------------------------------- #
# ingest: dedup + validation
# --------------------------------------------------------------------------- #

class TestIngest:
    def test_duplicate_obs_dropped_and_counted(self, tmp_path):
        p = tmp_path / "l.db"
        conn = _make_ledger(p)
        # same (underlying, event_at) twice; first by received_at wins
        _snap(conn, snapshot_id="a", decision_id="da", event_at_utc=_iso(T0),
              received_at_utc=_iso(T0), underlying="SPY",
              underlying_price=100.0)
        _snap(conn, snapshot_id="b", decision_id="db", event_at_utc=_iso(T0),
              received_at_utc=_iso(T0 + timedelta(seconds=1)),
              underlying="SPY", underlying_price=999.0)
        conn.commit(); conn.close()

        ro = ingest.connect(str(p))
        obs, report = ingest.load_underlying_obs(ro)
        ro.close()
        assert report["dropped_duplicate"] == 1
        assert len(obs["SPY"]) == 1
        assert obs["SPY"][0].price == 100.0          # keep-first
        assert obs["SPY"][0].source_snapshot_id == "a"

    def test_bad_timestamp_rows_dropped_not_guessed(self, tmp_path):
        p = tmp_path / "l.db"
        conn = _make_ledger(p)
        _snap(conn, snapshot_id="a", decision_id="da",
              event_at_utc="2026-11-04T15:00:00",     # naive -> defect
              received_at_utc=_iso(T0), underlying="SPY",
              underlying_price=100.0)
        conn.commit(); conn.close()

        ro = ingest.connect(str(p))
        obs, report = ingest.load_underlying_obs(ro)
        decs, dreport = ingest.load_decisions(ro)
        ro.close()
        assert report["dropped_bad_timestamp"] == 1
        assert obs == {}
        assert dreport["dropped_bad_timestamp"] == 1
        assert decs == []

    def test_duplicate_decision_id_keep_first(self, tmp_path):
        p = tmp_path / "l.db"
        conn = _make_ledger(p)
        _snap(conn, snapshot_id="a", decision_id="dup", event_at_utc=_iso(T0),
              underlying="SPY", underlying_price=100.0, score=1.0)
        _snap(conn, snapshot_id="b", decision_id="dup",
              event_at_utc=_iso(T0 + timedelta(minutes=1)),
              underlying="SPY", underlying_price=101.0, score=2.0)
        conn.commit(); conn.close()

        ro = ingest.connect(str(p))
        decs, report = ingest.load_decisions(ro)
        ro.close()
        assert report["dropped_duplicate_decision_id"] == 1
        assert len(decs) == 1 and decs[0].score == 1.0

    def test_event_at_fallback_to_received(self, tmp_path):
        p = tmp_path / "l.db"
        conn = _make_ledger(p)
        _event(conn, event_key="e1", decision_id="d1", event_type="outcome",
               event_at_utc=None, received_at_utc=_iso(T0),
               payload_json=json.dumps({"class": "realized"}))
        conn.commit(); conn.close()

        ro = ingest.connect(str(p))
        events, report = ingest.load_events(ro)
        ro.close()
        assert report["event_at_fallback_received"] == 1
        assert events["d1"][0].event_at == T0

    def test_ledger_opened_read_only(self, tmp_path):
        p = tmp_path / "l.db"
        conn = _make_ledger(p)
        conn.commit(); conn.close()
        ro = ingest.connect(str(p))
        with pytest.raises(sqlite3.OperationalError):
            ro.execute("INSERT INTO events (event_key, decision_id, "
                       "schema_version, received_at_utc) "
                       "VALUES ('x', 'y', 1, 'z')")
        ro.close()


# --------------------------------------------------------------------------- #
# features: strict point-in-time semantics
# --------------------------------------------------------------------------- #

def _mk_index(*obs):
    by_u = {}
    for o in obs:
        by_u.setdefault(o.underlying, []).append(o)
    for s in by_u.values():
        s.sort(key=lambda o: o.event_at)
    return ObsIndex(by_u)


def _obs(minutes, price, avail_lag_s=0, underlying="SPY"):
    t = T0 + timedelta(minutes=minutes)
    return UnderlyingObs(underlying=underlying, event_at=t,
                         available_at=t + timedelta(seconds=avail_lag_s),
                         price=price, source_snapshot_id=f"s{minutes}")


class TestNoFutureJoins:
    def test_future_event_never_used(self):
        idx = _mk_index(_obs(-30, 100.0), _obs(10, 200.0))
        got = idx.asof("SPY", T0)
        assert got is not None and got.price == 100.0

    def test_late_arriving_data_excluded_by_availability(self):
        # event happened before T0 but only became knowable after T0
        idx = _mk_index(_obs(-30, 100.0),
                        _obs(-5, 150.0, avail_lag_s=600))
        got = idx.asof("SPY", T0)
        assert got is not None and got.price == 100.0   # not 150

    def test_boundary_obs_at_exactly_as_of_included(self):
        idx = _mk_index(_obs(-30, 100.0), _obs(0, 111.0))
        got = idx.asof("SPY", T0)
        assert got is not None and got.price == 111.0

    def test_ret_15m_uses_only_past(self):
        from oracle.ml.schema import DecisionRecord
        idx = _mk_index(_obs(-15, 100.0), _obs(0, 101.0), _obs(15, 999.0))
        d = DecisionRecord(
            decision_id="d1", snapshot_id="s1", kind="candidate",
            accepted=True, reject_code=None, as_of=T0, available_at=T0,
            underlying="SPY", symbol=None, expiration=None, strike=None,
            option_type=None, multiplier=None, quote_bid=None, quote_ask=None,
            underlying_price=101.0, iv=None, delta=None, direction=None,
            score=None, p_win=None, ev=None)
        row = build_feature_row(d, idx, FeatureConfig())
        import math
        assert row.features["ret_15m"] == pytest.approx(math.log(101.0 / 100.0))

    def test_stale_base_obs_yields_none_not_a_guess(self):
        from oracle.ml.schema import DecisionRecord
        # only obs is 3 hours old; 15m return must be None
        idx = _mk_index(_obs(-180, 100.0))
        d = DecisionRecord(
            decision_id="d1", snapshot_id="s1", kind="candidate",
            accepted=True, reject_code=None, as_of=T0, available_at=T0,
            underlying="SPY", symbol=None, expiration=None, strike=None,
            option_type=None, multiplier=None, quote_bid=None, quote_ask=None,
            underlying_price=101.0, iv=None, delta=None, direction=None,
            score=None, p_win=None, ev=None)
        row = build_feature_row(d, idx, FeatureConfig())
        assert row.features["ret_15m"] is None
        assert row.features["ret_60m"] is None


# --------------------------------------------------------------------------- #
# labels: horizons, censoring, neutral band, first touch
# --------------------------------------------------------------------------- #

def _decision(decision_id="d1", as_of=T0, price=100.0, kind="candidate",
              accepted=True, reject_code=None, symbol=None, iv=None):
    from oracle.ml.schema import DecisionRecord
    return DecisionRecord(
        decision_id=decision_id, snapshot_id=f"s_{decision_id}", kind=kind,
        accepted=accepted, reject_code=reject_code, as_of=as_of,
        available_at=as_of, underlying="SPY", symbol=symbol, expiration=None,
        strike=None, option_type=None, multiplier=None, quote_bid=None,
        quote_ask=None, underlying_price=price, iv=iv, delta=None,
        direction=None, score=None, p_win=None, ev=None)


def _labels_by_key(labels):
    return {(l.horizon, l.label_type): l for l in labels}


class TestLabels:
    def test_log_return_and_direction_at_15m(self):
        idx = _mk_index(_obs(0, 100.0), _obs(15, 102.0), _obs(30, 103.0))
        cov = T0 + timedelta(minutes=30)
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        lr = by[("15m", "log_return")]
        import math
        assert lr.status is LabelStatus.OK
        assert lr.value == pytest.approx(math.log(102.0 / 100.0))
        assert lr.label_end == T0 + timedelta(minutes=15)
        assert lr.available_at is not None
        assert by[("15m", "direction")].value == 1.0

    def test_neutral_band_direction_zero(self):
        idx = _mk_index(_obs(0, 100.0), _obs(15, 100.0001),
                        _obs(30, 100.0))
        cov = T0 + timedelta(minutes=30)
        by = _labels_by_key(build_labels(
            _decision(), idx, cov, config=LabelConfig(neutral_band=0.001)))
        assert by[("15m", "direction")].value == 0.0

    def test_horizon_beyond_coverage_is_censored(self):
        idx = _mk_index(_obs(0, 100.0), _obs(5, 101.0))
        cov = T0 + timedelta(minutes=5)
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        assert by[("60m", "log_return")].status is LabelStatus.CENSORED
        assert by[("60m", "log_return")].value is None

    def test_gap_inside_coverage_is_missing_not_censored(self):
        # coverage runs long but there is no obs near t+15m
        idx = _mk_index(_obs(0, 100.0), _obs(120, 105.0))
        cov = T0 + timedelta(minutes=120)
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        lab = by[("15m", "log_return")]
        assert lab.status is LabelStatus.MISSING
        assert lab.meta.get("why") == "obs_gap"

    def test_session_label_censored_on_final_coverage_session(self):
        idx = _mk_index(_obs(0, 100.0), _obs(60, 101.0))
        cov = T0 + timedelta(minutes=60)            # same session
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        assert by[("1s", "log_return")].status is LabelStatus.CENSORED

    def test_session_label_ok_when_session_complete(self):
        next_day = T0 + timedelta(days=1)
        idx = _mk_index(_obs(0, 100.0), _obs(60, 110.0),
                        UnderlyingObs("SPY", next_day, next_day, 111.0, "x"))
        cov = next_day
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        lab = by[("1s", "log_return")]
        import math
        assert lab.status is LabelStatus.OK
        assert lab.value == pytest.approx(math.log(110.0 / 100.0))

    def test_iv_change_missing_without_iv_series(self):
        idx = _mk_index(_obs(0, 100.0), _obs(30, 101.0))
        cov = T0 + timedelta(minutes=30)
        by = _labels_by_key(build_labels(_decision(), idx, cov))
        lab = by[("15m", "iv_change")]
        assert lab.status in (LabelStatus.MISSING, LabelStatus.CENSORED)
        assert lab.value is None

    def test_first_touch_stop_from_outcome_event(self):
        idx = _mk_index(_obs(0, 100.0), _obs(30, 99.0))
        cov = T0 + timedelta(minutes=30)
        exit_t = T0 + timedelta(minutes=20)
        events = [ingest.LedgerEvent(
            event_key="e1", decision_id="d1", event_type="outcome",
            event_at=exit_t, available_at=exit_t, broker_order_id=None,
            qty=1.0, price=0.9, fees=None, status=None,
            payload={"class": "realized", "reason": "dynamic_stop_loss"})]
        by = _labels_by_key(build_labels(_decision(), idx, cov,
                                         events=events))
        lab = by[("event", "first_touch")]
        assert lab.status is LabelStatus.OK
        assert lab.value == pytest.approx(20.0)
        assert lab.meta["touch"] == "stop"

    def test_first_touch_censored_when_filled_but_unresolved(self):
        idx = _mk_index(_obs(0, 100.0))
        cov = T0 + timedelta(minutes=30)
        events = [ingest.LedgerEvent(
            event_key="e1", decision_id="d1", event_type="fill",
            event_at=T0, available_at=T0, broker_order_id=None,
            qty=1.0, price=1.0, fees=None, status=None, payload={})]
        by = _labels_by_key(build_labels(_decision(), idx, cov,
                                         events=events))
        lab = by[("event", "first_touch")]
        assert lab.status is LabelStatus.CENSORED
        assert lab.label_end == cov


# --------------------------------------------------------------------------- #
# outcome taxonomy
# --------------------------------------------------------------------------- #

def _ev(event_type, minutes=0, payload=None, price=None, qty=None,
        key=None):
    t = T0 + timedelta(minutes=minutes)
    return ingest.LedgerEvent(
        event_key=key or f"{event_type}{minutes}", decision_id="d1",
        event_type=event_type, event_at=t, available_at=t,
        broker_order_id=None, qty=qty, price=price, fees=None,
        status=None, payload=payload or {})


class TestOutcomeTaxonomy:
    def test_policy_skip(self):
        rec = classify_outcome(
            _decision(accepted=False, reject_code="ev_gate:x"), [])
        assert rec.status is OutcomeStatus.POLICY_SKIP
        assert rec.reason == "ev_gate:x"

    def test_counterfactual_shadow(self):
        rec = classify_outcome(
            _decision(kind="shadow_candidate", accepted=False),
            [_ev("outcome", 30, {"class": "simulated", "net_pnl_pct": 0.1})])
        assert rec.status is OutcomeStatus.COUNTERFACTUAL
        assert rec.net_pnl_pct is None            # simulated stays in meta
        assert rec.meta["simulated"]["net_pnl_pct"] == 0.1

    def test_mark_derived_when_net_dollars_null(self):
        rec = classify_outcome(_decision(), [
            _ev("submitted", 0), _ev("fill", 1, price=1.05, qty=1.0),
            _ev("outcome", 60, {"class": "realized", "gross_pnl_pct": 0.05,
                                "net_pnl_dollars": None,
                                "net_pnl_pct": 0.04, "reason": "take_profit"})])
        assert rec.status is OutcomeStatus.MARK_DERIVED
        assert rec.net_pnl_pct == 0.04
        assert rec.fill_price == 1.05

    def test_fills_actual_when_net_dollars_present(self):
        rec = classify_outcome(_decision(), [
            _ev("submitted", 0), _ev("fill", 1, price=1.05, qty=1.0),
            _ev("outcome", 60, {"class": "realized", "gross_pnl_pct": 0.05,
                                "net_pnl_dollars": 12.5,
                                "net_pnl_pct": 0.04, "reason": "exit"})])
        assert rec.status is OutcomeStatus.FILLS_ACTUAL
        assert rec.net_pnl_dollars == 12.5

    def test_confirmed_nonfill(self):
        rec = classify_outcome(_decision(), [
            _ev("submitted", 0), _ev("cancel", 5)])
        assert rec.status is OutcomeStatus.CONFIRMED_NONFILL

    def test_unknown_outcomes_stay_missing_never_losses(self):
        # accepted, never submitted
        rec = classify_outcome(_decision(), [])
        assert rec.status is OutcomeStatus.MISSING
        assert rec.meta["stage"] == "no_submission"
        assert rec.net_pnl_pct is None
        # submitted, no fill, no cancel
        rec = classify_outcome(_decision(), [_ev("submitted", 0)])
        assert rec.status is OutcomeStatus.MISSING
        assert rec.meta["stage"] == "submitted_unresolved"
        # filled, no outcome
        rec = classify_outcome(_decision(), [
            _ev("submitted", 0), _ev("fill", 1, price=1.0)])
        assert rec.status is OutcomeStatus.MISSING
        assert rec.meta["stage"] == "filled_no_outcome"


# --------------------------------------------------------------------------- #
# end-to-end build: reproducibility + CLI + coverage report
# --------------------------------------------------------------------------- #

def _populate_small_ledger(path):
    conn = _make_ledger(path)
    start = T0 - timedelta(minutes=60)
    prices = [100 + 0.1 * i for i in range(30)]       # 5-min grid, 145 min
    _obs_rows(conn, "SPY", start, prices)
    # one real candidate mid-series with quotes + option identity
    _snap(conn, snapshot_id="cand1", decision_id="dcand1",
          event_at_utc=_iso(T0), underlying="SPY",
          symbol="SPY261120C00100000", expiration="2026-11-20",
          strike=100.0, option_type="call", multiplier=100.0,
          quote_bid=1.0, quote_ask=1.1, underlying_price=100.5,
          iv=0.25, delta=0.5, direction="CALL", score=7.5, p_win=0.6,
          ev=10.0, signals_json=json.dumps({"momentum": 0.2,
                                            "volatility": 0.01}),
          sizing_json=json.dumps({"qty": 1}), regime="trending")
    # a rejected candidate
    _snap(conn, snapshot_id="cand2", decision_id="dcand2",
          event_at_utc=_iso(T0 + timedelta(minutes=5)), underlying="SPY",
          accepted=0, reject_code="profit_gate:block",
          underlying_price=100.6)
    # events for the accepted candidate -> mark-derived outcome
    _event(conn, event_key="e_sub", decision_id="dcand1",
           event_type="submitted", event_at_utc=_iso(T0))
    _event(conn, event_key="e_fill", decision_id="dcand1",
           event_type="fill", event_at_utc=_iso(T0 + timedelta(minutes=1)),
           qty=1.0, price=1.05)
    _event(conn, event_key="e_out", decision_id="dcand1",
           event_type="outcome",
           event_at_utc=_iso(T0 + timedelta(minutes=45)),
           payload_json=json.dumps({"class": "realized",
                                    "gross_pnl_pct": 0.05,
                                    "net_pnl_dollars": None,
                                    "net_pnl_pct": 0.04,
                                    "reason": "dynamic_stop_loss"}))
    conn.commit(); conn.close()


class TestBuildDataset:
    def test_reproducible_dataset_hash(self, tmp_path):
        ledger = tmp_path / "ledger.db"
        _populate_small_ledger(ledger)
        m1 = ds.build_dataset(str(ledger), str(tmp_path / "o1.db"),
                              str(tmp_path / "m1.json"))
        m2 = ds.build_dataset(str(ledger), str(tmp_path / "o2.db"),
                              str(tmp_path / "m2.json"))
        assert m1["dataset_hash"] == m2["dataset_hash"]
        assert m1["counts"] == m2["counts"]

    def test_store_contents_and_meta(self, tmp_path):
        ledger = tmp_path / "ledger.db"
        _populate_small_ledger(ledger)
        out = tmp_path / "out.db"
        manifest = ds.build_dataset(str(ledger), str(out),
                                    str(tmp_path / "m.json"))
        conn = sqlite3.connect(str(out))
        n_feat = conn.execute("SELECT COUNT(*) FROM features").fetchone()[0]
        n_lab = conn.execute("SELECT COUNT(*) FROM labels").fetchone()[0]
        n_out = conn.execute("SELECT COUNT(*) FROM outcomes").fetchone()[0]
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        statuses = {r[0] for r in conn.execute(
            "SELECT DISTINCT status FROM outcomes")}
        conn.close()
        assert n_feat == manifest["counts"]["feature_rows"] > 0
        assert n_lab == manifest["counts"]["label_rows"] > 0
        assert n_out == n_feat
        assert meta["dataset_hash"] == manifest["dataset_hash"]
        # mark-derived candidate + policy skip + counterfactual shadows
        assert {"mark_derived", "policy_skip", "counterfactual"} <= statuses

    def test_manifest_names_real_gaps(self, tmp_path):
        ledger = tmp_path / "ledger.db"
        _populate_small_ledger(ledger)
        manifest = ds.build_dataset(str(ledger), str(tmp_path / "o.db"),
                                    str(tmp_path / "m.json"))
        gaps = " ".join(manifest["coverage"]["known_gaps"])
        assert "iv_change" in gaps          # no same-contract IV series
        assert "mark_derived" in gaps       # fills-actual absent

    def test_window_filter_excludes_out_of_range_decisions(self, tmp_path):
        ledger = tmp_path / "ledger.db"
        _populate_small_ledger(ledger)
        m = ds.build_dataset(
            str(ledger), str(tmp_path / "o.db"), str(tmp_path / "m.json"),
            start=T0 + timedelta(days=5), end=T0 + timedelta(days=6))
        assert m["counts"]["decisions"] == 0
        assert m["ingest_reports"]["decisions"]["dropped_out_of_range"] > 0

    def test_cli_smoke(self, tmp_path, capsys):
        ledger = tmp_path / "ledger.db"
        _populate_small_ledger(ledger)
        out = tmp_path / "o.db"
        man = tmp_path / "m.json"
        rc = ds.main(["--ledger", str(ledger), "--output", str(out),
                      "--manifest", str(man), "--neutral-band", "0.002"])
        assert rc == 0
        assert out.exists() and man.exists()
        printed = capsys.readouterr().out
        assert "dataset_hash=" in printed
        assert "GAP:" in printed
        loaded = json.loads(man.read_text())
        assert loaded["config"]["label"]["neutral_band"] == 0.002
        assert loaded["input"]["ledger_sha256"]

    def test_rows_hash_order_independent(self):
        a = {"x": 1.0, "t": T0}
        b = {"x": 2.0, "t": T0}
        assert rows_hash([a, b]) == rows_hash([b, a])
        assert rows_hash([a]) != rows_hash([b])
