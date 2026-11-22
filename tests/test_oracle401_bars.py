"""Oracle 4.0.1 Stage A acceptance tests (master prompt battery).

Covers: late-arriving bar exclusion, completed-bar boundary, stale-gap
refusal, daily/intraday mismatch (never rescaled), no-synthetic-grid in
bars mode, DST + early-close calendar handling, overnight/intraday return
separation, duplicate timestamps, point-in-time availability in valuation
history, and the label-provenance fixes (exact-match exit reasons,
simulated outcomes never OK).  All hermetic (tmp sqlite stores).
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, timezone

import pytest

from oracle.ml import bars as barsmod
from oracle.ml import ingest
from oracle.ml import shadow as shmod
from oracle.ml import valuation as val
from oracle.ml.bars import Bar, BarStore, BarsUnavailable, intraday_log_returns
from oracle.ml.features import ObsIndex
from oracle.ml.labels import build_labels
from oracle.ml.schema import DecisionRecord, LabelStatus, UnderlyingObs

UTC = timezone.utc

# 2026-01-05 is a Monday under EST: RTH = 14:30-21:00 UTC.
S1_OPEN = datetime(2026, 1, 5, 14, 30, tzinfo=UTC)
# 2026-07-06 is a Monday under EDT: RTH = 13:30-20:00 UTC.
S2_OPEN = datetime(2026, 7, 6, 13, 30, tzinfo=UTC)


def _bar(ts: datetime, close: float, sym: str = "SPY", res: int = 60,
         src: str = "test") -> Bar:
    return Bar(symbol=sym, resolution_s=res, ts_utc=ts, open=close,
               high=close, low=close, close=close, volume=1.0,
               session_date=barsmod.session_date_of(ts),
               rth=barsmod.is_rth(ts), source=src)


def _minute_bars(start: datetime, n: int, base: float = 100.0):
    """n one-minute bars with mild price variation (nonzero vol)."""
    return [_bar(start + timedelta(minutes=i), base + (i % 5) * 0.2)
            for i in range(n)]


@pytest.fixture()
def store_path(tmp_path):
    return str(tmp_path / "bars.sqlite")


def _write(store_path, bars):
    st = BarStore(store_path, readonly=False)
    counts = st.insert_bars(bars)
    st.close()
    return counts


# --------------------------------------------------------------------------- #
# completed-bar point-in-time rule
# --------------------------------------------------------------------------- #

class TestCompletedBarPIT:
    def test_late_arriving_bar_excluded(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 10))
        st = BarStore(store_path)
        # as_of mid-way through the 14:35 bar: that bar has not completed
        as_of = S1_OPEN + timedelta(minutes=5, seconds=30)
        got = st.load_bars("SPY", 60, as_of=as_of)
        st.close()
        assert got[-1].ts_utc == S1_OPEN + timedelta(minutes=4)
        assert all(b.end_utc <= as_of for b in got)

    def test_completed_bar_boundary_inclusive(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 10))
        st = BarStore(store_path)
        # as_of exactly at a bar END: that bar exists; the next does not
        as_of = S1_OPEN + timedelta(minutes=5)
        got = st.load_bars("SPY", 60, as_of=as_of)
        st.close()
        assert got[-1].ts_utc == S1_OPEN + timedelta(minutes=4)
        assert got[-1].end_utc == as_of

    def test_mid_bar_never_visible(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 10))
        st = BarStore(store_path)
        as_of = S1_OPEN + timedelta(minutes=4, seconds=59)
        got = st.load_bars("SPY", 60, as_of=as_of)
        st.close()
        # the 14:34 bar ends at 14:35 > as_of -> excluded
        assert got[-1].ts_utc == S1_OPEN + timedelta(minutes=3)


# --------------------------------------------------------------------------- #
# structured refusals (never guess, never rescale)
# --------------------------------------------------------------------------- #

class TestStructuredRefusals:
    def test_store_missing(self, tmp_path):
        with pytest.raises(BarsUnavailable) as ei:
            BarStore(str(tmp_path / "nope.sqlite"), readonly=True)
        assert ei.value.reason == "STORE_MISSING"

    def test_symbol_unknown(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 5))
        st = BarStore(store_path)
        with pytest.raises(BarsUnavailable) as ei:
            st.load_bars("QQQ", 60)
        st.close()
        assert ei.value.reason == "SYMBOL_UNKNOWN"

    def test_daily_intraday_mismatch_refused_not_rescaled(self, store_path):
        # only DAILY bars exist; a 1-minute request must refuse with a
        # structured reason -- never return daily closes as intraday steps
        daily_ts = datetime(2026, 1, 5, 0, 0, tzinfo=UTC)
        _write(store_path, [_bar(daily_ts, 100.0, res=86400)])
        st = BarStore(store_path)
        with pytest.raises(BarsUnavailable) as ei:
            st.load_bars("SPY", 60, rth_only=False)
        st.close()
        assert ei.value.reason == "RESOLUTION_UNAVAILABLE"

    def test_no_coverage_window(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 5))
        st = BarStore(store_path)
        with pytest.raises(BarsUnavailable) as ei:
            st.load_bars("SPY", 60, start=S2_OPEN,
                         end=S2_OPEN + timedelta(hours=1))
        st.close()
        assert ei.value.reason == "NO_COVERAGE"

    def test_stale_gap_rejected(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 10))
        st = BarStore(store_path)
        as_of = S1_OPEN + timedelta(hours=3)
        with pytest.raises(BarsUnavailable) as ei:
            st.load_bars("SPY", 60, as_of=as_of,
                         max_staleness=timedelta(minutes=30))
        st.close()
        assert ei.value.reason == "STALE"

    def test_insufficient_history(self, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 5))
        st = BarStore(store_path)
        with pytest.raises(BarsUnavailable) as ei:
            st.load_bars("SPY", 60, min_bars=30)
        st.close()
        assert ei.value.reason == "INSUFFICIENT_HISTORY"


# --------------------------------------------------------------------------- #
# duplicate timestamps
# --------------------------------------------------------------------------- #

class TestDuplicates:
    def test_duplicate_timestamps_keep_first_and_counted(self, store_path):
        c1 = _write(store_path, [_bar(S1_OPEN, 100.0)])
        c2 = _write(store_path, [_bar(S1_OPEN, 999.0)])
        assert c1 == {"inserted": 1, "dropped_duplicate_ts": 0}
        assert c2 == {"inserted": 0, "dropped_duplicate_ts": 1}
        st = BarStore(store_path)
        got = st.load_bars("SPY", 60)
        st.close()
        assert len(got) == 1 and got[0].close == 100.0


# --------------------------------------------------------------------------- #
# exchange calendar: DST + early closes
# --------------------------------------------------------------------------- #

class TestCalendar:
    def test_dst_rth_window(self):
        o_jan, c_jan = barsmod.rth_window_utc(date(2026, 1, 5))   # EST
        o_jul, c_jul = barsmod.rth_window_utc(date(2026, 7, 6))   # EDT
        assert (o_jan.hour, o_jan.minute) == (14, 30)
        assert (c_jan.hour, c_jan.minute) == (21, 0)
        assert (o_jul.hour, o_jul.minute) == (13, 30)
        assert (c_jul.hour, c_jul.minute) == (20, 0)

    def test_is_rth_respects_dst(self):
        assert barsmod.is_rth(datetime(2026, 1, 5, 14, 30, tzinfo=UTC))
        assert not barsmod.is_rth(datetime(2026, 1, 5, 13, 30, tzinfo=UTC))
        assert barsmod.is_rth(datetime(2026, 7, 6, 13, 30, tzinfo=UTC))
        assert not barsmod.is_rth(datetime(2026, 7, 6, 20, 30, tzinfo=UTC))

    def test_early_close_observed_not_padded(self, store_path):
        # session 1: last bar 18:00 UTC (13:00 ET) -> early close observed
        early = [_bar(S1_OPEN, 100.0),
                 _bar(S1_OPEN + timedelta(minutes=30), 100.5),
                 _bar(datetime(2026, 1, 5, 18, 0, tzinfo=UTC), 101.0)]
        # session 2: last bar starts 20:59 UTC (15:59 ET) -> full session
        s2 = datetime(2026, 1, 6, 14, 30, tzinfo=UTC)
        full = [_bar(s2, 100.0),
                _bar(datetime(2026, 1, 6, 20, 59, tzinfo=UTC), 100.2)]
        _write(store_path, early + full)
        st = BarStore(store_path)
        cov = st.coverage("SPY", 60)
        st.close()
        assert cov["sessions"] == 2
        assert cov["early_close_observed"] == ["2026-01-05"]


# --------------------------------------------------------------------------- #
# session-aware returns: overnight separation, gaps
# --------------------------------------------------------------------------- #

class TestReturnSeparation:
    def test_overnight_returns_kept_distinct(self):
        d1_late = datetime(2026, 1, 5, 20, 58, tzinfo=UTC)
        d2 = datetime(2026, 1, 6, 14, 30, tzinfo=UTC)
        bars = [_bar(d1_late, 100.0),
                _bar(d1_late + timedelta(minutes=1), 100.5),
                _bar(d2, 102.0),
                _bar(d2 + timedelta(minutes=1), 102.2)]
        rs = intraday_log_returns(bars)
        assert len(rs.intraday) == 2
        assert len(rs.overnight) == 1
        assert rs.n_sessions == 2

    def test_gap_bridged_vs_chain_break(self):
        t0 = S1_OPEN
        bars = [_bar(t0, 100.0),
                _bar(t0 + timedelta(minutes=1), 100.1),    # 1 step: intraday
                _bar(t0 + timedelta(minutes=3), 100.3),    # 2 steps: bridged
                _bar(t0 + timedelta(minutes=13), 100.9)]   # 10 steps: break
        rs = intraday_log_returns(bars, max_gap_bars=5)
        assert len(rs.intraday) == 2
        assert rs.n_gap_bridged == 1
        assert rs.n_chain_breaks == 1
        assert rs.overnight == []


# --------------------------------------------------------------------------- #
# valuation history: PIT availability, no daily-as-intraday, no grid
# --------------------------------------------------------------------------- #

def _pit_obs(base: datetime, minutes: float, price: float,
             avail_lag_s: float = 0.0):
    t = base + timedelta(minutes=minutes)
    return (t, t + timedelta(seconds=avail_lag_s), price)


class TestValuationHistoryPIT:
    BASE = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)

    def test_late_available_obs_excluded(self):
        obs = [_pit_obs(self.BASE, m, 100.0 + m * 0.1) for m in range(5)]
        # event before as_of, but only knowable afterwards
        obs.append(_pit_obs(self.BASE, 5, 150.0, avail_lag_s=900))
        as_of = self.BASE + timedelta(minutes=6)
        rets, diag = val._step_returns(obs, as_of, step_minutes=1.0)
        assert diag["n_obs_pit"] == 5
        assert diag["n_returns"] == 4           # the 150.0 jump never enters

    def test_daily_spacing_yields_no_intraday_returns(self):
        # daily closes with an intraday step request: refuse via zero
        # returns (insufficient_history), never rescale
        obs = [_pit_obs(self.BASE, 1440.0 * d, 100.0 + d) for d in range(40)]
        as_of = self.BASE + timedelta(days=41)
        rets, diag = val._step_returns(obs, as_of, step_minutes=5.0)
        assert diag["n_returns"] == 0
        scen, reason = val.generate_scenarios(
            obs, as_of, 100.0, val.ValuationConfig(), seed=1)
        assert scen is None and reason == "insufficient_history"

    def test_overnight_gap_never_spanned(self):
        d1 = datetime(2026, 1, 5, 20, 58, tzinfo=UTC)
        d2 = datetime(2026, 1, 6, 14, 30, tzinfo=UTC)
        obs = [(d1, d1, 100.0),
               (d1 + timedelta(minutes=1), d1 + timedelta(minutes=1), 100.5),
               (d2, d2, 102.0),
               (d2 + timedelta(minutes=1), d2 + timedelta(minutes=1), 102.2)]
        as_of = d2 + timedelta(minutes=5)
        rets, diag = val._step_returns(obs, as_of, step_minutes=1.0)
        assert diag["n_sessions"] == 2
        assert diag["n_returns"] == 2           # one per session, no bridge

    def test_wide_intraday_gap_breaks_chain(self):
        obs = [_pit_obs(self.BASE, 0, 100.0),
               _pit_obs(self.BASE, 30, 101.0)]  # 30 steps at 1-min step
        as_of = self.BASE + timedelta(minutes=31)
        rets, diag = val._step_returns(obs, as_of, step_minutes=1.0,
                                       max_gap_steps=3.0)
        assert diag["n_returns"] == 0
        assert diag["n_chain_breaks"] == 1


# --------------------------------------------------------------------------- #
# shadow bars mode: real bars or structured abstention -- no synthetic grid
# --------------------------------------------------------------------------- #

def _cand(i: int = 0):
    return {"sym": "SPY", "score": 1.0, "underlying_price": 100.5,
            "opt": {"symbol": "SPY260105C00100000", "bid": 1.0, "ask": 1.1,
                    "dte": 1}}


class TestShadowBarsMode:
    def test_missing_store_abstains_no_synthetic_fallback(self, tmp_path):
        cfg = shmod.ShadowConfig(enabled=True, max_scenarios=32,
                                 bars_db=str(tmp_path / "missing.sqlite"))
        rec = shmod.shadow_scan(
            [_cand()], scan_id="t1",
            now=S1_OPEN + timedelta(hours=1), cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        meta = rec["scenarios"]["SPY"]
        assert meta["status"].startswith("bars_unavailable:STORE_MISSING")
        assert meta["bars_unavailable"]["reason"] == "STORE_MISSING"
        assert meta["history_source"] == "bars"
        assert "history_grid_assumed" not in rec   # 4.0.2: key removed

    def test_bars_mode_uses_real_completed_bars(self, tmp_path, store_path):
        _write(store_path, _minute_bars(S1_OPEN, 41))
        cfg = shmod.ShadowConfig(enabled=True, max_scenarios=64,
                                 bars_db=store_path)
        now = S1_OPEN + timedelta(minutes=50)
        rec = shmod.shadow_scan(
            [_cand()], scan_id="t2",
            step_minutes=1.0, now=now, cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        meta = rec["scenarios"]["SPY"]
        assert meta["history_source"] == "bars"
        assert meta["status"] == "ok"
        assert meta["n_history_points"] == 41

    def test_no_bars_db_abstains_with_bars_db_unset(self, tmp_path):
        # 4.0.2 P0-2: the synthetic-grid fallback is gone; without a bars
        # store every underlying abstains with a structured reason.
        cfg = shmod.ShadowConfig(enabled=True, max_scenarios=32)  # no bars_db
        rec = shmod.shadow_scan(
            [_cand()], scan_id="t3",
            now=S1_OPEN + timedelta(hours=1), cfg=cfg,
            out_path=str(tmp_path / "out.jsonl"))
        meta = rec["scenarios"]["SPY"]
        assert meta["history_source"] == "bars"
        assert meta["bars_unavailable"]["reason"] == "BARS_DB_UNSET"
        assert meta["status"].startswith("bars_unavailable:BARS_DB_UNSET")
        assert "history_grid_assumed" not in rec

    def test_env_flag_round_trip(self, monkeypatch):
        monkeypatch.setenv("ORACLE4_BARS_DB", r"artifacts\oracle4\b.sqlite")
        monkeypatch.setenv("ORACLE4_BARS_MAX_STALENESS_MIN", "30")
        cfg = shmod.ShadowConfig.from_env()
        assert cfg.bars_db == r"artifacts\oracle4\b.sqlite"
        assert cfg.bars_max_staleness_min == 30.0
        monkeypatch.delenv("ORACLE4_BARS_DB")
        monkeypatch.delenv("ORACLE4_BARS_MAX_STALENESS_MIN")
        cfg = shmod.ShadowConfig.from_env()
        assert cfg.bars_db == ""
        assert cfg.bars_max_staleness_min == 15.0


# --------------------------------------------------------------------------- #
# label provenance: exact-match exit reasons, simulated never OK
# --------------------------------------------------------------------------- #

T0 = datetime(2026, 11, 4, 15, 0, tzinfo=UTC)


def _decision(decision_id="d1"):
    return DecisionRecord(
        decision_id=decision_id, snapshot_id=f"s_{decision_id}",
        kind="candidate", accepted=True, reject_code=None, as_of=T0,
        available_at=T0, underlying="SPY", symbol=None, expiration=None,
        strike=None, option_type=None, multiplier=None, quote_bid=None,
        quote_ask=None, underlying_price=100.0, iv=None, delta=None,
        direction=None, score=None, p_win=None, ev=None)


def _obs_idx():
    o = [UnderlyingObs("SPY", T0 + timedelta(minutes=m),
                       T0 + timedelta(minutes=m), 100.0 + m * 0.01, f"s{m}")
         for m in (0, 15, 30)]
    return ObsIndex({"SPY": o})


def _outcome_event(reason, klass="realized", minutes=20):
    t = T0 + timedelta(minutes=minutes)
    return [ingest.LedgerEvent(
        event_key="e1", decision_id="d1", event_type="outcome",
        event_at=t, available_at=t, broker_order_id=None, qty=1.0,
        price=0.9, fees=None, status=None,
        payload={"class": klass, "reason": reason})]


def _first_touch(labels):
    return {(l.horizon, l.label_type): l for l in labels}[
        ("event", "first_touch")]


class TestLabelProvenance:
    def _label(self, reason, klass="realized"):
        cov = T0 + timedelta(minutes=30)
        return _first_touch(build_labels(
            _decision(), _obs_idx(), cov,
            events=_outcome_event(reason, klass)))

    def test_time_stop_is_not_a_price_stop(self):
        lab = self._label("TIME_STOP")
        assert lab.status is LabelStatus.OK
        assert lab.meta["touch"] == "other_exit"
        assert lab.meta["touch_detail"] == "time_stop"

    def test_trailing_stop_is_not_the_loss_barrier(self):
        lab = self._label("dynamic_trailing_stop")
        assert lab.meta["touch"] == "other_exit"
        assert lab.meta["touch_detail"] == "trailing_stop"

    def test_roll_failed_take_profit_is_not_a_target_touch(self):
        lab = self._label("roll_failed_take_profit")
        assert lab.meta["touch"] == "other_exit"
        assert lab.meta["touch_detail"] == "roll_failed_target"

    def test_configured_stop_still_a_stop(self):
        lab = self._label("dynamic_stop_loss")
        assert lab.status is LabelStatus.OK
        assert lab.meta["touch"] == "stop"
        assert lab.meta["touch_detail"] == "stop"

    def test_simulated_outcome_never_resolves_ok(self):
        lab = self._label("dynamic_stop_loss", klass="simulated")
        assert lab.status is LabelStatus.SIMULATED
        assert lab.status is not LabelStatus.OK
        assert lab.meta["simulated"] is True

    def test_unmapped_reason_flagged_not_guessed(self):
        lab = self._label("brand_new_exit_kind")
        assert lab.meta["touch"] == "other_exit"
        assert lab.meta["touch_detail"] == "unknown"
        assert lab.meta["reason_unmapped"] is True

    def test_censored_when_filled_but_unresolved(self):
        cov = T0 + timedelta(minutes=30)
        fill = [ingest.LedgerEvent(
            event_key="f1", decision_id="d1", event_type="fill",
            event_at=T0, available_at=T0, broker_order_id=None, qty=1.0,
            price=1.0, fees=None, status=None, payload={})]
        lab = _first_touch(build_labels(_decision(), _obs_idx(), cov,
                                        events=fill))
        assert lab.status is LabelStatus.CENSORED
        assert lab.label_end == cov
