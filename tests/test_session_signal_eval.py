"""Offline tests for session_signal_eval (synthetic fixtures only)."""

from datetime import date, datetime, timezone

import pytest

from session_signal_eval import (
    aggregate_legs,
    direction_of,
    eval_decision_sessions,
    parse_as_of_utc,
    realized_by_hold,
)

UTC = timezone.utc
D1, D2, D3 = date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 8)  # Jan 7 = holiday
CAL = [D1, D2, D3]


def _dec(ts, sym="XYZ", direction=1, decision_id="d1"):
    return {"decision_id": decision_id, "as_of": ts, "underlying": sym, "direction": direction}


def _ctx():
    # Signal day D1: minute bars at 15:00, 15:01, 15:02 UTC
    minute = {
        ("XYZ", D1): [
            (datetime(2026, 1, 5, 15, 0, tzinfo=UTC), 100.0),
            (datetime(2026, 1, 5, 15, 1, tzinfo=UTC), 100.5),
            (datetime(2026, 1, 5, 15, 2, tzinfo=UTC), 101.0),
        ]
    }
    daily_raw = {"XYZ": {D1: {"open": 99.0, "close": 102.0}, D2: {"open": 103.0, "close": 104.0}}}
    daily_adj = {"XYZ": {D1: {"open": 99.0, "close": 102.0}, D2: {"open": 103.0, "close": 104.0},
                         D3: {"open": 105.0, "close": 106.0}}}
    return minute, daily_raw, daily_adj


class TestDirection:
    def test_mapping(self):
        assert direction_of("CALL", "long_call") == 1
        assert direction_of("PUT", "long_put") == -1
        assert direction_of("SPREAD", "bullish_put_credit_spread") == 1
        assert direction_of("SKIP", "x") is None
        assert direction_of(None, None) is None


class TestTimestamps:
    def test_naive_as_of_is_utc(self):
        dt = parse_as_of_utc("2026-06-17T14:18:29.669454")
        assert dt.tzinfo is not None and dt.utcoffset().total_seconds() == 0

    def test_aware_as_of_preserved(self):
        dt = parse_as_of_utc("2026-05-01T13:30:04.465399+00:00")
        assert dt.hour == 13


class TestLeakage:
    def test_bar_at_exact_signal_time_not_used(self):
        """A minute bar stamped AT the signal ts may contain the trade that
        produced the signal -- entry must be the NEXT bar."""
        minute, daily_raw, daily_adj = _ctx()
        sig = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)  # == first bar ts
        recs = eval_decision_sessions(
            [_dec(sig)], minute, daily_raw, daily_adj, CAL, cost_bps=0.0
        )
        leg = recs[0]["intraday"]
        assert leg is not None
        # entry is the 15:01 bar (100.5), never the 15:00 bar (100.0)
        assert leg["gross"] == pytest.approx(102.0 / 100.5 - 1.0)
        assert "15:01" in leg["entry_ts_utc"]

    def test_no_bar_after_signal_skips_leg(self):
        minute, daily_raw, daily_adj = _ctx()
        sig = datetime(2026, 1, 5, 20, 30, tzinfo=UTC)  # after last bar
        recs = eval_decision_sessions([_dec(sig)], minute, daily_raw, daily_adj, CAL)
        assert recs[0]["intraday"] is None

    def test_stale_entry_bar_rejected(self):
        """If the first available bar is > MAX_ENTRY_DELAY_MIN after the
        signal, the fill assumption is not credible -> leg skipped."""
        minute, daily_raw, daily_adj = _ctx()
        minute[("XYZ", D1)] = [(datetime(2026, 1, 5, 16, 0, tzinfo=UTC), 100.0)]
        sig = datetime(2026, 1, 5, 15, 0, tzinfo=UTC)
        recs = eval_decision_sessions([_dec(sig)], minute, daily_raw, daily_adj, CAL)
        assert recs[0]["intraday"] is None
        assert "entry_bar_stale" in recs[0]["flags"]


class TestLegs:
    def test_intraday_and_overnight_values_long(self):
        minute, daily_raw, daily_adj = _ctx()
        sig = datetime(2026, 1, 5, 15, 0, 30, tzinfo=UTC)
        recs = eval_decision_sessions(
            [_dec(sig)], minute, daily_raw, daily_adj, CAL, cost_bps=0.0
        )
        r = recs[0]
        assert r["intraday"]["gross"] == pytest.approx(102.0 / 100.5 - 1)  # 15:01 bar entry
        assert r["overnight"]["gross"] == pytest.approx(103.0 / 102.0 - 1)  # D1 close -> D2 open

    def test_put_direction_flips_sign(self):
        minute, daily_raw, daily_adj = _ctx()
        sig = datetime(2026, 1, 5, 15, 0, 30, tzinfo=UTC)
        recs = eval_decision_sessions(
            [_dec(sig, direction=-1)], minute, daily_raw, daily_adj, CAL, cost_bps=0.0
        )
        r = recs[0]
        assert r["intraday"]["gross"] == pytest.approx(-(102.0 / 100.5 - 1))
        assert r["overnight"]["gross"] == pytest.approx(-(103.0 / 102.0 - 1))

    def test_costs_subtracted_round_trip(self):
        minute, daily_raw, daily_adj = _ctx()
        sig = datetime(2026, 1, 5, 15, 0, 30, tzinfo=UTC)
        recs = eval_decision_sessions(
            [_dec(sig)], minute, daily_raw, daily_adj, CAL, cost_bps=5.0
        )
        r = recs[0]
        assert r["intraday"]["net"] == pytest.approx(r["intraday"]["gross"] - 10.0 / 1e4)
        assert r["overnight"]["net"] == pytest.approx(r["overnight"]["gross"] - 10.0 / 1e4)

    def test_overnight_spans_holiday_to_next_session(self):
        """Signal on D2; next session is D3 (D7 holiday skipped)."""
        minute, daily_raw, daily_adj = _ctx()
        minute[("XYZ", D2)] = [(datetime(2026, 1, 6, 15, 0, tzinfo=UTC), 103.5)]
        sig = datetime(2026, 1, 6, 14, 59, tzinfo=UTC)
        recs = eval_decision_sessions(
            [_dec(sig)], minute, daily_raw, daily_adj, CAL, cost_bps=0.0
        )
        on = recs[0]["overnight"]
        assert on["gross"] == pytest.approx(105.0 / 104.0 - 1)  # D2 close -> D3 open
        assert "2026-01-08" in on["exit_assumption"]

    def test_last_calendar_session_has_no_overnight(self):
        minute, daily_raw, daily_adj = _ctx()
        minute[("XYZ", D3)] = [(datetime(2026, 1, 8, 15, 0, tzinfo=UTC), 105.2)]
        sig = datetime(2026, 1, 8, 14, 59, tzinfo=UTC)
        recs = eval_decision_sessions([_dec(sig)], minute, daily_raw, daily_adj, CAL)
        assert recs[0]["overnight"] is None
        assert "overnight_bars_missing" in recs[0]["flags"]


class TestAggregation:
    def test_aggregate_and_realized_strata(self):
        recs = [
            {"intraday": {"net": 0.01, "gross": 0.01}, "overnight": None},
            {"intraday": {"net": -0.01, "gross": -0.01}, "overnight": None},
        ]
        agg = aggregate_legs(recs, "intraday")
        assert agg["n"] == 2 and agg["mean"] == pytest.approx(0.0)
        assert agg["pos_freq"] == pytest.approx(0.5)

        rows = [
            {"net_pnl_pct": 10.0, "hold_days": 0},
            {"net_pnl_pct": -5.0, "hold_days": 0},
            {"net_pnl_pct": 20.0, "hold_days": 2},
        ]
        strata = realized_by_hold(rows)
        assert strata["same_day (hold_days=0)"]["n"] == 2
        assert strata["same_day (hold_days=0)"]["mean_net_pnl_pct"] == pytest.approx(2.5)
        assert strata["held_overnight (hold_days>=1)"]["n"] == 1
        assert strata["held_overnight (hold_days>=1)"]["mean_net_pnl_pct"] == pytest.approx(20.0)
