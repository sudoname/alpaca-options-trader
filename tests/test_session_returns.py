"""Offline tests for session_returns (synthetic fixtures only).

These tests prove software correctness (identity, calendar handling,
adjustment consistency, data-quality rejection). They say NOTHING about
market edge.
"""

import math
from datetime import date

import pandas as pd
import pytest

from session_returns import (
    IDENTITY_TOL,
    compute_session_returns,
    session_stats,
)


def _bars(rows):
    return pd.DataFrame(rows, columns=["date", "open", "high", "low", "close", "volume"])


def _row(d, o, c, h=None, l=None, v=1000.0):
    h = h if h is not None else max(o, c)
    l = l if l is not None else min(o, c)
    return {"date": d, "open": o, "high": h, "low": l, "close": c, "volume": v}


CAL = [date(2026, 1, 5), date(2026, 1, 6), date(2026, 1, 7), date(2026, 1, 8), date(2026, 1, 9)]


class TestIdentity:
    def test_decomposition_matches_hand_computed(self):
        bars = _bars([_row(CAL[0], 100.0, 102.0), _row(CAL[1], 103.02, 101.0)])
        rets, q = compute_session_returns(bars, CAL)
        assert q.clean
        assert len(rets) == 1
        r = rets.iloc[0]
        assert r["overnight"] == pytest.approx(103.02 / 102.0 - 1)
        assert r["intraday"] == pytest.approx(101.0 / 103.02 - 1)
        assert r["daily"] == pytest.approx(101.0 / 102.0 - 1)

    def test_identity_holds_on_every_row(self):
        prices = [(100, 101), (102, 99.5), (99, 103.7), (104, 104), (103.9, 98.2)]
        bars = _bars([_row(d, o, c) for d, (o, c) in zip(CAL, prices)])
        rets, q = compute_session_returns(bars, CAL)
        assert q.clean and len(rets) == 4
        for _, r in rets.iterrows():
            assert abs((1 + r["overnight"]) * (1 + r["intraday"]) - (1 + r["daily"])) <= IDENTITY_TOL


class TestCalendar:
    def test_prev_close_spans_holiday_not_calendar_day(self):
        # Calendar has a holiday between Jan 6 and Jan 8: prev close for
        # Jan 8 must be Jan 6's close.
        cal = [date(2026, 1, 6), date(2026, 1, 8)]
        bars = _bars([_row(cal[0], 100, 110), _row(cal[1], 111, 112)])
        rets, q = compute_session_returns(bars, cal)
        assert q.clean and len(rets) == 1
        assert rets.iloc[0]["prev_date"] == cal[0]
        assert rets.iloc[0]["overnight"] == pytest.approx(111 / 110 - 1)

    def test_missing_session_flagged_and_pairs_skipped(self):
        bars = _bars([_row(CAL[0], 100, 101), _row(CAL[2], 102, 103)])  # Jan 6 missing
        rets, q = compute_session_returns(bars, CAL)
        assert "2026-01-06" in q.missing_sessions
        assert not q.clean
        # No pair may bridge the hole as if Jan 6 didn't exist in the calendar.
        assert len(rets) == 0

    def test_bar_on_non_calendar_date_is_dropped_and_flagged(self):
        sat = date(2026, 1, 10)
        bars = _bars([_row(CAL[3], 100, 101), _row(CAL[4], 101, 102), _row(sat, 50, 51)])
        rets, q = compute_session_returns(bars, CAL)
        assert str(sat) in q.non_calendar_bars
        assert len(rets) == 1  # Jan 8 -> Jan 9 pair unaffected


class TestDataQuality:
    def test_nonpositive_price_rejected(self):
        bars = _bars([_row(CAL[0], 100, 101), _row(CAL[1], -5, 101)])
        rets, q = compute_session_returns(bars, CAL)
        assert str(CAL[1]) in q.bad_price_bars
        assert len(rets) == 0

    def test_open_outside_low_high_rejected(self):
        # Simulates mixing an unadjusted open with an adjusted close series
        # (e.g., post 2:1 split): open=100 but adjusted range is ~50.
        bars = _bars(
            [
                _row(CAL[0], 50.0, 50.5, h=50.6, l=49.9),
                {"date": CAL[1], "open": 100.0, "high": 51.0, "low": 50.0, "close": 50.8, "volume": 1e3},
            ]
        )
        rets, q = compute_session_returns(bars, CAL)
        assert str(CAL[1]) in q.bad_price_bars
        assert len(rets) == 0

    def test_consistently_adjusted_split_produces_no_spurious_return(self):
        # 2:1 split effective Jan 7. A CONSISTENTLY adjusted series halves
        # all pre-split prices, so overnight return stays small.
        bars = _bars(
            [
                _row(CAL[0], 50.0, 50.5),   # adjusted (was 100/101)
                _row(CAL[1], 50.6, 50.4),
            ]
        )
        rets, q = compute_session_returns(bars, CAL)
        assert q.clean
        assert abs(rets.iloc[0]["overnight"]) < 0.01  # not ~-50%

    def test_stale_flat_zero_volume_bar_flagged(self):
        bars = _bars(
            [
                _row(CAL[0], 100, 101),
                {"date": CAL[1], "open": 101.0, "high": 101.0, "low": 101.0, "close": 101.0, "volume": 0.0},
            ]
        )
        rets, q = compute_session_returns(bars, CAL)
        assert str(CAL[1]) in q.stale_bars
        assert len(rets) == 1  # flagged, still usable


class TestStats:
    def test_stats_values_and_uncertainty(self):
        rets = pd.DataFrame(
            {
                "date": CAL[1:5],
                "prev_date": CAL[0:4],
                "overnight": [0.01, -0.01, 0.02, 0.0],
                "intraday": [0.0, 0.0, 0.0, 0.0],
                "daily": [0.01, -0.01, 0.02, 0.0],
            }
        )
        st = session_stats(rets)
        on = st["overnight"]
        assert on["n"] == 4
        assert on["mean"] == pytest.approx(0.005)
        assert on["std"] == pytest.approx(pd.Series([0.01, -0.01, 0.02, 0.0]).std(ddof=1))
        assert on["se_mean"] == pytest.approx(on["std"] / math.sqrt(4))
        assert on["pos_freq"] == pytest.approx(0.5)
        assert on["cum_growth"] == pytest.approx(1.01 * 0.99 * 1.02 * 1.0)

    def test_trailing_window(self):
        rets = pd.DataFrame(
            {
                "date": CAL[1:5],
                "prev_date": CAL[0:4],
                "overnight": [0.10, 0.10, 0.0, 0.0],
                "intraday": [0.0] * 4,
                "daily": [0.10, 0.10, 0.0, 0.0],
            }
        )
        st = session_stats(rets, window=2)
        assert st["overnight"]["n"] == 2
        assert st["overnight"]["mean"] == pytest.approx(0.0)

    def test_empty_input(self):
        rets = pd.DataFrame(columns=["date", "prev_date", "overnight", "intraday", "daily"])
        st = session_stats(rets)
        assert st["overnight"]["n"] == 0
