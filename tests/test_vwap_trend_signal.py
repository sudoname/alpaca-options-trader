"""Hermetic tests for vwap_trend_signal.py (research-only trend signal).

Covers the regression surface the research prompt requires: completed-bar
timing, session VWAP reset/basis labeling, neutral vs missing inputs, Wilder
RSI hand calculations, EMA warm-up, staleness, and direction alignment
(True/False vs None so filter skips stay distinct from missing inputs).
No network, no files, no env.
"""
from datetime import datetime, timezone

import pytest

import vwap_trend_signal as vts


def _bar(ts, c, v=100, vwap=None, o=None, h=None, l=None):
    b = {"ts": ts, "o": o if o is not None else c,
         "h": h if h is not None else c + 0.5,
         "l": l if l is not None else c - 0.5,
         "c": c, "v": v}
    if vwap is not None:
        b["vwap"] = vwap
    return b


def _session(closes, start="2026-11-04T14:30:00+00:00", vendor_vwap=True):
    t0 = datetime.fromisoformat(start)
    bars = []
    for i, c in enumerate(closes):
        ts = (t0.replace(tzinfo=timezone.utc)
              if t0.tzinfo is None else t0)
        ts = ts.timestamp() + i * 300
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        bars.append(_bar(iso, c, vwap=(c if vendor_vwap else None)))
    return bars


def test_selftest_passes(capsys):
    assert vts._selftest() == 0
    assert "SELFTEST PASS" in capsys.readouterr().out


# ---------------------------------------------------------------- timing ---

def test_bar_eligible_only_after_close():
    bars = _session([100.0])
    # decision 1s before bar close -> no completed bar
    r = vts.signal_at(bars, "2026-11-04T14:34:59+00:00")
    assert r["state_vwap_only"] == vts.UNAVAILABLE
    assert r["reason"] == "no_completed_bar"
    # exactly at close -> eligible
    r = vts.signal_at(bars, "2026-11-04T14:35:00+00:00")
    assert r["bar_ts"] == "2026-11-04T14:30:00+00:00"
    assert r["bar_lag_s"] == 0.0
    assert r["state_vwap_only"] != vts.UNAVAILABLE


def test_uses_latest_completed_bar_not_in_progress():
    bars = _session([100.0, 101.0, 102.0])
    # 14:41 -> bars 14:30 and 14:35 completed; 14:40 bar still open
    r = vts.signal_at(bars, "2026-11-04T14:41:00+00:00")
    assert r["bar_ts"] == "2026-11-04T14:35:00+00:00"
    assert r["close"] == 101.0


def test_stale_bar_rejected():
    bars = _session([100.0])
    # last close 14:35, decision 14:46 -> lag 11 min > 10 min cap
    r = vts.signal_at(bars, "2026-11-04T14:46:00+00:00")
    assert r["reason"] == "stale_bar"
    assert r["state_vwap_only"] == vts.UNAVAILABLE
    assert r["state_full"] == vts.UNAVAILABLE


def test_bad_decision_ts():
    r = vts.signal_at(_session([100.0]), "not-a-timestamp")
    assert r["reason"] == "bad_decision_ts"


# ------------------------------------------------------------------ vwap ---

def test_session_vwap_resets_per_session():
    """VWAP is computed only from the bars passed in: feeding a new session's
    bars yields a VWAP untouched by the prior session's prices."""
    day1 = _session([100.0, 100.0], start="2026-11-04T14:30:00+00:00")
    day2 = _session([50.0, 50.0], start="2026-11-05T14:30:00+00:00")
    r1 = vts.signal_at(day1, "2026-11-04T14:40:00+00:00")
    r2 = vts.signal_at(day2, "2026-11-05T14:40:00+00:00")
    assert r1["vwap"] == pytest.approx(100.0)
    assert r2["vwap"] == pytest.approx(50.0)  # no bleed from day 1


def test_vwap_basis_trade_derived_vs_typical_price():
    vendor = vts.session_vwap_series(_session([10.0, 11.0]))
    assert vendor["basis"] == "trade_derived"
    assert vendor["series"][-1] == pytest.approx((10 * 100 + 11 * 100) / 200)
    approx = vts.session_vwap_series(_session([10.0, 11.0],
                                              vendor_vwap=False))
    assert approx["basis"] == "typical_price"
    # typical price = (h + l + c)/3 with h=c+.5, l=c-.5 -> equals c
    assert approx["series"][-1] == pytest.approx(10.5)


def test_vwap_basis_degrades_if_any_bar_lacks_vendor_vwap():
    bars = _session([10.0, 11.0])
    del bars[1]["vwap"]
    assert vts.session_vwap_series(bars)["basis"] == "typical_price"


def test_zero_volume_session_has_no_vwap():
    bars = _session([10.0])
    bars[0]["v"] = 0
    r = vts.signal_at(bars, "2026-11-04T14:40:00+00:00")
    assert r["reason"] == "no_session_vwap"
    assert r["state_vwap_only"] == vts.UNAVAILABLE


# -------------------------------------------------------- states/neutral ---

def test_close_equal_vwap_is_neutral_not_unavailable():
    assert vts._state_vwap_only(100.0, 100.0) == vts.NEUTRAL


def test_full_state_mixed_readings_are_neutral():
    # valid inputs, bullish on vwap but bearish rsi -> neutral
    assert vts._state_full(101.0, 100.0, 100.5, 45.0) == vts.NEUTRAL
    # all three bearish
    assert vts._state_full(99.0, 100.0, 99.5, 40.0) == vts.BEARISH


def test_full_state_missing_input_is_unavailable():
    assert vts._state_full(101.0, 100.0, None, 55.0) == vts.UNAVAILABLE


def test_full_warming_up_before_ema_period():
    bars = _session([100.0 + i for i in range(5)])
    r = vts.signal_at(bars, "2026-11-04T14:56:00+00:00")
    assert r["state_vwap_only"] == vts.BULLISH
    assert r["state_full"] == vts.UNAVAILABLE
    assert r["reason"] == "warming_up"


def test_continuation_prior_closes_warm_full_variant():
    bars = _session([100.0 + 0.1 * i for i in range(3)])
    prior = [99.0 + 0.05 * i for i in range(30)]
    r = vts.signal_at(bars, "2026-11-04T14:46:00+00:00",
                      prior_closes=prior)
    assert r["warmup"] == "continuation"
    assert r["state_full"] == vts.BULLISH


# --------------------------------------------------------------- rsi/ema ---

def test_rsi_warmup_returns_none_not_fifty():
    # unlike telegram_bot.calculate_rsi, no fabricated neutral 50
    out = vts.rsi_series([100, 101, 102], period=14)
    assert out == [None, None, None]


def test_wilder_rsi_hand_calc_period_3():
    closes = [10.0, 11.0, 10.5, 11.5, 12.0]
    out = vts.rsi_series(closes, period=3)
    assert out[:3] == [None, None, None]
    # seed: gains (1, 0, 1)/3, losses (0, .5, 0)/3 -> RS=4 -> RSI=80
    assert out[3] == pytest.approx(80.0)
    # smooth: ag=(2/3*2+0.5)/3=0.611..., al=(1/6*2)/3=0.111...
    ag = (2.0 / 3.0 * 2 + 0.5) / 3.0
    al = (0.5 / 3.0 * 2 + 0.0) / 3.0
    assert out[4] == pytest.approx(100 - 100 / (1 + ag / al))


def test_rsi_all_gains_is_100():
    out = vts.rsi_series([1, 2, 3, 4, 5], period=3)
    assert out[-1] == 100.0


def test_ema_sma_seed_matches_intraday_features_contract():
    out = vts.ema_series([1.0, 2.0, 3.0, 4.0], period=3)
    assert out[0] is None and out[1] is None
    assert out[2] == pytest.approx(2.0)          # SMA seed
    assert out[3] == pytest.approx(2.0 + (4 - 2) * 0.5)  # k=2/(3+1)


# ------------------------------------------------------------- alignment ---

@pytest.mark.parametrize("state,direction,expected", [
    (vts.BULLISH, "CALL", True),
    (vts.BULLISH, "call", True),
    (vts.BULLISH, "PUT", False),
    (vts.BEARISH, "PUT", True),
    (vts.BEARISH, "CALL", False),
    (vts.NEUTRAL, "CALL", False),
    (vts.NEUTRAL, "PUT", False),
    (vts.UNAVAILABLE, "CALL", None),
    (vts.BULLISH, "IRON_CONDOR", None),
    (vts.BULLISH, "", None),
    (vts.BULLISH, None, None),
])
def test_alignment_mapping(state, direction, expected):
    assert vts.aligned(state, direction) is expected


def test_parse_ts_variants():
    assert vts.parse_ts("2026-11-04T14:30:00Z") is not None
    assert vts.parse_ts("2026-11-04T14:30:00+00:00") is not None
    assert vts.parse_ts(None) is None
    assert vts.parse_ts("garbage") is None


def test_parse_ts_nanosecond_fraction():
    # ledger timestamps carry >6 fractional digits (regression: these were
    # rejected, flagging every real opportunity signal unavailable)
    dt = vts.parse_ts("2026-11-04T14:53:27.03192073Z")
    assert dt is not None
    assert dt.microsecond == 31920
    assert vts.parse_ts("2026-11-04T14:53:27.123Z").microsecond == 123000
