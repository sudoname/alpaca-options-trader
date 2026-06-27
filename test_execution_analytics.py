"""Unit tests for execution_analytics (pure, in-memory, no network)."""

import execution_analytics as ex


def test_self_test_passes():
    assert ex._self_test() == 0


def test_entry_quality_spread_slippage_fillquality():
    eps = [
        {"quote_bid": 1.00, "quote_ask": 1.20, "fill_price": 1.15},  # mid 1.10
        {"quote_bid": 2.00, "quote_ask": 2.10, "fill_price": 2.05},  # mid 2.05
    ]
    eq = ex.entry_quality(eps)
    assert eq["samples"] == 2
    exp_spread = ((0.20 / 1.10) + (0.10 / 2.05)) / 2 * 100
    assert abs(eq["avg_spread_pct"] - exp_spread) < 1e-9
    # fill 1.15 > mid 1.10 (outside); fill 2.05 == mid (inside) -> 50%
    assert abs(eq["fill_quality_pct"] - 50.0) < 1e-9


def test_entry_quality_skips_rows_without_quotes():
    eps = [{"fill_price": 9.9}, {"quote_bid": 1.0, "quote_ask": None,
                                 "fill_price": 1.0}]
    eq = ex.entry_quality(eps)
    assert eq["samples"] == 0
    assert eq["avg_spread_pct"] is None
    assert eq["fill_quality_pct"] is None


def test_entry_quality_rejects_crossed_quotes():
    eps = [{"quote_bid": 2.0, "quote_ask": 1.0, "fill_price": 1.5}]  # ask < bid
    assert ex.entry_quality(eps)["samples"] == 0


def test_holding_stats_days_and_same_day():
    eps = [{"hold_days": 0}, {"hold_days": 2}, {"hold_days": 4}]
    hs = ex.holding_stats(eps)
    assert hs["samples"] == 3
    assert abs(hs["avg_hold_days"] - 2.0) < 1e-9
    assert abs(hs["median_hold_days"] - 2.0) < 1e-9
    assert abs(hs["same_day_pct"] - (1 / 3 * 100)) < 1e-9


def test_round_trips_fifo_and_same_day_flag():
    fills = [
        {"symbol": "X", "side": "buy", "qty": "1",
         "transaction_time": "2026-06-26T10:00:00Z"},
        {"symbol": "X", "side": "sell", "qty": "1",
         "transaction_time": "2026-06-26T14:00:00Z"},  # same day, +4h
        {"symbol": "Y", "side": "buy", "qty": "1",
         "transaction_time": "2026-06-25T15:00:00Z"},
        {"symbol": "Y", "side": "sell", "qty": "1",
         "transaction_time": "2026-06-26T15:00:00Z"},  # next day, +24h
    ]
    rts = ex.round_trips_from_fills(fills)
    assert len(rts) == 2
    by_hours = sorted(rts)
    assert abs(by_hours[0][0] - 4.0) < 1e-9 and by_hours[0][1] is True
    assert abs(by_hours[1][0] - 24.0) < 1e-9 and by_hours[1][1] is False


def test_round_trips_partial_fill_matching():
    fills = [
        {"symbol": "Z", "side": "buy", "qty": "2",
         "transaction_time": "2026-06-26T10:00:00Z"},
        {"symbol": "Z", "side": "sell", "qty": "1",
         "transaction_time": "2026-06-26T11:00:00Z"},
        {"symbol": "Z", "side": "sell", "qty": "1",
         "transaction_time": "2026-06-26T12:00:00Z"},
    ]
    rts = ex.round_trips_from_fills(fills)
    assert len(rts) == 2
    assert abs(rts[0][0] - 1.0) < 1e-9
    assert abs(rts[1][0] - 2.0) < 1e-9


def test_compute_and_format_never_raise_on_empty():
    rep = ex.compute_execution([])
    md = ex.format_markdown(rep)
    assert "Execution Quality" in md
    assert ex.holding_stats([])["avg_hold_days"] is None
