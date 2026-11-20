"""Stage G tests: strategy books, group IDs, hedge-aware order investigation.

The "November 18" fixtures below are MODELED examples (clearly labeled):
server data for 2026-11-18 contained no actual 0DTE entries, so these rows
model the described scenario for reconciliation-logic testing only.  They
are fixtures, not trading rules fitted to one day.
"""
from datetime import datetime, timezone

import pytest

from oracle.ml import books as bk

# 2026-11-18 15:00 UTC == 10:00 ET -> ET session date 2026-11-18
NOW = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)
LATE = datetime(2026, 11, 18, 21, 30, tzinfo=timezone.utc)  # 16:30 ET


# --------------------------------------------------------------------------- #
# dte buckets / expiry clock / book classification
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("dte,expect", [
    (None, None), (float("nan"), None), ("bogus", None),
    (0, "0"), (-1, "0"),
    (0.5, "1-5"), (1, "1-5"), (5, "1-5"),
    (5.2, "6-21"), (6, "6-21"), (21, "6-21"),
    (22, "22-60"), (60, "22-60"),
    (61, ">60"), (365, ">60"),
])
def test_dte_bucket(dte, expect):
    assert bk.dte_bucket(dte) == expect


def test_remaining_expiry_minutes_same_day():
    # 10:00 ET -> 16:00 ET close on the same date = 360 minutes left
    assert bk.remaining_expiry_minutes("2026-11-18", NOW) == pytest.approx(360.0)


def test_remaining_expiry_minutes_past_close_is_negative():
    assert bk.remaining_expiry_minutes("2026-11-18", LATE) == pytest.approx(-30.0)


def test_remaining_expiry_minutes_unparseable():
    assert bk.remaining_expiry_minutes(None, NOW) is None
    assert bk.remaining_expiry_minutes("not-a-date", NOW) is None


def test_classify_book_by_expiration():
    assert bk.classify_book("2026-11-18", NOW) == bk.BOOK_0DTE
    assert bk.classify_book("2026-12-19", NOW) == bk.BOOK_SWING
    # already-expired contract: not a live book
    assert bk.classify_book("2026-11-17", NOW) == bk.BOOK_UNKNOWN


def test_classify_book_dte_fallback():
    assert bk.classify_book(None, NOW, dte_days=0.3) == bk.BOOK_0DTE
    assert bk.classify_book(None, NOW, dte_days=5.0) == bk.BOOK_SWING
    assert bk.classify_book(None, NOW, dte_days=float("nan")) == bk.BOOK_UNKNOWN
    assert bk.classify_book(None, NOW) == bk.BOOK_UNKNOWN


def test_group_and_strategy_ids():
    assert bk.candidate_group_id("s1", "SPY", "2026-11-18") == \
        "s1|SPY|2026-11-18"
    assert bk.candidate_group_id("s1", None, None) == "s1|UNKNOWN|UNKNOWN"
    assert bk.strategy_id(bk.BOOK_0DTE, "SPY", "call") == "0dte:SPY:call"
    assert bk.strategy_id(bk.BOOK_SWING, None, None) == "swing:UNKNOWN:na"


# --------------------------------------------------------------------------- #
# assign_books / summarize_books
# --------------------------------------------------------------------------- #

def _cand(decision_id, symbol, **features):
    return {"decision_id": decision_id, "symbol": symbol,
            "features": features}


def test_assign_books_paired_call_put_is_one_strategy_group():
    cands = [
        _cand("s1:SPY:0", "SPY261118C00595000", quote_mid=1.0, spread_abs=0.02),
        _cand("s1:SPY:1", "SPY261118P00595000", quote_mid=1.1, spread_abs=0.02),
        _cand("s1:QQQ:2", "QQQ261219C00500000", dte_days=31.0),
    ]
    out = bk.assign_books(cands, now=NOW, scan_id="s1", bars_mode="bars")
    call, put = out["s1:SPY:0"], out["s1:SPY:1"]
    swing = out["s1:QQQ:2"]
    assert call["book"] == put["book"] == bk.BOOK_0DTE
    assert call["group_id"] == put["group_id"] == "s1|SPY|2026-11-18"
    assert call["paired"] is True and put["paired"] is True
    assert call["group_role"] == "call_leg"
    assert put["group_role"] == "put_leg"
    assert call["strategy_id"] == "0dte:SPY:call"
    assert put["strategy_id"] == "0dte:SPY:put"
    # the swing leg is its own single-leg strategy with a dte bucket
    assert swing["book"] == bk.BOOK_SWING
    assert swing["paired"] is False and swing["group_role"] == "single"
    assert swing["dte_bucket"] == "22-60"
    # 0dte legs never get swing buckets
    assert call["dte_bucket"] is None

    summ = bk.summarize_books(out)
    assert summ["candidates_by_book"] == {"0dte": 2, "swing": 1}
    assert summ["groups_by_book"] == {"0dte": 1, "swing": 1}
    assert summ["paired_groups"] == 1     # the pair counts ONCE, not twice
    assert summ["swing_dte_buckets"] == {"22-60": 1}


def test_assign_books_0dte_data_quality_annotations():
    cands = [
        # synthetic-grid bars + wide spread + stale quote, all flagged
        _cand("s2:SPY:0", "SPY261118C00600000",
              quote_mid=1.0, spread_abs=0.2, quote_age_s=300.0),
    ]
    out = bk.assign_books(cands, now=NOW, scan_id="s2",
                          bars_mode="synthetic_grid")
    notes = out["s2:SPY:0"]["annotations"]
    assert "0dte_requires_minute_bars" in notes
    assert "0dte_wide_spread_near_expiry" in notes
    assert "0dte_stale_quote" in notes
    assert "0dte_past_close" not in notes
    assert out["s2:SPY:0"]["remaining_expiry_min"] == pytest.approx(360.0)


def test_assign_books_0dte_past_close_annotation():
    out = bk.assign_books(
        [_cand("s3:SPY:0", "SPY261118C00600000", quote_mid=1.0,
               spread_abs=0.02)],
        now=LATE, scan_id="s3", bars_mode="bars")
    notes = out["s3:SPY:0"]["annotations"]
    assert "0dte_past_close" in notes
    assert "0dte_requires_minute_bars" not in notes  # real bars
    assert "0dte_wide_spread_near_expiry" not in notes  # 2% spread


def test_assign_books_swing_has_no_0dte_annotations():
    out = bk.assign_books(
        [_cand("s4:QQQ:0", "QQQ261219C00500000", quote_mid=1.0,
               spread_abs=0.9, quote_age_s=999.0)],
        now=NOW, scan_id="s4", bars_mode="synthetic_grid")
    assert out["s4:QQQ:0"]["annotations"] == []


def test_assign_books_skips_missing_decision_id():
    out = bk.assign_books(
        [{"decision_id": "", "symbol": "SPY261118C00600000"},
         _cand("ok", "SPY261118C00600000")],
        now=NOW, scan_id="s5")
    assert set(out) == {"ok"}


# --------------------------------------------------------------------------- #
# investigate_orders — hedge-aware duplicate investigation
# --------------------------------------------------------------------------- #

def _order(order_id, symbol, at, side="buy", qty=1):
    return {"order_id": order_id, "symbol": symbol, "side": side,
            "qty": qty, "at": at}


def test_hedge_pair_is_not_a_duplicate():
    # MODELED Nov-18 0DTE example: simultaneous call+put legs on the same
    # underlying+expiration are one intentional paired strategy.
    rep = bk.investigate_orders([
        _order("o1", "SPY261118C00595000", "2026-11-18T14:32:05Z"),
        _order("o2", "SPY261118P00595000", "2026-11-18T14:32:07Z"),
    ])
    assert rep["n_duplicate_suspects"] == 0
    assert rep["n_hedge_pairs"] == 1
    assert rep["hedge_pairs"][0]["classification"] == "intentional_hedge_pair"
    assert rep["hedge_pairs"][0]["order_ids"] == ["o1", "o2"]
    assert rep["n_concurrent_distinct"] == 0
    assert "never counted as duplicates" in rep["note"]


def test_true_duplicate_same_contract_side_qty():
    rep = bk.investigate_orders([
        _order("d1", "SPY261118C00600000", "2026-11-18T15:01:00Z"),
        _order("d2", "SPY261118C00600000", "2026-11-18T15:01:30Z"),
    ])
    assert rep["n_duplicate_suspects"] == 1
    dup = rep["duplicate_suspects"][0]
    assert dup["symbol"] == "SPY261118C00600000"
    assert dup["order_ids"] == ["d1", "d2"]
    assert dup["seconds_apart"] == pytest.approx(30.0)
    assert rep["n_hedge_pairs"] == 0


def test_same_order_id_echo_is_not_a_duplicate():
    # Ledger echoes of one broker order share the order_id -> not a defect.
    rep = bk.investigate_orders([
        _order("same", "SPY261118C00600000", "2026-11-18T15:01:00Z"),
        _order("same", "SPY261118C00600000", "2026-11-18T15:01:05Z"),
    ])
    assert rep["n_duplicate_suspects"] == 0


def test_laddered_strikes_are_concurrent_distinct():
    rep = bk.investigate_orders([
        _order("l1", "SPY261118C00600000", "2026-11-18T15:10:00Z"),
        _order("l2", "SPY261118C00605000", "2026-11-18T15:10:20Z"),
    ])
    assert rep["n_concurrent_distinct"] == 1
    assert rep["n_duplicate_suspects"] == 0
    assert rep["n_hedge_pairs"] == 0


def test_window_and_unparseable_rows():
    rep = bk.investigate_orders([
        _order("w1", "SPY261118C00600000", "2026-11-18T15:00:00Z"),
        _order("w2", "SPY261118C00600000", "2026-11-18T15:05:00Z"),  # 300s
        {"order_id": "bad", "symbol": "", "at": "2026-11-18T15:00:00Z"},
        {"order_id": "bad2", "symbol": "SPY261118C00600000", "at": "junk"},
    ], window_s=120.0)
    assert rep["n_orders"] == 2
    assert rep["n_unparseable"] == 2
    assert rep["n_duplicate_suspects"] == 0  # outside the window


def test_modeled_nov18_combined_fixture():
    """MODELED 2026-11-18 session: hedge pair + true duplicate + ladder.

    Clusters are >120 s apart so classifications stay isolated — mirrors
    how the investigation separates intentional hedges from defects.
    """
    orders = [
        # 14:32 cluster: intentional straddle (call+put) — hedge, NOT dup
        _order("h1", "SPY261118C00595000", "2026-11-18T14:32:05Z"),
        _order("h2", "SPY261118P00595000", "2026-11-18T14:32:06Z"),
        # 15:01 cluster: accidental double-submit — the real defect
        _order("d1", "SPY261118C00600000", "2026-11-18T15:01:00Z"),
        _order("d2", "SPY261118C00600000", "2026-11-18T15:01:02Z"),
        # 15:10 cluster: laddered strikes, same type — distinct, noted only
        _order("l1", "SPY261118C00605000", "2026-11-18T15:10:00Z"),
        _order("l2", "SPY261118C00610000", "2026-11-18T15:10:30Z"),
    ]
    rep = bk.investigate_orders(orders, window_s=120.0)
    assert rep["n_orders"] == 6
    assert rep["n_duplicate_suspects"] == 1
    assert rep["duplicate_suspects"][0]["order_ids"] == ["d1", "d2"]
    assert rep["n_hedge_pairs"] == 1
    assert rep["hedge_pairs"][0]["order_ids"] == ["h1", "h2"]
    assert rep["n_concurrent_distinct"] == 1
    assert rep["concurrent_distinct"][0]["order_ids"] == ["l1", "l2"]


def test_investigate_orders_is_observe_only():
    import inspect
    src = inspect.getsource(bk)
    for forbidden in ("alpaca", "place_order", "submit_order",
                      "replace_order", "cancel_order", "import requests",
                      "TradingClient"):
        assert forbidden not in src
