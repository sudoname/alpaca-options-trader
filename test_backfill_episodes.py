"""Unit tests for backfill_episodes (pure, in-memory, no network)."""

import json

import backfill_episodes as bf
from episode_store import EpisodeStore


def test_self_test_passes():
    assert bf._self_test() == 0


def test_parse_occ():
    assert bf.parse_occ("KO260717C00081000") == ("KO", "2026-07-17", "C", 81.0)
    assert bf.parse_occ("SPY260109P00475000") == ("SPY", "2026-01-09", "P", 475.0)
    assert bf.parse_occ("nope") is None
    assert bf.parse_occ(None) is None


def test_fifo_partial_close():
    # Buy 3 @1.00, then sell 1 @2.00 and sell 2 @0.50 -> two round-trips.
    acts = [
        {"activity_type": "FILL", "symbol": "AAA260101C00100000",
         "side": "buy", "qty": "3", "price": "1.00", "order_id": "o1",
         "transaction_time": "2026-01-01T15:00:00Z"},
        {"activity_type": "FILL", "symbol": "AAA260101C00100000",
         "side": "sell", "qty": "1", "price": "2.00", "order_id": "o2",
         "transaction_time": "2026-01-02T15:00:00Z"},
        {"activity_type": "FILL", "symbol": "AAA260101C00100000",
         "side": "sell", "qty": "2", "price": "0.50", "order_id": "o3",
         "transaction_time": "2026-01-03T15:00:00Z"},
    ]
    trips = bf.match_round_trips(acts)
    assert len(trips) == 2
    assert sum(t["qty"] for t in trips) == 3


def test_pnl_long_and_short():
    long_trip = {"entry_price": 1.0, "exit_price": 1.5, "qty": 2,
                 "is_long": True, "fees": 0.0}
    gp, np_, nd = bf._trip_pnl(long_trip)
    assert abs(nd - 100.0) < 1e-9 and abs(gp - 50.0) < 1e-9
    short_trip = {"entry_price": 1.0, "exit_price": 0.5, "qty": 2,
                  "is_long": False, "fees": 0.0}
    _, _, nd2 = bf._trip_pnl(short_trip)
    assert abs(nd2 - 100.0) < 1e-9   # short profits when price falls


def test_end_to_end_persists_evidence():
    acts = [
        {"activity_type": "FILL", "symbol": "KO260717C00081000",
         "side": "buy", "qty": "1", "price": "1.00", "order_id": "o1",
         "transaction_time": "2026-06-01T15:00:00Z"},
        {"activity_type": "FILL", "symbol": "KO260717C00081000",
         "side": "sell", "qty": "1", "price": "1.50", "order_id": "o2",
         "transaction_time": "2026-06-03T15:00:00Z"},
    ]
    store = EpisodeStore(":memory:")
    try:
        assert bf.backfill_broker(store, acts) == 1
        row = store.completed()[0]
        ev = json.loads(row["features_json"])["evidence"]
        assert ev["strategy"] == "long_call"
        assert ev["direction"] == "up"
        assert row["mode"] == "backfill"
    finally:
        store.close()
