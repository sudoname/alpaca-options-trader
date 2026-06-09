"""
Offline tests for the scheduler's close-recording fixes (no creds / no network).

Covers the gap that left EOD-closed positions unrecorded and stuck as
'tracked open':
  * realized_from_fills      — quantity-weighted exit price from real sell fills,
                               split fills, recency, and fail-open guards.
  * _already_booked          — symbol+entry_time dedup against trading_history.
  * force_close_scheduler_positions — routes through enforce_exit (records the
                               outcome) AND prunes the closed row from
                               active_trades.json; leaves non-scheduler rows.
  * reconcile_closed_from_fills — books a vanished tracked position from its
                               real fill, prunes it, is idempotent, and skips
                               rows already booked.
"""

import json
import os
import tempfile
import unittest
from types import SimpleNamespace

import run_alpaca_intraday as ri
from run_alpaca_intraday import (
    realized_from_fills, _already_booked, IntradayScheduler, SCHEDULER_SOURCE,
)


class _FakeTrader:
    """Minimal trader: close + record into an in-memory trading_history."""

    def __init__(self, positions):
        self._positions = positions
        self.base_url = "https://paper-api.alpaca.markets"
        self.headers = {}
        self.trading_history = {"trades": []}
        self.closed = []
        self.recorded = []

    def get_positions(self):
        return self._positions

    def close_position(self, trade, position, reason):
        self.closed.append((trade["symbol"], reason))

    def record_trade_outcome(self, trade, outcome, pnl_percent=0):
        self.recorded.append((trade["symbol"], outcome, round(pnl_percent, 4)))
        self.trading_history["trades"].append(
            {"symbol": trade["symbol"], "entry_time": trade.get("entry_time"),
             "outcome": outcome, "pnl_percent": pnl_percent})


def _scheduler(trader):
    return IntradayScheduler(SimpleNamespace(symbols=[]), trader, pdt=None)


class TestRealizedFromFills(unittest.TestCase):
    def test_single_fill(self):
        trade = {"entry_price": 24.9, "quantity": 3}
        fills = [{"qty": 3, "price": 22.15, "transaction_time": "2026-06-09T19:45:23"}]
        ep, pnl = realized_from_fills(trade, fills)
        self.assertAlmostEqual(ep, 22.15, places=4)
        self.assertAlmostEqual(pnl, (22.15 - 24.9) / 24.9 * 100, places=4)

    def test_split_fills_weighted(self):
        # HD-style: two 1-lot fills cover a 2-lot position at the same price.
        trade = {"entry_price": 20.6, "quantity": 2}
        fills = [
            {"qty": 1, "price": 18.2, "transaction_time": "2026-06-09T19:45:24"},
            {"qty": 1, "price": 18.2, "transaction_time": "2026-06-09T19:45:23"},
        ]
        ep, pnl = realized_from_fills(trade, fills)
        self.assertAlmostEqual(ep, 18.2, places=4)
        self.assertAlmostEqual(pnl, (18.2 - 20.6) / 20.6 * 100, places=4)

    def test_uses_newest_fills_only_up_to_qty(self):
        # An older, unrelated sell at a very different price must be ignored
        # once the open quantity is covered by the newest fills.
        trade = {"entry_price": 10.0, "quantity": 2}
        fills = [
            {"qty": 2, "price": 12.0, "transaction_time": "2026-06-09T19:00:00"},
            {"qty": 2, "price": 99.0, "transaction_time": "2026-06-01T10:00:00"},
        ]
        ep, pnl = realized_from_fills(trade, fills)
        self.assertAlmostEqual(ep, 12.0, places=4)

    def test_weighted_average_across_prices(self):
        trade = {"entry_price": 10.0, "quantity": 3}
        fills = [
            {"qty": 1, "price": 12.0, "transaction_time": "2026-06-09T19:00:02"},
            {"qty": 2, "price": 9.0, "transaction_time": "2026-06-09T19:00:01"},
        ]
        ep, _ = realized_from_fills(trade, fills)
        self.assertAlmostEqual(ep, (12.0 * 1 + 9.0 * 2) / 3, places=4)

    def test_empty_fills_returns_none(self):
        self.assertEqual(realized_from_fills({"entry_price": 5, "quantity": 1}, []),
                         (None, None))

    def test_missing_entry_returns_none(self):
        self.assertEqual(
            realized_from_fills({"quantity": 1},
                                [{"qty": 1, "price": 2, "transaction_time": "t"}]),
            (None, None))

    def test_zero_quantity_returns_none(self):
        self.assertEqual(
            realized_from_fills({"entry_price": 5, "quantity": 0},
                                [{"qty": 1, "price": 2, "transaction_time": "t"}]),
            (None, None))

    def test_partial_coverage_uses_available(self):
        # Only 1 of 2 contracts has a fill -> book the 1 we can see.
        trade = {"entry_price": 10.0, "quantity": 2}
        fills = [{"qty": 1, "price": 11.0, "transaction_time": "2026-06-09T19:00:00"}]
        ep, _ = realized_from_fills(trade, fills)
        self.assertAlmostEqual(ep, 11.0, places=4)


class TestAlreadyBooked(unittest.TestCase):
    def test_matches_symbol_and_entry_time(self):
        t = _FakeTrader([])
        t.trading_history["trades"].append({"symbol": "X", "entry_time": "e1"})
        self.assertTrue(_already_booked(t, {"symbol": "X", "entry_time": "e1"}))
        self.assertFalse(_already_booked(t, {"symbol": "X", "entry_time": "e2"}))
        self.assertFalse(_already_booked(t, {"symbol": "Y", "entry_time": "e1"}))


class _FileBackedCase(unittest.TestCase):
    """Redirects the module's active-trades file to a temp path per test."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._path = os.path.join(self._dir, "active_trades.json")
        self._orig = ri.ACTIVE_TRADES_FILE
        ri.ACTIVE_TRADES_FILE = self._path

    def tearDown(self):
        ri.ACTIVE_TRADES_FILE = self._orig

    def _write(self, trades):
        with open(self._path, "w") as f:
            json.dump(trades, f)

    def _read(self):
        with open(self._path) as f:
            return json.load(f)


class TestForceClose(_FileBackedCase):
    def test_records_outcome_and_prunes(self):
        trade = {"symbol": "UNH...C", "entry_price": 24.9, "quantity": 3,
                 "entry_time": "t1", "source": SCHEDULER_SOURCE}
        other = {"symbol": "ZZZ...C", "entry_price": 1.0, "quantity": 1,
                 "entry_time": "t2", "source": "manual"}
        self._write([trade, other])
        trader = _FakeTrader([{"symbol": "UNH...C", "qty": "3",
                               "current_price": "22.15"}])
        _scheduler(trader).force_close_scheduler_positions()
        # Outcome recorded via enforce_exit (close + record).
        self.assertEqual(trader.closed, [("UNH...C", "EOD_CLOSE")])
        self.assertEqual(len(trader.recorded), 1)
        sym, outcome, pnl = trader.recorded[0]
        self.assertEqual((sym, outcome), ("UNH...C", "EOD_CLOSE"))
        self.assertAlmostEqual(pnl, (22.15 - 24.9) / 24.9 * 100, places=2)
        # The closed scheduler row is pruned; the manual row survives.
        survivors = self._read()
        self.assertEqual([t["symbol"] for t in survivors], ["ZZZ...C"])

    def test_ignores_non_scheduler_rows(self):
        self._write([{"symbol": "M...C", "entry_price": 1.0, "quantity": 1,
                      "entry_time": "t", "source": "manual"}])
        trader = _FakeTrader([{"symbol": "M...C", "qty": "1",
                               "current_price": "2.0"}])
        _scheduler(trader).force_close_scheduler_positions()
        self.assertEqual(trader.closed, [])
        self.assertEqual(len(self._read()), 1)


class TestReconcileFromFills(_FileBackedCase):
    def _patch_fills(self, mapping):
        self._orig_fetch = ri._fetch_sell_fills_by_symbol
        ri._fetch_sell_fills_by_symbol = lambda trader: mapping
        self.addCleanup(setattr, ri, "_fetch_sell_fills_by_symbol",
                        self._orig_fetch)

    def test_books_orphan_from_fill_and_prunes(self):
        orphan = {"symbol": "KO...C", "entry_price": 6.75, "quantity": 2,
                  "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([orphan])
        self._patch_fills({"KO...C": [
            {"qty": 2, "price": 6.05, "transaction_time": "2026-06-09T19:45:24"}]})
        trader = _FakeTrader([])  # nothing held -> orphan vanished
        _scheduler(trader).reconcile_closed_from_fills()
        self.assertEqual(len(trader.recorded), 1)
        sym, outcome, pnl = trader.recorded[0]
        self.assertEqual((sym, outcome), ("KO...C", "reconciled_fill_close"))
        self.assertAlmostEqual(pnl, (6.05 - 6.75) / 6.75 * 100, places=2)
        self.assertEqual(self._read(), [])

    def test_idempotent_second_pass_no_double_record(self):
        orphan = {"symbol": "KO...C", "entry_price": 6.75, "quantity": 2,
                  "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([orphan])
        self._patch_fills({"KO...C": [
            {"qty": 2, "price": 6.05, "transaction_time": "2026-06-09T19:45:24"}]})
        trader = _FakeTrader([])
        sched = _scheduler(trader)
        sched.reconcile_closed_from_fills()
        sched.reconcile_closed_from_fills()  # file already pruned
        self.assertEqual(len(trader.recorded), 1)

    def test_skips_already_booked_but_still_prunes(self):
        orphan = {"symbol": "KO...C", "entry_price": 6.75, "quantity": 2,
                  "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([orphan])
        self._patch_fills({"KO...C": [
            {"qty": 2, "price": 6.05, "transaction_time": "2026-06-09T19:45:24"}]})
        trader = _FakeTrader([])
        # Pretend the EOD path already recorded it.
        trader.trading_history["trades"].append(
            {"symbol": "KO...C", "entry_time": "t1"})
        _scheduler(trader).reconcile_closed_from_fills()
        self.assertEqual(trader.recorded, [])      # no double record
        self.assertEqual(self._read(), [])          # but pruned

    def test_no_fill_leaves_row_tracked(self):
        orphan = {"symbol": "KO...C", "entry_price": 6.75, "quantity": 2,
                  "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([orphan])
        self._patch_fills({})  # no fills found
        trader = _FakeTrader([])
        _scheduler(trader).reconcile_closed_from_fills()
        self.assertEqual(trader.recorded, [])
        self.assertEqual([t["symbol"] for t in self._read()], ["KO...C"])

    def test_held_position_is_not_touched(self):
        held = {"symbol": "KO...C", "entry_price": 6.75, "quantity": 2,
                "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([held])
        self._patch_fills({})
        trader = _FakeTrader([{"symbol": "KO...C", "qty": "2",
                               "current_price": "6.0"}])
        _scheduler(trader).reconcile_closed_from_fills()
        self.assertEqual(trader.recorded, [])
        self.assertEqual(len(self._read()), 1)


if __name__ == "__main__":
    unittest.main()
