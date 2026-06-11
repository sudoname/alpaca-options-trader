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


class TestTimeStop(_FileBackedCase):
    """Hold-overnight mode: force_close with reason + `only` predicate."""

    def _rows(self):
        old = {"symbol": "OLD...C", "entry_price": 2.0, "quantity": 1,
               "entry_time": "2026-06-01T10:00:00",
               "source": SCHEDULER_SOURCE}
        fresh = {"symbol": "NEW...C", "entry_price": 2.0, "quantity": 1,
                 "entry_time": "2026-06-09T10:00:00",
                 "source": SCHEDULER_SOURCE}
        manual_old = {"symbol": "MAN...C", "entry_price": 2.0, "quantity": 1,
                      "entry_time": "2026-06-01T10:00:00", "source": "manual"}
        return old, fresh, manual_old

    def test_closes_only_aged_scheduler_rows(self):
        from datetime import datetime
        old, fresh, manual_old = self._rows()
        self._write([old, fresh, manual_old])
        trader = _FakeTrader([
            {"symbol": "OLD...C", "qty": "1", "current_price": "1.5"},
            {"symbol": "NEW...C", "qty": "1", "current_price": "2.5"},
            {"symbol": "MAN...C", "qty": "1", "current_price": "2.5"},
        ])
        now = datetime.fromisoformat("2026-06-10T15:00:00")
        _scheduler(trader).force_close_scheduler_positions(
            reason="TIME_STOP",
            only=lambda t: ri.held_past_max_days(
                t.get("entry_time"), now, 5))
        # Only the aged scheduler row closes, recorded as TIME_STOP.
        self.assertEqual(trader.closed, [("OLD...C", "TIME_STOP")])
        self.assertEqual([r[:2] for r in trader.recorded],
                         [("OLD...C", "TIME_STOP")])
        # Fresh scheduler row and the manual row survive in the file.
        self.assertEqual([t["symbol"] for t in self._read()],
                         ["NEW...C", "MAN...C"])

    def test_default_reason_still_eod_close(self):
        old, _, _ = self._rows()
        self._write([old])
        trader = _FakeTrader([{"symbol": "OLD...C", "qty": "1",
                               "current_price": "1.5"}])
        _scheduler(trader).force_close_scheduler_positions()
        self.assertEqual(trader.closed, [("OLD...C", "EOD_CLOSE")])


class TestHeldPastMaxDays(unittest.TestCase):
    def test_boundaries_and_fail_open(self):
        from datetime import datetime
        now = datetime.fromisoformat("2026-06-10T15:00:00")
        held = ri.held_past_max_days
        self.assertFalse(held("2026-06-10T09:00:00", now, 5))  # same day
        self.assertFalse(held("2026-06-05T09:00:00", now, 5))  # exactly 5
        self.assertTrue(held("2026-06-04T09:00:00", now, 5))   # 6 days
        self.assertFalse(held("2020-01-01T09:00:00", now, 0))  # disabled
        self.assertFalse(held("garbage", now, 5))              # fail-open
        self.assertFalse(held(None, now, 5))
        self.assertFalse(held("2020-01-01T09:00:00", None, 5))


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


class TestEODScopeAudit(_FileBackedCase):
    """Regression guards for the EOD force-close blast radius.

    The EOD close must only ever touch intraday long-option rows the scheduler
    itself opened (source == SCHEDULER_SOURCE in active_trades.json). It must
    not close Telegram/manual rows (which carry no 'source' or a foreign one),
    must not sweep broker positions it doesn't track (e.g. future live spread
    legs), and must stay decoupled from the spread-paper / advisory / learning
    stores, which have their own lifecycles.
    """

    def test_closes_only_scheduler_source_rows(self):
        sched_row = {"symbol": "SCH...C", "entry_price": 2.0, "quantity": 1,
                     "entry_time": "t1", "source": SCHEDULER_SOURCE}
        no_source = {"symbol": "TG...C", "entry_price": 3.0, "quantity": 1,
                     "entry_time": "t2"}  # telegram/manual default: no key
        foreign = {"symbol": "FT...C", "entry_price": 4.0, "quantity": 1,
                   "entry_time": "t3", "source": "telegram"}
        self._write([sched_row, no_source, foreign])
        trader = _FakeTrader([
            {"symbol": "SCH...C", "qty": "1", "current_price": "1.5"},
            {"symbol": "TG...C", "qty": "1", "current_price": "2.5"},
            {"symbol": "FT...C", "qty": "1", "current_price": "3.5"},
        ])
        _scheduler(trader).force_close_scheduler_positions()
        self.assertEqual(trader.closed, [("SCH...C", "EOD_CLOSE")])
        self.assertEqual([r[0] for r in trader.recorded], ["SCH...C"])
        self.assertEqual([t["symbol"] for t in self._read()],
                         ["TG...C", "FT...C"])

    def test_untracked_broker_positions_are_never_swept(self):
        # A broker position with NO active_trades row (e.g. a future live
        # spread leg) must be invisible to the EOD close, which iterates the
        # tracked file — never raw broker positions.
        sched_row = {"symbol": "SCH...C", "entry_price": 2.0, "quantity": 1,
                     "entry_time": "t1", "source": SCHEDULER_SOURCE}
        self._write([sched_row])
        trader = _FakeTrader([
            {"symbol": "SCH...C", "qty": "1", "current_price": "1.5"},
            {"symbol": "SPRDLEG...P", "qty": "1", "current_price": "9.9"},
        ])
        _scheduler(trader).force_close_scheduler_positions()
        self.assertEqual(trader.closed, [("SCH...C", "EOD_CLOSE")])

    def test_spread_paper_positions_file_untouched(self):
        # The simulated spread book lives in its own file with its own
        # lifecycle; neither the EOD close nor the fill reconcile may open it.
        spread_path = os.path.join(self._dir, "spread_paper_positions.json")
        payload = json.dumps([{"id": "sp1", "symbol": "SPY", "legs": 2,
                               "opened": "2026-06-08"}])
        with open(spread_path, "w") as f:
            f.write(payload)
        self._write([{"symbol": "SCH...C", "entry_price": 2.0, "quantity": 1,
                      "entry_time": "t1", "source": SCHEDULER_SOURCE}])
        trader = _FakeTrader([{"symbol": "SCH...C", "qty": "1",
                               "current_price": "1.5"}])
        sched = _scheduler(trader)
        sched.force_close_scheduler_positions()
        orig_fetch = ri._fetch_sell_fills_by_symbol
        ri._fetch_sell_fills_by_symbol = lambda trader: {}
        try:
            sched.reconcile_closed_from_fills()
        finally:
            ri._fetch_sell_fills_by_symbol = orig_fetch
        with open(spread_path) as f:
            self.assertEqual(f.read(), payload)

    def test_eod_close_routes_through_enforce_exit(self):
        # enforce_exit = close_position + record_trade_outcome, exactly once
        # each, with the EOD_CLOSE reason — the recording contract.
        self._write([{"symbol": "SCH...C", "entry_price": 2.0, "quantity": 1,
                      "entry_time": "t1", "source": SCHEDULER_SOURCE}])
        trader = _FakeTrader([{"symbol": "SCH...C", "qty": "1",
                               "current_price": "1.5"}])
        _scheduler(trader).force_close_scheduler_positions()
        self.assertEqual(trader.closed, [("SCH...C", "EOD_CLOSE")])
        self.assertEqual(len(trader.recorded), 1)
        self.assertEqual(trader.recorded[0][1], "EOD_CLOSE")
        self.assertEqual(len(trader.trading_history["trades"]), 1)

    def test_reconcile_ignores_non_scheduler_orphans(self):
        # Vanished rows that the scheduler does not own (no source / foreign
        # source) are left alone even when a matching sell fill exists.
        no_source = {"symbol": "TG...C", "entry_price": 3.0, "quantity": 1,
                     "entry_time": "t2"}
        foreign = {"symbol": "FT...C", "entry_price": 4.0, "quantity": 1,
                   "entry_time": "t3", "source": "telegram"}
        self._write([no_source, foreign])
        orig_fetch = ri._fetch_sell_fills_by_symbol
        ri._fetch_sell_fills_by_symbol = lambda trader: {
            "TG...C": [{"qty": 1, "price": 2.0,
                        "transaction_time": "2026-06-09T19:00:00"}],
            "FT...C": [{"qty": 1, "price": 3.0,
                        "transaction_time": "2026-06-09T19:00:00"}],
        }
        trader = _FakeTrader([])
        try:
            _scheduler(trader).reconcile_closed_from_fills()
        finally:
            ri._fetch_sell_fills_by_symbol = orig_fetch
        self.assertEqual(trader.recorded, [])
        self.assertEqual([t["symbol"] for t in self._read()],
                         ["TG...C", "FT...C"])

    def test_scheduler_module_has_no_advisory_or_spread_coupling(self):
        # Static isolation guard: the scheduler module must not reference the
        # spread-paper / advisory-attribution / learning-shadow stores, whose
        # multi-day lifecycles are owned elsewhere. If a future change wires
        # them in, this fails and forces an explicit scope review.
        with open(ri.__file__, encoding="utf-8") as f:
            src = f.read()
        for needle in ("spread_paper", "advisory_attribution",
                       "learning_shadow", "hypothesis_engine"):
            self.assertNotIn(needle, src,
                             f"run_alpaca_intraday must not couple to {needle}")


if __name__ == "__main__":
    unittest.main()
