"""
Offline tests for the fills-aware realized-log reconcile (no creds / no network).

Covers the pure layer that turns broker FILL activities into corrected
``realized_pnl_log.json`` rows, and the file-level merge/apply that replaces
only the target day while preserving every other day:
  * realized_by_symbol  — weighted cost basis, day-scoped sells, split fills,
                          other-day exclusion, no-buy skip, fail-open rows.
  * build_rows          — row shape, ordering, source tag, default timestamp.
  * merge_log           — preserves other days, replaces the target day.
  * reconcile(apply)    — backs up, rewrites only the target day, verifies.
"""

import json
import os
import tempfile
import unittest

import reconcile_realized as rr
from reconcile_realized import (
    realized_by_symbol, build_rows, merge_log, RECONCILE_SOURCE,
)

DAY = "2026-06-09"


class TestRealizedBySymbol(unittest.TestCase):
    def test_single_symbol(self):
        acts = [
            {"symbol": "UNH", "side": "buy", "qty": 3, "price": 24.9,
             "transaction_time": "2026-06-02T15:00:00Z"},
            {"symbol": "UNH", "side": "sell", "qty": 3, "price": 22.15,
             "transaction_time": f"{DAY}T19:45:23Z"},
        ]
        realized, ts = realized_by_symbol(acts, DAY)
        self.assertAlmostEqual(realized["UNH"], (22.15 - 24.9) * 3 * 100, places=2)
        self.assertEqual(ts["UNH"], f"{DAY}T19:45:23Z")

    def test_split_sell_fills_weighted(self):
        acts = [
            {"symbol": "HD", "side": "buy", "qty": 2, "price": 20.6,
             "transaction_time": "2026-06-01T15:00:00Z"},
            {"symbol": "HD", "side": "sell", "qty": 1, "price": 18.2,
             "transaction_time": f"{DAY}T19:45:23Z"},
            {"symbol": "HD", "side": "sell", "qty": 1, "price": 18.2,
             "transaction_time": f"{DAY}T19:45:24Z"},
        ]
        realized, ts = realized_by_symbol(acts, DAY)
        self.assertAlmostEqual(realized["HD"], (18.2 * 2 - 20.6 * 2) * 100, places=2)
        # Newest sell timestamp wins.
        self.assertEqual(ts["HD"], f"{DAY}T19:45:24Z")

    def test_weighted_avg_buy_cost(self):
        # Buys at two prices; avg cost basis is qty-weighted.
        acts = [
            {"symbol": "Z", "side": "buy", "qty": 1, "price": 12.0,
             "transaction_time": "2026-06-01T15:00:00Z"},
            {"symbol": "Z", "side": "buy", "qty": 2, "price": 9.0,
             "transaction_time": "2026-06-02T15:00:00Z"},
            {"symbol": "Z", "side": "sell", "qty": 3, "price": 11.0,
             "transaction_time": f"{DAY}T19:00:00Z"},
        ]
        realized, _ = realized_by_symbol(acts, DAY)
        avg = (12.0 * 1 + 9.0 * 2) / 3
        self.assertAlmostEqual(realized["Z"], (11.0 * 3 - avg * 3) * 100, places=2)

    def test_other_day_sell_excluded(self):
        acts = [
            {"symbol": "X", "side": "buy", "qty": 1, "price": 10.0,
             "transaction_time": "2026-06-01T15:00:00Z"},
            {"symbol": "X", "side": "sell", "qty": 1, "price": 12.0,
             "transaction_time": "2026-06-08T19:00:00Z"},
        ]
        realized, _ = realized_by_symbol(acts, DAY)
        self.assertEqual(realized, {})

    def test_no_buy_in_window_skipped(self):
        acts = [{"symbol": "Y", "side": "sell", "qty": 1, "price": 5.0,
                 "transaction_time": f"{DAY}T19:00:00Z"}]
        realized, _ = realized_by_symbol(acts, DAY)
        self.assertEqual(realized, {})

    def test_alt_side_names(self):
        acts = [
            {"symbol": "Q", "side": "buy_to_open", "qty": 1, "price": 4.0,
             "transaction_time": "2026-06-08T15:00:00Z"},
            {"symbol": "Q", "side": "sell_to_close", "qty": 1, "price": 5.0,
             "transaction_time": f"{DAY}T19:00:00Z"},
        ]
        realized, _ = realized_by_symbol(acts, DAY)
        self.assertAlmostEqual(realized["Q"], (5.0 - 4.0) * 1 * 100, places=2)

    def test_malformed_rows_are_skipped(self):
        acts = [
            {"symbol": "G", "side": "buy", "qty": "oops", "price": 4.0,
             "transaction_time": "2026-06-08T15:00:00Z"},
            None,
            {"symbol": "G", "side": "buy", "qty": 1, "price": 4.0,
             "transaction_time": "2026-06-08T15:00:00Z"},
            {"symbol": "G", "side": "sell", "qty": 1, "price": 5.0,
             "transaction_time": f"{DAY}T19:00:00Z"},
        ]
        realized, _ = realized_by_symbol(acts, DAY)
        self.assertAlmostEqual(realized["G"], (5.0 - 4.0) * 100, places=2)

    def test_empty_returns_empty(self):
        self.assertEqual(realized_by_symbol([], DAY), ({}, {}))
        self.assertEqual(realized_by_symbol(None, DAY), ({}, {}))


class TestBuildRows(unittest.TestCase):
    def test_shape_order_tag(self):
        rows = build_rows({"B": 5.0, "A": -10.0}, {"A": f"{DAY}T19:00:00Z"}, DAY)
        self.assertEqual([r["symbol"] for r in rows], ["A", "B"])  # sorted
        self.assertTrue(all(r["source"] == RECONCILE_SOURCE for r in rows))
        self.assertTrue(all(r["date"] == DAY for r in rows))
        self.assertEqual(rows[0]["timestamp"], f"{DAY}T19:00:00Z")
        self.assertEqual(rows[1]["timestamp"], f"{DAY}T00:00:00")  # default


class TestMergeLog(unittest.TestCase):
    def test_replaces_target_day_preserves_others(self):
        existing = [
            {"date": "2026-06-08", "amount": -50.0, "symbol": "OLD"},
            {"date": DAY, "amount": 999.0, "symbol": "STALE"},
        ]
        new_rows = build_rows({"NEW": -5.0}, {}, DAY)
        merged = merge_log(existing, DAY, new_rows)
        syms = [r["symbol"] for r in merged]
        self.assertIn("OLD", syms)
        self.assertIn("NEW", syms)
        self.assertNotIn("STALE", syms)


class TestReconcileApply(unittest.TestCase):
    """End-to-end of the file layer with the network call stubbed out."""

    def setUp(self):
        self._dir = tempfile.mkdtemp()
        self._log = os.path.join(self._dir, "realized_pnl_log.json")
        self._orig_creds = rr._load_credentials
        self._orig_fetch = rr._paginate_fills
        rr._load_credentials = lambda: ("http://x", {})
        self.addCleanup(setattr, rr, "_load_credentials", self._orig_creds)
        self.addCleanup(setattr, rr, "_paginate_fills", self._orig_fetch)

    def _stub_fills(self, acts):
        rr._paginate_fills = lambda base, headers, after: acts

    def _write_log(self, rows):
        with open(self._log, "w") as f:
            json.dump(rows, f)

    def _read_log(self):
        with open(self._log) as f:
            return json.load(f)

    def test_dry_run_writes_nothing(self):
        self._stub_fills([
            {"symbol": "UNH", "side": "buy", "qty": 1, "price": 10.0,
             "transaction_time": "2026-06-08T15:00:00Z"},
            {"symbol": "UNH", "side": "sell", "qty": 1, "price": 8.0,
             "transaction_time": f"{DAY}T19:00:00Z"},
        ])
        self._write_log([{"date": DAY, "amount": 123.0, "symbol": "STALE"}])
        result = rr.reconcile(DAY, 14, self._log, apply=False)
        self.assertFalse(result["applied"])
        self.assertAlmostEqual(result["new_total"], (8.0 - 10.0) * 100, places=2)
        self.assertAlmostEqual(result["prior_day_total"], 123.0, places=2)
        # File untouched.
        self.assertEqual(self._read_log(),
                         [{"date": DAY, "amount": 123.0, "symbol": "STALE"}])

    def test_apply_rewrites_day_and_backs_up(self):
        self._stub_fills([
            {"symbol": "KO", "side": "buy", "qty": 2, "price": 6.75,
             "transaction_time": "2026-06-08T15:00:00Z"},
            {"symbol": "KO", "side": "sell", "qty": 2, "price": 6.05,
             "transaction_time": f"{DAY}T19:45:24Z"},
        ])
        self._write_log([
            {"date": "2026-06-08", "amount": -50.0, "symbol": "OLD"},
            {"date": DAY, "amount": 999.0, "symbol": "STALE"},
        ])
        result = rr.reconcile(DAY, 14, self._log, apply=True)
        self.assertTrue(result["applied"])
        self.assertTrue(os.path.exists(result["backup"]))
        rows = self._read_log()
        syms = [r["symbol"] for r in rows]
        self.assertIn("OLD", syms)       # other day preserved
        self.assertIn("KO", syms)        # rebuilt from fills
        self.assertNotIn("STALE", syms)  # stale day row gone
        ko = next(r for r in rows if r["symbol"] == "KO")
        self.assertAlmostEqual(ko["amount"], (6.05 - 6.75) * 2 * 100, places=2)
        self.assertEqual(ko["source"], RECONCILE_SOURCE)


if __name__ == "__main__":
    unittest.main()
