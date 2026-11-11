"""
Oracle — Phase 2 acceptance tests: pre-submission order persistence,
broker-outcome join by stable IDs, and fill-rate/slippage calibration.

Pins the invariants of the order_intent / order_outcome record family in
``oracle.execution.calibration``:

  1. Intents are persisted BEFORE submission — including no-trade decisions
     (with a reason) and unfilled opportunities — keyed by a stable
     client_order_id (fallback trade_id, else generated).
  2. Outcomes join back to intents purely by that stable id; the join is
     last-non-null-wins so intent-time quote context (bid/ask/mid, source,
     fill_probability) survives next to the terminal broker verdict.
  3. Missing evidence is marked EXPLICITLY (avg_fill_price / order_id /
     fees / exit_fills), never silently fabricated.
  4. The lifecycle adapter records only terminal orders (working/unknown
     return None with NO write) and detects late fills from event labels.
  5. ``fit_fill_and_slippage`` uses nonfills in every denominator, reports
     signed per-share slippage vs the intent-time mid (positive = worse:
     buy fill−mid, sell mid−fill), buckets by fill-probability and spread,
     and flags low_sample buckets so the pooled row is the fallback.
  6. Fail-open: missing ledger -> [], junk lines ignored, empty fit -> zeros.

Hand-math anchor (the FitScenario fixture):
  c1 buy  qty 1, quote 1.00/1.04 (mid 1.02), p_fill 0.8 -> filled 1 @ 1.04,
     fees 0.02, exit 1.20.      slippage = 1.04 - 1.02 = +0.02. Clean row.
  c2 sell qty 2, quote 1.00/1.04 (mid 1.02), p_fill 0.6 -> filled 1 @ 1.01
     (partial; canceled remainder). slippage = 1.02 - 1.01 = +0.01,
     fill fraction 0.5, missing evidence = fees + exit_fills.
  c3 buy  qty 1, quote 2.00/2.08, p_fill 0.4 -> expired, 0 fills (nonfill).
  nt1 no-trade intent, reason "stale_quote".
  c4 submit intent, never resolved (unresolved).

  Pooled over {c1, c2, c3}: predicted_fill_rate = (0.8+0.6+0.4)/3 = 0.6,
  actual any-fill rate = 2/3, bias = 0.6 - 2/3 = -0.066667,
  mean_fill_fraction = (1.0 + 0.5 + 0.0)/3 = 0.5,
  mean_slippage = (0.02 + 0.01)/2 = +0.015 over n_slippage = 2.

Offline: temp dirs only; no creds, no network.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.execution.calibration import (
    EVIDENCE_EXIT,
    EVIDENCE_FEES,
    EVIDENCE_FILL_PRICE,
    EVIDENCE_ORDER_ID,
    RECORD_TYPE_INTENT,
    RECORD_TYPE_OUTCOME,
    fit_fill_and_slippage,
    load_order_records,
    record_order_intent,
    record_order_outcome,
    record_order_outcome_from_lifecycle,
)


def _tmp_ledger():
    return os.path.join(tempfile.mkdtemp(), "orders.jsonl")


class TestOrderIntentPersistence(unittest.TestCase):
    """Invariant 1: intents persist pre-submission with stable keys."""

    def test_intent_returns_client_order_id_as_key(self):
        path = _tmp_ledger()
        key = record_order_intent(
            {"client_order_id": "c1", "symbol": "SPY"}, jsonl_path=path)
        self.assertEqual(key, "c1")
        rows = load_order_records(path)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["record_type"], RECORD_TYPE_INTENT)
        self.assertEqual(rows[0]["client_order_id"], "c1")

    def test_intent_falls_back_to_trade_id_then_generated(self):
        path = _tmp_ledger()
        self.assertEqual(
            record_order_intent({"trade_id": "t9"}, jsonl_path=path), "t9")
        gen = record_order_intent({}, jsonl_path=path)
        self.assertIsInstance(gen, str)
        self.assertEqual(len(gen), 12)  # uuid4().hex[:12]

    def test_intent_auto_mid_from_bid_ask(self):
        path = _tmp_ledger()
        record_order_intent(
            {"client_order_id": "c1", "bid": 1.00, "ask": 1.04},
            jsonl_path=path)
        self.assertAlmostEqual(load_order_records(path)[0]["mid"], 1.02,
                               places=6)

    def test_intent_explicit_mid_preserved(self):
        path = _tmp_ledger()
        record_order_intent(
            {"client_order_id": "c1", "bid": 1.00, "ask": 1.04, "mid": 1.03},
            jsonl_path=path)
        self.assertAlmostEqual(load_order_records(path)[0]["mid"], 1.03,
                               places=6)

    def test_intent_qty_alias_and_quote_provenance(self):
        path = _tmp_ledger()
        record_order_intent({
            "client_order_id": "c1", "symbol": "spy", "side": "buy",
            "order_type": "limit", "qty": 2, "limit_price": 1.02,
            "expiry_s": 30, "quote_source": "sim", "quote_ts": "t0",
            "quote_age_s": 0.5, "fill_probability": 0.8,
        }, jsonl_path=path)
        row = load_order_records(path)[0]
        self.assertEqual(row["symbol"], "SPY")
        self.assertEqual(row["requested_qty"], 2.0)   # qty alias
        self.assertEqual(row["quote_source"], "sim")
        self.assertEqual(row["quote_ts"], "t0")
        self.assertEqual(row["quote_age_s"], 0.5)
        self.assertIsNone(row["outcome_status"])      # not terminal yet

    def test_no_trade_intent_is_evidence_too(self):
        path = _tmp_ledger()
        key = record_order_intent({
            "trade_id": "nt1", "symbol": "SPY",
            "decision": "no_trade", "reason": "stale_quote",
        }, jsonl_path=path)
        self.assertEqual(key, "nt1")
        row = load_order_records(path)[0]
        self.assertEqual(row["decision"], "no_trade")
        self.assertEqual(row["reason"], "stale_quote")


class TestOrderOutcomeAndJoin(unittest.TestCase):
    """Invariants 2–3: join by stable id; explicit missing evidence."""

    def test_outcome_requires_client_order_id(self):
        path = _tmp_ledger()
        self.assertIsNone(record_order_outcome("", {"terminal_state": "x"},
                                               jsonl_path=path))
        self.assertEqual(load_order_records(path), [])  # nothing written

    def test_full_evidence_row_has_no_missing_evidence(self):
        path = _tmp_ledger()
        record_order_outcome("c1", {
            "order_id": "o1", "terminal_state": "filled",
            "requested_qty": 1, "filled_qty": 1, "avg_fill_price": 1.04,
            "fees": 0.02, "exit_price": 1.20,
        }, jsonl_path=path)
        row = load_order_records(path)[0]
        self.assertEqual(row["missing_evidence"], [])
        self.assertEqual(row["outcome_status"], "terminal")

    def test_missing_evidence_flags(self):
        path = _tmp_ledger()
        # Filled but no price, no broker id, no fees, no exit -> all four.
        record_order_outcome("c1", {"terminal_state": "filled",
                                    "filled_qty": 1}, jsonl_path=path)
        missing = load_order_records(path)[0]["missing_evidence"]
        self.assertCountEqual(missing, [EVIDENCE_FILL_PRICE,
                                        EVIDENCE_ORDER_ID,
                                        EVIDENCE_FEES, EVIDENCE_EXIT])

    def test_realized_pnl_satisfies_exit_evidence(self):
        path = _tmp_ledger()
        record_order_outcome("c1", {
            "order_id": "o1", "terminal_state": "filled", "filled_qty": 1,
            "avg_fill_price": 1.04, "fees": 0.02, "realized_pnl": 16.0,
        }, jsonl_path=path)
        self.assertEqual(load_order_records(path)[0]["missing_evidence"], [])

    def test_nonfill_outcome_flags_nothing_fill_related(self):
        path = _tmp_ledger()
        record_order_outcome("c3", {
            "order_id": "o3", "terminal_state": "expired", "filled_qty": 0,
        }, jsonl_path=path)
        self.assertEqual(load_order_records(path)[0]["missing_evidence"], [])

    def test_join_folds_intent_and_outcome_last_non_null_wins(self):
        path = _tmp_ledger()
        record_order_intent({
            "client_order_id": "c1", "symbol": "SPY", "side": "buy",
            "order_type": "limit", "qty": 1, "limit_price": 1.04,
            "bid": 1.00, "ask": 1.04, "quote_source": "kernel_shadow",
            "fill_probability": 0.8,
        }, jsonl_path=path)
        record_order_outcome("c1", {
            "order_id": "broker-1", "terminal_state": "filled",
            "requested_qty": 1, "filled_qty": 1, "avg_fill_price": 1.04,
            "fees": 0.02, "exit_price": 1.20,
        }, jsonl_path=path)
        rows = load_order_records(path)
        self.assertEqual(len(rows), 1)  # one logical order, two lines
        row = rows[0]
        # intent-time context survives the join...
        self.assertAlmostEqual(row["mid"], 1.02, places=6)
        self.assertEqual(row["fill_probability"], 0.8)
        self.assertEqual(row["quote_source"], "kernel_shadow")
        # ...next to the terminal broker verdict.
        self.assertEqual(row["terminal_state"], "filled")
        self.assertEqual(row["order_id"], "broker-1")
        self.assertEqual(row["outcome_status"], "terminal")
        self.assertEqual(row["missing_evidence"], [])

    def test_load_ignores_junk_and_foreign_record_types(self):
        path = _tmp_ledger()
        with open(path, "w", encoding="utf-8") as fh:
            fh.write("not json\n")
            fh.write('{"record_type": "execution_estimate", "trade_id": "A"}\n')
            fh.write('[1, 2, 3]\n')
        record_order_intent({"client_order_id": "c1"}, jsonl_path=path)
        rows = load_order_records(path)
        self.assertEqual([r["client_order_id"] for r in rows], ["c1"])

    def test_load_missing_file_fails_open(self):
        self.assertEqual(
            load_order_records(os.path.join(tempfile.mkdtemp(), "nope.jsonl")),
            [])


class TestLifecycleAdapter(unittest.TestCase):
    """Invariant 4: ManagedOrder.to_dict() -> outcome; terminal-only."""

    @staticmethod
    def _order(**over):
        base = {
            "client_order_id": "c1", "order_id": "b1", "symbol": "OPT",
            "side": "buy", "qty": 1, "order_type": "limit",
            "limit_price": 1.04, "state": "filled", "filled_qty": 1,
            "avg_fill_price": 1.04, "events": [[0.0, "submitted", ""],
                                               [1.0, "filled", ""]],
        }
        base.update(over)
        return base

    def test_filled_order_recorded_with_extras(self):
        path = _tmp_ledger()
        key = record_order_outcome_from_lifecycle(
            self._order(), fees=0.02, exit_price=1.20, jsonl_path=path)
        self.assertEqual(key, "c1")
        row = load_order_records(path)[0]
        self.assertEqual(row["terminal_state"], "filled")
        self.assertEqual(row["filled_qty"], 1.0)
        self.assertEqual(row["fees"], 0.02)
        self.assertFalse(row["late_fill"])
        self.assertEqual(row["missing_evidence"], [])

    def test_working_and_unknown_return_none_without_write(self):
        path = _tmp_ledger()
        self.assertIsNone(record_order_outcome_from_lifecycle(
            self._order(state="working"), jsonl_path=path))
        self.assertIsNone(record_order_outcome_from_lifecycle(
            self._order(state="unknown"), jsonl_path=path))
        self.assertIsNone(record_order_outcome_from_lifecycle(
            self._order(state=""), jsonl_path=path))
        self.assertEqual(load_order_records(path), [])

    def test_expired_nonfill_recorded(self):
        path = _tmp_ledger()
        record_order_outcome_from_lifecycle(
            self._order(state="expired", filled_qty=0, avg_fill_price=None,
                        events=[[0.0, "submitted", ""],
                                [30.0, "expired", ""]]),
            jsonl_path=path)
        row = load_order_records(path)[0]
        self.assertEqual(row["terminal_state"], "expired")
        self.assertEqual(row["filled_qty"], 0.0)
        self.assertEqual(row["missing_evidence"], [])

    def test_late_fill_detected_from_event_label(self):
        path = _tmp_ledger()
        record_order_outcome_from_lifecycle(
            self._order(state="partially_filled", filled_qty=1, qty=2,
                        events=[[0.0, "submitted", ""],
                                [1.0, "cancel_requested", ""],
                                [2.0, "late_fill_after_cancel_request", ""]]),
            fees=0.01, realized_pnl=-2.0, jsonl_path=path)
        row = load_order_records(path)[0]
        self.assertTrue(row["late_fill"])
        self.assertEqual(row["terminal_state"], "partially_filled")


class _FitScenario:
    """Builds the hand-math ledger from the module docstring once."""

    @staticmethod
    def build(path):
        record_order_intent({
            "client_order_id": "c1", "symbol": "SPY", "side": "buy",
            "order_type": "limit", "qty": 1, "limit_price": 1.04,
            "bid": 1.00, "ask": 1.04, "fill_probability": 0.8,
        }, jsonl_path=path)
        record_order_intent({
            "client_order_id": "c2", "symbol": "SPY", "side": "sell",
            "order_type": "limit", "qty": 2, "limit_price": 1.01,
            "bid": 1.00, "ask": 1.04, "fill_probability": 0.6,
        }, jsonl_path=path)
        record_order_intent({
            "client_order_id": "c3", "symbol": "SPY", "side": "buy",
            "order_type": "limit", "qty": 1, "limit_price": 2.02,
            "bid": 2.00, "ask": 2.08, "fill_probability": 0.4,
        }, jsonl_path=path)
        record_order_intent({
            "trade_id": "nt1", "symbol": "SPY",
            "decision": "no_trade", "reason": "stale_quote",
        }, jsonl_path=path)
        record_order_intent({  # c4: submitted but never resolved
            "client_order_id": "c4", "symbol": "SPY", "side": "buy",
            "order_type": "limit", "qty": 1, "bid": 1.00, "ask": 1.04,
            "fill_probability": 0.9,
        }, jsonl_path=path)
        record_order_outcome("c1", {
            "order_id": "b1", "terminal_state": "filled",
            "requested_qty": 1, "filled_qty": 1, "avg_fill_price": 1.04,
            "fees": 0.02, "exit_price": 1.20,
        }, jsonl_path=path)
        record_order_outcome("c2", {  # partial sell; fees/exit unknown
            "order_id": "b2", "terminal_state": "partially_filled",
            "requested_qty": 2, "filled_qty": 1, "avg_fill_price": 1.01,
        }, jsonl_path=path)
        record_order_outcome("c3", {  # expired nonfill
            "order_id": "b3", "terminal_state": "expired", "filled_qty": 0,
        }, jsonl_path=path)
        return load_order_records(path)


class TestFitFillAndSlippage(unittest.TestCase):
    """Invariant 5: hand-math fit; nonfills in denominators; buckets."""

    @classmethod
    def setUpClass(cls):
        cls.rows = _FitScenario.build(_tmp_ledger())
        cls.fit = fit_fill_and_slippage(cls.rows, min_n=2)

    def test_counts(self):
        f = self.fit
        self.assertEqual(f["n_intents"], 5)
        self.assertEqual(f["n_no_trade"], 1)
        self.assertEqual(f["n_submitted_terminal"], 3)
        self.assertEqual(f["n_unresolved"], 1)           # c4
        self.assertEqual(f["n_with_missing_evidence"], 1)  # c2 fees+exit

    def test_pooled_hand_math(self):
        p = self.fit["pooled"]
        self.assertEqual(p["n"], 3)
        self.assertAlmostEqual(p["predicted_fill_rate"], 0.6, places=6)
        self.assertAlmostEqual(p["actual_fill_rate"], 2.0 / 3.0, places=6)
        self.assertAlmostEqual(p["fill_rate_bias"], 0.6 - 2.0 / 3.0, places=5)
        self.assertAlmostEqual(p["mean_fill_fraction"], 0.5, places=6)
        self.assertAlmostEqual(p["mean_slippage"], 0.015, places=6)
        self.assertEqual(p["n_slippage"], 2)
        self.assertFalse(p["low_sample"])  # 3 >= min_n=2

    def test_nonfill_counts_in_denominator(self):
        """Drop c3 (the nonfill) and the actual fill rate jumps to 1.0 —
        proving the real fit's 2/3 came from counting the nonfill."""
        fills_only = [r for r in self.rows
                      if r.get("client_order_id") != "c3"]
        p = fit_fill_and_slippage(fills_only, min_n=2)["pooled"]
        self.assertEqual(p["n"], 2)
        self.assertAlmostEqual(p["actual_fill_rate"], 1.0, places=6)

    def test_fill_probability_buckets(self):
        by = {b["bucket"]: b for b in self.fit["by_fill_probability"]}
        self.assertEqual(by["(0.75, 1.00]"]["n"], 1)   # c1 @ 0.8
        self.assertEqual(by["(0.50, 0.75]"]["n"], 1)   # c2 @ 0.6
        self.assertEqual(by["(0.25, 0.50]"]["n"], 1)   # c3 @ 0.4
        self.assertEqual(by["(0.00, 0.25]"]["n"], 0)
        for b in by.values():  # every bucket n=1 or 0 < min_n=2
            self.assertTrue(b["low_sample"])

    def test_spread_buckets(self):
        by = {b["bucket"]: b for b in self.fit["by_spread"]}
        tight = by["(0.02, 0.05]"]          # c1, c2: spread $0.04
        self.assertEqual(tight["n"], 2)
        self.assertAlmostEqual(tight["mean_slippage"], 0.015, places=6)
        self.assertFalse(tight["low_sample"])
        wide = by["(0.05, 0.10]"]           # c3: spread $0.08
        self.assertEqual(wide["n"], 1)
        self.assertAlmostEqual(wide["actual_fill_rate"], 0.0, places=6)
        self.assertTrue(wide["low_sample"])
        self.assertEqual(by["(0.00, 0.02]"]["n"], 0)

    def test_slippage_signs(self):
        """Buy worse = +; sell worse = +; a price improvement is negative."""
        rows = [
            {"record_type": RECORD_TYPE_OUTCOME, "outcome_status": "terminal",
             "side": "buy", "requested_qty": 1, "filled_qty": 1,
             "avg_fill_price": 1.00, "mid": 1.02},   # buy BELOW mid: -0.02
            {"record_type": RECORD_TYPE_OUTCOME, "outcome_status": "terminal",
             "side": "sell", "requested_qty": 1, "filled_qty": 1,
             "avg_fill_price": 1.00, "mid": 1.02},   # sell below mid: +0.02
        ]
        slips = [fit_fill_and_slippage([r], min_n=1)["pooled"]["mean_slippage"]
                 for r in rows]
        self.assertAlmostEqual(slips[0], -0.02, places=6)
        self.assertAlmostEqual(slips[1], +0.02, places=6)

    def test_empty_records_fail_open(self):
        f = fit_fill_and_slippage([])
        self.assertEqual(f["n_intents"], 0)
        self.assertEqual(f["n_submitted_terminal"], 0)
        self.assertEqual(f["pooled"]["n"], 0)
        self.assertIsNone(f["pooled"]["mean_slippage"])
        self.assertIsNone(f["pooled"]["actual_fill_rate"])
        self.assertTrue(f["pooled"]["low_sample"])

    def test_determinism(self):
        self.assertEqual(self.fit, fit_fill_and_slippage(self.rows, min_n=2))


if __name__ == "__main__":
    unittest.main(verbosity=2)
