"""
Oracle 3.0 — Upgrade C acceptance tests: executable-EV shadow -> paper veto.

These pin the promotion invariants the live gate in ``smart_trader.py`` relies
on, WITHOUT instantiating the full trader (no creds / no network / no order
placement). The gate calls exactly two pure entry points from
``oracle.execution.executable_ev`` (``compute_executable_ev`` +
``should_veto_entry``) and the append-only ledger in
``oracle.execution.calibration``; all three are exercised directly here.

Invariants:
  1. Wide spread strictly lowers executable_EV (more friction eaten).
  2. A +theoretical / −executable flip is ``would_reject`` and only VETOES when
     shadow is OFF *and* the account is paper.
  3. Shadow ON is byte-identical to pre-Phase-2: it never vetoes.
  4. Live account never vetoes even with shadow OFF (paper-first promotion).
  5. The conservative entry never fills BELOW mid (a limit at ``max_entry_price``
     is honoured — the fill model never flatters the entry).
  6. The calibration ledger records estimate + realization keyed by the same
     trade_id and folds to one resolved row with computable capture ratios.

No creds, no network, no order placement. File writes go to a tmp path only.
"""

import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model import CostModel
from oracle.execution.client import Quote
from oracle.execution.executable_ev import (
    capture_ratios,
    compute_executable_ev,
    compute_realized_ev,
    should_veto_entry,
)
from oracle.execution.fill_model import FillModel
from oracle.execution import calibration as calib


CONTRACT = {"symbol": "OPT", "expected_value": 15.0,
            "probability_of_profit": 0.55}
THIN = {"symbol": "OPT", "expected_value": 3.0}
TIGHT = Quote("OPT", bid=1.00, ask=1.02, ts="t")   # 2c spread, mid 1.01
WIDE = Quote("OPT", bid=1.00, ask=1.40, ts="t")    # 40c spread, mid 1.20


class TestSpreadFriction(unittest.TestCase):
    def test_wide_spread_lowers_executable_ev(self):
        t = compute_executable_ev(CONTRACT, 1.01, TIGHT,
                                  fill_model=FillModel(), cost_model=CostModel())
        w = compute_executable_ev(CONTRACT, 1.20, WIDE,
                                  fill_model=FillModel(), cost_model=CostModel())
        self.assertIsNotNone(t.executable_EV)
        self.assertIsNotNone(w.executable_EV)
        self.assertLess(w.executable_EV, t.executable_EV)
        self.assertGreater(w.spread_cost, t.spread_cost)


class TestVetoArming(unittest.TestCase):
    def _flip(self):
        # A thin theoretical edge on a wide market flips +theoretical/−executable.
        return compute_executable_ev(THIN, 1.20, WIDE,
                                     fill_model=FillModel(), cost_model=CostModel())

    def test_flip_is_would_reject(self):
        flip = self._flip()
        self.assertIsNotNone(flip.executable_EV)
        self.assertLessEqual(flip.executable_EV, 0)
        self.assertGreater(flip.theoretical_EV, 0)

    def test_veto_when_shadow_off_and_paper(self):
        flip = self._flip()
        dec = should_veto_entry(flip.theoretical_EV, flip.executable_EV,
                                shadow_mode=False, is_paper=True)
        self.assertTrue(dec["would_reject"])
        self.assertTrue(dec["veto_armed"])
        self.assertTrue(dec["veto"])

    def test_shadow_on_never_vetoes(self):
        flip = self._flip()
        dec = should_veto_entry(flip.theoretical_EV, flip.executable_EV,
                                shadow_mode=True, is_paper=True)
        self.assertTrue(dec["would_reject"])   # still recorded
        self.assertFalse(dec["veto"])          # but never acts

    def test_live_account_never_vetoes(self):
        flip = self._flip()
        dec = should_veto_entry(flip.theoretical_EV, flip.executable_EV,
                                shadow_mode=False, is_paper=False)
        self.assertTrue(dec["would_reject"])
        self.assertFalse(dec["veto_armed"])
        self.assertFalse(dec["veto"])

    def test_positive_executable_never_flags(self):
        t = compute_executable_ev(CONTRACT, 1.01, TIGHT,
                                  fill_model=FillModel(), cost_model=CostModel())
        self.assertGreater(t.executable_EV, 0)
        dec = should_veto_entry(t.theoretical_EV, t.executable_EV,
                                shadow_mode=False, is_paper=True)
        self.assertFalse(dec["would_reject"])
        self.assertFalse(dec["veto"])


class TestConservativeEntry(unittest.TestCase):
    def test_entry_never_below_mid(self):
        # The BUY fill is conservative: you never get filled below mid, so a
        # max_entry_price at/above the expected entry is always honoured.
        for q in (TIGHT, WIDE):
            xev = compute_executable_ev(CONTRACT, None, q,
                                        fill_model=FillModel(), cost_model=CostModel())
            self.assertIsNotNone(xev.expected_entry_price)
            self.assertIsNotNone(xev.mid_price)
            self.assertGreaterEqual(xev.expected_entry_price, xev.mid_price)


class TestCalibrationLedger(unittest.TestCase):
    def test_estimate_then_realization_folds_by_trade_id(self):
        xev = compute_executable_ev(CONTRACT, 1.01, TIGHT,
                                    fill_model=FillModel(), cost_model=CostModel())
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "exec_calib.jsonl")
            fields = dict(xev.to_dict())
            fields.update({"trade_id": "ord-1", "symbol": "OPT", "qty": 1})
            tid = calib.record_execution_estimate(fields, jsonl_path=path)
            self.assertEqual(tid, "ord-1")

            realized = compute_realized_ev(1.04, 1.20, qty=1, cost_model=CostModel())
            calib.record_execution_realization(
                "ord-1", {"filled": True, "actual_entry_price": 1.04,
                          "actual_exit_price": 1.20, "realized_EV": realized},
                jsonl_path=path)

            rows = calib.load_records(jsonl_path=path)
            folded = [r for r in rows if r.get("trade_id") == "ord-1"]
            self.assertEqual(len(folded), 1)
            row = folded[0]
            self.assertEqual(row["resolution_status"], "resolved")
            self.assertIsNotNone(row["theoretical_EV"])   # from the estimate
            self.assertIsNotNone(row["realized_EV"])      # from the realization
            ratios = capture_ratios(row["theoretical_EV"],
                                    row["executable_EV"], row["realized_EV"])
            self.assertIsNotNone(ratios["execution_capture_ratio"])

    def test_record_is_fail_open_on_junk(self):
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "j.jsonl")
            # Junk fields must never raise; estimate returns a trade_id.
            self.assertIsNotNone(
                calib.record_execution_estimate({"symbol": "OPT"}, jsonl_path=path))
            # Missing trade_id on realization -> None, no raise.
            self.assertIsNone(
                calib.record_execution_realization("", {}, jsonl_path=path))


if __name__ == "__main__":
    unittest.main()
