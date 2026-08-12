"""
Oracle 3.0 — Upgrade D acceptance tests: fill-model calibration.

These pin the promotion invariants for ``calibrated_params_from`` — the loader
that turns the realized-fill calibration ledger into a tuned ``FillModelConfig``
the executable-EV path consumes when ``ENABLE_FILL_MODEL`` is ON.

Invariants:
  1. Fail-open / byte-identical: junk, empty, or fewer than
     ``MIN_CALIBRATION_SAMPLES`` filled trades -> the conservative default
     config UNCHANGED, so the model behaves exactly like today.
  2. Conservative-only: calibration can only make a fill WORSE — slippage never
     drops, fill-probability base rates never rise.
  3. Captures reality: with enough records showing a constant under-charged
     entry, the calibrated model PREDICTS the realized entry price on the
     calibration set (model capture ratio -> 1.0).
  4. Over-optimistic fill rate is corrected DOWN toward the realized rate.
  5. Determinism: identical records -> byte-identical config.

No creds, no network, no file writes (records are passed in directly).
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.execution.client import OrderRequest, Quote
from oracle.execution.fill_model import (
    MIN_CALIBRATION_SAMPLES,
    FillModel,
    FillModelConfig,
    calibrated_params_from,
)


TIGHT = Quote("OPT", bid=1.00, ask=1.02, ts="t")   # 2c spread, mid 1.01
BUY = OrderRequest("OPT", "buy", 1, order_type="market")


def _resolved(n, *, expected_entry, actual_entry, fill_prob=1.0,
              theo=15.0, exe=11.0, realized=11.0):
    return [{
        "trade_id": f"c{i}",
        "resolution_status": "resolved",
        "filled": True,
        "fill_probability": fill_prob,
        "expected_entry_price": expected_entry,
        "actual_entry_price": actual_entry,
        "theoretical_EV": theo,
        "executable_EV": exe,
        "realized_EV": realized,
    } for i in range(n)]


class TestFailOpen(unittest.TestCase):
    def test_junk_and_empty_return_default(self):
        default = FillModelConfig()
        for junk in (None, [], "x", 42, [{"bad": 1}]):
            self.assertEqual(calibrated_params_from(junk, base=default), default)

    def test_below_threshold_is_byte_identical(self):
        default = FillModelConfig()
        ref = FillModel().estimate_fill(BUY, TIGHT).expected_fill_price
        recs = _resolved(MIN_CALIBRATION_SAMPLES - 1,
                         expected_entry=ref, actual_entry=round(ref + 0.05, 6))
        cfg = calibrated_params_from(recs, base=default)
        self.assertEqual(cfg, default)
        self.assertFalse(cfg.calibrated)
        # And the estimate itself is identical to the default model's.
        self.assertEqual(
            FillModel(cfg).estimate_fill(BUY, TIGHT),
            FillModel().estimate_fill(BUY, TIGHT))


class TestCapturesReality(unittest.TestCase):
    def test_sufficient_records_predict_realized_entry(self):
        default = FillModelConfig()
        ref = FillModel().estimate_fill(BUY, TIGHT).expected_fill_price
        delta = 0.05
        recs = _resolved(MIN_CALIBRATION_SAMPLES,
                         expected_entry=ref, actual_entry=round(ref + delta, 6))
        cfg = calibrated_params_from(recs, base=default)

        self.assertTrue(cfg.calibrated)
        self.assertEqual(cfg.n_samples, MIN_CALIBRATION_SAMPLES)
        # Slippage was RAISED (never lowered).
        self.assertGreater(cfg.base_slippage, default.base_slippage)
        # The calibrated model now predicts the realized entry -> capture ~1.0.
        got = FillModel(cfg).estimate_fill(BUY, TIGHT).expected_fill_price
        self.assertIsNotNone(got)
        self.assertAlmostEqual(got, ref + delta, places=6)


class TestConservativeOnly(unittest.TestCase):
    def test_fill_probability_only_scales_down(self):
        default = FillModelConfig()
        ref = FillModel().estimate_fill(BUY, TIGHT).expected_fill_price
        # Model predicted a 0.90 fill rate but reality only filled half. Keep the
        # FILLED count at/above threshold and add an equal number of unfilled
        # resolutions so actual_fill_rate (0.5) < predicted (0.90).
        recs = _resolved(MIN_CALIBRATION_SAMPLES,
                         expected_entry=ref, actual_entry=ref, fill_prob=0.90)
        for i in range(MIN_CALIBRATION_SAMPLES):
            recs.append({"trade_id": f"u{i}", "resolution_status": "resolved",
                         "filled": False, "fill_probability": 0.90})
        cfg = calibrated_params_from(recs, base=default)
        self.assertLess(cfg.p_market, default.p_market)
        self.assertLess(cfg.p_marketable_limit, default.p_marketable_limit)
        self.assertLess(cfg.p_passive_limit, default.p_passive_limit)

    def test_slippage_never_drops_when_model_overcharged(self):
        # Model over-charged the entry (actual cheaper than expected). A negative
        # slippage error must NOT lower base_slippage (conservative floor).
        default = FillModelConfig()
        ref = FillModel().estimate_fill(BUY, TIGHT).expected_fill_price
        recs = _resolved(MIN_CALIBRATION_SAMPLES,
                         expected_entry=ref, actual_entry=round(ref - 0.05, 6))
        cfg = calibrated_params_from(recs, base=default)
        self.assertGreaterEqual(cfg.base_slippage, default.base_slippage)


class TestDeterminism(unittest.TestCase):
    def test_same_records_identical_config(self):
        default = FillModelConfig()
        ref = FillModel().estimate_fill(BUY, TIGHT).expected_fill_price
        recs = _resolved(MIN_CALIBRATION_SAMPLES,
                         expected_entry=ref, actual_entry=round(ref + 0.03, 6))
        self.assertEqual(
            calibrated_params_from(recs, base=default),
            calibrated_params_from(recs, base=default))


if __name__ == "__main__":
    unittest.main()
