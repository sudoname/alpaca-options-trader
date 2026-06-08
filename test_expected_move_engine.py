"""
Offline tests for Phase 7A expected-move / volatility-edge engine.

No creds, no network. All inputs are injected; CSV output goes to temp files.
"""

import csv
import math
import os
import tempfile
import unittest

from expected_move_engine import (
    EDGE_FAIR, EDGE_NA, EDGE_OVERPRICED, EDGE_UNDERPRICED,
    ExpectedMoveConfig, ExpectedMoveEngine, ExpectedMoveInputs,
    average_true_range, classify_vol_edge, gather_inputs_from_trader,
    realized_volatility,
)


class RealizedVolTests(unittest.TestCase):
    def test_none_when_too_few_closes(self):
        self.assertIsNone(realized_volatility([100.0]))
        self.assertIsNone(realized_volatility([]))

    def test_flat_series_is_zero(self):
        self.assertEqual(realized_volatility([100.0, 100.0, 100.0, 100.0]), 0.0)

    def test_positive_for_moving_series(self):
        closes = [100.0, 102.0, 101.0, 103.0, 99.0, 104.0]
        v = realized_volatility(closes)
        self.assertIsNotNone(v)
        self.assertGreater(v, 0.0)

    def test_window_uses_only_trailing_closes(self):
        closes = [100.0] * 50 + [100.0, 110.0, 95.0, 120.0, 90.0]
        full = realized_volatility(closes)
        windowed = realized_volatility(closes, window=5)
        # The recent window is far more volatile than the flat history.
        self.assertGreater(windowed, full)

    def test_unannualized_smaller_than_annualized(self):
        closes = [100.0, 102.0, 98.0, 103.0, 97.0]
        ann = realized_volatility(closes, annualize=True)
        raw = realized_volatility(closes, annualize=False)
        self.assertAlmostEqual(ann, raw * math.sqrt(252), places=6)


class ATRTests(unittest.TestCase):
    def test_none_when_insufficient_bars(self):
        bars = [{"h": 10, "l": 9, "c": 9.5}]
        self.assertIsNone(average_true_range(bars, period=14))

    def test_simple_atr(self):
        # 16 bars, each with a true range of 2.0 -> ATR 2.0.
        bars = [{"h": 12, "l": 10, "c": 11} for _ in range(16)]
        atr = average_true_range(bars, period=14)
        self.assertAlmostEqual(atr, 2.0, places=6)


class ClassifyEdgeTests(unittest.TestCase):
    def test_labels(self):
        self.assertEqual(classify_vol_edge(0.5), EDGE_OVERPRICED)
        self.assertEqual(classify_vol_edge(-0.5), EDGE_UNDERPRICED)
        self.assertEqual(classify_vol_edge(0.0), EDGE_FAIR)
        self.assertEqual(classify_vol_edge(None), EDGE_NA)


class ComputeTests(unittest.TestCase):
    def setUp(self):
        self.engine = ExpectedMoveEngine(ExpectedMoveConfig(enabled=True))

    def test_horizon_scaling_sqrt_time(self):
        inp = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                 recent_realized_vol=0.20, vix=20.0)
        res = self.engine.compute(inp, symbol="SPY")
        self.assertEqual(res.status, "ok")
        # 3d move ~ sqrt(3) * 1d move (no earnings multiplier here).
        self.assertAlmostEqual(res.expected_move_3d,
                               res.expected_move_1d * math.sqrt(3), places=4)
        self.assertAlmostEqual(res.expected_move_7d,
                               res.expected_move_1d * math.sqrt(7), places=4)

    def test_edge_positive_when_vix_rich(self):
        inp = ExpectedMoveInputs(hv20=0.15, hv60=0.15, hv90=0.15,
                                 recent_realized_vol=0.15, vix=30.0)
        res = self.engine.compute(inp, symbol="SPY")
        self.assertGreater(res.volatility_edge, 0.0)
        self.assertEqual(res.edge_label, EDGE_OVERPRICED)

    def test_edge_negative_when_vix_cheap(self):
        inp = ExpectedMoveInputs(hv20=0.40, hv60=0.40, hv90=0.40,
                                 recent_realized_vol=0.40, vix=12.0)
        res = self.engine.compute(inp, symbol="SPY")
        self.assertLess(res.volatility_edge, 0.0)
        self.assertEqual(res.edge_label, EDGE_UNDERPRICED)

    def test_edge_na_without_vix(self):
        inp = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                 recent_realized_vol=0.20, vix=None)
        res = self.engine.compute(inp, symbol="SPY")
        self.assertEqual(res.status, "ok")
        self.assertIsNone(res.volatility_edge)
        self.assertEqual(res.edge_label, EDGE_NA)

    def test_insufficient_data(self):
        inp = ExpectedMoveInputs()  # nothing -> no forecast vol
        res = self.engine.compute(inp, symbol="SPY")
        self.assertEqual(res.status, "insufficient_data")

    def test_dollars_when_price_given(self):
        inp = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                 recent_realized_vol=0.20, vix=20.0, price=400.0)
        res = self.engine.compute(inp, symbol="SPY")
        self.assertTrue(res.in_dollars)
        # A 20% annual vol at $400 -> 1d move well under $10.
        self.assertGreater(res.expected_move_1d, 0.0)
        self.assertLess(res.expected_move_1d, 20.0)

    def test_earnings_inflates_near_term(self):
        base = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                  recent_realized_vol=0.20, vix=20.0)
        bumped = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                    recent_realized_vol=0.20, vix=20.0,
                                    earnings_days=1)
        r0 = self.engine.compute(base)
        r1 = self.engine.compute(bumped)
        self.assertGreater(r1.expected_move_1d, r0.expected_move_1d)


class RecordTests(unittest.TestCase):
    def test_record_round_trip(self):
        d = tempfile.mkdtemp()
        path = os.path.join(d, "em.csv")
        engine = ExpectedMoveEngine(ExpectedMoveConfig(
            enabled=True, history_file=path))
        inp = ExpectedMoveInputs(hv20=0.20, hv60=0.20, hv90=0.20,
                                 recent_realized_vol=0.20, vix=22.0, price=400.0)
        res = engine.compute(inp, symbol="SPY")
        engine.record(res, inp)
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self.assertEqual(len(rows), 1)
        self.assertIn("in_hv20", rows[0])
        self.assertIn("volatility_edge", rows[0])
        self.assertEqual(rows[0]["symbol"], "SPY")


class GatherInputsTests(unittest.TestCase):
    class _FakeTrader:
        def __init__(self, closes):
            self._closes = closes

        def get_price_history(self, symbol, days=130):
            return self._closes

    def test_gathers_from_closes(self):
        closes = [100.0 + (i % 5) for i in range(130)]
        inp = gather_inputs_from_trader(self._FakeTrader(closes), "SPY", vix=18.0)
        self.assertIsNotNone(inp.hv20)
        self.assertIsNotNone(inp.hv90)
        self.assertEqual(inp.vix, 18.0)
        self.assertEqual(inp.price, closes[-1])

    def test_failopen_on_error(self):
        class Boom:
            def get_price_history(self, *a, **k):
                raise RuntimeError("no network")
        inp = gather_inputs_from_trader(Boom(), "SPY")
        self.assertIsNone(inp.hv20)
        self.assertIsNone(inp.price)


if __name__ == "__main__":
    unittest.main()
