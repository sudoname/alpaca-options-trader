"""
Offline tests for explain_context.build_explain_context (the read-only evidence
builder behind the dashboard's /api/explain endpoint).

No creds, no network: every test injects a ``market_view_factory`` that returns
either ``None``, a raising stub, or an offline ``HistoricalMarketView`` seeded
with synthetic bars. The contract pinned here:

  1. FAIL-OPEN. No market view / no bars / any raise -> ``{}`` (never raises),
     so explain degrades to INSUFFICIENT_DATA exactly as before.
  2. REAL EVIDENCE. Populated bars yield a ctx with trend/momentum/realized_vol/
     volume_ratio/rel_strength, and that ctx moves the agents off neutral
     (compute_oracle_explain -> verdict OK).
  3. SPY relative strength is pinned to 0.0 (no self-spread).
  4. READ-ONLY BY CONSTRUCTION. The module issues only HTTP GETs and imports no
     execution path -- asserted by source grep.
  5. ``_self_test()`` returns 0.
"""

import inspect
import unittest
from datetime import datetime

import explain_context as ec
from explain_context import build_explain_context
from market_view import HistoricalMarketView, make_bar


def _uptrend(n=12):
    return [make_bar(f"2026-01-{i + 1:02d}", 100 + i, 100 + i + 0.6,
                     99.6 + i, 100.5 + i,
                     1_000_000 + (50_000 if i == n - 1 else 0))
            for i in range(n)]


def _flat(price=400.0, n=12):
    return [make_bar(f"2026-01-{i + 1:02d}", price, price + 1, price - 1,
                     price, 1_000_000) for i in range(n)]


def _mv(daily):
    return HistoricalMarketView(datetime(2026, 1, 31, 16, 0), daily=daily)


class TestFailOpen(unittest.TestCase):
    def test_none_market_view_yields_empty(self):
        self.assertEqual(
            build_explain_context("SPY", market_view_factory=lambda: None), {})

    def test_raising_factory_yields_empty(self):
        def boom():
            raise RuntimeError("network down")
        self.assertEqual(
            build_explain_context("SPY", market_view_factory=boom), {})

    def test_no_bars_yields_empty(self):
        mv = _mv({"AAA": []})
        self.assertEqual(
            build_explain_context("AAA", market_view_factory=lambda: mv), {})


class TestRealEvidence(unittest.TestCase):
    def setUp(self):
        self.mv = _mv({"AAA": _uptrend(), "SPY": _flat()})
        self.ctx = build_explain_context("AAA", market_view_factory=lambda: self.mv)

    def test_ctx_is_populated(self):
        self.assertTrue(self.ctx)

    def test_trend_up(self):
        self.assertEqual(self.ctx.get("trend"), "up")

    def test_positive_momentum(self):
        self.assertIn("momentum", self.ctx)
        self.assertGreater(self.ctx["momentum"], 0)

    def test_carries_realized_vol_and_volume_ratio(self):
        self.assertIn("realized_vol", self.ctx)
        self.assertIn("volume_ratio", self.ctx)

    def test_positive_rel_strength_vs_flat_spy(self):
        self.assertIn("rel_strength", self.ctx)
        self.assertGreater(self.ctx["rel_strength"], 0)

    def test_ctx_moves_agents_off_neutral(self):
        import oracle_intelligence_reports as oir
        rep = oir.compute_oracle_explain("AAA", ctx=self.ctx)
        self.assertEqual(rep.get("verdict"), "OK")


class TestSpyRelStrength(unittest.TestCase):
    def test_spy_rel_strength_is_zero(self):
        mv = _mv({"SPY": _uptrend()})
        ctx = build_explain_context("SPY", market_view_factory=lambda: mv)
        self.assertEqual(ctx.get("rel_strength"), 0.0)


class TestReadOnlyByConstruction(unittest.TestCase):
    def test_no_execution_symbols_in_source(self):
        src = inspect.getsource(ec)
        forbidden = ("place_option_order(", "submit_order(", "execute_trade(",
                     "open_position(", "close_position(", "record_outcome(",
                     "requests.post(", "requests.put(", "requests.delete(",
                     "requests.patch(")
        for token in forbidden:
            self.assertNotIn(token, src, msg=f"forbidden token {token!r} present")


class TestSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(ec._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
