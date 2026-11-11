"""
Phase 2 — EV-derived buy-limit entry variant (opt-in) regression tests.

Covers the master-prompt requirements for the variant:
  * the cap is derived from the expected edge after costs and the EV floor,
    with single-charge accounting identical to ``executable_ev``;
  * valid option tick sizes (penny / nickel-dime grids, explicit override),
    rounded DOWN (conservative) so a fill at the cap still clears the floor;
  * quote age and spread/size constraints respected;
  * explicit expiry stamped on the result;
  * any permitted repricing stays within the cap;
  * a reason is logged whenever a valid cap cannot be formed;
  * comparison fields against the current market-entry baseline
    (``net_ev_at_market`` vs ``net_ev_at_limit``) are always carried;
  * the kernel-mode adapter rewrites the intent as a limit order at the policy
    cap when configured, abstains (or falls back to market) on no-cap, and
    fails open to the baseline market order on any policy error.

All hand-math anchors were verified independently:
  zero-friction config, quote 1.00/1.02, qty 1: basis 1.02, round trip $2.
    gross 10  -> cap 1.10 (marketable), net@limit 0.00
    gross 7.5 -> cap 1.07,              net@limit 0.50
    floor 5   -> cap 1.05
    gross 1   -> cap 1.01 (NON-marketable resting order), net@limit 0.00
  default config (slip .02 / occ .02 each side): basis 1.04, cost $6.04.
    gross 15        -> raw 1.1296 -> cap 1.12, net@limit 0.96
    qty 2, gross 30 -> same cap (scale invariance), net@limit 1.92
    nickel grid     -> 1.10;  explicit tick 0.25 -> 1.00

No creds, no network, no order placement, no file writes.
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model import CostConfig, CostModel
from oracle.engine.events import BarEvent
from oracle.engine.strategy import OracleStrategyAdapter, StrategyConfig
from oracle.execution.client import (
    STATUS_FILLED,
    STATUS_PENDING,
    OrderRequest,
    Quote,
    SimExecutionClient,
)
from oracle.execution.limit_policy import (
    R_CAP_NONPOSITIVE,
    R_CROSSED_QUOTE,
    R_EV_UNAVAILABLE,
    R_INVALID_QTY,
    R_NO_QUOTE,
    R_NONPOSITIVE_QUOTE,
    R_ONE_SIDED_QUOTE,
    R_SPREAD_TOO_WIDE,
    R_STALE_QUOTE,
    compute_entry_limit,
    gross_ev_from_contract,
    option_tick,
    reprice_within_cap,
    round_down_to_tick,
)

FREE = CostModel(CostConfig(slippage_per_contract=0.0,
                            occ_fee_per_contract=0.0,
                            commission_per_contract=0.0))
DEFAULT = CostModel()  # slip .02, occ .02 each side


# --------------------------------------------------------------------------- #
# Pure policy math
# --------------------------------------------------------------------------- #
class TestCapMath(unittest.TestCase):
    """Hand-verified anchors: cap = basis + (net@market − floor)/(100·qty)."""

    def test_zero_friction_exact_tick(self):
        el = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=FREE)
        self.assertEqual(el.status, "ok")
        self.assertAlmostEqual(el.buy_basis, 1.02)
        self.assertAlmostEqual(el.execution_cost, 2.0)
        self.assertAlmostEqual(el.net_ev_at_market, 8.0)
        self.assertAlmostEqual(el.limit_price, 1.10)
        self.assertTrue(el.marketable)
        # Filled exactly at the cap leaves exactly the floor.
        self.assertAlmostEqual(el.net_ev_at_limit, 0.0)

    def test_down_rounding_is_conservative(self):
        el = compute_entry_limit(gross_ev=7.5, qty=1, bid=1.00, ask=1.02,
                                 cost_model=FREE)
        self.assertAlmostEqual(el.limit_price, 1.07)  # raw 1.075 rounds DOWN
        self.assertAlmostEqual(el.net_ev_at_limit, 0.5)
        self.assertGreaterEqual(el.net_ev_at_limit, el.ev_floor)

    def test_ev_floor_shifts_cap_down(self):
        el = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=FREE, ev_floor=5.0)
        self.assertAlmostEqual(el.limit_price, 1.05)
        self.assertGreaterEqual(el.net_ev_at_limit, 5.0)

    def test_thin_edge_rests_below_ask(self):
        el = compute_entry_limit(gross_ev=1.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=FREE)
        self.assertEqual(el.status, "ok")
        self.assertAlmostEqual(el.limit_price, 1.01)
        self.assertFalse(el.marketable)   # resting order, not a rejection
        self.assertGreaterEqual(el.net_ev_at_limit, 0.0)

    def test_default_cost_model_charges_once(self):
        el = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=DEFAULT)
        self.assertAlmostEqual(el.buy_basis, 1.04)       # ask + slippage
        self.assertAlmostEqual(el.execution_cost, 6.04)  # spread+2slip+2fee
        self.assertAlmostEqual(el.net_ev_at_market, 8.96)
        self.assertAlmostEqual(el.limit_price, 1.12)     # raw 1.1296 DOWN
        self.assertAlmostEqual(el.net_ev_at_limit, 0.96)
        # Identity: net@limit = net@market + (basis − cap)·100·qty.
        self.assertAlmostEqual(
            el.net_ev_at_limit,
            el.net_ev_at_market + (el.buy_basis - el.limit_price) * 100.0)

    def test_qty_scale_invariance(self):
        el1 = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                  cost_model=DEFAULT)
        el2 = compute_entry_limit(gross_ev=30.0, qty=2, bid=1.00, ask=1.02,
                                  cost_model=DEFAULT)
        self.assertAlmostEqual(el1.limit_price, el2.limit_price)
        self.assertAlmostEqual(el2.net_ev_at_limit, 2 * el1.net_ev_at_limit)

    def test_floor_guarantee_property_sweep(self):
        """Whenever a cap forms, a fill AT the cap clears the floor."""
        for gross in (0.5, 1.0, 2.5, 5.0, 7.3, 10.0, 25.0, 123.45):
            for floor in (0.0, 1.0, 3.0):
                el = compute_entry_limit(gross_ev=gross, qty=1,
                                         bid=1.00, ask=1.02,
                                         cost_model=DEFAULT, ev_floor=floor)
                if el.status == "ok":
                    self.assertGreaterEqual(
                        el.net_ev_at_limit, floor - 1e-9,
                        f"gross={gross} floor={floor} -> {el}")

    def test_determinism(self):
        a = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                cost_model=DEFAULT)
        b = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                cost_model=DEFAULT)
        self.assertEqual(a, b)

    def test_expiry_stamped(self):
        el = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=FREE, expiry_s=45.0)
        self.assertEqual(el.expiry_s, 45.0)
        self.assertEqual(
            compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                                cost_model=FREE).expiry_s, 30.0)


class TestTicks(unittest.TestCase):
    def test_grids(self):
        self.assertEqual(option_tick(1.50, penny_quoted=True), 0.01)
        self.assertEqual(option_tick(2.99, penny_quoted=False), 0.05)
        self.assertEqual(option_tick(3.00, penny_quoted=False), 0.10)

    def test_nickel_grid_rounds_down(self):
        el = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=DEFAULT, penny_quoted=False)
        self.assertAlmostEqual(el.limit_price, 1.10)  # raw 1.1296 on nickels

    def test_explicit_tick_override(self):
        el = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                                 cost_model=DEFAULT, tick=0.25)
        self.assertAlmostEqual(el.limit_price, 1.00)
        self.assertEqual(el.tick, 0.25)

    def test_exact_multiple_not_knocked_down(self):
        self.assertAlmostEqual(round_down_to_tick(1.10, 0.01), 1.10, places=12)
        self.assertAlmostEqual(round_down_to_tick(0.3, 0.05), 0.30, places=12)


class TestNoCapReasons(unittest.TestCase):
    """Every refusal carries a reason code and no price; never raises."""

    CASES = {
        R_INVALID_QTY: dict(gross_ev=10.0, qty=0, bid=1.0, ask=1.02),
        R_EV_UNAVAILABLE: dict(gross_ev=None, qty=1, bid=1.0, ask=1.02),
        R_NO_QUOTE: dict(gross_ev=10.0, qty=1, bid=None, ask=None),
        R_NONPOSITIVE_QUOTE: dict(gross_ev=10.0, qty=1, bid=0.0, ask=0.0),
        R_ONE_SIDED_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.0, ask=None),
        R_CROSSED_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.10, ask=1.02),
        R_STALE_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                            quote_age_s=12.0, max_quote_age_s=5.0),
        R_SPREAD_TOO_WIDE: dict(gross_ev=50.0, qty=1, bid=1.00, ask=1.40,
                                max_spread_pct=20.0),
        R_CAP_NONPOSITIVE: dict(gross_ev=-500.0, qty=1, bid=1.0, ask=1.02),
    }

    def test_each_reason(self):
        for reason, kw in self.CASES.items():
            el = compute_entry_limit(cost_model=FREE, **kw)
            self.assertEqual(el.status, "no_cap", reason)
            self.assertIn(reason, el.reasons)
            self.assertIsNone(el.limit_price, reason)

    def test_fresh_quote_passes_age_cap(self):
        el = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                                 cost_model=FREE, quote_age_s=2.0,
                                 max_quote_age_s=5.0)
        self.assertEqual(el.status, "ok")

    def test_junk_inputs_never_raise(self):
        el = compute_entry_limit(gross_ev="x", qty="y", bid=object(), ask="z")
        self.assertEqual(el.status, "no_cap")


class TestReprice(unittest.TestCase):
    """Permitted repricing chases the market but never exceeds the cap."""

    def setUp(self):
        self.el = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00,
                                      ask=1.02, cost_model=DEFAULT)  # cap 1.12

    def test_clamped_to_cap(self):
        self.assertEqual(reprice_within_cap(self.el, 1.25), self.el.limit_price)

    def test_tick_rounds_down(self):
        self.assertAlmostEqual(reprice_within_cap(self.el, 1.1049), 1.10)

    def test_below_cap_passes(self):
        self.assertAlmostEqual(reprice_within_cap(self.el, 1.05), 1.05)

    def test_no_cap_no_reprice(self):
        nocap = compute_entry_limit(gross_ev=None, qty=1, bid=1.0, ask=1.02)
        self.assertIsNone(reprice_within_cap(nocap, 1.05))
        self.assertIsNone(reprice_within_cap(self.el, None))


class TestGrossEvSourcing(unittest.TestCase):
    """v3 gross preferred; net+costs reconstructed; net-only lower bound."""

    def test_chain(self):
        self.assertEqual(gross_ev_from_contract(
            {"gross_expected_value": 12.5, "expected_value": 6.0}), 12.5)
        self.assertEqual(gross_ev_from_contract(
            {"expected_value": 6.0, "round_trip_costs": 6.04}), 12.04)
        self.assertEqual(gross_ev_from_contract({"expected_value": 6.0}), 6.0)
        self.assertIsNone(gross_ev_from_contract({}))
        self.assertIsNone(gross_ev_from_contract(None))
        self.assertIsNone(gross_ev_from_contract(
            {"gross_expected_value": "nan"}))

    def test_net_fallback_only_lowers_the_cap(self):
        """NET <= GROSS, so the fallback can never overpay vs the true cap."""
        true_cap = compute_entry_limit(gross_ev=12.04, qty=1, bid=1.00,
                                       ask=1.02, cost_model=DEFAULT)
        net_cap = compute_entry_limit(
            gross_ev=gross_ev_from_contract({"expected_value": 6.0}),
            qty=1, bid=1.00, ask=1.02, cost_model=DEFAULT)
        self.assertLessEqual(net_cap.limit_price, true_cap.limit_price)


# --------------------------------------------------------------------------- #
# Adapter integration (kernel mode + entry_limit_config)
# --------------------------------------------------------------------------- #
from oracle.decision.snapshot_builder import (  # noqa: E402
    CTX_MARKET_REGIME,
    CTX_MOMENTUM,
    CTX_PRICES,
    CTX_QUOTE,
    CTX_VOLATILITY,
)

T0 = datetime(2024, 1, 2, 16, 0, 0)
CONTRACTS = (
    {"symbol": "AAA_C1", "type": "call", "iv": 0.30, "dte": 5,
     "score": 8.0, "confidence": 2, "expected_value": 15.0,
     "probability_of_profit": 0.55, "ev_per_dollar_risk": 0.12,
     "bid": 1.00, "ask": 1.05},
)


def _ctx_builder(symbol, event, clock):
    return {"expected_move_pct": 3.0, "underlying_price": 498.0,
            CTX_PRICES: [100.0, 101.0, 102.0, 104.0, 106.0],
            CTX_MOMENTUM: 0.05, CTX_VOLATILITY: 0.2,
            CTX_MARKET_REGIME: "trending",
            CTX_QUOTE: {"bid": 1.00, "ask": 1.05, "quote_age_s": 1.0}}


def _contracts_provider(symbol, event, clock):
    return CONTRACTS


def _bar():
    return BarEvent(symbol="AAA", event_timestamp=T0, o=100.0, c=106.0,
                    strategy_mode="intraday")


def _adapter(emitted, entry_limit_config):
    return OracleStrategyAdapter(
        emit=emitted.append, config=StrategyConfig(), use_kernel=True,
        ctx_builder=_ctx_builder, contracts_provider=_contracts_provider,
        entry_limit_config=entry_limit_config)


def _intents(emitted):
    return [e for e in emitted if e.event_type == "TRADE_INTENT"]


class TestAdapterEntryLimit(unittest.TestCase):
    def test_default_is_market_baseline(self):
        """Without the opt-in config the intent is unchanged (market)."""
        emitted = []
        _adapter(emitted, entry_limit_config=None).handle(_bar())
        it = _intents(emitted)[0]
        self.assertEqual(it.order_type, "market")
        self.assertIsNone(it.limit_price)

    def test_opt_in_emits_policy_cap(self):
        """The intent becomes a limit at exactly the pure-policy cap, and the
        full EntryLimit (baseline comparison included) is logged."""
        emitted = []
        adapter = _adapter(emitted, {"cost_model": FREE, "ev_floor": 0.0,
                                     "expiry_s": 20.0})
        adapter.handle(_bar())
        its = _intents(emitted)
        self.assertEqual(len(its), 1)
        it = its[0]

        expected = compute_entry_limit(
            gross_ev=gross_ev_from_contract(CONTRACTS[0]), qty=it.qty,
            bid=1.00, ask=1.05, quote_age_s=1.0, cost_model=FREE,
            ev_floor=0.0, expiry_s=20.0)
        self.assertEqual(expected.status, "ok")
        self.assertEqual(it.order_type, "limit")
        self.assertAlmostEqual(it.limit_price, expected.limit_price)

        logged = adapter.signals[-1]["entry_limit"]
        self.assertEqual(logged["status"], "ok")
        self.assertAlmostEqual(logged["limit_price"], expected.limit_price)
        # Baseline-vs-variant comparison fields always carried.
        self.assertAlmostEqual(logged["net_ev_at_market"],
                               expected.net_ev_at_market)
        self.assertAlmostEqual(logged["net_ev_at_limit"],
                               expected.net_ev_at_limit)
        self.assertEqual(logged["expiry_s"], 20.0)

    def test_no_cap_abstains_with_reason(self):
        """Impossible floor -> no cap -> no intent; reason logged."""
        emitted = []
        adapter = _adapter(emitted, {"cost_model": FREE, "ev_floor": 1e9})
        adapter.handle(_bar())
        self.assertEqual(_intents(emitted), [])
        logged = adapter.signals[-1]["entry_limit"]
        self.assertEqual(logged["status"], "no_cap")
        self.assertIn(R_CAP_NONPOSITIVE, logged["reasons"])

    def test_no_cap_market_fallback(self):
        """fallback='market' falls through to the baseline market order."""
        emitted = []
        adapter = _adapter(emitted, {"cost_model": FREE, "ev_floor": 1e9,
                                     "fallback": "market"})
        adapter.handle(_bar())
        it = _intents(emitted)[0]
        self.assertEqual(it.order_type, "market")
        self.assertIsNone(it.limit_price)
        self.assertEqual(
            adapter.signals[-1]["entry_limit"]["status"], "no_cap")

    def test_stale_quote_abstains(self):
        emitted = []
        adapter = _adapter(emitted, {"cost_model": FREE,
                                     "max_quote_age_s": 0.5})
        adapter.handle(_bar())  # ctx quote_age_s = 1.0 > 0.5
        self.assertEqual(_intents(emitted), [])
        self.assertIn(R_STALE_QUOTE,
                      adapter.signals[-1]["entry_limit"]["reasons"])

    def test_policy_error_fails_open_to_market(self):
        """A broken cost_model must never block the baseline entry."""
        emitted = []
        adapter = _adapter(emitted, {"cost_model": object()})
        adapter.handle(_bar())
        it = _intents(emitted)[0]
        self.assertEqual(it.order_type, "market")
        self.assertIsNone(it.limit_price)


class TestExecutionOfCappedIntent(unittest.TestCase):
    """The Sim broker's behaviour on the capped order matches the policy's
    marketable flag: marketable caps fill (with price improvement), resting
    caps pend — they never fill above the cap."""

    def test_marketable_cap_fills_at_ask(self):
        el = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.05,
                                 cost_model=FREE)
        self.assertTrue(el.marketable)
        sim = SimExecutionClient(
            quotes={"AAA_C1": Quote("AAA_C1", bid=1.00, ask=1.05, ts="t")})
        res = sim.submit_order(OrderRequest(
            "AAA_C1", "buy", 1, order_type="limit",
            limit_price=el.limit_price))
        self.assertEqual(res.status, STATUS_FILLED)
        self.assertLessEqual(res.filled_avg_price, el.limit_price)

    def test_resting_cap_pends_never_overpays(self):
        el = compute_entry_limit(gross_ev=1.0, qty=1, bid=1.00, ask=1.05,
                                 cost_model=FREE)
        self.assertEqual(el.status, "ok")
        self.assertFalse(el.marketable)
        sim = SimExecutionClient(
            quotes={"AAA_C1": Quote("AAA_C1", bid=1.00, ask=1.05, ts="t")})
        res = sim.submit_order(OrderRequest(
            "AAA_C1", "buy", 1, order_type="limit",
            limit_price=el.limit_price))
        self.assertEqual(res.status, STATUS_PENDING)
        self.assertEqual(res.filled_qty, 0.0)


# --------------------------------------------------------------------------- #
# Standalone runner (run_selftests.py executes this without pytest).
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print("tests.test_entry_limit_policy self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
