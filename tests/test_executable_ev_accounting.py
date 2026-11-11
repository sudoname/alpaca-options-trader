"""
Phase 1 regression tests — executable EV accounting (costs charged once).

The reviewed bug: smart_trader passed the entry stamp's `expected_value`
(already NET of the stamp's modeled round-trip costs) into
compute_executable_ev as `theoretical_ev`, which subtracts its own execution
costs — charging costs twice. Review fixture: 1 contract, bid 0.99 / ask 1.01,
entry basis 1.01, confidence 3, delta 0.55, take-profit 25%, stop 15% —
executable EV about -$3.77 double-charged vs +$2.21 charged once.

These tests pin the SIGN and the accounting identities rather than exact
dollar values, so they stay valid if cost-model defaults change:
  * stamp:  gross_expected_value == expected_value + round_trip_costs
  * exec:   net_ev_after_costs  == theoretical_EV - execution_cost
  * exec:   executable_EV       == round(fill_probability * net_ev_after_costs, 6)
Plus: multi-contract scaling, zero/negative edge, unusable quotes (crossed /
one-sided / missing / stale -> EV unavailable with a reason code), explicit
fees, and the smart_trader gross-derivation logic (new stamp, legacy stamp,
empty stamp).

Run: python -m unittest tests.test_executable_ev_accounting -v
"""

import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model import CostConfig, CostModel
from entry_ev_stamp import compute_entry_stamp
from oracle.execution.client import Quote
from oracle.execution.executable_ev import compute_executable_ev, should_veto_entry
from oracle.execution.fill_model import FillModel

# Review fixture ------------------------------------------------------------
OPTION = {"symbol": "SPY260101C00600000", "confidence": 3, "delta": 0.55}
LEVELS = {"take_profit_percent": 0.25, "stop_loss_percent": 0.15}
BID, ASK, ENTRY, QTY = 0.99, 1.01, 1.01, 1


def _xev(theoretical_ev, qty=QTY, quote=None, cost_model=None, market=None,
         max_quote_age_s=None):
    return compute_executable_ev(
        OPTION, fair_value=None,
        quote=quote if quote is not None else Quote(OPTION["symbol"], bid=BID, ask=ASK),
        fill_model=FillModel(), cost_model=cost_model or CostModel(),
        qty=qty, theoretical_ev=theoretical_ev, market=market,
        max_quote_age_s=max_quote_age_s)


def _derive_theo(stamp):
    """Replicates smart_trader's gross-EV derivation (smart_trader.py ~2961).

    smart_trader can't be imported without broker credentials, so the exact
    logic is mirrored here; if the wiring there changes, update both.
    """
    theo = (stamp or {}).get("gross_expected_value")
    if theo is None and stamp:
        net = stamp.get("expected_value")
        rt = stamp.get("round_trip_costs")
        theo = (net + rt) if (net is not None and rt is not None) else net
    return theo


class TestReviewFixtureSign(unittest.TestCase):
    """The reviewed -$3.77 vs +$2.21 sign flip, pinned by sign not value."""

    def setUp(self):
        self.stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                         bid=BID, ask=ASK)
        self.assertTrue(self.stamp, "stamp must compute for the fixture")

    def test_old_wiring_double_charges_and_flips_sign(self):
        net = self.stamp["expected_value"]            # already cost-netted
        old = _xev(theoretical_ev=net)                # the pre-fix wiring
        self.assertEqual(old.status, "ok")
        self.assertIsNotNone(old.executable_EV)
        self.assertLess(old.executable_EV, 0,
                        "net-as-theoretical must double-charge to negative")

    def test_new_wiring_charges_once_and_stays_positive(self):
        gross = self.stamp["gross_expected_value"]
        new = _xev(theoretical_ev=gross)
        self.assertEqual(new.status, "ok")
        self.assertIsNotNone(new.executable_EV)
        self.assertGreater(new.executable_EV, 0,
                           "gross-as-theoretical charged once stays positive")

    def test_fix_recovers_roughly_reviewed_magnitudes(self):
        # Loose bands: defaults may drift, but the fixture should stay in the
        # neighborhood of -3.77 / +2.21 while CostModel defaults hold.
        old = _xev(theoretical_ev=self.stamp["expected_value"]).executable_EV
        new = _xev(theoretical_ev=self.stamp["gross_expected_value"]).executable_EV
        self.assertLess(old, -1.0)
        self.assertGreater(new, 1.0)


class TestAccountingIdentities(unittest.TestCase):
    def test_stamp_gross_equals_net_plus_costs(self):
        stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                    bid=BID, ask=ASK)
        self.assertAlmostEqual(
            stamp["gross_expected_value"],
            stamp["expected_value"] + stamp["round_trip_costs"], places=2)

    def test_stamp_gross_matches_pop_barrier_formula(self):
        stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                    bid=BID, ask=ASK)
        pop, tp, sl = (stamp["probability_of_profit"],
                       stamp["take_profit_pct"], stamp["stop_loss_pct"])
        expected = (pop * tp - (1.0 - pop) * sl) * stamp["max_loss"]
        self.assertAlmostEqual(stamp["gross_expected_value"], expected,
                               delta=0.05)  # rounding of pop to 4dp

    def test_stamp_units_and_quantity_explicit(self):
        stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                    bid=BID, ask=ASK)
        self.assertEqual(stamp["ev_units"], "usd_total")
        self.assertEqual(stamp["quantity"], QTY)
        self.assertGreaterEqual(stamp["stamp_version"], 3)

    def test_executable_single_charge_identity(self):
        stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                    bid=BID, ask=ASK)
        r = _xev(theoretical_ev=stamp["gross_expected_value"])
        self.assertIsNotNone(r.net_ev_after_costs)
        self.assertAlmostEqual(
            r.net_ev_after_costs, r.theoretical_EV - r.execution_cost,
            places=6)
        self.assertAlmostEqual(
            r.executable_EV,
            round(r.fill_probability * r.net_ev_after_costs, 6), places=6)


class TestMultiContract(unittest.TestCase):
    def test_stamp_scales_linearly_with_qty(self):
        s1 = compute_entry_stamp(OPTION, LEVELS, ENTRY, 1, bid=BID, ask=ASK)
        s3 = compute_entry_stamp(OPTION, LEVELS, ENTRY, 3, bid=BID, ask=ASK)
        self.assertAlmostEqual(s3["max_loss"], 3 * s1["max_loss"], places=2)
        self.assertAlmostEqual(s3["gross_expected_value"],
                               3 * s1["gross_expected_value"], delta=0.02)
        self.assertEqual(s3["quantity"], 3)

    def test_execution_costs_scale_with_qty(self):
        s1 = compute_entry_stamp(OPTION, LEVELS, ENTRY, 1, bid=BID, ask=ASK)
        s3 = compute_entry_stamp(OPTION, LEVELS, ENTRY, 3, bid=BID, ask=ASK)
        r1 = _xev(theoretical_ev=s1["gross_expected_value"], qty=1)
        r3 = _xev(theoretical_ev=s3["gross_expected_value"], qty=3)
        self.assertAlmostEqual(r3.execution_cost, 3 * r1.execution_cost,
                               places=6)
        self.assertAlmostEqual(r3.fees, 3 * r1.fees, places=6)
        # identity holds at size too
        self.assertAlmostEqual(
            r3.executable_EV,
            round(r3.fill_probability * (r3.theoretical_EV - r3.execution_cost), 6),
            places=6)


class TestZeroAndNegativeEdge(unittest.TestCase):
    def test_no_signal_means_ev_neutral_gross(self):
        flat = {"symbol": OPTION["symbol"], "confidence": 0, "delta": 0.40}
        stamp = compute_entry_stamp(flat, LEVELS, ENTRY, QTY, bid=BID, ask=ASK)
        self.assertAlmostEqual(stamp["gross_expected_value"], 0.0, places=2)
        # gambler's-ruin baseline pop with no tilt: sl / (tp + sl) = 0.375
        self.assertAlmostEqual(stamp["probability_of_profit"], 0.375, places=4)
        # net goes negative by exactly the modeled costs
        self.assertAlmostEqual(stamp["expected_value"],
                               -stamp["round_trip_costs"], places=2)

    def test_negative_gross_stays_negative_no_flip(self):
        # A negative theoretical edge is not the veto's job (upstream EV floor
        # rejects it); the veto flags only the +theo / -executable FLIP.
        r = _xev(theoretical_ev=-5.0)
        self.assertEqual(r.status, "ok")
        self.assertLess(r.executable_EV, 0)
        v = should_veto_entry(r.theoretical_EV, r.executable_EV,
                              shadow_mode=True, is_paper=True)
        self.assertFalse(v["would_reject"])
        self.assertFalse(v["veto"])

    def test_small_positive_gross_flips_and_would_reject(self):
        # Gross edge smaller than execution costs: the single-charged net goes
        # negative -> the flip the veto exists to catch.
        r = _xev(theoretical_ev=2.0)
        self.assertEqual(r.status, "ok")
        self.assertGreater(r.theoretical_EV, 0)
        self.assertLess(r.executable_EV, 0)
        shadow = should_veto_entry(r.theoretical_EV, r.executable_EV,
                                   shadow_mode=True, is_paper=True)
        self.assertTrue(shadow["would_reject"])
        self.assertFalse(shadow["veto"])          # shadow never acts
        armed = should_veto_entry(r.theoretical_EV, r.executable_EV,
                                  shadow_mode=False, is_paper=True)
        self.assertTrue(armed["veto"])            # armed only paper + no shadow
        live = should_veto_entry(r.theoretical_EV, r.executable_EV,
                                 shadow_mode=False, is_paper=False)
        self.assertFalse(live["veto"])            # never live-money


class TestUnusableQuotes(unittest.TestCase):
    """Unusable quotes -> EV unavailable (None) with a reason code."""

    def test_crossed_quote(self):
        r = _xev(theoretical_ev=10.0,
                 quote=Quote(OPTION["symbol"], bid=1.10, ask=1.02))
        self.assertEqual(r.status, "unusable_quote")
        self.assertIn("crossed_quote", r.reasons)
        self.assertIsNone(r.executable_EV)
        self.assertIsNone(r.net_ev_after_costs)

    def test_one_sided_quote(self):
        r = _xev(theoretical_ev=10.0,
                 quote=Quote(OPTION["symbol"], bid=1.00, ask=None))
        self.assertEqual(r.status, "unusable_quote")
        self.assertIn("one_sided_quote", r.reasons)
        self.assertIsNone(r.executable_EV)

    def test_nonpositive_quote(self):
        r = _xev(theoretical_ev=10.0,
                 quote=Quote(OPTION["symbol"], bid=0.50, ask=0.0))
        self.assertEqual(r.status, "unusable_quote")
        self.assertIn("nonpositive_quote", r.reasons)
        self.assertIsNone(r.executable_EV)

    def test_missing_quote(self):
        r = compute_executable_ev(OPTION, fair_value=None, quote=None,
                                  fill_model=FillModel(),
                                  cost_model=CostModel(), qty=QTY,
                                  theoretical_ev=10.0)
        self.assertEqual(r.status, "no_quote")
        self.assertIsNone(r.executable_EV)
        self.assertIsNone(r.net_ev_after_costs)

    def test_stale_quote_opt_in(self):
        stale = _xev(theoretical_ev=10.0,
                     market={"quote_age_s": 12.0}, max_quote_age_s=5.0)
        self.assertEqual(stale.status, "unusable_quote")
        self.assertIn("stale_quote", stale.reasons)
        self.assertIsNone(stale.executable_EV)
        fresh = _xev(theoretical_ev=10.0,
                     market={"quote_age_s": 2.0}, max_quote_age_s=5.0)
        self.assertEqual(fresh.status, "ok")
        self.assertIsNotNone(fresh.executable_EV)

    def test_no_age_cap_means_age_ignored(self):
        r = _xev(theoretical_ev=10.0, market={"quote_age_s": 9999.0})
        self.assertEqual(r.status, "ok")


class TestExplicitFees(unittest.TestCase):
    def test_commission_flows_into_fees_once(self):
        base = CostModel()
        pricey = CostModel(CostConfig(commission_per_contract=0.65))
        r0 = _xev(theoretical_ev=10.0, cost_model=base)
        r1 = _xev(theoretical_ev=10.0, cost_model=pricey)
        # fees = 2 sides * (occ + commission) * qty
        self.assertAlmostEqual(r1.fees - r0.fees, 2 * 0.65 * QTY, places=6)
        # the extra fee is charged exactly once in the net
        self.assertAlmostEqual(
            r0.net_ev_after_costs - r1.net_ev_after_costs,
            2 * 0.65 * QTY, places=6)


class TestSmartTraderGrossDerivation(unittest.TestCase):
    """The _theo derivation smart_trader now performs before compute call."""

    def test_new_stamp_uses_gross_field(self):
        stamp = compute_entry_stamp(OPTION, LEVELS, ENTRY, QTY,
                                    bid=BID, ask=ASK)
        self.assertEqual(_derive_theo(stamp), stamp["gross_expected_value"])

    def test_legacy_stamp_reconstructs_gross(self):
        legacy = {"expected_value": 0.63, "round_trip_costs": 6.04}  # v2 shape
        self.assertAlmostEqual(_derive_theo(legacy), 6.67, places=2)

    def test_legacy_stamp_without_costs_falls_back_to_net(self):
        legacy = {"expected_value": 0.63}
        self.assertAlmostEqual(_derive_theo(legacy), 0.63, places=2)

    def test_empty_stamp_yields_none(self):
        self.assertIsNone(_derive_theo({}))
        self.assertIsNone(_derive_theo(None))


if __name__ == "__main__":
    unittest.main(verbosity=2)
