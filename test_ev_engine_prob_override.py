"""
Regression lock for Oracle 3.0 — Phase 5's single live-touching plug:
``ev_engine.evaluate_proposal(..., prob_override=None)``.

No creds, no network, no broker. The contract this file pins down:

  1. DEFAULT-OFF IS BYTE-IDENTICAL. Omitting ``prob_override`` and passing it
     explicitly as ``None`` produce field-for-field identical ``EVResult``s, and
     those results still equal the modeled (driftless log-normal) PoP/EV — i.e.
     the pre-Oracle behavior is unchanged for every strategy family.
  2. INVALID OVERRIDES ARE IGNORED. NaN / out-of-range / non-numeric overrides
     fall through to the unchanged path (no-op, never raises).
  3. A VALID OVERRIDE ONLY CHANGES THE PROBABILITY INPUT. When supplied, the
     EVResult's ``probability_of_profit`` becomes the override and EV is
     recomputed from it via the same per-strategy formula — EV still arbitrates,
     nothing here sizes, prices, blocks, or submits a trade.

Mirrors test_ev_engine.py's proposal builders so the two stay in lock-step.
"""

import math
import unittest
from datetime import date, timedelta

import ev_engine as ev
from ev_engine import (
    evaluate_proposal, credit_spread_ev, iron_condor_ev,
    estimate_structure_costs, STATUS_OK,
)
from cost_model import CostModel, CostConfig
from spread_builder import (
    SpreadLeg, SpreadProposal,
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR,
)

SPOT, SIGMA, DAYS = 100.0, 0.25, 30
EXP = (date.today() + timedelta(days=DAYS)).isoformat()

FREE = CostModel(CostConfig(slippage_per_contract=0.0, occ_fee_per_contract=0.0,
                            commission_per_contract=0.0,
                            overnight_carry_per_contract_per_day=0.0,
                            min_spread_floor=0.0))


def leg(action, otype, strike, bid, ask):
    return SpreadLeg(action=action, option_type=otype, strike=strike,
                     bid=bid, ask=ask, expiration=EXP)


def bull_put():
    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol="SPY",
        legs=[leg("sell", "put", 95, 1.00, 1.10), leg("buy", "put", 90, 0.40, 0.50)],
        net_credit_or_debit=0.50, max_profit=50.0, max_loss=450.0,
        breakeven=94.5, width=5.0, oracle_score=62.0)


def bear_call():
    return SpreadProposal(
        strategy_name=BEARISH_CALL_CREDIT_SPREAD, symbol="SPY",
        legs=[leg("sell", "call", 105, 1.00, 1.10), leg("buy", "call", 110, 0.40, 0.50)],
        net_credit_or_debit=0.50, max_profit=50.0, max_loss=450.0,
        breakeven=105.5, width=5.0)


def debit_call():
    return SpreadProposal(
        strategy_name=DEBIT_CALL_SPREAD, symbol="SPY",
        legs=[leg("buy", "call", 100, 2.00, 2.20), leg("sell", "call", 105, 0.80, 0.90)],
        net_credit_or_debit=-1.40, max_profit=360.0, max_loss=140.0,
        breakeven=101.4, width=5.0)


def debit_put():
    return SpreadProposal(
        strategy_name=DEBIT_PUT_SPREAD, symbol="SPY",
        legs=[leg("buy", "put", 100, 2.00, 2.20), leg("sell", "put", 95, 0.80, 0.90)],
        net_credit_or_debit=-1.40, max_profit=360.0, max_loss=140.0,
        breakeven=98.6, width=5.0)


def condor():
    return SpreadProposal(
        strategy_name=IRON_CONDOR, symbol="SPY",
        legs=[leg("buy", "put", 90, 0.30, 0.40), leg("sell", "put", 95, 0.80, 0.90),
              leg("sell", "call", 105, 0.80, 0.90), leg("buy", "call", 110, 0.30, 0.40)],
        net_credit_or_debit=1.00, max_profit=100.0, max_loss=400.0,
        breakeven=[94.0, 106.0], width=5.0)


ALL = (bull_put, bear_call, debit_call, debit_put, condor)


class TestDefaultIsByteIdentical(unittest.TestCase):
    def test_omitted_equals_explicit_none(self):
        for build in ALL:
            p = build()
            base = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE)
            explicit = evaluate_proposal(p, SPOT, SIGMA, days=DAYS,
                                         cost_model=FREE, prob_override=None)
            self.assertEqual(vars(base), vars(explicit), msg=build.__name__)

    def test_none_path_matches_modeled_pop(self):
        # Bull put: modeled PoP via the log-normal; default None must reproduce it.
        r = evaluate_proposal(bull_put(), SPOT, SIGMA, days=DAYS, cost_model=FREE,
                              prob_override=None)
        self.assertEqual(r.status, STATUS_OK)
        pop = ev._p_terminal_above(SPOT, 94.5, SIGMA, 0.0, DAYS)
        costs = estimate_structure_costs(bull_put().legs, days=DAYS, model=FREE)
        self.assertAlmostEqual(r.probability_of_profit, round(pop, 4), places=4)
        self.assertAlmostEqual(r.expected_value,
                               round(credit_spread_ev(pop, 50.0, 450.0, costs), 2),
                               delta=0.011)


class TestInvalidOverridesAreNoOps(unittest.TestCase):
    def test_invalid_overrides_ignored(self):
        for build in ALL:
            p = build()
            base = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE)
            for bad in (float("nan"), float("inf"), -0.1, 1.2, "x", [], {}):
                got = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                                        prob_override=bad)  # type: ignore[arg-type]
                self.assertEqual(vars(base), vars(got),
                                 msg=f"{build.__name__} bad={bad!r}")


class TestValidOverrideChangesProbabilityOnly(unittest.TestCase):
    def test_credit_override_exact(self):
        p = bull_put()
        costs = estimate_structure_costs(p.legs, days=DAYS, model=FREE)
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                              prob_override=0.80)
        self.assertEqual(r.status, STATUS_OK)
        self.assertAlmostEqual(r.probability_of_profit, 0.80, places=4)
        self.assertAlmostEqual(r.expected_value,
                               round(credit_spread_ev(0.80, 50.0, 450.0, costs), 2),
                               delta=0.011)

    def test_condor_override_exact(self):
        p = condor()
        costs = estimate_structure_costs(p.legs, days=DAYS, model=FREE)
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                              prob_override=0.65)
        self.assertEqual(r.status, STATUS_OK)
        self.assertAlmostEqual(r.probability_of_profit, 0.65, places=4)
        self.assertAlmostEqual(r.expected_value,
                               round(iron_condor_ev(0.65, 0.35, 100.0, 400.0, costs), 2),
                               delta=0.011)

    def test_higher_pop_raises_credit_ev(self):
        p = bull_put()
        lo = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                               prob_override=0.40)
        hi = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                               prob_override=0.90)
        self.assertGreater(hi.expected_value, lo.expected_value)

    def test_debit_override_changes_ev_and_pop(self):
        for build in (debit_call, debit_put):
            p = build()
            base = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE)
            got = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE,
                                    prob_override=0.85)
            self.assertEqual(got.status, STATUS_OK, msg=build.__name__)
            self.assertAlmostEqual(got.probability_of_profit, 0.85, places=4,
                                   msg=build.__name__)
            self.assertNotEqual(got.expected_value, base.expected_value,
                                msg=build.__name__)

    def test_override_never_raises_on_garbage_proposal(self):
        for junk_prob in (0.5, None, "x", float("nan")):
            ev.evaluate_proposal(None, SPOT, SIGMA, prob_override=junk_prob)  # type: ignore[arg-type]


if __name__ == "__main__":
    unittest.main()
