"""
Offline tests for Phase 10A — the Expected Value (EV) engine.

No creds, no network, no broker. Covers the ten required areas:
  1. Credit spread EV          5. EV per dollar risk      8. Invalid probabilities
  2. Debit spread EV           6. Recommendation tiers    9. Telegram output
  3. Iron condor EV            7. Empty data handling    10. No execution path touched
  4. Cost adjustments

ev_engine is ADVISORY ONLY: the last test class statically proves the live
execution modules (run_alpaca_intraday, smart_trader) do not consume it, and
that ev_engine itself never imports the trader or talks to the network.
"""

import math
import os
import unittest
from datetime import date, timedelta

import ev_engine as ev
from ev_engine import (
    EVConfig, EVResult,
    credit_spread_ev, debit_spread_ev, iron_condor_ev,
    ev_per_dollar_risk, classify_ev, estimate_structure_costs,
    evaluate_proposal, evaluate_for_symbol, format_ev_report,
    STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE,
    STATUS_OK, STATUS_INSUFFICIENT,
)
from cost_model import CostModel, CostConfig
from spread_builder import (
    SpreadLeg, SpreadProposal,
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
)

HERE = os.path.dirname(os.path.abspath(__file__))
SPOT, SIGMA, DAYS = 100.0, 0.25, 30
EXP = (date.today() + timedelta(days=DAYS)).isoformat()

# A cost model with zero frictions (and bid==ask quotes) -> costs == 0.
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


# --------------------------------------------------------------------------- #
# 1. Credit spread EV
# --------------------------------------------------------------------------- #
class TestCreditSpreadEV(unittest.TestCase):
    def test_formula_exact(self):
        # EV = 0.7*45 - 0.3*155 - 9 = 31.5 - 46.5 - 9 = -24.0
        self.assertAlmostEqual(credit_spread_ev(0.7, 45.0, 155.0, 9.0), -24.0, places=6)

    def test_zero_costs_default(self):
        self.assertAlmostEqual(credit_spread_ev(0.5, 100.0, 100.0), 0.0, places=6)

    def test_bull_put_evaluate_self_consistent(self):
        p = bull_put()
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE)
        self.assertEqual(r.status, STATUS_OK)
        # Bull put with breakeven below spot: PoP must be > 0.5.
        self.assertGreater(r.probability_of_profit, 0.5)
        pop = ev._p_terminal_above(SPOT, 94.5, SIGMA, 0.0, DAYS)
        costs = estimate_structure_costs(p.legs, days=DAYS, model=FREE)
        expected = credit_spread_ev(pop, 50.0, 450.0, costs)
        self.assertAlmostEqual(r.expected_value, expected, delta=0.011)

    def test_bear_call_pop_is_below_breakeven_prob(self):
        r = evaluate_proposal(bear_call(), SPOT, SIGMA, days=DAYS, cost_model=FREE)
        self.assertEqual(r.status, STATUS_OK)
        pop = ev._p_terminal_below(SPOT, 105.5, SIGMA, 0.0, DAYS)
        self.assertAlmostEqual(r.probability_of_profit, pop, places=4)


# --------------------------------------------------------------------------- #
# 2. Debit spread EV
# --------------------------------------------------------------------------- #
class TestDebitSpreadEV(unittest.TestCase):
    def test_formula_exact(self):
        # 0.2*360 + 0.3*180 - 0.5*140 - 10 = 72 + 54 - 70 - 10 = 46.0
        out = debit_spread_ev(0.2, 0.3, 0.5, 360.0, 180.0, 140.0, 10.0)
        self.assertAlmostEqual(out, 46.0, places=6)

    def test_regions_partition_to_one(self):
        for proposal, short_k, be, upper in (
            (debit_call(), 105.0, 101.4, True),
            (debit_put(), 95.0, 98.6, False),
        ):
            r = evaluate_proposal(proposal, SPOT, SIGMA, days=DAYS, cost_model=FREE)
            self.assertEqual(r.status, STATUS_OK)
            if upper:
                p_max = ev._p_terminal_above(SPOT, short_k, SIGMA, 0.0, DAYS)
                p_loss = ev._p_terminal_below(SPOT, be, SIGMA, 0.0, DAYS)
            else:
                p_max = ev._p_terminal_below(SPOT, short_k, SIGMA, 0.0, DAYS)
                p_loss = ev._p_terminal_above(SPOT, be, SIGMA, 0.0, DAYS)
            p_partial = 1.0 - p_max - p_loss
            self.assertGreaterEqual(p_partial, 0.0)
            self.assertAlmostEqual(r.probability_of_profit, p_max + p_partial, places=4)
            expected = debit_spread_ev(p_max, p_partial, p_loss, 360.0, 180.0, 140.0,
                                       estimate_structure_costs(proposal.legs,
                                                                days=DAYS, model=FREE))
            self.assertAlmostEqual(r.expected_value, expected, delta=0.011)

    def test_partial_payout_is_half_max_profit(self):
        # With p_partial = 1 (force via pure function): EV = partial_payout - costs.
        out = debit_spread_ev(0.0, 1.0, 0.0, 360.0, 180.0, 140.0, 0.0)
        self.assertAlmostEqual(out, 180.0, places=6)


# --------------------------------------------------------------------------- #
# 3. Iron condor EV
# --------------------------------------------------------------------------- #
class TestIronCondorEV(unittest.TestCase):
    def test_formula_exact(self):
        # 0.8*100 - 0.2*400 - 5 = 80 - 80 - 5 = -5.0
        self.assertAlmostEqual(iron_condor_ev(0.8, 0.2, 100.0, 400.0, 5.0), -5.0, places=6)

    def test_evaluate_range_probability(self):
        r = evaluate_proposal(condor(), SPOT, SIGMA, days=DAYS, cost_model=FREE)
        self.assertEqual(r.status, STATUS_OK)
        range_p = (ev._p_terminal_below(SPOT, 106.0, SIGMA, 0.0, DAYS)
                   - ev._p_terminal_below(SPOT, 94.0, SIGMA, 0.0, DAYS))
        self.assertAlmostEqual(r.probability_of_profit, range_p, places=4)
        costs = estimate_structure_costs(condor().legs, days=DAYS, model=FREE)
        expected = iron_condor_ev(range_p, 1.0 - range_p, 100.0, 400.0, costs)
        self.assertAlmostEqual(r.expected_value, expected, delta=0.011)

    def test_tail_plus_range_must_not_exceed_one(self):
        self.assertIsNone(iron_condor_ev(0.8, 0.3, 100.0, 400.0, 0.0))


# --------------------------------------------------------------------------- #
# 4. Cost adjustments
# --------------------------------------------------------------------------- #
class TestCostAdjustments(unittest.TestCase):
    def test_quoted_legs_sum_round_trip(self):
        model = CostModel()
        legs = bull_put().legs
        want = sum(model.round_trip_cost(l.bid, l.ask, qty=1,
                                         hold_days=DAYS)["cost_dollars"]
                   for l in legs)
        self.assertAlmostEqual(estimate_structure_costs(legs, days=DAYS, model=model),
                               want, places=6)

    def test_missing_quote_uses_conservative_floor(self):
        legs = [SpreadLeg(action="sell", option_type="put", strike=95)]  # no quotes
        cost = estimate_structure_costs(legs, days=0, model=CostModel())
        self.assertGreater(cost, 0.0)

    def test_higher_costs_lower_ev(self):
        p = bull_put()
        free = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=FREE)
        # Default model is identical math but with real frictions -> lower EV.
        # Use bid==ask so only the model's frictions differ, not the quotes.
        paid = evaluate_proposal(p, SPOT, SIGMA, days=DAYS, cost_model=CostModel())
        self.assertGreater(free.expected_value, paid.expected_value)
        self.assertGreater(paid.estimated_costs, free.estimated_costs)

    def test_costs_reduce_ev_by_exact_amount(self):
        self.assertAlmostEqual(
            credit_spread_ev(0.7, 50.0, 450.0, 0.0)
            - credit_spread_ev(0.7, 50.0, 450.0, 12.5),
            12.5, places=6)


# --------------------------------------------------------------------------- #
# 5. EV per dollar risk
# --------------------------------------------------------------------------- #
class TestEVPerRisk(unittest.TestCase):
    def test_basic_ratio(self):
        self.assertAlmostEqual(ev_per_dollar_risk(20.0, 100.0), 0.2, places=6)

    def test_undefined_cases(self):
        self.assertIsNone(ev_per_dollar_risk(None, 100.0))
        self.assertIsNone(ev_per_dollar_risk(20.0, 0.0))
        self.assertIsNone(ev_per_dollar_risk(20.0, -5.0))
        self.assertIsNone(ev_per_dollar_risk(float("nan"), 100.0))


# --------------------------------------------------------------------------- #
# 6. Recommendation thresholds
# --------------------------------------------------------------------------- #
class TestRecommendationThresholds(unittest.TestCase):
    def test_default_tiers(self):
        cases = [
            (0.30, STRONG_ACCEPT), (0.15, STRONG_ACCEPT),
            (0.149, ACCEPT), (0.05, ACCEPT),
            (0.049, NEUTRAL), (0.0, NEUTRAL),
            (-0.01, WEAK_SETUP), (-0.05, WEAK_SETUP),
            (-0.051, REJECT_CANDIDATE), (-0.50, REJECT_CANDIDATE),
        ]
        for ratio, want in cases:
            self.assertEqual(classify_ev(ratio), want, msg=f"ratio={ratio}")

    def test_none_and_nan_are_neutral(self):
        self.assertEqual(classify_ev(None), NEUTRAL)
        self.assertEqual(classify_ev(float("nan")), NEUTRAL)

    def test_custom_config(self):
        cfg = EVConfig(strong_accept_min=0.5, accept_min=0.25, weak_min=-0.25)
        self.assertEqual(classify_ev(0.3, cfg), ACCEPT)
        self.assertEqual(classify_ev(-0.2, cfg), WEAK_SETUP)


# --------------------------------------------------------------------------- #
# 7. Empty data handling
# --------------------------------------------------------------------------- #
class TestEmptyDataHandling(unittest.TestCase):
    def test_none_proposal(self):
        r = evaluate_proposal(None, SPOT, SIGMA)
        self.assertEqual(r.status, STATUS_INSUFFICIENT)
        self.assertIsNone(r.expected_value)

    def test_no_trade_proposal_passes_reason_through(self):
        p = SpreadProposal(strategy_name=NO_TRADE, symbol="SPY", reason="weak_vol_edge")
        r = evaluate_proposal(p, SPOT, SIGMA)
        self.assertEqual(r.status, STATUS_INSUFFICIENT)
        self.assertEqual(r.reason, "weak_vol_edge")

    def test_missing_breakeven(self):
        p = bull_put()
        p.breakeven = None
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS)
        self.assertEqual(r.status, STATUS_INSUFFICIENT)

    def test_condor_needs_two_breakevens(self):
        p = condor()
        p.breakeven = 100.0  # scalar instead of [low, high]
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS)
        self.assertEqual(r.status, STATUS_INSUFFICIENT)

    def test_nonpositive_max_loss(self):
        p = bull_put()
        p.max_loss = 0.0
        r = evaluate_proposal(p, SPOT, SIGMA, days=DAYS)
        self.assertEqual(r.status, STATUS_INSUFFICIENT)


# --------------------------------------------------------------------------- #
# 8. Invalid probabilities
# --------------------------------------------------------------------------- #
class TestInvalidProbabilities(unittest.TestCase):
    def test_pure_formulas_reject_bad_probs(self):
        for bad in (1.2, -0.1, float("nan"), None, "x"):
            self.assertIsNone(credit_spread_ev(bad, 50.0, 450.0, 0.0),
                              msg=f"pop={bad}")
        self.assertIsNone(debit_spread_ev(0.6, 0.6, 0.6, 360.0, 180.0, 140.0, 0.0))
        self.assertIsNone(iron_condor_ev(1.5, -0.5, 100.0, 400.0, 0.0))

    def test_evaluate_rejects_bad_spot_or_sigma(self):
        p = bull_put()
        for spot, sigma in ((None, SIGMA), (0.0, SIGMA), (-5.0, SIGMA),
                            (SPOT, None), (SPOT, 0.0), (SPOT, -0.2)):
            r = evaluate_proposal(p, spot, sigma, days=DAYS)
            self.assertEqual(r.status, STATUS_INSUFFICIENT,
                             msg=f"spot={spot} sigma={sigma}")

    def test_bad_days_falls_back_to_default(self):
        p = bull_put()
        r = evaluate_proposal(p, SPOT, SIGMA, days=-3, cost_model=FREE)
        self.assertEqual(r.status, STATUS_OK)
        self.assertEqual(r.days, EVConfig().default_days)


# --------------------------------------------------------------------------- #
# 9. Telegram output
# --------------------------------------------------------------------------- #
class _FakeTrader:
    """Duck-typed trader: spread proposal + price history only. No network."""

    def __init__(self, proposal):
        self._proposal = proposal
        # 130 closes with mild variation -> realized vol > 0 for the EM engine.
        self._closes = [100.0 + math.sin(i / 3.0) for i in range(130)]

    def propose_spread(self, symbol):
        return self._proposal

    def get_price_history(self, symbol, days=130):
        return self._closes[-days:]


class TestTelegramOutput(unittest.TestCase):
    def test_report_contains_required_fields(self):
        result = evaluate_for_symbol(_FakeTrader(bull_put()), "SPY")
        self.assertEqual(result.status, STATUS_OK)
        text = format_ev_report(result)
        for needle in ("EV ANALYSIS — SPY", "Strategy: Bull Put Credit Spread",
                       "Expected Value:", "Probability of Profit:",
                       "EV / Risk:", "Recommendation:", "Advisory analytics only"):
            self.assertIn(needle, text)
        # Recommendation must be one of the five tiers.
        self.assertIn(result.recommendation,
                      {STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE})

    def test_signed_ev_formatting(self):
        r = EVResult(symbol="SPY", strategy=BULLISH_PUT_CREDIT_SPREAD,
                     expected_value=18.4, probability_of_profit=0.71,
                     ev_per_dollar_risk=0.23, max_profit=45.0, max_loss=155.0,
                     estimated_costs=9.0, days=30, recommendation=STRONG_ACCEPT)
        text = format_ev_report(r)
        self.assertIn("Expected Value: +18.40", text)
        self.assertIn("Probability of Profit: 71%", text)
        self.assertIn("EV / Risk: 0.23", text)
        self.assertIn("Recommendation: STRONG_ACCEPT", text)

    def test_insufficient_report(self):
        r = EVResult(symbol="SPY", status=STATUS_INSUFFICIENT, reason="weak_vol_edge")
        text = format_ev_report(r)
        self.assertIn("No EV analysis for SPY", text)
        self.assertIn("weak_vol_edge", text)
        self.assertIn("nothing was traded", text)

    def test_no_trade_symbol_yields_insufficient(self):
        p = SpreadProposal(strategy_name=NO_TRADE, symbol="SPY", reason="missing_chain")
        result = evaluate_for_symbol(_FakeTrader(p), "SPY")
        self.assertEqual(result.status, STATUS_INSUFFICIENT)
        self.assertEqual(result.reason, "missing_chain")


# --------------------------------------------------------------------------- #
# 10. No execution path touched (static guards)
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    @staticmethod
    def _src(name):
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as f:
            return f.read()

    def test_scheduler_does_not_consume_ev_engine(self):
        self.assertNotIn("ev_engine", self._src("run_alpaca_intraday.py"))

    def test_trader_does_not_consume_ev_engine(self):
        self.assertNotIn("ev_engine", self._src("smart_trader.py"))

    def test_ev_engine_has_no_execution_or_network_coupling(self):
        src = self._src("ev_engine.py")
        self.assertNotIn("import smart_trader", src)
        self.assertNotIn("from smart_trader", src)
        self.assertNotIn("import requests", src)
        self.assertNotIn("place_order", src)
        self.assertNotIn("submit_order", src)

    def test_telegram_surface_exists_and_is_analytics_only(self):
        src = self._src("telegram_bot.py")
        self.assertIn("EV_ANALYSIS", src)          # the analytics command exists
        self.assertIn("def ev_analysis", src)      # with its handler


if __name__ == "__main__":
    unittest.main()
