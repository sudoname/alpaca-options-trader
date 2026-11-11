"""Tests for pricing_diagnostics (Phase 5c of the expectancy review).

Covered requirements:
* Quote taxonomy reuses the Phase-1 unusable-quote vocabulary
  (nonpositive_quote | one_sided_quote | crossed_quote | stale_quote).
* Black-Scholes price/delta/vega sanity and degenerate inputs.
* IV solving: round-trip recovery, no_price / expired / below_intrinsic /
  above_bound statuses, bid/mid/ask diagnostics gated on quote usability.
* Put-call parity: European residual ~ 0; American flagged with a caveat,
  never asserted.
* Model-implied probability is labeled risk-neutral and explicitly distinct
  from empirical probability of profit.
* Exercise-appropriate reference pricing: CRR American put >= European BSM
  put; engine name recorded; QuantLib strictly optional.

No credentials, no network.
"""
import math
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pricing_diagnostics as pd  # noqa: E402


class TestClassifyQuote(unittest.TestCase):
    def test_ok(self):
        self.assertEqual(pd.classify_quote(0.99, 1.01), "ok")

    def test_nonpositive_when_both_sides_bad(self):
        self.assertEqual(pd.classify_quote(0.0, 0.0), "nonpositive_quote")
        self.assertEqual(pd.classify_quote(None, None), "nonpositive_quote")
        self.assertEqual(pd.classify_quote(-0.5, 0.0), "nonpositive_quote")

    def test_one_sided(self):
        self.assertEqual(pd.classify_quote(None, 1.01), "one_sided_quote")
        self.assertEqual(pd.classify_quote(0.99, None), "one_sided_quote")
        self.assertEqual(pd.classify_quote(0.0, 1.01), "one_sided_quote")
        self.assertEqual(pd.classify_quote(0.99, 0.0), "one_sided_quote")

    def test_crossed(self):
        self.assertEqual(pd.classify_quote(1.05, 1.01), "crossed_quote")

    def test_stale(self):
        self.assertEqual(
            pd.classify_quote(0.99, 1.01, quote_age_s=120.0, max_age_s=60.0),
            "stale_quote")
        self.assertEqual(
            pd.classify_quote(0.99, 1.01, quote_age_s=30.0, max_age_s=60.0),
            "ok")

    def test_bool_is_not_a_price(self):
        self.assertEqual(pd.classify_quote(True, 1.01), "one_sided_quote")

    def test_priority_bad_sides_before_stale(self):
        self.assertEqual(
            pd.classify_quote(0.0, 0.0, quote_age_s=999.0),
            "nonpositive_quote")
        self.assertEqual(
            pd.classify_quote(0.0, 1.01, quote_age_s=999.0),
            "one_sided_quote")


class TestBlackScholes(unittest.TestCase):
    S, K, T, R, Q = 450.0, 455.0, 30.0 / 365.0, 0.05, 0.013

    def test_call_put_positive(self):
        c = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "call")
        p = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "put")
        self.assertGreater(c, 0.0)
        self.assertGreater(p, 0.0)

    def test_expired_is_intrinsic(self):
        self.assertAlmostEqual(
            pd.bs_price(460.0, 455.0, 0.0, 0.05, 0.0, 0.22, "call"), 5.0)
        self.assertAlmostEqual(
            pd.bs_price(460.0, 455.0, 0.0, 0.05, 0.0, 0.22, "put"), 0.0)

    def test_zero_vol_discounted_forward_intrinsic(self):
        v = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.0, "call")
        expected = max(0.0, self.S * math.exp(-self.Q * self.T)
                       - self.K * math.exp(-self.R * self.T))
        self.assertAlmostEqual(v, expected, places=10)

    def test_bad_inputs_raise(self):
        with self.assertRaises(ValueError):
            pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "straddle")
        with self.assertRaises(ValueError):
            pd.bs_price(-1.0, self.K, self.T, self.R, self.Q, 0.22, "call")

    def test_delta_bounds(self):
        dc = pd.bs_delta(self.S, self.K, self.T, self.R, self.Q, 0.22, "call")
        dp = pd.bs_delta(self.S, self.K, self.T, self.R, self.Q, 0.22, "put")
        self.assertTrue(0.0 < dc < 1.0)
        self.assertTrue(-1.0 < dp < 0.0)

    def test_vega_positive(self):
        self.assertGreater(
            pd.bs_vega(self.S, self.K, self.T, self.R, self.Q, 0.22), 0.0)

    def test_monotone_in_vol(self):
        lo = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.10, "call")
        hi = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.40, "call")
        self.assertLess(lo, hi)


class TestSolveIV(unittest.TestCase):
    S, K, T, R, Q = 450.0, 455.0, 30.0 / 365.0, 0.05, 0.013

    def test_round_trip(self):
        for sigma in (0.12, 0.22, 0.55):
            for opt in ("call", "put"):
                price = pd.bs_price(self.S, self.K, self.T, self.R, self.Q,
                                    sigma, opt)
                res = pd.solve_iv(price, self.S, self.K, self.T, self.R,
                                  self.Q, opt)
                self.assertEqual(res["status"], "ok", res)
                self.assertAlmostEqual(res["iv"], sigma, places=4)

    def test_no_price(self):
        self.assertEqual(
            pd.solve_iv(None, self.S, self.K, self.T, self.R, self.Q,
                        "call")["status"], "no_price")
        self.assertEqual(
            pd.solve_iv(0.0, self.S, self.K, self.T, self.R, self.Q,
                        "call")["status"], "no_price")

    def test_expired(self):
        self.assertEqual(
            pd.solve_iv(1.0, self.S, self.K, 0.0, self.R, self.Q,
                        "call")["status"], "expired")

    def test_below_intrinsic(self):
        # Deep ITM call priced below its zero-vol lower bound.
        res = pd.solve_iv(1.0, 500.0, 455.0, self.T, self.R, self.Q, "call")
        self.assertEqual(res["status"], "below_intrinsic")
        self.assertIsNone(res["iv"])

    def test_above_bound(self):
        res = pd.solve_iv(self.S * 2.0, self.S, self.K, self.T, self.R,
                          self.Q, "call")
        self.assertEqual(res["status"], "above_bound")
        self.assertIsNone(res["iv"])


class TestQuoteIVDiagnostics(unittest.TestCase):
    S, K, T, R, Q = 450.0, 455.0, 30.0 / 365.0, 0.05, 0.013

    def _mk(self, bid, ask, **kw):
        return pd.quote_iv_diagnostics(bid=bid, ask=ask, S=self.S, K=self.K,
                                       T=self.T, r=self.R, q=self.Q,
                                       opt_type="call", **kw)

    def test_usable_quote_solves_three_sides(self):
        bid = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.21, "call")
        ask = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.23, "call")
        d = self._mk(bid, ask)
        self.assertEqual(d["quote_status"], "ok")
        self.assertAlmostEqual(d["iv_bid"], 0.21, places=3)
        self.assertAlmostEqual(d["iv_ask"], 0.23, places=3)
        self.assertTrue(d["iv_bid"] < d["iv_mid"] < d["iv_ask"])

    def test_unusable_quote_yields_no_ivs(self):
        d = self._mk(1.05, 1.01)  # crossed
        self.assertEqual(d["quote_status"], "crossed_quote")
        self.assertIsNone(d["iv_bid"])
        self.assertIsNone(d["iv_mid"])
        self.assertIsNone(d["iv_ask"])

    def test_stale_quote_blocked(self):
        d = self._mk(0.99, 1.01, quote_age_s=600.0, max_age_s=60.0)
        self.assertEqual(d["quote_status"], "stale_quote")
        self.assertIsNone(d["iv_mid"])

    def test_reasons_recorded_per_side(self):
        bid = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.21, "call")
        d = self._mk(bid, self.S * 3.0)  # ask absurdly above no-arb bound
        self.assertEqual(d["quote_status"], "ok")
        self.assertEqual(d["reasons"]["ask"], "above_bound")
        self.assertIsNone(d["iv_ask"])
        self.assertIsNotNone(d["iv_bid"])


class TestPutCallParity(unittest.TestCase):
    S, K, T, R, Q = 450.0, 455.0, 30.0 / 365.0, 0.05, 0.013

    def test_european_consistent_mids(self):
        c = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "call")
        p = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "put")
        res = pd.put_call_parity_check(call_mid=c, put_mid=p, S=self.S,
                                       K=self.K, T=self.T, r=self.R, q=self.Q,
                                       exercise="european")
        self.assertEqual(res["status"], "ok")
        self.assertAlmostEqual(res["residual"], 0.0, places=8)
        self.assertTrue(res["within_tolerance"])

    def test_violation_detected(self):
        res = pd.put_call_parity_check(call_mid=10.0, put_mid=1.0, S=self.S,
                                       K=self.K, T=self.T, r=self.R, q=self.Q,
                                       exercise="european", tolerance=0.25)
        self.assertEqual(res["status"], "ok")
        self.assertFalse(res["within_tolerance"])

    def test_american_carries_caveat(self):
        c = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "call")
        p = pd.bs_price(self.S, self.K, self.T, self.R, self.Q, 0.22, "put")
        res = pd.put_call_parity_check(call_mid=c, put_mid=p, S=self.S,
                                       K=self.K, T=self.T, r=self.R, q=self.Q,
                                       exercise="american")
        self.assertEqual(res["exercise"], "american")
        self.assertTrue(res["caveat"])  # diagnostic only, never an assertion

    def test_missing_mid_unavailable(self):
        res = pd.put_call_parity_check(call_mid=None, put_mid=1.0, S=self.S,
                                       K=self.K, T=self.T)
        self.assertEqual(res["status"], "unavailable")


class TestModelProbabilityLabeling(unittest.TestCase):
    def test_risk_neutral_label_and_separation(self):
        res = pd.model_itm_prob(S=450.0, K=455.0, T=30.0 / 365.0, r=0.05,
                                q=0.013, sigma=0.22, opt_type="call")
        self.assertEqual(res["probability_kind"], "model_implied_risk_neutral")
        self.assertEqual(res["not_equal_to"],
                         "empirical_probability_of_profit")
        self.assertTrue(0.0 < res["probability"] < 1.0)

    def test_call_put_probs_sum_to_one(self):
        kw = dict(S=450.0, K=455.0, T=30.0 / 365.0, r=0.05, q=0.013,
                  sigma=0.22)
        pc = pd.model_itm_prob(opt_type="call", **kw)["probability"]
        pp = pd.model_itm_prob(opt_type="put", **kw)["probability"]
        self.assertAlmostEqual(pc + pp, 1.0, places=10)


class TestReferencePricing(unittest.TestCase):
    S, K, T, R, Q = 450.0, 455.0, 90.0 / 365.0, 0.05, 0.013

    def test_american_put_premium_over_european(self):
        # High rate amplifies the early-exercise premium on puts.
        amer = pd.crr_american_price(100.0, 110.0, 1.0, 0.08, 0.0, 0.25,
                                     "put", steps=400)
        euro = pd.bs_price(100.0, 110.0, 1.0, 0.08, 0.0, 0.25, "put")
        self.assertGreater(amer, euro)

    def test_crr_converges_to_bsm_for_calls_no_dividend(self):
        # American call, q=0: never optimal to exercise early -> ~ BSM.
        amer = pd.crr_american_price(self.S, self.K, self.T, self.R, 0.0,
                                     0.22, "call", steps=1000)
        euro = pd.bs_price(self.S, self.K, self.T, self.R, 0.0, 0.22, "call")
        self.assertAlmostEqual(amer, euro, delta=0.05)

    def test_reference_price_engine_recorded(self):
        res_a = pd.reference_price(S=self.S, K=self.K, T=self.T, r=self.R,
                                   q=self.Q, sigma=0.22, opt_type="put",
                                   exercise="american", steps=200)
        res_e = pd.reference_price(S=self.S, K=self.K, T=self.T, r=self.R,
                                   q=self.Q, sigma=0.22, opt_type="put",
                                   exercise="european")
        self.assertIn(res_a["engine"], ("crr_binomial", "quantlib_crr"))
        self.assertIn(res_e["engine"], ("black_scholes", "quantlib_analytic"))
        self.assertGreaterEqual(res_a["price"], res_e["price"] - 1e-6)

    def test_quantlib_strictly_optional(self):
        # Module imports and prices without QuantLib installed.
        self.assertIsInstance(pd._HAS_QUANTLIB, bool)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(pd._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
