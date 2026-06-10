"""
Offline tests for Phase 10C — Advisory Gate consumes EV (soft signal only).

No creds, no network, no broker. Covers the eight required areas:
  1. Advisory behavior unchanged when EV missing
  2. Positive EV strengthens the recommendation
  3. Negative EV weakens the recommendation
  4. Strong negative EV prevents STRONG_ACCEPT
  5. EV fields appear in the output object (ev_summary / ev_checks)
  6. Telegram ADVISORY_CHECK EV section
  7. Attribution captures EV fields at open (never recomputed at close)
  8. No execution path touched (static guards)
"""

import os
import tempfile
import unittest
from types import SimpleNamespace

import advisory_attribution as aa
import advisory_gate as ag
from advisory_gate import (
    evaluate_setup, format_advisory_check, generate_advisory_check_text,
    ev_fields_from_result, ev_thresholds, ADVISORY_FOOTER,
    STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE,
)
from oracle_analytics import AnalyticsConfig

HERE = os.path.dirname(os.path.abspath(__file__))

# Explicit thresholds -> no env reads inside evaluate_setup during tests.
THR = {"min_ev": 0.00, "min_ev_per_risk": 0.05, "min_pop": 0.50}


def _empty_cfg():
    return AnalyticsConfig(spread_trades_file="/nonexistent/ev_t.json",
                           spread_positions_file="/nonexistent/ev_p.json",
                           expected_move_file="/nonexistent/ev_e.csv",
                           training_dataset_file="/nonexistent/ev_d.csv")


def _eval(**ev_kwargs):
    """evaluate_setup on empty data (base = NEUTRAL) with EV kwargs."""
    return evaluate_setup(config=_empty_cfg(), ev_thresholds_override=THR,
                          **ev_kwargs)


def _eval_strong_base(**ev_kwargs):
    """Deterministic STRONG_ACCEPT base: all thresholds undefined (pass) and
    enough history that _classify maps 5 passed checks to STRONG_ACCEPT."""
    return evaluate_setup(config=_empty_cfg(),
                          recommendations={"n_trades": 10}, trades=[],
                          ev_thresholds_override=THR, **ev_kwargs)


GOOD_EV = dict(expected_value=18.40, ev_per_dollar_risk=0.23,
               probability_of_profit=0.71, estimated_costs=5.0,
               ev_recommendation=STRONG_ACCEPT)


# --------------------------------------------------------------------------- #
# 1. Advisory behavior unchanged when EV missing
# --------------------------------------------------------------------------- #
class TestUnchangedWhenEVMissing(unittest.TestCase):
    def test_no_ev_kwargs_matches_pre_10c(self):
        res = _eval()
        self.assertEqual(res["recommendation"], res["base_recommendation"])
        self.assertFalse(res["ev_available"])
        self.assertEqual(res["ev_adjustment"], "none")
        self.assertEqual(res["ev_checks"], {})
        # Original check set untouched (EV checks are kept separate).
        self.assertEqual(set(res["checks"]),
                         {"oracle_score", "vol_edge", "dte", "iv_rank",
                          "strategy"})

    def test_explicit_none_ev_is_identical(self):
        base = _eval()
        with_none = _eval(expected_value=None, ev_per_dollar_risk=None,
                          probability_of_profit=None, estimated_costs=None,
                          ev_recommendation=None)
        self.assertEqual(with_none["recommendation"], base["recommendation"])
        self.assertEqual(with_none["confidence"], base["confidence"])
        self.assertEqual(with_none["checks"], base["checks"])

    def test_strong_base_unchanged_without_ev(self):
        res = _eval_strong_base()
        self.assertEqual(res["recommendation"], STRONG_ACCEPT)
        self.assertEqual(res["base_recommendation"], STRONG_ACCEPT)


# --------------------------------------------------------------------------- #
# 2. Positive EV strengthens
# --------------------------------------------------------------------------- #
class TestPositiveEVStrengthens(unittest.TestCase):
    def test_neutral_base_becomes_accept(self):
        res = _eval(**GOOD_EV)
        self.assertEqual(res["base_recommendation"], NEUTRAL)
        self.assertEqual(res["recommendation"], ACCEPT)
        self.assertEqual(res["ev_adjustment"], "strengthened")

    def test_strengthen_caps_at_strong_accept(self):
        res = _eval_strong_base(**GOOD_EV)
        self.assertEqual(res["recommendation"], STRONG_ACCEPT)

    def test_partial_pass_does_not_strengthen(self):
        # Positive EV but PoP below the floor -> soft signal stays quiet.
        res = _eval(expected_value=18.40, ev_per_dollar_risk=0.23,
                    probability_of_profit=0.30, estimated_costs=5.0)
        self.assertEqual(res["recommendation"], NEUTRAL)
        self.assertEqual(res["ev_adjustment"], "none")

    def test_unreasonable_costs_do_not_strengthen(self):
        # EV positive but costs eat most of the pre-cost edge.
        res = _eval(expected_value=10.0, ev_per_dollar_risk=0.10,
                    probability_of_profit=0.70, estimated_costs=100.0)
        self.assertFalse(res["ev_checks"]["costs_reasonable"])
        self.assertEqual(res["recommendation"], NEUTRAL)


# --------------------------------------------------------------------------- #
# 3. Negative EV weakens
# --------------------------------------------------------------------------- #
class TestNegativeEVWeakens(unittest.TestCase):
    def test_neutral_base_becomes_weak_setup(self):
        res = _eval(expected_value=-10.0, ev_per_dollar_risk=-0.02,
                    probability_of_profit=0.60)
        self.assertEqual(res["base_recommendation"], NEUTRAL)
        self.assertEqual(res["recommendation"], WEAK_SETUP)
        self.assertEqual(res["ev_adjustment"], "weakened")

    def test_only_one_tier_never_a_hard_gate(self):
        # Even a very negative EV moves the verdict by at most one tier.
        res = _eval_strong_base(expected_value=-500.0,
                                ev_per_dollar_risk=-0.90,
                                probability_of_profit=0.10)
        self.assertEqual(res["recommendation"], ACCEPT)
        self.assertNotEqual(res["recommendation"], REJECT_CANDIDATE)

    def test_zero_ev_neither_strengthens_nor_weakens(self):
        res = _eval(expected_value=0.0, ev_per_dollar_risk=0.0,
                    probability_of_profit=0.60)
        self.assertEqual(res["recommendation"], NEUTRAL)
        self.assertEqual(res["ev_adjustment"], "none")


# --------------------------------------------------------------------------- #
# 4. Strong negative EV prevents STRONG_ACCEPT
# --------------------------------------------------------------------------- #
class TestStrongNegativePreventsStrongAccept(unittest.TestCase):
    def test_strong_accept_base_is_demoted(self):
        res = _eval_strong_base(expected_value=-50.0,
                                ev_per_dollar_risk=-0.20,
                                probability_of_profit=0.40)
        self.assertEqual(res["base_recommendation"], STRONG_ACCEPT)
        self.assertNotEqual(res["recommendation"], STRONG_ACCEPT)

    def test_ev_reject_recommendation_also_demotes(self):
        res = _eval_strong_base(expected_value=-5.0,
                                ev_per_dollar_risk=-0.01,
                                ev_recommendation=REJECT_CANDIDATE)
        self.assertNotEqual(res["recommendation"], STRONG_ACCEPT)

    def test_invariant_no_base_yields_strong_accept(self):
        for base in (REJECT_CANDIDATE, WEAK_SETUP, NEUTRAL, ACCEPT,
                     STRONG_ACCEPT):
            adjusted, _ = ag._apply_ev_adjustment(
                base, -50.0, -0.20, REJECT_CANDIDATE, {})
            self.assertNotEqual(adjusted, STRONG_ACCEPT, f"base={base}")


# --------------------------------------------------------------------------- #
# 5. EV fields appear in the output object
# --------------------------------------------------------------------------- #
class TestOutputObject(unittest.TestCase):
    def test_ev_summary_fields(self):
        res = _eval(**GOOD_EV)
        self.assertEqual(res["ev_summary"], {
            "expected_value": 18.40,
            "ev_per_dollar_risk": 0.23,
            "probability_of_profit": 0.71,
            "estimated_costs": 5.0,
            "ev_recommendation": STRONG_ACCEPT,
        })
        self.assertTrue(res["ev_available"])

    def test_ev_checks_names_and_values(self):
        res = _eval(**GOOD_EV)
        self.assertEqual(set(res["ev_checks"]),
                         {"ev_positive", "ev_per_risk_ok", "pop_ok",
                          "costs_reasonable"})
        self.assertTrue(all(res["ev_checks"].values()))

    def test_ev_summary_present_even_without_ev(self):
        res = _eval()
        self.assertIn("ev_summary", res)
        self.assertTrue(all(v is None for v in res["ev_summary"].values()))

    def test_threshold_defaults_from_loader(self):
        class _Loader:
            def get_float(self, key, default):
                return default
        thr = ev_thresholds(loader=_Loader())
        self.assertEqual(thr, {"min_ev": 0.00, "min_ev_per_risk": 0.05,
                               "min_pop": 0.50})

    def test_costs_reasonable_fail_open_when_unknown(self):
        self.assertTrue(ag._costs_reasonable(10.0, None))
        self.assertTrue(ag._costs_reasonable(None, 50.0))
        self.assertTrue(ag._costs_reasonable(25.0, 5.0))     # 5 <= 0.5*30
        self.assertFalse(ag._costs_reasonable(10.0, 100.0))  # 100 > 0.5*110

    def test_ev_fields_from_result_object_dict_and_errors(self):
        ns = SimpleNamespace(status="ok", expected_value=18.4,
                             ev_per_dollar_risk=0.23,
                             probability_of_profit=0.71, estimated_costs=9.6,
                             recommendation=STRONG_ACCEPT)
        out = ev_fields_from_result(ns)
        self.assertEqual(out["expected_value"], 18.4)
        self.assertEqual(out["ev_recommendation"], STRONG_ACCEPT)
        out = ev_fields_from_result({"status": "ok", "expected_value": 1.0,
                                     "recommendation": ACCEPT})
        self.assertEqual(out["expected_value"], 1.0)
        # None / errored EVResult -> all None (fail-open)
        self.assertTrue(all(v is None
                            for v in ev_fields_from_result(None).values()))
        bad = ev_fields_from_result({"status": "insufficient_data",
                                     "expected_value": 99.0})
        self.assertTrue(all(v is None for v in bad.values()))


# --------------------------------------------------------------------------- #
# 6. Telegram EV section
# --------------------------------------------------------------------------- #
class TestTelegramEVSection(unittest.TestCase):
    FEATURES = {"oracle_score": 88, "volatility_edge": 0.04, "dte": 38,
                "iv_rank": 60, "strategy": "bullish_put_credit_spread"}

    def test_ev_section_rendered(self):
        res = _eval(**GOOD_EV)
        text = format_advisory_check("SPY", self.FEATURES, res)
        self.assertIn("*EV analysis:*", text)
        self.assertIn("Expected Value: +$18.40", text)
        self.assertIn("EV/Risk: 0.23", text)
        self.assertIn("Probability of Profit: 71%", text)
        self.assertIn(f"EV Recommendation: {STRONG_ACCEPT}", text)
        self.assertIn(ADVISORY_FOOTER, text)

    def test_negative_ev_formatting(self):
        res = _eval(expected_value=-12.5, ev_per_dollar_risk=-0.10,
                    probability_of_profit=0.40,
                    ev_recommendation=REJECT_CANDIDATE)
        text = format_advisory_check("SPY", self.FEATURES, res)
        self.assertIn("Expected Value: -$12.50", text)
        self.assertIn("Probability of Profit: 40%", text)

    def test_missing_ev_shows_na(self):
        text = format_advisory_check("SPY", self.FEATURES, _eval())
        self.assertIn("Expected Value: n/a", text)
        self.assertIn("EV Recommendation: n/a", text)
        self.assertIn(ADVISORY_FOOTER, text)

    def test_generate_text_with_ev_result(self):
        ev = SimpleNamespace(status="ok", expected_value=18.4,
                             ev_per_dollar_risk=0.23,
                             probability_of_profit=0.71, estimated_costs=9.6,
                             recommendation=STRONG_ACCEPT)
        text = generate_advisory_check_text("SPY", config=_empty_cfg(),
                                            ev_result=ev)
        self.assertIn("Advisory Check — SPY", text)
        self.assertIn("Expected Value: +$18.40", text)
        self.assertIn(ADVISORY_FOOTER, text)


# --------------------------------------------------------------------------- #
# 7. Attribution captures EV fields at open
# --------------------------------------------------------------------------- #
class TestAttributionCapturesEV(unittest.TestCase):
    def _pos(self, **kw):
        pos = {"id": "t1", "symbol": "SPY",
               "strategy": "bullish_put_credit_spread", "oracle_score": 85,
               "timestamp": "2026-06-09T10:00:00",
               "expected_value": 18.4, "ev_per_dollar_risk": 0.23,
               "probability_of_profit": 0.71,
               "ev_recommendation": STRONG_ACCEPT}
        pos.update(kw)
        return pos

    def _store(self):
        return os.path.join(tempfile.mkdtemp(), "advisory_attribution.json")

    def test_open_fields_include_ev(self):
        for f in ("expected_value", "ev_per_dollar_risk",
                  "probability_of_profit", "ev_recommendation"):
            self.assertIn(f, aa.OPEN_FIELDS)
            self.assertNotIn(f, aa.CLOSE_FIELDS)

    def test_snapshot_captures_ev_at_open(self):
        snap = aa.build_open_snapshot(self._pos(), config=_empty_cfg())
        self.assertEqual(snap["expected_value"], 18.4)
        self.assertEqual(snap["ev_per_dollar_risk"], 0.23)
        self.assertEqual(snap["probability_of_profit"], 0.71)
        self.assertEqual(snap["ev_recommendation"], STRONG_ACCEPT)
        self.assertIsNone(snap["pnl"])

    def test_snapshot_without_ev_is_fail_open(self):
        pos = {"id": "t2", "symbol": "SPY",
               "strategy": "bullish_put_credit_spread",
               "timestamp": "2026-06-09T10:00:00"}
        snap = aa.build_open_snapshot(pos, config=_empty_cfg())
        for f in ("expected_value", "ev_per_dollar_risk",
                  "probability_of_profit", "ev_recommendation"):
            self.assertIsNone(snap[f])
        self.assertIsNotNone(snap["advisory_recommendation"])

    def test_close_never_recomputes_ev(self):
        store = self._store()
        aa.record_open(self._pos(), config=_empty_cfg(), path=store)
        closed = aa.record_close({"id": "t1", "pnl": -40.0,
                                  "exit_reason": "stop_loss",
                                  "closed_at": "2026-06-10T15:00:00"},
                                 path=store)
        # Outcome appended ...
        self.assertEqual(closed["pnl"], -40.0)
        self.assertEqual(closed["win_loss"], "loss")
        # ... EV belief frozen exactly as captured at open.
        self.assertEqual(closed["expected_value"], 18.4)
        self.assertEqual(closed["ev_per_dollar_risk"], 0.23)
        self.assertEqual(closed["probability_of_profit"], 0.71)
        self.assertEqual(closed["ev_recommendation"], STRONG_ACCEPT)


# --------------------------------------------------------------------------- #
# 8. No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(HERE, name), encoding="utf-8") as f:
            return f.read()

    def test_live_modules_do_not_consume_gate_or_ev(self):
        for mod in ("run_alpaca_intraday.py", "smart_trader.py"):
            src = self._read(mod)
            self.assertNotIn("advisory_gate", src,
                             f"{mod} must not consume advisory_gate")
            self.assertNotIn("ev_engine", src,
                             f"{mod} must not consume ev_engine")

    def test_gate_has_no_execution_or_network(self):
        src = self._read("advisory_gate.py")
        for forbidden in ("import smart_trader", "from smart_trader",
                          "import ev_engine", "from ev_engine",
                          "import requests", "place_order", "submit_order"):
            self.assertNotIn(forbidden, src)

    def test_attribution_has_no_execution(self):
        src = self._read("advisory_attribution.py")
        for forbidden in ("import smart_trader", "from smart_trader",
                          "import ev_engine", "import requests",
                          "place_order", "submit_order"):
            self.assertNotIn(forbidden, src)

    def test_telegram_surface_exists(self):
        src = self._read("telegram_bot.py")
        self.assertIn("def advisory_check", src)
        self.assertIn("generate_advisory_check_text", src)


if __name__ == "__main__":
    unittest.main()
