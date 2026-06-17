"""
Offline tests for Phase 11A-1 — Triple Gap (pure, no I/O, no network).

Covers:
  - raw gap math + signs (vol / move / ev)
  - zero-baseline vs model-baseline EV tagging
  - component normalisation, clamping and the EV floor
  - move-gap scaling by the oracle move (scale-free) vs fixed fallback
  - weighted combination + renormalisation over present components
  - custom weights from config
  - missing inputs -> insufficient_data
  - to_dict round-trips every field
"""

import unittest

import triple_gap as tg
from triple_gap import (
    TripleGapConfig, compute_triple_gap,
    normalize_vol_gap, normalize_move_gap, normalize_ev_gap,
    STATUS_OK, STATUS_INSUFFICIENT,
    EV_GAP_ZERO_BASELINE, EV_GAP_MODEL_BASELINE,
)


class TestRawGaps(unittest.TestCase):
    def test_signed_gaps_and_status(self):
        r = compute_triple_gap(
            symbol="spy", strategy="debit_call_spread",
            market_iv=0.25, forecast_vol=0.20,
            market_expected_move=12.0, oracle_expected_move=10.0,
            oracle_expected_value=25.0)
        self.assertAlmostEqual(r.vol_gap, 0.05)
        self.assertAlmostEqual(r.move_gap, 2.0)
        self.assertAlmostEqual(r.ev_gap, 25.0)
        self.assertEqual(r.symbol, "spy")
        self.assertEqual(r.status, STATUS_OK)

    def test_negative_gaps(self):
        r = compute_triple_gap(market_iv=0.20, forecast_vol=0.30,
                               market_expected_move=8.0,
                               oracle_expected_move=10.0)
        self.assertAlmostEqual(r.vol_gap, -0.10)
        self.assertAlmostEqual(r.move_gap, -2.0)


class TestEvBaselineTag(unittest.TestCase):
    def test_zero_baseline_when_market_neutral_missing(self):
        r = compute_triple_gap(oracle_expected_value=25.0)
        self.assertEqual(r.ev_gap, 25.0)
        self.assertEqual(r.ev_gap_source, EV_GAP_ZERO_BASELINE)

    def test_model_baseline_when_supplied(self):
        r = compute_triple_gap(oracle_expected_value=20.0,
                               market_neutral_expected_value=5.0)
        self.assertEqual(r.ev_gap, 15.0)
        self.assertEqual(r.ev_gap_source, EV_GAP_MODEL_BASELINE)


class TestNormalisation(unittest.TestCase):
    def test_vol_score(self):
        self.assertEqual(normalize_vol_gap(0.05), 50.0)   # 0.05 / 0.10
        self.assertEqual(normalize_vol_gap(-0.05), 50.0)  # magnitude
        self.assertEqual(normalize_vol_gap(0.20), 100.0)  # clamps
        self.assertIsNone(normalize_vol_gap(None))

    def test_move_score_scale_free(self):
        # |2| / oracle move 10 -> 20.
        self.assertEqual(normalize_move_gap(2.0, 10.0), 20.0)
        # clamps when gap exceeds the oracle move.
        self.assertEqual(normalize_move_gap(15.0, 10.0), 100.0)

    def test_move_score_fixed_fallback(self):
        # No oracle move -> fixed MOVE_GAP_FULL_SCALE (=1.0) scaling.
        self.assertEqual(normalize_move_gap(0.5, None), 50.0)
        self.assertEqual(normalize_move_gap(0.5, 0.0), 50.0)

    def test_ev_score_floor_and_clamp(self):
        self.assertEqual(normalize_ev_gap(25.0), 50.0)    # 25 / 50
        self.assertEqual(normalize_ev_gap(100.0), 100.0)  # clamps
        self.assertEqual(normalize_ev_gap(-30.0), 0.0)    # floored
        self.assertIsNone(normalize_ev_gap(None))


class TestWeightedCombination(unittest.TestCase):
    def test_default_weights(self):
        r = compute_triple_gap(market_iv=0.25, forecast_vol=0.20,
                               market_expected_move=12.0,
                               oracle_expected_move=10.0,
                               oracle_expected_value=25.0)
        # vol 50, move 20, ev 50; (.3*50+.3*20+.4*50)/1.0 = 41.0
        self.assertEqual(r.vol_gap_score, 50.0)
        self.assertEqual(r.move_gap_score, 20.0)
        self.assertEqual(r.ev_gap_score, 50.0)
        self.assertEqual(r.triple_gap_score, 41.0)

    def test_single_component_renormalises(self):
        r = compute_triple_gap(market_iv=0.30, forecast_vol=0.20)
        self.assertIsNone(r.move_gap_score)
        self.assertIsNone(r.ev_gap_score)
        self.assertEqual(r.triple_gap_score, r.vol_gap_score)

    def test_two_components_renormalise(self):
        # vol 100 (0.10/0.10), ev 50; (.3*100+.4*50)/0.7 = 71.4
        r = compute_triple_gap(market_iv=0.30, forecast_vol=0.20,
                               oracle_expected_value=25.0)
        self.assertEqual(r.vol_gap_score, 100.0)
        self.assertEqual(r.triple_gap_score, 71.4)


class TestConfig(unittest.TestCase):
    def test_custom_weights(self):
        cfg = TripleGapConfig(vol_weight=1.0, move_weight=0.0, ev_weight=1.0)
        # vol 100, move present(20) weight 0, ev 50 -> (1*100+0*20+1*50)/2 = 75
        r = compute_triple_gap(market_iv=0.30, forecast_vol=0.20,
                               market_expected_move=12.0,
                               oracle_expected_move=10.0,
                               oracle_expected_value=25.0, config=cfg)
        self.assertEqual(r.triple_gap_score, 75.0)

    def test_from_env_fail_open(self):
        # Bad path -> defaults, never raises.
        cfg = TripleGapConfig.from_env(path="/nonexistent/.env")
        self.assertIsInstance(cfg, TripleGapConfig)
        self.assertEqual(cfg.vol_weight, 0.30)


class TestInsufficientAndDict(unittest.TestCase):
    def test_no_inputs_insufficient(self):
        r = compute_triple_gap()
        self.assertEqual(r.status, STATUS_INSUFFICIENT)
        self.assertIsNone(r.triple_gap_score)

    def test_non_numeric_inputs_drop_out(self):
        r = compute_triple_gap(market_iv="n/a", forecast_vol=None,
                               oracle_expected_value="bad")
        self.assertEqual(r.status, STATUS_INSUFFICIENT)

    def test_to_dict_has_all_fields(self):
        r = compute_triple_gap(market_iv=0.25, forecast_vol=0.20)
        d = r.to_dict()
        for key in ("symbol", "strategy", "vol_gap", "move_gap", "ev_gap",
                    "vol_gap_score", "move_gap_score", "ev_gap_score",
                    "triple_gap_score", "ev_gap_source", "status", "reason"):
            self.assertIn(key, d)

    def test_self_test_passes(self):
        self.assertEqual(tg._self_test(), 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
