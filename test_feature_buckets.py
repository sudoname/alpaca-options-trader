"""
Offline tests for Phase 13 — shared feature bucketing.

No creds, no network, no broker. Covers:
  1. Volatility / strength / dte / delta boundaries (half-open).
  2. Regime + direction normalization (incl. synonyms).
  3. Pattern as an OPTIONAL dimension (present-only-when-stamped).
  4. Legacy rows degrade to None dims; episode raw -> direction/volatility.
  5. Setup-key string / tuple stability and the none-token.

feature_buckets is pure: it only LABELS trades. It never reads files, never
touches the network, and never opens, sizes or alters any trade. Every function
is None-tolerant and returns a label (or None) without raising.
"""

import unittest

import feature_buckets as fb


class TestBoundaries(unittest.TestCase):
    def test_volatility_half_open(self):
        self.assertEqual(fb.volatility_bucket(0.10), "low")
        self.assertEqual(fb.volatility_bucket(0.149), "low")
        self.assertEqual(fb.volatility_bucket(0.15), "normal")
        self.assertEqual(fb.volatility_bucket(0.29), "normal")
        self.assertEqual(fb.volatility_bucket(0.30), "elevated")
        self.assertEqual(fb.volatility_bucket(0.49), "elevated")
        self.assertEqual(fb.volatility_bucket(0.50), "extreme")
        self.assertIsNone(fb.volatility_bucket(None))

    def test_strength(self):
        self.assertEqual(fb.strength_bucket(0), "weak")
        self.assertEqual(fb.strength_bucket(1), "medium")
        self.assertEqual(fb.strength_bucket(2), "medium")
        self.assertEqual(fb.strength_bucket(3), "strong")
        self.assertIsNone(fb.strength_bucket(None))

    def test_dte(self):
        self.assertEqual(fb.dte_bucket(7), "0-7")
        self.assertEqual(fb.dte_bucket(8), "8-21")
        self.assertEqual(fb.dte_bucket(21), "8-21")
        self.assertEqual(fb.dte_bucket(22), "22-45")
        self.assertEqual(fb.dte_bucket(45), "22-45")
        self.assertEqual(fb.dte_bucket(46), "45+")
        self.assertIsNone(fb.dte_bucket(None))

    def test_delta_uses_abs(self):
        self.assertEqual(fb.delta_bucket(0.29), "<0.30")
        self.assertEqual(fb.delta_bucket(0.30), "0.30-0.50")
        self.assertEqual(fb.delta_bucket(0.50), "0.30-0.50")
        self.assertEqual(fb.delta_bucket(0.51), ">0.50")
        self.assertEqual(fb.delta_bucket(-0.55), ">0.50")
        self.assertIsNone(fb.delta_bucket(None))


class TestNormalization(unittest.TestCase):
    def test_regime(self):
        self.assertEqual(fb.regime_bucket("TRENDING"), "trending")
        self.assertEqual(fb.regime_bucket("Volatile"), "volatile")
        self.assertIsNone(fb.regime_bucket("garbage"))
        self.assertIsNone(fb.regime_bucket(None))

    def test_direction_synonyms(self):
        self.assertEqual(fb.direction_bucket("up"), "up")
        self.assertEqual(fb.direction_bucket("bullish"), "up")
        self.assertEqual(fb.direction_bucket("bearish"), "down")
        self.assertEqual(fb.direction_bucket("neutral"), "flat")
        self.assertIsNone(fb.direction_bucket("none"))


class TestPatternOptional(unittest.TestCase):
    def test_pattern_absent_omitted(self):
        key = fb.make_setup_key({"regime": "trending", "dte": 30,
                                 "entry_delta": 0.4})
        self.assertNotIn("pattern", key)

    def test_pattern_present_included(self):
        key = fb.make_setup_key({"regime": "trending",
                                 "candlestick_pattern": "Hammer"})
        self.assertEqual(key.get("pattern"), "hammer")

    def test_neutral_pattern_treated_as_absent(self):
        self.assertIsNone(fb.pattern_bucket("neutral"))
        self.assertIsNone(fb.pattern_bucket(""))
        key = fb.make_setup_key({"candlestick_pattern": "neutral"})
        self.assertNotIn("pattern", key)


class TestExtraction(unittest.TestCase):
    def test_legacy_metrics_row(self):
        feats = fb.extract_features(
            {"metrics": {"signal_strength": 3, "entry_delta": 0.25}})
        self.assertEqual(feats["strength"], "strong")
        self.assertEqual(feats["delta_bucket"], "<0.30")
        # Legacy rows carry no regime/vol/direction.
        self.assertIsNone(feats["regime"])
        self.assertIsNone(feats["volatility"])

    def test_episode_raw_direction_and_vol(self):
        feats = fb.extract_features(
            {"features_json": {"raw": {"momentum": 0.08, "realized_vol": 0.4}}})
        self.assertEqual(feats["direction"], "up")
        self.assertEqual(feats["volatility"], "elevated")

    def test_episode_raw_as_json_string(self):
        feats = fb.extract_features(
            {"features_json": '{"raw": {"momentum": -0.09}}'})
        self.assertEqual(feats["direction"], "down")


class TestKeyRendering(unittest.TestCase):
    def test_key_str_none_token(self):
        ks = fb.setup_key_str({"regime": "trending", "volatility": None})
        self.assertEqual(ks, "regime=trending|volatility=none")

    def test_key_tuple_preserves_present_none_drops_absent(self):
        kt = fb.setup_key_tuple({"regime": "ranging", "direction": None})
        self.assertEqual(kt, (("regime", "ranging"), ("direction", None)))

    def test_empty_key_str(self):
        self.assertEqual(fb.setup_key_str({}), "(empty)")


class TestNeverRaises(unittest.TestCase):
    def test_garbage(self):
        for junk in (None, 42, "nonsense", [], {"x": object()}):
            fb.extract_features(junk)  # type: ignore[arg-type]
        # make_setup_key only takes dicts; ensure dict garbage is safe.
        fb.make_setup_key({"x": object()})


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(fb._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
