"""
Offline tests for Phase 13B — Oracle Score v2 blending.

No creds, no network, no broker. Covers:
  1. v1 and v2 weights each sum to 1.0.
  2. v2 manual arithmetic (neutral edge, perfect edge, partial sub-scores).
  3. Edge clamping and fail-open on garbage.
  4. Version parsing defaults to v1 and rejects unknown values.

oracle_score_v2 is pure arithmetic: no I/O, no network, never raises.
"""

import unittest

import oracle_score_v2 as v2
from oracle_score_v2 import V1, V2, V1_WEIGHTS, V2_WEIGHTS, blend_v1, blend_v2


class TestWeights(unittest.TestCase):
    def test_weights_sum_to_one(self):
        self.assertAlmostEqual(sum(V1_WEIGHTS.values()), 1.0, places=6)
        self.assertAlmostEqual(sum(V2_WEIGHTS.values()), 1.0, places=6)

    def test_v2_learned_edge_weight(self):
        self.assertEqual(V2_WEIGHTS["learned_edge"], 0.30)


class TestBlendV2(unittest.TestCase):
    def setUp(self):
        self.perfect = {"vol_edge": 1.0, "liquidity": 1.0, "risk_reward": 1.0,
                        "cost": 1.0, "trend_align": 1.0}

    def test_neutral_edge(self):
        # 0.70 sub-score weight at 1.0 + 0.30 * 0.5 edge = 0.85 -> 85.0.
        self.assertEqual(blend_v2(self.perfect, 0.5), 85.0)

    def test_perfect_edge(self):
        self.assertEqual(blend_v2(self.perfect, 1.0), 100.0)

    def test_manual_partial(self):
        subs = {"vol_edge": 0.8, "liquidity": 0.6, "risk_reward": 0.5,
                "cost": 0.4, "trend_align": 0.2}
        manual = (0.20 * 0.8 + 0.15 * 0.6 + 0.15 * 0.5 + 0.10 * 0.4
                  + 0.10 * 0.2 + 0.30 * 0.7) * 100.0
        self.assertEqual(blend_v2(subs, 0.7), round(manual, 1))

    def test_edge_clamped(self):
        self.assertEqual(blend_v2(self.perfect, 5.0), 100.0)
        self.assertEqual(blend_v2(self.perfect, -5.0), 70.0)

    def test_fail_open_on_garbage(self):
        self.assertEqual(blend_v2({}, None), 0.0)  # type: ignore[arg-type]


class TestBlendV1(unittest.TestCase):
    def test_v1_reference_perfect(self):
        perfect = {"vol_edge": 1.0, "liquidity": 1.0, "risk_reward": 1.0,
                   "cost": 1.0, "trend_align": 1.0}
        self.assertEqual(blend_v1(perfect), 100.0)


class TestVersionParsing(unittest.TestCase):
    class _Loader:
        def __init__(self, value):
            self._value = value

        def get_str(self, name, default=""):
            return self._value

    def test_parse_v2(self):
        self.assertEqual(v2.score_version_from_env(self._Loader("v2")), V2)

    def test_parse_v1_upper(self):
        self.assertEqual(v2.score_version_from_env(self._Loader("V1")), V1)

    def test_unknown_defaults_v1(self):
        self.assertEqual(v2.score_version_from_env(self._Loader("garbage")), V1)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(v2._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
