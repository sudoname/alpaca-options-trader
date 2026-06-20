"""
Offline tests for Oracle 3.0 — the Adaptive Feature Weighting head (oracle_weights).

No creds, no network, no broker. Every test injects synthetic closed records.
Covers:
  1. compute_weights never raises and returns uniform + INSUFFICIENT_DATA on
     empty / garbage / sub-threshold input.
  2. Every learned weight is clamped to [w_min, w_max] — no agent can dominate.
  3. A selectively-predictive agent out-weighs a never-convicting (neutral) one;
     a neutral agent keeps a positive weight.
  4. Results are deterministic; weights always cover the full agent roster.
  5. Persistence round-trips through a temp file (current + history) and drift is
     computable over >=2 snapshots.

oracle_weights is SHADOW credit-assignment over agents: it never opens, sizes,
prices, blocks or alters any trade, and never learns a trade direction.
"""

import os
import tempfile
import unittest
import uuid

import oracle_weights as ow
from oracle_agents import AGENT_NAMES


class TestComputeWeights(unittest.TestCase):
    def test_empty_is_uniform_insufficient(self):
        r = ow.compute_weights([])
        self.assertEqual(r["verdict"], "INSUFFICIENT_DATA")
        self.assertEqual(set(r["weights"]), set(AGENT_NAMES))
        self.assertTrue(all(abs(w - 1.0) < 1e-9 for w in r["weights"].values()))

    def test_subthreshold_is_uniform(self):
        recs = ow._synthetic_records(ow.MIN_SAMPLES - 1)
        r = ow.compute_weights(recs)
        self.assertEqual(r["verdict"], "INSUFFICIENT_DATA")
        self.assertTrue(all(abs(w - 1.0) < 1e-9 for w in r["weights"].values()))

    def test_sufficient_is_ok_and_bounded(self):
        cfg = ow.OracleWeightsConfig()
        r = ow.compute_weights(ow._synthetic_records(40), cfg)
        self.assertEqual(r["verdict"], "OK")
        self.assertEqual(set(r["weights"]), set(AGENT_NAMES))
        for w in r["weights"].values():
            self.assertGreaterEqual(w, cfg.w_min - 1e-9)
            self.assertLessEqual(w, cfg.w_max + 1e-9)

    def test_predictive_agent_outweighs_neutral(self):
        r = ow.compute_weights(ow._synthetic_records(40))
        self.assertGreater(r["weights"]["trend"], r["weights"]["liquidity"])
        self.assertGreater(r["weights"]["liquidity"], 0.0)

    def test_deterministic(self):
        a = ow.compute_weights(ow._synthetic_records(40))["weights"]
        b = ow.compute_weights(ow._synthetic_records(40))["weights"]
        self.assertEqual(a, b)


class TestFailOpen(unittest.TestCase):
    def test_garbage_never_raises(self):
        for junk in (None, 42, "x", [None, 42], [{"weird": object()}]):
            r = ow.compute_weights(junk)  # type: ignore[arg-type]
            self.assertEqual(set(r["weights"]), set(AGENT_NAMES))
            self.assertEqual(r["verdict"], "INSUFFICIENT_DATA")


class TestPersistence(unittest.TestCase):
    def setUp(self):
        self.tmp = os.path.join(tempfile.gettempdir(),
                                f"ow_test_{uuid.uuid4().hex}.json")
        self.cfg = ow.OracleWeightsConfig(weights_file=self.tmp)

    def tearDown(self):
        try:
            os.remove(self.tmp)
        except OSError:
            pass

    def test_missing_file_reads_uniform(self):
        self.assertEqual(ow.current_weights(self.cfg), ow.uniform_weights())
        self.assertEqual(ow.weight_history(self.cfg), [])

    def test_save_and_reload(self):
        res = ow.compute_weights(ow._synthetic_records(40), self.cfg)
        self.assertTrue(ow.save_weights(res, self.cfg))
        self.assertEqual(ow.current_weights(self.cfg), res["weights"])
        self.assertEqual(len(ow.weight_history(self.cfg)), 1)

    def test_history_appends_and_drift(self):
        ow.save_weights(ow.compute_weights(ow._synthetic_records(40), self.cfg),
                        self.cfg)
        ow.save_weights(ow.compute_weights(ow._synthetic_records(60), self.cfg),
                        self.cfg)
        hist = ow.weight_history(self.cfg)
        self.assertEqual(len(hist), 2)
        self.assertIsNotNone(ow.weight_drift(hist))
        self.assertIsNone(ow.weight_drift(hist[:1]))

    def test_update_weights_persists_only_on_ok(self):
        # sub-threshold -> INSUFFICIENT -> not persisted.
        ow.update_weights(ow._synthetic_records(3), self.cfg)
        self.assertEqual(ow.weight_history(self.cfg), [])
        # sufficient -> persisted.
        ow.update_weights(ow._synthetic_records(40), self.cfg)
        self.assertEqual(len(ow.weight_history(self.cfg)), 1)


class TestDrift(unittest.TestCase):
    def test_none_on_short_history(self):
        self.assertIsNone(ow.weight_drift(None))
        self.assertIsNone(ow.weight_drift([]))
        self.assertIsNone(ow.weight_drift([{"weights": {"trend": 1.0}}]))

    def test_sum_abs_diff(self):
        hist = [{"weights": {"a": 1.0, "b": 1.0}},
                {"weights": {"a": 1.5, "b": 0.5}}]
        self.assertAlmostEqual(ow.weight_drift(hist), 1.0, places=6)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(ow._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
