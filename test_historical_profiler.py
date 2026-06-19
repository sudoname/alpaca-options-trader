"""
Offline tests for Phase 13C — the Historical Setup Profiler.

No creds, no network, no broker. All records are injected and the cache write
targets a temp file. Covers:
  1. Aggregation: per-setup blocks + a global roll-up with the right counts.
  2. Pattern dimension present only when the cohort is stamped.
  3. Atomic cache write + JSON readback round-trip, no stray temp file.
  4. Fail-open: a bad write target returns False and never raises.
  5. Stable setup-key strings.

historical_profiler is ANALYTICS / READ-ONLY: it never opens, sizes, prices,
blocks or alters any trade; the single cache write is fail-open.
"""

import os
import unittest

import historical_profiler as hp
import learned_edge as le
import oracle_analytics as oa
from historical_profiler import (
    ProfilerConfig, compute_profile, save_profile, SCHEMA_VERSION,
)


def _mixed_records():
    # 8 trending/up (no pattern) + 6 ranging/down (hammer pattern).
    return (
        [le._rec("trending", "up", 0.20, i % 4 != 0, rid=f"a{i}")
         for i in range(8)]
        + [le._rec("ranging", "down", 0.10, i % 2 == 0, rid=f"b{i}",
                   pattern="hammer") for i in range(6)]
    )


class TestAggregation(unittest.TestCase):
    def test_schema_and_global_count(self):
        prof = compute_profile(records=_mixed_records())
        self.assertEqual(prof["schema_version"], SCHEMA_VERSION)
        self.assertEqual(prof["global"]["count"], 14)
        self.assertTrue(prof["profiles"])
        self.assertIn("generated_at", prof)

    def test_block_fields(self):
        prof = compute_profile(records=_mixed_records())
        block = prof["global"]
        for field in ("count", "win_rate", "avg_pnl", "avg_pnl_pct", "avg_ev",
                      "avg_holding_time", "profit_factor", "max_loss_observed"):
            self.assertIn(field, block)

    def test_empty_records_clean(self):
        prof = compute_profile(records=[])
        self.assertEqual(prof["profiles"], {})
        self.assertEqual(prof["global"]["count"], 0)


class TestPatternDimension(unittest.TestCase):
    def test_pattern_present_only_when_stamped(self):
        prof = compute_profile(records=_mixed_records())
        keys = list(prof["profiles"])
        # The stamped cohort carries pattern=hammer.
        self.assertTrue(any("pattern=hammer" in ks for ks in keys))
        # The unstamped trending cohort omits the pattern dimension.
        self.assertTrue(any("pattern=" not in ks and "regime=trending" in ks
                            for ks in keys))


class TestCacheRoundTrip(unittest.TestCase):
    def test_atomic_write_and_readback(self):
        prof = compute_profile(records=_mixed_records())
        tmp = f"_test_profiler_{os.getpid()}.json"
        cfg = ProfilerConfig(profile_file=tmp)
        try:
            self.assertTrue(save_profile(prof, cfg))
            back = oa.read_json(tmp)
            self.assertIsInstance(back, dict)
            self.assertEqual(back["global"]["count"], 14)
            # No stray temp files left behind by the atomic write.
            self.assertFalse(any(f.startswith(f"{tmp}.tmp")
                                 for f in os.listdir(".")))
        finally:
            for f in (tmp,):
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except OSError:
                    pass

    def test_bad_target_fails_open(self):
        prof = compute_profile(records=_mixed_records())
        # Empty path -> reported failure, no exception.
        self.assertIs(save_profile(prof, ProfilerConfig(profile_file="")),
                      False)


class TestStableKeys(unittest.TestCase):
    def test_same_records_same_keys(self):
        recs = _mixed_records()
        p1 = compute_profile(records=recs)
        p2 = compute_profile(records=list(reversed(recs)))
        self.assertEqual(set(p1["profiles"]), set(p2["profiles"]))


class TestNeverRaises(unittest.TestCase):
    def test_garbage(self):
        compute_profile(records=[None, 42, "x", {"junk": object()}])  # type: ignore[list-item]


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(hp._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
