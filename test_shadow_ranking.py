"""
Offline tests for Phase 13D — the Shadow EV-portfolio replay.

No creds, no network, no broker. All records are injected. Covers:
  1. Deterministic replay: the same record set always yields the same picks.
  2. The three ranking systems pick the expected top-1 per decision set.
  3. Hand-checked aggregates (total P/L, win rate) per system.
  4. Singleton-set fallback for legacy rows with no batch/date.
  5. Empty / garbage input never raises.

shadow_ranking is ANALYTICS / OFFLINE: it never opens, sizes, prices, blocks or
alters any trade; ranking is a stable sort with deterministic tie-breakers.
"""

import unittest

import shadow_ranking as sr
import learned_edge as le
from shadow_ranking import (
    replay, group_decision_sets, RANK_ORACLE, RANK_BEST_EV, RANK_LEARNED,
)

LECFG = le.LearnedEdgeConfig()


def _two_day_records():
    day1 = [
        sr._cand("AAA", 90.0, 0.05, 3.0, -40.0, date="2025-01-02", rid="a1"),
        sr._cand("BBB", 60.0, 0.30, 18.0, +50.0, date="2025-01-02", rid="b1"),
    ]
    day2 = [
        sr._cand("CCC", 80.0, 0.25, 15.0, +30.0, date="2025-01-03", rid="c1"),
        sr._cand("DDD", 40.0, 0.02, 1.0, -20.0, date="2025-01-03", rid="d1"),
    ]
    return day1 + day2


class TestPicks(unittest.TestCase):
    def setUp(self):
        self.rep = replay(records=_two_day_records(), config=LECFG)

    def test_two_decision_sets(self):
        self.assertEqual(self.rep["num_decision_sets"], 2)

    def test_oracle_picks_high_score(self):
        picks = [c["symbol"] for c in self.rep["systems"][RANK_ORACLE]["choices"]]
        self.assertEqual(picks, ["AAA", "CCC"])

    def test_best_ev_picks_high_ev(self):
        picks = [c["symbol"]
                 for c in self.rep["systems"][RANK_BEST_EV]["choices"]]
        self.assertEqual(picks, ["BBB", "CCC"])

    def test_learned_picks_valid(self):
        picks = [c["symbol"]
                 for c in self.rep["systems"][RANK_LEARNED]["choices"]]
        self.assertEqual(len(picks), 2)
        for p in picks:
            self.assertIn(p, ("AAA", "BBB", "CCC", "DDD"))


class TestAggregates(unittest.TestCase):
    def setUp(self):
        self.rep = replay(records=_two_day_records(), config=LECFG)

    def test_oracle_aggregate(self):
        st = self.rep["systems"][RANK_ORACLE]["stats"]
        # AAA(-40) + CCC(+30) = -10, 1 win of 2.
        self.assertEqual(st["decisions"], 2)
        self.assertAlmostEqual(st["total_pnl"], -10.0, places=6)
        self.assertAlmostEqual(st["win_rate"], 0.5, places=6)

    def test_best_ev_aggregate(self):
        st = self.rep["systems"][RANK_BEST_EV]["stats"]
        # BBB(+50) + CCC(+30) = 80, 2 wins of 2.
        self.assertAlmostEqual(st["total_pnl"], 80.0, places=6)
        self.assertEqual(st["win_rate"], 1.0)


class TestDeterminism(unittest.TestCase):
    def test_repeated_replay_identical(self):
        recs = _two_day_records()
        r1 = replay(records=recs, config=LECFG)
        r2 = replay(records=list(recs), config=LECFG)
        for system in (RANK_ORACLE, RANK_BEST_EV, RANK_LEARNED):
            p1 = [c["symbol"] for c in r1["systems"][system]["choices"]]
            p2 = [c["symbol"] for c in r2["systems"][system]["choices"]]
            self.assertEqual(p1, p2)


class TestGrouping(unittest.TestCase):
    def test_group_by_date(self):
        sets = group_decision_sets(_two_day_records())
        self.assertEqual(len(sets), 2)
        self.assertEqual(len(sets[0]), 2)

    def test_singleton_fallback(self):
        legacy = [
            {"id": "x1", "symbol": "X", "oracle_score": 70.0, "pnl": 5.0,
             "max_loss": 100.0},
            {"id": "y1", "symbol": "Y", "oracle_score": 30.0, "pnl": -5.0,
             "max_loss": 100.0},
        ]
        sets = group_decision_sets(legacy)
        self.assertEqual(len(sets), 2)
        rep = replay(records=legacy, config=LECFG)
        self.assertEqual(rep["num_decision_sets"], 2)
        # Singleton sets -> every system picks the same trade -> equal totals.
        self.assertEqual(
            rep["systems"][RANK_ORACLE]["stats"]["total_pnl"],
            rep["systems"][RANK_BEST_EV]["stats"]["total_pnl"])


class TestNeverRaises(unittest.TestCase):
    def test_empty(self):
        rep = replay(records=[], config=LECFG)
        self.assertEqual(rep["num_decision_sets"], 0)

    def test_garbage(self):
        replay(records=[None, 42, "x", {"junk": 1}], config=LECFG)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(sr._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
