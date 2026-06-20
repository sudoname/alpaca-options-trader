"""
Offline tests for single_leg_reports — the read-only analytics over the
single-leg intraday bot's on-disk stores.

No creds, no network, no broker: each test points the loaders at temp files /
an in-temp SQLite store. Contract pinned here:

  1. FAIL-OPEN. Missing/corrupt sources -> verdict INSUFFICIENT_DATA, never raise.
  2. CORRECT AGGREGATES. Realized P/L (all-time + today), open/closed counts,
     win rate, and the episode action/outcome mix match the inputs.
  3. READ-ONLY PROVIDERS. The three ``compute_*`` provider functions contain no
     file-write or DB-mutation calls (writes are confined to the self-test).
"""

import json
import os
import tempfile
import unittest
from datetime import datetime

import single_leg_reports as slr


class TestKpis(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.today = datetime.now().date().isoformat()

    def _write(self, name, obj):
        p = os.path.join(self.d, name)
        with open(p, "w") as f:
            json.dump(obj, f)
        return p

    def test_empty_is_insufficient(self):
        k = slr.compute_single_leg_kpis(
            active_path=os.path.join(self.d, "no1.json"),
            history_path=os.path.join(self.d, "no2.json"),
            realized_path=os.path.join(self.d, "no3.json"))
        self.assertEqual(k["verdict"], "INSUFFICIENT_DATA")
        self.assertEqual(k["open_positions"], 0)
        self.assertEqual(k["closed_trades"], 0)

    def test_aggregates(self):
        active = self._write("a.json", [{"symbol": "X"}, {"symbol": "Y"}])
        hist = self._write("h.json", {"trades": [
            {"pnl_percent": 20.0}, {"pnl_percent": -10.0}, {"pnl_percent": 5.0}]})
        real = self._write("r.json", [
            {"date": self.today, "amount": 120.0},
            {"date": self.today, "amount": -30.0},
            {"date": "2000-01-01", "amount": 999.0}])
        k = slr.compute_single_leg_kpis(active_path=active, history_path=hist,
                                        realized_path=real, today=self.today)
        self.assertEqual(k["verdict"], "OK")
        self.assertEqual(k["open_positions"], 2)
        self.assertEqual(k["closed_trades"], 3)
        self.assertAlmostEqual(k["realized_total"], 1089.0)
        self.assertAlmostEqual(k["today_realized"], 90.0)
        self.assertEqual(k["wins"], 2)
        self.assertEqual(k["losses"], 1)
        self.assertAlmostEqual(k["win_rate"], 2.0 / 3.0)

    def test_corrupt_files_fail_open(self):
        p = os.path.join(self.d, "bad.json")
        with open(p, "w") as f:
            f.write("{not json")
        k = slr.compute_single_leg_kpis(active_path=p, history_path=p,
                                        realized_path=p)
        self.assertEqual(k["verdict"], "INSUFFICIENT_DATA")


class TestPositions(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_empty(self):
        out = slr.compute_single_leg_positions(
            active_path=os.path.join(self.d, "none.json"))
        self.assertEqual(out["verdict"], "INSUFFICIENT_DATA")
        self.assertEqual(out["positions"], [])

    def test_shape(self):
        p = os.path.join(self.d, "a.json")
        with open(p, "w") as f:
            json.dump([{"symbol": "SPY260101C00500000",
                        "underlying_symbol": "SPY", "quantity": 2,
                        "entry_price": 1.5, "entry_time": "2026-01-01T10:00:00",
                        "metrics": {"expected_value": 0.2,
                                    "probability_of_profit": 0.55}}], f)
        out = slr.compute_single_leg_positions(active_path=p)
        self.assertEqual(out["verdict"], "OK")
        self.assertEqual(out["count"], 1)
        row = out["positions"][0]
        self.assertEqual(row["underlying"], "SPY")
        self.assertEqual(row["quantity"], 2)
        self.assertAlmostEqual(row["expected_value"], 0.2)
        self.assertAlmostEqual(row["probability_of_profit"], 0.55)


class TestEpisodes(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()

    def test_missing_db_insufficient(self):
        out = slr.compute_single_leg_episodes(
            episode_db=os.path.join(self.d, "none.db"))
        self.assertEqual(out["verdict"], "INSUFFICIENT_DATA")

    def test_does_not_create_db(self):
        # Guarded so we never materialize an empty episodes.db on a box without one.
        path = os.path.join(self.d, "none.db")
        slr.compute_single_leg_episodes(episode_db=path)
        self.assertFalse(os.path.exists(path))

    def test_completed_episode_counts(self):
        from episode_store import EpisodeStore
        path = os.path.join(self.d, "episodes.db")
        store = EpisodeStore(path)
        did = store.log_decision(
            symbol="SPY", underlying="SPY", strat="intraday",
            features={"x": 1}, quote={"bid": 1.0, "ask": 1.1},
            modeled_cost=None, rule_action="CALL", rule_confidence=0.6,
            gate=None, risk=None, chosen_action="CALL", qty=1, mode="0DTE")
        store.record_outcome(decision_id=did, fill_price=1.0, exit_price=1.3,
                             gross_pnl_pct=30.0, net_pnl_pct=28.0,
                             net_pnl_dollars=28.0, hold_days=0,
                             outcome="take_profit")
        store.close()
        out = slr.compute_single_leg_episodes(episode_db=path)
        self.assertEqual(out["verdict"], "OK")
        self.assertEqual(out["stats"]["completed"], 1)
        self.assertEqual(out["chosen_action_counts"].get("CALL"), 1)
        self.assertEqual(out["outcome_counts"].get("take_profit"), 1)


class TestReadOnlyProviders(unittest.TestCase):
    def test_provider_bodies_have_no_writes(self):
        # The public providers must not mutate any store. (Loaders legitimately
        # open files in read mode; the self-test legitimately writes fixtures.)
        import inspect
        forbidden = ("log_decision(", "record_outcome(", "json.dump(",
                     ".commit(", "_save(", '"w"', "'w'")
        for fn in (slr.compute_single_leg_kpis,
                   slr.compute_single_leg_positions,
                   slr.compute_single_leg_episodes):
            src = inspect.getsource(fn)
            for tok in forbidden:
                self.assertNotIn(tok, src,
                                 msg=f"{fn.__name__} contains write token {tok!r}")


class TestSelfTest(unittest.TestCase):
    def test_self_test_passes(self):
        self.assertEqual(slr._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
