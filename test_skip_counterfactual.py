"""
Tests for skip_counterfactual: the forward-underlying-return resolver that gives
SKIP episodes a learnable counterfactual outcome.

No network: a tmp in-memory EpisodeStore + a stub price_fn drive everything.
"""

import unittest
from datetime import datetime, timedelta

from episode_store import EpisodeStore
from skip_counterfactual import counterfactual_return, resolve_due_skips


class TestCounterfactualReturn(unittest.TestCase):
    """Value == the would-be position's forward return (the foregone trade)."""

    def test_call_sign(self):
        # The foregone CALL would gain when price rises, lose when it falls.
        self.assertGreater(counterfactual_return("CALL", 100.0, 105.0), 0)
        self.assertLess(counterfactual_return("CALL", 100.0, 95.0), 0)

    def test_put_inverted(self):
        # The foregone PUT would gain when price falls, lose when it rises.
        self.assertGreater(counterfactual_return("PUT", 100.0, 95.0), 0)
        self.assertLess(counterfactual_return("PUT", 100.0, 105.0), 0)

    def test_magnitude(self):
        # CALL, price 100->90 (fell): foregone call would have lost => -10%.
        self.assertAlmostEqual(counterfactual_return("CALL", 100.0, 90.0), -10.0)
        # PUT, price 100->90 (fell): foregone put would have gained => +10%.
        self.assertAlmostEqual(counterfactual_return("PUT", 100.0, 90.0), 10.0)

    def test_case_insensitive(self):
        # CALL, price 100->110 (rose): foregone call would have gained => +10%.
        self.assertAlmostEqual(counterfactual_return("call", 100.0, 110.0), 10.0)

    def test_bad_input_returns_none(self):
        self.assertIsNone(counterfactual_return("CALL", None, 105.0))
        self.assertIsNone(counterfactual_return("CALL", 0.0, 105.0))
        self.assertIsNone(counterfactual_return("CALL", -5.0, 105.0))
        self.assertIsNone(counterfactual_return("CALL", 100.0, None))
        self.assertIsNone(counterfactual_return("CALL", 100.0, 0.0))
        self.assertIsNone(counterfactual_return("HOLD", 100.0, 105.0))
        self.assertIsNone(counterfactual_return("", 100.0, 105.0))


class TestResolveDueSkips(unittest.TestCase):
    def setUp(self):
        self.store = EpisodeStore(":memory:")

    def tearDown(self):
        self.store.close()

    def _log_skip(self, underlying, direction, entry_px, age_min, *, raw=None):
        feats = {"raw": raw if raw is not None else {"underlying_price": entry_px},
                 "state_key": "k"}
        did = self.store.log_decision(
            symbol=f"{underlying}260101C00500000", underlying=underlying, strat="t",
            features=feats, quote=None, modeled_cost=None, rule_action=direction,
            rule_confidence=0.0, gate=None, chosen_action="SKIP", qty=1,
            mode="live-paper-blocked")
        # Backdate created_at by age_min minutes.
        self.store.conn.execute(
            "UPDATE episodes SET created_at=? WHERE decision_id=?",
            ((datetime.now() - timedelta(minutes=age_min)).isoformat(), did))
        self.store.conn.commit()
        return did

    def _row(self, did):
        rows = self.store._rows("SELECT * FROM episodes WHERE decision_id=?", (did,))
        return rows[0] if rows else None

    def test_due_skip_resolves_with_correct_sign(self):
        did = self._log_skip("SPY", "CALL", 100.0, age_min=500)
        n = resolve_due_skips(self.store, lambda s: 90.0, horizon_min=390)
        self.assertEqual(n, 1)
        row = self._row(did)
        self.assertEqual(row["outcome"], "skip_resolved")
        # Foregone CALL, price 100->90: would have lost => -10%.
        self.assertAlmostEqual(row["net_pnl_pct"], -10.0)
        self.assertAlmostEqual(row["gross_pnl_pct"], -10.0)
        self.assertEqual(row["exit_price"], 90.0)

    def test_too_recent_skip_left_open(self):
        did = self._log_skip("QQQ", "PUT", 100.0, age_min=10)
        n = resolve_due_skips(self.store, lambda s: 90.0, horizon_min=390)
        self.assertEqual(n, 0)
        self.assertIsNone(self._row(did)["outcome"])

    def test_non_skip_open_rows_untouched(self):
        # A real CALL decision still open must not be resolved by this path.
        did = self.store.log_decision(
            symbol="IWM260101C00500000", underlying="IWM", strat="t",
            features={"raw": {"underlying_price": 100.0}, "state_key": "k"},
            quote=None, modeled_cost=None, rule_action="CALL", rule_confidence=0.0,
            gate=None, chosen_action="CALL", qty=1, mode="1DTE")
        self.store.conn.execute(
            "UPDATE episodes SET created_at=? WHERE decision_id=?",
            ((datetime.now() - timedelta(minutes=500)).isoformat(), did))
        self.store.conn.commit()
        n = resolve_due_skips(self.store, lambda s: 90.0, horizon_min=390)
        self.assertEqual(n, 0)
        self.assertIsNone(self._row(did)["outcome"])

    def test_missing_entry_price_is_skipped_not_crashed(self):
        did = self._log_skip("DIA", "CALL", 100.0, age_min=500, raw={})
        n = resolve_due_skips(self.store, lambda s: 90.0, horizon_min=390)
        self.assertEqual(n, 0)
        self.assertIsNone(self._row(did)["outcome"])

    def test_price_fn_failure_is_defensive(self):
        did = self._log_skip("SPY", "CALL", 100.0, age_min=500)

        def boom(_sym):
            raise RuntimeError("quote service down")

        n = resolve_due_skips(self.store, boom, horizon_min=390)
        self.assertEqual(n, 0)
        self.assertIsNone(self._row(did)["outcome"])

    def test_price_fn_none_is_defensive(self):
        did = self._log_skip("SPY", "CALL", 100.0, age_min=500)
        n = resolve_due_skips(self.store, lambda s: None, horizon_min=390)
        self.assertEqual(n, 0)
        self.assertIsNone(self._row(did)["outcome"])

    def test_mixed_batch_resolves_only_due(self):
        due = self._log_skip("SPY", "CALL", 100.0, age_min=500)
        fresh = self._log_skip("QQQ", "PUT", 200.0, age_min=5)
        prices = {"SPY": 90.0, "QQQ": 190.0}
        n = resolve_due_skips(self.store, lambda s: prices.get(s), horizon_min=390)
        self.assertEqual(n, 1)
        self.assertEqual(self._row(due)["outcome"], "skip_resolved")
        # Foregone CALL, 100->90 (price fell): would have lost => -10%.
        self.assertAlmostEqual(self._row(due)["net_pnl_pct"], -10.0)
        self.assertIsNone(self._row(fresh)["outcome"])

    def test_open_skips_helper(self):
        self._log_skip("SPY", "CALL", 100.0, age_min=5)
        skips = self.store.open_skips()
        self.assertEqual(len(skips), 1)
        self.assertEqual(skips[0]["underlying"], "SPY")
        self.assertEqual(skips[0]["rule_action"], "CALL")


if __name__ == "__main__":
    unittest.main()
