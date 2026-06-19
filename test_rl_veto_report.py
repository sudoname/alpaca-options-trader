"""
Offline tests for Phase 12 — RL Veto Report.

No creds, no network, no broker. Covers:
  1. CALL states with negative Q are flagged (RL learned calls lose there).
  2. PUT states with positive Q are flagged (RL learned puts win there).
  3. Rule/RL disagreements are detected with realized PnL attached.
  4. The visit floor suppresses low-evidence flags.
  5. Empty Q-table -> clean INSUFFICIENT report carrying the shadow footer.
  6. The report is advisory only — verdicts/thresholds, never an action.

rl_veto_report is ANALYTICS ONLY and READ-ONLY: it never writes a Q-table, opens,
sizes or blocks any trade, and never reaches the network. The agent and episode
store are injected, so the whole module is deterministic.
"""

import unittest

import rl_veto_report as rvr
from rl_veto_report import (
    compute_rl_veto_report, format_rl_veto_report, generate_rl_veto_report_text,
    VERDICT_DISAGREES, VERDICT_ALIGNED, VERDICT_INSUFFICIENT,
    RL_VETO_FOOTER, CALL, PUT, SKIP,
)


class _Agent:
    """Stand-in exposing policy_report() like QLearningAgent."""

    def __init__(self, policy):
        self._policy = policy

    def policy_report(self):
        return {"num_states": len(self._policy), "policy": self._policy}


class _Store:
    def __init__(self, rows):
        self._rows = rows

    def completed(self):
        return self._rows


def _policy():
    return {
        # Aligned: RL likes CALL, rule fired CALL.
        "strat=t|change=up_strong": {
            "best_action": CALL,
            "q_values": {CALL: 0.30, PUT: -0.20, SKIP: 0.0},
            "visits": {CALL: 20, PUT: 8, SKIP: 5}},
        # RL learned CALL loses (negative Q) and PUT wins (positive Q); rule
        # still fired CALL while RL best is SKIP -> a disagreement.
        "strat=t|change=dn_strong": {
            "best_action": SKIP,
            "q_values": {CALL: -0.25, PUT: 0.18, SKIP: 0.0},
            "visits": {CALL: 15, PUT: 12, SKIP: 6}},
    }


def _store():
    return _Store([
        {"state_key": "strat=t|change=up_strong", "rule_action": "CALL",
         "net_pnl_pct": 12.0},
        {"state_key": "strat=t|change=dn_strong", "rule_action": "CALL",
         "net_pnl_pct": -18.0},
        {"state_key": "strat=t|change=dn_strong", "rule_action": "CALL",
         "net_pnl_pct": -9.0},
    ])


class TestFlags(unittest.TestCase):
    def setUp(self):
        self.rep = compute_rl_veto_report(agent=_Agent(_policy()),
                                          store=_store(), min_visits=5)

    def test_negative_call_q_flagged(self):
        keys = {d["state_key"] for d in self.rep["call_negative_states"]}
        self.assertIn("strat=t|change=dn_strong", keys)
        # The aligned positive-Q CALL state must NOT be flagged.
        self.assertNotIn("strat=t|change=up_strong", keys)

    def test_positive_put_q_flagged(self):
        keys = {d["state_key"] for d in self.rep["put_positive_states"]}
        self.assertIn("strat=t|change=dn_strong", keys)

    def test_disagreement_detected_with_realized_pnl(self):
        dis = [d for d in self.rep["disagreements"]
               if d["state_key"] == "strat=t|change=dn_strong"]
        self.assertTrue(dis)
        self.assertEqual(dis[0]["rule_action"], "CALL")
        self.assertEqual(dis[0]["rl_best_action"], SKIP)
        # Mean of -18 and -9 = -13.5.
        self.assertAlmostEqual(dis[0]["rule_realized_avg_pnl_pct"], -13.5)

    def test_aligned_state_is_not_a_disagreement(self):
        keys = {d["state_key"] for d in self.rep["disagreements"]}
        self.assertNotIn("strat=t|change=up_strong", keys)

    def test_verdict_disagrees(self):
        self.assertEqual(self.rep["verdict"], VERDICT_DISAGREES)

    def test_suggested_thresholds_are_advisory(self):
        s = self.rep["suggested_thresholds"]
        self.assertEqual(s["min_visits"], 5)
        self.assertEqual(s["veto_call_when_q_below"], 0.0)
        self.assertEqual(s["favor_put_when_q_above"], 0.0)
        self.assertEqual(s["note"], RL_VETO_FOOTER)
        self.assertEqual(s["strongest_negative_call_q"], -0.25)


class TestVisitFloor(unittest.TestCase):
    def test_high_floor_suppresses_flags(self):
        rep = compute_rl_veto_report(agent=_Agent(_policy()), store=_store(),
                                     min_visits=999)
        self.assertFalse(rep["call_negative_states"])
        self.assertFalse(rep["put_positive_states"])
        self.assertFalse(rep["disagreements"])

    def test_no_episodes_means_no_disagreements(self):
        # Without an episode store there is no rule action to disagree with.
        rep = compute_rl_veto_report(agent=_Agent(_policy()),
                                     store=_Store([]), min_visits=5)
        self.assertFalse(rep["disagreements"])
        # But Q-table-only flags still work.
        self.assertTrue(rep["call_negative_states"])


class TestVerdictAndEmpty(unittest.TestCase):
    def test_aligned_when_no_disagreements(self):
        policy = {
            "s|a": {"best_action": CALL,
                    "q_values": {CALL: 0.3, PUT: -0.1, SKIP: 0.0},
                    "visits": {CALL: 20, PUT: 5, SKIP: 5}},
        }
        store = _Store([{"state_key": "s|a", "rule_action": "CALL",
                         "net_pnl_pct": 5.0}])
        rep = compute_rl_veto_report(agent=_Agent(policy), store=store,
                                     min_visits=5)
        self.assertEqual(rep["verdict"], VERDICT_ALIGNED)

    def test_empty_qtable_is_insufficient(self):
        rep = compute_rl_veto_report(agent=_Agent({}), store=_Store([]))
        self.assertEqual(rep["num_states"], 0)
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)


class TestFormatting(unittest.TestCase):
    def test_empty_report_clean_with_footer(self):
        text = format_rl_veto_report(
            compute_rl_veto_report(agent=_Agent({})))
        self.assertIn(RL_VETO_FOOTER, text)
        self.assertIn(VERDICT_INSUFFICIENT, text)

    def test_populated_report_never_raises(self):
        rep = compute_rl_veto_report(agent=_Agent(_policy()), store=_store(),
                                     min_visits=5)
        text = format_rl_veto_report(rep)
        self.assertIn(RL_VETO_FOOTER, text)
        self.assertIn("RL Veto Report", text)
        self.assertIn(VERDICT_DISAGREES, text)

    def test_generate_text_smoke(self):
        # No injected sources -> reads disk fail-open; must not raise.
        text = generate_rl_veto_report_text()
        self.assertIn(RL_VETO_FOOTER, text)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(rvr._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
