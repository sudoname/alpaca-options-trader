"""
Offline tests for decision_reports — the read-only analytics over the RL
episode store (episodes.db).

No creds, no network, no broker: each test builds a temp episodes.db through
EpisodeStore (the same writer the live bot uses) and points the providers at
it. Contract pinned here:

  1. FAIL-OPEN. Missing/corrupt db -> verdict INSUFFICIENT_DATA, never raise,
     and the db file is NEVER created by a provider.
  2. CORRECT SHAPES. Decision log rows carry gate/block-reason/votes parsed
     from the stored JSON; SKIP rows are flagged pnl_is_counterfactual.
  3. CORRECT AGGREGATES. Funnel counts and per-agent agreement/hit rates match
     hand-computed values over the fixture.
  4. READ-ONLY PROVIDERS. The compute_* bodies contain no write tokens.
"""

import os
import tempfile
import unittest

import decision_reports as dr
from episode_store import EpisodeStore

LONG_REASON = "profitability gate: " + "x" * 300


def _votes_trade1():
    return {
        "trend": {"bull": 0.8, "bear": 0.1, "conf": 0.9, "net": 0.7},
        "volatility": {"bull": 0.2, "bear": 0.6, "conf": 0.5, "net": -0.4},
        "news": {"bull": 0.0, "bear": 0.0, "conf": 0.3, "net": 0.0},
    }


def _votes_trade2():
    return {
        "trend": {"bull": 0.1, "bear": 0.6, "conf": 0.8, "net": -0.5},
        "volatility": {"bull": 0.3, "bear": 0.5, "conf": 0.5, "net": -0.2},
        "news": {"bull": 0.0, "bear": 0.0, "conf": 0.3, "net": 0.0},
    }


def build_fixture(path):
    """5 rows: CALL win, PUT loss, SKIP (CF pnl), open CALL, cap-skip SKIP
    without any evidence/votes. Returns the store's decision_ids in order."""
    store = EpisodeStore(path)
    ids = []
    # 1. traded CALL, resolved win (+30%).
    ids.append(store.log_decision(
        symbol="SPY260101C00500000", underlying="SPY", strat="intraday",
        features={"evidence": {"agent_votes": _votes_trade1(),
                               "regime_label": "trend_up",
                               "regime_confidence": 0.8,
                               "conviction": 0.6, "direction": "CALL"}},
        quote={"bid": 1.0, "ask": 1.1}, modeled_cost=None,
        rule_action="CALL", rule_confidence=65.0,
        gate={"veto": False, "reason": "ok", "q": 0.12, "visits": 7,
              "confidence": 0.7},
        risk=None, chosen_action="CALL", qty=1, mode="live-paper",
        as_of="2026-11-20T10:00:00"))
    store.record_outcome(ids[-1], fill_price=1.0, exit_price=1.3,
                         gross_pnl_pct=30.0, net_pnl_pct=30.0,
                         net_pnl_dollars=30.0, hold_days=0,
                         outcome="take_profit")
    # 2. traded PUT, resolved loss (-10%).
    ids.append(store.log_decision(
        symbol="QQQ260101P00400000", underlying="QQQ", strat="intraday",
        features={"evidence": {"agent_votes": _votes_trade2(),
                               "regime_label": "trend_down",
                               "regime_confidence": 0.6,
                               "conviction": 0.4, "direction": "PUT"}},
        quote={"bid": 2.0, "ask": 2.1}, modeled_cost=None,
        rule_action="PUT", rule_confidence=55.0,
        gate={"veto": False, "reason": "ok", "q": 0.05, "visits": 3,
              "confidence": 0.6},
        risk=None, chosen_action="PUT", qty=1, mode="live-paper",
        as_of="2026-11-20T11:00:00"))
    store.record_outcome(ids[-1], fill_price=2.0, exit_price=1.8,
                         gross_pnl_pct=-10.0, net_pnl_pct=-10.0,
                         net_pnl_dollars=-20.0, hold_days=0,
                         outcome="stop_loss")
    # 3. blocked SKIP with a long ":"-form reason; counterfactual +5%.
    ids.append(store.log_decision(
        symbol="SPY", underlying="SPY", strat="intraday",
        features={"evidence": {"regime_label": "chop",
                               "regime_confidence": 0.3}},
        quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=40.0,
        gate={"veto": True, "reason": "q below floor", "q": -0.02,
              "visits": 9, "confidence": 0.4},
        risk={"block_reason": LONG_REASON},
        chosen_action="SKIP", qty=0, mode="live-paper-blocked",
        as_of="2026-11-20T12:00:00"))
    store.record_outcome(ids[-1], net_pnl_pct=5.0, hold_days=0,
                         outcome="skip_resolved")
    # 4. open traded CALL (no outcome yet).
    ids.append(store.log_decision(
        symbol="IWM260101C00200000", underlying="IWM", strat="intraday",
        features={"evidence": {"agent_votes": _votes_trade1()}},
        quote={"bid": 0.5, "ask": 0.6}, modeled_cost=None,
        rule_action="CALL", rule_confidence=70.0, gate=None, risk=None,
        chosen_action="CALL", qty=2, mode="live-paper",
        as_of="2026-11-20T13:00:00"))
    # 5. cap-skip SKIP, "("-form reason, no evidence at all; CF +2%.
    ids.append(store.log_decision(
        symbol="SPY", underlying="SPY", strat="intraday",
        features={}, quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=80.0, gate=None,
        risk={"block_reason": "cap reached (3/3)"},
        chosen_action="SKIP", qty=0, mode="cap-skip-cf",
        as_of="2026-11-20T14:00:00"))
    store.record_outcome(ids[-1], net_pnl_pct=2.0, hold_days=0,
                         outcome="capskip_resolved")
    store.close()
    return ids


class WithFixture(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.db = os.path.join(self.d, "episodes.db")
        self.ids = build_fixture(self.db)


class TestFailOpen(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.missing = os.path.join(self.d, "none.db")

    def test_missing_db_insufficient(self):
        self.assertEqual(
            dr.compute_decision_log(episode_db=self.missing)["verdict"],
            "INSUFFICIENT_DATA")
        self.assertEqual(
            dr.compute_decision_funnel(episode_db=self.missing)["verdict"],
            "INSUFFICIENT_DATA")
        self.assertEqual(
            dr.compute_agent_contributions(episode_db=self.missing)["verdict"],
            "INSUFFICIENT_DATA")

    def test_providers_never_create_db(self):
        dr.compute_decision_log(episode_db=self.missing)
        dr.compute_decision_funnel(episode_db=self.missing)
        dr.compute_agent_contributions(episode_db=self.missing)
        self.assertFalse(os.path.exists(self.missing))

    def test_corrupt_db_fails_open(self):
        bad = os.path.join(self.d, "bad.db")
        with open(bad, "wb") as f:
            f.write(b"this is not a sqlite database at all")
        self.assertEqual(
            dr.compute_decision_log(episode_db=bad)["verdict"],
            "INSUFFICIENT_DATA")
        self.assertEqual(
            dr.compute_decision_funnel(episode_db=bad)["verdict"],
            "INSUFFICIENT_DATA")
        self.assertEqual(
            dr.compute_agent_contributions(episode_db=bad)["verdict"],
            "INSUFFICIENT_DATA")


class TestDecisionLog(WithFixture):
    def test_all_scope_newest_first(self):
        out = dr.compute_decision_log(episode_db=self.db)
        self.assertEqual(out["verdict"], "OK")
        self.assertEqual(out["scope"], "all")
        self.assertEqual(out["total"], 5)
        self.assertEqual(out["count"], 5)
        as_ofs = [r["as_of"] for r in out["rows"]]
        self.assertEqual(as_ofs, sorted(as_ofs, reverse=True))
        self.assertEqual(out["rows"][0]["as_of"], "2026-11-20T14:00:00")

    def test_limit_clamp_and_paging(self):
        out = dr.compute_decision_log(episode_db=self.db, limit=2)
        self.assertEqual(out["count"], 2)
        self.assertEqual(out["total"], 5)
        self.assertEqual(dr.compute_decision_log(
            episode_db=self.db, limit=0)["limit"], 1)
        self.assertEqual(dr.compute_decision_log(
            episode_db=self.db, limit=9999)["limit"], 200)
        self.assertEqual(dr.compute_decision_log(
            episode_db=self.db, limit="junk")["limit"], 50)

    def test_scope_filters(self):
        traded = dr.compute_decision_log(episode_db=self.db, scope="traded")
        self.assertEqual(traded["total"], 3)
        self.assertTrue(all(r["chosen_action"] in ("CALL", "PUT")
                            for r in traded["rows"]))
        blocked = dr.compute_decision_log(episode_db=self.db, scope="blocked")
        self.assertEqual(blocked["total"], 2)
        self.assertTrue(all(r["chosen_action"] == "SKIP"
                            for r in blocked["rows"]))
        # Invalid scope falls back to "all".
        bad = dr.compute_decision_log(episode_db=self.db, scope="nonsense")
        self.assertEqual(bad["scope"], "all")

    def test_counterfactual_flag_only_on_skip(self):
        out = dr.compute_decision_log(episode_db=self.db)
        for r in out["rows"]:
            self.assertEqual(r["pnl_is_counterfactual"],
                             r["chosen_action"] == "SKIP")

    def test_votes_gate_and_evidence_parsed(self):
        out = dr.compute_decision_log(episode_db=self.db, scope="traded")
        call = next(r for r in out["rows"] if r["underlying"] == "SPY")
        votes = call["agent_votes"]
        self.assertEqual([v["agent"] for v in votes],
                         ["news", "trend", "volatility"])
        trend = next(v for v in votes if v["agent"] == "trend")
        self.assertAlmostEqual(trend["net"], 0.7)
        self.assertAlmostEqual(trend["conf"], 0.9)
        self.assertEqual(call["gate"]["veto"], False)
        self.assertAlmostEqual(call["gate"]["q"], 0.12)
        self.assertEqual(call["regime_label"], "trend_up")
        self.assertAlmostEqual(call["conviction"], 0.6)
        self.assertEqual(call["direction"], "CALL")
        self.assertAlmostEqual(call["net_pnl_pct"], 30.0)
        self.assertEqual(call["outcome"], "take_profit")

    def test_open_row_has_null_outcome(self):
        out = dr.compute_decision_log(episode_db=self.db, scope="traded")
        iwm = next(r for r in out["rows"] if r["underlying"] == "IWM")
        self.assertIsNone(iwm["outcome"])
        self.assertIsNone(iwm["net_pnl_pct"])
        self.assertIsNone(iwm["gate"])

    def test_block_reason_truncation_and_prefix(self):
        out = dr.compute_decision_log(episode_db=self.db, scope="blocked")
        long_row = next(r for r in out["rows"]
                        if r["as_of"] == "2026-11-20T12:00:00")
        self.assertEqual(len(long_row["block_reason"]), 200)
        self.assertEqual(long_row["block_reason_full"], LONG_REASON)
        self.assertEqual(long_row["block_reason_prefix"],
                         "profitability gate")
        cap_row = next(r for r in out["rows"]
                       if r["as_of"] == "2026-11-20T14:00:00")
        self.assertEqual(cap_row["block_reason_prefix"], "cap reached")
        self.assertIsNone(cap_row["agent_votes"])


class TestFunnel(WithFixture):
    def test_counts(self):
        out = dr.compute_decision_funnel(episode_db=self.db)
        self.assertEqual(out["verdict"], "OK")
        self.assertEqual(out["total"], 5)
        self.assertEqual(out["completed"], 4)
        self.assertEqual(out["open"], 1)
        self.assertEqual(out["by_chosen_action"],
                         {"CALL": 2, "PUT": 1, "SKIP": 2})
        self.assertEqual(out["by_mode"]["live-paper"], 3)
        self.assertEqual(out["by_mode"]["live-paper-blocked"], 1)
        self.assertEqual(out["by_mode"]["cap-skip-cf"], 1)
        self.assertEqual(out["by_outcome"]["take_profit"], 1)
        self.assertEqual(out["by_outcome"]["open"], 1)

    def test_funnel_stages_and_reasons(self):
        out = dr.compute_decision_funnel(episode_db=self.db)
        stages = {s["stage"]: s["count"] for s in out["funnel"]}
        self.assertEqual(stages, {"decisions": 5, "traded": 3,
                                  "blocked": 2, "cap_skipped": 1})
        reasons = {r["prefix"]: r["count"] for r in out["block_reasons"]}
        self.assertEqual(reasons,
                         {"profitability gate": 1, "cap reached": 1})
        # Equal counts tie-break alphabetically.
        self.assertEqual(out["block_reasons"][0]["prefix"], "cap reached")


class TestContributions(WithFixture):
    def test_hand_computed_rates(self):
        out = dr.compute_agent_contributions(episode_db=self.db)
        self.assertEqual(out["verdict"], "OK")
        # Only the 2 resolved CALL/PUT rows qualify (open row has NULL pnl,
        # SKIP counterfactuals are excluded by chosen_action).
        self.assertEqual(out["n_trades"], 2)
        self.assertEqual(out["n_with_votes"], 2)
        agents = {a["agent"]: a for a in out["agents"]}
        self.assertEqual(sorted(agents), ["news", "trend", "volatility"])

        t = agents["trend"]  # agrees on both trades; win then loss
        self.assertEqual(t["n"], 2)
        self.assertAlmostEqual(t["agreement_rate"], 1.0)
        self.assertAlmostEqual(t["hit_rate_when_agreeing"], 0.5)
        self.assertEqual(t["n_bullish"], 1)
        self.assertAlmostEqual(t["hit_rate_when_bullish"], 1.0)
        self.assertEqual(t["n_bearish"], 1)
        self.assertAlmostEqual(t["hit_rate_when_bearish"], 0.0)
        self.assertAlmostEqual(t["mean_net_win"], 30.0)
        self.assertAlmostEqual(t["mean_net_loss"], -10.0)
        self.assertAlmostEqual(t["mean_conf"], 0.85)

        v = agents["volatility"]  # disagrees on the win, agrees on the loss
        self.assertAlmostEqual(v["agreement_rate"], 0.5)
        self.assertAlmostEqual(v["hit_rate_when_agreeing"], 0.0)
        self.assertEqual(v["n_bullish"], 0)
        self.assertIsNone(v["hit_rate_when_bullish"])
        self.assertEqual(v["n_bearish"], 2)
        self.assertAlmostEqual(v["hit_rate_when_bearish"], 0.5)

        n = agents["news"]  # net == 0 everywhere: never agrees, no sides
        self.assertAlmostEqual(n["agreement_rate"], 0.0)
        self.assertIsNone(n["hit_rate_when_agreeing"])
        self.assertEqual(n["n_bullish"], 0)
        self.assertEqual(n["n_bearish"], 0)

    def test_cap_clamp_and_echo(self):
        out = dr.compute_agent_contributions(episode_db=self.db, cap=1)
        self.assertEqual(out["cap"], 1)
        self.assertEqual(out["n_trades"], 1)
        self.assertEqual(dr.compute_agent_contributions(
            episode_db=self.db, cap=10 ** 9)["cap"], 20000)
        self.assertEqual(dr.compute_agent_contributions(
            episode_db=self.db, cap="junk")["cap"], 5000)


class TestReadOnlyProviders(unittest.TestCase):
    def test_provider_bodies_have_no_writes(self):
        # The public providers must not mutate the store (loaders legitimately
        # open the db read-style; fixtures live in this test file only).
        import inspect
        forbidden = ("log_decision(", "record_outcome(", "json.dump(",
                     ".commit(", "_save(", '"w"', "'w'", "INSERT", "UPDATE",
                     "DELETE", "executescript")
        for fn in (dr.compute_decision_log,
                   dr.compute_decision_funnel,
                   dr.compute_agent_contributions):
            src = inspect.getsource(fn)
            for tok in forbidden:
                self.assertNotIn(tok, src,
                                 msg=f"{fn.__name__} contains write token {tok!r}")


if __name__ == "__main__":
    unittest.main()
