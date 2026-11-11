"""Tests for ranking_eval, skeptic_impact, and the Phase-5b quality helpers.

Covered requirements (master prompt, Phase 5):
* Contract ranking evaluated by NET replay outcomes with nonfills counted as
  $0 at the opportunity level; variants for observed-only greeks, liquidity
  tilt, and delta band; insufficient-evidence verdict on empty input.
* Skeptic incremental impact measured beyond deterministic checks (quiet vs
  doubt groups among deterministic-pass rows), sample sizes always reported,
  insufficient_evidence below the resolved-count floor, counterfactual veto
  P&L, ledger round-trip join by decision_id, fail-open on a missing db.
* smart_trader per-field quality labels: broker only when a numeric snapshot
  value exists, fallback otherwise; unlabeled snapshot -> all fallback.
  _candidate_extra forwards only present fields and returns None when empty.

No credentials, no network.
"""
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import ranking_eval as re_mod  # noqa: E402
import skeptic_impact as si  # noqa: E402


def _cand(symbol, score, *, iv=0.22, delta=0.55, gq=None, bid=1.00, ask=1.04,
          age=5.0, vol=500, oi=400, net=0.0):
    return {"symbol": symbol, "score": score, "iv": iv, "delta": delta,
            "greeks_quality": gq, "bid": bid, "ask": ask, "quote_age_s": age,
            "volume": vol, "open_interest": oi, "net_outcome": net}


class TestRankingVariants(unittest.TestCase):
    def test_baseline_picks_argmax_score(self):
        g = {"group_id": "g", "candidates": [
            _cand("A", 1.0, net=10.0), _cand("B", 2.0, net=-5.0)]}
        rep = re_mod.evaluate_ranking([g], variants=("baseline",))
        self.assertEqual(rep["variants"]["baseline"]["picks"], ["B"])
        self.assertEqual(rep["variants"]["baseline"]["total_net"], -5.0)

    def test_observed_only_penalizes_fallback_fields(self):
        gq_fb = {"iv": "fallback", "delta": "fallback"}
        gq_ob = {"iv": "broker", "delta": "broker"}
        g = {"group_id": "g", "candidates": [
            _cand("FB", 2.0, gq=gq_fb, net=-20.0),
            _cand("OB", 1.2, gq=gq_ob, net=15.0)]}
        rep = re_mod.evaluate_ranking([g])
        # baseline takes the higher raw score; observed_only discounts it.
        self.assertEqual(rep["variants"]["baseline"]["picks"], ["FB"])
        self.assertEqual(rep["variants"]["observed_only"]["picks"], ["OB"])

    def test_unlabeled_treated_as_fallback(self):
        g = {"group_id": "g", "candidates": [
            _cand("UN", 2.0, gq=None, net=0.0),
            _cand("OB", 1.2, gq={"iv": "broker", "delta": "broker"},
                  net=5.0)]}
        rep = re_mod.evaluate_ranking([g], variants=("observed_only",))
        # 2.0 * 0.25 (two unobserved fields) < 1.2 -> observed wins.
        self.assertEqual(rep["variants"]["observed_only"]["picks"], ["OB"])

    def test_liquidity_tilt_prefers_tight_spread_and_oi(self):
        g = {"group_id": "g", "candidates": [
            _cand("WIDE", 1.1, bid=1.00, ask=1.20, oi=10, net=-10.0),
            _cand("TIGHT", 1.0, bid=1.00, ask=1.02, oi=500, net=10.0)]}
        rep = re_mod.evaluate_ranking([g], variants=("liquidity_tilt",))
        self.assertEqual(rep["variants"]["liquidity_tilt"]["picks"], ["TIGHT"])

    def test_delta_band_restricts_then_falls_back(self):
        in_band = _cand("IN", 1.0, delta=0.50, net=5.0)
        out_band = _cand("OUT", 2.0, delta=0.90, net=-5.0)
        g1 = {"group_id": "g1", "candidates": [in_band, out_band]}
        rep = re_mod.evaluate_ranking([g1], variants=("delta_band",))
        self.assertEqual(rep["variants"]["delta_band"]["picks"], ["IN"])
        # No qualifying delta -> falls back to the baseline pick.
        g2 = {"group_id": "g2", "candidates": [
            _cand("X", 1.0, delta=0.90, net=1.0),
            _cand("Y", 2.0, delta=0.10, net=2.0)]}
        rep2 = re_mod.evaluate_ranking([g2], variants=("delta_band",))
        self.assertEqual(rep2["variants"]["delta_band"]["picks"], ["Y"])

    def test_abs_delta_for_puts(self):
        g = {"group_id": "g", "candidates": [
            _cand("PUT", 1.0, delta=-0.55, net=5.0),
            _cand("FAR", 2.0, delta=-0.05, net=-5.0)]}
        rep = re_mod.evaluate_ranking([g], variants=("delta_band",))
        self.assertEqual(rep["variants"]["delta_band"]["picks"], ["PUT"])


class TestRankingAccounting(unittest.TestCase):
    def test_nonfill_counts_zero_at_opportunity_level(self):
        groups = [
            {"group_id": "g1", "candidates": [_cand("A", 1.0, net=30.0)]},
            {"group_id": "g2", "candidates": [_cand("B", 1.0, net=None)]},
        ]
        rep = re_mod.evaluate_ranking(groups, variants=("baseline",))
        v = rep["variants"]["baseline"]
        self.assertEqual(v["n_groups"], 2)
        self.assertEqual(v["n_filled"], 1)
        self.assertEqual(v["fill_rate"], 0.5)
        self.assertEqual(v["total_net"], 30.0)
        self.assertEqual(v["expectancy_per_opportunity"], 15.0)  # nonfill = 0
        self.assertEqual(v["expectancy_per_fill"], 30.0)

    def test_profit_factor_and_win_rate(self):
        groups = [
            {"group_id": f"g{i}", "candidates": [_cand("A", 1.0, net=n)]}
            for i, n in enumerate((20.0, -10.0, 30.0, -5.0))
        ]
        rep = re_mod.evaluate_ranking(groups, variants=("baseline",))
        v = rep["variants"]["baseline"]
        self.assertAlmostEqual(v["profit_factor"], 50.0 / 15.0)
        self.assertAlmostEqual(v["win_rate_filled"], 0.5)

    def test_empty_input_insufficient_evidence(self):
        rep = re_mod.evaluate_ranking([])
        self.assertEqual(rep["verdict"], "insufficient_evidence")

    def test_fixture_and_selftest(self):
        self.assertEqual(re_mod._self_test(), 0)
        rep = re_mod.evaluate_ranking(re_mod.fixture_groups())
        self.assertEqual(rep["variants"]["baseline"]["picks"],
                         ["A1", "B1", "C1", "D1"])
        # format_report renders every variant row without raising
        text = re_mod.format_report(rep)
        for name in re_mod.VARIANT_ORDER:
            self.assertIn(name, text)

    def test_jsonl_round_trip(self):
        import json
        groups = re_mod.fixture_groups()
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "groups.jsonl")
            with open(path, "w", encoding="utf-8") as fh:
                for g in groups:
                    fh.write(json.dumps(g) + "\n")
            loaded = re_mod.load_groups_jsonl(path)
        self.assertEqual(len(loaded), len(groups))
        rep_a = re_mod.evaluate_ranking(groups)
        rep_b = re_mod.evaluate_ranking(loaded)
        self.assertEqual(rep_a["variants"]["baseline"]["picks"],
                         rep_b["variants"]["baseline"]["picks"])


class TestSkepticImpact(unittest.TestCase):
    def test_fixture_adds_value(self):
        rep = si.evaluate_skeptic_impact(si.fixture_rows())
        self.assertEqual(rep["verdict"], "skeptic_adds_value")
        self.assertEqual(rep["quiet"]["n_resolved"], 6)
        self.assertEqual(rep["doubt"]["n_resolved"], 5)
        self.assertAlmostEqual(rep["avg_net_quiet_minus_doubt"], 28.0)
        self.assertAlmostEqual(rep["doubt_veto_avoided_pnl"], 80.0)
        self.assertEqual(rep["redundant_doubt_on_rejects"], 2)

    def test_redundant_when_no_outcome_difference(self):
        rows = []
        for i in range(6):
            rows.append(si.make_row(f"q{i}", deterministic_pass=True,
                                    net_outcome=5.0))
            rows.append(si.make_row(f"d{i}", deterministic_pass=True,
                                    skeptic_flags=["pile_on"],
                                    net_outcome=5.0))
        rep = si.evaluate_skeptic_impact(rows)
        self.assertEqual(rep["verdict"], "skeptic_redundant")

    def test_insufficient_evidence_below_floor(self):
        rep = si.evaluate_skeptic_impact(si.fixture_rows()[:4])
        self.assertEqual(rep["verdict"], "insufficient_evidence")
        self.assertIsNone(rep["avg_net_quiet_minus_doubt"])
        # Sample sizes are still reported.
        self.assertIn("n", rep["quiet"])
        self.assertIn("n_resolved", rep["doubt"])

    def test_unresolved_outcomes_never_imputed(self):
        rows = [si.make_row("u", deterministic_pass=True,
                            skeptic_flags=["x"], net_outcome=None)]
        rep = si.evaluate_skeptic_impact(rows)
        self.assertEqual(rep["doubt"]["n_unresolved"], 1)
        self.assertEqual(rep["doubt"]["total_net"], 0.0)
        self.assertIsNone(rep["doubt"]["avg_net"])

    def test_nan_outcome_treated_unresolved(self):
        rep = si.evaluate_skeptic_impact(
            [{"decision_id": "x", "deterministic_pass": True,
              "net_outcome": float("nan")}])
        self.assertEqual(rep["quiet"]["n_resolved"], 0)

    def test_boost_only_doubt_counts(self):
        row = si.make_row("b", deterministic_pass=True, no_trade_boost=0.03,
                          net_outcome=1.0)
        rep = si.evaluate_skeptic_impact([row])
        self.assertEqual(rep["doubt"]["n"], 1)
        self.assertEqual(rep["quiet"]["n"], 0)

    def test_ledger_round_trip_and_fail_open(self):
        from research_ledger import ResearchLedger
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "led.db")
            led = ResearchLedger(db)
            review = {"flags": ["stale_catalyst"], "no_trade_boost": 0.05}
            did = led.record_snapshot(kind="candidate", accepted=True,
                                      symbol="SPY261231C00450000",
                                      extra={"thesis_review": review})
            led.record_snapshot(kind="candidate", accepted=True,
                                symbol="SPY261231P00440000")  # no review
            led.record_event(decision_id=did, event_type="outcome",
                             payload={"net_pnl": -12.0})
            led.close()
            rows = si.rows_from_ledger(db)
            self.assertEqual(len(rows), 1)
            self.assertEqual(rows[0]["decision_id"], did)
            self.assertEqual(rows[0]["net_outcome"], -12.0)
            self.assertEqual(rows[0]["skeptic_flags"], ["stale_catalyst"])
        self.assertEqual(si.rows_from_ledger(
            os.path.join(td, "led.db")), [])  # deleted -> fail-open []
        self.assertEqual(si.rows_from_ledger(None), [])

    def test_module_self_test(self):
        self.assertEqual(si._self_test(), 0)


class TestSmartTraderQualityHelpers(unittest.TestCase):
    """Pure staticmethods — no trader instance, no credentials needed."""

    @classmethod
    def setUpClass(cls):
        from smart_trader import SmartOptionsTrader
        cls.T = SmartOptionsTrader

    def test_broker_labels_require_numeric_values(self):
        snap = {"delta": 0.55, "gamma": 0.02, "theta": -0.04, "vega": 0.11,
                "iv": 0.21}
        q = self.T._greeks_quality(snap)
        self.assertEqual(q, {f: "broker" for f in
                             ("delta", "gamma", "theta", "vega", "iv")})

    def test_partial_snapshot_mixed_labels(self):
        q = self.T._greeks_quality({"delta": 0.55, "iv": None,
                                    "vega": "bad"})
        self.assertEqual(q["delta"], "broker")
        self.assertEqual(q["iv"], "fallback")
        self.assertEqual(q["vega"], "fallback")
        self.assertEqual(q["gamma"], "fallback")

    def test_empty_or_none_all_fallback(self):
        for snap in ({}, None):
            q = self.T._greeks_quality(snap)
            self.assertEqual(set(q.values()), {"fallback"})
            self.assertEqual(set(q), {"delta", "gamma", "theta", "vega",
                                      "iv"})

    def test_candidate_extra_forwards_present_fields_only(self):
        opt = {"greeks_quality": {"iv": "broker"}, "quote_ts": "2026-11-11T15:00:00Z",
               "thesis_review": {"flags": []}, "bid": 1.0}
        extra = self.T._candidate_extra(opt)
        self.assertEqual(set(extra), {"greeks_quality", "quote_ts",
                                      "thesis_review"})

    def test_candidate_extra_none_when_empty(self):
        self.assertIsNone(self.T._candidate_extra({}))
        self.assertIsNone(self.T._candidate_extra(None))
        self.assertIsNone(self.T._candidate_extra({"bid": 1.0,
                                                   "quote_ts": None}))


if __name__ == "__main__":
    unittest.main()
