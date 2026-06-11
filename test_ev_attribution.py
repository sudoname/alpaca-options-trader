"""
Offline tests for Phase 10E — EV Attribution.

No creds, no network, no broker. Covers:
  - EV / EV-Risk / PoP / Oracle / Vol-Edge / Advisory bucket calculations
  - per-bucket profit factor and the full stat block
  - monotonicity, best/worst bucket, separation score, verdicts
  - record merging (trades file + attribution snapshots)
  - empty datasets and malformed rows (never raises)
  - Telegram output
  - no execution path touched (static guards)
"""

import os
import unittest

import advisory_gate as ag
import ev_attribution as eva
from ev_attribution import (
    EV_BUCKETS, EV_RISK_BUCKETS, POP_BUCKETS, ORACLE_BUCKETS,
    VOL_EDGE_BUCKETS, ADVISORY_ORDER,
    VERDICT_YES, VERDICT_NO, VERDICT_INCONCLUSIVE, PF_CAP,
    ANALYTICS_FOOTER,
    bucket_label, bucket_stats, compute_bucket_table, compute_category_table,
    compute_predictiveness, load_closed_records, compute_ev_attribution,
    format_ev_attribution,
)

HERE = os.path.dirname(os.path.abspath(__file__))


def mk(i, pnl, ev=None, ratio=None, pop=None, oracle=None, edge=None,
       adv=None, strategy="bullish_put_credit_spread", max_loss=400.0):
    """Synthetic closed paper-spread record with frozen entry beliefs."""
    return {"id": f"t{i}", "pnl": pnl, "expected_value": ev,
            "ev_per_dollar_risk": ratio, "probability_of_profit": pop,
            "oracle_score": oracle, "volatility_edge": edge,
            "advisory_recommendation": adv, "strategy": strategy,
            "max_loss": max_loss}


def pf_pair(i, value_field, value, win, loss):
    """Two records in the same bucket engineering PF = win/loss exactly."""
    a = mk(f"{i}a", float(win))
    b = mk(f"{i}b", -float(loss))
    a[value_field] = value
    b[value_field] = value
    return [a, b]


# --------------------------------------------------------------------------- #
# Bucket label assignment (half-open [lo, hi))
# --------------------------------------------------------------------------- #
class TestBucketLabels(unittest.TestCase):
    def test_ev_buckets(self):
        cases = [(-0.01, "EV < 0"), (0.0, "EV 0-10"), (9.99, "EV 0-10"),
                 (10.0, "EV 10-20"), (20.0, "EV 20-50"), (49.99, "EV 20-50"),
                 (50.0, "EV 50+"), (500.0, "EV 50+")]
        for value, expected in cases:
            self.assertEqual(bucket_label(value, EV_BUCKETS), expected, value)

    def test_ev_risk_buckets(self):
        cases = [(-0.5, "EV/Risk < 0"), (0.0, "EV/Risk 0-0.05"),
                 (0.05, "EV/Risk 0.05-0.10"), (0.10, "EV/Risk 0.10-0.20"),
                 (0.1999, "EV/Risk 0.10-0.20"), (0.20, "EV/Risk 0.20+")]
        for value, expected in cases:
            self.assertEqual(bucket_label(value, EV_RISK_BUCKETS), expected,
                             value)

    def test_pop_buckets(self):
        cases = [(0.49, "PoP <50%"), (0.50, "PoP 50-60%"),
                 (0.60, "PoP 60-70%"), (0.70, "PoP 70-80%"),
                 (0.80, "PoP 80%+"), (0.99, "PoP 80%+")]
        for value, expected in cases:
            self.assertEqual(bucket_label(value, POP_BUCKETS), expected, value)

    def test_oracle_buckets(self):
        cases = [(0, "Oracle 0-39"), (39.9, "Oracle 0-39"),
                 (40, "Oracle 40-59"), (60, "Oracle 60-79"),
                 (80, "Oracle 80-100"), (100, "Oracle 80-100")]
        for value, expected in cases:
            self.assertEqual(bucket_label(value, ORACLE_BUCKETS), expected,
                             value)

    def test_vol_edge_buckets_take_percent_values(self):
        # _edge_pct converts the stored fraction to percent before bucketing.
        self.assertAlmostEqual(eva._edge_pct({"volatility_edge": 0.034}), 3.4)
        cases = [(-0.1, "Edge <0%"), (0.0, "Edge 0-1%"), (1.0, "Edge 1-2%"),
                 (2.5, "Edge 2-3%"), (3.0, "Edge 3%+")]
        for value, expected in cases:
            self.assertEqual(bucket_label(value, VOL_EDGE_BUCKETS), expected,
                             value)

    def test_missing_or_non_numeric_is_unbucketed(self):
        self.assertIsNone(bucket_label(None, EV_BUCKETS))
        self.assertIsNone(bucket_label("garbage", EV_BUCKETS))


# --------------------------------------------------------------------------- #
# Per-bucket stat block (incl. profit factor)
# --------------------------------------------------------------------------- #
class TestBucketStats(unittest.TestCase):
    def test_full_stat_block(self):
        rows = [mk(1, 100.0, max_loss=400.0), mk(2, -50.0, max_loss=200.0)]
        m = bucket_stats(rows)
        self.assertEqual(m["trades"], 2)
        self.assertEqual(m["wins"], 1)
        self.assertEqual(m["losses"], 1)
        self.assertAlmostEqual(m["win_rate"], 0.5)
        self.assertAlmostEqual(m["total_pnl"], 50.0)
        self.assertAlmostEqual(m["average_pnl"], 25.0)
        self.assertAlmostEqual(m["profit_factor"], 2.0)
        self.assertAlmostEqual(m["max_loss_observed"], 50.0)
        # (100/400 + -50/200) / 2 = (0.25 - 0.25) / 2 = 0.0
        self.assertAlmostEqual(m["average_return_on_risk"], 0.0)

    def test_profit_factor_all_wins_is_inf(self):
        m = bucket_stats([mk(1, 10.0), mk(2, 20.0)])
        self.assertEqual(m["profit_factor"], float("inf"))
        self.assertEqual(m["max_loss_observed"], 0.0)

    def test_empty_bucket(self):
        m = bucket_stats([])
        self.assertEqual(m["trades"], 0)
        self.assertEqual(m["losses"], 0)
        self.assertIsNone(m["profit_factor"])
        self.assertIsNone(m["average_return_on_risk"])
        self.assertEqual(m["average_pnl"], 0.0)

    def test_return_on_risk_skips_missing_max_loss(self):
        rows = [mk(1, 100.0, max_loss=400.0), mk(2, 50.0, max_loss=None)]
        m = bucket_stats(rows)
        self.assertAlmostEqual(m["average_return_on_risk"], 0.25)

    def test_bucket_table_routes_rows(self):
        rows = [mk(1, 10.0, ev=60.0), mk(2, -5.0, ev=60.0),
                mk(3, 7.0, ev=15.0), mk(4, 1.0, ev=None)]  # last unbucketed
        table = compute_bucket_table(rows, eva._ev, EV_BUCKETS)
        self.assertEqual(table["EV 50+"]["trades"], 2)
        self.assertEqual(table["EV 10-20"]["trades"], 1)
        self.assertEqual(table["EV < 0"]["trades"], 0)
        self.assertEqual(sum(m["trades"] for m in table.values()), 3)

    def test_advisory_category_table_zero_filled(self):
        rows = [mk(1, 10.0, adv=ag.STRONG_ACCEPT),
                mk(2, -5.0, adv=ag.STRONG_ACCEPT),
                mk(3, 3.0, adv=ag.WEAK_SETUP)]
        table = compute_category_table(rows, "advisory_recommendation",
                                       ADVISORY_ORDER)
        self.assertEqual(set(table), set(ADVISORY_ORDER))
        self.assertEqual(table[ag.STRONG_ACCEPT]["trades"], 2)
        self.assertEqual(table[ag.WEAK_SETUP]["trades"], 1)
        self.assertEqual(table[ag.NEUTRAL]["trades"], 0)


# --------------------------------------------------------------------------- #
# Predictiveness: monotonicity, separation, verdicts
# --------------------------------------------------------------------------- #
class TestPredictiveness(unittest.TestCase):
    def _table(self, pf_by_value):
        """EV bucket table where bucket of `value` has PF = win/loss."""
        rows = []
        for i, (value, (win, loss)) in enumerate(pf_by_value.items()):
            rows += pf_pair(i, "expected_value", value, win, loss)
        return compute_bucket_table(rows, eva._ev, EV_BUCKETS)

    def test_perfectly_rising_is_yes(self):
        # PFs: EV<0 -> 0.5, EV 10-20 -> 1.0, EV 50+ -> 2.0
        table = self._table({-5.0: (50, 100), 15.0: (100, 100),
                             60.0: (200, 100)})
        p = compute_predictiveness(table, [b[0] for b in EV_BUCKETS])
        self.assertEqual(p["buckets_with_data"], 3)
        self.assertAlmostEqual(p["monotonicity"], 1.0)
        self.assertAlmostEqual(p["separation"], 1.5)
        self.assertEqual(p["best_bucket"], "EV 50+")
        self.assertEqual(p["worst_bucket"], "EV < 0")
        self.assertEqual(p["verdict"], VERDICT_YES)

    def test_inverted_is_no(self):
        # PFs: EV<0 -> 2.0, EV 10-20 -> 1.0, EV 50+ -> 0.5
        table = self._table({-5.0: (200, 100), 15.0: (100, 100),
                             60.0: (50, 100)})
        p = compute_predictiveness(table, [b[0] for b in EV_BUCKETS])
        self.assertAlmostEqual(p["monotonicity"], 0.0)
        self.assertAlmostEqual(p["separation"], -1.5)
        self.assertEqual(p["verdict"], VERDICT_NO)

    def test_separation_matches_spec_example(self):
        # EV/Risk 0.20+ PF 2.4 vs EV/Risk <0 PF 0.6 -> separation 1.8.
        rows = (pf_pair(1, "ev_per_dollar_risk", -0.10, 60, 100)
                + pf_pair(2, "ev_per_dollar_risk", 0.30, 240, 100))
        table = compute_bucket_table(rows, eva._ev_risk, EV_RISK_BUCKETS)
        p = compute_predictiveness(table, [b[0] for b in EV_RISK_BUCKETS])
        self.assertAlmostEqual(p["separation"], 1.8)
        self.assertEqual(p["best_bucket"], "EV/Risk 0.20+")
        self.assertEqual(p["worst_bucket"], "EV/Risk < 0")
        self.assertEqual(p["verdict"], VERDICT_YES)

    def test_positive_separation_but_choppy_is_inconclusive(self):
        # PFs: 1.0, 0.4, 0.3, 1.2 -> mono 1/3 < 0.5, separation +0.2 > 0.
        table = self._table({-5.0: (100, 100), 5.0: (40, 100),
                             15.0: (30, 100), 60.0: (120, 100)})
        p = compute_predictiveness(table, [b[0] for b in EV_BUCKETS])
        self.assertAlmostEqual(p["monotonicity"], round(1 / 3, 2))
        self.assertAlmostEqual(p["separation"], 0.2)
        self.assertEqual(p["verdict"], VERDICT_INCONCLUSIVE)

    def test_single_bucket_is_inconclusive(self):
        table = self._table({15.0: (100, 50)})
        p = compute_predictiveness(table, [b[0] for b in EV_BUCKETS])
        self.assertEqual(p["buckets_with_data"], 1)
        self.assertEqual(p["best_bucket"], "EV 10-20")
        self.assertEqual(p["worst_bucket"], "EV 10-20")
        self.assertIsNone(p["separation"])
        self.assertEqual(p["verdict"], VERDICT_INCONCLUSIVE)

    def test_empty_table_is_inconclusive(self):
        p = compute_predictiveness(
            compute_bucket_table([], eva._ev, EV_BUCKETS),
            [b[0] for b in EV_BUCKETS])
        self.assertEqual(p["buckets_with_data"], 0)
        self.assertIsNone(p["best_bucket"])
        self.assertEqual(p["verdict"], VERDICT_INCONCLUSIVE)

    def test_infinite_pf_capped_for_scoring(self):
        # All-win top bucket: PF inf -> measured as PF_CAP, still YES.
        rows = pf_pair(1, "expected_value", -5.0, 50, 100)  # PF 0.5
        winner = mk("w", 80.0)
        winner["expected_value"] = 60.0
        rows.append(winner)                                  # PF inf
        table = compute_bucket_table(rows, eva._ev, EV_BUCKETS)
        p = compute_predictiveness(table, [b[0] for b in EV_BUCKETS])
        self.assertAlmostEqual(p["separation"], round(PF_CAP - 0.5, 2))
        self.assertEqual(p["verdict"], VERDICT_YES)


# --------------------------------------------------------------------------- #
# Record loading / merging
# --------------------------------------------------------------------------- #
class TestLoadClosedRecords(unittest.TestCase):
    def test_merges_trade_row_with_snapshot_by_id(self):
        trades = [{"id": "t1", "pnl": 50.0, "max_loss": 400.0,
                   "strategy": "bullish_put_credit_spread"}]
        snapshots = [{"trade_id": "t1", "pnl": 50.0,
                      "advisory_recommendation": ag.ACCEPT,
                      "expected_value": 12.0}]
        records = load_closed_records(trades=trades, snapshots=snapshots)
        self.assertEqual(len(records), 1)
        rec = records[0]
        self.assertEqual(rec["max_loss"], 400.0)                # from trade
        self.assertEqual(rec["advisory_recommendation"], ag.ACCEPT)  # snap
        self.assertEqual(rec["expected_value"], 12.0)           # from snap

    def test_snapshot_advisory_wins_over_trade_field(self):
        trades = [{"id": "t1", "pnl": 50.0,
                   "advisory_recommendation": ag.NEUTRAL}]
        snapshots = [{"trade_id": "t1", "pnl": 50.0,
                      "advisory_recommendation": ag.STRONG_ACCEPT}]
        records = load_closed_records(trades=trades, snapshots=snapshots)
        self.assertEqual(records[0]["advisory_recommendation"],
                         ag.STRONG_ACCEPT)

    def test_snapshot_only_record_still_counts(self):
        records = load_closed_records(
            trades=[],
            snapshots=[{"trade_id": "s1", "pnl": -20.0,
                        "advisory_recommendation": ag.WEAK_SETUP}])
        self.assertEqual(len(records), 1)

    def test_open_rows_excluded(self):
        records = load_closed_records(
            trades=[{"id": "t1", "pnl": None}],
            snapshots=[{"trade_id": "t1", "pnl": None}])
        self.assertEqual(records, [])

    def test_missing_files_fail_open(self):
        from oracle_analytics import AnalyticsConfig
        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/evat_t.json",
            spread_positions_file="/nonexistent/evat_p.json",
            expected_move_file="/nonexistent/evat_e.csv",
            training_dataset_file="/nonexistent/evat_d.csv",
            trade_history_file="/nonexistent/evat_h.json")
        records = load_closed_records(
            config=cfg, attribution_path="/nonexistent/evat_a.json")
        self.assertEqual(records, [])


# --------------------------------------------------------------------------- #
# Empty datasets and malformed rows
# --------------------------------------------------------------------------- #
class TestEmptyAndMalformed(unittest.TestCase):
    def test_empty_report(self):
        report = compute_ev_attribution(records=[])
        self.assertEqual(report["sample_size"], 0)
        self.assertEqual(report["confidence"], "Low")
        self.assertEqual(report["ev_predictiveness"]["verdict"],
                         VERDICT_INCONCLUSIVE)
        text = format_ev_attribution(report)
        self.assertIn("No closed paper spread trades", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_malformed_rows_never_raise(self):
        report = compute_ev_attribution(
            trades=["garbage", 42, None, {},
                    {"id": "a", "pnl": "not-a-number"},
                    {"id": "b", "pnl": 10.0, "expected_value": "junk"}],
            snapshots=[3, "x", {"trade_id": "c"}])
        # Only the one parsable closed row survives; junk EV is unbucketed.
        self.assertEqual(report["sample_size"], 1)
        self.assertEqual(
            sum(m["trades"] for m in report["ev_buckets"].values()), 0)
        # Formatting the imperfect report never raises either.
        self.assertIn(ANALYTICS_FOOTER, format_ev_attribution(report))

    def test_confidence_tiers(self):
        small = compute_ev_attribution(records=[mk(i, 1.0) for i in range(5)])
        medium = compute_ev_attribution(records=[mk(i, 1.0) for i in range(60)])
        large = compute_ev_attribution(records=[mk(i, 1.0)
                                                for i in range(250)])
        self.assertEqual(small["confidence"], "Low")
        self.assertEqual(medium["confidence"], "Medium")
        self.assertEqual(large["confidence"], "High")


# --------------------------------------------------------------------------- #
# Telegram output
# --------------------------------------------------------------------------- #
class TestTelegramOutput(unittest.TestCase):
    def _records(self):
        return [
            mk(1, 80.0, ev=60.0, ratio=0.25, pop=0.82, oracle=85, edge=0.034,
               adv=ag.STRONG_ACCEPT),
            mk(2, 60.0, ev=55.0, ratio=0.22, pop=0.81, oracle=82, edge=0.031,
               adv=ag.STRONG_ACCEPT),
            mk(3, -40.0, ev=25.0, ratio=0.12, pop=0.72, oracle=70, edge=0.015,
               adv=ag.ACCEPT),
            mk(4, -55.0, ev=-5.0, ratio=-0.02, pop=0.45, oracle=35,
               edge=-0.004, adv=ag.WEAK_SETUP),
        ]

    def test_report_sections(self):
        text = format_ev_attribution(
            compute_ev_attribution(records=self._records()))
        self.assertIn("EV Attribution", text)
        self.assertIn("*Expected Value buckets:*", text)
        self.assertIn("*EV/Risk buckets:*", text)
        self.assertIn("*Predictiveness:*", text)
        self.assertIn("`EV 50+`", text)
        self.assertIn("`EV < 0`", text)
        self.assertIn("Higher buckets outperform lower buckets:", text)
        self.assertIn("Sample size: `4`", text)
        self.assertIn("Confidence: *Low*", text)
        self.assertIn(ANALYTICS_FOOTER, text)

    def test_empty_buckets_hidden(self):
        text = format_ev_attribution(
            compute_ev_attribution(records=self._records()))
        self.assertNotIn("`EV 10-20`", text)  # no record lands there

    def test_telegram_bot_wires_the_command(self):
        with open(os.path.join(HERE, "telegram_bot.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("EV_ATTRIBUTION", src)
        self.assertIn("def ev_attribution", src)


# --------------------------------------------------------------------------- #
# No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_live_modules_do_not_consume_attribution(self):
        for name in ("run_alpaca_intraday.py", "smart_trader.py"):
            src = self._read(name)
            self.assertNotIn("ev_attribution", src,
                             f"{name} must not import ev_attribution")

    def test_module_never_imports_live_trader_or_network(self):
        src = self._read("ev_attribution.py")
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
