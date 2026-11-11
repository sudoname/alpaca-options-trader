"""
Phase 3 — exit replay on the quote path: trigger vs execution regression tests.

Master-prompt requirements covered (exit_replay.py):
  * the exit TRIGGER (exit_manager.evaluate_exit — the live policy) and the
    exit EXECUTION (research_replay.simulate_fill — sell at bid minus modeled
    slippage, fees, depth caps) are modeled separately;
  * gap-through stops fill at the gapped quote, never at the stop level;
  * stop-and-recovery winners: the stop is honored (no lookahead undo) and the
    recovery is reported only as a diagnostic flag;
  * stale quotes neither trigger nor fill — skipped and counted;
  * partial exits: depth caps leave quantity working on later ticks; the
    un-exited remainder is reported (exited_partial / exit_unfilled);
  * forced time exits close on the last valid tick (reason 'session_end');
  * ambiguous ticks (quote spans both the loss and profit levels) are flagged
    and resolved by an explicit policy, pessimistic by default;
  * provenance of the result is the WEAKEST tick label consumed;
  * exit-side accounting identity: proceeds - entry notional, fees once;
  * byte-identical determinism and a JSONL import path for real data.

Deterministic fixtures only. No creds, no network; the JSONL test writes to a
tempdir. Default execution model: sell fill = bid - 0.02 slippage, OCC fee
0.02/contract, quotes staler than 60s unusable.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from exit_replay import QuoteTick, replay_exit_path, ticks_from_jsonl
from research_replay import ExecutionModel

LEVELS = {"stop_loss_percent": 10.0, "take_profit_percent": 20.0,
          "trailing_stop_distance": 0.05}
TRADE = {"entry_price": 1.00}


def tick(ts, bid, ask, **kw):
    return QuoteTick(ts=ts, bid=bid, ask=ask, **kw)


class TriggerVsExecutionTest(unittest.TestCase):
    def test_take_profit_triggers_at_mid_fills_at_bid_minus_slippage(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("2026-01-02T10:00:00", 1.04, 1.06),
            tick("2026-01-02T10:01:00", 1.21, 1.23),
        ])
        self.assertEqual(r["status"], "exited")
        self.assertEqual(r["trigger"]["action"], "take_profit")
        self.assertAlmostEqual(r["trigger"]["price"], 1.22)      # observed mid
        self.assertAlmostEqual(r["trigger"]["level_price"], 1.20)
        self.assertAlmostEqual(r["avg_exit_price"], 1.19)        # bid - 0.02
        self.assertEqual(r["filled_qty"], 1)

    def test_bid_trigger_source(self):
        # With bid-triggered stops the 0.89 bid fires even though mid holds.
        r = replay_exit_path(TRADE, LEVELS, [tick("t1", 0.89, 0.95)],
                             trigger_source="bid")
        self.assertEqual(r["trigger"]["action"], "stop_loss")
        self.assertAlmostEqual(r["trigger"]["price"], 0.89)
        r2 = replay_exit_path(TRADE, LEVELS, [tick("t1", 0.89, 0.95)],
                              force_exit_at_end=False)
        self.assertEqual(r2["status"], "open")                   # mid 0.92 holds

    def test_exit_fees_charged_once_and_identity(self):
        r = replay_exit_path(TRADE, LEVELS, [tick("t1", 1.21, 1.23)], qty=3)
        self.assertEqual(r["filled_qty"], 3)
        self.assertAlmostEqual(r["exit_fees"], 0.06)             # 0.02 x 3, once
        self.assertAlmostEqual(r["proceeds"], 1.19 * 100 * 3 - 0.06)
        # net_pnl_vs_entry = proceeds - entry notional (entry costs NOT re-charged)
        self.assertAlmostEqual(r["net_pnl_vs_entry"],
                               r["proceeds"] - 1.00 * 100 * 3)


class GapThroughAndRecoveryTest(unittest.TestCase):
    SERIES = [tick("t1", 0.99, 1.01),
              tick("t2", 0.60, 0.62),           # gap through the 0.90 stop
              tick("t3", 0.95, 0.97)]           # recovery above the level

    def test_gap_through_stop_fills_at_gapped_quote(self):
        r = replay_exit_path(TRADE, LEVELS, self.SERIES)
        self.assertEqual(r["trigger"]["action"], "stop_loss")
        self.assertTrue(r["gap_through"])
        self.assertAlmostEqual(r["trigger"]["level_price"], 0.90)
        self.assertAlmostEqual(r["trigger"]["price"], 0.61)      # observed, not 0.90
        self.assertAlmostEqual(r["avg_exit_price"], 0.58)        # 0.60 bid - 0.02
        self.assertAlmostEqual(r["net_pnl_vs_entry"], 0.58 * 100 - 0.02 - 100.0)

    def test_recovery_is_diagnostic_only_no_lookahead_undo(self):
        r = replay_exit_path(TRADE, LEVELS, self.SERIES)
        self.assertTrue(r["recovered_after_stop"])
        self.assertEqual(r["status"], "exited")                  # still stopped out
        self.assertEqual(r["fills"][0]["ts"], "t2")              # exit at the gap
        no_recovery = replay_exit_path(TRADE, LEVELS, self.SERIES[:2])
        self.assertFalse(no_recovery["recovered_after_stop"])
        self.assertAlmostEqual(no_recovery["avg_exit_price"], 0.58)  # same exit


class StaleQuoteTest(unittest.TestCase):
    def test_stale_quote_does_not_trigger(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", 0.99, 1.01),
            tick("t2", 0.60, 0.62, quote_age_s=120.0),   # would stop, but stale
            tick("t3", 1.00, 1.02),
        ], force_exit_at_end=False)
        self.assertEqual(r["status"], "open")
        self.assertEqual(r["counts"]["stale"], 1)
        self.assertIsNone(r["trigger"])

    def test_stale_quote_does_not_fill_committed_exit(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", 0.60, 0.62, bid_size=0),            # stop fires, no depth
            tick("t2", 0.95, 0.97, quote_age_s=120.0),     # better price but stale
            tick("t3", 0.70, 0.72),
        ])
        self.assertEqual(r["status"], "exited")
        self.assertEqual(r["fills"][0]["ts"], "t3")        # skipped the stale tick
        self.assertAlmostEqual(r["avg_exit_price"], 0.68)
        self.assertEqual(r["counts"]["stale"], 1)

    def test_tighter_model_staleness(self):
        model = ExecutionModel(name="stale_quotes", max_quote_age_s=5.0)
        r = replay_exit_path(TRADE, LEVELS,
                             [tick("t1", 0.60, 0.62, quote_age_s=10.0)],
                             model=model, force_exit_at_end=False)
        self.assertEqual(r["status"], "unreplayable")      # no valid tick at all
        self.assertEqual(r["counts"]["stale"], 1)


class PartialExitTest(unittest.TestCase):
    def test_depth_cap_accumulates_across_ticks(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", 0.99, 1.01),
            tick("t2", 0.60, 0.62, bid_size=2),
            tick("t3", 0.55, 0.57, bid_size=10),
        ], qty=5)
        self.assertEqual(r["status"], "exited")
        self.assertEqual([f["qty"] for f in r["fills"]], [2, 3])
        self.assertEqual([f["price"] for f in r["fills"]], [0.58, 0.53])
        self.assertAlmostEqual(r["avg_exit_price"], (0.58 * 2 + 0.53 * 3) / 5)
        self.assertAlmostEqual(r["exit_fees"], 0.10)
        self.assertAlmostEqual(r["net_pnl_vs_entry"], r["proceeds"] - 500.0)

    def test_unexited_remainder_is_reported(self):
        r = replay_exit_path(TRADE, LEVELS,
                             [tick("t1", 0.60, 0.62, bid_size=1)], qty=3)
        self.assertEqual(r["status"], "exited_partial")
        self.assertEqual(r["filled_qty"], 1)
        self.assertEqual(r["remaining_qty"], 2)

    def test_exit_unfilled_when_no_depth_ever(self):
        r = replay_exit_path(TRADE, LEVELS,
                             [tick("t1", 0.60, 0.62, bid_size=0)], qty=2)
        self.assertEqual(r["status"], "exit_unfilled")
        self.assertEqual(r["filled_qty"], 0)
        self.assertEqual(r["remaining_qty"], 2)
        self.assertIsNotNone(r["trigger"])                 # the instruction stands
        self.assertIsNone(r["avg_exit_price"])
        self.assertIsNone(r["net_pnl_vs_entry"])           # no fabricated P&L


class ForcedTimeExitTest(unittest.TestCase):
    def test_session_end_closes_on_last_valid_tick(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", 1.02, 1.04),
            tick("t2", 1.05, 1.07),
            tick("t3", 0.50, 0.40),                        # crossed: invalid
        ])
        self.assertTrue(r["forced_time_exit"])
        self.assertEqual(r["trigger"]["action"], "forced_time_exit")
        self.assertEqual(r["trigger"]["reason"], "session_end")
        self.assertEqual(r["trigger"]["ts"], "t2")         # last VALID tick
        self.assertAlmostEqual(r["avg_exit_price"], 1.03)
        self.assertEqual(r["counts"]["invalid"], 1)

    def test_opt_out_keeps_position_open(self):
        r = replay_exit_path(TRADE, LEVELS, [tick("t1", 1.02, 1.04)],
                             force_exit_at_end=False)
        self.assertEqual(r["status"], "open")
        self.assertFalse(r["forced_time_exit"])
        self.assertEqual(r["fills"], [])


class AmbiguityTest(unittest.TestCase):
    # Quote spans both levels: bid 0.85 alone says stop (-15%), ask 1.25 alone
    # says take (+25%), mid 1.05 says hold.
    WIDE = [tick("t1", 0.85, 1.25)]

    def test_default_policy_is_pessimistic_stop_first(self):
        r = replay_exit_path(TRADE, LEVELS, self.WIDE)
        self.assertTrue(r["ambiguous"])
        self.assertIn("stop_loss", r["ambiguity_detail"])
        self.assertEqual(r["trigger"]["action"], "stop_loss")
        self.assertAlmostEqual(r["trigger"]["price"], 0.85)
        self.assertAlmostEqual(r["avg_exit_price"], 0.83)

    def test_take_first_policy_still_fills_on_the_bid(self):
        r = replay_exit_path(TRADE, LEVELS, self.WIDE,
                             ambiguity_policy="take_first")
        self.assertTrue(r["ambiguous"])
        self.assertEqual(r["trigger"]["action"], "take_profit")
        self.assertAlmostEqual(r["trigger"]["price"], 1.25)
        self.assertAlmostEqual(r["avg_exit_price"], 0.83)  # execution is honest

    def test_unambiguous_tick_not_flagged(self):
        r = replay_exit_path(TRADE, LEVELS, [tick("t1", 1.21, 1.23)])
        self.assertFalse(r["ambiguous"])
        self.assertIsNone(r["ambiguity_detail"])


class TrailingStopReplayTest(unittest.TestCase):
    def test_arm_ride_and_trail_out(self):
        levels = dict(LEVELS, take_profit_percent=200.0)   # let the trail decide
        r = replay_exit_path(TRADE, levels, [
            tick("t1", 1.00, 1.02),    # mid 1.01 arms trailing, high 1.01
            tick("t2", 1.30, 1.32),    # high 1.31
            tick("t3", 1.22, 1.24),    # mid 1.23 < 1.31*0.95 = 1.2445 -> trail
        ])
        self.assertEqual(r["trigger"]["action"], "trailing_stop")
        self.assertAlmostEqual(r["trigger"]["level_price"], 1.2445)
        self.assertAlmostEqual(r["avg_exit_price"], 1.20)
        self.assertGreater(r["net_pnl_vs_entry"], 0)       # stop-and-ride winner

    def test_same_tick_cannot_set_high_and_fire(self):
        levels = dict(LEVELS, take_profit_percent=200.0)
        r = replay_exit_path(TRADE, levels, [tick("t1", 1.30, 1.32)],
                             force_exit_at_end=False)
        self.assertEqual(r["status"], "open")              # high == cur: no fire

    def test_arm_profit_threshold_respected(self):
        levels = dict(LEVELS, take_profit_percent=200.0)
        r = replay_exit_path(TRADE, levels, [
            tick("t1", 1.04, 1.06),    # +5% < 25% arm threshold: not armed
            tick("t2", 0.98, 1.00),    # drop would trail-fire if armed
        ], arm_profit_pct=0.25, force_exit_at_end=False)
        self.assertEqual(r["status"], "open")


class DataQualityAndProvenanceTest(unittest.TestCase):
    def test_invalid_quotes_counted_never_trigger(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", None, 1.01),            # one-sided
            tick("t2", 0.70, 0.60),            # crossed
            tick("t3", 0.0, 0.62),             # non-positive
        ], force_exit_at_end=False)
        self.assertEqual(r["status"], "unreplayable")
        self.assertEqual(r["counts"]["invalid"], 3)
        self.assertEqual(r["counts"]["valid"], 0)

    def test_missing_entry_price_unreplayable(self):
        r = replay_exit_path({}, LEVELS, [tick("t1", 1.21, 1.23)])
        self.assertEqual(r["status"], "unreplayable")
        self.assertEqual(r["reason"], "missing_entry_price")

    def test_weakest_provenance_propagates(self):
        r = replay_exit_path(TRADE, LEVELS, [
            tick("t1", 0.99, 1.01, provenance="actual"),
            tick("t2", 0.60, 0.62, provenance="synthetic"),
        ])
        self.assertEqual(r["provenance"], "synthetic")
        r2 = replay_exit_path(TRADE, LEVELS,
                              [tick("t1", 1.21, 1.23, provenance="actual")])
        self.assertEqual(r2["provenance"], "actual")

    def test_scenario_label_carried(self):
        model = ExecutionModel(name="heavy_slippage",
                               extra_slippage_per_contract=0.05)
        r = replay_exit_path(TRADE, LEVELS, [tick("t1", 1.21, 1.23)],
                             model=model)
        self.assertEqual(r["scenario"], "heavy_slippage")
        self.assertAlmostEqual(r["avg_exit_price"], 1.14)  # bid - 0.07


class DeterminismAndImportTest(unittest.TestCase):
    def test_replay_is_deterministic(self):
        series = [tick("t1", 0.99, 1.01), tick("t2", 0.60, 0.62, bid_size=2),
                  tick("t3", 0.55, 0.57, bid_size=10)]
        a = replay_exit_path(TRADE, LEVELS, series, qty=5)
        b = replay_exit_path(TRADE, LEVELS, series, qty=5)
        self.assertEqual(a, b)

    def test_caller_trade_dict_not_mutated(self):
        trade = {"entry_price": 1.00}
        levels = dict(LEVELS, take_profit_percent=200.0)
        replay_exit_path(trade, levels, [tick("t1", 1.00, 1.02),
                                         tick("t2", 1.30, 1.32)])
        self.assertEqual(trade, {"entry_price": 1.00})     # no trailing state leak

    def test_jsonl_import_path(self):
        rows = [
            {"ts": "2026-11-07T14:31:00+00:00", "bid": 1.19, "ask": 1.21,
             "bid_size": 25, "ask_size": 40, "quote_age_s": 0.4,
             "provenance": "historical"},
            {"ts": "2026-11-07T14:32:00+00:00", "bid": 1.25, "ask": 1.27},
        ]
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "quotes.jsonl")
            with open(path, "w", encoding="utf-8") as f:
                for rec in rows:
                    f.write(json.dumps(rec) + "\n")
                f.write("not json\n")                      # junk skipped
                f.write(json.dumps({"bid": 1.0, "ask": 1.1}) + "\n")  # no ts
            ticks = ticks_from_jsonl(path)
        self.assertEqual(len(ticks), 2)
        self.assertEqual(ticks[0].bid_size, 25)
        self.assertEqual(ticks[1].provenance, "historical")
        r = replay_exit_path(TRADE, LEVELS, ticks)
        self.assertEqual(r["status"], "exited")
        self.assertEqual(r["provenance"], "historical")

    def test_missing_file_returns_empty(self):
        self.assertEqual(ticks_from_jsonl(
            os.path.join(tempfile.gettempdir(), "no_such_quotes.jsonl")), [])


if __name__ == "__main__":
    unittest.main()
