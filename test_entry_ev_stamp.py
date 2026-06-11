"""
Offline tests for Phase 10H — Entry EV stamp + trading_history adapter.

No creds, no network, no broker. Covers:
  - compute_entry_stamp fields, max_loss = premium, POP clamping
  - gambler's-ruin baseline is EV-zero gross; tilt makes it positive
  - tilt monotonic in confidence and |delta|; capped at +0.25
  - wider spreads -> lower net EV (costs)
  - fail-open: bad inputs return {} and never raise
  - load_closed_records merges stamped trading_history rows, skips
    stampless/unfinished rows, and spread-trade rows keep priority
  - end-to-end: ev_calibration + pop_calibration read scheduler trades
    from a temp trading_history.json via AnalyticsConfig
  - static guards: smart_trader wires the stamp; module stays off the
    execution path
"""

import json
import os
import tempfile
import unittest

import entry_ev_stamp as es
from entry_ev_stamp import compute_entry_stamp, _round_trip_costs
from ev_attribution import load_closed_records
from oracle_analytics import AnalyticsConfig

HERE = os.path.dirname(os.path.abspath(__file__))

OPT = {"confidence": 2, "delta": 0.55, "type": "call"}
LEVELS = {"take_profit_percent": 0.25, "stop_loss_percent": 0.15}


def stamp(option=OPT, levels=LEVELS, price=2.50, qty=2, bid=2.45, ask=2.50):
    return compute_entry_stamp(option, levels, price, qty, bid=bid, ask=ask)


# --------------------------------------------------------------------------- #
# Stamp contents
# --------------------------------------------------------------------------- #
class TestStampFields(unittest.TestCase):
    def test_required_keys_and_max_loss(self):
        s = stamp()
        for key in ("stamp_version", "pop_model", "expected_value",
                    "probability_of_profit", "max_loss", "ev_per_dollar_risk",
                    "take_profit_pct", "stop_loss_pct", "signal_strength",
                    "entry_delta", "round_trip_costs"):
            self.assertIn(key, s)
        self.assertEqual(s["max_loss"], 2.50 * 100 * 2)  # premium paid
        self.assertEqual(s["stamp_version"], es.STAMP_VERSION)
        self.assertEqual(s["pop_model"], es.POP_MODEL)
        json.dumps(s)  # JSON-serializable

    def test_pop_formula_and_clamp(self):
        # No edge at all: pop == sl/(tp+sl) baseline exactly.
        s = stamp(option={"confidence": 0, "delta": 0.0})
        self.assertAlmostEqual(s["probability_of_profit"], 0.15 / 0.40)
        # Huge confidence: tilt caps at +0.25.
        s = stamp(option={"confidence": 50, "delta": 0.9})
        self.assertAlmostEqual(s["probability_of_profit"],
                               0.15 / 0.40 + 0.25)
        # Clamps stay inside [0.02, 0.98].
        lo = stamp(option={"confidence": 0, "delta": 0.0},
                   levels={"take_profit_percent": 99.0,
                           "stop_loss_percent": 0.01})
        self.assertGreaterEqual(lo["probability_of_profit"], 0.02)
        hi = stamp(option={"confidence": 50, "delta": 0.9},
                   levels={"take_profit_percent": 0.01,
                           "stop_loss_percent": 99.0})
        self.assertLessEqual(hi["probability_of_profit"], 0.98)

    def test_baseline_is_gross_ev_zero(self):
        # pop = sl/(tp+sl) makes gross EV exactly 0; net EV = -costs.
        s = stamp(option={"confidence": 0, "delta": 0.0})
        costs = s["round_trip_costs"]
        self.assertAlmostEqual(s["expected_value"], -costs, places=2)

    def test_tilt_monotonic_in_confidence_and_delta(self):
        evs = [stamp(option={"confidence": c, "delta": 0.0})["expected_value"]
               for c in (0, 1, 2, 3)]
        self.assertEqual(evs, sorted(evs))
        self.assertLess(evs[0], evs[-1])
        by_delta = [stamp(option={"confidence": 0, "delta": d})
                    ["expected_value"] for d in (0.30, 0.50, 0.70)]
        self.assertEqual(by_delta, sorted(by_delta))
        # Delta below 0.40 contributes nothing.
        self.assertEqual(stamp(option={"confidence": 0, "delta": 0.10})
                         ["expected_value"],
                         stamp(option={"confidence": 0, "delta": 0.39})
                         ["expected_value"])

    def test_wider_spread_lowers_net_ev(self):
        tight = stamp(bid=2.48, ask=2.50)
        wide = stamp(bid=2.20, ask=2.50)
        self.assertLess(wide["expected_value"], tight["expected_value"])
        self.assertGreater(wide["round_trip_costs"],
                           tight["round_trip_costs"])

    def test_default_levels_when_missing(self):
        # Missing dynamic levels -> tp 0.25 / sl 0.15 defaults.
        s = stamp(levels=None)
        self.assertAlmostEqual(s["take_profit_pct"], 0.25)
        self.assertAlmostEqual(s["stop_loss_pct"], 0.15)

    def test_costs_never_negative(self):
        # Missing quotes may still carry fixed fees, but never go negative.
        self.assertGreaterEqual(_round_trip_costs(None, None, 1), 0.0)
        self.assertGreaterEqual(_round_trip_costs(0, 0, 3), 0.0)
        self.assertGreaterEqual(_round_trip_costs(2.45, 2.50, 2), 0.0)


# --------------------------------------------------------------------------- #
# Fail-open
# --------------------------------------------------------------------------- #
class TestFailOpen(unittest.TestCase):
    def test_bad_inputs_return_empty(self):
        self.assertEqual(compute_entry_stamp(OPT, LEVELS, None, 1), {})
        self.assertEqual(compute_entry_stamp(OPT, LEVELS, 0, 1), {})
        self.assertEqual(compute_entry_stamp(OPT, LEVELS, -2.0, 1), {})
        self.assertEqual(compute_entry_stamp(OPT, LEVELS, 2.5, 0), {})
        self.assertEqual(compute_entry_stamp(OPT, LEVELS, "junk", "junk"), {})

    def test_zero_levels_fall_back_to_defaults(self):
        # Explicit 0 tp/sl is nonsensical -> treated as missing (defaults).
        s = compute_entry_stamp(
            OPT, {"take_profit_percent": 0, "stop_loss_percent": 0}, 2.5, 1)
        self.assertAlmostEqual(s["take_profit_pct"], 0.25)
        self.assertAlmostEqual(s["stop_loss_pct"], 0.15)

    def test_garbage_option_and_levels_never_raise(self):
        for opt in (None, "junk", 7, {"confidence": "x", "delta": "y"}):
            for lv in (None, "junk", {"take_profit_percent": "z"}):
                s = compute_entry_stamp(opt, lv, 2.5, 1)
                self.assertIsInstance(s, dict)
                if s:
                    self.assertGreater(s["max_loss"], 0)


# --------------------------------------------------------------------------- #
# Adapter: trading_history rows into load_closed_records
# --------------------------------------------------------------------------- #
def hist_row(i, pnl=25.0, ev=10.0, pop=0.7, **extra):
    row = {"order_id": f"h{i}", "symbol": "SPY260612C00600000",
           "pnl": pnl, "expected_value": ev, "probability_of_profit": pop,
           "max_loss": 500.0}
    row.update(extra)
    return row


class TestHistoryAdapter(unittest.TestCase):
    def test_stamped_history_rows_are_loaded(self):
        rows = [hist_row(1), hist_row(2, pnl=-40.0, ev=5.0, pop=0.6)]
        recs = load_closed_records(trades=[], snapshots=[],
                                   history_trades=rows)
        self.assertEqual(len(recs), 2)
        self.assertEqual(recs[0]["order_id"], "h1")

    def test_stampless_and_open_rows_are_skipped(self):
        rows = [
            hist_row(1),                                   # good
            {"order_id": "h2", "pnl": 10.0},               # no stamp -> skip
            {"order_id": "h3", "expected_value": 5.0},     # no pnl -> skip
            "junk", None, 7,                               # malformed -> skip
        ]
        recs = load_closed_records(trades=[], snapshots=[],
                                   history_trades=rows)
        self.assertEqual([r["order_id"] for r in recs], ["h1"])

    def test_pop_only_stamp_still_counts(self):
        row = hist_row(1)
        del row["expected_value"]
        recs = load_closed_records(trades=[], snapshots=[],
                                   history_trades=[row])
        self.assertEqual(len(recs), 1)

    def test_spread_trades_keep_priority_over_history(self):
        spread = {"trade_id": "h1", "pnl": 99.0, "expected_value": 1.0,
                  "max_loss": 100.0}
        recs = load_closed_records(trades=[spread], snapshots=[],
                                   history_trades=[hist_row(1)])
        self.assertEqual(len(recs), 1)
        self.assertEqual(recs[0]["pnl"], 99.0)

    def test_reads_trading_history_file_via_config(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "trading_history.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"trades": [hist_row(i) for i in range(3)],
                           "performance_metrics": {}}, fh)
            cfg = AnalyticsConfig(
                spread_trades_file=os.path.join(td, "no_t.json"),
                spread_positions_file=os.path.join(td, "no_p.json"),
                expected_move_file=os.path.join(td, "no_e.csv"),
                training_dataset_file=os.path.join(td, "no_d.csv"),
                trade_history_file=path)
            recs = load_closed_records(
                config=cfg, attribution_path=os.path.join(td, "no_a.json"))
            self.assertEqual(len(recs), 3)

    def test_end_to_end_calibrations_see_scheduler_trades(self):
        from ev_calibration import compute_ev_calibration
        from pop_calibration import compute_pop_calibration
        # 12 closed scheduler trades, all stamped.
        rows = [hist_row(i, pnl=(30.0 if i % 3 else -30.0),
                         ev=8.0, pop=0.7) for i in range(12)]
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "trading_history.json")
            with open(path, "w", encoding="utf-8") as fh:
                json.dump({"trades": rows}, fh)
            cfg = AnalyticsConfig(
                spread_trades_file=os.path.join(td, "no_t.json"),
                spread_positions_file=os.path.join(td, "no_p.json"),
                expected_move_file=os.path.join(td, "no_e.csv"),
                training_dataset_file=os.path.join(td, "no_d.csv"),
                trade_history_file=path)
            apath = os.path.join(td, "no_a.json")
            ev_rep = compute_ev_calibration(config=cfg,
                                            attribution_path=apath)
            self.assertEqual(ev_rep["sample_size"], 12)
            pop_rep = compute_pop_calibration(config=cfg,
                                              attribution_path=apath)
            self.assertEqual(pop_rep["sample_size"], 12)
            b = pop_rep["buckets"]["PoP 70-80%"]
            self.assertEqual(b["trades"], 12)


# --------------------------------------------------------------------------- #
# Wiring + no execution path touched
# --------------------------------------------------------------------------- #
class TestWiringAndGuards(unittest.TestCase):
    def test_smart_trader_wires_the_stamp(self):
        with open(os.path.join(HERE, "smart_trader.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("compute_entry_stamp", src)
        self.assertIn("[EV STAMP]", src)

    def test_module_never_imports_live_trader_or_network(self):
        with open(os.path.join(HERE, "entry_ev_stamp.py"), "r",
                  encoding="utf-8") as fh:
            src = fh.read()
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order",
                       "open_position", "close_position"):
            self.assertNotIn(banned, src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
