"""Phase 4 regression tests — VWAP pullback features, order flow, study.

Covers the master-plan requirements:
  * 9/20/50 EMAs with documented warm-up behavior (None until seed, SMA seed,
    recursive EMA after; fail-open on garbage).
  * Reproducible VWAP pullback + confirmation sequence with explicit
    tolerance and expiry; mirrored short side as a separate hypothesis;
    decisive breaks reset; stale setups expire.
  * compute_intraday_features emits ema_9/ema_20/ema_9_over_20, omits ema_50
    during warm-up, gates pullback flags on a usable session VWAP, and stays
    fail-open ({}) on junk input.
  * Timestamped trade-print (Time & Sales) features: quote-rule first,
    tick-rule fallback; missing/stale/future-stamped data explicit.
  * Freshness-aware L2 and top-of-book: ok/stale/crossed/missing statuses;
    an unstamped book can never qualify as verified.
  * Confirmation variants (prints_only / top_of_book / verified_depth) apply
    ONLY when their required inputs qualify (confirmed is None otherwise).
  * Matched-comparison study: same sessions, fixed exits/costs, ladder
    monotonicity, calls/puts reported separately, 50-EMA as a separate
    hypothesis, unqualified-input accounting, cost charged once,
    conservative same-bar stop, JSONL round-trip, fail-open empty runs.

No credentials, no network.
"""

import json
import os
import sys
import tempfile
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vwap_pullback_study as vps
from intraday_features import (
    compute_intraday_features,
    compute_vwap_pullback,
    ema_series,
)
from market_view import make_intraday_bar
from order_flow_features import (
    assess_book,
    assess_top_of_book,
    compute_print_features,
    confirm_order_flow,
)

NOW = 1_700_000_100.0


def _bar(i, o, h, l, c, v=1000):
    return make_intraday_bar(f"2026-01-06T10:{i:02d}:00Z", o, h, l, c, v)


def _rally(n=30, step=0.1):
    return [_bar(i, 100 + step * i, 100.15 + step * i, 99.95 + step * i,
                 100.1 + step * i) for i in range(n)]


def _downtrend(n=30, step=0.1):
    return [_bar(i, 100 - step * i, 100.05 - step * i, 99.85 - step * i,
                 99.9 - step * i) for i in range(n)]


class TestEmaSeries(unittest.TestCase):
    def test_warmup_and_seed(self):
        vals = [1.0, 2.0, 3.0, 4.0, 5.0]
        out = ema_series(vals, 3)
        self.assertEqual(out[:2], [None, None])
        self.assertAlmostEqual(out[2], 2.0)  # SMA seed of first 3
        k = 2.0 / 4.0
        self.assertAlmostEqual(out[3], 4.0 * k + 2.0 * (1 - k))
        self.assertAlmostEqual(out[4], 5.0 * k + out[3] * (1 - k))

    def test_too_short_is_all_none(self):
        self.assertEqual(ema_series([1.0, 2.0], 9), [None, None])

    def test_garbage_fails_open(self):
        self.assertEqual(ema_series([1.0, "x", 3.0], 2), [None, None, None])
        self.assertEqual(ema_series([1.0, 2.0], 0), [None, None])
        self.assertEqual(ema_series([1.0, 2.0], -1), [None, None])
        self.assertEqual(ema_series(None, 3), [])
        self.assertEqual(ema_series([], 3), [])


class TestVwapPullback(unittest.TestCase):
    def _long_trend(self):
        return [_bar(i, 100 + 0.5 * i, 100.6 + 0.5 * i, 99.9 + 0.5 * i,
                     100.5 + 0.5 * i) for i in range(5)]

    def test_long_confirmation(self):
        bars = self._long_trend()
        bars.append(_bar(5, 102.4, 102.6, 101.0, 101.4))   # touch, holds
        bars.append(_bar(6, 101.5, 102.2, 101.4, 102.0))   # reclaim
        pb = compute_vwap_pullback(bars, direction="long")
        self.assertEqual(pb["state"], "confirmed")
        self.assertTrue(pb["confirmed_now"])
        self.assertEqual(pb["pullback_index"], 5)
        self.assertEqual(pb["trigger_index"], 6)
        self.assertIsNotNone(pb["trigger_vwap"])

    def test_confirmed_now_only_on_final_bar(self):
        bars = self._long_trend()
        bars.append(_bar(5, 102.4, 102.6, 101.0, 101.4))
        bars.append(_bar(6, 101.5, 102.2, 101.4, 102.0))
        bars.append(_bar(7, 102.0, 102.3, 101.9, 102.1))   # one bar later
        pb = compute_vwap_pullback(bars, direction="long")
        self.assertFalse(pb["confirmed_now"])

    def test_decisive_break_resets(self):
        bars = self._long_trend()
        bars.append(_bar(5, 102.4, 102.6, 101.0, 101.4))
        bars.append(_bar(6, 101.3, 101.4, 99.8, 100.0))    # close through band
        pb = compute_vwap_pullback(bars, direction="long")
        self.assertEqual(pb["state"], "seeking_trend")
        self.assertFalse(pb["confirmed_now"])

    def test_stale_setup_expires(self):
        bars = self._long_trend()
        bars.append(_bar(5, 102.4, 102.6, 101.0, 101.4))
        for i in range(6, 18):  # drift: never reclaims, never breaks
            bars.append(_bar(i, 101.39, 101.39, 101.39, 101.39))
        pb = compute_vwap_pullback(bars, direction="long", expiry_bars=10)
        self.assertEqual(pb["state"], "expired")
        self.assertFalse(pb["confirmed_now"])

    def test_short_mirror_confirmation(self):
        bars = [_bar(i, 100 - 0.5 * i, 100.1 - 0.5 * i, 99.4 - 0.5 * i,
                     99.5 - 0.5 * i) for i in range(5)]
        bars.append(_bar(5, 97.6, 98.8, 97.4, 97.8))       # high re-touches
        bars.append(_bar(6, 97.7, 97.9, 97.3, 97.4))       # close back below
        pb = compute_vwap_pullback(bars, direction="short")
        self.assertEqual(pb["state"], "confirmed")
        self.assertTrue(pb["confirmed_now"])
        self.assertEqual(pb["trigger_index"], 6)

    def test_bad_inputs_fail_open(self):
        self.assertEqual(compute_vwap_pullback([], direction="long"), {})
        self.assertEqual(compute_vwap_pullback(None, direction="long"), {})
        self.assertEqual(
            compute_vwap_pullback(self._long_trend(), direction="up"), {})


class TestIntradayFeatureKeys(unittest.TestCase):
    def test_rally_emits_ema_stack(self):
        feats = compute_intraday_features(_rally(30))
        self.assertIn("ema_9", feats)
        self.assertIn("ema_20", feats)
        self.assertEqual(feats["ema_9_over_20"], 1)
        # 50-EMA still warming up at 30 bars: omitted, not guessed.
        self.assertNotIn("ema_50", feats)
        self.assertNotIn("price_vs_ema_50", feats)
        # Pullback flags present (session has usable VWAP), 0/1 valued.
        self.assertIn("vwap_pullback_long", feats)
        self.assertIn("vwap_pullback_short", feats)
        self.assertIn(feats["vwap_pullback_long"], (0, 1))

    def test_downtrend_ema_cross_sign(self):
        feats = compute_intraday_features(_downtrend(30))
        self.assertEqual(feats["ema_9_over_20"], -1)

    def test_ema50_present_with_enough_bars(self):
        feats = compute_intraday_features(_rally(60))
        self.assertIn("ema_50", feats)
        self.assertEqual(feats["price_vs_ema_50"], 1)

    def test_junk_input_fails_open(self):
        self.assertEqual(compute_intraday_features("x"), {})
        self.assertEqual(compute_intraday_features(None), {})


class TestPrintFeatures(unittest.TestCase):
    def test_quote_rule_classification(self):
        tape = [
            {"ts": NOW - 5, "price": 10.05, "size": 500,
             "bid": 10.00, "ask": 10.05},   # at ask -> buy
            {"ts": NOW - 4, "price": 10.00, "size": 100,
             "bid": 10.00, "ask": 10.05},   # at bid -> sell
            {"ts": NOW - 3, "price": 10.05, "size": 150,
             "bid": 10.00, "ask": 10.05},   # at ask -> buy
        ]
        pf = compute_print_features(tape, now_ts=NOW)
        self.assertEqual(pf["status"], "ok")
        self.assertEqual(pf["buy_volume"], 650.0)
        self.assertEqual(pf["sell_volume"], 100.0)
        self.assertAlmostEqual(pf["print_imbalance"], 550.0 / 750.0, places=5)
        self.assertEqual(pf["n_classified"], 3)

    def test_tick_rule_fallback(self):
        tape = [
            {"ts": NOW - 6, "price": 10.00, "size": 100},  # no prior: skipped
            {"ts": NOW - 5, "price": 10.05, "size": 200},  # up-tick -> buy
            {"ts": NOW - 4, "price": 10.00, "size": 50},   # down-tick -> sell
        ]
        pf = compute_print_features(tape, now_ts=NOW)
        self.assertEqual(pf["status"], "ok")
        self.assertEqual(pf["n_classified"], 2)
        self.assertEqual(pf["buy_volume"], 200.0)
        self.assertEqual(pf["sell_volume"], 50.0)

    def test_stale_tape(self):
        pf = compute_print_features(
            [{"ts": NOW - 120, "price": 10.0, "size": 100}], now_ts=NOW)
        self.assertEqual(pf["status"], "stale")

    def test_missing_and_future(self):
        self.assertEqual(compute_print_features([], now_ts=NOW)["status"],
                         "missing")
        pf = compute_print_features(
            [{"ts": NOW + 5, "price": 10.0, "size": 100}], now_ts=NOW)
        self.assertEqual(pf["status"], "missing")  # future print: no evidence
        self.assertEqual(
            compute_print_features([{"ts": NOW - 1, "price": 10.0,
                                     "size": 100}], now_ts=None)["status"],
            "missing")


class TestBookAssessment(unittest.TestCase):
    def _book(self, **kw):
        book = {
            "ts": NOW - 1,
            "bids": [{"price": {"amount": "10.00"}, "quantity": "900"},
                     {"price": {"amount": "9.95"}, "quantity": "700"}],
            "asks": [{"price": {"amount": "10.05"}, "quantity": "300"},
                     {"price": {"amount": "10.10"}, "quantity": "200"}],
        }
        book.update(kw)
        return book

    def test_fresh_book_ok(self):
        bf = assess_book(self._book(), now_ts=NOW)
        self.assertEqual(bf["status"], "ok")
        self.assertAlmostEqual(bf["orderbook_imbalance"],
                               (1600.0 - 500.0) / 2100.0, places=5)
        self.assertEqual(bf["orderbook_levels"], 2)

    def test_unstamped_book_can_never_verify(self):
        bf = assess_book(self._book(ts=None), now_ts=NOW)
        self.assertEqual(bf["status"], "stale")
        self.assertEqual(bf["reason"], "no_timestamp")
        bf2 = assess_book(self._book(), now_ts=None)
        self.assertEqual(bf2["status"], "stale")

    def test_old_book_stale(self):
        bf = assess_book(self._book(ts=NOW - 60), now_ts=NOW)
        self.assertEqual(bf["status"], "stale")
        self.assertEqual(bf["reason"], "age_exceeded")

    def test_crossed_book(self):
        crossed = self._book(
            bids=[{"price": {"amount": "10.10"}, "quantity": "900"}])
        self.assertEqual(assess_book(crossed, now_ts=NOW)["status"], "crossed")

    def test_missing_side_and_no_book(self):
        self.assertEqual(assess_book(self._book(asks=[]),
                                     now_ts=NOW)["status"], "missing")
        self.assertEqual(assess_book(None, now_ts=NOW)["status"], "missing")

    def test_top_of_book(self):
        q = {"ts": NOW - 1, "bid": 10.0, "ask": 10.05,
             "bid_size": 300, "ask_size": 100}
        tf = assess_top_of_book(q, now_ts=NOW)
        self.assertEqual(tf["status"], "ok")
        self.assertAlmostEqual(tf["tob_imbalance"], 0.5)
        self.assertEqual(
            assess_top_of_book(dict(q, bid=10.10), now_ts=NOW)["status"],
            "crossed")
        self.assertEqual(
            assess_top_of_book(dict(q, bid_size=0), now_ts=NOW)["status"],
            "missing")


class TestConfirmOrderFlow(unittest.TestCase):
    def setUp(self):
        self.buy_tape = compute_print_features(
            [{"ts": NOW - 3, "price": 10.05, "size": 500,
              "bid": 10.00, "ask": 10.05},
             {"ts": NOW - 2, "price": 10.00, "size": 100,
              "bid": 10.00, "ask": 10.05}], now_ts=NOW)
        self.fresh_book = assess_book(
            {"ts": NOW - 1,
             "bids": [{"price": {"amount": "10.00"}, "quantity": "900"}],
             "asks": [{"price": {"amount": "10.05"}, "quantity": "300"}]},
            now_ts=NOW)

    def test_prints_confirm_long_not_short(self):
        long_r = confirm_order_flow("long", variant="prints_only",
                                    prints_feats=self.buy_tape)
        self.assertTrue(long_r["applicable"])
        self.assertTrue(long_r["confirmed"])
        short_r = confirm_order_flow("short", variant="prints_only",
                                     prints_feats=self.buy_tape)
        self.assertTrue(short_r["applicable"])
        self.assertFalse(short_r["confirmed"])

    def test_verified_depth_confirms(self):
        r = confirm_order_flow("long", variant="verified_depth",
                               book_feats=self.fresh_book)
        self.assertTrue(r["applicable"])
        self.assertTrue(r["confirmed"])

    def test_unqualified_inputs_not_applied(self):
        stale = dict(self.fresh_book, status="stale")
        r = confirm_order_flow("long", variant="verified_depth",
                               book_feats=stale)
        self.assertFalse(r["applicable"])
        self.assertEqual(r["reason"], "input_stale")
        self.assertIsNone(r["confirmed"])
        r2 = confirm_order_flow("long", variant="prints_only",
                                prints_feats=None)
        self.assertEqual(r2["reason"], "input_missing")
        self.assertIsNone(r2["confirmed"])

    def test_bad_arguments_and_no_imbalance(self):
        r = confirm_order_flow("up", variant="prints_only",
                               prints_feats=self.buy_tape)
        self.assertFalse(r["applicable"])
        self.assertEqual(r["reason"], "bad_arguments")
        # ok tape with zero classified volume -> imbalance None -> not applied
        lone = compute_print_features(
            [{"ts": NOW - 1, "price": 10.0, "size": 100}], now_ts=NOW)
        self.assertEqual(lone["status"], "ok")
        self.assertIsNone(lone["print_imbalance"])
        r3 = confirm_order_flow("long", variant="prints_only",
                                prints_feats=lone)
        self.assertEqual(r3["reason"], "no_imbalance")


class TestMatchedComparisonStudy(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.sessions, cls.flow = vps.make_fixture_sessions(2)
        cls.report = vps.run_study(cls.sessions, cls.flow)

    def test_report_shape_and_labels(self):
        rep = self.report
        self.assertEqual(rep["n_sessions"], 2)
        self.assertEqual(rep["diagnostic"], "underlying_returns")
        self.assertIn("data_gap", rep)
        for v in vps.VARIANT_ORDER:
            self.assertIn("calls", rep["variants"][v])
            self.assertIn("puts", rep["variants"][v])

    def test_ladder_monotone_and_counts(self):
        n = {v: self.report["variants"][v]["calls"]["n_signals"]
             for v in vps.VARIANT_ORDER}
        self.assertGreaterEqual(n["baseline"], 1)
        self.assertEqual(n["pullback"], 4)  # incl. warm-up confirmation
        self.assertTrue(n["pullback"] >= n["pullback_ema"]
                        >= n["pullback_ema_prints"]
                        >= n["pullback_ema_prints_depth"])
        self.assertEqual(n["pullback_ema"], 2)
        self.assertEqual(n["pullback_ema_prints"], 2)
        self.assertEqual(n["pullback_ema_prints_depth"], 2)
        # Separate hypothesis: 50 EMA never warms up in the fixture.
        self.assertEqual(n["pullback_ema50"], 0)

    def test_puts_reported_separately(self):
        self.assertEqual(
            self.report["variants"]["pullback"]["puts"]["n_signals"], 0)

    def test_fixed_exits_and_single_cost_charge(self):
        calls = self.report["variants"]["pullback_ema_prints_depth"]["calls"]
        self.assertEqual(calls["n_trades"], 2)
        self.assertEqual(calls["win_rate"], 1.0)
        self.assertEqual(calls["exit_reasons"]["target"], 2)
        self.assertAlmostEqual(calls["avg_return_pct"],
                               vps.TARGET_PCT - vps.COST_PCT, places=9)

    def test_unqualified_input_accounting(self):
        rep = vps.run_study(self.sessions, None)
        pf = rep["variants"]["pullback_ema_prints"]["calls"]
        self.assertEqual(pf["n_signals"], 0)
        self.assertEqual(pf["n_unqualified_inputs"], 2)
        stale_flow = {
            ts: dict(row, book=dict(row["book"], ts=row["book"]["ts"] - 999))
            for ts, row in self.flow.items()}
        rep2 = vps.run_study(self.sessions, stale_flow)
        df = rep2["variants"]["pullback_ema_prints_depth"]["calls"]
        self.assertEqual(df["n_signals"], 0)
        self.assertEqual(df["n_unqualified_inputs"], 2)

    def test_same_bar_stop_is_conservative(self):
        bars = [vps.StudyBar(60.0 * i, 100.0, 100.05, 99.0, 99.2, 1000.0)
                for i in range(3)]
        tr = vps.simulate_trades(
            bars, [0], "long",
            vps.StudyConfig(target_pct=0.001, stop_pct=0.001))
        self.assertEqual(len(tr), 1)
        self.assertEqual(tr[0]["reason"], "stop")

    def test_short_simulation(self):
        bars = [vps.StudyBar(60.0 * i, 100.0 - 0.3 * i, 100.1 - 0.3 * i,
                             99.8 - 0.3 * i, 99.9 - 0.3 * i, 1000.0)
                for i in range(10)]
        tr = vps.simulate_trades(bars, [0], "short", vps.StudyConfig())
        self.assertEqual(len(tr), 1)
        self.assertGreater(tr[0]["return_pct"], 0)

    def test_jsonl_round_trip(self):
        tmpdir = tempfile.mkdtemp()
        bars_path = os.path.join(tmpdir, "bars.jsonl")
        flow_path = os.path.join(tmpdir, "flow.jsonl")
        with open(bars_path, "w", encoding="utf-8") as fh:
            for bars in self.sessions:
                for b in bars:
                    fh.write(json.dumps({"ts": b.ts, "o": b.o, "h": b.h,
                                         "l": b.l, "c": b.c, "v": b.v}) + "\n")
            fh.write("not json\n")
            fh.write(json.dumps({"ts": "garbage", "o": 1}) + "\n")
        with open(flow_path, "w", encoding="utf-8") as fh:
            for row in self.flow.values():
                fh.write(json.dumps(row) + "\n")
        s2 = vps.load_bars_jsonl(bars_path)
        f2 = vps.load_flow_jsonl(flow_path)
        self.assertEqual(len(s2), 2)
        rep2 = vps.run_study(s2, f2)
        self.assertEqual(rep2["variants"], self.report["variants"])

    def test_empty_study_fails_open(self):
        rep = vps.run_study([])
        self.assertEqual(rep["n_sessions"], 0)
        self.assertEqual(
            rep["variants"]["baseline"]["calls"]["n_trades"], 0)
        self.assertEqual(vps.run_study([[None], []])["n_bars"], 0)


if __name__ == "__main__":
    unittest.main()
