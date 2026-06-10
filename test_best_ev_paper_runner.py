"""
Offline tests for Phase 10D — Paper trade the highest-EV structures.

No creds, no network, no broker. Covers the ten required areas:
  1. Default OFF does nothing       6. Rejects low EV/risk
  2. Opens valid top EV proposal    7. Rejects duplicate symbol
  3. Respects max trades per run    8. Stores EV context in paper position
  4. Rejects low recommendation     9. Telegram output
  5. Rejects low EV                10. No execution path touched

best_ev_paper_runner is SIMULATION ONLY: the last test class statically
proves the live execution modules (run_alpaca_intraday, smart_trader) do not
consume it, and that the runner itself never imports the live trader, the
network stack, or any order-placement API.
"""

import math
import os
import shutil
import tempfile
import unittest
from datetime import date, timedelta

import best_ev_paper_runner as bpr
from best_ev_paper_runner import (
    BestEVPaperConfig, passes_paper_thresholds, open_candidates,
    run_paper_from_best_ev, format_paper_run_report, log_decision,
    SKIP_BELOW_RECOMMENDATION, SKIP_EV_BELOW_MIN, SKIP_EV_PER_RISK_BELOW_MIN,
    SKIP_MAX_TRADES, SKIP_PROPOSAL_CHANGED, SKIP_PROPOSAL_ERROR,
    EV_CONTEXT_FIELDS, ADVISORY_CONTEXT_FIELDS, PAPER_FOOTER, LOG_TAG,
)
from ev_engine import (
    EVResult,
    STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE,
    STATUS_OK,
)
from spread_builder import SpreadLeg, SpreadProposal, BULLISH_PUT_CREDIT_SPREAD
from spread_paper_trader import SpreadPaperTrader, SpreadPaperConfig

HERE = os.path.dirname(os.path.abspath(__file__))
EXP = (date.today() + timedelta(days=30)).isoformat()


def row(sym="SPY", strategy=BULLISH_PUT_CREDIT_SPREAD, ev=10.0, pop=0.7,
        ratio=0.10, mp=50.0, ml=450.0, costs=9.0, score=60.0, days=30,
        rec=ACCEPT, status=STATUS_OK, vol_edge=None):
    """Hand-built EVResult row for threshold/open-loop/format tests."""
    return EVResult(symbol=sym, strategy=strategy, expected_value=ev,
                    probability_of_profit=pop, ev_per_dollar_risk=ratio,
                    max_profit=mp, max_loss=ml, estimated_costs=costs,
                    oracle_score=score, days=days, recommendation=rec,
                    status=status, volatility_edge=vol_edge)


def leg(action, otype, strike, bid, ask):
    return SpreadLeg(action=action, option_type=otype, strike=strike,
                     bid=bid, ask=ask, expiration=EXP)


def far_otm_bull_put(symbol, credit=1.50, mp=150.0, ml=350.0, be=78.5):
    """A deep-OTM fat-credit bull put -> PoP ~ 1 -> genuinely EV-positive.

    End-to-end via ev_engine this scores STRONG_ACCEPT with EV ~ +$122 and
    EV/risk ~ 0.35, so it passes the default paper thresholds.
    """
    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol=symbol,
        legs=[leg("sell", "put", 80, credit + 0.40, credit + 0.50),
              leg("buy", "put", 75, 0.40, 0.50)],
        net_credit_or_debit=credit, max_profit=mp, max_loss=ml,
        breakeven=be, width=5.0, oracle_score=70.0)


class _FakeTrader:
    """Duck-typed trader: propose_spread + get_price_history (no network)."""

    def __init__(self, symbol, proposal):
        self.symbol = symbol
        self._proposal = proposal

    def propose_spread(self, symbol):
        return self._proposal

    def get_price_history(self, symbol, days=130):
        return [100.0 + math.sin(i / 3.0) for i in range(130)]


def make_factory(proposals):
    """trader_factory over a {symbol: proposal} dict; unknown symbol raises."""
    def factory(symbol):
        if symbol not in proposals:
            raise RuntimeError(f"no data for {symbol}")
        return _FakeTrader(symbol, proposals[symbol])
    return factory


class _FakeLoader:
    def __init__(self, vals):
        self.v = vals

    def get_str(self, k, d=""):
        return str(self.v.get(k, d))

    def get_int(self, k, d=0):
        return int(self.v.get(k, d))

    def get_bool(self, k, d=False):
        val = self.v.get(k, d)
        if isinstance(val, bool):
            return val
        return str(val).strip().lower() in ("1", "true", "yes", "on")

    def get_float(self, k, d=0.0):
        return float(self.v.get(k, d))


class _TmpCwdCase(unittest.TestCase):
    """Run inside a temp dir so attribution/cache side-files never touch the
    repo, with a paper trader writing to temp JSON files."""

    def setUp(self):
        self._old_cwd = os.getcwd()
        self.tmp = tempfile.mkdtemp()
        os.chdir(self.tmp)
        self.pt = SpreadPaperTrader(SpreadPaperConfig(
            enabled=True, min_oracle_score=0.0,
            positions_file=os.path.join(self.tmp, "pos.json"),
            trades_file=os.path.join(self.tmp, "trades.json")))

    def tearDown(self):
        os.chdir(self._old_cwd)
        shutil.rmtree(self.tmp, ignore_errors=True)


def enabled_cfg(**kw):
    return BestEVPaperConfig(enabled=True, **kw)


# --------------------------------------------------------------------------- #
# 1. Default OFF does nothing
# --------------------------------------------------------------------------- #
class TestDefaultOff(unittest.TestCase):
    def test_config_defaults(self):
        cfg = BestEVPaperConfig()
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.max_trades_per_run, 3)
        self.assertEqual(cfg.min_recommendation, ACCEPT)
        self.assertAlmostEqual(cfg.min_ev_per_risk, 0.05)
        self.assertAlmostEqual(cfg.min_ev, 0.00)

    def test_from_env_defaults_off(self):
        cfg = BestEVPaperConfig.from_env(loader=_FakeLoader({}))
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.max_trades_per_run, 3)
        self.assertEqual(cfg.min_recommendation, ACCEPT)

    def test_from_env_reads_keys(self):
        cfg = BestEVPaperConfig.from_env(loader=_FakeLoader({
            "ENABLE_BEST_EV_PAPER_TRADING": "true",
            "BEST_EV_PAPER_MAX_TRADES_PER_RUN": "5",
            "BEST_EV_PAPER_MIN_RECOMMENDATION": "strong_accept",
            "BEST_EV_PAPER_MIN_EV_PER_RISK": "0.10",
            "BEST_EV_PAPER_MIN_EV": "1.5",
        }))
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.max_trades_per_run, 5)
        self.assertEqual(cfg.min_recommendation, STRONG_ACCEPT)
        self.assertAlmostEqual(cfg.min_ev_per_risk, 0.10)
        self.assertAlmostEqual(cfg.min_ev, 1.5)

    def test_from_env_invalid_tier_falls_back(self):
        cfg = BestEVPaperConfig.from_env(loader=_FakeLoader(
            {"BEST_EV_PAPER_MIN_RECOMMENDATION": "BANANAS"}))
        self.assertEqual(cfg.min_recommendation, ACCEPT)

    def test_disabled_run_does_nothing(self):
        calls = []

        def factory(symbol):  # must never be invoked when disabled
            calls.append(symbol)
            raise AssertionError("factory called while disabled")

        summary = run_paper_from_best_ev(
            "AAA,BBB", factory, config=BestEVPaperConfig(enabled=False))
        self.assertFalse(summary["enabled"])
        self.assertEqual(summary["scanned"], 0)
        self.assertEqual(summary["candidates"], 0)
        self.assertEqual(summary["opened"], [])
        self.assertEqual(summary["skipped"], [])
        self.assertEqual(calls, [])

    def test_no_factory_fails_open(self):
        summary = run_paper_from_best_ev("AAA", None, config=enabled_cfg())
        self.assertEqual(summary["opened"], [])
        self.assertEqual(summary["skipped"][0]["reason"], "no_trader_factory")


# --------------------------------------------------------------------------- #
# 2. Opens valid top EV proposal
# --------------------------------------------------------------------------- #
class TestOpensTopCandidate(_TmpCwdCase):
    def test_end_to_end_opens_simulated_trade(self):
        proposals = {"AAA": far_otm_bull_put("AAA")}
        summary = run_paper_from_best_ev(
            "AAA", make_factory(proposals), config=enabled_cfg(),
            paper_trader=self.pt)
        self.assertTrue(summary["enabled"])
        self.assertEqual(summary["scanned"], 1)
        self.assertEqual(summary["candidates"], 1)
        self.assertEqual(len(summary["opened"]), 1)
        pos = summary["opened"][0]
        self.assertEqual(pos["symbol"], "AAA")
        self.assertEqual(pos["strategy"], BULLISH_PUT_CREDIT_SPREAD)
        # ... and it landed in the simulator's positions file.
        saved = self.pt.load_positions()
        self.assertEqual(len(saved), 1)
        self.assertEqual(saved[0]["id"], pos["id"])

    def test_best_candidate_opened_first(self):
        proposals = {"AAA": far_otm_bull_put("AAA", credit=1.50, mp=150.0,
                                             ml=350.0),
                     "BBB": far_otm_bull_put("BBB", credit=0.60, mp=60.0,
                                             ml=440.0, be=79.4)}
        summary = run_paper_from_best_ev(
            "AAA,BBB", make_factory(proposals),
            config=enabled_cfg(max_trades_per_run=1), paper_trader=self.pt)
        self.assertEqual([p["symbol"] for p in summary["opened"]], ["AAA"])

    def test_proposal_changed_between_rank_and_open(self):
        # Ranked as bull put, but the trader now proposes something else.
        changed = far_otm_bull_put("AAA")
        changed.strategy_name = "bear_call_credit_spread"
        opened, skipped = open_candidates(
            [row(sym="AAA")], make_factory({"AAA": changed}), self.pt,
            enabled_cfg())
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], SKIP_PROPOSAL_CHANGED)

    def test_proposal_error_is_skipped_not_fatal(self):
        opened, skipped = open_candidates(
            [row(sym="ZZZ")], make_factory({}), self.pt, enabled_cfg())
        self.assertEqual(opened, [])
        self.assertTrue(skipped[0]["reason"].startswith(SKIP_PROPOSAL_ERROR))


# --------------------------------------------------------------------------- #
# 3. Respects max trades per run
# --------------------------------------------------------------------------- #
class TestMaxTradesPerRun(_TmpCwdCase):
    def test_caps_opens_and_reports_rest_skipped(self):
        rows = [row(sym=s) for s in ("AAA", "BBB", "CCC")]
        proposals = {s: far_otm_bull_put(s) for s in ("AAA", "BBB", "CCC")}
        opened, skipped = open_candidates(
            rows, make_factory(proposals), self.pt,
            enabled_cfg(max_trades_per_run=1))
        self.assertEqual(len(opened), 1)
        self.assertEqual(opened[0]["symbol"], "AAA")
        self.assertEqual([(r["symbol"], r["reason"]) for r in skipped],
                         [("BBB", SKIP_MAX_TRADES), ("CCC", SKIP_MAX_TRADES)])

    def test_zero_max_trades_opens_nothing(self):
        opened, skipped = open_candidates(
            [row(sym="AAA")], make_factory({"AAA": far_otm_bull_put("AAA")}),
            self.pt, enabled_cfg(max_trades_per_run=0))
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], SKIP_MAX_TRADES)
        self.assertEqual(self.pt.load_positions(), [])


# --------------------------------------------------------------------------- #
# 4. Rejects low recommendation
# --------------------------------------------------------------------------- #
class TestRejectsLowRecommendation(_TmpCwdCase):
    def test_threshold_check(self):
        cfg = enabled_cfg(min_recommendation=ACCEPT)
        for rec in (NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE):
            self.assertEqual(passes_paper_thresholds(row(rec=rec), cfg),
                             SKIP_BELOW_RECOMMENDATION)
        for rec in (ACCEPT, STRONG_ACCEPT):
            self.assertIsNone(passes_paper_thresholds(row(rec=rec), cfg))

    def test_open_loop_skips_low_recommendation(self):
        opened, skipped = open_candidates(
            [row(sym="AAA", rec=NEUTRAL)],
            make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], SKIP_BELOW_RECOMMENDATION)
        self.assertEqual(self.pt.load_positions(), [])


# --------------------------------------------------------------------------- #
# 5. Rejects low EV
# --------------------------------------------------------------------------- #
class TestRejectsLowEV(_TmpCwdCase):
    def test_threshold_check(self):
        cfg = enabled_cfg(min_ev=0.0)
        self.assertEqual(passes_paper_thresholds(row(ev=-5.0), cfg),
                         SKIP_EV_BELOW_MIN)
        self.assertEqual(passes_paper_thresholds(row(ev=None), cfg),
                         SKIP_EV_BELOW_MIN)
        self.assertIsNone(passes_paper_thresholds(row(ev=0.0), cfg))

    def test_open_loop_skips_low_ev(self):
        opened, skipped = open_candidates(
            [row(sym="AAA", ev=-1.0)],
            make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], SKIP_EV_BELOW_MIN)


# --------------------------------------------------------------------------- #
# 6. Rejects low EV/risk
# --------------------------------------------------------------------------- #
class TestRejectsLowEVPerRisk(_TmpCwdCase):
    def test_threshold_check(self):
        cfg = enabled_cfg(min_ev_per_risk=0.05)
        self.assertEqual(passes_paper_thresholds(row(ratio=0.01), cfg),
                         SKIP_EV_PER_RISK_BELOW_MIN)
        self.assertEqual(passes_paper_thresholds(row(ratio=None), cfg),
                         SKIP_EV_PER_RISK_BELOW_MIN)
        self.assertIsNone(passes_paper_thresholds(row(ratio=0.05), cfg))

    def test_open_loop_skips_low_ratio(self):
        opened, skipped = open_candidates(
            [row(sym="AAA", ratio=0.01)],
            make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], SKIP_EV_PER_RISK_BELOW_MIN)


# --------------------------------------------------------------------------- #
# 7. Rejects duplicate symbol
# --------------------------------------------------------------------------- #
class TestRejectsDuplicateSymbol(_TmpCwdCase):
    def test_existing_open_position_blocks_second_open(self):
        first = self.pt.open_position(far_otm_bull_put("AAA"))
        self.assertTrue(first["allowed"])
        opened, skipped = open_candidates(
            [row(sym="AAA")], make_factory({"AAA": far_otm_bull_put("AAA")}),
            self.pt, enabled_cfg())
        self.assertEqual(opened, [])
        self.assertEqual(skipped[0]["reason"], "duplicate_position")
        self.assertEqual(len(self.pt.load_positions()), 1)

    def test_same_run_never_doubles_a_symbol(self):
        # Two ranked rows for the same symbol in one run: second is rejected
        # by the simulator's duplicate check.
        rows = [row(sym="AAA", ratio=0.30), row(sym="AAA", ratio=0.10)]
        opened, skipped = open_candidates(
            rows, make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        self.assertEqual(len(opened), 1)
        self.assertEqual(skipped[0]["reason"], "duplicate_position")


# --------------------------------------------------------------------------- #
# 8. Stores EV context in the spread paper position
# --------------------------------------------------------------------------- #
class TestStoresEVContext(_TmpCwdCase):
    def test_ev_fields_on_position_and_persisted(self):
        r = row(sym="AAA", ev=18.40, ratio=0.23, pop=0.71, costs=5.0,
                rec=STRONG_ACCEPT, days=30, vol_edge=0.12)
        opened, skipped = open_candidates(
            [r], make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        self.assertEqual(len(opened), 1)
        pos = opened[0]
        # The EV belief at entry is stamped onto the simulated position.
        self.assertAlmostEqual(pos["expected_value"], 18.40)
        self.assertAlmostEqual(pos["ev_per_dollar_risk"], 0.23)
        self.assertAlmostEqual(pos["probability_of_profit"], 0.71)
        self.assertEqual(pos["ev_recommendation"], STRONG_ACCEPT)
        self.assertAlmostEqual(pos["estimated_costs"], 5.0)
        # DTE / vol edge travel through open_position's context.
        self.assertEqual(pos.get("dte"), 30)
        self.assertAlmostEqual(pos.get("volatility_edge"), 0.12)
        # Advisory snapshot fields are copied back onto the record.
        for field_name in ADVISORY_CONTEXT_FIELDS:
            self.assertIn(field_name, pos)
        self.assertIsNotNone(pos["advisory_recommendation"])
        # The enriched record (not the bare one) is what got persisted.
        saved = self.pt.load_positions()[0]
        for field_name in EV_CONTEXT_FIELDS:
            self.assertEqual(saved.get(field_name), pos.get(field_name),
                             f"persisted row missing {field_name}")

    def test_attribution_snapshot_is_ev_aware(self):
        r = row(sym="AAA", ev=18.40, ratio=0.23, pop=0.71, costs=5.0,
                rec=STRONG_ACCEPT)
        opened, _ = open_candidates(
            [r], make_factory({"AAA": far_otm_bull_put("AAA")}), self.pt,
            enabled_cfg())
        import advisory_attribution as aa
        records = aa.load_snapshots(aa.DEFAULT_ATTRIBUTION_FILE)
        snap = next(rec for rec in records
                    if rec.get("trade_id") == opened[0]["id"])
        self.assertAlmostEqual(snap["expected_value"], 18.40)
        self.assertAlmostEqual(snap["ev_per_dollar_risk"], 0.23)
        self.assertAlmostEqual(snap["probability_of_profit"], 0.71)
        self.assertEqual(snap["ev_recommendation"], STRONG_ACCEPT)


# --------------------------------------------------------------------------- #
# 9. Telegram output (+ log line format)
# --------------------------------------------------------------------------- #
class TestTelegramOutput(unittest.TestCase):
    def test_disabled_report(self):
        text = format_paper_run_report({"enabled": False})
        self.assertIn("Best EV Paper Run", text)
        self.assertIn("disabled", text)
        self.assertIn("ENABLE_BEST_EV_PAPER_TRADING", text)
        self.assertIn(PAPER_FOOTER, text)

    def test_enabled_report_sections(self):
        summary = {
            "enabled": True, "scanned": 3, "candidates": 2,
            "opened": [{"symbol": "NVDA",
                        "strategy": BULLISH_PUT_CREDIT_SPREAD,
                        "expected_value": 121.92,
                        "ev_per_dollar_risk": 0.35}],
            "skipped": [{"symbol": "QQQ", "strategy": None,
                         "reason": SKIP_EV_BELOW_MIN}],
        }
        text = format_paper_run_report(summary)
        self.assertIn("Scanned: 3 symbol(s)", text)
        self.assertIn("Candidates: 2", text)
        self.assertIn("Opened: 1 simulated trade(s)", text)
        self.assertIn("*Opened:*", text)
        self.assertIn("1. NVDA", text)
        self.assertIn("EV +$121.92", text)
        self.assertIn("EV/Risk 0.35", text)
        self.assertIn("*Skipped:*", text)
        self.assertIn(f"QQQ — {SKIP_EV_BELOW_MIN}", text)
        self.assertIn(PAPER_FOOTER, text)

    def test_no_opened_no_section(self):
        text = format_paper_run_report(
            {"enabled": True, "scanned": 1, "candidates": 0,
             "opened": [], "skipped": []})
        self.assertNotIn("*Opened:*", text)
        self.assertNotIn("*Skipped:*", text)
        self.assertIn("Opened: 0 simulated trade(s)", text)
        self.assertIn(PAPER_FOOTER, text)

    def test_log_decision_format(self):
        line = log_decision(row(sym="NVDA", ev=121.92, ratio=0.35,
                                rec=STRONG_ACCEPT),
                            "opened", "opened")
        self.assertTrue(line.startswith(LOG_TAG))
        for token in ("symbol=NVDA", "strategy=", "expected_value=121.92",
                      "ev_per_dollar_risk=0.35",
                      f"recommendation={STRONG_ACCEPT}",
                      "action=opened", "reason=opened"):
            self.assertIn(token, line)

    def test_telegram_bot_wires_the_command(self):
        src_path = os.path.join(HERE, "telegram_bot.py")
        with open(src_path, "r", encoding="utf-8") as fh:
            src = fh.read()
        self.assertIn("BEST_EV_PAPER_RUN", src)
        self.assertIn("def best_ev_paper_run", src)


# --------------------------------------------------------------------------- #
# 10. No execution path touched
# --------------------------------------------------------------------------- #
class TestNoExecutionPathTouched(unittest.TestCase):
    def _read(self, name):
        with open(os.path.join(HERE, name), "r", encoding="utf-8") as fh:
            return fh.read()

    def test_live_modules_do_not_consume_the_runner(self):
        for name in ("run_alpaca_intraday.py", "smart_trader.py"):
            src = self._read(name)
            self.assertNotIn("best_ev_paper_runner", src,
                             f"{name} must not import the paper runner")

    def test_runner_never_imports_live_trader_or_network(self):
        src = self._read("best_ev_paper_runner.py")
        for banned in ("import smart_trader", "from smart_trader",
                       "import requests", "place_order", "submit_order"):
            self.assertNotIn(banned, src)
        self.assertNotIn("alpaca", src.lower(),
                         "runner must not reference the broker API at all")

    def test_runner_only_writes_through_the_paper_simulator(self):
        # The only open path is SpreadPaperTrader.open_position.
        src = self._read("best_ev_paper_runner.py")
        self.assertIn("paper_trader.open_position", src)
        self.assertNotIn("place_order_with_stops", src)


if __name__ == "__main__":
    unittest.main(verbosity=2)
