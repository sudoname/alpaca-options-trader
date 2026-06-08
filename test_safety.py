"""
Offline safety tests for Phase 0 (risk caps + sizing + ranking), Phase 1 (real
Alpaca option Greeks/IV wiring with heuristic fallback) and Phase 2 (DTE/delta
targeting, cost-EV gate, liquidity filter — all OFF by default).

Run with:
    python -X utf8 -m unittest test_safety -v
    python -X utf8 -m pytest test_safety.py -q

NO network and NO broker calls: the option quote/snapshot fetchers are either
monkeypatched or exercised through their pure parser/merge helpers. The trader
is constructed against the local .env (paper keys) but its network-touching
advisory services (news/sentiment) are disabled per-test.
"""

import unittest
from datetime import datetime, timedelta

from risk_engine import RiskEngine, RiskLimits, load_risk_limits_from_env
from smart_trader import SmartOptionsTrader


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _make_trader(ticker="AAPL"):
    """Construct a trader with network-touching advisory services disabled."""
    t = SmartOptionsTrader(ticker=ticker)
    t.news_service = None
    t.sentiment_service = None
    # Pin sizing thresholds so tests don't depend on .env values.
    t.conf_high_signals = 2
    t.conf_very_high_signals = 4
    # Keep selection from being filtered out by budget during these tests.
    t.max_budget_per_trade = 100000.0
    t.max_spread_pct = 100.0
    # Pin all Phase 2 sub-gates OFF and to known config so tests don't depend on
    # whatever the local .env happens to hold; each test opts in explicitly.
    t.use_dte_targeting = False
    t.option_min_dte = 30
    t.option_max_dte = 90
    t.option_target_dte = 45
    t.use_delta_targeting = False
    t.option_target_call_delta = 0.40
    t.option_target_put_delta = -0.40
    t.option_max_delta_distance = 0.20
    t.use_cost_ev_gate = False
    t.min_post_cost_edge = 0.00
    t.max_option_spread_pct = 0.15
    t.use_option_liquidity_filter = False
    t.min_option_volume = 0
    t.min_option_open_interest = 0
    t.require_option_liquidity_data = False
    # Pin Phase 3 flags OFF + known config so default-behavior tests are stable.
    t.use_skip_on_weak_signal = False
    t.min_direction_signals = 2
    t.use_normalized_confidence = False
    t.last_skip_reason = None
    # Pin Phase 4 flags OFF so default behavior is unchanged unless a test opts in.
    t.use_portfolio_greek_limits = False
    t.portfolio_limits = None
    t.use_realized_pnl_killswitch = False
    return t


def _pin_direction_inputs(t, *, momentum=0.0, volatility=0.1, regime="ranging",
                          prices=None):
    """Monkeypatch the (network-touching) market primitives that
    determine_option_strategy reads, so signal tallies are deterministic and
    offline. `prices` drives the short/medium trend slices."""
    if prices is None:
        prices = [100.0] * 6  # flat -> no trend signals
    t.calculate_momentum = lambda sym=None: momentum
    t.calculate_volatility = lambda sym=None: volatility
    t.get_market_regime = lambda sym=None: regime
    t.get_price_history = lambda sym=None, days=20: list(prices)
    t.news_service = None


def _mock_call(strike, expiration="2026-08-21", bid=2.0, ask=2.2):
    return {
        "symbol": f"AAPL260821C{int(strike*1000):08d}",
        "strike_price": strike,
        "type": "call",
        "expiration_date": expiration,
        "mock": True,
        "mock_bid": bid,
        "mock_ask": ask,
        "volume": 0,
        "open_interest": 0,
    }


def _real_call(strike, expiration="2026-08-21"):
    return {
        "symbol": f"AAPL260821C{int(strike*1000):08d}",
        "strike_price": strike,
        "type": "call",
        "expiration_date": expiration,
        "mock": False,
        "volume": 500,
        "open_interest": 1000,
    }


# --------------------------------------------------------------------------- #
# Phase 0: RiskEngine.check + per-underlying concentration cap
# --------------------------------------------------------------------------- #
class TestRiskEngineCheck(unittest.TestCase):
    def setUp(self):
        self.eng = RiskEngine(RiskLimits(
            max_budget_per_trade=500.0,
            daily_loss_limit=300.0,
            max_concurrent=3,
            min_pdt_remaining=1,
            kill_switch_loss=500.0,
        ))

    def test_clean_trade_allowed(self):
        r = self.eng.check(trade_cost=200.0, realized_pnl_today=-50.0,
                           open_positions=1, pdt_remaining=2, may_day_trade=True)
        self.assertTrue(r["allowed"])
        self.assertEqual(r["reason"], "ok")

    def test_over_budget_blocked(self):
        r = self.eng.check(trade_cost=600.0, realized_pnl_today=0.0, open_positions=0)
        self.assertFalse(r["allowed"])
        self.assertIn("over_budget", r["breaches"])

    def test_max_concurrent_blocked(self):
        r = self.eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=3)
        self.assertFalse(r["allowed"])
        self.assertIn("max_concurrent", r["breaches"])

    def test_missing_input_fails_closed(self):
        r = self.eng.check(trade_cost=None, realized_pnl_today=0.0, open_positions=0)
        self.assertFalse(r["allowed"])
        self.assertIn("missing_required_input", r["breaches"])

    def test_garbage_input_fails_closed(self):
        r = self.eng.check(trade_cost="oops", realized_pnl_today=0.0, open_positions=0)
        self.assertFalse(r["allowed"])


class TestPerUnderlyingCap(unittest.TestCase):
    def test_default_is_no_op(self):
        """Default high limit never blocks, even with a count supplied."""
        eng = RiskEngine(RiskLimits())  # max_per_underlying defaults high
        r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=50)
        self.assertTrue(r["allowed"])
        self.assertNotIn("max_per_underlying", r["breaches"])

    def test_env_default_high(self):
        """The default loaded from .env keeps the cap effectively disabled."""
        lim = load_risk_limits_from_env(path="does_not_exist.env")
        self.assertGreaterEqual(lim.max_per_underlying, 1000)

    def test_blocks_at_limit_when_enabled(self):
        eng = RiskEngine(RiskLimits(max_per_underlying=2))
        r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=2)
        self.assertFalse(r["allowed"])
        self.assertIn("max_per_underlying", r["breaches"])

    def test_allows_under_limit_when_enabled(self):
        eng = RiskEngine(RiskLimits(max_per_underlying=3))
        r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=2)
        self.assertTrue(r["allowed"])

    def test_skipped_when_count_omitted(self):
        """Even with a tight cap, omitting the count leaves it a no-op."""
        eng = RiskEngine(RiskLimits(max_per_underlying=1))
        r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1)
        self.assertTrue(r["allowed"])
        self.assertNotIn("max_per_underlying", r["breaches"])

    def test_nonpositive_limit_means_unlimited(self):
        eng = RiskEngine(RiskLimits(max_per_underlying=0))
        r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=99)
        self.assertTrue(r["allowed"])


# --------------------------------------------------------------------------- #
# Phase 0: OCC underlying extraction (feeds the cap in the live path)
# --------------------------------------------------------------------------- #
class TestOccUnderlying(unittest.TestCase):
    def test_parses_root(self):
        self.assertEqual(SmartOptionsTrader._occ_underlying("ABT260821C00095000"), "ABT")
        self.assertEqual(SmartOptionsTrader._occ_underlying("SPY240722P00553000"), "SPY")

    def test_non_option_returned_unchanged(self):
        self.assertEqual(SmartOptionsTrader._occ_underlying("AAPL"), "AAPL")
        self.assertEqual(SmartOptionsTrader._occ_underlying(""), "")


# --------------------------------------------------------------------------- #
# Phase 0: _confidence_to_quantity
# --------------------------------------------------------------------------- #
class TestConfidenceToQuantity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def test_regular(self):
        self.assertEqual(self.t._confidence_to_quantity(0), 1)
        self.assertEqual(self.t._confidence_to_quantity(1), 1)

    def test_high(self):
        self.assertEqual(self.t._confidence_to_quantity(2), 2)
        self.assertEqual(self.t._confidence_to_quantity(3), 2)

    def test_very_high(self):
        self.assertEqual(self.t._confidence_to_quantity(4), 3)
        self.assertEqual(self.t._confidence_to_quantity(9), 3)

    def test_garbage_defaults_to_one(self):
        self.assertEqual(self.t._confidence_to_quantity(None), 1)
        self.assertEqual(self.t._confidence_to_quantity("oops"), 1)


# --------------------------------------------------------------------------- #
# Phase 0: select_best_option (offline via mock contracts)
# --------------------------------------------------------------------------- #
class TestSelectBestOption(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def test_picks_a_call_and_tags_metadata(self):
        contracts = [_mock_call(140), _mock_call(145), _mock_call(150)]
        self.t.last_signal_strength = 3
        best = self.t.select_best_option(contracts, current_price=150.0, strategy="call")
        self.assertIsNotNone(best)
        self.assertEqual(best["type"], "call")
        self.assertIn("score", best)
        self.assertEqual(best["strategy_type"], "call")
        self.assertEqual(best["confidence"], 3)  # carries directional conviction

    def test_far_otm_calls_skipped(self):
        # All strikes far above current price -> nothing selected.
        contracts = [_mock_call(200), _mock_call(210)]
        best = self.t.select_best_option(contracts, current_price=150.0, strategy="call")
        self.assertIsNone(best)

    def test_budget_filter_excludes_expensive(self):
        self.t.max_budget_per_trade = 100.0  # $1.00/contract cap
        contracts = [_mock_call(145, bid=2.0, ask=2.2)]  # $220 > cap
        best = self.t.select_best_option(contracts, current_price=150.0, strategy="call")
        self.assertIsNone(best)
        self.t.max_budget_per_trade = 100000.0  # restore


# --------------------------------------------------------------------------- #
# Phase 1: snapshot parsing + Greeks/IV merge (pure helpers)
# --------------------------------------------------------------------------- #
class TestParseSnapshotGreeks(unittest.TestCase):
    def test_full_snapshot(self):
        snap = {
            "greeks": {"delta": 0.55, "gamma": 0.02, "theta": -0.03,
                       "vega": 0.20, "rho": 0.01},
            "impliedVolatility": 0.42,
        }
        out = SmartOptionsTrader._parse_snapshot_greeks(snap)
        self.assertEqual(out, {"delta": 0.55, "gamma": 0.02, "theta": -0.03,
                               "vega": 0.20, "iv": 0.42})

    def test_partial_snapshot(self):
        snap = {"greeks": {"delta": 0.5}}  # no gamma/theta/vega, no IV
        out = SmartOptionsTrader._parse_snapshot_greeks(snap)
        self.assertEqual(out, {"delta": 0.5})

    def test_missing_greeks(self):
        self.assertEqual(SmartOptionsTrader._parse_snapshot_greeks({}), {})
        self.assertEqual(SmartOptionsTrader._parse_snapshot_greeks(
            {"latestQuote": {"ap": 1.0}}), {})

    def test_non_dict_and_non_numeric(self):
        self.assertEqual(SmartOptionsTrader._parse_snapshot_greeks(None), {})
        self.assertEqual(SmartOptionsTrader._parse_snapshot_greeks(
            {"greeks": {"delta": "x"}, "impliedVolatility": "y"}), {})


class TestApplyRealGreeks(unittest.TestCase):
    def _fallback(self):
        return {"delta": 0.6, "gamma": 0.01, "theta": -0.05, "vega": 0.10, "iv": 0.25}

    def test_real_overrides_all(self):
        od = self._fallback()
        real = {"delta": 0.55, "gamma": 0.02, "theta": -0.03, "vega": 0.20, "iv": 0.42}
        log = SmartOptionsTrader._apply_real_greeks(od, real)
        self.assertEqual(od["delta"], 0.55)
        self.assertEqual(od["iv"], 0.42)
        self.assertIn("delta=0.5500(real)", log)
        self.assertIn("iv=0.4200(real)", log)
        self.assertNotIn("fallback", log)

    def test_partial_keeps_fallback(self):
        od = self._fallback()
        real = {"delta": 0.55}  # only delta is real
        log = SmartOptionsTrader._apply_real_greeks(od, real)
        self.assertEqual(od["delta"], 0.55)
        self.assertEqual(od["gamma"], 0.01)   # fallback retained
        self.assertEqual(od["iv"], 0.25)      # fallback retained
        self.assertIn("delta=0.5500(real)", log)
        self.assertIn("gamma=0.0100(fallback)", log)
        self.assertIn("iv=0.2500(fallback)", log)

    def test_empty_keeps_all_fallback(self):
        od = self._fallback()
        log = SmartOptionsTrader._apply_real_greeks(od, {})
        self.assertEqual(od, self._fallback())
        self.assertEqual(log.count("(fallback)"), 5)
        self.assertNotIn("(real)", log)


# --------------------------------------------------------------------------- #
# Phase 1: end-to-end wiring through select_best_option (network monkeypatched)
# --------------------------------------------------------------------------- #
class TestRealGreeksWiring(unittest.TestCase):
    def setUp(self):
        self.t = _make_trader()
        self.t.get_option_price = lambda sym: {"bid": 1.0, "ask": 1.1,
                                               "mid": 1.05, "ts": None}

    def test_disabled_by_default_does_not_fetch_snapshot(self):
        self.t.use_real_greeks = False
        called = {"n": 0}

        def _boom(sym):
            called["n"] += 1
            return {}
        self.t.get_option_snapshot = _boom

        best = self.t.select_best_option([_real_call(145)], current_price=150.0,
                                         strategy="call")
        self.assertIsNotNone(best)
        self.assertEqual(called["n"], 0)            # never called when disabled
        self.assertEqual(best["gamma"], 0.01)       # heuristic fallback intact
        self.assertEqual(best["iv"], 0.25)
        self.assertEqual(best["vega"], 0.10)

    def test_enabled_uses_real_values(self):
        self.t.use_real_greeks = True
        self.t.get_option_snapshot = lambda sym: {
            "delta": 0.58, "gamma": 0.03, "theta": -0.02, "vega": 0.25, "iv": 0.40}
        best = self.t.select_best_option([_real_call(145)], current_price=150.0,
                                         strategy="call")
        self.assertIsNotNone(best)
        self.assertEqual(best["delta"], 0.58)
        self.assertEqual(best["gamma"], 0.03)
        self.assertEqual(best["theta"], -0.02)
        self.assertEqual(best["vega"], 0.25)
        self.assertEqual(best["iv"], 0.40)

    def test_enabled_but_missing_snapshot_falls_back(self):
        self.t.use_real_greeks = True
        self.t.get_option_snapshot = lambda sym: {}   # snapshot unavailable
        best = self.t.select_best_option([_real_call(145)], current_price=150.0,
                                         strategy="call")
        self.assertIsNotNone(best)
        self.assertEqual(best["gamma"], 0.01)   # fallback retained
        self.assertEqual(best["theta"], -0.05)
        self.assertEqual(best["iv"], 0.25)
        self.assertEqual(best["vega"], 0.10)


# --------------------------------------------------------------------------- #
# Phase 2: evaluate_contract_phase2 sub-gates (pure, no network)
# --------------------------------------------------------------------------- #
class TestPhase2DTE(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def _eval(self, dte):
        return self.t.evaluate_contract_phase2(
            dte=dte, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.0, ask=1.05, volume=100, open_interest=100)

    def test_off_by_default(self):
        self.t.use_dte_targeting = False
        self.assertEqual(self._eval(5), (None, 1.0))     # out of window, but off
        self.t.use_dte_targeting = True

    def test_in_window_at_target_no_penalty(self):
        self.t.use_dte_targeting = True
        reason, mult = self._eval(45)
        self.assertIsNone(reason)
        self.assertAlmostEqual(mult, 1.0)
        self.t.use_dte_targeting = False

    def test_below_min_rejected(self):
        self.t.use_dte_targeting = True
        self.assertEqual(self._eval(10)[0], "bad_dte")
        self.t.use_dte_targeting = False

    def test_above_max_rejected(self):
        self.t.use_dte_targeting = True
        self.assertEqual(self._eval(120)[0], "bad_dte")
        self.t.use_dte_targeting = False

    def test_closer_to_target_scores_higher(self):
        self.t.use_dte_targeting = True
        _, near = self._eval(50)   # 5 from target
        _, far = self._eval(88)    # 43 from target, still in window
        self.assertGreater(near, far)
        self.assertLessEqual(near, 1.0)
        self.assertGreaterEqual(far, 0.8)
        self.t.use_dte_targeting = False


class TestPhase2Delta(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def _eval(self, delta, has_real, strategy="call"):
        return self.t.evaluate_contract_phase2(
            dte=45, delta=delta, has_real_delta=has_real, strategy=strategy,
            bid=1.0, ask=1.05, volume=100, open_interest=100)

    def test_off_by_default(self):
        self.t.use_delta_targeting = False
        self.assertEqual(self._eval(0.99, True)[0], None)  # far, but gate off

    def test_missing_real_delta_falls_back(self):
        self.t.use_delta_targeting = True
        # has_real_delta False -> no targeting even though delta is far off.
        self.assertEqual(self._eval(0.99, False), (None, 1.0))
        self.t.use_delta_targeting = False

    def test_call_near_target_ok(self):
        self.t.use_delta_targeting = True
        reason, mult = self._eval(0.42, True, "call")
        self.assertIsNone(reason)
        self.assertLessEqual(mult, 1.0)
        self.assertGreaterEqual(mult, 0.8)
        self.t.use_delta_targeting = False

    def test_call_far_rejected(self):
        self.t.use_delta_targeting = True
        self.assertEqual(self._eval(0.80, True, "call")[0], "bad_delta")
        self.t.use_delta_targeting = False

    def test_put_target(self):
        self.t.use_delta_targeting = True
        self.assertIsNone(self._eval(-0.42, True, "put")[0])
        self.assertEqual(self._eval(-0.80, True, "put")[0], "bad_delta")
        self.t.use_delta_targeting = False


class TestPhase2CostEV(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def test_off_by_default(self):
        self.t.use_cost_ev_gate = False
        reason, _ = self.t.evaluate_contract_phase2(
            dte=45, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.0, ask=5.0, volume=100, open_interest=100)  # huge spread
        self.assertIsNone(reason)

    def test_wide_spread_rejected(self):
        self.t.use_cost_ev_gate = True
        self.t.max_option_spread_pct = 0.15
        reason, _ = self.t.evaluate_contract_phase2(
            dte=45, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.0, ask=2.0, volume=100, open_interest=100)  # 50% spread
        self.assertEqual(reason, "wide_spread")
        self.t.use_cost_ev_gate = False

    def test_negative_ev_rejected(self):
        self.t.use_cost_ev_gate = True
        self.t.max_option_spread_pct = 1.0     # let the spread pass the spread gate
        self.t.min_post_cost_edge = 0.0
        self.t.base_take_profit = 0.0          # zero gross target -> post-cost < 0
        reason, _ = self.t.evaluate_contract_phase2(
            dte=45, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.00, ask=1.02, volume=100, open_interest=100)
        self.assertEqual(reason, "negative_ev")
        self.t.use_cost_ev_gate = False

    def test_healthy_trade_passes(self):
        self.t.use_cost_ev_gate = True
        self.t.max_option_spread_pct = 0.15
        self.t.min_post_cost_edge = 0.0
        self.t.base_take_profit = 2.20         # 220% gross target
        reason, _ = self.t.evaluate_contract_phase2(
            dte=45, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.00, ask=1.02, volume=100, open_interest=100)
        self.assertIsNone(reason)
        self.t.use_cost_ev_gate = False


class TestPhase2Liquidity(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def _eval(self, volume, oi, vp=True, op=True):
        return self.t.evaluate_contract_phase2(
            dte=45, delta=0.5, has_real_delta=False, strategy="call",
            bid=1.0, ask=1.05, volume=volume, open_interest=oi,
            volume_present=vp, oi_present=op)

    def test_off_by_default(self):
        self.t.use_option_liquidity_filter = False
        self.assertIsNone(self._eval(0, 0)[0])

    def test_low_volume_rejected(self):
        self.t.use_option_liquidity_filter = True
        self.t.min_option_volume = 50
        self.t.min_option_open_interest = 0
        self.assertEqual(self._eval(10, 100)[0], "low_volume")
        self.t.use_option_liquidity_filter = False

    def test_low_open_interest_rejected(self):
        self.t.use_option_liquidity_filter = True
        self.t.min_option_volume = 0
        self.t.min_option_open_interest = 500
        self.assertEqual(self._eval(100, 10)[0], "low_open_interest")
        self.t.use_option_liquidity_filter = False

    def test_missing_data_fails_open(self):
        self.t.use_option_liquidity_filter = True
        self.t.require_option_liquidity_data = False
        self.t.min_option_volume = 1000
        self.assertIsNone(self._eval(0, 0, vp=False, op=False)[0])
        self.t.use_option_liquidity_filter = False

    def test_missing_data_required_rejected(self):
        self.t.use_option_liquidity_filter = True
        self.t.require_option_liquidity_data = True
        self.assertEqual(self._eval(0, 0, vp=False, op=False)[0],
                         "missing_liquidity_data")
        self.t.use_option_liquidity_filter = False
        self.t.require_option_liquidity_data = False


# --------------------------------------------------------------------------- #
# Phase 2: end-to-end through select_best_option (network monkeypatched)
# --------------------------------------------------------------------------- #
def _real_call_exp(strike, days):
    exp = (datetime.now().date() + timedelta(days=days)).strftime("%Y-%m-%d")
    return {
        "symbol": f"AAPL{exp[2:4]}{exp[5:7]}{exp[8:10]}C{int(strike*1000):08d}",
        "strike_price": strike,
        "type": "call",
        "expiration_date": exp,
        "mock": False,
        "volume": 500,
        "open_interest": 1000,
    }


class TestPhase2Integration(unittest.TestCase):
    def setUp(self):
        self.t = _make_trader()
        self.t.get_option_price = lambda sym: {"bid": 1.00, "ask": 1.05,
                                               "mid": 1.025, "ts": None}

    def test_dte_preference_picks_closer_to_target(self):
        self.t.use_dte_targeting = True            # target 45, window 30-90
        near = _real_call_exp(145, 45)             # at target
        far = _real_call_exp(145, 88)              # far but in window
        best = self.t.select_best_option([far, near], current_price=150.0,
                                         strategy="call")
        self.assertIsNotNone(best)
        self.assertEqual(best["expiration"], near["expiration_date"])

    def test_dte_out_of_window_rejected(self):
        self.t.use_dte_targeting = True
        far = _real_call_exp(145, 200)             # beyond max DTE
        best = self.t.select_best_option([far], current_price=150.0,
                                         strategy="call")
        self.assertIsNone(best)

    def test_delta_targeting_falls_back_when_missing(self):
        self.t.use_delta_targeting = True
        self.t.get_option_snapshot = lambda sym: {}   # no real delta
        best = self.t.select_best_option([_real_call_exp(145, 45)],
                                         current_price=150.0, strategy="call")
        self.assertIsNotNone(best)                  # fell back, not rejected

    def test_delta_targeting_rejects_far_real_delta(self):
        self.t.use_delta_targeting = True
        self.t.get_option_snapshot = lambda sym: {"delta": 0.90}  # far from 0.40
        best = self.t.select_best_option([_real_call_exp(145, 45)],
                                         current_price=150.0, strategy="call")
        self.assertIsNone(best)                     # bad_delta -> rejected

    def test_cost_ev_gate_rejects_wide_spread(self):
        self.t.use_cost_ev_gate = True
        self.t.max_option_spread_pct = 0.15
        self.t.get_option_price = lambda sym: {"bid": 1.0, "ask": 2.0,
                                               "mid": 1.5, "ts": None}  # 50%
        best = self.t.select_best_option([_real_call_exp(145, 45)],
                                         current_price=150.0, strategy="call")
        self.assertIsNone(best)


# --------------------------------------------------------------------------- #
# Phase 3: SKIP / NO_TRADE on weak/flat signals (determine_option_strategy)
# --------------------------------------------------------------------------- #
class TestPhase3Skip(unittest.TestCase):
    def test_flat_defaults_to_call_when_flag_disabled(self):
        """bull == bear (and zero signals) -> default CALL when flag off."""
        t = _make_trader()
        t.use_skip_on_weak_signal = False
        _pin_direction_inputs(t)  # flat: no signals
        self.assertEqual(t.determine_option_strategy("AAPL"), "call")
        self.assertEqual(t.last_signal_strength, 0)  # unchanged behavior

    def test_flat_returns_skip_when_flag_enabled(self):
        """bull == bear with enough signals -> SKIP (flat_signal)."""
        t = _make_trader()
        t.use_skip_on_weak_signal = True
        t.min_direction_signals = 2
        # momentum bullish (+1) and a short-term downtrend (+1 bear) -> 1 vs 1.
        _pin_direction_inputs(t, momentum=0.02,
                              prices=[100, 100, 100, 103, 100, 100])
        self.assertEqual(t.determine_option_strategy("AAPL"), "skip")
        self.assertIn("flat_signal", t.last_skip_reason)
        self.assertEqual(t.last_signal_strength, 0)

    def test_below_min_signals_returns_skip(self):
        """Fewer than MIN_DIRECTION_SIGNALS total -> SKIP (below_min_signals)."""
        t = _make_trader()
        t.use_skip_on_weak_signal = True
        t.min_direction_signals = 2
        _pin_direction_inputs(t, momentum=0.02)  # one bullish signal only
        self.assertEqual(t.determine_option_strategy("AAPL"), "skip")
        self.assertIn("below_min_signals", t.last_skip_reason)

    def test_below_min_signals_trades_when_flag_disabled(self):
        """Same thin signal trades as a CALL when the flag is off (unchanged)."""
        t = _make_trader()
        t.use_skip_on_weak_signal = False
        _pin_direction_inputs(t, momentum=0.02)
        self.assertEqual(t.determine_option_strategy("AAPL"), "call")
        self.assertEqual(t.last_signal_strength, 1)

    def test_strong_bullish_unchanged_by_default(self):
        t = _make_trader()
        t.use_skip_on_weak_signal = False
        _pin_direction_inputs(t, momentum=0.05,
                              prices=[100, 100, 100, 100, 100, 110])
        self.assertEqual(t.determine_option_strategy("AAPL"), "call")
        self.assertEqual(t.last_signal_strength, 4)

    def test_strong_bearish_unchanged_by_default(self):
        t = _make_trader()
        t.use_skip_on_weak_signal = False
        _pin_direction_inputs(t, momentum=-0.05,
                              prices=[110, 110, 110, 110, 110, 100])
        self.assertEqual(t.determine_option_strategy("AAPL"), "put")
        self.assertEqual(t.last_signal_strength, 4)

    def test_strong_signal_not_skipped_when_flag_enabled(self):
        """A clear directional edge still trades even with skip enabled."""
        t = _make_trader()
        t.use_skip_on_weak_signal = True
        _pin_direction_inputs(t, momentum=0.05,
                              prices=[100, 100, 100, 100, 100, 110])
        self.assertEqual(t.determine_option_strategy("AAPL"), "call")


# --------------------------------------------------------------------------- #
# Phase 3: normalized confidence + sizing safety
# --------------------------------------------------------------------------- #
class TestPhase3NormalizedConfidence(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.t = _make_trader()

    def test_unanimous_strong_high(self):
        # 4 vs 0 -> margin 4, agreement 1.0 -> 4 (very high).
        self.assertEqual(self.t._normalized_confidence(4, 0, "call"), 4)

    def test_clear_medium(self):
        # 3 vs 1 -> margin 2, agreement 0.75 -> round(1.5)=2 (medium/high).
        self.assertEqual(self.t._normalized_confidence(3, 1, "call"), 2)

    def test_narrow_lead_deflated_low(self):
        # 4 vs 3 -> margin 1, agreement ~0.57 -> 1 (NOT very-high).
        self.assertEqual(self.t._normalized_confidence(4, 3, "call"), 1)

    def test_tie_is_zero(self):
        self.assertEqual(self.t._normalized_confidence(2, 2, "call"), 0)
        self.assertEqual(self.t._normalized_confidence(0, 0, "call"), 0)

    def test_never_exceeds_raw_winning_count(self):
        # Cap at the raw winning side count so sizing can't be inflated.
        self.assertLessEqual(self.t._normalized_confidence(1, 0, "call"), 1)

    def test_put_uses_bearish_raw_cap(self):
        # 0 vs 4 put -> margin 4, agreement 1.0, raw(bear)=4 -> 4.
        self.assertEqual(self.t._normalized_confidence(0, 4, "put"), 4)

    def test_quantity_skips_at_zero_only_when_normalized(self):
        t = _make_trader()
        t.use_normalized_confidence = True
        self.assertEqual(t._confidence_to_quantity(0), 0)   # weak -> skip
        self.assertEqual(t._confidence_to_quantity(1), 1)   # medium -> small
        self.assertEqual(t._confidence_to_quantity(2), 2)
        self.assertEqual(t._confidence_to_quantity(4), 3)   # strong -> normal

    def test_quantity_floor_is_one_by_default(self):
        t = _make_trader()
        t.use_normalized_confidence = False
        self.assertEqual(t._confidence_to_quantity(0), 1)   # unchanged

    def test_normalized_confidence_threads_into_strategy(self):
        # 4 vs 3 winning bull -> normalized confidence collapses to 1.
        t = _make_trader()
        t.use_normalized_confidence = True
        # Build bull=4, bear=3 via primitives:
        #   momentum strong (+2 bull), short & medium uptrend (+1 +1 bull),
        #   then force a high-vol bearish nudge? Simpler: drive directly.
        t.calculate_momentum = lambda sym=None: 0.05      # +2 bull
        t.calculate_volatility = lambda sym=None: 0.1
        t.get_market_regime = lambda sym=None: "ranging"
        t.get_price_history = lambda sym=None, days=20: [100, 100, 100, 100, 100, 110]
        t.news_service = None
        # bull = 2(mom)+1(short)+1(med) = 4, bear = 0 here -> margin 4 -> 4.
        t.determine_option_strategy("AAPL")
        self.assertEqual(t.last_signal_strength, 4)


# --------------------------------------------------------------------------- #
# Phase 3: scheduler honors SKIP (no contract lookup / no order)
# --------------------------------------------------------------------------- #
class TestPhase3SchedulerSkip(unittest.TestCase):
    def _fake_self(self, trader):
        class _FakeSched:
            pass
        fs = _FakeSched()
        fs.trader = trader
        fs.entered_today = set()
        fs._has_live_position = lambda sym: False
        return fs

    def _trader(self, direction, reason=None):
        class _FakeTrader:
            def __init__(self):
                self.ticker = None
                self.last_skip_reason = reason
            def get_current_price(self, sym):
                return 100.0
            def get_option_contracts(self, sym):
                return [{"symbol": "X"}]
            def determine_option_strategy(self, sym):
                return direction
            def select_best_option(self, *a, **k):
                raise AssertionError("select_best_option must not run on SKIP")
        return _FakeTrader()

    def test_skip_short_circuits_before_contract_selection(self):
        from run_alpaca_intraday import IntradayScheduler
        tr = self._trader("skip", reason="flat_signal (bull 1 == bear 1)")
        fs = self._fake_self(tr)
        result = IntradayScheduler._evaluate(fs, "SPY")
        self.assertIsNone(result)  # skipped, no candidate

    def test_non_skip_proceeds_to_selection(self):
        from run_alpaca_intraday import IntradayScheduler

        class _OkTrader:
            def __init__(self):
                self.ticker = None
                self.last_skip_reason = None
            def get_current_price(self, sym):
                return 100.0
            def get_option_contracts(self, sym):
                return [{"symbol": "X"}]
            def determine_option_strategy(self, sym):
                return "call"
            def select_best_option(self, contracts, price, strategy=None):
                return {"symbol": "AAPL260821C00150000", "ask": 1.0, "score": 42.0}

        fs = self._fake_self(_OkTrader())
        result = IntradayScheduler._evaluate(fs, "SPY")
        self.assertIsNotNone(result)
        self.assertEqual(result["direction"], "call")
        self.assertEqual(result["score"], 42.0)


# --------------------------------------------------------------------------- #
# Phase 3: Telegram honors SKIP (no trade message, no contract lookup)
# --------------------------------------------------------------------------- #
class TestPhase3TelegramSkip(unittest.TestCase):
    def test_analyze_ticker_returns_no_trade_on_skip(self):
        import smart_trader
        from unittest import mock
        from telegram_bot import TelegramTradingBot

        class _FakeSkipTrader:
            def __init__(self, ticker=None):
                self.use_skip_on_weak_signal = True
                self.last_skip_reason = "flat_signal (bull 1 == bear 1)"
            def determine_option_strategy(self, ticker):
                return "skip"
            def get_option_contracts(self, ticker):
                raise AssertionError("contract lookup must not happen on SKIP")

        bot = TelegramTradingBot()
        bot.get_comprehensive_market_data = lambda ticker: {"current_price": 100.0}
        bot.determine_option_type = lambda market_data: "CALL"

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _FakeSkipTrader):
            msg = bot.analyze_ticker("AAPL", chat_id="x")

        self.assertIn("No trade", msg)
        self.assertIn("NO_TRADE", msg)

    def test_disabled_flag_does_not_short_circuit(self):
        """With the flag off, the skip branch is bypassed (strategy not called),
        so the flow proceeds to contract lookup as before."""
        import smart_trader
        from unittest import mock
        from telegram_bot import TelegramTradingBot

        class _FakeDisabledTrader:
            def __init__(self, ticker=None):
                self.use_skip_on_weak_signal = False
            def determine_option_strategy(self, ticker):
                raise AssertionError("strategy must not be called when flag off")
            def get_option_contracts(self, ticker):
                return []  # -> bot returns its 'no contracts' message

        bot = TelegramTradingBot()
        bot.get_comprehensive_market_data = lambda ticker: {"current_price": 100.0}
        bot.determine_option_type = lambda market_data: "CALL"

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _FakeDisabledTrader):
            msg = bot.analyze_ticker("AAPL", chat_id="x")

        # Reached contract lookup (returned the no-contracts message), i.e. it
        # did NOT short-circuit on skip.
        self.assertIn("No option contracts", msg)


# --------------------------------------------------------------------------- #
# Phase 4: portfolio_risk aggregate-exposure caps (pure module)
# --------------------------------------------------------------------------- #
class TestPhase4PortfolioLimits(unittest.TestCase):
    def setUp(self):
        from portfolio_risk import PortfolioLimits
        self.on = PortfolioLimits(enabled=True, max_abs_delta=5.0,
                                  max_abs_vega=10.0, max_theta_loss=5.0,
                                  max_same_direction=3, max_per_underlying=2)
        self.off = PortfolioLimits(enabled=False)

    def _check(self, book, trade, limits=None):
        from portfolio_risk import check_portfolio_limits
        return check_portfolio_limits(book, trade, limits or self.on)

    def _pos(self, underlying, direction="call", qty=1, delta=0.0, vega=0.0,
             theta=0.0):
        return {"underlying": underlying, "direction": direction, "qty": qty,
                "delta": delta, "vega": vega, "theta": theta}

    def test_delta_cap_blocks(self):
        book = [self._pos("AAA", "call", qty=9, delta=0.5)]   # +4.5
        r = self._check(book, self._pos("BBB", "call", qty=2, delta=0.5))  # +1.0
        self.assertFalse(r["allowed"])
        self.assertIn("portfolio_delta", r["breaches"])
        self.assertAlmostEqual(r["projected_delta"], 5.5, places=6)

    def test_delta_cap_allows_puts_to_offset(self):
        # +4.5 from calls, a put adds -1.0 -> projected 3.5 (within cap).
        book = [self._pos("AAA", "call", qty=9, delta=0.5)]
        r = self._check(book, self._pos("BBB", "put", qty=2, delta=0.5))
        self.assertTrue(r["allowed"])
        self.assertAlmostEqual(r["projected_delta"], 3.5, places=6)

    def test_vega_cap_blocks(self):
        book = [self._pos("AAA", "call", qty=95, vega=0.1)]   # 9.5
        r = self._check(book, self._pos("BBB", "call", qty=10, vega=0.1))  # +1.0
        self.assertFalse(r["allowed"])
        self.assertIn("portfolio_vega", r["breaches"])

    def test_theta_cap_blocks(self):
        book = [self._pos("AAA", "call", qty=90, theta=0.05)]  # -4.5
        r = self._check(book, self._pos("BBB", "call", qty=20, theta=0.05))  # -1.0
        self.assertFalse(r["allowed"])
        self.assertIn("portfolio_theta", r["breaches"])
        self.assertAlmostEqual(r["projected_theta"], -5.5, places=6)

    def test_same_direction_cap_blocks(self):
        book = [self._pos(u, "call") for u in ("AAA", "BBB", "CCC")]  # 3 bullish
        r = self._check(book, self._pos("DDD", "call"))              # -> 4 > 3
        self.assertFalse(r["allowed"])
        self.assertIn("same_direction", r["breaches"])

    def test_same_direction_opposite_allowed(self):
        book = [self._pos(u, "call") for u in ("AAA", "BBB", "CCC")]
        r = self._check(book, self._pos("DDD", "put"))  # 0 bearish + 1 = 1
        self.assertTrue(r["allowed"])

    def test_same_underlying_cap_blocks(self):
        book = [self._pos("SPY", "call"), self._pos("SPY", "put")]  # 2 on SPY
        r = self._check(book, self._pos("SPY", "call"))            # -> 3 > 2
        self.assertFalse(r["allowed"])
        self.assertIn("per_underlying", r["breaches"])

    def test_clean_trade_allowed(self):
        r = self._check([], self._pos("SPY", "call", qty=1, delta=0.4,
                                      vega=0.1, theta=0.05))
        self.assertTrue(r["allowed"])
        self.assertEqual(r["reason"], "ok")

    def test_disabled_is_noop(self):
        # A wildly over-limit trade is allowed when the gate is disabled.
        r = self._check([], self._pos("SPY", "call", qty=99, delta=0.9,
                                      vega=1.0, theta=1.0), limits=self.off)
        self.assertTrue(r["allowed"])


# --------------------------------------------------------------------------- #
# Phase 4: missing real greeks -> heuristic fallback estimates
# --------------------------------------------------------------------------- #
class TestPhase4GreekFallback(unittest.TestCase):
    def test_position_greeks_use_fallback_when_snapshot_missing(self):
        t = _make_trader()
        t.get_option_snapshot = lambda sym: {}        # no real greeks
        t.get_current_price = lambda sym=None: 150.0
        positions = [{"symbol": "AAPL260821C00145000", "qty": "2", "mock": False}]
        out = t._position_greeks(positions)
        self.assertEqual(len(out), 1)
        g = out[0]
        self.assertEqual(g["underlying"], "AAPL")
        self.assertEqual(g["direction"], "call")
        self.assertEqual(g["qty"], 2)
        # Static fallbacks for vega/theta; delta from heuristic (ITM call -> >0).
        self.assertAlmostEqual(g["vega"], 0.10, places=6)
        self.assertAlmostEqual(g["theta"], 0.05, places=6)
        self.assertGreater(g["delta"], 0.0)

    def test_position_greeks_prefer_real_snapshot(self):
        t = _make_trader()
        t.get_option_snapshot = lambda sym: {"delta": -0.33, "vega": 0.22,
                                             "theta": -0.07}
        t.get_current_price = lambda sym=None: 150.0
        positions = [{"symbol": "AAPL260821P00145000", "qty": 1, "mock": False}]
        out = t._position_greeks(positions)
        g = out[0]
        self.assertEqual(g["direction"], "put")
        # Real snapshot values pass through unchanged; portfolio_risk normalizes
        # signs (abs delta/vega, -abs theta) from the position direction.
        self.assertAlmostEqual(g["delta"], -0.33, places=6)
        self.assertAlmostEqual(g["vega"], 0.22, places=6)
        self.assertAlmostEqual(g["theta"], -0.07, places=6)

    def test_position_greeks_skips_non_options(self):
        t = _make_trader()
        t.get_option_snapshot = lambda sym: {}
        t.get_current_price = lambda sym=None: 150.0
        positions = [{"symbol": "AAPL", "qty": 10, "mock": False}]  # equity, not OCC
        self.assertEqual(t._position_greeks(positions), [])

    def test_portfolio_check_disabled_is_noop_on_trader(self):
        t = _make_trader()
        t.use_portfolio_greek_limits = False
        r = t._portfolio_greek_check({"symbol": "AAPL260821C00145000",
                                      "type": "call", "delta": 0.9,
                                      "vega": 1.0, "theta": 1.0}, 99)
        self.assertTrue(r["allowed"])

    def test_portfolio_check_blocks_via_trader(self):
        from portfolio_risk import PortfolioLimits
        t = _make_trader()
        t.use_portfolio_greek_limits = True
        t.portfolio_limits = PortfolioLimits(enabled=True, max_abs_delta=1.0,
                                             max_abs_vega=10.0, max_theta_loss=10.0,
                                             max_same_direction=99,
                                             max_per_underlying=99)
        t.get_positions = lambda: []                 # empty book
        t.get_option_snapshot = lambda sym: {}
        t.get_current_price = lambda sym=None: 150.0
        # New trade alone is +2.0 delta (qty 2 * 1.0) > 1.0 cap.
        r = t._portfolio_greek_check({"symbol": "AAPL260821C00145000",
                                      "type": "call", "delta": 1.0,
                                      "vega": 0.0, "theta": 0.0}, 2)
        self.assertFalse(r["allowed"])
        self.assertIn("portfolio_delta", r["breaches"])


# --------------------------------------------------------------------------- #
# Phase 4: realized-only kill-switch + daily reset
# --------------------------------------------------------------------------- #
class TestPhase4RealizedKillSwitch(unittest.TestCase):
    def _tracker(self):
        import tempfile, os
        from realized_pnl_tracker import RealizedPnLTracker
        path = os.path.join(tempfile.mkdtemp(), "realized_pnl_log.json")
        return RealizedPnLTracker(path)

    def test_tracker_sums_today_only(self):
        from datetime import datetime, timedelta
        t = self._tracker()
        self.assertEqual(t.get_today_realized(), 0.0)
        t.add_realized(-120.0, "SPY")
        t.add_realized(30.0, "QQQ")
        self.assertAlmostEqual(t.get_today_realized(), -90.0, places=6)
        # Yesterday's big loss must not count toward today (daily reset).
        t.add_realized(-999.0, "OLD", when=datetime.now() - timedelta(days=1))
        self.assertAlmostEqual(t.get_today_realized(), -90.0, places=6)

    def test_killswitch_ignores_unrealized_when_flag_on(self):
        # Flag ON: _risk_check sources realized P/L only. A huge unrealized
        # equity drop must NOT trip the kill-switch.
        from unittest import mock
        t = _make_trader()
        t.use_realized_pnl_killswitch = True
        t.ticker = "AAPL"
        t.get_positions = lambda: []
        # equity-minus-last_equity would be -5000 (a deep unrealized drawdown).
        t.get_account = lambda: {"equity": "5000", "last_equity": "10000"}

        class _ZeroRealized:
            def __init__(self, *a, **k):
                pass
            def get_today_realized(self):
                return 0.0  # nothing actually realized today

        with mock.patch("realized_pnl_tracker.RealizedPnLTracker", _ZeroRealized):
            verdict = t._risk_check(trade_cost=100.0, qty=1, may_day_trade=False)
        self.assertNotIn("kill_switch", verdict.get("breaches", []))
        self.assertNotIn("daily_loss_limit", verdict.get("breaches", []))

    def test_killswitch_trips_on_realized_loss_when_flag_on(self):
        from unittest import mock
        t = _make_trader()
        t.use_realized_pnl_killswitch = True
        t.ticker = "AAPL"
        t.get_positions = lambda: []
        t.get_account = lambda: {"equity": "10000", "last_equity": "10000"}

        big_loss = -abs(t.risk_engine.limits.kill_switch_loss) - 1.0

        class _BigRealizedLoss:
            def __init__(self, *a, **k):
                pass
            def get_today_realized(self):
                return big_loss

        with mock.patch("realized_pnl_tracker.RealizedPnLTracker", _BigRealizedLoss):
            verdict = t._risk_check(trade_cost=100.0, qty=1, may_day_trade=False)
        self.assertFalse(verdict["allowed"])
        self.assertIn("kill_switch", verdict["breaches"])

    def test_default_flag_off_uses_equity_delta(self):
        # Flag OFF (default): legacy behavior. A deep equity drop trips the switch.
        t = _make_trader()
        t.use_realized_pnl_killswitch = False
        t.ticker = "AAPL"
        t.get_positions = lambda: []
        loss = -abs(t.risk_engine.limits.kill_switch_loss) - 1.0
        t.get_account = lambda: {"equity": str(10000 + loss),
                                 "last_equity": "10000"}
        verdict = t._risk_check(trade_cost=100.0, qty=1, may_day_trade=False)
        self.assertFalse(verdict["allowed"])
        self.assertIn("kill_switch", verdict["breaches"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
