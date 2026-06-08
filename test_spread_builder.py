"""
Offline tests for Phase 6A defined-risk spread PROPOSALS.

No creds / no network. Covers (Requirement 8):
  * builds each spread type: bull put credit, bear call credit, debit call,
    debit put, iron condor — with correct max_profit / max_loss / breakeven.
  * safety rejections: undefined risk, missing bid/ask, wide per-leg spread,
    max loss above the risk limit, illiquid legs.
  * selection rules map (vol_state, trend) -> strategy exactly.
  * the [SPREAD_PROPOSAL] log block carries every required field.
  * Telegram `SPREAD_PROPOSAL TICKER` returns a PROPOSAL ONLY (no execution),
    and the default (flag off) behavior is unchanged (feature disabled message,
    no trader instantiated).
"""

import unittest
from unittest import mock

from spread_builder import (
    SpreadConfig, SpreadLeg, SpreadProposal,
    build_bull_put_credit_spread, build_bear_call_credit_spread,
    build_debit_call_spread, build_debit_put_spread, build_iron_condor,
    build_spread, validate_legs, validate_defined_risk,
    classify_volatility, classify_trend, select_spread_strategy,
    format_proposal_log,
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
)


# Generous gates so the *build math* tests never trip a liquidity/spread guard;
# individual rejection tests tighten a single knob.
CFG = SpreadConfig(enabled=True, max_loss_limit=1000.0, max_leg_spread_pct=50.0,
                   min_open_interest=0.0, min_volume=0.0)


def leg(action, otype, strike, bid, ask, oi=500, vol=500):
    return SpreadLeg(action=action, option_type=otype, strike=strike,
                     bid=bid, ask=ask, open_interest=oi, volume=vol)


# --------------------------------------------------------------------------- #
# Builds — the five supported defined-risk structures
# --------------------------------------------------------------------------- #
class TestBuilds(unittest.TestCase):
    def test_bull_put_credit(self):
        # sell 100P @1.20, buy 95P @0.45 -> credit 0.75, width 5.
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 1.20, 1.25),
            leg("buy", "put", 95, 0.40, 0.45), CFG, "SPY")
        self.assertEqual(p.strategy_name, BULLISH_PUT_CREDIT_SPREAD)
        self.assertTrue(p.is_credit)
        self.assertAlmostEqual(p.net_credit_or_debit, 0.75, places=2)
        self.assertAlmostEqual(p.max_profit, 75.0, places=2)
        self.assertAlmostEqual(p.max_loss, 425.0, places=2)   # (5 - 0.75)*100
        self.assertAlmostEqual(p.breakeven, 99.25, places=2)  # 100 - 0.75
        self.assertEqual(p.width, 5)

    def test_bear_call_credit(self):
        # sell 100C @1.20, buy 105C @0.45 -> credit 0.75, width 5.
        p = build_bear_call_credit_spread(
            leg("sell", "call", 100, 1.20, 1.25),
            leg("buy", "call", 105, 0.40, 0.45), CFG, "SPY")
        self.assertEqual(p.strategy_name, BEARISH_CALL_CREDIT_SPREAD)
        self.assertTrue(p.is_credit)
        self.assertAlmostEqual(p.max_profit, 75.0, places=2)
        self.assertAlmostEqual(p.max_loss, 425.0, places=2)
        self.assertAlmostEqual(p.breakeven, 100.75, places=2)  # 100 + 0.75

    def test_debit_call(self):
        # buy 100C @2.00, sell 105C @0.50 -> debit 1.50, width 5.
        p = build_debit_call_spread(
            leg("buy", "call", 100, 1.95, 2.00),
            leg("sell", "call", 105, 0.50, 0.55), CFG, "SPY")
        self.assertEqual(p.strategy_name, DEBIT_CALL_SPREAD)
        self.assertFalse(p.is_credit)
        self.assertLess(p.net_credit_or_debit, 0)              # signed debit
        self.assertAlmostEqual(p.net_credit_or_debit, -1.50, places=2)
        self.assertAlmostEqual(p.max_loss, 150.0, places=2)    # debit*100
        self.assertAlmostEqual(p.max_profit, 350.0, places=2)  # (5-1.5)*100
        self.assertAlmostEqual(p.breakeven, 101.50, places=2)  # 100 + 1.50

    def test_debit_put(self):
        # buy 100P @2.00, sell 95P @0.50 -> debit 1.50, width 5.
        p = build_debit_put_spread(
            leg("buy", "put", 100, 1.95, 2.00),
            leg("sell", "put", 95, 0.50, 0.55), CFG, "SPY")
        self.assertEqual(p.strategy_name, DEBIT_PUT_SPREAD)
        self.assertFalse(p.is_credit)
        self.assertAlmostEqual(p.max_loss, 150.0, places=2)
        self.assertAlmostEqual(p.max_profit, 350.0, places=2)
        self.assertAlmostEqual(p.breakeven, 98.50, places=2)   # 100 - 1.50

    def test_iron_condor(self):
        # buy 90P/sell 95P + sell 105C/buy 110C. Each credit 0.60 -> total 1.20.
        p = build_iron_condor(
            leg("buy", "put", 90, 0.30, 0.35),
            leg("sell", "put", 95, 0.95, 1.00),
            leg("sell", "call", 105, 0.95, 1.00),
            leg("buy", "call", 110, 0.30, 0.35), CFG, "SPY")
        self.assertEqual(p.strategy_name, IRON_CONDOR)
        self.assertTrue(p.is_credit)
        self.assertIsInstance(p.breakeven, list)
        self.assertEqual(len(p.breakeven), 2)
        # net credit = (0.95-0.35) + (0.95-0.35) = 1.20; width 5.
        self.assertAlmostEqual(p.net_credit_or_debit, 1.20, places=2)
        self.assertAlmostEqual(p.max_profit, 120.0, places=2)
        self.assertAlmostEqual(p.max_loss, 380.0, places=2)    # (5-1.2)*100
        self.assertAlmostEqual(p.breakeven[0], 93.80, places=2)  # 95 - 1.2
        self.assertAlmostEqual(p.breakeven[1], 106.20, places=2)  # 105 + 1.2

    def test_build_spread_dispatch_unordered_legs(self):
        # build_spread infers roles regardless of leg order.
        legs = [leg("buy", "call", 105, 0.50, 0.55),
                leg("buy", "call", 100, 1.95, 2.00)]
        # role inference makes the lower strike the long, higher the short.
        legs[0].action = legs[1].action = "buy"  # actions get overwritten anyway
        p = build_spread(DEBIT_CALL_SPREAD, legs, CFG, "SPY")
        self.assertEqual(p.strategy_name, DEBIT_CALL_SPREAD)


# --------------------------------------------------------------------------- #
# Safety rejections (Requirement 5)
# --------------------------------------------------------------------------- #
class TestRejections(unittest.TestCase):
    def test_reject_undefined_risk_naked_short(self):
        # A lone short put has no long put -> undefined risk.
        self.assertEqual(validate_defined_risk([leg("sell", "put", 100, 1.0, 1.1)]),
                         "undefined_risk")
        p = build_spread(BULLISH_PUT_CREDIT_SPREAD,
                         [leg("sell", "put", 100, 1.0, 1.1)], CFG, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "undefined_risk")

    def test_reject_undefined_risk_short_heavy_ratio(self):
        # 2 short calls but only 1 long call -> undefined risk.
        legs = [leg("sell", "call", 100, 1.0, 1.1),
                leg("sell", "call", 101, 1.0, 1.1),
                leg("buy", "call", 105, 0.4, 0.45)]
        self.assertEqual(validate_defined_risk(legs), "undefined_risk")

    def test_reject_missing_bid_ask(self):
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, None, None),   # no quote
            leg("buy", "put", 95, 0.40, 0.45), CFG, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "missing_quote")

    def test_reject_wide_spread(self):
        narrow = SpreadConfig(enabled=True, max_loss_limit=1000.0,
                              max_leg_spread_pct=5.0)
        # short put bid 1.00 / ask 1.50 -> 33% wide, exceeds 5%.
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 1.00, 1.50),
            leg("buy", "put", 95, 0.40, 0.45), narrow, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "wide_spread")

    def test_reject_max_loss_above_limit(self):
        tight = SpreadConfig(enabled=True, max_loss_limit=100.0,
                             max_leg_spread_pct=50.0)
        # width 5, credit 0.75 -> max loss 425 > 100 limit.
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 1.20, 1.25),
            leg("buy", "put", 95, 0.40, 0.45), tight, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "max_loss_exceeds_limit")

    def test_reject_illiquid_leg(self):
        liq = SpreadConfig(enabled=True, max_loss_limit=1000.0,
                           max_leg_spread_pct=50.0, min_open_interest=100.0)
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 1.20, 1.25, oi=10),   # OI below floor
            leg("buy", "put", 95, 0.40, 0.45, oi=500), liq, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "illiquid_leg")

    def test_illiquid_fails_open_when_data_missing(self):
        liq = SpreadConfig(enabled=True, max_loss_limit=1000.0,
                           max_leg_spread_pct=50.0, min_open_interest=100.0)
        # OI is None on both legs -> liquidity gate fails OPEN (builds).
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 1.20, 1.25, oi=None),
            leg("buy", "put", 95, 0.40, 0.45, oi=None), liq, "SPY")
        self.assertEqual(p.strategy_name, BULLISH_PUT_CREDIT_SPREAD)

    def test_reject_non_positive_credit(self):
        # short bid below long ask -> no real credit.
        p = build_bull_put_credit_spread(
            leg("sell", "put", 100, 0.30, 0.35),
            leg("buy", "put", 95, 0.40, 0.45), CFG, "SPY")
        self.assertEqual(p.strategy_name, NO_TRADE)
        self.assertEqual(p.reason, "non_positive_credit")


# --------------------------------------------------------------------------- #
# Selection rules (Requirement 4)
# --------------------------------------------------------------------------- #
class TestSelection(unittest.TestCase):
    def test_volatility_classification(self):
        self.assertEqual(classify_volatility(0.40, 0.20, CFG), "overpriced")  # 2.0
        self.assertEqual(classify_volatility(0.15, 0.20, CFG), "underpriced")  # 0.75
        self.assertEqual(classify_volatility(0.21, 0.20, CFG), "fair")        # 1.05
        self.assertEqual(classify_volatility(None, 0.20, CFG), "unknown")
        self.assertEqual(classify_volatility(0.40, 0, CFG), "unknown")

    def test_trend_classification(self):
        self.assertEqual(classify_trend(0.05, CFG), "bullish")
        self.assertEqual(classify_trend(-0.05, CFG), "bearish")
        self.assertEqual(classify_trend(0.0, CFG), "neutral")
        self.assertEqual(classify_trend(None, CFG), "neutral")

    def test_selection_matrix(self):
        cases = {
            ("overpriced", "neutral"): IRON_CONDOR,
            ("overpriced", "bullish"): BULLISH_PUT_CREDIT_SPREAD,
            ("overpriced", "bearish"): BEARISH_CALL_CREDIT_SPREAD,
            ("underpriced", "bullish"): DEBIT_CALL_SPREAD,
            ("underpriced", "bearish"): DEBIT_PUT_SPREAD,
            ("fair", "bullish"): NO_TRADE,
            ("underpriced", "neutral"): NO_TRADE,
            ("unknown", "bullish"): NO_TRADE,
        }
        for (vs, tr), exp in cases.items():
            self.assertEqual(select_spread_strategy(vs, tr), exp, (vs, tr))

    def test_weak_edge_is_no_trade(self):
        self.assertEqual(
            select_spread_strategy("overpriced", "bullish", edge_ok=False), NO_TRADE)


# --------------------------------------------------------------------------- #
# Logging (Requirement 6)
# --------------------------------------------------------------------------- #
class TestLog(unittest.TestCase):
    def test_log_has_all_fields(self):
        p = build_iron_condor(
            leg("buy", "put", 90, 0.30, 0.35),
            leg("sell", "put", 95, 0.95, 1.00),
            leg("sell", "call", 105, 0.95, 1.00),
            leg("buy", "call", 110, 0.30, 0.35), CFG, "SPY")
        log = format_proposal_log(p)
        for tok in ("[SPREAD_PROPOSAL]", "strategy=", "symbol=SPY", "legs=",
                    "net_credit_or_debit=", "max_profit=", "max_loss=",
                    "breakeven=", "reason="):
            self.assertIn(tok, log)


# --------------------------------------------------------------------------- #
# Telegram command — proposal ONLY, default disabled (Requirement 7 + 8)
# --------------------------------------------------------------------------- #
class TestTelegramSpreadCommand(unittest.TestCase):
    def _bot(self):
        from telegram_bot import TelegramTradingBot
        return TelegramTradingBot()

    def test_disabled_by_default_no_trader(self):
        """Flag off -> disabled message and the trader is NEVER instantiated."""
        import smart_trader
        bot = self._bot()
        bot.spread_proposals_enabled = False

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("trader must not be built when disabled")

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _Boom):
            msg = bot.get_spread_proposal("SPY", chat_id="x")
        self.assertIn("disabled", msg.lower())

    def test_enabled_returns_proposal_only(self):
        """Flag on -> returns the proposal text; never calls any order method."""
        import smart_trader
        bot = self._bot()
        bot.spread_proposals_enabled = True

        sample = SpreadProposal(
            strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol="SPY",
            legs=[SpreadLeg("sell", "put", 100, 1.20, 1.25),
                  SpreadLeg("buy", "put", 95, 0.40, 0.45)],
            net_credit_or_debit=0.75, max_profit=75.0, max_loss=425.0,
            breakeven=99.25, width=5.0, estimated_probability=0.85,
            reason="vol overpriced + bullish")

        class _FakeTrader:
            def __init__(self, ticker=None, **k):
                self.ticker = ticker
            def propose_spread(self, symbol=None, config=None):
                return sample
            def place_order_with_stops(self, *a, **k):
                raise AssertionError("proposal command must NOT place orders")
            def execute_trade(self, *a, **k):
                raise AssertionError("proposal command must NOT execute")

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _FakeTrader):
            msg = bot.get_spread_proposal("SPY", chat_id="x")

        self.assertIn("Spread Proposal", msg)
        self.assertIn("bullish_put_credit_spread", msg)
        self.assertIn("425.00", msg)        # max loss surfaced
        self.assertIn("nothing was traded", msg.lower())

    def test_enabled_no_trade_message(self):
        import smart_trader
        bot = self._bot()
        bot.spread_proposals_enabled = True

        nope = SpreadProposal(strategy_name=NO_TRADE, symbol="SPY",
                              reason="no_edge vol=fair trend=neutral")

        class _FakeTrader:
            def __init__(self, ticker=None, **k):
                pass
            def propose_spread(self, symbol=None, config=None):
                return nope

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _FakeTrader):
            msg = bot.get_spread_proposal("SPY", chat_id="x")
        self.assertIn("No spread proposal", msg)
        self.assertIn("no_edge", msg)

    def test_invalid_symbol_rejected(self):
        bot = self._bot()
        bot.spread_proposals_enabled = True
        msg = bot.get_spread_proposal("NOT A TICKER", chat_id="x")
        self.assertIn("Invalid symbol", msg)


if __name__ == "__main__":
    unittest.main()
