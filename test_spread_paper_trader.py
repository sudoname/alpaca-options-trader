"""
Offline tests for Phase 6C spread PAPER trading (simulation only).

No creds, no network, no broker calls. All quotes are passed in. JSON state is
written to per-test temp files so nothing touches the repo's real paper files.
"""

import os
import tempfile
import unittest
from unittest import mock

from config_loader import ConfigLoader
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, DEBIT_CALL_SPREAD, NO_TRADE,
    SpreadLeg, SpreadProposal,
)
from spread_paper_trader import (
    REASON_DISABLED, REASON_DUPLICATE_POSITION, REASON_INVALID_MAX_LOSS,
    REASON_LOW_ORACLE_SCORE, REASON_NO_TRADE, REASON_OPENED,
    STATUS_CLOSED, STATUS_OPEN,
    SpreadPaperConfig, SpreadPaperTrader, compute_mark,
)


# --------------------------------------------------------------------------- #
# Proposal fixtures
# --------------------------------------------------------------------------- #
def credit_proposal(symbol="SPY", oracle=80.0):
    """Bull put credit: SELL 100p @2.00, BUY 95p @1.25 -> net credit 0.75."""
    legs = [
        SpreadLeg("sell", "put", 100, bid=1.95, ask=2.05),
        SpreadLeg("buy", "put", 95, bid=1.20, ask=1.30),
    ]
    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=0.75, max_profit=75.0, max_loss=425.0,
        breakeven=99.25, width=5.0, oracle_score=oracle)


def debit_proposal(symbol="QQQ", oracle=80.0):
    """Debit call: BUY 100c @2.00, SELL 105c @0.75 -> net debit 1.25."""
    legs = [
        SpreadLeg("buy", "call", 100, bid=1.95, ask=2.05),
        SpreadLeg("sell", "call", 105, bid=0.70, ask=0.80),
    ]
    return SpreadProposal(
        strategy_name=DEBIT_CALL_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=-1.25, max_profit=375.0, max_loss=125.0,
        breakeven=101.25, width=5.0, oracle_score=oracle)


def _enabled_trader(tmpdir):
    cfg = SpreadPaperConfig(
        enabled=True, min_oracle_score=70.0,
        positions_file=os.path.join(tmpdir, "positions.json"),
        trades_file=os.path.join(tmpdir, "trades.json"))
    return SpreadPaperTrader(cfg)


# --------------------------------------------------------------------------- #
# compute_mark (pure)
# --------------------------------------------------------------------------- #
class TestComputeMark(unittest.TestCase):
    def test_credit_entry_mark_is_negative(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        # +mid(buy 95p=1.25) - mid(sell 100p=2.00) = -0.75
        self.assertAlmostEqual(compute_mark(legs), -0.75, places=6)

    def test_debit_entry_mark_is_positive(self):
        legs = [l.as_dict() for l in debit_proposal().legs]
        # +mid(buy 100c=2.00) - mid(sell 105c=0.75) = +1.25
        self.assertAlmostEqual(compute_mark(legs), 1.25, places=6)

    def test_external_quotes_override_stored(self):
        legs = [l.as_dict() for l in credit_proposal().legs]
        quotes = {"sell:put:100": {"bid": 0.95, "ask": 1.05},
                  "buy:put:95": {"bid": 0.45, "ask": 0.55}}
        # +0.50 - 1.00 = -0.50
        self.assertAlmostEqual(compute_mark(legs, quotes), -0.50, places=6)

    def test_float_quote_supported(self):
        legs = [l.as_dict() for l in debit_proposal().legs]
        quotes = {"buy:call:100": 3.0, "sell:call:105": 1.0}
        self.assertAlmostEqual(compute_mark(legs, quotes), 2.0, places=6)


# --------------------------------------------------------------------------- #
# open_position (safety + persistence)
# --------------------------------------------------------------------------- #
class TestOpenPosition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trader = _enabled_trader(self.tmp)

    def test_open_valid_proposal(self):
        res = self.trader.open_position(credit_proposal())
        self.assertTrue(res["allowed"])
        self.assertEqual(res["reason"], REASON_OPENED)
        pos = res["position"]
        self.assertEqual(pos["status"], STATUS_OPEN)
        self.assertAlmostEqual(pos["entry_mark"], -0.75, places=6)
        # persisted and discoverable
        self.assertEqual(len(self.trader.get_open_positions()), 1)
        self.assertIsNotNone(self.trader.find_position(pos["id"]))

    def test_position_has_all_required_fields(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        for key in ("id", "timestamp", "symbol", "strategy", "oracle_score",
                    "legs", "net_credit_or_debit", "max_profit", "max_loss",
                    "breakeven", "status", "entry_mark", "current_mark",
                    "pnl", "pnl_percent", "exit_reason"):
            self.assertIn(key, pos)

    def test_reject_no_trade(self):
        nt = SpreadProposal(strategy_name=NO_TRADE, symbol="SPY",
                            max_loss=100.0, oracle_score=99.0)
        res = self.trader.open_position(nt)
        self.assertFalse(res["allowed"])
        self.assertEqual(res["reason"], REASON_NO_TRADE)
        self.assertEqual(self.trader.get_open_positions(), [])

    def test_reject_invalid_max_loss_zero(self):
        p = credit_proposal()
        p.max_loss = 0.0
        self.assertEqual(self.trader.open_position(p)["reason"],
                         REASON_INVALID_MAX_LOSS)

    def test_reject_invalid_max_loss_missing(self):
        p = credit_proposal()
        p.max_loss = None
        self.assertEqual(self.trader.open_position(p)["reason"],
                         REASON_INVALID_MAX_LOSS)

    def test_reject_low_oracle_score(self):
        res = self.trader.open_position(credit_proposal(oracle=50.0))
        self.assertFalse(res["allowed"])
        self.assertEqual(res["reason"], REASON_LOW_ORACLE_SCORE)

    def test_reject_duplicate_symbol(self):
        self.assertTrue(self.trader.open_position(credit_proposal())["allowed"])
        dup = self.trader.open_position(credit_proposal())
        self.assertFalse(dup["allowed"])
        self.assertEqual(dup["reason"], REASON_DUPLICATE_POSITION)
        self.assertEqual(len(self.trader.get_open_positions()), 1)

    def test_different_symbols_allowed(self):
        self.assertTrue(self.trader.open_position(credit_proposal("SPY"))["allowed"])
        self.assertTrue(self.trader.open_position(debit_proposal("QQQ"))["allowed"])
        self.assertEqual(len(self.trader.get_open_positions()), 2)

    def test_reject_when_disabled(self):
        cfg = SpreadPaperConfig(
            enabled=False,
            positions_file=os.path.join(self.tmp, "d.json"),
            trades_file=os.path.join(self.tmp, "dt.json"))
        disabled = SpreadPaperTrader(cfg)
        res = disabled.open_position(credit_proposal())
        self.assertFalse(res["allowed"])
        self.assertEqual(res["reason"], REASON_DISABLED)
        self.assertFalse(os.path.exists(cfg.positions_file))


# --------------------------------------------------------------------------- #
# mark-to-market
# --------------------------------------------------------------------------- #
class TestMarkToMarket(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trader = _enabled_trader(self.tmp)

    def test_credit_spread_gain(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        # spread narrows -> profit for the credit seller
        quotes = {"sell:put:100": {"bid": 0.95, "ask": 1.05},
                  "buy:put:95": {"bid": 0.45, "ask": 0.55}}
        marked = self.trader.mark_to_market(pos["id"], quotes)
        self.assertAlmostEqual(marked["current_mark"], -0.50, places=6)
        self.assertAlmostEqual(marked["pnl"], 25.0, places=6)

    def test_credit_spread_full_loss_equals_max_loss(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        # pinned at full width (5 wide): SELL 100p deep ITM 6.00, BUY 95p 1.00
        quotes = {"sell:put:100": {"bid": 5.95, "ask": 6.05},
                  "buy:put:95": {"bid": 0.95, "ask": 1.05}}
        marked = self.trader.mark_to_market(pos["id"], quotes)
        self.assertAlmostEqual(marked["pnl"], -pos["max_loss"], places=2)
        self.assertAlmostEqual(marked["pnl_percent"], -100.0, places=2)

    def test_debit_spread_gain(self):
        pos = self.trader.open_position(debit_proposal())["position"]
        # spread appreciates -> profit for the debit buyer
        quotes = {"buy:call:100": {"bid": 2.95, "ask": 3.05},
                  "sell:call:105": {"bid": 0.95, "ask": 1.05}}
        marked = self.trader.mark_to_market(pos["id"], quotes)
        self.assertAlmostEqual(marked["current_mark"], 2.0, places=6)
        self.assertAlmostEqual(marked["pnl"], 75.0, places=6)

    def test_mtm_persists_to_store(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        quotes = {"sell:put:100": {"bid": 0.95, "ask": 1.05},
                  "buy:put:95": {"bid": 0.45, "ask": 0.55}}
        self.trader.mark_to_market(pos["id"], quotes)
        reread = self.trader.find_position(pos["id"])
        self.assertAlmostEqual(reread["pnl"], 25.0, places=6)

    def test_mtm_unknown_id_returns_none(self):
        self.assertIsNone(self.trader.mark_to_market("nope", {}))


# --------------------------------------------------------------------------- #
# close_position (writes trade history)
# --------------------------------------------------------------------------- #
class TestClosePosition(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        self.trader = _enabled_trader(self.tmp)

    def test_close_writes_trade_history(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        quotes = {"sell:put:100": {"bid": 0.95, "ask": 1.05},
                  "buy:put:95": {"bid": 0.45, "ask": 0.55}}
        closed = self.trader.close_position(pos["id"], quotes,
                                            exit_reason="take_profit")
        self.assertEqual(closed["status"], STATUS_CLOSED)
        self.assertEqual(closed["exit_reason"], "take_profit")
        self.assertAlmostEqual(closed["pnl"], 25.0, places=6)
        # open store pruned, history written
        self.assertEqual(self.trader.get_open_positions(), [])
        trades = self.trader.load_trades()
        self.assertEqual(len(trades), 1)
        self.assertEqual(trades[0]["id"], pos["id"])

    def test_close_unknown_returns_none(self):
        self.assertIsNone(self.trader.close_position("nope"))

    def test_close_twice_returns_none(self):
        pos = self.trader.open_position(credit_proposal())["position"]
        self.assertIsNotNone(self.trader.close_position(pos["id"]))
        self.assertIsNone(self.trader.close_position(pos["id"]))
        self.assertEqual(len(self.trader.load_trades()), 1)


# --------------------------------------------------------------------------- #
# Config defaults (default behavior unchanged)
# --------------------------------------------------------------------------- #
class TestConfigDefaults(unittest.TestCase):
    def test_disabled_and_floor_by_default(self):
        cfg = SpreadPaperConfig.from_env(
            loader=ConfigLoader(file_values={}, environ={}))
        self.assertFalse(cfg.enabled)
        self.assertEqual(cfg.min_oracle_score, 70.0)

    def test_env_can_enable(self):
        cfg = SpreadPaperConfig.from_env(loader=ConfigLoader(
            file_values={}, environ={"USE_SPREAD_PAPER_TRADING": "true",
                                     "SPREAD_MIN_ORACLE_SCORE": "85"}))
        self.assertTrue(cfg.enabled)
        self.assertEqual(cfg.min_oracle_score, 85.0)


# --------------------------------------------------------------------------- #
# Telegram commands never call Alpaca / never place orders
# --------------------------------------------------------------------------- #
class TestTelegramPaperCommands(unittest.TestCase):
    def _bot(self, tmp, enabled=True):
        from telegram_bot import TelegramTradingBot
        bot = TelegramTradingBot()
        bot.spread_paper_enabled = enabled
        # Point the bot's paper trader at temp files (no repo pollution).
        bot._spread_paper_trader = lambda: _enabled_trader(tmp)
        return bot

    def _fake_trader_cls(self, proposal):
        class _FakeTrader:
            def __init__(self, ticker=None, **k):
                self.ticker = ticker
            def propose_spread(self, symbol=None, config=None):
                return proposal
            # Any of these being called means a live path leaked into paper mode.
            def place_order_with_stops(self, *a, **k):
                raise AssertionError("paper command must NOT place orders")
            def execute_trade(self, *a, **k):
                raise AssertionError("paper command must NOT execute")
            def submit_order(self, *a, **k):
                raise AssertionError("paper command must NOT submit orders")
        return _FakeTrader

    def test_open_positions_close_flow_no_alpaca(self):
        import smart_trader
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp)
        fake = self._fake_trader_cls(credit_proposal())

        with mock.patch.object(smart_trader, "SmartOptionsTrader", fake):
            opened = bot.spread_paper_open("SPY", chat_id="x")
        self.assertIn("OPENED", opened)
        self.assertIn("SIMULATED", opened)
        self.assertIn("NO broker order", opened)

        listed = bot.spread_paper_positions(chat_id="x")
        self.assertIn("SPY", listed)
        self.assertIn("SIMULATED", listed)

        # Recover the generated id from the store and close it.
        trader = _enabled_trader(tmp)
        pid = trader.get_open_positions()[0]["id"]
        closed = bot.spread_paper_close(pid, chat_id="x")
        self.assertIn("CLOSED", closed)
        self.assertIn("SIMULATED", closed)
        self.assertEqual(len(trader.load_trades()), 1)

    def test_open_disabled_never_builds_trader(self):
        import smart_trader
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp, enabled=False)

        class _Boom:
            def __init__(self, *a, **k):
                raise AssertionError("trader must not be built when disabled")

        with mock.patch.object(smart_trader, "SmartOptionsTrader", _Boom):
            msg = bot.spread_paper_open("SPY", chat_id="x")
        self.assertIn("disabled", msg.lower())

    def test_positions_disabled_message(self):
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp, enabled=False)
        self.assertIn("disabled", bot.spread_paper_positions(chat_id="x").lower())

    def test_open_rejected_low_score_message(self):
        import smart_trader
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp)
        fake = self._fake_trader_cls(credit_proposal(oracle=50.0))
        with mock.patch.object(smart_trader, "SmartOptionsTrader", fake):
            msg = bot.spread_paper_open("SPY", chat_id="x")
        self.assertIn("rejected", msg.lower())
        self.assertIn(REASON_LOW_ORACLE_SCORE, msg)

    def test_invalid_symbol_rejected(self):
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp)
        self.assertIn("Invalid symbol", bot.spread_paper_open("NOT A TICKER", chat_id="x"))

    def test_close_requires_id(self):
        tmp = tempfile.mkdtemp()
        bot = self._bot(tmp)
        self.assertIn("Usage", bot.spread_paper_close("", chat_id="x"))


if __name__ == "__main__":
    unittest.main()
