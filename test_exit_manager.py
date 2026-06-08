"""
Offline tests for the Phase 5 unified exit manager.

No creds / no network. Covers:
  * evaluate_exit: stop / take / trailing / expiration / stale / hold, precedence,
    roll suppression of take-profit, and fail-open on bad prices.
  * format_exit_log: contains every Phase-5 required field.
  * enforce_exit: closes and records the outcome EXACTLY ONCE (via a fake trader),
    and uses the supplied reason code (legacy 'dynamic_*' for the scheduler).
"""

import unittest
from datetime import datetime, timedelta

from exit_manager import evaluate_exit, format_exit_log, enforce_exit, ExitDecision


LEVELS = {'stop_loss_percent': 10.0, 'take_profit_percent': 20.0,
          'trailing_stop_distance': 0.05}


class TestEvaluateExit(unittest.TestCase):
    def test_stop_loss_fires(self):
        d = evaluate_exit({'entry_price': 1.00}, 0.85, LEVELS)
        self.assertEqual(d.action, 'stop_loss')
        self.assertTrue(d.should_exit)
        self.assertAlmostEqual(d.pnl_percent, -15.0, places=4)

    def test_take_profit_fires(self):
        d = evaluate_exit({'entry_price': 1.00}, 1.25, LEVELS)
        self.assertEqual(d.action, 'take_profit')
        self.assertTrue(d.should_exit)

    def test_take_profit_suppressed_when_rolling(self):
        d = evaluate_exit({'entry_price': 1.00}, 1.25, LEVELS, roll_enabled=True)
        self.assertEqual(d.action, 'hold')
        self.assertFalse(d.should_exit)

    def test_trailing_stop_fires_below_take(self):
        # pnl +10% (below the 20% take) but pulled back 6.8% from the high.
        trade = {'entry_price': 1.00, 'trailing_stop_active': True,
                 'highest_price': 1.18}
        d = evaluate_exit(trade, 1.10, LEVELS)
        self.assertEqual(d.action, 'trailing_stop')

    def test_trailing_not_armed_holds(self):
        trade = {'entry_price': 1.00, 'trailing_stop_active': False,
                 'highest_price': 1.18}
        d = evaluate_exit(trade, 1.10, LEVELS)
        self.assertEqual(d.action, 'hold')

    def test_trailing_armed_but_not_pulled_back_holds(self):
        trade = {'entry_price': 1.00, 'trailing_stop_active': True,
                 'highest_price': 1.12}
        d = evaluate_exit(trade, 1.11, LEVELS)  # only 0.9% off the high
        self.assertEqual(d.action, 'hold')

    def test_stop_precedence_over_take(self):
        # A catastrophic move is read as a stop, never a take.
        d = evaluate_exit({'entry_price': 1.00}, 0.40, LEVELS)
        self.assertEqual(d.action, 'stop_loss')

    def test_take_precedence_over_trailing(self):
        # +25% qualifies for take; trailing is armed too, but take wins.
        trade = {'entry_price': 1.00, 'trailing_stop_active': True,
                 'highest_price': 1.40}
        d = evaluate_exit(trade, 1.25, LEVELS)
        self.assertEqual(d.action, 'take_profit')

    def test_expiration_fires_when_enabled(self):
        now = datetime(2025, 1, 10, 12, 0, 0)
        exp = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        d = evaluate_exit({'entry_price': 1.00, 'expiration': exp}, 1.00, LEVELS,
                          now=now, check_expiration=True)
        self.assertEqual(d.action, 'expiration')

    def test_expiration_skipped_when_disabled(self):
        now = datetime(2025, 1, 10, 12, 0, 0)
        exp = (now + timedelta(days=1)).strftime('%Y-%m-%d')
        d = evaluate_exit({'entry_price': 1.00, 'expiration': exp}, 1.00, LEVELS,
                          now=now, check_expiration=False)
        self.assertEqual(d.action, 'hold')

    def test_expiration_far_holds(self):
        now = datetime(2025, 1, 10, 12, 0, 0)
        exp = (now + timedelta(days=30)).strftime('%Y-%m-%d')
        d = evaluate_exit({'entry_price': 1.00, 'expiration': exp}, 1.00, LEVELS,
                          now=now, check_expiration=True)
        self.assertEqual(d.action, 'hold')

    def test_stale_fires_past_cap(self):
        now = datetime(2025, 1, 10, 12, 0, 0)
        entry = (now - timedelta(days=6)).isoformat()
        d = evaluate_exit({'entry_price': 1.00, 'entry_time': entry}, 1.02, LEVELS,
                          now=now, max_hold_days=5)
        self.assertEqual(d.action, 'stale')

    def test_stale_not_fired_within_cap(self):
        now = datetime(2025, 1, 10, 12, 0, 0)
        entry = (now - timedelta(days=2)).isoformat()
        d = evaluate_exit({'entry_price': 1.00, 'entry_time': entry}, 1.02, LEVELS,
                          now=now, max_hold_days=5)
        self.assertEqual(d.action, 'hold')

    def test_hold_when_nothing_triggers(self):
        d = evaluate_exit({'entry_price': 1.00}, 1.05, LEVELS)
        self.assertEqual(d.action, 'hold')
        self.assertFalse(d.should_exit)

    def test_failopen_on_zero_entry(self):
        d = evaluate_exit({'entry_price': 0}, 1.0, LEVELS)
        self.assertFalse(d.should_exit)

    def test_failopen_on_bad_price(self):
        d = evaluate_exit({'entry_price': 'oops'}, 1.0, LEVELS)
        self.assertFalse(d.should_exit)

    def test_failopen_on_zero_current(self):
        d = evaluate_exit({'entry_price': 1.00}, 0, LEVELS)
        self.assertFalse(d.should_exit)


class TestFormatExitLog(unittest.TestCase):
    def test_contains_all_required_fields(self):
        trade = {'underlying_symbol': 'SPY', 'symbol': 'SPY250117C00500000',
                 'entry_price': 1.50}
        line = format_exit_log('telegram', trade, 1.20, 'stop_loss', -20.0)
        for token in ('source=telegram', 'symbol=SPY',
                      'contract=SPY250117C00500000', 'entry_price=1.50',
                      'current_price=1.20', 'pnl_percent=-20.00',
                      'exit_reason=stop_loss'):
            self.assertIn(token, line)

    def test_falls_back_to_ticker_when_no_underlying(self):
        trade = {'ticker': 'AAPL', 'symbol': 'AAPL...C', 'entry_price': 2.0}
        line = format_exit_log('scheduler', trade, 1.0, 'dynamic_take_profit', 5.0)
        self.assertIn('symbol=AAPL', line)
        self.assertIn('exit_reason=dynamic_take_profit', line)


class _FakeTrader:
    """Records calls so we can assert close + record happen exactly once."""
    def __init__(self):
        self.closed = []
        self.recorded = []

    def close_position(self, trade, position, reason):
        self.closed.append((trade['symbol'], reason))

    def record_trade_outcome(self, trade, outcome, pnl_percent=0):
        self.recorded.append((trade['symbol'], outcome, pnl_percent))


class TestEnforceExit(unittest.TestCase):
    def test_closes_and_records_once_with_reason_code(self):
        trader = _FakeTrader()
        trade = {'symbol': 'SPY...C', 'entry_price': 1.0, 'entry_time': 'x'}
        position = {'qty': '1', 'current_price': '0.9'}
        enforce_exit(trader, trade, position, 'dynamic_stop_loss', -10.0,
                     'scheduler', current_price=0.9)
        self.assertEqual(trader.closed, [('SPY...C', 'dynamic_stop_loss')])
        self.assertEqual(trader.recorded, [('SPY...C', 'dynamic_stop_loss', -10.0)])
        # exactly once each
        self.assertEqual(len(trader.closed), 1)
        self.assertEqual(len(trader.recorded), 1)

    def test_telegram_uses_action_reason(self):
        trader = _FakeTrader()
        trade = {'symbol': 'AAPL...P', 'entry_price': 2.0, 'entry_time': 'x'}
        position = {'qty': '2', 'current_price': '2.5'}
        enforce_exit(trader, trade, position, 'take_profit', 25.0, 'telegram')
        self.assertEqual(trader.closed, [('AAPL...P', 'take_profit')])
        self.assertEqual(trader.recorded, [('AAPL...P', 'take_profit', 25.0)])


if __name__ == '__main__':
    unittest.main()
