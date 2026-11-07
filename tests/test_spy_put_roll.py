"""Regression coverage for delayed fills, exit-before-roll and buying power."""

import unittest
from dataclasses import replace
from datetime import timedelta

from oracle.engine.events import BarEvent, OptionQuoteEvent
from oracle.lab.spy_put_roll import Config, Session, replay, timestamp


OPEN = timestamp("2026-11-05T09:30:00-05:00")
CLOSE = OPEN + timedelta(hours=6, minutes=30)
SESSION = Session(OPEN, CLOSE, 765)
FIRST = "SPY261105P00751000"
SECOND = "SPY261105P00748000"


def bar(minute, close):
    return BarEvent(symbol="SPY", event_timestamp=OPEN + timedelta(minutes=minute),
                    o=760 if minute == 1 else close + 0.1,
                    h=760.2 if minute == 1 else close + 0.2,
                    l=close - 0.1, c=close, v=1000, timeframe="1Min")


def quote(minute, second, symbol, bid, ask, *, size=20):
    return OptionQuoteEvent(
        symbol=symbol, underlying="SPY",
        event_timestamp=OPEN + timedelta(minutes=minute, seconds=second),
        bid=bid, ask=ask, data={"bid_size": size, "ask_size": size},
    )


def feed():
    return [
        *[bar(minute, 760 - minute * 0.22) for minute in range(1, 11)],
        quote(10, 59, FIRST, 1.24, 1.26), bar(11, 755),
        quote(11, 0, FIRST, 1.24, 1.26),
        quote(11, 1, FIRST, 1.24, 1.26),
        quote(11, 59, FIRST, 3.50, 3.53), bar(12, 752.5),
        quote(12, 1, FIRST, 3.50, 3.53),
        quote(12, 2, FIRST, 3.50, 3.53),
        quote(12, 59, SECOND, 0.78, 0.81), bar(13, 751.5),
        quote(13, 1, SECOND, 0.78, 0.81),
        quote(350, 1, FIRST, 4.00, 4.03),
        quote(350, 1, SECOND, 1.75, 1.78),
    ]


def fills(result):
    return [order for order in result["orders"] if order["filled_qty"]]


class TestPutReplay(unittest.TestCase):
    def test_roll_after_fill_and_compound_realized_proceeds(self):
        result = replay(SESSION, reversed(feed()))
        trades = fills(result)
        self.assertEqual([(t["side"], t["contract"], t["filled_qty"]) for t in trades],
                         [("buy", FIRST, 1), ("sell", FIRST, 1),
                          ("buy", SECOND, 3), ("sell", SECOND, 3)])
        self.assertEqual(timestamp(trades[0]["ts"]), OPEN + timedelta(minutes=11, seconds=1))
        self.assertTrue(result["flat_at_end"])
        self.assertAlmostEqual(result["net_realized_pnl"], result["ending_cash"] - 700, places=2)
        self.assertEqual(replay(SESSION, feed()), result)

    def test_fixed_and_hold_use_same_entry_but_distinct_roll_rules(self):
        fixed = fills(replay(SESSION, feed(), profile="fixed_roll"))
        hold = fills(replay(SESSION, feed(), profile="hold"))
        self.assertEqual(fixed[0], hold[0])
        self.assertEqual(fixed[2]["filled_qty"], 2)
        self.assertEqual([trade["reason"] for trade in hold], ["opening_breakdown", "time_exit"])

    def test_partial_exit_and_missing_liquidity_prevent_reentry(self):
        events = [
            replace(event, data={"bid_size": 1, "ask_size": 20})
            if event == quote(12, 1, FIRST, 3.50, 3.53) else event
            for event in feed()
            if event not in (quote(12, 2, FIRST, 3.50, 3.53),
                             quote(350, 1, FIRST, 4.00, 4.03))
        ]
        config = Config(base_premium=300, daily_loss=300)
        result = replay(SESSION, events, config)
        self.assertEqual([order["filled_qty"] for order in fills(result)], [2, 1])
        self.assertEqual(result["open_positions"], {FIRST: 1.0})
        self.assertFalse(result["flat_at_end"])

    def test_cash_profile_caps_redeployment_at_remaining_settled_budget(self):
        config = Config(starting_cash=200, reuse_sale_proceeds=False)
        trades = fills(replay(SESSION, feed(), config))
        self.assertEqual([order["contract"] for order in trades], [FIRST, FIRST])

    def test_delayed_quote_cannot_fill_and_late_option_is_filtered(self):
        events = feed()[:12] + [
            replace(quote(11, 0, FIRST, 1.24, 1.26),
                    received_timestamp=OPEN + timedelta(minutes=11, seconds=2)),
            quote(11, 3, "SPY261106P00751000", 1.24, 1.26),
        ]
        self.assertEqual(fills(replay(SESSION, events)), [])

    def test_small_balance_and_incomplete_opening_range_abstain(self):
        self.assertEqual(fills(replay(SESSION, feed(), Config(starting_cash=50))), [])
        self.assertEqual(fills(replay(SESSION, feed()[1:])), [])
