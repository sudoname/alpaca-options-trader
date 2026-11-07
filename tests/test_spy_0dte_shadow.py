"""Validate the read-only adapter and timed shadow loop with a synthetic feed."""

import unittest
from datetime import timedelta
from dataclasses import replace
from unittest.mock import patch

from oracle.lab.spy_0dte_shadow import AlpacaMarketData, FeedError, market_time, next_session, run_session
from oracle.lab.spy_put_roll import Config
from tests.test_spy_put_roll import CLOSE, FIRST, OPEN, SECOND, SESSION, bar, quote


class Clock:
    def __init__(self):
        self.time = OPEN + timedelta(minutes=10, seconds=55)

    def now(self):
        return self.time

    def sleep(self, seconds):
        self.time += timedelta(seconds=seconds)


class FakeFeed:
    def __init__(self, clock):
        self.clock = clock
        self.bars_for_day = [
            *[bar(minute, 760 - minute * 0.22) for minute in range(1, 11)],
            bar(11, 755), bar(12, 752.5), bar(13, 751.5),
        ]

    def bars(self, start, end):
        if end >= OPEN + timedelta(minutes=20):
            raise FeedError("/v2/stocks/bars", 503)
        for event in self.bars_for_day:
            if start < event.event_timestamp <= end:
                yield replace(event, received_timestamp=self.clock.now())

    def quotes(self, symbols):
        now = self.clock.now()
        elapsed = (now - OPEN).total_seconds()
        minute, second = divmod(int(elapsed), 60)
        prices = {
            FIRST: (1.24, 1.26) if elapsed < 12 * 60 else (3.50, 3.53),
            SECOND: (0.78, 0.81) if elapsed < 14 * 60 else (1.75, 1.78),
        }
        for symbol in symbols:
            event = quote(minute, second, symbol, *prices[symbol])
            yield replace(event, received_timestamp=now)


class TestShadow(unittest.TestCase):
    def test_calendar_uses_holidays_early_close_and_eastern_timezone(self):
        feed = AlpacaMarketData("key", "secret", "sip", "opra")
        with patch.object(feed, "calendar", return_value=[
            {"date": "2026-11-25", "open": "09:30", "close": "16:00"},
            {"date": "2026-11-27", "open": "09:30", "close": "13:00"},
        ]):
            previous, date, opened, closed = next_session(feed, market_time("2026-11-26", "09:30"))
        self.assertEqual(previous["date"], "2026-11-25")
        self.assertEqual(date, "2026-11-27")
        self.assertEqual(closed - opened, timedelta(hours=3, minutes=30))
        self.assertEqual(opened.hour, 14)

    def test_alpaca_bar_start_is_converted_to_completed_time(self):
        feed = AlpacaMarketData("key", "secret", "sip", "opra")
        with patch.object(feed, "get", return_value={"bars": {"SPY": [
            {"t": OPEN.isoformat(), "o": 755, "h": 756, "l": 754, "c": 755, "v": 100},
            {"t": (OPEN + timedelta(minutes=1)).isoformat(),
             "o": 755, "h": 756, "l": 754, "c": 755, "v": 100},
        ]}}), patch("oracle.lab.spy_0dte_shadow.utcnow", return_value=OPEN + timedelta(seconds=62)):
            bars = list(feed.bars(OPEN, OPEN + timedelta(minutes=1)))
        self.assertEqual([bar.event_timestamp for bar in bars], [OPEN + timedelta(minutes=1)])

    def test_automatic_session_logs_fill_sequence_and_directional_abstention(self):
        clock = Clock()
        journal = []
        # A short synthetic session keeps the integration test small.
        session = replace(SESSION, closed=CLOSE - timedelta(minutes=300))
        run_session(FakeFeed(clock), session, {FIRST: 751, SECOND: 748}, ("put", "call"),
                    Config(), "capped_roll", journal.append, 3,
                    now=clock.now, sleep=clock.sleep)
        summary = journal[-1]
        self.assertEqual(summary["event"], "session_summary")
        self.assertEqual(summary["outcome"], "completed")
        self.assertGreater(summary["feed_errors"], 0)
        put, call = summary["results"]
        self.assertTrue(put["flat_at_end"])
        self.assertEqual([order["reason"] for order in put["orders"] if order["filled_qty"]],
                         ["opening_breakdown", "roll_winner", "roll_entry", "time_exit"])
        self.assertEqual(call["entries"], 0)
        self.assertTrue(all(item["direction"] == "put" for item in journal
                            if item["event"].startswith("simulated_")))
        self.assertEqual(sum(item["event"] == "simulated_intent" for item in journal), 4)
