"""
Upgrade 4 — temporal-integrity / look-ahead tests.

These pin the ``oracle.temporal`` contract AND the ``market_view.MarketView``
point-in-time guarantee it formalizes:

  1. A future daily bar, earnings, analyst rating and option quote are each
     rejected as features at a decision ``as_of`` that precedes their
     availability.
  2. A rolling indicator (SMA) computed through a ``HistoricalMarketView`` uses
     ONLY bars whose session close <= as_of.
  3. A candlestick/intraday feature requires a COMPLETED bar (the window must
     fully elapse before the bar is knowable).
  4. A realized outcome (forward return, measured after the decision) cannot
     enter the feature set — the label-leakage guard.
  5. ``assert_no_lookahead`` folds a ``MarketView.audit`` and raises in strict
     mode, logs + returns False (fail-open) otherwise.

No creds, no network, no order placement.
"""

import os
import sys
import unittest
from datetime import datetime, time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from market_view import HistoricalMarketView, Quote, make_bar, make_intraday_bar
from oracle.temporal import (
    LookAheadError,
    TemporalGuard,
    TemporalStamp,
    assert_no_lookahead,
    conservative_available_ts,
    validate_feature,
)

DECISION = datetime(2024, 1, 5, 16, 0)  # Friday close


class TestFutureDatumRejected(unittest.TestCase):
    def test_future_bar_rejected(self):
        s = TemporalStamp.make("2024-01-08", DECISION, kind="daily_bar")
        ok, reason = validate_feature(s)
        self.assertFalse(ok)
        self.assertEqual(reason, "available_after_decision")

    def test_future_earnings_rejected(self):
        s = TemporalStamp.make("2024-01-06T08:00:00", DECISION, kind="earnings")
        self.assertFalse(s.is_valid())

    def test_future_analyst_rating_rejected(self):
        # Published after Friday's close -> not usable in the Friday decision.
        s = TemporalStamp.make("2024-01-05T16:30:00", DECISION,
                               kind="analyst_rating")
        self.assertFalse(s.is_valid())

    def test_future_option_quote_rejected(self):
        s = TemporalStamp.make("2024-01-05T16:00:01", DECISION,
                               kind="option_quote")
        self.assertFalse(s.is_valid())

    def test_prior_datum_valid(self):
        self.assertTrue(
            TemporalStamp.make("2024-01-04", DECISION, kind="daily_bar")
            .is_valid())
        self.assertTrue(
            TemporalStamp.make("2024-01-05T15:59:59", DECISION,
                               kind="option_quote").is_valid())


class TestRollingIndicatorNoLeak(unittest.TestCase):
    def _daily(self):
        return {"AAA": [
            make_bar("2024-01-02", 100, 101, 99, 100, 1e6),
            make_bar("2024-01-03", 100, 102, 99, 101, 1e6),
            make_bar("2024-01-04", 101, 103, 100, 102, 1e6),
            make_bar("2024-01-05", 102, 104, 101, 103, 1e6),   # decision day
            make_bar("2024-01-08", 103, 106, 102, 105, 1e6),   # FUTURE
        ]}

    def test_sma_uses_only_known_bars(self):
        # Decide at Friday CLOSE -> Friday's bar is known, Monday's is not.
        mv = HistoricalMarketView(DECISION, daily=self._daily())
        bars = mv.daily_bars("AAA", 30)
        self.assertEqual([b.date for b in bars],
                         ["2024-01-02", "2024-01-03", "2024-01-04",
                          "2024-01-05"])
        closes = [b.c for b in bars]
        sma = sum(closes) / len(closes)
        # Monday's 105 close must NOT be in the average.
        self.assertNotIn(105.0, closes)
        self.assertAlmostEqual(sma, (100 + 101 + 102 + 103) / 4.0)
        # And the audit is clean at this as_of.
        self.assertTrue(assert_no_lookahead(mv, strict=True))

    def test_decision_at_open_excludes_current_day(self):
        # Decide at Friday's OPEN (09:30) -> Friday's completed daily bar is not
        # yet knowable; the rolling window ends Thursday.
        mv = HistoricalMarketView(datetime(2024, 1, 5, 9, 30),
                                  daily=self._daily())
        bars = mv.daily_bars("AAA", 30)
        self.assertEqual(bars[-1].date, "2024-01-04")


class TestCandlestickRequiresCompletedBar(unittest.TestCase):
    def test_intraday_bar_requires_full_window(self):
        # A 5-min bar starting 09:30 only becomes available at 09:35.
        av = conservative_available_ts("2024-01-05T09:30:00", "intraday_bar",
                                       interval_minutes=5)
        self.assertEqual(av, datetime(2024, 1, 5, 9, 35))
        # Deciding at 09:34 -> not yet complete -> invalid.
        dec_early = datetime(2024, 1, 5, 9, 34)
        s_early = TemporalStamp.make("2024-01-05T09:30:00", dec_early,
                                     kind="intraday_bar", interval_minutes=5)
        self.assertFalse(s_early.is_valid())
        # Deciding at 09:35 -> complete -> valid.
        dec_ok = datetime(2024, 1, 5, 9, 35)
        s_ok = TemporalStamp.make("2024-01-05T09:30:00", dec_ok,
                                  kind="intraday_bar", interval_minutes=5)
        self.assertTrue(s_ok.is_valid())

    def test_marketview_intraday_bar_none_during_warmup(self):
        bar = make_bar("2024-01-05", 100, 101, 99, 100.5, 1e5)
        mv = HistoricalMarketView(datetime(2024, 1, 5, 9, 45),
                                  intraday={"AAA": bar})
        # 30-min opening range not complete at 09:45.
        self.assertIsNone(mv.intraday_bar("AAA", 30))
        mv2 = HistoricalMarketView(datetime(2024, 1, 5, 10, 5),
                                   intraday={"AAA": bar})
        self.assertIsNotNone(mv2.intraday_bar("AAA", 30))


class TestOutcomeCannotLeak(unittest.TestCase):
    def test_forward_return_is_future(self):
        # Outcome measured at horizon end (next week) is a future datum.
        s = TemporalStamp.make("2024-01-12T16:00:00", DECISION,
                               kind="daily_bar", ident="AAA:forward_return")
        self.assertFalse(s.is_valid())

    def test_option_quote_after_as_of_filtered_by_marketview(self):
        q = Quote(1.10, 1.20, datetime(2024, 1, 5, 16, 0, 1))  # 1s post-close
        mv = HistoricalMarketView(DECISION, quotes={"AAA240105C00100000": q})
        self.assertIsNone(mv.option_quote("AAA240105C00100000"))


class TestAssertNoLookahead(unittest.TestCase):
    class _FakeMV:
        def __init__(self, as_of, audit):
            self.as_of = as_of
            self.audit = audit

    def test_clean_passes(self):
        mv = self._FakeMV(DECISION, [{"kind": "daily_bar",
                                      "ts": datetime(2024, 1, 4, 16),
                                      "id": "AAA"}])
        self.assertTrue(assert_no_lookahead(mv, strict=True))

    def test_leak_strict_raises(self):
        mv = self._FakeMV(DECISION, [{"kind": "daily_bar",
                                      "ts": datetime(2024, 1, 9, 16),
                                      "id": "AAA"}])
        with self.assertRaises(LookAheadError):
            assert_no_lookahead(mv, strict=True)

    def test_leak_nonstrict_returns_false(self):
        mv = self._FakeMV(DECISION, [{"kind": "daily_bar",
                                      "ts": datetime(2024, 1, 9, 16),
                                      "id": "AAA"}])
        self.assertFalse(assert_no_lookahead(mv, strict=False))


class TestTemporalGuard(unittest.TestCase):
    def test_guard_collects_violation_nonstrict(self):
        with TemporalGuard(DECISION, strict=False) as g:
            self.assertTrue(g.check("daily_bar", "2024-01-04"))
            self.assertFalse(g.check("daily_bar", "2024-01-09"))
        self.assertFalse(g.ok())
        self.assertEqual(len(g.violations), 1)

    def test_guard_strict_raises(self):
        with self.assertRaises(LookAheadError):
            with TemporalGuard(DECISION, strict=True) as g:
                g.check("daily_bar", "2024-01-09")


def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print("tests.test_temporal_integrity self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
