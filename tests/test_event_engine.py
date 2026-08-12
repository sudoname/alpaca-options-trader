"""
Phase-2 Upgrade A — unified event-driven simulation engine tests.

These pin the engine's Upgrade-A invariants:

  1. Determinism: the same ``(feed, config, seed)`` produces a byte-identical
     order log + realized P&L (the acceptance invariant).
  2. Bus ordering: events published out of order dispatch in timestamp order,
     and equal timestamps keep publish order (seq tie-break).
  3. Clock control: a ``SimulationClock`` advances only from the event stream,
     monotonically; ``now()`` during dispatch is the event's own time.
  4. Adapter philosophy: direction is an OUTPUT of the probability — a flat read
     abstains (NO_TRADE), a bullish read emits a CALL, a bearish read a PUT.
  5. Sim/paper parity: the SAME adapter body driven by two different feed
     orderings (a "backtest" feed vs a shuffled "paper-mock" feed with identical
     timestamps) yields identical signals + orders.
  6. LLM determinism guard: the driver refuses an implicit live LLM binding.
  7. Look-ahead guard: a future-timestamped feature is rejected by
     ``oracle.temporal`` (the engine never lets a not-yet-available bar decide).

No creds, no network, no order placement.
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.engine.backtest_driver import BacktestDriver, DriverConfig
from oracle.engine.bus import EventBus, _ts_key
from oracle.engine.clock import SimulationClock
from oracle.engine.events import (
    BarEvent,
    QuoteEvent,
    TimerEvent,
    TRADE_INTENT,
)
from oracle.engine.strategy import OracleStrategyAdapter, StrategyConfig

T0 = datetime(2026, 8, 11, 13, 30, 0)
T1 = datetime(2026, 8, 11, 13, 31, 0)
T2 = datetime(2026, 8, 11, 13, 32, 0)


# --- shared cached/mocked Oracle head + option quote (deterministic) -------- #
def _stub_explain(ticker, ctx, prior):
    trend = (ctx or {}).get("trend", 0.0)
    if trend > 0:
        prob = {"p_call": 0.7, "p_put": 0.2, "p_no_trade": 0.1}
    elif trend < 0:
        prob = {"p_call": 0.2, "p_put": 0.7, "p_no_trade": 0.1}
    else:
        prob = {"p_call": 0.3, "p_put": 0.3, "p_no_trade": 0.4}
    return {"ticker": ticker, "probability": prob, "verdict": "OK"}


def _resolver(symbol, direction, event, clock):
    c = getattr(event, "c", 100.0) or 100.0
    px = round(c / 100.0, 4)
    return (f"{symbol}-{direction}", px, round(px + 0.02, 4),
            {"ask_size": 100, "bid_size": 100})


def _feed():
    return [
        BarEvent(symbol="AAPL", event_timestamp=T0, o=100.0, c=101.0),   # CALL
        BarEvent(symbol="MSFT", event_timestamp=T1, o=100.0, c=100.0),   # flat
        BarEvent(symbol="TSLA", event_timestamp=T2, o=100.0, c=98.0),    # PUT
    ]


def _run(seed, feed=None):
    drv = BacktestDriver(
        config=DriverConfig(seed=seed,
                            strategy=StrategyConfig(min_directional_p=0.55)),
        explain_fn=_stub_explain, contract_resolver=_resolver)
    return drv.run(feed if feed is not None else _feed())


class TestDeterminism(unittest.TestCase):
    def test_same_seed_identical_order_log_and_pnl(self):
        r1 = _run(seed=1)
        r2 = _run(seed=1)
        self.assertEqual(r1.order_log, r2.order_log)
        self.assertEqual(r1.realized_pnl, r2.realized_pnl)
        self.assertEqual(r1.to_dict(), r2.to_dict())

    def test_two_intents_and_one_abstain(self):
        r = _run(seed=1)
        self.assertEqual(r.orders, 2)
        dirs = sorted(s["direction"] for s in r.signals)
        self.assertEqual(dirs, ["CALL", "NO_TRADE", "PUT"])
        self.assertEqual(set(r.open_positions), {"AAPL-CALL", "TSLA-PUT"})

    def test_no_handler_errors(self):
        self.assertEqual(_run(seed=1).errors, [])


class TestBusOrdering(unittest.TestCase):
    def test_dispatch_is_timestamp_ordered(self):
        bus = EventBus()
        seen = []
        bus.subscribe_all(lambda e: seen.append(e.event_timestamp))
        bus.publish(QuoteEvent(symbol="X", event_timestamp=T2, bid=1, ask=2))
        bus.publish(BarEvent(symbol="X", event_timestamp=T0, c=1.0))
        bus.publish(QuoteEvent(symbol="X", event_timestamp=T1, bid=1, ask=2))
        n = bus.run()
        self.assertEqual(n, 3)
        self.assertEqual(seen, [T0, T1, T2])

    def test_equal_timestamp_keeps_publish_order(self):
        bus = EventBus()
        tags = []
        bus.subscribe("TIMER", lambda e: tags.append(e.data.get("tag")))
        for tag in ("a", "b", "c"):
            bus.publish(TimerEvent(event_timestamp=T0, data={"tag": tag}))
        bus.run()
        self.assertEqual(tags, ["a", "b", "c"])

    def test_ts_key_none_sorts_first(self):
        self.assertLess(_ts_key(None), _ts_key(T0))


class TestClockControl(unittest.TestCase):
    def test_clock_advances_from_stream_monotonically(self):
        clock = SimulationClock()
        bus = EventBus(clock)
        nows = []
        bus.subscribe_all(lambda e: nows.append(clock.now()))
        # Publish out of order; the clock must still see each event's own time
        # and never rewind.
        bus.publish(BarEvent(symbol="X", event_timestamp=T1, c=1.0))
        bus.publish(BarEvent(symbol="X", event_timestamp=T0, c=1.0))
        bus.publish(BarEvent(symbol="X", event_timestamp=T2, c=1.0))
        bus.run()
        self.assertEqual(nows, [T0, T1, T2])
        self.assertEqual(clock.now(), T2)

    def test_clock_does_not_rewind(self):
        clock = SimulationClock(T1)
        clock.advance_to(T0)
        self.assertEqual(clock.now(), T1)


class TestAdapterPhilosophy(unittest.TestCase):
    def _decide(self, bar):
        emitted = []
        a = OracleStrategyAdapter(emit=emitted.append,
                                  config=StrategyConfig(min_directional_p=0.55),
                                  explain_fn=_stub_explain,
                                  contract_resolver=_resolver)
        a.handle(bar)
        return emitted

    def test_bullish_emits_call(self):
        out = self._decide(BarEvent(symbol="AAPL", event_timestamp=T0,
                                    o=100.0, c=101.0))
        intents = [e for e in out if e.event_type == TRADE_INTENT]
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].direction, "CALL")

    def test_bearish_emits_put(self):
        out = self._decide(BarEvent(symbol="TSLA", event_timestamp=T0,
                                    o=100.0, c=98.0))
        intents = [e for e in out if e.event_type == TRADE_INTENT]
        self.assertEqual(len(intents), 1)
        self.assertEqual(intents[0].direction, "PUT")

    def test_flat_abstains(self):
        out = self._decide(BarEvent(symbol="MSFT", event_timestamp=T0,
                                    o=100.0, c=100.0))
        self.assertFalse([e for e in out if e.event_type == TRADE_INTENT])

    def test_one_intent_per_symbol(self):
        emitted = []
        a = OracleStrategyAdapter(emit=emitted.append,
                                  config=StrategyConfig(min_directional_p=0.55),
                                  explain_fn=_stub_explain,
                                  contract_resolver=_resolver)
        a.handle(BarEvent(symbol="AAPL", event_timestamp=T0, o=100.0, c=101.0))
        a.handle(BarEvent(symbol="AAPL", event_timestamp=T1, o=100.0, c=102.0))
        intents = [e for e in emitted if e.event_type == TRADE_INTENT]
        self.assertEqual(len(intents), 1)


class TestSimPaperParity(unittest.TestCase):
    def test_shuffled_feed_same_timestamps_identical_result(self):
        # "backtest" feed in natural order vs a "paper-mock" feed published in a
        # different publish order but with identical timestamps. The bus orders
        # by timestamp, so the SAME adapter body must produce identical output.
        ordered = _feed()
        shuffled = [ordered[2], ordered[0], ordered[1]]
        r_ord = _run(seed=1, feed=ordered)
        r_shuf = _run(seed=1, feed=shuffled)
        self.assertEqual(r_ord.order_log, r_shuf.order_log)
        self.assertEqual(r_ord.realized_pnl, r_shuf.realized_pnl)
        self.assertEqual([s["direction"] for s in r_ord.signals],
                         [s["direction"] for s in r_shuf.signals])


class TestLLMGuard(unittest.TestCase):
    def test_forbid_live_llm_blocks_implicit_binding(self):
        with self.assertRaises(ValueError):
            BacktestDriver(config=DriverConfig())  # no explain_fn injected

    def test_opt_out_allows_construction(self):
        # forbid_live_llm=False is the explicit escape hatch.
        BacktestDriver(config=DriverConfig(forbid_live_llm=False),
                       contract_resolver=_resolver)


class TestLookaheadGuard(unittest.TestCase):
    def test_future_feature_rejected(self):
        # The engine must never let a not-yet-available bar decide. We assert the
        # temporal contract the driver relies on: a feature available AFTER the
        # decision is invalid.
        from oracle.temporal import TemporalStamp, validate_feature
        stamp = TemporalStamp(event_timestamp=T2, available_timestamp=T2,
                              decision_timestamp=T0)
        ok, _reason = validate_feature(stamp)
        self.assertFalse(ok)

    def test_available_before_decision_ok(self):
        from oracle.temporal import TemporalStamp, validate_feature
        stamp = TemporalStamp(event_timestamp=T0, available_timestamp=T0,
                              decision_timestamp=T2)
        ok, _reason = validate_feature(stamp)
        self.assertTrue(ok)


if __name__ == "__main__":
    unittest.main()
