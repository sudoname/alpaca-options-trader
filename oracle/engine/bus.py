"""
Oracle Engine — Upgrade A: deterministic in-memory event bus.

The bus is the spine of the simulation engine. It is intentionally boring:

  * A single-threaded, synchronous priority queue ordered by
    ``(event_timestamp, seq)`` where ``seq`` is a monotonically increasing
    publish counter. No threads, no async, no wall-clock -> the same events
    published in the same order always dispatch in the same order. That is the
    Upgrade-1 replay invariant.
  * Handlers subscribe to an ``event_type`` (or to ALL events). Dispatch calls
    every matching handler in subscription order. A handler may publish new
    events (a strategy emits TRADE_INTENT; a broker emits ORDER_FILLED); those
    are queued with a fresh ``seq`` and processed in timestamp order, so an
    order filled "later" than a subsequent bar is still processed after it.
  * When a ``SimulationClock`` is attached, the bus advances it to each event's
    timestamp BEFORE dispatch, so any handler reading ``clock.now()`` sees the
    event's own time.
  * FAIL-OPEN: a handler that raises is logged into ``self.errors`` and does not
    abort the run (a single broken subscriber never takes down a replay).

Ordering key detail: ``event_timestamp`` may be a ``datetime`` or ``None``.
``None`` sorts before any real time (treated as "beginning of time") so
un-timestamped control events lead. Two events with equal timestamps keep
publish order via ``seq``.
"""

from __future__ import annotations

import heapq
from collections import defaultdict
from datetime import datetime
from typing import Any, Callable, DefaultDict, List, Optional, Tuple

from oracle.engine.events import Event

Handler = Callable[[Event], None]


def _ts_key(ts: Any) -> Tuple[int, float]:
    """Total order over (possibly None) timestamps.

    Returns a (bucket, value) pair: bucket 0 = None (sorts first), bucket 1 =
    real time as an ordinal float. datetimes are converted to an ordinal via
    ``.toordinal()`` + seconds-of-day so naive datetimes compare correctly
    without timezone/epoch ambiguity.
    """
    if ts is None:
        return (0, 0.0)
    if isinstance(ts, datetime):
        secs = (ts.hour * 3600 + ts.minute * 60 + ts.second
                + ts.microsecond / 1_000_000.0)
        return (1, ts.toordinal() * 86400.0 + secs)
    try:
        return (1, float(ts))
    except (TypeError, ValueError):
        return (0, 0.0)


class EventBus:
    """Deterministic synchronous event bus."""

    def __init__(self, clock: Any = None) -> None:
        self.clock = clock
        self._heap: List[Tuple[Tuple[int, float], int, Event]] = []
        self._seq = 0
        self._subs: DefaultDict[str, List[Handler]] = defaultdict(list)
        self._all: List[Handler] = []
        self.dispatched: List[Event] = []
        self.errors: List[Tuple[str, str]] = []   # (event_type, repr(exc))
        self._running = False

    # -- subscription ----------------------------------------------------- #
    def subscribe(self, event_type: str, handler: Handler) -> None:
        self._subs[event_type].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        self._all.append(handler)

    # -- publishing ------------------------------------------------------- #
    def publish(self, event: Event) -> None:
        """Queue an event. ``seq`` preserves publish order for equal timestamps."""
        self._seq += 1
        key = _ts_key(getattr(event, "event_timestamp", None))
        heapq.heappush(self._heap, (key, self._seq, event))

    def publish_all(self, events) -> None:
        for ev in events or ():
            if isinstance(ev, Event):
                self.publish(ev)

    # -- draining --------------------------------------------------------- #
    def _dispatch(self, event: Event) -> None:
        if isinstance(self.clock, object) and hasattr(self.clock, "advance_to"):
            try:
                self.clock.advance_to(getattr(event, "event_timestamp", None))
            except Exception:
                pass
        self.dispatched.append(event)
        for handler in list(self._all) + list(self._subs.get(event.event_type, ())):
            try:
                handler(event)
            except Exception as exc:  # fail-open: one bad handler != dead run
                self.errors.append((event.event_type, repr(exc)))

    def step(self) -> Optional[Event]:
        """Process exactly one event (the earliest). Returns it, or None when
        the queue is empty."""
        if not self._heap:
            return None
        _, _, event = heapq.heappop(self._heap)
        self._dispatch(event)
        return event

    def run(self, max_events: int = 1_000_000) -> int:
        """Drain the queue in deterministic order until empty (or the safety cap
        is hit). Handlers publishing new events extend the drain. Returns the
        number of events dispatched."""
        if self._running:  # re-entrancy guard
            return 0
        self._running = True
        n = 0
        try:
            while self._heap and n < max_events:
                _, _, event = heapq.heappop(self._heap)
                self._dispatch(event)
                n += 1
        finally:
            self._running = False
        return n

    @property
    def pending(self) -> int:
        return len(self._heap)


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from oracle.engine.clock import SimulationClock
    from oracle.engine.events import BarEvent, QuoteEvent, TIMER, TimerEvent

    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)
    t1 = datetime(2026, 8, 11, 13, 31, 0)
    t2 = datetime(2026, 8, 11, 13, 32, 0)

    clock = SimulationClock()
    bus = EventBus(clock)
    seen: List[str] = []

    bus.subscribe_all(lambda e: seen.append(e.event_type))

    # Publish OUT OF ORDER; the bus must dispatch in timestamp order.
    bus.publish(QuoteEvent(symbol="X", event_timestamp=t2, bid=1, ask=2))
    bus.publish(BarEvent(symbol="X", event_timestamp=t0, c=1.0))
    bus.publish(QuoteEvent(symbol="X", event_timestamp=t1, bid=1, ask=2))
    n = bus.run()
    if n != 3:
        print("FAIL: dispatched count", n); ok = False
    if [e.event_timestamp for e in bus.dispatched] != [t0, t1, t2]:
        print("FAIL: not timestamp-ordered",
              [e.event_timestamp for e in bus.dispatched]); ok = False
    # Clock advanced to the last event time.
    if clock.now() != t2:
        print("FAIL: clock not advanced by bus", clock.now()); ok = False

    # Equal-timestamp events keep PUBLISH order (seq tie-break).
    bus2 = EventBus()
    order: List[str] = []
    bus2.subscribe(TIMER, lambda e: order.append(e.data.get("tag", "")))
    bus2.publish(TimerEvent(event_timestamp=t0, data={"tag": "a"}))
    bus2.publish(TimerEvent(event_timestamp=t0, data={"tag": "b"}))
    bus2.publish(TimerEvent(event_timestamp=t0, data={"tag": "c"}))
    bus2.run()
    if order != ["a", "b", "c"]:
        print("FAIL: seq tie-break order", order); ok = False

    # Type routing: a BAR handler must not see QUOTE events.
    bus3 = EventBus()
    bars: List[Event] = []
    bus3.subscribe("BAR", lambda e: bars.append(e))
    bus3.publish(BarEvent(symbol="X", event_timestamp=t0, c=1.0))
    bus3.publish(QuoteEvent(symbol="X", event_timestamp=t1, bid=1, ask=2))
    bus3.run()
    if len(bars) != 1 or bars[0].event_type != "BAR":
        print("FAIL: type routing", bars); ok = False

    # A handler that publishes MORE events extends the drain deterministically.
    bus4 = EventBus()
    tally: List[str] = []

    def cascader(e: Event) -> None:
        tally.append(e.event_type)
        if e.event_type == "BAR":
            bus4.publish(TimerEvent(event_timestamp=e.event_timestamp,
                                    data={"tag": "derived"}))

    bus4.subscribe_all(cascader)
    bus4.publish(BarEvent(symbol="X", event_timestamp=t0, c=1.0))
    total = bus4.run()
    if total != 2 or tally != ["BAR", "TIMER"]:
        print("FAIL: cascade", total, tally); ok = False

    # Fail-open: a raising handler is recorded, not fatal.
    bus5 = EventBus()

    def boom(e: Event) -> None:
        raise ValueError("kaboom")

    reached: List[str] = []
    bus5.subscribe("BAR", boom)
    bus5.subscribe("BAR", lambda e: reached.append("after"))
    bus5.publish(BarEvent(symbol="X", event_timestamp=t0, c=1.0))
    bus5.run()
    if not bus5.errors or reached != ["after"]:
        print("FAIL: fail-open handler", bus5.errors, reached); ok = False

    # Determinism: same publishes -> same dispatch order every time.
    def build_order():
        b = EventBus()
        out: List[Tuple[int, float]] = []
        b.subscribe_all(lambda e: out.append(_ts_key(e.event_timestamp)))
        for ts in (t2, t0, t1, t0):
            b.publish(BarEvent(symbol="X", event_timestamp=ts, c=1.0))
        b.run()
        return out

    if build_order() != build_order():
        print("FAIL: non-deterministic ordering"); ok = False

    print("engine.bus self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
