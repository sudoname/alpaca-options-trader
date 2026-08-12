"""
Oracle Engine — Upgrade A: the clock abstraction.

All time-based logic in the SIM path reads a ``Clock``; it never calls
``datetime.now()`` directly. That is what makes a backtest deterministic and a
replay reproducible: the ``SimulationClock`` advances only as the event stream
tells it to, so "now" during a replay is exactly the event time being processed.

  * ``LiveClock``       — wall-clock; ``now()`` == ``datetime.now()``. Live/paper.
  * ``SimulationClock`` — deterministic; ``now()`` returns the current simulated
    time, advanced monotonically via ``advance_to(ts)`` by the bus/driver.

Both satisfy the ``Clock`` protocol so decision code can hold a ``Clock`` and be
agnostic to whether it is live or replaying.
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional, Protocol, runtime_checkable


@runtime_checkable
class Clock(Protocol):
    def now(self) -> datetime:
        ...


class LiveClock:
    """Wall-clock. Used by the live/paper feed."""

    name = "live"

    def now(self) -> datetime:
        return datetime.now()


class SimulationClock:
    """Deterministic clock driven by the event stream.

    Time only ever moves FORWARD: ``advance_to`` ignores timestamps earlier than
    the current time (a late-arriving lower-timestamp event does not rewind the
    world). ``now()`` before the first advance returns the configured start.
    """

    name = "sim"

    def __init__(self, start: Optional[datetime] = None) -> None:
        self._t: Optional[datetime] = start

    def now(self) -> datetime:
        # Fail-open: if never seeded, fall back to epoch-min so callers always
        # get a datetime rather than None.
        return self._t if self._t is not None else datetime.min

    def advance_to(self, ts: Optional[datetime]) -> datetime:
        """Move the clock forward to ``ts`` (monotonic). Returns the new time."""
        if isinstance(ts, datetime):
            if self._t is None or ts > self._t:
                self._t = ts
        return self.now()

    def reset(self, start: Optional[datetime] = None) -> None:
        self._t = start


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)
    t1 = datetime(2026, 8, 11, 14, 0, 0)

    sim = SimulationClock(t0)
    if sim.now() != t0:
        print("FAIL: sim start", sim.now()); ok = False

    sim.advance_to(t1)
    if sim.now() != t1:
        print("FAIL: sim advance", sim.now()); ok = False

    # Monotonic: an earlier timestamp must not rewind the clock.
    sim.advance_to(t0)
    if sim.now() != t1:
        print("FAIL: sim rewound", sim.now()); ok = False

    # Unseeded clock is fail-open (datetime.min, never None).
    empty = SimulationClock()
    if empty.now() != datetime.min:
        print("FAIL: unseeded sim clock", empty.now()); ok = False

    # advance_to ignores non-datetime input (fail-open).
    empty.advance_to("not-a-date")
    if empty.now() != datetime.min:
        print("FAIL: bad advance input", empty.now()); ok = False

    # Protocol conformance.
    if not isinstance(sim, Clock) or not isinstance(LiveClock(), Clock):
        print("FAIL: Clock protocol conformance"); ok = False

    # LiveClock returns a datetime.
    if not isinstance(LiveClock().now(), datetime):
        print("FAIL: live clock now()"); ok = False

    print("engine.clock self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
