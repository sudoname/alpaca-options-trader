"""
Oracle Engine — Upgrade A: the unified event-driven simulation engine.

An ADAPTER over the existing pure decision functions. Backtest, shadow, paper,
and live all speak one event language and share one strategy body; they differ
only by feed + execution adapter + clock. The engine owns no network, no creds,
and (in sim) no wall-clock, so a replay is deterministic and reproducible.

Flag-gated by ``ENABLE_EVENT_DRIVEN_SIMULATION`` (default OFF). Importing this
package has NO effect on the live ``smart_trader`` decision path.

Layers:
  * ``events``           — frozen event dataclasses + the event-type vocabulary.
  * ``clock``            — ``Clock`` protocol, ``LiveClock``, ``SimulationClock``.
  * ``bus``              — deterministic timestamp-ordered ``EventBus``.
  * ``strategy``         — ``Strategy`` protocol + ``OracleStrategyAdapter``.
  * ``sim_broker``       — ``SimBroker`` (deterministic fills via fill_simulator).
  * ``backtest_driver``  — wires the above into a reproducible offline run.
"""

from __future__ import annotations

from oracle.engine.backtest_driver import (
    BacktestDriver,
    BacktestResult,
    DriverConfig,
)
from oracle.engine.bus import EventBus
from oracle.engine.clock import Clock, LiveClock, SimulationClock
from oracle.engine.events import (
    BAR,
    CATALYST,
    EVENT_TYPES,
    NEWS,
    OPTION_QUOTE,
    ORDER_CANCELLED,
    ORDER_FILLED,
    ORDER_PARTIAL_FILL,
    ORDER_SUBMITTED,
    POSITION_UPDATED,
    QUOTE,
    SESSION_CLOSE,
    SESSION_OPEN,
    SIGNAL,
    TIMER,
    TRADE_INTENT,
    BarEvent,
    CatalystEvent,
    Event,
    NewsEvent,
    OptionQuoteEvent,
    OrderEvent,
    PositionEvent,
    QuoteEvent,
    SessionEvent,
    SignalEvent,
    TimerEvent,
    TradeIntentEvent,
)
from oracle.engine.sim_broker import BrokerFill, SimBroker
from oracle.engine.strategy import (
    OracleStrategyAdapter,
    Strategy,
    StrategyConfig,
)

__all__ = [
    # events
    "Event", "BarEvent", "QuoteEvent", "OptionQuoteEvent", "NewsEvent",
    "CatalystEvent", "TimerEvent", "SessionEvent", "SignalEvent",
    "TradeIntentEvent", "OrderEvent", "PositionEvent", "EVENT_TYPES",
    "BAR", "QUOTE", "OPTION_QUOTE", "NEWS", "CATALYST", "SIGNAL",
    "TRADE_INTENT", "ORDER_SUBMITTED", "ORDER_FILLED", "ORDER_PARTIAL_FILL",
    "ORDER_CANCELLED", "POSITION_UPDATED", "SESSION_OPEN", "SESSION_CLOSE",
    "TIMER",
    # clock
    "Clock", "LiveClock", "SimulationClock",
    # bus
    "EventBus",
    # strategy
    "Strategy", "StrategyConfig", "OracleStrategyAdapter",
    # broker
    "SimBroker", "BrokerFill",
    # driver
    "BacktestDriver", "DriverConfig", "BacktestResult",
]
