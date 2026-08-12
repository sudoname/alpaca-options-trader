"""
Oracle Engine — Upgrade A: event vocabulary for the unified simulation engine.

The engine is an ADAPTER over the existing pure decision functions. Backtest,
shadow, paper, and live all speak the same event language; they differ only by
feed + execution adapter + clock. Every event is a frozen dataclass so that a
replay is reproducible and comparable (two runs with the same inputs emit
byte-identical event streams).

Common fields on every event (per the Phase-2 plan):
    event_id            stable identifier for this event
    event_type          one of the EVENT_TYPES constants below
    event_timestamp     when the thing happened (drives bus ordering)
    received_timestamp  when the engine became aware of it (>= event_timestamp)
    source              free-form origin tag ("feed", "strategy", "broker", ...)
    symbol              underlying / contract symbol, "" when N/A
    strategy_mode       "intraday" | "swing" | "" (mode that produced it)
    correlation_id      links an intent -> orders -> fills -> position updates

Type-specific payload lives in typed subclasses (BarEvent, QuoteEvent, ...) plus
a generic ``data`` mapping for anything not promoted to a field. Nothing here
touches a broker, a clock, or the network; it is pure data.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Any, Mapping, Optional, Tuple

# --------------------------------------------------------------------------- #
# Event-type vocabulary (string constants -> stable across processes/replays)
# --------------------------------------------------------------------------- #
MARKET_DATA = "MARKET_DATA"
BAR = "BAR"
QUOTE = "QUOTE"
OPTION_QUOTE = "OPTION_QUOTE"
NEWS = "NEWS"
CATALYST = "CATALYST"
SIGNAL = "SIGNAL"
TRADE_INTENT = "TRADE_INTENT"
ORDER_SUBMITTED = "ORDER_SUBMITTED"
ORDER_ACCEPTED = "ORDER_ACCEPTED"
ORDER_REJECTED = "ORDER_REJECTED"
ORDER_PARTIAL_FILL = "ORDER_PARTIAL_FILL"
ORDER_FILLED = "ORDER_FILLED"
ORDER_CANCELLED = "ORDER_CANCELLED"
ORDER_REPLACED = "ORDER_REPLACED"
POSITION_UPDATED = "POSITION_UPDATED"
RISK_EVENT = "RISK_EVENT"
EXIT_SIGNAL = "EXIT_SIGNAL"
SESSION_OPEN = "SESSION_OPEN"
SESSION_CLOSE = "SESSION_CLOSE"
TIMER = "TIMER"

EVENT_TYPES = frozenset({
    MARKET_DATA, BAR, QUOTE, OPTION_QUOTE, NEWS, CATALYST, SIGNAL, TRADE_INTENT,
    ORDER_SUBMITTED, ORDER_ACCEPTED, ORDER_REJECTED, ORDER_PARTIAL_FILL,
    ORDER_FILLED, ORDER_CANCELLED, ORDER_REPLACED, POSITION_UPDATED, RISK_EVENT,
    EXIT_SIGNAL, SESSION_OPEN, SESSION_CLOSE, TIMER,
})


# --------------------------------------------------------------------------- #
# Base event
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Event:
    """Common envelope. All fields default so frozen subclasses can add their
    own fields without dataclass field-ordering pain."""

    event_type: str = ""
    event_timestamp: Optional[datetime] = None
    received_timestamp: Optional[datetime] = None
    source: str = ""
    symbol: str = ""
    strategy_mode: str = ""
    correlation_id: str = ""
    event_id: str = ""
    data: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "event_type": self.event_type,
            "event_timestamp": _iso(self.event_timestamp),
            "received_timestamp": _iso(self.received_timestamp),
            "source": self.source,
            "symbol": self.symbol,
            "strategy_mode": self.strategy_mode,
            "correlation_id": self.correlation_id,
            "event_id": self.event_id,
            "data": dict(self.data or {}),
        }


def _iso(ts: Any) -> Optional[str]:
    if isinstance(ts, datetime):
        return ts.isoformat()
    return ts if ts is None else str(ts)


# --------------------------------------------------------------------------- #
# Market-data events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class BarEvent(Event):
    """A completed OHLCV bar. ``event_timestamp`` is the bar CLOSE (the moment
    the bar became known) — never the bar open — to keep the temporal contract."""

    event_type: str = BAR
    o: Optional[float] = None
    h: Optional[float] = None
    l: Optional[float] = None
    c: Optional[float] = None
    v: Optional[float] = None
    timeframe: str = "1Day"


@dataclass(frozen=True)
class QuoteEvent(Event):
    """A top-of-book equity/underlying quote."""

    event_type: str = QUOTE
    bid: Optional[float] = None
    ask: Optional[float] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return round((self.bid + self.ask) / 2.0, 6)


@dataclass(frozen=True)
class OptionQuoteEvent(Event):
    """A top-of-book option-contract quote. ``symbol`` is the OCC symbol."""

    event_type: str = OPTION_QUOTE
    bid: Optional[float] = None
    ask: Optional[float] = None
    underlying: str = ""

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return round((self.bid + self.ask) / 2.0, 6)


@dataclass(frozen=True)
class NewsEvent(Event):
    event_type: str = NEWS
    headline: str = ""
    score: Optional[float] = None


@dataclass(frozen=True)
class CatalystEvent(Event):
    event_type: str = CATALYST
    catalyst_type: str = ""
    score: Optional[float] = None


@dataclass(frozen=True)
class TimerEvent(Event):
    event_type: str = TIMER


@dataclass(frozen=True)
class SessionEvent(Event):
    """SESSION_OPEN / SESSION_CLOSE — set ``event_type`` explicitly."""


# --------------------------------------------------------------------------- #
# Decision / execution events
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class SignalEvent(Event):
    """A directional read produced by the strategy (pre-intent)."""

    event_type: str = SIGNAL
    direction: str = ""            # "CALL" | "PUT" | "NO_TRADE"
    p_call: Optional[float] = None
    p_put: Optional[float] = None
    p_no_trade: Optional[float] = None


@dataclass(frozen=True)
class TradeIntentEvent(Event):
    """The strategy's request to trade. The strategy emits this and NEVER touches
    the broker; a broker adapter consumes it. Carries the quote + market context
    the fill model needs so the SimBroker stays self-contained + deterministic."""

    event_type: str = TRADE_INTENT
    contract: str = ""             # OCC symbol to trade
    direction: str = ""            # "CALL" | "PUT"
    side: str = "buy"              # "buy" | "sell"
    qty: int = 1
    order_type: str = "market"
    limit_price: Optional[float] = None
    max_entry_price: Optional[float] = None
    bid: Optional[float] = None
    ask: Optional[float] = None
    market: Mapping[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderEvent(Event):
    """ORDER_SUBMITTED / ACCEPTED / REJECTED / PARTIAL_FILL / FILLED / CANCELLED
    / REPLACED — set ``event_type`` explicitly."""

    order_id: Optional[str] = None
    side: str = ""
    qty: float = 0.0
    filled_qty: float = 0.0
    fill_price: Optional[float] = None
    status: str = ""
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class PositionEvent(Event):
    event_type: str = POSITION_UPDATED
    qty: float = 0.0
    avg_price: Optional[float] = None
    realized_pnl: Optional[float] = None


@dataclass(frozen=True)
class RiskEvent(Event):
    event_type: str = RISK_EVENT
    reason: str = ""


@dataclass(frozen=True)
class ExitEvent(Event):
    event_type: str = EXIT_SIGNAL
    reason: str = ""


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic, no network / creds / file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)

    bar = BarEvent(symbol="AAPL", event_timestamp=t0, o=1, h=2, l=0.5, c=1.5,
                   v=1000, source="feed")
    if bar.event_type != BAR or bar.c != 1.5:
        print("FAIL: bar event fields", bar); ok = False

    q = QuoteEvent(symbol="AAPL", event_timestamp=t0, bid=1.00, ask=1.02)
    if q.event_type != QUOTE or q.mid != 1.01:
        print("FAIL: quote mid", q); ok = False

    oq = OptionQuoteEvent(symbol="AAPL260911C00200000", bid=1.0, ask=1.2,
                          underlying="AAPL", event_timestamp=t0)
    if oq.event_type != OPTION_QUOTE or oq.mid != 1.1:
        print("FAIL: option quote mid", oq); ok = False

    intent = TradeIntentEvent(symbol="AAPL", contract="AAPL260911C00200000",
                              direction="CALL", side="buy", qty=2,
                              bid=1.0, ask=1.2, event_timestamp=t0,
                              correlation_id="c1")
    if intent.event_type != TRADE_INTENT or intent.qty != 2:
        print("FAIL: intent fields", intent); ok = False

    sess = SessionEvent(event_type=SESSION_OPEN, event_timestamp=t0)
    if sess.event_type != SESSION_OPEN:
        print("FAIL: session event", sess); ok = False

    # Frozen (immutable) — mutation must raise.
    try:
        bar.c = 9.0  # type: ignore[misc]
        print("FAIL: events must be frozen"); ok = False
    except Exception:
        pass

    # Every promoted event type is in the vocabulary set.
    for ev in (bar, q, oq, intent, sess):
        if ev.event_type not in EVENT_TYPES:
            print("FAIL: unknown event_type", ev.event_type); ok = False

    # to_dict is JSON-friendly (ISO timestamps, plain dict).
    d = bar.to_dict()
    if d["event_timestamp"] != t0.isoformat() or d["event_type"] != BAR:
        print("FAIL: to_dict", d); ok = False

    print("engine.events self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
