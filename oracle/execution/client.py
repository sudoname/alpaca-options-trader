"""
Oracle Execution — ExecutionClient interface + value objects + Sim/Shadow.

This module defines the narrow broker interface every decision path shares and
two fully-offline adapters:

  * ``SimExecutionClient`` — deterministic in-memory broker for backtest/replay.
    Immediate marketable fills by default (BUY at ask, SELL at bid); an optional
    ``fill_model`` callable lets Upgrade 2 inject slippage / partial / no-fill.
  * ``ShadowExecutionClient`` — delegates READS to a wrapped real client but
    NEVER writes; ``submit/cancel/replace`` are logged and return a synthetic
    ``status='shadow'`` result. This is how a new decision path runs beside live
    without placing a single order.

The real broker adapter lives in ``oracle.execution.alpaca`` (transport-injected
so it is testable offline).
"""

from __future__ import annotations

import abc
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional, Tuple

# Order status vocabulary (superset of Alpaca's + our shadow marker).
STATUS_ACCEPTED = "accepted"
STATUS_FILLED = "filled"
STATUS_PARTIAL = "partially_filled"
STATUS_REJECTED = "rejected"
STATUS_CANCELED = "canceled"
STATUS_PENDING = "pending"
STATUS_SHADOW = "shadow"


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Value objects (frozen -> reproducible / comparable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Quote:
    symbol: str
    bid: Optional[float] = None
    ask: Optional[float] = None
    ts: Optional[str] = None

    @property
    def mid(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return round((self.bid + self.ask) / 2.0, 6)

    @property
    def spread(self) -> Optional[float]:
        if self.bid is None or self.ask is None:
            return None
        return round(self.ask - self.bid, 6)


@dataclass(frozen=True)
class OrderRequest:
    symbol: str
    side: str                       # 'buy' | 'sell'
    qty: int
    order_type: str = "market"      # 'market' | 'limit'
    limit_price: Optional[float] = None
    tif: str = "day"
    asset_class: str = "us_option"
    client_order_id: Optional[str] = None
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class OrderResult:
    order_id: Optional[str]
    status: str
    symbol: str
    side: str
    filled_qty: float = 0.0
    filled_avg_price: Optional[float] = None
    client_order_id: Optional[str] = None
    submitted_at: Optional[str] = None
    raw: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.status in (STATUS_FILLED, STATUS_PARTIAL) and self.filled_qty > 0

    @property
    def is_terminal_no_fill(self) -> bool:
        return (self.status in (STATUS_REJECTED, STATUS_CANCELED)
                and self.filled_qty <= 0)


@dataclass(frozen=True)
class Fill:
    order_id: Optional[str]
    symbol: str
    qty: float
    price: float
    ts: Optional[str] = None


@dataclass(frozen=True)
class Position:
    symbol: str
    qty: float
    avg_price: Optional[float] = None
    market_value: Optional[float] = None


# --------------------------------------------------------------------------- #
# Interface
# --------------------------------------------------------------------------- #
class ExecutionClient(abc.ABC):
    """The narrow broker contract. Decision code depends ONLY on this surface,
    so swapping live<->paper<->sim<->shadow never changes the decision."""

    name: str = "abstract"

    # -- reads (fail-open: None/empty on error) --------------------------- #
    @abc.abstractmethod
    def get_quote(self, symbol: str) -> Optional[Quote]: ...

    @abc.abstractmethod
    def get_positions(self) -> List[Position]: ...

    @abc.abstractmethod
    def get_buying_power(self) -> Optional[float]: ...

    @abc.abstractmethod
    def get_order(self, order_id: str) -> Optional[OrderResult]: ...

    @abc.abstractmethod
    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]: ...

    # -- writes (surface a rejected result rather than raising) ----------- #
    @abc.abstractmethod
    def submit_order(self, order: OrderRequest) -> OrderResult: ...

    @abc.abstractmethod
    def cancel_order(self, order_id: str) -> OrderResult: ...

    @abc.abstractmethod
    def replace_order(self, order_id: str, **changes: Any) -> OrderResult: ...


# --------------------------------------------------------------------------- #
# Deterministic simulation broker
# --------------------------------------------------------------------------- #
# A fill_model maps (OrderRequest, Quote) -> (fill_price, filled_qty, status).
FillModelFn = Callable[[OrderRequest, Optional[Quote]],
                       Tuple[Optional[float], float, str]]


def _default_fill(order: OrderRequest,
                  quote: Optional[Quote]) -> Tuple[Optional[float], float, str]:
    """Conservative marketable fill: BUY at ask, SELL at bid; limit fills only
    when marketable. No quote -> pending. This is the Upgrade-1 baseline that
    Upgrade 2's FillModel replaces with spread/slippage/partial behaviour."""
    if quote is None:
        return None, 0.0, STATUS_PENDING
    buy = order.side == "buy"
    ref = quote.ask if buy else quote.bid
    if ref is None:
        ref = quote.mid
    if ref is None:
        return None, 0.0, STATUS_PENDING
    if order.order_type == "limit" and order.limit_price is not None:
        marketable = (order.limit_price >= ref) if buy else (order.limit_price <= ref)
        if not marketable:
            return None, 0.0, STATUS_PENDING
        ref = order.limit_price
    return round(float(ref), 6), float(order.qty), STATUS_FILLED


class SimExecutionClient(ExecutionClient):
    """In-memory, deterministic broker. Same inputs + same fill_model -> byte
    identical fills, so a replay is reproducible and comparable to live."""

    name = "sim"

    def __init__(self, *, quotes: Optional[Dict[str, Quote]] = None,
                 buying_power: Optional[float] = 100000.0,
                 fill_model: Optional[FillModelFn] = None,
                 clock: Optional[Callable[[], str]] = None) -> None:
        self._quotes: Dict[str, Quote] = dict(quotes or {})
        self._buying_power = buying_power
        self._fill_model = fill_model or _default_fill
        self._clock = clock
        self._orders: Dict[str, OrderResult] = {}
        self._fills: List[Fill] = []
        self._positions: Dict[str, Position] = {}
        self._seq = 0

    # -- test/replay helpers --------------------------------------------- #
    def set_quote(self, quote: Quote) -> None:
        self._quotes[quote.symbol] = quote

    def _now(self) -> Optional[str]:
        try:
            return self._clock() if self._clock else None
        except Exception:
            return None

    def _next_id(self) -> str:
        self._seq += 1
        return f"sim-{self._seq:06d}"

    # -- reads ------------------------------------------------------------ #
    def get_quote(self, symbol: str) -> Optional[Quote]:
        return self._quotes.get(symbol)

    def get_positions(self) -> List[Position]:
        return [p for p in self._positions.values() if p.qty]

    def get_buying_power(self) -> Optional[float]:
        return self._buying_power

    def get_order(self, order_id: str) -> Optional[OrderResult]:
        return self._orders.get(order_id)

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        if order_id is None:
            return list(self._fills)
        return [f for f in self._fills if f.order_id == order_id]

    # -- writes ----------------------------------------------------------- #
    def submit_order(self, order: OrderRequest) -> OrderResult:
        oid = self._next_id()
        quote = self._quotes.get(order.symbol)
        try:
            price, filled_qty, status = self._fill_model(order, quote)
        except Exception:
            price, filled_qty, status = None, 0.0, STATUS_REJECTED
        result = OrderResult(
            order_id=oid, status=status, symbol=order.symbol, side=order.side,
            filled_qty=float(filled_qty or 0.0), filled_avg_price=price,
            client_order_id=order.client_order_id, submitted_at=self._now(),
            raw={"order_type": order.order_type,
                 "limit_price": order.limit_price})
        self._orders[oid] = result
        if result.is_filled and price is not None:
            self._fills.append(Fill(order_id=oid, symbol=order.symbol,
                                    qty=result.filled_qty, price=price,
                                    ts=result.submitted_at))
            self._apply_fill(order, result.filled_qty, price)
        return result

    def _apply_fill(self, order: OrderRequest, qty: float, price: float) -> None:
        signed = qty if order.side == "buy" else -qty
        cur = self._positions.get(order.symbol)
        if cur is None:
            new_qty = signed
            avg = price if new_qty else None
        else:
            new_qty = cur.qty + signed
            if (cur.qty >= 0) == (signed >= 0) and new_qty:
                # adding to the same side -> weighted average
                prev_cost = (cur.avg_price or price) * abs(cur.qty)
                avg = round((prev_cost + price * abs(signed)) / abs(new_qty), 6)
            else:
                avg = cur.avg_price if new_qty else None
        self._positions[order.symbol] = Position(
            symbol=order.symbol, qty=round(new_qty, 6), avg_price=avg,
            market_value=(round(new_qty * price * 100.0, 6)
                          if order.asset_class == "us_option"
                          else round(new_qty * price, 6)))
        if self._buying_power is not None:
            self._buying_power = round(
                self._buying_power - signed * price *
                (100.0 if order.asset_class == "us_option" else 1.0), 6)

    def cancel_order(self, order_id: str) -> OrderResult:
        cur = self._orders.get(order_id)
        if cur is None:
            return OrderResult(order_id=order_id, status=STATUS_REJECTED,
                               symbol="", side="", raw={"error": "unknown_order"})
        if cur.is_filled:
            return cur  # already filled -> cancel is a no-op
        canceled = replace(cur, status=STATUS_CANCELED)
        self._orders[order_id] = canceled
        return canceled

    def replace_order(self, order_id: str, **changes: Any) -> OrderResult:
        cur = self._orders.get(order_id)
        if cur is None or cur.is_filled:
            return cur or OrderResult(order_id=order_id, status=STATUS_REJECTED,
                                      symbol="", side="",
                                      raw={"error": "unknown_order"})
        # Re-submit as a fresh order carrying the requested changes.
        new_req = OrderRequest(
            symbol=cur.symbol, side=cur.side,
            qty=int(changes.get("qty", cur.filled_qty or 1)),
            order_type=changes.get("order_type", cur.raw.get("order_type", "market")),
            limit_price=changes.get("limit_price", cur.raw.get("limit_price")),
            client_order_id=cur.client_order_id)
        self.cancel_order(order_id)
        return self.submit_order(new_req)


# --------------------------------------------------------------------------- #
# Shadow broker (reads real, writes nothing)
# --------------------------------------------------------------------------- #
class ShadowExecutionClient(ExecutionClient):
    """Wraps a real ExecutionClient for READS but never places an order. Every
    write is recorded in ``self.submitted`` and returns a ``status='shadow'``
    result, so a candidate decision path can run beside live and its would-do
    orders can be compared to reality without any market effect."""

    name = "shadow"

    def __init__(self, inner: Optional[ExecutionClient] = None,
                 *, clock: Optional[Callable[[], str]] = None) -> None:
        self._inner = inner
        self._clock = clock
        self.submitted: List[OrderRequest] = []
        self._seq = 0

    def _now(self) -> Optional[str]:
        try:
            return self._clock() if self._clock else None
        except Exception:
            return None

    # -- reads delegate to inner (fail-open) ------------------------------ #
    def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            return self._inner.get_quote(symbol) if self._inner else None
        except Exception:
            return None

    def get_positions(self) -> List[Position]:
        try:
            return self._inner.get_positions() if self._inner else []
        except Exception:
            return []

    def get_buying_power(self) -> Optional[float]:
        try:
            return self._inner.get_buying_power() if self._inner else None
        except Exception:
            return None

    def get_order(self, order_id: str) -> Optional[OrderResult]:
        try:
            return self._inner.get_order(order_id) if self._inner else None
        except Exception:
            return None

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        return []  # a shadow path has no fills of its own

    # -- writes are inert (never touch inner) ----------------------------- #
    def _shadow_result(self, order: OrderRequest) -> OrderResult:
        self._seq += 1
        return OrderResult(order_id=f"shadow-{self._seq:06d}",
                           status=STATUS_SHADOW, symbol=order.symbol,
                           side=order.side, client_order_id=order.client_order_id,
                           submitted_at=self._now(),
                           raw={"would_submit": True,
                                "order_type": order.order_type,
                                "qty": order.qty,
                                "limit_price": order.limit_price})

    def submit_order(self, order: OrderRequest) -> OrderResult:
        self.submitted.append(order)
        return self._shadow_result(order)

    def cancel_order(self, order_id: str) -> OrderResult:
        return OrderResult(order_id=order_id, status=STATUS_SHADOW, symbol="",
                           side="", raw={"would_cancel": True})

    def replace_order(self, order_id: str, **changes: Any) -> OrderResult:
        return OrderResult(order_id=order_id, status=STATUS_SHADOW, symbol="",
                           side="", raw={"would_replace": changes})


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    sim = SimExecutionClient(
        quotes={"OPT1": Quote("OPT1", bid=1.00, ask=1.10, ts="t")},
        buying_power=10000.0)

    # Marketable BUY fills at the ask; position + BP update deterministically.
    r = sim.submit_order(OrderRequest("OPT1", "buy", 2, order_type="market"))
    if not r.is_filled or r.filled_avg_price != 1.10 or r.filled_qty != 2:
        print("FAIL: market buy fill", r); ok = False
    pos = {p.symbol: p for p in sim.get_positions()}
    if pos.get("OPT1") is None or pos["OPT1"].qty != 2:
        print("FAIL: position after buy", pos); ok = False
    # BP debited by 2 * 1.10 * 100 = 220.
    if sim.get_buying_power() != 10000.0 - 220.0:
        print("FAIL: buying power", sim.get_buying_power()); ok = False
    if len(sim.get_fills()) != 1 or len(sim.get_fills(r.order_id)) != 1:
        print("FAIL: fills ledger", sim.get_fills()); ok = False

    # Non-marketable LIMIT buy stays pending (no fill, no position change).
    r2 = sim.submit_order(OrderRequest("OPT1", "buy", 1, order_type="limit",
                                       limit_price=1.00))
    if r2.status != STATUS_PENDING or r2.is_filled:
        print("FAIL: non-marketable limit should pend", r2); ok = False
    # Cancel the pending order.
    c = sim.cancel_order(r2.order_id)
    if c.status != STATUS_CANCELED:
        print("FAIL: cancel pending", c); ok = False

    # Missing quote -> pending (fail-open), never raises.
    r3 = sim.submit_order(OrderRequest("NOPE", "buy", 1))
    if r3.status != STATUS_PENDING:
        print("FAIL: missing quote should pend", r3); ok = False

    # Determinism: a fresh sim with identical inputs yields the same first id +
    # fill price.
    sim_b = SimExecutionClient(
        quotes={"OPT1": Quote("OPT1", bid=1.00, ask=1.10, ts="t")},
        buying_power=10000.0)
    rb = sim_b.submit_order(OrderRequest("OPT1", "buy", 2, order_type="market"))
    if (rb.order_id, rb.filled_avg_price, rb.filled_qty) != \
            ("sim-000001", 1.10, 2):
        print("FAIL: non-deterministic sim", rb); ok = False

    # Injected fill_model overrides the default (e.g. slippage).
    def slip(order, quote):
        base = quote.ask if order.side == "buy" else quote.bid
        return round(base + 0.05, 6), float(order.qty), STATUS_FILLED
    sim_s = SimExecutionClient(quotes={"OPT1": Quote("OPT1", 1.0, 1.1)},
                               fill_model=slip)
    rs = sim_s.submit_order(OrderRequest("OPT1", "buy", 1))
    if rs.filled_avg_price != 1.15:
        print("FAIL: injected fill_model", rs); ok = False

    # Shadow client: reads delegate, writes are inert and recorded.
    shadow = ShadowExecutionClient(inner=sim)
    if shadow.get_quote("OPT1") is None or shadow.get_buying_power() is None:
        print("FAIL: shadow reads should delegate"); ok = False
    before_orders = len(sim._orders)
    sr = shadow.submit_order(OrderRequest("OPT1", "buy", 5))
    if sr.status != STATUS_SHADOW or sr.raw.get("would_submit") is not True:
        print("FAIL: shadow submit not inert", sr); ok = False
    if len(sim._orders) != before_orders:
        print("FAIL: shadow leaked an order into inner"); ok = False
    if len(shadow.submitted) != 1 or shadow.submitted[0].qty != 5:
        print("FAIL: shadow did not record intent", shadow.submitted); ok = False
    if shadow.get_fills() != []:
        print("FAIL: shadow should have no fills"); ok = False

    # Quote helpers.
    q = Quote("X", 1.0, 1.2)
    if q.mid != 1.1 or q.spread != 0.2:
        print("FAIL: quote mid/spread", q.mid, q.spread); ok = False

    print("execution.client self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
