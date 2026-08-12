"""
Oracle Engine — Upgrade A: deterministic simulation broker.

``SimBroker`` consumes ``TRADE_INTENT`` events and turns them into realistic,
DETERMINISTIC fills using the existing ``oracle.execution.fill_simulator``
(latency + partial fills, priced by the conservative ``fill_model``). It emits
the order lifecycle back onto the bus (``ORDER_SUBMITTED`` -> ``ORDER_FILLED`` /
``ORDER_PARTIAL_FILL`` / ``ORDER_CANCELLED``) plus ``POSITION_UPDATED`` with
running realized P&L, and maintains an in-memory position book + cash.

It reuses (does not re-implement) the friction model, and it never touches a
real broker or the network. Two runs with the same intents + seed produce a
byte-identical order log and P&L — the Upgrade-A acceptance invariant.

Position/PnL accounting:
  * BUY opens/adds to a long (weighted-average cost).
  * SELL reduces/closes the long; realized P&L on the closed quantity is
    ``(exit_price - avg_cost) * closed_qty * CONTRACT_MULTIPLIER`` for options.
  * Round-trip fees/slippage are already inside each fill price via the
    conservative fill model, so realized P&L is executable, not theoretical.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional

from cost_model import CONTRACT_MULTIPLIER
from oracle.engine.events import (
    ORDER_CANCELLED,
    ORDER_FILLED,
    ORDER_PARTIAL_FILL,
    ORDER_SUBMITTED,
    Event,
    OrderEvent,
    PositionEvent,
)
from oracle.execution.client import STATUS_FILLED, STATUS_PARTIAL, OrderRequest, Quote
from oracle.execution.fill_simulator import FillSimulator


@dataclass
class _Book:
    qty: float = 0.0
    avg_price: Optional[float] = None


@dataclass(frozen=True)
class BrokerFill:
    """A record of one executed intent (for the driver's outcome ledger)."""

    contract: str
    side: str
    order_qty: int
    filled_qty: float
    avg_fill_price: Optional[float]
    status: str
    realized_pnl: float
    correlation_id: str
    reasons: tuple = ()


class SimBroker:
    """Deterministic in-memory options broker driven by TRADE_INTENT events."""

    def __init__(
        self,
        emit: Callable[[Event], None],
        *,
        fill_simulator: Optional[FillSimulator] = None,
        seed: int = 0,
        asset_multiplier: float = CONTRACT_MULTIPLIER,
    ) -> None:
        self.emit = emit
        self.sim = fill_simulator or FillSimulator()
        self.seed = int(seed)
        self.multiplier = float(asset_multiplier)
        self.books: Dict[str, _Book] = {}
        self.realized_pnl: float = 0.0
        self.fills: List[BrokerFill] = []
        self.order_log: List[dict] = []
        self._seq = 0

    # -- bus entry point -------------------------------------------------- #
    def handle(self, event: Event) -> None:
        if event.event_type == "TRADE_INTENT":
            self.on_trade_intent(event)

    def on_trade_intent(self, intent: Event) -> None:
        contract = getattr(intent, "contract", "") or intent.symbol
        side = getattr(intent, "side", "buy")
        qty = max(1, int(getattr(intent, "qty", 1) or 1))
        order_type = getattr(intent, "order_type", "market")
        limit_price = getattr(intent, "limit_price", None)
        bid = getattr(intent, "bid", None)
        ask = getattr(intent, "ask", None)
        market = getattr(intent, "market", {}) or {}
        corr = intent.correlation_id or contract

        self._seq += 1
        order_id = f"sim-{self._seq:06d}"
        quote = Quote(contract, bid=bid, ask=ask, ts=str(intent.event_timestamp))
        order = OrderRequest(
            contract, side, qty, order_type=order_type, limit_price=limit_price,
            client_order_id=corr)

        self.emit(OrderEvent(
            event_type=ORDER_SUBMITTED, symbol=contract,
            event_timestamp=intent.event_timestamp,
            received_timestamp=intent.received_timestamp, source="broker",
            correlation_id=corr, order_id=order_id, side=side, qty=float(qty),
            status="submitted"))

        sim = self.sim.simulate(order, quote, market=dict(market), seed=self.seed)

        realized = 0.0
        if sim.is_filled and sim.avg_fill_price is not None:
            realized = self._apply_fill(contract, side, sim.filled_qty,
                                        sim.avg_fill_price)
            et = ORDER_FILLED if sim.status == STATUS_FILLED else ORDER_PARTIAL_FILL
            self.emit(OrderEvent(
                event_type=et, symbol=contract,
                event_timestamp=intent.event_timestamp,
                received_timestamp=intent.received_timestamp, source="broker",
                correlation_id=corr, order_id=order_id, side=side,
                qty=float(qty), filled_qty=float(sim.filled_qty),
                fill_price=sim.avg_fill_price, status=sim.status,
                reasons=sim.reasons))
            book = self.books.get(contract, _Book())
            self.emit(PositionEvent(
                symbol=contract, event_timestamp=intent.event_timestamp,
                source="broker", correlation_id=corr, qty=book.qty,
                avg_price=book.avg_price, realized_pnl=round(self.realized_pnl, 6)))
        else:
            self.emit(OrderEvent(
                event_type=ORDER_CANCELLED, symbol=contract,
                event_timestamp=intent.event_timestamp,
                received_timestamp=intent.received_timestamp, source="broker",
                correlation_id=corr, order_id=order_id, side=side,
                qty=float(qty), status=sim.status, reasons=sim.reasons))

        record = {
            "order_id": order_id, "contract": contract, "side": side,
            "order_qty": qty, "filled_qty": round(sim.filled_qty, 6),
            "avg_fill_price": sim.avg_fill_price, "status": sim.status,
            "realized_pnl": round(realized, 6), "correlation_id": corr,
            "reasons": list(sim.reasons)}
        self.order_log.append(record)
        self.fills.append(BrokerFill(
            contract=contract, side=side, order_qty=qty,
            filled_qty=round(sim.filled_qty, 6),
            avg_fill_price=sim.avg_fill_price, status=sim.status,
            realized_pnl=round(realized, 6), correlation_id=corr,
            reasons=sim.reasons))

    # -- position book ---------------------------------------------------- #
    def _apply_fill(self, contract: str, side: str, qty: float,
                    price: float) -> float:
        """Update the book; return realized P&L generated by THIS fill ($)."""
        book = self.books.get(contract) or _Book()
        signed = qty if side == "buy" else -qty
        realized = 0.0

        if book.qty == 0 or (book.qty > 0) == (signed > 0):
            # Opening or adding to the same side -> weighted-average cost.
            new_qty = book.qty + signed
            if new_qty == 0:
                book.avg_price = None
            else:
                prev_cost = (book.avg_price or price) * abs(book.qty)
                book.avg_price = round(
                    (prev_cost + price * abs(signed)) / abs(new_qty), 6)
            book.qty = round(new_qty, 6)
        else:
            # Reducing/closing the existing side -> realize P&L on closed qty.
            closing = min(abs(signed), abs(book.qty))
            entry = book.avg_price if book.avg_price is not None else price
            if book.qty > 0:      # long closed by a sell
                realized = (price - entry) * closing * self.multiplier
            else:                 # short closed by a buy
                realized = (entry - price) * closing * self.multiplier
            new_qty = round(book.qty + signed, 6)
            book.qty = new_qty
            if new_qty == 0:
                book.avg_price = None
            elif (new_qty > 0) != (book.qty - signed > 0):
                # flipped through zero -> remainder opens the opposite side
                book.avg_price = price

        self.books[contract] = book
        self.realized_pnl = round(self.realized_pnl + realized, 6)
        return realized

    # -- summary ---------------------------------------------------------- #
    def summary(self) -> dict:
        return {
            "realized_pnl": round(self.realized_pnl, 6),
            "orders": len(self.order_log),
            "open_positions": {c: b.qty for c, b in self.books.items() if b.qty},
        }


# --------------------------------------------------------------------------- #
# Self-test (offline, deterministic — no network / creds / file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime

    from oracle.engine.events import TradeIntentEvent

    ok = True
    t0 = datetime(2026, 8, 11, 13, 30, 0)

    def run_once(seed: int):
        emitted: List[Event] = []
        broker = SimBroker(emitted.append, seed=seed)
        # OPEN long: market buy, deep book -> full fill at/above mid.
        broker.handle(TradeIntentEvent(
            symbol="AAPL", contract="AAPL-CALL", direction="CALL", side="buy",
            qty=2, order_type="market", bid=1.00, ask=1.02,
            market={"ask_size": 100}, event_timestamp=t0, correlation_id="c1"))
        # CLOSE long into a higher book -> positive realized P&L.
        broker.handle(TradeIntentEvent(
            symbol="AAPL", contract="AAPL-CALL", direction="CALL", side="sell",
            qty=2, order_type="market", bid=1.50, ask=1.52,
            market={"bid_size": 100}, event_timestamp=t0, correlation_id="c2"))
        return broker, emitted

    broker, emitted = run_once(seed=1)

    # Two fills recorded, position flat at the end.
    if len(broker.order_log) != 2:
        print("FAIL: expected 2 orders", broker.order_log); ok = False
    if broker.summary()["open_positions"]:
        print("FAIL: position should be flat", broker.summary()); ok = False

    # Buying at ~1.02 and selling at ~1.50 must realize a positive P&L.
    if broker.realized_pnl <= 0:
        print("FAIL: round trip should be profitable", broker.realized_pnl)
        ok = False

    # Lifecycle events emitted: SUBMITTED + FILLED + POSITION_UPDATED per leg.
    types = [e.event_type for e in emitted]
    if types.count(ORDER_SUBMITTED) != 2 or types.count(ORDER_FILLED) < 1:
        print("FAIL: order lifecycle events", types); ok = False
    if types.count("POSITION_UPDATED") < 2:
        print("FAIL: position updates", types); ok = False

    # Determinism: same seed -> identical order log + realized P&L.
    broker2, _ = run_once(seed=1)
    if broker2.order_log != broker.order_log:
        print("FAIL: non-deterministic order log"); ok = False
    if broker2.realized_pnl != broker.realized_pnl:
        print("FAIL: non-deterministic pnl", broker.realized_pnl,
              broker2.realized_pnl); ok = False

    # No-quote intent -> cancelled, no fill, no position, never raises.
    emitted3: List[Event] = []
    broker3 = SimBroker(emitted3.append, seed=0)
    broker3.handle(TradeIntentEvent(
        symbol="NVDA", contract="NVDA-CALL", direction="CALL", side="buy",
        qty=1, order_type="market", event_timestamp=t0, correlation_id="c3"))
    if broker3.summary()["open_positions"]:
        print("FAIL: no-quote should not open a position"); ok = False
    if ORDER_CANCELLED not in [e.event_type for e in emitted3]:
        print("FAIL: no-quote should cancel", emitted3); ok = False

    # Losing round trip realizes a negative P&L.
    emitted4: List[Event] = []
    broker4 = SimBroker(emitted4.append, seed=1)
    broker4.handle(TradeIntentEvent(
        symbol="X", contract="X-CALL", side="buy", qty=1, order_type="market",
        bid=2.00, ask=2.02, market={"ask_size": 100}, event_timestamp=t0,
        correlation_id="l1"))
    broker4.handle(TradeIntentEvent(
        symbol="X", contract="X-CALL", side="sell", qty=1, order_type="market",
        bid=1.00, ask=1.02, market={"bid_size": 100}, event_timestamp=t0,
        correlation_id="l2"))
    if broker4.realized_pnl >= 0:
        print("FAIL: losing round trip should be negative", broker4.realized_pnl)
        ok = False

    print("engine.sim_broker self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
