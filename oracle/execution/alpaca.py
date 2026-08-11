"""
Oracle Execution — Alpaca adapter (the real broker behind the interface).

``AlpacaExecutionClient`` is the ONLY place a decision path touches the live /
paper broker. It implements ``client.ExecutionClient`` by translating our frozen
value objects (Quote / OrderRequest / OrderResult / Position / Fill) to and from
the ``alpaca-py`` SDK — the same SDK ``alpaca_client.py`` and ``smart_trader``
already use, so nothing about the broker behaviour changes.

Swapping live<->paper is a single flag:

    live   = AlpacaExecutionClient(paper=False)
    paper  = AlpacaExecutionClient(paper=True)     # identical code path

Testability / offline discipline (mirrors the rest of Oracle):
  * The SDK clients are INJECTABLE. ``_self_test`` hands in fakes, so it runs
    with no network, no creds and places no order.
  * The real SDK is imported LAZILY inside the default client factories, so
    importing this module never needs credentials and never hits the network.
  * Fail-open reads: any error -> None / [] (never raises in the read path).
  * Writes surface a ``status='rejected'`` OrderResult rather than raising, so a
    broker error can never crash the decision loop.

This adapter deliberately duplicates NO strategy logic. Sizing, direction, EV
and every hard risk control stay upstream in the kernel / risk engine; this file
only submits, cancels, replaces and reads.
"""

from __future__ import annotations

import os
from typing import Any, Callable, List, Optional

from oracle.execution.client import (
    STATUS_ACCEPTED,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    STATUS_REJECTED,
    ExecutionClient,
    Fill,
    OrderRequest,
    OrderResult,
    Position,
    Quote,
)


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _to_str(value: Any) -> Optional[str]:
    if value is None:
        return None
    # Alpaca enums stringify as e.g. "OrderStatus.FILLED"; prefer their .value.
    val = getattr(value, "value", value)
    try:
        return str(val)
    except Exception:  # pragma: no cover
        return None


# --------------------------------------------------------------------------- #
# SDK status / side normalization -> our vocabulary
# --------------------------------------------------------------------------- #
_STATUS_MAP = {
    "filled": STATUS_FILLED,
    "partially_filled": STATUS_PARTIAL,
    "canceled": STATUS_CANCELED,
    "cancelled": STATUS_CANCELED,
    "expired": STATUS_CANCELED,
    "done_for_day": STATUS_CANCELED,
    "rejected": STATUS_REJECTED,
    "suspended": STATUS_REJECTED,
    "stopped": STATUS_REJECTED,
    "accepted": STATUS_ACCEPTED,
    "new": STATUS_ACCEPTED,
    "pending_new": STATUS_ACCEPTED,
    "accepted_for_bidding": STATUS_ACCEPTED,
    "calculated": STATUS_ACCEPTED,
    "held": STATUS_ACCEPTED,
    "replaced": STATUS_ACCEPTED,
    "pending_replace": STATUS_PENDING,
    "pending_cancel": STATUS_PENDING,
}


def _norm_status(native: Any) -> str:
    s = (_to_str(native) or "").strip().lower()
    # ``OrderStatus.FILLED`` -> "orderstatus.filled" -> take the tail.
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return _STATUS_MAP.get(s, STATUS_PENDING)


def _norm_side(native: Any) -> str:
    s = (_to_str(native) or "").strip().lower()
    if "." in s:
        s = s.rsplit(".", 1)[-1]
    return "sell" if s == "sell" else "buy"


# --------------------------------------------------------------------------- #
# native SDK object -> our value objects (all getattr-duck-typed so fakes work)
# --------------------------------------------------------------------------- #
def _map_quote(native: Any, symbol: str) -> Optional[Quote]:
    if native is None:
        return None
    bid = _to_float(getattr(native, "bid_price", None))
    ask = _to_float(getattr(native, "ask_price", None))
    ts = _to_str(getattr(native, "timestamp", None))
    if bid is None and ask is None:
        return None
    return Quote(symbol=symbol, bid=bid, ask=ask, ts=ts)


def _map_position(native: Any) -> Optional[Position]:
    sym = _to_str(getattr(native, "symbol", None))
    if not sym:
        return None
    return Position(
        symbol=sym,
        qty=_to_float(getattr(native, "qty", None)) or 0.0,
        avg_price=_to_float(getattr(native, "avg_entry_price", None)),
        market_value=_to_float(getattr(native, "market_value", None)),
    )


def _map_order(native: Any) -> Optional[OrderResult]:
    if native is None:
        return None
    return OrderResult(
        order_id=_to_str(getattr(native, "id", None)),
        status=_norm_status(getattr(native, "status", None)),
        symbol=_to_str(getattr(native, "symbol", None)) or "",
        side=_norm_side(getattr(native, "side", None)),
        filled_qty=_to_float(getattr(native, "filled_qty", None)) or 0.0,
        filled_avg_price=_to_float(getattr(native, "filled_avg_price", None)),
        client_order_id=_to_str(getattr(native, "client_order_id", None)),
        submitted_at=_to_str(getattr(native, "submitted_at", None)),
        raw={"native_status": _to_str(getattr(native, "status", None))},
    )


# --------------------------------------------------------------------------- #
# default SDK-backed request builder (imported lazily -> offline-safe module)
# --------------------------------------------------------------------------- #
def _build_sdk_order(order: OrderRequest) -> Any:
    """Translate our OrderRequest to an alpaca-py request object. Imported here
    (not at module top) so the module imports with no SDK / creds present."""
    from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
    from alpaca.trading.requests import LimitOrderRequest, MarketOrderRequest

    side = OrderSide.BUY if order.side == "buy" else OrderSide.SELL
    tif = TimeInForce.GTC if str(order.tif).lower() == "gtc" else TimeInForce.DAY
    common = dict(symbol=order.symbol, qty=order.qty, side=side, time_in_force=tif)
    if order.asset_class == "us_option":
        common["asset_class"] = AssetClass.US_OPTION
    if order.order_type == "limit":
        return LimitOrderRequest(limit_price=order.limit_price, **common)
    return MarketOrderRequest(**common)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
class AlpacaExecutionClient(ExecutionClient):
    """Live/paper broker behind the ExecutionClient interface.

    All SDK objects are injectable for offline testing:
      * ``trading``  — object exposing ``get_account/get_all_positions/
        get_order_by_id/submit_order/cancel_order_by_id/replace_order_by_id``.
        Defaults to a lazily-built ``alpaca.trading.client.TradingClient``.
      * ``quotes``   — object exposing ``get_latest_quote(symbol)`` (the shape of
        the existing ``AlpacaOptionsClient``). Lazily built by default.
      * ``order_builder`` — maps an OrderRequest to whatever ``trading.submit_order``
        expects; defaults to the SDK builder. Tests pass ``lambda o: o``.
    """

    def __init__(self, *, paper: bool = True,
                 trading: Optional[Any] = None,
                 quotes: Optional[Any] = None,
                 order_builder: Optional[Callable[[OrderRequest], Any]] = None
                 ) -> None:
        self._paper = bool(paper)
        self._trading = trading
        self._quotes = quotes
        self._order_builder = order_builder or _build_sdk_order
        self.name = "alpaca-paper" if self._paper else "alpaca-live"

    # -- lazy default SDK clients (only built if not injected) ------------ #
    def _make_trading(self) -> Any:
        from alpaca.trading.client import TradingClient
        api_key = os.getenv("ALPACA_API_KEY")
        secret_key = os.getenv("ALPACA_SECRET_KEY")
        if not api_key or not secret_key:
            raise RuntimeError("ALPACA_API_KEY / ALPACA_SECRET_KEY not set")
        return TradingClient(api_key, secret_key, paper=self._paper)

    def _make_quotes(self) -> Any:
        # Reuse the existing options data client wrapper (has get_latest_quote).
        from alpaca_client import AlpacaOptionsClient
        return AlpacaOptionsClient()

    def _trading_client(self) -> Any:
        if self._trading is None:
            self._trading = self._make_trading()
        return self._trading

    def _quotes_client(self) -> Any:
        if self._quotes is None:
            self._quotes = self._make_quotes()
        return self._quotes

    # -- reads (fail-open: None / [] on any error) ------------------------ #
    def get_quote(self, symbol: str) -> Optional[Quote]:
        try:
            native = self._quotes_client().get_latest_quote(symbol)
            return _map_quote(native, symbol)
        except Exception:
            return None

    def get_positions(self) -> List[Position]:
        try:
            raw = self._trading_client().get_all_positions()
            out = [_map_position(p) for p in (raw or [])]
            return [p for p in out if p is not None]
        except Exception:
            return []

    def get_buying_power(self) -> Optional[float]:
        try:
            acct = self._trading_client().get_account()
            return _to_float(getattr(acct, "buying_power", None))
        except Exception:
            return None

    def get_order(self, order_id: str) -> Optional[OrderResult]:
        try:
            native = self._trading_client().get_order_by_id(order_id)
            return _map_order(native)
        except Exception:
            return None

    def get_fills(self, order_id: Optional[str] = None) -> List[Fill]:
        # Alpaca exposes fills via the activities API; the order object already
        # carries filled_qty / filled_avg_price, so synthesize a single fill for
        # a filled order rather than pulling a second endpoint. Fail-open [].
        if order_id is None:
            return []
        try:
            res = self.get_order(order_id)
            if res is None or not res.is_filled or res.filled_avg_price is None:
                return []
            return [Fill(order_id=res.order_id, symbol=res.symbol,
                         qty=res.filled_qty, price=res.filled_avg_price,
                         ts=res.submitted_at)]
        except Exception:
            return []

    # -- writes (surface a rejected result rather than raising) ----------- #
    def submit_order(self, order: OrderRequest) -> OrderResult:
        try:
            req = self._order_builder(order)
            native = self._trading_client().submit_order(req)
            mapped = _map_order(native)
            if mapped is None:
                raise RuntimeError("empty order response")
            return mapped
        except Exception as exc:
            return OrderResult(order_id=None, status=STATUS_REJECTED,
                               symbol=order.symbol, side=order.side,
                               client_order_id=order.client_order_id,
                               raw={"error": str(exc)})

    def cancel_order(self, order_id: str) -> OrderResult:
        try:
            self._trading_client().cancel_order_by_id(order_id)
            return OrderResult(order_id=order_id, status=STATUS_CANCELED,
                               symbol="", side="", raw={"canceled": True})
        except Exception as exc:
            return OrderResult(order_id=order_id, status=STATUS_REJECTED,
                               symbol="", side="", raw={"error": str(exc)})

    def replace_order(self, order_id: str, **changes: Any) -> OrderResult:
        try:
            from alpaca.trading.requests import ReplaceOrderRequest
            req = ReplaceOrderRequest(
                qty=changes.get("qty"),
                limit_price=changes.get("limit_price"),
            )
            native = self._trading_client().replace_order_by_id(order_id, req)
            mapped = _map_order(native)
            if mapped is None:
                raise RuntimeError("empty replace response")
            return mapped
        except Exception as exc:
            return OrderResult(order_id=order_id, status=STATUS_REJECTED,
                               symbol="", side="", raw={"error": str(exc)})


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no order placement — injected fakes only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    class _NativeQuote:
        def __init__(self, bid, ask, ts):
            self.bid_price, self.ask_price, self.timestamp = bid, ask, ts

    class _NativePos:
        def __init__(self, sym, qty, avg, mv):
            self.symbol, self.qty = sym, qty
            self.avg_entry_price, self.market_value = avg, mv

    class _NativeAcct:
        buying_power = "25000.50"

    class _NativeOrder:
        def __init__(self, oid, status, sym, side, fq=0.0, fap=None):
            self.id, self.status, self.symbol, self.side = oid, status, sym, side
            self.filled_qty, self.filled_avg_price = fq, fap
            self.client_order_id, self.submitted_at = "coid-1", "2024-01-02T15:00:00Z"

    class _FakeTrading:
        def __init__(self):
            self.submitted = []
            self.canceled = []
        def get_account(self):
            return _NativeAcct()
        def get_all_positions(self):
            return [_NativePos("AAPL240119C00190000", "2", "1.20", "240.0")]
        def get_order_by_id(self, oid):
            return _NativeOrder(oid, "filled", "OPT1", "buy", fq=1.0, fap=1.05)
        def submit_order(self, req):
            self.submitted.append(req)
            return _NativeOrder("ord-1", "accepted", req.symbol, req.side)
        def cancel_order_by_id(self, oid):
            self.canceled.append(oid)
        def replace_order_by_id(self, oid, req):
            return _NativeOrder(oid, "replaced", "OPT1", "buy")

    class _FakeQuotes:
        def get_latest_quote(self, symbol):
            return _NativeQuote(1.00, 1.10, "2024-01-02T15:00:00Z")

    trading = _FakeTrading()
    ac = AlpacaExecutionClient(paper=True, trading=trading,
                               quotes=_FakeQuotes(),
                               order_builder=lambda o: o)  # identity -> offline

    if ac.name != "alpaca-paper":
        print("FAIL: paper name", ac.name); ok = False
    if AlpacaExecutionClient(paper=False, trading=trading,
                             quotes=_FakeQuotes()).name != "alpaca-live":
        print("FAIL: live name"); ok = False

    # Quote mapping (bid/ask/mid/spread).
    q = ac.get_quote("OPT1")
    if q is None or q.bid != 1.00 or q.ask != 1.10 or q.mid != 1.05:
        print("FAIL: quote map", q); ok = False

    # Buying power coerced from a string field.
    if ac.get_buying_power() != 25000.50:
        print("FAIL: buying power", ac.get_buying_power()); ok = False

    # Positions mapping.
    pos = ac.get_positions()
    if len(pos) != 1 or pos[0].qty != 2.0 or pos[0].avg_price != 1.20:
        print("FAIL: positions", pos); ok = False

    # Submit maps native 'accepted' -> our STATUS_ACCEPTED; builder is identity
    # so the fake receives our OrderRequest unchanged.
    res = ac.submit_order(OrderRequest("OPT1", "buy", 1, order_type="market"))
    if res.status != STATUS_ACCEPTED or res.order_id != "ord-1":
        print("FAIL: submit status", res); ok = False
    if not trading.submitted or trading.submitted[0].symbol != "OPT1":
        print("FAIL: submit not forwarded", trading.submitted); ok = False

    # get_order maps 'filled' and get_fills synthesizes one fill.
    got = ac.get_order("ord-9")
    if got is None or got.status != STATUS_FILLED or not got.is_filled:
        print("FAIL: get_order", got); ok = False
    fills = ac.get_fills("ord-9")
    if len(fills) != 1 or fills[0].price != 1.05 or fills[0].qty != 1.0:
        print("FAIL: get_fills synth", fills); ok = False

    # Cancel returns a canceled result and forwards the id.
    c = ac.cancel_order("ord-1")
    if c.status != STATUS_CANCELED or trading.canceled != ["ord-1"]:
        print("FAIL: cancel", c, trading.canceled); ok = False

    # Replace maps native 'replaced' -> STATUS_ACCEPTED (uses real SDK request;
    # tolerate SDK-absent envs by accepting a rejected fail-open too).
    r = ac.replace_order("ord-1", qty=2)
    if r.status not in (STATUS_ACCEPTED, STATUS_REJECTED):
        print("FAIL: replace status", r); ok = False

    # Writes fail-open: a trading client that raises yields a rejected result,
    # never an exception.
    class _Boom:
        def submit_order(self, req):
            raise RuntimeError("broker down")
    boom = AlpacaExecutionClient(paper=True, trading=_Boom(),
                                 quotes=_FakeQuotes(), order_builder=lambda o: o)
    rb = boom.submit_order(OrderRequest("OPT1", "buy", 1))
    if rb.status != STATUS_REJECTED or "broker down" not in str(rb.raw):
        print("FAIL: submit fail-open", rb); ok = False

    # Reads fail-open: a raising client yields None / [] (never raises).
    class _BoomReads:
        def get_account(self):
            raise RuntimeError("x")
        def get_all_positions(self):
            raise RuntimeError("x")
    br = AlpacaExecutionClient(paper=True, trading=_BoomReads(),
                               quotes=_FakeQuotes())
    if br.get_buying_power() is not None or br.get_positions() != []:
        print("FAIL: reads fail-open"); ok = False

    # Status / side normalization edge cases.
    if _norm_status("OrderStatus.PARTIALLY_FILLED") != STATUS_PARTIAL:
        print("FAIL: enum-style status norm"); ok = False
    if _norm_side("OrderSide.SELL") != "sell" or _norm_side(None) != "buy":
        print("FAIL: side norm"); ok = False

    print("execution.alpaca self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
