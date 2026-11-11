"""
Oracle Execution — deterministic order lifecycle (Phase 2c).

``OrderLifecycle`` is the stateful layer between a decision path's TradeIntent
and an ``ExecutionClient`` (sim, shadow, paper or live). The client layer below
is stateless per call; this layer owns the order's LIFE: submission, broker
acknowledgement, rejection, cancellation, local expiry, partial fills, late
fills, and retries — with three hard guarantees the master prompt requires:

  1. POSITION = FILLS ONLY. The open position reported here is derived solely
     from broker-confirmed filled quantity. An accepted-but-unfilled order, a
     pending cancel, or an expired intent contributes exactly zero.
  2. RECONCILE BEFORE RETRY. ``retry`` refuses to create a replacement order
     until the broker state of the previous attempt (and every attempt in the
     chain) is re-read and confirmed terminal. The replacement is sized to the
     REMAINING quantity (original intent minus everything already filled across
     the chain), so a retry can never produce an excess position.
  3. NO DUPLICATES / IDEMPOTENT REPLAY. Every order carries a caller-supplied
     stable ``client_order_id``. Submitting the same id twice returns the
     existing record without touching the broker; retries derive deterministic
     child ids (``<root>#r2``, ``<root>#r3``...). Replaying the same call
     sequence against the same client reproduces byte-identical histories.

Races and ambiguity are handled conservatively:
  * CANCEL/FILL RACE — once a cancel is requested, a later poll that shows the
    broker filled the order anyway ADOPTS the fill (position reflects reality,
    an explicit ``late_fill_after_cancel_request`` event is recorded). A fill
    always beats our local intent to cancel.
  * AMBIGUOUS SUBMIT — if the transport raised (or the client surfaced a
    rejection with no broker order id), the order MAY exist at the broker.
    Such records block ``retry`` until an injected
    ``lookup_by_client_order_id`` resolves them (found -> adopt; not found ->
    confirmed never-submitted and the retry may proceed). With no lookup
    available the retry stays blocked — duplicate protection wins.
  * EXPIRY — an order with ``expiry_s`` (e.g. the EV-capped entry variant's
    cap expiry) is cancelled by ``poll`` once its age reaches the horizon; a
    fill reported afterwards is still adopted (late fill).

Time is an explicit ``now: float`` argument on every call — no wall clock is
read — so replay is deterministic and testable at any speed.
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from oracle.execution.client import (
    STATUS_ACCEPTED,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    STATUS_REJECTED,
    ExecutionClient,
    OrderRequest,
    OrderResult,
)

# Local lifecycle states (a superset of broker status: adds expiry/ambiguity).
STATE_WORKING = "working"
STATE_FILLED = "filled"
STATE_PARTIAL = "partially_filled"   # terminal: canceled/expired WITH some fill
STATE_CANCELED = "canceled"          # terminal: canceled, zero fill
STATE_REJECTED = "rejected"          # terminal: broker rejected, zero fill
STATE_EXPIRED = "expired"            # terminal: our expiry cancel, zero fill
STATE_UNKNOWN = "unknown"            # submit outcome ambiguous (blocks retry)

TERMINAL_STATES = (STATE_FILLED, STATE_PARTIAL, STATE_CANCELED,
                   STATE_REJECTED, STATE_EXPIRED)


@dataclass
class ManagedOrder:
    """One order attempt and everything we learned about it, in event order."""

    client_order_id: str
    request: OrderRequest
    submitted_ts: float
    expiry_s: Optional[float] = None
    order_id: Optional[str] = None
    state: str = STATE_WORKING
    filled_qty: float = 0.0
    avg_fill_price: Optional[float] = None
    cancel_requested: bool = False
    expiry_cancel: bool = False
    ambiguous: bool = False
    parent: Optional[str] = None
    root: Optional[str] = None
    events: List[Tuple[float, str, str]] = field(default_factory=list)

    @property
    def is_terminal(self) -> bool:
        return self.state in TERMINAL_STATES

    @property
    def signed_filled(self) -> float:
        sign = 1.0 if str(self.request.side).lower() != "sell" else -1.0
        return sign * self.filled_qty

    def to_dict(self) -> Dict[str, Any]:
        return {
            "client_order_id": self.client_order_id,
            "order_id": self.order_id,
            "symbol": self.request.symbol,
            "side": self.request.side,
            "qty": self.request.qty,
            "order_type": self.request.order_type,
            "limit_price": self.request.limit_price,
            "state": self.state,
            "filled_qty": self.filled_qty,
            "avg_fill_price": self.avg_fill_price,
            "submitted_ts": self.submitted_ts,
            "expiry_s": self.expiry_s,
            "cancel_requested": self.cancel_requested,
            "expiry_cancel": self.expiry_cancel,
            "ambiguous": self.ambiguous,
            "parent": self.parent,
            "root": self.root or self.client_order_id,
            "events": [list(e) for e in self.events],
        }


class OrderLifecycle:
    """Deterministic order-state machine over an ``ExecutionClient``.

    ``lookup_by_client_order_id`` (optional) resolves an ambiguous submit:
    given a client_order_id it returns the broker's ``OrderResult`` if the
    order exists, else ``None``. Without it, ambiguous records permanently
    block retries (conservative duplicate protection).
    """

    def __init__(self, client: ExecutionClient, *,
                 lookup_by_client_order_id: Optional[
                     Callable[[str], Optional[OrderResult]]] = None) -> None:
        self._client = client
        self._lookup = lookup_by_client_order_id
        self._orders: Dict[str, ManagedOrder] = {}

    # ------------------------------------------------------------------ reads
    def record(self, client_order_id: str) -> Optional[ManagedOrder]:
        return self._orders.get(client_order_id)

    def records(self) -> List[ManagedOrder]:
        return list(self._orders.values())

    def open_positions(self) -> Dict[str, float]:
        """Positions derived ONLY from broker-confirmed fills."""
        out: Dict[str, float] = {}
        for rec in self._orders.values():
            if rec.filled_qty <= 0:
                continue
            sym = rec.request.symbol
            out[sym] = round(out.get(sym, 0.0) + rec.signed_filled, 6)
        return {k: v for k, v in out.items() if v}

    def chain_filled(self, root: str) -> float:
        return round(sum(r.filled_qty for r in self._orders.values()
                         if (r.root or r.client_order_id) == root), 6)

    def history(self, client_order_id: str) -> List[Tuple[float, str, str]]:
        rec = self._orders.get(client_order_id)
        return list(rec.events) if rec else []

    # ----------------------------------------------------------------- writes
    def submit(self, request: OrderRequest, *, now: float,
               expiry_s: Optional[float] = None) -> ManagedOrder:
        """Submit an order. Idempotent on ``client_order_id``: a second submit
        with the same id returns the existing record and never re-sends."""
        coid = request.client_order_id
        if not coid:
            raise ValueError("OrderRequest.client_order_id is required for "
                             "lifecycle management (stable, joinable ids)")
        existing = self._orders.get(coid)
        if existing is not None:
            existing.events.append((round(now, 6),
                                    "duplicate_submit_ignored", ""))
            return existing

        rec = ManagedOrder(client_order_id=coid, request=request,
                           submitted_ts=float(now), expiry_s=expiry_s,
                           root=coid)
        self._orders[coid] = rec
        rec.events.append((round(now, 6), "submit", request.symbol))
        try:
            res = self._client.submit_order(request)
        except Exception as exc:  # transport raised -> order MAY exist
            rec.state = STATE_UNKNOWN
            rec.ambiguous = True
            rec.events.append((round(now, 6), "submit_error", str(exc)))
            return rec
        self._absorb(rec, res, now, "submit_ack")
        return rec

    def cancel(self, client_order_id: str, *, now: float) -> Optional[ManagedOrder]:
        """Request a manual cancel. A fill that races the cancel still wins."""
        rec = self._orders.get(client_order_id)
        if rec is None or rec.is_terminal or rec.order_id is None:
            return rec
        self._request_cancel(rec, now, expiry=False)
        return rec

    def poll(self, now: float) -> List[ManagedOrder]:
        """Re-read broker state for every order that could still change, apply
        transitions, and enforce expiry. Returns the records that changed state
        or quantity during this poll (deterministic, insertion-ordered)."""
        changed: List[ManagedOrder] = []
        for rec in list(self._orders.values()):
            # FILLED and REJECTED cannot change; CANCELED/EXPIRED/PARTIAL can
            # (a late fill may still arrive after a cancel ack).
            if rec.state in (STATE_FILLED, STATE_REJECTED) and not rec.ambiguous:
                continue
            before = (rec.state, rec.filled_qty)
            if rec.order_id is None:
                self._try_lookup(rec, now)
            else:
                try:
                    res = self._client.get_order(rec.order_id)
                except Exception:
                    res = None
                if res is not None:
                    self._absorb(rec, res, now, "poll")
                if (rec.state == STATE_WORKING
                        and rec.expiry_s is not None
                        and not rec.cancel_requested
                        and (now - rec.submitted_ts) >= rec.expiry_s - 1e-9):
                    self._request_cancel(rec, now, expiry=True)
            if (rec.state, rec.filled_qty) != before:
                changed.append(rec)
        return changed

    def retry(self, client_order_id: str, *, now: float,
              request: Optional[OrderRequest] = None) -> Optional[ManagedOrder]:
        """Create a replacement order for a failed attempt — but ONLY after
        reconciling broker state and ONLY for the remaining quantity. Returns
        the new child record, or None when the retry is refused (still
        working, ambiguous, or nothing left to fill)."""
        rec = self._orders.get(client_order_id)
        if rec is None:
            return None

        # 1) Reconcile this attempt with the broker before anything else.
        if rec.order_id is not None:
            try:
                res = self._client.get_order(rec.order_id)
            except Exception:
                res = None
            if res is not None:
                self._absorb(rec, res, now, "reconcile")
        elif rec.ambiguous:
            self._try_lookup(rec, now)

        if rec.ambiguous or rec.state == STATE_UNKNOWN:
            rec.events.append((round(now, 6), "retry_blocked_ambiguous", ""))
            return None
        if rec.state == STATE_WORKING:
            rec.events.append((round(now, 6), "retry_blocked_working", ""))
            return None

        # 2) The whole chain must be settled; size the child to what remains.
        root = rec.root or rec.client_order_id
        chain = [r for r in self._orders.values()
                 if (r.root or r.client_order_id) == root]
        if any(r.state == STATE_WORKING or r.ambiguous
               or r.state == STATE_UNKNOWN for r in chain):
            rec.events.append((round(now, 6), "retry_blocked_chain_open", ""))
            return None
        root_rec = self._orders.get(root, rec)
        target = int(root_rec.request.qty)
        filled = sum(r.filled_qty for r in chain)
        remaining = int(round(target - filled))
        if remaining <= 0:
            rec.events.append((round(now, 6), "retry_refused_complete",
                               f"filled={filled}"))
            return None

        # 3) Deterministic child id; submit() stays the single write path.
        child_coid = f"{root}#r{len(chain) + 1}"
        if child_coid in self._orders:  # idempotent replay guard
            return self._orders[child_coid]
        base = request if request is not None else rec.request
        child_req = dataclasses.replace(base, qty=remaining,
                                        client_order_id=child_coid)
        child = self.submit(child_req, now=now, expiry_s=rec.expiry_s)
        child.parent = rec.client_order_id
        child.root = root
        return child

    # --------------------------------------------------------------- internals
    def _request_cancel(self, rec: ManagedOrder, now: float, *,
                        expiry: bool) -> None:
        rec.cancel_requested = True
        rec.expiry_cancel = rec.expiry_cancel or expiry
        rec.events.append((round(now, 6),
                           "expiry_cancel_requested" if expiry
                           else "cancel_requested",
                           f"age={round(now - rec.submitted_ts, 6)}s"))
        try:
            res = self._client.cancel_order(rec.order_id)
        except Exception as exc:
            rec.events.append((round(now, 6), "cancel_error", str(exc)))
            return
        if res is not None:
            self._absorb(rec, res, now, "cancel_response")

    def _try_lookup(self, rec: ManagedOrder, now: float) -> None:
        """Resolve an ambiguous submit by client_order_id, if a lookup exists."""
        if self._lookup is None:
            return
        try:
            res = self._lookup(rec.client_order_id)
        except Exception as exc:
            rec.events.append((round(now, 6), "lookup_error", str(exc)))
            return
        if res is not None:
            rec.ambiguous = False
            self._absorb(rec, res, now, "reconcile_lookup")
        else:
            # Confirmed: the broker never saw this order.
            rec.ambiguous = False
            if rec.state == STATE_UNKNOWN:
                rec.state = STATE_REJECTED
            rec.events.append((round(now, 6), "reconcile_not_found", ""))

    def _absorb(self, rec: ManagedOrder, res: OrderResult, now: float,
                label: str) -> None:
        """Fold one broker observation into the record. Monotone on fills;
        a confirmed fill is never downgraded by a later stale read."""
        if res.order_id and not rec.order_id:
            rec.order_id = res.order_id
        # Broker filled_qty is cumulative; guard against out-of-order reads.
        fq = float(res.filled_qty or 0.0)
        if fq > rec.filled_qty:
            rec.filled_qty = fq
            if res.filled_avg_price is not None:
                rec.avg_fill_price = float(res.filled_avg_price)
        status = res.status
        rec.events.append((round(now, 6), label, status))

        if rec.state == STATE_FILLED:
            return  # fills are final
        if status == STATUS_FILLED:
            if rec.cancel_requested:
                rec.events.append((round(now, 6),
                                   "late_fill_after_cancel_request", ""))
            rec.state = STATE_FILLED
            rec.ambiguous = False
        elif status == STATUS_PARTIAL:
            rec.state = STATE_WORKING  # still live; terminal only on cancel
            rec.ambiguous = False
        elif status == STATUS_CANCELED:
            if rec.filled_qty > 0:
                rec.state = STATE_PARTIAL
            else:
                rec.state = STATE_EXPIRED if rec.expiry_cancel else STATE_CANCELED
            rec.ambiguous = False
        elif status == STATUS_REJECTED:
            if label == "cancel_response":
                # The CANCEL was rejected (too late / unknown) — the order
                # itself is unchanged; a later poll decides fill vs cancel.
                return
            rec.state = STATE_REJECTED
            if res.order_id is None:
                rec.ambiguous = True  # may still exist at the broker
        elif status in (STATUS_ACCEPTED, STATUS_PENDING):
            if rec.state in (STATE_CANCELED, STATE_EXPIRED, STATE_PARTIAL):
                return  # never resurrect a confirmed terminal from a stale read
            rec.state = STATE_WORKING
            rec.ambiguous = False


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from oracle.execution.client import Quote, SimExecutionClient

    ok = True

    # 1) Instant marketable fill -> FILLED; position == fills only.
    sim = SimExecutionClient(quotes={"OPT": Quote("OPT", bid=1.00, ask=1.05)})
    life = OrderLifecycle(sim)
    rec = life.submit(OrderRequest("OPT", "buy", 2, order_type="market",
                                   client_order_id="c1"), now=0.0)
    if rec.state != STATE_FILLED or rec.filled_qty != 2.0:
        print("FAIL: instant fill", rec.to_dict()); ok = False
    if life.open_positions() != {"OPT": 2.0}:
        print("FAIL: position from fills", life.open_positions()); ok = False

    # 2) Duplicate submit is idempotent (one broker order only).
    again = life.submit(OrderRequest("OPT", "buy", 2, order_type="market",
                                     client_order_id="c1"), now=1.0)
    if again is not rec or len(sim._orders) != 1:
        print("FAIL: duplicate submit not idempotent"); ok = False

    # 3) Resting limit -> working; expiry cancels it; position unchanged.
    rest = life.submit(OrderRequest("OPT", "buy", 1, order_type="limit",
                                    limit_price=0.90, client_order_id="c2"),
                       now=2.0, expiry_s=30.0)
    life.poll(now=10.0)
    if rest.state != STATE_WORKING:
        print("FAIL: resting should be working", rest.state); ok = False
    life.poll(now=32.0)   # age 30s -> expiry cancel
    if rest.state != STATE_EXPIRED or rest.filled_qty != 0.0:
        print("FAIL: expiry", rest.to_dict()); ok = False
    if life.open_positions() != {"OPT": 2.0}:
        print("FAIL: expired order leaked into position"); ok = False

    # 4) Reconcile-before-retry: child sized to remaining, deterministic id.
    child = life.retry("c2", now=33.0,
                       request=dataclasses.replace(rest.request,
                                                   order_type="market",
                                                   limit_price=None))
    if child is None or child.client_order_id != "c2#r2" or \
            child.request.qty != 1 or child.state != STATE_FILLED:
        print("FAIL: retry child", child and child.to_dict()); ok = False
    if life.retry("c2", now=34.0) is not None:   # chain complete -> refuse
        print("FAIL: retry after complete chain"); ok = False

    # 5) Missing client_order_id is a programming error.
    try:
        life.submit(OrderRequest("OPT", "buy", 1), now=0.0)
        print("FAIL: submit without client_order_id should raise"); ok = False
    except ValueError:
        pass

    # 6) Determinism: identical script on fresh sims -> identical histories.
    def run_script():
        s = SimExecutionClient(quotes={"OPT": Quote("OPT", 1.00, 1.05)})
        lf = OrderLifecycle(s)
        lf.submit(OrderRequest("OPT", "buy", 1, order_type="limit",
                               limit_price=0.90, client_order_id="k1"),
                  now=0.0, expiry_s=5.0)
        lf.poll(now=3.0)
        lf.poll(now=6.0)
        lf.retry("k1", now=7.0)
        return [r.to_dict() for r in lf.records()]
    if run_script() != run_script():
        print("FAIL: replay not deterministic"); ok = False

    print("execution.lifecycle self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
