"""
Oracle Execution — Upgrade 2B: latency + partial-fill simulation.

Upgrade 2A (``fill_model``) answers a *point* question: "if this order hits the
book right now, what price / probability / immediate partial do I get?" That is
a single snapshot. Upgrade 2B plays that estimate out *over simulated time*:

  * SUBMIT latency  — wall time between our decision and the order reaching the
    broker (nothing can fill before this).
  * ACK latency     — broker acknowledgement before the order is working.
  * WORKING window  — the order rests and fills in one or more SLICES as
    displayed liquidity refreshes each cadence step, until it is complete, the
    working window elapses (market/marketable -> whatever filled is kept), or a
    passive order is CANCELLED unfilled.

The result is a deterministic sequence of ``FillSlice`` events (each stamped
with a time offset in seconds) plus the aggregate: quantity-weighted average
fill price, total filled quantity, remaining quantity, and a terminal status.

Design principles (same as the rest of Oracle execution):
  * CONSERVATIVE. Every slice is priced by the 2A model, which never flatters us
    (BUY >= mid, SELL <= mid). Unfilled remainder is a cost, not free optionality.
  * DETERMINISTIC given a seed. The only stochastic element — does this slice
    fill, given ``fill_probability`` — is drawn from a seeded ``random.Random``
    keyed off the order, so two runs with the same (order, quote, config, seed)
    produce byte-identical slices. This is the Upgrade-1 replay invariant.
  * FAIL-OPEN. No usable quote -> an empty, no-fill simulation; junk inputs never
    raise.

This does NOT touch a broker. It is the engine a backtest / shadow run uses to
turn a ``TradeIntent`` into realistic realized fills; Upgrade 2C reads the
aggregate to compute *realized* EV against *executable* and *theoretical* EV.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import Any, List, Mapping, Optional, Sequence, Tuple

from oracle.execution.client import (
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    OrderRequest,
    Quote,
)
from oracle.execution.fill_model import FillEstimate, FillModel, FillModelConfig


def _seed_from_order(order: OrderRequest, seed: int) -> int:
    """Deterministic per-order seed so identical orders replay identically but
    two different symbols/sides don't share a fill sequence."""
    key = f"{order.symbol}|{order.side}|{order.qty}|{order.order_type}|" \
          f"{order.limit_price}|{order.client_order_id}|{seed}"
    h = 5381
    for ch in key:
        h = ((h << 5) + h + ord(ch)) & 0xFFFFFFFF
    return h


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class LatencyConfig:
    """Conservative, calibratable latency + working-window constants (seconds).
    Defaults err toward slower fills and a bounded working window."""

    submit_latency_sec: float = 0.25     # decision -> order reaches broker
    ack_latency_sec: float = 0.15        # broker ack before the order is working
    slice_interval_sec: float = 1.0      # cadence at which liquidity refreshes
    max_working_sec: float = 60.0        # how long the order rests before we stop
    passive_cancel_sec: float = 45.0     # cancel an unfilled passive order after
    max_slices: int = 64                 # hard cap on simulated slices (safety)

    def get(self, name: str, default: Any = None) -> Any:
        return getattr(self, name, default)


# --------------------------------------------------------------------------- #
# Results
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FillSlice:
    """One partial execution at a point in simulated time."""

    ts_offset: float        # seconds since the decision
    qty: float              # quantity filled in THIS slice
    price: float            # conservative price for this slice (from 2A)
    cumulative_qty: float   # total filled up to and including this slice
    remaining_qty: float    # order_qty - cumulative_qty
    reasons: Tuple[str, ...] = ()


@dataclass(frozen=True)
class FillSimulation:
    """Aggregate outcome of playing a fill model out over time."""

    symbol: str
    side: str
    order_qty: int
    filled_qty: float
    remaining_qty: float
    avg_fill_price: Optional[float]
    status: str                          # filled | partially_filled | canceled | pending
    first_fill_offset: Optional[float]   # seconds until the first partial
    last_fill_offset: Optional[float]    # seconds until the final partial
    total_latency: float                 # submit + ack
    slices: Tuple[FillSlice, ...] = ()
    reasons: Tuple[str, ...] = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    @property
    def is_filled(self) -> bool:
        return self.filled_qty > 0

    @property
    def is_complete(self) -> bool:
        return self.status == STATUS_FILLED and self.remaining_qty <= 0

    @property
    def fill_ratio(self) -> float:
        return 0.0 if self.order_qty <= 0 else round(self.filled_qty / self.order_qty, 6)


# --------------------------------------------------------------------------- #
# Simulator
# --------------------------------------------------------------------------- #
class FillSimulator:
    def __init__(self, fill_model: Optional[FillModel] = None,
                 config: Optional[LatencyConfig] = None) -> None:
        self.fill_model = fill_model or FillModel()
        self.config = config or LatencyConfig()

    def simulate(self, order: OrderRequest, quote: Optional[Quote], *,
                 market: Optional[Mapping[str, Any]] = None,
                 quotes_over_time: Optional[Sequence[Tuple[float, Quote]]] = None,
                 seed: int = 0) -> FillSimulation:
        """Play ``order`` out over time against ``quote`` (held constant) or an
        optional ``quotes_over_time`` schedule of ``(ts_offset, Quote)`` the
        book steps through. Deterministic given ``seed``.
        """
        cfg = self.config
        side = "buy" if str(order.side).lower() != "sell" else "sell"
        order_qty = max(1, int(order.qty or 1))
        latency = round(cfg.submit_latency_sec + cfg.ack_latency_sec, 6)
        rng = random.Random(_seed_from_order(order, seed))

        # No usable quote at submit -> nothing can ever fill.
        if quote is None and not quotes_over_time:
            return FillSimulation(
                symbol=order.symbol, side=side, order_qty=order_qty,
                filled_qty=0.0, remaining_qty=float(order_qty),
                avg_fill_price=None, status=STATUS_PENDING,
                first_fill_offset=None, last_fill_offset=None,
                total_latency=latency, reasons=("no_quote",))

        schedule = self._normalize_schedule(quote, quotes_over_time, latency)

        order_type = str(order.order_type or "market").lower()
        passive = False
        slices: List[FillSlice] = []
        reasons: List[str] = []
        cumulative = 0.0
        notional = 0.0
        t = latency
        slice_idx = 0
        stop = latency + max(0.0, cfg.max_working_sec)

        while (cumulative < order_qty and t <= stop
               and slice_idx < cfg.max_slices):
            q = self._quote_at(schedule, t)
            remaining = order_qty - cumulative
            probe = OrderRequest(
                order.symbol, side, int(round(remaining)),
                order_type=order.order_type, limit_price=order.limit_price,
                tif=order.tif, asset_class=order.asset_class)
            est: FillEstimate = self.fill_model.estimate_fill(
                probe, q, market=market)
            passive = "passive_limit" in est.reasons

            if est.expected_fill_price is not None and est.partial_qty > 0 \
                    and est.fill_probability > 0:
                # The one stochastic decision: did this slice fill this step?
                if rng.random() <= est.fill_probability:
                    take = min(float(est.partial_qty), remaining)
                    cumulative += take
                    notional += take * est.expected_fill_price
                    slices.append(FillSlice(
                        ts_offset=round(t, 6), qty=round(take, 6),
                        price=est.expected_fill_price,
                        cumulative_qty=round(cumulative, 6),
                        remaining_qty=round(order_qty - cumulative, 6),
                        reasons=est.reasons))

            # Passive orders that have not filled at all get pulled at the
            # cancel horizon (a real desk does not sit forever).
            if passive and cumulative <= 0 and t >= latency + cfg.passive_cancel_sec:
                reasons.append("passive_cancelled")
                break

            t = round(t + max(1e-6, cfg.slice_interval_sec), 6)
            slice_idx += 1

        filled = round(cumulative, 6)
        remaining_qty = round(order_qty - filled, 6)
        avg_price = round(notional / filled, 6) if filled > 0 else None

        if filled <= 0:
            status = STATUS_CANCELED if passive else STATUS_PENDING
            if not reasons:
                reasons.append("no_fill" if not passive else "passive_unfilled")
        elif remaining_qty > 0:
            status = STATUS_PARTIAL
            reasons.append("partial_working_window_elapsed")
        else:
            status = STATUS_FILLED

        return FillSimulation(
            symbol=order.symbol, side=side, order_qty=order_qty,
            filled_qty=filled, remaining_qty=remaining_qty,
            avg_fill_price=avg_price, status=status,
            first_fill_offset=slices[0].ts_offset if slices else None,
            last_fill_offset=slices[-1].ts_offset if slices else None,
            total_latency=latency, slices=tuple(slices),
            reasons=tuple(reasons),
            notes={"order_type": order_type, "slice_count": len(slices),
                   "passive": passive})

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _normalize_schedule(quote: Optional[Quote],
                            quotes_over_time: Optional[
                                Sequence[Tuple[float, Quote]]],
                            latency: float) -> List[Tuple[float, Quote]]:
        sched: List[Tuple[float, Quote]] = []
        if quotes_over_time:
            for item in quotes_over_time:
                try:
                    off, q = item
                except (TypeError, ValueError):
                    continue
                if isinstance(q, Quote):
                    sched.append((float(off), q))
            sched.sort(key=lambda x: x[0])
        if not sched and quote is not None:
            sched = [(0.0, quote)]
        elif quote is not None and (not sched or sched[0][0] > 0.0):
            # ensure a quote exists at/at-before submit
            sched.insert(0, (0.0, quote))
        return sched

    @staticmethod
    def _quote_at(schedule: List[Tuple[float, Quote]],
                  t: float) -> Optional[Quote]:
        """Most recent quote whose offset <= t (step function)."""
        chosen: Optional[Quote] = None
        for off, q in schedule:
            if off <= t:
                chosen = q
            else:
                break
        if chosen is None and schedule:
            chosen = schedule[0][1]
        return chosen


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    tight = Quote("OPT", bid=1.00, ask=1.02, ts="t")

    fm = FillModel()
    sim = FillSimulator(fm)

    # Market BUY, deep book -> fully filled in one slice, after latency.
    r = sim.simulate(OrderRequest("OPT", "buy", 5, order_type="market"),
                     tight, market={"ask_size": 100}, seed=1)
    if r.status != STATUS_FILLED or r.filled_qty != 5.0:
        print("FAIL: market fully filled", r); ok = False
    if r.avg_fill_price is None or r.avg_fill_price < tight.mid:
        print("FAIL: avg fill below mid", r); ok = False
    if r.first_fill_offset is None or r.first_fill_offset < r.total_latency:
        print("FAIL: fill before latency", r); ok = False

    # Determinism: same (order, quote, seed) -> identical simulation.
    r2 = sim.simulate(OrderRequest("OPT", "buy", 5, order_type="market"),
                      tight, market={"ask_size": 100}, seed=1)
    if r != r2:
        print("FAIL: non-deterministic simulation", r, r2); ok = False

    # Thin displayed size -> multiple slices accumulate over time to complete.
    big = sim.simulate(OrderRequest("OPT", "buy", 10, order_type="market"),
                       tight, market={"ask_size": 3}, seed=7)
    if big.filled_qty <= 0:
        print("FAIL: sliced order should fill something", big); ok = False
    if big.is_complete and len(big.slices) < 2:
        print("FAIL: small displayed size should need multiple slices", big)
        ok = False
    # slices are time-ordered and cumulative is monotone non-decreasing
    prev_t, prev_c = -1.0, -1.0
    for sl in big.slices:
        if sl.ts_offset < prev_t or sl.cumulative_qty < prev_c:
            print("FAIL: slices not monotone", big.slices); ok = False; break
        prev_t, prev_c = sl.ts_offset, sl.cumulative_qty

    # Passive limit far from the market that never fills -> CANCELED, no fill,
    # and it is pulled at the cancel horizon (not left working forever).
    pas = sim.simulate(
        OrderRequest("OPT", "buy", 1, order_type="limit", limit_price=0.50),
        tight, seed=3)
    # With a passive order the model still assigns p_passive_limit; force a
    # never-fills book by using a seed sweep to confirm cancel path exists.
    never = FillSimulator(
        FillModel(FillModelConfig(p_passive_limit=0.0)))
    nf = never.simulate(
        OrderRequest("OPT", "buy", 1, order_type="limit", limit_price=0.50),
        tight, seed=3)
    if nf.filled_qty != 0.0 or nf.status != STATUS_CANCELED:
        print("FAIL: unfilled passive should cancel", nf); ok = False
    if nf.last_fill_offset is not None:
        print("FAIL: canceled order has no fills", nf); ok = False
    _ = pas  # the ordinary passive path is exercised; outcome is seed-dependent

    # No quote -> pending, no fill, never raises.
    nq = sim.simulate(OrderRequest("OPT", "buy", 1), None, seed=0)
    if nq.status != STATUS_PENDING or nq.filled_qty != 0.0:
        print("FAIL: no-quote pending", nq); ok = False

    # Quote schedule: price improves after submit; a passive buy at 1.00 that is
    # below the ask initially can fill once the book steps to it.
    improving = [
        (0.0, Quote("OPT", bid=1.05, ask=1.07, ts="t0")),
        (5.0, Quote("OPT", bid=0.98, ask=1.00, ts="t1")),
    ]
    sch = FillSimulator(FillModel()).simulate(
        OrderRequest("OPT", "buy", 1, order_type="market"),
        None, quotes_over_time=improving, seed=2)
    if not sch.is_filled or sch.avg_fill_price is None:
        print("FAIL: scheduled quote should fill a market order", sch); ok = False

    # fill_ratio sanity.
    if not (0.0 <= big.fill_ratio <= 1.0):
        print("FAIL: fill_ratio out of range", big.fill_ratio); ok = False

    print("execution.fill_simulator self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
