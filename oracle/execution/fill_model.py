"""
Oracle Execution — Upgrade 2A: fill / slippage model.

``estimate_fill`` turns an ``OrderRequest`` + a ``Quote`` (+ optional market
microstructure context) into a ``FillEstimate`` — the realistic price and
probability a real broker would give you, not the friendly mid. This is what
separates *theoretical* EV (priced at mid / fair value) from *executable* EV
(priced at what you can actually get filled at), which Upgrade 2C consumes.

Design principles (mirror ``cost_model`` and the rest of Oracle):
  * CONSERVATIVE by construction. A BUY never fills below the mid; a SELL never
    fills above it. Any modeling uncertainty costs YOU, never flatters the EV.
  * Deterministic. Same (order, quote, market) -> byte-identical estimate, so a
    replay is reproducible and comparable to live (the Upgrade-1 invariant).
  * Calibratable + documented. ``fill_probability`` / ``expected_fill_delay`` /
    the slippage components are transparent heuristics with conservative default
    constants meant to be tuned from real fills (backtest -> paper -> live).
  * Fail-open. Missing microstructure (sizes / OI / volume) degrades to the
    quote-only estimate; junk inputs never raise.

Microstructure context (all optional, passed via ``market=``):
  bid_size, ask_size, volume, open_interest, quote_age_sec, underlying_vol,
  iv, dte, delta, gamma, theta.

The model plugs straight into ``SimExecutionClient`` via ``as_sim_fill_model()``
so backtest fills use exactly this logic.

NOTE (staging): Upgrade 2A covers the price + slippage + liquidity model and the
fields for fill-probability / delay / partial. The full latency + partial-fill
*event simulation* (multi-slice fills over time) lands in Upgrade 2B on top of
these same fields.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Mapping, Optional, Tuple

from oracle.execution.client import (
    STATUS_FILLED,
    STATUS_PARTIAL,
    STATUS_PENDING,
    OrderRequest,
    Quote,
)


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


# --------------------------------------------------------------------------- #
# Config (frozen -> a fill model instance is reproducible / comparable)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FillModelConfig:
    """Conservative, calibratable constants. Defaults err toward WORSE fills."""

    tick_size: float = 0.01
    # slippage (per contract, in $) beyond crossing the spread
    base_slippage: float = 0.01           # fixed adverse selection at the touch
    spread_penalty_frac: float = 0.10     # extra = frac * spread
    # size impact: crossing more than the displayed size costs progressively more
    size_impact_frac: float = 0.50        # of spread, per unit of excess-size ratio
    size_impact_cap_frac: float = 1.0     # cap size impact at frac * spread
    # liquidity penalty for thin names
    oi_floor: float = 500.0               # OI below this is "thin"
    volume_floor: float = 100.0           # daily contract volume below this is thin
    illiquidity_penalty_frac: float = 0.25  # of spread, added when thin
    # fill probability base rates (before thinness / passiveness adjustment)
    p_market: float = 0.99
    p_marketable_limit: float = 0.95
    p_passive_limit: float = 0.55
    # expected fill delay (seconds)
    delay_marketable: float = 1.0
    delay_passive: float = 30.0
    # a spread wider than this fraction of mid is treated as a "wide" market
    wide_spread_frac: float = 0.15
    # provenance (additive, behaviourally inert): stamped by the calibrator so a
    # caller / report can tell a learned config from the conservative default.
    # ``estimate_fill`` never reads these, so a calibrated config with the same
    # numeric constants produces byte-identical estimates to the default.
    calibrated: bool = False
    n_samples: int = 0

    def get(self, name: str, default: Any = None) -> Any:  # convenience
        return getattr(self, name, default)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class FillEstimate:
    """What a realistic fill looks like. ``expected_fill_price`` is the
    conservative price you should assume; ``fill_probability`` is the chance the
    order fills at all in its intended window; ``partial_qty`` is the quantity
    you should expect to get immediately at the touch."""

    symbol: str
    side: str
    order_qty: int
    expected_fill_price: Optional[float]
    fill_probability: float
    expected_fill_delay: float            # seconds
    slippage_estimate: float              # $ per contract vs the touch
    liquidity_penalty: float              # $ per contract from thinness/size
    partial_qty: float
    status: str                           # filled | partially_filled | pending
    mid: Optional[float] = None
    touch: Optional[float] = None         # ask for a buy, bid for a sell
    reasons: Tuple[str, ...] = ()
    notes: Dict[str, Any] = field(default_factory=dict)

    @property
    def is_fillable(self) -> bool:
        return self.status in (STATUS_FILLED, STATUS_PARTIAL) and self.partial_qty > 0


# --------------------------------------------------------------------------- #
# Model
# --------------------------------------------------------------------------- #
class FillModel:
    def __init__(self, config: Optional[FillModelConfig] = None) -> None:
        self.config = config or FillModelConfig()

    # ------------------------------------------------------------------ core
    def estimate_fill(self, order: OrderRequest, quote: Optional[Quote],
                      *, market: Optional[Mapping[str, Any]] = None
                      ) -> FillEstimate:
        cfg = self.config
        mkt = dict(market or {})
        side = "buy" if str(order.side).lower() != "sell" else "sell"
        buy = side == "buy"
        qty = max(1, int(order.qty or 1))

        bid = _to_float(quote.bid) if quote else None
        ask = _to_float(quote.ask) if quote else None

        # No usable quote -> nothing to model. Pending, zero fill probability.
        if bid is None and ask is None:
            return FillEstimate(
                symbol=order.symbol, side=side, order_qty=qty,
                expected_fill_price=None, fill_probability=0.0,
                expected_fill_delay=cfg.delay_passive, slippage_estimate=0.0,
                liquidity_penalty=0.0, partial_qty=0.0, status=STATUS_PENDING,
                reasons=("no_quote",))

        # Fill in a one-sided quote conservatively.
        if bid is None:
            bid = ask
        if ask is None:
            ask = bid
        mid = round((bid + ask) / 2.0, 6)
        spread = max(0.0, round(ask - bid, 6))
        touch = ask if buy else bid

        # ---- slippage components (all >= 0, all in $ per contract) --------
        base = max(0.0, cfg.base_slippage)
        spread_comp = cfg.spread_penalty_frac * spread

        displayed = _to_float(mkt.get("ask_size" if buy else "bid_size"))
        size_comp = 0.0
        if displayed is not None and displayed > 0 and qty > displayed:
            excess_ratio = (qty - displayed) / displayed
            size_comp = min(cfg.size_impact_frac * spread * excess_ratio,
                            cfg.size_impact_cap_frac * spread)

        oi = _to_float(mkt.get("open_interest"))
        vol = _to_float(mkt.get("volume"))
        thin = ((oi is not None and oi < cfg.oi_floor) or
                (vol is not None and vol < cfg.volume_floor))
        liq_penalty = cfg.illiquidity_penalty_frac * spread if thin else 0.0

        slippage = round(base + spread_comp + size_comp, 6)

        # ---- expected fill price (conservative: worse for us) -------------
        if buy:
            price = touch + slippage + liq_penalty
            price = max(price, mid)                 # never fill a buy below mid
        else:
            price = touch - slippage - liq_penalty
            price = min(price, mid)                 # never fill a sell above mid
            price = max(0.0, price)
        price = round(price, 6)

        # ---- order-type gating (market / marketable-limit / passive) ------
        order_type = str(order.order_type or "market").lower()
        limit = _to_float(order.limit_price)
        reasons = []
        if order_type == "market":
            p_base = cfg.p_market
            delay = cfg.delay_marketable
            reasons.append("market")
        else:  # limit
            marketable = (limit is not None and
                          (limit >= ask if buy else limit <= bid))
            if limit is None:
                # a limit order with no price is treated as non-marketable
                return _pending(order, side, qty, mid, touch, spread,
                                slippage, liq_penalty, cfg,
                                reason="limit_without_price")
            if marketable:
                # fill no worse than the limit, no better than our conservative
                # crossed price.
                if buy:
                    price = round(min(price, max(limit, mid)), 6)
                else:
                    price = round(max(price, min(limit, mid)), 6)
                p_base = cfg.p_marketable_limit
                delay = cfg.delay_marketable
                reasons.append("marketable_limit")
            else:
                # passive: we would sit inside/behind the spread. Model a real
                # chance of NOT filling; the price, if it fills, is our limit.
                price = round(limit, 6)
                p_base = cfg.p_passive_limit
                delay = cfg.delay_passive
                reasons.append("passive_limit")

        # ---- fill probability + partial quantity --------------------------
        prob = float(p_base)
        # thin book / wide spread lowers the odds of a clean fill
        if thin:
            prob *= 0.85
            reasons.append("thin_liquidity")
        if mid and spread / mid > cfg.wide_spread_frac:
            prob *= 0.85
            reasons.append("wide_spread")

        # displayed size gates the immediate quantity + whole-order odds
        if displayed is not None and displayed > 0 and displayed < qty:
            prob *= _clamp(displayed / qty, 0.0, 1.0)
            partial = float(displayed)
            reasons.append("size_gated")
        else:
            partial = float(qty)

        prob = round(_clamp(prob, 0.0, 1.0), 6)
        delay = round(delay * (1.15 if thin else 1.0), 6)

        if partial <= 0:
            status = STATUS_PENDING
        elif partial < qty:
            status = STATUS_PARTIAL
        else:
            status = STATUS_FILLED

        return FillEstimate(
            symbol=order.symbol, side=side, order_qty=qty,
            expected_fill_price=price, fill_probability=prob,
            expected_fill_delay=delay, slippage_estimate=slippage,
            liquidity_penalty=round(liq_penalty, 6), partial_qty=partial,
            status=status, mid=mid, touch=round(touch, 6),
            reasons=tuple(reasons),
            notes={"spread": spread, "displayed_size": displayed,
                   "thin": thin, "order_type": order_type})

    # ------------------------------------------------ SimExecutionClient hook
    def as_sim_fill_model(self
                          ) -> Callable[[OrderRequest, Optional[Quote]],
                                        Tuple[Optional[float], float, str]]:
        """Adapt to ``SimExecutionClient``'s ``FillModelFn`` signature so a
        backtest fills using exactly this model. Deterministic: fills the
        expected partial quantity at the expected price; pending -> no fill."""
        def _fn(order: OrderRequest, quote: Optional[Quote]
                ) -> Tuple[Optional[float], float, str]:
            est = self.estimate_fill(order, quote)
            if not est.is_fillable or est.expected_fill_price is None:
                return None, 0.0, STATUS_PENDING
            return est.expected_fill_price, est.partial_qty, est.status
        return _fn


# --------------------------------------------------------------------------- #
# Calibration — learn conservative constants from realized fills (Upgrade 2 / D)
# --------------------------------------------------------------------------- #
# Below this many resolved+filled trades the ledger is too thin to trust, so we
# keep the conservative defaults and stay byte-identical to today (fail-open).
MIN_CALIBRATION_SAMPLES = 20


def calibrated_params_from(records: Optional[List[dict]] = None, *,
                           base: Optional[FillModelConfig] = None,
                           min_samples: int = MIN_CALIBRATION_SAMPLES
                           ) -> FillModelConfig:
    """Return a ``FillModelConfig`` tuned from the execution-calibration ledger.

    Reads the folded calibration records (``oracle.execution.calibration``) and
    nudges the conservative defaults toward what really happened:

      * ``base_slippage`` is raised by the mean *signed* slippage error
        (actual entry - expected entry) so the model stops under-charging the
        entry. Only ever RAISED — calibration never flatters a fill.
      * the fill-probability base rates are scaled DOWN toward the realized
        fill rate when the model was over-optimistic (``fill_rate_bias`` > 0).
        Only ever LOWERED.
      * ``illiquidity_penalty_frac`` is raised when executable_EV over-stated
        realized_EV (``model_capture_ratio`` < 1), capped so it stays sane.

    Contract (the Upgrade-D invariants the tests pin):
      * Fail-open + conservative: junk / ``None`` / fewer than ``min_samples``
        filled trades -> the ``base`` config UNCHANGED, so the model is
        byte-identical to today. Any error is swallowed the same way.
      * Deterministic: identical ``records`` -> identical config.
      * Monotone-pessimistic: every adjustment can only make a fill WORSE.

    ``records`` may be passed directly (already folded) or omitted to load the
    live ledger. ``base`` defaults to the conservative :class:`FillModelConfig`.
    """
    cfg = base or FillModelConfig()
    try:
        from oracle.execution import calibration as _calib
        recs = records if records is not None else _calib.load_records()
        if not isinstance(recs, list) or not recs:
            return cfg
        stats = _calib.compute_calibration(recs)
        n = int(stats.get("n_filled") or 0)
        if n < int(min_samples):
            return cfg  # too thin to trust -> conservative defaults, unchanged

        # ---- slippage: raise base by the mean signed error (increase only) ---
        base_slip = cfg.base_slippage
        slip_err = stats.get("mean_slippage_error")
        if isinstance(slip_err, (int, float)) and slip_err > 0:
            base_slip = round(cfg.base_slippage + float(slip_err), 6)

        # ---- fill probability: scale toward the realized fill rate (down only)
        p_scale = 1.0
        pred = stats.get("predicted_fill_rate")
        act = stats.get("actual_fill_rate")
        if (isinstance(pred, (int, float)) and pred > 0
                and isinstance(act, (int, float))):
            p_scale = _clamp(float(act) / float(pred), 0.0, 1.0)

        # ---- liquidity penalty: raise when the model over-stated realized EV -
        illiq = cfg.illiquidity_penalty_frac
        cap = stats.get("model_capture_ratio")
        if isinstance(cap, (int, float)) and cap < 1.0:
            shortfall = _clamp(1.0 - float(cap), 0.0, 1.0)
            illiq = round(_clamp(cfg.illiquidity_penalty_frac * (1.0 + shortfall),
                                 cfg.illiquidity_penalty_frac, 1.0), 6)

        return replace(
            cfg,
            base_slippage=base_slip,
            illiquidity_penalty_frac=illiq,
            p_market=round(cfg.p_market * p_scale, 6),
            p_marketable_limit=round(cfg.p_marketable_limit * p_scale, 6),
            p_passive_limit=round(cfg.p_passive_limit * p_scale, 6),
            calibrated=True,
            n_samples=n,
        )
    except Exception:  # fail-open: any hiccup -> conservative defaults unchanged
        return base or FillModelConfig()


def _pending(order: OrderRequest, side: str, qty: int, mid, touch, spread,
             slippage, liq_penalty, cfg: FillModelConfig,
             *, reason: str) -> FillEstimate:
    return FillEstimate(
        symbol=order.symbol, side=side, order_qty=qty,
        expected_fill_price=None, fill_probability=0.0,
        expected_fill_delay=cfg.delay_passive, slippage_estimate=slippage,
        liquidity_penalty=round(liq_penalty, 6), partial_qty=0.0,
        status=STATUS_PENDING, mid=mid, touch=round(touch, 6),
        reasons=(reason,), notes={"spread": spread})


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    fm = FillModel()

    tight = Quote("OPT", bid=1.00, ask=1.02, ts="t")   # 2c spread, mid 1.01
    wide = Quote("OPT", bid=1.00, ask=1.40, ts="t")    # 40c spread, mid 1.20

    # Marketable BUY fills at/above the touch and strictly at/above mid.
    b = fm.estimate_fill(OrderRequest("OPT", "buy", 1, order_type="market"),
                         tight)
    if b.expected_fill_price is None or b.expected_fill_price < tight.mid:
        print("FAIL: buy below mid", b); ok = False
    if b.expected_fill_price < tight.ask:
        print("FAIL: market buy should be >= ask", b); ok = False
    if b.status != STATUS_FILLED or b.fill_probability <= 0.9:
        print("FAIL: market fill status/prob", b); ok = False

    # Marketable SELL fills at/below touch and strictly at/below mid.
    s = fm.estimate_fill(OrderRequest("OPT", "sell", 1, order_type="market"),
                         tight)
    if s.expected_fill_price is None or s.expected_fill_price > tight.mid:
        print("FAIL: sell above mid", s); ok = False
    if s.expected_fill_price > tight.bid:
        print("FAIL: market sell should be <= bid", s); ok = False

    # Wider spread -> strictly more slippage than the tight market.
    bw = fm.estimate_fill(OrderRequest("OPT", "buy", 1, order_type="market"),
                          wide)
    if not (bw.slippage_estimate > b.slippage_estimate):
        print("FAIL: wide spread should slip more", bw.slippage_estimate,
              b.slippage_estimate); ok = False

    # Determinism.
    if fm.estimate_fill(OrderRequest("OPT", "buy", 1), tight) != \
            fm.estimate_fill(OrderRequest("OPT", "buy", 1), tight):
        print("FAIL: non-deterministic estimate"); ok = False

    # Thin liquidity adds a penalty and lowers fill probability.
    thin = fm.estimate_fill(OrderRequest("OPT", "buy", 1, order_type="market"),
                            tight, market={"open_interest": 10, "volume": 5})
    if thin.liquidity_penalty <= 0:
        print("FAIL: thin should add liquidity penalty", thin); ok = False
    if not (thin.fill_probability < b.fill_probability):
        print("FAIL: thin should lower fill prob", thin.fill_probability,
              b.fill_probability); ok = False

    # Size larger than the displayed ask_size -> partial fill + size impact.
    part = fm.estimate_fill(OrderRequest("OPT", "buy", 10, order_type="market"),
                            tight, market={"ask_size": 3})
    if part.status != STATUS_PARTIAL or part.partial_qty != 3.0:
        print("FAIL: size-gated partial", part); ok = False
    if not (part.slippage_estimate >= b.slippage_estimate):
        print("FAIL: size impact should not reduce slippage", part); ok = False
    if not (part.fill_probability < b.fill_probability):
        print("FAIL: partial should lower whole-order prob", part); ok = False

    # Non-marketable passive limit BUY (limit below the bid) -> lower prob,
    # fills only at the limit, longer delay.
    pas = fm.estimate_fill(
        OrderRequest("OPT", "buy", 1, order_type="limit", limit_price=0.95),
        tight)
    if pas.expected_fill_price != 0.95:
        print("FAIL: passive fills at limit", pas); ok = False
    if not (pas.fill_probability < b.fill_probability):
        print("FAIL: passive prob should be lower", pas); ok = False
    if pas.expected_fill_delay <= b.expected_fill_delay:
        print("FAIL: passive delay should be longer", pas); ok = False

    # Marketable limit BUY (limit above ask) fills, capped by the limit.
    ml = fm.estimate_fill(
        OrderRequest("OPT", "buy", 1, order_type="limit", limit_price=1.10),
        tight)
    if ml.status != STATUS_FILLED or ml.expected_fill_price > 1.10:
        print("FAIL: marketable limit", ml); ok = False
    if ml.expected_fill_price < tight.mid:
        print("FAIL: marketable limit below mid", ml); ok = False

    # No quote -> pending, zero fill probability, never raises.
    nq = fm.estimate_fill(OrderRequest("OPT", "buy", 1), None)
    if nq.status != STATUS_PENDING or nq.fill_probability != 0.0:
        print("FAIL: no-quote pending", nq); ok = False

    # SimExecutionClient hook: a backtest fills using this model.
    from oracle.execution.client import SimExecutionClient
    sim = SimExecutionClient(quotes={"OPT": tight},
                             fill_model=fm.as_sim_fill_model())
    r = sim.submit_order(OrderRequest("OPT", "buy", 1, order_type="market"))
    if not r.is_filled or r.filled_avg_price is None or \
            r.filled_avg_price < tight.mid:
        print("FAIL: sim hook fill", r); ok = False

    # ---- calibration (Upgrade D): fail-open + conservative + captures reality
    default_cfg = FillModelConfig()
    # Insufficient / junk -> conservative default UNCHANGED (byte-identical).
    for junk in (None, [], [{"bad": 1}], "x", 42):
        if calibrated_params_from(junk, base=default_cfg) != default_cfg:
            print("FAIL: thin/junk calibration should be default", junk); ok = False

    # A reference buy at the tight market and the price the DEFAULT model paid.
    order = OrderRequest("OPT", "buy", 1, order_type="market")
    ref = fm.estimate_fill(order, tight).expected_fill_price   # 1.032
    delta = 0.05                                                # under-charged 5c
    recs = [{
        "trade_id": f"c{i}", "resolution_status": "resolved", "filled": True,
        "fill_probability": 1.0, "expected_entry_price": ref,
        "actual_entry_price": round(ref + delta, 6),
        "theoretical_EV": 15.0, "executable_EV": 11.0, "realized_EV": 11.0,
    } for i in range(MIN_CALIBRATION_SAMPLES)]

    cal = calibrated_params_from(recs, base=default_cfg)
    if not cal.calibrated or cal.n_samples != MIN_CALIBRATION_SAMPLES:
        print("FAIL: calibrated flag/n_samples", cal); ok = False
    # Slippage only ever RISES.
    if cal.base_slippage < default_cfg.base_slippage:
        print("FAIL: calibration lowered slippage", cal); ok = False
    # After calibration the model PREDICTS the realized entry (capture -> 1.0).
    got = FillModel(cal).estimate_fill(order, tight).expected_fill_price
    if got is None or abs(got - (ref + delta)) > 1e-6:
        print("FAIL: calibrated model should predict realized entry", got); ok = False
    # Determinism.
    if calibrated_params_from(recs, base=default_cfg) != cal:
        print("FAIL: non-deterministic calibration"); ok = False

    print("execution.fill_model self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
