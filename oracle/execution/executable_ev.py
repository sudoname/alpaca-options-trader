"""
Oracle Execution — Upgrade 2C: theoretical vs executable vs realized EV.

Oracle already computes a *theoretical* EV for a candidate — probability-weighted
payoff priced at fair value / mid (``ev_engine``). But you never trade at the
mid. This module makes the gap explicit by producing THREE numbers for the same
contract:

  * theoretical_EV — the model's edge priced at fair value (what the screener
    ranks on). Unchanged; sourced from the contract / ``ev_engine``.
  * executable_EV  — the edge AFTER realistic execution: entry above mid, exit
    below mid (Upgrade-2A ``fill_model``), broker fees (``cost_model``), and the
    probability the order fills at all. This is the number a shadow/paper run
    should actually believe.
  * realized_EV    — computed post-hoc from the ACTUAL fills once a trade closes
    (Upgrade-2D telemetry feeds this back to calibrate the two above).

The point of the split (per the spec): a candidate can be +theoretical but
−executable once the spread / slippage / fill-probability are paid — and that is
a *successful* rejection, not a missed trade. Nothing here places an order or
gates a trade; it only computes. The live gate consumes it flag-gated and
shadow-first in Upgrade 2D.

Design principles (same as the rest of Oracle execution):
  * CONSERVATIVE. Execution frictions come from the 2A model, which never
    flatters us. Unknown fill probability is treated as certain-fill only when
    the order is marketable; otherwise it degrades the number.
  * REUSE, don't duplicate. Frictions = ``fill_model``; fees = ``cost_model``;
    theoretical EV = whatever the contract already carries (``ev_engine`` output).
  * DETERMINISTIC + FAIL-OPEN. Same inputs -> same numbers; missing microstructure
    degrades to a quote-only estimate; junk never raises.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping, Optional

from cost_model import CONTRACT_MULTIPLIER, CostModel
from oracle.execution.client import OrderRequest, Quote
from oracle.execution.fill_model import FillEstimate, FillModel

_EPS = 1e-9


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def _first(mapping: Mapping[str, Any], *names: str) -> Optional[float]:
    """First numeric value among ``names`` in a contract-like mapping."""
    if not isinstance(mapping, Mapping):
        return None
    for n in names:
        v = _num(mapping.get(n))
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class ExecutableEV:
    """The three-EV view of one candidate contract. ``execution_risk`` is the
    fraction of the theoretical edge consumed by realistic execution (0 = free,
    1 = the entire edge is eaten, >1 = execution turns the edge negative)."""

    symbol: str
    theoretical_fair_value: Optional[float]
    displayed_bid: Optional[float]
    displayed_ask: Optional[float]
    mid_price: Optional[float]
    expected_entry_price: Optional[float]
    expected_exit_price: Optional[float]
    spread_cost: float                 # $ round-trip from crossing the spread
    slippage_estimate: float           # $ per contract (entry, from 2A)
    fill_probability: float
    expected_fill_delay: float
    liquidity_penalty: float           # $ per contract (from 2A)
    fees: float                        # $ round-trip broker/OCC fees
    execution_cost: float              # $ total round-trip friction
    theoretical_EV: Optional[float]
    executable_EV: Optional[float]
    execution_risk: Optional[float]
    qty: int = 1
    status: str = "ok"
    reasons: tuple = ()
    notes: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d

    @property
    def is_executable_positive(self) -> bool:
        return self.executable_EV is not None and self.executable_EV > 0


# --------------------------------------------------------------------------- #
# Theoretical / executable
# --------------------------------------------------------------------------- #
def compute_executable_ev(contract: Mapping[str, Any],
                          fair_value: Optional[float],
                          quote: Optional[Quote], *,
                          fill_model: Optional[FillModel] = None,
                          cost_model: Optional[CostModel] = None,
                          qty: int = 1,
                          theoretical_ev: Optional[float] = None,
                          market: Optional[Mapping[str, Any]] = None,
                          order_type: str = "market",
                          limit_price: Optional[float] = None) -> ExecutableEV:
    """Turn a candidate ``contract`` + ``quote`` into the theoretical/executable
    EV split. Never raises.

    ``theoretical_EV`` is taken from ``theoretical_ev`` if given, else the
    contract's ``expected_value`` (``ev_engine`` output). Execution frictions are
    modeled as a round trip: pay ``expected_entry_price`` (>= mid) to open and
    receive ``expected_exit_price`` (<= mid) to close AT the entry-time quote —
    a conservative same-quote round trip — plus broker/OCC fees. The whole thing
    is then scaled by ``fill_probability`` because an order that does not fill
    earns nothing.
    """
    fm = fill_model or FillModel()
    cm = cost_model or CostModel()
    qty = max(1, int(qty or 1))
    sym = str((contract or {}).get("symbol") or (quote.symbol if quote else "") or "")
    reasons = []

    bid = _num(quote.bid) if quote else None
    ask = _num(quote.ask) if quote else None
    mid = quote.mid if quote else None
    if mid is None and bid is not None and ask is not None:
        mid = round((bid + ask) / 2.0, 6)

    theo = theoretical_ev if theoretical_ev is not None else \
        _first(contract or {}, "expected_value", "theoretical_EV", "ev")
    if fair_value is None:
        fair_value = _first(contract or {}, "fair_value", "theoretical_fair_value")
        if fair_value is None:
            fair_value = mid

    # No usable quote -> executable EV is undefined (can't model a fill).
    if bid is None and ask is None:
        return ExecutableEV(
            symbol=sym, theoretical_fair_value=fair_value,
            displayed_bid=bid, displayed_ask=ask, mid_price=mid,
            expected_entry_price=None, expected_exit_price=None,
            spread_cost=0.0, slippage_estimate=0.0, fill_probability=0.0,
            expected_fill_delay=0.0, liquidity_penalty=0.0, fees=0.0,
            execution_cost=0.0, theoretical_EV=theo, executable_EV=None,
            execution_risk=None, qty=qty, status="no_quote",
            reasons=("no_quote",))

    # ---- entry fill (BUY): conservative price >= mid --------------------- #
    entry_order = OrderRequest(sym or "OPT", "buy", qty, order_type=order_type,
                               limit_price=limit_price)
    entry: FillEstimate = fm.estimate_fill(entry_order, quote, market=market)
    # ---- exit fill (SELL) at the same quote: conservative price <= mid --- #
    exit_order = OrderRequest(sym or "OPT", "sell", qty, order_type="market")
    exit_: FillEstimate = fm.estimate_fill(exit_order, quote, market=market)

    entry_px = entry.expected_fill_price
    exit_px = exit_.expected_fill_price

    # ---- round-trip friction in dollars ---------------------------------- #
    # spread crossing + both-side slippage is captured by (entry-mid)+(mid-exit)
    # priced per contract in option points -> dollars via the multiplier.
    entry_pen = max(0.0, (entry_px - mid)) if (entry_px is not None and mid is not None) else 0.0
    exit_pen = max(0.0, (mid - exit_px)) if (exit_px is not None and mid is not None) else 0.0
    spread_cost = round((entry_pen + exit_pen) * CONTRACT_MULTIPLIER * qty, 6)

    # broker/OCC fees, both sides, from the shared cost model.
    fcfg = cm.config
    fees = round(2.0 * (fcfg.occ_fee_per_contract + fcfg.commission_per_contract) * qty, 6)

    # liquidity penalty (already inside entry price above; surface it too).
    liq = round(entry.liquidity_penalty * CONTRACT_MULTIPLIER * qty, 6)
    execution_cost = round(spread_cost + fees, 6)

    fill_prob = float(entry.fill_probability)
    if fill_prob < 1.0:
        reasons.append("fill_probability<1")

    # ---- executable EV --------------------------------------------------- #
    executable = execution_risk = None
    if theo is not None:
        edge_after_cost = theo - execution_cost
        # An order that does not fill earns nothing (no position, no edge).
        executable = round(fill_prob * edge_after_cost, 6)
        denom = abs(theo) if abs(theo) > _EPS else _EPS
        # fraction of the theoretical edge consumed by execution (incl. no-fill)
        execution_risk = round(1.0 - (executable / theo), 6) if abs(theo) > _EPS \
            else None
        if executable <= 0 < (theo or 0):
            reasons.append("executable_negative")
    else:
        reasons.append("no_theoretical_ev")

    return ExecutableEV(
        symbol=sym,
        theoretical_fair_value=round(fair_value, 6) if fair_value is not None else None,
        displayed_bid=bid, displayed_ask=ask,
        mid_price=round(mid, 6) if mid is not None else None,
        expected_entry_price=entry_px, expected_exit_price=exit_px,
        spread_cost=spread_cost, slippage_estimate=entry.slippage_estimate,
        fill_probability=round(fill_prob, 6),
        expected_fill_delay=entry.expected_fill_delay,
        liquidity_penalty=liq, fees=fees, execution_cost=execution_cost,
        theoretical_EV=round(theo, 6) if theo is not None else None,
        executable_EV=executable, execution_risk=execution_risk,
        qty=qty, status="ok", reasons=tuple(reasons),
        notes={"entry_reasons": list(entry.reasons),
               "exit_reasons": list(exit_.reasons)})


# --------------------------------------------------------------------------- #
# Realized (post-hoc, from actual fills)
# --------------------------------------------------------------------------- #
def compute_realized_ev(entry_fill_price: float, exit_fill_price: float, *,
                        qty: int = 1,
                        cost_model: Optional[CostModel] = None) -> Optional[float]:
    """Realized $ P&L of a closed long-option round trip from ACTUAL fills.

    This is the ground-truth ``realized_EV`` the calibration loop compares
    against ``theoretical_EV`` / ``executable_EV``. Fees come from the shared
    cost model (fills are already the real prices, so no slippage is re-applied).
    """
    ep = _num(entry_fill_price)
    xp = _num(exit_fill_price)
    if ep is None or xp is None:
        return None
    qty = max(1, int(qty or 1))
    cm = cost_model or CostModel()
    fcfg = cm.config
    fees = 2.0 * (fcfg.occ_fee_per_contract + fcfg.commission_per_contract) * qty
    gross = (xp - ep) * CONTRACT_MULTIPLIER * qty
    return round(gross - fees, 6)


def capture_ratios(theoretical_ev: Optional[float],
                   executable_ev: Optional[float],
                   realized_ev: Optional[float] = None) -> dict:
    """The EV-degradation ladder used in the calibration report:

      execution_capture_ratio = executable / theoretical  (how much survives model->exec)
      realized_capture_ratio  = realized  / theoretical    (how much survives model->reality)
      model_capture_ratio     = realized  / executable     (did the exec model predict reality)
    """
    def _ratio(num, den):
        n, d = _num(num), _num(den)
        if n is None or d is None or abs(d) <= _EPS:
            return None
        return round(n / d, 6)

    return {
        "execution_capture_ratio": _ratio(executable_ev, theoretical_ev),
        "realized_capture_ratio": _ratio(realized_ev, theoretical_ev),
        "model_capture_ratio": _ratio(realized_ev, executable_ev),
    }


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    fm = FillModel()
    cm = CostModel()

    tight = Quote("OPT", bid=1.00, ask=1.02, ts="t")   # 2c spread, mid 1.01
    wide = Quote("OPT", bid=1.00, ask=1.40, ts="t")    # 40c spread, mid 1.20

    contract = {"symbol": "OPT", "expected_value": 15.0,
                "probability_of_profit": 0.55}

    # Tight market: executable EV is below theoretical but stays positive.
    t = compute_executable_ev(contract, fair_value=1.01, quote=tight,
                              fill_model=fm, cost_model=cm)
    if t.theoretical_EV != 15.0:
        print("FAIL: theoretical passthrough", t); ok = False
    if t.executable_EV is None or not (t.executable_EV < t.theoretical_EV):
        print("FAIL: executable should be below theoretical", t); ok = False
    if not (t.executable_EV > 0):
        print("FAIL: tight market should stay positive", t); ok = False
    if not (0.0 < (t.execution_risk or 0) < 1.0):
        print("FAIL: execution_risk in (0,1) for tight", t.execution_risk); ok = False
    if t.expected_entry_price is None or t.expected_entry_price < t.mid_price:
        print("FAIL: entry below mid", t); ok = False
    if t.expected_exit_price is None or t.expected_exit_price > t.mid_price:
        print("FAIL: exit above mid", t); ok = False

    # Wider spread strictly lowers executable EV (more friction).
    w = compute_executable_ev(contract, fair_value=1.20, quote=wide,
                              fill_model=fm, cost_model=cm)
    if not (w.executable_EV < t.executable_EV):
        print("FAIL: wide spread should lower executable EV",
              w.executable_EV, t.executable_EV); ok = False
    if not (w.spread_cost > t.spread_cost):
        print("FAIL: wide spread should cost more", w.spread_cost, t.spread_cost)
        ok = False

    # A thin, small theoretical edge on a wide market flips to −executable ->
    # this is a SUCCESSFUL rejection (the whole point of the split).
    thin_edge = {"symbol": "OPT", "expected_value": 3.0}
    flip = compute_executable_ev(thin_edge, fair_value=1.20, quote=wide,
                                 fill_model=fm, cost_model=cm)
    if flip.executable_EV is None or flip.executable_EV > 0:
        print("FAIL: +theoretical/−executable should reject", flip); ok = False
    if "executable_negative" not in flip.reasons:
        print("FAIL: expected executable_negative reason", flip.reasons); ok = False
    if not (flip.execution_risk is None or flip.execution_risk > 1.0):
        print("FAIL: execution_risk should exceed 1 when edge flips",
              flip.execution_risk); ok = False

    # Lower fill probability (passive limit far from market) lowers executable EV.
    passive = compute_executable_ev(
        contract, fair_value=1.01, quote=tight, fill_model=fm, cost_model=cm,
        order_type="limit", limit_price=0.95)
    if not (passive.fill_probability < t.fill_probability):
        print("FAIL: passive should lower fill prob", passive.fill_probability)
        ok = False
    if not (passive.executable_EV < t.executable_EV):
        print("FAIL: lower fill prob should lower executable EV", passive); ok = False

    # No quote -> undefined executable EV, no raise.
    nq = compute_executable_ev(contract, fair_value=None, quote=None,
                               fill_model=fm, cost_model=cm)
    if nq.status != "no_quote" or nq.executable_EV is not None:
        print("FAIL: no-quote handling", nq); ok = False

    # Determinism.
    if compute_executable_ev(contract, 1.01, tight, fill_model=fm) != \
            compute_executable_ev(contract, 1.01, tight, fill_model=fm):
        print("FAIL: non-deterministic executable EV"); ok = False

    # Realized EV from actual fills + capture ratios.
    realized = compute_realized_ev(1.04, 1.20, qty=1, cost_model=cm)
    if realized is None or realized <= 0:
        print("FAIL: winning round trip should realize > 0", realized); ok = False
    losing = compute_realized_ev(1.04, 0.90, qty=1, cost_model=cm)
    if losing is None or losing >= 0:
        print("FAIL: losing round trip should realize < 0", losing); ok = False

    ratios = capture_ratios(15.0, t.executable_EV, realized)
    if ratios["execution_capture_ratio"] is None or \
            not (0.0 < ratios["execution_capture_ratio"] < 1.0):
        print("FAIL: execution_capture_ratio", ratios); ok = False
    if ratios["realized_capture_ratio"] is None:
        print("FAIL: realized_capture_ratio should compute", ratios); ok = False
    # None-safe when a leg is missing.
    if capture_ratios(None, None, None)["execution_capture_ratio"] is not None:
        print("FAIL: None inputs should yield None ratio"); ok = False

    print("execution.executable_ev self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
