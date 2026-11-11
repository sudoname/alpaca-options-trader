"""
Oracle Execution — EV-derived buy-limit entry policy (opt-in variant).

The current baseline enters with MARKET orders: it pays whatever the ask (plus
modeled slippage) happens to be. This module derives the highest acceptable
BUY LIMIT from the expected edge after costs and a configured EV floor:

    the most we can pay per contract such that, if filled exactly at that
    limit, the trade's net EV (gross strategy EV minus round-trip execution
    costs) still clears the floor.

Single-charge accounting, consistent with ``executable_ev``:

    buy_basis          = CostModel's modeled market entry price (ask + slippage,
                         after the min-spread-floor adjustment)
    net_ev_at_market   = gross_ev − round_trip_cost(bid, ask, qty)   # the
                         market-entry baseline this variant is compared against
    L_cap(raw)         = buy_basis + (net_ev_at_market − ev_floor) / (100 × qty)
    limit_price        = L_cap rounded DOWN to the valid option tick

Down-rounding is the conservative direction: filling exactly at the limit then
guarantees ``net_ev_at_limit >= ev_floor`` (worst case; any price improvement
only helps). ``net_ev_at_limit`` is that worst-case number:

    net_ev_at_limit = net_ev_at_market + (buy_basis − limit_price) × 100 × qty

Respecting the spec:
  * valid option tick sizes — penny-quoted names tick $0.01; classic grid is
    $0.05 below $3.00 premium and $0.10 at/above (``option_tick``);
  * quote age — ``max_quote_age_s`` mirrors executable_ev's ``stale_quote``;
  * existing spread/size constraints — ``max_spread_pct`` and qty validation;
  * conservative rounding — always DOWN to the tick;
  * explicit expiry — ``expiry_s`` stamped on the result for the order's TIF
    window (the lifecycle layer cancels at expiry; this module only computes);
  * repricing stays within the cap — ``reprice_within_cap``;
  * reason logging — ``status='no_cap'`` always carries a reason code naming
    exactly why a valid cap could not be formed.

Pure, deterministic, fail-open: junk inputs yield a reason-coded ``no_cap``,
never an exception. Nothing here places an order or gates a trade.
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Optional

from cost_model import CONTRACT_MULTIPLIER, CostModel

_EPS = 1e-9

# Reason codes (mirror executable_ev where the concept is shared)
R_INVALID_QTY = "invalid_qty"
R_EV_UNAVAILABLE = "ev_unavailable"
R_NO_QUOTE = "no_quote"
R_NONPOSITIVE_QUOTE = "nonpositive_quote"
R_ONE_SIDED_QUOTE = "one_sided_quote"
R_CROSSED_QUOTE = "crossed_quote"
R_STALE_QUOTE = "stale_quote"
R_SPREAD_TOO_WIDE = "spread_too_wide"
R_CAP_NONPOSITIVE = "cap_nonpositive"


def _num(x: Any) -> Optional[float]:
    if x is None or isinstance(x, bool):
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return f if f == f and f not in (float("inf"), float("-inf")) else None


def gross_ev_from_contract(contract: Any) -> Optional[float]:
    """GROSS strategy EV ($ total) from a contract/stamp-like mapping.

    Prefers the explicit v3 ``gross_expected_value``; reconstructs it from the
    legacy NET ``expected_value`` + ``round_trip_costs``; else falls back to
    the NET figure alone as a conservative LOWER bound (NET <= GROSS, so a
    smaller gross only lowers the cap — it can never overpay)."""
    if not hasattr(contract, "get"):
        return None
    g = _num(contract.get("gross_expected_value"))
    if g is not None:
        return g
    net = _num(contract.get("expected_value"))
    rt = _num(contract.get("round_trip_costs"))
    if net is not None and rt is not None:
        return net + rt
    return net


# --------------------------------------------------------------------------- #
# Tick helpers
# --------------------------------------------------------------------------- #
def option_tick(price: float, penny_quoted: bool = True) -> float:
    """Minimum price increment for a US option at ``price`` premium.

    Penny-quoted names (SPY/QQQ/most liquid classes under the penny pilot /
    its successor) tick $0.01 at any premium. The classic grid is $0.05 below
    a $3.00 premium and $0.10 at or above it.
    """
    if penny_quoted:
        return 0.01
    p = _num(price)
    return 0.05 if (p is None or p < 3.0) else 0.10


def round_down_to_tick(price: float, tick: float) -> float:
    """Largest tick multiple <= ``price`` (conservative for a buy limit)."""
    if tick <= 0:
        return float(price)
    # +EPS guards float dust so an exact multiple is not knocked down a tick.
    steps = math.floor(price / tick + 1e-9)
    # round() kills binary-float dust like 1.1300000000000001
    return round(steps * tick, 10)


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class EntryLimit:
    """The EV-capped buy-limit for one candidate, or a reason-coded ``no_cap``.

    ``net_ev_at_market`` is the current market-entry baseline (what the
    existing policy expects to make); ``net_ev_at_limit`` is the worst-case
    net EV if the limit fills exactly at ``limit_price``. By construction
    ``net_ev_at_limit >= ev_floor`` whenever ``status == 'ok'``.
    """

    status: str                          # "ok" | "no_cap"
    reasons: tuple = ()
    limit_price: Optional[float] = None  # the cap, tick-rounded DOWN
    tick: Optional[float] = None
    marketable: Optional[bool] = None    # cap >= displayed ask?
    expiry_s: float = 30.0               # explicit order lifetime
    qty: int = 1
    bid: Optional[float] = None
    ask: Optional[float] = None
    buy_basis: Optional[float] = None    # modeled market fill (ask+slip, floored spread)
    gross_ev: Optional[float] = None     # strategy EV before execution costs ($ total)
    execution_cost: Optional[float] = None  # modeled round-trip friction ($ total)
    ev_floor: float = 0.0
    net_ev_at_market: Optional[float] = None  # gross − cost: the baseline comparison
    net_ev_at_limit: Optional[float] = None   # worst case filled at the cap

    def to_dict(self) -> dict:
        d = asdict(self)
        d["reasons"] = list(self.reasons)
        return d

    @property
    def is_ok(self) -> bool:
        return self.status == "ok" and self.limit_price is not None


def _no_cap(reason: str, *, qty: int = 1, bid=None, ask=None, gross_ev=None,
            ev_floor: float = 0.0, expiry_s: float = 30.0,
            execution_cost=None, net_ev_at_market=None,
            buy_basis=None, tick=None) -> EntryLimit:
    return EntryLimit(status="no_cap", reasons=(reason,), limit_price=None,
                      tick=tick, marketable=None, expiry_s=expiry_s, qty=qty,
                      bid=bid, ask=ask, buy_basis=buy_basis, gross_ev=gross_ev,
                      execution_cost=execution_cost, ev_floor=ev_floor,
                      net_ev_at_market=net_ev_at_market, net_ev_at_limit=None)


# --------------------------------------------------------------------------- #
# Core
# --------------------------------------------------------------------------- #
def compute_entry_limit(*,
                        gross_ev: Optional[float],
                        qty: int,
                        bid: Optional[float],
                        ask: Optional[float],
                        quote_age_s: Optional[float] = None,
                        cost_model: Optional[CostModel] = None,
                        ev_floor: float = 0.0,
                        tick: Optional[float] = None,
                        penny_quoted: bool = True,
                        max_quote_age_s: Optional[float] = None,
                        max_spread_pct: Optional[float] = None,
                        expiry_s: float = 30.0) -> EntryLimit:
    """Highest acceptable buy limit for a long-option entry. Never raises.

    ``gross_ev`` MUST be the GROSS strategy EV in total dollars (multiplier and
    quantity applied, NO execution costs subtracted — same contract as
    ``compute_executable_ev``'s ``theoretical_ev``); this function charges the
    round trip itself, exactly once. ``ev_floor`` is in the same total-dollar
    units. ``tick`` overrides ``option_tick`` when the instrument's increment
    is known explicitly.
    """
    cm = cost_model or CostModel()
    ev_floor = _num(ev_floor) or 0.0
    expiry_s = _num(expiry_s) or 30.0

    # ---- size -------------------------------------------------------------
    try:
        q = int(qty)
    except (TypeError, ValueError):
        q = 0
    if q < 1:
        return _no_cap(R_INVALID_QTY, qty=max(q, 0), bid=_num(bid), ask=_num(ask),
                       gross_ev=_num(gross_ev), ev_floor=ev_floor, expiry_s=expiry_s)

    # ---- EV ---------------------------------------------------------------
    gev = _num(gross_ev)
    if gev is None:
        return _no_cap(R_EV_UNAVAILABLE, qty=q, bid=_num(bid), ask=_num(ask),
                       ev_floor=ev_floor, expiry_s=expiry_s)

    # ---- quote (same ladder as executable_ev) -----------------------------
    b, a = _num(bid), _num(ask)
    if b is None and a is None:
        return _no_cap(R_NO_QUOTE, qty=q, gross_ev=gev, ev_floor=ev_floor,
                       expiry_s=expiry_s)
    if (a is not None and a <= 0) or (b is not None and b < 0):
        return _no_cap(R_NONPOSITIVE_QUOTE, qty=q, bid=b, ask=a, gross_ev=gev,
                       ev_floor=ev_floor, expiry_s=expiry_s)
    if b is None or a is None:
        return _no_cap(R_ONE_SIDED_QUOTE, qty=q, bid=b, ask=a, gross_ev=gev,
                       ev_floor=ev_floor, expiry_s=expiry_s)
    if b > a + _EPS:
        return _no_cap(R_CROSSED_QUOTE, qty=q, bid=b, ask=a, gross_ev=gev,
                       ev_floor=ev_floor, expiry_s=expiry_s)
    if max_quote_age_s is not None:
        age = _num(quote_age_s)
        if age is not None and age > float(max_quote_age_s):
            return _no_cap(R_STALE_QUOTE, qty=q, bid=b, ask=a, gross_ev=gev,
                           ev_floor=ev_floor, expiry_s=expiry_s)

    # ---- spread constraint -------------------------------------------------
    mid = (a + b) / 2.0
    if max_spread_pct is not None and mid > _EPS:
        spread_pct = (a - b) / mid * 100.0
        if spread_pct > float(max_spread_pct) + _EPS:
            return _no_cap(R_SPREAD_TOO_WIDE, qty=q, bid=b, ask=a, gross_ev=gev,
                           ev_floor=ev_floor, expiry_s=expiry_s)

    # ---- cap math (single-charge, shared cost model) -----------------------
    rt = cm.round_trip_cost(b, a, qty=q)
    execution_cost = round(float(rt["cost_dollars"]), 6)
    buy_basis = round(float(rt["entry_price"]), 6)
    net_at_market = round(gev - execution_cost, 6)

    per_dollar = CONTRACT_MULTIPLIER * q
    l_raw = buy_basis + (net_at_market - ev_floor) / per_dollar

    tk = _num(tick)
    if tk is None or tk <= 0:
        tk = option_tick(l_raw, penny_quoted=penny_quoted)
    cap = round_down_to_tick(l_raw, tk)

    if cap <= 0:
        return _no_cap(R_CAP_NONPOSITIVE, qty=q, bid=b, ask=a, gross_ev=gev,
                       ev_floor=ev_floor, expiry_s=expiry_s,
                       execution_cost=execution_cost,
                       net_ev_at_market=net_at_market,
                       buy_basis=buy_basis, tick=tk)

    net_at_limit = round(net_at_market + (buy_basis - cap) * per_dollar, 6)
    return EntryLimit(
        status="ok", reasons=(), limit_price=cap, tick=tk,
        marketable=bool(cap >= a - _EPS), expiry_s=expiry_s, qty=q,
        bid=b, ask=a, buy_basis=buy_basis, gross_ev=round(gev, 6),
        execution_cost=execution_cost, ev_floor=ev_floor,
        net_ev_at_market=net_at_market, net_ev_at_limit=net_at_limit)


# --------------------------------------------------------------------------- #
# Repricing
# --------------------------------------------------------------------------- #
def reprice_within_cap(entry_limit: EntryLimit,
                       desired_price: float) -> Optional[float]:
    """Clamp a requested reprice to the EV cap, tick-rounded DOWN.

    Returns the permitted limit price, or None when no valid cap exists (or
    the clamped price would be non-positive). Repricing can chase the market
    upward but NEVER beyond ``entry_limit.limit_price`` — the EV floor stays
    guaranteed for any fill at or below the cap.
    """
    if entry_limit is None or not entry_limit.is_ok:
        return None
    d = _num(desired_price)
    if d is None:
        return None
    tk = entry_limit.tick or 0.01
    clamped = round_down_to_tick(min(d, entry_limit.limit_price), tk)
    return clamped if clamped > 0 else None


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:  # pragma: no cover - exercised via __main__
    from cost_model import CostConfig
    ok = True

    def check(cond, msg, *detail):
        nonlocal ok
        if not cond:
            print("FAIL:", msg, *detail)
            ok = False

    # --- hand math with a zero-friction config (only the spread is paid) ----
    free = CostModel(CostConfig(slippage_per_contract=0.0,
                                occ_fee_per_contract=0.0,
                                commission_per_contract=0.0))
    # bid 1.00 / ask 1.02, qty 1: basis 1.02, round trip = $2. gross 10 ->
    # net@market 8 -> cap = 1.02 + 0.08 = 1.10 (exact tick), net@limit = floor.
    el = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                             cost_model=free)
    check(el.status == "ok" and abs(el.limit_price - 1.10) < 1e-9,
          "hand cap 1.10", el.limit_price)
    check(el.marketable is True, "1.10 >= ask is marketable", el)
    check(abs(el.net_ev_at_market - 8.0) < 1e-6, "net@market 8", el)
    check(abs(el.net_ev_at_limit - 0.0) < 1e-6, "net@limit == floor", el)
    check(abs(el.buy_basis - 1.02) < 1e-9, "basis = ask (no slippage)", el)

    # gross 7.5 -> raw 1.075 -> rounds DOWN to 1.07, net@limit 0.5 >= floor.
    el2 = compute_entry_limit(gross_ev=7.5, qty=1, bid=1.00, ask=1.02,
                              cost_model=free)
    check(abs(el2.limit_price - 1.07) < 1e-9, "down-round 1.075 -> 1.07", el2.limit_price)
    check(abs(el2.net_ev_at_limit - 0.5) < 1e-6, "conservative rounding adds EV", el2)
    check(el2.net_ev_at_limit >= el2.ev_floor - 1e-9, "floor guaranteed", el2)

    # EV floor shifts the cap down: floor 5 -> cap = 1.02 + 0.03 = 1.05.
    el3 = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.00, ask=1.02,
                              cost_model=free, ev_floor=5.0)
    check(abs(el3.limit_price - 1.05) < 1e-9, "floor shifts cap", el3.limit_price)
    check(el3.net_ev_at_limit >= 5.0 - 1e-9, "floor respected", el3)

    # Thin edge -> cap below the ask: resting (non-marketable) but valid.
    el4 = compute_entry_limit(gross_ev=1.0, qty=1, bid=1.00, ask=1.02,
                              cost_model=free)
    # net@market = -1 -> raw = 1.02 - 0.01 = 1.01 < ask
    check(el4.status == "ok" and abs(el4.limit_price - 1.01) < 1e-9,
          "thin edge caps below ask", el4.limit_price)
    check(el4.marketable is False, "below-ask cap is non-marketable", el4)
    check(el4.net_ev_at_limit >= 0.0 - 1e-9, "resting cap still clears floor", el4)

    # --- default (realistic) cost config ------------------------------------
    cm = CostModel()  # slip .02, occ .02 each side
    # basis 1.04, cost = 104.02 - 97.98 = 6.04; gross 15 -> net 8.96
    # raw = 1.04 + .0896 = 1.1296 -> 1.12; net@limit = 8.96 + (1.04-1.12)*100 = .96
    el5 = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                              cost_model=cm)
    check(abs(el5.buy_basis - 1.04) < 1e-9, "basis includes slippage", el5.buy_basis)
    check(abs(el5.execution_cost - 6.04) < 1e-6, "round-trip cost 6.04", el5.execution_cost)
    check(abs(el5.limit_price - 1.12) < 1e-9, "realistic cap 1.12", el5.limit_price)
    check(abs(el5.net_ev_at_limit - 0.96) < 1e-6, "net@limit .96", el5.net_ev_at_limit)

    # Per-contract scale invariance: qty 2 with 2x gross -> same cap.
    el6 = compute_entry_limit(gross_ev=30.0, qty=2, bid=1.00, ask=1.02,
                              cost_model=cm)
    check(abs(el6.limit_price - el5.limit_price) < 1e-9, "qty scale invariance",
          el6.limit_price, el5.limit_price)
    check(abs(el6.net_ev_at_limit - 2 * el5.net_ev_at_limit) < 1e-6,
          "net@limit scales with qty", el6.net_ev_at_limit)

    # --- tick grids ---------------------------------------------------------
    check(option_tick(1.50, penny_quoted=True) == 0.01, "penny tick")
    check(option_tick(2.99, penny_quoted=False) == 0.05, "nickel tick < $3")
    check(option_tick(3.00, penny_quoted=False) == 0.10, "dime tick >= $3")
    # raw 1.1296 on nickel grid -> 1.10
    el7 = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                              cost_model=cm, penny_quoted=False)
    check(abs(el7.limit_price - 1.10) < 1e-9, "nickel grid rounds to 1.10",
          el7.limit_price)
    # explicit tick override wins
    el8 = compute_entry_limit(gross_ev=15.0, qty=1, bid=1.00, ask=1.02,
                              cost_model=cm, tick=0.25)
    check(abs(el8.limit_price - 1.00) < 1e-9, "explicit tick override",
          el8.limit_price)
    check(abs(round_down_to_tick(1.10, 0.01) - 1.10) < 1e-12,
          "exact multiple not knocked down a tick")

    # --- every no-cap reason is coded ---------------------------------------
    cases = {
        R_INVALID_QTY: dict(gross_ev=10.0, qty=0, bid=1.0, ask=1.02),
        R_EV_UNAVAILABLE: dict(gross_ev=None, qty=1, bid=1.0, ask=1.02),
        R_NO_QUOTE: dict(gross_ev=10.0, qty=1, bid=None, ask=None),
        R_NONPOSITIVE_QUOTE: dict(gross_ev=10.0, qty=1, bid=0.0, ask=0.0),
        R_ONE_SIDED_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.0, ask=None),
        R_CROSSED_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.10, ask=1.02),
        R_STALE_QUOTE: dict(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                            quote_age_s=12.0, max_quote_age_s=5.0),
        R_SPREAD_TOO_WIDE: dict(gross_ev=50.0, qty=1, bid=1.00, ask=1.40,
                                max_spread_pct=20.0),
        R_CAP_NONPOSITIVE: dict(gross_ev=-500.0, qty=1, bid=1.0, ask=1.02),
    }
    for reason, kw in cases.items():
        r = compute_entry_limit(cost_model=free, **kw)
        check(r.status == "no_cap" and reason in r.reasons,
              f"reason-coded no_cap {reason}", r.status, r.reasons)
        check(r.limit_price is None, f"{reason} yields no price", r.limit_price)

    # fresh quote passes the age cap
    fresh = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                                cost_model=free, quote_age_s=2.0,
                                max_quote_age_s=5.0)
    check(fresh.status == "ok", "fresh quote passes age cap", fresh.status)

    # --- repricing stays within the cap -------------------------------------
    check(reprice_within_cap(el5, 1.25) == el5.limit_price,
          "reprice clamped to cap", reprice_within_cap(el5, 1.25))
    check(abs(reprice_within_cap(el5, 1.1049) - 1.10) < 1e-9,
          "reprice tick-rounds down")
    check(reprice_within_cap(el5, 1.05) == 1.05, "reprice below cap passes")
    nocap = compute_entry_limit(gross_ev=None, qty=1, bid=1.0, ask=1.02)
    check(reprice_within_cap(nocap, 1.05) is None, "no cap -> no reprice")
    check(reprice_within_cap(el5, None) is None, "junk desired -> None")

    # --- explicit expiry + determinism + fail-open --------------------------
    elx = compute_entry_limit(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                              cost_model=free, expiry_s=45.0)
    check(elx.expiry_s == 45.0, "expiry stamped", elx.expiry_s)
    check(compute_entry_limit(gross_ev=10.0, qty=1, bid=1.0, ask=1.02,
                              cost_model=free) == el, "deterministic")
    junk = compute_entry_limit(gross_ev="x", qty="y", bid=object(), ask="z")
    check(junk.status == "no_cap", "junk never raises", junk.status)

    # to_dict round trip
    d = el5.to_dict()
    check(isinstance(d["reasons"], list) and d["limit_price"] == el5.limit_price,
          "to_dict", d)

    # --- gross-EV sourcing chain --------------------------------------------
    check(gross_ev_from_contract({"gross_expected_value": 12.5,
                                  "expected_value": 6.0}) == 12.5,
          "v3 gross field preferred")
    check(gross_ev_from_contract({"expected_value": 6.0,
                                  "round_trip_costs": 6.04}) == 12.04,
          "net + costs reconstruction")
    check(gross_ev_from_contract({"expected_value": 6.0}) == 6.0,
          "net-only conservative lower bound")
    check(gross_ev_from_contract({}) is None, "empty mapping -> None")
    check(gross_ev_from_contract(None) is None, "non-mapping -> None")
    check(gross_ev_from_contract({"gross_expected_value": "nan"}) is None,
          "junk gross -> None")

    print("execution.limit_policy self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
