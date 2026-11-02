"""
Deterministic replay + execution-cost evaluation over the research ledger.

Reconstructs what a decision saw (the immutable snapshot) and what an order
would have done under an explicit, configurable execution model. Rules:

  1. EXACT RECORDED REPLAY ONLY uses data frozen in the snapshot at decision
     time. Nothing is fetched, interpolated, or fabricated: a snapshot without
     an executable bid/ask is reported as ``unreplayable``, never guessed.
  2. HISTORICAL SIMULATION (mode='historical_sim') is only entered when the
     caller explicitly injects a ``quote_fn``; results are labeled so they can
     never be confused with exact replay.
  3. The execution model prices fills off the executable side of the book
     (buy = ask + slippage, sell = bid - slippage, via cost_model semantics),
     pays per-contract fees, honors the actual contract multiplier, caps size
     against displayed depth, and can leave orders partial or unfilled. The
     spread is paid exactly once per side — replay never double-counts it.
  4. Adverse scenarios (wider spreads, heavier slippage, staler quotes,
     thinner books) are first-class: the same decisions re-priced under
     stress, deterministically.
  5. STRUCTURAL ISOLATION: this module never imports a broker client or
     ``requests``; ``ReplayBroker`` exposes read-only ledger views and raises
     ``ReplayIsolationError`` on ANY undefined attribute (submit_order,
     place_order, ...). Research commands cannot submit orders through it.

Honesty note (documented limitation): a modeled fill — and even an Alpaca
paper fill — is evidence of a *plausible* execution, not proof of live
executability. Treat replay P/L as an estimate bounded by the scenario grid.
"""

import json
from dataclasses import dataclass, field, replace
from typing import Callable, Dict, List, Optional

from cost_model import CONTRACT_MULTIPLIER, CostConfig, CostModel


# --------------------------------------------------------------------------- #
# Structural isolation
# --------------------------------------------------------------------------- #
class ReplayIsolationError(RuntimeError):
    """Raised when replay code touches anything resembling order flow."""


class ReplayBroker:
    """Read-only ledger facade. There is no order-submission code path here:
    every attribute that is not an explicitly defined read method raises."""

    _ALLOWED = ("get_decision", "snapshots", "events_for_order", "stats")

    def __init__(self, ledger):
        object.__setattr__(self, "_ledger", ledger)

    def get_decision(self, decision_id):
        return self._ledger.get_decision(decision_id)

    def snapshots(self, **kw):
        return self._ledger.snapshots(**kw)

    def events_for_order(self, broker_order_id):
        return self._ledger.events_for_order(broker_order_id)

    def stats(self):
        return self._ledger.stats()

    def __getattr__(self, name):
        raise ReplayIsolationError(
            f"ReplayBroker has no '{name}': replay is read-only and cannot "
            f"submit, cancel, or modify orders (allowed: {self._ALLOWED})")

    def __setattr__(self, name, value):
        raise ReplayIsolationError("ReplayBroker is immutable")


# --------------------------------------------------------------------------- #
# Execution model
# --------------------------------------------------------------------------- #
@dataclass
class ExecutionModel:
    """Explicit assumptions used to price a replayed order. Deterministic:
    the same inputs always produce the same fill."""
    name: str = "base"
    latency_ms: float = 250.0            # decision -> order arrival (documentation; no price drift modeled)
    max_quote_age_s: float = 60.0        # quotes staler than this do not fill
    extra_slippage_per_contract: float = 0.0   # adverse add-on over cost-model slippage
    spread_widen_frac: float = 0.0       # widen half-spread around mid by this fraction
    max_participation: float = 1.0       # max fraction of displayed size we may take
    allow_partial: bool = True           # False -> all-or-none against depth cap
    cost: CostConfig = field(default_factory=CostConfig)

    def effective_cost(self) -> CostConfig:
        return replace(self.cost,
                       slippage_per_contract=self.cost.slippage_per_contract
                       + self.extra_slippage_per_contract)


#: Standard stress grid. Same decisions, harsher micro-structure assumptions.
ADVERSE_SCENARIOS: Dict[str, ExecutionModel] = {
    "base": ExecutionModel(name="base"),
    "wide_spread": ExecutionModel(name="wide_spread", spread_widen_frac=0.5),
    "heavy_slippage": ExecutionModel(name="heavy_slippage",
                                     extra_slippage_per_contract=0.05),
    "stale_quotes": ExecutionModel(name="stale_quotes", max_quote_age_s=5.0),
    "thin_book": ExecutionModel(name="thin_book", max_participation=0.25),
}


def _widen(bid: float, ask: float, frac: float):
    """Symmetrically widen the spread around the mid (mid preserved)."""
    if not frac:
        return bid, ask
    mid = (bid + ask) / 2.0
    half = max(0.0, (ask - bid) / 2.0) * (1.0 + frac)
    return max(0.0, mid - half), mid + half


def simulate_fill(side: str, quote: Optional[Dict], qty: int,
                  model: ExecutionModel,
                  multiplier: float = CONTRACT_MULTIPLIER) -> Dict:
    """Price one side of an order against a recorded quote.

    Returns a dict with ``status`` in:
      unreplayable - no executable two-sided quote was recorded (never guessed)
      unfilled     - quote stale, or depth cap leaves nothing (all-or-none)
      partial      - depth cap reduced size (allow_partial)
      filled       - full size at the modeled price

    ``liquidity`` reports whether depth was 'observed' (sizes recorded) or
    'unobserved' (no sizes; depth cap not applied, flagged honestly).
    """
    q = quote or {}
    bid, ask = q.get("bid"), q.get("ask")
    try:
        bid = float(bid) if bid is not None else None
        ask = float(ask) if ask is not None else None
    except (TypeError, ValueError):
        bid = ask = None
    if bid is None or ask is None or bid < 0 or ask <= 0:
        return {"status": "unreplayable", "filled_qty": 0, "price": None,
                "fees": 0.0, "cash": 0.0, "liquidity": "missing",
                "reason": "no_executable_quote_recorded"}

    age = q.get("quote_age_s")
    if age is not None and float(age) > model.max_quote_age_s:
        return {"status": "unfilled", "filled_qty": 0, "price": None,
                "fees": 0.0, "cash": 0.0, "liquidity": "stale",
                "reason": f"quote_age_{age}s>max_{model.max_quote_age_s}s"}

    # Liquidity cap against the displayed size on the side we take.
    depth = q.get("ask_size") if side == "buy" else q.get("bid_size")
    liquidity = "unobserved"
    fill_qty = int(qty)
    if depth is not None:
        liquidity = "observed"
        cap = int(float(depth) * model.max_participation)
        if cap < fill_qty:
            if not model.allow_partial or cap <= 0:
                return {"status": "unfilled", "filled_qty": 0, "price": None,
                        "fees": 0.0, "cash": 0.0, "liquidity": liquidity,
                        "reason": f"depth_cap_{cap}<qty_{qty}"}
            fill_qty = cap

    w_bid, w_ask = _widen(bid, ask, model.spread_widen_frac)
    cm = CostModel(model.effective_cost())
    base = cm.estimate_fill(side, w_bid, w_ask, qty=fill_qty)
    # Re-derive dollars with the ACTUAL contract multiplier.
    notional = base["price"] * multiplier * fill_qty
    cash = -(notional + base["fees"]) if side == "buy" else (notional - base["fees"])
    return {
        "status": "filled" if fill_qty == int(qty) else "partial",
        "filled_qty": fill_qty,
        "price": base["price"],
        "fees": base["fees"],
        "notional": notional,
        "cash": cash,
        "liquidity": liquidity,
        "reason": None,
    }


def execution_shortfall(modeled_price, actual_price, side: str = "buy") -> Optional[float]:
    """Signed $/contract difference, positive = reality was WORSE than model
    (paid more on a buy, received less on a sell)."""
    try:
        m, a = float(modeled_price), float(actual_price)
    except (TypeError, ValueError):
        return None
    return (a - m) if side == "buy" else (m - a)


# --------------------------------------------------------------------------- #
# Per-decision replay
# --------------------------------------------------------------------------- #
def _snapshot_quote(snap: Dict) -> Dict:
    return {
        "bid": snap.get("quote_bid"),
        "ask": snap.get("quote_ask"),
        "bid_size": snap.get("quote_bid_size"),
        "ask_size": snap.get("quote_ask_size"),
        "quote_age_s": snap.get("quote_age_s"),
    }


def _first_event(events: List[Dict], *types) -> Optional[Dict]:
    for e in events:
        if e.get("event_type") in types:
            return e
    return None


def replay_decision(record: Dict, model: ExecutionModel,
                    quote_fn: Optional[Callable[[Dict], Optional[Dict]]] = None,
                    qty: Optional[int] = None) -> Dict:
    """Replay one ledger decision (output of ``ResearchLedger.get_decision``).

    mode='exact_recorded' when the entry quote comes from the frozen snapshot;
    mode='historical_sim' when a caller-supplied ``quote_fn(snapshot)`` had to
    provide it. No quote from either source -> status='unreplayable'.

    Exit pricing, in order of honesty:
      * an observed exit/fill-to-close price from the lifecycle events is used
        directly (fees still paid; 'observed');
      * otherwise, if ``quote_fn`` yields an exit quote, the model sells at
        bid - slippage ('modeled');
      * otherwise the replay reports entry-side costs only ('none').
    """
    snaps = record.get("snapshots") or []
    events = record.get("events") or []
    if not snaps:
        return {"decision_id": record.get("decision_id"),
                "status": "unreplayable", "reason": "no_snapshot"}
    snap = snaps[0]
    did = snap.get("decision_id")
    mult = float(snap.get("multiplier") or CONTRACT_MULTIPLIER)
    want_qty = qty
    if want_qty is None:
        try:
            want_qty = int((json.loads(snap.get("sizing_json") or "{}") or {}).get("qty") or 1)
        except (TypeError, ValueError):
            want_qty = 1

    quote = _snapshot_quote(snap)
    mode = "exact_recorded"
    if (quote.get("bid") is None or quote.get("ask") is None) and quote_fn is not None:
        fetched = None
        try:
            fetched = quote_fn(snap)
        except Exception:
            fetched = None
        if fetched:
            quote, mode = fetched, "historical_sim"

    entry = simulate_fill("buy", quote, want_qty, model, multiplier=mult)
    result = {
        "decision_id": did,
        "symbol": snap.get("symbol"),
        "mode": mode,
        "scenario": model.name,
        "status": entry["status"],
        "entry": entry,
        "exit": None,
        "exit_source": "none",
        "net_pnl_dollars": None,
        "net_pnl_pct": None,
        "shortfall_entry": None,
    }
    if entry["status"] in ("unreplayable", "unfilled"):
        result["reason"] = entry.get("reason")
        return result

    # Execution shortfall vs the ACTUAL recorded fill, when one exists.
    actual_fill = _first_event(events, "fill", "partial_fill")
    if actual_fill and actual_fill.get("price") is not None:
        result["shortfall_entry"] = execution_shortfall(
            entry["price"], actual_fill["price"], side="buy")

    # Exit: observed event price first; modeled quote only if explicitly given.
    exit_ev = _first_event(events, "exit")
    fees_cfg = model.effective_cost()
    per_side_fees = (fees_cfg.occ_fee_per_contract
                     + fees_cfg.commission_per_contract) * entry["filled_qty"]
    if exit_ev and exit_ev.get("price") is not None:
        px = float(exit_ev["price"])
        proceeds = px * mult * entry["filled_qty"] - per_side_fees
        result["exit"] = {"price": px, "fees": per_side_fees,
                          "cash": proceeds, "filled_qty": entry["filled_qty"]}
        result["exit_source"] = "observed"
    elif quote_fn is not None:
        xq = None
        try:
            xq = quote_fn(snap)
        except Exception:
            xq = None
        if xq:
            x = simulate_fill("sell", xq, entry["filled_qty"], model, multiplier=mult)
            if x["status"] in ("filled", "partial"):
                result["exit"] = x
                result["exit_source"] = "modeled"

    if result["exit"] is not None:
        outlay = -entry["cash"]
        proceeds = result["exit"]["cash"]
        net = proceeds - outlay
        result["net_pnl_dollars"] = net
        result["net_pnl_pct"] = (net / outlay * 100.0) if outlay else None
    return result


# --------------------------------------------------------------------------- #
# Cohort replay + scenarios
# --------------------------------------------------------------------------- #
def replay_cohort(ledger, model: ExecutionModel,
                  since: Optional[str] = None, until: Optional[str] = None,
                  accepted: Optional[int] = 1,
                  quote_fn: Optional[Callable] = None,
                  kind: str = "candidate") -> Dict:
    """Replay every matching decision; aggregate honestly (counts by status,
    replayable-only P/L, shortfall stats). Deterministic given the ledger.
    Shadow snapshots are excluded unless ``kind`` is overridden — a cohort is
    one strategy's decisions, never a mix."""
    snaps = ledger.snapshots(since=since, until=until, accepted=accepted,
                             kind=kind)
    results, by_status = [], {}
    pnl, shortfalls = [], []
    for s in snaps:
        rec = ledger.get_decision(s["decision_id"])
        r = replay_decision(rec, model, quote_fn=quote_fn)
        results.append(r)
        by_status[r["status"]] = by_status.get(r["status"], 0) + 1
        if r.get("net_pnl_pct") is not None:
            pnl.append(r["net_pnl_pct"])
        if r.get("shortfall_entry") is not None:
            shortfalls.append(r["shortfall_entry"])
    n = len(results)
    summary = {
        "scenario": model.name,
        "decisions": n,
        "by_status": by_status,
        "fill_rate": (by_status.get("filled", 0) + by_status.get("partial", 0)) / n if n else None,
        "with_pnl": len(pnl),
        "avg_net_pnl_pct": sum(pnl) / len(pnl) if pnl else None,
        "avg_entry_shortfall": sum(shortfalls) / len(shortfalls) if shortfalls else None,
        "shortfall_samples": len(shortfalls),
    }
    return {"summary": summary, "results": results}


def run_scenarios(ledger, scenarios: Optional[Dict[str, ExecutionModel]] = None,
                  **cohort_kw) -> Dict[str, Dict]:
    """Replay the same cohort under each scenario; returns name -> summary."""
    out = {}
    for name, model in (scenarios or ADVERSE_SCENARIOS).items():
        out[name] = replay_cohort(ledger, model, **cohort_kw)["summary"]
    return out


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from research_ledger import ResearchLedger

    ok = True
    led = ResearchLedger(":memory:")

    # --- isolation: the broker facade cannot submit anything -------------- #
    rb = ReplayBroker(led)
    for bad in ("submit_order", "place_order", "cancel_order", "create_order"):
        try:
            getattr(rb, bad)
            print(f"FAIL: ReplayBroker.{bad} must raise"); ok = False
        except ReplayIsolationError:
            pass
    try:
        rb.x = 1
        print("FAIL: ReplayBroker must be immutable"); ok = False
    except ReplayIsolationError:
        pass
    if rb.stats()["snapshots"] != 0:
        print("FAIL: read path should work"); ok = False

    # --- fill mechanics ---------------------------------------------------- #
    base = ExecutionModel(cost=CostConfig(slippage_per_contract=0.02,
                                          occ_fee_per_contract=0.02,
                                          commission_per_contract=0.0))
    q = {"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 40, "quote_age_s": 2.0}

    f = simulate_fill("buy", q, 2, base)
    if f["status"] != "filled" or abs(f["price"] - 1.12) > 1e-9:
        print("FAIL: buy = ask + slippage", f); ok = False
    if abs(f["notional"] - 1.12 * 100 * 2) > 1e-9:
        print("FAIL: multiplier applied", f); ok = False

    # Missing quote is unreplayable — never fabricated.
    if simulate_fill("buy", {"bid": None, "ask": None}, 1, base)["status"] != "unreplayable":
        print("FAIL: missing quote must be unreplayable"); ok = False
    if simulate_fill("buy", None, 1, base)["status"] != "unreplayable":
        print("FAIL: absent quote must be unreplayable"); ok = False

    # Stale quote does not fill.
    stale = dict(q, quote_age_s=120.0)
    if simulate_fill("buy", stale, 1, base)["status"] != "unfilled":
        print("FAIL: stale quote must not fill"); ok = False

    # Depth cap -> partial (allow_partial) or unfilled (all-or-none).
    thin = ExecutionModel(max_participation=0.25, cost=base.cost)  # 40*0.25=10
    p = simulate_fill("buy", q, 20, thin)
    if p["status"] != "partial" or p["filled_qty"] != 10:
        print("FAIL: depth cap should partial-fill 10", p); ok = False
    aon = ExecutionModel(max_participation=0.25, allow_partial=False, cost=base.cost)
    if simulate_fill("buy", q, 20, aon)["status"] != "unfilled":
        print("FAIL: all-or-none should go unfilled", ); ok = False

    # Unknown depth is honest: fills, but flagged 'unobserved'.
    nodepth = {"bid": 1.00, "ask": 1.10}
    nd = simulate_fill("buy", nodepth, 5, thin)
    if nd["status"] != "filled" or nd["liquidity"] != "unobserved":
        print("FAIL: unknown depth must fill flagged unobserved", nd); ok = False

    # Adverse widening keeps the mid, worsens the buy price.
    wide = ExecutionModel(spread_widen_frac=1.0, cost=base.cost)  # 1.00/1.10 -> 0.95/1.15
    wf = simulate_fill("buy", q, 1, wide)
    if abs(wf["price"] - 1.17) > 1e-9:
        print("FAIL: widened buy should be 1.15+0.02", wf["price"]); ok = False

    # No spread double-counting: modeled buy-then-sell on one quote matches
    # cost_model.round_trip_cost exactly.
    buy = simulate_fill("buy", q, 1, base)
    sell = simulate_fill("sell", q, 1, base)
    rt = CostModel(base.cost).round_trip_cost(1.00, 1.10, qty=1)
    modeled_cost = (-buy["cash"]) - sell["cash"]
    if abs(modeled_cost - rt["cost_dollars"]) > 1e-9:
        print("FAIL: spread must be paid once per side",
              modeled_cost, rt["cost_dollars"]); ok = False

    # Shortfall sign: paying more than modeled is positive (worse).
    if not (execution_shortfall(1.12, 1.15, "buy") > 0 >
            execution_shortfall(1.12, 1.10, "buy")):
        print("FAIL: shortfall sign convention"); ok = False

    # --- full-decision replay against the ledger --------------------------- #
    did1 = led.record_snapshot(
        decision_id="r-1", accepted=True, symbol="SPY261219C00500000",
        quote={"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 40,
               "quote_age_s": 2.0},
        underlying_price=498.0, direction="CALL", sizing={"qty": 2})
    led.record_event(decision_id=did1, event_type="submitted",
                     broker_order_id="o1", qty=2, price=1.10)
    led.record_event(decision_id=did1, event_type="fill", broker_order_id="o1",
                     qty=2, price=1.15, status="filled")
    led.record_event(decision_id=did1, event_type="exit", broker_order_id="o2",
                     qty=2, price=1.40)

    r1 = replay_decision(led.get_decision(did1), base)
    if r1["mode"] != "exact_recorded" or r1["status"] != "filled":
        print("FAIL: r-1 should replay exactly", r1["mode"], r1["status"]); ok = False
    if abs(r1["shortfall_entry"] - (1.15 - 1.12)) > 1e-9:
        print("FAIL: entry shortfall vs actual fill", r1["shortfall_entry"]); ok = False
    if r1["exit_source"] != "observed" or r1["net_pnl_pct"] is None:
        print("FAIL: observed exit should produce net P/L", r1); ok = False
    # Net check: outlay = 2*(1.12*100)+0.04; proceeds = 2*(1.40*100)-0.04.
    outlay = 1.12 * 200 + 0.04
    proceeds = 1.40 * 200 - 0.04
    if abs(r1["net_pnl_dollars"] - (proceeds - outlay)) > 1e-6:
        print("FAIL: net dollars", r1["net_pnl_dollars"]); ok = False

    # Snapshot with no quote: unreplayable without quote_fn, historical_sim with.
    did2 = led.record_snapshot(decision_id="r-2", accepted=True,
                               symbol="QQQ261219P00400000", direction="PUT",
                               sizing={"qty": 1})
    r2 = replay_decision(led.get_decision(did2), base)
    if r2["status"] != "unreplayable":
        print("FAIL: quoteless snapshot must be unreplayable", r2); ok = False
    r2b = replay_decision(led.get_decision(did2), base,
                          quote_fn=lambda s: {"bid": 2.00, "ask": 2.10})
    if r2b["mode"] != "historical_sim" or r2b["status"] != "filled":
        print("FAIL: injected quote must be labeled historical_sim", r2b); ok = False

    # Cohort + scenarios: deterministic, honest counts.
    c = replay_cohort(led, base)
    if c["summary"]["decisions"] != 2 or c["summary"]["by_status"].get("unreplayable") != 1:
        print("FAIL: cohort counts", c["summary"]); ok = False
    c2 = replay_cohort(led, base)
    if json.dumps(c["summary"], sort_keys=True) != json.dumps(c2["summary"], sort_keys=True):
        print("FAIL: replay must be deterministic"); ok = False
    sc = run_scenarios(led)
    if set(sc) != set(ADVERSE_SCENARIOS):
        print("FAIL: scenario grid", list(sc)); ok = False
    if sc["stale_quotes"]["by_status"].get("filled", 0) != 1:
        # age 2.0s < 5s cap, so r-1 still fills under stale_quotes
        print("FAIL: stale scenario", sc["stale_quotes"]); ok = False

    led.close()
    print("research_replay self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
