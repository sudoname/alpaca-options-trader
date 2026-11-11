"""
Oracle 3.0 — Phase 3: replay exits on a quote path, with the exit TRIGGER and
the exit EXECUTION modeled separately.

Why: live monitoring and prior replay conflated "the rule fired at price X"
with "we sold at price X". In reality a stop observed at X fills at the bid
minus slippage of whatever quote is actually there — which, on a gap, can be
far below the stop level. This module walks an intra-trade option quote series
and reports both truths:

  * trigger  — WHEN and WHY the shared exit policy (`exit_manager.evaluate_exit`,
    the exact function the live scheduler uses) decided to exit, and at what
    observed trigger price;
  * execution — WHAT a deterministic execution model
    (`research_replay.simulate_fill` + `cost_model.CostModel`) says the exit
    actually filled at: sell at bid - slippage, fees paid, depth caps honored,
    partial fills accumulated across subsequent ticks.

Honesty rules (mirrors research_replay):
  1. No lookahead: a stop that fires is taken even if later ticks recover; the
     recovery is reported as a diagnostic flag (`recovered_after_stop`), never
     used to undo the exit.
  2. Gap-through stops fill at the gapped quote, not at the stop level
     (`gap_through=True` when the trigger was observed beyond the level).
  3. Stale quotes (older than the model's `max_quote_age_s`) neither trigger
     nor fill — they are counted and skipped, never trusted.
  4. Ambiguous ticks — where the bid alone implies a loss-side exit while the
     ask alone implies a profit-side exit (very wide quote spanning both
     levels) — are flagged and resolved by an explicit policy
     (default `stop_first`, the pessimistic choice).
  5. Partial exits: depth caps can leave quantity working; remaining size keeps
     trying on later ticks. Un-exited remainder is reported, not hidden.
  6. Forced time exits: if no rule fires by the end of the series, the position
     is closed on the last valid tick (reason `session_end`) unless the caller
     opts out (`force_exit_at_end=False` -> status 'open').
  7. Provenance: every tick carries a label in
     actual > paper > historical > indicative > synthetic; the path result is
     labeled with the WEAKEST provenance consumed, so synthetic data can never
     masquerade as history.

Accounting boundary (Phase 1 invariant — costs charged once): this module
reports EXIT-side economics only. `net_pnl_vs_entry` is proceeds minus the
entry PRICE notional; entry-side slippage/fees are the entry path's job
(`compute_executable_ev` / `research_replay.replay_decision`) and are NOT
re-charged here.

DATA GAP (reported per the master prompt): the research ledger stores exactly
ONE quote per decision (entry time) — no intra-trade option quote series exists
anywhere in the repo today. Until historical option quotes are imported, use
deterministic fixtures (see tests/test_exit_path_replay.py) or the JSONL import
path below.

Import path for real data — one JSON object per line, exact fields:

    {"ts": "2026-11-07T14:31:00+00:00",   # ISO-8601 quote timestamp (required)
     "bid": 1.19, "ask": 1.21,            # executable two-sided quote (required)
     "bid_size": 25, "ask_size": 40,      # displayed contracts (optional)
     "quote_age_s": 0.4,                  # staleness at observation (optional)
     "provenance": "historical"}          # actual|paper|historical|indicative|synthetic

Command (offline, deterministic):

    python -c "import exit_replay as xr; \
ticks = xr.ticks_from_jsonl('quotes/SPY251107C00680000.jsonl'); \
print(xr.replay_exit_path({'entry_price': 1.00}, \
{'stop_loss_percent': 10.0, 'take_profit_percent': 20.0, \
 'trailing_stop_distance': 0.05}, ticks))"

Pure and deterministic: no network, no broker imports, no I/O except
`ticks_from_jsonl`. Same inputs -> same output.
"""

import json
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional

from cost_model import CONTRACT_MULTIPLIER
from exit_manager import evaluate_exit, should_arm_trailing, _parse_dt
from research_replay import ExecutionModel, simulate_fill


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #
#: Strong -> weak. A replay result is labeled with the weakest tick consumed.
PROVENANCE_RANK = {
    "actual": 5, "paper": 4, "historical": 3, "indicative": 2, "synthetic": 1,
}

_LOSS_SIDE = ("stop_loss", "trailing_stop")
_PROFIT_SIDE = ("take_profit",)

#: Deterministic fallback clock when a tick timestamp cannot be parsed, so
#: expiration/stale checks never consult the wall clock during replay.
_EPOCH = datetime(1970, 1, 1)


@dataclass
class QuoteTick:
    """One observation of the option's market. ``ts`` is an ISO-8601 string."""
    ts: str
    bid: Optional[float]
    ask: Optional[float]
    bid_size: Optional[float] = None
    ask_size: Optional[float] = None
    quote_age_s: Optional[float] = None
    provenance: str = "historical"

    def quote(self) -> Dict:
        return {"bid": self.bid, "ask": self.ask, "bid_size": self.bid_size,
                "ask_size": self.ask_size, "quote_age_s": self.quote_age_s}


def ticks_from_jsonl(path: str) -> List[QuoteTick]:
    """Load a quote series from JSONL (fields documented in the module
    docstring). Junk lines are skipped; a missing file returns []. The caller
    is responsible for the series being in chronological order."""
    ticks: List[QuoteTick] = []
    try:
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except (ValueError, TypeError):
                    continue
                if not isinstance(rec, dict) or rec.get("ts") is None:
                    continue
                ticks.append(QuoteTick(
                    ts=str(rec.get("ts")),
                    bid=rec.get("bid"), ask=rec.get("ask"),
                    bid_size=rec.get("bid_size"), ask_size=rec.get("ask_size"),
                    quote_age_s=rec.get("quote_age_s"),
                    provenance=str(rec.get("provenance") or "historical"),
                ))
    except FileNotFoundError:
        return []
    return ticks


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #
def _valid_quote(tick: QuoteTick) -> Optional[Dict]:
    """Executable two-sided quote or None. Crossed/one-sided/non-positive
    quotes are unusable for both triggering and execution."""
    try:
        bid = float(tick.bid) if tick.bid is not None else None
        ask = float(tick.ask) if tick.ask is not None else None
    except (TypeError, ValueError):
        return None
    if bid is None or ask is None or bid <= 0 or ask <= 0 or ask < bid:
        return None
    return {"bid": bid, "ask": ask, "bid_size": tick.bid_size,
            "ask_size": tick.ask_size, "quote_age_s": tick.quote_age_s}


def _is_stale(tick: QuoteTick, model: ExecutionModel) -> bool:
    age = tick.quote_age_s
    if age is None:
        return False
    try:
        return float(age) > model.max_quote_age_s
    except (TypeError, ValueError):
        return False


def _level_price(action: str, entry: float, levels: Dict,
                 highest: float) -> Optional[float]:
    """The price the rule nominally fires at (what an idealized exit would
    assume it exits at). Used only to diagnose gap-through and recovery."""
    if action == "stop_loss":
        return entry * (1.0 - float(levels.get("stop_loss_percent", 10.0)) / 100.0)
    if action == "take_profit":
        return entry * (1.0 + float(levels.get("take_profit_percent", 20.0)) / 100.0)
    if action == "trailing_stop" and highest > 0:
        return highest * (1.0 - float(levels.get("trailing_stop_distance", 0.05)))
    return None


# --------------------------------------------------------------------------- #
# The replay
# --------------------------------------------------------------------------- #
def replay_exit_path(
    trade: Dict,
    levels: Dict,
    ticks: List[QuoteTick],
    *,
    model: Optional[ExecutionModel] = None,
    qty: int = 1,
    trigger_source: str = "mid",          # 'mid' | 'bid' (live bid-triggered stops)
    arm_profit_pct: float = 0.0,
    roll_enabled: bool = False,
    check_expiration: bool = False,
    max_hold_days: Optional[int] = None,
    ambiguity_policy: str = "stop_first",  # 'stop_first' (pessimistic) | 'take_first'
    force_exit_at_end: bool = True,
    multiplier: float = CONTRACT_MULTIPLIER,
) -> Dict:
    """Walk ``ticks`` chronologically for one open long-option position and
    replay the exit — trigger and execution separately.

    ``trade`` needs ``entry_price`` (and optionally ``expiration``,
    ``entry_time``, pre-armed ``trailing_stop_active``/``highest_price``).
    ``levels`` uses `exit_manager` units: stop/take in PERCENT, trailing
    distance a FRACTION. Triggers are decided by ``exit_manager.evaluate_exit``
    on the ``trigger_source`` price; fills are priced by
    ``research_replay.simulate_fill('sell', ...)`` on the triggering tick and,
    while quantity remains, on each later usable tick.

    Returns a dict (see module docstring for semantics):
      status: 'exited' | 'exited_partial' | 'exit_unfilled' | 'open' | 'unreplayable'
      trigger {ts, price, action, reason, level_price, index} | None
      fills [...], filled_qty, remaining_qty, avg_exit_price, proceeds, exit_fees
      flags: ambiguous, gap_through, recovered_after_stop, forced_time_exit
      counts {ticks, valid, stale, invalid}; provenance; scenario
    """
    model = model or ExecutionModel()
    qty = int(qty)
    try:
        entry = float(trade.get("entry_price") or 0)
    except (TypeError, ValueError):
        entry = 0.0

    # Internal mutable state; never mutate the caller's dict.
    state = dict(trade)
    highest = 0.0
    try:
        highest = float(state.get("highest_price") or 0)
    except (TypeError, ValueError):
        highest = 0.0

    counts = {"ticks": len(ticks), "valid": 0, "stale": 0, "invalid": 0}
    prov_rank: Optional[int] = None

    trigger: Optional[Dict] = None
    ambiguous = False
    ambiguity_detail: Optional[str] = None
    gap_through = False
    forced = False
    fills: List[Dict] = []
    remaining = qty
    last_valid_index: Optional[int] = None
    exit_done_index: Optional[int] = None

    def _consume(tick: QuoteTick) -> None:
        nonlocal prov_rank
        r = PROVENANCE_RANK.get(str(tick.provenance), 1)
        prov_rank = r if prov_rank is None else min(prov_rank, r)

    def _attempt_fill(i: int, tick: QuoteTick, quote: Dict) -> None:
        nonlocal remaining, exit_done_index
        res = simulate_fill("sell", quote, remaining, model, multiplier=multiplier)
        if res["status"] in ("filled", "partial") and res["filled_qty"] > 0:
            _consume(tick)
            fills.append({"ts": tick.ts, "index": i,
                          "qty": res["filled_qty"], "price": res["price"],
                          "fees": res["fees"], "cash": res["cash"],
                          "liquidity": res["liquidity"]})
            remaining -= res["filled_qty"]
            if remaining <= 0:
                exit_done_index = i

    if entry <= 0:
        return _result("unreplayable", qty, remaining, trigger, fills, entry,
                       multiplier, counts, prov_rank, model, ambiguous,
                       ambiguity_detail, gap_through, False, forced,
                       reason="missing_entry_price")

    for i, tick in enumerate(ticks):
        if remaining <= 0:
            break
        quote = _valid_quote(tick)
        if quote is None:
            counts["invalid"] += 1
            continue
        if _is_stale(tick, model):
            counts["stale"] += 1
            continue
        counts["valid"] += 1
        last_valid_index = i
        bid, ask = quote["bid"], quote["ask"]
        trig_price = bid if trigger_source == "bid" else (bid + ask) / 2.0
        now = _parse_dt(tick.ts) or _EPOCH

        if trigger is not None:
            # Committed to exiting: keep working the remaining quantity.
            _attempt_fill(i, tick, quote)
            continue

        # Trailing bookkeeping BEFORE the decision (same tick cannot both set
        # the new high and fire off it: cur == highest never trips the stop).
        if not state.get("trailing_stop_active") and should_arm_trailing(
                entry, trig_price, arm_profit_pct):
            state["trailing_stop_active"] = True
        if state.get("trailing_stop_active"):
            highest = max(highest, trig_price)
            state["highest_price"] = highest

        kw = dict(now=now, roll_enabled=roll_enabled,
                  check_expiration=check_expiration, max_hold_days=max_hold_days)
        decision = evaluate_exit(state, trig_price, levels, **kw)

        # Ambiguity: the bid alone says loss-side exit while the ask alone says
        # profit-side exit (quote spans both levels). Resolve explicitly.
        d_bid = evaluate_exit(state, bid, levels, **kw)
        d_ask = evaluate_exit(state, ask, levels, **kw)
        if d_bid.action in _LOSS_SIDE and d_ask.action in _PROFIT_SIDE:
            ambiguous = True
            ambiguity_detail = (f"bid {bid} -> {d_bid.action}, "
                                f"ask {ask} -> {d_ask.action}")
            if ambiguity_policy == "take_first":
                decision, trig_price = d_ask, ask
            else:                                     # pessimistic default
                decision, trig_price = d_bid, bid

        if not decision.should_exit:
            continue

        level = _level_price(decision.action, entry, levels, highest)
        if decision.action in _LOSS_SIDE and level is not None \
                and trig_price < level - 1e-9:
            gap_through = True
        _consume(tick)
        trigger = {"ts": tick.ts, "index": i, "price": round(trig_price, 6),
                   "action": decision.action, "reason": decision.reason,
                   "level_price": round(level, 6) if level is not None else None}
        _attempt_fill(i, tick, quote)

    # Forced time exit: nothing fired, series over, position still open.
    if trigger is None and force_exit_at_end and last_valid_index is not None:
        tick = ticks[last_valid_index]
        quote = _valid_quote(tick)
        bid, ask = quote["bid"], quote["ask"]
        trig_price = bid if trigger_source == "bid" else (bid + ask) / 2.0
        forced = True
        _consume(tick)
        trigger = {"ts": tick.ts, "index": last_valid_index,
                   "price": round(trig_price, 6), "action": "forced_time_exit",
                   "reason": "session_end", "level_price": None}
        _attempt_fill(last_valid_index, tick, quote)

    # Diagnostic only — computed AFTER all exit decisions are frozen, so it can
    # never feed back into them (no lookahead): did the market come back above
    # the level we stopped out at?
    recovered = False
    if (trigger is not None and trigger["action"] in _LOSS_SIDE
            and exit_done_index is not None
            and trigger.get("level_price") is not None):
        for tick in ticks[exit_done_index + 1:]:
            q = _valid_quote(tick)
            if q is None or _is_stale(tick, model):
                continue
            if (q["bid"] + q["ask"]) / 2.0 > trigger["level_price"]:
                recovered = True
                break

    if trigger is None:
        status = "open" if counts["valid"] else "unreplayable"
    elif remaining <= 0:
        status = "exited"
    elif fills:
        status = "exited_partial"
    else:
        status = "exit_unfilled"
    return _result(status, qty, remaining, trigger, fills, entry, multiplier,
                   counts, prov_rank, model, ambiguous, ambiguity_detail,
                   gap_through, recovered, forced)


def _result(status, qty, remaining, trigger, fills, entry, multiplier, counts,
            prov_rank, model, ambiguous, ambiguity_detail, gap_through,
            recovered, forced, reason=None) -> Dict:
    filled_qty = qty - remaining
    proceeds = round(sum(f["cash"] for f in fills), 6)
    exit_fees = round(sum(f["fees"] for f in fills), 6)
    avg_exit = (round(sum(f["price"] * f["qty"] for f in fills) / filled_qty, 6)
                if filled_qty > 0 else None)
    gross_exit_value = round(sum(f["price"] * multiplier * f["qty"]
                                 for f in fills), 6)
    # Exit-side economics only: entry fees/slippage live on the entry path and
    # are charged exactly once there (Phase 1 invariant).
    net_vs_entry = (round(proceeds - entry * multiplier * filled_qty, 6)
                    if filled_qty > 0 else None)
    pnl_pct = (round((avg_exit - entry) / entry * 100.0, 6)
               if (filled_qty > 0 and entry > 0) else None)
    prov = None
    if prov_rank is not None:
        prov = {v: k for k, v in PROVENANCE_RANK.items()}[prov_rank]
    return {
        "status": status, "reason": reason,
        "qty": qty, "filled_qty": filled_qty, "remaining_qty": remaining,
        "trigger": trigger, "fills": fills,
        "avg_exit_price": avg_exit, "proceeds": proceeds, "exit_fees": exit_fees,
        "gross_exit_value": gross_exit_value,
        "net_pnl_vs_entry": net_vs_entry, "pnl_percent": pnl_pct,
        "ambiguous": ambiguous, "ambiguity_detail": ambiguity_detail,
        "gap_through": gap_through, "recovered_after_stop": recovered,
        "forced_time_exit": forced,
        "counts": counts, "provenance": prov, "scenario": model.name,
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network, no files)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    levels = {"stop_loss_percent": 10.0, "take_profit_percent": 20.0,
              "trailing_stop_distance": 0.05}
    trade = {"entry_price": 1.00}

    # Trigger vs execution: take fires at mid 1.22, fills at bid-slip 1.19.
    # (Avoid exact-threshold quotes: mid of 1.19/1.21 is 1.2 which floats just
    # BELOW 20.0% and does not fire — faithful to evaluate_exit semantics.)
    r = replay_exit_path(trade, levels, [
        QuoteTick("2026-01-02T10:00:00", 1.04, 1.06),
        QuoteTick("2026-01-02T10:01:00", 1.21, 1.23),
    ])
    if r["status"] != "exited" or r["trigger"]["action"] != "take_profit":
        print("FAIL: take trigger", r); ok = False
    if abs(r["trigger"]["price"] - 1.22) > 1e-9 or abs(r["avg_exit_price"] - 1.19) > 1e-9:
        print("FAIL: trigger/fill separation", r); ok = False

    # Gap-through stop: fills at the gapped bid, far below the 0.90 level.
    r = replay_exit_path(trade, levels, [
        QuoteTick("t1", 0.99, 1.01),
        QuoteTick("t2", 0.60, 0.62),
        QuoteTick("t3", 0.95, 0.97),   # recovery after the stop
    ])
    if not (r["gap_through"] and r["recovered_after_stop"]
            and r["trigger"]["action"] == "stop_loss"
            and abs(r["avg_exit_price"] - 0.58) < 1e-9):
        print("FAIL: gap-through stop", r); ok = False

    # Stale tick that would have stopped us out is skipped, not trusted.
    r = replay_exit_path(trade, levels, [
        QuoteTick("t1", 0.99, 1.01),
        QuoteTick("t2", 0.60, 0.62, quote_age_s=120.0),
        QuoteTick("t3", 1.00, 1.02),
    ], force_exit_at_end=False)
    if r["status"] != "open" or r["counts"]["stale"] != 1:
        print("FAIL: stale skip", r); ok = False

    # Forced time exit on the last valid tick when nothing fires.
    r = replay_exit_path(trade, levels, [QuoteTick("t1", 1.00, 1.02)])
    if not (r["forced_time_exit"] and r["trigger"]["action"] == "forced_time_exit"):
        print("FAIL: forced exit", r); ok = False

    # Synthetic provenance wins (weakest label propagates).
    r = replay_exit_path(trade, levels, [
        QuoteTick("t1", 1.19, 1.21, provenance="synthetic")])
    if r["provenance"] != "synthetic":
        print("FAIL: provenance", r); ok = False

    print("exit_replay self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
