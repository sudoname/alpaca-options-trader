"""Matched, quote-derived market versus EV-capped limit entry research.

Input: JSONL candidates with decision_id, decision_time (ISO UTC), symbol,
qty, gross_expected_value (total dollars), and levels (exit_manager units).
Quotes: JSONL at <quotes-dir>/<symbol>.jsonl using exit_replay. No broker
orders are placed. Quote touches are only a *scenario*, not proven fills.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import datetime, timedelta, timezone
from typing import Optional

from cost_model import CONTRACT_MULTIPLIER, CostModel
from exit_replay import QuoteTick, replay_exit_path, ticks_from_jsonl
from oracle.execution.limit_policy import compute_entry_limit
from research_replay import ExecutionModel, simulate_fill


def _time(value: str) -> datetime:
    dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        raise ValueError("quote/decision timestamps must include timezone")
    return dt.astimezone(timezone.utc)


def _valid(tick: QuoteTick, model: ExecutionModel) -> bool:
    try:
        return (tick.bid is not None and tick.ask is not None
                and 0 < float(tick.bid) <= float(tick.ask)
                and (tick.quote_age_s is None or
                     float(tick.quote_age_s) <= model.max_quote_age_s))
    except (TypeError, ValueError):
        return False


def compare_candidate(candidate: dict, ticks: list[QuoteTick], *,
                      model: Optional[ExecutionModel] = None,
                      max_gap_s: float = 10.0) -> dict:
    """Full-size entries only; missing quote coverage remains unresolved."""
    model = model or ExecutionModel()
    did = candidate["decision_id"]
    decision = _time(candidate["decision_time"])
    try:
        qty = int(candidate["qty"])
    except (TypeError, ValueError, KeyError):
        return {"decision_id": did, "status": "UNRESOLVED_QUANTITY"}
    if qty < 1:
        return {"decision_id": did, "status": "UNRESOLVED_QUANTITY"}
    series = sorted(((_time(t.ts), t) for t in ticks), key=lambda pair: pair[0])
    prior = [(ts, t) for ts, t in series if ts <= decision and _valid(t, model)]
    dq = candidate.get("decision_quote") or {}
    if dq.get("bid") is not None and dq.get("ask") is not None:
        frozen = QuoteTick(ts=decision.isoformat(), bid=dq["bid"], ask=dq["ask"],
                           bid_size=dq.get("bid_size"), ask_size=dq.get("ask_size"),
                           quote_age_s=dq.get("quote_age_s"), provenance="decision_snapshot")
        if _valid(frozen, model):
            prior.append((decision, frozen))
    prior.sort(key=lambda pair: pair[0])
    if not prior or (decision - prior[-1][0]).total_seconds() > model.max_quote_age_s:
        return {"decision_id": did, "status": "UNRESOLVED_ENTRY_QUOTE"}
    quote = prior[-1][1]
    cap = compute_entry_limit(gross_ev=candidate.get("gross_expected_value"),
                              qty=qty, bid=quote.bid, ask=quote.ask,
                              quote_age_s=quote.quote_age_s,
                              cost_model=CostModel(model.effective_cost()),
                              max_quote_age_s=model.max_quote_age_s,
                              expiry_s=float(candidate.get("limit_expiry_s", 30)))
    arrival = decision + timedelta(milliseconds=model.latency_ms)
    expiry = decision + timedelta(seconds=cap.expiry_s)
    arriving = [(ts, t) for ts, t in series if ts >= arrival]
    if not arriving or (arriving[0][0] - arrival).total_seconds() > max_gap_s:
        return {"decision_id": did, "status": "UNRESOLVED_ARRIVAL_QUOTE"}
    market_tick = arriving[0][1]
    market = simulate_fill("buy", market_tick.quote(), qty, model) if _valid(market_tick, model) else {"status": "unreplayable"}

    def exit_result(ts, entry_price, entry_cash):
        future = [t for tick_time, t in arriving if tick_time > ts]
        result = replay_exit_path({"entry_price": entry_price, "entry_time": ts.isoformat()},
                                  candidate["levels"], future, model=model,
                                  qty=qty, trigger_source="bid",
                                  force_exit_at_end=False)
        return {"exit_status": result["status"], "exit": result,
                "after_cost_pnl": round(entry_cash + result["proceeds"], 6)
                if result["status"] == "exited" else None}

    market_result = {"entry": market, "exit_status": None, "after_cost_pnl": None}
    if market["status"] == "filled":
        market_result.update(exit_result(arriving[0][0], market["price"], market["cash"]))
    limit_result = {"cap": cap.limit_price, "cap_status": cap.status,
                    "entry": None, "exit_status": None, "after_cost_pnl": None}
    if cap.status == "ok":
        within = [(ts, t) for ts, t in arriving if ts <= expiry]
        touch = next(((ts, t) for ts, t in within if _valid(t, model)
                      and float(t.ask) <= cap.limit_price
                      and t.ask_size is not None
                      and float(t.ask_size) * model.max_participation >= qty), None)
        if touch:
            ts, t = touch
            fee = (model.cost.occ_fee_per_contract + model.cost.commission_per_contract) * qty
            entry = {"status": "quote_touch_scenario", "ts": t.ts,
                     "price": cap.limit_price, "filled_qty": qty,
                     "fees": fee, "cash": -(cap.limit_price * CONTRACT_MULTIPLIER * qty + fee)}
            limit_result["entry"] = entry
            limit_result.update(exit_result(ts, cap.limit_price, entry["cash"]))
        else:
            obs = [ts for ts, t in series if arrival <= ts <= expiry
                   and _valid(t, model)]
            if series[-1][0] >= expiry and obs and max(
                    [(obs[0] - arrival).total_seconds(),
                     *[(b - a).total_seconds() for a, b in zip(obs, obs[1:])],
                     (expiry - obs[-1]).total_seconds()]) <= max_gap_s:
                limit_result["entry"] = {"status": "missed_trade", "cash": 0.0}
                limit_result["after_cost_pnl"] = 0.0
            else:
                limit_result["entry"] = {"status": "unresolved_quote_coverage"}
    return {"decision_id": did, "status": "MODELLED_QUOTE_SCENARIO",
            "market": market_result, "limit": limit_result,
            "paired_after_cost_delta": round(limit_result["after_cost_pnl"] - market_result["after_cost_pnl"], 6)
            if market_result["after_cost_pnl"] is not None and limit_result["after_cost_pnl"] is not None else None,
            "note": "A quote touch does not prove a limit order would have filled; exits require observed triggers."}


def run_study(candidates_path: str, quotes_dir: str) -> dict:
    if not os.path.isfile(candidates_path) or not os.path.isdir(quotes_dir):
        return {"status": "INSUFFICIENT_DATA", "missing": [
            name for name, available in (("candidates", os.path.isfile(candidates_path)),
                                    ("quotes_dir", os.path.isdir(quotes_dir))) if not available]}
    with open(candidates_path, encoding="utf-8") as fh:
        candidates = [json.loads(line) for line in fh if line.strip()]
    if not candidates:
        return {"status": "INSUFFICIENT_DATA", "missing": ["candidate_rows"]}
    rows = []
    for c in candidates:
        symbol = str(c["symbol"])
        if os.path.basename(symbol) != symbol or symbol in (".", ".."):
            raise ValueError("invalid symbol filename")
        rows.append(compare_candidate(c, ticks_from_jsonl(os.path.join(quotes_dir, symbol + ".jsonl"))))
    paired = [r for r in rows if r.get("paired_after_cost_delta") is not None]
    return {"status": "MODELLED_QUOTE_SCENARIO", "n_candidates": len(rows),
            "n_paired_resolved": len(paired), "n_missing_or_unresolved": len(rows)-len(paired),
            "market_pnl_on_paired": round(sum(r["market"]["after_cost_pnl"] for r in paired), 6),
            "limit_pnl_on_paired": round(sum(r["limit"]["after_cost_pnl"] for r in paired), 6),
            "rows": rows}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--quotes-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    result = run_study(args.candidates, args.quotes_dir)
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps({k: v for k, v in result.items() if k != "rows"}, indent=2))
    return 0 if result["status"] != "INSUFFICIENT_DATA" else 2


if __name__ == "__main__":
    raise SystemExit(main())
