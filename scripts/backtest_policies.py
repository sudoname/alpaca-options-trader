#!/usr/bin/env python
"""Policy backtest over the episode store — read-only, research-side.

Replays every resolved decision in episodes.db and compares entry policies:

  actual     what the system really did (gates on) — realized P&L
  trade_all  take every candidate (gates off) — realized + counterfactual P&L
  qtable     take a candidate iff the Q-table's best action at the episode's
             state matches the candidate's direction
  qtable_pos same, but additionally require Q(best) > 0

Each SKIP row carries a *counterfactual* net_pnl_pct (what the skipped
contract would have returned), so policies that take trades the live system
blocked can still be scored.

Assumptions (printed with the report):
  * equal notional per trade (default $1,000), no compounding, no concurrency
    or capital limits;
  * counterfactual P&L is mark-derived — real fills would be worse;
  * net_pnl_pct is in percent units (e.g. 24.4 == +24.4%).

Usage:
  python scripts/backtest_policies.py --db server_data/2026-11-19/episodes.db \
      --qtable rl_qtable.json [--notional 1000] [--json out.json]
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import defaultdict


# ---------------------------------------------------------------- data loading

def load_episodes(db_path: str, since: str | None = None,
                  until: str | None = None) -> list[dict]:
    db = sqlite3.connect(db_path)
    db.row_factory = sqlite3.Row
    where = ["net_pnl_pct IS NOT NULL"]
    params: list[str] = []
    if since:
        where.append("as_of >= ?"); params.append(since)
    if until:
        where.append("as_of < ?"); params.append(until)
    rows = db.execute(
        "SELECT as_of, symbol, underlying, strat, chosen_action, rule_action,"
        "       net_pnl_pct, outcome, mode, gate_json, features_json"
        f" FROM episodes WHERE {' AND '.join(where)} ORDER BY as_of",
        params,
    ).fetchall()
    db.close()
    eps = []
    for r in rows:
        try:
            gate = json.loads(r["gate_json"]) if r["gate_json"] else {}
        except (TypeError, ValueError):
            gate = {}
        try:
            feats = json.loads(r["features_json"]) if r["features_json"] else {}
        except (TypeError, ValueError):
            feats = {}
        ev = feats.get("evidence") or {}
        chosen = r["chosen_action"]
        candidate = chosen if chosen in ("CALL", "PUT") else r["rule_action"]
        if candidate not in ("CALL", "PUT"):
            continue  # no candidate direction -> nothing to score
        # live episodes store dow=na, but the Q-table trainer recomputes the
        # weekday from the timestamp — substitute it so lookups line up.
        state = gate.get("state_key")
        if state and "dow=na" in state and r["as_of"]:
            try:
                from datetime import datetime
                dow = datetime.fromisoformat(str(r["as_of"])[:19]).weekday()
                state = state.replace("dow=na", f"dow={dow}")
            except ValueError:
                pass
        eps.append({
            "as_of": r["as_of"],
            "underlying": r["underlying"] or r["symbol"],
            "chosen": chosen,
            "candidate": candidate,
            "pnl": float(r["net_pnl_pct"]),
            "counterfactual": chosen == "SKIP",
            "state_key": state,
            "state_key_raw": gate.get("state_key"),
            "regime": ev.get("regime_label"),
            "outcome": r["outcome"],
        })
    return eps


ACTIONS = ("SKIP", "CALL", "PUT")


def load_qtable(path: str) -> dict:
    """Return {state_key: (best_action, best_q)} from a flat 'state|ACTION' table."""
    with open(path, encoding="utf-8") as fh:
        raw = json.load(fh)
    table = raw.get("q", raw)
    per_state: dict[str, dict[str, float]] = defaultdict(dict)
    for key, val in table.items():
        state, _, action = key.rpartition("|")
        if action in ACTIONS and isinstance(val, (int, float)):
            per_state[state][action] = float(val)
    out = {}
    for state, vals in per_state.items():
        best = max(vals, key=vals.get)
        out[state] = (best, vals[best])
    return out


# ------------------------------------------------------------------- policies

def decide(policy: str, ep: dict, qtable: dict) -> bool:
    if policy == "actual":
        return ep["chosen"] in ("CALL", "PUT")
    if policy == "trade_all":
        return True
    best = qtable.get(ep["state_key"]) or qtable.get(ep["state_key_raw"])
    if best is None:
        return False
    action, q = best
    if policy == "qtable":
        return action == ep["candidate"]
    if policy == "qtable_pos":
        return action == ep["candidate"] and q > 0.0
    raise ValueError(policy)


# ---------------------------------------------------------------------- stats

def score(trades: list[dict], notional: float) -> dict:
    if not trades:
        return {"n": 0}
    pnls = [t["pnl"] for t in trades]
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    equity, peak, max_dd = 0.0, 0.0, 0.0
    for p in pnls:  # trades are as_of-ordered
        equity += notional * p / 100.0
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
    return {
        "n": len(pnls),
        "n_counterfactual": sum(1 for t in trades if t["counterfactual"]),
        "win_rate": len(wins) / len(pnls),
        "avg_win_pct": gross_win / len(wins) if wins else None,
        "avg_loss_pct": -gross_loss / len(losses) if losses else None,
        "expectancy_pct": sum(pnls) / len(pnls),
        "profit_factor": gross_win / gross_loss if gross_loss > 0 else None,
        "total_pnl_dollars": notional * sum(pnls) / 100.0,
        "max_drawdown_dollars": max_dd,
        "final_equity_dollars": equity,
    }


def breakdown(trades: list[dict], key: str, top: int = 8) -> list[tuple]:
    buckets = defaultdict(list)
    for t in trades:
        buckets[t.get(key) or "?"].append(t["pnl"])
    rows = []
    for name, pnls in buckets.items():
        rows.append((name, len(pnls), sum(pnls) / len(pnls),
                     sum(1 for p in pnls if p > 0) / len(pnls)))
    rows.sort(key=lambda r: -r[1])
    return rows[:top]


# --------------------------------------------------------------------- report

def fmt(v, pat="{:+.1f}"):
    return "—" if v is None else pat.format(v)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="episodes.db")
    ap.add_argument("--qtable", default="rl_qtable.json")
    ap.add_argument("--notional", type=float, default=1000.0)
    ap.add_argument("--since", help="only episodes with as_of >= this (ISO)")
    ap.add_argument("--until", help="only episodes with as_of < this (ISO)")
    ap.add_argument("--json", help="also write full results to this JSON file")
    args = ap.parse_args()

    eps = load_episodes(args.db, args.since, args.until)
    if eps:
        print(f"window: {eps[0]['as_of']} .. {eps[-1]['as_of']}")
    qtable = load_qtable(args.qtable)
    print(f"episodes with resolved P&L and a candidate direction: {len(eps)}")
    print(f"Q-table states loaded: {len(qtable)}")

    policies = ("actual", "trade_all", "qtable", "qtable_pos")
    results, taken = {}, {}
    for pol in policies:
        taken[pol] = [e for e in eps if decide(pol, e, qtable)]
        results[pol] = score(taken[pol], args.notional)

    hdr = (f"{'policy':<11}{'trades':>7}{'CF%':>6}{'win%':>7}{'avgW%':>8}"
           f"{'avgL%':>8}{'exp%':>7}{'PF':>6}{'P&L$':>11}{'maxDD$':>9}")
    print("\n== policy comparison "
          f"(equal ${args.notional:.0f}/trade, no compounding) ==")
    print(hdr)
    print("-" * len(hdr))
    for pol in policies:
        s = results[pol]
        if s["n"] == 0:
            print(f"{pol:<11}{0:>7}")
            continue
        cf = 100.0 * s["n_counterfactual"] / s["n"]
        pf = "—" if s["profit_factor"] is None else f"{s['profit_factor']:.2f}"
        print(f"{pol:<11}{s['n']:>7}{cf:>6.0f}{100*s['win_rate']:>7.1f}"
              f"{fmt(s['avg_win_pct']):>8}{fmt(s['avg_loss_pct']):>8}"
              f"{s['expectancy_pct']:>+7.2f}{pf:>6}"
              f"{s['total_pnl_dollars']:>+11.0f}"
              f"{s['max_drawdown_dollars']:>9.0f}")

    # gate value: what the gates let through vs what they blocked
    traded = [e for e in eps if not e["counterfactual"]]
    blocked = [e for e in eps if e["counterfactual"]]
    if traded and blocked:
        mt = sum(e["pnl"] for e in traded) / len(traded)
        mb = sum(e["pnl"] for e in blocked) / len(blocked)
        print("\n== gate value ==")
        print(f"traded  mean P&L: {mt:+.2f}%  (n={len(traded)})")
        print(f"blocked mean P&L: {mb:+.2f}%  (n={len(blocked)}, counterfactual)")
        print(f"gate edge per decision: {mt - mb:+.2f} pct-points")

    print("\n== realized trades by underlying (top 8) ==")
    for name, n, mean, wr in breakdown(traded, "underlying"):
        print(f"  {name:<8} n={n:<5} mean={mean:+7.2f}%  win={100*wr:4.1f}%")
    print("\n== realized trades by regime ==")
    for name, n, mean, wr in breakdown(traded, "regime"):
        print(f"  {str(name):<16} n={n:<5} mean={mean:+7.2f}%  win={100*wr:4.1f}%")

    print("\ncaveats: counterfactual P&L is mark-derived (optimistic); equal "
          "notional per trade;\nno capital/concurrency limits; paper fills.")

    if args.json:
        with open(args.json, "w", encoding="utf-8") as fh:
            json.dump({"db": args.db, "qtable": args.qtable,
                       "notional": args.notional, "results": results},
                      fh, indent=2)
        print(f"wrote {args.json}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
