"""Evaluate Oracle's EXISTING signals split by trading session.

Reads historical decisions from episodes.db (point-in-time store written by
smart_trader at decision time) and measures, per decision:

  1. REALIZED option P&L (net_pnl_pct from episode outcomes), stratified by
     holding horizon (same-day close vs held overnight).  This is the only
     options-level evidence; everything else below is an UNDERLYING proxy.
  2. "Remaining intraday" leg: signed underlying return from the first
     executable price AFTER the signal timestamp (next minute-bar open,
     strictly later than as_of) to the SAME session's official close.
  3. "Next overnight" leg: signed underlying return from the signal
     session's close to the NEXT exchange session's open (an MOC entry is
     executable because every Oracle signal fires >= 60 minutes before the
     close, per the scheduler's ENTRY_CUTOFF).

Point-in-time rules
-------------------
* The feature cutoff equals the signal timestamp (smart_trader computes
  features from live quotes at scan time; nothing later is used).
* The minute-bar entry must be STRICTLY after as_of -- a bar stamped at the
  signal time could contain the trade that produced the signal.
* A session close may be used as an entry price only because it occurs
  AFTER the signal was observed (signal intraday -> MOC order is
  executable).  A close is never used to imply a fill at that same close
  for a signal generated FROM that close.

Adjustment consistency
----------------------
* Intraday leg: RAW minute open -> RAW daily close of the SAME session
  (no corporate action can occur inside one session; raw/raw is consistent).
* Overnight leg: close and next open BOTH come from one 'all'-adjusted
  daily series (splits/dividends handled consistently across the night).
Adjusted and unadjusted prices are never mixed within a leg.

Costs: ``--cost-bps`` (default 1.0) is charged PER SIDE on the underlying
proxy legs (2x per round trip), representing half-spread + slippage for
liquid US equities.  Realized option P&L already includes real spreads.

This script is read-only research: it never changes decisions, places no
orders, and does not alter any trading behavior.

Usage:
    python session_signal_eval.py --db episodes_server.db --mode live-paper \
        --cost-bps 1 --out session_signal_eval.json
"""

from __future__ import annotations

import argparse
import json
import math
import sqlite3
import sys
from bisect import bisect_right
from datetime import date, datetime, time, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

ET = ZoneInfo("America/New_York")
MAX_ENTRY_DELAY_MIN = 30  # give up on the intraday leg if no minute bar within this


# ---------------------------------------------------------------------------
# Pure core (no I/O) -- unit-testable with synthetic fixtures
# ---------------------------------------------------------------------------


def direction_of(chosen_action: Optional[str], strat: Optional[str]) -> Optional[int]:
    """Map a decision to signed underlying exposure (+1 long / -1 short)."""
    a = (chosen_action or "").upper()
    if a == "CALL":
        return 1
    if a == "PUT":
        return -1
    if a == "SPREAD":
        # only spread strat in the data is bullish_put_credit_spread
        return 1 if "bullish" in (strat or "") else None
    return None


def parse_as_of_utc(s: str) -> datetime:
    """episodes.as_of is ISO; naive values are UTC by store convention."""
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=timezone.utc) if dt.tzinfo is None else dt.astimezone(timezone.utc)


def eval_decision_sessions(
    decisions: Sequence[dict],
    minute_bars: Dict[Tuple[str, date], List[Tuple[datetime, float]]],
    daily_raw: Dict[str, Dict[date, dict]],
    daily_adj: Dict[str, Dict[date, dict]],
    cal_dates: Sequence[date],
    cost_bps: float = 1.0,
) -> List[dict]:
    """Compute per-decision session legs.  Pure function.

    decisions: dicts with decision_id, as_of (aware UTC datetime),
        underlying, direction (+1/-1).
    minute_bars: (symbol, et_date) -> [(aware-UTC ts, open)] sorted by ts.
    daily_raw / daily_adj: symbol -> et_date -> {"open":..,"close":..}.
    cal_dates: sorted exchange session dates.
    cost_bps: per-side cost on underlying proxy legs.
    """
    cal = sorted(set(cal_dates))
    next_session = {d: cal[i + 1] for i, d in enumerate(cal[:-1])}
    rt_cost = 2.0 * cost_bps / 1e4
    out = []
    for dec in decisions:
        as_of: datetime = dec["as_of"]
        sym = dec["underlying"]
        direction = dec["direction"]
        et_dt = as_of.astimezone(ET)
        d = et_dt.date()
        rec = {
            "decision_id": dec["decision_id"],
            "underlying": sym,
            "direction": direction,
            "signal_ts_utc": as_of.isoformat(),
            "signal_ts_et": et_dt.isoformat(),
            "feature_cutoff_utc": as_of.isoformat(),  # features from live scan data
            "session_date": str(d),
            "intraday": None,
            "overnight": None,
            "flags": [],
        }
        if d not in next_session and d not in set(cal):
            rec["flags"].append("signal_date_not_in_calendar")
            out.append(rec)
            continue

        # ---- remaining-intraday leg (raw minute entry -> raw daily close)
        bars = minute_bars.get((sym, d), [])
        entry_px = entry_ts = None
        if bars:
            ts_list = [b[0] for b in bars]
            i = bisect_right(ts_list, as_of)  # first bar strictly after as_of
            if i < len(bars):
                cand_ts, cand_px = bars[i]
                if (cand_ts - as_of) <= timedelta(minutes=MAX_ENTRY_DELAY_MIN):
                    entry_ts, entry_px = cand_ts, cand_px
                else:
                    rec["flags"].append("entry_bar_stale")
        else:
            rec["flags"].append("no_minute_bars")
        close_raw = daily_raw.get(sym, {}).get(d, {}).get("close")
        if entry_px and close_raw and entry_px > 0:
            gross = direction * (close_raw / entry_px - 1.0)
            rec["intraday"] = {
                "entry_ts_utc": entry_ts.isoformat(),
                "entry_assumption": "next minute-bar open strictly after signal (raw)",
                "exit_assumption": "same-session official close (raw, MOC)",
                "gross": gross,
                "net": gross - rt_cost,
                "cost_rt": rt_cost,
            }
        elif close_raw is None:
            rec["flags"].append("no_daily_raw_close")

        # ---- next-overnight leg ('all'-adjusted close -> next open)
        nd = next_session.get(d)
        c_adj = daily_adj.get(sym, {}).get(d, {}).get("close")
        o_adj = daily_adj.get(sym, {}).get(nd, {}).get("open") if nd else None
        if nd and c_adj and o_adj and c_adj > 0:
            gross = direction * (o_adj / c_adj - 1.0)
            rec["overnight"] = {
                "entry_assumption": "same-session close via MOC (signal >=60min before close)",
                "exit_assumption": f"next session ({nd}) official open (MOO)",
                "gross": gross,
                "net": gross - rt_cost,
                "cost_rt": rt_cost,
            }
        else:
            rec["flags"].append("overnight_bars_missing")
        out.append(rec)
    return out


def aggregate_legs(records: Sequence[dict], leg: str, key: str = "net") -> dict:
    vals = [r[leg][key] for r in records if r.get(leg)]
    n = len(vals)
    if n == 0:
        return {"n": 0}
    mean = sum(vals) / n
    if n > 1:
        var = sum((v - mean) ** 2 for v in vals) / (n - 1)
        std = math.sqrt(var)
        se = std / math.sqrt(n)
    else:
        std = se = float("nan")
    return {
        "n": n,
        "mean": mean,
        "std": std,
        "se_mean": se,
        "pos_freq": sum(1 for v in vals if v > 0) / n,
    }


def realized_by_hold(rows: Sequence[dict]) -> dict:
    """Stratify realized option net_pnl_pct by holding horizon (REAL data)."""
    strata = {"same_day (hold_days=0)": [], "held_overnight (hold_days>=1)": []}
    for r in rows:
        pnl = r.get("net_pnl_pct")
        hd = r.get("hold_days")
        if pnl is None or hd is None:
            continue
        key = "same_day (hold_days=0)" if hd == 0 else "held_overnight (hold_days>=1)"
        strata[key].append(float(pnl))
    out = {}
    for k, vals in strata.items():
        n = len(vals)
        if n == 0:
            out[k] = {"n": 0}
            continue
        mean = sum(vals) / n
        std = math.sqrt(sum((v - mean) ** 2 for v in vals) / (n - 1)) if n > 1 else float("nan")
        out[k] = {
            "n": n,
            "mean_net_pnl_pct": mean,
            "se_mean": std / math.sqrt(n) if n > 1 else float("nan"),
            "win_rate": sum(1 for v in vals if v > 0) / n,
        }
    return out


# ---------------------------------------------------------------------------
# Data loading (I/O)
# ---------------------------------------------------------------------------


def load_decisions(db_path: str, mode: str) -> List[dict]:
    con = sqlite3.connect(db_path)
    con.row_factory = sqlite3.Row
    rows = con.execute(
        "SELECT decision_id, as_of, underlying, strat, chosen_action, "
        "net_pnl_pct, hold_days, outcome FROM episodes WHERE mode = ? "
        "ORDER BY as_of",
        (mode,),
    ).fetchall()
    con.close()
    out = []
    for r in rows:
        direction = direction_of(r["chosen_action"], r["strat"])
        if direction is None:
            continue
        out.append(
            {
                "decision_id": r["decision_id"],
                "as_of": parse_as_of_utc(r["as_of"]),
                "underlying": r["underlying"],
                "direction": direction,
                "net_pnl_pct": r["net_pnl_pct"],
                "hold_days": r["hold_days"],
                "outcome": r["outcome"],
            }
        )
    return out


def fetch_minute_bars_for(
    decisions: Sequence[dict],
) -> Dict[Tuple[str, date], List[Tuple[datetime, float]]]:
    """One multi-symbol RAW minute-bar request per decision date."""
    import os

    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    from session_returns import _load_env

    _load_env()
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )
    by_date: Dict[date, set] = {}
    for dec in decisions:
        d = dec["as_of"].astimezone(ET).date()
        by_date.setdefault(d, set()).add(dec["underlying"])

    out: Dict[Tuple[str, date], List[Tuple[datetime, float]]] = {}
    for d in sorted(by_date):
        syms = sorted(by_date[d])
        start = datetime.combine(d, time(9, 30), tzinfo=ET).astimezone(timezone.utc)
        end = datetime.combine(d, time(16, 5), tzinfo=ET).astimezone(timezone.utc)
        # free-plan SIP restriction: never query the last ~15 minutes
        sip_limit = datetime.now(timezone.utc) - timedelta(minutes=16)
        if start >= sip_limit:
            continue
        end = min(end, sip_limit)
        req = StockBarsRequest(
            symbol_or_symbols=syms,
            timeframe=TimeFrame.Minute,
            start=start,
            end=end,
            adjustment=Adjustment.RAW,
        )
        raw = client.get_stock_bars(req)
        for sym in syms:
            bars = []
            for b in raw.data.get(sym, []):
                ts = b.timestamp
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                bars.append((ts.astimezone(timezone.utc), float(b.open)))
            bars.sort(key=lambda x: x[0])
            out[(sym, d)] = bars
    return out


def _daily_lookup(df) -> Dict[date, dict]:
    return {
        row.date: {"open": float(row.open), "close": float(row.close)}
        for row in df.itertuples(index=False)
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--db", default="episodes.db")
    p.add_argument("--mode", default="live-paper", help="episodes.mode filter")
    p.add_argument("--cost-bps", type=float, default=1.0, help="per-side underlying cost")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    from session_returns import fetch_calendar, fetch_daily_bars

    decisions = load_decisions(args.db, args.mode)
    if not decisions:
        print(f"no decisions with mode={args.mode} in {args.db}")
        return 1
    d0 = min(d["as_of"] for d in decisions).astimezone(ET).date() - timedelta(days=7)
    d1 = max(d["as_of"] for d in decisions).astimezone(ET).date() + timedelta(days=7)
    syms = sorted({d["underlying"] for d in decisions})
    print(f"{len(decisions)} decisions, {len(syms)} underlyings, {d0}..{d1}")

    calendar = fetch_calendar(d0, d1)
    cal_dates = [c["date"] for c in calendar]
    raw_bars = fetch_daily_bars(syms, d0, d1, adjustment="raw")
    adj_bars = fetch_daily_bars(syms, d0, d1, adjustment="all")
    daily_raw = {s: _daily_lookup(raw_bars[s]) for s in syms}
    daily_adj = {s: _daily_lookup(adj_bars[s]) for s in syms}
    print("fetching minute bars per decision date ...")
    minute_bars = fetch_minute_bars_for(decisions)

    records = eval_decision_sessions(
        decisions, minute_bars, daily_raw, daily_adj, cal_dates, cost_bps=args.cost_bps
    )

    result = {
        "meta": {
            "db": args.db,
            "mode": args.mode,
            "decisions": len(decisions),
            "underlyings": len(syms),
            "date_range": [str(d0), str(d1)],
            "cost_bps_per_side": args.cost_bps,
            "note": (
                "intraday/overnight legs are UNDERLYING proxies with stated "
                "executable-entry assumptions; realized_options is real "
                "episode P&L (options, net)"
            ),
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "realized_options_by_hold": realized_by_hold(decisions),
        "underlying_proxy": {
            "remaining_intraday_net": aggregate_legs(records, "intraday"),
            "next_overnight_net": aggregate_legs(records, "overnight"),
            "remaining_intraday_gross": aggregate_legs(records, "intraday", "gross"),
            "next_overnight_gross": aggregate_legs(records, "overnight", "gross"),
        },
        "by_direction": {
            "long (CALL)": {
                "intraday_net": aggregate_legs([r for r in records if r["direction"] == 1], "intraday"),
                "overnight_net": aggregate_legs([r for r in records if r["direction"] == 1], "overnight"),
            },
            "short (PUT)": {
                "intraday_net": aggregate_legs([r for r in records if r["direction"] == -1], "intraday"),
                "overnight_net": aggregate_legs([r for r in records if r["direction"] == -1], "overnight"),
            },
        },
        "flag_counts": {},
    }
    flag_counts: Dict[str, int] = {}
    for r in records:
        for f in r["flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1
    result["flag_counts"] = flag_counts

    text = json.dumps(result, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")

    def _fmt(s):
        if not s.get("n"):
            return "n=0"
        return (
            f"n={s['n']}  mean={s['mean']*1e4:+.2f}bp  se={s['se_mean']*1e4:.2f}bp  "
            f"pos={s['pos_freq']:.1%}"
        )

    print("\nRealized option P&L by holding horizon (REAL, net %/trade):")
    for k, v in result["realized_options_by_hold"].items():
        if v.get("n"):
            print(
                f"  {k:32s} n={v['n']:5d}  mean={v['mean_net_pnl_pct']:+.2f}%  "
                f"se={v['se_mean']:.2f}%  win={v['win_rate']:.1%}"
            )
        else:
            print(f"  {k:32s} n=0")
    up = result["underlying_proxy"]
    print("\nUnderlying proxy (signed, net of costs):")
    print(f"  remaining intraday : {_fmt(up['remaining_intraday_net'])}")
    print(f"  next overnight     : {_fmt(up['next_overnight_net'])}")
    print(f"  flags: {flag_counts}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
