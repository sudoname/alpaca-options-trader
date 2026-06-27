"""
Execution-quality analytics over ``episodes.db`` + broker fills (offline, fail-open).

Reports four execution metrics the daily report consumes:

* **Avg spread paid** -- mean ``(ask-bid)/mid`` at entry (from episode entry quotes).
* **Avg slippage**    -- mean ``(fill_price-mid)`` in bps (signed; + = paid up).
* **Avg holding time**-- mean ``hold_days`` (+ intraday hours paired from broker fills).
* **Fill quality**    -- % of fills at/inside mid; avg distance to mid in bps.

Spread / slippage / fill-quality need entry quotes (``quote_bid`` / ``quote_ask``).
Backfilled rows lack quotes, so those metrics report ``None`` with ``samples=0``
until live evidence-stamped trades close -- holding time works immediately off
``hold_days``. The module is pure over injected rows; loaders fail open to empty
lists and the whole thing is unit-testable with no network.
"""

import datetime as dt
import json
import os
from collections import defaultdict, deque
from typing import Dict, List, Optional, Sequence

DEFAULT_DB = "episodes.db"


# --------------------------------------------------------------------------- #
# numeric helpers
# --------------------------------------------------------------------------- #
def _f(v) -> Optional[float]:
    """Best-effort float; None on anything non-numeric."""
    if v is None:
        return None
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _mean(xs: Sequence[float]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _median(xs: Sequence[float]) -> Optional[float]:
    xs = sorted(x for x in xs if x is not None)
    if not xs:
        return None
    n = len(xs)
    mid = n // 2
    return xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0


def _round(v, nd=2):
    return round(v, nd) if isinstance(v, (int, float)) else v


# --------------------------------------------------------------------------- #
# pure core
# --------------------------------------------------------------------------- #
def entry_quality(episodes: Sequence[dict]) -> dict:
    """Spread paid, slippage-to-mid (bps), and fill quality over rows with quotes."""
    spreads, slippages, dist_bps = [], [], []
    inside = 0
    n = 0
    for e in episodes:
        bid = _f(e.get("quote_bid"))
        ask = _f(e.get("quote_ask"))
        fill = _f(e.get("fill_price"))
        if bid is None or ask is None or fill is None:
            continue
        if ask <= 0 or bid < 0 or ask < bid:
            continue
        mid = (bid + ask) / 2.0
        if mid <= 0:
            continue
        n += 1
        spreads.append((ask - bid) / mid)
        slippages.append((fill - mid) / mid * 1e4)        # signed bps
        dist_bps.append(abs(fill - mid) / mid * 1e4)
        if fill <= mid + 1e-9:                              # at/inside mid
            inside += 1
    avg_spread = _mean(spreads)
    return {
        "samples": n,
        "avg_spread_pct": (avg_spread * 100) if avg_spread is not None else None,
        "avg_slippage_bps": _mean(slippages),
        "fill_quality_pct": (inside / n * 100) if n else None,
        "avg_dist_to_mid_bps": _mean(dist_bps),
    }


def holding_stats(episodes: Sequence[dict],
                  round_trips: Optional[Sequence[tuple]] = None) -> dict:
    """Holding-time distribution from ``hold_days`` (+ optional broker round-trips).

    ``round_trips`` is a list of ``(hours, same_day_bool)`` from
    :func:`round_trips_from_fills`. ``avg_intraday_hours`` averages only the
    same-day pairs; ``avg_roundtrip_hours`` averages all of them.
    """
    days = [_f(e.get("hold_days")) for e in episodes]
    days = [d for d in days if d is not None]
    same_day = sum(1 for d in days if d <= 0)
    rt = list(round_trips or [])
    intraday = [h for h, sd in rt if sd]
    all_hours = [h for h, _ in rt]
    return {
        "samples": len(days),
        "avg_hold_days": _mean(days),
        "median_hold_days": _median(days),
        "same_day_pct": (same_day / len(days) * 100) if days else None,
        "avg_intraday_hours": _mean(intraday) if intraday else None,
        "intraday_samples": len(intraday),
        "avg_roundtrip_hours": _mean(all_hours) if all_hours else None,
        "roundtrip_samples": len(all_hours),
    }


def _fill_ts(f: dict) -> Optional[dt.datetime]:
    raw = f.get("transaction_time") or f.get("time")
    if not isinstance(raw, str):
        return None
    try:
        return dt.datetime.fromisoformat(raw.replace("Z", "+00:00"))
    except (ValueError, AttributeError):
        return None


def round_trips_from_fills(fills: Sequence[dict]) -> List[tuple]:
    """FIFO-pair buy(open)/sell(close) fills per symbol -> ``(hours, same_day)``.

    The live system holds *long* single-leg options, so buy = open, sell = close.
    Partial fills are matched proportionally; unmatched legs are ignored.
    ``same_day`` is True when open and close fall on the same UTC calendar date.
    """
    timed = [(t, f) for f in fills if (t := _fill_ts(f)) is not None]
    timed.sort(key=lambda tf: tf[0])
    opens: Dict[str, deque] = defaultdict(deque)
    pairs: List[tuple] = []
    for t, f in timed:
        side = (f.get("side") or "").lower()
        sym = f.get("symbol")
        qty = _f(f.get("qty")) or 0.0
        if qty <= 0 or not sym:
            continue
        if side == "buy":
            opens[sym].append([t, qty])
        elif side == "sell":
            remaining = qty
            while remaining > 1e-9 and opens[sym]:
                ot, oq = opens[sym][0]
                take = min(oq, remaining)
                hours = (t - ot).total_seconds() / 3600.0
                pairs.append((hours, ot.date() == t.date()))
                remaining -= take
                oq -= take
                if oq <= 1e-9:
                    opens[sym].popleft()
                else:
                    opens[sym][0][1] = oq
    return pairs


def compute_execution(episodes: Sequence[dict],
                      fills: Optional[Sequence[dict]] = None) -> dict:
    """Assemble the execution-quality report from episodes (+ optional broker fills)."""
    intraday = round_trips_from_fills(fills) if fills else []
    return {
        "episodes": len(episodes),
        "entry": entry_quality(episodes),
        "holding": holding_stats(episodes, intraday),
    }


# --------------------------------------------------------------------------- #
# formatting
# --------------------------------------------------------------------------- #
def _fmt(v, suffix="", nd=2):
    if v is None:
        return "n/a"
    return f"{round(v, nd)}{suffix}"


def format_markdown(report: dict) -> str:
    eq = report.get("entry", {})
    hs = report.get("holding", {})
    lines = ["### Execution Quality", ""]
    lines.append(f"- Episodes analyzed: **{report.get('episodes', 0)}**")
    lines.append(
        f"- Avg spread paid: **{_fmt(eq.get('avg_spread_pct'), '%')}** "
        f"| Avg slippage: **{_fmt(eq.get('avg_slippage_bps'), ' bps')}** "
        f"(entry-quote samples: {eq.get('samples', 0)})"
    )
    lines.append(
        f"- Fill quality (at/inside mid): **{_fmt(eq.get('fill_quality_pct'), '%')}** "
        f"| Avg dist to mid: **{_fmt(eq.get('avg_dist_to_mid_bps'), ' bps')}**"
    )
    lines.append(
        f"- Avg holding time: **{_fmt(hs.get('avg_hold_days'), ' days')}** "
        f"(median {_fmt(hs.get('median_hold_days'), ' days')}, "
        f"same-day {_fmt(hs.get('same_day_pct'), '%')})"
    )
    if hs.get("roundtrip_samples"):
        lines.append(
            f"- Avg round-trip hold: **{_fmt(hs.get('avg_roundtrip_hours'), ' h')}** "
            f"({hs.get('roundtrip_samples')} round-trips; "
            f"intraday {_fmt(hs.get('avg_intraday_hours'), ' h')} "
            f"over {hs.get('intraday_samples')})"
        )
    if not eq.get("samples"):
        lines.append("")
        lines.append("_Spread/slippage/fill-quality await live entry quotes "
                     "(backfilled rows carry no bid/ask)._")
    return "\n".join(lines)


def to_json(report: dict) -> str:
    return json.dumps(report, indent=2, default=str)


# --------------------------------------------------------------------------- #
# fail-open loaders (network/disk live only here)
# --------------------------------------------------------------------------- #
def load_episodes(db_path: str = DEFAULT_DB) -> List[dict]:
    """Completed episodes from the store; [] on any failure."""
    try:
        from episode_store import EpisodeStore
        return EpisodeStore(db_path).completed()
    except Exception:
        return []


def load_fills(path: str) -> List[dict]:
    """Broker FILL activities from an ``account_activities.json`` file or export dir."""
    try:
        if os.path.isdir(path):
            path = os.path.join(path, "account_activities.json")
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        if not isinstance(data, list):
            return []
        return [a for a in data
                if str(a.get("activity_type", "")).upper() == "FILL"]
    except Exception:
        return []


def generate_live_report(db_path: str = DEFAULT_DB,
                         fills_path: Optional[str] = None) -> dict:
    episodes = load_episodes(db_path)
    fills = load_fills(fills_path) if fills_path else []
    return compute_execution(episodes, fills)


# --------------------------------------------------------------------------- #
# self-test (no network/disk)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # entry_quality: bid 1.00 ask 1.20 mid 1.10, fill 1.15 -> paid up
    eps = [
        {"quote_bid": 1.00, "quote_ask": 1.20, "fill_price": 1.15, "hold_days": 0},
        {"quote_bid": 2.00, "quote_ask": 2.10, "fill_price": 2.05, "hold_days": 3},
        {"quote_bid": None, "quote_ask": None, "fill_price": 9.9, "hold_days": 5},
    ]
    eq = entry_quality(eps)
    if eq["samples"] != 2:
        print("FAIL: entry samples", eq["samples"]); ok = False
    # spread1=0.20/1.10, spread2=0.10/2.05 -> mean*100
    exp_spread = ((0.20 / 1.10) + (0.10 / 2.05)) / 2 * 100
    if eq["avg_spread_pct"] is None or abs(eq["avg_spread_pct"] - exp_spread) > 1e-6:
        print("FAIL: spread", eq["avg_spread_pct"], exp_spread); ok = False
    # row2 fill 2.05 == mid 2.05 -> inside; row1 fill 1.15 > mid 1.10 -> outside
    if abs(eq["fill_quality_pct"] - 50.0) > 1e-6:
        print("FAIL: fill quality", eq["fill_quality_pct"]); ok = False

    hs = holding_stats(eps)
    if hs["samples"] != 3 or abs(hs["avg_hold_days"] - (0 + 3 + 5) / 3) > 1e-6:
        print("FAIL: holding", hs); ok = False
    if abs(hs["same_day_pct"] - (1 / 3 * 100)) > 1e-6:
        print("FAIL: same_day_pct", hs["same_day_pct"]); ok = False

    # round_trips_from_fills: buy 10:00 -> sell 14:00 = 4h
    fills = [
        {"symbol": "X", "side": "buy", "qty": "1",
         "transaction_time": "2026-06-26T10:00:00Z"},
        {"symbol": "X", "side": "sell", "qty": "1",
         "transaction_time": "2026-06-26T14:00:00Z"},
    ]
    rts = round_trips_from_fills(fills)
    if len(rts) != 1 or abs(rts[0][0] - 4.0) > 1e-6 or rts[0][1] is not True:
        print("FAIL: round trip hours", rts); ok = False
    hs2 = holding_stats(eps, rts)
    if hs2["intraday_samples"] != 1 or abs(hs2["avg_intraday_hours"] - 4.0) > 1e-6:
        print("FAIL: intraday agg", hs2); ok = False

    # empty inputs never raise
    compute_execution([])
    format_markdown(compute_execution([]))

    print("execution_analytics self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        fp = None
        for a in sys.argv[1:]:
            if not a.startswith("--"):
                fp = a
        rep = generate_live_report(fills_path=fp)
        print(format_markdown(rep))
        print()
        print(to_json(rep))
        sys.exit(0)
    sys.exit(_self_test())
