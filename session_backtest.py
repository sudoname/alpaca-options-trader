"""Honest session-strategy backtest on UNDERLYING daily session returns.

Tests the Knuteson (arXiv:2010.01727) overnight/intraday hypothesis on the
symbols Oracle actually trades, with strict chronological hygiene:

* Chronological train (60%) / validation (20%) / test (20%) splits with a
  one-session embargo at each boundary.  Session labels span at most one
  session, so a one-session purge removes all label overlap.
* The session-aware model's parameters (symbol-selection t-stat threshold)
  are fitted on train, chosen on validation, and FROZEN before the test
  segment is touched.  Test data never influences any choice.
* Costs: ``--cost-bps`` per side (default 1.0bp) is charged on every leg
  entry and exit (2 sides/session for a session strategy).  At daily
  frequency this is a material drag and is reported explicitly.
* Fills: overnight leg = MOC at close, MOO at next open; intraday leg =
  MOO at open, MOC at close.  Official auction prices are used as fills --
  realistic for liquid US large caps in small size, optimistic for size.
* Gap risk: the overnight leg bears full gap exposure between close and
  open; max drawdown on the net series includes realized gaps in-sample.
  No stop-loss can protect an overnight position intra-gap.

Annualization: 252 sessions/year.  Each session contributes exactly one
leg return, so ann_mean = 252 * mean, ann_vol = sqrt(252) * std.

OPTIONS ARE UNTESTED HERE: this backtest uses underlying prices only.  No
historical option-quote dataset is wired in, so nothing here demonstrates
that an options implementation of a session strategy is profitable.

Strategies compared (all long-only, equal-weight):
  overnight_only   hold close -> next open, flat intraday
  intraday_only    hold open -> close, flat overnight
  buy_hold         daily close-to-close (one entry, negligible turnover)
  session_model    overnight-only restricted to symbols whose TRAIN-period
                   overnight t-stat exceeds a threshold chosen on VALIDATION

Read-only research: no orders, no trading-behavior changes.

Usage:
    python session_backtest.py --symbols SPY QQQ ... --start 2024-10-01 \
        --end 2026-10-03 --cost-bps 1 --out session_backtest.json
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from datetime import datetime, timezone
from typing import Dict, List, Optional, Sequence, Tuple

import pandas as pd

ANN_SESSIONS = 252
THRESH_GRID = [0.0, 0.5, 1.0, 1.5, 2.0]


# ---------------------------------------------------------------------------
# Pure core (no I/O)
# ---------------------------------------------------------------------------


def chrono_splits(
    n: int, train_frac: float = 0.6, val_frac: float = 0.2, embargo: int = 1
) -> Dict[str, Tuple[int, int]]:
    """Chronological [start, end) index ranges with an embargo gap.

    Session returns are non-overlapping (each label spans one session), so a
    one-session embargo fully purges boundary overlap.
    """
    tr_end = int(n * train_frac)
    va_end = int(n * (train_frac + val_frac))
    return {
        "train": (0, tr_end),
        "val": (tr_end + embargo, va_end),
        "test": (va_end + embargo, n),
    }


def net_leg(gross: pd.Series, cost_bps: float, sides_per_session: float = 2.0) -> pd.Series:
    """Subtract per-side costs from a per-session leg return series."""
    return gross - sides_per_session * cost_bps / 1e4


def metrics(net: pd.Series) -> dict:
    """Net performance metrics with uncertainty.  Annualization: 252."""
    s = net.dropna().astype(float)
    n = len(s)
    if n == 0:
        return {"n": 0}
    mean = float(s.mean())
    std = float(s.std(ddof=1)) if n > 1 else float("nan")
    se = std / math.sqrt(n) if n > 1 else float("nan")
    equity = (1.0 + s).cumprod()
    peak = equity.cummax()
    mdd = float((equity / peak - 1.0).min())
    sharpe = (mean / std * math.sqrt(ANN_SESSIONS)) if std and std > 0 else float("nan")
    return {
        "n_sessions": n,
        "mean_per_session": mean,
        "se_mean": se,
        "ann_mean": mean * ANN_SESSIONS,
        "ann_vol": std * math.sqrt(ANN_SESSIONS) if n > 1 else float("nan"),
        "sharpe_net": sharpe,
        "cum_net_return": float(equity.iloc[-1] - 1.0),
        "max_drawdown": mdd,
        "pos_freq": float((s > 0).mean()),
    }


def equal_weight(leg_returns: Dict[str, pd.Series], symbols: Sequence[str]) -> pd.Series:
    """Equal-weight portfolio across symbols, aligned on session date index."""
    cols = {s: leg_returns[s] for s in symbols if s in leg_returns}
    if not cols:
        return pd.Series(dtype=float)
    df = pd.DataFrame(cols)
    return df.mean(axis=1, skipna=True)


def select_symbols(
    overnight: Dict[str, pd.Series], train_rng: Tuple[int, int], threshold: float
) -> List[str]:
    """Symbols whose TRAIN-segment overnight t-stat exceeds threshold.

    Only indices inside train_rng are read -- validation/test rows never
    influence selection.
    """
    lo, hi = train_rng
    picked = []
    for sym, s in overnight.items():
        seg = s.iloc[lo:hi].dropna()
        n = len(seg)
        if n < 20:
            continue
        std = seg.std(ddof=1)
        if not std or std <= 0:
            continue
        t = seg.mean() / (std / math.sqrt(n))
        if t > threshold:
            picked.append(sym)
    return sorted(picked)


def fit_threshold(
    overnight_net: Dict[str, pd.Series],
    splits: Dict[str, Tuple[int, int]],
    grid: Sequence[float] = THRESH_GRID,
) -> Tuple[float, List[str], dict]:
    """Pick the t-stat threshold on VALIDATION only; freeze before test.

    Selection stats come from TRAIN; candidate portfolios are scored on the
    VALIDATION segment by net Sharpe.  Returns (threshold, selected symbols
    re-estimated on train+val for the test period, val scores).
    """
    val_lo, val_hi = splits["val"]
    scores = {}
    for th in grid:
        syms = select_symbols(overnight_net, splits["train"], th)
        if not syms:
            scores[th] = {"n_symbols": 0, "val_sharpe": float("nan")}
            continue
        port = equal_weight(overnight_net, syms).iloc[val_lo:val_hi]
        m = metrics(port)
        scores[th] = {"n_symbols": len(syms), "val_sharpe": m.get("sharpe_net", float("nan"))}
    valid = {t: v for t, v in scores.items() if v["n_symbols"] > 0 and not math.isnan(v["val_sharpe"])}
    if not valid:
        return float("nan"), [], scores
    best = max(valid, key=lambda t: valid[t]["val_sharpe"])
    # Re-estimate selection on train+val (all pre-test data) with the frozen
    # threshold; the test segment is never read here.
    trainval_rng = (splits["train"][0], splits["val"][1])
    final_syms = select_symbols(overnight_net, trainval_rng, best)
    return best, final_syms, scores


# ---------------------------------------------------------------------------
# Runner (I/O)
# ---------------------------------------------------------------------------


def run_backtest(
    symbols: Sequence[str],
    start: str,
    end: str,
    cost_bps: float = 1.0,
    benchmark: str = "SPY",
) -> dict:
    from datetime import datetime as _dt

    from session_returns import compute_session_returns, fetch_calendar, fetch_daily_bars

    d0 = _dt.strptime(start, "%Y-%m-%d").date()
    d1 = _dt.strptime(end, "%Y-%m-%d").date()
    all_syms = sorted(set(symbols) | {benchmark})
    calendar = fetch_calendar(d0, d1)
    cal_dates = [c["date"] for c in calendar]
    bars = fetch_daily_bars(all_syms, d0, d1, adjustment="all")

    overnight: Dict[str, pd.Series] = {}
    intraday: Dict[str, pd.Series] = {}
    daily: Dict[str, pd.Series] = {}
    quality = {}
    for sym in all_syms:
        rets, q = compute_session_returns(bars[sym], cal_dates)
        quality[sym] = q.as_dict()
        idx = pd.Index(rets["date"])
        overnight[sym] = pd.Series(rets["overnight"].values, index=idx)
        intraday[sym] = pd.Series(rets["intraday"].values, index=idx)
        daily[sym] = pd.Series(rets["daily"].values, index=idx)

    # master session index = benchmark's sessions (most complete)
    master = overnight[benchmark].index
    for d in (overnight, intraday, daily):
        for sym in d:
            d[sym] = d[sym].reindex(master)

    n = len(master)
    splits = chrono_splits(n)
    te_lo, te_hi = splits["test"]

    # cost-adjusted per-symbol legs (2 sides per session)
    on_net = {s: net_leg(v, cost_bps) for s, v in overnight.items()}
    id_net = {s: net_leg(v, cost_bps) for s, v in intraday.items()}
    # buy & hold: one entry over the whole period -> amortized cost ~0; use gross
    universe = [s for s in symbols if s != benchmark] or [benchmark]

    threshold, model_syms, val_scores = fit_threshold(on_net, splits)

    def test_slice(series: pd.Series) -> pd.Series:
        return series.iloc[te_lo:te_hi]

    strategies = {
        f"{benchmark}_overnight_only": test_slice(on_net[benchmark]),
        f"{benchmark}_intraday_only": test_slice(id_net[benchmark]),
        f"{benchmark}_buy_hold": test_slice(daily[benchmark]),
        "universe_overnight_only": test_slice(equal_weight(on_net, universe)),
        "universe_intraday_only": test_slice(equal_weight(id_net, universe)),
        "universe_buy_hold": test_slice(equal_weight(daily, universe)),
    }
    if model_syms:
        strategies["session_model_overnight"] = test_slice(equal_weight(on_net, model_syms))

    results = {name: metrics(s) for name, s in strategies.items()}
    for name in results:
        if "buy_hold" in name:
            results[name]["turnover_sides_per_session"] = 0.0
            results[name]["exposure"] = "fully invested close-to-close"
        else:
            results[name]["turnover_sides_per_session"] = 2.0
            results[name]["exposure"] = (
                "overnight leg only (~17.5h/day)" if "overnight" in name else "intraday leg only (~6.5h/day)"
            )

    return {
        "meta": {
            "symbols": list(symbols),
            "benchmark": benchmark,
            "start": start,
            "end": end,
            "sessions_total": n,
            "splits": {k: list(v) for k, v in splits.items()},
            "test_dates": [str(master[te_lo]), str(master[te_hi - 1])] if te_hi > te_lo else None,
            "cost_bps_per_side": cost_bps,
            "annualization": f"{ANN_SESSIONS} sessions/year",
            "adjustment": "all (split+dividend)",
            "options_status": "UNTESTED - underlying prices only",
            "generated_at": datetime.now(timezone.utc).isoformat(),
        },
        "session_model": {
            "threshold_chosen_on_val": threshold,
            "val_scores_by_threshold": val_scores,
            "selected_symbols": model_syms,
            "n_selected": len(model_syms),
        },
        "test_results_net": results,
        "quality_flags": {s: q for s, q in quality.items() if any(q.values())},
    }


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--cost-bps", type=float, default=1.0)
    p.add_argument("--benchmark", default="SPY")
    p.add_argument("--out", default=None)
    args = p.parse_args(argv)

    result = run_backtest(args.symbols, args.start, args.end, args.cost_bps, args.benchmark)
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")

    meta = result["meta"]
    print(
        f"\nTest segment: {meta['test_dates']}  "
        f"({meta['splits']['test'][1] - meta['splits']['test'][0]} sessions, "
        f"costs {meta['cost_bps_per_side']}bp/side)  [{meta['options_status']}]"
    )
    sm = result["session_model"]
    print(
        f"session_model: threshold={sm['threshold_chosen_on_val']} "
        f"(picked on val), {sm['n_selected']} symbols"
    )
    print(f"{'strategy':28s} {'ann_mean':>9s} {'ann_vol':>8s} {'sharpe':>7s} {'cum':>8s} {'mdd':>7s} {'n':>4s}")
    for name, m in result["test_results_net"].items():
        if not m.get("n_sessions"):
            continue
        print(
            f"{name:28s} {m['ann_mean']:+8.2%} {m['ann_vol']:8.2%} "
            f"{m['sharpe_net']:7.2f} {m['cum_net_return']:+8.2%} "
            f"{m['max_drawdown']:7.2%} {m['n_sessions']:4d}"
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
