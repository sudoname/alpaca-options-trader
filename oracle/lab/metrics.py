"""
Oracle Lab — performance & robustness metrics (PURE, no I/O).

Every function here is a pure statistic over a list of *trade records*. A trade
record is a plain dict; the only field that must be present for the PnL-based
statistics is ``pnl`` (realized dollar PnL, signed). Everything else is optional
and only unlocks a richer breakdown when present:

  pnl              realized dollar PnL (signed)          [required for PnL stats]
  return_pct       realized return in percent (signed)   [Sharpe/Sortino prefer]
  direction        'call' | 'put' | 'no_trade'
  regime           regime label (e.g. 'trending')
  catalyst_type    catalyst label or None
  strategy_mode    'intraday' | 'swing'
  conviction       0.0-1.0 conviction score
  pop              modelled probability of profit at entry (0-1)
  win              bool/int outcome (derived from pnl when absent)
  theoretical_ev   modelled EV at decision time (dollars)
  executable_ev    EV after fill/slippage modelling (dollars)
  realized_ev      realized PnL used for EV-capture (defaults to pnl)
  mfe / mae        max favourable / adverse excursion (dollars, mae <= 0)
  hold_minutes     holding time in minutes (or use hold_days)
  hold_days        holding time in days
  entry_ts/exit_ts ISO timestamps (used for exposure/turnover when present)

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer a safe default (0.0 / None / {}) over a guess; never raise on
    malformed input. An empty trade list yields zeroed stats, not an error.
  * Deterministic. No randomness, no wall-clock reads.
"""

from __future__ import annotations

import math
from typing import Any, Callable, Dict, List, Optional, Sequence

# Direction / bucket labels.
DIR_CALL = "call"
DIR_PUT = "put"
DIR_NO_TRADE = "no_trade"

# Trading periods per year for annualization (US equity/options sessions).
_TRADING_DAYS_PER_YEAR = 252


# --------------------------------------------------------------------------- #
# Coercion helpers
# --------------------------------------------------------------------------- #
def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _iter(trades: Any) -> List[dict]:
    """Coerce arbitrary input into a list of dict records. Fail-open: a
    non-iterable (or a string) yields an empty list rather than raising."""
    if isinstance(trades, (str, bytes, dict)) or trades is None:
        return []
    try:
        return [t for t in trades if isinstance(t, dict)]
    except TypeError:
        return []


def _pnls(trades: Sequence[dict]) -> List[float]:
    """Signed realized PnL for every record that carries a numeric ``pnl``."""
    out: List[float] = []
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        v = _to_float(t.get("pnl"))
        if v is not None:
            out.append(v)
    return out


def _returns(trades: Sequence[dict]) -> List[float]:
    """Signed return fraction per trade. Uses ``return_pct``/100 when present,
    else falls back to ``pnl`` (dollar) so Sharpe still ranks consistently."""
    out: List[float] = []
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        rp = _to_float(t.get("return_pct"))
        if rp is not None:
            out.append(rp / 100.0)
            continue
        pv = _to_float(t.get("pnl"))
        if pv is not None:
            out.append(pv)
    return out


def _is_win(t: dict) -> Optional[bool]:
    """Outcome of a trade. Explicit ``win`` wins; else derived from ``pnl``."""
    if not isinstance(t, dict):
        return None
    if "win" in t and t["win"] is not None:
        try:
            return bool(t["win"])
        except Exception:  # pragma: no cover - defensive
            return None
    v = _to_float(t.get("pnl"))
    if v is None:
        return None
    return v > 0.0


# --------------------------------------------------------------------------- #
# Core scalar statistics
# --------------------------------------------------------------------------- #
def trade_count(trades: Sequence[dict]) -> int:
    return sum(1 for t in _iter(trades) if isinstance(t, dict))


def win_rate(trades: Sequence[dict]) -> float:
    """Fraction of resolved trades with pnl > 0, in [0, 1]. 0.0 when empty."""
    wins = 0
    total = 0
    for t in _iter(trades):
        w = _is_win(t)
        if w is None:
            continue
        total += 1
        if w:
            wins += 1
    return round(wins / total, 6) if total else 0.0


def avg_win(trades: Sequence[dict]) -> float:
    wins = [p for p in _pnls(trades) if p > 0.0]
    return round(sum(wins) / len(wins), 6) if wins else 0.0


def avg_loss(trades: Sequence[dict]) -> float:
    """Average of losing PnLs (returned as a NEGATIVE number). 0.0 when none."""
    losses = [p for p in _pnls(trades) if p < 0.0]
    return round(sum(losses) / len(losses), 6) if losses else 0.0


def expectancy(trades: Sequence[dict]) -> float:
    """Mean realized PnL per trade (dollars). 0.0 when empty."""
    pnls = _pnls(trades)
    return round(sum(pnls) / len(pnls), 6) if pnls else 0.0


def total_pnl(trades: Sequence[dict]) -> float:
    return round(sum(_pnls(trades)), 6)


def profit_factor(trades: Sequence[dict]) -> Optional[float]:
    """Gross profit / gross loss. None when there are no losses (undefined)."""
    pnls = _pnls(trades)
    gross_profit = sum(p for p in pnls if p > 0.0)
    gross_loss = -sum(p for p in pnls if p < 0.0)
    if gross_loss <= 0.0:
        return None
    return round(gross_profit / gross_loss, 6)


def _mean(xs: Sequence[float]) -> float:
    return sum(xs) / len(xs) if xs else 0.0


def _stdev(xs: Sequence[float], sample: bool = True) -> float:
    n = len(xs)
    if n < 2:
        return 0.0
    mu = _mean(xs)
    var = sum((x - mu) ** 2 for x in xs) / ((n - 1) if sample else n)
    return math.sqrt(var)


def sharpe(trades: Sequence[dict], *, rf: float = 0.0,
           annualize: bool = False) -> float:
    """Per-trade Sharpe over the return series (mean-rf)/stdev.

    ``annualize`` scales by sqrt(252) as a coarse per-session proxy. 0.0 when
    fewer than two returns or zero dispersion (never divides by zero)."""
    rets = _returns(trades)
    if len(rets) < 2:
        return 0.0
    sd = _stdev(rets)
    if sd <= 0.0:
        return 0.0
    s = (_mean(rets) - rf) / sd
    if annualize:
        s *= math.sqrt(_TRADING_DAYS_PER_YEAR)
    return round(s, 6)


def sortino(trades: Sequence[dict], *, rf: float = 0.0,
            annualize: bool = False) -> float:
    """Like Sharpe but penalizes only downside dispersion. 0.0 when no
    downside deviation (never divides by zero)."""
    rets = _returns(trades)
    if len(rets) < 2:
        return 0.0
    downside = [min(0.0, r - rf) for r in rets]
    dd = math.sqrt(_mean([d * d for d in downside]))
    if dd <= 0.0:
        return 0.0
    s = (_mean(rets) - rf) / dd
    if annualize:
        s *= math.sqrt(_TRADING_DAYS_PER_YEAR)
    return round(s, 6)


def equity_curve(trades: Sequence[dict], *, starting: float = 0.0) -> List[float]:
    """Cumulative PnL curve (one point per trade, in list order)."""
    curve: List[float] = []
    running = starting
    for p in _pnls(trades):
        running += p
        curve.append(round(running, 6))
    return curve


def max_drawdown(trades: Sequence[dict]) -> float:
    """Largest peak-to-trough drop on the cumulative PnL curve (returned as a
    NON-NEGATIVE dollar amount). 0.0 when empty or monotonically rising."""
    peak = 0.0
    running = 0.0
    max_dd = 0.0
    for p in _pnls(trades):
        running += p
        if running > peak:
            peak = running
        dd = peak - running
        if dd > max_dd:
            max_dd = dd
    return round(max_dd, 6)


def calmar(trades: Sequence[dict]) -> Optional[float]:
    """Total PnL / max drawdown. None when there is no drawdown (undefined)."""
    dd = max_drawdown(trades)
    if dd <= 0.0:
        return None
    return round(total_pnl(trades) / dd, 6)


def avg_mfe(trades: Sequence[dict]) -> float:
    xs = [f for f in (_to_float(t.get("mfe")) for t in _iter(trades)
          if isinstance(t, dict)) if f is not None]
    return round(_mean(xs), 6) if xs else 0.0


def avg_mae(trades: Sequence[dict]) -> float:
    xs = [f for f in (_to_float(t.get("mae")) for t in _iter(trades)
          if isinstance(t, dict)) if f is not None]
    return round(_mean(xs), 6) if xs else 0.0


def _holds(trades: Sequence[dict]) -> List[float]:
    """Holding time per trade in MINUTES (hold_minutes, else hold_days*1440)."""
    out: List[float] = []
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        m = _to_float(t.get("hold_minutes"))
        if m is None:
            d = _to_float(t.get("hold_days"))
            if d is not None:
                m = d * 1440.0
        if m is not None:
            out.append(m)
    return out


def avg_hold_minutes(trades: Sequence[dict]) -> float:
    xs = _holds(trades)
    return round(_mean(xs), 6) if xs else 0.0


def median_hold_minutes(trades: Sequence[dict]) -> float:
    xs = sorted(_holds(trades))
    if not xs:
        return 0.0
    n = len(xs)
    mid = n // 2
    med = xs[mid] if n % 2 else (xs[mid - 1] + xs[mid]) / 2.0
    return round(med, 6)


def exposure(trades: Sequence[dict], *, window_minutes: Optional[float] = None
             ) -> Optional[float]:
    """Fraction of a window spent in trades = sum(holds)/window. None when the
    window is unknown or non-positive (cannot be inferred without a clock)."""
    if window_minutes is None or window_minutes <= 0:
        return None
    return round(min(1.0, sum(_holds(trades)) / window_minutes), 6)


def turnover(trades: Sequence[dict], *, window_minutes: Optional[float] = None
             ) -> Optional[float]:
    """Trades per day over the window. None when the window is unknown."""
    if window_minutes is None or window_minutes <= 0:
        return None
    days = window_minutes / 1440.0
    if days <= 0:
        return None
    return round(trade_count(trades) / days, 6)


def realized_ev_capture(trades: Sequence[dict]) -> Optional[float]:
    """Realized PnL / modelled (executable, else theoretical) EV, aggregate.

    A capture ratio near 1.0 means realized outcomes matched the EV model; well
    below 1.0 means the model over-promised. None when no modelled EV present.
    """
    model_sum = 0.0
    realized_sum = 0.0
    have_model = False
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        ev = _to_float(t.get("executable_ev"))
        if ev is None:
            ev = _to_float(t.get("theoretical_ev"))
        if ev is None:
            continue
        have_model = True
        model_sum += ev
        r = _to_float(t.get("realized_ev"))
        if r is None:
            r = _to_float(t.get("pnl")) or 0.0
        realized_sum += r
    if not have_model or model_sum == 0.0:
        return None
    return round(realized_sum / model_sum, 6)


def pop_calibration(trades: Sequence[dict], *, bins: int = 5) -> List[dict]:
    """Reliability table: for each modelled-PoP bin, the empirical win rate.

    Returns one row per non-empty bin: {bin_low, bin_high, n, mean_pop,
    empirical_win_rate, gap}. ``gap`` = empirical - mean_pop (calibration
    error). Empty list when no trade carries both ``pop`` and an outcome."""
    return _calibration_table(
        trades,
        pred_fn=lambda t: _to_float(t.get("pop")),
        outcome_fn=lambda t: (1.0 if _is_win(t) else 0.0)
        if _is_win(t) is not None else None,
        bins=bins,
        pred_key="mean_pop",
    )


def direction_calibration(trades: Sequence[dict]) -> Dict[str, dict]:
    """Per-direction hit stats: {call:{n,win_rate,expectancy}, put:{...}}.

    'Hit' for a directional trade == pnl > 0 (the option made money in the
    predicted direction). NO_TRADE rows are excluded (nothing to be right about).
    """
    out: Dict[str, dict] = {}
    for side in (DIR_CALL, DIR_PUT):
        sub = [t for t in _iter(trades) if isinstance(t, dict)
               and str(t.get("direction") or "").lower() == side]
        if not sub:
            continue
        out[side] = {
            "n": trade_count(sub),
            "win_rate": win_rate(sub),
            "expectancy": expectancy(sub),
        }
    return out


def _calibration_table(trades: Sequence[dict], *,
                       pred_fn: Callable[[dict], Optional[float]],
                       outcome_fn: Callable[[dict], Optional[float]],
                       bins: int, pred_key: str) -> List[dict]:
    bins = max(1, int(bins))
    buckets: List[List[tuple]] = [[] for _ in range(bins)]
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        p = pred_fn(t)
        o = outcome_fn(t)
        if p is None or o is None:
            continue
        p = min(1.0, max(0.0, p))
        idx = min(bins - 1, int(p * bins))
        buckets[idx].append((p, o))
    rows: List[dict] = []
    for i, b in enumerate(buckets):
        if not b:
            continue
        preds = [x[0] for x in b]
        outs = [x[1] for x in b]
        mean_pred = _mean(preds)
        emp = _mean(outs)
        rows.append({
            "bin_low": round(i / bins, 6),
            "bin_high": round((i + 1) / bins, 6),
            "n": len(b),
            pred_key: round(mean_pred, 6),
            "empirical_win_rate": round(emp, 6),
            "gap": round(emp - mean_pred, 6),
        })
    return rows


# --------------------------------------------------------------------------- #
# Aggregate summary + breakdowns
# --------------------------------------------------------------------------- #
def compute_metrics(trades: Sequence[dict], *,
                    window_minutes: Optional[float] = None,
                    rf: float = 0.0) -> Dict[str, Any]:
    """One-call summary of every scalar statistic. Deterministic, never raises.

    ``window_minutes`` (optional) unlocks exposure/turnover; without it those
    keys are None. Result is a flat dict safe to persist as JSON."""
    trades = [t for t in _iter(trades) if isinstance(t, dict)]
    return {
        "trade_count": trade_count(trades),
        "win_rate": win_rate(trades),
        "avg_win": avg_win(trades),
        "avg_loss": avg_loss(trades),
        "expectancy": expectancy(trades),
        "total_pnl": total_pnl(trades),
        "profit_factor": profit_factor(trades),
        "sharpe": sharpe(trades, rf=rf),
        "sortino": sortino(trades, rf=rf),
        "max_drawdown": max_drawdown(trades),
        "calmar": calmar(trades),
        "avg_mfe": avg_mfe(trades),
        "avg_mae": avg_mae(trades),
        "avg_hold_minutes": avg_hold_minutes(trades),
        "median_hold_minutes": median_hold_minutes(trades),
        "exposure": exposure(trades, window_minutes=window_minutes),
        "turnover": turnover(trades, window_minutes=window_minutes),
        "realized_ev_capture": realized_ev_capture(trades),
    }


def _group_by(trades: Sequence[dict], key: str,
              *, default: str = "unknown") -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        raw = t.get(key)
        label = str(raw).lower() if raw not in (None, "") else default
        groups.setdefault(label, []).append(t)
    return groups


def _bucketize(trades: Sequence[dict], key: str, edges: Sequence[float],
               labels: Sequence[str]) -> Dict[str, List[dict]]:
    """Group trades by which [edge_i, edge_{i+1}) band ``key`` falls in."""
    groups: Dict[str, List[dict]] = {}
    for t in _iter(trades):
        if not isinstance(t, dict):
            continue
        v = _to_float(t.get(key))
        if v is None:
            groups.setdefault("unknown", []).append(t)
            continue
        placed = False
        for i in range(len(edges) - 1):
            if edges[i] <= v < edges[i + 1]:
                groups.setdefault(labels[i], []).append(t)
                placed = True
                break
        if not placed:
            groups.setdefault(labels[-1], []).append(t)
    return groups


def breakdown(trades: Sequence[dict], key: str) -> Dict[str, Dict[str, Any]]:
    """Per-value ``compute_metrics`` for a categorical field (regime,
    direction, catalyst_type, strategy_mode, ...). Deterministic key order."""
    groups = _group_by(trades, key)
    return {label: compute_metrics(groups[label])
            for label in sorted(groups.keys())}


def breakdown_by_regime(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    return breakdown(trades, "regime")


def breakdown_by_direction(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    return breakdown(trades, "direction")


def breakdown_by_catalyst(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    return breakdown(trades, "catalyst_type")


def breakdown_by_mode(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    return breakdown(trades, "strategy_mode")


def breakdown_by_conviction(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    edges = [0.0, 0.34, 0.67, 1.0001]
    labels = ["low", "medium", "high"]
    groups = _bucketize(trades, "conviction", edges, labels)
    return {label: compute_metrics(groups[label])
            for label in sorted(groups.keys())}


def breakdown_by_ev(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    edges = [-1e18, 0.0, 1e18]
    labels = ["negative_ev", "positive_ev"]
    key = "executable_ev" if any(
        isinstance(t, dict) and _to_float(t.get("executable_ev")) is not None
        for t in _iter(trades)) else "theoretical_ev"
    groups = _bucketize(trades, key, edges, labels)
    return {label: compute_metrics(groups[label])
            for label in sorted(groups.keys())}


def breakdown_by_pop(trades: Sequence[dict]) -> Dict[str, Dict[str, Any]]:
    edges = [0.0, 0.4, 0.6, 1.0001]
    labels = ["low", "medium", "high"]
    groups = _bucketize(trades, "pop", edges, labels)
    return {label: compute_metrics(groups[label])
            for label in sorted(groups.keys())}


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Empty input -> zeroed stats, never raises.
    m0 = compute_metrics([])
    if m0["trade_count"] != 0 or m0["win_rate"] != 0.0:
        print("FAIL: empty stats", m0); ok = False
    if m0["profit_factor"] is not None or m0["calmar"] is not None:
        print("FAIL: empty undefined -> None", m0); ok = False

    trades = [
        {"pnl": 100.0, "return_pct": 10.0, "direction": "call",
         "regime": "trending", "pop": 0.7, "conviction": 0.9,
         "theoretical_ev": 80.0, "executable_ev": 60.0, "mfe": 150.0,
         "mae": -20.0, "hold_minutes": 30.0, "strategy_mode": "intraday",
         "catalyst_type": "earnings"},
        {"pnl": -50.0, "return_pct": -5.0, "direction": "put",
         "regime": "choppy", "pop": 0.55, "conviction": 0.4,
         "theoretical_ev": 20.0, "executable_ev": -5.0, "mfe": 40.0,
         "mae": -60.0, "hold_minutes": 90.0, "strategy_mode": "swing",
         "catalyst_type": None},
        {"pnl": 200.0, "return_pct": 20.0, "direction": "call",
         "regime": "trending", "pop": 0.8, "conviction": 0.95,
         "theoretical_ev": 120.0, "executable_ev": 100.0, "mfe": 220.0,
         "mae": -10.0, "hold_minutes": 45.0, "strategy_mode": "intraday",
         "catalyst_type": "earnings"},
        {"pnl": -80.0, "return_pct": -8.0, "direction": "put",
         "regime": "choppy", "pop": 0.5, "conviction": 0.3,
         "theoretical_ev": -10.0, "executable_ev": -30.0, "mfe": 15.0,
         "mae": -95.0, "hold_minutes": 120.0, "strategy_mode": "swing",
         "catalyst_type": "news"},
    ]

    if trade_count(trades) != 4:
        print("FAIL: trade_count", trade_count(trades)); ok = False
    if win_rate(trades) != 0.5:
        print("FAIL: win_rate", win_rate(trades)); ok = False
    if abs(avg_win(trades) - 150.0) > 1e-9:
        print("FAIL: avg_win", avg_win(trades)); ok = False
    if abs(avg_loss(trades) - (-65.0)) > 1e-9:
        print("FAIL: avg_loss", avg_loss(trades)); ok = False
    if abs(expectancy(trades) - 42.5) > 1e-9:
        print("FAIL: expectancy", expectancy(trades)); ok = False
    if abs(total_pnl(trades) - 170.0) > 1e-9:
        print("FAIL: total_pnl", total_pnl(trades)); ok = False

    # profit factor = 300 / 130
    pf = profit_factor(trades)
    if pf is None or abs(pf - (300.0 / 130.0)) > 1e-6:
        print("FAIL: profit_factor", pf); ok = False

    # Drawdown: curve = 100, 50, 250, 170. Peak 250 -> trough 170 = 80.
    if abs(max_drawdown(trades) - 80.0) > 1e-9:
        print("FAIL: max_drawdown", max_drawdown(trades)); ok = False
    cal = calmar(trades)
    if cal is None or abs(cal - (170.0 / 80.0)) > 1e-6:
        print("FAIL: calmar", cal); ok = False

    # Sharpe/Sortino finite & non-zero on dispersed returns.
    if sharpe(trades) == 0.0 or sortino(trades) == 0.0:
        print("FAIL: sharpe/sortino zero", sharpe(trades), sortino(trades))
        ok = False

    # No division by zero when all returns identical.
    flat = [{"pnl": 10.0, "return_pct": 1.0} for _ in range(3)]
    if sharpe(flat) != 0.0 or sortino(flat) != 0.0:
        print("FAIL: zero-dispersion should be 0.0"); ok = False

    # EV capture: realized 170 / executable 125 = 1.36
    cap = realized_ev_capture(trades)
    if cap is None or abs(cap - (170.0 / 125.0)) > 1e-6:
        print("FAIL: realized_ev_capture", cap); ok = False

    # Exposure/turnover need a window; None without one.
    if exposure(trades) is not None or turnover(trades) is not None:
        print("FAIL: exposure/turnover need window"); ok = False
    exp = exposure(trades, window_minutes=1440.0)
    if exp is None or not (0.0 <= exp <= 1.0):
        print("FAIL: exposure windowed", exp); ok = False

    # Direction calibration: two calls (both win), two puts (both lose).
    dc = direction_calibration(trades)
    if dc.get("call", {}).get("win_rate") != 1.0:
        print("FAIL: call win_rate", dc); ok = False
    if dc.get("put", {}).get("win_rate") != 0.0:
        print("FAIL: put win_rate", dc); ok = False

    # PoP calibration returns rows summing to n=4.
    pc = pop_calibration(trades, bins=5)
    if sum(r["n"] for r in pc) != 4:
        print("FAIL: pop_calibration coverage", pc); ok = False

    # Breakdowns partition the trades.
    br = breakdown_by_regime(trades)
    if br.get("trending", {}).get("trade_count") != 2:
        print("FAIL: regime breakdown", br); ok = False
    bev = breakdown_by_ev(trades)
    if "positive_ev" not in bev or "negative_ev" not in bev:
        print("FAIL: ev breakdown", bev); ok = False
    if bev["negative_ev"]["trade_count"] != 2:  # two executable_ev < 0
        print("FAIL: ev breakdown split", bev); ok = False

    # Determinism + junk tolerance.
    if compute_metrics(trades) != compute_metrics(trades):
        print("FAIL: non-deterministic"); ok = False
    for junk in (None, 42, "x", [None, 7, {"pnl": "bad"}]):
        try:
            compute_metrics(junk)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("lab.metrics self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
