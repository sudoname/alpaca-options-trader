"""Session-return decomposition: overnight vs intraday returns.

Motivated by Knuteson (2020), "Strikingly Suspicious Overnight and Intraday
Returns" (arXiv:2010.01727).  This module decomposes daily equity returns into

    overnight_return = open_t / close_{t-1} - 1          (prior close -> open)
    intraday_return  = close_t / open_t - 1              (open -> close)
    daily_return     = close_t / close_{t-1} - 1

and verifies the identity (1 + overnight) * (1 + intraday) == (1 + daily).

Correctness rules enforced here
-------------------------------
* Exchange calendar: session dates come from the Alpaca /v2/calendar endpoint
  (holidays and early closes included).  A "previous close" is the close of
  the immediately preceding *calendar session*, never simply the previous
  bar row.  Pairs that span a missing bar are dropped and flagged.
* Time zones: Alpaca bar timestamps are UTC; they are converted to
  America/New_York before being mapped to a session date.
* Corporate actions: open and close come from the SAME bar series fetched in
  a SINGLE request with ONE explicit adjustment setting, so opens and closes
  are always adjusted identically.  Adjusted closes are never combined with
  unadjusted opens.
* Price vs total returns: with adjustment="all" (default) the series is
  split- AND dividend-adjusted, so computed returns approximate total
  returns (dividends accrue to the overnight leg by construction, because
  the adjustment is applied to prices before the ex-date).  With
  adjustment="split" the series is split-adjusted only and returns are
  price returns.  The choice is recorded in every output.
* Data quality: non-positive prices, open/close outside [low, high], bars on
  non-calendar dates, and calendar sessions with missing bars are rejected
  or flagged -- never silently used.

The pure functions (``compute_session_returns``, ``session_stats``) take
plain pandas inputs and perform no I/O, so they are testable offline with
synthetic fixtures.  Synthetic fixtures prove software correctness only --
they say nothing about market edge.

CLI (requires ALPACA_API_KEY / ALPACA_SECRET_KEY in .env):

    python session_returns.py --symbols SPY QQQ --start 2024-01-01 \
        --end 2026-10-03 --adjustment all --out session_stats.json

This module is read-only research tooling: it places no orders and changes
no trading behavior.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional, Sequence

import pandas as pd

ET_TZ = "America/New_York"
IDENTITY_TOL = 1e-9

# ---------------------------------------------------------------------------
# Pure core (no I/O)
# ---------------------------------------------------------------------------


@dataclass
class SessionQuality:
    """Data-quality flags raised while computing session returns."""

    missing_sessions: List[str] = field(default_factory=list)   # calendar day, no bar
    non_calendar_bars: List[str] = field(default_factory=list)  # bar day not in calendar
    bad_price_bars: List[str] = field(default_factory=list)     # <=0 or outside [low, high]
    stale_bars: List[str] = field(default_factory=list)         # zero volume, flat bar
    identity_failures: List[str] = field(default_factory=list)  # (1+on)(1+id) != 1+daily

    def as_dict(self) -> Dict[str, List[str]]:
        return {
            "missing_sessions": self.missing_sessions,
            "non_calendar_bars": self.non_calendar_bars,
            "bad_price_bars": self.bad_price_bars,
            "stale_bars": self.stale_bars,
            "identity_failures": self.identity_failures,
        }

    @property
    def clean(self) -> bool:
        return not (
            self.missing_sessions
            or self.non_calendar_bars
            or self.bad_price_bars
            or self.identity_failures
        )


def compute_session_returns(
    bars: pd.DataFrame,
    calendar_dates: Sequence[date],
) -> tuple[pd.DataFrame, SessionQuality]:
    """Decompose one symbol's daily bars into session returns.

    Parameters
    ----------
    bars : DataFrame with columns [date, open, high, low, close, volume],
        one row per session, a SINGLE consistent adjustment already applied.
        ``date`` must be python ``datetime.date`` in exchange (ET) terms.
    calendar_dates : ordered sequence of valid exchange session dates
        covering the bar range (holidays excluded, early closes included --
        an early close is still one session).

    Returns
    -------
    (returns_df, quality)
        returns_df columns: date, prev_date, overnight, intraday, daily.
        Only rows where BOTH the session bar and the previous session bar
        exist and pass quality checks are emitted.
    """
    q = SessionQuality()
    cal = sorted(set(calendar_dates))
    cal_set = set(cal)

    by_date: Dict[date, dict] = {}
    for row in bars.itertuples(index=False):
        d = row.date
        if d not in cal_set:
            q.non_calendar_bars.append(str(d))
            continue
        o, h, l, c = float(row.open), float(row.high), float(row.low), float(row.close)
        if o <= 0 or c <= 0 or h <= 0 or l <= 0:
            q.bad_price_bars.append(str(d))
            continue
        # open/close must lie inside the bar's own range (tiny tolerance for
        # vendor rounding on adjusted series)
        tol = 1e-6 * c
        if not (l - tol <= o <= h + tol) or not (l - tol <= c <= h + tol):
            q.bad_price_bars.append(str(d))
            continue
        vol = float(getattr(row, "volume", float("nan")))
        if vol == 0 and o == c == h == l:
            q.stale_bars.append(str(d))  # flagged but still usable
        by_date[d] = {"open": o, "close": c}

    if by_date:
        lo, hi = min(by_date), max(by_date)
        for d in cal:
            if lo <= d <= hi and d not in by_date:
                q.missing_sessions.append(str(d))

    out = []
    for prev_d, d in zip(cal, cal[1:]):
        if d not in by_date or prev_d not in by_date:
            continue  # gap already flagged via missing_sessions
        pc = by_date[prev_d]["close"]
        o = by_date[d]["open"]
        c = by_date[d]["close"]
        overnight = o / pc - 1.0
        intraday = c / o - 1.0
        daily = c / pc - 1.0
        if abs((1.0 + overnight) * (1.0 + intraday) - (1.0 + daily)) > IDENTITY_TOL:
            q.identity_failures.append(str(d))
            continue
        out.append(
            {
                "date": d,
                "prev_date": prev_d,
                "overnight": overnight,
                "intraday": intraday,
                "daily": daily,
            }
        )

    cols = ["date", "prev_date", "overnight", "intraday", "daily"]
    return pd.DataFrame(out, columns=cols), q


def session_stats(returns: pd.DataFrame, window: Optional[int] = None) -> Dict[str, dict]:
    """Per-session summary stats with uncertainty.

    Returns, for each of overnight/intraday/daily: n, mean, std (sample),
    standard error of the mean, positive frequency, and cumulative growth
    of $1 allocated only to that session.  If ``window`` is given, stats are
    computed on the trailing ``window`` rows only (rolling snapshot).
    """
    df = returns.tail(window) if window else returns
    stats: Dict[str, dict] = {}
    for col in ("overnight", "intraday", "daily"):
        s = df[col].astype(float)
        n = int(s.count())
        if n == 0:
            stats[col] = {"n": 0}
            continue
        mean = float(s.mean())
        std = float(s.std(ddof=1)) if n > 1 else float("nan")
        se = std / math.sqrt(n) if n > 1 else float("nan")
        stats[col] = {
            "n": n,
            "mean": mean,
            "std": std,
            "se_mean": se,
            "pos_freq": float((s > 0).mean()),
            "cum_growth": float((1.0 + s).prod()),
        }
    return stats


# ---------------------------------------------------------------------------
# Alpaca fetchers (network I/O; used by CLI and downstream scripts)
# ---------------------------------------------------------------------------


def _load_env() -> None:
    try:
        from dotenv import load_dotenv

        load_dotenv()
    except Exception:
        pass


def fetch_calendar(start: date, end: date) -> List[dict]:
    """Exchange sessions from Alpaca /v2/calendar (holidays + early closes).

    Returns list of {"date": date, "open": datetime, "close": datetime}
    (open/close are naive ET wall-clock times as provided by Alpaca).
    """
    import os

    from alpaca.trading.client import TradingClient
    from alpaca.trading.requests import GetCalendarRequest

    _load_env()
    client = TradingClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY"), paper=True
    )
    days = client.get_calendar(GetCalendarRequest(start=start, end=end))
    return [{"date": d.date, "open": d.open, "close": d.close} for d in days]


def fetch_daily_bars(
    symbols: Sequence[str],
    start: date,
    end: date,
    adjustment: str = "all",
) -> Dict[str, pd.DataFrame]:
    """Daily bars per symbol with ONE explicit adjustment for the whole series.

    adjustment: "all" (split+dividend; ~total returns), "split" (price
    returns) or "raw".  Open and close always come from the same adjusted
    series -- adjusted closes are never mixed with unadjusted opens.
    """
    import os

    from alpaca.data.enums import Adjustment
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame

    _load_env()
    client = StockHistoricalDataClient(
        os.getenv("ALPACA_API_KEY"), os.getenv("ALPACA_SECRET_KEY")
    )
    # Free data plans reject SIP queries touching the last ~15 minutes;
    # clamp the end so we keep official SIP opens/closes (never fall back
    # to IEX-only prints, which are not official auction prices).
    end_dt = min(
        datetime.combine(end, datetime.max.time().replace(microsecond=0)),
        datetime.utcnow() - timedelta(minutes=16),
    )
    req = StockBarsRequest(
        symbol_or_symbols=list(symbols),
        timeframe=TimeFrame.Day,
        start=datetime.combine(start, datetime.min.time()),
        end=end_dt,
        adjustment=Adjustment(adjustment),
    )
    raw = client.get_stock_bars(req)
    out: Dict[str, pd.DataFrame] = {}
    for sym in symbols:
        rows = []
        for bar in raw.data.get(sym, []):
            ts = pd.Timestamp(bar.timestamp)
            ts = ts.tz_localize("UTC") if ts.tzinfo is None else ts
            et_date = ts.tz_convert(ET_TZ).date()
            rows.append(
                {
                    "date": et_date,
                    "open": float(bar.open),
                    "high": float(bar.high),
                    "low": float(bar.low),
                    "close": float(bar.close),
                    "volume": float(bar.volume),
                }
            )
        out[sym] = pd.DataFrame(
            rows, columns=["date", "open", "high", "low", "close", "volume"]
        )
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run_analysis(
    symbols: Sequence[str],
    start: date,
    end: date,
    adjustment: str = "all",
    rolling_window: int = 63,
) -> dict:
    """Fetch real data and compute per-symbol session stats."""
    calendar = fetch_calendar(start, end)
    cal_dates = [c["date"] for c in calendar]
    early_closes = [
        str(c["date"]) for c in calendar if c["close"].time() < datetime(2000, 1, 1, 16).time()
    ]
    bars = fetch_daily_bars(symbols, start, end, adjustment=adjustment)

    result = {
        "meta": {
            "symbols": list(symbols),
            "start": str(start),
            "end": str(end),
            "adjustment": adjustment,
            "return_type": "total-return approximation (split+dividend adjusted)"
            if adjustment == "all"
            else ("price return (split-adjusted)" if adjustment == "split" else "raw"),
            "calendar_sessions": len(cal_dates),
            "early_close_sessions": early_closes,
            "generated_at": datetime.utcnow().isoformat() + "Z",
        },
        "symbols": {},
    }
    for sym in symbols:
        rets, quality = compute_session_returns(bars[sym], cal_dates)
        result["symbols"][sym] = {
            "quality": quality.as_dict(),
            "quality_clean": quality.clean,
            "full_sample": session_stats(rets),
            f"trailing_{rolling_window}d": session_stats(rets, window=rolling_window),
            "first_date": str(rets["date"].iloc[0]) if len(rets) else None,
            "last_date": str(rets["date"].iloc[-1]) if len(rets) else None,
        }
    return result


def main(argv: Optional[Sequence[str]] = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--symbols", nargs="+", required=True)
    p.add_argument("--start", required=True, help="YYYY-MM-DD")
    p.add_argument("--end", required=True, help="YYYY-MM-DD")
    p.add_argument("--adjustment", default="all", choices=["all", "split", "raw"])
    p.add_argument("--window", type=int, default=63, help="trailing window size")
    p.add_argument("--out", default=None, help="write JSON to this path")
    args = p.parse_args(argv)

    result = run_analysis(
        args.symbols,
        datetime.strptime(args.start, "%Y-%m-%d").date(),
        datetime.strptime(args.end, "%Y-%m-%d").date(),
        adjustment=args.adjustment,
        rolling_window=args.window,
    )
    text = json.dumps(result, indent=2, default=str)
    if args.out:
        with open(args.out, "w") as fh:
            fh.write(text)
        print(f"wrote {args.out}")
    # concise console summary
    for sym, info in result["symbols"].items():
        fs = info["full_sample"]
        print(f"\n{sym}  (n={fs['overnight'].get('n', 0)}, clean={info['quality_clean']})")
        for leg in ("overnight", "intraday", "daily"):
            st = fs[leg]
            if st.get("n"):
                print(
                    f"  {leg:9s} mean={st['mean']*1e4:+7.2f}bp  se={st['se_mean']*1e4:5.2f}bp  "
                    f"pos={st['pos_freq']:.1%}  $1->{st['cum_growth']:.4f}"
                )
    return 0


if __name__ == "__main__":
    sys.exit(main())
