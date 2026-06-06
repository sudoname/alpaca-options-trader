"""
Point-in-time market data accessor.

`MarketView(as_of)` exposes only data that was knowable at or before `as_of`.
This is the foundation for no-lookahead feature computation: the same accessor
is used by backtests (a `HistoricalMarketView` over pre-fetched series) and by
live code (a `LiveMarketView` whose `as_of` is `datetime.now()`), so the feature
path cannot accidentally peek at the future.

Key rules enforced here:
  * A daily bar for calendar date D is "known" only after that session's close
    (D + close_time). Deciding at the open of day T therefore cannot see day T's
    high/low/close (this is exactly the lookahead bug in the older enhanced
    backtest, which read the full day's high/low to size an open-time entry).
  * Option quotes are returned only if their quote timestamp <= as_of.
  * Every datum handed out is recorded in an audit log so an integrity test can
    assert nothing stamped after as_of ever escaped.

No network access happens in `HistoricalMarketView`, so it is fully testable
with no credentials. `LiveMarketView` mirrors the raw-`requests` endpoints the
rest of the project already uses (see smart_trader.get_price_history /
get_option_price).
"""

from datetime import datetime, time, timedelta
from typing import Dict, List, NamedTuple, Optional


# --------------------------------------------------------------------------- #
# Bar
# --------------------------------------------------------------------------- #
class Bar(NamedTuple):
    date: str            # "YYYY-MM-DD"
    o: float
    h: float
    l: float
    c: float
    v: float
    close_dt: datetime   # when this bar became known (session close)


def _parse_date(d: str) -> datetime.date:
    return datetime.strptime(d[:10], "%Y-%m-%d").date()


def make_bar(date: str, o, h, l, c, v=0.0, close_time: time = time(16, 0)) -> Bar:
    """Build a Bar, computing the time it becomes known (its session close)."""
    day = _parse_date(date)
    close_dt = datetime.combine(day, close_time)
    return Bar(date[:10], float(o), float(h), float(l), float(c), float(v), close_dt)


# --------------------------------------------------------------------------- #
# Quote
# --------------------------------------------------------------------------- #
class Quote(NamedTuple):
    bid: float
    ask: float
    ts: datetime

    @property
    def mid(self) -> float:
        if self.bid > 0 and self.ask > 0:
            return (self.bid + self.ask) / 2.0
        return self.ask or self.bid


# --------------------------------------------------------------------------- #
# Base MarketView
# --------------------------------------------------------------------------- #
class MarketView:
    """
    Base accessor. Subclasses provide candidate series; this base applies the
    point-in-time (<= as_of) filter and the audit trail.
    """

    def __init__(
        self,
        as_of: datetime,
        *,
        close_time: time = time(16, 0),
        warmup_days: int = 30,
    ):
        if not isinstance(as_of, datetime):
            raise TypeError("as_of must be a datetime")
        self._as_of = as_of
        self.close_time = close_time
        self.warmup_days = warmup_days
        self.audit: List[Dict] = []  # records every datum returned, for integrity tests

    @property
    def as_of(self) -> datetime:
        return self._as_of

    # --- to be provided by subclasses ------------------------------------- #
    def _candidate_daily_bars(self, symbol: str) -> List[Bar]:
        raise NotImplementedError

    def _candidate_intraday_bar(self, symbol: str, minutes: int) -> Optional[Bar]:
        return None

    def _candidate_option_quote(self, occ_symbol: str) -> Optional[Quote]:
        return None

    def _candidate_vix(self, symbol: str) -> List[Bar]:
        return []

    # --- public, point-in-time-filtered API ------------------------------- #
    def _record(self, kind: str, when: datetime, ident: str) -> None:
        self.audit.append({"kind": kind, "ts": when, "id": ident})

    def daily_bars(self, symbol: str, lookback: int = 30) -> List[Bar]:
        """Most recent `lookback` daily bars whose session close <= as_of."""
        known = [b for b in self._candidate_daily_bars(symbol) if b.close_dt <= self._as_of]
        known.sort(key=lambda b: b.date)
        out = known[-lookback:] if lookback else known
        for b in out:
            self._record("daily_bar", b.close_dt, f"{symbol}:{b.date}")
        return out

    def last_close(self, symbol: str) -> Optional[float]:
        bars = self.daily_bars(symbol, 1)
        return bars[-1].c if bars else None

    def intraday_bar(self, symbol: str, minutes: int = 30) -> Optional[Bar]:
        """
        First `minutes`-bar of the as_of session, but only once as_of is past
        that window (None during the warmup window). Never used to derive a
        full-day high/low; that completed-day info comes from daily_bars.
        """
        session_open = datetime.combine(self._as_of.date(), time(9, 30))
        if self._as_of < session_open + timedelta(minutes=minutes):
            return None
        bar = self._candidate_intraday_bar(symbol, minutes)
        if bar is not None:
            self._record("intraday_bar", bar.close_dt, f"{symbol}:{bar.date}:{minutes}m")
        return bar

    def option_quote(self, occ_symbol: str) -> Optional[Dict]:
        q = self._candidate_option_quote(occ_symbol)
        if q is None or q.ts > self._as_of:
            return None
        self._record("option_quote", q.ts, occ_symbol)
        return {"bid": q.bid, "ask": q.ask, "mid": q.mid, "ts": q.ts}

    def vix(self, symbol: str = "^VIX") -> Optional[float]:
        known = [b for b in self._candidate_vix(symbol) if b.close_dt <= self._as_of]
        known.sort(key=lambda b: b.date)
        if not known:
            return None
        b = known[-1]
        self._record("vix", b.close_dt, f"{symbol}:{b.date}")
        return b.c

    def vix_bars(self, symbol: str = "^VIX", lookback: int = 30) -> List[Bar]:
        known = [b for b in self._candidate_vix(symbol) if b.close_dt <= self._as_of]
        known.sort(key=lambda b: b.date)
        out = known[-lookback:] if lookback else known
        for b in out:
            self._record("vix_bar", b.close_dt, f"{symbol}:{b.date}")
        return out


# --------------------------------------------------------------------------- #
# Historical (no network) — backtests and tests
# --------------------------------------------------------------------------- #
class HistoricalMarketView(MarketView):
    """
    Backed by pre-fetched series. `daily` maps symbol -> list[Bar];
    `vix_series` maps symbol -> list[Bar]; `intraday` maps symbol -> Bar;
    `quotes` maps occ_symbol -> Quote.
    """

    def __init__(
        self,
        as_of: datetime,
        *,
        daily: Optional[Dict[str, List[Bar]]] = None,
        vix_series: Optional[Dict[str, List[Bar]]] = None,
        intraday: Optional[Dict[str, Bar]] = None,
        quotes: Optional[Dict[str, Quote]] = None,
        **kwargs,
    ):
        super().__init__(as_of, **kwargs)
        self._daily = daily or {}
        self._vix = vix_series or {}
        self._intraday = intraday or {}
        self._quotes = quotes or {}

    def _candidate_daily_bars(self, symbol):
        return list(self._daily.get(symbol, []))

    def _candidate_intraday_bar(self, symbol, minutes):
        return self._intraday.get(symbol)

    def _candidate_option_quote(self, occ_symbol):
        return self._quotes.get(occ_symbol)

    def _candidate_vix(self, symbol):
        return list(self._vix.get(symbol, []))


# --------------------------------------------------------------------------- #
# Live — thin wrapper over the raw Alpaca endpoints the project already uses
# --------------------------------------------------------------------------- #
class LiveMarketView(MarketView):
    """
    as_of defaults to now(). Fetches daily bars and option quotes from Alpaca
    using the same endpoints as smart_trader. Network-bound; not exercised in
    self-tests. Provided so the live path and the backtest share one feature
    code path.
    """

    def __init__(
        self,
        *,
        headers: Dict[str, str],
        data_url: str = "https://data.alpaca.markets",
        feed: str = "iex",
        as_of: Optional[datetime] = None,
        **kwargs,
    ):
        super().__init__(as_of or datetime.now(), **kwargs)
        self.headers = headers
        self.data_url = data_url
        self.feed = feed

    def _candidate_daily_bars(self, symbol):
        import requests

        end = self._as_of
        start = end - timedelta(days=max(self.warmup_days, 10) + 10)
        try:
            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                headers=self.headers,
                params={
                    "timeframe": "1Day",
                    "start": start.strftime("%Y-%m-%d"),
                    "end": end.strftime("%Y-%m-%d"),
                    "limit": 10000,
                    "feed": self.feed,
                    "adjustment": "raw",
                },
                timeout=30,
            )
        except Exception:
            return []
        if resp.status_code != 200:
            return []
        out = []
        for b in resp.json().get("bars", []) or []:
            out.append(
                make_bar(b["t"], b["o"], b["h"], b["l"], b["c"], b.get("v", 0), self.close_time)
            )
        return out

    def _candidate_option_quote(self, occ_symbol):
        import requests

        try:
            resp = requests.get(
                f"{self.data_url}/v1beta1/options/quotes/latest",
                headers=self.headers,
                params={"symbols": occ_symbol, "feed": "indicative"},
                timeout=30,
            )
        except Exception:
            return None
        if resp.status_code != 200:
            return None
        data = resp.json()
        q = (data.get("quotes") or {}).get(occ_symbol)
        if not q:
            return None
        ts_raw = q.get("t")
        try:
            ts = datetime.fromisoformat(ts_raw.replace("Z", "")) if ts_raw else self._as_of
        except (ValueError, AttributeError):
            ts = self._as_of
        return Quote(float(q.get("bp", 0)), float(q.get("ap", 0)), ts)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    daily = {
        "SPY": [
            make_bar("2026-01-02", 470, 472, 469, 471, 1e6),
            make_bar("2026-01-05", 471, 474, 470, 473, 1e6),
            make_bar("2026-01-06", 473, 475, 472, 474, 1e6),  # the "current" day
        ]
    }

    # as_of = close of 2026-01-06 -> all three bars known.
    mv_close = HistoricalMarketView(datetime(2026, 1, 6, 16, 0), daily=daily)
    bars = mv_close.daily_bars("SPY", 30)
    if len(bars) != 3 or bars[-1].date != "2026-01-06":
        print("FAIL: close-time view should know all 3 bars"); ok = False

    # as_of = open of 2026-01-06 (09:30) -> current day NOT yet known.
    mv_open = HistoricalMarketView(datetime(2026, 1, 6, 9, 30), daily=daily)
    bars_open = mv_open.daily_bars("SPY", 30)
    if len(bars_open) != 2 or bars_open[-1].date != "2026-01-05":
        print("FAIL: open-time view must not see the current day's bar"); ok = False

    # intraday_bar None during warmup, present after the window.
    intraday_bar = make_bar("2026-01-06", 473, 473.5, 472.5, 473.2, 1e5)
    mv_warm = HistoricalMarketView(
        datetime(2026, 1, 6, 9, 45), daily=daily, intraday={"SPY": intraday_bar}
    )
    if mv_warm.intraday_bar("SPY", 30) is not None:
        print("FAIL: intraday_bar must be None before the 30-min mark"); ok = False
    mv_after = HistoricalMarketView(
        datetime(2026, 1, 6, 10, 5), daily=daily, intraday={"SPY": intraday_bar}
    )
    if mv_after.intraday_bar("SPY", 30) is None:
        print("FAIL: intraday_bar should be available after the 30-min mark"); ok = False

    # option quote filtered by ts.
    q = Quote(1.10, 1.20, datetime(2026, 1, 6, 15, 0))
    mv_q = HistoricalMarketView(datetime(2026, 1, 6, 16, 0), quotes={"SPY260106C00475000": q})
    got = mv_q.option_quote("SPY260106C00475000")
    if not got or abs(got["mid"] - 1.15) > 1e-9:
        print("FAIL: option_quote mid wrong"); ok = False
    mv_q_early = HistoricalMarketView(datetime(2026, 1, 6, 14, 0), quotes={"SPY260106C00475000": q})
    if mv_q_early.option_quote("SPY260106C00475000") is not None:
        print("FAIL: option_quote must be None when quote ts > as_of"); ok = False

    # audit: nothing returned should be stamped after as_of.
    if any(rec["ts"] > mv_close.as_of for rec in mv_close.audit):
        print("FAIL: audit found a datum stamped after as_of"); ok = False

    print("market_view self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
