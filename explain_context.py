"""
explain_context.py — READ-ONLY evidence-context builder for Oracle "explain".

Given a ticker symbol, assemble the ``ctx`` dict the Oracle agents
(``oracle_agents.run_agents``) consume, using only read-only Alpaca market data:
daily bars fetched via ``GET /v2/stocks/{symbol}/bars`` through a
``market_view.LiveMarketView``. From those bars it derives

  * trend / momentum / realized_vol / regime  (via ``regime.detect_regime``),
  * a recent volume ratio (last bar vs the trailing average),
  * relative strength vs SPY (n-day return spread), and
  * the primary candlestick pattern (``oracle.signals.candlestick_patterns``).

Every field is optional — the agents tolerate missing keys and vote neutral for
whatever is absent. Without this context the dashboard's explain endpoint always
returned INSUFFICIENT_DATA because no evidence was ever assembled.

This module is read-only: it issues only HTTP GETs for market data and never
writes, trades, or mutates any state. On missing creds / no network / any error
it FAILS OPEN to ``{}`` so explain degrades exactly as before. The market-view
factory is injectable so unit tests run fully offline.
"""

from statistics import mean
from typing import Callable, Dict, List, Optional

DEFAULT_LOOKBACK = 30
REL_STRENGTH_WINDOW = 10
VOLUME_WINDOW = 5


def _alpaca_headers() -> Optional[Dict[str, str]]:
    """Read-only Alpaca auth headers from config, or None when creds are absent."""
    try:
        from config_loader import ConfigLoader
        env = ConfigLoader()
        key = env.get("ALPACA_API_KEY", "")
        secret = env.get("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return None
        return {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
    except Exception:
        return None


def _default_market_view_factory():
    """Build a live, read-only market view (daily bars via GET). None w/o creds."""
    headers = _alpaca_headers()
    if not headers:
        return None
    try:
        from config_loader import ConfigLoader
        feed = ConfigLoader().get("SCREENER_ALPACA_FEED", "iex") or "iex"
    except Exception:
        feed = "iex"
    from market_view import LiveMarketView
    return LiveMarketView(headers=headers, feed=feed)


def _closes(bars) -> List[float]:
    return [b.c for b in bars if getattr(b, "c", None) is not None]


def _n_day_return(bars, n: int) -> Optional[float]:
    closes = _closes(bars)
    if len(closes) <= n or not closes[-1 - n]:
        return None
    return (closes[-1] - closes[-1 - n]) / closes[-1 - n]


def _volume_ratio(bars) -> Optional[float]:
    vols = [b.v for b in bars if getattr(b, "v", None)]
    if len(vols) < VOLUME_WINDOW + 1:
        return None
    trailing = vols[-(VOLUME_WINDOW + 1):-1]
    avg = mean(trailing) if trailing else 0.0
    if not avg:
        return None
    return vols[-1] / avg


def _candlestick(bars) -> Optional[dict]:
    try:
        from oracle.signals.candlestick_patterns import detect_primary
        stamp = detect_primary(bars)
        return stamp.to_dict() if stamp is not None else None
    except Exception:
        return None


def build_explain_context(
    symbol: str,
    *,
    market_view_factory: Optional[Callable[[], object]] = None,
) -> Dict:
    """Assemble the agent evidence ``ctx`` for ``symbol`` from read-only data.

    Returns a ``ctx`` dict (possibly partial) or ``{}`` when no data/creds are
    available. Never raises. ``market_view_factory`` is injectable for offline
    tests; it must return an object exposing ``daily_bars(symbol, lookback)``.
    """
    try:
        factory = market_view_factory or _default_market_view_factory
        mv = factory()
        if mv is None:
            return {}

        bars = mv.daily_bars(symbol, DEFAULT_LOOKBACK)
        if not bars:
            return {}

        ctx: Dict = {}

        # trend / momentum / realized_vol / regime — reuse the project's labeler.
        try:
            from regime import detect_regime
            reg = detect_regime(mv, symbol)
            if reg.get("trend") in ("up", "down"):
                ctx["trend"] = reg["trend"]
            if reg.get("momentum") is not None:
                ctx["momentum"] = reg["momentum"]
            if reg.get("realized_vol") is not None:
                ctx["realized_vol"] = reg["realized_vol"]
            if reg.get("regime"):
                ctx["regime"] = reg["regime"]
        except Exception:
            pass

        vr = _volume_ratio(bars)
        if vr is not None:
            ctx["volume_ratio"] = round(vr, 4)

        # Relative strength vs SPY (skip the spread when the symbol IS SPY).
        sym_ret = _n_day_return(bars, REL_STRENGTH_WINDOW)
        if sym_ret is not None:
            if symbol.upper() == "SPY":
                ctx["rel_strength"] = 0.0
            else:
                try:
                    spy_bars = mv.daily_bars("SPY", DEFAULT_LOOKBACK)
                    spy_ret = _n_day_return(spy_bars, REL_STRENGTH_WINDOW)
                    if spy_ret is not None:
                        ctx["rel_strength"] = round(sym_ret - spy_ret, 4)
                except Exception:
                    pass

        cs = _candlestick(bars)
        if cs:
            ctx["candlestick"] = cs

        return ctx
    except Exception:
        return {}


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses a synthetic offline market view)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime
    from market_view import HistoricalMarketView, make_bar

    ok = True

    # No creds / no factory output -> fail open to {} (no network).
    if build_explain_context("SPY", market_view_factory=lambda: None) != {}:
        print("FAIL: None market view should yield {}"); ok = False

    # A raising factory must still fail open.
    def _boom():
        raise RuntimeError("network down")
    if build_explain_context("SPY", market_view_factory=_boom) != {}:
        print("FAIL: raising factory should yield {}"); ok = False

    # Synthetic steady uptrend for the symbol, flat SPY -> populated ctx with a
    # real trend, a volume ratio, and positive relative strength.
    up = [make_bar(f"2026-01-{i+1:02d}", 100 + i, 100 + i + 0.6,
                   99.6 + i, 100.5 + i, 1_000_000 + (50_000 if i == 11 else 0))
          for i in range(12)]
    spy = [make_bar(f"2026-01-{i+1:02d}", 400, 401, 399, 400, 1_000_000)
           for i in range(12)]
    mv = HistoricalMarketView(datetime(2026, 1, 31, 16, 0),
                              daily={"AAA": up, "SPY": spy})

    ctx = build_explain_context("AAA", market_view_factory=lambda: mv)
    if not ctx:
        print("FAIL: populated bars should yield a non-empty ctx:", ctx); ok = False
    if ctx.get("trend") != "up":
        print("FAIL: steady rise should give trend=up:", ctx); ok = False
    if "momentum" not in ctx or ctx["momentum"] <= 0:
        print("FAIL: rising series should have positive momentum:", ctx); ok = False
    if "realized_vol" not in ctx:
        print("FAIL: ctx should carry realized_vol:", ctx); ok = False
    if "volume_ratio" not in ctx:
        print("FAIL: ctx should carry volume_ratio:", ctx); ok = False
    if "rel_strength" not in ctx or ctx["rel_strength"] <= 0:
        print("FAIL: rising symbol vs flat SPY -> positive rel_strength:", ctx)
        ok = False

    # The assembled ctx must actually move the agents off neutral.
    try:
        import oracle_intelligence_reports as oir
        rep = oir.compute_oracle_explain("AAA", ctx=ctx)
        if rep.get("verdict") != "OK":
            print("FAIL: explain with real ctx should be OK:", rep.get("verdict"))
            ok = False
    except Exception as ex:
        print("FAIL: compute_oracle_explain integration:", ex); ok = False

    # SPY relative strength is pinned to 0.0 (no self-spread).
    ctx_spy = build_explain_context("SPY", market_view_factory=lambda: mv)
    if ctx_spy.get("rel_strength") != 0.0:
        print("FAIL: SPY rel_strength should be 0.0:", ctx_spy); ok = False

    print("explain_context self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
