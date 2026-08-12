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
    mode: Optional[str] = None,
    mode_aware_trend: bool = False,
    intraday_features: bool = False,
    orderbook_imbalance: bool = False,
    level_reclaim: bool = False,
    options_flow: bool = False,
    dealer_gamma: bool = False,
    market_view_factory: Optional[Callable[[], object]] = None,
    chain_fetch: Optional[Callable[[str], List[dict]]] = None,
) -> Dict:
    """Assemble the agent evidence ``ctx`` for ``symbol`` from read-only data.

    Returns a ``ctx`` dict (possibly partial) or ``{}`` when no data/creds are
    available. Never raises. ``market_view_factory`` is injectable for offline
    tests; it must return an object exposing ``daily_bars(symbol, lookback)``.

    Every enrichment is independently gated so any single feature flag can be
    toggled without perturbing the others:

      * ``mode`` (``"intraday"``/``"swing"``) — when set, stamps ``ctx["mode"]``
        + ``ctx["trend_horizon"]`` so mode-gated agents can self-gate. This is
        INERT for trend numbers on its own (agents merely read the stamp).
      * ``mode_aware_trend`` — when True *and* ``mode`` is set, derives the
        trend/momentum from the mode's feature source/span (the trend-horizon
        bug fix). This is the ONLY switch that changes the trend numbers.
      * ``intraday_features`` — when True and the resolved mode is intraday,
        folds :func:`intraday_features.compute_intraday_features` into ctx.
      * ``orderbook_imbalance`` — when True and the resolved mode is intraday,
        folds Robinhood price-book imbalance into ctx.
      * ``level_reclaim`` — when True, folds :func:`level_reclaim`.
        ``compute_level_reclaim`` (reclaim/loss of SMA / prior-day H-L / VWAP)
        into ctx. Works in any mode; uses intraday bars + VWAP when available.
      * ``options_flow`` — when True, fetches an option-chain snapshot and folds
        :func:`options_flow.compute_flow_features` (call/put skew, unusual
        volume, IV term structure) into ctx.
      * ``dealer_gamma`` — when True, folds :func:`oracle.gex.compute_gex`
        (net dealer GEX, flip point, regime) into ctx from the same snapshot.

    ``chain_fetch`` is an injectable ``symbol -> [chain_row dict]`` used by the
    options_flow / dealer_gamma enrichment; when omitted a read-only Alpaca
    options-snapshot fetch is used (and skipped without creds). With every flag
    at its default (``mode=None`` and all booleans False) the result is
    byte-identical to the historical daily-only ctx.
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

        # Resolve the mode descriptor once; stamping and detection are decoupled.
        desc = None
        if mode is not None:
            try:
                from regime import bars_for_mode
                desc = bars_for_mode(mode)
                ctx["mode"] = desc["mode"]
                ctx["trend_horizon"] = desc["mode"]
            except Exception:
                desc = None
        resolved_mode = desc["mode"] if desc else None

        # trend / momentum / realized_vol / regime — reuse the project's labeler.
        try:
            from regime import detect_regime
            if mode_aware_trend and desc is not None:
                reg = detect_regime(mv, symbol,
                                    vol_lookback=desc["vol_lookback"],
                                    momentum_bars=desc["momentum_bars"],
                                    source=desc["source"])
            else:
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

        # --- Intraday session features (intraday mode only) --------------- #
        if intraday_features and resolved_mode == "intraday":
            try:
                from intraday_features import compute_intraday_features
                ibars = (mv.intraday_bars(symbol)
                         if hasattr(mv, "intraday_bars") else None)
                if ibars:
                    prior_close = prior_high = prior_low = None
                    try:
                        d2 = mv.daily_bars(symbol, 2)
                        if d2 and len(d2) >= 2:
                            pd_bar = d2[-2]
                            prior_close = getattr(pd_bar, "c", None)
                            prior_high = getattr(pd_bar, "h", None)
                            prior_low = getattr(pd_bar, "l", None)
                    except Exception:
                        pass
                    feats = compute_intraday_features(
                        ibars, prior_close=prior_close,
                        prior_high=prior_high, prior_low=prior_low)
                    if feats:
                        ctx.update(feats)
            except Exception:
                pass

        # --- Order-book imbalance (intraday mode only) -------------------- #
        if orderbook_imbalance and resolved_mode == "intraday":
            try:
                from rh_price_book import get_order_book_imbalance
                ob = get_order_book_imbalance(symbol)
                if ob:
                    ctx.update(ob)
            except Exception:
                pass

        # --- Key-level reclaim / loss (any mode) -------------------------- #
        if level_reclaim:
            try:
                from level_reclaim import compute_level_reclaim
                ibars = None
                if resolved_mode == "intraday" and hasattr(mv, "intraday_bars"):
                    try:
                        ibars = mv.intraday_bars(symbol) or None
                    except Exception:
                        ibars = None
                lr = compute_level_reclaim(bars, intraday_bars=ibars,
                                           vwap=ctx.get("vwap"))
                if lr:
                    ctx.update(lr)
            except Exception:
                pass

        # --- Options flow / dealer gamma (any mode) ----------------------- #
        if options_flow or dealer_gamma:
            try:
                spot = None
                try:
                    spot = float(getattr(bars[-1], "c", None))
                except (TypeError, ValueError):
                    spot = None

                chain = None
                if chain_fetch is not None:
                    try:
                        chain = chain_fetch(symbol)
                    except Exception:
                        chain = None
                else:
                    headers = _alpaca_headers()
                    if headers:
                        try:
                            from config_loader import ConfigLoader
                            feed = (ConfigLoader().get(
                                "ALPACA_OPTIONS_FEED", "indicative")
                                or "indicative")
                        except Exception:
                            feed = "indicative"
                        try:
                            from options_flow import fetch_chain_snapshot
                            chain = fetch_chain_snapshot(
                                symbol, headers=headers, feed=feed, spot=spot)
                        except Exception:
                            chain = None

                if chain:
                    if options_flow:
                        try:
                            from options_flow import compute_flow_features
                            ff = compute_flow_features(chain)
                            if ff.get("flow_status") == "OK":
                                ctx.update({k: v for k, v in ff.items()
                                            if k != "flow_status"})
                        except Exception:
                            pass
                    if dealer_gamma:
                        try:
                            from oracle.gex import compute_gex
                            gx = compute_gex(chain, spot)
                            if gx.get("gex_status") == "OK":
                                ctx.update({k: v for k, v in gx.items()
                                            if k != "gex_status"})
                        except Exception:
                            pass
            except Exception:
                pass

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

    # mode=None (default) must NOT stamp mode/trend_horizon (byte-identical).
    if "mode" in ctx or "trend_horizon" in ctx:
        print("FAIL: default mode should not stamp mode/trend_horizon:", ctx); ok = False

    # mode='intraday' stamps the horizon so the intraday agent can self-gate.
    ctx_intra = build_explain_context("AAA", mode="intraday",
                                      market_view_factory=lambda: mv)
    if ctx_intra.get("mode") != "intraday" or ctx_intra.get("trend_horizon") != "intraday":
        print("FAIL: intraday mode should stamp mode/trend_horizon:", ctx_intra); ok = False

    # mode alone must NOT change the trend numbers (stamp/detection decoupled).
    if ctx_intra.get("trend") != ctx.get("trend") or \
            ctx_intra.get("momentum") != ctx.get("momentum"):
        print("FAIL: mode alone should not change trend detection:", ctx_intra); ok = False

    # mode='swing' resolves to the swing horizon.
    ctx_swing = build_explain_context("AAA", mode="swing",
                                      market_view_factory=lambda: mv)
    if ctx_swing.get("trend_horizon") != "swing":
        print("FAIL: swing mode should stamp trend_horizon=swing:", ctx_swing); ok = False

    # mode_aware_trend=True still stamps the horizon (detection may fall back to
    # daily here because mv carries no intraday series).
    ctx_mat = build_explain_context("AAA", mode="intraday", mode_aware_trend=True,
                                    market_view_factory=lambda: mv)
    if ctx_mat.get("trend_horizon") != "intraday":
        print("FAIL: mode_aware_trend should still stamp horizon:", ctx_mat); ok = False

    # intraday_features flag folds intraday session keys into ctx (intraday mode
    # only) when the market view exposes an intraday series.
    from market_view import make_intraday_bar
    ibars = [make_intraday_bar(
                 f"2026-01-31T{9 + (30 + i) // 60:02d}:{(30 + i) % 60:02d}:00Z",
                 100 + i * 0.05, 100 + i * 0.05 + 0.03,
                 100 + i * 0.05 - 0.03, 100 + i * 0.05 + 0.02,
                 1000 + i * 10)
             for i in range(30)]
    mv_intra = HistoricalMarketView(datetime(2026, 1, 31, 16, 0),
                                    daily={"AAA": up, "SPY": spy},
                                    intraday_series={"AAA": ibars})
    ctx_if = build_explain_context("AAA", mode="intraday", intraday_features=True,
                                   market_view_factory=lambda: mv_intra)
    if "vwap" not in ctx_if or "opening_range_break" not in ctx_if:
        print("FAIL: intraday_features should fold intraday keys into ctx:", ctx_if)
        ok = False

    # intraday_features must be inert in swing mode (no intraday keys leak in).
    ctx_sw_if = build_explain_context("AAA", mode="swing", intraday_features=True,
                                      market_view_factory=lambda: mv_intra)
    if "vwap" in ctx_sw_if or "opening_range_break" in ctx_sw_if:
        print("FAIL: intraday_features must not populate in swing mode:", ctx_sw_if)
        ok = False

    # level_reclaim flag folds a reclaim signal into ctx (steady uptrend holds
    # above its levels -> non-negative signal).
    ctx_lr = build_explain_context("AAA", level_reclaim=True,
                                   market_view_factory=lambda: mv)
    if "reclaim_signal" not in ctx_lr or "reclaim_levels" not in ctx_lr:
        print("FAIL: level_reclaim should fold reclaim keys into ctx:", ctx_lr)
        ok = False

    # level_reclaim OFF (default) must not stamp reclaim keys.
    if "reclaim_signal" in ctx:
        print("FAIL: default ctx should not carry reclaim_signal:", ctx); ok = False

    # options_flow / dealer_gamma fold namespaced chain features into ctx via an
    # injected chain fetch (no network). Spot ~= last close of `up` (111.5); the
    # synthetic chain is call-heavy with a cumulative-GEX crossing near spot.
    def _fake_chain(_symbol):
        return [
            {"type": "call", "strike": 110, "expiration_date": "2026-02-20",
             "volume": 9000, "open_interest": 4000, "gamma": 0.05, "iv": 0.35},
            {"type": "put", "strike": 110, "expiration_date": "2026-02-20",
             "volume": 1000, "open_interest": 1500, "gamma": 0.05, "iv": 0.34},
            {"type": "call", "strike": 115, "expiration_date": "2026-03-20",
             "volume": 500, "open_interest": 6000, "gamma": 0.05, "iv": 0.28},
            {"type": "put", "strike": 105, "expiration_date": "2026-03-20",
             "volume": 300, "open_interest": 3000, "gamma": 0.05, "iv": 0.29},
        ]
    ctx_of = build_explain_context("AAA", options_flow=True, dealer_gamma=True,
                                   market_view_factory=lambda: mv,
                                   chain_fetch=_fake_chain)
    if "cp_volume_skew" not in ctx_of or ctx_of["cp_volume_skew"] <= 0:
        print("FAIL: options_flow should fold call-heavy skew into ctx:", ctx_of)
        ok = False
    if "gex_regime" not in ctx_of or "gex_total" not in ctx_of:
        print("FAIL: dealer_gamma should fold GEX keys into ctx:", ctx_of); ok = False

    # Both flags OFF (default) must not stamp any flow / gamma keys.
    for k in ("cp_volume_skew", "cp_oi_skew", "unusual_volume",
              "iv_term_structure", "gex_regime", "gex_total"):
        if k in ctx:
            print(f"FAIL: default ctx should not carry {k}:", ctx); ok = False

    # A raising chain_fetch must fail open (flags on, but no keys leak / no raise).
    def _bad_chain(_symbol):
        raise RuntimeError("chain down")
    ctx_ofb = build_explain_context("AAA", options_flow=True, dealer_gamma=True,
                                    market_view_factory=lambda: mv,
                                    chain_fetch=_bad_chain)
    if "cp_volume_skew" in ctx_ofb or "gex_regime" in ctx_ofb:
        print("FAIL: failing chain_fetch should leave flow/gamma keys absent:",
              ctx_ofb); ok = False

    print("explain_context self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
