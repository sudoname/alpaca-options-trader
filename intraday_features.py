"""
Oracle 2.1 — intraday session features (ANALYTICS ONLY, pure, no I/O).

``compute_intraday_features`` turns a list of intraday (1-minute) bars for the
CURRENT session into the handful of same-day signals the intraday trading
profile actually cares about — opening-range break, overnight gap and whether
it filled, session VWAP and which side price sits on, and 1-/5-bar momentum.
These replace the coarse daily-only inputs that made the engine label a fresh
intraday setup off a 5-day trend.

Design rules (mirror oracle/mode.py, regime.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer a safe omission over a guess; never raise on malformed input.
  * Offline-testable with synthetic bars.

Bars are duck-typed: each element only needs ``.o/.h/.l/.c/.v`` attributes
(``market_view.Bar`` satisfies this), so the same code runs live and in backtest.
Every field is optional — a consumer (``IntradayAgent``) votes neutral for
whatever is absent.
"""

from typing import List, Optional

# First N bars (== minutes for 1-min bars) that define the opening range.
OPENING_RANGE_BARS = 15


def _num(x) -> Optional[float]:
    try:
        f = float(x)
        return f
    except (TypeError, ValueError):
        return None


def _g(bar, attr: str) -> Optional[float]:
    return _num(getattr(bar, attr, None))


def _typical(bar) -> Optional[float]:
    h, l, c = _g(bar, "h"), _g(bar, "l"), _g(bar, "c")
    if h is None or l is None or c is None:
        return None
    return (h + l + c) / 3.0


def _highs(bars) -> List[float]:
    return [v for v in (_g(b, "h") for b in bars) if v is not None]


def _lows(bars) -> List[float]:
    return [v for v in (_g(b, "l") for b in bars) if v is not None]


def compute_intraday_features(
    bars,
    prior_close: Optional[float] = None,
    prior_high: Optional[float] = None,
    prior_low: Optional[float] = None,
    *,
    opening_range_bars: int = OPENING_RANGE_BARS,
) -> dict:
    """Same-session intraday features from ``bars`` (current session, ascending).

    Returns a possibly-partial dict; ``{}`` on empty/garbage input. Keys:
      * ``vwap`` / ``vwap_reclaim`` (+1 above, -1 below, 0 on it)
      * ``gap_pct`` / ``gap_filled`` (needs ``prior_close``)
      * ``opening_range_break`` (+1 above OR high, -1 below OR low, 0 inside)
      * ``intraday_momentum_1m`` / ``intraday_momentum_5m``
      * ``prior_day_break`` (+1 above prior-day high, -1 below prior-day low)
    Never raises.
    """
    try:
        bars = [b for b in (bars or []) if b is not None]
        if not bars:
            return {}
        feats: dict = {}
        last_close = _g(bars[-1], "c")

        # --- VWAP + which side price sits on ------------------------------ #
        num = den = 0.0
        for b in bars:
            v = _g(b, "v") or 0.0
            tp = _typical(b)
            if v and tp is not None:
                num += tp * v
                den += v
        if den > 0:
            vwap = num / den
            feats["vwap"] = round(vwap, 4)
            if last_close is not None:
                feats["vwap_reclaim"] = (
                    1 if last_close > vwap else (-1 if last_close < vwap else 0))

        # --- Overnight gap + whether it filled ---------------------------- #
        first_open = _g(bars[0], "o")
        pc = _num(prior_close)
        if pc and first_open is not None:
            feats["gap_pct"] = round((first_open - pc) / pc, 6)
            highs, lows = _highs(bars), _lows(bars)
            if first_open > pc:
                feats["gap_filled"] = bool(lows) and min(lows) <= pc
            elif first_open < pc:
                feats["gap_filled"] = bool(highs) and max(highs) >= pc
            else:
                feats["gap_filled"] = True

        # --- Opening-range break ------------------------------------------ #
        orb = bars[:opening_range_bars] if len(bars) > opening_range_bars else bars
        or_highs, or_lows = _highs(orb), _lows(orb)
        if last_close is not None and or_highs and or_lows:
            or_hi, or_lo = max(or_highs), min(or_lows)
            feats["opening_range_break"] = (
                1 if last_close > or_hi else (-1 if last_close < or_lo else 0))

        # --- Short-horizon momentum --------------------------------------- #
        closes = [v for v in (_g(b, "c") for b in bars) if v is not None]

        def _mom(n: int) -> Optional[float]:
            if len(closes) <= n or not closes[-1 - n]:
                return None
            return round((closes[-1] - closes[-1 - n]) / closes[-1 - n], 6)

        m1, m5 = _mom(1), _mom(5)
        if m1 is not None:
            feats["intraday_momentum_1m"] = m1
        if m5 is not None:
            feats["intraday_momentum_5m"] = m5

        # --- Prior-day range breach (uses prior_high / prior_low) --------- #
        ph, pl = _num(prior_high), _num(prior_low)
        if last_close is not None and ph is not None and pl is not None:
            feats["prior_day_break"] = (
                1 if last_close > ph else (-1 if last_close < pl else 0))

        return feats
    except Exception:  # pragma: no cover - fail-open
        return {}


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from market_view import make_intraday_bar

    ok = True

    # Empty / garbage -> {}.
    for junk in (None, [], [None], "x", 7):
        if compute_intraday_features(junk) != {}:
            print("FAIL: junk should yield {}", junk); ok = False

    # A steady intraday rally from 100 to ~101 with rising volume.
    up = [make_intraday_bar(f"2026-01-06T{9 + (30 + i) // 60:02d}:"
                            f"{(30 + i) % 60:02d}:00Z",
                            100 + i * 0.05, 100 + i * 0.05 + 0.03,
                            100 + i * 0.05 - 0.03, 100 + i * 0.05 + 0.02,
                            1000 + i * 10)
          for i in range(30)]
    f = compute_intraday_features(up, prior_close=99.5,
                                  prior_high=100.2, prior_low=99.0)
    if "vwap" not in f:
        print("FAIL: rising session should have a vwap", f); ok = False
    if f.get("vwap_reclaim") != 1:
        print("FAIL: last close above vwap -> reclaim=1", f); ok = False
    if f.get("gap_pct") is None or f["gap_pct"] <= 0:
        print("FAIL: open above prior close -> positive gap", f); ok = False
    if f.get("opening_range_break") != 1:
        print("FAIL: rally should break the opening range up", f); ok = False
    if f.get("intraday_momentum_5m") is None or f["intraday_momentum_5m"] <= 0:
        print("FAIL: rising series -> positive 5m momentum", f); ok = False
    if f.get("prior_day_break") != 1:
        print("FAIL: close above prior-day high -> prior_day_break=1", f); ok = False

    # Gap up that fills: open above prior close, then trades back below it.
    fill = [make_intraday_bar("2026-01-06T09:30:00Z", 101, 101.2, 100.9, 101.0, 1000),
            make_intraday_bar("2026-01-06T09:31:00Z", 101, 101.1, 99.8, 99.9, 1200)]
    ff = compute_intraday_features(fill, prior_close=100.0)
    if ff.get("gap_pct") is None or ff["gap_pct"] <= 0:
        print("FAIL: gap up expected", ff); ok = False
    if ff.get("gap_filled") is not True:
        print("FAIL: price traded back through prior close -> gap_filled", ff); ok = False

    # A downtrend sits below VWAP with negative momentum.
    down = [make_intraday_bar(f"2026-01-06T{9 + (30 + i) // 60:02d}:"
                              f"{(30 + i) % 60:02d}:00Z",
                              100 - i * 0.05, 100 - i * 0.05 + 0.03,
                              100 - i * 0.05 - 0.03, 100 - i * 0.05 - 0.02,
                              1000 + i * 10)
            for i in range(30)]
    fd = compute_intraday_features(down, prior_close=100.5)
    if fd.get("vwap_reclaim") != -1:
        print("FAIL: downtrend last close below vwap -> reclaim=-1", fd); ok = False
    if fd.get("opening_range_break") != -1:
        print("FAIL: downtrend should break the opening range down", fd); ok = False

    print("intraday_features self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
