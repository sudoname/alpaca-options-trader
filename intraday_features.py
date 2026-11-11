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

Phase 4 additions (same design rules):
  * ``ema_series`` — 9/20/50-period EMAs on session closes, SMA-seeded with
    documented warm-up (None until ``period`` completed bars exist).
  * ``compute_vwap_pullback`` — a reproducible VWAP-pullback-and-confirmation
    state machine with explicit tolerance and expiry.

SOURCE SETUP (kept identifiable): a VWAP-pullback continuation day-trade —
price trends on one side of session VWAP, pulls back to touch VWAP, and the
entry is the reclaim; the source uses the 9-EMA-over-20-EMA condition as the
long-side trend filter and plots the 50 EMA for context. ADAPTATIONS made
here: an explicit touch tolerance and confirmation expiry (the source leaves
both discretionary); the 50-EMA entry rule and the mirrored put (short) setup
are carried as SEPARATE hypotheses, not folded into the base signal. Only
completed bars are used; the session anchor is the bar list itself (callers
pass current-session bars only, as ``explain_context`` already does).
"""

from typing import List, Optional

# First N bars (== minutes for 1-min bars) that define the opening range.
OPENING_RANGE_BARS = 15

# Phase 4 defaults — explicit so replays and live read the same numbers.
EMA_PERIODS = (9, 20, 50)
VWAP_PULLBACK_TOLERANCE_PCT = 0.0005   # 0.05% band around VWAP counts as touch
VWAP_PULLBACK_EXPIRY_BARS = 10         # confirmation must arrive within N bars


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


def ema_series(values, period: int) -> List[Optional[float]]:
    """EMA over ``values`` with a documented, deterministic warm-up.

    Warm-up: entries ``0..period-2`` are ``None``; entry ``period-1`` is the
    SMA of the first ``period`` values (the seed); afterwards the standard
    recursive EMA with ``k = 2 / (period + 1)``. Returns a list the same
    length as ``values``. Fail-open: a non-numeric value anywhere, a bad
    period, or any error yields an all-``None`` list. Never raises.
    """
    try:
        vals = [_num(v) for v in (values or [])]
        n = len(vals)
        if not isinstance(period, int) or period <= 0 or any(v is None for v in vals):
            return [None] * n
        out: List[Optional[float]] = [None] * n
        if n < period:
            return out
        seed = sum(vals[:period]) / period
        out[period - 1] = seed
        k = 2.0 / (period + 1)
        prev = seed
        for i in range(period, n):
            prev = vals[i] * k + prev * (1.0 - k)
            out[i] = prev
        return out
    except Exception:  # pragma: no cover - fail-open
        return [None] * (len(values) if isinstance(values, (list, tuple)) else 0)


def _running_vwap(bars) -> List[Optional[float]]:
    """Per-bar running session VWAP (includes that bar); None until volume."""
    out: List[Optional[float]] = []
    num = den = 0.0
    for b in bars:
        v = _g(b, "v") or 0.0
        tp = _typical(b)
        if v and tp is not None:
            num += tp * v
            den += v
        out.append(num / den if den > 0 else None)
    return out


def compute_vwap_pullback(
    bars,
    *,
    direction: str = "long",
    tolerance_pct: float = VWAP_PULLBACK_TOLERANCE_PCT,
    expiry_bars: int = VWAP_PULLBACK_EXPIRY_BARS,
) -> dict:
    """VWAP-pullback-and-confirmation state machine over current-session bars.

    SOURCE SETUP: price trends on one side of session VWAP, pulls back to
    touch VWAP, and the entry is the reclaim. ADAPTATIONS: the touch is an
    explicit ``tolerance_pct`` band around the running VWAP and confirmation
    must arrive within ``expiry_bars`` bars of the touch (the source leaves
    both discretionary). ``direction="short"`` is the mirrored put-side
    hypothesis, carried separately per the master plan.

    Evaluated per completed bar (running VWAP includes that bar):
      * trend      — close beyond the far band (above it for long, below for
                     short)
      * pullback   — from trend/confirmed: the bar's extreme re-touches the
                     band while the close holds at or beyond the near band
                     (a touch without a decisive break)
      * confirmed  — from pullback: a close back beyond the far band within
                     ``expiry_bars`` bars of the touch
      * expired    — a pullback older than ``expiry_bars`` with no
                     confirmation
      * a decisive close through the opposite band resets to seeking_trend
    Repeated setups are allowed; the FINAL state is reported.

    Returns ``{}`` on empty/garbage input or a bad direction, else:
      ``direction``, ``state`` (seeking_trend|trend|pullback|confirmed|
      expired), ``confirmed_now`` (confirmation landed ON the final bar —
      the actionable entry), ``trend_index``, ``pullback_index``,
      ``trigger_index``, ``trigger_vwap``, ``tolerance_pct``,
      ``expiry_bars``. Never raises.
    """
    try:
        bars = [b for b in (bars or []) if b is not None]
        if not bars or direction not in ("long", "short"):
            return {}
        vwaps = _running_vwap(bars)
        state = "seeking_trend"
        trend_i: Optional[int] = None
        pullback_i: Optional[int] = None
        trigger_i: Optional[int] = None
        trigger_vwap: Optional[float] = None
        confirmed_now = False
        last_i = len(bars) - 1
        for i, b in enumerate(bars):
            vw, c = vwaps[i], _g(b, "c")
            if vw is None or c is None:
                continue
            hi_band = vw * (1.0 + tolerance_pct)
            lo_band = vw * (1.0 - tolerance_pct)
            if direction == "long":
                beyond = c > hi_band          # trending side holds
                broken = c < lo_band          # decisive break against setup
                ext = _g(b, "l")
                touched = ext is not None and ext <= hi_band and c >= lo_band
            else:
                beyond = c < lo_band
                broken = c > hi_band
                ext = _g(b, "h")
                touched = ext is not None and ext >= lo_band and c <= hi_band
            if state in ("seeking_trend", "expired"):
                if beyond:
                    state, trend_i = "trend", i
            elif state in ("trend", "confirmed"):
                if broken:
                    state = "seeking_trend"
                elif touched:
                    state, pullback_i = "pullback", i
            elif state == "pullback":
                if broken:
                    state = "seeking_trend"
                elif beyond and (i - pullback_i) <= expiry_bars:
                    state, trigger_i, trigger_vwap = "confirmed", i, vw
                    confirmed_now = i == last_i
                elif (i - pullback_i) > expiry_bars:
                    # Stale setup; a close beyond the band restarts a trend.
                    state = "expired"
                    if beyond:
                        state, trend_i = "trend", i
        return {
            "direction": direction,
            "state": state,
            "confirmed_now": bool(confirmed_now and state == "confirmed"),
            "trend_index": trend_i,
            "pullback_index": pullback_i,
            "trigger_index": trigger_i,
            "trigger_vwap": round(trigger_vwap, 4) if trigger_vwap is not None else None,
            "tolerance_pct": tolerance_pct,
            "expiry_bars": expiry_bars,
        }
    except Exception:  # pragma: no cover - fail-open
        return {}


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
      * ``ema_9`` / ``ema_20`` / ``ema_50`` (omitted during warm-up)
      * ``ema_9_over_20`` (+1/-1/0) and ``price_vs_ema_50`` (+1/-1/0;
        separate hypothesis, not folded into the base signal)
      * ``vwap_pullback_long`` / ``vwap_pullback_short`` (1 only when the
        pullback confirmation landed on the FINAL bar — actionable entry)
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

        # --- Phase 4: EMA stack (9/20/50) on session closes ---------------- #
        emas: dict = {}
        for period in EMA_PERIODS:
            series = ema_series(closes, period)
            if series and series[-1] is not None:
                emas[period] = series[-1]
                feats[f"ema_{period}"] = round(series[-1], 4)
        e9, e20, e50 = emas.get(9), emas.get(20), emas.get(50)
        if e9 is not None and e20 is not None:
            feats["ema_9_over_20"] = 1 if e9 > e20 else (-1 if e9 < e20 else 0)
        if last_close is not None and e50 is not None:
            feats["price_vs_ema_50"] = (
                1 if last_close > e50 else (-1 if last_close < e50 else 0))

        # --- Phase 4: VWAP pullback confirmation flags --------------------- #
        # Only meaningful when the session has a usable VWAP.
        if "vwap" in feats:
            for d, key in (("long", "vwap_pullback_long"),
                           ("short", "vwap_pullback_short")):
                pb = compute_vwap_pullback(bars, direction=d)
                if pb:
                    feats[key] = 1 if pb.get("confirmed_now") else 0

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

    # --- Phase 4: EMA warm-up, seeding, and ordering ----------------------- #
    es = ema_series([1.0] * 12, 9)
    if es[:8] != [None] * 8 or es[8] != 1.0 or es[-1] != 1.0:
        print("FAIL: constant series EMA == constant after warm-up", es); ok = False
    if ema_series([1.0, "x", 3.0], 2) != [None, None, None]:
        print("FAIL: garbage value should void the EMA series"); ok = False
    if ema_series([1.0, 2.0], 0) != [None, None]:
        print("FAIL: nonpositive period -> all-None series"); ok = False
    if ema_series([], 9) != []:
        print("FAIL: empty values -> empty series"); ok = False
    if f.get("ema_9") is None or f.get("ema_20") is None:
        print("FAIL: 30-bar rally should carry ema_9/ema_20", f); ok = False
    if "ema_50" in f:
        print("FAIL: ema_50 must stay absent during warm-up (30 bars)", f); ok = False
    if f.get("ema_9_over_20") != 1:
        print("FAIL: rally -> ema_9_over_20 = 1", f); ok = False
    if fd.get("ema_9_over_20") != -1:
        print("FAIL: downtrend -> ema_9_over_20 = -1", fd); ok = False
    if f.get("vwap_pullback_long") != 0:
        print("FAIL: no final-bar confirmation -> vwap_pullback_long = 0", f); ok = False

    # --- Phase 4: VWAP pullback state machine ------------------------------ #
    def _bar(i, o, h, l, c):
        return make_intraday_bar(f"2026-01-06T10:{i:02d}:00Z", o, h, l, c, 1000)

    trend = [_bar(i, 100 + 0.5 * i, 100.6 + 0.5 * i, 99.9 + 0.5 * i,
                  100.5 + 0.5 * i) for i in range(5)]
    pull = [_bar(5, 102.4, 102.6, 101.0, 101.4)]       # touch, close holds band
    confirm = [_bar(6, 101.5, 102.2, 101.4, 102.0)]    # reclaim above band

    pb = compute_vwap_pullback(trend + pull + confirm)
    if pb.get("state") != "confirmed" or pb.get("confirmed_now") is not True:
        print("FAIL: trend->touch->reclaim should confirm on last bar", pb); ok = False
    if pb.get("pullback_index") != 5 or pb.get("trigger_index") != 6:
        print("FAIL: pullback/trigger indexes", pb); ok = False

    fpb = compute_intraday_features(trend + pull + confirm)
    if fpb.get("vwap_pullback_long") != 1:
        print("FAIL: final-bar confirmation -> vwap_pullback_long = 1", fpb); ok = False

    # Confirmation must arrive within the expiry window.
    drift = [_bar(7 + j, 101.39, 101.39, 101.39, 101.39) for j in range(12)]
    pe = compute_vwap_pullback(trend + pull + drift)
    if pe.get("state") != "expired":
        print("FAIL: stale pullback should expire", pe); ok = False

    # A decisive close through the far band invalidates the setup.
    brk = [_bar(6, 101.3, 101.4, 99.8, 100.0)]
    pi = compute_vwap_pullback(trend + pull + brk)
    if pi.get("state") != "seeking_trend" or pi.get("confirmed_now"):
        print("FAIL: break below vwap should reset the setup", pi); ok = False

    # Mirrored short setup (separate hypothesis) confirms on the way down.
    dtrend = [_bar(i, 100 - 0.5 * i, 100.1 - 0.5 * i, 99.4 - 0.5 * i,
                   99.5 - 0.5 * i) for i in range(5)]
    dpull = [_bar(5, 97.6, 99.0, 97.4, 98.6)]
    dconf = [_bar(6, 98.5, 98.6, 97.8, 98.0)]
    ps = compute_vwap_pullback(dtrend + dpull + dconf, direction="short")
    if ps.get("state") != "confirmed" or ps.get("confirmed_now") is not True:
        print("FAIL: mirrored short pullback should confirm", ps); ok = False

    if compute_vwap_pullback(trend, direction="sideways") != {}:
        print("FAIL: bad direction -> {}"); ok = False
    if compute_vwap_pullback(None) != {}:
        print("FAIL: no bars -> {}"); ok = False

    print("intraday_features self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
