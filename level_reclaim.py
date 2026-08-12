"""
Oracle 2.1 — Feature 1: key-level reclaim / loss (ANALYTICS ONLY, pure).

``compute_level_reclaim`` turns a symbol's price relative to a handful of
"decision" levels into a single directional signal. Traders lean bullish when
price *reclaims* a level it had been below (crosses back above the 20-SMA,
prior-day high, or VWAP) and bearish when it *loses* one it had been holding.

Levels covered:
  * 10-bar and 20-bar simple moving averages (from ``daily_bars`` closes),
  * the prior session's high and low,
  * the session VWAP (when supplied, e.g. from ``intraday_features``).

Design rules (mirror regime.py / intraday_features.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer a safe omission over a guess; never raise on malformed input.
  * Offline-testable with synthetic bars.

Cross detection needs a "before" and "after" price. When ``intraday_bars`` is
supplied the current session is in progress, so ``cur``/``prev`` come from the
last two intraday closes (this captures an intraday reclaim of a *daily* level)
and the prior session is ``daily_bars[-1]``. With daily bars only, ``cur``/
``prev`` are the last two daily closes and the prior session is ``daily_bars[-2]``.

Bars are duck-typed: each element only needs ``.o/.h/.l/.c`` attributes.
"""

from typing import List, Optional

# How many bars back defines each SMA level.
SMA_FAST = 10
SMA_SLOW = 20

# Signal weight per level status. A fresh cross (reclaim/loss) counts far more
# than merely sitting on the right side of a level.
_SCORE = {
    "reclaimed": 1.0,
    "lost": -1.0,
    "holding_above": 0.25,
    "holding_below": -0.25,
}


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _g(bar, attr: str) -> Optional[float]:
    return _num(getattr(bar, attr, None))


def _closes(bars) -> List[float]:
    return [v for v in (_g(b, "c") for b in bars) if v is not None]


def _sma(closes: List[float], n: int) -> Optional[float]:
    if len(closes) < n or n <= 0:
        return None
    window = closes[-n:]
    return sum(window) / len(window)


def _status(prev: Optional[float], cur: Optional[float],
            level: Optional[float]) -> Optional[str]:
    """Classify ``cur`` (with the previous price ``prev``) against ``level``.

    Returns one of ``reclaimed`` / ``lost`` / ``holding_above`` /
    ``holding_below``, or ``None`` when undecidable (missing inputs or price
    sitting exactly on the level with no prior context).
    """
    if cur is None or level is None:
        return None
    if prev is not None:
        if prev <= level < cur:
            return "reclaimed"
        if prev >= level > cur:
            return "lost"
    if cur > level:
        return "holding_above"
    if cur < level:
        return "holding_below"
    return None


def compute_level_reclaim(daily_bars, intraday_bars=None,
                          vwap: Optional[float] = None) -> dict:
    """Directional key-level reclaim signal for a symbol.

    Returns ``{"reclaim_signal": float in [-1, 1], "reclaim_levels": {...}}``
    or ``{}`` when there is too little data. ``reclaim_levels`` maps each usable
    level name to ``{"level": float, "status": str}``. Never raises.
    """
    try:
        dbars = [b for b in (daily_bars or []) if b is not None]
        ibars = [b for b in (intraday_bars or []) if b is not None]
        closes = _closes(dbars)

        # Current / previous price and the prior-session bar depend on whether
        # a live intraday session is supplied.
        if ibars:
            iclose = _closes(ibars)
            if not iclose:
                return {}
            cur = iclose[-1]
            prev = iclose[-2] if len(iclose) >= 2 else None
            prior_session = dbars[-1] if dbars else None
        else:
            if len(closes) < 2:
                return {}
            cur = closes[-1]
            prev = closes[-2]
            prior_session = dbars[-2] if len(dbars) >= 2 else None

        levels = []  # (name, value)
        sma_fast = _sma(closes, SMA_FAST)
        sma_slow = _sma(closes, SMA_SLOW)
        if sma_fast is not None:
            levels.append(("sma10", sma_fast))
        if sma_slow is not None:
            levels.append(("sma20", sma_slow))
        if prior_session is not None:
            ph, pl = _g(prior_session, "h"), _g(prior_session, "l")
            if ph is not None:
                levels.append(("prior_high", ph))
            if pl is not None:
                levels.append(("prior_low", pl))
        vw = _num(vwap)
        if vw is not None:
            levels.append(("vwap", vw))

        detail = {}
        scores = []
        for name, value in levels:
            st = _status(prev, cur, value)
            if st is None:
                continue
            detail[name] = {"level": round(value, 4), "status": st}
            scores.append(_SCORE.get(st, 0.0))

        if not scores:
            return {}

        signal = max(-1.0, min(1.0, sum(scores) / len(scores)))
        return {"reclaim_signal": round(signal, 6), "reclaim_levels": detail}
    except Exception:  # pragma: no cover - fail-open
        return {}


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from market_view import make_bar, make_intraday_bar

    ok = True

    # Empty / garbage -> {}.
    for junk in (None, [], [None], "x", 7):
        if compute_level_reclaim(junk) != {}:
            print("FAIL: junk should yield {}", junk); ok = False

    # Daily-only: a steady climb sits above both SMAs (holding_above) -> +.
    up = [make_bar(f"2026-01-{i+1:02d}", 100 + i, 100 + i + 0.5,
                   99.5 + i, 100.5 + i) for i in range(22)]
    r_up = compute_level_reclaim(up)
    if not r_up or r_up["reclaim_signal"] <= 0:
        print("FAIL: uptrend should hold above its levels (positive)", r_up); ok = False
    if "sma20" not in r_up["reclaim_levels"]:
        print("FAIL: 22 bars should expose the 20-SMA level", r_up); ok = False

    # A fresh reclaim: price was below the SMAs, last close pops back above.
    flat = [make_bar(f"2026-02-{i+1:02d}", 100, 100.5, 99.5, 100.0)
            for i in range(20)]
    reclaim = flat + [make_bar("2026-02-21", 99, 101.5, 98.9, 101.4)]
    # Bring the prior close below the SMA, then cross above on the last bar.
    reclaim[-2] = make_bar("2026-02-20", 100, 100.1, 98.0, 98.2)
    r_rc = compute_level_reclaim(reclaim)
    if not r_rc or r_rc["reclaim_signal"] <= 0:
        print("FAIL: reclaim of the SMA should be bullish", r_rc); ok = False
    if not any(v["status"] == "reclaimed" for v in r_rc["reclaim_levels"].values()):
        print("FAIL: expected at least one 'reclaimed' level", r_rc); ok = False

    # A fresh loss: price was above, last close breaks below -> bearish.
    loss = flat + [make_bar("2026-02-21", 100, 100.1, 98.4, 98.5)]
    loss[-2] = make_bar("2026-02-20", 100, 102.0, 101.5, 101.8)
    r_ls = compute_level_reclaim(loss)
    if not r_ls or r_ls["reclaim_signal"] >= 0:
        print("FAIL: loss of the SMA should be bearish", r_ls); ok = False
    if not any(v["status"] == "lost" for v in r_ls["reclaim_levels"].values()):
        print("FAIL: expected at least one 'lost' level", r_ls); ok = False

    # Intraday session reclaiming VWAP + prior-day high.
    daily = [make_bar(f"2026-03-{i+1:02d}", 50, 50.6, 49.6, 50.2)
             for i in range(20)]
    # Prior session (last daily bar) high = 50.6; VWAP = 50.0.
    intra = [make_intraday_bar("2026-03-21T09:30:00Z", 49.8, 49.9, 49.7, 49.8, 1000),
             make_intraday_bar("2026-03-21T09:31:00Z", 49.8, 50.8, 49.8, 50.75, 1500)]
    r_int = compute_level_reclaim(daily, intraday_bars=intra, vwap=50.0)
    if not r_int or r_int["reclaim_signal"] <= 0:
        print("FAIL: intraday reclaim of VWAP/PDH should be bullish", r_int); ok = False
    if "vwap" not in r_int["reclaim_levels"]:
        print("FAIL: vwap level should appear when supplied", r_int); ok = False
    if r_int["reclaim_levels"]["vwap"]["status"] != "reclaimed":
        print("FAIL: intraday cross above VWAP -> reclaimed", r_int); ok = False

    # Bounded output.
    if not (-1.0 <= r_up["reclaim_signal"] <= 1.0):
        print("FAIL: signal out of range", r_up); ok = False

    print("level_reclaim self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
