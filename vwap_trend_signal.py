"""Point-in-time VWAP / EMA / RSI trend state on the underlying (RESEARCH ONLY).

Computes the directional state of the underlying at an Oracle decision
timestamp from 5-minute bars, using only candles that had COMPLETED by that
timestamp. Pure functions: no network, no credentials, no file writes, no env
reads. Never submits orders. Mirrors the design rules of intraday_features.py.

Signal definitions (research defaults, recorded before any holdout look):

  * Candle timeframe: 5 minutes (explicit default; the source chart leaves
    the timeframe open).
  * Session VWAP: cumulative sum(bar_vwap * volume) / sum(volume) over the
    CURRENT regular session only, resetting at the session open. When the
    vendor per-bar ``vwap`` field is present on every contributing bar the
    series is labeled ``trade_derived``; otherwise the typical price
    (h+l+c)/3 substitutes and the series is labeled ``typical_price``.
  * EMA(20) on completed-bar closes, SMA-seeded with the same warm-up
    contract as intraday_features.ema_series (None until ``period`` closes).
  * RSI(14), Wilder smoothing: first value after ``period+1`` closes using
    the simple average of the first ``period`` gains/losses as the seed,
    then ``avg = (prev*(period-1) + current) / period``. None during
    warm-up. NOTE: the only pre-existing RSI in this checkout
    (telegram_bot.calculate_rsi) is a display-only helper that averages the
    last ``period`` deltas and substitutes a neutral 50 when data is
    insufficient; that fabricated default is unacceptable for research, so
    this module uses the standard Wilder formula and reports warm-up as
    UNAVAILABLE instead of guessing.

States (per the source material):

  vwap_only:  bullish  = close > session VWAP
              bearish  = close < session VWAP
              neutral  = close == session VWAP
  full:       bullish  = close > VWAP and close > EMA and RSI > 50
              bearish  = close < VWAP and close < EMA and RSI < 50
              neutral  = all three inputs valid, neither side satisfied
  unavailable = required input missing, stale, or still warming up
                (reason recorded).

Completed-bar timing: vendor bars are labeled by their START time and no
ingestion timestamp is available, so a bar becomes eligible exactly at
``start + bar_minutes`` (documented precision limit: real ingestion adds
unknown positive latency; treating availability as the bar close is the
EARLIEST defensible point and is applied identically to every variant).

Warm-up scope: indicators are computed on current-session closes only
(primary; consistent with intraday_features), with an optional
``prior_closes`` continuation mode (pre-registered sensitivity) that seeds
EMA/RSI history with the previous regular session's closes. Session VWAP
always resets at the session open in both modes.
"""
import re
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

BAR_MINUTES_DEFAULT = 5
EMA_PERIOD_DEFAULT = 20
RSI_PERIOD_DEFAULT = 14
RSI_MIDPOINT = 50.0
MAX_STALENESS_MIN_DEFAULT = 10.0  # one missing bar tolerated, two is stale

BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"
UNAVAILABLE = "unavailable"


def _num(x) -> Optional[float]:
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def parse_ts(value) -> Optional[datetime]:
    """ISO-8601 (``Z`` accepted) -> aware UTC datetime; None on garbage.

    Ledger timestamps carry nanosecond fractions (e.g. ``.03192073Z``);
    ``datetime.fromisoformat`` accepts at most 6 fractional digits, so the
    fraction is truncated to microseconds before parsing.
    """
    if isinstance(value, datetime):
        return value if value.tzinfo else value.replace(tzinfo=timezone.utc)
    if not isinstance(value, str) or not value:
        return None
    s = value.replace("Z", "+00:00")
    m = re.match(r"^([^.]*)\.(\d+)(.*)$", s)
    if m:
        s = f"{m.group(1)}.{m.group(2)[:6]}{m.group(3)}"
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)


def ema_series(values: Sequence[float], period: int) -> List[Optional[float]]:
    """Same contract as intraday_features.ema_series (SMA seed, k=2/(p+1))."""
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


def rsi_series(closes: Sequence[float], period: int = RSI_PERIOD_DEFAULT
               ) -> List[Optional[float]]:
    """Wilder RSI. None until ``period + 1`` closes exist; never guesses."""
    vals = [_num(v) for v in (closes or [])]
    n = len(vals)
    if not isinstance(period, int) or period <= 0 or any(v is None for v in vals):
        return [None] * n
    out: List[Optional[float]] = [None] * n
    if n < period + 1:
        return out
    gains = [max(vals[i] - vals[i - 1], 0.0) for i in range(1, n)]
    losses = [max(vals[i - 1] - vals[i], 0.0) for i in range(1, n)]
    avg_gain = sum(gains[:period]) / period
    avg_loss = sum(losses[:period]) / period

    def _rsi(g: float, l: float) -> float:
        if l == 0.0:
            return 100.0 if g > 0.0 else 50.0
        rs = g / l
        return 100.0 - (100.0 / (1.0 + rs))

    out[period] = _rsi(avg_gain, avg_loss)
    for i in range(period + 1, n):
        avg_gain = (avg_gain * (period - 1) + gains[i - 1]) / period
        avg_loss = (avg_loss * (period - 1) + losses[i - 1]) / period
        out[i] = _rsi(avg_gain, avg_loss)
    return out


def session_vwap_series(bars: Sequence[Dict]) -> Dict:
    """Running session VWAP per bar.

    Returns ``{"series": [...], "basis": "trade_derived"|"typical_price"}``.
    Basis degrades to ``typical_price`` if ANY contributing bar lacks the
    vendor ``vwap`` field.
    """
    out: List[Optional[float]] = []
    num = den = 0.0
    trade_derived = True
    for b in bars or []:
        v = _num(b.get("v")) or 0.0
        bv = _num(b.get("vwap"))
        if bv is None:
            h, l, c = _num(b.get("h")), _num(b.get("l")), _num(b.get("c"))
            bv = (h + l + c) / 3.0 if None not in (h, l, c) else None
            if v and bv is not None:
                trade_derived = False
        if v and bv is not None:
            num += bv * v
            den += v
        out.append(num / den if den > 0 else None)
    return {"series": out,
            "basis": "trade_derived" if trade_derived else "typical_price"}


def _state_vwap_only(close: Optional[float], vwap: Optional[float]) -> str:
    if close is None or vwap is None:
        return UNAVAILABLE
    if close > vwap:
        return BULLISH
    if close < vwap:
        return BEARISH
    return NEUTRAL


def _state_full(close: Optional[float], vwap: Optional[float],
                ema: Optional[float], rsi: Optional[float]) -> str:
    if None in (close, vwap, ema, rsi):
        return UNAVAILABLE
    if close > vwap and close > ema and rsi > RSI_MIDPOINT:
        return BULLISH
    if close < vwap and close < ema and rsi < RSI_MIDPOINT:
        return BEARISH
    return NEUTRAL


def signal_at(session_bars: Sequence[Dict], decision_ts,
              *, prior_closes: Optional[Sequence[float]] = None,
              ema_period: int = EMA_PERIOD_DEFAULT,
              rsi_period: int = RSI_PERIOD_DEFAULT,
              bar_minutes: int = BAR_MINUTES_DEFAULT,
              max_staleness_min: float = MAX_STALENESS_MIN_DEFAULT) -> Dict:
    """Point-in-time trend state at ``decision_ts`` from current-session bars.

    ``session_bars``: dicts with ``ts`` (ISO-8601 UTC bar START), ``o h l c
    v`` and optionally vendor ``vwap``; CURRENT regular session only, in time
    order. ``prior_closes``: optional previous-session closes used ONLY to
    warm EMA/RSI (continuation sensitivity); VWAP always session-resets.

    Returns a dict with ``state_vwap_only``, ``state_full``, the input
    values, the source bar timestamp, ``bar_lag_s`` (decision time minus bar
    close), ``vwap_basis``, and ``reason`` when either state is unavailable.
    """
    dts = parse_ts(decision_ts)
    base = {"state_vwap_only": UNAVAILABLE, "state_full": UNAVAILABLE,
            "close": None, "vwap": None, "ema": None, "rsi": None,
            "bar_ts": None, "bar_lag_s": None, "vwap_basis": None,
            "reason": None, "warmup": "continuation" if prior_closes else "session"}
    if dts is None:
        base["reason"] = "bad_decision_ts"
        return base

    completed = []
    for b in session_bars or []:
        bts = parse_ts(b.get("ts"))
        if bts is None:
            continue
        if bts + timedelta(minutes=bar_minutes) <= dts:
            completed.append((bts, b))
    if not completed:
        base["reason"] = "no_completed_bar"
        return base
    completed.sort(key=lambda p: p[0])
    last_ts, _last = completed[-1]
    lag_s = (dts - (last_ts + timedelta(minutes=bar_minutes))).total_seconds()
    base["bar_ts"] = last_ts.isoformat()
    base["bar_lag_s"] = lag_s
    if lag_s > max_staleness_min * 60.0:
        base["reason"] = "stale_bar"
        return base

    bars = [b for _, b in completed]
    closes = [_num(b.get("c")) for b in bars]
    if closes[-1] is None:
        base["reason"] = "bad_close"
        return base
    vw = session_vwap_series(bars)
    base["vwap_basis"] = vw["basis"]
    close = closes[-1]
    vwap = vw["series"][-1]
    base["close"] = close
    base["vwap"] = vwap
    base["state_vwap_only"] = _state_vwap_only(close, vwap)
    if base["state_vwap_only"] == UNAVAILABLE:
        base["reason"] = "no_session_vwap"
        return base

    hist = list(prior_closes or []) + closes
    ema = ema_series(hist, ema_period)[-1]
    rsi = rsi_series(hist, rsi_period)[-1]
    base["ema"] = ema
    base["rsi"] = rsi
    base["state_full"] = _state_full(close, vwap, ema, rsi)
    if base["state_full"] == UNAVAILABLE:
        base["reason"] = "warming_up"
    return base


def aligned(state: str, direction: str) -> Optional[bool]:
    """Alignment of a signal state with a single-leg long option's exposure.

    Oracle's recorded decisions are long single-leg contracts, so net
    directional exposure is CALL -> bullish, PUT -> bearish. Returns True /
    False for a definite answer, None when the state is unavailable (missing
    input, reported separately from a deliberate filter skip). Unknown
    directions are out of scope -> None.
    """
    d = (direction or "").upper()
    if state == UNAVAILABLE or d not in ("CALL", "PUT"):
        return None
    want = BULLISH if d == "CALL" else BEARISH
    return state == want


def _selftest() -> int:  # pragma: no cover - exercised via tests
    failures = []

    def check(name, cond):
        print(("PASS" if cond else "FAIL") + f": {name}")
        if not cond:
            failures.append(name)

    def bar(ts, c, v=100.0, vwap=None, o=None, h=None, l=None):
        return {"ts": ts, "o": o if o is not None else c,
                "h": h if h is not None else c + 0.5,
                "l": l if l is not None else c - 0.5,
                "c": c, "v": v, **({"vwap": vwap} if vwap is not None else {})}

    # 25 rising 5-min bars from 14:30 UTC, vendor vwap slightly below close
    bars = [bar(f"2026-11-04T{14 + (30 + 5 * i) // 60:02d}:"
                f"{(30 + 5 * i) % 60:02d}:00Z", 100.0 + i, vwap=99.5 + i)
            for i in range(25)]

    # completed-bar timing: at 14:34:59 the first bar has not closed
    s = signal_at(bars, "2026-11-04T14:34:59Z")
    check("bar not eligible before its close", s["reason"] == "no_completed_bar")
    s = signal_at(bars, "2026-11-04T14:35:00Z")
    check("bar eligible exactly at close", s["state_vwap_only"] == BULLISH)
    check("full variant warming up early", s["state_full"] == UNAVAILABLE
          and s["reason"] == "warming_up")

    # after warm-up (>=21 closes for RSI14+EMA20): rising tape -> bullish
    late = "2026-11-04T16:35:00Z"  # all 25 bars completed
    s = signal_at(bars, late)
    check("full bullish after warm-up", s["state_full"] == BULLISH
          and s["rsi"] is not None and s["rsi"] > 50 and s["ema"] is not None)
    check("vwap basis trade_derived", s["vwap_basis"] == "trade_derived")
    check("bar_ts is last completed", s["bar_ts"].startswith("2026-11-04T16:30"))

    # falling tape -> bearish both variants
    fbars = [bar(b["ts"], 200.0 - i, vwap=200.4 - i) for i, b in enumerate(bars)]
    s = signal_at(fbars, late)
    check("full bearish on falling tape", s["state_full"] == BEARISH
          and s["state_vwap_only"] == BEARISH and s["rsi"] < 50)

    # equality -> neutral (vwap_only); mixed -> neutral (full)
    nbars = [bar(b["ts"], 100.0, v=10.0, vwap=100.0) for b in bars]
    s = signal_at(nbars, late)
    check("close == vwap is neutral", s["state_vwap_only"] == NEUTRAL)
    check("flat tape full neutral (rsi=50)", s["state_full"] == NEUTRAL)

    # staleness: decision 11 min after last bar close -> unavailable
    s = signal_at(bars[:3], "2026-11-04T15:00:00Z")   # last close 14:45
    check("stale bar rejected", s["reason"] == "stale_bar"
          and s["state_vwap_only"] == UNAVAILABLE)

    # typical-price fallback labeled
    tbars = [{k: v for k, v in b.items() if k != "vwap"} for b in bars]
    s = signal_at(tbars, late)
    check("typical-price basis labeled", s["vwap_basis"] == "typical_price")

    # continuation warm-up: early decision becomes available with prior closes
    s = signal_at(bars, "2026-11-04T14:40:00Z",
                  prior_closes=[99.0 + 0.1 * i for i in range(30)])
    check("continuation mode warms early decision",
          s["state_full"] in (BULLISH, BEARISH, NEUTRAL)
          and s["warmup"] == "continuation")

    # wilder RSI reference: constant rises -> 100; warm-up None
    r = rsi_series([100 + i for i in range(20)], 14)
    check("rsi warmup None then 100 on monotone rise",
          r[13] is None and r[14] == 100.0)
    # textbook wilder check (period 3)
    r3 = rsi_series([10.0, 11.0, 10.5, 11.5, 12.0], 3)
    g1, l1 = (1.0 + 0.0 + 1.0) / 3, (0.0 + 0.5 + 0.0) / 3
    exp3 = 100 - 100 / (1 + g1 / l1)
    check("wilder seed matches hand calc", abs(r3[3] - exp3) < 1e-9)
    g2, l2 = (g1 * 2 + 0.5) / 3, (l1 * 2 + 0.0) / 3
    exp4 = 100 - 100 / (1 + g2 / l2)
    check("wilder smoothing matches hand calc", abs(r3[4] - exp4) < 1e-9)

    # alignment mapping
    check("CALL aligns with bullish", aligned(BULLISH, "CALL") is True)
    check("PUT aligns with bearish", aligned(BEARISH, "put") is True)
    check("neutral never aligns", aligned(NEUTRAL, "CALL") is False)
    check("unavailable -> None", aligned(UNAVAILABLE, "CALL") is None)
    check("undefined exposure -> None", aligned(BULLISH, "IRON_CONDOR") is None)

    if failures:
        print(f"SELFTEST FAIL ({len(failures)}): {failures}")
        return 1
    print("SELFTEST PASS")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(_selftest())
