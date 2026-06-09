"""Barrier touch-probability engine (advisory, pure-math, no network).

Answers the question "will <SYMBOL> reach <TARGET> within <DAYS>, and why?"
using a geometric-Brownian-motion (GBM) first-passage model. Two probabilities
are produced for each horizon:

  * P(touch)  - probability the price trades THROUGH the target at any point in
                the window (first-passage / barrier-hit). This is the headline
                number for "will it reach".
  * P(close)  - probability the price CLOSES beyond the target at the horizon
                (terminal distribution). Always <= P(touch).

A directional drift can be layered in. The drift is supplied by the caller
(typically derived from the bot's own call/put signal via ``signal_to_drift``)
so the report can show a neutral "driftless" baseline next to a
"signal-adjusted" view.

Everything here is deterministic and unit-testable; the only dependency is the
standard library ``math`` module. Nothing in this file ever places a trade.
"""

from __future__ import annotations

import math
from typing import Optional

# Calendar-day basis: a horizon of "30 days" means 30 calendar days, converted
# to years for the annualized-vol math. Markets only move on trading days, so
# this is mildly conservative (slightly overstates available time), which is
# the prudent direction for a "could it reach" question.
DAYS_PER_YEAR = 365.0

# Default horizons (calendar days) used by the report when the caller asks for
# a single target window; the requested window is always included.
DEFAULT_HORIZONS = (1, 7, 30, 90)


def norm_cdf(x: float) -> float:
    """Standard-normal CDF via the error function (no SciPy dependency)."""
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _ann_to_horizon(sigma: float, days: float) -> float:
    """Return (T_years, sigma*sqrt(T)) for an annualized sigma over ``days``."""
    t = max(days, 0.0) / DAYS_PER_YEAR
    return t, sigma * math.sqrt(t) if t > 0 else 0.0


def touch_probability(spot: float, barrier: float, sigma: float,
                      mu: float = 0.0, days: float = 30.0) -> float:
    """GBM first-passage probability that ``spot`` touches ``barrier``.

    Uses the reflection-principle closed form for the running max/min of a
    Brownian motion with drift. ``sigma`` and ``mu`` are annualized (sigma is
    the volatility, mu the expected arithmetic log-price drift before the
    Ito term). Returns a probability in [0, 1].

    A lower barrier (barrier < spot) uses the running-minimum law; an upper
    barrier uses the running-maximum law. Degenerate inputs fail safe to 0.
    """
    if spot <= 0 or barrier <= 0 or sigma <= 0 or days <= 0:
        return 0.0

    t, v = _ann_to_horizon(sigma, days)
    if v <= 0:
        return 0.0

    a = math.log(barrier / spot)        # log-distance to the barrier
    m = mu - 0.5 * sigma * sigma        # drift of log-price (Ito-corrected)

    if barrier < spot:
        # P(min_{s<=T} X_s <= a),  a < 0
        p = norm_cdf((a - m * t) / v) + \
            math.exp(2.0 * m * a / (sigma * sigma)) * norm_cdf((a + m * t) / v)
    else:
        # P(max_{s<=T} X_s >= a),  a > 0
        p = norm_cdf((m * t - a) / v) + \
            math.exp(2.0 * m * a / (sigma * sigma)) * norm_cdf((-m * t - a) / v)

    # Clamp: the exponential term can nudge slightly outside [0,1] numerically.
    return min(1.0, max(0.0, p))


def prob_close_beyond(spot: float, barrier: float, sigma: float,
                      mu: float = 0.0, days: float = 30.0) -> float:
    """Probability the terminal price closes beyond ``barrier`` at the horizon.

    For a lower barrier: P(S_T <= barrier). For an upper barrier:
    P(S_T >= barrier). This is the terminal (log-normal) tail, always <=
    ``touch_probability`` for the same inputs.
    """
    if spot <= 0 or barrier <= 0 or sigma <= 0 or days <= 0:
        return 0.0

    t, v = _ann_to_horizon(sigma, days)
    if v <= 0:
        return 0.0

    a = math.log(barrier / spot)
    m = mu - 0.5 * sigma * sigma
    z = (a - m * t) / v
    if barrier < spot:
        return min(1.0, max(0.0, norm_cdf(z)))       # P(S_T <= barrier)
    return min(1.0, max(0.0, 1.0 - norm_cdf(z)))      # P(S_T >= barrier)


def signal_to_drift(strategy: Optional[str], strength: Optional[float],
                    momentum: Optional[float], sigma: float) -> float:
    """Convert the bot's directional signal into an annualized drift (mu).

    The drift is expressed as a fraction of one volatility unit so it scales
    with the symbol's own variability:

      * call  -> +conviction * sigma   (bullish push up)
      * put   -> -conviction * sigma   (bearish push down)
      * skip  -> a faint lean in the direction of recent momentum
                 (sign(momentum) * 0.25 * sigma); 0 if momentum is flat.

    ``conviction`` is the signal strength capped at 4 and normalized to [0,1],
    matching the bot's confidence scale. Returns 0.0 for unusable inputs
    (fail-open: a zero drift reduces to the neutral, driftless model).
    """
    if sigma is None or sigma <= 0:
        return 0.0
    strat = (strategy or '').strip().lower()
    conv = 0.0
    if strength is not None:
        try:
            conv = min(abs(float(strength)), 4.0) / 4.0
        except (TypeError, ValueError):
            conv = 0.0

    if strat == 'call':
        return +conv * sigma
    if strat == 'put':
        return -conv * sigma
    # skip / unknown: faint momentum lean only
    if momentum:
        try:
            return math.copysign(0.25 * sigma, float(momentum))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def classify_probability(p: float) -> str:
    """Plain-English bucket for a probability in [0, 1]."""
    if p < 0.10:
        return "near-impossible"
    if p < 0.30:
        return "unlikely"
    if p < 0.45:
        return "possible but less likely than not"
    if p <= 0.55:
        return "roughly a coin-flip"
    if p < 0.75:
        return "more likely than not"
    if p < 0.90:
        return "probable"
    return "very likely"


def analyze(spot: float, barrier: float, sigma: float, mu: float = 0.0,
            horizons=DEFAULT_HORIZONS) -> list:
    """Per-horizon barrier analysis.

    Returns a list of dicts (one per horizon, ascending) with both the
    driftless and drift-adjusted touch probabilities plus the drift-adjusted
    terminal-close probability::

        {days, p_touch_driftless, p_touch_drift, p_close_drift}
    """
    rows = []
    for d in sorted(set(int(h) for h in horizons if int(h) > 0)):
        rows.append({
            'days': d,
            'p_touch_driftless': touch_probability(spot, barrier, sigma, 0.0, d),
            'p_touch_drift': touch_probability(spot, barrier, sigma, mu, d),
            'p_close_drift': prob_close_beyond(spot, barrier, sigma, mu, d),
        })
    return rows


def _fmt_pct(p: float) -> str:
    return f"{p * 100:.0f}%"


def format_report(symbol: str, spot: float, target: float, sigma: float,
                  days: int, strategy: Optional[str] = None,
                  strength: Optional[float] = None,
                  momentum: Optional[float] = None) -> str:
    """Build the Telegram (Markdown) barrier report for ``symbol``.

    Shows the distance to target, the per-horizon touch probabilities
    (driftless vs signal-adjusted), the terminal-close probability at the
    requested window, and a plain-English "will it & why" verdict. Advisory
    only - this function never trades and has no side effects.
    """
    symbol = (symbol or '').upper()
    direction = "below" if target < spot else "above"
    pct_away = (target - spot) / spot * 100.0
    # How many sigma is the move, scaled to the requested horizon?
    _, v_window = _ann_to_horizon(sigma, days)
    sigmas_away = (abs(math.log(target / spot)) / v_window) if v_window > 0 else float('inf')

    mu = signal_to_drift(strategy, strength, momentum, sigma)
    # Build a horizon set that always includes the requested window.
    horizons = sorted(set(list(DEFAULT_HORIZONS) + [int(days)]))
    rows = analyze(spot, target, sigma, mu, horizons)

    # Signal description
    strat = (strategy or '').strip().lower()
    if strat == 'call':
        lean = "bullish (up)"
    elif strat == 'put':
        lean = "bearish (down)"
    elif strat == 'skip':
        lean = "no-trade / sub-conviction"
    else:
        lean = "n/a"
    conv = min(abs(float(strength)), 4.0) / 4.0 if strength is not None else 0.0
    drift_pct = mu * 100.0

    # Headline verdict at the requested horizon (signal-adjusted touch prob).
    req = next((r for r in rows if r['days'] == int(days)), rows[-1])
    verdict_p = req['p_touch_drift']
    verdict = classify_probability(verdict_p)

    lines = []
    lines.append(f"🎯 *Barrier Analysis — {symbol}*")
    lines.append(
        f"Spot `{spot:,.2f}` → target `{target:,.2f}` "
        f"({pct_away:+.1f}%, {direction}, ~{sigmas_away:.2f}σ at {int(days)}d)")
    lines.append(
        f"Vol `{sigma:.0%}` ann.  Signal: *{lean}*"
        + (f", conv `{conv:.0%}`" if strat in ('call', 'put') else "")
        + f"  → drift `{drift_pct:+.1f}%/yr`")
    lines.append("")
    lines.append("*P(touch) — reaches target at any point*")
    lines.append("`days  driftless  +signal`")
    for r in rows:
        mark = "  ←" if r['days'] == int(days) else ""
        lines.append(
            f"`{r['days']:>4}    {_fmt_pct(r['p_touch_driftless']):>6}    "
            f"{_fmt_pct(r['p_touch_drift']):>6}`{mark}")
    lines.append("")
    lines.append(
        f"P(close beyond at {int(days)}d, +signal): "
        f"`{_fmt_pct(req['p_close_drift'])}`")
    lines.append("")
    lines.append(
        f"*Verdict:* reaching `{target:,.2f}` within *{int(days)} days* is "
        f"*{verdict}* (~{_fmt_pct(verdict_p)}).")
    # Why
    why = (
        f"At {sigma:.0%} annualized vol the target is ~{sigmas_away:.2f}σ "
        f"{direction} spot over {int(days)} days. "
    )
    if abs(drift_pct) < 1e-9:
        why += "No directional drift applied (neutral baseline)."
    else:
        agrees = (
            (target < spot and mu < 0) or (target > spot and mu > 0))
        why += (
            f"The bot's {lean} signal "
            + ("supports" if agrees else "works against")
            + f" the move, shifting the odds "
            + ("up" if agrees else "down")
            + " versus the driftless case.")
    lines.append("_" + why + "_")
    lines.append("_(Advisory only — math from live vol + signal; nothing traded.)_")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network): `python barrier_engine.py`
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            ok = False
            print(f"FAIL: {msg}")

    # norm_cdf sanity
    check(abs(norm_cdf(0.0) - 0.5) < 1e-9, "norm_cdf(0)=0.5")
    check(norm_cdf(5) > 0.999 and norm_cdf(-5) < 0.001, "norm_cdf tails")

    # Touch prob: monotonic in time, bounded, and >= close prob.
    p7 = touch_probability(100, 90, 0.2, 0.0, 7)
    p30 = touch_probability(100, 90, 0.2, 0.0, 30)
    p90 = touch_probability(100, 90, 0.2, 0.0, 90)
    check(0 <= p7 <= p30 <= p90 <= 1, f"touch monotonic in t ({p7:.3f},{p30:.3f},{p90:.3f})")
    c30 = prob_close_beyond(100, 90, 0.2, 0.0, 30)
    check(c30 <= p30 + 1e-9, f"close<=touch ({c30:.3f} vs {p30:.3f})")

    # Downward drift raises P(touch) of a LOWER barrier; upward lowers it.
    base = touch_probability(100, 90, 0.2, 0.0, 30)
    down = touch_probability(100, 90, 0.2, -0.10, 30)
    up = touch_probability(100, 90, 0.2, +0.10, 30)
    check(down > base > up, f"drift moves lower-barrier touch ({down:.3f}>{base:.3f}>{up:.3f})")

    # Upper barrier: upward drift raises P(touch).
    ub = touch_probability(100, 110, 0.2, 0.0, 30)
    uup = touch_probability(100, 110, 0.2, +0.10, 30)
    udn = touch_probability(100, 110, 0.2, -0.10, 30)
    check(uup > ub > udn, f"drift moves upper-barrier touch ({uup:.3f}>{ub:.3f}>{udn:.3f})")

    # signal_to_drift sign/scale
    check(signal_to_drift('call', 4, 0.0, 0.2) > 0, "call -> +drift")
    check(signal_to_drift('put', 4, 0.0, 0.2) < 0, "put -> -drift")
    check(abs(signal_to_drift('call', 4, 0, 0.2) - 0.2) < 1e-9, "full conv = 1 sigma")
    check(abs(signal_to_drift('call', 2, 0, 0.2) - 0.1) < 1e-9, "half conv = 0.5 sigma")
    check(signal_to_drift('skip', 0, -0.02, 0.2) < 0, "skip leans with neg momentum")
    check(signal_to_drift('skip', 0, 0.0, 0.2) == 0.0, "skip+flat = 0 drift")
    check(signal_to_drift('call', 4, 0, 0.0) == 0.0, "zero vol -> 0 drift")

    # classify buckets
    check(classify_probability(0.05) == "near-impossible", "classify low")
    check(classify_probability(0.50) == "roughly a coin-flip", "classify mid")
    check(classify_probability(0.95) == "very likely", "classify high")

    # analyze shape
    rows = analyze(100, 90, 0.2, -0.05, [7, 30])
    check(len(rows) == 2 and rows[0]['days'] == 7, "analyze rows")
    check(all(set(r) == {'days', 'p_touch_driftless', 'p_touch_drift', 'p_close_drift'}
              for r in rows), "analyze keys")

    # format_report returns a non-trivial string, requested horizon included.
    rep = format_report("SPY", 732.0, 710.0, 0.14, 30,
                         strategy='put', strength=1, momentum=-0.013)
    check("Barrier Analysis" in rep and "SPY" in rep, "report header")
    check("Verdict" in rep and "30" in rep, "report verdict/horizon")

    print("OK" if ok else "FAILED")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_self_test())
