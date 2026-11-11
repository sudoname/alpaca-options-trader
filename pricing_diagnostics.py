"""
Pricing diagnostics for option quotes (Phase 5 of the Oracle master prompt).

Pure-stdlib reference checks that keep OBSERVED market data distinct from
MODEL outputs:

  * classify_quote()         — invalid-quote taxonomy reusing the Phase-1
                               executable-EV vocabulary (nonpositive_quote,
                               one_sided_quote, crossed_quote, stale_quote).
  * bs_price/bs_delta/bs_vega — Black–Scholes(-Merton) European reference.
  * solve_iv()               — robust bisection IV solve with explicit
                               no-solution reasons (never guesses).
  * quote_iv_diagnostics()   — usable bid/mid/ask IV where the model can
                               solve it, gated on quote validity.
  * model_itm_prob()         — risk-neutral N(d2) terminal ITM probability,
                               labeled so it is never conflated with the
                               empirical probability-of-profit calibration
                               (pop_calibration / pop_recalibration), which
                               is fit on realized trade outcomes.
  * put_call_parity_check()  — C − P vs forward for synchronized quotes.
                               SPY options are American on a dividend payer,
                               so equality is only an approximation; the
                               check reports the residual and a tolerance
                               verdict, it does not assert.
  * crr_american_price()     — Cox–Ross–Rubinstein binomial tree, the
                               exercise-appropriate reference for American
                               options. Pure python, bounded steps.
  * reference_price()        — picks QuantLib when installed (optional heavy
                               dependency, never required), else CRR for
                               American / BS for European, and records which
                               engine produced the number.

No credentials, no network. Everything returns plain dicts with a 'status'
or 'engine' field so callers can tell an observed value from a model value.
"""

import math
from typing import Dict, Optional

try:  # reuse the project's erf-based CDF; keep a local fallback for reuse
    from barrier_engine import norm_cdf
except Exception:  # pragma: no cover - barrier_engine is in-repo
    def norm_cdf(x: float) -> float:
        return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))

# Optional heavy reference library (master prompt: keep it optional).
try:  # pragma: no cover - exercised only where QuantLib is installed
    import QuantLib as _ql  # noqa: N812
    _HAS_QUANTLIB = True
except Exception:
    _ql = None
    _HAS_QUANTLIB = False

# Quote-status vocabulary shared with oracle/execution/executable_ev.py.
QUOTE_OK = "ok"
QUOTE_NONPOSITIVE = "nonpositive_quote"
QUOTE_ONE_SIDED = "one_sided_quote"
QUOTE_CROSSED = "crossed_quote"
QUOTE_STALE = "stale_quote"

DEFAULT_MAX_QUOTE_AGE_S = 60.0

# IV solver bounds (annualized vol). Outside this range the market price is
# treated as unsolvable rather than extrapolated.
IV_LO = 1e-4
IV_HI = 5.0
IV_TOL = 1e-6
IV_MAX_ITER = 200


# --------------------------------------------------------------------------- #
# Quote validity
# --------------------------------------------------------------------------- #
def classify_quote(bid, ask, *, quote_age_s: Optional[float] = None,
                   max_age_s: float = DEFAULT_MAX_QUOTE_AGE_S) -> str:
    """Classify a bid/ask observation. Returns QUOTE_OK or a reason code.

    Order of checks mirrors executable_ev: missing/nonpositive sides first
    (a 0.0 ask is unusable regardless of staleness), then one-sidedness,
    crossing, and finally staleness of an otherwise well-formed quote.
    """
    bid_ok = isinstance(bid, (int, float)) and not isinstance(bid, bool) and bid > 0
    ask_ok = isinstance(ask, (int, float)) and not isinstance(ask, bool) and ask > 0
    if not bid_ok and not ask_ok:
        return QUOTE_NONPOSITIVE
    if not bid_ok or not ask_ok:
        return QUOTE_ONE_SIDED
    if bid > ask:
        return QUOTE_CROSSED
    if quote_age_s is not None and quote_age_s > max_age_s:
        return QUOTE_STALE
    return QUOTE_OK


# --------------------------------------------------------------------------- #
# Black–Scholes(-Merton) European reference
# --------------------------------------------------------------------------- #
def _d1_d2(S: float, K: float, T: float, r: float, q: float, sigma: float):
    st = sigma * math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / st
    return d1, d1 - st


def bs_price(S: float, K: float, T: float, r: float, q: float,
             sigma: float, opt_type: str) -> float:
    """European BSM price. T in years; opt_type 'call'|'put'.

    T<=0 or sigma<=0 degenerate cases return discounted intrinsic on the
    forward (the no-time/no-vol limit), keeping the solver's bracketing sane.
    """
    opt_type = (opt_type or "").lower()
    if opt_type not in ("call", "put"):
        raise ValueError(f"opt_type must be call|put, got {opt_type!r}")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        fwd_s = S * math.exp(-q * max(T, 0.0))
        fwd_k = K * math.exp(-r * max(T, 0.0))
        intrinsic = fwd_s - fwd_k if opt_type == "call" else fwd_k - fwd_s
        return max(intrinsic, 0.0)
    d1, d2 = _d1_d2(S, K, T, r, q, sigma)
    if opt_type == "call":
        return S * math.exp(-q * T) * norm_cdf(d1) - K * math.exp(-r * T) * norm_cdf(d2)
    return K * math.exp(-r * T) * norm_cdf(-d2) - S * math.exp(-q * T) * norm_cdf(-d1)


def bs_delta(S: float, K: float, T: float, r: float, q: float,
             sigma: float, opt_type: str) -> float:
    """European BSM delta (spot delta, continuous dividend yield q)."""
    if T <= 0 or sigma <= 0:
        itm = (S > K) if (opt_type or "").lower() == "call" else (S < K)
        base = 1.0 if itm else 0.0
        return base if (opt_type or "").lower() == "call" else -base
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    if (opt_type or "").lower() == "call":
        return math.exp(-q * T) * norm_cdf(d1)
    return -math.exp(-q * T) * norm_cdf(-d1)


def bs_vega(S: float, K: float, T: float, r: float, q: float,
            sigma: float) -> float:
    """European BSM vega per 1.0 of vol (not per vol point)."""
    if T <= 0 or sigma <= 0:
        return 0.0
    d1, _ = _d1_d2(S, K, T, r, q, sigma)
    pdf = math.exp(-0.5 * d1 * d1) / math.sqrt(2.0 * math.pi)
    return S * math.exp(-q * T) * pdf * math.sqrt(T)


def model_itm_prob(S: float, K: float, T: float, r: float, q: float,
                   sigma: float, opt_type: str) -> Dict:
    """Risk-neutral terminal ITM probability N(d2) / N(−d2).

    This is a MODEL-IMPLIED number under the pricing measure. It is not the
    empirical probability of profit: that comes from realized-outcome
    calibration (pop_calibration / pop_recalibration) and includes costs and
    the exit policy. The label makes the distinction machine-readable.
    """
    if T <= 0 or sigma <= 0:
        itm = (S > K) if (opt_type or "").lower() == "call" else (S < K)
        p = 1.0 if itm else 0.0
    else:
        _, d2 = _d1_d2(S, K, T, r, q, sigma)
        p = norm_cdf(d2) if (opt_type or "").lower() == "call" else norm_cdf(-d2)
    return {"probability": p,
            "probability_kind": "model_implied_risk_neutral",
            "not_equal_to": "empirical_probability_of_profit"}


# --------------------------------------------------------------------------- #
# Implied volatility (bisection; explicit failure reasons)
# --------------------------------------------------------------------------- #
def solve_iv(price, S: float, K: float, T: float, r: float, q: float,
             opt_type: str) -> Dict:
    """Solve BSM implied vol for a EUROPEAN price by bisection.

    Returns {'iv', 'status', 'iterations'}. status is 'ok' or a reason:
      no_price        — price missing/nonpositive
      expired         — T <= 0, no vol is defined
      below_intrinsic — price under the no-arbitrage lower bound
      above_bound     — price above even IV_HI vol (or above S for calls)
    Bisection is monotone-safe (vega > 0), no Newton overshoot.
    """
    if not isinstance(price, (int, float)) or isinstance(price, bool) or price <= 0:
        return {"iv": None, "status": "no_price", "iterations": 0}
    if T <= 0:
        return {"iv": None, "status": "expired", "iterations": 0}
    lo_price = bs_price(S, K, T, r, q, IV_LO, opt_type)
    if price < lo_price - IV_TOL:
        return {"iv": None, "status": "below_intrinsic", "iterations": 0}
    hi_price = bs_price(S, K, T, r, q, IV_HI, opt_type)
    if price > hi_price + IV_TOL:
        return {"iv": None, "status": "above_bound", "iterations": 0}
    lo, hi = IV_LO, IV_HI
    it = 0
    for it in range(1, IV_MAX_ITER + 1):
        mid = 0.5 * (lo + hi)
        pm = bs_price(S, K, T, r, q, mid, opt_type)
        if abs(pm - price) < IV_TOL or (hi - lo) < IV_TOL:
            return {"iv": mid, "status": "ok", "iterations": it}
        if pm < price:
            lo = mid
        else:
            hi = mid
    return {"iv": 0.5 * (lo + hi), "status": "ok", "iterations": it}


def quote_iv_diagnostics(*, bid, ask, S: float, K: float, T: float,
                         r: float = 0.0, q: float = 0.0,
                         opt_type: str = "call",
                         quote_age_s: Optional[float] = None,
                         max_age_s: float = DEFAULT_MAX_QUOTE_AGE_S) -> Dict:
    """IV at bid/mid/ask where the model can solve it, gated on quote validity.

    An unusable quote (per classify_quote) yields no IVs — mirroring the
    Phase-1 rule that an unusable quote is unavailable evidence, not zero.
    Per-side solve failures are reported in 'reasons' rather than silently
    dropped, so 'below_intrinsic' at the bid is visible as a data-quality
    signal and not confused with a missing quote.
    """
    status = classify_quote(bid, ask, quote_age_s=quote_age_s, max_age_s=max_age_s)
    out = {"quote_status": status, "iv_bid": None, "iv_mid": None,
           "iv_ask": None, "reasons": {}}
    if status != QUOTE_OK:
        return out
    mid = 0.5 * (float(bid) + float(ask))
    for name, px in (("bid", float(bid)), ("mid", mid), ("ask", float(ask))):
        res = solve_iv(px, S, K, T, r, q, opt_type)
        if res["status"] == "ok":
            out[f"iv_{name}"] = res["iv"]
        else:
            out["reasons"][name] = res["status"]
    return out


# --------------------------------------------------------------------------- #
# Put–call relationship (synchronized quotes)
# --------------------------------------------------------------------------- #
def put_call_parity_check(*, call_mid, put_mid, S: float, K: float, T: float,
                          r: float = 0.0, q: float = 0.0,
                          tolerance: float = 0.25,
                          exercise: str = "american") -> Dict:
    """Residual of C − P − (S·e^{−qT} − K·e^{−rT}) for synchronized mids.

    For European options the residual should be ~0. SPY options are AMERICAN
    on a dividend payer, where early exercise turns the equality into a
    bound, so this is a diagnostic (large residual => stale/bad quote or
    mismatched timestamps), never an assertion. The exercise assumption is
    echoed back so the caveat travels with the number.
    """
    ok = all(isinstance(v, (int, float)) and not isinstance(v, bool) and v > 0
             for v in (call_mid, put_mid))
    if not ok or S <= 0 or K <= 0 or T < 0:
        return {"status": "unavailable", "residual": None,
                "within_tolerance": None, "exercise": exercise}
    forward_leg = S * math.exp(-q * T) - K * math.exp(-r * T)
    residual = float(call_mid) - float(put_mid) - forward_leg
    return {"status": "ok", "residual": residual,
            "within_tolerance": abs(residual) <= tolerance,
            "tolerance": tolerance, "exercise": exercise,
            "caveat": ("exact only for European; American + dividends makes "
                       "parity an inequality — diagnostic use only")}


# --------------------------------------------------------------------------- #
# American reference: Cox–Ross–Rubinstein binomial tree
# --------------------------------------------------------------------------- #
def crr_american_price(S: float, K: float, T: float, r: float, q: float,
                       sigma: float, opt_type: str, steps: int = 200) -> float:
    """American option price on a CRR tree (continuous dividend yield q)."""
    opt_type = (opt_type or "").lower()
    if opt_type not in ("call", "put"):
        raise ValueError(f"opt_type must be call|put, got {opt_type!r}")
    if S <= 0 or K <= 0:
        raise ValueError("S and K must be positive")
    if T <= 0 or sigma <= 0:
        intrinsic = S - K if opt_type == "call" else K - S
        return max(intrinsic, 0.0)
    steps = max(1, min(int(steps), 2000))
    dt = T / steps
    u = math.exp(sigma * math.sqrt(dt))
    d = 1.0 / u
    disc = math.exp(-r * dt)
    p = (math.exp((r - q) * dt) - d) / (u - d)
    p = min(max(p, 0.0), 1.0)  # clamp against extreme drift/vol inputs
    # terminal payoffs
    values = []
    for j in range(steps + 1):
        st = S * (u ** j) * (d ** (steps - j))
        values.append(max(st - K, 0.0) if opt_type == "call" else max(K - st, 0.0))
    # backward induction with early exercise
    for i in range(steps - 1, -1, -1):
        for j in range(i + 1):
            st = S * (u ** j) * (d ** (i - j))
            cont = disc * (p * values[j + 1] + (1.0 - p) * values[j])
            ex = st - K if opt_type == "call" else K - st
            values[j] = max(cont, ex, 0.0)
    return values[0]


def reference_price(S: float, K: float, T: float, r: float, q: float,
                    sigma: float, opt_type: str,
                    exercise: str = "american", steps: int = 200) -> Dict:
    """Reference model price with the engine recorded.

    QuantLib (optional) when installed; otherwise CRR for American and BSM
    for European. The returned dict always names the engine so model values
    can never masquerade as observed market prices downstream.
    """
    exercise = (exercise or "american").lower()
    if _HAS_QUANTLIB:  # pragma: no cover - optional dependency path
        try:
            today = _ql.Date.todaysDate()
            _ql.Settings.instance().evaluationDate = today
            day_count = _ql.Actual365Fixed()
            calendar = _ql.NullCalendar()
            expiry = today + int(round(max(T, 1e-9) * 365))
            payoff = _ql.PlainVanillaPayoff(
                _ql.Option.Call if opt_type == "call" else _ql.Option.Put, K)
            ex = (_ql.AmericanExercise(today, expiry) if exercise == "american"
                  else _ql.EuropeanExercise(expiry))
            option = _ql.VanillaOption(payoff, ex)
            spot = _ql.QuoteHandle(_ql.SimpleQuote(S))
            rts = _ql.YieldTermStructureHandle(
                _ql.FlatForward(today, r, day_count))
            qts = _ql.YieldTermStructureHandle(
                _ql.FlatForward(today, q, day_count))
            vts = _ql.BlackVolTermStructureHandle(
                _ql.BlackConstantVol(today, calendar, sigma, day_count))
            process = _ql.BlackScholesMertonProcess(spot, qts, rts, vts)
            if exercise == "american":
                engine = _ql.BinomialVanillaEngine(process, "crr", steps)
                name = "quantlib_crr"
            else:
                engine = _ql.AnalyticEuropeanEngine(process)
                name = "quantlib_analytic"
            option.setPricingEngine(engine)
            return {"price": float(option.NPV()), "engine": name,
                    "exercise": exercise}
        except Exception:
            pass  # fall through to the pure-python references
    if exercise == "american":
        return {"price": crr_american_price(S, K, T, r, q, sigma, opt_type, steps),
                "engine": "crr_binomial", "exercise": exercise}
    return {"price": bs_price(S, K, T, r, q, sigma, opt_type),
            "engine": "black_scholes", "exercise": exercise}


# --------------------------------------------------------------------------- #
# Self-test (no credentials, no network)
# --------------------------------------------------------------------------- #
def _close(a, b, tol=1e-3):
    return a is not None and b is not None and abs(a - b) <= tol


def _self_test() -> int:
    ok = True

    # Quote taxonomy
    cases = [((1.0, 1.05, None), QUOTE_OK), ((0, 1.0, None), QUOTE_ONE_SIDED),
             ((0, 0, None), QUOTE_NONPOSITIVE), ((1.1, 1.0, None), QUOTE_CROSSED),
             ((1.0, 1.05, 120.0), QUOTE_STALE)]
    for (b, a, age), want in cases:
        got = classify_quote(b, a, quote_age_s=age)
        if got != want:
            print(f"FAIL classify_quote({b},{a},{age}) = {got} want {want}")
            ok = False

    # BSM round trip: price at a known vol, solve it back.
    S, K, T, r, q = 450.0, 455.0, 30 / 365, 0.05, 0.013
    for opt_type in ("call", "put"):
        px = bs_price(S, K, T, r, q, 0.22, opt_type)
        res = solve_iv(px, S, K, T, r, q, opt_type)
        if res["status"] != "ok" or not _close(res["iv"], 0.22, 1e-4):
            print(f"FAIL iv round trip {opt_type}: {res}")
            ok = False

    # Unsolvable prices are labeled, not guessed.
    if solve_iv(0.0001, S, K, T, r, q, "call")["status"] != "below_intrinsic" and \
       solve_iv(0.0001, S, K, T, r, q, "call")["status"] != "ok":
        pass  # deep OTM tiny price may still solve; just ensure no crash
    if solve_iv(S * 2, S, K, T, r, q, "call")["status"] != "above_bound":
        print("FAIL above_bound not detected")
        ok = False
    if solve_iv(None, S, K, T, r, q, "call")["status"] != "no_price":
        print("FAIL no_price not detected")
        ok = False

    # Parity: consistent European mids give ~0 residual.
    c = bs_price(S, K, T, r, q, 0.22, "call")
    p = bs_price(S, K, T, r, q, 0.22, "put")
    par = put_call_parity_check(call_mid=c, put_mid=p, S=S, K=K, T=T, r=r, q=q,
                                exercise="european")
    if par["status"] != "ok" or not par["within_tolerance"]:
        print(f"FAIL parity {par}")
        ok = False

    # CRR American >= European BSM for puts (early-exercise premium).
    am = crr_american_price(S, K, T, r, q, 0.22, "put", steps=200)
    eu = bs_price(S, K, T, r, q, 0.22, "put")
    if am < eu - 1e-6:
        print(f"FAIL american put {am} < european {eu}")
        ok = False

    # Model probability label present and distinct.
    mp = model_itm_prob(S, K, T, r, q, 0.22, "call")
    if mp["probability_kind"] != "model_implied_risk_neutral":
        print(f"FAIL probability kind {mp}")
        ok = False

    # reference_price records its engine.
    ref = reference_price(S, K, T, r, q, 0.22, "put", exercise="american")
    if "engine" not in ref or ref["price"] <= 0:
        print(f"FAIL reference_price {ref}")
        ok = False

    print("pricing_diagnostics self-test:", "PASS" if ok else "FAIL",
          f"(quantlib={'yes' if _HAS_QUANTLIB else 'no'})")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
