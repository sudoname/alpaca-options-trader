"""
Oracle 2.1 — Expected-move engine (ANALYTICS ONLY, pure, no I/O).

The expected move is the 1-sigma price range implied for an underlying over the
option's holding horizon. It is a *feature*, never a trade trigger: it feeds the
machine-readable trade thesis (oracle/thesis.py), repricing/chase detection
(later phases), and the shadow learning slate. It NEVER opens, sizes, prices,
blocks, or alters a real/paper trade.

Two derivations, both fail-open:
  * IV method (default): sigma1 = IV * sqrt(DTE / 365). Given an underlying
    price, that becomes a 1-sigma dollar band [price*(1-s), price*(1+s)].
  * Straddle method: the ATM straddle price is the market's own expected-move
    quote; expected_move_pct ~= straddle / price * 100. Used when a straddle
    price is supplied.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess: missing/malformed inputs -> None, never raise.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from math import sqrt
from typing import Optional

# Method labels.
METHOD_IV = "iv"
METHOD_STRADDLE = "straddle"

# Trading days are irrelevant here; expected move is a calendar-time diffusion.
DAYS_PER_YEAR = 365.0

# An annualized IV above this is assumed to be on a 0-100 percentage scale
# (e.g. 35 meaning 35%) and normalized to a decimal. Real annualized IVs above
# 300% are vanishingly rare, so this is a safe demarcation.
_IV_PERCENT_CUTOFF = 3.0


def _to_float(value) -> Optional[float]:
    """Coerce to float; bools and junk -> None (never raises)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_iv(iv) -> Optional[float]:
    """Normalize IV to an annualized decimal. Tolerates a 0-100 pct scale."""
    v = _to_float(iv)
    if v is None or v < 0.0:
        return None
    if v > _IV_PERCENT_CUTOFF:
        v = v / 100.0
    return v


@dataclass
class ExpectedMove:
    """One expected-move estimate. All fields analytics-only."""

    method: str
    sigma1_pct: float                 # 1-sigma move, percent of spot
    iv: Optional[float]               # normalized annualized IV (decimal)
    dte: Optional[float]              # days to expiration used
    price: Optional[float]            # spot used for the dollar band
    sigma1_dollars: Optional[float]   # 1-sigma move in dollars (needs price)
    lower: Optional[float]            # price - sigma1_dollars
    upper: Optional[float]            # price + sigma1_dollars

    def to_dict(self) -> dict:
        return {
            "method": self.method,
            "sigma1_pct": self.sigma1_pct,
            "iv": self.iv,
            "dte": self.dte,
            "price": self.price,
            "sigma1_dollars": self.sigma1_dollars,
            "lower": self.lower,
            "upper": self.upper,
        }


def compute_expected_move(iv=None, dte=None, price=None,
                          straddle_price=None) -> Optional[dict]:
    """Return an expected-move dict, or None when inputs are insufficient.

    Straddle method wins when a positive straddle price and spot are supplied
    (the market's own expected-move quote); otherwise the IV method is used.
    Never raises.
    """
    p = _to_float(price)
    strad = _to_float(straddle_price)

    # --- Straddle method: ATM straddle ~= market expected move. ---
    if strad is not None and strad > 0 and p is not None and p > 0:
        sigma1_pct = (strad / p) * 100.0
        s_dollars = strad
        return ExpectedMove(
            method=METHOD_STRADDLE,
            sigma1_pct=round(sigma1_pct, 6),
            iv=_norm_iv(iv),
            dte=_to_float(dte),
            price=round(p, 6),
            sigma1_dollars=round(s_dollars, 6),
            lower=round(p - s_dollars, 6),
            upper=round(p + s_dollars, 6),
        ).to_dict()

    # --- IV method: sigma1 = IV * sqrt(DTE / 365). ---
    ivn = _norm_iv(iv)
    d = _to_float(dte)
    if ivn is None or d is None or d < 0.0:
        return None
    frac = sqrt(d / DAYS_PER_YEAR)
    sigma1 = ivn * frac                    # decimal 1-sigma move
    sigma1_pct = sigma1 * 100.0
    s_dollars = lower = upper = None
    if p is not None and p > 0:
        s_dollars = round(p * sigma1, 6)
        lower = round(p - s_dollars, 6)
        upper = round(p + s_dollars, 6)
    return ExpectedMove(
        method=METHOD_IV,
        sigma1_pct=round(sigma1_pct, 6),
        iv=round(ivn, 6),
        dte=d,
        price=(round(p, 6) if p is not None else None),
        sigma1_dollars=s_dollars,
        lower=lower,
        upper=upper,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # IV method, no price: sigma1% = 0.35 * sqrt(30/365) * 100 ~= 10.036%.
    em = compute_expected_move(iv=0.35, dte=30)
    if em is None or em["method"] != METHOD_IV:
        print("FAIL: iv method missing", em); ok = False
    else:
        expect = 0.35 * sqrt(30 / 365.0) * 100.0
        if abs(em["sigma1_pct"] - expect) > 1e-4:
            print("FAIL: sigma1_pct", em["sigma1_pct"], "want", expect); ok = False
        if em["sigma1_dollars"] is not None:
            print("FAIL: no price -> no dollar band", em); ok = False

    # IV method with price -> symmetric dollar band around spot.
    emp = compute_expected_move(iv=0.35, dte=30, price=100.0)
    if emp is None or emp["sigma1_dollars"] is None:
        print("FAIL: priced iv method", emp); ok = False
    else:
        s = emp["sigma1_dollars"]
        if abs((emp["upper"] - emp["lower"]) - 2 * s) > 1e-6:
            print("FAIL: band not symmetric", emp); ok = False
        if abs(emp["lower"] - (100.0 - s)) > 1e-6:
            print("FAIL: lower band", emp); ok = False

    # IV given on a 0-100 percentage scale is normalized to decimal.
    em_pct = compute_expected_move(iv=35, dte=30)
    if em_pct is None or abs(em_pct["iv"] - 0.35) > 1e-9:
        print("FAIL: iv percent normalization", em_pct); ok = False
    if em_pct and abs(em_pct["sigma1_pct"] - em["sigma1_pct"]) > 1e-6:
        print("FAIL: pct-scale IV should match decimal IV", em_pct, em); ok = False

    # Straddle method: straddle 6 on spot 100 -> 6% expected move, band +/-6.
    ems = compute_expected_move(iv=0.35, dte=30, price=100.0, straddle_price=6.0)
    if ems is None or ems["method"] != METHOD_STRADDLE:
        print("FAIL: straddle method", ems); ok = False
    else:
        if abs(ems["sigma1_pct"] - 6.0) > 1e-9:
            print("FAIL: straddle pct", ems); ok = False
        if abs(ems["upper"] - 106.0) > 1e-9 or abs(ems["lower"] - 94.0) > 1e-9:
            print("FAIL: straddle band", ems); ok = False

    # Longer DTE -> larger 1-sigma move (monotonic in time).
    short = compute_expected_move(iv=0.4, dte=7)
    long = compute_expected_move(iv=0.4, dte=60)
    if not (short and long and long["sigma1_pct"] > short["sigma1_pct"]):
        print("FAIL: dte monotonicity", short, long); ok = False

    # Determinism.
    if compute_expected_move(iv=0.35, dte=30, price=100.0) != \
            compute_expected_move(iv=0.35, dte=30, price=100.0):
        print("FAIL: non-deterministic"); ok = False

    # Insufficient / garbage inputs -> None, never raise.
    for bad in (
        dict(iv=None, dte=30),
        dict(iv=0.35, dte=None),
        dict(iv=0.35, dte=-5),
        dict(iv="x", dte="y"),
        dict(iv=True, dte=30),
    ):
        try:
            if compute_expected_move(**bad) is not None:
                print("FAIL: bad inputs should be None", bad); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on bad inputs", bad, exc); ok = False

    # dte == 0 is degenerate but valid: zero expected move.
    z = compute_expected_move(iv=0.35, dte=0)
    if z is None or abs(z["sigma1_pct"]) > 1e-12:
        print("FAIL: dte=0 should be zero move", z); ok = False

    print("expected_move self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
