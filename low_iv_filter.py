"""low_iv_filter — throttle new option entries in a low-volatility ("calm
grind / melt-up") regime.

Motivation: a long-premium book bleeds on quiet, drifting tape — puts lose on
direction while calls lose to theta/IV-crush. On those days the bot should take
*fewer* and/or *smaller* new positions rather than keep stacking long premium.

Signal: per-underlying annualized realized volatility (the value produced by
``smart_trader.calculate_volatility`` — already capped to [0.10, 0.80]). When it
sits at/below ``vol_threshold`` the underlying is in a low-IV regime.

Two levers (both advisory, both fail-open):
  * ``effective_cap``     — shrink the per-underlying open-position cap so a calm
                            name maxes out at fewer concurrent contracts.
  * ``adjusted_quantity`` — scale the order size down. Floored at 1 contract, so
                            it is a no-op for a qty=1 bot but bites once sizing
                            grows.

Everything here is pure and deterministic (no network, no clock); the live path
supplies the realized-vol number. Any bad/missing input fails open (treats the
regime as NOT low-IV → no throttling).
"""
from __future__ import annotations

# Defaults mirror the .env flag defaults wired into smart_trader.load_credentials
# (LOW_IV_VOL_THRESHOLD / LOW_IV_SIZE_FACTOR / LOW_IV_CAP_DELTA).
DEFAULT_VOL_THRESHOLD = 0.15  # annualized realized vol; at/below -> low-IV
DEFAULT_SIZE_FACTOR = 0.5     # multiply order size by this when low-IV
DEFAULT_CAP_DELTA = 1         # subtract this from the per-underlying cap when low-IV


def is_low_iv(realized_vol, threshold: float = DEFAULT_VOL_THRESHOLD) -> bool:
    """True when ``realized_vol`` is a finite number at/below ``threshold``.

    Fail-open: ``None``, NaN, or anything non-numeric -> ``False`` (not low-IV),
    so a missing vol reading never throttles a trade.
    """
    try:
        v = float(realized_vol)
    except (TypeError, ValueError):
        return False
    if v != v:  # NaN
        return False
    return v <= float(threshold)


def effective_cap(base_cap: int, low_iv: bool,
                  cap_delta: int = DEFAULT_CAP_DELTA) -> int:
    """Per-underlying open-position cap after the low-IV adjustment.

    When ``low_iv`` is True, drop the cap by ``cap_delta`` but never below 1 —
    a calm name can still take its first position, it just can't stack as deep.
    When not low-IV (or cap_delta<=0) the base cap is returned unchanged.
    """
    try:
        base = int(base_cap)
    except (TypeError, ValueError):
        return base_cap
    if not low_iv:
        return base
    try:
        delta = int(cap_delta)
    except (TypeError, ValueError):
        return base
    if delta <= 0:
        return base
    return max(1, base - delta)


def adjusted_quantity(qty: int, low_iv: bool,
                      size_factor: float = DEFAULT_SIZE_FACTOR) -> int:
    """Order size after the low-IV adjustment.

    When ``low_iv`` is True, scale ``qty`` by ``size_factor`` and round, but
    never below 1 contract. No-op when not low-IV, when qty<=1, or when
    size_factor>=1. A non-positive/garbage factor fails open to the input qty.
    """
    try:
        q = int(qty)
    except (TypeError, ValueError):
        return qty
    if not low_iv or q <= 1:
        return q
    try:
        f = float(size_factor)
    except (TypeError, ValueError):
        return q
    if f <= 0 or f >= 1:
        return q
    return max(1, int(round(q * f)))


def _self_test() -> int:
    ok = True

    # --- is_low_iv: threshold boundary + fail-open --------------------------
    if not is_low_iv(0.10):
        print("FAIL: 0.10 <= 0.15 should be low-IV"); ok = False
    if not is_low_iv(0.15):  # boundary is inclusive
        print("FAIL: 0.15 == threshold should be low-IV"); ok = False
    if is_low_iv(0.20):
        print("FAIL: 0.20 > 0.15 should NOT be low-IV"); ok = False
    if is_low_iv(None):
        print("FAIL: None must fail open to NOT low-IV"); ok = False
    if is_low_iv("abc"):
        print("FAIL: garbage must fail open to NOT low-IV"); ok = False
    if is_low_iv(float("nan")):
        print("FAIL: NaN must fail open to NOT low-IV"); ok = False
    if not is_low_iv(0.25, threshold=0.30):
        print("FAIL: custom threshold not honored"); ok = False

    # --- effective_cap: drop by delta, floor at 1 ---------------------------
    if effective_cap(2, low_iv=False) != 2:
        print("FAIL: not low-IV must leave cap unchanged"); ok = False
    if effective_cap(2, low_iv=True) != 1:
        print("FAIL: cap 2 low-IV -> 1"); ok = False
    if effective_cap(1, low_iv=True) != 1:
        print("FAIL: cap must never drop below 1"); ok = False
    if effective_cap(3, low_iv=True, cap_delta=2) != 1:
        print("FAIL: cap 3 delta 2 -> 1"); ok = False
    if effective_cap(5, low_iv=True, cap_delta=2) != 3:
        print("FAIL: cap 5 delta 2 -> 3"); ok = False
    if effective_cap(2, low_iv=True, cap_delta=0) != 2:
        print("FAIL: cap_delta 0 must be a no-op"); ok = False

    # --- adjusted_quantity: scale, floor at 1, no-op cases ------------------
    if adjusted_quantity(1, low_iv=True) != 1:
        print("FAIL: qty 1 must stay 1 (no-op for qty=1 bot)"); ok = False
    if adjusted_quantity(4, low_iv=False) != 4:
        print("FAIL: not low-IV must leave qty unchanged"); ok = False
    if adjusted_quantity(4, low_iv=True) != 2:
        print("FAIL: qty 4 *0.5 -> 2"); ok = False
    if adjusted_quantity(3, low_iv=True) != 2:  # round(1.5) -> 2
        print("FAIL: qty 3 *0.5 -> 2 (rounded)"); ok = False
    if adjusted_quantity(2, low_iv=True) != 1:
        print("FAIL: qty 2 *0.5 -> 1"); ok = False
    if adjusted_quantity(10, low_iv=True, size_factor=0.0) != 10:
        print("FAIL: factor 0 must fail open"); ok = False
    if adjusted_quantity(10, low_iv=True, size_factor=1.0) != 10:
        print("FAIL: factor 1 must be a no-op"); ok = False

    print("low_iv_filter self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
