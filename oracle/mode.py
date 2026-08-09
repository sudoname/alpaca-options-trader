"""
Oracle 2.1 — Strategy-mode profiles (ANALYTICS ONLY, pure, no I/O).

A *mode* is a named trading profile that bundles the DTE (days-to-expiry)
selection window and the holding horizon that go together for a style of trade:

  * intraday — short-dated contracts, close same-session. Matches the bot's
    historical default behavior.
  * swing    — multi-week contracts, ride the thesis over several days.

This module ONLY describes those profiles. It never computes direction, never
opens/sizes/prices a trade, and never reads env/creds/network. Two consumers,
both flag-gated upstream:

  * thesis / evidence SHADOW: the resolved mode label is threaded into
    ``build_thesis`` so the decay horizon recorded in the learning slate matches
    the profile in force. Default 'intraday' == the existing hardcoded default,
    so the thesis record is byte-identical unless the caller opts in.
  * DTE window (ACTIVE, flag-gated ``USE_ORACLE_MODE_DTE``, default OFF): the
    caller may override its contract-selection window from ``mode_profile``.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer a safe default over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from typing import Optional

# Mode labels (kept in sync with oracle.thesis MODE_INTRADAY / MODE_SWING).
MODE_INTRADAY = "intraday"
MODE_SWING = "swing"

# Canonical profiles. horizon_days is the holding horizon after which the
# thesis is considered stale; None for swing means "ride to the contract's DTE"
# (build_thesis derives the horizon from dte in swing mode).
_PROFILES = {
    MODE_INTRADAY: {
        "mode": MODE_INTRADAY,
        "min_dte": 0,
        "target_dte": 1,
        "max_dte": 7,
        "horizon_days": 1,
    },
    MODE_SWING: {
        "mode": MODE_SWING,
        "min_dte": 21,
        "target_dte": 45,
        "max_dte": 90,
        "horizon_days": None,
    },
}

# Accepted aliases -> canonical mode label.
_ALIASES = {
    "intraday": MODE_INTRADAY,
    "intra": MODE_INTRADAY,
    "day": MODE_INTRADAY,
    "daytrade": MODE_INTRADAY,
    "scalp": MODE_INTRADAY,
    "0dte": MODE_INTRADAY,
    "swing": MODE_SWING,
    "swings": MODE_SWING,
    "position": MODE_SWING,
    "multiday": MODE_SWING,
}


def resolve_mode(value) -> str:
    """Normalize an arbitrary label to a canonical mode. Fail-open to intraday.

    Accepts the canonical labels, common aliases, and any case/whitespace.
    Unknown or malformed input returns MODE_INTRADAY (the historical default),
    so a caller that passes junk keeps byte-identical behavior.
    """
    if not isinstance(value, str):
        return MODE_INTRADAY
    v = value.strip().lower()
    return _ALIASES.get(v, MODE_INTRADAY)


def mode_profile(value) -> dict:
    """Return a COPY of the resolved mode's profile dict. Never raises.

    The copy keeps callers from mutating the module-level canonical profile.
    """
    resolved = resolve_mode(value)
    return dict(_PROFILES[resolved])


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Canonical labels resolve to themselves.
    if resolve_mode("intraday") != MODE_INTRADAY:
        print("FAIL: intraday label"); ok = False
    if resolve_mode("swing") != MODE_SWING:
        print("FAIL: swing label"); ok = False

    # Case / whitespace insensitivity.
    if resolve_mode("  SWING  ") != MODE_SWING:
        print("FAIL: case/whitespace"); ok = False

    # Aliases.
    if resolve_mode("0dte") != MODE_INTRADAY:
        print("FAIL: 0dte alias"); ok = False
    if resolve_mode("position") != MODE_SWING:
        print("FAIL: position alias"); ok = False

    # Fail-open: junk / wrong type -> intraday (historical default).
    for junk in (None, "", "nonsense", 42, [], {}):
        if resolve_mode(junk) != MODE_INTRADAY:
            print("FAIL: junk should fail open to intraday", junk); ok = False

    # Profiles have the expected shape and sane ordering.
    for label in (MODE_INTRADAY, MODE_SWING):
        p = mode_profile(label)
        for key in ("mode", "min_dte", "target_dte", "max_dte", "horizon_days"):
            if key not in p:
                print("FAIL: profile missing key", label, key); ok = False
        if p["mode"] != label:
            print("FAIL: profile mode label", p); ok = False
        if not (p["min_dte"] <= p["target_dte"] <= p["max_dte"]):
            print("FAIL: dte window ordering", p); ok = False

    # Intraday is shorter-dated than swing across the whole window.
    pi = mode_profile(MODE_INTRADAY)
    ps = mode_profile(MODE_SWING)
    if not (pi["max_dte"] <= ps["min_dte"]):
        print("FAIL: intraday should be shorter-dated than swing", pi, ps)
        ok = False

    # Intraday holds one day; swing rides to dte (horizon None).
    if pi["horizon_days"] != 1:
        print("FAIL: intraday horizon", pi); ok = False
    if ps["horizon_days"] is not None:
        print("FAIL: swing horizon should be None (ride to dte)", ps); ok = False

    # Returned profile is a copy — mutation must not leak into the module.
    p = mode_profile(MODE_SWING)
    p["max_dte"] = 999
    if mode_profile(MODE_SWING)["max_dte"] == 999:
        print("FAIL: profile not defensively copied"); ok = False

    # Determinism.
    if mode_profile("swing") != mode_profile("swing"):
        print("FAIL: non-deterministic"); ok = False

    print("mode self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
