"""
Oracle 2.1 — Chase / extension detector (ANALYTICS ONLY, pure, no I/O).

An entry is a *chase* when the underlying has ALREADY run in the trade's
direction by an outsized fraction of its expected move: buying now pays up for a
move that has largely happened, with worse risk/reward. This module measures
that extension. It is a FEATURE, never a standalone trigger and never a
direction source:

  * SHADOW: stamped into ``features_json.evidence`` for leaderboard analysis.
  * ACTIVE (flag-gated upstream, default OFF): the caller may DISCOUNT the
    conviction that feeds sizing when ``extended`` is True. It never flips the
    direction (an OUTPUT of the rule engine) and never opens/prices a trade.

Definition:
  extension_ratio = |recent_move_pct| / expected_move_pct   (same horizon)
  aligned         = recent move is in the SAME direction as the trade
  extended        = aligned AND extension_ratio >= chase_ratio

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

# Default chase multiple: a move >= 2x the 1-sigma expected move in the trade's
# own direction is treated as extended. Env-tunable by the caller.
CHASE_RATIO = 2.0


def _to_float(value) -> Optional[float]:
    """Coerce to float; bools and junk -> None (never raises)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _dir_sign(direction) -> Optional[int]:
    """+1 for a bullish trade, -1 for bearish, None if unknown."""
    if not isinstance(direction, str):
        return None
    d = direction.strip().lower()
    if d in ("call", "up", "bull", "bullish", "long"):
        return 1
    if d in ("put", "down", "bear", "bearish", "short"):
        return -1
    return None


@dataclass
class ExtensionStamp:
    """One chase/extension reading. All fields analytics-only."""

    extended: bool                     # aligned chase beyond chase_ratio
    aligned: Optional[bool]            # recent move in the trade's direction
    extension_ratio: Optional[float]   # |recent| / expected (same horizon)
    direction: Optional[str]           # trade direction as given
    reason: str

    def to_dict(self) -> dict:
        return {
            "extended": self.extended,
            "aligned": self.aligned,
            "extension_ratio": self.extension_ratio,
            "direction": self.direction,
            "reason": self.reason,
        }


def detect_extension(recent_move_pct, expected_move_pct, direction,
                     chase_ratio: float = CHASE_RATIO) -> dict:
    """Return a chase/extension stamp dict. Never raises.

    ``recent_move_pct`` and ``expected_move_pct`` must describe the SAME horizon
    (e.g. a 3-day realized move vs a 3-day 1-sigma expected move), both in
    percent. Fails open (extended False, ratio None) on missing/degenerate input.
    """
    recent = _to_float(recent_move_pct)
    expected = _to_float(expected_move_pct)
    sign = _dir_sign(direction)

    if recent is None or expected is None or expected <= 0.0:
        return ExtensionStamp(
            extended=False, aligned=None, extension_ratio=None,
            direction=(direction if isinstance(direction, str) else None),
            reason="insufficient input",
        ).to_dict()

    ratio = round(abs(recent) / expected, 6)
    aligned = None if sign is None else ((recent > 0) == (sign > 0))
    extended = bool(aligned) and ratio >= chase_ratio

    if aligned is None:
        reason = f"ratio={ratio} dir=unknown"
    elif extended:
        reason = f"chase: {ratio}x expected move in trade direction"
    elif aligned:
        reason = f"aligned but only {ratio}x expected (< {chase_ratio}x)"
    else:
        reason = f"counter-trend move ({ratio}x), not a chase"

    return ExtensionStamp(
        extended=extended,
        aligned=aligned,
        extension_ratio=ratio,
        direction=(direction if isinstance(direction, str) else None),
        reason=reason,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Aligned 3x chase for a call (moved +6% vs 2% expected) -> extended.
    t = detect_extension(6.0, 2.0, "call")
    if not t["extended"] or t["aligned"] is not True:
        print("FAIL: aligned chase should be extended", t); ok = False
    if abs(t["extension_ratio"] - 3.0) > 1e-9:
        print("FAIL: extension ratio", t); ok = False

    # Aligned but modest (1.5x < 2x) -> not extended.
    tm = detect_extension(3.0, 2.0, "call")
    if tm["extended"] or tm["aligned"] is not True:
        print("FAIL: modest aligned should not be extended", tm); ok = False

    # Put with a big down-move is an aligned chase.
    tp = detect_extension(-6.0, 2.0, "put")
    if not tp["extended"] or tp["aligned"] is not True:
        print("FAIL: put chase should be extended", tp); ok = False

    # Counter-trend move (call after a down-move) -> aligned False, not extended.
    tc = detect_extension(-6.0, 2.0, "call")
    if tc["extended"] or tc["aligned"] is not False:
        print("FAIL: counter-trend not a chase", tc); ok = False

    # Ratio monotonicity: bigger move -> bigger ratio.
    small = detect_extension(4.0, 2.0, "call")
    big = detect_extension(10.0, 2.0, "call")
    if not (big["extension_ratio"] > small["extension_ratio"]):
        print("FAIL: ratio monotonicity", small, big); ok = False

    # Tunable chase_ratio: a 3x move with a 4x threshold is not a chase.
    tt = detect_extension(6.0, 2.0, "call", chase_ratio=4.0)
    if tt["extended"]:
        print("FAIL: higher chase_ratio should not fire", tt); ok = False

    # Unknown direction -> aligned None, extended False, still returns a ratio.
    tu = detect_extension(6.0, 2.0, None)
    if tu["extended"] or tu["aligned"] is not None:
        print("FAIL: unknown direction", tu); ok = False
    if abs(tu["extension_ratio"] - 3.0) > 1e-9:
        print("FAIL: ratio computed without direction", tu); ok = False

    # Determinism + never-raise on junk / degenerate input.
    if detect_extension(6.0, 2.0, "call") != detect_extension(6.0, 2.0, "call"):
        print("FAIL: non-deterministic"); ok = False
    for args in ((None, 2.0, "call"), (6.0, None, "call"), (6.0, 0.0, "call"),
                 ("x", "y", "call"), (True, 2.0, "call")):
        try:
            r = detect_extension(*args)
            if r["extended"] is not False or r["extension_ratio"] is not None:
                print("FAIL: degenerate should fail open", args, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on bad input", args, exc); ok = False

    print("extension self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
