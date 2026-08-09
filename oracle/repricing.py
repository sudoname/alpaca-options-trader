"""
Oracle 2.1 — Repricing-opportunity detector (ANALYTICS ONLY, pure, no I/O).

The mirror image of a chase: a *repricing opportunity* is a modest pullback
AGAINST the trade's direction — a dip for a bullish thesis, a bounce for a
bearish one — that offers a cheaper entry while the thesis is still intact. A
shallow counter-move (roughly 0.5x-1.5x the expected move) is a healthy
pullback; a deeper one is thesis damage, not an opportunity. This module is a
FEATURE, never a standalone trigger and never a direction source:

  * SHADOW: stamped into ``features_json.evidence`` for leaderboard analysis.
  * ACTIVE (flag-gated upstream, default OFF): the caller may add a small
    conviction BONUS when ``opportunity`` is True. It never flips direction (an
    OUTPUT of the rule engine) and never opens/prices a trade.

Definition:
  pullback_ratio = |recent_move_pct| / expected_move_pct   (same horizon)
  counter        = recent move is OPPOSITE the trade direction
  opportunity    = counter AND PULLBACK_MIN <= pullback_ratio <= PULLBACK_MAX

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

# A healthy pullback sits in this band (fraction of the 1-sigma expected move).
# Below MIN it is noise; above MAX it is thesis damage rather than a dip.
PULLBACK_MIN = 0.5
PULLBACK_MAX = 1.5

KIND_PULLBACK = "pullback"


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
class RepricingStamp:
    """One repricing-opportunity reading. All fields analytics-only."""

    opportunity: bool                  # healthy counter-move within the band
    kind: Optional[str]                # 'pullback' when opportunity, else None
    pullback_ratio: Optional[float]    # |recent| / expected (same horizon)
    direction: Optional[str]           # trade direction as given
    reason: str

    def to_dict(self) -> dict:
        return {
            "opportunity": self.opportunity,
            "kind": self.kind,
            "pullback_ratio": self.pullback_ratio,
            "direction": self.direction,
            "reason": self.reason,
        }


def detect_repricing(recent_move_pct, expected_move_pct, direction,
                     pullback_min: float = PULLBACK_MIN,
                     pullback_max: float = PULLBACK_MAX) -> dict:
    """Return a repricing-opportunity stamp dict. Never raises.

    ``recent_move_pct`` and ``expected_move_pct`` must describe the SAME horizon,
    both in percent. Fails open (opportunity False, ratio None) on missing or
    degenerate input.
    """
    recent = _to_float(recent_move_pct)
    expected = _to_float(expected_move_pct)
    sign = _dir_sign(direction)

    if recent is None or expected is None or expected <= 0.0 or sign is None:
        return RepricingStamp(
            opportunity=False, kind=None, pullback_ratio=None,
            direction=(direction if isinstance(direction, str) else None),
            reason="insufficient input",
        ).to_dict()

    ratio = round(abs(recent) / expected, 6)
    counter = (recent < 0) == (sign > 0)  # move opposite the trade direction
    opportunity = counter and (pullback_min <= ratio <= pullback_max)

    if not counter:
        reason = f"move with trade direction ({ratio}x), not a pullback"
    elif opportunity:
        reason = f"healthy pullback: {ratio}x expected against the thesis"
    elif ratio < pullback_min:
        reason = f"pullback too shallow ({ratio}x < {pullback_min}x)"
    else:
        reason = f"pullback too deep ({ratio}x > {pullback_max}x) — thesis risk"

    return RepricingStamp(
        opportunity=opportunity,
        kind=(KIND_PULLBACK if opportunity else None),
        pullback_ratio=ratio,
        direction=(direction if isinstance(direction, str) else None),
        reason=reason,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Bullish thesis, a 1x-expected dip -> healthy pullback opportunity.
    t = detect_repricing(-2.0, 2.0, "call")
    if not t["opportunity"] or t["kind"] != KIND_PULLBACK:
        print("FAIL: dip should be opportunity", t); ok = False
    if abs(t["pullback_ratio"] - 1.0) > 1e-9:
        print("FAIL: pullback ratio", t); ok = False

    # Bearish thesis, a bounce up within the band -> opportunity.
    tb = detect_repricing(1.5, 2.0, "put")
    if not tb["opportunity"]:
        print("FAIL: bounce should be opportunity for put", tb); ok = False

    # Move WITH the trade direction is not a pullback (that is a chase).
    tw = detect_repricing(2.0, 2.0, "call")
    if tw["opportunity"] or tw["kind"] is not None:
        print("FAIL: aligned move is not a pullback", tw); ok = False

    # Too shallow (0.25x) -> noise, no opportunity.
    ts = detect_repricing(-0.5, 2.0, "call")
    if ts["opportunity"]:
        print("FAIL: shallow pullback should not fire", ts); ok = False

    # Too deep (2.5x against thesis) -> thesis damage, no opportunity.
    td = detect_repricing(-5.0, 2.0, "call")
    if td["opportunity"]:
        print("FAIL: deep pullback is thesis risk, not opportunity", td); ok = False

    # Band edges are inclusive.
    lo = detect_repricing(-1.0, 2.0, "call")  # 0.5x
    hi = detect_repricing(-3.0, 2.0, "call")  # 1.5x
    if not (lo["opportunity"] and hi["opportunity"]):
        print("FAIL: band edges inclusive", lo, hi); ok = False

    # Unknown direction -> fail open (cannot tell counter vs aligned).
    tu = detect_repricing(-2.0, 2.0, None)
    if tu["opportunity"] or tu["pullback_ratio"] is not None:
        print("FAIL: unknown direction fails open", tu); ok = False

    # Determinism + never-raise on junk / degenerate input.
    if detect_repricing(-2.0, 2.0, "call") != detect_repricing(-2.0, 2.0, "call"):
        print("FAIL: non-deterministic"); ok = False
    for args in ((None, 2.0, "call"), (-2.0, None, "call"), (-2.0, 0.0, "call"),
                 ("x", "y", "call"), (True, 2.0, "call")):
        try:
            r = detect_repricing(*args)
            if r["opportunity"] is not False or r["pullback_ratio"] is not None:
                print("FAIL: degenerate should fail open", args, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on bad input", args, exc); ok = False

    print("repricing self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
