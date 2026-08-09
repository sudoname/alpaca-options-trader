"""
Oracle 2.1 — Thesis-decay exit signal (ANALYTICS ONLY, pure, no I/O).

A trade thesis has a finite shelf life and a level at which it is simply wrong.
This module reads a FROZEN entry thesis (direction, decay_horizon_days,
invalidation_pct) against the position's current state and reports whether the
thesis has decayed enough to exit. Two independent triggers:

  * invalidation — the UNDERLYING has moved AGAINST the trade direction by at
    least ``invalidation_pct`` (a full adverse 1-sigma expected move). The
    thesis was wrong; cut it. Checked first (being wrong beats being stale).
  * horizon      — the position has been held past ``decay_horizon_days``
    (plus an optional grace). The edge the thesis described has passed.

It is a FEATURE, never a standalone trigger and never a direction source:

  * SHADOW: can be stamped for leaderboard analysis.
  * ACTIVE (flag-gated upstream ``USE_THESIS_DECAY_EXIT``, default OFF): the
    monitor may close a position when ``exit`` is True. It never opens, sizes,
    prices, or flips the direction of a trade — it only proposes a close of an
    ALREADY-OPEN position, and hard stops / take-profit keep priority upstream.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
    Datetime parsing is left to the caller — this module takes scalars only.
  * Prefer no-exit over a guess; never raise on malformed input (fail-open:
    a missing input disables only the check that needs it).
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

KIND_INVALIDATION = "invalidation"
KIND_HORIZON = "horizon"


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
class DecayStamp:
    """One thesis-decay reading. All fields analytics-only."""

    exit: bool                         # True when the thesis has decayed enough
    kind: Optional[str]                # 'invalidation' / 'horizon' / None
    hold_days: Optional[float]         # days held (as supplied)
    adverse_move_pct: Optional[float]  # signed: >0 = underlying moved AGAINST
    reason: str

    def to_dict(self) -> dict:
        return {
            "exit": self.exit,
            "kind": self.kind,
            "hold_days": self.hold_days,
            "adverse_move_pct": self.adverse_move_pct,
            "reason": self.reason,
        }


def evaluate_thesis_decay(hold_days, decay_horizon_days, direction,
                          invalidation_pct, entry_underlying_price,
                          current_underlying_price,
                          horizon_grace_days: float = 0.0) -> dict:
    """Return a thesis-decay stamp dict. Never raises.

    Args (all optional; a missing arg disables only its own check):
      hold_days                days the position has been open (caller-computed)
      decay_horizon_days       thesis shelf life in days (from the thesis)
      direction                'call'/'put' (the trade direction, an INPUT)
      invalidation_pct         adverse %-move that voids the thesis (>0)
      entry_underlying_price   underlying price at entry
      current_underlying_price underlying price now
      horizon_grace_days       extra days added to the horizon before it fires

    Precedence: invalidation (thesis wrong) before horizon (thesis stale).
    Fails open to ``exit=False`` whenever the needed inputs are absent/degenerate.
    """
    held = _to_float(hold_days)
    horizon = _to_float(decay_horizon_days)
    inval = _to_float(invalidation_pct)
    entry_u = _to_float(entry_underlying_price)
    cur_u = _to_float(current_underlying_price)
    grace = _to_float(horizon_grace_days) or 0.0
    sign = _dir_sign(direction)

    # Adverse move: signed so that >0 means the underlying moved AGAINST the
    # trade. call (sign +1) is hurt by a down-move; put (sign -1) by an up-move.
    adverse_pct = None
    if sign is not None and entry_u is not None and cur_u is not None \
            and entry_u > 0.0:
        raw_pct = (cur_u - entry_u) / entry_u * 100.0
        adverse_pct = round(-sign * raw_pct, 6)

    # 1) Invalidation — thesis was wrong. Requires a positive threshold and a
    #    measurable adverse move.
    if inval is not None and inval > 0.0 and adverse_pct is not None \
            and adverse_pct >= inval:
        return DecayStamp(
            exit=True, kind=KIND_INVALIDATION, hold_days=held,
            adverse_move_pct=adverse_pct,
            reason=(f"invalidated: underlying {adverse_pct:.2f}% against "
                    f"thesis >= {inval:.2f}%"),
        ).to_dict()

    # 2) Horizon — thesis is stale. Requires a non-negative horizon and a known
    #    hold time.
    if horizon is not None and horizon >= 0.0 and held is not None \
            and held >= horizon + grace:
        return DecayStamp(
            exit=True, kind=KIND_HORIZON, hold_days=held,
            adverse_move_pct=adverse_pct,
            reason=(f"stale: held {held:.2f}d >= horizon {horizon:.0f}d "
                    f"+ grace {grace:.0f}d"),
        ).to_dict()

    # No decay trigger fired (or inputs insufficient) -> hold.
    if adverse_pct is None and held is None:
        reason = "insufficient input"
    else:
        reason = "thesis intact"
    return DecayStamp(
        exit=False, kind=None, hold_days=held,
        adverse_move_pct=adverse_pct, reason=reason,
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Invalidation: a call whose underlying fell 3% vs a 2% invalidation.
    t = evaluate_thesis_decay(hold_days=0.2, decay_horizon_days=1,
                              direction="call", invalidation_pct=2.0,
                              entry_underlying_price=100.0,
                              current_underlying_price=97.0)
    if not t["exit"] or t["kind"] != KIND_INVALIDATION:
        print("FAIL: call down-move should invalidate", t); ok = False
    if abs(t["adverse_move_pct"] - 3.0) > 1e-6:
        print("FAIL: adverse move pct (call)", t); ok = False

    # Invalidation for a put: underlying rose 3% vs a 2% invalidation.
    tp = evaluate_thesis_decay(hold_days=0.2, decay_horizon_days=1,
                               direction="put", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=103.0)
    if not tp["exit"] or tp["kind"] != KIND_INVALIDATION:
        print("FAIL: put up-move should invalidate", tp); ok = False
    if abs(tp["adverse_move_pct"] - 3.0) > 1e-6:
        print("FAIL: adverse move pct (put)", tp); ok = False

    # A FAVORABLE move is not invalidation (call up 3% -> adverse negative).
    tf = evaluate_thesis_decay(hold_days=0.2, decay_horizon_days=1,
                               direction="call", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=103.0)
    if tf["exit"] or tf["adverse_move_pct"] >= 0:
        print("FAIL: favorable move must not invalidate", tf); ok = False

    # An adverse move smaller than the threshold does not fire.
    ts = evaluate_thesis_decay(hold_days=0.2, decay_horizon_days=5,
                               direction="call", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=99.0)
    if ts["exit"]:
        print("FAIL: sub-threshold adverse move should hold", ts); ok = False

    # Horizon: held past the decay horizon (no invalidation).
    th = evaluate_thesis_decay(hold_days=2.0, decay_horizon_days=1,
                               direction="call", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=100.5)
    if not th["exit"] or th["kind"] != KIND_HORIZON:
        print("FAIL: past-horizon should exit stale", th); ok = False

    # Grace defers the horizon trigger.
    tg = evaluate_thesis_decay(hold_days=1.5, decay_horizon_days=1,
                               direction="call", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=100.5,
                               horizon_grace_days=1.0)
    if tg["exit"]:
        print("FAIL: grace should defer horizon exit", tg); ok = False

    # Precedence: invalidation wins even when also past horizon.
    tprec = evaluate_thesis_decay(hold_days=5.0, decay_horizon_days=1,
                                  direction="call", invalidation_pct=2.0,
                                  entry_underlying_price=100.0,
                                  current_underlying_price=95.0)
    if tprec["kind"] != KIND_INVALIDATION:
        print("FAIL: invalidation should win over horizon", tprec); ok = False

    # Intraday horizon=1: a same-session hold (0.3d) with no adverse move holds.
    ti = evaluate_thesis_decay(hold_days=0.3, decay_horizon_days=1,
                               direction="call", invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=100.0)
    if ti["exit"]:
        print("FAIL: fresh intraday hold should not decay", ti); ok = False

    # Fail-open: unknown direction disables the invalidation check.
    tu = evaluate_thesis_decay(hold_days=0.2, decay_horizon_days=5,
                               direction=None, invalidation_pct=2.0,
                               entry_underlying_price=100.0,
                               current_underlying_price=90.0)
    if tu["exit"] or tu["adverse_move_pct"] is not None:
        print("FAIL: unknown direction disables invalidation", tu); ok = False

    # Fail-open: no horizon and no prices -> insufficient input, no exit.
    tn = evaluate_thesis_decay(hold_days=None, decay_horizon_days=None,
                               direction="call", invalidation_pct=None,
                               entry_underlying_price=None,
                               current_underlying_price=None)
    if tn["exit"] or tn["reason"] != "insufficient input":
        print("FAIL: empty inputs should fail open", tn); ok = False

    # Determinism + never-raise on junk / degenerate input.
    a = evaluate_thesis_decay(2.0, 1, "call", 2.0, 100.0, 95.0)
    b = evaluate_thesis_decay(2.0, 1, "call", 2.0, 100.0, 95.0)
    if a != b:
        print("FAIL: non-deterministic"); ok = False
    for args in ((None, None, "call", None, None, None),
                 ("x", "y", "call", "z", "p", "q"),
                 (True, 1, "call", 2.0, 100.0, 95.0),
                 (2.0, 1, "call", 2.0, 0.0, 95.0)):  # entry price 0 -> no adverse
        try:
            r = evaluate_thesis_decay(*args)
            if not isinstance(r, dict) or "exit" not in r:
                print("FAIL: bad shape on junk", args, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on bad input", args, exc); ok = False

    print("decay self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
