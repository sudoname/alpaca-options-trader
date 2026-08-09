"""
Oracle 2.1 — Machine-readable trade thesis (ANALYTICS ONLY, pure, no I/O).

A ``TradeThesis`` is a structured, serializable snapshot of *why* a trade was
taken and *what would invalidate it*. It is a downstream RECORD, never a signal
source:

  * Direction is an INPUT here, copied verbatim from the first-principles
    engine (determine_option_strategy). This module NEVER computes or overrides
    direction — it only records what was already decided.
  * The thesis feeds the shadow learning slate now, and (in later, flag-gated
    phases) thesis-decay exits and repricing/conviction logic. On its own it
    NEVER opens, sizes, prices, blocks, or alters a real/paper trade.

Fields:
  direction            'call' / 'put' (normalized from call/put or up/down)
  mode                 'intraday' / 'swing' — the strategy profile in force
  conviction           0.0-1.0, normalized from the rule signal strength
  expected_move_pct    1-sigma move for the horizon (from oracle.expected_move)
  invalidation_pct     adverse % move against the position that voids the thesis
  decay_horizon_days   holding horizon after which the thesis is stale
  entry_reason         short human-readable rationale
  catalyst_ref         id/label of a driving catalyst (None until Phase B)

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

# Mode labels.
MODE_INTRADAY = "intraday"
MODE_SWING = "swing"

# Direction labels (normalized).
DIR_CALL = "call"
DIR_PUT = "put"

# Signal strength that maps to full (1.0) conviction. The rule engine emits a
# small integer vote tally (roughly 0-4); 4 convicted votes == full conviction.
CONVICTION_STRENGTH_FULL = 4.0

# Default intraday holding horizon in days (same-session close).
INTRADAY_HORIZON_DAYS = 1


def _to_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_direction(value) -> Optional[str]:
    """Map call/put or up/down (any case) to 'call'/'put'. Else None."""
    if not isinstance(value, str):
        return None
    v = value.strip().lower()
    if v in ("call", "up", "bull", "bullish", "long"):
        return DIR_CALL
    if v in ("put", "down", "bear", "bearish", "short"):
        return DIR_PUT
    return None


def _norm_conviction(strength) -> float:
    """Normalize a rule signal strength to conviction in [0, 1]. Fail-open 0."""
    s = _to_float(strength)
    if s is None:
        return 0.0
    c = abs(s) / CONVICTION_STRENGTH_FULL
    return 0.0 if c < 0.0 else 1.0 if c > 1.0 else round(c, 6)


@dataclass
class TradeThesis:
    """A structured, analytics-only trade thesis."""

    direction: Optional[str]
    mode: str
    conviction: float
    expected_move_pct: Optional[float]
    invalidation_pct: Optional[float]
    decay_horizon_days: Optional[int]
    entry_reason: str
    catalyst_ref: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "direction": self.direction,
            "mode": self.mode,
            "conviction": self.conviction,
            "expected_move_pct": self.expected_move_pct,
            "invalidation_pct": self.invalidation_pct,
            "decay_horizon_days": self.decay_horizon_days,
            "entry_reason": self.entry_reason,
            "catalyst_ref": self.catalyst_ref,
        }


def build_thesis(ctx: Optional[dict], expected_move: Optional[dict] = None,
                 mode: str = MODE_INTRADAY) -> dict:
    """Assemble a thesis dict from a decision ``ctx`` + optional expected-move.

    ``ctx`` fields consumed (all optional): direction, signal_strength (or
    confidence), regime, dte, catalyst_ref. Direction is recorded as-is; this
    function never decides it. Never raises.
    """
    if not isinstance(ctx, dict):
        ctx = {}
    mode = mode if mode in (MODE_INTRADAY, MODE_SWING) else MODE_INTRADAY

    direction = _norm_direction(ctx.get("direction"))
    strength = ctx.get("signal_strength")
    if strength is None:
        strength = ctx.get("confidence")
    conviction = _norm_conviction(strength)

    em_pct = None
    if isinstance(expected_move, dict):
        em_pct = _to_float(expected_move.get("sigma1_pct"))

    # A trade is invalidated by a 1-sigma move against it (the expected move is
    # the natural noise band; a full adverse sigma means the thesis was wrong).
    invalidation_pct = em_pct

    # Holding horizon: intraday closes same session; swing rides to (near)
    # expiration, capped by the contract's remaining DTE when known.
    dte = _to_float(ctx.get("dte"))
    if mode == MODE_INTRADAY:
        decay_horizon = INTRADAY_HORIZON_DAYS
    else:
        decay_horizon = int(dte) if dte is not None and dte >= 0 else None

    regime = ctx.get("regime")
    dir_txt = direction or "n/a"
    em_txt = f"{em_pct:.1f}%" if em_pct is not None else "n/a"
    entry_reason = (f"{mode} {dir_txt} conv={conviction:.2f} "
                    f"1sigma={em_txt} regime={regime or 'n/a'}")

    return TradeThesis(
        direction=direction,
        mode=mode,
        conviction=conviction,
        expected_move_pct=(round(em_pct, 6) if em_pct is not None else None),
        invalidation_pct=(round(invalidation_pct, 6)
                          if invalidation_pct is not None else None),
        decay_horizon_days=decay_horizon,
        entry_reason=entry_reason,
        catalyst_ref=ctx.get("catalyst_ref"),
    ).to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    em = {"sigma1_pct": 10.0}

    # Intraday call, strong conviction (4 votes -> 1.0), horizon 1 day.
    t = build_thesis({"direction": "call", "signal_strength": 4,
                      "regime": "trending", "dte": 30}, em)
    if t["direction"] != DIR_CALL:
        print("FAIL: direction call", t); ok = False
    if abs(t["conviction"] - 1.0) > 1e-9:
        print("FAIL: conviction full", t); ok = False
    if t["mode"] != MODE_INTRADAY or t["decay_horizon_days"] != 1:
        print("FAIL: intraday horizon", t); ok = False
    if abs(t["expected_move_pct"] - 10.0) > 1e-9:
        print("FAIL: expected move copy", t); ok = False
    if abs(t["invalidation_pct"] - 10.0) > 1e-9:
        print("FAIL: invalidation = 1 sigma", t); ok = False

    # 'down' normalizes to put; half strength -> 0.5 conviction.
    t2 = build_thesis({"direction": "down", "signal_strength": 2}, em)
    if t2["direction"] != DIR_PUT:
        print("FAIL: direction down->put", t2); ok = False
    if abs(t2["conviction"] - 0.5) > 1e-9:
        print("FAIL: conviction half", t2); ok = False

    # Swing mode rides to DTE horizon.
    ts = build_thesis({"direction": "call", "signal_strength": 3, "dte": 21},
                      em, mode=MODE_SWING)
    if ts["mode"] != MODE_SWING or ts["decay_horizon_days"] != 21:
        print("FAIL: swing horizon = dte", ts); ok = False

    # This module NEVER invents a direction: unknown/missing -> None.
    for bad_dir in (None, "sideways", 42, "", "flat"):
        td = build_thesis({"direction": bad_dir, "signal_strength": 3}, em)
        if td["direction"] is not None:
            print("FAIL: must not invent direction", bad_dir, td); ok = False

    # No expected move -> None move/invalidation, still builds.
    tn = build_thesis({"direction": "call", "signal_strength": 1})
    if tn["expected_move_pct"] is not None or tn["invalidation_pct"] is not None:
        print("FAIL: no em -> None fields", tn); ok = False
    if abs(tn["conviction"] - 0.25) > 1e-9:
        print("FAIL: conviction quarter", tn); ok = False

    # 'confidence' is an accepted fallback for signal_strength.
    tc = build_thesis({"direction": "call", "confidence": 4}, em)
    if abs(tc["conviction"] - 1.0) > 1e-9:
        print("FAIL: confidence fallback", tc); ok = False

    # Conviction clamps to [0, 1]; catalyst_ref passes through.
    tclip = build_thesis({"direction": "call", "signal_strength": 99,
                          "catalyst_ref": "earnings:AAPL"}, em)
    if tclip["conviction"] != 1.0:
        print("FAIL: conviction clamp", tclip); ok = False
    if tclip["catalyst_ref"] != "earnings:AAPL":
        print("FAIL: catalyst_ref passthrough", tclip); ok = False

    # Determinism + never-raise on junk.
    if build_thesis({"direction": "call", "signal_strength": 3}, em) != \
            build_thesis({"direction": "call", "signal_strength": 3}, em):
        print("FAIL: non-deterministic"); ok = False
    for junk in (None, 42, "x", [], {"weird": object()}):
        try:
            r = build_thesis(junk, em)  # type: ignore[arg-type]
            if "direction" not in r or "mode" not in r:
                print("FAIL: junk shape", junk, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("thesis self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
