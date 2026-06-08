"""
Phase 4 — portfolio-level options risk controls.

A small, network-free decision module that mirrors the conventions of
``risk_engine.py``: a frozen-ish limits dataclass, a manual ``.env`` loader, and
a single pure ``check_portfolio_limits`` that returns a verdict dict. It is the
aggregate-exposure complement to the per-trade risk engine.

Caps enforced (all OFF unless ``USE_PORTFOLIO_GREEK_LIMITS`` is true):
  * max |portfolio delta|   (directional exposure, contract-delta units)
  * max |portfolio vega|    (vol exposure)
  * max portfolio theta loss(daily decay; theta is negative for long options)
  * max same-direction positions (bullish=long calls, bearish=long puts)
  * max positions per underlying (concentration)

Unit convention (kept deliberately simple and matching the small paper account):
exposure is measured in *contract* greeks, i.e. ``greek_per_contract * contracts``
(NOT multiplied by 100). With the defaults (delta cap 5.0) that is roughly ten
0.5-delta contracts — sane for this account; a *100 share-equivalent scale would
make the caps nonsensical.

Sign normalization is done HERE from the position ``direction`` so callers only
pass greek *magnitudes* plus 'call'/'put':
  * delta_signed  = +|delta| for calls, -|delta| for puts
  * vega_contrib  = +|vega|              (long options are vega-positive)
  * theta_contrib = -|theta|             (long options decay; theta-negative)

This module never raises for normal inputs and does no I/O beyond the env read.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
@dataclass
class PortfolioLimits:
    enabled: bool = False              # USE_PORTFOLIO_GREEK_LIMITS
    max_abs_delta: float = 5.0         # MAX_PORTFOLIO_ABS_DELTA
    max_abs_vega: float = 10.0         # MAX_PORTFOLIO_ABS_VEGA
    max_theta_loss: float = 5.0        # MAX_PORTFOLIO_THETA_LOSS (positive $/units)
    max_same_direction: int = 3        # MAX_SAME_DIRECTION_POSITIONS
    max_per_underlying: int = 2        # MAX_POSITIONS_PER_UNDERLYING


def _flag(env: Dict[str, str], key: str, default: bool = False) -> bool:
    return str(env.get(key, str(default))).strip().lower() in (
        "1", "true", "yes", "on")


def _f(env: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _i(env: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(env.get(key, default)))
    except (TypeError, ValueError):
        return default


def load_portfolio_limits_from_env(path: str = ".env") -> PortfolioLimits:
    """Resolve limits via the shared loader (shell env > .env > default).

    Phase 4.5: ``ConfigLoader`` is a drop-in for the parsed-``.env`` dict this
    used to build, so ``_flag``/``_f``/``_i`` work unchanged while a shell
    ``KEY=... python ...`` now overrides ``.env``.
    """
    from config_loader import ConfigLoader
    env = ConfigLoader(path)
    return PortfolioLimits(
        enabled=_flag(env, "USE_PORTFOLIO_GREEK_LIMITS", False),
        max_abs_delta=_f(env, "MAX_PORTFOLIO_ABS_DELTA", 5.0),
        max_abs_vega=_f(env, "MAX_PORTFOLIO_ABS_VEGA", 10.0),
        max_theta_loss=_f(env, "MAX_PORTFOLIO_THETA_LOSS", 5.0),
        max_same_direction=_i(env, "MAX_SAME_DIRECTION_POSITIONS", 3),
        max_per_underlying=_i(env, "MAX_POSITIONS_PER_UNDERLYING", 2),
    )


# --------------------------------------------------------------------------- #
# Greek aggregation
# --------------------------------------------------------------------------- #
def _is_bullish(direction: str) -> bool:
    """Long call -> bullish; long put -> bearish. The bot only buys options."""
    return str(direction).strip().lower() in ("call", "c", "bull", "bullish")


def _norm_position(pos: Dict) -> Dict:
    """Normalize one position dict into signed contract-greek contributions.

    Input keys (magnitudes ok): underlying, direction('call'/'put'), qty,
    delta, vega, theta. Missing greeks default to 0.0. Returns
    {underlying, bullish, qty, delta, vega, theta} with signs applied.
    """
    direction = pos.get("direction", "call")
    bullish = _is_bullish(direction)
    try:
        qty = abs(int(float(pos.get("qty", 1) or 0)))
    except (TypeError, ValueError):
        qty = 0
    delta = abs(float(pos.get("delta", 0.0) or 0.0))
    vega = abs(float(pos.get("vega", 0.0) or 0.0))
    theta = abs(float(pos.get("theta", 0.0) or 0.0))
    return {
        "underlying": str(pos.get("underlying", "")).upper(),
        "bullish": bullish,
        "qty": qty,
        "delta": (delta if bullish else -delta) * qty,
        "vega": vega * qty,            # long options: vega-positive
        "theta": -theta * qty,         # long options: theta-negative (decay)
    }


def aggregate_greeks(positions: List[Dict]) -> Dict:
    """Sum signed contract greeks across positions.

    Returns {delta, vega, theta, count, bullish, bearish, by_underlying}.
    """
    delta = vega = theta = 0.0
    bullish = bearish = 0
    by_underlying: Dict[str, int] = {}
    for raw in positions or []:
        p = _norm_position(raw)
        if p["qty"] <= 0:
            continue
        delta += p["delta"]
        vega += p["vega"]
        theta += p["theta"]
        if p["bullish"]:
            bullish += 1
        else:
            bearish += 1
        by_underlying[p["underlying"]] = by_underlying.get(p["underlying"], 0) + 1
    return {
        "delta": delta,
        "vega": vega,
        "theta": theta,
        "count": (bullish + bearish),
        "bullish": bullish,
        "bearish": bearish,
        "by_underlying": by_underlying,
    }


# --------------------------------------------------------------------------- #
# The check
# --------------------------------------------------------------------------- #
def check_portfolio_limits(
    current_positions: List[Dict],
    new_trade: Dict,
    limits: PortfolioLimits,
) -> Dict:
    """Project the new trade onto the current book and test every aggregate cap.

    ``current_positions`` and ``new_trade`` use the same dict shape understood by
    ``_norm_position``. Returns a verdict dict:
      {allowed, reason, breaches,
       current_delta, projected_delta, current_vega, projected_vega,
       current_theta, projected_theta,
       same_direction, projected_same_direction,
       per_underlying, projected_per_underlying}

    A trade is allowed only when the *projected* book (current + new) stays
    within every cap. When ``limits.enabled`` is False this is a no-op that
    always allows (so default behavior is unchanged).
    """
    cur = aggregate_greeks(current_positions)
    new = _norm_position(new_trade)

    projected_delta = cur["delta"] + new["delta"]
    projected_vega = cur["vega"] + new["vega"]
    projected_theta = cur["theta"] + new["theta"]

    new_bullish = new["bullish"]
    cur_same_direction = cur["bullish"] if new_bullish else cur["bearish"]
    projected_same_direction = cur_same_direction + 1

    cur_per_underlying = cur["by_underlying"].get(new["underlying"], 0)
    projected_per_underlying = cur_per_underlying + 1

    verdict = {
        "allowed": True,
        "reason": "ok",
        "breaches": [],
        "current_delta": cur["delta"],
        "projected_delta": projected_delta,
        "current_vega": cur["vega"],
        "projected_vega": projected_vega,
        "current_theta": cur["theta"],
        "projected_theta": projected_theta,
        "same_direction": cur_same_direction,
        "projected_same_direction": projected_same_direction,
        "per_underlying": cur_per_underlying,
        "projected_per_underlying": projected_per_underlying,
    }

    if not limits.enabled:
        return verdict  # disabled -> always allow, no behavior change

    breaches: List[str] = []
    if abs(projected_delta) > limits.max_abs_delta:
        breaches.append("portfolio_delta")
    if abs(projected_vega) > limits.max_abs_vega:
        breaches.append("portfolio_vega")
    # theta is negative for long options; "theta loss" is its magnitude.
    if (-projected_theta) > limits.max_theta_loss:
        breaches.append("portfolio_theta")
    if limits.max_same_direction > 0 and projected_same_direction > limits.max_same_direction:
        breaches.append("same_direction")
    if limits.max_per_underlying > 0 and projected_per_underlying > limits.max_per_underlying:
        breaches.append("per_underlying")

    verdict["breaches"] = breaches
    verdict["allowed"] = not breaches
    verdict["reason"] = "; ".join(breaches) if breaches else "ok"
    return verdict


def summarize_for_log(v: Dict) -> str:
    """Compact one-liner for the trade log (mirrors the requested log keys)."""
    return (
        f"current_delta={v['current_delta']:.2f} "
        f"projected_delta={v['projected_delta']:.2f} "
        f"current_vega={v['current_vega']:.2f} "
        f"projected_vega={v['projected_vega']:.2f} "
        f"current_theta={v['current_theta']:.2f} "
        f"projected_theta={v['projected_theta']:.2f}"
    )


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    on = PortfolioLimits(enabled=True, max_abs_delta=5.0, max_abs_vega=10.0,
                         max_theta_loss=5.0, max_same_direction=3,
                         max_per_underlying=2)

    # Disabled -> always allowed even with a wild trade.
    off = PortfolioLimits(enabled=False)
    r = check_portfolio_limits(
        [], {"underlying": "SPY", "direction": "call", "qty": 99,
             "delta": 0.9, "vega": 1.0, "theta": 1.0}, off)
    if not r["allowed"]:
        print("FAIL: disabled limits should always allow", r); ok = False

    # Delta cap: book at +4.5 delta, new +1.0 -> 5.5 > 5.0 blocks.
    book = [{"underlying": "AAA", "direction": "call", "qty": 9, "delta": 0.5,
             "vega": 0.1, "theta": 0.05}]  # delta 4.5
    r = check_portfolio_limits(
        book, {"underlying": "BBB", "direction": "call", "qty": 2, "delta": 0.5,
               "vega": 0.1, "theta": 0.05}, on)  # +1.0 -> 5.5
    if r["allowed"] or "portfolio_delta" not in r["breaches"]:
        print("FAIL: delta cap should block", r); ok = False

    # Vega cap.
    book = [{"underlying": "AAA", "direction": "call", "qty": 95, "delta": 0.0,
             "vega": 0.1, "theta": 0.0}]  # vega 9.5
    r = check_portfolio_limits(
        book, {"underlying": "BBB", "direction": "call", "qty": 10, "delta": 0.0,
               "vega": 0.1, "theta": 0.0}, on)  # +1.0 -> 10.5
    if r["allowed"] or "portfolio_vega" not in r["breaches"]:
        print("FAIL: vega cap should block", r); ok = False

    # Theta-loss cap.
    book = [{"underlying": "AAA", "direction": "call", "qty": 90, "delta": 0.0,
             "vega": 0.0, "theta": 0.05}]  # theta -4.5
    r = check_portfolio_limits(
        book, {"underlying": "BBB", "direction": "call", "qty": 20, "delta": 0.0,
               "vega": 0.0, "theta": 0.05}, on)  # -1.0 -> -5.5 -> loss 5.5
    if r["allowed"] or "portfolio_theta" not in r["breaches"]:
        print("FAIL: theta cap should block", r); ok = False

    # Same-direction cap (3 bullish already, new bullish -> projected 4 > 3).
    book = [{"underlying": u, "direction": "call", "qty": 1, "delta": 0.0,
             "vega": 0.0, "theta": 0.0} for u in ("AAA", "BBB", "CCC")]
    r = check_portfolio_limits(
        book, {"underlying": "DDD", "direction": "call", "qty": 1, "delta": 0.0,
               "vega": 0.0, "theta": 0.0}, on)
    if r["allowed"] or "same_direction" not in r["breaches"]:
        print("FAIL: same-direction cap should block", r); ok = False
    # ...but the opposite direction is fine (0 bearish + 1 = 1 <= 3).
    r = check_portfolio_limits(
        book, {"underlying": "DDD", "direction": "put", "qty": 1, "delta": 0.0,
               "vega": 0.0, "theta": 0.0}, on)
    if not r["allowed"]:
        print("FAIL: opposite-direction trade should be allowed", r); ok = False

    # Per-underlying cap (2 already on SPY, new SPY -> projected 3 > 2).
    book = [{"underlying": "SPY", "direction": "call", "qty": 1, "delta": 0.0,
             "vega": 0.0, "theta": 0.0},
            {"underlying": "SPY", "direction": "put", "qty": 1, "delta": 0.0,
             "vega": 0.0, "theta": 0.0}]
    r = check_portfolio_limits(
        book, {"underlying": "SPY", "direction": "call", "qty": 1, "delta": 0.0,
               "vega": 0.0, "theta": 0.0}, on)
    if r["allowed"] or "per_underlying" not in r["breaches"]:
        print("FAIL: per-underlying cap should block", r); ok = False

    # Clean trade within all caps -> allowed.
    r = check_portfolio_limits(
        [], {"underlying": "SPY", "direction": "call", "qty": 1, "delta": 0.4,
             "vega": 0.1, "theta": 0.05}, on)
    if not r["allowed"] or r["reason"] != "ok":
        print("FAIL: clean trade should be allowed", r); ok = False

    print("portfolio_risk self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
