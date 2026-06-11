"""
Phase 10H — Entry EV stamp for single-leg scheduler trades.

The Phase 10G calibration analytics (ev_calibration / pop_calibration) need
each closed trade to carry the *frozen entry-time belief*: expected dollar EV,
probability of profit, and max loss. The spread runner stamps these, but the
single-leg scheduler path never did — so scheduler trades produced outcomes
with no beliefs to calibrate against.

This module computes that stamp at order time. It is ANALYTICS-ONLY metadata:
nothing here gates, sizes, or prices a trade, and the caller treats any
failure as "no stamp" (fail-open).

Model (deterministic, unit-testable, no network):
  * max_loss   = premium paid = entry_price * 100 * qty (long option worst case)
  * POP        = gambler's-ruin baseline + signal tilt.
                 A driftless walk on the premium hits +tp before -sl with
                 p0 = sl / (tp + sl) (exactly EV-zero by construction). The
                 directional signal that justified the entry tilts that up:
                 tilt = min(0.25, 0.05 * signal_strength
                                  + 0.10 * max(0, |delta| - 0.40)).
                 Because an absolute probability tilt buys a gross EV edge of
                 tilt * (tp + sl) * premium, wide barriers (e.g. tp 2.2 /
                 sl 0.6 in production) would let a +0.25 tilt claim more EV
                 than the premium itself. So the applied tilt is also capped
                 by the edge it creates: tilt <= MAX_EDGE / (tp + sl), keeping
                 gross EV <= MAX_EDGE (25%) of premium. With the default
                 0.25/0.15 barriers that cap (0.625) is inactive.
                 pop = clamp(p0 + tilt, 0.02, 0.98).
  * EV (gross) = (pop * tp - (1 - pop) * sl) * premium
  * EV (net)   = gross - round-trip costs (CostModel when available, else a
                 full-spread + per-contract-fee fallback, else 0).

Whether this belief is any good is precisely what the calibration reports
measure — the stamp only has to be frozen, deterministic, and honest.
"""

from typing import Dict, Optional

STAMP_VERSION = 2
POP_MODEL = "tp_sl_race_v2"

# Gross EV may never exceed this fraction of the premium at risk. The signal
# tilt is an absolute probability bump, so its EV edge scales with the barrier
# width (tp + sl); this cap keeps the frozen belief honest at wide barriers.
MAX_EDGE = 0.25

# Fallback round-trip friction per contract (2 sides of slippage + OCC fees)
# used only when the cost model itself is unavailable.
_FALLBACK_FEES_PER_CONTRACT = 0.08


def _f(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _round_trip_costs(bid: Optional[float], ask: Optional[float],
                      qty: int) -> float:
    """Dollar cost of buying then selling at the same quote. Fail-open to 0."""
    b, a = _f(bid, 0.0), _f(ask, 0.0)
    try:
        from cost_model import CostModel
        rt = CostModel().round_trip_cost(b, a, qty=qty)
        cost = _f(rt.get("cost_dollars"))
        if cost is not None:
            return max(0.0, cost)
    except Exception:
        pass
    if a > 0 and b > 0 and a >= b:
        return ((a - b) * 100.0 + _FALLBACK_FEES_PER_CONTRACT) * qty
    return 0.0


def compute_entry_stamp(option: Optional[Dict], dynamic_levels: Optional[Dict],
                        entry_price, qty,
                        bid=None, ask=None) -> Dict:
    """Frozen entry-time belief for one long single-leg option trade.

    Returns {} when the inputs can't support a stamp (no premium/size); never
    raises. The result is JSON-serializable and goes into trade_info['metrics'].
    """
    try:
        price = _f(entry_price)
        n = int(_f(qty, 0) or 0)
        if not price or price <= 0 or n <= 0:
            return {}

        levels = dynamic_levels if isinstance(dynamic_levels, dict) else {}
        opt = option if isinstance(option, dict) else {}

        tp = _f(levels.get("take_profit_percent"), 0.25) or 0.25
        sl = _f(levels.get("stop_loss_percent"), 0.15) or 0.15
        tp, sl = abs(tp), abs(sl)
        if tp + sl <= 0:
            return {}

        premium = price * 100.0 * n  # dollars at risk (long option max loss)

        # Gambler's-ruin baseline: EV-neutral if the entry had no edge at all.
        p0 = sl / (tp + sl)
        strength = max(0, int(_f(opt.get("confidence"), 0) or 0))
        delta = abs(_f(opt.get("delta"), 0.0) or 0.0)
        tilt = min(0.25, 0.05 * strength + 0.10 * max(0.0, delta - 0.40))
        # Edge cap: gross EV = tilt * (tp + sl) * premium, so bound the tilt
        # by the edge it buys. Inactive at narrow barriers (tp + sl <= 1).
        tilt = min(tilt, MAX_EDGE / (tp + sl))
        pop = min(0.98, max(0.02, p0 + tilt))

        gross_ev = (pop * tp - (1.0 - pop) * sl) * premium
        costs = _round_trip_costs(bid, ask, n)
        ev = gross_ev - costs

        return {
            "stamp_version": STAMP_VERSION,
            "pop_model": POP_MODEL,
            "expected_value": round(ev, 2),
            "probability_of_profit": round(pop, 4),
            "max_loss": round(premium, 2),
            "ev_per_dollar_risk": round(ev / premium, 4) if premium else None,
            "take_profit_pct": tp,
            "stop_loss_pct": sl,
            "signal_strength": strength,
            "entry_delta": delta,
            "round_trip_costs": round(costs, 2),
        }
    except Exception:
        return {}
