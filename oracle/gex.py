"""
Oracle 2.1 — Feature 3b: dealer gamma exposure (GEX). PURE, offline-testable.

``compute_gex(chain_rows, spot)`` turns an option chain snapshot into the
market-maker gamma-positioning picture the intraday profile cares about:

  * ``gex_total`` — net dealer gamma in dollar terms across the chain,
  * ``gex_flip_point`` — the "zero-gamma" strike where cumulative GEX crosses
    zero (below it dealers are typically short gamma, above it long),
  * ``gex_regime`` — ``"positive"`` (dealers long gamma -> they buy dips / sell
    rips -> vol-suppressing, mean-reverting) or ``"negative"`` (dealers short
    gamma -> they chase -> vol-amplifying, trend-extending),
  * ``spot_vs_gex_flip`` — +1 when spot is above the flip, -1 below, 0 on it.

Convention (documented so the sign is unambiguous): a CALL contributes
``+gamma·OI·100·spot`` and a PUT ``-gamma·OI·100·spot`` — i.e. dealers are
assumed long calls / short puts. Only the SIGN of the aggregate matters for the
regime label, so the exact scaling constant is irrelevant.

Fails to ``{"gex_status": "INSUFFICIENT", ...}`` (never raises) whenever the
chain lacks the open-interest / gamma needed to compute a meaningful exposure.
Chain rows are duck-typed dicts (see ``option_chain._process_contract``):
``type`` (call/put), ``strike``, ``gamma``, ``open_interest``.
"""

from typing import List, Optional


def _num(x) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def _row_get(row, *keys):
    if not isinstance(row, dict):
        return None
    for k in keys:
        if k in row and row[k] is not None:
            return row[k]
    return None


def _row_type(row) -> Optional[str]:
    t = _row_get(row, "type", "option_type", "right")
    if t is None:
        return None
    t = str(t).strip().lower()
    if t in ("c", "call"):
        return "call"
    if t in ("p", "put"):
        return "put"
    return None


def _parse_rows(chain_rows) -> List[dict]:
    out = []
    for r in (chain_rows or []):
        t = _row_type(r)
        strike = _num(_row_get(r, "strike", "strike_price"))
        gamma = _num(_row_get(r, "gamma"))
        oi = _num(_row_get(r, "open_interest", "oi", "openInterest"))
        if t is None or strike is None:
            continue
        out.append({"type": t, "strike": strike, "gamma": gamma,
                    "oi": oi if oi is not None else 0.0})
    return out


def compute_gex(chain_rows, spot) -> dict:
    """Net dealer gamma exposure for an option chain around ``spot``.

    Returns ``{gex_status: "OK", gex_total, gex_flip_point, gex_regime,
    spot_vs_gex_flip}`` or ``{gex_status: "INSUFFICIENT", gex_reason}``. Never
    raises.
    """
    try:
        spot_f = _num(spot)
        rows = _parse_rows(chain_rows)
        if spot_f is None or spot_f <= 0 or not rows:
            return {"gex_status": "INSUFFICIENT", "gex_reason": "no spot/rows"}

        usable = [r for r in rows if r["gamma"] is not None and r["oi"] > 0]
        if not usable:
            return {"gex_status": "INSUFFICIENT",
                    "gex_reason": "no open interest / gamma"}

        by_strike: dict = {}
        total = 0.0
        for r in usable:
            sign = 1.0 if r["type"] == "call" else -1.0
            g = sign * r["gamma"] * r["oi"] * 100.0 * spot_f
            by_strike[r["strike"]] = by_strike.get(r["strike"], 0.0) + g
            total += g

        # Cumulative-crossing estimate of the zero-gamma (flip) strike.
        strikes = sorted(by_strike)
        cum = 0.0
        pts = []
        for k in strikes:
            cum += by_strike[k]
            pts.append((k, cum))
        flip = None
        for i in range(1, len(pts)):
            k0, c0 = pts[i - 1]
            k1, c1 = pts[i]
            if c0 == 0.0:
                flip = k0
                break
            if (c0 < 0.0 < c1) or (c0 > 0.0 > c1):
                denom = (c1 - c0)
                frac = (-c0 / denom) if denom != 0.0 else 0.0
                flip = k0 + frac * (k1 - k0)
                break

        regime = "positive" if total > 0 else ("negative" if total < 0 else "neutral")
        if flip is None:
            spot_vs = 0
        elif spot_f > flip:
            spot_vs = 1
        elif spot_f < flip:
            spot_vs = -1
        else:
            spot_vs = 0

        return {
            "gex_status": "OK",
            "gex_total": round(total, 2),
            "gex_flip_point": round(flip, 4) if flip is not None else None,
            "gex_regime": regime,
            "spot_vs_gex_flip": spot_vs,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"gex_status": "INSUFFICIENT", "gex_reason": "error"}


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # No spot / no rows / empty -> INSUFFICIENT.
    for bad in (compute_gex([], 100), compute_gex(None, 100),
                compute_gex([{"type": "call", "strike": 100, "gamma": 0.01}], 0)):
        if bad.get("gex_status") != "INSUFFICIENT":
            print("FAIL: bad input should be INSUFFICIENT", bad); ok = False

    # Rows without OI -> INSUFFICIENT (gamma alone can't weight exposure).
    no_oi = [{"type": "call", "strike": 100, "gamma": 0.02, "open_interest": 0},
             {"type": "put", "strike": 100, "gamma": 0.02, "open_interest": 0}]
    if compute_gex(no_oi, 100).get("gex_status") != "INSUFFICIENT":
        print("FAIL: zero OI should be INSUFFICIENT"); ok = False

    # Call-dominated OI -> positive net GEX (dealers long gamma).
    call_heavy = [
        {"type": "call", "strike": 100, "gamma": 0.05, "open_interest": 5000},
        {"type": "call", "strike": 105, "gamma": 0.04, "open_interest": 4000},
        {"type": "put", "strike": 95, "gamma": 0.03, "open_interest": 500},
    ]
    r_ch = compute_gex(call_heavy, 100)
    if r_ch.get("gex_status") != "OK":
        print("FAIL: usable chain should be OK", r_ch); ok = False
    if r_ch.get("gex_total", 0) <= 0 or r_ch.get("gex_regime") != "positive":
        print("FAIL: call-heavy chain should be positive GEX", r_ch); ok = False

    # Put-dominated OI -> negative net GEX (dealers short gamma).
    put_heavy = [
        {"type": "put", "strike": 100, "gamma": 0.05, "open_interest": 6000},
        {"type": "put", "strike": 95, "gamma": 0.04, "open_interest": 5000},
        {"type": "call", "strike": 105, "gamma": 0.03, "open_interest": 400},
    ]
    r_ph = compute_gex(put_heavy, 100)
    if r_ph.get("gex_total", 0) >= 0 or r_ph.get("gex_regime") != "negative":
        print("FAIL: put-heavy chain should be negative GEX", r_ph); ok = False

    # A chain whose cumulative gamma crosses zero in the interior exposes a
    # flip point. Puts sit below, calls above, with heavier call OI so the
    # running sum turns positive between the 105 and 110 strikes.
    balanced = [
        {"type": "put", "strike": 90, "gamma": 0.05, "open_interest": 4000},
        {"type": "put", "strike": 95, "gamma": 0.05, "open_interest": 3000},
        {"type": "call", "strike": 105, "gamma": 0.05, "open_interest": 6000},
        {"type": "call", "strike": 110, "gamma": 0.05, "open_interest": 8000},
    ]
    r_bal = compute_gex(balanced, 100)
    if r_bal.get("gex_status") != "OK" or r_bal.get("gex_flip_point") is None:
        print("FAIL: balanced chain should expose a flip point", r_bal); ok = False
    if r_bal.get("spot_vs_gex_flip") not in (-1, 0, 1):
        print("FAIL: spot_vs_gex_flip must be a sign", r_bal); ok = False

    # Garbage rows never raise.
    for junk in ("x", 7, [object()], [{"weird": 1}]):
        try:
            compute_gex(junk, 100)
        except Exception as ex:  # pragma: no cover
            print("FAIL: raised on junk", junk, ex); ok = False

    print("oracle.gex self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
