"""
SKIP counterfactual resolver.

A SKIP decision (the bot declined a setup: EV-gate / portfolio / risk / budget /
duplicate) has no fill, so it cannot have a realized P/L. To give the abstention
class a learnable label we attach a *counterfactual* outcome: a signed "skip
quality" score derived from the forward UNDERLYING move, where POSITIVE means
skipping was the right call (we avoided an adverse move) and negative means we
missed a winner.

    CALL skip -> (entry - now) / entry * 100     (good/positive if price fell)
    PUT  skip -> (now - entry) / entry * 100     (good/positive if price rose)

This sign convention keeps every action on one scale for the RL/ML loop: higher
is better for CALL, PUT, *and* SKIP alike, so a SKIP earns positive reward in
exactly the states where trading would have lost money.

The episode row already carries the would-be direction (`rule_action`) and the
underlying price at decision time (`features.raw.underlying_price`, stamped by
the shadow recorder). After `horizon_min` wall-clock minutes the resolver fetches
the current underlying price and records the counterfactual as the row's outcome
(`gross_pnl_pct == net_pnl_pct`, since a skip has no execution cost).

Pure stdlib + the EpisodeStore interface; the math is fully testable with no
network (inject a stub `price_fn`). Every step is defensive: a single bad row
never aborts the batch.
"""

from datetime import datetime
import json
from typing import Callable, Dict, List, Optional


def counterfactual_return(direction, entry_px, now_px) -> Optional[float]:
    """Signed "skip quality" (%) from the forward underlying move.

    Positive => skipping was correct (we avoided an adverse move); negative =>
    we missed a winner. CALL skip is good when the price fell; PUT skip is good
    when it rose. Returns None on missing/non-positive entry, missing now price,
    or an unrecognised direction.
    """
    try:
        entry = float(entry_px)
        now = float(now_px)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or now <= 0:
        return None
    d = str(direction or "").upper()
    if d == "CALL":
        return (entry - now) / entry * 100.0
    if d == "PUT":
        return (now - entry) / entry * 100.0
    return None


def _entry_price_from_features(features_json) -> Optional[float]:
    """Pull features.raw.underlying_price out of a stored features blob."""
    try:
        feats = features_json if isinstance(features_json, dict) else json.loads(features_json)
    except (TypeError, ValueError):
        return None
    raw = (feats or {}).get("raw") or {}
    px = raw.get("underlying_price")
    try:
        return float(px) if px is not None else None
    except (TypeError, ValueError):
        return None


def _age_minutes(created_at, now: datetime) -> Optional[float]:
    """Minutes elapsed between an ISO `created_at` and `now` (None if unparsable)."""
    if not created_at:
        return None
    try:
        ts = datetime.fromisoformat(str(created_at))
    except ValueError:
        return None
    # Compare on the same naive/aware footing as `now`.
    if ts.tzinfo is not None and now.tzinfo is None:
        ts = ts.replace(tzinfo=None)
    elif ts.tzinfo is None and now.tzinfo is not None:
        ts = ts.replace(tzinfo=now.tzinfo)
    return (now - ts).total_seconds() / 60.0


def resolve_due_skips(
    store,
    price_fn: Callable[[str], Optional[float]],
    horizon_min: float = 390.0,
    now: Optional[datetime] = None,
) -> int:
    """Resolve open SKIP rows older than `horizon_min` with a counterfactual.

    For each due row: read the would-be direction + the entry underlying price,
    fetch the current underlying price via `price_fn`, compute the forward
    return, and write it as the outcome. Returns the number resolved.
    """
    now = now or datetime.now()
    try:
        rows: List[Dict] = store.open_skips()
    except Exception:
        return 0

    resolved = 0
    price_cache: Dict[str, Optional[float]] = {}
    for row in rows:
        try:
            age = _age_minutes(row.get("created_at"), now)
            if age is None or age < horizon_min:
                continue
            entry = _entry_price_from_features(row.get("features_json"))
            if entry is None:
                continue
            underlying = row.get("underlying")
            if not underlying:
                continue
            if underlying not in price_cache:
                try:
                    price_cache[underlying] = price_fn(underlying)
                except Exception:
                    price_cache[underlying] = None
            now_px = price_cache[underlying]
            cf = counterfactual_return(row.get("rule_action"), entry, now_px)
            if cf is None:
                continue
            ok = store.record_outcome(
                row.get("decision_id"),
                exit_price=now_px,
                gross_pnl_pct=cf,
                net_pnl_pct=cf,
                net_pnl_dollars=0.0,
                hold_days=int(round(horizon_min / 390.0)) or 1,
                outcome="skip_resolved",
                closed_at=now.isoformat(),
            )
            if ok:
                resolved += 1
        except Exception:
            continue
    return resolved


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import timedelta
    from episode_store import EpisodeStore

    ok = True

    # --- pure math (positive == skipping was correct) --------------------- #
    if not (counterfactual_return("CALL", 100.0, 95.0) > 0):
        print("FAIL: CALL skip with price falling should be good (positive)"); ok = False
    if not (counterfactual_return("CALL", 100.0, 105.0) < 0):
        print("FAIL: CALL skip with price rising should be bad (negative)"); ok = False
    if not (counterfactual_return("PUT", 100.0, 105.0) > 0):
        print("FAIL: PUT skip with price rising should be good (positive)"); ok = False
    if not (counterfactual_return("PUT", 100.0, 95.0) < 0):
        print("FAIL: PUT skip with price falling should be bad (negative)"); ok = False
    for bad in (
        counterfactual_return("CALL", None, 105.0),
        counterfactual_return("CALL", 0.0, 105.0),
        counterfactual_return("CALL", 100.0, None),
        counterfactual_return("HOLD", 100.0, 105.0),
    ):
        if bad is not None:
            print("FAIL: bad input should be None", bad); ok = False

    # --- resolver against a temp store ------------------------------------ #
    store = EpisodeStore(":memory:")

    def feats(px):
        return {"feature_version": "t", "raw": {"underlying_price": px}, "state_key": "k"}

    # An old CALL skip on SPY (entry 100); price now 90 -> skip was good (+10%).
    old_skip = store.log_decision(
        symbol="SPY", underlying="SPY", strat="t", features=feats(100.0),
        quote=None, modeled_cost=None, rule_action="CALL", rule_confidence=0.0,
        gate=None, chosen_action="SKIP", qty=1, mode="live-paper-blocked",
    )
    # Backdate created_at well past the horizon.
    store.conn.execute(
        "UPDATE episodes SET created_at=? WHERE decision_id=?",
        ((datetime.now() - timedelta(minutes=500)).isoformat(), old_skip),
    )
    store.conn.commit()

    # A too-recent skip should be left open.
    fresh_skip = store.log_decision(
        symbol="QQQ", underlying="QQQ", strat="t", features=feats(100.0),
        quote=None, modeled_cost=None, rule_action="PUT", rule_confidence=0.0,
        gate=None, chosen_action="SKIP", qty=1, mode="live-paper-blocked",
    )

    # A real (non-SKIP) open decision must be untouched.
    real = store.log_decision(
        symbol="IWM", underlying="IWM", strat="t", features=feats(100.0),
        quote=None, modeled_cost=None, rule_action="CALL", rule_confidence=0.0,
        gate=None, chosen_action="CALL", qty=1, mode="1DTE",
    )

    prices = {"SPY": 90.0, "QQQ": 110.0, "IWM": 105.0}
    n = resolve_due_skips(store, lambda s: prices.get(s), horizon_min=390)
    if n != 1:
        print("FAIL: exactly one due skip should resolve", n); ok = False

    rows = {r["decision_id"]: r for r in store._rows("SELECT * FROM episodes")}
    if rows[old_skip]["outcome"] != "skip_resolved":
        print("FAIL: old skip should be resolved"); ok = False
    if abs((rows[old_skip]["net_pnl_pct"] or 0) - 10.0) > 1e-9:
        print("FAIL: CALL skip with price 100->90 should be +10%",
              rows[old_skip]["net_pnl_pct"]); ok = False
    if rows[fresh_skip]["outcome"] is not None:
        print("FAIL: fresh skip should stay open"); ok = False
    if rows[real]["outcome"] is not None:
        print("FAIL: non-skip decision must not be resolved here"); ok = False

    # Missing entry price -> skipped, not crashed.
    no_px = store.log_decision(
        symbol="DIA", underlying="DIA", strat="t",
        features={"raw": {}}, quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="live-paper-blocked",
    )
    store.conn.execute(
        "UPDATE episodes SET created_at=? WHERE decision_id=?",
        ((datetime.now() - timedelta(minutes=500)).isoformat(), no_px),
    )
    store.conn.commit()
    n2 = resolve_due_skips(store, lambda s: 100.0, horizon_min=390)
    if n2 != 0:
        print("FAIL: row with no entry price should not resolve", n2); ok = False

    store.close()
    print("skip_counterfactual self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
