"""Per-underlying cap-skip counterfactual resolver (option-repriced).

When the scheduler declines a 2nd+ position on an underlying because of the
per-underlying capacity gate (`cfg.max_per_underlying`, optionally tightened by
the low-IV regime filter), the setup we *would* have bought is recorded as a
SKIP episode tagged ``mode="cap-skip-cf"`` carrying the concrete contract and
the entry ask we'd have paid (stamped under ``features.capskip``).

Unlike ``skip_counterfactual`` (which scores the forgone *underlying* move), this
resolver reprices the actual *contract* at end of session and books the honest
long-option return:

    return% = (exit_mid - entry_ask) / entry_ask * 100

We always go long the option (buy-to-open a call or a put), so the sign is
independent of direction: a positive number means the blocked contract would
have gained (the cap cost us), negative means the cap saved us. This is the fair
test of the throttle because it captures the theta / IV-crush the filter is
meant to avoid — the very thing an underlying-move counterfactual misses.

Pure stdlib + the EpisodeStore interface; fully testable with no network (inject
a stub ``option_price_fn``). Every step is defensive: one bad row never aborts
the batch. Resolving is advisory-only; it never touches the broker.
"""

from datetime import datetime
import json
from typing import Callable, Dict, List, Optional


# --------------------------------------------------------------------------- #
# Pure math
# --------------------------------------------------------------------------- #
def option_cf_return(entry_ask, exit_mid) -> Optional[float]:
    """Long-option return (%) of the blocked contract: (exit-entry)/entry*100.

    We buy-to-open, so the sign does not depend on call/put. Returns None on a
    missing / non-positive entry ask, or a missing / negative exit price.
    """
    try:
        entry = float(entry_ask)
        exit_px = float(exit_mid)
    except (TypeError, ValueError):
        return None
    if entry <= 0 or exit_px < 0:
        return None
    return (exit_px - entry) / entry * 100.0


def _capskip_block(features_json) -> Dict:
    """Pull the ``features.capskip`` blob out of a stored features record."""
    try:
        feats = features_json if isinstance(features_json, dict) else json.loads(features_json)
    except (TypeError, ValueError):
        return {}
    return (feats or {}).get("capskip") or {}


def _exit_mid_from_quote(quote) -> Optional[float]:
    """Normalise an option_price_fn result to a single exit price.

    Accepts a ``{bid,ask,mid}`` dict (uses mid, else ask, else bid) or a bare
    number. Returns None when nothing usable is present.
    """
    if quote is None:
        return None
    if isinstance(quote, dict):
        for key in ("mid", "ask", "bid"):
            val = quote.get(key)
            try:
                if val is not None and float(val) > 0:
                    return float(val)
            except (TypeError, ValueError):
                continue
        return None
    try:
        return float(quote)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# Resolver
# --------------------------------------------------------------------------- #
def resolve_due_capskips(
    store,
    option_price_fn: Callable[[str], object],
    now: Optional[datetime] = None,
) -> int:
    """Reprice every open cap-skip contract and book its option counterfactual.

    Intended to run at end of session (the caller gates *when*). For each open
    ``mode='cap-skip-cf'`` row: read the entry ask, reprice the contract via
    ``option_price_fn(occ_symbol)``, compute the long-option return, and write it
    as the outcome (``outcome='capskip_resolved'``). Returns the number resolved.
    """
    now = now or datetime.now()
    try:
        rows: List[Dict] = store.open_capskips()
    except Exception:
        return 0

    resolved = 0
    price_cache: Dict[str, Optional[float]] = {}
    for row in rows:
        try:
            block = _capskip_block(row.get("features_json"))
            entry_ask = block.get("entry_ask")
            if entry_ask is None:
                continue
            occ = row.get("symbol")
            if not occ:
                continue
            if occ not in price_cache:
                try:
                    price_cache[occ] = _exit_mid_from_quote(option_price_fn(occ))
                except Exception:
                    price_cache[occ] = None
            exit_mid = price_cache[occ]
            ret = option_cf_return(entry_ask, exit_mid)
            if ret is None:
                continue
            ok = store.record_outcome(
                row.get("decision_id"),
                exit_price=exit_mid,
                gross_pnl_pct=ret,
                net_pnl_pct=ret,
                net_pnl_dollars=0.0,
                hold_days=0,
                outcome="capskip_resolved",
                closed_at=now.isoformat(),
            )
            if ok:
                resolved += 1
        except Exception:
            continue
    return resolved


# --------------------------------------------------------------------------- #
# Reporting
# --------------------------------------------------------------------------- #
def summarize(store) -> Dict:
    """Aggregate resolved cap-skip counterfactuals, split low-IV vs regular.

    Returns ``{"all": {...}, "low_iv": {...}, "regular": {...}}`` where each
    bucket has ``trades``, ``win_rate`` (fraction with return>0), ``avg_return``
    (mean %), and ``total_return`` (sum %). "win" here means the blocked trade
    would have won, i.e. the cap cost us; a low win_rate is the filter earning
    its keep.
    """
    def _empty():
        return {"trades": 0, "wins": 0, "win_rate": 0.0, "avg_return": 0.0,
                "total_return": 0.0}

    buckets = {"all": _empty(), "low_iv": _empty(), "regular": _empty()}
    try:
        rows = store.completed()
    except Exception:
        return buckets

    for row in rows:
        if row.get("outcome") != "capskip_resolved":
            continue
        ret = row.get("net_pnl_pct")
        if ret is None:
            continue
        ret = float(ret)
        block = _capskip_block(row.get("features_json"))
        low_iv = bool(block.get("low_iv"))
        targets = ["all", "low_iv" if low_iv else "regular"]
        for name in targets:
            b = buckets[name]
            b["trades"] += 1
            if ret > 0:
                b["wins"] += 1
            b["total_return"] += ret

    for b in buckets.values():
        if b["trades"]:
            b["win_rate"] = b["wins"] / b["trades"]
            b["avg_return"] = b["total_return"] / b["trades"]
    return buckets


def format_summary(summary: Dict) -> str:
    """Human-readable one-block report of a ``summarize()`` result."""
    lines = ["=== Cap-skip counterfactual (option-repriced) ==="]
    for name in ("all", "low_iv", "regular"):
        b = summary.get(name) or {}
        n = b.get("trades", 0)
        if not n:
            lines.append(f"{name:8s}: no resolved cap-skips yet")
            continue
        lines.append(
            f"{name:8s}: n={n:4d}  would-win%={b['win_rate']*100:5.1f}  "
            f"avg_opt_return%={b['avg_return']:+6.1f}  "
            f"total%={b['total_return']:+8.1f}"
        )
    lines.append(
        "(would-win% = share of BLOCKED contracts that would have gained; "
        "lower is better for the throttle)"
    )
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from episode_store import EpisodeStore

    ok = True

    # --- pure math -------------------------------------------------------- #
    if abs(option_cf_return(1.00, 1.50) - 50.0) > 1e-9:
        print("FAIL: 1.00->1.50 should be +50%"); ok = False
    if abs(option_cf_return(2.00, 1.00) - (-50.0)) > 1e-9:
        print("FAIL: 2.00->1.00 should be -50%"); ok = False
    if abs(option_cf_return(1.00, 0.0) - (-100.0)) > 1e-9:
        print("FAIL: expiry-worthless should be -100%"); ok = False
    for bad in (
        option_cf_return(None, 1.5),
        option_cf_return(0.0, 1.5),
        option_cf_return(1.0, None),
        option_cf_return(1.0, -1.0),
    ):
        if bad is not None:
            print("FAIL: bad input should be None", bad); ok = False

    # --- quote normalisation --------------------------------------------- #
    if _exit_mid_from_quote({"bid": 1.0, "ask": 2.0, "mid": 1.5}) != 1.5:
        print("FAIL: mid should win"); ok = False
    if _exit_mid_from_quote({"bid": 1.0, "ask": 2.0}) != 2.0:
        print("FAIL: ask fallback when no mid"); ok = False
    if _exit_mid_from_quote(1.25) != 1.25:
        print("FAIL: bare float"); ok = False
    if _exit_mid_from_quote(None) is not None:
        print("FAIL: None quote -> None"); ok = False

    # --- resolver against a temp store ----------------------------------- #
    store = EpisodeStore(":memory:")

    def feats(entry_ask, low_iv):
        return {
            "feature_version": "t",
            "raw": {"underlying_price": 100.0},
            "state_key": "k",
            "capskip": {"entry_ask": entry_ask, "low_iv": low_iv,
                        "base_cap": 2, "eff_cap": 1, "realized_vol": 0.12,
                        "direction": "call", "contract": "SPY260101C00500000"},
        }

    # A low-IV cap-skip: entry ask 1.00, will reprice to 1.50 -> +50% (a "would-win").
    win = store.log_decision(
        symbol="SPY260101C00500000", underlying="SPY", strat="t",
        features=feats(1.00, True), quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="cap-skip-cf",
    )
    # A regular cap-skip: entry ask 2.00, reprices to 1.00 -> -50% (throttle helped).
    loss = store.log_decision(
        symbol="QQQ260101P00400000", underlying="QQQ", strat="t",
        features=feats(2.00, False), quote=None, modeled_cost=None,
        rule_action="PUT", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="cap-skip-cf",
    )
    # A non-capskip SKIP must be untouched by this resolver.
    other = store.log_decision(
        symbol="IWM260101C00200000", underlying="IWM", strat="t",
        features=feats(1.00, False), quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="live-paper-blocked",
    )

    quotes = {
        "SPY260101C00500000": {"bid": 1.4, "ask": 1.6, "mid": 1.5},
        "QQQ260101P00400000": {"bid": 0.9, "ask": 1.1, "mid": 1.0},
        "IWM260101C00200000": {"bid": 5.0, "ask": 5.2, "mid": 5.1},
    }
    n = resolve_due_capskips(store, lambda s: quotes.get(s))
    if n != 2:
        print("FAIL: exactly two cap-skips should resolve", n); ok = False

    rows = {r["decision_id"]: r for r in store._rows("SELECT * FROM episodes")}
    if rows[win]["outcome"] != "capskip_resolved":
        print("FAIL: win row should resolve"); ok = False
    if abs((rows[win]["net_pnl_pct"] or 0) - 50.0) > 1e-9:
        print("FAIL: win return should be +50%", rows[win]["net_pnl_pct"]); ok = False
    if abs((rows[loss]["net_pnl_pct"] or 0) - (-50.0)) > 1e-9:
        print("FAIL: loss return should be -50%", rows[loss]["net_pnl_pct"]); ok = False
    if rows[other]["outcome"] is not None:
        print("FAIL: non-capskip skip must stay open"); ok = False

    # --- summary ---------------------------------------------------------- #
    summary = summarize(store)
    if summary["all"]["trades"] != 2:
        print("FAIL: summary all trades should be 2", summary["all"]); ok = False
    if summary["low_iv"]["trades"] != 1 or summary["regular"]["trades"] != 1:
        print("FAIL: summary split wrong", summary); ok = False
    if abs(summary["low_iv"]["win_rate"] - 1.0) > 1e-9:
        print("FAIL: low_iv would-win% should be 100", summary["low_iv"]); ok = False
    if abs(summary["regular"]["win_rate"] - 0.0) > 1e-9:
        print("FAIL: regular would-win% should be 0", summary["regular"]); ok = False

    # Missing entry ask -> skipped, not crashed.
    no_ask = store.log_decision(
        symbol="DIA260101C00300000", underlying="DIA", strat="t",
        features={"capskip": {}}, quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="cap-skip-cf",
    )
    n2 = resolve_due_capskips(store, lambda s: {"mid": 1.0})
    if n2 != 0:
        print("FAIL: row with no entry ask should not resolve", n2); ok = False
    _ = no_ask

    store.close()
    print("capskip_cf self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--summary" in sys.argv:
        from episode_store import EpisodeStore
        st = EpisodeStore("episodes.db")
        print(format_summary(summarize(st)))
        st.close()
        sys.exit(0)
    sys.exit(_self_test())
