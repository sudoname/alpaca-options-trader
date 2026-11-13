"""Explicit outcome states + fills-based net-dollar accounting (pure, no I/O).

Distinguishes a policy skip from contract outcomes so promotion-grade
reports never conflate "we chose not to trade" with "the trade lost money"
or "we don't know what happened":

  - FILLED_RESOLVED    entry filled, exits cover it, net P&L resolved
                       (pnl_source: 'reported' | 'fills' | 'mark').
  - CONFIRMED_NO_FILL  order reached the broker and terminated with zero
                       fills (reject/cancel/expired order). Zero exposure,
                       zero P&L; the broker's reason is retained.
  - POLICY_SKIP        candidate was never accepted (gate block, ranking
                       loss, unsubmitted). Zero P&L by definition.
  - UNRESOLVED         open residual quantity, incomplete exits, or a
                       submitted order with no fill evidence yet.
  - MISSING_COVERAGE   filled but the data needed to settle net P&L is
                       absent (e.g. expired without observed settlement).

Zero-P&L states are CONFIRMED_NO_FILL and POLICY_SKIP only. Unresolved and
missing-coverage rows must be excluded (and counted) rather than treated as
zero. Simulated/counterfactual outcome events never resolve an accepted row.

Net dollars come from actual entry and exit fills: quantity-weighted
average prices over matched quantity, times the contract multiplier, minus
total fees (charged once, not per fill).

Run `python outcome_states.py --selftest` for a hermetic self-check.
"""
import json
import math

FILLED_RESOLVED = "filled_resolved"
CONFIRMED_NO_FILL = "confirmed_no_fill"
POLICY_SKIP = "policy_skip"
UNRESOLVED = "unresolved"
MISSING_COVERAGE = "missing_coverage"

ALL_STATES = (FILLED_RESOLVED, CONFIRMED_NO_FILL, POLICY_SKIP,
              UNRESOLVED, MISSING_COVERAGE)
ZERO_PNL_STATES = (CONFIRMED_NO_FILL, POLICY_SKIP)

CONTRACT_MULTIPLIER = 100.0
_EPS = 1e-9

# Broker-evidence markers that justify counting a legacy null as a
# confirmed no-fill (provenance requirement).
_LEGACY_NO_FILL_MARKERS = frozenset((
    "canceled", "cancelled", "rejected", "expired_unfilled",
    "broker_canceled_no_fill", "broker_cancelled_no_fill",
    "broker_expired_no_fill", "broker_rejected_no_fill",
    "broker_done_for_day_no_fill",
))


def _num(value):
    """float(value) if it is a finite number, else None."""
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    return f if math.isfinite(f) else None


def _fill_qty_price(fill):
    """Extract (qty, price) from a fill dict; (None, None) if unusable."""
    if not isinstance(fill, dict):
        return (None, None)
    q = _num(fill.get("qty", fill.get("quantity")))
    p = _num(fill.get("price"))
    if q is None or p is None or q <= 0 or p <= 0:
        return (None, None)
    return (q, p)


def _avg_and_qty(fills):
    """Quantity-weighted average price and total qty over valid fills."""
    total_qty = 0.0
    total_cost = 0.0
    for f in fills or ():
        q, p = _fill_qty_price(f)
        if q is None:
            continue
        total_qty += q
        total_cost += q * p
    if total_qty <= _EPS:
        return (None, 0.0)
    return (total_cost / total_qty, total_qty)


def net_dollars_from_fills(entry_fills, exit_fills,
                           multiplier=CONTRACT_MULTIPLIER, fees=0.0):
    """Realized net dollars from actual entry and exit fills.

    net = (exit_avg - entry_avg) * matched_qty * multiplier - fees
    where matched_qty = min(entry_qty, exit_qty) and the averages are
    quantity-weighted over valid (qty>0, price>0) fills. `fees` is the
    TOTAL fee for the round trip — charged once, never per fill.

    Returns a dict with net_pnl_dollars, matched_qty, entry_qty, exit_qty,
    residual_qty, entry_avg, exit_avg, fees — or None when the inputs
    cannot support the computation (no valid entry or exit fills).
    """
    entry_avg, entry_qty = _avg_and_qty(entry_fills)
    exit_avg, exit_qty = _avg_and_qty(exit_fills)
    if entry_avg is None or exit_avg is None:
        return None
    mult = _num(multiplier)
    fee = _num(fees)
    if mult is None or mult <= 0:
        return None
    fee = fee if fee is not None and fee >= 0 else 0.0
    matched = min(entry_qty, exit_qty)
    return {
        "net_pnl_dollars": (exit_avg - entry_avg) * matched * mult - fee,
        "matched_qty": matched,
        "entry_qty": entry_qty,
        "exit_qty": exit_qty,
        "residual_qty": entry_qty - matched,
        "entry_avg": entry_avg,
        "exit_avg": exit_avg,
        "fees": fee,
    }


def _event_payload(event):
    """Tolerant payload read: dict `payload` or JSON `payload_json`."""
    payload = event.get("payload")
    if isinstance(payload, dict):
        return payload
    raw = event.get("payload_json")
    if isinstance(raw, str) and raw:
        try:
            parsed = json.loads(raw)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {}


def _result(state, reason, *, entry_qty=0.0, exit_qty=0.0,
            matched_qty=0.0, net=None, pnl_source=None):
    return {
        "state": state,
        "reason": reason,
        "entry_qty": entry_qty,
        "exit_qty": exit_qty,
        "residual_qty": max(entry_qty - matched_qty, 0.0),
        "matched_qty": matched_qty,
        "net_pnl_dollars": net,
        "pnl_source": pnl_source,
    }


def classify_decision(snapshots, events):
    """Classify one ledger decision into an explicit outcome state.

    `snapshots` / `events` are the rows from ResearchLedger.get_decision.
    Only broker evidence resolves an accepted row: fills resolve P&L,
    reject/cancel events confirm a no-fill, expiry without observed
    settlement is MISSING_COVERAGE, and anything in between stays
    UNRESOLVED. Simulated (counterfactual) outcome events are ignored.
    """
    snapshots = snapshots or ()
    events = events or ()

    accepted = [s for s in snapshots if s.get("accepted") == 1]
    if not accepted:
        reason = None
        for s in snapshots:
            if s.get("reject_code"):
                reason = s.get("reject_code")
                break
        return _result(POLICY_SKIP, reason or "not_accepted",
                       net=0.0, pnl_source="policy")

    entry_fills = []
    exit_fills = []
    submitted = False
    terminal_nofill_reason = None
    expiry_terminal = False
    reported_net = None

    for e in events:
        etype = e.get("event_type")
        payload = _event_payload(e)
        if etype == "submitted":
            submitted = True
        elif etype in ("fill", "partial_fill"):
            side = str(payload.get("side") or "").lower()
            fill = {"qty": e.get("qty"), "price": e.get("price")}
            if side.startswith("sell"):
                exit_fills.append(fill)
            else:
                entry_fills.append(fill)
        elif etype == "exit":
            exit_fills.append({"qty": e.get("qty"), "price": e.get("price")})
        elif etype in ("reject", "cancel"):
            if terminal_nofill_reason is None:
                terminal_nofill_reason = (
                    payload.get("reason") or f"broker_{etype}")
        elif etype in ("expiry", "assignment"):
            expiry_terminal = True
        elif etype == "outcome":
            if payload.get("class") == "realized":
                net = _num(payload.get("net_pnl_dollars"))
                if net is not None:
                    reported_net = net
            # simulated outcomes never resolve an accepted row

    _, entry_qty = _avg_and_qty(entry_fills)
    _, exit_qty = _avg_and_qty(exit_fills)

    if entry_qty <= _EPS:
        if terminal_nofill_reason is not None:
            return _result(CONFIRMED_NO_FILL, terminal_nofill_reason,
                           net=0.0, pnl_source="broker")
        reason = "no_fill_evidence" if submitted else "not_submitted"
        return _result(UNRESOLVED, reason)

    matched = min(entry_qty, exit_qty)
    residual = entry_qty - matched
    if residual > _EPS:
        if expiry_terminal:
            return _result(MISSING_COVERAGE,
                           "expired_without_observed_settlement",
                           entry_qty=entry_qty, exit_qty=exit_qty,
                           matched_qty=matched)
        return _result(UNRESOLVED, "open_residual_qty",
                       entry_qty=entry_qty, exit_qty=exit_qty,
                       matched_qty=matched)

    if reported_net is not None:
        return _result(FILLED_RESOLVED, "reported_outcome",
                       entry_qty=entry_qty, exit_qty=exit_qty,
                       matched_qty=matched, net=reported_net,
                       pnl_source="reported")

    computed = net_dollars_from_fills(entry_fills, exit_fills)
    if computed is not None:
        return _result(FILLED_RESOLVED, "computed_from_fills",
                       entry_qty=entry_qty, exit_qty=exit_qty,
                       matched_qty=matched,
                       net=computed["net_pnl_dollars"],
                       pnl_source="fills")

    return _result(MISSING_COVERAGE, "net_not_computable",
                   entry_qty=entry_qty, exit_qty=exit_qty,
                   matched_qty=matched)


def classify_legacy_record(record):
    """Classify a legacy trading_history row (JSON dict) into a state.

    Legacy rows carry mark-derived P&L; a null P&L only counts as a
    confirmed no-fill when the row carries explicit terminal no-fill
    provenance — otherwise it stays unresolved.
    """
    record = record or {}
    net = _num(record.get("net_pnl_dollars"))
    if net is not None and record.get("pnl_source") == "fills":
        return _result(FILLED_RESOLVED, "legacy_fills",
                       net=net, pnl_source="fills")
    pnl = _num(record.get("pnl"))
    pnl_pct = _num(record.get("pnl_percent"))
    if pnl is not None or pnl_pct is not None:
        return _result(FILLED_RESOLVED, "legacy_mark",
                       net=pnl, pnl_source="mark")
    for key in ("status", "outcome", "reason"):
        marker = str(record.get(key) or "").lower()
        if marker in _LEGACY_NO_FILL_MARKERS:
            return _result(CONFIRMED_NO_FILL, marker,
                           net=0.0, pnl_source="legacy")
    return _result(UNRESOLVED, "legacy_null_without_provenance")


# --------------------------------------------------------------------------- #
# self-test                                                                   #
# --------------------------------------------------------------------------- #

def _selftest():
    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    # --- net_dollars_from_fills ---
    r = net_dollars_from_fills([{"qty": 2, "price": 1.0}],
                               [{"qty": 2, "price": 1.25}])
    check("net basic 2x(1.00->1.25) = +50",
          r and abs(r["net_pnl_dollars"] - 50.0) < 1e-9
          and r["matched_qty"] == 2.0 and r["residual_qty"] == 0.0)

    r = net_dollars_from_fills([{"qty": 2, "price": 1.0}],
                               [{"qty": 1, "price": 1.5},
                                {"qty": 1, "price": 1.0}],
                               fees=1.3)
    check("fees charged once: 2x(1.00->1.25) - 1.30 = 48.70",
          r and abs(r["net_pnl_dollars"] - 48.7) < 1e-9)

    r = net_dollars_from_fills([{"qty": 3, "price": 2.0}],
                               [{"qty": 1, "price": 2.5}])
    check("partial match: matched=1 residual=2 net=+50",
          r and r["matched_qty"] == 1.0 and r["residual_qty"] == 2.0
          and abs(r["net_pnl_dollars"] - 50.0) < 1e-9)

    check("unusable fills -> None",
          net_dollars_from_fills([], [{"qty": 1, "price": 1.0}]) is None
          and net_dollars_from_fills([{"qty": 0, "price": 1}], []) is None
          and net_dollars_from_fills(
              [{"qty": "x", "price": 1}], [{"qty": 1, "price": 1}]) is None)

    # --- classify_decision ---
    skip = classify_decision(
        [{"accepted": 0, "reject_code": "gate:pop_too_low"}], [])
    check("policy skip retains reason, zero P&L",
          skip["state"] == POLICY_SKIP
          and skip["reason"] == "gate:pop_too_low"
          and skip["net_pnl_dollars"] == 0.0)

    acc = [{"accepted": 1, "reject_code": None}]
    nofill = classify_decision(acc, [
        {"event_type": "submitted", "payload": {}},
        {"event_type": "cancel",
         "payload": {"reason": "broker_canceled_no_fill"}},
    ])
    check("confirmed no-fill from broker cancel, zero P&L",
          nofill["state"] == CONFIRMED_NO_FILL
          and nofill["reason"] == "broker_canceled_no_fill"
          and nofill["net_pnl_dollars"] == 0.0
          and nofill["pnl_source"] == "broker")

    rej = classify_decision(acc, [
        {"event_type": "reject",
         "payload_json": json.dumps({"reason": "broker_rejected_no_fill"})},
    ])
    check("broker reject (payload_json) retains its own reason",
          rej["state"] == CONFIRMED_NO_FILL
          and rej["reason"] == "broker_rejected_no_fill")

    unres = classify_decision(acc, [{"event_type": "submitted"}])
    check("submitted without fills is unresolved (not zero)",
          unres["state"] == UNRESOLVED
          and unres["reason"] == "no_fill_evidence"
          and unres["net_pnl_dollars"] is None)

    resid = classify_decision(acc, [
        {"event_type": "fill", "qty": 2, "price": 1.0,
         "payload": {"side": "buy"}},
        {"event_type": "exit", "qty": 1, "price": 1.2},
    ])
    check("open residual is unresolved",
          resid["state"] == UNRESOLVED
          and resid["reason"] == "open_residual_qty"
          and resid["residual_qty"] == 1.0)

    filled = classify_decision(acc, [
        {"event_type": "fill", "qty": 2, "price": 1.0,
         "payload": {"side": "buy"}},
        {"event_type": "fill", "qty": 2, "price": 1.25,
         "payload": {"side": "sell_to_close"}},
    ])
    check("filled resolved from fills: +50, source=fills",
          filled["state"] == FILLED_RESOLVED
          and abs(filled["net_pnl_dollars"] - 50.0) < 1e-9
          and filled["pnl_source"] == "fills")

    reported = classify_decision(acc, [
        {"event_type": "fill", "qty": 1, "price": 2.0,
         "payload": {"side": "buy"}},
        {"event_type": "exit", "qty": 1, "price": 2.1},
        {"event_type": "outcome",
         "payload": {"class": "realized", "net_pnl_dollars": 9.5}},
    ])
    check("reported realized outcome wins: 9.5, source=reported",
          reported["state"] == FILLED_RESOLVED
          and reported["net_pnl_dollars"] == 9.5
          and reported["pnl_source"] == "reported")

    expired = classify_decision(acc, [
        {"event_type": "fill", "qty": 1, "price": 2.0,
         "payload": {"side": "buy"}},
        {"event_type": "expiry", "payload": {"class": "realized"}},
    ])
    check("expiry without observed settlement = missing coverage",
          expired["state"] == MISSING_COVERAGE
          and expired["reason"] == "expired_without_observed_settlement")

    sim = classify_decision(acc, [
        {"event_type": "submitted"},
        {"event_type": "outcome",
         "payload": {"class": "simulated",
                     "counterfactual_pnl_pct": 3.0}},
    ])
    check("simulated outcome never resolves an accepted row",
          sim["state"] == UNRESOLVED)

    # --- classify_legacy_record ---
    leg = classify_legacy_record({"pnl": -120.0, "pnl_percent": -10.0})
    check("legacy mark P&L resolves with source=mark",
          leg["state"] == FILLED_RESOLVED and leg["pnl_source"] == "mark")
    leg = classify_legacy_record({"status": "canceled"})
    check("legacy explicit cancel = confirmed no-fill",
          leg["state"] == CONFIRMED_NO_FILL)
    leg = classify_legacy_record({"pnl": None})
    check("legacy null without provenance stays unresolved",
          leg["state"] == UNRESOLVED
          and leg["reason"] == "legacy_null_without_provenance")

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(_selftest())
