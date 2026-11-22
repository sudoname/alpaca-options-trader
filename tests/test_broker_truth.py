"""Broker activities are matched by IDs, never by symbols or nearby times."""
from datetime import datetime, timezone

from oracle.ml.broker_truth import audit_activities
from oracle.ml.ingest import LedgerEvent


T = datetime(2026, 11, 20, tzinfo=timezone.utc)


def event(kind, decision, oid):
    return LedgerEvent(event_key=f"{decision}-{kind}", decision_id=decision,
                       event_type=kind, event_at=T, available_at=T,
                       broker_order_id=oid, qty=None, price=None, fees=None,
                       status=None, source="paper", payload={})


def fill(aid, oid, side, qty, price, symbol="SPY261120C00500000"):
    return dict(id=aid, order_id=oid, activity_type="FILL", side=side,
                qty=str(qty), price=str(price), symbol=symbol)


def fee(aid, oid, amount):
    return dict(id=aid, order_id=oid, activity_type="FEE",
                net_amount=str(-amount))


def test_exact_partial_fills_and_fees_charged_once():
    events = {"d": [event("submitted", "d", "entry"),
                    event("exit", "d", "exit")]}
    rows = [fill("x1", "entry", "buy", 1, 2.00),
            fill("x2", "entry", "buy", 1, 2.10),
            fill("x3", "exit", "sell", 2, 2.50),
            fill("x3", "exit", "sell", 2, 2.50),
            fee("f1", "entry", .04), fee("f2", "exit", .06)]
    report = audit_activities(events, rows)
    row = report["decisions"][0]
    assert row["state"] == "fill_backed_after_order_fees"
    assert row["pnl_before_fees"] == 90.0
    assert row["pnl_after_order_fees"] == 89.9
    assert row["entry_qty"] == row["exit_qty"] == 2
    assert report["activity_counts"]["duplicate_activity_id"] == 1


def test_no_fee_row_is_unknown_not_zero():
    events = {"d": [event("submitted", "d", "entry"),
                    event("exit", "d", "exit")]}
    report = audit_activities(events, [fill("x1", "entry", "buy", 1, 1),
                                       fill("x2", "exit", "sell", 1, 2),
                                       fee("f1", "entry", 0)])
    assert report["decisions"][0]["state"] == "fill_backed_fee_unknown"
    assert report["decisions"][0]["pnl_before_fees"] == 100.0
    assert report["sum_after_order_fees"] == 0


def test_partial_exit_and_unallocated_fee_do_not_create_profit():
    events = {"d": [event("submitted", "d", "entry"),
                    event("exit", "d", "exit")]}
    report = audit_activities(events, [fill("x1", "entry", "buy", 2, 1),
                                       fill("x2", "exit", "sell", 1, 2),
                                       fee("f1", "entry", .02),
                                       fee("f2", "exit", .02),
                                       fee("f3", "", .01)])
    assert report["decisions"][0]["state"] == "unresolved"
    assert report["decisions"][0]["reason"] == "unmatched_position_quantity"
    assert report["unallocated_account_fees"] == .01
    assert report["decisions"][0]["unmatched_qty"] == 1
    assert not report["account_fee_reconciliation_complete"]


def test_two_exact_exit_orders_reconcile_one_decision():
    events = {"d": [event("submitted", "d", "entry"),
                    event("exit", "d", "exit-1"),
                    event("exit", "d", "exit-2")]}
    rows = [fill("x1", "entry", "buy", 2, 1),
            fill("x2", "exit-1", "sell", 1, 2),
            fill("x3", "exit-2", "sell", 1, 3),
            fee("f1", "entry", .02), fee("f2", "exit-1", .01),
            fee("f3", "exit-2", .01)]
    result = audit_activities(events, rows)["decisions"][0]
    assert result["state"] == "fill_backed_after_order_fees"
    assert result["pnl_after_order_fees"] == 299.96


def test_no_exit_order_is_mark_derived_only_with_unambiguous_history():
    events = {"d": [event("submitted", "d", "entry")]}
    row = {"order_id": "entry", "symbol": "SPY261120C00500000", "pnl": 999}
    report = audit_activities(events, [fill("x1", "entry", "buy", 1, 1)], [row])
    assert report["decisions"][0]["state"] == "mark_derived"
    report = audit_activities(events, [fill("x1", "entry", "buy", 1, 1)],
                              [row, row])
    assert report["decisions"][0]["state"] == "unresolved"


def test_shared_order_id_cannot_claim_two_decisions():
    events = {"d1": [event("submitted", "d1", "entry")],
              "d2": [event("submitted", "d2", "entry")]}
    report = audit_activities(events, [fill("x1", "entry", "buy", 1, 1)],
                              [{"order_id": "entry", "pnl": 50}])
    assert report["counts"]["order_id_shared_by_decisions"] == 2
    assert report["counts"]["unresolved"] == 2
