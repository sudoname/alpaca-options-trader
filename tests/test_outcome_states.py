"""Hermetic tests for explicit outcome states + fills-based net dollars.

No credentials, no network, no files outside tmp: traders are built via
``__new__`` with only the attributes ``record_trade_outcome`` reads, the
research ledger is in-memory SQLite, and the realized-P&L kill-switch
tracker is replaced by a fake module so nothing writes
realized_pnl_log.json.

Covers:
  * classify_decision against a REAL ledger round-trip (payload_json path);
  * partial-fill accumulation and policy-skip semantics;
  * classify_legacy_record provenance branches;
  * record_trade_outcome threading — mark-derived legacy ``pnl`` stays
    distinct from fills-based ``net_pnl_dollars``; the kill-switch prefers
    net; the ledger outcome event carries gross/net separately;
  * skeptic_impact reader prefers the new net_pnl_dollars key and skips
    honest-None values.
"""
import json
import sys
import types

import pytest

from outcome_states import (
    FILLED_RESOLVED, CONFIRMED_NO_FILL, POLICY_SKIP, UNRESOLVED,
    ZERO_PNL_STATES,
    classify_decision, classify_legacy_record, net_dollars_from_fills,
)
from research_capture import ResearchCapture
from research_ledger import ResearchLedger


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _make_trader(ledger):
    """SmartOptionsTrader via __new__ with only the attrs the method reads."""
    from smart_trader import SmartOptionsTrader
    t = SmartOptionsTrader.__new__(SmartOptionsTrader)
    t.trading_history = {"trades": []}
    t.ml_model = {"success_patterns": [], "failure_patterns": []}
    t.update_model_weights = lambda *a, **k: None
    t.save_trading_history = lambda *a, **k: None
    t.save_ml_model = lambda *a, **k: None
    t.research_capture = ResearchCapture(ledger=ledger, enabled=True)
    return t


@pytest.fixture
def fake_tracker(monkeypatch):
    """Replace realized_pnl_tracker so the kill-switch writes nowhere."""
    calls = []
    mod = types.ModuleType("realized_pnl_tracker")

    class RealizedPnLTracker:
        def add_realized(self, dollars, symbol=None):
            calls.append((dollars, symbol))

    mod.RealizedPnLTracker = RealizedPnLTracker
    monkeypatch.setitem(sys.modules, "realized_pnl_tracker", mod)
    return calls


# --------------------------------------------------------------------------- #
# classify_decision — ledger round-trip (payload_json path)
# --------------------------------------------------------------------------- #

def test_classify_from_real_ledger_round_trip():
    ledger = ResearchLedger(":memory:")
    did = ledger.record_snapshot(accepted=True,
                                 symbol="KO260116C00060000",
                                 direction="CALL")
    ledger.record_event(decision_id=did, event_type="submitted",
                        broker_order_id="oid-1")
    ledger.record_event(decision_id=did, event_type="fill",
                        broker_order_id="oid-1", qty=2, price=6.75,
                        payload={"side": "buy"})
    ledger.record_event(decision_id=did, event_type="fill",
                        broker_order_id="oid-2", qty=2, price=6.05,
                        payload={"side": "sell_to_close"})
    d = ledger.get_decision(did)
    r = classify_decision(d["snapshots"], d["events"])
    assert r["state"] == FILLED_RESOLVED
    assert r["pnl_source"] == "fills"
    assert r["net_pnl_dollars"] == pytest.approx((6.05 - 6.75) * 2 * 100)
    assert r["matched_qty"] == pytest.approx(2.0)
    assert r["residual_qty"] == pytest.approx(0.0)


def test_classify_partial_fills_accumulate():
    acc = [{"accepted": 1}]
    events = [
        {"event_type": "partial_fill", "qty": 1, "price": 1.0,
         "payload": {"side": "buy"}},
        {"event_type": "partial_fill", "qty": 1, "price": 1.2,
         "payload": {"side": "buy"}},
        {"event_type": "fill", "qty": 2, "price": 1.5,
         "payload": {"side": "sell"}},
    ]
    r = classify_decision(acc, events)
    assert r["state"] == FILLED_RESOLVED
    # entry avg = 1.10, exit 1.50 -> (0.40) * 2 * 100 = 80
    assert r["net_pnl_dollars"] == pytest.approx(80.0)


def test_policy_skip_is_zero_pnl_and_keeps_reason():
    r = classify_decision([{"accepted": 0, "reject_code": "gate:ev_low"}], [])
    assert r["state"] == POLICY_SKIP
    assert r["state"] in ZERO_PNL_STATES
    assert r["reason"] == "gate:ev_low"
    assert r["net_pnl_dollars"] == 0.0


def test_broker_reject_zero_exposure_keeps_broker_reason():
    acc = [{"accepted": 1}]
    r = classify_decision(acc, [
        {"event_type": "submitted"},
        {"event_type": "reject",
         "payload_json": json.dumps({"reason": "broker_rejected_no_fill"})},
    ])
    assert r["state"] == CONFIRMED_NO_FILL
    assert r["state"] in ZERO_PNL_STATES
    assert r["reason"] == "broker_rejected_no_fill"
    assert r["net_pnl_dollars"] == 0.0


# --------------------------------------------------------------------------- #
# classify_legacy_record — provenance requirement
# --------------------------------------------------------------------------- #

def test_legacy_fills_provenance_preferred():
    r = classify_legacy_record({"net_pnl_dollars": -140.0,
                                "pnl_source": "fills",
                                "pnl": -135.0})
    assert r["state"] == FILLED_RESOLVED
    assert r["pnl_source"] == "fills"
    assert r["net_pnl_dollars"] == pytest.approx(-140.0)


def test_legacy_null_needs_provenance():
    assert classify_legacy_record({})["state"] == UNRESOLVED
    assert (classify_legacy_record({"pnl": None})["reason"]
            == "legacy_null_without_provenance")
    assert (classify_legacy_record({"outcome": "cancelled"})["state"]
            == CONFIRMED_NO_FILL)


# --------------------------------------------------------------------------- #
# record_trade_outcome — net-dollar threading
# --------------------------------------------------------------------------- #

def test_record_trade_outcome_threads_fills_net(fake_tracker):
    ledger = ResearchLedger(":memory:")
    trader = _make_trader(ledger)
    did = ledger.record_snapshot(accepted=True,
                                 symbol="KO260116C00060000",
                                 direction="CALL")
    trade = {"symbol": "KO260116C00060000", "entry_time": "t1",
             "entry_price": 6.75, "quantity": 2,
             "decision_id": did, "order_id": "oid-1", "exit_price": 6.05}

    trader.record_trade_outcome(trade, "reconciled_fill_close", -10.0,
                                net_pnl_dollars=-140.0, pnl_source="fills")

    rec = trader.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] == pytest.approx(-140.0)
    assert rec["pnl_source"] == "fills"
    assert rec["pnl_percent"] == pytest.approx(-10.0)  # gross % untouched

    # Kill-switch prefers NET dollars over the mark-derived -135.
    assert fake_tracker == [
        (pytest.approx(-140.0), "KO260116C00060000")]

    # Ledger outcome event keeps gross and net distinct.
    events = ledger.get_decision(did)["events"]
    outcomes = [e for e in events if e["event_type"] == "outcome"]
    assert len(outcomes) == 1
    payload = json.loads(outcomes[0]["payload_json"])
    assert payload["net_pnl_dollars"] == pytest.approx(-140.0)
    assert payload["gross_pnl_pct"] == pytest.approx(-10.0)
    # net % computed off entry notional 6.75*100*2 = 1350
    assert payload["net_pnl_pct"] == pytest.approx(-140.0 / 1350.0 * 100.0)

    # And the ledger decision now classifies as resolved (reported wins).
    d = ledger.get_decision(did)
    r = classify_decision(d["snapshots"], d["events"])
    # no fill events were written here, only exit+outcome — the exit event
    # supplies exit qty but there is no entry fill, so this specific path
    # stays unresolved until the broker fill events land. That is the
    # honest answer: record_trade_outcome alone is not fill evidence.
    assert r["state"] == UNRESOLVED


def test_record_trade_outcome_mark_fallback(fake_tracker):
    ledger = ResearchLedger(":memory:")
    trader = _make_trader(ledger)
    trade = {"symbol": "XYZ", "entry_time": "t1",
             "entry_price": 6.75, "quantity": 2}

    trader.record_trade_outcome(trade, "stop_loss", -10.0)

    rec = trader.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] is None
    assert rec["pnl_source"] == "mark"
    # Kill-switch falls back to mark notional: 6.75*100*2*(-10%) = -135.
    assert fake_tracker == [(pytest.approx(-135.0), "XYZ")]


def test_record_trade_outcome_bad_net_degrades_to_mark(fake_tracker):
    ledger = ResearchLedger(":memory:")
    trader = _make_trader(ledger)
    trade = {"symbol": "XYZ", "entry_time": "t1",
             "entry_price": 6.75, "quantity": 2}

    # Garbage net must never raise and must not poison the record.
    trader.record_trade_outcome(trade, "stop_loss", -10.0,
                                net_pnl_dollars="garbage",
                                pnl_source="fills")

    rec = trader.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] is None
    assert fake_tracker == [(pytest.approx(-135.0), "XYZ")]


# --------------------------------------------------------------------------- #
# reader audit — skeptic_impact prefers the new key, skips honest None
# --------------------------------------------------------------------------- #

def test_skeptic_impact_reads_net_pnl_dollars_first():
    import skeptic_impact as si
    assert si._OUTCOME_KEYS[0] == "net_pnl_dollars"
    assert si._extract_net_outcome(
        {"net_pnl_dollars": -140.0, "pnl": -135.0}, None, None) == \
        pytest.approx(-140.0)
    # Honest-None net must not shadow a usable legacy pnl.
    assert si._extract_net_outcome(
        {"net_pnl_dollars": None, "pnl": -135.0}, None, None) == \
        pytest.approx(-135.0)


# --------------------------------------------------------------------------- #
# net_dollars_from_fills — fee + reconcile-shape checks
# --------------------------------------------------------------------------- #

def test_net_dollars_reconcile_shape_matches_booking():
    # Exactly the shapes reconcile_closed_from_fills builds.
    r = net_dollars_from_fills(
        [{"qty": 2, "price": 6.75}],
        [{"qty": 2, "price": 6.05}])
    assert r is not None
    assert r["net_pnl_dollars"] == pytest.approx(-140.0)


def test_net_dollars_fees_charged_once():
    r = net_dollars_from_fills(
        [{"qty": 2, "price": 1.0}],
        [{"qty": 1, "price": 1.5}, {"qty": 1, "price": 1.0}],
        fees=1.30)
    assert r["net_pnl_dollars"] == pytest.approx(48.70)
    assert r["fees"] == pytest.approx(1.30)
