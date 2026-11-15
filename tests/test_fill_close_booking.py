"""Hermetic tests for fills-based close booking (record_close_with_fills).

No credentials, no network: traders are built via ``__new__`` with only the
attributes the methods under test read; _await_fill is stubbed; requests.post
is patched; the realized-P&L kill-switch tracker is replaced by a fake module
so nothing writes realized_pnl_log.json.

Covers:
  * record_close_with_fills upgrade path — filled order supplies the REAL
    exit_price and fills-based net dollars (pnl_source='fills');
  * fail-open degradations — no/non-str order id, unfilled within the wait,
    zero fill fields, _await_fill raising: all book legacy mark-derived and
    never touch trade['exit_price'];
  * close_position returns the sell order id from the POST response, None on
    unparseable/absent id (fail-open);
  * enforce_exit prefers record_close_with_fills when present and threads the
    order id; falls back to record_trade_outcome on traders without it.
"""
import sys
import types
from unittest.mock import MagicMock, patch

import pytest

import exit_manager
from research_capture import ResearchCapture
from research_ledger import ResearchLedger


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

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


def _make_trader():
    """SmartOptionsTrader via __new__ with only the attrs the methods read."""
    from smart_trader import SmartOptionsTrader
    t = SmartOptionsTrader.__new__(SmartOptionsTrader)
    t.trading_history = {"trades": []}
    t.ml_model = {"success_patterns": [], "failure_patterns": []}
    t.update_model_weights = lambda *a, **k: None
    t.save_trading_history = lambda *a, **k: None
    t.save_ml_model = lambda *a, **k: None
    t.research_capture = ResearchCapture(ledger=ResearchLedger(":memory:"),
                                         enabled=True)
    return t


def _trade():
    return {"symbol": "KO260116C00060000", "entry_time": "t1",
            "entry_price": 6.75, "quantity": 2}


# --------------------------------------------------------------------------- #
# record_close_with_fills — upgrade path
# --------------------------------------------------------------------------- #

def test_filled_order_books_fills_net_and_real_exit_price(fake_tracker):
    t = _make_trader()
    t._await_fill = lambda oid: {"status": "filled",
                                 "filled_avg_price": "6.05",
                                 "filled_qty": "2"}
    trade = _trade()

    t.record_close_with_fills(trade, "dynamic_exit", -10.0,
                              exit_order_id="oid-1")

    assert trade["exit_price"] == pytest.approx(6.05)
    rec = t.trading_history["trades"][-1]
    assert rec["pnl_source"] == "fills"
    # (6.05 - 6.75) * 2 * 100 = -140 — matches reconcile_closed_from_fills.
    assert rec["net_pnl_dollars"] == pytest.approx(-140.0)
    assert rec["pnl_percent"] == pytest.approx(-10.0)  # legacy mark % kept
    # Kill-switch received NET dollars, not mark notional.
    assert fake_tracker == [(pytest.approx(-140.0), "KO260116C00060000")]


def test_partial_qty_uses_min_matched(fake_tracker):
    t = _make_trader()
    t._await_fill = lambda oid: {"status": "filled",
                                 "filled_avg_price": "7.00",
                                 "filled_qty": "1"}
    trade = _trade()  # entry qty 2, exit fill qty 1

    t.record_close_with_fills(trade, "dynamic_exit", 3.7,
                              exit_order_id="oid-1")

    rec = t.trading_history["trades"][-1]
    # matched_qty = min(2, 1) = 1 -> (7.00 - 6.75) * 1 * 100 = +25
    assert rec["net_pnl_dollars"] == pytest.approx(25.0)
    assert rec["pnl_source"] == "fills"


# --------------------------------------------------------------------------- #
# record_close_with_fills — fail-open degradations
# --------------------------------------------------------------------------- #

@pytest.mark.parametrize("oid", [None, "", 42, MagicMock()])
def test_non_str_or_missing_id_books_mark(fake_tracker, oid):
    t = _make_trader()
    t._await_fill = lambda o: (_ for _ in ()).throw(
        AssertionError("must not poll without a real id"))
    trade = _trade()

    t.record_close_with_fills(trade, "stop_loss", -10.0, exit_order_id=oid)

    assert "exit_price" not in trade  # honest-None preserved
    rec = t.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] is None
    assert rec["pnl_source"] == "mark"
    # Kill-switch fell back to mark notional: 6.75*100*2*(-10%) = -135.
    assert fake_tracker == [(pytest.approx(-135.0), "KO260116C00060000")]


def test_unfilled_within_wait_books_mark(fake_tracker):
    t = _make_trader()
    t._await_fill = lambda oid: {"status": "accepted"}
    trade = _trade()

    t.record_close_with_fills(trade, "stop_loss", -10.0, exit_order_id="oid-1")

    assert "exit_price" not in trade
    rec = t.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] is None and rec["pnl_source"] == "mark"


def test_zero_fill_fields_books_mark(fake_tracker):
    t = _make_trader()
    t._await_fill = lambda oid: {"status": "filled",
                                 "filled_avg_price": None,
                                 "filled_qty": "0"}
    trade = _trade()

    t.record_close_with_fills(trade, "stop_loss", -10.0, exit_order_id="oid-1")

    assert "exit_price" not in trade
    assert t.trading_history["trades"][-1]["net_pnl_dollars"] is None


def test_await_fill_raising_books_mark(fake_tracker):
    t = _make_trader()

    def _boom(oid):
        raise RuntimeError("api down")

    t._await_fill = _boom
    trade = _trade()

    t.record_close_with_fills(trade, "stop_loss", -10.0, exit_order_id="oid-1")

    rec = t.trading_history["trades"][-1]
    assert rec["net_pnl_dollars"] is None and rec["pnl_source"] == "mark"
    assert fake_tracker == [(pytest.approx(-135.0), "KO260116C00060000")]


# --------------------------------------------------------------------------- #
# close_position — order-id extraction (fail-open)
# --------------------------------------------------------------------------- #

def _close_trader():
    from smart_trader import SmartOptionsTrader
    t = SmartOptionsTrader.__new__(SmartOptionsTrader)
    t.base_url = "https://paper.example"
    t.headers = {}
    t.use_limit_exits = False
    return t


def test_close_position_returns_order_id():
    t = _close_trader()
    resp = MagicMock()
    resp.json.return_value = {"id": "sell-oid-9"}
    with patch("smart_trader.requests.post", return_value=resp) as post:
        oid = t.close_position({"symbol": "KO260116C00060000"},
                               {"qty": 2}, "dynamic_exit")
    assert oid == "sell-oid-9"
    assert post.call_args.kwargs["json"]["side"] == "sell"
    assert post.call_args.kwargs["json"]["type"] == "market"


@pytest.mark.parametrize("payload", [{}, {"id": None}, {"id": 42}, None])
def test_close_position_bad_payload_returns_none(payload):
    t = _close_trader()
    resp = MagicMock()
    if payload is None:
        resp.json.side_effect = ValueError("not json")
    else:
        resp.json.return_value = payload
    with patch("smart_trader.requests.post", return_value=resp):
        oid = t.close_position({"symbol": "KO260116C00060000"},
                               {"qty": 2}, "dynamic_exit")
    assert oid is None


# --------------------------------------------------------------------------- #
# enforce_exit — wiring + legacy fallback
# --------------------------------------------------------------------------- #

class _FillsTrader:
    def __init__(self):
        self.calls = []

    def close_position(self, trade, position, reason):
        self.calls.append(("close", reason))
        return "oid-7"

    def record_close_with_fills(self, trade, reason, pnl_percent,
                                exit_order_id=None):
        self.calls.append(("fills", reason, pnl_percent, exit_order_id))

    def record_trade_outcome(self, trade, reason, pnl_percent):
        self.calls.append(("legacy", reason))


class _LegacyTrader:
    def __init__(self):
        self.calls = []

    def close_position(self, trade, position, reason):
        self.calls.append(("close", reason))
        return None

    def record_trade_outcome(self, trade, reason, pnl_percent):
        self.calls.append(("legacy", reason, pnl_percent))


def test_enforce_exit_prefers_fills_booking_and_threads_id():
    t = _FillsTrader()
    exit_manager.enforce_exit(t, {"symbol": "X"}, {"current_price": 1.0},
                              "dynamic_stop_loss", -30.0, "scheduler")
    assert t.calls == [("close", "dynamic_stop_loss"),
                       ("fills", "dynamic_stop_loss", -30.0, "oid-7")]


def test_enforce_exit_legacy_fallback_without_method():
    t = _LegacyTrader()
    exit_manager.enforce_exit(t, {"symbol": "X"}, {"current_price": 1.0},
                              "dynamic_take_profit", 25.0, "telegram")
    assert t.calls == [("close", "dynamic_take_profit"),
                       ("legacy", "dynamic_take_profit", 25.0)]
