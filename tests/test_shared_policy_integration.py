"""
Phase 2 — shared decision/execution policy integration test.

The master-prompt acceptance criterion: feed the SAME captured session facts
through BOTH consumption paths and assert they produce the same decisions,
contract choice, quantity, veto reason, and intended order fields.

  Path A (scheduler shadow shape) — ``snapshot_from_candidate`` over the
      scheduler's candidate dict (``run_alpaca_intraday._evaluate`` shape),
      then ``decide`` and ``intended_order``. This is exactly what
      ``IntradayScheduler._kernel_shadow`` records to the research ledger.
  Path B (replay/engine shape)    — ``OracleStrategyAdapter(use_kernel=True)``
      consuming a BarEvent, with the series transported via the reserved
      ctx keys and the contract list via ``contracts_provider``. The emitted
      TRADE_INTENT is the replay engine's would-be order.

Assertions (per master prompt):
  * Snapshots are EQUAL across paths (same facts -> same frozen input).
  * Decisions are EQUAL by canonical fingerprint (action, direction,
    contract choice, quantity, vetoes all pinned by it) — and the individual
    fields are additionally compared explicitly for a readable failure.
  * The TRADE_INTENT's (contract, qty, side, order_type, limit_price) match
    ``intended_order(decision)`` field-for-field.
  * Veto case (EV floor): both paths SKIP with the identical veto string,
    no intent is emitted, and ``intended_order`` is None.
  * No-trade case (weak signal): both paths abstain identically.
  * Execution: a SimExecutionClient fills exactly the intended quantity; the
    recorded fill equals ``decision.size`` (broker never re-sizes policy).
  * Idempotent replay: a fresh adapter fed the same events reproduces
    byte-identical intent fields and decision fingerprints.

No creds, no network, no order placement, no file writes.
"""

import os
import sys
import unittest
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.decision.kernel import decide
from oracle.decision.schema import DecisionConfig
from oracle.decision.snapshot_builder import (
    CTX_MARKET_REGIME,
    CTX_MOMENTUM,
    CTX_PRICES,
    CTX_QUOTE,
    CTX_VOLATILITY,
    intended_order,
    snapshot_from_candidate,
    snapshot_from_event,
)
from oracle.engine.events import BarEvent
from oracle.engine.strategy import OracleStrategyAdapter, StrategyConfig
from oracle.execution.client import (
    STATUS_FILLED,
    OrderRequest,
    Quote,
    SimExecutionClient,
)

# --------------------------------------------------------------------------- #
# One set of session facts, two shapes.
# --------------------------------------------------------------------------- #
T0 = datetime(2024, 1, 2, 16, 0, 0)

CONTRACTS = (
    {"symbol": "AAA_C1", "type": "call", "iv": 0.30, "dte": 5,
     "score": 8.0, "confidence": 2, "expected_value": 15.0,
     "probability_of_profit": 0.55, "ev_per_dollar_risk": 0.12,
     "bid": 1.00, "ask": 1.05},
    {"symbol": "AAA_C2", "type": "call", "iv": 0.30, "dte": 5,
     "score": 6.0, "confidence": 2},
    {"symbol": "AAA_P1", "type": "put", "iv": 0.30, "dte": 5, "score": 9.0},
)
FACTS = {"expected_move_pct": 3.0, "underlying_price": 498.0}
PRICES = (100.0, 101.0, 102.0, 104.0, 106.0)
MOMENTUM, VOLATILITY, REGIME = 0.05, 0.2, "trending"

FLAT_PRICES = (100.0, 100.0, 100.0, 100.0, 100.0)


def _candidate() -> dict:
    """Scheduler candidate dict (run_alpaca_intraday._evaluate shape)."""
    return {"sym": "AAA", "direction": "call", "opt": dict(CONTRACTS[0]),
            "score": 8.0, "underlying_price": 498.0}


def _scheduler_snapshot(prices=PRICES, momentum=MOMENTUM,
                        volatility=VOLATILITY):
    """Path A: exactly what _kernel_shadow builds for a ranked candidate,
    but with the full contract list so both paths select from the same set."""
    return snapshot_from_candidate(
        _candidate(), timestamp=T0.isoformat(),
        ctx={"expected_move_pct": 3.0},
        prices=prices, momentum=momentum, volatility=volatility,
        market_regime=REGIME, contracts=CONTRACTS)


def _ctx_builder(prices=PRICES, momentum=MOMENTUM, volatility=VOLATILITY):
    """Path B: engine ctx_builder transporting the series via reserved keys."""
    def build(symbol, event, clock):
        return {**FACTS,
                CTX_PRICES: list(prices),
                CTX_MOMENTUM: momentum,
                CTX_VOLATILITY: volatility,
                CTX_MARKET_REGIME: REGIME,
                CTX_QUOTE: {"bid": 1.00, "ask": 1.05}}
    return build


def _contracts_provider(symbol, event, clock):
    return CONTRACTS


def _bar(symbol="AAA") -> BarEvent:
    return BarEvent(symbol=symbol, event_timestamp=T0, o=100.0, c=106.0,
                    strategy_mode="intraday")


def _adapter(emitted, decision_config=None):
    return OracleStrategyAdapter(
        emit=emitted.append, config=StrategyConfig(),
        use_kernel=True, decision_config=decision_config,
        ctx_builder=_ctx_builder(), contracts_provider=_contracts_provider)


def _intents(emitted):
    return [e for e in emitted if e.event_type == "TRADE_INTENT"]


# --------------------------------------------------------------------------- #
# Tests
# --------------------------------------------------------------------------- #
class TestSameDecisionAcrossPaths(unittest.TestCase):
    """Same facts -> same Snapshot -> same Decision on both paths."""

    def test_snapshots_equal(self):
        snap_a = _scheduler_snapshot()
        snap_b = snapshot_from_event(
            _bar(), ctx=_ctx_builder()("AAA", _bar(), None),
            contracts=CONTRACTS)
        self.assertEqual(snap_a, snap_b)

    def test_decisions_equal_by_fingerprint_and_fields(self):
        d_a = decide(_scheduler_snapshot())

        emitted = []
        adapter = _adapter(emitted)
        adapter.handle(_bar())
        self.assertEqual(len(adapter.decisions), 1)
        d_b = adapter.decisions[0]

        self.assertEqual(d_a.fingerprint(), d_b.fingerprint())
        self.assertEqual(d_a, d_b)
        # Explicit field comparison for readable failures (master prompt:
        # decisions, contract choice, quantity, veto reason).
        self.assertEqual(d_a.action, d_b.action)
        self.assertEqual(d_a.direction, d_b.direction)
        self.assertEqual((d_a.selected_contract or {}).get("symbol"),
                         (d_b.selected_contract or {}).get("symbol"))
        self.assertEqual(d_a.size, d_b.size)
        self.assertEqual(d_a.vetoes, d_b.vetoes)

        # And the fixture is genuinely actionable (not vacuous parity).
        self.assertEqual(d_a.action, "enter")
        self.assertEqual(d_a.direction, "call")
        self.assertEqual((d_a.selected_contract or {}).get("symbol"), "AAA_C1")
        self.assertGreaterEqual(d_a.size, 1)


class TestIntendedOrderParity(unittest.TestCase):
    """The replay TRADE_INTENT and the scheduler shadow's recorded order are
    the same projection, field-for-field."""

    def test_trade_intent_matches_intended_order(self):
        d_a = decide(_scheduler_snapshot())
        order = intended_order(d_a)
        self.assertIsNotNone(order)

        emitted = []
        adapter = _adapter(emitted)
        adapter.handle(_bar())
        intents = _intents(emitted)
        self.assertEqual(len(intents), 1)
        it = intents[0]

        self.assertEqual(it.contract, order["symbol"])
        self.assertEqual(it.qty, order["qty"])
        self.assertEqual(it.side, order["side"])
        self.assertEqual(it.order_type, order["order_type"])
        self.assertEqual(it.limit_price, order["limit_price"])
        self.assertEqual(it.direction, (order["direction"] or "").upper())
        self.assertEqual(it.symbol, order["underlying"])
        # Quantity is the kernel's size, never a config default.
        self.assertEqual(it.qty, d_a.size)
        self.assertEqual(order["symbol"], "AAA_C1")
        self.assertEqual(order["side"], "buy")
        self.assertEqual(order["order_type"], "market")
        # Quote plumbed from the selected contract.
        self.assertEqual(it.bid, 1.00)
        self.assertEqual(it.ask, 1.05)

    def test_one_intent_per_symbol_still_holds_in_kernel_mode(self):
        emitted = []
        adapter = _adapter(emitted)
        adapter.handle(_bar())
        adapter.handle(_bar())
        self.assertEqual(len(_intents(emitted)), 1)


class TestVetoParity(unittest.TestCase):
    """EV-floor veto: both paths SKIP with the identical veto string, no
    intent, no intended order."""

    CFG = {"USE_EV_ENTRY_GATE": True, "MIN_EV_PER_DOLLAR_RISK": 0.50}

    def test_veto_reason_identical_and_no_order(self):
        cfg = DecisionConfig.make(self.CFG)
        d_a = decide(_scheduler_snapshot(), config=cfg)
        self.assertEqual(d_a.action, "skip")
        self.assertTrue(any("ev_gate" in v for v in d_a.vetoes))
        self.assertIsNone(intended_order(d_a))

        emitted = []
        adapter = _adapter(emitted, decision_config=DecisionConfig.make(self.CFG))
        adapter.handle(_bar())
        d_b = adapter.decisions[0]

        self.assertEqual(d_a.fingerprint(), d_b.fingerprint())
        self.assertEqual(d_a.vetoes, d_b.vetoes)
        self.assertEqual(_intents(emitted), [])
        # Telemetry records the veto + a NO_TRADE signal direction.
        self.assertEqual(adapter.signals[0]["vetoes"], list(d_a.vetoes))
        self.assertEqual(adapter.signals[0]["direction"], "NO_TRADE")


class TestNoTradeParity(unittest.TestCase):
    """Weak-signal abstain: both paths produce the identical no-trade."""

    CFG = {"USE_SKIP_ON_WEAK_SIGNAL": True, "MIN_DIRECTION_SIGNALS": 2}

    def test_no_trade_identical_and_no_order(self):
        cfg = DecisionConfig.make(self.CFG)
        d_a = decide(_scheduler_snapshot(prices=FLAT_PRICES, momentum=0.0,
                                         volatility=0.1), config=cfg)
        self.assertEqual(d_a.action, "no_trade")
        self.assertIsNone(intended_order(d_a))

        emitted = []
        adapter = OracleStrategyAdapter(
            emit=emitted.append, config=StrategyConfig(),
            use_kernel=True, decision_config=DecisionConfig.make(self.CFG),
            ctx_builder=_ctx_builder(prices=FLAT_PRICES, momentum=0.0,
                                     volatility=0.1),
            contracts_provider=_contracts_provider)
        adapter.handle(_bar())
        d_b = adapter.decisions[0]

        self.assertEqual(d_a.fingerprint(), d_b.fingerprint())
        self.assertEqual(d_b.action, "no_trade")
        self.assertEqual(_intents(emitted), [])


class TestExecutionOfSharedIntent(unittest.TestCase):
    """The deterministic Sim broker fills exactly the shared intent — the
    broker executes policy, it never re-sizes it."""

    def test_sim_fills_intended_quantity(self):
        d = decide(_scheduler_snapshot())
        order = intended_order(d)
        sym = order["symbol"]
        sim = SimExecutionClient(
            quotes={sym: Quote(sym, bid=1.00, ask=1.05, ts="t")},
            buying_power=10000.0)
        res = sim.submit_order(OrderRequest(sym, order["side"], order["qty"],
                                            order_type=order["order_type"]))
        self.assertEqual(res.status, STATUS_FILLED)
        self.assertEqual(res.filled_qty, float(d.size))
        self.assertEqual(res.filled_avg_price, 1.05)   # conservative: at ask


class TestIdempotentReplay(unittest.TestCase):
    """Replaying the same events through a fresh adapter reproduces identical
    decisions and intent fields (no hidden state, no wall-clock)."""

    def _run(self):
        emitted = []
        adapter = _adapter(emitted)
        adapter.handle(_bar())
        it = _intents(emitted)[0]
        d = adapter.decisions[0]
        return (d.fingerprint(),
                (it.contract, it.qty, it.side, it.order_type, it.limit_price,
                 it.direction, it.symbol, it.correlation_id))

    def test_two_runs_identical(self):
        self.assertEqual(self._run(), self._run())


# --------------------------------------------------------------------------- #
# Standalone runner (so run_selftests.py can execute this without pytest).
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print("tests.test_shared_policy_integration self-test:",
          "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(_self_test())
