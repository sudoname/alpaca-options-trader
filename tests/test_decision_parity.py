"""
Upgrade 1D — parity / regression tests: ``decision_live == decision_backtest``.

The whole point of the unified kernel (``oracle.decision.decide``) is that the
live scan, a paper scan and a historical replay run the SAME decision code and
differ ONLY by how the ``Snapshot`` is built and which ``ExecutionClient`` is
handed the result. These tests pin that invariant:

  1. Determinism         — decide(snap) == decide(snap), by canonical fingerprint.
  2. Path parity         — a Snapshot built from a "live-scan" source and one
                           built from a "historical-replay" row (string-typed,
                           reordered keys, extra junk fields) yield an EQUAL
                           Decision. Same facts -> same decision, regardless of
                           feed.
  3. Execution independence — routing an actionable Decision through the Sim
                           broker (fills) vs the Shadow broker (no-op) never
                           changes the Decision; the Sim fills deterministically
                           and the Shadow places nothing.
  4. Config parity       — equal config -> equal config_version -> equal
                           Decision; a single differing knob breaks equality.
  5. Sim determinism     — two independent SimExecutionClients with identical
                           quotes + orders produce identical fills / positions.

No creds, no network, no order placement.
"""

import os
import sys
import unittest

# Allow ``python tests/test_decision_parity.py`` (repo root not auto on path
# when a file inside tests/ is run directly). Harmless under -m / pytest.
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.decision import decide
from oracle.decision.schema import (
    ACTION_ENTER,
    Decision,
    DecisionConfig,
    Snapshot,
)
from oracle.execution.client import (
    STATUS_FILLED,
    STATUS_SHADOW,
    OrderRequest,
    Quote,
    ShadowExecutionClient,
    SimExecutionClient,
)


# --------------------------------------------------------------------------- #
# Two views of the SAME market fact.
#
# A live scan hands us already-typed floats + a tidy contract list; a historical
# replay row is reconstructed from storage and is messier (strings, reordered
# keys, extra columns). Both must fold to an identical Snapshot -> identical
# Decision. That equality IS decision_live == decision_backtest.
# --------------------------------------------------------------------------- #
def _live_snapshot() -> Snapshot:
    return Snapshot.make(
        "AAA", "2024-01-02T16:00:00", strategy_mode="intraday",
        prices=[100.0, 101.0, 102.0, 104.0, 106.0],
        momentum=0.05, volatility=0.2, market_regime="trending",
        ctx={"expected_move_pct": 3.0},
        contracts=[
            {"symbol": "AAA_C1", "type": "call", "iv": 0.30, "dte": 5,
             "score": 8.0, "confidence": 2, "expected_value": 15.0,
             "probability_of_profit": 0.55, "ev_per_dollar_risk": 0.12},
            {"symbol": "AAA_C2", "type": "call", "iv": 0.30, "dte": 5,
             "score": 6.0, "confidence": 2},
            {"symbol": "AAA_P1", "type": "put", "iv": 0.30, "dte": 5,
             "score": 9.0},
        ])


def _backtest_snapshot() -> Snapshot:
    # Same facts, reconstructed from a storage row: numbers as strings, contract
    # keys reordered, an extra ignored column, prices with a junk element that
    # coercion drops back to the same 5-close series.
    return Snapshot.make(
        "aaa", "2024-01-02T16:00:00", strategy_mode="intraday",
        prices=["100", "101", "102", "104", "106", None],
        momentum="0.05", volatility="0.2", market_regime="trending",
        ctx={"expected_move_pct": "3.0", "_ignored_col": "whatever"},
        contracts=[
            {"type": "call", "symbol": "AAA_C1", "dte": 5, "iv": 0.30,
             "confidence": 2, "score": 8.0, "ev_per_dollar_risk": 0.12,
             "probability_of_profit": 0.55, "expected_value": 15.0,
             "_row_id": 991},
            {"type": "call", "symbol": "AAA_C2", "dte": 5, "iv": 0.30,
             "score": 6.0, "confidence": 2},
            {"type": "put", "symbol": "AAA_P1", "dte": 5, "iv": 0.30,
             "score": 9.0},
        ])


class TestDeterminism(unittest.TestCase):
    def test_decide_is_deterministic(self):
        s = _live_snapshot()
        self.assertEqual(decide(s), decide(s))
        self.assertEqual(decide(s).fingerprint(), decide(s).fingerprint())


class TestPathParity(unittest.TestCase):
    def test_live_equals_backtest(self):
        d_live = decide(_live_snapshot())
        d_back = decide(_backtest_snapshot())
        # The acceptance invariant, asserted by canonical fingerprint.
        self.assertEqual(
            d_live, d_back,
            msg=f"\nlive={d_live.fingerprint()}\nback={d_back.fingerprint()}")
        self.assertEqual(d_live.fingerprint(), d_back.fingerprint())

    def test_parity_holds_actionable_with_contract(self):
        d = decide(_live_snapshot())
        self.assertTrue(d.is_actionable())
        self.assertEqual(d.direction, "call")
        # Direction filter wins over the higher-scored PUT.
        self.assertIsNotNone(d.selected_contract)
        self.assertEqual(d.selected_contract.get("symbol"), "AAA_C1")

    def test_parity_survives_extra_snapshot_fields(self):
        # Junk columns on the replay row must not perturb the decision.
        self.assertEqual(decide(_live_snapshot()), decide(_backtest_snapshot()))


class TestConfigParity(unittest.TestCase):
    def test_equal_config_equal_decision(self):
        c1 = DecisionConfig.make({"USE_EV_ENTRY_GATE": True,
                                  "MIN_EV_PER_DOLLAR_RISK": 0.05})
        c2 = DecisionConfig.make({"MIN_EV_PER_DOLLAR_RISK": 0.05,
                                  "USE_EV_ENTRY_GATE": True})
        self.assertEqual(c1.config_version, c2.config_version)
        self.assertEqual(decide(_live_snapshot(), config=c1),
                         decide(_backtest_snapshot(), config=c2))

    def test_differing_knob_breaks_equality(self):
        base = decide(_live_snapshot())
        # A gate that vetoes (floor above the contract's ev/$risk) must change
        # the Decision -> parity is sensitive, not vacuously true.
        veto_cfg = DecisionConfig.make({"USE_EV_ENTRY_GATE": True,
                                        "MIN_EV_PER_DOLLAR_RISK": 0.50})
        vetoed = decide(_live_snapshot(), config=veto_cfg)
        self.assertNotEqual(base, vetoed)
        self.assertNotEqual(base.action, vetoed.action)
        self.assertEqual(vetoed.action, "skip")


class TestExecutionIndependence(unittest.TestCase):
    """The Decision must not depend on which broker executes it. Execution is a
    separate, swappable concern (that is exactly what lets a backtest and live
    share one decision)."""

    def _actionable(self) -> Decision:
        d = decide(_live_snapshot())
        self.assertEqual(d.action, ACTION_ENTER)
        return d

    def test_decision_unchanged_by_broker_choice(self):
        # Decide once; the same Decision is what both brokers receive.
        d = self._actionable()
        d_again = decide(_live_snapshot())
        self.assertEqual(d, d_again)

    def test_sim_fills_shadow_does_not(self):
        d = self._actionable()
        sym = d.selected_contract["symbol"]
        quote = Quote(sym, bid=1.00, ask=1.10, ts="t")
        order = OrderRequest(sym, "buy", d.size, order_type="market")

        sim = SimExecutionClient(quotes={sym: quote}, buying_power=10000.0)
        sim_res = sim.submit_order(order)
        self.assertEqual(sim_res.status, STATUS_FILLED)
        self.assertEqual(sim_res.filled_avg_price, 1.10)   # conservative: at ask
        self.assertEqual(sim_res.filled_qty, float(d.size))

        shadow = ShadowExecutionClient(inner=sim)
        before = len(sim._orders)
        sh_res = shadow.submit_order(order)
        self.assertEqual(sh_res.status, STATUS_SHADOW)
        # Shadow read delegates but the write never touched the inner broker.
        self.assertEqual(len(sim._orders), before)
        self.assertEqual(len(shadow.submitted), 1)
        self.assertEqual(shadow.get_fills(), [])

    def test_sim_is_deterministic_across_instances(self):
        d = self._actionable()
        sym = d.selected_contract["symbol"]
        q = Quote(sym, bid=1.00, ask=1.10, ts="t")
        order = OrderRequest(sym, "buy", d.size, order_type="market")

        a = SimExecutionClient(quotes={sym: q}, buying_power=10000.0)
        b = SimExecutionClient(quotes={sym: q}, buying_power=10000.0)
        ra, rb = a.submit_order(order), b.submit_order(order)
        self.assertEqual(
            (ra.order_id, ra.filled_avg_price, ra.filled_qty),
            (rb.order_id, rb.filled_avg_price, rb.filled_qty))
        self.assertEqual(a.get_buying_power(), b.get_buying_power())


# --------------------------------------------------------------------------- #
# Standalone runner (so run_selftests.py can execute this without pytest).
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    suite = unittest.defaultTestLoader.loadTestsFromModule(
        __import__(__name__))
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    ok = result.wasSuccessful()
    print("tests.test_decision_parity self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
