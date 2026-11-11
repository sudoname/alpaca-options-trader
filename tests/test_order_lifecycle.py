"""
Phase 2 — deterministic order lifecycle regression tests.

Master-prompt requirements covered (oracle/execution/lifecycle.py):
  * acknowledgments, rejects, cancellations, expiry, partial fills and late
    fills are handled deterministically;
  * only actual broker-confirmed filled quantity becomes an open position —
    accepted-but-unfilled, pending-cancel and expired orders contribute zero;
  * broker state is reconciled before any retry, and a retry is sized to the
    REMAINING quantity of the chain so duplicate/excess orders are impossible;
  * ambiguous submits (transport error / reject without a broker order id)
    block retries until a client_order_id lookup resolves them;
  * cancel/fill races: a fill always beats our intent to cancel, and is
    recorded as an explicit late fill;
  * replaying the same call sequence against the same scripted broker yields
    byte-identical order histories (idempotent replay).

The broker is a scripted ExecutionClient test double, so every race and
out-of-order read is reproduced exactly. No creds, no network, no file writes.
"""

import dataclasses
import os
import sys
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.execution.client import (
    STATUS_ACCEPTED,
    STATUS_CANCELED,
    STATUS_FILLED,
    STATUS_PARTIAL,
    STATUS_REJECTED,
    ExecutionClient,
    OrderRequest,
    OrderResult,
    Quote,
    SimExecutionClient,
)
from oracle.execution.lifecycle import (
    STATE_CANCELED,
    STATE_EXPIRED,
    STATE_FILLED,
    STATE_PARTIAL,
    STATE_REJECTED,
    STATE_UNKNOWN,
    STATE_WORKING,
    OrderLifecycle,
)


def res(order_id, status, *, symbol="OPT", side="buy", filled=0.0,
        price=None, coid=None):
    return OrderResult(order_id=order_id, status=status, symbol=symbol,
                       side=side, filled_qty=filled, filled_avg_price=price,
                       client_order_id=coid)


class ScriptedClient(ExecutionClient):
    """Deterministic broker double. ``submit_q`` is a FIFO of OrderResult or
    Exception consumed per submit; ``polls[order_id]`` is a FIFO of snapshots
    returned by get_order (the last one repeats forever); ``cancels[order_id]``
    is the response to cancel_order (OrderResult or Exception)."""

    name = "scripted"

    def __init__(self):
        self.submit_q = []
        self.polls = {}
        self.cancels = {}
        self.submits = []
        self.cancel_calls = []

    # -- reads -------------------------------------------------------------
    def get_quote(self, symbol):
        return None

    def get_positions(self):
        return []

    def get_buying_power(self):
        return None

    def get_order(self, order_id):
        script = self.polls.get(order_id)
        if not script:
            return None
        return script.pop(0) if len(script) > 1 else script[0]

    def get_fills(self, order_id=None):
        return []

    # -- writes ------------------------------------------------------------
    def submit_order(self, order):
        self.submits.append(order)
        out = self.submit_q.pop(0)
        if isinstance(out, Exception):
            raise out
        if out.client_order_id is None and order.client_order_id:
            out = dataclasses.replace(out,
                                      client_order_id=order.client_order_id)
        return out

    def cancel_order(self, order_id):
        self.cancel_calls.append(order_id)
        out = self.cancels.get(order_id)
        if isinstance(out, Exception):
            raise out
        if out is None:
            return res(order_id, STATUS_CANCELED)
        return out

    def replace_order(self, order_id, **changes):
        raise NotImplementedError


def req(coid, qty=1, side="buy", order_type="limit", limit_price=1.00):
    return OrderRequest("OPT", side, qty, order_type=order_type,
                        limit_price=limit_price, client_order_id=coid)


class TestAckRejectFill(unittest.TestCase):
    def test_ack_then_fill_via_poll(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.polls["o1"] = [res("o1", STATUS_ACCEPTED),
                              res("o1", STATUS_FILLED, filled=2, price=1.01)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1", qty=2), now=0.0)
        self.assertEqual(rec.state, STATE_WORKING)
        self.assertEqual(life.open_positions(), {})   # ack is not a position
        life.poll(now=1.0)
        self.assertEqual(rec.state, STATE_WORKING)
        changed = life.poll(now=2.0)
        self.assertEqual([r.client_order_id for r in changed], ["c1"])
        self.assertEqual(rec.state, STATE_FILLED)
        self.assertEqual(rec.filled_qty, 2.0)
        self.assertEqual(rec.avg_fill_price, 1.01)
        self.assertEqual(life.open_positions(), {"OPT": 2.0})

    def test_reject_with_order_id_is_terminal_and_retryable(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_REJECTED),
                           res("o2", STATUS_FILLED, filled=1, price=1.02)]
        broker.polls["o1"] = [res("o1", STATUS_REJECTED)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        self.assertEqual(rec.state, STATE_REJECTED)
        self.assertFalse(rec.ambiguous)
        self.assertEqual(life.open_positions(), {})
        child = life.retry("c1", now=1.0)
        self.assertIsNotNone(child)
        self.assertEqual(child.client_order_id, "c1#r2")
        self.assertEqual(child.request.qty, 1)        # full qty: nothing filled
        self.assertEqual(child.state, STATE_FILLED)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})

    def test_sell_fill_signs_negative(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_FILLED, side="sell",
                               filled=3, price=0.98)]
        life = OrderLifecycle(broker)
        life.submit(req("c1", qty=3, side="sell"), now=0.0)
        self.assertEqual(life.open_positions(), {"OPT": -3.0})


class TestCancelFillRaces(unittest.TestCase):
    def test_fill_beats_cancel_seen_on_later_poll(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.cancels["o1"] = res("o1", STATUS_CANCELED)        # our view
        broker.polls["o1"] = [res("o1", STATUS_FILLED, filled=1,  # reality
                                  price=1.00)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        life.cancel("c1", now=1.0)
        self.assertEqual(rec.state, STATE_CANCELED)
        self.assertEqual(life.open_positions(), {})
        life.poll(now=2.0)                                        # truth lands
        self.assertEqual(rec.state, STATE_FILLED)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})
        self.assertIn("late_fill_after_cancel_request",
                      [e[1] for e in rec.events])

    def test_cancel_response_itself_reports_fill(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.cancels["o1"] = res("o1", STATUS_FILLED, filled=1, price=1.00)
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        life.cancel("c1", now=1.0)
        self.assertEqual(rec.state, STATE_FILLED)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})
        self.assertIn("late_fill_after_cancel_request",
                      [e[1] for e in rec.events])

    def test_rejected_cancel_leaves_order_working(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.cancels["o1"] = res("o1", STATUS_REJECTED)   # "too late"
        broker.polls["o1"] = [res("o1", STATUS_FILLED, filled=1, price=1.01)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        life.cancel("c1", now=1.0)
        self.assertEqual(rec.state, STATE_WORKING)  # cancel reject != order reject
        life.poll(now=2.0)
        self.assertEqual(rec.state, STATE_FILLED)

    def test_stale_accepted_read_never_resurrects_canceled(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.cancels["o1"] = res("o1", STATUS_CANCELED)
        broker.polls["o1"] = [res("o1", STATUS_ACCEPTED)]   # stale snapshot
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        life.cancel("c1", now=1.0)
        self.assertEqual(rec.state, STATE_CANCELED)
        life.poll(now=2.0)
        self.assertEqual(rec.state, STATE_CANCELED)


class TestExpiry(unittest.TestCase):
    def test_expiry_cancels_resting_order(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.polls["o1"] = [res("o1", STATUS_ACCEPTED)]
        broker.cancels["o1"] = res("o1", STATUS_CANCELED)
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=100.0, expiry_s=30.0)
        life.poll(now=120.0)                       # age 20s < 30s
        self.assertEqual(rec.state, STATE_WORKING)
        self.assertEqual(broker.cancel_calls, [])
        life.poll(now=130.0)                       # age 30s -> expiry
        self.assertEqual(rec.state, STATE_EXPIRED)
        self.assertTrue(rec.expiry_cancel)
        self.assertEqual(broker.cancel_calls, ["o1"])
        self.assertEqual(life.open_positions(), {})

    def test_late_fill_after_expiry_is_adopted(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.polls["o1"] = [res("o1", STATUS_ACCEPTED),
                              res("o1", STATUS_FILLED, filled=1, price=1.00)]
        broker.cancels["o1"] = res("o1", STATUS_CANCELED)
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0, expiry_s=10.0)
        life.poll(now=10.0)
        self.assertEqual(rec.state, STATE_EXPIRED)
        life.poll(now=11.0)                        # broker filled it anyway
        self.assertEqual(rec.state, STATE_FILLED)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})
        self.assertIn("late_fill_after_cancel_request",
                      [e[1] for e in rec.events])


class TestPartialFillsAndRetry(unittest.TestCase):
    def _partial_then_cancel(self):
        """3-lot order fills 1, is cancelled -> STATE_PARTIAL, position 1."""
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.polls["o1"] = [
            res("o1", STATUS_PARTIAL, filled=1, price=1.00),
            res("o1", STATUS_CANCELED, filled=1, price=1.00),
        ]
        broker.cancels["o1"] = res("o1", STATUS_PARTIAL, filled=1, price=1.00)
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1", qty=3), now=0.0)
        life.poll(now=1.0)
        self.assertEqual(rec.state, STATE_WORKING)   # partial is still live
        self.assertEqual(rec.filled_qty, 1.0)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})
        life.cancel("c1", now=2.0)
        life.poll(now=3.0)                           # broker confirms cancel
        self.assertEqual(rec.state, STATE_PARTIAL)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})
        return broker, life, rec

    def test_partial_fill_counts_only_filled_quantity(self):
        self._partial_then_cancel()

    def test_retry_sized_to_remaining_never_excess(self):
        broker, life, _ = self._partial_then_cancel()
        broker.submit_q = [res("o2", STATUS_FILLED, filled=2, price=1.01)]
        child = life.retry("c1", now=4.0)
        self.assertIsNotNone(child)
        self.assertEqual(child.client_order_id, "c1#r2")
        self.assertEqual(child.request.qty, 2)        # 3 target - 1 filled
        self.assertEqual(life.chain_filled("c1"), 3.0)
        self.assertEqual(life.open_positions(), {"OPT": 3.0})
        # Chain complete -> any further retry is refused (no excess orders).
        self.assertIsNone(life.retry("c1", now=5.0))
        self.assertIsNone(life.retry("c1#r2", now=6.0))
        self.assertEqual(len(broker.submits), 2)      # exactly two broker writes

    def test_retry_blocked_while_working_and_reconciles_first(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        # Reconcile inside retry() discovers the fill -> retry refused.
        broker.polls["o1"] = [res("o1", STATUS_FILLED, filled=1, price=1.00)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        self.assertIsNone(life.retry("c1", now=1.0))
        self.assertEqual(rec.state, STATE_FILLED)     # reconciled before retry
        self.assertIn("retry_refused_complete", [e[1] for e in rec.events])
        self.assertEqual(len(broker.submits), 1)      # no duplicate order

    def test_monotone_fills_ignore_stale_out_of_order_reads(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        broker.polls["o1"] = [
            res("o1", STATUS_PARTIAL, filled=2, price=1.00),
            res("o1", STATUS_PARTIAL, filled=1, price=0.90),  # stale read
        ]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1", qty=3), now=0.0)
        life.poll(now=1.0)
        self.assertEqual(rec.filled_qty, 2.0)
        life.poll(now=2.0)
        self.assertEqual(rec.filled_qty, 2.0)         # never downgraded
        self.assertEqual(rec.avg_fill_price, 1.00)


class TestAmbiguousSubmit(unittest.TestCase):
    def test_transport_error_blocks_retry_without_lookup(self):
        broker = ScriptedClient()
        broker.submit_q = [ConnectionError("timeout")]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        self.assertEqual(rec.state, STATE_UNKNOWN)
        self.assertTrue(rec.ambiguous)
        self.assertIsNone(life.retry("c1", now=1.0))  # conservative block
        self.assertIn("retry_blocked_ambiguous", [e[1] for e in rec.events])
        self.assertEqual(len(broker.submits), 1)

    def test_lookup_adopts_order_that_did_reach_broker(self):
        broker = ScriptedClient()
        broker.submit_q = [ConnectionError("timeout")]
        broker.polls["o1"] = [res("o1", STATUS_FILLED, filled=1, price=1.00)]
        life = OrderLifecycle(
            broker,
            lookup_by_client_order_id=lambda coid:
                res("o1", STATUS_ACCEPTED, coid=coid))
        rec = life.submit(req("c1"), now=0.0)
        life.poll(now=1.0)                            # lookup resolves -> adopt
        self.assertFalse(rec.ambiguous)
        self.assertEqual(rec.order_id, "o1")
        life.poll(now=2.0)
        self.assertEqual(rec.state, STATE_FILLED)
        self.assertIsNone(life.retry("c1", now=3.0))  # filled -> no duplicate

    def test_lookup_not_found_unblocks_retry(self):
        broker = ScriptedClient()
        broker.submit_q = [ConnectionError("timeout"),
                           res("o2", STATUS_FILLED, filled=1, price=1.00)]
        life = OrderLifecycle(broker,
                              lookup_by_client_order_id=lambda coid: None)
        rec = life.submit(req("c1"), now=0.0)
        child = life.retry("c1", now=1.0)             # reconcile: never existed
        self.assertEqual(rec.state, STATE_REJECTED)
        self.assertIn("reconcile_not_found", [e[1] for e in rec.events])
        self.assertIsNotNone(child)
        self.assertEqual(child.request.qty, 1)
        self.assertEqual(life.open_positions(), {"OPT": 1.0})

    def test_reject_without_order_id_is_ambiguous(self):
        broker = ScriptedClient()
        broker.submit_q = [res(None, STATUS_REJECTED)]
        life = OrderLifecycle(broker)
        rec = life.submit(req("c1"), now=0.0)
        self.assertEqual(rec.state, STATE_REJECTED)
        self.assertTrue(rec.ambiguous)                # may still exist at broker
        self.assertIsNone(life.retry("c1", now=1.0))


class TestIdempotentReplay(unittest.TestCase):
    def test_duplicate_submit_returns_existing_without_broker_write(self):
        broker = ScriptedClient()
        broker.submit_q = [res("o1", STATUS_ACCEPTED)]
        life = OrderLifecycle(broker)
        a = life.submit(req("c1"), now=0.0)
        b = life.submit(req("c1"), now=1.0)
        self.assertIs(a, b)
        self.assertEqual(len(broker.submits), 1)
        self.assertIn("duplicate_submit_ignored", [e[1] for e in a.events])

    def test_replaying_same_event_sequence_is_byte_identical(self):
        def run():
            broker = ScriptedClient()
            broker.submit_q = [res("o1", STATUS_ACCEPTED),
                               res("o2", STATUS_FILLED, filled=2, price=1.01)]
            broker.polls["o1"] = [
                res("o1", STATUS_PARTIAL, filled=1, price=1.00),
                res("o1", STATUS_CANCELED, filled=1, price=1.00),
            ]
            broker.cancels["o1"] = res("o1", STATUS_PARTIAL, filled=1,
                                       price=1.00)
            life = OrderLifecycle(broker)
            life.submit(req("c1", qty=3), now=0.0, expiry_s=60.0)
            life.poll(now=1.0)
            life.cancel("c1", now=2.0)
            life.poll(now=3.0)
            life.retry("c1", now=4.0)
            life.submit(req("c1", qty=3), now=5.0)    # replayed duplicate
            life.poll(now=6.0)
            return ([r.to_dict() for r in life.records()],
                    life.open_positions(), len(broker.submits))
        first, second = run(), run()
        self.assertEqual(first, second)
        self.assertEqual(first[1], {"OPT": 3.0})
        self.assertEqual(first[2], 2)                 # exactly two broker writes


class TestSimIntegration(unittest.TestCase):
    """End-to-end over the real SimExecutionClient (no scripting)."""

    def test_rest_expire_retry_market(self):
        sim = SimExecutionClient(
            quotes={"OPT": Quote("OPT", bid=1.00, ask=1.05)})
        life = OrderLifecycle(sim)
        rec = life.submit(req("c1", qty=2, limit_price=0.90), now=0.0,
                          expiry_s=30.0)
        self.assertEqual(rec.state, STATE_WORKING)
        self.assertEqual(life.open_positions(), {})
        life.poll(now=30.0)
        self.assertEqual(rec.state, STATE_EXPIRED)
        child = life.retry(
            "c1", now=31.0,
            request=dataclasses.replace(rec.request, order_type="market",
                                        limit_price=None))
        self.assertEqual(child.state, STATE_FILLED)
        self.assertEqual(child.filled_qty, 2.0)
        self.assertEqual(child.avg_fill_price, 1.05)
        self.assertEqual(life.open_positions(), {"OPT": 2.0})
        self.assertIsNone(life.retry("c1", now=32.0))


if __name__ == "__main__":
    unittest.main(verbosity=2)
