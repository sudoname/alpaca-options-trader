"""
Tests for the research/validation layer (decision ledger, capture hooks,
reconciliation, replay, shadow harness, chronological validation).

Spec coverage (section 9 of the research-layer requirements):
  * disabled-mode equivalence  - flag off means no files, no DB, no behavior
  * idempotency                - duplicate events / reconciler re-runs are no-ops
  * restart recovery           - on-disk ledger survives close/reopen without
                                 double counting
  * partial fills              - broker partial fills and depth-capped sim fills
  * missing/stale data         - never fabricate quotes; stale quotes don't fill
  * cost accounting            - sim fills reconcile exactly with cost_model;
                                 spread paid once per side; real multiplier
  * leakage prevention         - purge/embargo, chronological ordering, the
                                 final test slice is never evaluated
  * shadow isolation           - replay broker cannot place orders; shadow
                                 harness never blocks/raises/mutates and never
                                 leaks into baseline analytics

Offline by design: in-memory or tmp_path SQLite, injected fakes, no creds,
no network, no broker.
"""

import copy
from datetime import datetime, timedelta, timezone

import pytest

from cost_model import CostModel
from research_capture import ResearchCapture
from research_experiments import ShadowComparator
from research_ledger import ResearchLedger
from research_outcomes import OutcomeReconciler
from research_replay import (
    ExecutionModel,
    ReplayBroker,
    ReplayIsolationError,
    replay_cohort,
    simulate_fill,
)
from research_validation import (
    chronological_split,
    observations_from_ledger,
    purge_overlapping,
    validation_report,
    walk_forward,
)

# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
QUOTE = {"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 50, "age_s": 1.0}


def _ts(day: int, hour: int = 14, minute: int = 30) -> str:
    return datetime(2026, 10, 1 + day, hour, minute,
                    tzinfo=timezone.utc).isoformat()


def seed_lifecycle(led: ResearchLedger, i: int, day: int = 0) -> str:
    """One complete decision: snapshot -> submitted -> fill -> exit -> outcome."""
    did, oid = f"d-{i}", f"o-{i}"
    t0, t1 = _ts(day), _ts(day, 18)
    led.record_snapshot(
        decision_id=did, accepted=True, symbol="SPY261218C00500000",
        underlying="SPY", quote=dict(QUOTE, ts=t0), underlying_price=498.0,
        direction="CALL", score=7.0, p_win=0.6, ev=5.0, regime="trend",
        sizing={"qty": 1}, event_at=t0)
    led.record_event(decision_id=did, event_type="submitted",
                     broker_order_id=oid, qty=1, price=1.10, event_at=t0)
    led.record_event(decision_id=did, event_type="fill", broker_order_id=oid,
                     qty=1, price=1.10, status="filled", event_at=t0)
    led.record_event(decision_id=did, event_type="exit",
                     broker_order_id=f"x-{i}", qty=1, price=1.21, event_at=t1)
    led.record_event(decision_id=did, event_type="outcome",
                     broker_order_id=f"x-{i}", qty=1, price=1.21, event_at=t1,
                     payload={"class": "realized", "net_pnl_pct": 10.0,
                              "net_pnl_dollars": 11.0})
    return did


def seed_open_submission(led: ResearchLedger, i: int, day: int = 0) -> str:
    """A decision stuck at 'submitted' (what the reconciler works on)."""
    did, oid = f"d-{i}", f"o-{i}"
    t0 = _ts(day)
    led.record_snapshot(
        decision_id=did, accepted=True, symbol="SPY261218C00500000",
        underlying="SPY", quote=dict(QUOTE, ts=t0), direction="CALL",
        sizing={"qty": 2}, event_at=t0)
    led.record_event(decision_id=did, event_type="submitted",
                     broker_order_id=oid, qty=2, price=1.10, event_at=t0)
    return did


def _obs(i: int, day: int, net=1.0, hold_minutes: float = 240.0,
         t_outcome="auto", p=None):
    """A validation observation shaped like observations_from_ledger output."""
    t0 = datetime(2026, 10, 1, 14, 30, tzinfo=timezone.utc) + timedelta(days=day)
    if t_outcome == "auto":
        t_out = (t0 + timedelta(minutes=hold_minutes)).isoformat()
    else:
        t_out = t_outcome
    return {"decision_id": f"obs-{i}", "t_decision": t0.isoformat(),
            "t_outcome": t_out, "net_pnl_pct": net, "net_pnl_dollars": None,
            "gross_pnl_pct": None, "p_win": p, "ev": None, "regime": "trend",
            "direction": "CALL", "filled": True, "outlay": 100.0,
            "shortfall": None, "simulated": False}


class BoomLedger:
    """A ledger whose writes always fail (disk full, schema drift, ...)."""

    def record_snapshot(self, **kw):
        raise RuntimeError("boom")

    def record_event(self, **kw):
        raise RuntimeError("boom")


# --------------------------------------------------------------------------- #
# 1) Disabled-mode equivalence
# --------------------------------------------------------------------------- #
class TestDisabledMode:
    def test_disabled_hooks_are_noops(self):
        cap = ResearchCapture(enabled=False)
        assert cap.enabled is False
        assert cap._ledger is None
        assert cap.on_candidate(decision_id="x", accepted=True) is None
        assert cap.on_order_submitted("x", "o-1") is None
        assert cap.on_fill("x", "o-1", qty=1, price=1.0) is None
        assert cap.on_broker_reject("x", "o-1", reason="nope") is None
        assert cap.on_exit("x", exit_price=1.2) is None
        assert cap.dropped == {}

    def test_env_flag_off_creates_no_file(self, monkeypatch, tmp_path):
        db = tmp_path / "ledger.db"
        # Shell env wins over .env in ConfigLoader, so this is deterministic
        # even if the repo .env sets the flag.
        monkeypatch.setenv("ENABLE_DECISION_LEDGER", "0")
        monkeypatch.setenv("DECISION_LEDGER_PATH", str(db))
        cap = ResearchCapture()
        assert cap.enabled is False
        cap.on_candidate(decision_id="x", accepted=True)
        assert not db.exists()

    def test_env_flag_on_enables_capture(self, monkeypatch, tmp_path):
        db = tmp_path / "ledger.db"
        monkeypatch.setenv("ENABLE_DECISION_LEDGER", "1")
        monkeypatch.setenv("DECISION_LEDGER_PATH", str(db))
        cap = ResearchCapture()
        try:
            assert cap.enabled is True
            assert cap.on_candidate(decision_id="e-1", accepted=True) == "e-1"
            assert db.exists()
        finally:
            if cap._ledger is not None:
                cap._ledger.close()

    def test_broken_ledger_counts_drops_never_raises(self):
        cap = ResearchCapture(ledger=BoomLedger(), enabled=True)
        assert cap.on_candidate(decision_id="z", accepted=True) is None
        cap.on_order_submitted("z", "o")
        cap.on_fill("z", "o", qty=1, price=1.0)
        cap.on_exit("z", exit_price=1.1)
        assert cap.dropped.get("on_candidate") == 1
        assert cap.dropped.get("on_order_submitted") == 1
        assert cap.dropped.get("on_fill") == 1
        assert cap.dropped.get("on_exit") == 1


# --------------------------------------------------------------------------- #
# 2) Idempotency
# --------------------------------------------------------------------------- #
class TestIdempotency:
    def test_duplicate_event_is_ignored(self):
        led = ResearchLedger(":memory:")
        seed_open_submission(led, 1)
        kw = dict(decision_id="d-1", event_type="fill", broker_order_id="o-1",
                  qty=2, price=1.08, event_at=_ts(0, 15))
        _, first = led.record_event(**kw)
        before = led.stats()["events"]
        _, second = led.record_event(**kw)
        assert first is True
        assert second is False
        assert led.stats()["events"] == before
        led.close()

    def test_reconciler_rerun_writes_nothing_new(self):
        led = ResearchLedger(":memory:")
        seed_open_submission(led, 1)
        fetch = lambda oid: {"status": "filled", "filled_qty": 2,
                             "filled_avg_price": 1.08,
                             "filled_at": _ts(0, 15)}
        s1 = OutcomeReconciler(led, fetch_order=fetch,
                               apply=True).reconcile_orders()
        assert s1["fills"] == 1
        n = led.stats()["events"]
        # Fresh reconciler, same broker truth: deterministic event_key makes
        # the re-run a no-op.
        s2 = OutcomeReconciler(led, fetch_order=fetch,
                               apply=True).reconcile_orders()
        assert s2["fills"] == 0
        assert led.stats()["events"] == n
        led.close()

    def test_dry_run_plans_but_writes_nothing(self):
        led = ResearchLedger(":memory:")
        seed_open_submission(led, 1)
        before = led.stats()["events"]
        rec = OutcomeReconciler(
            led, apply=False,
            fetch_order=lambda oid: {"status": "filled", "filled_qty": 2,
                                     "filled_avg_price": 1.08,
                                     "filled_at": _ts(0, 15)})
        summary = rec.reconcile_orders()
        assert summary["fills"] == 0            # nothing WRITTEN
        assert len(rec.planned) == 1            # but the action was planned
        assert led.stats()["events"] == before
        led.close()

    def test_activity_dedupe_by_activity_id(self):
        led = ResearchLedger(":memory:")
        seed_open_submission(led, 1)
        # Two partial fills with IDENTICAL qty/price but distinct activity ids
        # must record as two events; re-running records nothing.
        acts = [
            {"id": "a1", "order_id": "o-1", "qty": 1, "price": 1.08,
             "side": "buy", "type": "partial_fill",
             "transaction_time": _ts(0, 15)},
            {"id": "a2", "order_id": "o-1", "qty": 1, "price": 1.08,
             "side": "buy", "type": "partial_fill",
             "transaction_time": _ts(0, 15)},
        ]
        rec = OutcomeReconciler(led, fetch_activities=lambda after: acts,
                                apply=True)
        s1 = rec.reconcile_activities("2026-10-01T00:00:00+00:00")
        assert s1["recorded"] == 2
        s2 = OutcomeReconciler(led, fetch_activities=lambda after: acts,
                               apply=True).reconcile_activities(
                                   "2026-10-01T00:00:00+00:00")
        assert s2["recorded"] == 0
        led.close()


# --------------------------------------------------------------------------- #
# 3) Restart recovery (on-disk DB survives close/reopen, no double counting)
# --------------------------------------------------------------------------- #
class TestRestartRecovery:
    def test_reconcile_survives_restart(self, tmp_path):
        path = str(tmp_path / "ledger.db")
        fetch = lambda oid: {"status": "filled", "filled_qty": 2,
                             "filled_avg_price": 1.08,
                             "filled_at": _ts(0, 15)}

        led = ResearchLedger(path)
        seed_open_submission(led, 1)
        led.close()

        # "Process restart" #1: reconcile against broker truth.
        led = ResearchLedger(path)
        s1 = OutcomeReconciler(led, fetch_order=fetch,
                               apply=True).reconcile_orders()
        assert s1["fills"] == 1
        n = led.stats()["events"]
        led.close()

        # "Process restart" #2: a retry after a crash must not double count.
        led = ResearchLedger(path)
        s2 = OutcomeReconciler(led, fetch_order=fetch,
                               apply=True).reconcile_orders()
        assert s2["fills"] == 0
        assert led.stats()["events"] == n
        types = [e["event_type"] for e in led.get_decision("d-1")["events"]]
        assert types.count("fill") == 1
        led.close()


# --------------------------------------------------------------------------- #
# 4) Partial fills
# --------------------------------------------------------------------------- #
class TestPartialFills:
    def test_broker_partial_fill_records_partial_event(self):
        led = ResearchLedger(":memory:")
        seed_open_submission(led, 1)
        fetch = lambda oid: {"status": "partially_filled", "filled_qty": 1,
                             "filled_avg_price": 1.09,
                             "updated_at": _ts(0, 15)}
        OutcomeReconciler(led, fetch_order=fetch, apply=True).reconcile_orders()
        evs = led.get_decision("d-1")["events"]
        partials = [e for e in evs if e["event_type"] == "partial_fill"]
        assert len(partials) == 1
        assert float(partials[0]["qty"]) == 1.0
        assert float(partials[0]["price"]) == pytest.approx(1.09)
        led.close()

    def test_depth_cap_produces_partial(self):
        q = {"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 2,
             "quote_age_s": 1.0}
        m = ExecutionModel(allow_partial=True, max_participation=1.0)
        r = simulate_fill("buy", q, 5, m)
        assert r["status"] == "partial"
        assert r["filled_qty"] == 2
        assert r["liquidity"] == "observed"

    def test_all_or_none_goes_unfilled(self):
        q = {"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 2,
             "quote_age_s": 1.0}
        m = ExecutionModel(allow_partial=False)
        r = simulate_fill("buy", q, 5, m)
        assert r["status"] == "unfilled"
        assert r["filled_qty"] == 0
        assert "depth_cap" in (r["reason"] or "")

    def test_unobserved_depth_is_flagged_not_capped(self):
        q = {"bid": 1.00, "ask": 1.10, "quote_age_s": 1.0}  # no sizes recorded
        r = simulate_fill("buy", q, 5, ExecutionModel())
        assert r["status"] == "filled"
        assert r["liquidity"] == "unobserved"


# --------------------------------------------------------------------------- #
# 5) Missing / stale data: never fabricate
# --------------------------------------------------------------------------- #
class TestMissingStaleData:
    def test_missing_quote_is_unreplayable(self):
        m = ExecutionModel()
        for q in (None, {}, {"bid": None, "ask": 1.10}, {"bid": 1.0}):
            r = simulate_fill("buy", q, 1, m)
            assert r["status"] == "unreplayable"
            assert r["price"] is None
            assert r["liquidity"] == "missing"

    def test_stale_quote_does_not_fill(self):
        q = dict(QUOTE, quote_age_s=120.0)
        r = simulate_fill("buy", q, 1, ExecutionModel(max_quote_age_s=60.0))
        assert r["status"] == "unfilled"
        assert r["liquidity"] == "stale"

    def test_unknown_age_is_not_treated_as_stale(self):
        q = {"bid": 1.00, "ask": 1.10}  # age unrecorded -> can't prove stale
        r = simulate_fill("buy", q, 1, ExecutionModel(max_quote_age_s=60.0))
        assert r["status"] == "filled"

    def test_replay_never_invents_a_quote(self):
        led = ResearchLedger(":memory:")
        led.record_snapshot(decision_id="nq-1", accepted=True,
                            symbol="SPY261218C00500000", quote=None,
                            sizing={"qty": 1}, event_at=_ts(0))
        out = replay_cohort(led, ExecutionModel())
        assert out["summary"]["decisions"] == 1
        assert out["summary"]["by_status"].get("unreplayable") == 1
        assert out["summary"]["with_pnl"] == 0
        led.close()


# --------------------------------------------------------------------------- #
# 6) Cost accounting
# --------------------------------------------------------------------------- #
class TestCostAccounting:
    def test_sim_round_trip_equals_cost_model(self):
        """Buy + sell at the same quote must cost exactly what cost_model says
        a round trip costs: spread once per side, slippage and fees twice, no
        double counting anywhere."""
        m = ExecutionModel()
        q = {"bid": 1.00, "ask": 1.10, "bid_size": 500, "ask_size": 500,
             "quote_age_s": 1.0}
        qty = 2
        buy = simulate_fill("buy", q, qty, m)
        sell = simulate_fill("sell", q, qty, m)
        assert buy["status"] == sell["status"] == "filled"
        total_cost = (-buy["cash"]) - sell["cash"]  # cash out minus cash in
        rt = CostModel(m.effective_cost()).round_trip_cost(1.00, 1.10, qty=qty)
        assert total_cost == pytest.approx(rt["cost_dollars"], abs=1e-9)
        assert buy["price"] == pytest.approx(rt["entry_price"])
        assert sell["price"] == pytest.approx(rt["exit_price"])

    def test_actual_contract_multiplier_used(self):
        q = {"bid": 1.00, "ask": 1.10, "quote_age_s": 1.0}
        r100 = simulate_fill("buy", q, 1, ExecutionModel(), multiplier=100.0)
        r10 = simulate_fill("buy", q, 1, ExecutionModel(), multiplier=10.0)
        assert r100["notional"] == pytest.approx(r100["price"] * 100.0)
        assert r10["notional"] == pytest.approx(r10["price"] * 10.0)

    def test_adverse_scenarios_cost_more(self):
        q = {"bid": 1.00, "ask": 1.10, "quote_age_s": 1.0}
        base = simulate_fill("buy", q, 1, ExecutionModel())
        wide = simulate_fill("buy", q, 1,
                             ExecutionModel(name="wide", spread_widen_frac=0.5))
        slip = simulate_fill(
            "buy", q, 1,
            ExecutionModel(name="slip", extra_slippage_per_contract=0.05))
        assert wide["price"] > base["price"]
        assert slip["price"] == pytest.approx(base["price"] + 0.05)


# --------------------------------------------------------------------------- #
# 7) Leakage prevention
# --------------------------------------------------------------------------- #
class TestLeakagePrevention:
    def test_purge_drops_overlapping_and_open_ended(self):
        boundary = _obs(99, day=5)["t_decision"]
        train = [
            _obs(1, day=1),                       # closed well before boundary
            _obs(2, day=4, hold_minutes=2880.0),  # outcome lands AFTER boundary
            _obs(3, day=2, t_outcome=None),       # open-ended: can't prove closed
        ]
        kept, purged = purge_overlapping(train, boundary)
        assert [o["decision_id"] for o in kept] == ["obs-1"]
        assert purged == 2

    def test_embargo_widens_the_purge(self):
        boundary = _obs(99, day=1)["t_decision"]        # day1 14:30
        row = _obs(1, day=0, hold_minutes=23.5 * 60)    # closes day1 14:00
        kept, purged = purge_overlapping([row], boundary, embargo_minutes=0.0)
        assert purged == 0 and len(kept) == 1
        kept, purged = purge_overlapping([row], boundary, embargo_minutes=60.0)
        assert purged == 1 and kept == []

    def test_split_is_chronological_and_test_untouched(self):
        obs = [_obs(i, day=i) for i in range(10)]
        split = chronological_split(obs, train_frac=0.6, val_frac=0.2)
        assert split["test_untouched"] is True
        assert split["counts"] == {"train": 6, "val": 2, "test": 2}
        assert max(o["t_decision"] for o in split["train"]) \
            < min(o["t_decision"] for o in split["val"])
        assert max(o["t_decision"] for o in split["val"]) \
            < min(o["t_decision"] for o in split["test"])
        # The final slice is the chronologically last rows, verbatim.
        assert split["test"] == sorted(obs, key=lambda o: o["t_decision"])[-2:]

    def test_split_purges_train_rows_that_bleed_into_val(self):
        obs = [_obs(i, day=i) for i in range(10)]
        obs[5] = _obs(5, day=5, hold_minutes=3 * 24 * 60)  # closes inside val
        split = chronological_split(obs, train_frac=0.6, val_frac=0.2)
        assert split["purged"]["train"] >= 1
        assert all(o["decision_id"] != "obs-5" for o in split["train"])

    def test_walk_forward_trains_strictly_before_each_fold(self):
        obs = [_obs(i, day=i) for i in range(12)]
        wf = walk_forward(obs, n_folds=3)
        assert len(wf["folds"]) == 3
        fold_size = 12 // 4
        prev_start = ""
        for f in wf["folds"]:
            start_idx = f["fold"] * fold_size
            # Everything before the fold is either trained on or purged;
            # nothing from the fold or later ever reaches training.
            assert f["train_n"] + f["purged"] == start_idx
            assert f["test_start"] > prev_start
            prev_start = f["test_start"]
            assert f["test_n"] == fold_size
            assert f["coverage_days"] == fold_size

    def test_walk_forward_purges_overlapping_outcomes(self):
        # Multi-day holds: outcomes straddle fold boundaries, so the rows
        # nearest each boundary must be purged from training.
        obs = [_obs(i, day=i, hold_minutes=2 * 24 * 60) for i in range(12)]
        wf = walk_forward(obs, n_folds=3)
        fold_size = 12 // 4
        assert all(f["train_n"] + f["purged"] == f["fold"] * fold_size
                   for f in wf["folds"])
        assert any(f["purged"] > 0 for f in wf["folds"])

    def test_validation_report_never_evaluates_final_test(self):
        obs = [_obs(i, day=i, net=(1.0 if i % 2 else -0.5), p=0.6)
               for i in range(12)]
        report = validation_report(obs, embargo_minutes=0.0, n_folds=3)
        assert "test_metrics" not in report
        assert report["test_untouched"] is True
        assert "train_metrics" in report and "val_metrics" in report
        assert "not evaluated" in report["note"]


# --------------------------------------------------------------------------- #
# 8) Shadow isolation + replay isolation
# --------------------------------------------------------------------------- #
class TestIsolation:
    def test_replay_broker_cannot_place_orders(self):
        led = ResearchLedger(":memory:")
        seed_lifecycle(led, 1)
        rb = ReplayBroker(led)
        # Read access works...
        assert rb.stats()["events"] >= 4
        assert rb.get_decision("d-1")["snapshots"]
        assert rb.events_for_order("o-1")
        # ...every order-like or unknown access is structurally impossible.
        for name in ("submit_order", "place_order", "cancel_order",
                     "record_event", "record_snapshot", "conn"):
            with pytest.raises(ReplayIsolationError):
                getattr(rb, name)
        with pytest.raises(ReplayIsolationError):
            rb.anything = 1
        led.close()

    def test_shadow_backpressure_is_counted_not_blocking(self):
        led = ResearchLedger(":memory:")
        sc = ShadowComparator(led, variant="v2", maxsize=1, start_worker=False)
        assert sc.submit({"decision_id": "s-1", "accepted": True}) is True
        assert sc.submit({"decision_id": "s-2", "accepted": True}) is False
        acc = sc.accounting()
        assert acc["dropped"] == 1 and acc["queued"] == 1
        assert sc.drain() == 1
        acc = sc.accounting()
        assert acc["written"] == 1 and acc["queued"] == 0
        snaps = led.snapshots(kind="shadow_candidate")
        assert len(snaps) == 1
        assert '"shadow_variant": "v2"' in (snaps[0].get("extra_json") or "")
        led.close()

    def test_shadow_write_failure_counted_never_raises(self):
        sc = ShadowComparator(BoomLedger(), variant="v2", maxsize=4,
                              start_worker=False)
        assert sc.submit({"decision_id": "s-1"}) is True
        assert sc.drain() == 1
        acc = sc.accounting()
        assert acc["write_errors"] == 1 and acc["written"] == 0

    def test_shadow_never_mutates_its_input(self):
        led = ResearchLedger(":memory:")
        sc = ShadowComparator(led, variant="v2", maxsize=4, start_worker=False)
        cand = {"decision_id": "s-1", "accepted": True, "extra": {"a": 1}}
        before = copy.deepcopy(cand)
        sc.submit(cand)
        sc.drain()
        assert cand == before   # shadow_variant was injected into a COPY
        led.close()

    def test_shadow_rows_never_leak_into_baseline_analytics(self):
        led = ResearchLedger(":memory:")
        seed_lifecycle(led, 1)
        sc = ShadowComparator(led, variant="v2", maxsize=4, start_worker=False)
        # Shadow variant evaluated the SAME decision differently.
        sc.submit({"decision_id": "d-1", "accepted": True, "score": 9.9})
        sc.drain()
        assert len(led.snapshots(kind="shadow_candidate")) == 1
        # Baseline consumers see exactly one decision, not two.
        obs = observations_from_ledger(led)
        assert [o["decision_id"] for o in obs] == ["d-1"]
        cohort = replay_cohort(led, ExecutionModel())
        assert cohort["summary"]["decisions"] == 1
        # The baseline snapshot row itself is untouched (immutable insert).
        kinds = sorted(s["kind"] for s in led.get_decision("d-1")["snapshots"])
        assert kinds == ["candidate", "shadow_candidate"]
        base = [s for s in led.get_decision("d-1")["snapshots"]
                if s["kind"] == "candidate"][0]
        assert base["score"] == pytest.approx(7.0)   # not overwritten by 9.9
        led.close()
