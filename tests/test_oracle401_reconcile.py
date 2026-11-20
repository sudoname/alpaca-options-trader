"""Oracle 4.0.1 Stage D acceptance tests.

Hand fixtures for ledger reconciliation (fills-weighted prices, duplicate
guard, unresolved accounting, fees-once/never-inferred, partial fills,
proceeds-is-not-profit, marks-vs-fills, paper-vs-live stratification) and
for the EV/PoP calibration study (training-only fits, no future data,
same-unit enforcement, rank-power verdicts, shrinkage math).
"""
from __future__ import annotations

import json
import math
import random
import sqlite3
from datetime import date, datetime, timedelta, timezone

import pytest

from oracle.ml import pnl_calibration as pc
from oracle.ml import reconcile as rc
from oracle.ml.ingest import LedgerEvent

UTC = timezone.utc
T0 = datetime(2026, 11, 10, 15, 0, tzinfo=UTC)


def _ev(etype, decision="d1", key=None, minutes=0, qty=None, price=None,
        fees=None, order=None, source="paper", payload=None, status=None):
    return LedgerEvent(
        event_key=key or f"{decision}:{etype}:{minutes}",
        decision_id=decision,
        event_type=etype,
        event_at=T0 + timedelta(minutes=minutes),
        available_at=T0 + timedelta(minutes=minutes),
        broker_order_id=order,
        qty=qty, price=price, fees=fees,
        status=status, source=source, payload=payload or {},
    )


# =========================================================================== #
# reconcile_positions
# =========================================================================== #

class TestReconcilePositions:
    def test_fills_weighted_entry_hand_fixture(self):
        """2 @ 1.00 + 1 @ 1.30 -> 3 @ 1.10 exactly."""
        events = {"d1": [
            _ev("submitted", qty=3.0),
            _ev("fill", minutes=1, qty=2.0, price=1.00, fees=1.0, order="o1"),
            _ev("fill", minutes=2, qty=1.0, price=1.30, fees=0.5, order="o1"),
            _ev("exit", minutes=30, qty=3.0, price=1.50, fees=1.0),
            _ev("outcome", minutes=31,
                payload={"gross_pnl_pct": 36.4, "class": "realized"}),
        ]}
        positions, report = rc.reconcile_positions(events)
        p = positions["d1"]
        assert p["matched_qty"] == 3.0
        assert p["entry_price_weighted"] == pytest.approx(1.10)
        assert p["n_entry_fills"] == 2
        assert report["multi_fill_entries"] == 1
        # cash identity with multiplier 100 and fees exactly once:
        # 450 - 330 - 1.0 - 0.5 - 1.0 = 117.5
        assert p["cost_basis"] == pytest.approx(330.0)
        assert p["proceeds"] == pytest.approx(450.0)
        assert p["realized_cash_pnl"] == pytest.approx(117.5)
        assert p["cash_pnl_basis"] == "fills"
        assert p["fees_fully_known"] is True
        assert p["status"] == "resolved"

    def test_proceeds_are_never_profit(self):
        """Realized P&L must subtract cost basis and fees, not equal
        proceeds; without an entry fill no cash P&L is claimed."""
        resolved = {"d1": [
            _ev("fill", qty=1.0, price=2.0, fees=0.5, order="o1"),
            _ev("exit", minutes=5, qty=1.0, price=2.0, fees=0.5),
            _ev("outcome", minutes=6, payload={}),
        ]}
        positions, _ = rc.reconcile_positions(resolved)
        p = positions["d1"]
        assert p["proceeds"] == pytest.approx(200.0)
        assert p["realized_cash_pnl"] == pytest.approx(-1.0)  # fees only
        assert p["realized_cash_pnl"] != p["proceeds"]

        orphan = {"d2": [
            _ev("exit", decision="d2", qty=1.0, price=3.0),
            _ev("outcome", decision="d2", minutes=1, payload={}),
        ]}
        positions, report = rc.reconcile_positions(orphan)
        assert positions["d2"]["status"] == "exit_without_entry"
        assert positions["d2"]["realized_cash_pnl"] is None
        assert report["exit_without_entry"] == 1

    def test_duplicate_outcome_keep_first_and_counted(self):
        events = {"d1": [
            _ev("fill", qty=1.0, price=1.0, order="o1"),
            _ev("outcome", minutes=10, payload={"gross_pnl_pct": -5.0}),
            _ev("outcome", minutes=11, key="dup1",
                payload={"gross_pnl_pct": 99.0}),
            _ev("outcome", minutes=12, key="dup2",
                payload={"gross_pnl_pct": 77.0}),
        ]}
        positions, report = rc.reconcile_positions(events)
        p = positions["d1"]
        assert p["ledger_gross_pnl_pct"] == pytest.approx(-5.0)  # first wins
        assert p["duplicate_outcomes"] == 2
        assert report["duplicate_outcomes_dropped"] == 2

    def test_unresolved_and_unfilled_accounting(self):
        events = {
            "open1": [_ev("fill", decision="open1", qty=1.0, price=1.0)],
            "sub1": [_ev("submitted", decision="sub1", qty=1.0)],
        }
        positions, report = rc.reconcile_positions(events)
        assert positions["open1"]["status"] == "open_or_unresolved"
        assert positions["sub1"]["status"] == "submitted_unfilled"
        assert report["open_or_unresolved"] == 1
        assert report["submitted_unfilled"] == 1
        assert report["resolved"] == 0

    def test_missing_fees_stay_unknown_never_inferred(self):
        events = {"d1": [
            _ev("fill", qty=2.0, price=1.0, fees=None, order="o1"),
            _ev("exit", minutes=5, qty=2.0, price=1.2, fees=None),
            _ev("outcome", minutes=6, payload={}),
        ]}
        positions, report = rc.reconcile_positions(events)
        p = positions["d1"]
        assert p["fees_entry_known"] == 0.0
        assert p["fees_exit_known"] == 0.0
        assert p["fees_unknown_events"] == 2
        assert p["fees_fully_known"] is False
        # cash P&L is still reported, but on an explicit ex-fees basis
        assert p["realized_cash_pnl"] == pytest.approx(40.0)
        assert p["cash_pnl_basis"] == "fills_ex_unknown_fees"
        assert report["fees_unknown_events"] == 2
        assert report["fees_reported_events"] == 0

    def test_unallocated_fees_tracked_separately(self):
        events = {"d1": [
            _ev("submitted", qty=1.0, fees=0.75),
            _ev("fill", minutes=1, qty=1.0, price=1.0, fees=0.5),
        ]}
        positions, report = rc.reconcile_positions(events)
        p = positions["d1"]
        assert p["fees_unallocated"] == pytest.approx(0.75)
        assert p["fees_entry_known"] == pytest.approx(0.5)
        assert report["unallocated_fee_events"] == 1

    def test_partial_fill_detected(self):
        events = {"d1": [
            _ev("submitted", qty=5.0),
            _ev("fill", minutes=1, qty=2.0, price=1.0),
        ]}
        _, report = rc.reconcile_positions(events)
        assert report["partial_entries"] == 1

    def test_fill_missing_price_counted_not_guessed(self):
        events = {"d1": [
            _ev("fill", qty=1.0, price=None),
        ]}
        positions, report = rc.reconcile_positions(events)
        assert report["fills_missing_qty_or_price"] == 1
        assert positions["d1"]["entry_price_weighted"] is None
        assert positions["d1"]["matched_qty"] == 0.0

    def test_paper_vs_live_provenance_stratification(self):
        events = {
            "live1": [_ev("fill", decision="live1", qty=1.0, price=1.0,
                          source="live_hook")],
            "pap1": [_ev("fill", decision="pap1", qty=1.0, price=1.0,
                         source="paper")],
        }
        positions, _ = rc.reconcile_positions(events)
        assert positions["live1"]["provenance_sources"] == ["live_hook"]
        assert positions["pap1"]["provenance_sources"] == ["paper"]


# =========================================================================== #
# match_history / marks_vs_fills
# =========================================================================== #

class TestHistoryMatching:
    def _positions(self):
        events = {"d1": [
            _ev("fill", qty=2.0, price=8.25, order="ord-abc"),
            _ev("outcome", minutes=20, payload={"gross_pnl_pct": -21.78}),
        ]}
        positions, _ = rc.reconcile_positions(events)
        return positions

    def test_match_by_order_id_with_pct_delta(self):
        history = [
            {"order_id": "ord-abc", "pnl_percent": -20.0},
            {"order_id": "ord-missing", "pnl_percent": 5.0},
            {"pnl_percent": 1.0},  # no order id
        ]
        out = rc.match_history(history, self._positions())
        assert out["n_history_trades"] == 3
        assert out["n_history_with_order_id"] == 2
        assert out["n_matched"] == 1
        assert out["coverage_history"] == pytest.approx(0.5)
        delta = out["pnl_pct_delta_history_minus_ledger"]
        assert delta["n"] == 1
        assert delta["mean"] == pytest.approx(-20.0 - (-21.78))

    def test_marks_vs_fills_deltas_and_coverage(self):
        history = [
            {"symbol": "SPY261120C00600000", "exit_time": "2026-11-10T15:00:00",
             "pnl": 12.0},
            {"symbol": "SPY261120C00600000", "exit_time": "2026-11-10T16:00:00",
             "pnl": -3.0},
            {"symbol": "QQQ261120P00500000", "exit_time": "2026-11-10T15:00:00",
             "pnl": 7.0},  # no log row -> unmatched history
        ]
        log = [
            {"symbol": "SPY261120C00600000",
             "timestamp": "2026-11-10T15:01:00", "amount": 10.0},
            {"symbol": "SPY261120C00600000",
             "timestamp": "2026-11-10T16:00:30", "amount": -3.0},
            {"symbol": "SPY261120C00600000",
             "timestamp": "2026-11-10T20:00:00", "amount": 99.0},  # unmatched
        ]
        out = rc.marks_vs_fills(history, log)
        assert out["n_matched"] == 2
        assert out["coverage_history"] == pytest.approx(2 / 3)
        assert out["unmatched_log_rows"] == 1
        stats = out["delta_history_pnl_minus_fills_amount"]
        assert stats["mean"] == pytest.approx(1.0)      # (2.0 + 0.0) / 2
        assert stats["frac_exact"] == pytest.approx(0.5)
        # fewer than min_matched_for_ci -> CI suppressed, explicitly
        assert stats["ci95"] is None
        assert "ci_note" in stats

    def test_marks_vs_fills_repeated_close_consumed_once(self):
        """Two identical history closes cannot both claim one fill row."""
        history = [
            {"symbol": "S261120C00100000", "exit_time": "2026-11-10T15:00:00",
             "pnl": 5.0},
            {"symbol": "S261120C00100000", "exit_time": "2026-11-10T15:00:10",
             "pnl": 5.0},
        ]
        log = [{"symbol": "S261120C00100000",
                "timestamp": "2026-11-10T15:00:05", "amount": 5.0}]
        out = rc.marks_vs_fills(history, log)
        assert out["n_matched"] == 1
        assert out["unmatched_log_rows"] == 0
        assert out["coverage_history"] == pytest.approx(0.5)

    def test_marks_vs_fills_ci_when_enough_matches(self):
        history = [{"symbol": "S261120C00100000", "pnl": 1.0 + 0.1 * i,
                    "exit_time": f"2026-11-10T15:{i:02d}:00"}
                   for i in range(12)]
        log = [{"symbol": "S261120C00100000", "amount": 1.0,
                "timestamp": f"2026-11-10T15:{i:02d}:01"}
               for i in range(12)]
        out = rc.marks_vs_fills(history, log)
        stats = out["delta_history_pnl_minus_fills_amount"]
        assert stats["n"] == 12
        assert stats["ci95"] is not None
        lo, hi = stats["ci95"]
        assert lo < stats["mean"] < hi


# =========================================================================== #
# run_reconciliation end-to-end (tmp sqlite ledger, production columns)
# =========================================================================== #

def _make_ledger(path):
    conn = sqlite3.connect(str(path))
    conn.execute(
        "CREATE TABLE snapshots (snapshot_id, decision_id, kind, accepted, "
        "reject_code, event_at_utc, received_at_utc, symbol, underlying, "
        "underlying_price)")
    conn.execute(
        "CREATE TABLE events (event_key, decision_id, event_type, "
        "event_at_utc, received_at_utc, broker_order_id, qty, price, fees, "
        "status, source, payload_json)")
    t = "2026-11-10T15:00:00+00:00"
    t2 = "2026-11-10T15:30:00+00:00"
    conn.execute(
        "INSERT INTO snapshots VALUES ('s1','d1','candidate',1,NULL,?,?,"
        "'SPY261120C00600000','SPY',600.0)", (t, t))
    conn.execute(
        "INSERT INTO snapshots VALUES ('s2','d2','shadow_candidate',0,"
        "'LOW_SCORE',?,?,'SPY261120P00590000','SPY',600.0)", (t, t))
    rows = [
        ("e1", "d1", "submitted", t, t, "ord-1", 2.0, None, None,
         "accepted", "paper", "{}"),
        ("e2", "d1", "fill", t, t, "ord-1", 2.0, 8.25, None,
         "filled", "live_hook", "{}"),
        ("e3", "d1", "exit", t2, t2, "ord-1", 2.0, None, None,
         None, "paper", '{"reason": "dynamic_stop_loss"}'),
        ("e4", "d1", "outcome", t2, t2, None, None, None, None,
         None, "paper", '{"gross_pnl_pct": -21.78, "class": "realized"}'),
    ]
    conn.executemany(
        "INSERT INTO events VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", rows)
    conn.commit()
    conn.close()


class TestRunReconciliation:
    def test_end_to_end_report(self, tmp_path):
        ledger = tmp_path / "ledger.db"
        _make_ledger(ledger)
        history = tmp_path / "history.json"
        history.write_text(json.dumps({"trades": [
            {"order_id": "ord-1", "pnl_percent": -21.0, "pnl": -36.0,
             "symbol": "SPY261120C00600000",
             "exit_time": "2026-11-10T15:30:10"},
        ]}), encoding="utf-8")
        realized = tmp_path / "realized.json"
        realized.write_text(json.dumps([
            {"date": "2026-11-10", "timestamp": "2026-11-10T15:30:12",
             "amount": -35.5, "symbol": "SPY261120C00600000"},
        ]), encoding="utf-8")
        out = tmp_path / "report.json"

        rep = rc.run_reconciliation(
            str(ledger), str(history), str(realized), out_path=str(out))
        assert rep["verdict"] == "OK"
        assert rep["ingest"]["events_kept"] == 4
        assert rep["positions"]["resolved"] == 1
        assert rep["provenance_breakdown"].get("live_hook") == 1
        assert rep["history_match"]["n_matched"] == 1
        assert rep["marks_vs_fills"]["n_matched"] == 1
        assert rep["capture_groups"]["by_kind_accepted"][
            "candidate|accepted=1"] == 1
        assert rep["capture_groups"]["rejected_by_code"]["LOW_SCORE"] == 1
        # the old +$51.9 EV must be explicitly invalidated
        inv = rep["invalidated_estimates"][0]
        assert inv["value"] == 51.9 and inv["status"] == "INVALIDATED"
        assert out.exists()
        on_disk = json.loads(out.read_text(encoding="utf-8"))
        assert on_disk["verdict"] == "OK"

    def test_missing_ledger_fails_open(self, tmp_path):
        rep = rc.run_reconciliation(str(tmp_path / "nope.db"))
        assert rep["verdict"] == "INSUFFICIENT_DATA"
        assert "error" in rep


# =========================================================================== #
# pnl_calibration: extraction + split guards
# =========================================================================== #

def _trade(symbol="SPY261113C00600000", entry="2026-11-10T14:30:00",
           pnl=5.0, pop=0.6, ev=4.0, units="usd_total", regime="trend"):
    return {
        "symbol": symbol, "entry_time": entry, "pnl": pnl,
        "probability_of_profit": pop, "expected_value": ev,
        "metrics": {"ev_units": units}, "regime_label": regime,
        "order_id": "x",
    }


class TestOpportunityExtraction:
    def test_same_unit_enforcement(self):
        trades = [_trade(), _trade(units="usd_per_contract"),
                  _trade(units=None)]
        opps, report = pc.opportunities_from_history(trades)
        assert report["kept"] == 1
        assert report["dropped_unit_mismatch"] == 2
        assert len(opps) == 1

    def test_missing_fields_and_bad_timestamp_counted(self):
        trades = [_trade(pnl=None), _trade(pop=None),
                  _trade(entry="not-a-time"), _trade()]
        opps, report = pc.opportunities_from_history(trades)
        assert report["dropped_missing_fields"] == 2
        assert report["dropped_bad_timestamp"] == 1
        assert report["kept"] == 1

    def test_occ_parsing_side_and_dte(self):
        opps, _ = pc.opportunities_from_history(
            [_trade(symbol="QQQ261113P00500000",
                    entry="2026-11-10T14:30:00")])
        o = opps[0]
        assert o["underlying"] == "QQQ"
        assert o["side"] == "put"
        assert o["dte"] == 3
        assert o["dte_bucket"] == "2-7"
        assert o["provenance"] == "paper_marks"


def _opps(n_sessions, per_session, ev_fn, realized_fn, start=date(2026, 1, 5)):
    out = []
    for s in range(n_sessions):
        day = start + timedelta(days=s)
        for i in range(per_session):
            ev = ev_fn(s, i)
            realized = realized_fn(s, i, ev)
            entry = datetime(day.year, day.month, day.day, 15, 0,
                             tzinfo=UTC) + timedelta(minutes=i)
            out.append({
                "entry_time": entry, "session": day.isoformat(),
                "symbol": f"SPY{s}_{i}", "underlying": "SPY",
                "side": "call" if i % 2 == 0 else "put",
                "dte": 3, "dte_bucket": "2-7",
                "regime": "trend" if i % 3 else "chop",
                "p_pred": pc._sigmoid(ev / 20.0),
                "ev_pred": ev, "realized": realized,
                "win": realized > 0.0, "provenance": "paper_marks",
            })
    return out


class TestChronologicalSplit:
    def test_split_is_strictly_forward(self):
        opps = _opps(8, 3, lambda s, i: float(i), lambda s, i, e: e)
        train, eval_, meta = pc.chronological_split(opps, 0.25)
        assert meta["n_eval_sessions"] == 2
        assert max(o["session"] for o in train) < \
            min(o["session"] for o in eval_)
        assert meta["n_train"] == len(train) == 18
        assert meta["n_eval"] == len(eval_) == 6

    def test_future_in_train_raises(self):
        opps = _opps(4, 2, lambda s, i: float(i), lambda s, i, e: e)
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        # smuggle an eval-session row into train -> hard failure
        with pytest.raises(ValueError, match="strictly before"):
            pc.validate_split(train + [eval_[0]], eval_)

    def test_single_session_rejected(self):
        opps = _opps(1, 5, lambda s, i: float(i), lambda s, i, e: e)
        with pytest.raises(ValueError, match="at least 2 sessions"):
            pc.chronological_split(opps)


# =========================================================================== #
# pnl_calibration: model components
# =========================================================================== #

class TestModelComponents:
    def test_logistic_fit_recovers_known_coefficients(self):
        rng = random.Random(0)
        a_true, b_true = 0.5, 2.0
        xs, ys = [], []
        for _ in range(3000):
            x = rng.uniform(-2.0, 2.0)
            xs.append(x)
            ys.append(rng.random() < pc._sigmoid(a_true + b_true * x))
        a, b = pc.fit_logistic(xs, ys)
        assert abs(a - a_true) < 0.2
        assert abs(b - b_true) < 0.3

    def test_cohort_shrinkage_math_exact(self):
        train = (
            [{"regime": "trend", "side": "call", "dte_bucket": "2-7",
              "win": True}] * 10
            + [{"regime": "chop", "side": "put", "dte_bucket": "0-1",
                "win": False}] * 30
        )
        model = pc.cohort_rates(train, k=20.0)
        g = 10 / 40
        assert model["global_rate"] == pytest.approx(g)
        # (n*r + k*g) / (n + k) = (10*1 + 20*0.25) / 30 = 0.5
        assert model["rates"]["trend|call|2-7"] == pytest.approx(0.5)
        assert model["rates"]["chop|put|0-1"] == pytest.approx(
            (30 * 0.0 + 20.0 * g) / 50.0)
        # unseen cohort falls back to the global rate
        assert pc._cohort_predict(
            model, {"regime": "x", "side": "y", "dte_bucket": "z"}
        ) == pytest.approx(g)

    def test_ev_bucket_edges_and_assignment_monotone(self):
        train = [{"ev_pred": float(v)} for v in range(100)]
        edges = pc.ev_bucket_edges(train, n_buckets=5)
        assert len(edges) == 4
        assert edges == sorted(edges)
        buckets = [pc._bucket_of(float(v), edges) for v in range(100)]
        assert buckets == sorted(buckets)
        assert set(buckets) == {0, 1, 2, 3, 4}

    def test_spearman_perfect_and_ties(self):
        assert pc.spearman([1, 2, 3, 4], [10, 20, 30, 40]) == \
            pytest.approx(1.0)
        assert pc.spearman([1, 2, 3, 4], [40, 30, 20, 10]) == \
            pytest.approx(-1.0)
        assert pc.spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]) is None
        rho = pc.spearman([1, 2, 2, 3], [1, 2, 2, 3])
        assert rho == pytest.approx(1.0)


# =========================================================================== #
# pnl_calibration: evaluation + verdicts
# =========================================================================== #

class TestEvaluateAndVerdict:
    def test_positive_rank_power_validated(self):
        rng = random.Random(1)
        opps = _opps(
            20, 12,
            ev_fn=lambda s, i: -50.0 + 100.0 * ((s * 12 + i) % 24) / 23.0,
            realized_fn=lambda s, i, e: e + rng.gauss(0.0, 5.0))
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        assert len(eval_) >= pc.MIN_EVAL_N
        metrics = pc.evaluate(train, eval_)
        verdict = pc._verdict(metrics)
        assert verdict["verdict"] == "VALIDATED_RANK_POWER"
        assert metrics["ev_rank"]["spearman"] > 0.8
        assert metrics["ev_rank"]["session_bootstrap"]["lower_bound"] > 0
        assert metrics["ev_rank"]["ols_slope_realized_on_ev"] > 0.5

    def test_anti_correlated_fixture_is_no_validated_edge(self):
        rng = random.Random(2)
        opps = _opps(
            20, 12,
            ev_fn=lambda s, i: -50.0 + 100.0 * ((s * 12 + i) % 24) / 23.0,
            realized_fn=lambda s, i, e: -e + rng.gauss(0.0, 5.0))
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        metrics = pc.evaluate(train, eval_)
        verdict = pc._verdict(metrics)
        assert verdict["verdict"] == "NO_VALIDATED_EDGE"
        assert verdict["challenger_policy"] == "abstain"
        assert metrics["ev_rank"]["spearman"] < 0

    def test_small_eval_is_insufficient_evidence(self):
        opps = _opps(8, 4, lambda s, i: float(i),
                     lambda s, i, e: e - 1.5)
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        metrics = pc.evaluate(train, eval_)
        assert metrics["n_eval"] < pc.MIN_EVAL_N
        verdict = pc._verdict(metrics)
        assert verdict["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert verdict["challenger_policy"] == "abstain"

    def test_evaluate_rejects_leaky_split(self):
        opps = _opps(6, 4, lambda s, i: float(i), lambda s, i, e: e)
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        with pytest.raises(ValueError, match="strictly before"):
            pc.evaluate(train + [eval_[0]], eval_)

    def test_per_fill_equals_per_opportunity_with_note(self):
        opps = _opps(8, 8, lambda s, i: float(i) - 3.0,
                     lambda s, i, e: e + 0.5)
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        metrics = pc.evaluate(train, eval_)
        assert metrics["expectancy_per_fill_usd"] == \
            metrics["expectancy_per_opportunity_usd"]
        assert "survivorship" in metrics["per_fill_note"]

    def test_metrics_shapes_and_sanity(self):
        rng = random.Random(3)
        opps = _opps(
            16, 8,
            ev_fn=lambda s, i: -20.0 + 5.0 * i,
            realized_fn=lambda s, i, e: e + rng.gauss(0.0, 10.0))
        train, eval_, _ = pc.chronological_split(opps, 0.25)
        m = pc.evaluate(train, eval_)
        for name in ("stamped_pop", "base_rate", "logistic_recal",
                     "cohort_shrunk"):
            assert 0.0 <= m["prob_metrics"][name]["brier"] <= 1.0
            assert m["prob_metrics"][name]["log_loss"] >= 0.0
        # constant base-rate predictions cannot have a calibration slope
        assert m["prob_metrics"]["base_rate"]["calibration_slope"] is None
        assert sum(b["n"] for b in m["reliability_bins"]) == m["n_eval"]
        assert sum(r["n"] for r in m["ev_buckets"]["rows"]) == m["n_eval"]
        res = m["signed_residual_usd"]
        assert res["n"] == m["n_eval"] and res["ci95"] is not None


# =========================================================================== #
# run_pnl_calibration end-to-end
# =========================================================================== #

class TestRunPnlCalibration:
    def test_report_written_with_preregistration(self, tmp_path):
        trades = []
        for s in range(6):
            day = f"2026-11-{3 + s:02d}"
            for i in range(4):
                trades.append(_trade(
                    entry=f"{day}T14:{30 + i}:00",
                    pnl=(1.0 if i % 2 else -1.0), pop=0.55, ev=0.5))
        hist = tmp_path / "history.json"
        hist.write_text(json.dumps({"trades": trades}), encoding="utf-8")
        out = tmp_path / "cal.json"
        rep = pc.run_pnl_calibration(str(hist), out_path=str(out))
        assert rep["extraction"]["kept"] == 24
        assert rep["split"]["n_eval_sessions"] >= 1
        # tiny fixture -> must refuse to claim an edge
        assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert rep["challenger_policy"] == "abstain"
        assert rep["pre_registered"]["boot_seed"] == pc.BOOT_SEED
        assert out.exists()

    def test_missing_history_fails_open(self, tmp_path):
        rep = pc.run_pnl_calibration(str(tmp_path / "nope.json"))
        assert rep["verdict"] == "INSUFFICIENT_EVIDENCE"
        assert rep["challenger_policy"] == "abstain"
        assert "error" in rep
