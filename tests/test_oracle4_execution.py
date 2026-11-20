"""Phase 6a acceptance tests — execution calibration (hand-computed).

Hermetic: every fixture is synthetic and written to tmp_path. No broker,
no network, no production imports.
"""
from __future__ import annotations

import json
import math
import sqlite3

import pytest

from oracle.ml.execution import (
    CONSERVATIVE_FILL_PRIOR,
    PRIOR_STRENGTH,
    ev_calibration_report,
    fill_probability_report,
    jeffreys_interval,
    link_pairs,
    load_execution_log,
    load_history_trades,
    quantiles,
    run_execution_calibration,
    shrunk_fill_probability,
    spread_bucket,
    stop_overshoot_report,
    time_to_fill_report,
)


# --------------------------------------------------------------------------- #
# statistics
# --------------------------------------------------------------------------- #

def test_jeffreys_interval_symmetric_midpoint():
    lo, hi = jeffreys_interval(5, 10)
    assert 0.0 < lo < 0.5 < hi < 1.0
    # Beta(5.5, 5.5) is symmetric around 0.5
    assert abs((lo + hi) / 2.0 - 0.5) < 1e-6


def test_jeffreys_interval_all_filled_is_upper_bound_one():
    lo, hi = jeffreys_interval(10, 10)
    assert hi == 1.0
    assert 0.70 < lo < 0.90          # known ~0.78 for Jeffreys k=n=10
    lo9, _ = jeffreys_interval(9, 10)
    assert lo9 < lo                   # monotone in k


def test_jeffreys_interval_zero_n():
    assert jeffreys_interval(0, 0) == (0.0, 1.0)


def test_shrunk_fill_probability_exact():
    # (k + m*p0) / (n + m) with p0=0.8, m=10
    assert shrunk_fill_probability(0, 0) == pytest.approx(0.8)
    assert shrunk_fill_probability(100, 100) == pytest.approx(108.0 / 110.0)
    assert shrunk_fill_probability(0, 10) == pytest.approx(8.0 / 20.0)
    # thin all-filled group is pulled DOWN toward the conservative prior
    assert shrunk_fill_probability(3, 3) < 1.0
    assert CONSERVATIVE_FILL_PRIOR == 0.80 and PRIOR_STRENGTH == 10.0


def test_quantiles_hand_computed():
    q = quantiles([5.0, 1.0, 3.0, 2.0, 4.0])
    assert q["p50"] == pytest.approx(3.0)
    assert q["p25"] == pytest.approx(2.0)
    assert q["min"] == 1.0 and q["max"] == 5.0
    assert q["mean"] == pytest.approx(3.0)
    assert quantiles([]) == {}


def test_spread_bucket_boundaries():
    assert spread_bucket(None) == "unknown"
    assert spread_bucket(0.019) == "tight(<2%)"
    assert spread_bucket(0.02) == "normal(2-5%)"
    assert spread_bucket(0.05) == "wide(5-10%)"
    assert spread_bucket(0.10) == "very_wide(>=10%)"


# --------------------------------------------------------------------------- #
# loaders + linking
# --------------------------------------------------------------------------- #

def _write_jsonl(path, records):
    with open(path, "w", encoding="utf-8") as fh:
        for r in records:
            fh.write((r if isinstance(r, str) else json.dumps(r)) + "\n")


def _est(tid, **kw):
    base = {"record_type": "execution_estimate", "trade_id": tid,
            "recorded_at": "2026-11-10T15:00:00+00:00", "qty": 1,
            "mid_price": 10.0, "spread_cost": 50.0,
            "expected_entry_price": 10.2, "expected_exit_price": 10.5,
            "fill_probability": 0.99, "expected_fill_delay": 1.0,
            "executable_EV": 30.0, "theoretical_EV": 40.0}
    base.update(kw)
    return base


def _real(tid, **kw):
    base = {"record_type": "execution_realization", "trade_id": tid,
            "resolved_at": "2026-11-10T20:00:00+00:00", "filled": True,
            "actual_entry_price": 10.1, "actual_exit_price": 10.6,
            "actual_fill_delay": None, "realized_EV": 50.0,
            "resolution_status": "resolved"}
    base.update(kw)
    return base


def test_load_execution_log_validates_and_dedups(tmp_path):
    p = tmp_path / "exec.jsonl"
    _write_jsonl(p, [
        _est("a"), _est("a"),                       # duplicate
        _real("a"),
        {"record_type": "order_intent", "trade_id": "i1"},
        {"record_type": "mystery", "trade_id": "x"},  # unknown type
        {"record_type": "execution_estimate"},        # missing trade_id
        "{not json",                                   # bad json
    ])
    records, report = load_execution_log(str(p))
    assert report["kept_execution_estimate"] == 1
    assert report["kept_execution_realization"] == 1
    assert report["kept_order_intent"] == 1
    assert report["dropped_duplicate"] == 1
    assert report["dropped_unknown_type"] == 1
    assert report["dropped_missing_trade_id"] == 1
    assert report["dropped_bad_json"] == 1
    assert records["execution_estimate"][0]["trade_id"] == "a"


def test_link_pairs_counts_unlinked_both_sides():
    ests = [_est("a"), _est("b")]
    reals = [_real("a"), _real("orphan")]
    pairs, report = link_pairs(ests, reals)
    assert report == {"pairs_linked": 1, "estimates_unlinked": 1,
                      "realizations_unlinked": 1}
    pair = pairs[0]
    assert pair["trade_id"] == "a"
    assert pair["session"] == "2026-11-10"
    assert pair["filled"] is True
    assert pair["actual_fill_delay"] is None


def test_load_history_trades(tmp_path):
    p = tmp_path / "hist.json"
    p.write_text(json.dumps({"trades": [{"outcome": "x"}, "junk"]}),
                 encoding="utf-8")
    trades, report = load_history_trades(str(p))
    assert report == {"trades_kept": 1, "dropped_not_dict": 1}
    assert trades[0]["outcome"] == "x"


# --------------------------------------------------------------------------- #
# report blocks
# --------------------------------------------------------------------------- #

def test_fill_probability_report_limit_tp_and_survivorship():
    records = {
        "execution_estimate": [],
        "execution_realization": [_real(f"t{i}") for i in range(4)],
        "order_intent": [{"trade_id": "i"}] * 3,
    }
    history = ([{"outcome": "roll_take_profit"}] * 6
               + [{"outcome": "roll_failed_take_profit"}] * 4)
    rep = fill_probability_report(records, pairs=[], history=history,
                                  min_group=2)
    overall = rep["resolved_paper_overall"]
    assert overall["n"] == 4 and overall["filled"] == 4
    assert overall["raw_rate"] == 1.0
    assert "SURVIVORSHIP" in overall["note"]
    tp = rep["limit_take_profit"]
    assert tp["n"] == 10 and tp["filled"] == 6
    assert tp["raw_rate"] == pytest.approx(0.6)
    lo, hi = tp["jeffreys_95ci"]
    assert lo < 0.6 < hi
    intents = rep["indicative_intents"]
    assert intents["n"] == 3 and intents["submitted_to_broker"] == 0


def test_fill_probability_thin_groups_pool_and_shrink():
    # 3 pairs in one bucket, min_group=5 -> pooled, shrunk toward 0.8
    pairs = [{"filled": True, "spread_cost": 10.0, "mid_price": 10.0,
              "qty": 1.0}] * 3   # spread_pct=0.01 -> tight
    records = {"execution_estimate": [], "execution_realization": [],
               "order_intent": []}
    rep = fill_probability_report(records, pairs=pairs, history=[],
                                  min_group=5)
    pooled = rep["by_spread_bucket"]["pooled(<min_group)"]
    assert pooled["n"] == 3 and pooled["filled"] == 3
    assert pooled["shrunk_rate"] == pytest.approx(11.0 / 13.0)
    assert pooled["shrunk_rate"] < 1.0


def test_time_to_fill_unknown_coverage_not_zero():
    pairs = [{"expected_fill_delay": 1.0, "actual_fill_delay": None}] * 3
    rep = time_to_fill_report(pairs)
    assert rep["status"] == "UNKNOWN_COVERAGE"
    assert rep["n_expected"] == 3 and rep["n_actual"] == 0
    assert rep["actual_quantiles_s"] == {}


def test_ev_calibration_recovers_exact_linear_relation():
    # realized = 2*predicted + 1, across 3 sessions
    pairs = []
    for i, (pred, sess) in enumerate([(10.0, "2026-11-10"),
                                      (20.0, "2026-11-10"),
                                      (30.0, "2026-11-11"),
                                      (40.0, "2026-11-12")]):
        pairs.append({"executable_EV": pred, "realized_EV": 2 * pred + 1,
                      "session": sess})
    rep = ev_calibration_report(pairs, n_boot=100, seed=1)
    assert rep["n"] == 4 and rep["n_sessions"] == 3
    assert rep["ols_slope_realized_on_predicted"] == pytest.approx(2.0)
    assert rep["ols_intercept"] == pytest.approx(1.0)
    assert rep["correlation"] == pytest.approx(1.0)
    # error = realized - predicted = pred + 1
    assert rep["mean_error_realized_minus_predicted"] == pytest.approx(26.0)


def test_ev_calibration_empty_is_insufficient():
    assert ev_calibration_report([])["status"] == "INSUFFICIENT_EVIDENCE"


# --------------------------------------------------------------------------- #
# stop overshoot (synthetic ledger + dataset)
# --------------------------------------------------------------------------- #

def _make_ledger(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE snapshots "
                 "(decision_id TEXT, prediction_json TEXT)")
    conn.executemany("INSERT INTO snapshots VALUES (?, ?)", rows)
    conn.commit()
    conn.close()


def _make_dataset(path, rows):
    conn = sqlite3.connect(str(path))
    conn.execute("CREATE TABLE outcomes (decision_id TEXT, status TEXT, "
                 "gross_pnl_pct REAL, reason TEXT)")
    conn.executemany("INSERT INTO outcomes VALUES (?, ?, ?, ?)", rows)
    conn.commit()
    conn.close()


def test_stop_overshoot_hand_computed(tmp_path):
    ledger = tmp_path / "ledger.db"
    dataset = tmp_path / "ds.sqlite"
    pj = json.dumps({"exit_policy": {"stop_loss_pct": 0.15}})
    _make_ledger(ledger, [("d1", pj), ("d2", pj), ("d3", pj)])
    _make_dataset(dataset, [
        # stop floor -15; gross -20 -> overshoot +5 (worse than floor)
        ("d1", "mark_derived", -20.0, "dynamic_stop_loss"),
        # gross -16 -> overshoot +1
        ("d2", "mark_derived", -16.0, "dynamic_stop_loss"),
        # trailing stop exits above the floor: gross -5 -> overshoot -10
        ("d3", "mark_derived", -5.0, "dynamic_trailing_stop"),
        # no stop config in the ledger -> counted missing, not guessed
        ("d9", "mark_derived", -30.0, "dynamic_stop_loss"),
        # non-mark rows are excluded by the query
        ("d1", "counterfactual", None, "dynamic_stop_loss"),
    ])
    rep = stop_overshoot_report(str(dataset), str(ledger))
    hard = rep["dynamic_stop_loss"]
    assert hard["n"] == 2
    assert hard["frac_worse_than_floor"] == 1.0
    assert hard["overshoot_quantiles_pts"]["min"] == pytest.approx(1.0)
    assert hard["overshoot_quantiles_pts"]["max"] == pytest.approx(5.0)
    assert hard["overshoot_quantiles_pts"]["mean"] == pytest.approx(3.0)
    trail = rep["dynamic_trailing_stop"]
    assert trail["n"] == 1
    assert trail["frac_worse_than_floor"] == 0.0
    assert trail["overshoot_quantiles_pts"]["mean"] == pytest.approx(-10.0)
    assert rep["missing_stop_config"] == 1


# --------------------------------------------------------------------------- #
# end-to-end determinism
# --------------------------------------------------------------------------- #

def test_run_execution_calibration_deterministic(tmp_path):
    log = tmp_path / "exec.jsonl"
    _write_jsonl(log, [_est("a"), _real("a"), _est("b"), _real("b")])
    hist = tmp_path / "hist.json"
    hist.write_text(json.dumps({"trades": [
        {"outcome": "roll_take_profit"},
        {"outcome": "roll_failed_take_profit"},
    ]}), encoding="utf-8")
    r1 = run_execution_calibration(str(log), str(hist), n_boot=50, seed=7)
    r2 = run_execution_calibration(str(log), str(hist), n_boot=50, seed=7)
    assert r1["fingerprint"] == r2["fingerprint"]
    assert r1["link_report"]["pairs_linked"] == 2
    assert r1["stop_overshoot"]["status"] == "SKIPPED"
    assert r1["time_to_fill"]["status"] == "UNKNOWN_COVERAGE"
    assert "PAPER" in r1["provenance_note"]
