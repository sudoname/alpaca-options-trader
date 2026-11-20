"""Acceptance tests for oracle/ml/replay.py (Oracle 4.0 Phase 6b).

Hermetic and hand-computed: store fixtures are written through the
production writer ``oracle.ml.dataset._write_store`` so the loaders are
exercised against the real on-disk schema.  No network, no production
imports in the harness, no real market data.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timedelta, timezone

import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.ml import replay as rp  # noqa: E402
from oracle.ml.dataset import _write_store  # noqa: E402
from oracle.ml.schema import (  # noqa: E402
    FeatureRow,
    LabelRow,
    LabelStatus,
    OutcomeRecord,
    OutcomeStatus,
)

UTC = timezone.utc
SESSIONS = ["2026-01-05", "2026-01-06", "2026-01-07", "2026-01-08",
            "2026-01-09"]
HOLDOUT = ["2026-01-12"]


def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _trade(did: str, session: str, minute: int = 0, exit_minutes: int = 60,
           option_type: str = "call", fill_price: float = 10.0,
           fill_qty: float = 1.0, gross: float = 10.0,
           reason: str = "take_profit", spread_pct=0.03,
           has_quote=True) -> rp.Trade:
    entry = _ts(session, minute)
    return rp.Trade(
        decision_id=did, session=session, entry_at=entry,
        exit_at=entry + timedelta(minutes=exit_minutes),
        symbol=None, underlying="SPY", option_type=option_type,
        fill_price=fill_price, fill_qty=fill_qty, gross_pnl_pct=gross,
        reason=reason, spread_pct=spread_pct, has_quote=has_quote)


# --------------------------------------------------------------------------- #
# metrics helpers (hand-computed)
# --------------------------------------------------------------------------- #

class TestMetricHelpers:
    def test_sharpe_hand_computed(self):
        # mean 0.01, sample std over n-1 of [0.0,0.02] = sqrt(2e-4) ~ 0.01414
        r = [0.0, 0.02]
        expect = (0.01 / (0.0002 ** 0.5)) * (252 ** 0.5)
        assert rp.sharpe_ratio(r) == pytest.approx(expect)

    def test_sharpe_undefined_cases(self):
        assert rp.sharpe_ratio([]) is None
        assert rp.sharpe_ratio([0.01]) is None            # n < 2
        assert rp.sharpe_ratio([0.01, 0.01]) is None      # zero variance
        assert rp.sharpe_ratio([0.01, -0.01]) == pytest.approx(0.0)

    def test_max_drawdown(self):
        assert rp.max_drawdown([100.0, 110.0, 99.0, 105.0]) == pytest.approx(
            (99.0 - 110.0) / 110.0)
        assert rp.max_drawdown([100.0, 120.0]) == 0.0
        assert rp.max_drawdown([]) == 0.0

    def test_profit_factor(self):
        assert rp.profit_factor([10.0, -5.0]) == pytest.approx(2.0)
        assert rp.profit_factor([10.0, 5.0]) is None      # no losses
        assert rp.profit_factor([]) is None

    def test_gate_direction(self):
        # 3-class verdicts (Oracle 4.0.1): pass / gated_out / abstain.
        # call passes only when up is argmax AND edge >= margin
        assert rp.gate_direction("call", 0.1, 0.2, 0.7) == "pass"
        # argmax down -> opposite direction endorsed -> gated out
        assert rp.gate_direction("call", 0.7, 0.2, 0.1) == "gated_out"
        # up argmax but edge below margin -> abstain, not a pass
        assert rp.gate_direction("call", 0.32, 0.30, 0.38) == "abstain"
        # neutral argmax -> abstain for a call
        assert rp.gate_direction("call", 0.2, 0.6, 0.2) == "abstain"
        # put mirror
        assert rp.gate_direction("put", 0.7, 0.2, 0.1) == "pass"
        assert rp.gate_direction("put", 0.1, 0.2, 0.7) == "gated_out"
        assert rp.gate_direction("put", 0.38, 0.30, 0.32) == "abstain"
        # DEFECT COUNTEREXAMPLE: the old binary gate (p_up < 0.5 => put ok)
        # passed this put even though neutral is the most likely outcome.
        # Correct verdict: abstain (no-trade) despite low upward probability.
        assert rp.gate_direction("put", 0.25, 0.45, 0.30) == "abstain"
        # neutral is NEVER a put signal even with a big "not-up" margin
        assert rp.gate_direction("put", 0.30, 0.60, 0.10) == "abstain"
        # unknown option type -> gate cannot act, trade passes through
        assert rp.gate_direction(None, 0.0, 0.0, 0.0) == "pass"


# --------------------------------------------------------------------------- #
# portfolio engine (hand-computed money)
# --------------------------------------------------------------------------- #

class TestPortfolioEngine:
    def test_single_trade_frictionless(self):
        t = _trade("a", SESSIONS[0], gross=10.0)   # premium 1000 -> +100
        m = rp.run_portfolio([t], [SESSIONS[0]])
        assert m["n_executed"] == 1
        assert m["net_pnl"] == pytest.approx(100.0)
        assert m["gross_pnl"] == pytest.approx(100.0)
        assert m["friction_paid"] == 0.0
        assert m["win_rate"] == 1.0
        assert m["profit_factor"] is None                 # no losses
        assert m["turnover"] == pytest.approx(1000.0 / 100_000.0)
        assert m["final_equity"] == pytest.approx(100_100.0)
        assert m["per_trade_pnl_sum_check"] == pytest.approx(m["net_pnl"])
        assert m["pnl_by_exit_day"] == {SESSIONS[0]: pytest.approx(100.0)}

    def test_friction_round_trip(self):
        # one_way 1% x mult 2 = 2% per side on 1000 premium = 20 + 20
        t = _trade("a", SESSIONS[0], gross=10.0)
        m = rp.run_portfolio([t], [SESSIONS[0]],
                             one_way_frac=0.01, friction_mult=2.0)
        assert m["friction_paid"] == pytest.approx(40.0)
        # value 1100, exit fee 20, cost 1020 -> pnl 60
        assert m["net_pnl"] == pytest.approx(60.0)
        assert m["gross_pnl"] == pytest.approx(100.0)
        assert m["per_trade_pnl_sum_check"] == pytest.approx(60.0)

    def test_premium_floor_cannot_lose_more_than_premium(self):
        t = _trade("a", SESSIONS[0], gross=-150.0)
        m = rp.run_portfolio([t], [SESSIONS[0]])
        assert m["net_pnl"] == pytest.approx(-1000.0)     # floored at 0 value
        assert m["gross_pnl"] == pytest.approx(-1500.0)   # raw mark move

    def test_concurrency_skip_then_reentry(self):
        # a and b overlap; with max_concurrent=1, b is refused.
        # c enters after a exits and is accepted.
        a = _trade("a", SESSIONS[0], minute=0, exit_minutes=60)
        b = _trade("b", SESSIONS[0], minute=10, exit_minutes=60)
        c = _trade("c", SESSIONS[0], minute=120, exit_minutes=30)
        m = rp.run_portfolio([a, b, c], [SESSIONS[0]], max_concurrent=1)
        assert m["n_executed"] == 2
        assert m["skipped"] == {"concurrency": 1}
        assert m["net_pnl"] == pytest.approx(200.0)

    def test_cash_skip(self):
        a = _trade("a", SESSIONS[0], fill_price=10.0)     # needs 1000
        b = _trade("b", SESSIONS[0], minute=1, fill_price=10.0)
        m = rp.run_portfolio([a, b], [SESSIONS[0]], starting_cash=1500.0)
        assert m["n_executed"] == 1
        assert m["skipped"] == {"cash": 1}

    def test_multi_day_exit_and_zero_trade_day(self):
        # enter day1, exit on day3 (exit day NOT in sessions list);
        # day2 is a zero-trade day carried at cost -> 0 return.
        t = _trade("a", SESSIONS[0], exit_minutes=60 * 24 * 2)  # +2 days
        m = rp.run_portfolio([t], [SESSIONS[0], SESSIONS[1]])
        exit_day = t.exit_at.date().isoformat()
        assert m["sessions"] == [SESSIONS[0], SESSIONS[1], exit_day]
        curve = dict((d, e) for d, e in m["equity_curve"])
        assert curve[SESSIONS[0]] == pytest.approx(100_000.0)  # at cost
        assert curve[SESSIONS[1]] == pytest.approx(100_000.0)
        assert curve[exit_day] == pytest.approx(100_100.0)
        assert m["pnl_by_exit_day"] == {exit_day: pytest.approx(100.0)}
        # returns: [0, 0, +0.001] -> zero-trade day contributes 0
        assert m["sharpe_daily_ann"] is not None

    def test_overshoot_stress_hits_only_hard_stops(self):
        stop = _trade("s", SESSIONS[0], gross=-25.0, reason="dynamic_stop_loss")
        tp = _trade("t", SESSIONS[0], minute=1, gross=10.0,
                    reason="take_profit")
        base = rp.run_portfolio([stop, tp], [SESSIONS[0]])
        stressed = rp.run_portfolio([stop, tp], [SESSIONS[0]],
                                    overshoot_extra_pts=5.0)
        # only the stop trade degrades: 5 pts of its 1000 premium = -50
        assert stressed["net_pnl"] - base["net_pnl"] == pytest.approx(-50.0)

    def test_exposure_and_drawdown(self):
        lose = _trade("l", SESSIONS[0], gross=-40.0)
        m = rp.run_portfolio([lose], [SESSIONS[0]])
        assert m["max_drawdown"] == pytest.approx(-400.0 / 100_000.0)
        assert m["avg_exposure"] == 0.0  # closed within the session

    def test_deterministic(self):
        ts = [_trade(f"d{i}", SESSIONS[0], minute=i, gross=(-1.0) ** i * 10)
              for i in range(6)]
        m1 = rp.run_portfolio(ts, [SESSIONS[0]], one_way_frac=0.02)
        m2 = rp.run_portfolio(list(reversed(ts)), [SESSIONS[0]],
                              one_way_frac=0.02)
        assert m1 == m2   # input order cannot matter


# --------------------------------------------------------------------------- #
# variants (paired selection accounting)
# --------------------------------------------------------------------------- #

class TestBuildVariants:
    def _trades(self):
        return [
            _trade("good", SESSIONS[0]),
            _trade("wide", SESSIONS[0], minute=1, spread_pct=0.25),
            _trade("noquote", SESSIONS[0], minute=2, has_quote=False),
            _trade("putx", SESSIONS[0], minute=3, option_type="put"),
        ]

    def test_quality_gate(self):
        v = rp.build_variants(self._trades(), None, {}, {}, "unavailable")
        acct = v["baseline_quality"]["accounting"]
        kept = {t.decision_id for t in v["baseline_quality"]["trades"]}
        assert kept == {"good", "putx"}
        assert acct["dropped_quality"] == 2
        assert v["baseline_frozen"]["accounting"]["n"] == 4
        assert "valuation_ranked" not in v
        assert v["neural_gated"]["trades"] is None
        assert v["neural_gated"]["accounting"] == {"status": "unavailable"}

    def test_valuation_counts_unavailable_never_fabricates(self):
        # valuation picked one traded id and two ids with no recorded outcome
        v = rp.build_variants(
            self._trades(), {"good", "ghost1", "ghost2"}, {}, {},
            "unavailable")
        acct = v["valuation_ranked"]["accounting"]
        assert [t.decision_id for t in v["valuation_ranked"]["trades"]] == \
            ["good"]
        assert acct["outcome_unavailable"] == 2
        assert acct["selected_with_recorded_outcome"] == 1
        assert "never fabricated" in acct["note"]

    def test_direction_gate_with_partial_coverage(self):
        preds = {
            "good": (0.1, 0.2, 0.7),   # call, up argmax -> pass
            "putx": (0.2, 0.6, 0.2),   # put, NEUTRAL argmax -> abstain (drop)
            "wide": (0.7, 0.2, 0.1),   # call, down argmax -> gated_out
        }
        v = rp.build_variants(self._trades(), None, preds, preds, "ok")
        acct = v["classical_gated"]["accounting"]
        kept = {t.decision_id for t in v["classical_gated"]["trades"]}
        # good passes; putx abstains (neutral is NOT a put signal); wide is
        # gated out; noquote: NO prediction -> passes through and is counted.
        assert kept == {"good", "noquote"}
        assert acct == {"n": 2, "gated_out": 1, "no_prediction": 1,
                        "gate_passed": 1, "abstained": 1}
        assert v["neural_gated"]["accounting"]["n"] == 2

    def test_combined_is_intersection(self):
        preds = {"good": (0.1, 0.2, 0.7), "putx": (0.1, 0.2, 0.7)}
        v = rp.build_variants(self._trades(), {"putx", "good"}, preds, {},
                              "unavailable")
        kept = {t.decision_id for t in v["combined"]["trades"]}
        assert kept == {"good"}   # putx fails the gate, others not selected


# --------------------------------------------------------------------------- #
# store fixture (real writer schema)
# --------------------------------------------------------------------------- #

def _occ(opt: str) -> str:
    return "SPY260220%s00500000" % ("C" if opt == "call" else "P")


def _write_fixture(path: str, n_per_session: int = 16, no_trade_sessions=()):
    """Separable fixture: label sign == sign(feature x).

    Calls get x>0 (up) and puts x<0 (down), so every traded option agrees
    with a perfect direction forecast.  One decision per session per index.
    ``no_trade_sessions`` get features/labels but only policy_skip outcomes
    (a genuine no-trade day that must stay on the session calendar).
    """
    feats, labels, outcomes = [], [], []
    sessions = SESSIONS + HOLDOUT + list(no_trade_sessions)
    skip_all = set(no_trade_sessions)
    for session in sessions:
        for i in range(n_per_session):
            did = f"d-{session}-{i}"
            as_of = _ts(session, i)
            opt = "call" if i % 2 == 0 else "put"
            x = 1.0 + 0.01 * i if opt == "call" else -1.0 - 0.01 * i
            feats.append(FeatureRow(
                decision_id=did, snapshot_id=f"s-{did}", as_of=as_of,
                available_at=as_of, underlying="SPY", symbol=_occ(opt),
                kind="candidate", session=session,
                features={"x": x, "noise": float(i % 3),
                          "spread_pct": 0.25 if i == 1 else 0.03},
                quality={"has_quote": i != 2}))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="direction",
                value=1.0 if x > 0 else -1.0, status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
            if i == 3 or session in skip_all:   # policy skip: no P&L
                outcomes.append(OutcomeRecord(
                    decision_id=did, status=OutcomeStatus.POLICY_SKIP,
                    gross_pnl_pct=None, net_pnl_dollars=None,
                    net_pnl_pct=None, reason="gate", fill_price=None,
                    fill_qty=None, outcome_at=end, meta={}))
                continue
            if i == 4:   # missing fill fields -> dropped_missing_fill
                outcomes.append(OutcomeRecord(
                    decision_id=did, status=OutcomeStatus.MARK_DERIVED,
                    gross_pnl_pct=5.0, net_pnl_dollars=None,
                    net_pnl_pct=None, reason="take_profit", fill_price=None,
                    fill_qty=None, outcome_at=end, meta={}))
                continue
            gross = 12.0 if i % 2 == 0 else -8.0
            reason = "take_profit" if gross > 0 else "dynamic_stop_loss"
            outcomes.append(OutcomeRecord(
                decision_id=did, status=OutcomeStatus.MARK_DERIVED,
                gross_pnl_pct=gross, net_pnl_dollars=None, net_pnl_pct=None,
                reason=reason, fill_price=1.0 + 0.1 * i, fill_qty=1.0,
                outcome_at=as_of + timedelta(minutes=45), meta={}))
    _write_store(path, {}, feats, labels, outcomes,
                 {"dataset_hash": "replayfixture", "schema_version": "1"})


def _write_valuation(path: str, selected_ids, extra_ghosts=0):
    scans = [{"decision": "trade", "selected": {"decision_id": d}}
             for d in selected_ids]
    scans += [{"decision": "trade",
               "selected": {"decision_id": f"ghost-{i}"}}
              for i in range(extra_ghosts)]
    scans += [{"decision": "abstain", "selected": None} for _ in range(3)]
    with open(path, "w", encoding="utf-8") as fh:
        json.dump({"report_fingerprint": "valfp",
                   "result": {"scans": scans}}, fh)


class TestLoaders:
    def test_load_traded_outcomes(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        trades, rep = rp.load_traded_outcomes(store, holdout=HOLDOUT)
        # per session: 16 decisions - 1 policy_skip - 1 missing-fill = 14
        assert rep["trades_kept"] == 14 * len(SESSIONS)
        # whole holdout session: 15 gross-non-null mark_derived rows (the
        # missing-fill row still has gross, and the holdout check runs first)
        assert rep["holdout_excluded"] == 15
        assert rep["dropped_missing_fill"] == len(SESSIONS)
        assert rep["dropped_bad_timestamp"] == 0
        assert rep["outcome_status_counts"] == {
            "mark_derived": 15 * len(SESSIONS),
            "policy_skip": len(SESSIONS)}
        assert len(trades) == rep["trades_kept"]
        assert all(t.session not in set(HOLDOUT) for t in trades)
        assert {t.option_type for t in trades} == {"call", "put"}
        # spread/quality parsed from the feature/quality json
        by_id = {t.decision_id: t for t in trades}
        assert by_id[f"d-{SESSIONS[0]}-1"].spread_pct == pytest.approx(0.25)
        assert by_id[f"d-{SESSIONS[0]}-2"].has_quote is False
        assert by_id[f"d-{SESSIONS[0]}-0"].premium == pytest.approx(100.0)
        # chronological, exit never before entry
        assert all(t.exit_at >= t.entry_at for t in trades)
        assert trades == sorted(trades, key=lambda t: (t.entry_at,
                                                       t.decision_id))

    def test_exit_before_entry_is_clamped(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        as_of = _ts(SESSIONS[0])
        _write_store(
            store, {},
            [FeatureRow(decision_id="d1", snapshot_id="s1", as_of=as_of,
                        available_at=as_of, underlying="SPY",
                        symbol=_occ("call"), kind="candidate",
                        session=SESSIONS[0], features={}, quality={})],
            [],
            [OutcomeRecord(decision_id="d1",
                           status=OutcomeStatus.MARK_DERIVED,
                           gross_pnl_pct=1.0, net_pnl_dollars=None,
                           net_pnl_pct=None, reason="eod", fill_price=1.0,
                           fill_qty=1.0,
                           outcome_at=as_of - timedelta(minutes=5), meta={})],
            {"dataset_hash": "h"})
        trades, _ = rp.load_traded_outcomes(store, holdout=HOLDOUT)
        assert trades[0].exit_at == trades[0].entry_at

    def test_load_valuation_selection(self, tmp_path):
        path = str(tmp_path / "val.json")
        _write_valuation(path, ["a", "b"], extra_ghosts=1)
        ids, rep = rp.load_valuation_selection(path)
        assert ids == {"a", "b", "ghost-0"}
        assert rep["trade_scans"] == 3
        assert rep["scans"] == 6
        assert rep["report_fingerprint"] == "valfp"

    def test_load_friction_defaults_when_missing(self, tmp_path):
        f = rp.load_friction(str(tmp_path / "nope.json"))
        assert f["one_way_frac"] == 0.02
        assert f["overshoot_extra_pts"] == 10.0
        assert "missing" in f["note"]
        assert rp.load_friction(None)["one_way_frac"] == 0.02

    def test_load_friction_from_calibration_artifact(self, tmp_path):
        path = str(tmp_path / "exec.json")
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({
                "fingerprint": "fp",
                "slippage": {"entry_vs_mid": {"quantiles": {"mean": 0.019}}},
                "stop_overshoot": {"dynamic_stop_loss": {
                    "overshoot_quantiles_pts": {"p50": 2.5, "p90": 13.8}}},
            }, fh)
        f = rp.load_friction(path)
        assert f["one_way_frac"] == pytest.approx(0.019)
        assert f["overshoot_extra_pts"] == pytest.approx(11.3)
        assert f["source_fingerprint"] == "fp"


# --------------------------------------------------------------------------- #
# out-of-fold classical gate (separable fixture)
# --------------------------------------------------------------------------- #

class TestOOFClassical:
    def test_oof_predictions_cover_only_test_sessions(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        preds, rep = rp.compute_oof_classical(store, holdout=HOLDOUT)
        assert rep["model"] == "multinomial_logistic"
        assert rep["classes"] == ["down", "neutral", "up"]
        assert rep["folds_used"] >= 1
        assert rep["predictions"] == len(preds)
        # holdout decisions never receive a prediction
        assert not any(d.startswith(f"d-{HOLDOUT[0]}") for d in preds)
        # min_train=2 => the first two sessions are never testable
        for s in SESSIONS[:2]:
            assert not any(d.startswith(f"d-{s}-") for d in preds)
        # separable labels: x>0 -> up, x<0 -> down; the OOF forecast must
        # agree and each (p_down, p_neutral, p_up) tuple must be a simplex
        for did, p in preds.items():
            assert len(p) == 3
            assert sum(p) == pytest.approx(1.0, abs=1e-6)
            i = int(did.rsplit("-", 1)[1])
            argmax = max(range(3), key=lambda k: p[k])
            assert argmax == (2 if i % 2 == 0 else 0), (did, p)


# --------------------------------------------------------------------------- #
# end-to-end replay (CLI-level, neural disabled for speed)
# --------------------------------------------------------------------------- #

class TestRunReplay:
    def test_end_to_end_parity_and_accounting(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        val = str(tmp_path / "val.json")
        out = str(tmp_path / "replay.json")
        _write_fixture(store)
        trades, _ = rp.load_traded_outcomes(store, holdout=HOLDOUT)
        picked = [t.decision_id for t in trades[:5]]
        _write_valuation(val, picked, extra_ghosts=2)

        art = rp.run_replay(
            dataset_path=store, valuation_path=val,
            execution_calibration_path=None, out_path=out,
            holdout=HOLDOUT, n_boot=50, include_neural=False)

        # parity: frictionless unconstrained engine == outcome store
        assert art["parity"]["ok"] is True
        assert art["parity"]["n_trades_engine"] == 14 * len(SESSIONS)
        assert art["parity"]["max_abs_diff"] < 1e-6

        # holdout refused at the decision level
        assert art["holdout_refusal"]["decisions_excluded"] == 15
        assert art["holdout_refusal"]["sessions"] == HOLDOUT

        # every variant present; neural honestly unavailable
        v = art["variants"]
        assert set(v) == {"baseline_frozen", "baseline_quality",
                          "valuation_ranked", "classical_gated",
                          "neural_gated", "combined"}
        assert v["neural_gated"]["status"] == "unavailable"
        assert v["valuation_ranked"]["accounting"][
            "outcome_unavailable"] == 2

        # friction sweep ordering: more friction, never more pnl
        base = v["baseline_frozen"]["by_friction_mult"]
        assert set(base) == {"1.0", "1.5", "2.0"}
        assert base["1.0"]["net_pnl"] >= base["1.5"]["net_pnl"] \
            >= base["2.0"]["net_pnl"]
        # default friction (missing calibration) is 2% one way
        assert art["inputs"]["execution_calibration"][
            "one_way_frac"] == 0.02

        # stress only degrades (fixture has dynamic_stop_loss exits)
        stress = v["baseline_frozen"]["stop_overshoot_stress"]
        assert stress["delta_vs_unstressed"] < 0

        # paired deltas exist for every available challenger and carry the
        # Stage F clustered-uncertainty fields (effective sample size)
        paired = art["paired_vs_baseline"]
        assert set(paired) == {"baseline_quality", "valuation_ranked",
                               "classical_gated", "combined"}
        for p in paired.values():
            assert p["bootstrap"]["n_sessions"] >= 1
            assert p["n_nonzero_delta_sessions"] >= 0
            assert p["ess_abs_delta"] >= 0.0

        # Stage F: session calendar covers every dataset session; this
        # fixture trades on all of them, so no_trade_sessions is empty
        cal = art["session_calendar"]
        assert cal["n_sessions"] == len(SESSIONS)
        assert cal["n_traded_sessions"] == len(SESSIONS)
        assert cal["no_trade_sessions"] == []

        # Stage F: input crosschecks ran before use; the fixture valuation
        # carries no dataset_hash and there is no calibration artifact, so
        # both provenance checks are honestly "unknown", never fabricated
        xc = art["input_crosschecks"]
        assert xc["ok"] is True
        st = {c["name"]: c["status"] for c in xc["checks"]}
        assert st["valuation_dataset_hash_vs_dataset"] == "unknown"
        assert st["execution_calibration_fingerprint_recorded"] == "unknown"

        # Stage F: matched prediction/quote-eligible cohort accounting.
        # min_train=2 -> the first two sessions never get predictions
        # (14 trades each); per predicted session the wide-spread (i==1)
        # and no-quote (i==2) trades are quote-ineligible.
        mc = art["matched_cohort"]
        assert mc["status"] == "ok"
        assert mc["excluded_no_prediction"] == 14 * 2
        assert mc["excluded_quote_ineligible"] == 2 * 3
        assert mc["n_cohort"] == 14 * len(SESSIONS) - 14 * 2 - 2 * 3
        gate = mc["challenger_gate"]
        assert sum(gate.values()) == mc["n_cohort"]
        assert gate["gated_out"] == 0   # separable fixture: never contradicted
        for key in ("baseline_on_cohort", "challenger_on_cohort"):
            assert set(mc[key]) == {"n_executed", "net_pnl", "profit_factor",
                                    "sharpe_daily_ann", "max_drawdown",
                                    "pnl_by_exit_day"}
        assert mc["paired"]["n_nonzero_delta_sessions"] >= 0
        assert mc["paired"]["ess_abs_delta"] >= 0.0

        # Stage F: risk metrics flagged limited (no daily marks, <30
        # sessions) and deterministic event ordering documented
        rml = art["risk_metrics_limited"]
        assert rml["limited"] is True
        assert any("short sample" in r for r in rml["reasons"])
        assert "same-timestamp cash reuse" in art["event_ordering_note"]

        # missing data accounted, never zeroed
        mda = art["missing_data_accounting"]
        assert mda["outcome_status_counts_nonholdout"]["policy_skip"] == \
            len(SESSIONS)
        assert mda["classical_no_prediction"] is not None
        assert "outcome_unavailable" in mda["counterfactual_note"] \
            or "never fabricated" in mda["counterfactual_note"]

        # artifact written and fingerprinted
        assert os.path.exists(out)
        with open(out, encoding="utf-8") as fh:
            on_disk = json.load(fh)
        assert on_disk["fingerprint"] == art["fingerprint"]
        assert on_disk["inputs"]["dataset_meta"]["dataset_hash"] == \
            "replayfixture"

    def test_deterministic_fingerprint(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        kw = dict(dataset_path=store, valuation_path=None,
                  execution_calibration_path=None, holdout=HOLDOUT,
                  n_boot=25, include_neural=False)
        a1 = rp.run_replay(**kw)
        a2 = rp.run_replay(**kw)
        assert a1["fingerprint"] == a2["fingerprint"]
        assert "valuation_ranked" not in a1["variants"]

    def test_cli_smoke(self, tmp_path, capsys):
        store = str(tmp_path / "ds.sqlite")
        out = str(tmp_path / "replay.json")
        _write_fixture(store)
        rc = rp.main(["--dataset", store, "--out", out,
                      "--holdout", *HOLDOUT, "--n-boot", "10",
                      "--no-neural"])
        assert rc == 0
        text = capsys.readouterr().out
        assert "parity ok=True" in text
        assert "fingerprint=" in text
        assert os.path.exists(out)


# --------------------------------------------------------------------------- #
# Stage F (Oracle 4.0.1): session calendar + input crosschecks
# --------------------------------------------------------------------------- #

class TestStageF:
    def test_load_session_calendar_excludes_holdout(self, tmp_path):
        store = str(tmp_path / "ds.sqlite")
        _write_fixture(store)
        assert rp.load_session_calendar(store, HOLDOUT) == SESSIONS

    def test_calendar_includes_no_trade_sessions(self, tmp_path):
        # a session with features but zero executed trades must stay on the
        # calendar and show up as a no-trade day in the replay artifact
        store = str(tmp_path / "ds.sqlite")
        extra = "2026-01-02"
        _write_fixture(store, no_trade_sessions=[extra])
        cal = rp.load_session_calendar(store, HOLDOUT)
        assert cal == sorted([extra] + SESSIONS)

        art = rp.run_replay(
            dataset_path=store, valuation_path=None,
            execution_calibration_path=None, holdout=HOLDOUT,
            n_boot=10, include_neural=False)
        sc = art["session_calendar"]
        assert sc["n_sessions"] == len(SESSIONS) + 1
        assert sc["n_traded_sessions"] == len(SESSIONS)
        assert sc["no_trade_sessions"] == [extra]
        assert extra in art["sessions_replayed"]
        # parity is unaffected by a zero-trade day
        assert art["parity"]["ok"] is True

    def test_crosscheck_match_unknown_and_mismatch(self):
        meta = {"dataset_hash": "abc"}
        ok = rp.crosscheck_inputs(meta, {"dataset_hash": "abc"}, {})
        assert ok["ok"] is True
        st = {c["name"]: c["status"] for c in ok["checks"]}
        assert st["valuation_dataset_hash_vs_dataset"] == "match"
        # artifact predating provenance fields -> unknown, never guessed
        unk = rp.crosscheck_inputs(meta, {"dataset_hash": None}, {})
        st = {c["name"]: c["status"] for c in unk["checks"]}
        assert st["valuation_dataset_hash_vs_dataset"] == "unknown"
        # no valuation at all -> the check is simply absent
        none = rp.crosscheck_inputs(meta, None, {})
        assert "valuation_dataset_hash_vs_dataset" not in {
            c["name"] for c in none["checks"]}
        # positive mismatch refuses, loudly
        with pytest.raises(ValueError, match="refusing to mix artifacts"):
            rp.crosscheck_inputs(meta, {"dataset_hash": "WRONG"}, {})

    def test_crosscheck_records_friction_fingerprint(self):
        out = rp.crosscheck_inputs(
            {"dataset_hash": "abc"}, None, {"source_fingerprint": "fp"})
        st = {c["name"]: c["status"] for c in out["checks"]}
        assert st["execution_calibration_fingerprint_recorded"] == "recorded"

    def test_run_replay_refuses_mismatched_valuation(self, tmp_path):
        # end-to-end: a valuation selection computed on a DIFFERENT dataset
        # must be refused before any replay computation, not silently merged
        store = str(tmp_path / "ds.sqlite")
        val = str(tmp_path / "val.json")
        _write_fixture(store)
        with open(val, "w", encoding="utf-8") as fh:
            json.dump({"report_fingerprint": "valfp",
                       "dataset_hash": "WRONG-DATASET",
                       "result": {"scans": []}}, fh)
        with pytest.raises(ValueError, match="refusing to mix artifacts"):
            rp.run_replay(dataset_path=store, valuation_path=val,
                          execution_calibration_path=None,
                          holdout=HOLDOUT, n_boot=10, include_neural=False)
