"""Phase 5 tests: scenario valuation + all-candidate ranking (oracle/ml/valuation.py).

Covers the master-prompt acceptance list:
  * hand-computed payoff fixtures (stop / target / time exit / expiry settle)
  * monotonic fee/spread friction stress
  * contract multiplier and partial fills
  * expired / unusable / unparseable candidate abstention
  * same-side contract selection where the second-best legacy score has the
    higher executable EV and the challenger picks it
  * negative / unavailable EV can never beat abstention
  * CLI end-to-end on a fixture store with holdout-leak refusal

All fixtures are synthetic; nothing here is evidence of profitability.
"""
from __future__ import annotations

import json
import math
import os
import sys
from datetime import datetime, timedelta, timezone
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cost_model import CostConfig  # noqa: E402
from oracle.ml import valuation as val  # noqa: E402
from oracle.ml.dataset import _write_store  # noqa: E402
from oracle.ml.schema import (  # noqa: E402
    FeatureRow, LabelRow, LabelStatus, UnderlyingObs)
from pricing_diagnostics import bs_price  # noqa: E402

UTC = timezone.utc
SESSIONS = [f"2026-11-{d:02d}" for d in (2, 3, 4, 5, 6, 9, 10, 11)]

# CostConfig defaults the hand computations below rely on.
_CC = CostConfig()
assert _CC.slippage_per_contract == 0.02
assert _CC.occ_fee_per_contract == 0.02
assert _CC.commission_per_contract == 0.0
assert _CC.min_spread_floor == 0.01


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _cand(did="d1", *, symbol="SPY261218C00600000", underlying="SPY",
          session="2026-11-10", minute=0, **feat):
    """Candidate with a 1.90/2.10 quote and sane contract defaults.

    Feature overrides may be None to simulate missing fields.
    """
    features = {"quote_mid": 2.0, "spread_abs": 0.2, "dte_days": 29.0,
                "underlying_price": 600.0, "baseline_score": None}
    features.update(feat)
    return val.Candidate(
        decision_id=did, symbol=symbol, underlying=underlying,
        session=session, as_of=_ts(session, minute), kind="candidate",
        features=features, contract=val.parse_occ_symbol(symbol))


def _priced(marks, *, bid=1.9, ask=2.1, sigma=0.3, gap=0.5,
            expires_in_sim=False, n_steps=None):
    m = np.asarray(marks, dtype=np.float64)
    return {"marks": m, "bid": bid, "ask": ask, "mid": (bid + ask) / 2.0,
            "sigma": sigma, "american_gap_pct": gap,
            "expires_in_sim": expires_in_sim,
            "n_steps": n_steps if n_steps is not None else m.shape[1] - 1}


class _StubFill:
    """Deterministic FillModel stand-in for partial/no-fill paths."""

    def __init__(self, p, partial, status="filled", reasons=()):
        self.p, self.partial = p, partial
        self.status, self.reasons = status, tuple(reasons)

    def estimate_fill(self, order, quote, **kw):
        return SimpleNamespace(fill_probability=self.p,
                               partial_qty=self.partial,
                               status=self.status, reasons=self.reasons)


def _vrow(did, ev, *, mc_se=0.0, es95=-10.0, pop=0.5, capital=212.0,
          bid=1.9, ask=2.1, symbol=None, score=None,
          status="ok", reason=None):
    """Synthetic valuation dict shaped for rank_candidates."""
    mid = (bid + ask) / 2.0
    return {"decision_id": did, "symbol": symbol or f"SYM_{did}",
            "status": status, "reason": reason,
            "ev_opportunity": ev,
            "ev_per_dollar": (ev / capital) if capital else None,
            "capital_required": capital,
            "per_fill": {"mc_se": mc_se, "es95": es95, "pop": pop},
            "entry": {"bid": bid, "ask": ask, "mid": mid},
            "baseline_score": score}


def _obs_list(day="2026-11-10", n=200, base=600.0):
    """1-min-spaced underlying obs from 14:00 UTC (39 5-min grid returns)."""
    t0 = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(hours=14)
    return [(t0 + timedelta(minutes=i),
             base * (1.0 + 0.001 * math.sin(i * 0.7))) for i in range(n)]


AS_OF = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)  # after _obs_list end
_FLAT = val.ScenarioSet(prices=np.full((8, 7), 600.0), step_minutes=5.0)


# --------------------------------------------------------------------------- #
# A. OCC symbol parsing + vectorized BSM parity
# --------------------------------------------------------------------------- #

class TestOccAndBsm:
    def test_parse_call(self):
        c = val.parse_occ_symbol("SPY261218C00600000")
        assert c == {"underlying": "SPY", "expiration": "2026-12-18",
                     "option_type": "call", "strike": 600.0}

    def test_parse_put_fractional_strike(self):
        c = val.parse_occ_symbol("QQQ270115P00450500")
        assert c == {"underlying": "QQQ", "expiration": "2027-01-15",
                     "option_type": "put", "strike": 450.5}

    def test_parse_lowercase_and_whitespace(self):
        assert val.parse_occ_symbol(" spy261218c00600000 ") is not None

    @pytest.mark.parametrize("bad", [
        None, "", "JUNK", "SPY", "SPY261218X00600000",
        "SPY269918C00600000",      # month 99
        "SPY261218C00000000",      # zero strike
        "TOOLONGG261218C00600000",  # 8-char root
    ])
    def test_parse_rejects(self, bad):
        assert val.parse_occ_symbol(bad) is None

    def test_bs_vec_matches_scalar_including_degenerate(self):
        spots = np.array([80.0, 100.0, 120.0])
        for opt in ("call", "put"):
            for T in (0.25, 0.0):
                for sigma in (0.3, 0.0):
                    got = val._bs_price_vec(spots, 100.0, T, 0.04, 0.01,
                                            sigma, opt)
                    for s, g in zip(spots, got):
                        want = bs_price(float(s), 100.0, T, 0.04, 0.01,
                                        sigma, opt)
                        assert abs(float(g) - want) < 1e-9, (opt, T, sigma, s)


# --------------------------------------------------------------------------- #
# B. scenario generation
# --------------------------------------------------------------------------- #

class TestScenarios:
    CFG = val.ValuationConfig(n_paths=256, max_steps=12)

    def test_deterministic_same_seed(self):
        obs = _obs_list()
        a, ra = val.generate_scenarios(obs, AS_OF, 600.0, self.CFG, 123)
        b, rb = val.generate_scenarios(obs, AS_OF, 600.0, self.CFG, 123)
        assert ra == rb == "ok"
        assert np.array_equal(a.prices, b.prices)

    def test_different_seed_differs(self):
        obs = _obs_list()
        a, _ = val.generate_scenarios(obs, AS_OF, 600.0, self.CFG, 123)
        b, _ = val.generate_scenarios(obs, AS_OF, 600.0, self.CFG, 124)
        assert not np.array_equal(a.prices, b.prices)

    def test_past_only_future_obs_ignored(self):
        obs = _obs_list()
        a, _ = val.generate_scenarios(obs, AS_OF, 600.0, self.CFG, 7)
        tainted = obs + [(AS_OF + timedelta(minutes=30), 10000.0)]
        b, _ = val.generate_scenarios(tainted, AS_OF, 600.0, self.CFG, 7)
        assert np.array_equal(a.prices, b.prices)

    def test_insufficient_history_abstains(self):
        scen, why = val.generate_scenarios(_obs_list()[:10], AS_OF, 600.0,
                                           self.CFG, 7)
        assert scen is None and why == "insufficient_history"

    def test_missing_underlying_price(self):
        scen, why = val.generate_scenarios(_obs_list(), AS_OF, 0.0,
                                           self.CFG, 7)
        assert scen is None and why == "missing_underlying_price"

    def test_shape_anchor_and_positivity(self):
        scen, why = val.generate_scenarios(_obs_list(), AS_OF, 600.0,
                                           self.CFG, 7)
        assert why == "ok"
        assert scen.prices.shape == (256, 13)       # clamped paths, steps+1
        assert np.all(scen.prices[:, 0] == 600.0)
        assert np.all(scen.prices > 0)
        assert scen.n_steps == 12
        assert scen.meta["n_source_returns"] >= 30
        assert scen.meta["rv_forecast"] in ("ewma", "unavailable_scale_1")
        assert scen.meta["seed"] == 7

    def test_path_clamping(self):
        assert val.ValuationConfig(n_paths=10).clamped_paths() == 256
        assert val.ValuationConfig(n_paths=9999).clamped_paths() == 2048
        assert val.ValuationConfig(n_paths=512).clamped_paths() == 512

    def test_scenario_seed_stable_and_distinct(self):
        s1 = val.scenario_seed(42, "SPY", "12345")
        assert s1 == val.scenario_seed(42, "SPY", "12345")
        assert s1 != val.scenario_seed(42, "QQQ", "12345")
        assert s1 != val.scenario_seed(43, "SPY", "12345")
        assert 0 <= s1 < 2 ** 63 - 1


# --------------------------------------------------------------------------- #
# C. repricing + abstention reasons
# --------------------------------------------------------------------------- #

class TestReprice:
    CFG = val.ValuationConfig()

    def _abstain_reason(self, cand):
        priced, abst = val.reprice_candidate(cand, _FLAT, self.CFG)
        assert priced is None and abst is not None
        assert abst["status"] == "abstain"
        # 4.0.1-C EV taxonomy: confirmed nonfill is exactly 0; anything
        # unknown stays null and is labeled "unavailable" (never conflated).
        if abst["ev_basis"] == "confirmed_nonfill":
            assert abst["ev_opportunity"] == 0.0
        else:
            assert abst["ev_basis"] == "unavailable"
            assert abst["ev_opportunity"] is None
        return abst["reason"]

    def test_missing_quote(self):
        assert self._abstain_reason(_cand(quote_mid=None)) == "missing_quote"

    def test_one_sided_quote(self):
        # mid 0.05, spread 0.2 -> reconstructed bid is negative
        assert (self._abstain_reason(_cand(quote_mid=0.05))
                == "unusable_quote:one_sided_quote")

    def test_unparseable_symbol(self):
        assert (self._abstain_reason(_cand(symbol="JUNK", quote_mid=15.0))
                == "unparseable_symbol")

    def test_underlying_mismatch(self):
        # Contract root NVDA stored under an AMD scan: the shared scenario
        # paths are AMD prices, so repricing must abstain.
        assert (self._abstain_reason(
            _cand(symbol="NVDA261218C00600000", underlying="AMD",
                  quote_mid=15.0))
            == "underlying_mismatch")

    def test_missing_dte(self):
        assert (self._abstain_reason(_cand(quote_mid=15.0, dte_days=None))
                == "missing_dte")

    def test_expired(self):
        assert (self._abstain_reason(_cand(quote_mid=15.0, dte_days=0.0))
                == "expired")

    def test_missing_underlying_price(self):
        assert (self._abstain_reason(
            _cand(quote_mid=15.0, underlying_price=None))
            == "missing_underlying_price")

    def test_iv_below_intrinsic(self):
        # 700 put on a 600 spot quoted at 50 < intrinsic floor (~98)
        assert (self._abstain_reason(
            _cand(symbol="SPY261218P00700000", quote_mid=50.0))
            == "iv_unresolved:below_intrinsic")

    def test_happy_path_atm_call(self):
        priced, abst = val.reprice_candidate(_cand(quote_mid=15.0),
                                             _FLAT, self.CFG)
        assert abst is None
        assert priced["marks"].shape == (8, 7)
        assert np.all(priced["marks"][:, 0] == 15.0)
        assert 0.15 < priced["sigma"] < 0.30       # ~0.208 ATM 29d
        assert priced["n_steps"] == 6
        assert priced["expires_in_sim"] is False
        assert priced["american_gap_pct"] is not None

    def test_expiry_in_sim_settles_at_intrinsic(self):
        # 28.8 DTE-minutes -> 5 steps < 6 -> expires; ATM call settles at 0
        priced, abst = val.reprice_candidate(
            _cand(quote_mid=1.0, dte_days=0.02), _FLAT, self.CFG)
        assert abst is None
        assert priced["expires_in_sim"] is True
        assert priced["n_steps"] == 5
        assert priced["marks"].shape == (8, 6)
        assert np.all(priced["marks"][:, -1] == 0.0)


# --------------------------------------------------------------------------- #
# D. hand-computed cost + exit valuation
# --------------------------------------------------------------------------- #
# Entry quote 1.90/2.10: mid 2.00, spread 0.20, half 0.10, slip 0.02
# -> entry_fill 2.12, stop 1.908, target 2.544, capital 212, p_fill 0.99.

class TestValueCandidate:
    CFG = val.ValuationConfig()

    def test_take_profit_hand_computed(self):
        v = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), self.CFG)
        assert v["status"] == "ok"
        assert v["entry"]["fill_price"] == pytest.approx(2.12)
        # exit fill 3.0 - 0.12 = 2.88; net (2.88-2.12)*100 - 0.04 = 75.96
        assert v["per_fill"]["net_mean"] == pytest.approx(75.96)
        assert v["per_fill"]["pop"] == 1.0
        assert v["per_fill"]["exit_reasons"] == {
            "stop_loss": 0, "take_profit": 1, "time_exit": 0}
        assert v["costs"]["spread_entry"] == pytest.approx(10.0)
        assert v["costs"]["spread_exit_mean"] == pytest.approx(10.0)
        assert v["costs"]["slippage_round_trip_mean"] == pytest.approx(4.0)
        assert v["costs"]["fees_mean"] == pytest.approx(0.04)
        assert v["costs"]["exit_spread_source"] == "entry_spread_proxy"
        assert v["fill"]["p_fill"] == pytest.approx(0.99)
        assert v["fill"]["fill_frac"] == 1.0
        assert v["capital_required"] == pytest.approx(212.0)
        assert v["ev_opportunity"] == pytest.approx(0.99 * 75.96)
        assert v["ev_per_dollar"] == pytest.approx(0.99 * 75.96 / 212.0)
        assert v["n_paths"] == 1
        assert v["per_fill"]["mc_se"] == 0.0
        assert len(v["approximations"]) == 5

    def test_stop_loss_hand_computed(self):
        v = val.value_candidate(_cand(), _priced([[2.0, 1.5]]), self.CFG)
        # exit fill 1.5 - 0.12 = 1.38; net (1.38-2.12)*100 - 0.04 = -74.04
        assert v["per_fill"]["net_mean"] == pytest.approx(-74.04)
        assert v["per_fill"]["pop"] == 0.0
        assert v["per_fill"]["exit_reasons"]["stop_loss"] == 1

    def test_time_exit_hand_computed(self):
        v = val.value_candidate(_cand(), _priced([[2.0, 2.05]]), self.CFG)
        # exit fill 2.05 - 0.12 = 1.93; net (1.93-2.12)*100 - 0.04 = -19.04
        assert v["per_fill"]["net_mean"] == pytest.approx(-19.04)
        assert v["per_fill"]["exit_reasons"] == {
            "stop_loss": 0, "take_profit": 0, "time_exit": 1}
        assert v["status"] == "ok"      # negative EV is reported, not hidden

    def test_expiry_liquidation_charges_frictions(self):
        # 4.0.1-C: physically settled contracts are never cash-settled for
        # free — the documented research convention liquidates at the last
        # pre-expiry mark with FULL exit frictions and both fee legs.
        v = val.value_candidate(
            _cand(), _priced([[2.0, 2.2]], expires_in_sim=True), self.CFG)
        # exit fill 2.2 - 0.12 = 2.08; net (2.08-2.12)*100 - 0.04 = -4.04
        assert v["per_fill"]["net_mean"] == pytest.approx(-4.04)
        assert v["per_fill"]["exit_reasons"] == {
            "stop_loss": 0, "take_profit": 0, "pre_expiry_liquidation": 1}
        assert v["costs"]["spread_exit_mean"] == pytest.approx(10.0)
        assert v["costs"]["slippage_round_trip_mean"] == pytest.approx(4.0)
        assert v["costs"]["fees_mean"] == pytest.approx(0.04)
        assert v["settlement"]["convention"] == \
            "pre_expiry_liquidation_research"
        assert v["settlement"]["n_liquidated"] == 1

    def test_friction_stress_monotone_hand_computed(self):
        pr = _priced([[2.0, 2.05]])
        nets = [val.value_candidate(_cand(), pr, self.CFG,
                                    friction=f)["per_fill"]["net_mean"]
                for f in (1.0, 1.5, 2.0)]
        assert nets[0] == pytest.approx(-19.04)
        assert nets[1] == pytest.approx(-31.06)   # entry 2.18, exit 1.87
        assert nets[2] == pytest.approx(-43.08)   # entry 2.24, exit 1.81
        assert nets[0] > nets[1] > nets[2]

    def test_contract_multiplier_and_qty_scaling(self):
        cfg2 = val.ValuationConfig(qty=2)
        v = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), cfg2)
        assert v["per_fill"]["net_mean"] == pytest.approx(2 * 75.96)
        assert v["capital_required"] == pytest.approx(424.0)
        assert v["fill"]["fill_frac"] == 1.0
        assert v["ev_opportunity"] == pytest.approx(0.99 * 151.92)

    def test_wide_quote_haircuts_fill_probability(self):
        cand = _cand(quote_mid=1.0)                 # spread/mid = 0.2 > 0.15
        v = val.value_candidate(cand, _priced([[1.0, 1.6]], bid=0.9, ask=1.1),
                                self.CFG)
        # entry 1.12, target 1.344 -> tp at 1.6; exit 1.48; net 35.96
        assert v["per_fill"]["net_mean"] == pytest.approx(35.96)
        assert v["fill"]["p_fill"] == pytest.approx(0.99 * 0.85)
        assert v["ev_opportunity"] == pytest.approx(0.8415 * 35.96)

    def test_partial_fill_scales_opportunity_ev(self):
        stub = _StubFill(p=0.5, partial=0.5, status="partially_filled")
        v = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), self.CFG,
                                fill_model=stub)
        assert v["fill"]["p_fill"] == 0.5
        assert v["fill"]["fill_frac"] == 0.5
        assert v["ev_opportunity"] == pytest.approx(0.5 * 0.5 * 75.96)

    def test_known_nonfill_abstains_with_zero_ev(self):
        stub = _StubFill(p=0.0, partial=0.0, status="pending",
                         reasons=("no_quote",))
        v = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), self.CFG,
                                fill_model=stub)
        assert v["status"] == "abstain"
        assert v["reason"] == "no_fill_expected"
        assert v["ev_opportunity"] == 0.0
        assert v["fill_status"] == "pending"
        assert "no_quote" in v["fill_reasons"]

    def test_nonpositive_entry_fill_abstains(self):
        v = val.value_candidate(
            _cand(), _priced([[-1.0, -1.0]], bid=-1.1, ask=-0.9), self.CFG)
        assert v["status"] == "abstain"
        assert v["reason"] == "nonpositive_entry_fill"

    def test_mc_se_and_es95_on_mixed_paths(self):
        marks = np.array([[2.0, 3.0], [2.0, 1.5], [2.0, 2.05]])
        v = val.value_candidate(_cand(), _priced(marks), self.CFG)
        nets = np.array([75.96, -74.04, -19.04])
        assert v["per_fill"]["net_mean"] == pytest.approx(nets.mean())
        assert v["per_fill"]["mc_se"] == pytest.approx(
            nets.std(ddof=1) / math.sqrt(3))
        assert v["per_fill"]["es95"] == pytest.approx(-74.04)  # worst 5%
        assert v["per_fill"]["pop"] == pytest.approx(1.0 / 3.0)


# --------------------------------------------------------------------------- #
# E. exit-policy precedence
# --------------------------------------------------------------------------- #

class TestSimulateExit:
    def test_first_touch_wins(self):
        step, mark, reasons = val._simulate_exit(
            np.array([[2.0, 2.6, 1.5]]), 2.12, 10.0, 20.0, False)
        assert step[0] == 1 and mark[0] == 2.6
        assert reasons == {"stop_loss": 0, "take_profit": 1, "time_exit": 0}

    def test_stop_precedence_at_touch(self):
        step, mark, reasons = val._simulate_exit(
            np.array([[2.0, 1.9, 3.0]]), 2.12, 10.0, 20.0, False)
        assert step[0] == 1 and mark[0] == 1.9     # 1.9 <= 1.908 stop
        assert reasons["stop_loss"] == 1

    def test_time_exit_at_horizon(self):
        step, mark, reasons = val._simulate_exit(
            np.array([[2.0, 2.0, 2.0]]), 2.12, 10.0, 20.0, False)
        assert step[0] == 2 and mark[0] == 2.0
        assert reasons["time_exit"] == 1

    def test_expiry_liquidates_before_settlement(self):
        # 2-column path (1 step): degenerate case still exits at the last
        # available mark, labeled as the research liquidation convention.
        step, mark, reasons = val._simulate_exit(
            np.array([[2.0, 2.0]]), 2.12, 10.0, 20.0, True)
        assert step[0] == 1 and mark[0] == 2.0
        assert reasons == {"stop_loss": 0, "take_profit": 0,
                           "pre_expiry_liquidation": 1}

    def test_expiry_liquidation_column_precedes_intrinsic(self):
        # 4 columns, 3 steps, no trigger: exit at column 2 (last pre-expiry
        # mark), NOT at the intrinsic settlement column 3.
        step, mark, reasons = val._simulate_exit(
            np.array([[2.0, 2.1, 2.2, 0.0]]), 2.12, 10.0, 20.0, True)
        assert step[0] == 2 and mark[0] == 2.2
        assert reasons["pre_expiry_liquidation"] == 1


# --------------------------------------------------------------------------- #
# F. ranking gates + abstention
# --------------------------------------------------------------------------- #

class TestRanking:
    def _reasons(self, rank):
        return {e["decision_id"]: e["reason"] for e in rank["excluded"]}

    def test_negative_or_zero_ev_never_beats_abstention(self):
        rank = val.rank_candidates([_vrow("a", 0.0), _vrow("b", -5.0)])
        assert rank["decision"] == "abstain"
        assert rank["selected_decision_id"] is None
        assert rank["selected_ev"] == 0.0
        assert rank["abstention_ev"] == 0.0
        assert self._reasons(rank) == {"a": "ev_not_positive",
                                       "b": "ev_not_positive"}
        assert rank["n_eligible"] == 0 and rank["n_valued"] == 2

    def test_uncertain_ev_gate(self):
        rank = val.rank_candidates([_vrow("a", 1.0, mc_se=5.0)])
        assert self._reasons(rank) == {"a": "uncertain_ev"}

    def test_wide_spread_gate(self):
        rank = val.rank_candidates([_vrow("a", 10.0, bid=1.0, ask=2.0)])
        assert self._reasons(rank) == {"a": "wide_spread"}  # 66% > 40%

    def test_excess_downside_gate(self):
        rank = val.rank_candidates([_vrow("a", 10.0, es95=-500.0)])
        assert self._reasons(rank) == {"a": "excess_downside"}

    def test_abstained_valuation_passes_through_reason(self):
        rank = val.rank_candidates([
            _vrow("a", 0.0, status="abstain", reason="missing_quote")])
        assert self._reasons(rank) == {"a": "missing_quote"}

    def test_rank_order_and_tiebreaks(self):
        rows = [
            _vrow("d", 5.0, symbol="BBB"),
            _vrow("c", 5.0, symbol="AAA"),             # symbol breaks tie
            _vrow("b", 10.0, capital=400.0, symbol="ZZ"),
            _vrow("a", 10.0, capital=100.0, symbol="ZZ"),  # higher ev/$
        ]
        rank = val.rank_candidates(rows)
        assert rank["decision"] == "trade"
        assert [r["decision_id"] for r in rank["ranking"]] == \
            ["a", "b", "c", "d"]
        assert rank["selected_decision_id"] == "a"
        assert rank["selected_ev"] == 10.0

    def test_decision_id_final_tiebreak(self):
        rows = [_vrow("b", 5.0, symbol="S"), _vrow("a", 5.0, symbol="S")]
        rank = val.rank_candidates(rows)
        assert [r["decision_id"] for r in rank["ranking"]] == ["a", "b"]

    def test_baseline_order_includes_abstained_scored_rows(self):
        rows = [
            _vrow("ok1", 5.0, score=1.0),
            _vrow("ab1", 0.0, status="abstain", reason="missing_quote",
                  score=99.0),
        ]
        rank = val.rank_candidates(rows)
        assert rank["selected_decision_id"] == "ok1"
        assert rank["baseline_top_decision_id"] == "ab1"
        assert rank["baseline_order"] == ["ab1", "ok1"]

    def test_master_prompt_key_fixture_ev_beats_legacy_score(self):
        """Second-best legacy score has higher executable EV -> chosen."""
        cfg = val.ValuationConfig()
        cand_a = _cand("dA", symbol="SPY261218C00600000",
                       baseline_score=100.0)
        cand_b = _cand("dB", symbol="SPY261218C00605000",
                       baseline_score=10.0)
        marks_a = np.full((300, 2), 2.0)
        marks_a[:, 1] = 2.6                     # net 35.96 per path
        marks_b = np.full((300, 2), 2.0)
        marks_b[:, 1] = 3.0                     # net 75.96 per path
        v_a = val.value_candidate(cand_a, _priced(marks_a), cfg)
        v_b = val.value_candidate(cand_b, _priced(marks_b), cfg)
        assert v_a["ev_opportunity"] == pytest.approx(0.99 * 35.96)
        assert v_b["ev_opportunity"] == pytest.approx(0.99 * 75.96)
        assert v_a["per_fill"]["mc_se"] == 0.0  # constant paths

        rank = val.rank_candidates([v_a, v_b], cfg)
        assert rank["decision"] == "trade"
        assert rank["selected_decision_id"] == "dB"
        assert rank["baseline_top_decision_id"] == "dA"
        assert [r["decision_id"] for r in rank["ranking"]] == ["dB", "dA"]
        assert rank["n_eligible"] == 2


# --------------------------------------------------------------------------- #
# G. end-to-end run_valuation on a fixture store
# --------------------------------------------------------------------------- #

def _store_feature_row(did, session, minute, features, *, symbol=None,
                       underlying="SPY"):
    as_of = _ts(session, minute)
    return FeatureRow(decision_id=did, snapshot_id=f"s-{did}", as_of=as_of,
                      available_at=as_of, underlying=underlying,
                      symbol=symbol, kind="candidate", session=session,
                      features=features, quality={})


def _spy_obs_rows(day):
    return [UnderlyingObs(underlying="SPY", event_at=t, available_at=t,
                          price=p, source_snapshot_id="obs")
            for t, p in _obs_list(day)]


class TestRunValuation:
    def _store(self, tmp_path):
        path = str(tmp_path / "store.sqlite")
        feats = [
            _store_feature_row("c1", "2026-11-10", 240, {
                "quote_mid": 15.0, "spread_abs": 0.2, "dte_days": 29.0,
                "underlying_price": 600.0, "baseline_score": 10.0,
            }, symbol="SPY261218C00600000"),
            _store_feature_row("c2", "2026-11-10", 240, {
                "quote_mid": 12.0, "spread_abs": 0.2, "dte_days": 29.0,
                "underlying_price": 600.0, "baseline_score": 9.0,
            }, symbol="SPY261218C00605000"),
            _store_feature_row("c3", "2026-11-10", 240, {
                "dte_days": 29.0, "underlying_price": 600.0,
                "baseline_score": 8.0,
            }, symbol="SPY261218C00610000"),      # no quote -> abstain
        ]
        _write_store(path, {"SPY": _spy_obs_rows("2026-11-10")}, feats, [],
                     [], {"dataset_hash": "fixturehash",
                          "schema_version": "1"})
        return path

    def test_loaders(self, tmp_path):
        store = self._store(tmp_path)
        cands = val.load_candidates(store)
        assert [c.decision_id for c in cands] == ["c1", "c2", "c3"]
        assert val.load_candidates(store, sessions=["2026-11-02"]) == []
        obs = val.load_underlying_obs(store)
        assert len(obs["SPY"]) == 200
        assert obs["SPY"] == sorted(obs["SPY"])

    def test_end_to_end_counts_stress_and_determinism(self, tmp_path):
        store = self._store(tmp_path)
        cfg = val.ValuationConfig(n_paths=256, max_steps=12)
        res = val.run_valuation(store, ["2026-11-10"], cfg, seed=7)

        assert res["counts"] == {
            "candidates": 3, "scans": 1, "valued_ok": 2, "abstained": 1,
            "trade_decisions": res["counts"]["trade_decisions"],
            "abstain_decisions": 1 - res["counts"]["trade_decisions"],
        }
        assert res["abstain_reasons"] == {"missing_quote": 1}
        assert res["sessions_valued"] == ["2026-11-10"]
        assert res["ev_summary"]["n"] == 2
        assert res["pop_summary"]["n"] == 2

        stress = res["friction_stress"]
        assert sorted(stress) == ["1.0", "1.5", "2.0"]
        means = [stress[k]["net_mean"]["mean"] for k in ("1.0", "1.5", "2.0")]
        assert all(m is not None for m in means)
        assert means[0] >= means[1] >= means[2]
        assert res["friction_stress_monotone_decreasing"] is True

        scan = res["scans"][0]
        assert scan["n_candidates"] == 3
        assert scan["scenario"]["seed"] == val.scenario_seed(
            7, "SPY", str(scan["bucket"]))
        assert scan["baseline_top_decision_id"] == "c1"   # score 10 top
        assert res["latency"]["s_per_scan"] is not None

        # determinism: same store + seed -> same summary numbers
        res2 = val.run_valuation(store, ["2026-11-10"], cfg, seed=7)
        assert res2["counts"] == res["counts"]
        assert res2["ev_summary"] == res["ev_summary"]
        assert res2["scans"][0]["selected"] == res["scans"][0]["selected"]

    def test_no_obs_underlying_abstains_whole_scan(self, tmp_path):
        path = str(tmp_path / "noobs.sqlite")
        feats = [_store_feature_row("c1", "2026-11-10", 240, {
            "quote_mid": 15.0, "spread_abs": 0.2, "dte_days": 29.0,
            "underlying_price": 600.0}, symbol="SPY261218C00600000")]
        _write_store(path, {}, feats, [], [],
                     {"dataset_hash": "fixturehash", "schema_version": "1"})
        res = val.run_valuation(path, ["2026-11-10"],
                                val.ValuationConfig(n_paths=256,
                                                    max_steps=6), seed=7)
        assert res["counts"]["valued_ok"] == 0
        assert res["abstain_reasons"] == {"insufficient_history": 1}


# --------------------------------------------------------------------------- #
# H. CLI: fold-only sessions, holdout refusal, reproducible report
# --------------------------------------------------------------------------- #

def _cli_store(tmp_path):
    """8-session store: direction labels for fold derivation + 2 real
    candidates (with SPY obs) on fold session 2026-11-04."""
    path = str(tmp_path / "cli.sqlite")
    feats, labels = [], []
    for session in SESSIONS:
        for i in range(3):
            did = f"l-{session}-{i}"
            as_of = _ts(session, i)
            feats.append(_store_feature_row(did, session, i, {"x": 1.0}))
            end = as_of + timedelta(minutes=60)
            labels.append(LabelRow(
                decision_id=did, horizon="60m", label_type="direction",
                value=(1.0 if i % 2 else -1.0), status=LabelStatus.OK,
                label_end=end, available_at=end, meta={}))
    feats.append(_store_feature_row("v1", "2026-11-04", 240, {
        "quote_mid": 15.0, "spread_abs": 0.2, "dte_days": 29.0,
        "underlying_price": 600.0, "baseline_score": 5.0,
    }, symbol="SPY261218C00600000"))
    feats.append(_store_feature_row("v2", "2026-11-04", 240, {
        "quote_mid": 12.0, "spread_abs": 0.2, "dte_days": 29.0,
        "underlying_price": 600.0, "baseline_score": 4.0,
    }, symbol="SPY261218C00605000"))
    _write_store(path, {"SPY": _spy_obs_rows("2026-11-04")}, feats, labels,
                 [], {"dataset_hash": "fixturehash", "schema_version": "1"})
    return path


class TestCli:
    def test_cli_report(self, tmp_path, capsys):
        store = _cli_store(tmp_path)
        out = str(tmp_path / "valuation.json")
        rc = val.main(["--dataset", store, "--out", out, "--seed", "7",
                       "--n-paths", "256", "--max-steps", "6"])
        assert rc == 0
        with open(out, encoding="utf-8") as fh:
            rep = json.load(fh)
        assert rep["phase"] == "oracle4_phase5_valuation"
        assert rep["dataset_hash"] == "fixturehash"
        assert rep["seed"] == 7
        assert isinstance(rep["report_fingerprint"], str)
        assert rep["report_fingerprint"]
        assert rep["sessions"]["holdout"] == SESSIONS[-2:]
        assert not set(rep["sessions"]["requested"]) & \
            set(rep["sessions"]["holdout"])
        c = rep["result"]["counts"]
        # 6 fold sessions x 3 label-only rows + 2 real candidates
        assert c["candidates"] == 20
        assert c["valued_ok"] == 2
        assert c["abstained"] == 18
        assert rep["result"]["abstain_reasons"] == {
            "missing_underlying_price": 18}
        assert rep["cost_config"]["slippage_per_contract"] == 0.02
        captured = capsys.readouterr().out
        assert "fingerprint=" in captured

    def test_cli_refuses_holdout_sessions(self, tmp_path):
        store = _cli_store(tmp_path)
        out = str(tmp_path / "leak.json")
        with pytest.raises(SystemExit) as ei:
            val.main(["--dataset", store, "--out", out,
                      "--sessions", SESSIONS[-1]])
        assert ei.value.code == 2
        assert not os.path.exists(out)
