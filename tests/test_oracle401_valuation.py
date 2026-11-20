"""Oracle 4.0.1 Stage C tests: forecast-conditioned valuation + policy parity.

Master-prompt acceptance list covered here (torch-free; forecasts are plain
dicts in the ``Forecast.to_dict`` shape):
  * two intentionally different forecasts produce different scenario
    distributions AND can alter the challenger ranking (CALL vs PUT flip)
  * scenario quantile calibration: achieved ~= target, 90% interval coverage
  * realized-vol forecast conditions the scenario vol scale (with OOD clamp)
  * no hard-coded drift without a forecast; unavailable/unknown-horizon
    forecasts fall back to the labeled baseline BYTE-FOR-BYTE (CRN fail-open)
  * deterministic common random numbers with a forecast attached
  * exact exit policy / quantity from the immutable snapshot (vs flagged
    config defaults) changes simulated outcomes
  * quote age gate: stale quotes abstain with EV null ("unavailable"),
    distinct from confirmed non-fill EV == 0.0
  * fees charged exactly once (zero-fee config delta == 2 * fee_leg * qty)
  * snapshot quantity + partial fills scale EV_opportunity
  * hazard/CIF forecast output is a diagnostic only — EV is invariant
  * pre-expiry liquidation settlement diagnostic (intrinsic minus liq mark)
  * 0DTE vs swing horizon bucketing
  * all candidates valued; explicit no-trade alternative always on the menu

Synthetic fixtures only; nothing here is evidence of profitability.
"""
from __future__ import annotations

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

UTC = timezone.utc


# --------------------------------------------------------------------------- #
# helpers (mirroring tests/test_oracle4_valuation.py conventions)
# --------------------------------------------------------------------------- #

def _ts(session: str, minute: int = 0) -> datetime:
    return (datetime.fromisoformat(session).replace(tzinfo=UTC)
            + timedelta(hours=15, minutes=minute))


def _cand(did="d1", *, symbol="SPY261218C00600000", underlying="SPY",
          session="2026-11-10", minute=0, **feat):
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
    def __init__(self, p, partial, status="filled", reasons=()):
        self.p, self.partial = p, partial
        self.status, self.reasons = status, tuple(reasons)

    def estimate_fill(self, order, quote, **kw):
        return SimpleNamespace(fill_probability=self.p,
                               partial_qty=self.partial,
                               status=self.status, reasons=self.reasons)


def _obs_list(day="2026-11-10", n=200, base=600.0):
    t0 = datetime.fromisoformat(day).replace(tzinfo=UTC) + timedelta(hours=14)
    return [(t0 + timedelta(minutes=i),
             base * (1.0 + 0.001 * math.sin(i * 0.7))) for i in range(n)]


AS_OF = datetime(2026, 11, 10, 18, 0, tzinfo=UTC)
CFG = val.ValuationConfig(n_paths=256, max_steps=12)   # 60m horizon == 12 steps

Q_UP = {"0.05": -0.002, "0.25": 0.008, "0.5": 0.02,
        "0.75": 0.032, "0.95": 0.042}
Q_DOWN = {"0.05": -0.042, "0.25": -0.032, "0.5": -0.02,
          "0.75": -0.008, "0.95": 0.002}


def _fc(quantiles=None, *, rv=None, horizon="60m", available=True,
        p_up=0.6, p_down=0.2, model_id="fc_test", cif=None, **extra):
    """Plain-dict forecast in the Forecast.to_dict shape (no torch)."""
    d = {"available": available, "model_id": model_id, "horizon": horizon,
         "p_down": p_down, "p_neutral": round(1.0 - p_up - p_down, 6),
         "p_up": p_up, "quantiles": quantiles, "rv_forecast": rv,
         "cif": cif, "input_digest": "deadbeef"}
    d.update(extra)
    return d


def _scen(forecast, seed=123, cfg=CFG, obs=None):
    scen, why = val.generate_scenarios(obs or _obs_list(), AS_OF, 600.0,
                                       cfg, seed, forecast=forecast)
    assert why == "ok"
    return scen


# --------------------------------------------------------------------------- #
# A. forecast-conditioned scenario generation
# --------------------------------------------------------------------------- #

class TestForecastConditioning:
    def test_two_forecasts_produce_different_scenarios(self):
        up = _scen(_fc(Q_UP))
        dn = _scen(_fc(Q_DOWN, p_up=0.2, p_down=0.6))
        assert up.meta["mode"] == dn.meta["mode"] == "forecast_conditioned"
        assert not np.array_equal(up.prices, dn.prices)
        # directional shift at the forecast horizon (terminal column)
        assert up.prices[:, -1].mean() > 600.0 > dn.prices[:, -1].mean()

    def test_calibration_achieved_quantiles_and_coverage(self):
        scen = _scen(_fc(Q_UP))
        fm = scen.meta["forecast"]
        assert fm["quantile_remap_applied"] is True
        assert fm["k_star"] == 12 and fm["horizon_truncated"] is False
        for q, target in Q_UP.items():
            assert fm["target_quantiles"][q] == pytest.approx(float(target))
            assert fm["achieved_quantiles"][q] == pytest.approx(
                float(target), abs=1e-3), q
        # 90% interval coverage on the generated scenarios (256 paths)
        assert 0.85 <= fm["interval_coverage_05_95"] <= 0.95
        # and it matches a direct recount on the prices themselves
        r = np.log(scen.prices[:, 12] / 600.0)
        cov = float(np.mean((r >= -0.002) & (r <= 0.042)))
        assert fm["interval_coverage_05_95"] == pytest.approx(cov)

    def test_rv_forecast_conditions_vol_scale(self):
        base = _scen(None)
        hist = base.meta["hist_step_vol"]
        assert hist > 0
        scen = _scen(_fc(rv=1.5 * hist * math.sqrt(12)))
        assert scen.meta["rv_forecast"] == "model_forecast"
        assert scen.meta["vol_scale"] == pytest.approx(1.5)
        assert scen.meta["vol_scale_clamped_ood"] is False
        assert ("insufficient_quantiles_remap_skipped"
                in scen.meta["forecast"]["notes"])

    def test_rv_out_of_distribution_is_clamped(self):
        base = _scen(None)
        hist = base.meta["hist_step_vol"]
        scen = _scen(_fc(rv=10.0 * hist * math.sqrt(12)))
        assert scen.meta["vol_scale"] == CFG.vol_scale_bounds[1]
        assert scen.meta["vol_scale_clamped_ood"] is True

    def test_fail_open_baseline_byte_identity(self):
        """No forecast, unavailable forecast and unknown horizon all fall
        back to the SAME labeled baseline — no hard-coded drift, and CRN
        holds byte-for-byte across forecast variants."""
        none_ = _scen(None)
        unavail = _scen({"available": False, "reason": "model_refused"})
        badhz = _scen(_fc(Q_UP, horizon="7d"))
        assert np.array_equal(none_.prices, unavail.prices)
        assert np.array_equal(none_.prices, badhz.prices)
        for s in (none_, unavail, badhz):
            assert s.meta["mode"] == "baseline_bootstrap_ewma"
            assert s.meta["rv_forecast"] in ("ewma", "unavailable_scale_1")
        assert none_.meta["forecast"] is None
        assert unavail.meta["forecast"] == {
            "fallback_reason": "forecast_unavailable",
            "detail": "model_refused"}
        assert badhz.meta["forecast"]["fallback_reason"] == "unknown_horizon"

    def test_deterministic_crn_with_forecast(self):
        a = _scen(_fc(Q_UP), seed=77)
        b = _scen(_fc(Q_UP), seed=77)
        c = _scen(_fc(Q_UP), seed=78)
        assert np.array_equal(a.prices, b.prices)
        assert not np.array_equal(a.prices, c.prices)


# --------------------------------------------------------------------------- #
# B. forecasts alter the challenger ranking (Stage B acceptance, closed here)
# --------------------------------------------------------------------------- #

class TestForecastAltersRanking:
    def _rank(self, forecast):
        scen = _scen(forecast, seed=42)
        rows = []
        for did, sym in (("call", "SPY261218C00600000"),
                         ("put", "SPY261218P00600000")):
            cand = _cand(did, symbol=sym, quote_mid=15.0)
            priced, abst = val.reprice_candidate(cand, scen, CFG)
            assert abst is None, abst
            rows.append(val.value_candidate(cand, priced, CFG,
                                            forecast=forecast))
        ev = {r["decision_id"]: r["ev_opportunity"] for r in rows}
        return val.rank_candidates(rows, CFG), ev

    def test_forecast_direction_flips_call_put_ranking(self):
        rank_up, ev_up = self._rank(_fc(Q_UP))
        rank_dn, ev_dn = self._rank(_fc(Q_DOWN, p_up=0.2, p_down=0.6))
        assert ev_up["call"] > ev_up["put"]
        assert ev_dn["put"] > ev_dn["call"]
        assert rank_up["decision"] == "trade"
        assert rank_up["selected_decision_id"] == "call"
        assert rank_dn["decision"] == "trade"
        assert rank_dn["selected_decision_id"] == "put"


# --------------------------------------------------------------------------- #
# C. exact exit policy / quantity from the immutable snapshot
# --------------------------------------------------------------------------- #

class TestPolicyParity:
    VC = val.ValuationConfig()

    def test_snapshot_policy_and_qty_honored(self):
        cand = _cand(stop_loss_pct=5.0, take_profit_pct=10.0, qty=3)
        v = val.value_candidate(cand, _priced([[2.0, 2.4]]), self.VC)
        # snapshot target 2.12*1.10 = 2.332 -> 2.4 is a take-profit; exit
        # fill 2.28; net = (2.28-2.12)*100*3 - 2*0.02*3 = 48 - 0.12
        assert v["policy"] == {"stop_loss_pct": 5.0, "take_profit_pct": 10.0,
                               "qty": 3, "source": "snapshot",
                               "defaulted": False}
        assert v["per_fill"]["exit_reasons"]["take_profit"] == 1
        assert v["per_fill"]["net_mean"] == pytest.approx(47.88)
        assert v["costs"]["fees_mean"] == pytest.approx(0.12)
        assert v["capital_required"] == pytest.approx(636.0)

    def test_default_policy_is_flagged_and_differs(self):
        # same path under config defaults (TP 20% -> 2.544): time exit
        v = val.value_candidate(_cand(), _priced([[2.0, 2.4]]), self.VC)
        assert v["policy"]["source"] == "default_assumed"
        assert v["policy"]["defaulted"] is True
        assert v["policy"]["stop_loss_pct"] == 10.0
        assert v["policy"]["take_profit_pct"] == 20.0
        assert v["per_fill"]["exit_reasons"]["time_exit"] == 1

    def test_snapshot_stop_tighter_changes_exit(self):
        cand = _cand(stop_loss_pct=2.0)       # stop at 2.12*0.98 = 2.0776
        v = val.value_candidate(cand, _priced([[2.0, 2.05]]), self.VC)
        assert v["per_fill"]["exit_reasons"]["stop_loss"] == 1
        # default 10% stop (1.908) would have been a time exit instead
        v2 = val.value_candidate(_cand(), _priced([[2.0, 2.05]]), self.VC)
        assert v2["per_fill"]["exit_reasons"]["time_exit"] == 1

    def test_invalid_snapshot_policy_falls_back_flagged(self):
        cand = _cand(stop_loss_pct=-5.0, take_profit_pct="nan?", qty=0)
        stop, tp, qty, source, defaulted = val._resolve_policy(cand, self.VC)
        assert (stop, tp, qty) == (10.0, 20.0, 1)
        assert source == "default_assumed" and defaulted is True


# --------------------------------------------------------------------------- #
# D. quote age + EV taxonomy (null unavailable vs confirmed zero)
# --------------------------------------------------------------------------- #

class TestQuoteAgeAndEvTaxonomy:
    VC = val.ValuationConfig()
    FLAT = val.ScenarioSet(prices=np.full((8, 7), 600.0), step_minutes=5.0)

    def test_stale_quote_abstains_ev_unavailable(self):
        cand = _cand(quote_mid=15.0, quote_age_s=600.0)
        priced, abst = val.reprice_candidate(cand, self.FLAT, self.VC)
        assert priced is None
        assert abst["reason"] == "stale_quote"
        assert abst["ev_opportunity"] is None
        assert abst["ev_basis"] == "unavailable"
        assert abst["quote_age_s"] == 600.0
        assert abst["max_quote_age_s"] == 120.0

    def test_fresh_quote_passes_age_gate(self):
        cand = _cand(quote_mid=15.0, quote_age_s=30.0)
        priced, abst = val.reprice_candidate(cand, self.FLAT, self.VC)
        assert abst is None and priced is not None

    def test_ev_null_vs_abstention_zero_vs_nonfill_zero(self):
        # (1) estimated EV on a valued row
        ok = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), self.VC)
        assert ok["ev_basis"] == "estimated"
        assert ok["ev_opportunity"] > 0
        # (2) confirmed non-fill -> exactly 0.0
        stub = _StubFill(p=0.0, partial=0.0, status="pending",
                         reasons=("no_quote",))
        nf = val.value_candidate(_cand(), _priced([[2.0, 3.0]]), self.VC,
                                 fill_model=stub)
        assert nf["ev_opportunity"] == 0.0
        assert nf["ev_basis"] == "confirmed_nonfill"
        # (3) missing quote is also a confirmed unusable market -> 0.0
        _, mq = val.reprice_candidate(_cand(quote_mid=None), self.FLAT,
                                      self.VC)
        assert mq["ev_opportunity"] == 0.0
        assert mq["ev_basis"] == "confirmed_nonfill"
        # (4) stale quote EV is UNKNOWN -> null, never conflated with zero
        _, st = val.reprice_candidate(
            _cand(quote_mid=15.0, quote_age_s=999.0), self.FLAT, self.VC)
        assert st["ev_opportunity"] is None
        assert st["ev_basis"] == "unavailable"


# --------------------------------------------------------------------------- #
# E. costs charged once + quantity/partial-fill scaling
# --------------------------------------------------------------------------- #

class TestCostsAndFills:
    VC = val.ValuationConfig()

    def test_fees_charged_exactly_once(self):
        pr = _priced([[2.0, 2.05]])
        v_fee = val.value_candidate(_cand(), pr, self.VC)
        v_free = val.value_candidate(
            _cand(), pr, self.VC,
            cost_config=CostConfig(occ_fee_per_contract=0.0,
                                   commission_per_contract=0.0))
        # the ONLY difference is both fee legs, exactly once: 2 * 0.02 * qty
        delta = v_free["per_fill"]["net_mean"] - v_fee["per_fill"]["net_mean"]
        assert delta == pytest.approx(0.04)
        assert v_fee["costs"]["fees_mean"] == pytest.approx(0.04)
        assert v_free["costs"]["fees_mean"] == pytest.approx(0.0)

    def test_snapshot_qty_with_partial_fill_scales_ev(self):
        cand = _cand(qty=4)
        stub = _StubFill(p=0.8, partial=2, status="partially_filled")
        v = val.value_candidate(cand, _priced([[2.0, 3.0]]), self.VC,
                                fill_model=stub)
        assert v["policy"]["qty"] == 4
        assert v["fill"]["fill_frac"] == pytest.approx(0.5)
        # per-fill net scales with qty=4: 4 * 75.96 + fee diff
        # net = (2.88-2.12)*100*4 - 2*0.02*4 = 304 - 0.16
        assert v["per_fill"]["net_mean"] == pytest.approx(303.84)
        assert v["ev_opportunity"] == pytest.approx(0.8 * 0.5 * 303.84)

    def test_friction_stress_monotone_with_snapshot_policy(self):
        cand = _cand(stop_loss_pct=10.0, take_profit_pct=20.0, qty=1)
        pr = _priced([[2.0, 2.05]])
        nets = [val.value_candidate(cand, pr, self.VC,
                                    friction=f)["per_fill"]["net_mean"]
                for f in (1.0, 1.5, 2.0)]
        assert nets[0] > nets[1] > nets[2]
        assert nets[0] == pytest.approx(-19.04)


# --------------------------------------------------------------------------- #
# F. diagnostics: hazard CIF (no double counting), settlement, horizon
# --------------------------------------------------------------------------- #

class TestDiagnostics:
    VC = val.ValuationConfig()

    def test_hazard_diagnostic_is_ev_invariant(self):
        cif = {"stop": [0.01] * 7, "target": [0.02, 0.05, 0.1, 0.2, 0.3,
                                              0.4, 0.5],
               "other_exit": [0.0] * 7}
        pr = _priced([[2.0, 3.0]])
        with_fc = val.value_candidate(_cand(), pr, self.VC,
                                      forecast=_fc(Q_UP, cif=cif))
        without = val.value_candidate(_cand(), pr, self.VC)
        # the forecast CIF NEVER changes EV — diagnostic only
        assert with_fc["ev_opportunity"] == without["ev_opportunity"]
        assert with_fc["per_fill"]["net_mean"] == \
            without["per_fill"]["net_mean"]
        hd = with_fc["hazard_diagnostic"]
        assert hd["forecast_cif_terminal"] == {
            "stop": 0.01, "target": 0.5, "other_exit": 0.0}
        assert hd["sim_exit_reasons"] == with_fc["per_fill"]["exit_reasons"]
        assert "no double counting" in hd["note"]
        assert without["hazard_diagnostic"] is None

    def test_unavailable_forecast_has_no_hazard_diag(self):
        v = val.value_candidate(
            _cand(), _priced([[2.0, 3.0]]), self.VC,
            forecast={"available": False, "reason": "x"})
        assert v["hazard_diagnostic"] is None

    def test_settlement_residual_diagnostic(self):
        # 2 sim steps, expiry inside: liquidation at col 1 (mark 2.1); the
        # intrinsic settlement col 2 (0.05) is reported as a residual only.
        v = val.value_candidate(
            _cand(), _priced([[2.0, 2.1, 0.05]], expires_in_sim=True),
            self.VC)
        s = v["settlement"]
        assert s["convention"] == "pre_expiry_liquidation_research"
        assert s["n_liquidated"] == 1
        assert s["mean_intrinsic_minus_liq_mark"] == pytest.approx(-2.05)
        assert v["per_fill"]["exit_reasons"]["pre_expiry_liquidation"] == 1

    def test_horizon_class_buckets(self):
        v0 = val.value_candidate(_cand(dte_days=0.5),
                                 _priced([[2.0, 2.05]]), self.VC)
        vs = val.value_candidate(_cand(dte_days=29.0),
                                 _priced([[2.0, 2.05]]), self.VC)
        assert v0["horizon_class"] == "0dte"
        assert vs["horizon_class"] == "swing"


# --------------------------------------------------------------------------- #
# G. all candidates valued + explicit no-trade alternative
# --------------------------------------------------------------------------- #

class TestNoTradeAlternative:
    def test_all_candidates_valued_then_no_trade_wins(self):
        scen = _scen(_fc(Q_DOWN, p_up=0.2, p_down=0.6), seed=42)
        rows = []
        for did, sym in (("c600", "SPY261218C00600000"),
                         ("c605", "SPY261218C00605000")):
            cand = _cand(did, symbol=sym, quote_mid=15.0)
            priced, abst = val.reprice_candidate(cand, scen, CFG)
            assert abst is None
            rows.append(val.value_candidate(cand, priced, CFG))
        rank = val.rank_candidates(rows, CFG)
        # every candidate was valued before ranking, and under a strong
        # down forecast both calls lose -> the explicit no-trade option wins
        assert rank["n_valued"] == 2
        assert all(r["status"] == "ok" for r in rows)
        assert all(r["ev_opportunity"] < 0 for r in rows)
        assert rank["decision"] == "abstain"
        assert rank["abstention_ev"] == 0.0
        assert rank["selected_decision_id"] is None
