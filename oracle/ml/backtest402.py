"""Comprehensive backtest harness (Oracle 4.0.2 P10).

Extends the frozen phase-6 replay engine (``oracle.ml.replay``) with:

  * a **GBM direction challenger** (LightGBM, purged walk-forward OOF,
    native-NaN missingness) alongside the classical logistic and MLP;
  * an **ensemble gate** (renormalized mean of classical + GBM
    probabilities, computed only on the intersection of covered
    decisions -- uncovered trades pass through ungated, same as every
    other gate here);
  * **9 variants** on the identical trade universe: baseline_frozen,
    baseline_quality, valuation_ranked, classical_gated, neural_gated,
    combined, gbm_gated, ensemble_gated, gbm_combined;
  * **breakdowns** per variant by underlying / direction / DTE bucket /
    exit reason (unconstrained portfolio per cell -- membership, not
    ordering, drives the numbers; cells do NOT sum to the constrained
    portfolio and the artifact says so);
  * **matched cohorts** for both the classical and GBM gates;
  * the same friction sweep (1.0x/1.5x/2.0x), stop-overshoot stress and
    paired per-session bootstrap as the frozen replay.

Honesty rules carried over verbatim from replay.py: holdout refusal,
input crosschecks that raise rather than mix artifacts, counterfactual
notes on selection variants, no fabricated regimes (regime labels are
not recorded on traded-outcome rows, so the regime breakdown is an
explicit ``unavailable`` stub), parity check between the portfolio
engine and independent per-trade arithmetic.
"""
from __future__ import annotations

import argparse
import json
import os
from datetime import date
from typing import Any, Dict, List, Optional, Sequence, Set, Tuple

from oracle.lab.purged_validation import (
    dataset_meta,
    make_folds,
    metrics_fingerprint,
    paired_session_bootstrap,
)
from oracle.ml import gbm as gbm_mod
from oracle.ml.evidence import kish_ess
from oracle.ml.mlp import direction_class
from oracle.ml.replay import (
    COUNTERFACTUAL_NOTE,
    DEFAULT_HOLDOUT,
    FRICTION_MULTS,
    MTM_NOTE,
    QUALITY_MAX_SPREAD_PCT,
    Trade,
    _nonholdout_samples,
    _ok,
    _per_session_delta,
    _quality_pass,
    build_variants,
    compute_oof_classical,
    compute_oof_neural,
    crosscheck_inputs,
    gate_direction,
    load_friction,
    load_session_calendar,
    load_traded_outcomes,
    load_valuation_selection,
    run_portfolio,
)
from oracle.ml.valuation import parse_occ_symbol

__all__ = [
    "compute_oof_gbm",
    "ensemble_preds",
    "gate_trades",
    "build_variants_402",
    "compute_breakdowns",
    "run_backtest402",
    "main",
]

Pred = Tuple[float, float, float]

BREAKDOWN_NOTE = (
    "breakdown cells are UNCONSTRAINED portfolios (effectively infinite "
    "cash/slots, calibrated 1.0x friction) so that group membership -- "
    "not event ordering under concurrency/cash limits -- drives each "
    "cell; cells therefore do NOT sum to the constrained portfolio "
    "numbers in by_friction_mult.")

REGIME_NOTE = ("regime labels are not recorded on traded-outcome rows; "
               "no regime is fabricated")


# --------------------------------------------------------------------------- #
# GBM OOF challenger
# --------------------------------------------------------------------------- #
def compute_oof_gbm(
    dataset_path: str,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
    horizon: str = "60m",
) -> Tuple[Dict[str, Pred], Dict[str, Any]]:
    """Purged walk-forward OOF direction probabilities from LightGBM.

    Mirrors ``compute_oof_classical`` fold-for-fold; differences:
      * features go through ``gbm.build_matrix(..., mode="native")`` so
        missing values stay NaN (LightGBM handles them natively);
      * early stopping uses a validation tail = the last training
        session of each fold; when that tail is unusable the model
        falls back to a fixed boosting budget on the full training
        block (counted in ``folds_fixed_budget``) rather than early
        stopping on training data.
    """
    if not gbm_mod.LGBM_AVAILABLE:
        return {}, {"model": "lightgbm_direction", "status": "unavailable",
                    "why": gbm_mod.LGBM_WHY}
    samples = _nonholdout_samples(dataset_path, horizon, holdout)
    folds, fold_sessions, _ = make_folds(
        samples, test_block_sessions=1, min_train_sessions=2,
        embargo_sessions=1, holdout_sessions=0)
    by_id = {s.decision_id: s for s in samples}
    names = sorted({k for s in samples for k in s.features})

    preds: Dict[str, Pred] = {}
    folds_used = 0
    folds_skipped = 0
    folds_fixed_budget = 0
    for fold in folds:
        train_all = [by_id[d] for d in fold.train_ids if _ok(by_id[d])]
        test = [by_id[d] for d in fold.test_ids]
        y_all = [direction_class(float(s.label_value)) for s in train_all]
        if not test or len(train_all) < 10 or len(set(y_all)) < 2:
            folds_skipped += 1
            continue
        # validation tail: last training session; honest fallback is an
        # EMPTY validation set (fixed budget), never val == train.
        last_sess = max(s.session for s in train_all)
        fit_rows = [s for s in train_all if s.session != last_sess]
        val_rows = [s for s in train_all if s.session == last_sess]
        y_fit = [direction_class(float(s.label_value)) for s in fit_rows]
        if len(fit_rows) < 10 or len(val_rows) < 2 or len(set(y_fit)) < 2:
            fit_rows, val_rows = train_all, []
            y_fit = y_all
        y_val = [direction_class(float(s.label_value)) for s in val_rows]
        x_fit, _ = gbm_mod.build_matrix(fit_rows, names, "native")
        x_val, _ = gbm_mod.build_matrix(val_rows, names, "native")
        x_test, _ = gbm_mod.build_matrix(test, names, "native")
        booster, info = gbm_mod.fit_direction_gbm(x_fit, y_fit, x_val, y_val)
        if not info.get("val_used"):
            folds_fixed_budget += 1
        proba = gbm_mod.predict_direction_proba(booster, x_test)
        for s, row in zip(test, proba):
            preds[s.decision_id] = (
                float(row[0]), float(row[1]), float(row[2]))
        folds_used += 1

    report = {
        "model": "lightgbm_direction",
        "status": "ok",
        "classes": ["down", "neutral", "up"],
        "horizon": horizon,
        "lgbm_version": gbm_mod.LGBM_VERSION,
        "missingness_mode": "native",
        "folds_total": len(folds),
        "folds_used": folds_used,
        "folds_skipped": folds_skipped,
        "folds_fixed_budget": folds_fixed_budget,
        "sessions": fold_sessions,
        "predictions": len(preds),
    }
    return preds, report


def ensemble_preds(classical: Dict[str, Pred],
                   gbm: Dict[str, Pred]) -> Dict[str, Pred]:
    """Renormalized mean of two probability maps on their intersection.

    Decisions covered by only one model are NOT included -- the gate's
    no-prediction passthrough handles them, so the ensemble never
    silently degrades to a single model.
    """
    out: Dict[str, Pred] = {}
    for d in set(classical) & set(gbm):
        a, b = classical[d], gbm[d]
        p = tuple((x + y) / 2.0 for x, y in zip(a, b))
        s = sum(p)
        if s > 0:
            out[d] = (p[0] / s, p[1] / s, p[2] / s)
    return out


def gate_trades(trades: Sequence[Trade], preds: Dict[str, Pred]
                ) -> Tuple[List[Trade], Dict[str, int]]:
    """Directional gate over a trade list (same semantics as replay's
    internal ``_gated``): trades without a prediction are KEPT -- the
    gate only acts where evidence exists."""
    kept: List[Trade] = []
    acct = {"gated_out": 0, "no_prediction": 0,
            "gate_passed": 0, "abstained": 0}
    for t in trades:
        p = preds.get(t.decision_id)
        if p is None:
            acct["no_prediction"] += 1
            kept.append(t)
            continue
        verdict = gate_direction(t.option_type, p[0], p[1], p[2])
        if verdict == "pass":
            acct["gate_passed"] += 1
            kept.append(t)
        elif verdict == "abstain":
            acct["abstained"] += 1
        else:
            acct["gated_out"] += 1
    return kept, acct


# --------------------------------------------------------------------------- #
# Variants (frozen six + three GBM-era challengers)
# --------------------------------------------------------------------------- #
def build_variants_402(
    trades: List[Trade],
    valuation_ids: Optional[Set[str]],
    classical_preds: Dict[str, Pred],
    neural_preds: Dict[str, Pred],
    neural_status: str,
    gbm_preds: Dict[str, Pred],
    gbm_status: str,
) -> Dict[str, Dict[str, Any]]:
    """The six frozen replay variants plus gbm_gated / ensemble_gated /
    gbm_combined.  Unavailable challengers keep the replay stub shape
    ``{"trades": None, "accounting": {"status": "unavailable"}}``."""
    variants = build_variants(trades, valuation_ids, classical_preds,
                              neural_preds, neural_status)

    if gbm_status == "ok":
        kept_g, acct_g = gate_trades(trades, gbm_preds)
        variants["gbm_gated"] = {
            "trades": kept_g,
            "accounting": {"n": len(kept_g), **acct_g},
        }
        ens = ensemble_preds(classical_preds, gbm_preds)
        kept_e, acct_e = gate_trades(trades, ens)
        variants["ensemble_gated"] = {
            "trades": kept_e,
            "accounting": {
                "n": len(kept_e),
                "ensemble_covered": len(ens),
                "members": ["multinomial_logistic", "lightgbm_direction"],
                **acct_e,
            },
        }
        if valuation_ids is not None:
            kept_gc = [t for t in kept_g if t.decision_id in valuation_ids]
            variants["gbm_combined"] = {
                "trades": kept_gc,
                "accounting": {
                    "n": len(kept_gc),
                    "gbm_gate": acct_g,
                    "and_valuation_selected": True,
                    "note": COUNTERFACTUAL_NOTE,
                },
            }
        else:
            variants["gbm_combined"] = {
                "trades": None,
                "accounting": {"status": "unavailable",
                               "why": "no valuation selection artifact"},
            }
    else:
        stub = {"trades": None, "accounting": {"status": "unavailable"}}
        variants["gbm_gated"] = dict(stub)
        variants["ensemble_gated"] = dict(stub)
        variants["gbm_combined"] = dict(stub)
    return variants


# --------------------------------------------------------------------------- #
# Breakdowns
# --------------------------------------------------------------------------- #
def _dte_bucket(t: Trade) -> str:
    occ = parse_occ_symbol(t.symbol)
    if not occ or not occ.get("expiration"):
        return "unknown"
    try:
        dte = (date.fromisoformat(occ["expiration"])
               - t.entry_at.date()).days
    except (ValueError, TypeError):
        return "unknown"
    if dte <= 0:
        return "0dte"
    if dte <= 7:
        return "1-7d"
    if dte <= 30:
        return "8-30d"
    return ">30d"


def compute_breakdowns(trades: Sequence[Trade], sessions: Sequence[str],
                       one_way_frac: float) -> Dict[str, Any]:
    """Per-group UNCONSTRAINED portfolio metrics (see BREAKDOWN_NOTE)."""
    def _groups(keyfn) -> Dict[str, List[Trade]]:
        out: Dict[str, List[Trade]] = {}
        for t in trades:
            out.setdefault(keyfn(t), []).append(t)
        return out

    def _slim(group: List[Trade]) -> Dict[str, Any]:
        m = run_portfolio(group, sessions, starting_cash=1e12,
                          max_concurrent=10**9,
                          one_way_frac=one_way_frac, friction_mult=1.0)
        return {k: m[k] for k in ("n_executed", "net_pnl", "win_rate",
                                  "expectancy_per_trade", "profit_factor")}

    dims = {
        "by_underlying": _groups(lambda t: str(t.underlying or "unknown")),
        "by_direction": _groups(lambda t: str(t.option_type or "unknown")),
        "by_dte_bucket": _groups(_dte_bucket),
        "by_exit_reason": _groups(lambda t: str(t.reason or "unknown")),
    }
    out: Dict[str, Any] = {
        dim: {key: _slim(group) for key, group in sorted(groups.items())}
        for dim, groups in dims.items()
    }
    out["by_regime"] = {"status": "unavailable", "why": REGIME_NOTE}
    out["note"] = BREAKDOWN_NOTE
    return out


# --------------------------------------------------------------------------- #
# Matched cohort (generalized from replay for any prediction map)
# --------------------------------------------------------------------------- #
def _gate_cohort(trades: List[Trade], preds: Dict[str, Pred],
                 sessions: Sequence[str], friction: Dict[str, Any],
                 starting_cash: float, max_concurrent: int,
                 n_boot: int, seed: int) -> Dict[str, Any]:
    """Apples-to-apples cohort: trades with BOTH a prediction and an
    executable quote; baseline vs gate compared on that same cohort."""
    cohort = [t for t in trades
              if t.decision_id in preds and _quality_pass(t)]
    excluded_no_prediction = sum(
        1 for t in trades if t.decision_id not in preds)
    excluded_quote = sum(
        1 for t in trades
        if t.decision_id in preds and not _quality_pass(t))
    if not cohort:
        return {
            "status": "empty",
            "n_cohort": 0,
            "excluded_no_prediction": excluded_no_prediction,
            "excluded_quote_ineligible": excluded_quote,
            "note": ("no trades have both an OOF prediction and an "
                     "executable quote; gate value on a matched cohort "
                     "cannot be measured"),
        }
    cohort_kept: List[Trade] = []
    cohort_acct = {"gate_passed": 0, "abstained": 0, "gated_out": 0}
    for t in cohort:
        p = preds[t.decision_id]
        verdict = gate_direction(t.option_type, p[0], p[1], p[2])
        if verdict == "pass":
            cohort_acct["gate_passed"] += 1
            cohort_kept.append(t)
        elif verdict == "abstain":
            cohort_acct["abstained"] += 1
        else:
            cohort_acct["gated_out"] += 1

    def _slim(m: Dict[str, Any]) -> Dict[str, Any]:
        return {k: m[k] for k in (
            "n_executed", "net_pnl", "profit_factor",
            "sharpe_daily_ann", "max_drawdown", "pnl_by_exit_day")}

    base_m = _slim(run_portfolio(
        cohort, sessions, starting_cash=starting_cash,
        max_concurrent=max_concurrent,
        one_way_frac=friction["one_way_frac"], friction_mult=1.0))
    chal_m = _slim(run_portfolio(
        cohort_kept, sessions, starting_cash=starting_cash,
        max_concurrent=max_concurrent,
        one_way_frac=friction["one_way_frac"], friction_mult=1.0))
    delta = _per_session_delta(chal_m["pnl_by_exit_day"],
                               base_m["pnl_by_exit_day"])
    nonzero = [abs(v) for v in delta.values() if v != 0.0]
    return {
        "status": "ok",
        "n_cohort": len(cohort),
        "excluded_no_prediction": excluded_no_prediction,
        "excluded_quote_ineligible": excluded_quote,
        "challenger_gate": cohort_acct,
        "baseline_on_cohort": {k: v for k, v in base_m.items()
                               if k != "pnl_by_exit_day"},
        "challenger_on_cohort": {k: v for k, v in chal_m.items()
                                 if k != "pnl_by_exit_day"},
        "paired": {
            "per_session_delta": {k: round(v, 6)
                                  for k, v in delta.items()},
            "bootstrap": paired_session_bootstrap(
                delta, n_boot=n_boot, seed=seed),
            "n_nonzero_delta_sessions": len(nonzero),
            "ess_abs_delta": round(kish_ess(nonzero), 4),
        },
        "note": ("cohort = trades with BOTH an OOF prediction and an "
                 "executable quote; baseline and challenger are run on "
                 "the identical cohort so the delta isolates the gate"),
    }


# --------------------------------------------------------------------------- #
# Orchestration
# --------------------------------------------------------------------------- #
def run_backtest402(
    dataset_path: str,
    valuation_path: Optional[str] = None,
    execution_calibration_path: Optional[str] = None,
    out_path: Optional[str] = None,
    holdout: Sequence[str] = DEFAULT_HOLDOUT,
    starting_cash: float = 100_000.0,
    max_concurrent: int = 5,
    n_boot: int = 2000,
    seed: int = 42,
    include_neural: bool = True,
    include_gbm: bool = True,
    mlp_epochs: int = 20,
) -> Dict[str, Any]:
    trades, trade_report = load_traded_outcomes(dataset_path, holdout)
    traded_sessions = sorted({t.session for t in trades})
    sessions = load_session_calendar(dataset_path, holdout) \
        or traded_sessions
    friction = load_friction(execution_calibration_path)
    meta = dataset_meta(dataset_path)

    valuation_ids: Optional[Set[str]] = None
    valuation_report: Optional[Dict[str, Any]] = None
    if valuation_path and os.path.exists(valuation_path):
        valuation_ids, valuation_report = load_valuation_selection(
            valuation_path)
    crosschecks = crosscheck_inputs(meta, valuation_report, friction)

    classical_preds, classical_report = compute_oof_classical(
        dataset_path, holdout)
    if include_neural:
        neural_preds, neural_report = compute_oof_neural(
            dataset_path, holdout, max_epochs=mlp_epochs, seed=seed)
    else:
        neural_preds, neural_report = {}, {
            "model": "multitask_mlp", "status": "unavailable",
            "why": "disabled by --no-neural"}
    if include_gbm:
        gbm_preds, gbm_report = compute_oof_gbm(dataset_path, holdout)
    else:
        gbm_preds, gbm_report = {}, {
            "model": "lightgbm_direction", "status": "unavailable",
            "why": "disabled by --no-gbm"}

    variants = build_variants_402(
        trades, valuation_ids, classical_preds, neural_preds,
        neural_report.get("status", "unavailable"),
        gbm_preds, gbm_report.get("status", "unavailable"))

    # parity: engine vs independent per-trade arithmetic (frictionless,
    # unconstrained) -- must agree to float precision.
    parity_engine = run_portfolio(
        trades, sessions, starting_cash=1e12, max_concurrent=10**9,
        one_way_frac=0.0, friction_mult=1.0)
    independent_gross = sum(
        t.premium * t.gross_pnl_pct / 100.0 for t in trades)
    floored_gross = sum(
        max(t.premium * (1.0 + t.gross_pnl_pct / 100.0), 0.0) - t.premium
        for t in trades)
    parity = {
        "n_trades_engine": parity_engine["n_executed"],
        "n_trades_outcomes": len(trades),
        "engine_gross_pnl": parity_engine["gross_pnl"],
        "independent_gross_pnl": round(independent_gross, 6),
        "engine_net_pnl_frictionless": parity_engine["net_pnl"],
        "independent_net_pnl_floored": round(floored_gross, 6),
        "max_abs_diff": round(
            abs(parity_engine["net_pnl"] - floored_gross), 9),
        "ok": (parity_engine["n_executed"] == len(trades)
               and abs(parity_engine["net_pnl"] - floored_gross) < 1e-6),
    }

    results: Dict[str, Any] = {}
    per_session_base: Dict[str, Dict[str, float]] = {}
    breakdowns: Dict[str, Any] = {}
    for name, spec in variants.items():
        if spec["trades"] is None:
            results[name] = {"status": "unavailable",
                             "accounting": spec["accounting"]}
            continue
        by_mult: Dict[str, Any] = {}
        for mult in FRICTION_MULTS:
            m = run_portfolio(
                spec["trades"], sessions, starting_cash=starting_cash,
                max_concurrent=max_concurrent,
                one_way_frac=friction["one_way_frac"],
                friction_mult=mult)
            key = f"{mult:.1f}"
            if name == "baseline_frozen":
                per_session_base[key] = m["pnl_by_exit_day"]
            by_mult[key] = {k: v for k, v in m.items()
                            if k not in ("equity_curve",)}
        stress = run_portfolio(
            spec["trades"], sessions, starting_cash=starting_cash,
            max_concurrent=max_concurrent,
            one_way_frac=friction["one_way_frac"], friction_mult=1.0,
            overshoot_extra_pts=friction["overshoot_extra_pts"])
        results[name] = {
            "status": "ok",
            "accounting": spec["accounting"],
            "by_friction_mult": by_mult,
            "stop_overshoot_stress": {
                "extra_pts": friction["overshoot_extra_pts"],
                "net_pnl": stress["net_pnl"],
                "delta_vs_unstressed": round(
                    stress["net_pnl"] - by_mult["1.0"]["net_pnl"], 6),
            },
        }
        breakdowns[name] = compute_breakdowns(
            spec["trades"], sessions, friction["one_way_frac"])

    paired: Dict[str, Any] = {}
    base_days = per_session_base.get("1.0", {})
    for name, res in results.items():
        if name == "baseline_frozen" or res.get("status") != "ok":
            continue
        days = res["by_friction_mult"]["1.0"]["pnl_by_exit_day"]
        delta = _per_session_delta(days, base_days)
        nonzero = [abs(v) for v in delta.values() if v != 0.0]
        paired[name] = {
            "per_session_delta": {k: round(v, 6)
                                  for k, v in delta.items()},
            "bootstrap": paired_session_bootstrap(
                delta, n_boot=n_boot, seed=seed),
            "n_nonzero_delta_sessions": len(nonzero),
            "ess_abs_delta": round(kish_ess(nonzero), 4),
        }

    matched_cohorts = {
        "classical": _gate_cohort(
            trades, classical_preds, sessions, friction,
            starting_cash, max_concurrent, n_boot, seed),
        "gbm": (_gate_cohort(
            trades, gbm_preds, sessions, friction,
            starting_cash, max_concurrent, n_boot, seed)
            if gbm_report.get("status") == "ok"
            else {"status": "unavailable",
                  "why": gbm_report.get("why", "gbm not run")}),
    }

    artifact: Dict[str, Any] = {
        "generated_for": "oracle4.0.2_comprehensive_backtest",
        "phase": 10,
        "seed": seed,
        "config": {
            "starting_cash": starting_cash,
            "max_concurrent": max_concurrent,
            "friction_mults": list(FRICTION_MULTS),
            "n_boot": n_boot,
            "quality_max_spread_pct": QUALITY_MAX_SPREAD_PCT,
            "mlp_epochs": mlp_epochs,
        },
        "inputs": {
            "dataset": dataset_path,
            "dataset_meta": meta,
            "valuation": valuation_report,
            "execution_calibration": friction,
        },
        "holdout_refusal": {
            "sessions": list(holdout),
            "decisions_excluded": trade_report["holdout_excluded"],
            "note": ("holdout sessions are never replayed, never "
                     "scored, never selected on"),
        },
        "trade_universe": trade_report,
        "sessions_replayed": list(sessions),
        "session_calendar": {
            "n_sessions": len(sessions),
            "n_traded_sessions": len(traded_sessions),
            "no_trade_sessions": sorted(
                set(sessions) - set(traded_sessions)),
            "note": ("calendar = all non-holdout feature sessions; "
                     "sessions without resolved trades contribute "
                     "zero-P&L days to Sharpe/drawdown"),
        },
        "input_crosschecks": crosschecks,
        "mtm_note": MTM_NOTE,
        "missing_data_accounting": {
            "outcome_status_counts_nonholdout":
                trade_report["outcome_status_counts"],
            "counterfactual_note": COUNTERFACTUAL_NOTE,
            "classical_no_prediction": sum(
                1 for t in trades
                if t.decision_id not in classical_preds),
            "gbm_no_prediction": sum(
                1 for t in trades if t.decision_id not in gbm_preds),
            "neural": neural_report.get("status"),
            "selection_bias_note": (
                "only decisions that actually traded have resolved "
                "outcomes; gated variants can only REMOVE trades from "
                "that universe, so gate value is measured as avoided "
                "loss, not hypothetical new profit"),
        },
        "oof_gates": {
            "classical": classical_report,
            "neural": neural_report,
            "gbm": gbm_report,
        },
        "parity": parity,
        "variants": results,
        "breakdowns": breakdowns,
        "paired_vs_baseline": paired,
        "matched_cohorts": matched_cohorts,
        "risk_metrics_limited": {
            "limited": True,
            "reasons": [
                "single contiguous replay window; no regime diversity",
                "daily Sharpe annualized from few sessions is noisy",
            ] + (["fewer than 30 replay sessions; treat Sharpe and "
                  "max drawdown as descriptive only"]
                 if len(sessions) < 30 else []),
            "note": ("promotion decisions must rely on the paired "
                     "bootstrap deltas and forward evidence, not on "
                     "point risk metrics from this window"),
        },
        "event_ordering_note": (
            "portfolio events are ordered (timestamp, entry-before-"
            "exit, decision_id); identical inputs replay identically"),
        "provenance_note": (
            "all inputs are fingerprinted artifacts; crosschecks above "
            "refuse to mix mismatched dataset/valuation/calibration"),
        "breakdown_note": BREAKDOWN_NOTE,
        "regime_note": REGIME_NOTE,
    }
    artifact["fingerprint"] = metrics_fingerprint(artifact)

    if out_path:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)
        with open(out_path, "w", encoding="utf-8") as fh:
            json.dump(artifact, fh, indent=1, sort_keys=True)
    return artifact


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0.2 comprehensive backtest (P10)")
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--valuation", default=None)
    ap.add_argument("--execution-calibration", default=None)
    ap.add_argument("--out", default="artifacts/oracle4/replay402.json")
    ap.add_argument("--holdout", nargs="*", default=list(DEFAULT_HOLDOUT))
    ap.add_argument("--starting-cash", type=float, default=100_000.0)
    ap.add_argument("--max-concurrent", type=int, default=5)
    ap.add_argument("--n-boot", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--no-neural", action="store_true")
    ap.add_argument("--no-gbm", action="store_true")
    ap.add_argument("--mlp-epochs", type=int, default=20)
    args = ap.parse_args(argv)

    artifact = run_backtest402(
        args.dataset,
        valuation_path=args.valuation,
        execution_calibration_path=args.execution_calibration,
        out_path=args.out,
        holdout=tuple(args.holdout),
        starting_cash=args.starting_cash,
        max_concurrent=args.max_concurrent,
        n_boot=args.n_boot,
        seed=args.seed,
        include_neural=not args.no_neural,
        include_gbm=not args.no_gbm,
        mlp_epochs=args.mlp_epochs,
    )
    p = artifact["parity"]
    print(f"parity ok={p['ok']} (n={p['n_trades_engine']}, "
          f"max_abs_diff={p['max_abs_diff']})")
    for name, res in artifact["variants"].items():
        if res.get("status") == "ok":
            m = res["by_friction_mult"]["1.0"]
            print(f"{name:18s} n={m['n_executed']:4d} "
                  f"net={m['net_pnl']:12.2f} pf={m['profit_factor']} "
                  f"sharpe={m['sharpe_daily_ann']}")
        else:
            print(f"{name:18s} status={res.get('status')}")
    for gate, cohort in artifact["matched_cohorts"].items():
        if cohort.get("status") == "ok":
            bs = cohort["paired"]["bootstrap"]
            print(f"cohort[{gate}] n={cohort['n_cohort']} "
                  f"delta_mean={bs.get('mean')} "
                  f"ci=[{bs.get('ci_lo')}, {bs.get('ci_hi')}]")
        else:
            print(f"cohort[{gate}] status={cohort.get('status')}")
    print(f"fingerprint={artifact['fingerprint']}")
    if args.out:
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
