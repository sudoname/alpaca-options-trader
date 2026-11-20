"""Oracle 4.0.1 Stage F: promotion report with pre-registered gates.

Every forward performance measure, sample count, calibration result,
coverage figure and provenance claim is computed from ONE validated
immutable forward-evidence artifact (``oracle.ml.evidence``) for the exact
model, policy, baseline, data window and cost assumptions.  The historical
replay artifact contributes only replay-side invariants (accounting parity,
demonstrated holdout refusal) and context; its performance numbers are
development data and never satisfy forward gates.

Pre-registered gates (fixed before any final holdout was examined):

* evidence volume: >=60 forward sessions, >=200 resolved opportunities,
  >=100 actual (live-order) fills with effective sample size reported;
* expectancy: session-bootstrap 95% lower bound of daily net P&L > 0;
* paired uplift vs the frozen baseline: bootstrap 95% lower bound > 0,
  with clustered uncertainty (nonzero-delta sessions + Kish ESS reported);
* friction: net-positive at 1.5x; BOTH 2x friction AND stop-overshoot
  stress documented (an ``or`` here was a defect — fixed);
* annualized daily Sharpe (on equity returns) > 1, profit factor > 1.3
  computed from live fills;
* max drawdown within the predeclared limit AND no worse than the
  comparable baseline; a missing baseline drawdown is INSUFFICIENT —
  "no worse" can never pass vacuously;
* calibration improves (recomputed from brier scores, not a trusted bool);
* coverage disparity <= 10% (recomputed from per-group counts);
* evidence basis: only live order fills count; paper / mark-derived /
  counterfactual evidence FAILS this gate and can never reach ELIGIBLE;
* replay invariants: accounting parity and demonstrable holdout refusal.

Hard rules, enforced structurally:

* absent/invalid/thin evidence => ``INSUFFICIENT_EVIDENCE`` — thresholds
  are never relaxed to fit the data; known gate failures are still shown;
* adequate evidence with any failed gate => ``DO_NOT_PROMOTE``;
* the best possible verdict is ``ELIGIBLE_PENDING_HUMAN_REVIEW``; there is
  no PROMOTE output and no code path that changes the registry, a flag, or
  any trading behavior.  Promotion itself is a separate human action
  (``registry.approve_model`` + ``registry.set_champion``) and the named
  review must bind to this report's fingerprint
  (``oracle.ml.evidence.verify_human_review``).

Estimands are registered explicitly: the primary estimand is the
session-bootstrap 95% lower bound of per-session daily net P&L; expectancy
per opportunity and per fill are reported separately and are NOT
interchangeable.

No trading code imports this module.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from oracle.ml import evidence as ev

SCHEMA_VERSION = 2

# ---------------------------------------------------------------------------
# Pre-registered gates (fixed before any final holdout was examined).
# Changing these constants after seeing results invalidates the report;
# the values are embedded in every emitted report for auditability.
# ``min_actual_fills`` formalizes the previously implicit
# min_resolved_opportunities // 2 == 100; the numeric requirement is
# unchanged, now explicit.
# ---------------------------------------------------------------------------
GATES_PREREGISTERED: Dict[str, Any] = {
    "min_forward_sessions": 60,
    "min_resolved_opportunities": 200,
    "min_actual_fills": 100,
    "expectancy_bootstrap_lb_gt": 0.0,
    "paired_uplift_bootstrap_lb_gt": 0.0,
    "net_positive_at_friction_mult": 1.5,
    "stress_friction_mult_examined": 2.0,
    "sharpe_daily_ann_gt": 1.0,
    "profit_factor_gt": 1.3,
    "max_drawdown_frac_limit": 0.20,   # portfolio limit (fraction of equity)
    "drawdown_no_worse_than_baseline": True,
    "calibration_must_improve": True,
    "coverage_disparity_max": 0.10,    # max coverage-rate gap across groups
    "bootstrap_iters": 2000,
    "bootstrap_seed": 7,
}

PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT"

TRADING_DAYS_PER_YEAR = 252

_NO_FE_NOTE = "forward evidence absent or failed validation"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flt(x: Any) -> Optional[float]:
    if isinstance(x, bool):
        return None
    try:
        v = float(x)
        return v if math.isfinite(v) else None
    except (TypeError, ValueError):
        return None


def _session_bootstrap_lb(daily: List[float], iters: int, seed: int) -> Optional[float]:
    """95% lower bound on mean daily P&L via seeded session bootstrap."""
    n = len(daily)
    if n < 2:
        return None
    rng = random.Random(seed)
    means = []
    for _ in range(iters):
        sample = [daily[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[max(0, int(0.025 * iters) - 1)]


def _equity_max_drawdown(daily: List[float], capital_base: float) -> Optional[float]:
    """Peak-to-trough drawdown (negative fraction) on the equity curve."""
    if not daily or capital_base <= 0:
        return None
    equity = capital_base
    peak = capital_base
    worst = 0.0
    for d in daily:
        equity += d
        peak = max(peak, equity)
        if peak > 0:
            worst = min(worst, (equity - peak) / peak)
    return worst


def _gate(name: str, requirement: str, observed: Any, status: str,
          note: str = "") -> Dict[str, Any]:
    return {"name": name, "requirement": requirement,
            "observed": observed, "status": status, "note": note}


# ---------------------------------------------------------------------------
# core evaluation
# ---------------------------------------------------------------------------

def evaluate_promotion(replay_artifact: Dict[str, Any],
                       execution_artifact: Optional[Dict[str, Any]] = None,
                       *,
                       variant: str = "combined",
                       baseline: str = "baseline_frozen",
                       forward_evidence: Optional[Dict[str, Any]] = None,
                       expected: Optional[Dict[str, Any]] = None,
                       ) -> Dict[str, Any]:
    """Evaluate the pre-registered gates for ``variant`` vs ``baseline``.

    ``forward_evidence`` must be a validated immutable artifact matching the
    ``oracle.ml.evidence`` schema; it is validated here against ``expected``
    bindings (model/hash/baseline/policy/dataset/window).  Invalid or absent
    evidence makes every forward gate INSUFFICIENT — replay numbers never
    substitute for forward evidence.
    """
    g = GATES_PREREGISTERED
    gates: List[Dict[str, Any]] = []

    # ---- validate the single forward-evidence artifact ------------------ #
    exp = dict(expected or {})
    exp.setdefault("baseline_id", baseline)  # wrong comparator => mismatch
    fe_supplied = forward_evidence is not None
    fe_fp = forward_evidence.get("fingerprint") if isinstance(forward_evidence, dict) else None
    if fe_supplied:
        fe_errors = ev.validate_forward_evidence(forward_evidence, expected=exp)
    else:
        fe_errors = ["no forward evidence artifact supplied"]
    fe_valid = fe_supplied and not fe_errors
    fe: Optional[Dict[str, Any]] = forward_evidence if fe_valid else None

    # derived quantities from the validated artifact only
    sessions_rows: List[Dict[str, Any]] = fe["sessions"] if fe else []
    daily = [float(r["challenger_net_pnl"]) for r in sessions_rows]
    base_daily = [float(r["baseline_net_pnl"]) for r in sessions_rows]
    resolved = sum(int(r["n_opportunities"]) for r in sessions_rows)
    fills = fe.get("fills", []) if fe else []
    live_fills = [f for f in fills if f.get("source") in ev.FILL_SOURCES_ACTUAL]
    n_live = len(live_fills)
    capital_base = float(fe["capital_base"]) if fe else None

    # replay context (never satisfies forward gates)
    replay_sessions = replay_artifact.get("sessions_replayed")
    if isinstance(replay_sessions, list):
        replay_sessions = len(replay_sessions)

    # ---- evidence volume ------------------------------------------------ #
    if fe is None:
        gates.append(_gate(
            "forward_sessions",
            f">= {g['min_forward_sessions']} forward sessions",
            replay_sessions, INSUFFICIENT,
            _NO_FE_NOTE + "; replay sessions shown for context only — the "
            "gate counts FORWARD sessions"))
    else:
        n_sessions = len(sessions_rows)
        gates.append(_gate(
            "forward_sessions",
            f">= {g['min_forward_sessions']} forward sessions",
            n_sessions,
            PASS if n_sessions >= g["min_forward_sessions"] else INSUFFICIENT,
            "counted from validated per-session rows"))

    if fe is None:
        gates.append(_gate(
            "resolved_opportunities",
            f">= {g['min_resolved_opportunities']} resolved opportunities",
            None, INSUFFICIENT, _NO_FE_NOTE))
    else:
        gates.append(_gate(
            "resolved_opportunities",
            f">= {g['min_resolved_opportunities']} resolved opportunities",
            resolved,
            PASS if resolved >= g["min_resolved_opportunities"] else INSUFFICIENT,
            "summed from validated per-session rows"))

    if fe is None:
        gates.append(_gate(
            "actual_fills_effective_n",
            f">= {g['min_actual_fills']} actual fills with effective sample "
            "size reported",
            None, INSUFFICIENT,
            _NO_FE_NOTE + "; paper marks and kernel-shadow intents carry no "
            "fill evidence — only genuine order outcomes count"))
    else:
        live_per_session: Dict[str, int] = {}
        for f in live_fills:
            live_per_session[f["session"]] = live_per_session.get(f["session"], 0) + 1
        ess = ev.kish_ess(list(live_per_session.values()))
        gates.append(_gate(
            "actual_fills_effective_n",
            f">= {g['min_actual_fills']} actual fills with effective sample "
            "size reported",
            {"actual_fills": n_live,
             "effective_sample_size": round(ess, 2),
             "sessions_with_live_fills": len(live_per_session)},
            PASS if n_live >= g["min_actual_fills"] else INSUFFICIENT,
            "actual fills = live order outcomes only (source='live_order'); "
            "ESS is Kish over per-session live-fill counts (clustering)"))

    # ---- expectancy (session bootstrap on forward daily net P&L) -------- #
    if fe is None:
        gates.append(_gate(
            "expectancy_bootstrap_lb",
            f"session-bootstrap 95% lower bound of daily net P&L > "
            f"{g['expectancy_bootstrap_lb_gt']}",
            None, INSUFFICIENT, _NO_FE_NOTE))
        lb = None
    else:
        lb = _session_bootstrap_lb(daily, g["bootstrap_iters"], g["bootstrap_seed"])
        gates.append(_gate(
            "expectancy_bootstrap_lb",
            f"session-bootstrap 95% lower bound of daily net P&L > "
            f"{g['expectancy_bootstrap_lb_gt']}",
            lb,
            INSUFFICIENT if lb is None
            else (PASS if lb > g["expectancy_bootstrap_lb_gt"] else FAIL),
            f"n_days={len(daily)}; seeded bootstrap "
            f"({g['bootstrap_iters']} iters, seed {g['bootstrap_seed']})"))

    # ---- paired uplift vs baseline (clustered uncertainty) --------------- #
    if fe is None:
        gates.append(_gate(
            "paired_uplift_bootstrap_lb",
            f"paired incremental daily profit vs {baseline} with bootstrap "
            f"95% lower bound > {g['paired_uplift_bootstrap_lb_gt']}",
            None, INSUFFICIENT, _NO_FE_NOTE))
    else:
        deltas = [c - b for c, b in zip(daily, base_daily)]
        uplift_lb = _session_bootstrap_lb(
            deltas, g["bootstrap_iters"], g["bootstrap_seed"])
        nonzero = [d for d in deltas if d != 0.0]
        ess_delta = ev.kish_ess([abs(d) for d in nonzero])
        gates.append(_gate(
            "paired_uplift_bootstrap_lb",
            f"paired incremental daily profit vs {baseline} with bootstrap "
            f"95% lower bound > {g['paired_uplift_bootstrap_lb_gt']}",
            {"ci_lo": uplift_lb,
             "mean": (sum(deltas) / len(deltas)) if deltas else None,
             "n_sessions": len(deltas),
             "n_nonzero_delta_sessions": len(nonzero),
             "effective_sample_size_abs_delta": round(ess_delta, 2)},
            INSUFFICIENT if uplift_lb is None
            else (PASS if uplift_lb > g["paired_uplift_bootstrap_lb_gt"] else FAIL),
            "per-session paired deltas from the same validated artifact; "
            "ESS is Kish over |delta| (clustered uncertainty)"))

    # ---- friction gates --------------------------------------------------- #
    stress = (fe.get("stress") or {}) if fe else {}
    net15 = _flt(stress.get("net_pnl_1_5x")) if fe else None
    gates.append(_gate(
        "net_positive_at_1_5x_friction",
        "net P&L > 0 at 1.5x friction",
        net15,
        INSUFFICIENT if net15 is None else (PASS if net15 > 0 else FAIL),
        "" if fe else _NO_FE_NOTE))

    net20 = _flt(stress.get("net_pnl_2_0x")) if fe else None
    overshoot = _flt(stress.get("stop_overshoot_net_pnl")) if fe else None
    both_documented = net20 is not None and overshoot is not None
    gates.append(_gate(
        "stress_2x_and_overshoot_examined",
        "BOTH 2x friction AND stop-overshoot stress documented",
        {"net_pnl_2x": net20, "stop_overshoot_net_pnl": overshoot},
        PASS if both_documented else INSUFFICIENT,
        ("examination gate: both stresses must be documented for human "
         "review; either one missing is insufficient (an 'or' here was a "
         "defect — fixed)") if fe else _NO_FE_NOTE))

    # ---- performance gates (equity returns, live fills) ------------------- #
    if fe is None or len(daily) < 2:
        sharpe = None
        sharpe_note = _NO_FE_NOTE if fe is None else "fewer than 2 sessions"
    else:
        returns = [d / capital_base for d in daily]
        mean_r = sum(returns) / len(returns)
        var = sum((r - mean_r) ** 2 for r in returns) / (len(returns) - 1)
        sd = math.sqrt(var)
        sharpe = (mean_r / sd) * math.sqrt(TRADING_DAYS_PER_YEAR) if sd > 0 else None
        sharpe_note = ("computed on equity returns (daily P&L / capital_base), "
                       "annualized sqrt(252)")
    gates.append(_gate(
        "sharpe_daily_ann",
        f"daily Sharpe on equity returns (annualized) > {g['sharpe_daily_ann_gt']}",
        sharpe,
        INSUFFICIENT if sharpe is None
        else (PASS if sharpe > g["sharpe_daily_ann_gt"] else FAIL),
        sharpe_note))

    if fe is None:
        gates.append(_gate(
            "profit_factor",
            f"profit factor > {g['profit_factor_gt']} computed from actual fills",
            None, INSUFFICIENT, _NO_FE_NOTE))
    elif not live_fills:
        gates.append(_gate(
            "profit_factor",
            f"profit factor > {g['profit_factor_gt']} computed from actual fills",
            None, INSUFFICIENT,
            "no live order fills in evidence; profit factor from marks or "
            "counterfactuals does not count"))
    else:
        wins = sum(f["net_pnl"] for f in live_fills if f["net_pnl"] > 0)
        losses = sum(-f["net_pnl"] for f in live_fills if f["net_pnl"] < 0)
        if losses == 0.0:
            pf_obs: Any = {"wins": wins, "losses": 0.0, "profit_factor": None}
            pf_status = PASS if wins > 0 else INSUFFICIENT
            pf_note = ("no losing fills in sample; ratio undefined — passes "
                       "only with positive wins" if wins > 0
                       else "no wins and no losses: degenerate sample")
        else:
            pf = wins / losses
            pf_obs = {"wins": wins, "losses": losses, "profit_factor": pf}
            pf_status = PASS if pf > g["profit_factor_gt"] else FAIL
            pf_note = f"computed from {len(live_fills)} live fills"
        gates.append(_gate(
            "profit_factor",
            f"profit factor > {g['profit_factor_gt']} computed from actual fills",
            pf_obs, pf_status, pf_note))

    # ---- drawdown ---------------------------------------------------------- #
    v_dd = _equity_max_drawdown(daily, capital_base) if fe else None
    b_dd = _equity_max_drawdown(base_daily, capital_base) if fe else None
    if fe is None:
        dd_status, dd_note = INSUFFICIENT, _NO_FE_NOTE
    elif v_dd is None:
        dd_status, dd_note = INSUFFICIENT, "variant drawdown unavailable"
    elif b_dd is None:
        dd_status = INSUFFICIENT
        dd_note = ("baseline drawdown unavailable; 'no worse than baseline' "
                   "cannot pass vacuously — insufficient")
    else:
        within = abs(v_dd) <= g["max_drawdown_frac_limit"]
        no_worse = v_dd >= b_dd  # both negative fractions
        dd_status = PASS if (within and no_worse) else FAIL
        dd_note = (f"within_limit({g['max_drawdown_frac_limit']})={within}, "
                   f"no_worse_than_baseline={no_worse} (baseline={b_dd})")
    gates.append(_gate(
        "max_drawdown",
        f"|max drawdown| <= {g['max_drawdown_frac_limit']} of equity AND no "
        "worse than the comparable baseline",
        {"variant": v_dd, "baseline": b_dd}, dd_status, dd_note))

    # ---- calibration (recomputed, not a trusted boolean) ------------------- #
    cal = (fe.get("calibration") or {}) if fe else {}
    ch_brier = _flt((cal.get("challenger") or {}).get("brier"))
    ba_brier = _flt((cal.get("baseline") or {}).get("brier"))
    if fe is None:
        gates.append(_gate(
            "calibration_improved",
            "forecast calibration improves vs baseline on forward data "
            "(lower brier)", None, INSUFFICIENT, _NO_FE_NOTE))
    elif ch_brier is None or ba_brier is None:
        gates.append(_gate(
            "calibration_improved",
            "forecast calibration improves vs baseline on forward data "
            "(lower brier)",
            {"challenger_brier": ch_brier, "baseline_brier": ba_brier},
            INSUFFICIENT, "calibration block absent or incomplete"))
    else:
        improved = ch_brier < ba_brier
        gates.append(_gate(
            "calibration_improved",
            "forecast calibration improves vs baseline on forward data "
            "(lower brier)",
            {"challenger_brier": ch_brier, "baseline_brier": ba_brier},
            PASS if improved else FAIL,
            "recomputed comparison of brier scores from the validated "
            "artifact; a reported boolean is never trusted"))

    # ---- coverage disparity (recomputed from per-group counts) ------------- #
    cov_groups = ((fe.get("coverage") or {}).get("groups") or {}) if fe else {}
    rates = {name: grp["n_predictions"] / grp["n_opportunities"]
             for name, grp in cov_groups.items()
             if isinstance(grp, dict) and grp.get("n_opportunities")}
    if fe is None:
        gates.append(_gate(
            "coverage_disparity",
            f"max group coverage disparity <= {g['coverage_disparity_max']}",
            None, INSUFFICIENT, _NO_FE_NOTE))
    elif not rates:
        gates.append(_gate(
            "coverage_disparity",
            f"max group coverage disparity <= {g['coverage_disparity_max']}",
            None, INSUFFICIENT, "coverage block absent or has no usable groups"))
    else:
        disparity = max(rates.values()) - min(rates.values())
        gates.append(_gate(
            "coverage_disparity",
            f"max group coverage disparity <= {g['coverage_disparity_max']}",
            {"disparity": disparity,
             "rates": {k: round(v, 4) for k, v in rates.items()}},
            PASS if disparity <= g["coverage_disparity_max"] else FAIL,
            "recomputed from per-group opportunity/prediction counts"))

    # ---- evidence basis ------------------------------------------------------ #
    universe = replay_artifact.get("trade_universe", {}) or {}
    outcome_counts = universe.get("outcome_status_counts", {}) or {}
    mark_derived = outcome_counts.get("mark_derived")
    if fe is None:
        basis_status = INSUFFICIENT
        basis_obs: Any = {"mark_derived_replay_outcomes": mark_derived}
        basis_note = (_NO_FE_NOTE + "; replay outcomes are mark-derived paper "
                      "data — promotion may not rest on mark/synthetic evidence")
    else:
        src_counts: Dict[str, int] = {}
        for f in fills:
            src = f.get("source")
            src_counts[src] = src_counts.get(src, 0) + 1
        non_live = {k: v for k, v in src_counts.items()
                    if k not in ev.FILL_SOURCES_ACTUAL}
        basis_obs = {"fill_source_counts": src_counts,
                     "mark_derived_replay_outcomes": mark_derived}
        if not fills:
            basis_status = INSUFFICIENT
            basis_note = "no fills in evidence; basis cannot be established"
        elif non_live:
            basis_status = FAIL
            basis_note = ("paper/mark/counterfactual outcomes present "
                          f"({non_live}); they must not satisfy live-fill "
                          "promotion — a separately authorized trial would "
                          "collect its own policy-specific evidence")
        else:
            basis_status = PASS
            basis_note = "all fills are live order outcomes"
    gates.append(_gate(
        "evidence_basis",
        "compliance must rest on actual live order fills, never mark-derived, "
        "paper or counterfactual data",
        basis_obs, basis_status, basis_note))

    # ---- replay-side invariants ---------------------------------------------- #
    parity_ok = bool((replay_artifact.get("parity") or {}).get("ok"))
    gates.append(_gate(
        "accounting_parity",
        "replay accounting parity holds (portfolio engine vs independent sums)",
        {"parity_ok": parity_ok},
        PASS if parity_ok else FAIL))

    hr = replay_artifact.get("holdout_refusal") or {}
    hr_sessions = hr.get("sessions")
    hr_excluded = hr.get("decisions_excluded")
    hr_ok = (isinstance(hr_sessions, list) and len(hr_sessions) > 0
             and isinstance(hr_excluded, int)
             and not isinstance(hr_excluded, bool))
    gates.append(_gate(
        "holdout_refusal_demonstrated",
        "replay demonstrably refused registered holdout sessions "
        "(non-empty session list and an exclusion count)",
        {"sessions": hr_sessions, "decisions_excluded": hr_excluded},
        PASS if hr_ok else INSUFFICIENT,
        "parity alone is not enough; holdout refusal must be demonstrated"))

    # ---- verdict --------------------------------------------------------------- #
    statuses = [x["status"] for x in gates]
    n_pass = statuses.count(PASS)
    n_fail = statuses.count(FAIL)
    n_insufficient = statuses.count(INSUFFICIENT)
    if not fe_valid or n_insufficient > 0:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = ("forward evidence is absent, invalid, or one or more gates "
                  "lack sufficient evidence; thresholds are never relaxed to "
                  "fit thin samples (known gate failures are still shown)")
    elif n_fail > 0:
        verdict = "DO_NOT_PROMOTE"
        reason = "sufficient evidence and at least one pre-registered gate failed"
    else:
        verdict = "ELIGIBLE_PENDING_HUMAN_REVIEW"
        reason = ("all pre-registered gates pass; promotion still requires a "
                  "separate named human review bound to this report's "
                  "fingerprint and an explicit registry action")

    # ---- estimands (registered explicitly; not interchangeable) --------------- #
    if fe is not None:
        total_pnl = sum(daily)
        live_pnl = sum(f["net_pnl"] for f in live_fills)
        estimands: Optional[Dict[str, Any]] = {
            "registered_primary": (
                "session-bootstrap 95% lower bound of per-session daily net "
                "P&L (challenger)"),
            "per_session_mean": total_pnl / len(daily) if daily else None,
            "session_ci_lb": lb,
            "session_ci_config": {"iters": g["bootstrap_iters"],
                                  "seed": g["bootstrap_seed"]},
            "per_opportunity_expectancy": (total_pnl / resolved) if resolved else None,
            "per_fill_expectancy": (live_pnl / n_live) if n_live else None,
            "note": ("per-opportunity and per-fill expectancies are reported "
                     "separately and are NOT interchangeable; the estimand "
                     "was registered before evaluation"),
        }
    else:
        estimands = None

    exec_note = None
    if execution_artifact is not None:
        exec_note = {
            "fingerprint": execution_artifact.get("fingerprint"),
            "provenance_note": execution_artifact.get("provenance_note"),
            "ev_calibration_slope": (execution_artifact.get("ev_calibration") or {}).get(
                "ols_slope_realized_on_predicted"),
            "limit_tp_fill_rate_raw": ((execution_artifact.get("fill_probability") or {})
                                       .get("limit_take_profit") or {}).get("raw_rate"),
        }

    report = {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "generated_for": "oracle4_phase7_promotion_report",
        "variant": variant,
        "baseline": baseline,
        "gates_preregistered": dict(g),
        "gates": gates,
        "n_pass": n_pass,
        "n_fail": n_fail,
        "n_insufficient": n_insufficient,
        "verdict": verdict,
        "verdict_reason": reason,
        "human_review_required": True,
        "no_auto_promotion": True,
        "replay_fingerprint": replay_artifact.get("fingerprint"),
        "forward_evidence_validation": {
            "present": fe_supplied,
            "valid": fe_valid,
            "errors": fe_errors,
        },
        "forward_evidence_fingerprint": fe_fp,
        "expected_bindings": exp,
        "estimands": estimands,
        "execution_calibration": exec_note,
        "note": ("this report never changes the registry, flags, or trading "
                 "behavior; promotion is registry.approve_model + "
                 "registry.set_champion performed by a human"),
        "human_review_protocol": (
            "an explicit named review (reviewer_name, decision='approved', "
            "report_fingerprint matching this report's fingerprint) is "
            "required; verify with oracle.ml.evidence.verify_human_review — "
            "a bare approved_by string is not sufficient"),
    }
    # fingerprint the report so a human review can bind to this exact version
    report["fingerprint"] = ev.canonical_fingerprint(report)
    return report


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Oracle4 promotion report")
    ap.add_argument("--replay", default=os.path.join("artifacts", "oracle4", "replay.json"))
    ap.add_argument("--execution",
                    default=os.path.join("artifacts", "oracle4", "execution_calibration.json"))
    ap.add_argument("--variant", default="combined")
    ap.add_argument("--baseline", default="baseline_frozen")
    ap.add_argument("--forward-evidence", default=None,
                    help="JSON file: validated immutable forward-evidence artifact")
    ap.add_argument("--expected", default=None,
                    help="optional JSON file with expected bindings "
                         "(model_id/model_hash/baseline_id/policy_id/"
                         "dataset_fingerprint/window_start/window_end)")
    ap.add_argument("--out", default=os.path.join("artifacts", "oracle4",
                                                  "promotion_report.json"))
    args = ap.parse_args(argv)

    replay = _load_json(args.replay)
    execution = _load_json(args.execution) if os.path.exists(args.execution) else None
    fe = _load_json(args.forward_evidence) if args.forward_evidence else None
    expected = _load_json(args.expected) if args.expected else None

    report = evaluate_promotion(replay, execution, variant=args.variant,
                                baseline=args.baseline, forward_evidence=fe,
                                expected=expected)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps({
        "verdict": report["verdict"],
        "verdict_reason": report["verdict_reason"],
        "n_pass": report["n_pass"],
        "n_fail": report["n_fail"],
        "n_insufficient": report["n_insufficient"],
        "forward_evidence_valid": report["forward_evidence_validation"]["valid"],
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
