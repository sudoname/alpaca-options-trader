"""Oracle4 Phase 7c: promotion report with pre-registered gates.

Evaluates a challenger variant from the Phase 6 paired replay artifact
against gates that were fixed *before* looking at any final holdout:

* evidence volume: >=60 forward sessions and >=200 resolved opportunities;
* expectancy: session-bootstrap 95% lower bound of daily net P&L > 0;
* paired uplift: incremental profit vs the frozen baseline with a
  session-bootstrap lower bound > 0;
* friction: still net-positive at 1.5x friction; 2x stress examined;
* daily OOS Sharpe > 1 and profit factor > 1.3;
* drawdown within limits and no worse than baseline on matched exposure;
* improved forecast calibration (forward shadow evidence);
* no material group/coverage disparity (forward shadow evidence);
* evidence basis: compliance may not rest on mark-derived / synthetic data.

Hard rules, enforced structurally:

* thin samples => verdict ``INSUFFICIENT_EVIDENCE`` — thresholds are never
  relaxed to fit the data;
* any failed gate with sufficient samples => ``DO_NOT_PROMOTE``;
* the best possible verdict is ``ELIGIBLE_PENDING_HUMAN_REVIEW``; there is
  no PROMOTE output and no code path that changes the registry, a flag, or
  any trading behavior.  Promotion itself is a separate human action
  (``registry.approve_model`` + ``registry.set_champion``).

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

SCHEMA_VERSION = 1

# ---------------------------------------------------------------------------
# Pre-registered gates (fixed before any final holdout was examined).
# Changing these constants after seeing results invalidates the report;
# the values are embedded in every emitted report for auditability.
# ---------------------------------------------------------------------------
GATES_PREREGISTERED: Dict[str, Any] = {
    "min_forward_sessions": 60,
    "min_resolved_opportunities": 200,
    "expectancy_bootstrap_lb_gt": 0.0,
    "paired_uplift_bootstrap_lb_gt": 0.0,
    "net_positive_at_friction_mult": 1.5,
    "stress_friction_mult_examined": 2.0,
    "sharpe_daily_ann_gt": 1.0,
    "profit_factor_gt": 1.3,
    "max_drawdown_frac_limit": 0.20,   # portfolio limit (fraction of equity)
    "drawdown_no_worse_than_baseline": True,
    "calibration_must_improve": True,
    "coverage_disparity_max": 0.10,    # max abstention-rate gap across groups
    "bootstrap_iters": 2000,
    "bootstrap_seed": 7,
}

PASS, FAIL, INSUFFICIENT = "PASS", "FAIL", "INSUFFICIENT"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: str) -> Dict[str, Any]:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _flt(x: Any) -> Optional[float]:
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
                       ) -> Dict[str, Any]:
    """Evaluate the pre-registered gates for ``variant`` vs ``baseline``.

    ``forward_evidence`` is the (future) forward-shadow summary; until a
    real forward run exists it is absent and the gates that depend on it
    report INSUFFICIENT.  Expected keys when present:
    ``forward_sessions`` (int), ``resolved_opportunities`` (int),
    ``actual_fills`` (int), ``calibration_improved`` (bool),
    ``coverage_disparity`` (float), ``live_fill_based`` (bool).
    """
    g = GATES_PREREGISTERED
    gates: List[Dict[str, Any]] = []
    fe = forward_evidence or {}

    variants = replay_artifact.get("variants", {})
    v = variants.get(variant, {})
    b = variants.get(baseline, {})
    v_by = v.get("by_friction_mult", {})
    b_by = b.get("by_friction_mult", {})
    v10 = v_by.get("1.0", {})
    b10 = b_by.get("1.0", {})
    v15 = v_by.get("1.5", {})
    v20 = v_by.get("2.0", {})

    # ---- evidence volume ---------------------------------------------- #
    fwd_sessions = fe.get("forward_sessions")
    replay_sessions = replay_artifact.get("sessions_replayed")
    if isinstance(replay_sessions, list):
        replay_sessions = len(replay_sessions)
    if fwd_sessions is None:
        obs_sessions = replay_sessions
        sess_note = ("no forward shadow evidence yet; replay sessions shown "
                     "for context only — the gate counts FORWARD sessions")
    else:
        obs_sessions = fwd_sessions
        sess_note = "forward shadow sessions"
    sess_ok = isinstance(obs_sessions, (int, float)) and fwd_sessions is not None \
        and obs_sessions >= g["min_forward_sessions"]
    gates.append(_gate(
        "forward_sessions",
        f">= {g['min_forward_sessions']} forward sessions",
        obs_sessions,
        PASS if sess_ok else INSUFFICIENT,
        sess_note))

    resolved = fe.get("resolved_opportunities")
    if resolved is None:
        resolved = v10.get("n_executed")
        res_note = ("no forward resolved opportunities; replay executed-trade "
                    "count shown for context only")
        res_forward = False
    else:
        res_note = "forward resolved opportunities"
        res_forward = True
    res_ok = res_forward and isinstance(resolved, (int, float)) \
        and resolved >= g["min_resolved_opportunities"]
    gates.append(_gate(
        "resolved_opportunities",
        f">= {g['min_resolved_opportunities']} resolved opportunities",
        resolved,
        PASS if res_ok else INSUFFICIENT,
        res_note))

    fills = fe.get("actual_fills")
    gates.append(_gate(
        "actual_fills_effective_n",
        "enough actual fills for execution calibration (effective sample "
        "size reported, not assumed)",
        fills,
        PASS if isinstance(fills, (int, float)) and fills >= g["min_resolved_opportunities"] // 2
        else INSUFFICIENT,
        "paper marks and kernel-shadow intents carry no fill evidence; only "
        "genuine order outcomes count"))

    samples_sufficient = all(
        x["status"] == PASS for x in gates)  # the three volume gates so far

    # ---- expectancy (session bootstrap on variant daily P&L) ----------- #
    pnl_by_day = v10.get("pnl_by_exit_day") or {}
    daily = [f for f in (_flt(x) for x in pnl_by_day.values()) if f is not None]
    lb = _session_bootstrap_lb(daily, g["bootstrap_iters"], g["bootstrap_seed"])
    gates.append(_gate(
        "expectancy_bootstrap_lb",
        f"session-bootstrap 95% lower bound of daily net P&L > "
        f"{g['expectancy_bootstrap_lb_gt']}",
        lb,
        INSUFFICIENT if lb is None else (PASS if lb > g["expectancy_bootstrap_lb_gt"] else FAIL),
        f"n_days={len(daily)}; seeded bootstrap "
        f"({g['bootstrap_iters']} iters, seed {g['bootstrap_seed']})"))

    # ---- paired uplift vs baseline ------------------------------------- #
    paired = (replay_artifact.get("paired_vs_baseline") or {}).get(variant, {})
    boot = paired.get("bootstrap") or {}
    uplift_lb = _flt(boot.get("ci_lo"))
    gates.append(_gate(
        "paired_uplift_bootstrap_lb",
        f"paired incremental daily profit vs {baseline} with bootstrap 95% "
        f"lower bound > {g['paired_uplift_bootstrap_lb_gt']}",
        {"ci_lo": uplift_lb, "ci_hi": _flt(boot.get("ci_hi")),
         "mean": _flt(boot.get("mean")), "n_sessions": boot.get("n_sessions")},
        INSUFFICIENT if uplift_lb is None
        else (PASS if uplift_lb > g["paired_uplift_bootstrap_lb_gt"] else FAIL)))

    # ---- friction gates ------------------------------------------------ #
    net15 = _flt(v15.get("net_pnl"))
    gates.append(_gate(
        "net_positive_at_1_5x_friction",
        "net P&L > 0 at 1.5x friction",
        net15,
        INSUFFICIENT if net15 is None else (PASS if net15 > 0 else FAIL)))

    net20 = _flt(v20.get("net_pnl"))
    stress = v.get("stop_overshoot_stress") or {}
    gates.append(_gate(
        "stress_2x_examined",
        "2x friction and stop-overshoot stress examined and reported",
        {"net_pnl_2x": net20, "stop_overshoot_stress": stress or None},
        PASS if (net20 is not None or stress) else INSUFFICIENT,
        "examination gate: results are reported for human review, not "
        "thresholded"))

    # ---- performance gates --------------------------------------------- #
    sharpe = _flt(v10.get("sharpe_daily_ann"))
    gates.append(_gate(
        "sharpe_daily_ann",
        f"daily OOS Sharpe (annualized) > {g['sharpe_daily_ann_gt']}",
        sharpe,
        INSUFFICIENT if sharpe is None else (PASS if sharpe > g["sharpe_daily_ann_gt"] else FAIL)))

    pf = _flt(v10.get("profit_factor"))
    gates.append(_gate(
        "profit_factor",
        f"profit factor > {g['profit_factor_gt']}",
        pf,
        INSUFFICIENT if pf is None else (PASS if pf > g["profit_factor_gt"] else FAIL)))

    # ---- drawdown ------------------------------------------------------- #
    v_dd = _flt(v10.get("max_drawdown"))
    b_dd = _flt(b10.get("max_drawdown"))
    if v_dd is None:
        dd_status, dd_note = INSUFFICIENT, "variant drawdown missing"
    else:
        within = abs(v_dd) <= g["max_drawdown_frac_limit"]
        no_worse = (b_dd is None) or (v_dd >= b_dd)  # both negative fractions
        dd_status = PASS if (within and no_worse) else FAIL
        dd_note = (f"within_limit({g['max_drawdown_frac_limit']})={within}, "
                   f"no_worse_than_baseline={no_worse} (baseline={b_dd})")
    gates.append(_gate(
        "max_drawdown",
        f"|max drawdown| <= {g['max_drawdown_frac_limit']} of equity and no "
        "worse than baseline on matched exposure",
        v_dd, dd_status, dd_note))

    # ---- forward-shadow-only gates -------------------------------------- #
    cal = fe.get("calibration_improved")
    gates.append(_gate(
        "calibration_improved",
        "forecast calibration improved vs baseline on forward data",
        cal,
        INSUFFICIENT if cal is None else (PASS if bool(cal) else FAIL),
        "requires forward shadow predicted-vs-realized evidence; the Phase 6 "
        "paper EV calibration (slope -0.266) is context, not forward proof"))

    disparity = _flt(fe.get("coverage_disparity"))
    gates.append(_gate(
        "coverage_disparity",
        f"max group coverage/abstention disparity <= {g['coverage_disparity_max']}",
        disparity,
        INSUFFICIENT if disparity is None
        else (PASS if disparity <= g["coverage_disparity_max"] else FAIL)))

    # ---- evidence basis -------------------------------------------------- #
    live_based = fe.get("live_fill_based")
    universe = replay_artifact.get("trade_universe", {}) or {}
    outcome_counts = universe.get("outcome_status_counts", {}) or {}
    mark_derived = outcome_counts.get("mark_derived")
    if live_based is None:
        basis_status = INSUFFICIENT
        basis_note = ("replay outcomes are mark-derived paper data "
                      f"(mark_derived={mark_derived}); promotion may not rest "
                      "on mark/synthetic evidence — forward fills required")
    else:
        basis_status = PASS if bool(live_based) else FAIL
        basis_note = "forward evidence basis as reported by shadow pipeline"
    gates.append(_gate(
        "evidence_basis",
        "compliance must not depend on mark-derived or synthetic data",
        {"live_fill_based": live_based, "mark_derived_outcomes": mark_derived},
        basis_status, basis_note))

    # ---- mandatory invariants -------------------------------------------- #
    parity = (replay_artifact.get("parity") or {})
    parity_ok = bool(parity.get("ok"))
    gates.append(_gate(
        "mandatory_invariants",
        "all mandatory replay invariants pass (accounting parity, no "
        "holdout leakage)",
        {"parity_ok": parity_ok,
         "holdout_refusal": replay_artifact.get("holdout_refusal")},
        PASS if parity_ok else FAIL))

    # ---- verdict ---------------------------------------------------------- #
    statuses = [x["status"] for x in gates]
    if not samples_sufficient or INSUFFICIENT in statuses:
        verdict = "INSUFFICIENT_EVIDENCE"
        reason = ("one or more gates lack sufficient evidence; thresholds are "
                  "never relaxed to fit thin samples")
    elif FAIL in statuses:
        verdict = "DO_NOT_PROMOTE"
        reason = "sufficient evidence and at least one pre-registered gate failed"
    else:
        verdict = "ELIGIBLE_PENDING_HUMAN_REVIEW"
        reason = ("all pre-registered gates pass; promotion still requires a "
                  "separate human review and an explicit registry action")

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

    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _utc_now(),
        "generated_for": "oracle4_phase7_promotion_report",
        "variant": variant,
        "baseline": baseline,
        "gates_preregistered": dict(g),
        "gates": gates,
        "n_pass": statuses.count(PASS),
        "n_fail": statuses.count(FAIL),
        "n_insufficient": statuses.count(INSUFFICIENT),
        "verdict": verdict,
        "verdict_reason": reason,
        "human_review_required": True,
        "no_auto_promotion": True,
        "replay_fingerprint": replay_artifact.get("fingerprint"),
        "execution_calibration": exec_note,
        "note": ("this report never changes the registry, flags, or trading "
                 "behavior; promotion is registry.approve_model + "
                 "registry.set_champion performed by a human"),
    }


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
                    help="optional JSON file with forward shadow evidence")
    ap.add_argument("--out", default=os.path.join("artifacts", "oracle4",
                                                  "promotion_report.json"))
    args = ap.parse_args(argv)

    replay = _load_json(args.replay)
    execution = _load_json(args.execution) if os.path.exists(args.execution) else None
    fe = _load_json(args.forward_evidence) if args.forward_evidence else None

    report = evaluate_promotion(replay, execution, variant=args.variant,
                                baseline=args.baseline, forward_evidence=fe)

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, sort_keys=True)

    print(json.dumps({
        "verdict": report["verdict"],
        "verdict_reason": report["verdict_reason"],
        "n_pass": report["n_pass"],
        "n_fail": report["n_fail"],
        "n_insufficient": report["n_insufficient"],
        "out": args.out,
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
