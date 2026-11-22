"""Promotion gate for Oracle 4.0.2 (P11).

Reads a comprehensive-backtest artifact written by
``oracle.ml.backtest402`` and scores every scoreable challenger variant
against the frozen promotion criteria from the 4.0.2 charter.  This
module NEVER promotes anything: the best possible verdict is
``ELIGIBLE_PENDING_HUMAN_REVIEW`` -- a human reads the report and
decides.

Criteria (all must hold for eligibility):
  * ``C1_positive_expectancy``   expectancy/trade > 0 at calibrated 1.0x
  * ``C2_positive_incremental``  paired bootstrap mean vs frozen baseline > 0
  * ``C3_ci_lb_positive``        paired 95% CI lower bound > 0
  * ``C4_profit_factor``         PF > 1.3 at 1.0x
  * ``C5_sharpe``                annualized daily Sharpe > 1.0 at 1.0x
  * ``C6_survives_1_5x``         net P&L > 0 at 1.5x friction
  * ``E1_forward_sessions``      >= 60 replay sessions
  * ``E2_resolved_opportunities`` >= 200 resolved trades in the universe
  * ``E3_qualified_fills``       >= 100 executed fills at 1.0x

Verdict ladder (first match wins):
  * ``DO_NOT_PROMOTE``            paired CI upper bound <= 0
                                  (confidently *harmful* vs baseline)
  * ``INSUFFICIENT_EVIDENCE``     any evidence criterion (E1-E3) unmet
  * ``NO_VALIDATED_EDGE``         evidence met, a performance criterion fails
  * ``ELIGIBLE_PENDING_HUMAN_REVIEW``  every criterion passes

A variant with ``status != "ok"`` or without a paired-vs-baseline entry
is reported as ``NOT_SCOREABLE`` and never ranked.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from typing import Any, Dict, List, Optional

MIN_FORWARD_SESSIONS = 60
MIN_RESOLVED_OPPORTUNITIES = 200
MIN_QUALIFIED_FILLS = 100
MIN_PROFIT_FACTOR = 1.3
MIN_SHARPE = 1.0

VERDICTS = (
    "DO_NOT_PROMOTE",
    "INSUFFICIENT_EVIDENCE",
    "NO_VALIDATED_EDGE",
    "ELIGIBLE_PENDING_HUMAN_REVIEW",
)

NEVER_AUTO_PROMOTE_NOTE = (
    "this report never flips a production flag; the strongest possible "
    "outcome is ELIGIBLE_PENDING_HUMAN_REVIEW and a human makes the call")


def _crit(name: str, value: Optional[float], threshold: float,
          passed: Optional[bool]) -> Dict[str, Any]:
    return {"name": name, "value": value, "threshold": threshold,
            "pass": passed}


def evaluate_variant(artifact: Dict[str, Any], name: str
                     ) -> Dict[str, Any]:
    """Score one variant; returns verdict + per-criterion detail.

    Missing metrics (None profit factor, absent friction cell, ...)
    fail their criterion -- absence of evidence is never treated as a
    pass.
    """
    v = (artifact.get("variants") or {}).get(name) or {}
    paired = (artifact.get("paired_vs_baseline") or {}).get(name) or {}
    boot = paired.get("bootstrap") or {}
    if v.get("status") != "ok" or not boot:
        return {
            "variant": name,
            "verdict": "NOT_SCOREABLE",
            "why": ("variant unavailable in backtest artifact"
                    if v.get("status") != "ok"
                    else "no paired-vs-baseline bootstrap recorded"),
        }

    cell_10 = (v.get("by_friction_mult") or {}).get("1.0") or {}
    cell_15 = (v.get("by_friction_mult") or {}).get("1.5") or {}
    calendar = artifact.get("session_calendar") or {}
    universe = artifact.get("trade_universe") or {}

    exp = cell_10.get("expectancy_per_trade")
    pf = cell_10.get("profit_factor")
    sharpe = cell_10.get("sharpe_daily_ann")
    net_15 = cell_15.get("net_pnl")
    mean = boot.get("mean")
    ci_lo = boot.get("ci_lo")
    ci_hi = boot.get("ci_hi")
    n_sessions = calendar.get("n_sessions")
    n_resolved = universe.get("trades_kept")
    n_fills = cell_10.get("n_executed")

    def _gt(x: Optional[float], thr: float) -> Optional[bool]:
        return None if x is None else bool(x > thr)

    performance = [
        _crit("C1_positive_expectancy", exp, 0.0, _gt(exp, 0.0)),
        _crit("C2_positive_incremental", mean, 0.0, _gt(mean, 0.0)),
        _crit("C3_ci_lb_positive", ci_lo, 0.0, _gt(ci_lo, 0.0)),
        _crit("C4_profit_factor", pf, MIN_PROFIT_FACTOR,
              _gt(pf, MIN_PROFIT_FACTOR)),
        _crit("C5_sharpe", sharpe, MIN_SHARPE, _gt(sharpe, MIN_SHARPE)),
        _crit("C6_survives_1_5x", net_15, 0.0, _gt(net_15, 0.0)),
    ]
    evidence = [
        _crit("E1_forward_sessions", n_sessions, MIN_FORWARD_SESSIONS,
              None if n_sessions is None
              else bool(n_sessions >= MIN_FORWARD_SESSIONS)),
        _crit("E2_resolved_opportunities", n_resolved,
              MIN_RESOLVED_OPPORTUNITIES,
              None if n_resolved is None
              else bool(n_resolved >= MIN_RESOLVED_OPPORTUNITIES)),
        _crit("E3_qualified_fills", n_fills, MIN_QUALIFIED_FILLS,
              None if n_fills is None
              else bool(n_fills >= MIN_QUALIFIED_FILLS)),
    ]
    criteria = performance + evidence
    failed = [c["name"] for c in criteria if c["pass"] is not True]

    if ci_hi is not None and ci_hi <= 0.0:
        verdict = "DO_NOT_PROMOTE"
        why = ("paired 95% CI upper bound <= 0: the variant is "
               "confidently WORSE than the frozen baseline")
    elif any(c["pass"] is not True for c in evidence):
        verdict = "INSUFFICIENT_EVIDENCE"
        why = ("evidence thresholds unmet (sessions/opportunities/fills); "
               "no performance claim can be validated on this sample")
    elif failed:
        verdict = "NO_VALIDATED_EDGE"
        why = f"performance criteria failed: {', '.join(failed)}"
    else:
        verdict = "ELIGIBLE_PENDING_HUMAN_REVIEW"
        why = "all criteria pass; promotion still requires a human decision"

    return {
        "variant": name,
        "verdict": verdict,
        "why": why,
        "criteria": criteria,
        "failed": failed,
        "paired_bootstrap": dict(boot),
    }


def run_promotion(artifact_path: str,
                  out_path: Optional[str] = None) -> Dict[str, Any]:
    """Score every variant in the backtest artifact; write report."""
    with open(artifact_path, "r", encoding="utf-8") as fh:
        artifact = json.load(fh)

    names = sorted((artifact.get("variants") or {}).keys())
    results = {n: evaluate_variant(artifact, n) for n in names
               if n != "baseline_frozen"}

    scoreable = {n: r for n, r in results.items()
                 if r["verdict"] != "NOT_SCOREABLE"}
    best = None
    if scoreable:
        best = max(scoreable,
                   key=lambda n: scoreable[n]["paired_bootstrap"]
                   .get("mean") or float("-inf"))

    report: Dict[str, Any] = {
        "generated_for": "oracle4.0.2_promotion",
        "phase": 11,
        "inputs": {
            "backtest_artifact": os.path.abspath(artifact_path),
            "backtest_fingerprint": artifact.get("fingerprint"),
        },
        "criteria_spec": {
            "min_forward_sessions": MIN_FORWARD_SESSIONS,
            "min_resolved_opportunities": MIN_RESOLVED_OPPORTUNITIES,
            "min_qualified_fills": MIN_QUALIFIED_FILLS,
            "min_profit_factor": MIN_PROFIT_FACTOR,
            "min_sharpe": MIN_SHARPE,
            "verdicts": list(VERDICTS),
        },
        "risk_metrics_limited": artifact.get("risk_metrics_limited"),
        "matched_cohorts_summary": {
            g: (c.get("paired") or {}).get("bootstrap")
            for g, c in (artifact.get("matched_cohorts") or {}).items()
            if isinstance(c, dict)
        },
        "variants": results,
        "best_variant": best,
        "overall_verdict": (scoreable[best]["verdict"] if best
                            else "INSUFFICIENT_EVIDENCE"),
        "never_auto_promote": NEVER_AUTO_PROMOTE_NOTE,
    }
    blob = json.dumps(report, sort_keys=True).encode("utf-8")
    report["report_fingerprint"] = hashlib.sha256(blob).hexdigest()

    if out_path:
        d = os.path.dirname(os.path.abspath(out_path))
        if d:
            os.makedirs(d, exist_ok=True)
        tmp = f"{out_path}.tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=1, sort_keys=True)
        os.replace(tmp, out_path)
    return report


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        description="Oracle 4.0.2 promotion gate (P11): score backtest "
                    "variants against frozen criteria; never promotes.")
    ap.add_argument("--artifact", required=True,
                    help="replay402.json from oracle.ml.backtest402")
    ap.add_argument("--out", default=None, help="write report JSON here")
    args = ap.parse_args(argv)

    report = run_promotion(args.artifact, out_path=args.out)
    for name, r in sorted(report["variants"].items()):
        if r["verdict"] == "NOT_SCOREABLE":
            print(f"{name:18s} NOT_SCOREABLE ({r['why']})")
            continue
        boot = r["paired_bootstrap"]
        n_fail = len(r["failed"])
        print(f"{name:18s} {r['verdict']:28s} "
              f"delta_mean={boot.get('mean')} "
              f"ci=[{boot.get('ci_lo')}, {boot.get('ci_hi')}] "
              f"failed={n_fail}")
    print(f"best_variant={report['best_variant']} "
          f"overall={report['overall_verdict']}")
    print(f"report_fingerprint={report['report_fingerprint']}")
    if args.out:
        print(f"written: {args.out}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
