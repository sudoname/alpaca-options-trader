"""One read-only research run with explicit evidence and no stale fallback.

The GBM comparison requires the original point-in-time dataset and measured
execution calibration. Missing inputs produce INSUFFICIENT_DATA and never
reuse the pre-correction replay402.json or a hard-coded profit claim.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from typing import Any, Dict, Optional


def run_corrected_gbm(dataset: str, calibration: str, out_dir: str,
                      valuation: Optional[str] = None,
                      *, n_boot: int = 2000) -> Dict[str, Any]:
    required = {"dataset": dataset, "execution_calibration": calibration}
    if valuation:
        required["valuation"] = valuation
    missing = [name for name, path in required.items() if not os.path.isfile(path)]
    if missing:
        return {"status": "INSUFFICIENT_DATA", "missing_inputs": missing,
                "reason": "No historical report is substituted for a missing input"}

    with open(calibration, encoding="utf-8") as fh:
        calibrated = json.load(fh)
    entry_mean = (calibrated.get("slippage", {}).get("entry_vs_mid", {})
                  .get("quantiles", {}).get("mean"))
    overshoot = (calibrated.get("stop_overshoot", {})
                 .get("dynamic_stop_loss", {}).get("overshoot_quantiles_pts", {}))
    try:
        measured = [float(entry_mean), float(overshoot["p50"]),
                    float(overshoot["p90"])]
    except (TypeError, ValueError, KeyError):
        measured = []
    if (len(measured) != 3 or not all(math.isfinite(x) for x in measured)
            or measured[0] < 0 or measured[2] < measured[1]):
        return {"status": "INSUFFICIENT_DATA",
                "missing_inputs": ["measured_execution_calibration"],
                "reason": "The replay silently defaults missing calibration fields"}

    from oracle.ml import gbm
    if not gbm.LGBM_AVAILABLE or gbm.LGBM_VERSION != gbm.LGBM_PINNED:
        return {"status": "DEPENDENCY_UNAVAILABLE", "reason": gbm.LGBM_WHY}

    from oracle.ml.backtest402 import run_backtest402
    from oracle.ml.promotion402 import run_promotion
    os.makedirs(out_dir, exist_ok=True)
    replay_path = os.path.join(out_dir, "replay402_corrected.json")
    promotion_path = os.path.join(out_dir, "promotion402_corrected.json")
    replay = run_backtest402(
        dataset, valuation_path=valuation,
        execution_calibration_path=calibration,
        out_path=replay_path, n_boot=n_boot,
        include_gbm=True, include_neural=False)
    gate = (replay.get("variants", {}).get("gbm_gated") or {})
    acct = gate.get("accounting") or {}
    eligible = replay.get("trade_universe", {}).get("trades_kept")
    counted = sum(acct.get(k, 0) for k in
                  ("gate_passed", "gated_out", "abstained", "no_prediction"))
    if (not replay.get("parity", {}).get("ok") or gate.get("status") != "ok"
            or eligible is None or eligible != counted
            or acct.get("n") != acct.get("gate_passed")
            or not replay.get("input_crosschecks", {}).get("ok")):
        return {"status": "INVALID_REPLAY", "replay_path": replay_path,
                "eligible": eligible, "gate_accounting": acct,
                "reason": "Parity, model coverage, or fail-closed gate accounting failed"}
    promotion = run_promotion(replay_path, out_path=promotion_path)
    gbm_10 = gate["by_friction_mult"]["1.0"]
    gbm_15 = gate["by_friction_mult"]["1.5"]
    baseline = replay["variants"]["baseline_frozen"]["by_friction_mult"]["1.0"]
    paired = (replay.get("paired_vs_baseline", {}).get("gbm_gated") or {})
    from oracle.lab.purged_validation import paired_session_bootstrap
    paired_days = paired.get("per_session_delta") or {}
    calendar_days = set(replay.get("sessions_replayed") or []) | set(paired_days)
    calendar_paired = paired_session_bootstrap(
        {day: paired_days.get(day, 0.0) for day in calendar_days},
        n_boot=n_boot, seed=42)
    sources = (replay.get("trade_universe", {}).get("outcome_status_counts") or {})
    result = {
        "status": "COMPLETE",
        "fingerprints": {"replay": replay["fingerprint"],
                         "promotion": promotion.get("report_fingerprint")},
        "replay_path": replay_path, "promotion_path": promotion_path,
        "sessions": replay["session_calendar"]["n_sessions"],
        "eligible_opportunities": eligible,
        "replayed_outcome_source": "mark_derived_only",
        "dataset_outcome_status_counts": sources,
        "broker_fill_backed_in_replay": 0,
        "gate_accounting": acct,
        "baseline": {"n_modelled_executions": baseline["n_executed"],
                     "net_pnl": baseline["net_pnl"]},
        "gbm": {"n_modelled_executions": gbm_10["n_executed"],
                "net_pnl_1x": gbm_10["net_pnl"],
                "net_pnl_1_5x": gbm_15["net_pnl"],
                "profit_factor": gbm_10["profit_factor"],
                "expectancy_per_trade": gbm_10["expectancy_per_trade"]},
        "paired_bootstrap": paired.get("bootstrap"),
        "paired_calendar_including_zero_days": calendar_paired,
        "promotion": (promotion.get("variants") or {}).get("gbm_gated"),
        "limitation": "All replayed outcomes are mark-derived. n_executed "
                      "counts modeled portfolio orders, not broker fills. "
                      "The promotion E3 field inherits that proxy and cannot "
                      "establish its 100 fill-backed execution floor.",
    }
    return result


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--execution-calibration", required=True)
    parser.add_argument("--valuation")
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--n-boot", type=int, default=2000)
    args = parser.parse_args(argv)
    result = run_corrected_gbm(args.dataset, args.execution_calibration,
                               args.out_dir, args.valuation, n_boot=args.n_boot)
    os.makedirs(args.out_dir, exist_ok=True)
    with open(os.path.join(args.out_dir, "profitability_verdict.json"), "w",
              encoding="utf-8") as fh:
        json.dump(result, fh, indent=2, sort_keys=True)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0 if result["status"] == "COMPLETE" else 2


if __name__ == "__main__":
    raise SystemExit(main())
