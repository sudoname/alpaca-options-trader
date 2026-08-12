"""
Oracle 3.0 — Phase-2 Upgrade G: offline promotion-check runner (ADVISORY ONLY).

This is the human-facing front end for ``oracle.promotion.evaluate_promotion``.
It assembles the two evidence blocks a promotion depends on and prints an
auditable PROMOTE / HOLD report for each shadow layer:

    execution-calibration ledger  (oracle.execution.calibration.load_records
        -> compute_calibration, + injected n_sessions)         == calibration_stats
    walk-forward study JSON        (oracle.lab.run_phase2_study output, the
        ``walk_forward`` block)                                 == lab_result
    thresholds from .env (PROMO_* keys, resolved via ConfigLoader)

CRITICAL SAFETY POSTURE — this runner is OFFLINE ADVISORY ONLY:
  * It NEVER edits ``.env``, flips a flag, or touches the live path. A human
    reads the report and flips the flag by hand (paper first). That is the whole
    point of Upgrade G: make the promotion decision auditable & reproducible.
  * It only READS a JSONL ledger and a study JSON, and (optionally, opt-in)
    sends the same text to Telegram. No trading, no order, no creds required to
    run the analysis. Exit code is always 0 on a successful report — a HOLD is a
    normal, expected outcome, not an error.

Usage:
    python run_promotion_check.py --lab-json oracle/lab/results/<id>__phase2_study.json
    python run_promotion_check.py --layer executable_ev --calibration-jsonl execution_calibration.jsonl
    python run_promotion_check.py --self-test
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional

from oracle.promotion import LAYERS, evaluate_promotion, format_promotion_report

# .env key -> internal threshold key consumed by evaluate_promotion.
_PROMO_ENV_MAP = {
    "PROMO_MIN_EVALS": "min_evals",
    "PROMO_MIN_SESSIONS": "min_sessions",
    "PROMO_MIN_OOS_TRADES": "min_oos_trades",
    "PROMO_MIN_MARGIN": "min_margin",
    "PROMO_MIN_CAPTURE": "min_capture",
    "PROMO_MIN_REJECT_PRECISION": "min_reject_precision",
    "PROMO_MAX_FILL_RATE_BIAS": "max_fill_rate_bias",
    "PROMO_MAX_ABS_SLIPPAGE_ERROR": "max_abs_slippage_error",
}


# --------------------------------------------------------------------------- #
# Threshold resolution (.env PROMO_* -> internal keys). Missing keys are simply
# left out, so oracle.promotion falls back to its conservative built-in default.
# --------------------------------------------------------------------------- #
def resolve_thresholds(env_path: str = ".env") -> Dict[str, float]:
    try:
        from config_loader import ConfigLoader
        cfg = ConfigLoader(path=env_path)
    except Exception as exc:
        print(f"[promotion_check] threshold load skipped ({exc}); using defaults")
        return {}
    out: Dict[str, float] = {}
    for env_key, thr_key in _PROMO_ENV_MAP.items():
        raw = cfg.get(env_key, None)
        if raw is None:
            continue
        try:
            out[thr_key] = float(raw)
        except (TypeError, ValueError):
            continue
    return out


# --------------------------------------------------------------------------- #
# Evidence assembly
# --------------------------------------------------------------------------- #
def _count_sessions(records: List[dict]) -> int:
    """Distinct trading sessions among RESOLVED records (date prefix of the
    resolution timestamp, falling back to the estimate timestamp). Pure."""
    days = set()
    for r in records or []:
        if not isinstance(r, dict) or r.get("resolution_status") != "resolved":
            continue
        ts = r.get("resolved_at") or r.get("recorded_at")
        if isinstance(ts, str) and len(ts) >= 10:
            days.add(ts[:10])
    return len(days)


def load_calibration_stats(jsonl_path: Optional[str]) -> Optional[dict]:
    """Fold the execution-calibration ledger into stats, injecting ``n_sessions``
    (which ``compute_calibration`` does not track). Fail-open -> None on error."""
    try:
        from oracle.execution.calibration import compute_calibration, load_records
        records = load_records(jsonl_path)
        if not records:
            return None
        stats = compute_calibration(records)
        stats["n_sessions"] = _count_sessions(records)
        return stats
    except Exception as exc:
        print(f"[promotion_check] calibration load failed: {exc}")
        return None


def load_lab_result(lab_json_path: Optional[str]) -> Optional[dict]:
    """Read a Phase-2 study JSON and return its ``walk_forward`` block (the
    ``WalkForwardResult.to_dict()`` shape ``evaluate_promotion`` judges on). If
    the file already IS a walk-forward dict, return it as-is. Fail-open -> None."""
    if not lab_json_path:
        return None
    try:
        with open(lab_json_path, "r", encoding="utf-8") as fh:
            blob = json.load(fh)
    except Exception as exc:
        print(f"[promotion_check] lab JSON load failed: {exc}")
        return None
    if not isinstance(blob, dict):
        return None
    wf = blob.get("walk_forward")
    if isinstance(wf, dict):
        return wf
    # Already a bare walk-forward result?
    if "oos_collapse" in blob or "oos_metrics" in blob:
        return blob
    return None


# --------------------------------------------------------------------------- #
# Report
# --------------------------------------------------------------------------- #
def build_report(calibration_stats: Optional[dict],
                 lab_result: Optional[dict],
                 thresholds: Dict[str, float],
                 layers=LAYERS) -> Dict[str, Any]:
    """Evaluate each requested layer and return decisions + rendered text.
    Pure over its inputs (no env, no clock, no I/O)."""
    decisions: Dict[str, dict] = {}
    blocks: List[str] = []
    for layer in layers:
        dec = evaluate_promotion(layer, calibration_stats, lab_result, thresholds)
        decisions[layer] = dec
        blocks.append(format_promotion_report(dec))
    header = ("===== Oracle promotion check (ADVISORY — does NOT edit .env) =====")
    text = header + "\n\n" + "\n\n".join(blocks)
    return {"decisions": decisions, "text": text}


def _maybe_telegram(text: str, enabled: bool) -> None:
    """Opt-in, best-effort Telegram echo of the report. Fail-open."""
    if not enabled:
        return
    try:
        from telegram_bot import TelegramTradingBot
        TelegramTradingBot().send_message(text)
        print("[promotion_check] telegram summary sent")
    except Exception as exc:
        print(f"[promotion_check] telegram send skipped: {exc}")


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[List[str]] = None) -> int:
    p = argparse.ArgumentParser(
        description="Offline Oracle promotion check (advisory; never edits .env)")
    p.add_argument("--layer", choices=list(LAYERS) + ["all"], default="all",
                   help="which shadow layer to evaluate (default: all)")
    p.add_argument("--calibration-jsonl", default=None,
                   help="execution-calibration ledger (default: EXECUTION_"
                        "CALIBRATION_JSONL from .env, else execution_calibration.jsonl)")
    p.add_argument("--lab-json", default=None,
                   help="Phase-2 study JSON (its walk_forward block is judged)")
    p.add_argument("--env", default=".env", help="path to the .env for PROMO_* thresholds")
    p.add_argument("--telegram", action="store_true",
                   help="also send the report to Telegram (opt-in, best-effort)")
    p.add_argument("--self-test", action="store_true", help="run offline self-test and exit")
    args = p.parse_args(argv)

    if args.self_test:
        return _self_test()

    thresholds = resolve_thresholds(args.env)
    calibration_stats = load_calibration_stats(args.calibration_jsonl)
    lab_result = load_lab_result(args.lab_json)

    if calibration_stats is None:
        print("[promotion_check] note: no calibration evidence "
              "(ledger empty/missing) -> every layer will HOLD.")
    if lab_result is None:
        print("[promotion_check] note: no walk-forward evidence "
              "(--lab-json missing/unreadable) -> every layer will HOLD.")

    layers = LAYERS if args.layer == "all" else (args.layer,)
    rep = build_report(calibration_stats, lab_result, thresholds, layers=layers)
    _safe_print(rep["text"])
    _maybe_telegram(rep["text"], args.telegram)

    promoted = [ly for ly, d in rep["decisions"].items() if d.get("promote")]
    _safe_print(f"\nsummary: {len(promoted)}/{len(rep['decisions'])} layer(s) "
                f"clear ALL gates: {promoted or 'none'}")
    # Advisory: a HOLD is a normal outcome, not a failure. Always exit 0 on a
    # report that was produced without an internal error.
    return 0


def _safe_print(text: str) -> None:
    """Print tolerant of a narrow console encoding (Windows cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, "replace").decode(enc, "replace"))


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes; in-memory evidence only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # --- _count_sessions counts distinct resolved dates only. -------------- #
    recs = [
        {"resolution_status": "resolved", "resolved_at": "2024-01-01T15:00:00"},
        {"resolution_status": "resolved", "resolved_at": "2024-01-01T16:00:00"},
        {"resolution_status": "resolved", "resolved_at": "2024-01-02T16:00:00"},
        {"resolution_status": "resolved", "recorded_at": "2024-01-03T16:00:00"},
        {"resolution_status": "pending", "resolved_at": "2024-01-04T16:00:00"},
        {"bad": 1},
    ]
    if _count_sessions(recs) != 3:
        print("FAIL: session count", _count_sessions(recs)); ok = False

    # --- load_lab_result unwraps the walk_forward block. ------------------- #
    import tempfile
    with tempfile.TemporaryDirectory() as d:
        study = os.path.join(d, "study.json")
        wf = {"oos_collapse": False, "oos_expectancy": 9.0,
              "oos_capture_ratio": 0.7, "oos_metrics": {"trade_count": 25}}
        with open(study, "w", encoding="utf-8") as fh:
            json.dump({"walk_forward": wf, "report": {}}, fh)
        got = load_lab_result(study)
        if got != wf:
            print("FAIL: lab unwrap", got); ok = False
        # A bare walk-forward dict is returned as-is.
        bare = os.path.join(d, "bare.json")
        with open(bare, "w", encoding="utf-8") as fh:
            json.dump(wf, fh)
        if load_lab_result(bare) != wf:
            print("FAIL: bare lab passthrough"); ok = False
        # Missing / junk -> None (fail-open).
        if load_lab_result(os.path.join(d, "nope.json")) is not None:
            print("FAIL: missing lab not None"); ok = False
    if load_lab_result(None) is not None:
        print("FAIL: None lab path not None"); ok = False

    # --- build_report: missing evidence -> every layer HOLDs, text renders. - #
    rep = build_report(None, None, {})
    if any(dec.get("promote") for dec in rep["decisions"].values()):
        print("FAIL: promoted with no evidence"); ok = False
    if "does NOT edit .env" not in rep["text"] or "HOLD" not in rep["text"]:
        print("FAIL: report text missing guard/verdict"); ok = False

    # --- build_report: full passing evidence -> PROMOTE for a layer. ------- #
    cal = {"n_resolved": 40, "n_filled": 40, "n_sessions": 8,
           "n_would_reject": 6, "reject_precision": 0.83,
           "fill_rate_bias": 0.05, "mean_slippage_error": 0.03}
    lab = {"oos_collapse": False, "oos_expectancy": 12.0,
           "oos_capture_ratio": 0.8, "oos_ci_low": 4.0,
           "oos_metrics": {"trade_count": 30}}
    rep2 = build_report(cal, lab, {}, layers=("executable_ev",))
    if not rep2["decisions"]["executable_ev"]["promote"]:
        print("FAIL: full evidence did not promote",
              rep2["decisions"]["executable_ev"]["reasons"]); ok = False

    # --- resolve_thresholds maps PROMO_* -> internal keys from a fake .env. - #
    with tempfile.TemporaryDirectory() as d:
        envp = os.path.join(d, ".env")
        with open(envp, "w", encoding="utf-8") as fh:
            fh.write("PROMO_MIN_EVALS=50\nPROMO_MIN_CAPTURE=0.9\nJUNK=x\n")
        th = resolve_thresholds(envp)
        if th.get("min_evals") != 50.0 or th.get("min_capture") != 0.9:
            print("FAIL: threshold map", th); ok = False
        # A tightened .env threshold flips a borderline PROMOTE to HOLD.
        rep3 = build_report(cal, lab, th, layers=("executable_ev",))
        if rep3["decisions"]["executable_ev"]["promote"]:
            print("FAIL: tightened threshold still promoted"); ok = False

    print("run_promotion_check self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    sys.exit(main())
