"""
Oracle 3.0 — Phase-2 Upgrade G: automated promotion gates (Stage 4).

Upgrades C/D/E each ship a behavior that starts in SHADOW and is meant to be
promoted to a live veto only after it earns it on out-of-sample + realized data:

  * ``executable_ev``      EXECUTABLE_EV_SHADOW_MODE=false  (C)  — arm the veto.
  * ``fill_model``         ENABLE_FILL_MODEL=true           (D)  — trust the model.
  * ``adversarial_thesis`` ADVERSARIAL_THESIS_GATING=true   (E)  — feed the gate.

Today that promotion is a human judgement call. This module *codifies* the
thresholds so the decision is auditable and reproducible: given the realized
execution-calibration stats (``oracle.execution.calibration.compute_calibration``)
and the out-of-sample walk-forward result (``oracle.lab.walk_forward`` /
``run_phase2_study``), ``evaluate_promotion`` returns a hard yes/no plus the
reasons and the metrics it judged on.

CRITICAL SAFETY POSTURE — this module is OFFLINE ADVISORY ONLY:
  * It is a PURE, DETERMINISTIC function of its inputs. No env reads, no clock,
    no network, no file writes, no randomness.
  * It NEVER edits ``.env``, flips a flag, opens/sizes/prices/blocks a trade, or
    touches the live path. A human reads the report and flips the flag by hand.
  * FAIL-CLOSED on the decision, FAIL-OPEN on errors: any missing / malformed
    input yields ``promote=False`` with a reason — it can only ever WITHHOLD a
    promotion, never manufacture one.

``evaluate_promotion(layer, calibration_stats, lab_result, thresholds)``
    -> ``{"promote": bool, "reasons": [str], "metrics": {...}}``

``promote`` is True IFF every gate passes (``reasons`` is empty). Thresholds are
passed in (pure); the offline ``run_promotion_check.py`` runner resolves them
from ``.env`` and assembles the two stat blocks before calling this.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional

LOG_TAG = "[ORACLE_PROMOTION]"

# The three shadow layers Upgrade G can certify for promotion.
LAYERS = ("executable_ev", "fill_model", "adversarial_thesis")

# Conservative built-in thresholds. Every one can only make promotion HARDER;
# the runner may override from .env (PROMO_* keys) but the defaults already gate.
_DEFAULT_THRESHOLDS: Dict[str, float] = {
    # Generic (all layers) — sample size + out-of-sample survival.
    "min_evals": 30.0,           # resolved trades in the calibration ledger
    "min_sessions": 5.0,         # distinct trading sessions represented
    "min_oos_trades": 20.0,      # trades pooled into the OOS test windows
    "min_margin": 0.0,           # OOS edge lower-bound must clear this ($ expectancy)
    "min_capture": 0.5,          # OOS expectancy / IS expectancy floor
    # executable_ev — the would-be vetoes must have been net-correct.
    "min_reject_precision": 0.5,
    # fill_model — the model's fill-rate + slippage must be well-calibrated.
    "max_fill_rate_bias": 0.15,
    "max_abs_slippage_error": 0.10,
}


# --------------------------------------------------------------------------- #
# Coercion helpers (pure)
# --------------------------------------------------------------------------- #
def _to_num(v: Any) -> Optional[float]:
    """Coerce to a finite float, or None. Bools are NOT numbers here."""
    if v is None or isinstance(v, bool):
        return None
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


def _merge_thresholds(thresholds: Optional[dict]) -> Dict[str, float]:
    """Overlay caller thresholds on the conservative defaults. Unknown keys and
    non-numeric values are ignored (fail-open toward the stricter default)."""
    th = dict(_DEFAULT_THRESHOLDS)
    if isinstance(thresholds, dict):
        for k, v in thresholds.items():
            if k in th:
                fv = _to_num(v)
                if fv is not None:
                    th[k] = fv
    return th


# --------------------------------------------------------------------------- #
# The gate
# --------------------------------------------------------------------------- #
def evaluate_promotion(layer: str,
                       calibration_stats: Optional[dict],
                       lab_result: Optional[dict],
                       thresholds: Optional[dict] = None) -> Dict[str, Any]:
    """Decide whether ``layer`` may be promoted from shadow to a live veto.

    Pure / deterministic / fail-closed on the decision. Returns a dict with
    ``promote`` (True only when every gate passes), ``reasons`` (the gates that
    failed, sorted + de-duplicated), and ``metrics`` (the values judged on).
    """
    th = _merge_thresholds(thresholds)
    metrics: Dict[str, Any] = {"layer": layer, "thresholds": th}
    reasons: List[str] = []

    # ---- fail-open on missing / malformed inputs -> withhold promotion ----- #
    if layer not in LAYERS:
        return {"promote": False, "reasons": ["unknown_layer"], "metrics": metrics}
    if not isinstance(calibration_stats, dict) or not isinstance(lab_result, dict):
        return {"promote": False, "reasons": ["insufficient_input"],
                "metrics": metrics}

    # ---- pull the stats we judge on --------------------------------------- #
    n_resolved = _to_num(calibration_stats.get("n_resolved"))
    n_filled = _to_num(calibration_stats.get("n_filled"))
    n_sessions = _to_num(calibration_stats.get("n_sessions"))

    oos = lab_result.get("oos_metrics")
    oos = oos if isinstance(oos, dict) else {}
    oos_trades = _to_num(oos.get("trade_count"))
    oos_exp = _to_num(lab_result.get("oos_expectancy"))
    oos_capture = _to_num(lab_result.get("oos_capture_ratio"))
    # Optional explicit lower confidence bound on the OOS edge; when absent we
    # fall back to the point estimate (still must clear the margin).
    oos_ci_low = _to_num(lab_result.get("oos_ci_low"))
    collapse = bool(lab_result.get("oos_collapse"))
    lower_bound = oos_ci_low if oos_ci_low is not None else oos_exp

    metrics.update({
        "n_resolved": n_resolved,
        "n_filled": n_filled,
        "n_sessions": n_sessions,
        "oos_trades": oos_trades,
        "oos_expectancy": oos_exp,
        "oos_capture_ratio": oos_capture,
        "oos_ci_low": oos_ci_low,
        "oos_lower_bound": lower_bound,
        "oos_collapse": collapse,
    })

    # ---- generic gates (every layer must clear these) --------------------- #
    if collapse:
        reasons.append("oos_collapse")
    if n_resolved is None or n_resolved < th["min_evals"]:
        reasons.append("insufficient_evals")
    if th["min_sessions"] > 0 and (n_sessions is None
                                   or n_sessions < th["min_sessions"]):
        reasons.append("insufficient_sessions")
    if oos_trades is None or oos_trades < th["min_oos_trades"]:
        reasons.append("insufficient_oos_trades")
    # "margin inside the CI": the OOS edge is not distinguishable from the
    # margin, so the promotion is not earned.
    if lower_bound is None or lower_bound <= th["min_margin"]:
        reasons.append("margin_inside_ci")
    if oos_capture is None or oos_capture < th["min_capture"]:
        reasons.append("oos_edge_not_captured")

    # ---- layer-specific gates --------------------------------------------- #
    if layer == "executable_ev":
        n_wr = _to_num(calibration_stats.get("n_would_reject"))
        rp = _to_num(calibration_stats.get("reject_precision"))
        metrics["n_would_reject"] = n_wr
        metrics["reject_precision"] = rp
        if n_wr is None or n_wr < 1:
            reasons.append("no_would_reject_evidence")
        elif rp is None or rp < th["min_reject_precision"]:
            reasons.append("low_reject_precision")

    elif layer == "fill_model":
        bias = _to_num(calibration_stats.get("fill_rate_bias"))
        slip = _to_num(calibration_stats.get("mean_slippage_error"))
        metrics["fill_rate_bias"] = bias
        metrics["mean_slippage_error"] = slip
        if n_filled is None or n_filled < th["min_evals"]:
            reasons.append("insufficient_fills")
        if bias is None or abs(bias) > th["max_fill_rate_bias"]:
            reasons.append("fill_rate_bias_too_high")
        if slip is None or abs(slip) > th["max_abs_slippage_error"]:
            reasons.append("slippage_error_too_high")

    # adversarial_thesis rides purely on the generic OOS-survival gates: the
    # doubt layer is only promoted once its OOS edge is captured and not
    # collapsed, on a large-enough sample. No extra ledger requirement.

    promote = (len(reasons) == 0)
    return {"promote": promote,
            "reasons": sorted(set(reasons)),
            "metrics": metrics}


def format_promotion_report(decision: Dict[str, Any]) -> str:
    """Render one ``evaluate_promotion`` result as a human-readable block."""
    d = decision or {}
    m = d.get("metrics", {}) or {}
    layer = m.get("layer", "?")
    verdict = "PROMOTE" if d.get("promote") else "HOLD"

    def _f(v):
        return "n/a" if v is None else (f"{v:.4f}" if isinstance(v, float) else str(v))

    lines = [
        f"=== promotion gate: {layer} -> {verdict} ===",
        f"  resolved evals : {_f(m.get('n_resolved'))}  "
        f"(min {_f(m.get('thresholds', {}).get('min_evals'))})",
        f"  sessions       : {_f(m.get('n_sessions'))}  "
        f"(min {_f(m.get('thresholds', {}).get('min_sessions'))})",
        f"  OOS trades     : {_f(m.get('oos_trades'))}  "
        f"(min {_f(m.get('thresholds', {}).get('min_oos_trades'))})",
        f"  OOS expectancy : {_f(m.get('oos_expectancy'))}  "
        f"(lower-bound {_f(m.get('oos_lower_bound'))}, "
        f"margin {_f(m.get('thresholds', {}).get('min_margin'))})",
        f"  OOS capture    : {_f(m.get('oos_capture_ratio'))}  "
        f"(min {_f(m.get('thresholds', {}).get('min_capture'))})",
        f"  OOS collapse   : {_f(m.get('oos_collapse'))}",
    ]
    if layer == "executable_ev":
        lines.append(f"  reject precision: {_f(m.get('reject_precision'))}  "
                     f"(min {_f(m.get('thresholds', {}).get('min_reject_precision'))}, "
                     f"n={_f(m.get('n_would_reject'))})")
    elif layer == "fill_model":
        lines.append(f"  fill-rate bias : {_f(m.get('fill_rate_bias'))}  "
                     f"(max {_f(m.get('thresholds', {}).get('max_fill_rate_bias'))})")
        lines.append(f"  slippage error : {_f(m.get('mean_slippage_error'))}  "
                     f"(max {_f(m.get('thresholds', {}).get('max_abs_slippage_error'))})")
    if d.get("promote"):
        lines.append("  -> ALL GATES PASS. A human may now flip the flag "
                     "(paper first). This tool does NOT edit .env.")
    else:
        lines.append(f"  -> HOLD. failing gates: {', '.join(d.get('reasons', []))}")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (pure; no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _passing_stats(layer: str) -> dict:
    base = {
        "n_resolved": 40, "n_filled": 40, "n_sessions": 8,
        "n_would_reject": 6, "reject_precision": 0.83,
        "fill_rate_bias": 0.05, "mean_slippage_error": 0.03,
    }
    return base


def _passing_lab() -> dict:
    return {
        "oos_collapse": False,
        "oos_expectancy": 12.0,
        "oos_capture_ratio": 0.8,
        "oos_ci_low": 4.0,
        "oos_metrics": {"trade_count": 30, "expectancy": 12.0},
    }


def _self_test() -> int:
    ok = True

    # --- 1) All gates pass -> promote True for each layer. ------------------ #
    for layer in LAYERS:
        dec = evaluate_promotion(layer, _passing_stats(layer), _passing_lab())
        if not dec["promote"] or dec["reasons"]:
            print("FAIL: expected promote for", layer, dec["reasons"]); ok = False

    # --- 2) Missing / malformed inputs -> withhold (fail-open). ------------- #
    for bad in (None, 42, "x", []):
        d1 = evaluate_promotion("executable_ev", bad, _passing_lab())
        d2 = evaluate_promotion("executable_ev", _passing_stats("executable_ev"), bad)
        if d1["promote"] or d2["promote"]:
            print("FAIL: junk input promoted", bad); ok = False
        if d1["reasons"] != ["insufficient_input"]:
            print("FAIL: junk stats reason", d1["reasons"]); ok = False

    # --- 3) Unknown layer -> withhold. -------------------------------------- #
    du = evaluate_promotion("nope", _passing_stats("x"), _passing_lab())
    if du["promote"] or du["reasons"] != ["unknown_layer"]:
        print("FAIL: unknown layer", du); ok = False

    # --- 4) OOS collapse -> withhold regardless of everything else. --------- #
    lab_collapse = dict(_passing_lab()); lab_collapse["oos_collapse"] = True
    dc = evaluate_promotion("adversarial_thesis", _passing_stats("x"), lab_collapse)
    if dc["promote"] or "oos_collapse" not in dc["reasons"]:
        print("FAIL: collapse not caught", dc); ok = False

    # --- 5) Insufficient sample size -> withhold. --------------------------- #
    stats_small = dict(_passing_stats("executable_ev")); stats_small["n_resolved"] = 3
    ds = evaluate_promotion("executable_ev", stats_small, _passing_lab())
    if ds["promote"] or "insufficient_evals" not in ds["reasons"]:
        print("FAIL: small sample promoted", ds); ok = False

    # --- 6) Margin inside the CI (lower bound <= margin) -> withhold. ------- #
    lab_ci = dict(_passing_lab()); lab_ci["oos_ci_low"] = -1.0
    dm = evaluate_promotion("adversarial_thesis", _passing_stats("x"), lab_ci)
    if dm["promote"] or "margin_inside_ci" not in dm["reasons"]:
        print("FAIL: margin-inside-CI promoted", dm); ok = False

    # --- 7) OOS edge not captured -> withhold. ------------------------------ #
    lab_lowcap = dict(_passing_lab()); lab_lowcap["oos_capture_ratio"] = 0.2
    dcap = evaluate_promotion("adversarial_thesis", _passing_stats("x"), lab_lowcap)
    if dcap["promote"] or "oos_edge_not_captured" not in dcap["reasons"]:
        print("FAIL: low capture promoted", dcap); ok = False

    # --- 8) executable_ev with no would-reject evidence -> withhold. -------- #
    stats_nowr = dict(_passing_stats("executable_ev"))
    stats_nowr["n_would_reject"] = 0
    dnwr = evaluate_promotion("executable_ev", stats_nowr, _passing_lab())
    if dnwr["promote"] or "no_would_reject_evidence" not in dnwr["reasons"]:
        print("FAIL: no-reject-evidence promoted", dnwr); ok = False

    # --- 9) executable_ev with poor reject precision -> withhold. ----------- #
    stats_badrp = dict(_passing_stats("executable_ev"))
    stats_badrp["reject_precision"] = 0.2
    dbrp = evaluate_promotion("executable_ev", stats_badrp, _passing_lab())
    if dbrp["promote"] or "low_reject_precision" not in dbrp["reasons"]:
        print("FAIL: low reject precision promoted", dbrp); ok = False

    # --- 10) fill_model with biased fill rate / slippage -> withhold. ------- #
    stats_bias = dict(_passing_stats("fill_model"))
    stats_bias["fill_rate_bias"] = 0.40
    stats_bias["mean_slippage_error"] = 0.50
    dbias = evaluate_promotion("fill_model", stats_bias, _passing_lab())
    if dbias["promote"] or "fill_rate_bias_too_high" not in dbias["reasons"] \
            or "slippage_error_too_high" not in dbias["reasons"]:
        print("FAIL: biased fill model promoted", dbias); ok = False

    # --- 11) Threshold override can only tighten (raise the bar). ----------- #
    strict = evaluate_promotion("adversarial_thesis", _passing_stats("x"),
                                _passing_lab(), thresholds={"min_capture": 0.99})
    if strict["promote"] or "oos_edge_not_captured" not in strict["reasons"]:
        print("FAIL: strict override still promoted", strict); ok = False

    # --- 12) Determinism: identical inputs -> identical decision. ----------- #
    a = evaluate_promotion("executable_ev", _passing_stats("executable_ev"),
                           _passing_lab())
    b = evaluate_promotion("executable_ev", _passing_stats("executable_ev"),
                           _passing_lab())
    if a != b:
        print("FAIL: non-deterministic", a, b); ok = False

    # --- 13) Report renders for both verdicts. ------------------------------ #
    txt_pass = format_promotion_report(a)
    txt_hold = format_promotion_report(ds)
    if "PROMOTE" not in txt_pass or "HOLD" not in txt_hold:
        print("FAIL: report text", txt_pass[:80], txt_hold[:80]); ok = False

    print("oracle.promotion self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
