"""
Phase E 4 — Oracle probability calibration (analytics only, pure report).

Reads the folded predictions from ``oracle_prob_recorder`` and answers three
questions about the Oracle directional head, using ONLY resolved rows:

    reliability curve   For each p_call decile, does the empirical up-rate
                        track the predicted probability? (well-calibrated =
                        bucket mean p_call ~= realised up-rate.)
    direction Brier     Mean Brier of the call-head vs the 0.5 coin-flip
                        baseline (0.25). Lower is better; a head that beats
                        0.25 is adding directional information.
    no_trade diagnostic Of the beliefs the head called "no-trade" (neither
                        side favoured), how often did the underlying actually
                        move beyond a band? A high hit-rate means the head is
                        sitting out real moves.

STRICTLY analytics: pure functions over recorded rows, no network, no trading
side effects, fail-open to an INSUFFICIENT_DATA verdict.
"""

from typing import List, Optional

import oracle_prob_recorder as opr

VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

MIN_SAMPLES = 20                 # resolved directional rows for a verdict
BRIER_BASELINE = 0.25            # Brier of a p=0.5 coin flip
NO_TRADE_MOVE_BAND_PCT = 1.0     # |return| beyond this = a "real" move
ANALYTICS_FOOTER = "Analytics only — never sizes, blocks or alters a trade."
CALIBRATION_QUESTION = (
    "Is the Oracle direction head calibrated, and does it beat a coin flip?")

_DECILES = [(i / 10.0, (i + 1) / 10.0) for i in range(10)]


def _resolved_rows(records: Optional[List[dict]],
                   jsonl_path: Optional[str]) -> List[dict]:
    rows = records if records is not None else opr.load_predictions(jsonl_path)
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        if r.get("resolution_status") == opr.STATUS_RESOLVED:
            out.append(r)
    return out


def _decile_label(p: float) -> str:
    idx = min(9, max(0, int(p * 10)))
    lo, hi = _DECILES[idx]
    return f"{lo:.1f}-{hi:.1f}"


def compute_prob_calibration(records: Optional[List[dict]] = None,
                             jsonl_path: Optional[str] = None) -> dict:
    """Reliability curve + Brier-vs-baseline + no-trade diagnostic. Pure over
    the folded ledger. Never raises."""
    try:
        resolved = _resolved_rows(records, jsonl_path)

        # --- reliability curve + Brier over directional rows ---------------
        buckets = {_decile_label(lo): {"n": 0, "sum_p": 0.0, "up": 0}
                   for lo, _ in _DECILES}
        brier_sum = 0.0
        brier_n = 0
        directional_n = 0
        correct_n = 0
        graded_n = 0
        for r in resolved:
            pc = opr._to_float(r.get("p_call"))
            realized = r.get("realized_direction")
            if pc is not None and realized in ("up", "down"):
                lbl = _decile_label(pc)
                b = buckets[lbl]
                b["n"] += 1
                b["sum_p"] += pc
                if realized == "up":
                    b["up"] += 1
                directional_n += 1
            br = opr._to_float(r.get("brier_call"))
            if br is not None:
                brier_sum += br
                brier_n += 1
            corr = r.get("correct")
            if corr is not None:
                graded_n += 1
                if corr:
                    correct_n += 1

        reliability = []
        for lo, _ in _DECILES:
            lbl = _decile_label(lo)
            b = buckets[lbl]
            if b["n"] == 0:
                continue
            reliability.append({
                "bucket": lbl,
                "n": b["n"],
                "mean_p_call": round(b["sum_p"] / b["n"], 4),
                "empirical_up_rate": round(b["up"] / b["n"], 4),
            })

        brier = round(brier_sum / brier_n, 4) if brier_n else None
        brier_improvement = (round(BRIER_BASELINE - brier, 4)
                             if brier is not None else None)
        accuracy = round(correct_n / graded_n, 4) if graded_n else None

        # --- no-trade diagnostic ------------------------------------------
        nt_rows = [r for r in resolved
                   if r.get("predicted_direction") == "none"]
        nt_moved = 0
        nt_move_sum = 0.0
        nt_measured = 0
        for r in nt_rows:
            ret = opr._to_float(r.get("realized_return_pct"))
            if ret is None:
                continue
            nt_measured += 1
            nt_move_sum += abs(ret)
            if abs(ret) >= NO_TRADE_MOVE_BAND_PCT:
                nt_moved += 1
        no_trade_diagnostic = {
            "n": len(nt_rows),
            "measured": nt_measured,
            "moved_beyond_band": nt_moved,
            "move_band_pct": NO_TRADE_MOVE_BAND_PCT,
            "missed_move_rate": (round(nt_moved / nt_measured, 4)
                                 if nt_measured else None),
            "avg_abs_move_pct": (round(nt_move_sum / nt_measured, 4)
                                 if nt_measured else None),
        }

        verdict = (VERDICT_OK if directional_n >= MIN_SAMPLES
                   else VERDICT_INSUFFICIENT)
        return {
            "sample_size": len(resolved),
            "directional_n": directional_n,
            "reliability": reliability,
            "brier": brier,
            "brier_baseline": BRIER_BASELINE,
            "brier_improvement": brier_improvement,
            "direction_accuracy": accuracy,
            "no_trade_diagnostic": no_trade_diagnostic,
            "verdict": verdict,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "directional_n": 0, "reliability": [],
                "brier": None, "brier_baseline": BRIER_BASELINE,
                "brier_improvement": None, "direction_accuracy": None,
                "no_trade_diagnostic": {}, "verdict": VERDICT_INSUFFICIENT}


def _pct(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _num(value) -> str:
    return f"{value:.3f}" if value is not None else "n/a"


def format_prob_calibration(report: dict) -> str:
    """Telegram-ready ORACLE_PROB_CALIBRATION. Pure formatting."""
    header = "🎯 *Oracle Prob Calibration* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if not report or report.get("directional_n", 0) == 0:
        return "\n".join([
            header, "", f"_{CALIBRATION_QUESTION}_", "",
            "No resolved directional predictions yet.",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", footer,
        ])
    lines = [header, "", f"_{CALIBRATION_QUESTION}_", "",
             "*Reliability (p_call bucket -> mean pred / empirical up-rate):*"]
    for b in report.get("reliability", []):
        lines.append(
            f"`{b['bucket']}`: n `{b['n']}`, "
            f"pred `{_pct(b['mean_p_call'])}`, "
            f"actual `{_pct(b['empirical_up_rate'])}`")
    brier = report.get("brier")
    impr = report.get("brier_improvement")
    impr_str = (f"`{impr:+.3f}` vs 0.25" if impr is not None else "n/a")
    lines += [
        "",
        f"*Direction Brier:* `{_num(brier)}` ({impr_str})",
        f"*Direction accuracy:* `{_pct(report.get('direction_accuracy'))}`",
    ]
    nt = report.get("no_trade_diagnostic") or {}
    if nt.get("measured"):
        lines += [
            "",
            "*No-trade diagnostic:*",
            f"`{nt.get('n')}` no-trade calls, "
            f"`{nt.get('measured')}` measured; "
            f"missed-move `{_pct(nt.get('missed_move_rate'))}` "
            f"(>{nt.get('move_band_pct')}%), "
            f"avg |move| `{_num(nt.get('avg_abs_move_pct'))}%`",
        ]
    lines += [
        "",
        f"*Verdict:* `{report.get('verdict')}`",
        f"Resolved: `{report.get('sample_size')}` · "
        f"directional: `{report.get('directional_n')}`",
        "", footer,
    ]
    return "\n".join(lines)


def generate_prob_calibration_text(jsonl_path: Optional[str] = None) -> str:
    """Top-level entry for the ORACLE_PROB_CALIBRATION Telegram command."""
    return format_prob_calibration(
        compute_prob_calibration(jsonl_path=jsonl_path))


# --------------------------------------------------------------------------- #
# Self-test (offline; synthetic resolved rows)
# --------------------------------------------------------------------------- #
def _self_test() -> bool:
    # Empty -> INSUFFICIENT, no raise, renders.
    empty = compute_prob_calibration(records=[])
    assert empty["verdict"] == VERDICT_INSUFFICIENT
    assert empty["directional_n"] == 0
    txt = format_prob_calibration(empty)
    assert "INSUFFICIENT_DATA" in txt

    # A well-calibrated, informative head: high p_call rows go up, low go down.
    rows = []
    for _ in range(15):
        rows.append({"resolution_status": opr.STATUS_RESOLVED,
                     "p_call": 0.9, "realized_direction": "up",
                     "predicted_direction": "call", "correct": True,
                     "brier_call": (0.9 - 1.0) ** 2,
                     "realized_return_pct": 2.0})
    for _ in range(15):
        rows.append({"resolution_status": opr.STATUS_RESOLVED,
                     "p_call": 0.05, "realized_direction": "down",
                     "predicted_direction": "put", "correct": True,
                     "brier_call": (0.05 - 0.0) ** 2,
                     "realized_return_pct": -2.0})
    rep = compute_prob_calibration(records=rows)
    assert rep["verdict"] == VERDICT_OK, "30 directional rows -> OK"
    assert rep["directional_n"] == 30
    # High bucket up-rate ~1, low bucket up-rate ~0.
    by = {b["bucket"]: b for b in rep["reliability"]}
    assert by["0.9-1.0"]["empirical_up_rate"] == 1.0
    assert by["0.0-0.1"]["empirical_up_rate"] == 0.0
    # This head crushes the 0.25 baseline (Brier ~0.01).
    assert rep["brier"] is not None and rep["brier"] < BRIER_BASELINE
    assert rep["brier_improvement"] > 0.0
    assert rep["direction_accuracy"] == 1.0

    # No-trade diagnostic: a no-trade call that actually moved 3% is a miss.
    nt_rows = [{"resolution_status": opr.STATUS_RESOLVED,
                "p_call": 0.33, "predicted_direction": "none",
                "realized_direction": "up", "correct": None,
                "realized_return_pct": 3.0}]
    nt_rep = compute_prob_calibration(records=nt_rows)
    nd = nt_rep["no_trade_diagnostic"]
    assert nd["n"] == 1 and nd["measured"] == 1
    assert nd["moved_beyond_band"] == 1
    assert nd["missed_move_rate"] == 1.0
    assert abs(nd["avg_abs_move_pct"] - 3.0) < 1e-9

    # Renders a populated report without raising.
    assert "Reliability" in format_prob_calibration(rep)
    return True


if __name__ == "__main__":
    ok = _self_test()
    print("oracle_prob_calibration self-test:", "PASS" if ok else "FAIL")
    raise SystemExit(0 if ok else 1)
