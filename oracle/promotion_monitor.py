"""
Oracle 3.0 — Phase-2 Upgrade I: promotion regression monitor (ANALYTICS ONLY).

Upgrade H persists every ``run_promotion_check`` verdict to an append-only JSONL
history. Upgrade I turns that passive history into an ACTIVE watch: it scans the
per-layer verdict timeline and raises a regression alert when

  * PROMOTION LOST — a layer that previously cleared ALL gates (PROMOTE) is back
    to HOLD in the latest run, or
  * NEW FAILURE   — an already-holding layer developed a NEW failing gate that
    was not in its previous run.

This is the safety net for the human Stage-4 decision: a layer looking ready and
then quietly regressing (a fresh study, more fills, a threshold tighten) is
exactly the kind of drift a promotion process must catch.

    detect_regressions(rows, *, layers) -> report dict
        Pure fold over audit rows (from ``promotion_audit.load_promotion_history``).
    format_regression_alert(report) -> str
        Human/Telegram text; a clean scan renders a "no regressions" line.

CRITICAL SAFETY POSTURE — like the rest of the promotion stack this is OFFLINE
ANALYTICS ONLY. It only READS audit rows. It NEVER edits ``.env``, flips a flag,
opens/sizes/blocks a trade, reads creds, or hits the network. Every public entry
point is fail-open (returns an empty report / never raises). Deterministic:
identical rows -> identical report.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

LOG_TAG = "[PROMOTION_MONITOR]"

KIND_PROMOTION_LOST = "promotion_lost"
KIND_NEW_FAILURE = "new_failure"


# --------------------------------------------------------------------------- #
# Timeline fold (pure)
# --------------------------------------------------------------------------- #
def _timeline(rows: Optional[List[dict]]):
    """Return (chronological rows, {layer: [(ts, promote, reasons_list), ...]}),
    oldest-first, tolerant of junk rows/decisions."""
    clean = [r for r in (rows or [])
             if isinstance(r, dict) and isinstance(r.get("layers"), dict)]
    clean.sort(key=lambda r: str(r.get("recorded_at", "")))

    per: Dict[str, List] = {}
    for rec in clean:
        ts = str(rec.get("recorded_at", ""))
        for layer, dec in rec["layers"].items():
            if not isinstance(dec, dict):
                continue
            promote = bool(dec.get("promote"))
            raw = dec.get("reasons")
            reasons = [str(x) for x in raw] if isinstance(raw, list) else []
            per.setdefault(str(layer), []).append((ts, promote, reasons))
    return clean, per


# --------------------------------------------------------------------------- #
# detect_regressions
# --------------------------------------------------------------------------- #
def detect_regressions(rows: Optional[List[dict]], *,
                       layers: Optional[List[str]] = None) -> Dict[str, Any]:
    """Scan the audit timeline for per-layer regressions. Never raises."""
    try:
        clean, per = _timeline(rows)
        want = set(layers) if layers else None

        regressions: List[dict] = []
        layer_summ: Dict[str, dict] = {}

        for layer, seq in per.items():
            if want is not None and layer not in want:
                continue
            latest_ts, latest_promote, latest_reasons = seq[-1]
            ever_promoted = any(p for _, p, _ in seq)
            first_promoted_at = next((t for t, p, _ in seq if p), None)
            layer_summ[layer] = {
                "runs": len(seq),
                "latest_promote": latest_promote,
                "latest_reasons": latest_reasons,
                "ever_promoted": ever_promoted,
                "first_promoted_at": first_promoted_at,
            }

            # PROMOTION LOST — promoted at some point, now HOLD.
            if ever_promoted and not latest_promote:
                last_promote_idx = max(
                    i for i, (_, p, _) in enumerate(seq) if p)
                nxt = last_promote_idx + 1
                lost_at = seq[nxt][0] if nxt < len(seq) else latest_ts
                regressions.append({
                    "layer": layer,
                    "kind": KIND_PROMOTION_LOST,
                    "last_promoted_at": seq[last_promote_idx][0],
                    "lost_at": lost_at,
                    "latest_reasons": latest_reasons,
                })

            # NEW FAILURE — HOLD -> HOLD with a reason that was not there before.
            if len(seq) >= 2 and not latest_promote:
                _, prev_promote, prev_reasons = seq[-2]
                if not prev_promote:
                    added = sorted(set(latest_reasons) - set(prev_reasons))
                    if added:
                        regressions.append({
                            "layer": layer,
                            "kind": KIND_NEW_FAILURE,
                            "added_reasons": added,
                            "prev_at": seq[-2][0],
                            "latest_at": latest_ts,
                        })

        regressions.sort(key=lambda e: (e["layer"], e["kind"]))
        return {
            "n_runs": len(clean),
            "latest_at": clean[-1].get("recorded_at") if clean else None,
            "regressions": regressions,
            "layers": layer_summ,
        }
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} scan skipped: {exc}")
        return {"n_runs": 0, "latest_at": None, "regressions": [], "layers": {}}


# --------------------------------------------------------------------------- #
# format_regression_alert
# --------------------------------------------------------------------------- #
def format_regression_alert(report: Optional[dict]) -> str:
    """Render a regression report as a human/Telegram alert. Fail-open."""
    if not isinstance(report, dict):
        return "Oracle promotion monitor: no regressions detected."
    regs = report.get("regressions") or []
    if not regs:
        return "Oracle promotion monitor: no regressions detected."
    lines = [
        "===== Oracle promotion monitor — REGRESSION ALERT =====",
        f"runs scanned: {report.get('n_runs')} (latest {report.get('latest_at')})",
        "",
    ]
    for e in regs:
        if e.get("kind") == KIND_PROMOTION_LOST:
            reasons = ", ".join(e.get("latest_reasons") or []) or "n/a"
            lines.append(
                f"  [PROMOTION LOST] {e['layer']}: cleared all gates at "
                f"{e.get('last_promoted_at')}, back to HOLD at {e.get('lost_at')} "
                f"— now: {reasons}")
        elif e.get("kind") == KIND_NEW_FAILURE:
            added = ", ".join(e.get("added_reasons") or [])
            lines.append(
                f"  [NEW FAILURE] {e['layer']}: new failing gate(s) {added} "
                f"(since {e.get('prev_at')})")
    lines.append("")
    lines.append("ADVISORY ONLY — investigate before promoting.")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (pure; in-memory rows only; no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def _row(ts, layers):
        return {"type": "promotion_check", "recorded_at": ts, "layers": layers}

    def _dec(promote, reasons=()):
        return {"promote": promote, "reasons": list(reasons), "metrics": {}}

    # --- Empty / junk -> clean report. ------------------------------------- #
    empty = detect_regressions([])
    if empty["regressions"] or empty["n_runs"] != 0:
        print("FAIL: empty scan not clean", empty); ok = False
    if "no regressions" not in format_regression_alert(empty):
        print("FAIL: empty alert text"); ok = False
    for junk in (None, 42, "x", [{"bad": 1}, 7, None]):
        r = detect_regressions(junk)  # type: ignore[arg-type]
        if r["regressions"]:
            print("FAIL: junk produced regressions", junk, r); ok = False

    # --- No regression: steadily promoting. -------------------------------- #
    steady = detect_regressions([
        _row("2024-01-01", {"executable_ev": _dec(False, ["insufficient_evals"])}),
        _row("2024-01-02", {"executable_ev": _dec(True, [])}),
        _row("2024-01-03", {"executable_ev": _dec(True, [])}),
    ])
    if steady["regressions"]:
        print("FAIL: steady promote flagged", steady["regressions"]); ok = False
    ev = steady["layers"]["executable_ev"]
    if not ev["latest_promote"] or ev["first_promoted_at"] != "2024-01-02":
        print("FAIL: steady summary", ev); ok = False

    # --- PROMOTION LOST: promote then back to HOLD. ------------------------ #
    lost = detect_regressions([
        _row("2024-01-01", {"fill_model": _dec(True, [])}),
        _row("2024-01-02", {"fill_model": _dec(True, [])}),
        _row("2024-01-03", {"fill_model": _dec(False, ["fill_rate_bias_too_high"])}),
    ])
    pl = [e for e in lost["regressions"] if e["kind"] == KIND_PROMOTION_LOST]
    if len(pl) != 1:
        print("FAIL: promotion_lost not detected", lost["regressions"]); ok = False
    elif (pl[0]["layer"] != "fill_model" or
          pl[0]["last_promoted_at"] != "2024-01-02" or
          pl[0]["lost_at"] != "2024-01-03"):
        print("FAIL: promotion_lost fields", pl[0]); ok = False
    if "PROMOTION LOST" not in format_regression_alert(lost):
        print("FAIL: promotion_lost alert text"); ok = False

    # A later re-PROMOTE clears the regression.
    recovered = detect_regressions([
        _row("2024-01-01", {"fill_model": _dec(True, [])}),
        _row("2024-01-02", {"fill_model": _dec(False, ["x"])}),
        _row("2024-01-03", {"fill_model": _dec(True, [])}),
    ])
    if recovered["regressions"]:
        print("FAIL: recovered still flagged", recovered["regressions"]); ok = False

    # --- NEW FAILURE: HOLD -> HOLD with a new reason (never promoted). ----- #
    newfail = detect_regressions([
        _row("2024-01-01", {"adversarial_thesis": _dec(False, ["insufficient_oos_trades"])}),
        _row("2024-01-02", {"adversarial_thesis": _dec(
            False, ["insufficient_oos_trades", "oos_edge_not_captured"])}),
    ])
    nf = [e for e in newfail["regressions"] if e["kind"] == KIND_NEW_FAILURE]
    if len(nf) != 1 or nf[0]["added_reasons"] != ["oos_edge_not_captured"]:
        print("FAIL: new_failure detection", newfail["regressions"]); ok = False
    # A never-promoted, never-worsening HOLD is NOT a regression.
    if [e for e in newfail["regressions"] if e["kind"] == KIND_PROMOTION_LOST]:
        print("FAIL: HOLD-only flagged as promotion_lost"); ok = False

    # No NEW FAILURE when the previous run was a PROMOTE (that is a LOST, not a
    # drift) — avoids double-counting the transition reasons.
    trans = detect_regressions([
        _row("2024-01-01", {"executable_ev": _dec(True, [])}),
        _row("2024-01-02", {"executable_ev": _dec(False, ["low_reject_precision"])}),
    ])
    if [e for e in trans["regressions"] if e["kind"] == KIND_NEW_FAILURE]:
        print("FAIL: transition double-counted as new_failure"); ok = False
    if not [e for e in trans["regressions"] if e["kind"] == KIND_PROMOTION_LOST]:
        print("FAIL: transition missing promotion_lost"); ok = False

    # --- layers filter. ---------------------------------------------------- #
    scoped = detect_regressions([
        _row("2024-01-01", {"fill_model": _dec(True, []),
                            "executable_ev": _dec(True, [])}),
        _row("2024-01-02", {"fill_model": _dec(False, ["x"]),
                            "executable_ev": _dec(False, ["y"])}),
    ], layers=["fill_model"])
    if set(scoped["layers"]) != {"fill_model"}:
        print("FAIL: layers filter", list(scoped["layers"])); ok = False
    if any(e["layer"] != "fill_model" for e in scoped["regressions"]):
        print("FAIL: layers filter leaked", scoped["regressions"]); ok = False

    # --- Determinism. ------------------------------------------------------ #
    rows = [
        _row("2024-01-01", {"fill_model": _dec(True, [])}),
        _row("2024-01-02", {"fill_model": _dec(False, ["a", "b"])}),
    ]
    if detect_regressions(rows) != detect_regressions(list(reversed(rows))):
        print("FAIL: not order-independent / deterministic"); ok = False

    print("oracle.promotion_monitor self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
