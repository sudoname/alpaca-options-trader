"""
Oracle 3.0 — Phase-2 Upgrade H: promotion audit ledger (ANALYTICS ONLY).

Upgrade G turned the manual Stage-4 promotion decision into a PURE gate
(``oracle.promotion.evaluate_promotion``) and gave it an offline front end
(``run_promotion_check``). Upgrade H closes the audit loop: every time the
promotion check runs, its verdict is appended — verbatim and timestamped — to a
JSONL history, so the human promotion decision is TRACKED and REPRODUCIBLE over
time ("when did fill_model first clear all gates?", "did executable_ev flip back
to HOLD after the last study?").

    record_promotion_check(decisions, *, thresholds, sources, path, now)
        Append ONE row for a whole check run: per-layer promote/reasons/metrics
        plus a summary + provenance. Returns the stored record, or None on error.

    load_promotion_history(filters, *, path, limit)
        Read the ledger back, newest-first, optionally filtered by
        {layer, promote}. Context/audit lookup only.

    summarize_history(rows)  /  format_daily_summary(rows)
        Fold a set of rows into a per-layer "latest verdict + how long it has
        held + first-promoted timestamp" view for an optional daily Telegram
        digest.

CRITICAL SAFETY POSTURE — like the runner, this is OFFLINE AUDIT ONLY. It only
appends to / reads a local JSONL file. It NEVER edits ``.env``, flips a flag,
opens/sizes/blocks a trade, reads creds, or hits the network. Every public entry
point is fail-open (returns None / an empty result rather than raising).

Idiom mirrors ``oracle/trade_memory.py`` and ``oracle/execution/calibration.py``
(``_now`` / ``_jsonl_path`` / ``_append_jsonl``, append-only, fail-open).
"""

from __future__ import annotations

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

LOG_TAG = "[PROMOTION_AUDIT]"

JSONL_FILE_DEFAULT = "promotion_audit.jsonl"
ENV_KEY = "PROMOTION_AUDIT_JSONL"

RECORD_TYPE = "promotion_check"


# --------------------------------------------------------------------------- #
# Helpers (mirror trade_memory.py / calibration.py)
# --------------------------------------------------------------------------- #
def _now(now: Optional[datetime]) -> datetime:
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _jsonl_path(path: Optional[str] = None) -> str:
    """Resolve the ledger path (arg > env > default). Fail-open."""
    if path:
        return path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(ENV_KEY, JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _clean_decision(dec: Any) -> Optional[dict]:
    """Normalise one ``evaluate_promotion`` decision into a stored, JSON-safe
    shape. Returns None for anything that is not a dict."""
    if not isinstance(dec, dict):
        return None
    reasons = dec.get("reasons")
    if not isinstance(reasons, (list, tuple)):
        reasons = []
    metrics = dec.get("metrics")
    if not isinstance(metrics, dict):
        metrics = {}
    return {
        "promote": bool(dec.get("promote")),
        "reasons": [str(r) for r in reasons],
        "metrics": dict(metrics),
    }


# --------------------------------------------------------------------------- #
# record_promotion_check
# --------------------------------------------------------------------------- #
def record_promotion_check(decisions: Optional[dict], *,
                           thresholds: Optional[dict] = None,
                           sources: Optional[dict] = None,
                           path: Optional[str] = None,
                           now: Optional[datetime] = None) -> Optional[dict]:
    """Append one audit row for a whole promotion-check run.

    ``decisions`` is ``{layer: evaluate_promotion(...)}`` (exactly the
    ``build_report`` ``decisions`` map). Returns the stored record, or None on
    error (fail-open — the audit ledger must never break the runner)."""
    try:
        if not isinstance(decisions, dict) or not decisions:
            return None
        layers: Dict[str, dict] = {}
        for layer, dec in decisions.items():
            clean = _clean_decision(dec)
            if clean is not None:
                layers[str(layer)] = clean
        if not layers:
            return None
        promoted = sorted(ly for ly, d in layers.items() if d["promote"])
        rec: Dict[str, Any] = {
            "type": RECORD_TYPE,
            "id": str(uuid.uuid4()),
            "recorded_at": _now(now).isoformat(),
            "layers": layers,
            "promoted": promoted,
            "n_layers": len(layers),
            "n_promoted": len(promoted),
            "thresholds": dict(thresholds) if isinstance(thresholds, dict) else {},
            "sources": dict(sources) if isinstance(sources, dict) else {},
        }
        _append_jsonl(rec, _jsonl_path(path))
        return rec
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} audit row ignored: {exc}")
        return None


# --------------------------------------------------------------------------- #
# load_promotion_history
# --------------------------------------------------------------------------- #
def load_promotion_history(filters: Optional[dict] = None, *,
                           path: Optional[str] = None,
                           limit: Optional[int] = None) -> List[dict]:
    """Return audit rows, newest-first. Optional filters:
      * ``layer``   -> only rows that judged that layer
      * ``promote`` -> only rows whose ``layer`` verdict matches the bool
        (requires ``layer`` too; ignored otherwise)
    Audit lookup only; never raises."""
    try:
        p = _jsonl_path(path)
        if not os.path.exists(p):
            return []

        want_layer = None
        want_promote = None
        if isinstance(filters, dict):
            if filters.get("layer") is not None:
                want_layer = str(filters["layer"])
            if want_layer is not None and filters.get("promote") is not None:
                want_promote = bool(filters["promote"])

        rows: List[dict] = []
        with open(p, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict) or rec.get("type") != RECORD_TYPE:
                    continue
                layers = rec.get("layers")
                if not isinstance(layers, dict):
                    continue
                if want_layer is not None:
                    if want_layer not in layers:
                        continue
                    if want_promote is not None and \
                            bool(layers[want_layer].get("promote")) != want_promote:
                        continue
                rows.append(rec)

        rows.sort(key=lambda r: str(r.get("recorded_at", "")), reverse=True)
        if isinstance(limit, int) and limit >= 0:
            rows = rows[:limit]
        return rows
    except Exception:  # pragma: no cover - fail-open
        return []


# --------------------------------------------------------------------------- #
# summarize_history  /  format_daily_summary  (for an optional daily digest)
# --------------------------------------------------------------------------- #
def summarize_history(rows: Optional[List[dict]]) -> Dict[str, Any]:
    """Fold rows into a per-layer view: latest verdict, the timestamp of the
    earliest run where that layer cleared ALL gates (``first_promoted_at``), and
    a run-count. Pure; tolerant of junk rows. Newest-first or oldest-first input
    both work (sorted internally by ``recorded_at``)."""
    clean = [r for r in (rows or [])
             if isinstance(r, dict) and isinstance(r.get("layers"), dict)]
    # Oldest -> newest so "latest" and "first_promoted_at" are unambiguous.
    clean.sort(key=lambda r: str(r.get("recorded_at", "")))

    per_layer: Dict[str, Dict[str, Any]] = {}
    for rec in clean:
        ts = str(rec.get("recorded_at", ""))
        for layer, dec in rec["layers"].items():
            if not isinstance(dec, dict):
                continue
            promote = bool(dec.get("promote"))
            reasons = dec.get("reasons") if isinstance(dec.get("reasons"), list) else []
            slot = per_layer.setdefault(layer, {
                "runs": 0,
                "latest_promote": None,
                "latest_reasons": [],
                "latest_at": None,
                "first_promoted_at": None,
            })
            slot["runs"] += 1
            slot["latest_promote"] = promote
            slot["latest_reasons"] = [str(x) for x in reasons]
            slot["latest_at"] = ts
            if promote and slot["first_promoted_at"] is None:
                slot["first_promoted_at"] = ts

    return {
        "n_runs": len(clean),
        "latest_at": clean[-1].get("recorded_at") if clean else None,
        "layers": per_layer,
    }


def format_daily_summary(rows: Optional[List[dict]]) -> str:
    """Render ``summarize_history`` as a compact human/Telegram digest."""
    summ = summarize_history(rows)
    if summ["n_runs"] == 0:
        return "Oracle promotion audit: no checks recorded yet."
    lines = [
        "===== Oracle promotion audit — daily summary =====",
        f"runs recorded: {summ['n_runs']} (latest {summ['latest_at']})",
        "",
    ]
    for layer in sorted(summ["layers"]):
        s = summ["layers"][layer]
        verdict = "PROMOTE" if s["latest_promote"] else "HOLD"
        line = f"  {layer:20} {verdict:7} ({s['runs']} run(s))"
        if s["latest_promote"]:
            if s["first_promoted_at"]:
                line += f" — clear since {s['first_promoted_at']}"
        elif s["latest_reasons"]:
            line += " — " + ", ".join(s["latest_reasons"])
        lines.append(line)
    lines.append("")
    lines.append("ADVISORY ONLY — a human promotes by hand (paper first).")
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# Self-test (writes only to a throwaway temp path; no network, no creds)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    tmp = tempfile.mkdtemp(prefix="promotion_audit_")
    path = os.path.join(tmp, "audit.jsonl")

    def _dec(promote, reasons):
        return {"promote": promote, "reasons": list(reasons),
                "metrics": {"n_evals": 40}}

    try:
        # Empty ledger -> [] and an empty summary.
        if load_promotion_history({}, path=path) != []:
            print("FAIL: empty ledger should be []"); ok = False
        if summarize_history([])["n_runs"] != 0:
            print("FAIL: empty summary n_runs"); ok = False
        if "no checks recorded" not in format_daily_summary([]):
            print("FAIL: empty daily summary text"); ok = False

        # Run 1: executable_ev HOLDs, fill_model HOLDs.
        t1 = datetime(2024, 1, 1, tzinfo=timezone.utc)
        r1 = record_promotion_check(
            {"executable_ev": _dec(False, ["insufficient_evals"]),
             "fill_model": _dec(False, ["insufficient_fills"])},
            thresholds={"min_evals": 30.0},
            sources={"lab_json": "study.json"},
            path=path, now=t1)
        if not r1 or r1.get("id") is None:
            print("FAIL: record returned no row", r1); ok = False
        if r1 and (r1["n_promoted"] != 0 or r1["n_layers"] != 2):
            print("FAIL: run-1 counts", r1); ok = False
        if r1 and r1.get("promoted") != []:
            print("FAIL: run-1 promoted should be empty", r1); ok = False

        # Run 2 (later): executable_ev now PROMOTES; fill_model still HOLDs.
        t2 = datetime(2024, 1, 2, tzinfo=timezone.utc)
        r2 = record_promotion_check(
            {"executable_ev": _dec(True, []),
             "fill_model": _dec(False, ["fill_rate_bias_too_high"])},
            path=path, now=t2)
        if not r2 or r2["promoted"] != ["executable_ev"]:
            print("FAIL: run-2 promoted", r2); ok = False

        # History: newest-first, 2 rows.
        hist = load_promotion_history({}, path=path)
        if len(hist) != 2:
            print("FAIL: expected 2 rows", len(hist)); ok = False
        if hist and hist[0].get("recorded_at") != t2.isoformat():
            print("FAIL: not newest-first", hist[0].get("recorded_at")); ok = False

        # Filter by layer.
        ev_rows = load_promotion_history({"layer": "executable_ev"}, path=path)
        if len(ev_rows) != 2:
            print("FAIL: layer filter count", len(ev_rows)); ok = False
        # Filter by layer + promote=True -> only run 2.
        ev_prom = load_promotion_history(
            {"layer": "executable_ev", "promote": True}, path=path)
        if len(ev_prom) != 1 or ev_prom[0].get("recorded_at") != t2.isoformat():
            print("FAIL: layer+promote filter", ev_prom); ok = False
        # Unknown layer -> [].
        if load_promotion_history({"layer": "nope"}, path=path) != []:
            print("FAIL: unknown-layer filter should be []"); ok = False
        # limit.
        if len(load_promotion_history({}, path=path, limit=1)) != 1:
            print("FAIL: limit=1"); ok = False

        # Summary: executable_ev latest PROMOTE, first_promoted_at == t2;
        # fill_model latest HOLD.
        summ = summarize_history(hist)
        if summ["n_runs"] != 2:
            print("FAIL: summary n_runs", summ["n_runs"]); ok = False
        ev = summ["layers"].get("executable_ev", {})
        if not ev.get("latest_promote") or \
                ev.get("first_promoted_at") != t2.isoformat():
            print("FAIL: summary executable_ev", ev); ok = False
        fm = summ["layers"].get("fill_model", {})
        if fm.get("latest_promote") or fm.get("first_promoted_at") is not None:
            print("FAIL: summary fill_model", fm); ok = False
        txt = format_daily_summary(hist)
        if "PROMOTE" not in txt or "HOLD" not in txt or "ADVISORY ONLY" not in txt:
            print("FAIL: daily summary text", txt); ok = False

        # Fail-open: junk decisions -> None; junk filters never raise.
        if record_promotion_check(None, path=path) is not None:
            print("FAIL: None decisions should be None"); ok = False
        if record_promotion_check({}, path=path) is not None:
            print("FAIL: empty decisions should be None"); ok = False
        if record_promotion_check({"x": 42}, path=path) is not None:
            print("FAIL: non-dict decision -> None"); ok = False
        for junk in (None, 42, "x", []):
            try:
                load_promotion_history(junk, path=path)   # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                print("FAIL: history raised on junk", junk, exc); ok = False
        # summarize tolerates junk rows.
        if summarize_history([{"bad": 1}, 7, None])["n_runs"] != 0:  # type: ignore[list-item]
            print("FAIL: summary should skip junk rows"); ok = False

    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(tmp)
        except Exception:
            pass

    print("oracle.promotion_audit self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
