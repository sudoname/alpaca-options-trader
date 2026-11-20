"""Oracle 4.0.1 Stage F: validated, immutable forward-evidence artifacts.

Promotion gates must draw every forward performance measure, sample count,
calibration result, coverage figure and provenance claim from ONE validated
immutable forward-evidence artifact for the exact model, policy, baseline,
data window and cost assumptions.  This module defines that artifact's
schema, its canonical fingerprint, and a validator that rejects:

* mismatched model / hash / baseline / policy / dataset / window bindings,
* forged or summary-only evidence (no per-session rows),
* mark / paper / counterfactual outcomes mislabelled as actual fills,
* malformed types, NaN / non-finite values, inconsistent counts,
* broken cohort pairing (sessions without a baseline figure),
* out-of-order or out-of-window sessions, and stale reports.

It also hosts the prospective-holdout registry (register fresh holdout
sessions before tuning; refuse reuse after exposure) and the named human
review verification bound to a specific promotion report fingerprint.

Research-only: nothing here trades, deploys, or mutates production state.
"""
from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

EVIDENCE_KIND = "oracle4_forward_evidence"
HOLDOUT_KIND = "oracle4_prospective_holdout"
SCHEMA_VERSION = 1

# Only broker-confirmed live order fills count as actual fills.  Paper or
# counterfactual outcomes must never satisfy live-fill promotion gates.
FILL_SOURCES_ACTUAL = frozenset({"live_order"})
FILL_SOURCES_KNOWN = frozenset(
    {"live_order", "paper_order", "mark_derived", "counterfactual", "shadow_intent"}
)

_REQUIRED_STR_FIELDS = (
    "model_id",
    "model_hash",
    "baseline_id",
    "policy_id",
    "dataset_fingerprint",
    "code_git_sha",
    "generated_utc",
)

# Summary keys that, when present, must agree with the per-session rows.
_SUMMARY_KEYS = ("forward_sessions", "resolved_opportunities", "actual_fills")


# --------------------------------------------------------------------------- #
# primitives
# --------------------------------------------------------------------------- #

def _finite(x: Any) -> bool:
    """True for real finite numbers.  bool is explicitly rejected."""
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _nonneg_int(x: Any) -> bool:
    return isinstance(x, int) and not isinstance(x, bool) and x >= 0


def _utcnow() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def canonical_fingerprint(doc: Dict[str, Any]) -> str:
    """sha256 over canonical JSON of the document, excluding 'fingerprint'."""
    payload = {k: v for k, v in doc.items() if k != "fingerprint"}
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def sign(doc: Dict[str, Any]) -> Dict[str, Any]:
    """Return a copy of ``doc`` with its canonical fingerprint attached."""
    out = dict(doc)
    out["fingerprint"] = canonical_fingerprint(out)
    return out


def kish_ess(values: Sequence[float]) -> float:
    """Kish effective sample size (sum v)^2 / sum v^2; 0.0 when degenerate."""
    s1 = float(sum(values))
    s2 = float(sum(v * v for v in values))
    if s2 <= 0.0:
        return 0.0
    return (s1 * s1) / s2


# --------------------------------------------------------------------------- #
# forward-evidence validation
# --------------------------------------------------------------------------- #

def validate_forward_evidence(
    fe: Any, *, expected: Optional[Dict[str, Any]] = None
) -> List[str]:
    """Validate a forward-evidence artifact.  Returns a list of errors.

    An empty list means the artifact is structurally valid and consistent
    with the expected bindings; it does NOT mean the evidence passes any
    promotion gate.
    """
    errors: List[str] = []
    if not isinstance(fe, dict):
        return ["forward evidence must be a JSON object"]

    if fe.get("kind") != EVIDENCE_KIND:
        errors.append(
            f"kind must be {EVIDENCE_KIND!r} (got {fe.get('kind')!r})")
    if fe.get("schema_version") != SCHEMA_VERSION:
        errors.append(
            f"schema_version must be {SCHEMA_VERSION} "
            f"(got {fe.get('schema_version')!r})")

    for field in _REQUIRED_STR_FIELDS:
        v = fe.get(field)
        if not isinstance(v, str) or not v:
            errors.append(f"missing or empty required field: {field}")

    if not isinstance(fe.get("cost_assumptions"), dict) or not fe.get("cost_assumptions"):
        errors.append("cost_assumptions must be a non-empty object")

    cap = fe.get("capital_base")
    if not _finite(cap) or cap <= 0:
        errors.append("capital_base must be a finite positive number")

    # immutability: recompute the canonical fingerprint
    fp = fe.get("fingerprint")
    if not isinstance(fp, str) or canonical_fingerprint(fe) != fp:
        errors.append(
            "fingerprint mismatch: evidence altered after signing or forged")

    # window
    window = fe.get("window")
    w_start: Optional[str] = None
    w_end: Optional[str] = None
    if (not isinstance(window, dict)
            or not isinstance(window.get("start"), str)
            or not isinstance(window.get("end"), str)):
        errors.append("window must carry ISO string start and end")
    else:
        w_start, w_end = window["start"], window["end"]
        if w_start > w_end:
            errors.append("window start is after window end")

    # per-session rows (summary-only evidence is forgery-adjacent: reject)
    sessions = fe.get("sessions")
    session_ids: List[str] = []
    n_opp_total = 0
    n_fills_total = 0
    if not isinstance(sessions, list) or not sessions:
        present_summary = [k for k in _SUMMARY_KEYS if k in fe]
        if present_summary:
            errors.append(
                "summary-only evidence rejected: per-session rows are required "
                f"(summary keys present without sessions: {present_summary})")
        else:
            errors.append("sessions must be a non-empty list of per-session rows")
    else:
        prev: Optional[str] = None
        for i, row in enumerate(sessions):
            if not isinstance(row, dict):
                errors.append(f"sessions[{i}] is not an object")
                continue
            sid = row.get("session")
            if not isinstance(sid, str) or not sid:
                errors.append(f"sessions[{i}] missing session id")
                continue
            session_ids.append(sid)
            if prev is not None and sid <= prev:
                errors.append(
                    f"sessions out of chronological order at {sid!r} "
                    "(must be strictly increasing)")
            prev = sid
            if w_start is not None and (sid < w_start or sid > w_end):
                errors.append(f"session {sid} outside declared window")
            if not _finite(row.get("challenger_net_pnl")):
                errors.append(
                    f"session {sid}: challenger_net_pnl missing or non-finite")
            if not _finite(row.get("baseline_net_pnl")):
                errors.append(
                    f"session {sid}: cohort pairing broken — "
                    "baseline_net_pnl missing or non-finite")
            if not _nonneg_int(row.get("n_opportunities")):
                errors.append(
                    f"session {sid}: n_opportunities must be a non-negative integer")
            else:
                n_opp_total += row["n_opportunities"]
            if not _nonneg_int(row.get("n_fills")):
                errors.append(
                    f"session {sid}: n_fills must be a non-negative integer")
            else:
                n_fills_total += row["n_fills"]

    # staleness: a report generated before its window closed is stale/invalid
    gen = fe.get("generated_utc")
    if isinstance(gen, str) and gen and w_end is not None and gen[:10] < w_end[:10]:
        errors.append(
            "stale report: generated_utc precedes the evidence window end")

    # fills
    fills = fe.get("fills")
    if not isinstance(fills, list):
        errors.append("fills must be a list (may be empty)")
        fills = []
    sset = set(session_ids)
    n_live = 0
    for i, f in enumerate(fills):
        if not isinstance(f, dict):
            errors.append(f"fills[{i}] is not an object")
            continue
        src = f.get("source")
        if src not in FILL_SOURCES_KNOWN:
            errors.append(f"fills[{i}]: unknown fill source {src!r}")
        if session_ids and f.get("session") not in sset:
            errors.append(
                f"fills[{i}]: session {f.get('session')!r} not in per-session rows")
        if not _finite(f.get("net_pnl")):
            errors.append(f"fills[{i}]: net_pnl missing or non-finite")
        if f.get("fees_known") is not True:
            errors.append(
                f"fills[{i}]: fee coverage incomplete (fees_known is not true)")
        if src in FILL_SOURCES_ACTUAL:
            n_live += 1

    if session_ids and len(fills) != n_fills_total:
        errors.append(
            f"inconsistent counts: sum(sessions[].n_fills)={n_fills_total} "
            f"but len(fills)={len(fills)}")

    # summary keys, when present, must match what the rows derive
    if session_ids:
        derived = {
            "forward_sessions": len(session_ids),
            "resolved_opportunities": n_opp_total,
            "actual_fills": n_live,
        }
        for key, want in derived.items():
            if key in fe and fe[key] != want:
                errors.append(
                    f"inconsistent counts: {key}={fe[key]!r} but per-session "
                    f"rows derive {want}")

    # a bare boolean cannot upgrade evidence: claiming live-fill basis while
    # carrying non-live fills (or none) is an overclaim and invalidates it
    if fe.get("live_fill_based") is True:
        all_live = bool(fills) and all(
            isinstance(f, dict) and f.get("source") in FILL_SOURCES_ACTUAL
            for f in fills)
        if not all_live:
            errors.append(
                "live_fill_based overclaim: mark/paper/counterfactual outcomes "
                "may not be labelled as actual fills")

    # optional stress block: values present must be finite
    stress = fe.get("stress")
    if stress is not None:
        if not isinstance(stress, dict):
            errors.append("stress must be an object when present")
        else:
            for k in ("net_pnl_1_5x", "net_pnl_2_0x", "stop_overshoot_net_pnl"):
                if k in stress and stress[k] is not None and not _finite(stress[k]):
                    errors.append(f"stress.{k} must be finite when present")

    # optional calibration block
    cal = fe.get("calibration")
    if cal is not None:
        if not isinstance(cal, dict):
            errors.append("calibration must be an object when present")
        else:
            for side in ("challenger", "baseline"):
                blk = cal.get(side)
                if blk is not None and (
                        not isinstance(blk, dict) or not _finite(blk.get("brier"))):
                    errors.append(
                        f"calibration.{side} must carry a finite brier when present")

    # optional coverage block
    cov = fe.get("coverage")
    if cov is not None:
        groups = cov.get("groups") if isinstance(cov, dict) else None
        if not isinstance(groups, dict):
            errors.append("coverage.groups must be an object when coverage is present")
        else:
            for name, grp in groups.items():
                if (not isinstance(grp, dict)
                        or not _nonneg_int(grp.get("n_opportunities"))
                        or not _nonneg_int(grp.get("n_predictions"))):
                    errors.append(f"coverage group {name}: malformed counts")
                elif grp["n_predictions"] > grp["n_opportunities"]:
                    errors.append(
                        f"coverage group {name}: n_predictions exceeds "
                        "n_opportunities")

    # expected bindings: wrong model/hash/baseline/policy/dataset/window are
    # hard rejections, never warnings
    exp = expected or {}
    for key in ("model_id", "model_hash", "baseline_id", "policy_id",
                "dataset_fingerprint", "code_git_sha"):
        if key in exp and fe.get(key) != exp[key]:
            errors.append(
                f"binding mismatch: expected {key}={exp[key]!r} but evidence "
                f"carries {fe.get(key)!r}")
    if "window_start" in exp and w_start != exp["window_start"]:
        errors.append(
            f"binding mismatch: expected window start {exp['window_start']!r} "
            f"but evidence carries {w_start!r}")
    if "window_end" in exp and w_end != exp["window_end"]:
        errors.append(
            f"binding mismatch: expected window end {exp['window_end']!r} "
            f"but evidence carries {w_end!r}")

    return errors


# --------------------------------------------------------------------------- #
# prospective holdout registry
# --------------------------------------------------------------------------- #

def register_prospective_holdout(
    path: str | Path,
    sessions: Sequence[str],
    *,
    code_git_sha: str,
    artifact_hashes: Dict[str, str],
    registered_by: str,
    development_end: str,
) -> Dict[str, Any]:
    """Register a fresh prospective holdout BEFORE tuning.

    Every session must postdate ``development_end`` (all previously reviewed
    dates are development data).  The registry records code/artifact hashes
    and starts unexposed; once marked exposed it may never be reused.
    """
    sess = sorted(set(sessions))
    if not sess:
        raise ValueError("holdout must contain at least one session")
    bad = [s for s in sess if s <= development_end]
    if bad:
        raise ValueError(
            "prospective holdout must postdate development_end "
            f"{development_end!r}; offending sessions: {bad}")
    if not registered_by or not isinstance(registered_by, str):
        raise ValueError("registered_by must be a non-empty name")
    doc = sign({
        "kind": HOLDOUT_KIND,
        "schema_version": SCHEMA_VERSION,
        "sessions": sess,
        "code_git_sha": code_git_sha,
        "artifact_hashes": dict(artifact_hashes),
        "registered_by": registered_by,
        "development_end": development_end,
        "registered_utc": _utcnow(),
        "exposed": False,
        "exposure_log": [],
    })
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return doc


def load_holdout(path: str | Path) -> Dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def mark_holdout_exposed(path: str | Path, *, reason: str) -> Dict[str, Any]:
    """Record that the holdout has been looked at.  Irreversible by design."""
    doc = load_holdout(path)
    doc["exposed"] = True
    doc.setdefault("exposure_log", []).append(
        {"reason": reason, "at_utc": _utcnow()})
    doc = sign(doc)
    Path(path).write_text(
        json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return doc


def check_holdout_usable(reg: Any) -> List[str]:
    """Errors preventing use of a holdout registration (tamper / exposure)."""
    errors: List[str] = []
    if not isinstance(reg, dict):
        return ["holdout registration must be a JSON object"]
    if reg.get("kind") != HOLDOUT_KIND:
        errors.append(f"kind must be {HOLDOUT_KIND!r} (got {reg.get('kind')!r})")
    fp = reg.get("fingerprint")
    if not isinstance(fp, str) or canonical_fingerprint(reg) != fp:
        errors.append("holdout registration fingerprint mismatch: tampered")
    if reg.get("exposed"):
        errors.append(
            "holdout already exposed: refuse reuse after exposure "
            f"(log: {reg.get('exposure_log')})")
    if not reg.get("sessions"):
        errors.append("holdout registration carries no sessions")
    return errors


# --------------------------------------------------------------------------- #
# named human review, bound to a specific report fingerprint
# --------------------------------------------------------------------------- #

def verify_human_review(report: Any, review: Any) -> List[str]:
    """Verify an explicit named human review bound to a promotion report.

    A merely nonempty ``approved_by`` string is NOT sufficient: the review
    must name the reviewer, carry decision == 'approved', and bind to the
    exact report fingerprint it reviewed.
    """
    errors: List[str] = []
    if not isinstance(report, dict) or not isinstance(report.get("fingerprint"), str):
        errors.append("promotion report missing a fingerprint; cannot bind review")
        return errors
    if not isinstance(review, dict):
        return ["review must be a JSON object"]
    name = review.get("reviewer_name")
    if not isinstance(name, str) or not name.strip():
        if review.get("approved_by"):
            errors.append(
                "a bare approved_by string is not sufficient: review must "
                "carry reviewer_name")
        else:
            errors.append("review missing reviewer_name")
    if review.get("decision") != "approved":
        errors.append(
            f"review decision must be 'approved' (got {review.get('decision')!r})")
    if review.get("report_fingerprint") != report["fingerprint"]:
        errors.append(
            "review is not bound to this report: report_fingerprint mismatch")
    if ("report_schema_version" in review
            and review["report_schema_version"] != report.get("schema_version")):
        errors.append("review bound to a different report schema version")
    return errors
