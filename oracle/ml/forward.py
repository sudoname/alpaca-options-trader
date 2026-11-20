"""Per-book forward-evidence builder for Oracle 4.0.1 Stage G.

Turns resolved forward outcomes (from the shadow log joined to realized
trading history, or any other honest outcome resolver) into ONE validated,
signed, immutable forward-evidence artifact per strategy book, suitable for
``evidence.validate_forward_evidence`` and the promotion pipeline.

Honesty rules enforced here:

* Books never mix: an artifact is built for exactly one book; rows from
  other books are rejected, not silently merged.
* Cohort pairing: a session row requires BOTH challenger and baseline
  net P&L; outcomes that cannot be paired stay unresolved and are counted
  in comparison coverage instead of being dropped silently.
* Outcome sources keep their true labels (mark_derived / paper_order /
  counterfactual / shadow_intent); ``live_fill_based`` is True only when
  every fill is a broker-confirmed live order.
* The artifact is validated before it is returned or written; a builder
  that would emit invalid evidence raises instead.

Research-only: nothing here trades, deploys, or mutates production state.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from . import books as bk
from . import evidence as ev
from . import valuation as val

UTC = timezone.utc

RESOLVED = "resolved"
UNRESOLVED = "unresolved"


def _parse_utc(s: Any) -> Optional[datetime]:
    if isinstance(s, datetime):
        return s if s.tzinfo else s.replace(tzinfo=UTC)
    if not isinstance(s, str) or not s:
        return None
    try:
        dt = datetime.fromisoformat(s.replace("Z", "+00:00"))
    except ValueError:
        return None
    return dt if dt.tzinfo else dt.replace(tzinfo=UTC)


def _finite(x: Any) -> bool:
    return (isinstance(x, (int, float)) and not isinstance(x, bool)
            and x == x and abs(x) != float("inf"))


# --------------------------------------------------------------------------- #
# outcome resolution: shadow log x realized trading history
# --------------------------------------------------------------------------- #

def resolve_outcomes_from_history(
    scan_records: Iterable[Dict[str, Any]],
    trades: Iterable[Dict[str, Any]],
    *,
    source: str = "mark_derived",
    fees_known: bool = True,
) -> List[Dict[str, Any]]:
    """Join shadow scans to realized closed trades, per strategy book.

    For every scan whose challenger decided ``trade``, look for a closed
    trade on the challenger's selected symbol AND one on the baseline's top
    symbol, entered on the same session at or after the scan time.  Both
    legs must resolve for cohort pairing; otherwise the outcome row stays
    ``unresolved`` with a reason.  P&L figures come straight from the
    history rows under their true ``source`` label — nothing is invented.
    """
    if source not in ev.FILL_SOURCES_KNOWN:
        raise ValueError(f"unknown outcome source {source!r}")

    # index trades by (symbol, session) -> list of (entry_at, pnl)
    by_key: Dict[Tuple[str, str], List[Tuple[datetime, float]]] = {}
    for t in trades:
        sym = str(t.get("symbol") or "")
        at = _parse_utc(t.get("entry_time"))
        pnl = t.get("pnl")
        if not sym or at is None or not _finite(pnl):
            continue
        by_key.setdefault((sym, at.date().isoformat()), []).append(
            (at, float(pnl)))
    for rows in by_key.values():
        rows.sort(key=lambda r: r[0])

    def _match(sym: Optional[str], session: str,
               after: datetime) -> Optional[float]:
        if not sym:
            return None
        for at, pnl in by_key.get((str(sym), session), []):
            if at >= after:
                return pnl
        return None

    out: List[Dict[str, Any]] = []
    for rec in scan_records:
        if not isinstance(rec, dict):
            continue
        at = _parse_utc(rec.get("at"))
        if at is None:
            continue
        session = at.date().isoformat()
        scan_id = str(rec.get("scan_id") or "")
        ch = rec.get("challenger") or {}
        base = rec.get("baseline") or {}
        if ch.get("decision") != "trade":
            continue
        sel = ch.get("selected_symbol")
        top = base.get("top_symbol") or base.get("top_sym")
        contract = val.parse_occ_symbol(sel) or {}
        book = bk.classify_book(contract.get("expiration"), at)
        group_id = bk.candidate_group_id(
            scan_id, contract.get("underlying"), contract.get("expiration"))
        row: Dict[str, Any] = {
            "session": session,
            "scan_id": scan_id,
            "book": book,
            "group_id": group_id,
            "challenger_symbol": sel,
            "baseline_symbol": top,
        }
        ch_pnl = _match(sel, session, at)
        base_pnl = _match(top, session, at)
        if ch_pnl is None or base_pnl is None:
            missing = []
            if ch_pnl is None:
                missing.append("challenger")
            if base_pnl is None:
                missing.append("baseline")
            row.update({
                "resolution_state": UNRESOLVED,
                "reason": f"no realized trade for: {','.join(missing)}",
            })
        else:
            row.update({
                "resolution_state": RESOLVED,
                "challenger_net_pnl": ch_pnl,
                "baseline_net_pnl": base_pnl,
                "fill_source": source,
                "fees_known": bool(fees_known),
            })
        out.append(row)
    return out


# --------------------------------------------------------------------------- #
# per-book artifact builder
# --------------------------------------------------------------------------- #

def build_forward_evidence(
    outcomes: Iterable[Dict[str, Any]],
    *,
    book: str,
    model_id: str,
    model_hash: str,
    baseline_id: str,
    policy_id: str,
    dataset_fingerprint: str,
    code_git_sha: str,
    cost_assumptions: Dict[str, Any],
    capital_base: float,
    stress: Optional[Dict[str, Any]] = None,
    calibration: Optional[Dict[str, Any]] = None,
    generated_utc: Optional[str] = None,
) -> Dict[str, Any]:
    """Build ONE signed forward-evidence artifact for ONE strategy book.

    Raises ``ValueError`` on rows from other books (no mixing), malformed
    resolved rows, or when the finished artifact fails validation.
    """
    rows = list(outcomes)
    wrong = [r for r in rows if r.get("book") != book]
    if wrong:
        raise ValueError(
            f"book mixing rejected: {len(wrong)} outcome rows belong to "
            f"other books (artifact is for {book!r})")

    resolved: List[Dict[str, Any]] = []
    unresolved: List[Dict[str, Any]] = []
    for i, r in enumerate(rows):
        state = r.get("resolution_state")
        if state == RESOLVED:
            if not _finite(r.get("challenger_net_pnl")):
                raise ValueError(f"outcomes[{i}]: challenger_net_pnl not finite")
            if not _finite(r.get("baseline_net_pnl")):
                raise ValueError(
                    f"outcomes[{i}]: baseline_net_pnl not finite "
                    "(cohort pairing requires both arms)")
            if r.get("fill_source") not in ev.FILL_SOURCES_KNOWN:
                raise ValueError(
                    f"outcomes[{i}]: unknown fill_source "
                    f"{r.get('fill_source')!r}")
            if r.get("fees_known") is not True:
                raise ValueError(
                    f"outcomes[{i}]: fees_known must be True for a "
                    "resolved outcome")
            if not isinstance(r.get("session"), str) or not r["session"]:
                raise ValueError(f"outcomes[{i}]: missing session")
            resolved.append(r)
        elif state == UNRESOLVED:
            unresolved.append(r)
        else:
            raise ValueError(
                f"outcomes[{i}]: resolution_state must be "
                f"'{RESOLVED}' or '{UNRESOLVED}' (got {state!r})")

    if not resolved:
        raise ValueError(
            f"no resolved outcomes for book {book!r}: cannot build "
            "forward evidence (unresolved rows: %d)" % len(unresolved))

    # per-session aggregation (strictly increasing session ids by sort)
    by_session: Dict[str, List[Dict[str, Any]]] = {}
    for r in resolved:
        by_session.setdefault(r["session"], []).append(r)
    session_rows: List[Dict[str, Any]] = []
    fills: List[Dict[str, Any]] = []
    for sid in sorted(by_session):
        srows = by_session[sid]
        session_rows.append({
            "session": sid,
            "challenger_net_pnl": float(
                sum(r["challenger_net_pnl"] for r in srows)),
            "baseline_net_pnl": float(
                sum(r["baseline_net_pnl"] for r in srows)),
            "n_opportunities": len(srows),
            "n_fills": len(srows),
        })
        for r in srows:
            fills.append({
                "source": r["fill_source"],
                "session": sid,
                "symbol": r.get("challenger_symbol"),
                "net_pnl": float(r["challenger_net_pnl"]),
                "fees_known": True,
            })

    # comparison coverage per candidate group (resolved vs total)
    group_total: Dict[str, int] = {}
    group_resolved: Dict[str, int] = {}
    for r in rows:
        gid = str(r.get("group_id") or "UNKNOWN")
        group_total[gid] = group_total.get(gid, 0) + 1
        if r.get("resolution_state") == RESOLVED:
            group_resolved[gid] = group_resolved.get(gid, 0) + 1
    coverage = {
        "book": book,
        "groups": {
            gid: {"n_opportunities": n,
                  "n_predictions": group_resolved.get(gid, 0)}
            for gid, n in sorted(group_total.items())
        },
        "n_unresolved": len(unresolved),
        "resolution_rate": (len(resolved) / len(rows)) if rows else 0.0,
    }

    all_live = bool(fills) and all(
        f["source"] in ev.FILL_SOURCES_ACTUAL for f in fills)

    w_start = session_rows[0]["session"]
    w_end = session_rows[-1]["session"]
    gen = generated_utc or datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")

    doc: Dict[str, Any] = {
        "kind": ev.EVIDENCE_KIND,
        "schema_version": ev.SCHEMA_VERSION,
        "book": book,
        "model_id": model_id,
        "model_hash": model_hash,
        "baseline_id": baseline_id,
        "policy_id": policy_id,
        "dataset_fingerprint": dataset_fingerprint,
        "code_git_sha": code_git_sha,
        "generated_utc": gen,
        "cost_assumptions": dict(cost_assumptions),
        "capital_base": float(capital_base),
        "window": {"start": w_start, "end": w_end},
        "sessions": session_rows,
        "fills": fills,
        "forward_sessions": len(session_rows),
        "resolved_opportunities": sum(
            r["n_opportunities"] for r in session_rows),
        "actual_fills": sum(
            1 for f in fills if f["source"] in ev.FILL_SOURCES_ACTUAL),
        "live_fill_based": all_live,
        "coverage": coverage,
        "unresolved": [
            {"session": r.get("session"), "group_id": r.get("group_id"),
             "reason": r.get("reason")}
            for r in unresolved
        ],
        "note": ("per-book forward evidence; outcomes carry their true "
                 "source labels; unresolved outcomes counted, not dropped"),
    }
    if stress is not None:
        doc["stress"] = dict(stress)
    if calibration is not None:
        doc["calibration"] = dict(calibration)

    signed = ev.sign(doc)
    errors = ev.validate_forward_evidence(signed)
    if errors:
        raise ValueError(
            "builder produced invalid forward evidence: " + "; ".join(errors))
    return signed


def write_forward_evidence(doc: Dict[str, Any], out_dir: str | Path,
                           *, filename: Optional[str] = None) -> str:
    """Write a signed artifact as ``forward_evidence_<book>.json``."""
    book = str(doc.get("book") or "unknown")
    name = filename or f"forward_evidence_{book}.json"
    p = Path(out_dir) / name
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(doc, indent=2, sort_keys=True), encoding="utf-8")
    return str(p)


# --------------------------------------------------------------------------- #
# CLI: shadow log + trading history -> per-book artifacts (offline)
# --------------------------------------------------------------------------- #

def _load_jsonl(path: str) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return out


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--log", required=True, help="shadow scan JSONL")
    ap.add_argument("--history", required=True,
                    help="trading_history.json ({'trades': [...]})")
    ap.add_argument("--out-dir", default=os.path.join("artifacts", "oracle4"))
    ap.add_argument("--model-id", required=True)
    ap.add_argument("--model-hash", required=True)
    ap.add_argument("--baseline-id", default="production_score_rank")
    ap.add_argument("--policy-id", default="oracle4_shadow_rank_v1")
    ap.add_argument("--dataset-fingerprint", required=True)
    ap.add_argument("--code-git-sha", required=True)
    ap.add_argument("--capital-base", type=float, default=100_000.0)
    ap.add_argument("--source", default="mark_derived",
                    choices=sorted(ev.FILL_SOURCES_KNOWN))
    args = ap.parse_args(argv)

    records = _load_jsonl(args.log)
    with open(args.history, encoding="utf-8") as fh:
        trades = (json.load(fh) or {}).get("trades", [])
    outcomes = resolve_outcomes_from_history(records, trades,
                                             source=args.source)
    cost = {"friction_multiplier": 1.0,
            "basis": f"history net P&L as recorded ({args.source})"}
    n_written = 0
    for book in sorted({r.get("book") for r in outcomes if r.get("book")}):
        rows = [r for r in outcomes if r.get("book") == book]
        try:
            doc = build_forward_evidence(
                rows, book=book, model_id=args.model_id,
                model_hash=args.model_hash, baseline_id=args.baseline_id,
                policy_id=args.policy_id,
                dataset_fingerprint=args.dataset_fingerprint,
                code_git_sha=args.code_git_sha, cost_assumptions=cost,
                capital_base=args.capital_base)
        except ValueError as exc:
            print(f"[{book}] skipped: {exc}")
            continue
        path = write_forward_evidence(doc, args.out_dir)
        print(f"[{book}] wrote {path} "
              f"(sessions={doc['forward_sessions']}, "
              f"resolved={doc['resolved_opportunities']}, "
              f"actual_fills={doc['actual_fills']})")
        n_written += 1
    if not outcomes:
        print("no challenger trade decisions in the shadow log; "
              "nothing to resolve")
    return 0 if n_written or not outcomes else 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
