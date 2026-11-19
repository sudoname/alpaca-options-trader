"""Build the versioned Oracle 4.0 feature/label store from a research ledger.

Usage:
    python -m oracle.ml.dataset --ledger research_ledger_server_2026-11-13.db \
        --from 2026-11-04 --to 2026-11-14 \
        --output artifacts/oracle4_dataset.db \
        --manifest artifacts/oracle4_dataset_manifest.json

Outputs:
  * a sqlite store (underlying_obs, features, labels, outcomes, meta);
  * a JSON manifest with input hash, config, ingest reports, coverage report
    (naming what is missing and why), and a deterministic ``dataset_hash``.

Re-running on the same inputs with the same config reproduces the same
``dataset_hash`` bit-for-bit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sqlite3
from dataclasses import asdict
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .features import FeatureConfig, ObsIndex, build_feature_row
from .ingest import connect, load_decisions, load_events, load_underlying_obs
from .labels import LabelConfig, build_iv_index, build_labels, classify_outcome
from .schema import (
    SCHEMA_VERSION,
    FeatureRow,
    LabelRow,
    OutcomeRecord,
    iso_utc,
    rows_hash,
)

__all__ = ["build_dataset", "main"]


def _sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    return datetime.fromisoformat(s).replace(tzinfo=timezone.utc) \
        if "T" not in s else datetime.fromisoformat(s).astimezone(timezone.utc)


# --------------------------------------------------------------------------- #
# store writer
# --------------------------------------------------------------------------- #

_DDL = """
CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT);
CREATE TABLE underlying_obs (
    underlying TEXT NOT NULL, event_at TEXT NOT NULL,
    available_at TEXT NOT NULL, price REAL NOT NULL,
    source_snapshot_id TEXT, PRIMARY KEY (underlying, event_at));
CREATE TABLE features (
    decision_id TEXT PRIMARY KEY, snapshot_id TEXT, as_of TEXT NOT NULL,
    available_at TEXT NOT NULL, underlying TEXT NOT NULL, symbol TEXT,
    kind TEXT NOT NULL, session TEXT NOT NULL,
    features_json TEXT NOT NULL, quality_json TEXT NOT NULL);
CREATE TABLE labels (
    decision_id TEXT NOT NULL, horizon TEXT NOT NULL,
    label_type TEXT NOT NULL, value REAL, status TEXT NOT NULL,
    label_end TEXT, available_at TEXT, meta_json TEXT NOT NULL,
    PRIMARY KEY (decision_id, horizon, label_type));
CREATE TABLE outcomes (
    decision_id TEXT PRIMARY KEY, status TEXT NOT NULL,
    gross_pnl_pct REAL, net_pnl_dollars REAL, net_pnl_pct REAL,
    reason TEXT, fill_price REAL, fill_qty REAL, outcome_at TEXT,
    meta_json TEXT NOT NULL);
CREATE INDEX idx_labels_horizon ON labels (horizon, label_type, status);
CREATE INDEX idx_features_session ON features (session);
"""


def _write_store(path: str,
                 obs_by_u: Dict[str, list],
                 feature_rows: List[FeatureRow],
                 label_rows: List[LabelRow],
                 outcome_rows: List[OutcomeRecord],
                 meta: Dict[str, str]) -> None:
    if os.path.exists(path):
        os.remove(path)
    conn = sqlite3.connect(path)
    try:
        conn.executescript(_DDL)
        conn.executemany(
            "INSERT INTO meta VALUES (?, ?)", sorted(meta.items()))
        for series in obs_by_u.values():
            conn.executemany(
                "INSERT INTO underlying_obs VALUES (?, ?, ?, ?, ?)",
                [(o.underlying, iso_utc(o.event_at), iso_utc(o.available_at),
                  o.price, o.source_snapshot_id) for o in series])
        conn.executemany(
            "INSERT INTO features VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r.decision_id, r.snapshot_id, iso_utc(r.as_of),
              iso_utc(r.available_at), r.underlying, r.symbol, r.kind,
              r.session, json.dumps(r.features, sort_keys=True),
              json.dumps(r.quality, sort_keys=True, default=str))
             for r in feature_rows])
        conn.executemany(
            "INSERT INTO labels VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            [(r.decision_id, r.horizon, r.label_type, r.value, r.status.value,
              iso_utc(r.label_end), iso_utc(r.available_at),
              json.dumps(r.meta, sort_keys=True, default=str))
             for r in label_rows])
        conn.executemany(
            "INSERT INTO outcomes VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            [(r.decision_id, r.status.value, r.gross_pnl_pct,
              r.net_pnl_dollars, r.net_pnl_pct, r.reason, r.fill_price,
              r.fill_qty, iso_utc(r.outcome_at),
              json.dumps(r.meta, sort_keys=True, default=str))
             for r in outcome_rows])
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# coverage report
# --------------------------------------------------------------------------- #

def _coverage_report(feature_rows: List[FeatureRow],
                     label_rows: List[LabelRow],
                     outcome_rows: List[OutcomeRecord]) -> Dict[str, Any]:
    feat_counts: Dict[str, int] = {}
    n = len(feature_rows)
    for r in feature_rows:
        for k, v in r.features.items():
            if v is not None:
                feat_counts[k] = feat_counts.get(k, 0) + 1
    feature_coverage = {
        k: round(c / n, 4) for k, c in sorted(feat_counts.items())
    } if n else {}
    if n:
        all_keys = set()
        for r in feature_rows:
            all_keys.update(r.features.keys())
        for k in sorted(all_keys):
            feature_coverage.setdefault(k, 0.0)

    label_status: Dict[str, Dict[str, int]] = {}
    for r in label_rows:
        key = f"{r.horizon}/{r.label_type}"
        bucket = label_status.setdefault(key, {})
        bucket[r.status.value] = bucket.get(r.status.value, 0) + 1

    outcome_status: Dict[str, int] = {}
    for r in outcome_rows:
        outcome_status[r.status.value] = \
            outcome_status.get(r.status.value, 0) + 1

    gaps: List[str] = []
    for key, bucket in sorted(label_status.items()):
        total = sum(bucket.values())
        bad = bucket.get("missing", 0) + bucket.get("censored", 0)
        if total and bad / total > 0.5:
            gaps.append(
                f"{key}: {bad}/{total} unresolved "
                f"(missing={bucket.get('missing', 0)}, "
                f"censored={bucket.get('censored', 0)})")
    for k, frac in feature_coverage.items():
        if frac < 0.5:
            gaps.append(f"feature {k}: only {frac:.1%} coverage")
    if outcome_status.get("fills_actual", 0) < \
            outcome_status.get("mark_derived", 1):
        gaps.append(
            "outcomes dominated by mark_derived P&L; fills_actual booking "
            "only began at baseline freeze")

    return {
        "feature_coverage": feature_coverage,
        "label_status_by_horizon_type": {
            k: dict(sorted(v.items())) for k, v in sorted(label_status.items())
        },
        "outcome_status_counts": dict(sorted(outcome_status.items())),
        "known_gaps": gaps,
    }


# --------------------------------------------------------------------------- #
# build
# --------------------------------------------------------------------------- #

def build_dataset(
    ledger_path: str,
    output_path: str,
    manifest_path: str,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    kinds: Optional[List[str]] = None,
    feature_config: FeatureConfig = FeatureConfig(),
    label_config: LabelConfig = LabelConfig(),
    seed: int = 42,
) -> Dict[str, Any]:
    """Build store + manifest; returns the manifest dict."""
    ledger = connect(ledger_path)
    try:
        obs_by_u, obs_report = load_underlying_obs(ledger)  # full series:
        # features need pre-window history and labels need post-window future.
        decisions, dec_report = load_decisions(
            ledger, start=start, end=end, kinds=kinds)
        events_by_dec, ev_report = load_events(ledger)
    finally:
        ledger.close()

    index = ObsIndex(obs_by_u)
    coverage_end = max(
        (series[-1].event_at for series in obs_by_u.values() if series),
        default=datetime(1970, 1, 1, tzinfo=timezone.utc))
    iv_index = build_iv_index(decisions)

    feature_rows: List[FeatureRow] = []
    label_rows: List[LabelRow] = []
    outcome_rows: List[OutcomeRecord] = []
    for d in decisions:
        feature_rows.append(build_feature_row(d, index, feature_config))
        label_rows.extend(build_labels(
            d, index, coverage_end,
            events=events_by_dec.get(d.decision_id),
            iv_index=iv_index, config=label_config))
        outcome_rows.append(
            classify_outcome(d, events_by_dec.get(d.decision_id, [])))

    dataset_hash = rows_hash(
        list(feature_rows) + list(label_rows) + list(outcome_rows))

    coverage = _coverage_report(feature_rows, label_rows, outcome_rows)
    manifest: Dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": iso_utc(datetime.now(timezone.utc)),
        "input": {
            "ledger_path": ledger_path,
            "ledger_sha256": _sha256_file(ledger_path),
        },
        "window": {"from": iso_utc(start), "to": iso_utc(end),
                   "kinds": kinds or ["candidate", "shadow_candidate"],
                   "coverage_end": iso_utc(coverage_end)},
        "config": {
            "feature": asdict(feature_config),
            "label": asdict(label_config),
            "seed": seed,
        },
        "ingest_reports": {
            "underlying_obs": obs_report,
            "decisions": dec_report,
            "events": ev_report,
        },
        "counts": {
            "underlyings": len(obs_by_u),
            "underlying_obs": sum(len(s) for s in obs_by_u.values()),
            "decisions": len(decisions),
            "feature_rows": len(feature_rows),
            "label_rows": len(label_rows),
            "outcome_rows": len(outcome_rows),
            "sessions": len({r.session for r in feature_rows}),
        },
        "coverage": coverage,
        "dataset_hash": dataset_hash,
    }

    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "dataset_hash": dataset_hash,
        "ledger_sha256": manifest["input"]["ledger_sha256"],
        "generated_at_utc": manifest["generated_at_utc"],
    }
    _write_store(output_path, obs_by_u, feature_rows, label_rows,
                 outcome_rows, meta)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oracle.ml.dataset",
        description="Build the Oracle 4.0 feature/label store (research-only).")
    ap.add_argument("--ledger", required=True, help="research ledger sqlite")
    ap.add_argument("--from", dest="start", default=None,
                    help="UTC start date/time (inclusive), e.g. 2026-11-04")
    ap.add_argument("--to", dest="end", default=None,
                    help="UTC end date/time (exclusive)")
    ap.add_argument("--output", required=True, help="output sqlite store")
    ap.add_argument("--manifest", required=True, help="output manifest JSON")
    ap.add_argument("--kinds", nargs="*", default=None,
                    choices=["candidate", "shadow_candidate"],
                    help="decision kinds to include (default: both)")
    ap.add_argument("--neutral-band", type=float, default=0.001)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args(argv)

    label_config = LabelConfig(neutral_band=args.neutral_band)
    manifest = build_dataset(
        ledger_path=args.ledger,
        output_path=args.output,
        manifest_path=args.manifest,
        start=_parse_date(args.start),
        end=_parse_date(args.end),
        kinds=args.kinds,
        label_config=label_config,
        seed=args.seed,
    )
    c = manifest["counts"]
    print(f"dataset_hash={manifest['dataset_hash']}")
    print(f"decisions={c['decisions']} features={c['feature_rows']} "
          f"labels={c['label_rows']} outcomes={c['outcome_rows']} "
          f"sessions={c['sessions']}")
    for gap in manifest["coverage"]["known_gaps"]:
        print(f"GAP: {gap}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
