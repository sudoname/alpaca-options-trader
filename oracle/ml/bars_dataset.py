"""Bars-based training dataset builder for Oracle 4.0.2 (P1-5).

Turns multi-year underlying bar history (the Stage-A ``BarStore`` sqlite,
filled by ``bars_backfill.py`` / ``bars_feed.py``) into the SAME dataset
store schema that ``oracle.ml.dataset`` produces from the research ledger,
so every downstream consumer (``oracle.lab.purged_validation`` loaders,
the training CLI, registry, forecast providers) works unchanged on years
of real market history instead of a handful of live ledger sessions.

Pipeline:
  bars -> UnderlyingObs (event_at = available_at = bar END, price = close)
       -> ObsIndex -> synthesized ``DecisionRecord``s (kind="bars_synthetic")
          at sampled RTH times -> build_feature_row / build_labels /
          classify_outcome -> ``dataset._write_store`` + manifest.

Rules (same contract as the ledger dataset builder):
  * research-path only -- nothing in the production trading path imports
    this module; the BarStore is opened strictly read-only;
  * point-in-time correctness: an observation becomes available only at
    the bar's END (completed-bar rule), so ``ObsIndex.asof`` can never
    leak a forming close into features;
  * a sample time with no sufficiently fresh completed bar is skipped and
    counted (``skipped_no_asof`` / ``skipped_stale_asof``), never
    interpolated;
  * synthesized decisions carry no option/quote/score fields (``None``)
    and ``accepted=False`` -- ``classify_outcome`` marks every row
    COUNTERFACTUAL, and option-dependent labels (iv_change, first_touch)
    report MISSING with an explicit reason, never a guessed value;
  * determinism: identical bars yield an identical ``dataset_hash``.

Usage:
    python -m oracle.ml.bars_dataset \
        --bars-db artifacts/oracle4/bars/bars.sqlite \
        --symbols SPY QQQ IWM \
        --output artifacts/oracle402/bars_dataset.sqlite \
        --manifest artifacts/oracle402/bars_dataset_manifest.json
"""
from __future__ import annotations

import argparse
import json
import os
from dataclasses import asdict
from datetime import date, datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from .bars import BarStore, BarsUnavailable, rth_window_utc, sha256_file
from .dataset import _coverage_report, _write_store
from .features import FeatureConfig, ObsIndex, build_feature_row
from .labels import LabelConfig, build_labels, classify_outcome
from .schema import (SCHEMA_VERSION, DecisionRecord, FeatureRow, LabelRow,
                     OutcomeRecord, UnderlyingObs, iso_utc, rows_hash)

UTC = timezone.utc

DECISION_KIND = "bars_synthetic"

DEFAULT_RESOLUTION_S = 300
DEFAULT_SAMPLE_EVERY_MIN = 30
DEFAULT_WARMUP_MIN = 60
DEFAULT_END_OFFSET_MIN = 60


def _bars_to_obs(symbol: str, bars) -> List[UnderlyingObs]:
    """One observation per completed bar: price=close, time=bar END (PIT)."""
    out: List[UnderlyingObs] = []
    for b in bars:
        end = b.end_utc
        out.append(UnderlyingObs(
            underlying=symbol,
            event_at=end,
            available_at=end,
            price=float(b.close),
            source_snapshot_id=f"bar:{symbol}:{iso_utc(b.ts_utc)}",
        ))
    out.sort(key=lambda o: o.event_at)
    return out


def _sample_times(
    session_date_iso: str,
    *,
    sample_every_min: int,
    warmup_min: int,
    end_offset_min: int,
) -> List[datetime]:
    """RTH sampling grid for one session: [open+warmup, close-end_offset]."""
    open_utc, close_utc = rth_window_utc(date.fromisoformat(session_date_iso))
    t = open_utc + timedelta(minutes=int(warmup_min))
    last = close_utc - timedelta(minutes=int(end_offset_min))
    step = timedelta(minutes=max(1, int(sample_every_min)))
    out: List[datetime] = []
    while t <= last:
        out.append(t)
        t += step
    return out


def _make_decision(symbol: str, t: datetime) -> DecisionRecord:
    did = f"bars-{symbol}-{t.strftime('%Y%m%dT%H%M%SZ')}"
    return DecisionRecord(
        decision_id=did,
        snapshot_id=did,
        kind=DECISION_KIND,
        accepted=False,
        reject_code=None,
        as_of=t,
        available_at=t,
        underlying=symbol,
        symbol=None,
        expiration=None,
        strike=None,
        option_type=None,
        multiplier=None,
        quote_bid=None,
        quote_ask=None,
        underlying_price=None,  # spot comes from ObsIndex.asof (PIT)
        iv=None,
        delta=None,
        direction=None,
        score=None,
        p_win=None,
        ev=None,
    )


def build_bars_dataset(
    bars_db: str,
    output_path: str,
    manifest_path: str,
    symbols: Sequence[str],
    *,
    resolution_s: int = DEFAULT_RESOLUTION_S,
    start: Optional[datetime] = None,
    end: Optional[datetime] = None,
    sample_every_min: int = DEFAULT_SAMPLE_EVERY_MIN,
    warmup_min: int = DEFAULT_WARMUP_MIN,
    end_offset_min: int = DEFAULT_END_OFFSET_MIN,
    feature_config: FeatureConfig = FeatureConfig(),
    label_config: LabelConfig = LabelConfig(),
    seed: int = 42,
) -> Dict:
    """Build a ledger-schema dataset sqlite from BarStore history.

    Returns the manifest dict (also written to ``manifest_path``).
    Symbols whose bars are unavailable are skipped and reported in
    ``symbols.skipped`` -- never guessed.
    """
    syms = sorted({str(s).strip().upper() for s in symbols
                   if s and str(s).strip()})

    obs_by_u: Dict[str, List[UnderlyingObs]] = {}
    sessions_by_sym: Dict[str, List[str]] = {}
    bars_coverage: Dict[str, Dict] = {}
    skipped_symbols: Dict[str, str] = {}
    bars_total = 0

    with BarStore(bars_db, readonly=True) as store:
        for sym in syms:
            try:
                bars = store.load_bars(sym, resolution_s)
            except BarsUnavailable as exc:
                skipped_symbols[sym] = str(exc)
                continue
            obs_by_u[sym] = _bars_to_obs(sym, bars)
            sessions_by_sym[sym] = sorted({b.session_date for b in bars})
            bars_coverage[sym] = store.coverage(sym, resolution_s)
            bars_total += len(bars)

    index = ObsIndex(obs_by_u)
    coverage_end = max(
        (series[-1].event_at for series in obs_by_u.values() if series),
        default=datetime(1970, 1, 1, tzinfo=UTC),
    )

    sampling_report = {
        "candidate_times": 0,
        "sampled": 0,
        "skipped_no_asof": 0,
        "skipped_stale_asof": 0,
    }
    decisions: List[DecisionRecord] = []
    for sym in sorted(obs_by_u):
        for sd in sessions_by_sym.get(sym, []):
            for t in _sample_times(sd,
                                   sample_every_min=sample_every_min,
                                   warmup_min=warmup_min,
                                   end_offset_min=end_offset_min):
                if start is not None and t < start:
                    continue
                if end is not None and t >= end:
                    continue
                sampling_report["candidate_times"] += 1
                obs = index.asof(sym, t)
                if obs is None:
                    sampling_report["skipped_no_asof"] += 1
                    continue
                # freshness: the latest completed bar must be at most one
                # bar-interval old at the sample time, else skip (gap/halt)
                if (t - obs.event_at).total_seconds() > resolution_s:
                    sampling_report["skipped_stale_asof"] += 1
                    continue
                sampling_report["sampled"] += 1
                decisions.append(_make_decision(sym, t))
    decisions.sort(key=lambda d: (d.as_of, d.decision_id))

    feature_rows: List[FeatureRow] = []
    label_rows: List[LabelRow] = []
    outcome_rows: List[OutcomeRecord] = []
    for d in decisions:
        feature_rows.append(build_feature_row(d, index, feature_config))
        label_rows.extend(build_labels(d, index, coverage_end,
                                       events=None, iv_index=None,
                                       config=label_config))
        outcome_rows.append(classify_outcome(d, []))

    dataset_hash = rows_hash(
        list(feature_rows) + list(label_rows) + list(outcome_rows))

    generated_at = iso_utc(datetime.now(UTC))
    bars_db_sha256 = sha256_file(bars_db)
    manifest: Dict = {
        "schema_version": SCHEMA_VERSION,
        "builder": "bars_dataset",
        "generated_at_utc": generated_at,
        "input": {
            "bars_db": os.path.abspath(bars_db),
            "bars_db_sha256": bars_db_sha256,
            "resolution_s": resolution_s,
        },
        "window": {
            "from": iso_utc(start) if start else None,
            "to": iso_utc(end) if end else None,
            "coverage_end": iso_utc(coverage_end),
        },
        "sampling": {
            "sample_every_min": sample_every_min,
            "warmup_min": warmup_min,
            "end_offset_min": end_offset_min,
            "decision_kind": DECISION_KIND,
            "report": sampling_report,
        },
        "config": {
            "feature": asdict(feature_config),
            "label": asdict(label_config),
            "seed": seed,
        },
        "symbols": {
            "requested": syms,
            "built": sorted(obs_by_u),
            "skipped": skipped_symbols,
        },
        "bars_coverage": bars_coverage,
        "counts": {
            "underlyings": len(obs_by_u),
            "underlying_obs": sum(len(v) for v in obs_by_u.values()),
            "bars": bars_total,
            "decisions": len(decisions),
            "feature_rows": len(feature_rows),
            "label_rows": len(label_rows),
            "outcome_rows": len(outcome_rows),
            "sessions": len({r.session for r in feature_rows}),
        },
        "coverage": _coverage_report(feature_rows, label_rows, outcome_rows),
        "dataset_hash": dataset_hash,
    }

    for p in (output_path, manifest_path):
        d = os.path.dirname(os.path.abspath(p))
        if d:
            os.makedirs(d, exist_ok=True)
    meta = {
        "schema_version": str(SCHEMA_VERSION),
        "dataset_hash": dataset_hash,
        "bars_db_sha256": bars_db_sha256,
        "generated_at_utc": generated_at,
        "builder": "bars_dataset",
    }
    _write_store(output_path, obs_by_u, feature_rows, label_rows,
                 outcome_rows, meta)
    with open(manifest_path, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2, sort_keys=True)
    return manifest


def _parse_date(s: Optional[str]) -> Optional[datetime]:
    if not s:
        return None
    dt = datetime.fromisoformat(s)
    return dt.replace(tzinfo=UTC) if dt.tzinfo is None else dt.astimezone(UTC)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="python -m oracle.ml.bars_dataset",
        description=("Build a ledger-schema training dataset from BarStore "
                     "underlying history (research-only, read-only)."))
    ap.add_argument("--bars-db", required=True)
    ap.add_argument("--symbols", nargs="+", required=True)
    ap.add_argument("--output", required=True,
                    help="dataset sqlite output path")
    ap.add_argument("--manifest", required=True,
                    help="manifest json output path")
    ap.add_argument("--resolution", type=int, default=DEFAULT_RESOLUTION_S)
    ap.add_argument("--from", dest="start", default=None)
    ap.add_argument("--to", dest="end", default=None)
    ap.add_argument("--sample-every-min", type=int,
                    default=DEFAULT_SAMPLE_EVERY_MIN)
    ap.add_argument("--warmup-min", type=int, default=DEFAULT_WARMUP_MIN)
    ap.add_argument("--end-offset-min", type=int,
                    default=DEFAULT_END_OFFSET_MIN)
    args = ap.parse_args(argv)

    try:
        manifest = build_bars_dataset(
            args.bars_db,
            args.output,
            args.manifest,
            args.symbols,
            resolution_s=args.resolution,
            start=_parse_date(args.start),
            end=_parse_date(args.end),
            sample_every_min=args.sample_every_min,
            warmup_min=args.warmup_min,
            end_offset_min=args.end_offset_min,
        )
    except BarsUnavailable as exc:
        print(f"ERROR: bars unavailable: {exc}")
        return 2
    counts = manifest["counts"]
    print(f"dataset={args.output}")
    print(f"manifest={args.manifest}")
    for k in ("underlyings", "underlying_obs", "decisions", "feature_rows",
              "label_rows", "outcome_rows", "sessions"):
        print(f"{k}={counts[k]}")
    print(f"dataset_hash={manifest['dataset_hash']}")
    skipped = manifest["symbols"]["skipped"]
    if skipped:
        print(f"skipped_symbols={sorted(skipped)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
