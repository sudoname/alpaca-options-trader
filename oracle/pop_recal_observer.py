"""
Oracle 3.0 — scorer PoP RECALIBRATION: live SHADOW observer (analytics only).

The daily report's PoP calibration verdict is OVERCONFIDENT by ~26pp: buckets the
model stamps at (say) 80-90% PoP actually win far less often. ``pop_calibration``
already MEASURES that curve from closed trades. This observer applies it forward:
at each scoring decision it maps the raw stamped PoP to its calibration bucket and
records the bucket's MEASURED actual win-rate as the ``corrected_pop`` the scorer
*would* have used.

    corrected_pop = actual_win_rate of the bucket containing raw_pop
                    (only when that bucket is thick enough; else raw_pop)

STRICTLY analytics: this module records beliefs only. It NEVER opens, closes,
sizes, blocks or alters any trade, and NEVER feeds ``corrected_pop`` back into the
live scorer — ``observe_scoring`` returns the record it wrote (or None) and
NOTHING acts on that return. Every public entry point is fail-open (returns None /
{} / does nothing on error) and never raises. It is wired behind
``ENABLE_POP_RECAL_SHADOW`` (default OFF) so the live path is byte-identical when
disabled.
"""

import json
import os
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

LOG_TAG = "[POP_RECAL_SHADOW]"

JSONL_FILE_DEFAULT = "pop_recal_shadow.jsonl"
RECORD_RECAL = "pop_recal"

# Fallback minimum-trades threshold if pop_calibration cannot be imported.
_MIN_TRADES_FALLBACK = 10


# --------------------------------------------------------------------------- #
# Helpers (pure, fail-open)
# --------------------------------------------------------------------------- #
def _now(now: Optional[datetime]) -> datetime:
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _to_float(v) -> Optional[float]:
    try:
        if v is None:
            return None
        return float(v)
    except (TypeError, ValueError):
        return None


def _jsonl_path(path: Optional[str] = None) -> str:
    """Resolve the ledger path (arg > env > default). Fail-open."""
    if path:
        return path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(
            "POP_RECAL_SHADOW_JSONL", JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def load_records(path: Optional[str] = None) -> List[dict]:
    """Read every JSONL record. Fail-open -> []."""
    p = _jsonl_path(path)
    recs: List[dict] = []
    try:
        if not os.path.exists(p):
            return recs
        with open(p, encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    recs.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        pass
    return recs


def _keys_of_type(path: str, rec_type: str) -> set:
    keys = set()
    for r in load_records(path):
        if isinstance(r, dict) and r.get("type") == rec_type and r.get("key"):
            keys.add(r["key"])
    return keys


def _stable_key(candidate: dict, now: datetime) -> Optional[str]:
    """Stable per-decision key so repeated polls fold to one recal record."""
    try:
        did = candidate.get("decision_id")
        if did:
            return f"did:{did}"
        sym = candidate.get("symbol")
        if not sym:
            return None
        stamp = candidate.get("as_of") or now.date().isoformat()
        return f"{sym}|{stamp}"
    except Exception:
        return None


def _bucket_of(raw_pop: float) -> Optional[str]:
    """Calibration-bucket label for a raw PoP. Lazy import; fail-open -> None."""
    try:
        import ev_attribution as eva
        from pop_calibration import POP_CAL_BUCKETS
        return eva.bucket_label(raw_pop, POP_CAL_BUCKETS)
    except Exception:
        return None


def _min_trades() -> int:
    try:
        from pop_calibration import MIN_TRADES
        return int(MIN_TRADES)
    except Exception:
        return _MIN_TRADES_FALLBACK


# --------------------------------------------------------------------------- #
# Correction map + pure recalibration
# --------------------------------------------------------------------------- #
def bucket_correction_map(calibration_report: Optional[dict] = None) -> Dict[str, dict]:
    """Map each PoP bucket -> its MEASURED actual win-rate + usability.

    ``{label: {actual_win_rate, trades, usable}}``. ``usable`` is True only when
    the bucket carries >= MIN_TRADES resolved trades and a non-null win-rate.
    Pure over an injected report; fail-open -> {}.
    """
    try:
        buckets = (calibration_report or {}).get("buckets")
        if not isinstance(buckets, dict):
            return {}
        min_n = _min_trades()
        out: Dict[str, dict] = {}
        for label, blk in buckets.items():
            if not isinstance(blk, dict):
                continue
            trades = blk.get("trades") or 0
            awr = _to_float(blk.get("actual_win_rate"))
            out[label] = {
                "actual_win_rate": awr,
                "trades": trades,
                "usable": (trades >= min_n) and (awr is not None),
            }
        return out
    except Exception:
        return {}


def load_correction_map(config=None, attribution_path: Optional[str] = None
                        ) -> Dict[str, dict]:
    """Convenience loader: compute the calibration report and build the map.

    Touches the analytics readers (disk only); fail-open -> {}. Analytics-only.
    """
    try:
        import pop_calibration as pc
        report = pc.compute_pop_calibration(config=config,
                                            attribution_path=attribution_path)
        return bucket_correction_map(report)
    except Exception:
        return {}


def corrected_pop(raw_pop, correction_map: Optional[dict]
                  ) -> Tuple[Optional[float], dict]:
    """Return ``(corrected_pop, meta)`` for a raw stamped PoP. Pure; fail-open.

    ``corrected_pop`` is the bucket's measured win-rate when that bucket is
    usable, else the raw PoP unchanged. ``meta`` carries bucket / bucket_trades /
    applied so the record explains itself.
    """
    raw = _to_float(raw_pop)
    meta = {"bucket": None, "bucket_trades": None, "applied": False}
    if raw is None:
        return None, meta
    try:
        label = _bucket_of(raw)
        meta["bucket"] = label
        cm = correction_map if isinstance(correction_map, dict) else {}
        info = cm.get(label) if label else None
        if isinstance(info, dict):
            meta["bucket_trades"] = info.get("trades")
            if info.get("usable") and info.get("actual_win_rate") is not None:
                meta["applied"] = True
                return _to_float(info.get("actual_win_rate")), meta
        return raw, meta
    except Exception:
        return raw, meta


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def observe_scoring(candidate: dict, raw_pop, correction_map: Optional[dict] = None,
                    *, now: Optional[datetime] = None,
                    path: Optional[str] = None) -> Optional[dict]:
    """Record ONE recalibration line for a scoring candidate.

    Pure side-effect (append-only JSONL). Maps ``raw_pop`` through the measured
    bucket curve and records the ``corrected_pop`` the scorer WOULD have used.
    Returns the record it wrote, or None (nothing recorded / already logged / any
    error). Idempotent per stable key. NEVER feeds back, NEVER raises, NEVER
    alters ``candidate``.
    """
    try:
        if not isinstance(candidate, dict):
            return None
        raw = _to_float(raw_pop)
        if raw is None:
            return None
        n = _now(now)
        key = _stable_key(candidate, n)
        if not key:
            return None
        p = _jsonl_path(path)
        if key in _keys_of_type(p, RECORD_RECAL):
            return None  # idempotent: one recal per decision
        corr, meta = corrected_pop(raw, correction_map)
        delta_pp = ((corr - raw) * 100.0) if corr is not None else None
        rec = {
            "type": RECORD_RECAL,
            "key": key,
            "recorded_at": n.isoformat(),
            "symbol": candidate.get("symbol"),
            "raw_pop": raw,
            "bucket": meta.get("bucket"),
            "bucket_trades": meta.get("bucket_trades"),
            "corrected_pop": corr,
            "delta_pp": delta_pp,
            "applied": meta.get("applied", False),
        }
        did = candidate.get("decision_id")
        if did:
            rec["decision_id"] = did
        _append_jsonl(rec, p)
        return rec
    except Exception:
        return None


def summarize(path: Optional[str] = None) -> dict:
    """Aggregate the ledger. Pure; fail-open."""
    recs = [r for r in load_records(path)
            if isinstance(r, dict) and r.get("type") == RECORD_RECAL]
    applied = [r for r in recs if r.get("applied")]
    deltas = [d for d in (_to_float(r.get("delta_pp")) for r in applied)
              if d is not None]
    raws = [v for v in (_to_float(r.get("raw_pop")) for r in recs) if v is not None]
    corrs = [v for v in (_to_float(r.get("corrected_pop")) for r in recs)
             if v is not None]
    return {
        "n_records": len(recs),
        "n_applied": len(applied),
        "mean_delta_pp": (sum(deltas) / len(deltas)) if deltas else None,
        "mean_raw_pop": (sum(raws) / len(raws)) if raws else None,
        "mean_corrected_pop": (sum(corrs) / len(corrs)) if corrs else None,
    }


# --------------------------------------------------------------------------- #
# Offline self-test (no creds / no network; tmp files only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    try:
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)

        # 1) bucket_correction_map: usable only when trades >= MIN_TRADES.
        report = {"buckets": {
            "PoP 80-90%": {"trades": 40, "actual_win_rate": 0.60},
            "PoP 70-80%": {"trades": 3, "actual_win_rate": 0.55},  # too thin
            "PoP 90-100%": {"trades": 25, "actual_win_rate": None},  # no rate
        }}
        cm = bucket_correction_map(report)
        ok &= cm["PoP 80-90%"]["usable"] is True
        ok &= cm["PoP 70-80%"]["usable"] is False
        ok &= cm["PoP 90-100%"]["usable"] is False

        # 2) corrected_pop: raw 0.85 lands in 80-90% -> measured 0.60 (delta -25pp).
        corr, meta = corrected_pop(0.85, cm)
        ok &= corr is not None and abs(corr - 0.60) < 1e-9
        ok &= meta["applied"] is True and meta["bucket"] == "PoP 80-90%"
        # Thin bucket -> raw unchanged, not applied.
        corr2, meta2 = corrected_pop(0.75, cm)
        ok &= abs(corr2 - 0.75) < 1e-9 and meta2["applied"] is False
        # Junk raw -> None.
        ok &= corrected_pop(None, cm)[0] is None

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "recal.jsonl")

            # 3) applied correction recorded; repeat poll is a no-op.
            r1 = observe_scoring({"symbol": "AAPL", "decision_id": "S1"},
                                 0.85, cm, now=now, path=p)
            r2 = observe_scoring({"symbol": "AAPL", "decision_id": "S1"},
                                 0.85, cm, now=now, path=p)
            ok &= (r1 is not None) and (r1["applied"] is True)
            ok &= abs(r1["corrected_pop"] - 0.60) < 1e-9
            ok &= abs(r1["delta_pp"] - (-25.0)) < 1e-9
            ok &= (r2 is None)  # idempotent

            # 4) thin-bucket candidate recorded but not applied (distinct id).
            r3 = observe_scoring({"symbol": "TSLA", "decision_id": "S2"},
                                 0.75, cm, now=now, path=p)
            ok &= (r3 is not None) and (r3["applied"] is False)
            ok &= abs(r3["corrected_pop"] - 0.75) < 1e-9

            s = summarize(p)
            ok &= s["n_records"] == 2 and s["n_applied"] == 1
            ok &= abs(s["mean_delta_pp"] - (-25.0)) < 1e-9

        # 5) junk / missing inputs never raise.
        ok &= observe_scoring(None, 0.5, cm) is None
        ok &= observe_scoring({"symbol": "X"}, None, cm) is None
        ok &= bucket_correction_map(None) == {}
        ok &= isinstance(summarize(os.path.join(tempfile.gettempdir(),
                                                "nope_recal.jsonl")), dict)
    except Exception as exc:  # any raise is a hard fail
        print(f"pop_recal_observer self-test: FAIL ({exc!r})")
        return 1

    print("pop_recal_observer self-test: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
