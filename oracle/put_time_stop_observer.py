"""
Oracle 3.0 — 5-day put max-hold stop: live SHADOW observer (analytics only).

The realized-episode backtest (``oracle/lab/episodes_study.py --time-stop put``)
found a ~5-day put max-hold stop to be the sweet spot, but that estimate rode a
LINEAR-ACCRUAL assumption: episodes.db carries only the entry (``fill_price``)
and exit (``exit_price``) endpoints -- there is no mid-hold mark, so the day-5
counterfactual had to be *assumed*, not measured.

This module closes that gap going forward. It is a pure observer: as the live
monitor polls open positions, it records -- for each PUT that crosses the N-day
boundary -- the ACTUAL option mark at that boundary (the datum the backfill was
missing). Later, ``resolve_boundaries`` joins each boundary to the trade's real
close so ``shadow_delta_pct`` becomes a MEASURED answer to "would a 5-day put
stop have helped?".

STRICTLY analytics: this module records and resolves beliefs. It NEVER opens,
closes, sizes, blocks or alters any trade, and every public entry point is
fail-open (returns None / does nothing on error). It is wired behind
``ENABLE_PUT_TIME_STOP_SHADOW`` (default OFF) so the live path is byte-identical
when disabled.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

LOG_TAG = "[PUT_TIME_STOP_SHADOW]"

JSONL_FILE_DEFAULT = "put_time_stop_shadow.jsonl"
RECORD_BOUNDARY = "boundary"
RECORD_RESOLUTION = "resolution"
DEFAULT_CAP_DAYS = 5

# OCC option symbol tail: 6-digit expiry, C/P, 8-digit strike (e.g. ...250117P00150000)
_OCC_TAIL = re.compile(r"(\d{6})([CP])(\d{8})$")


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


def _parse_ts(v) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=timezone.utc)
    if not v:
        return None
    try:
        dt = datetime.fromisoformat(str(v).replace("Z", "+00:00"))
        return dt if dt.tzinfo else dt.replace(tzinfo=timezone.utc)
    except Exception:
        return None


def _jsonl_path(path: Optional[str] = None) -> str:
    """Resolve the ledger path (arg > env > default). Fail-open."""
    if path:
        return path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(
            "PUT_TIME_STOP_SHADOW_JSONL", JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _hold_days(entry_time, now: datetime) -> Optional[int]:
    """Whole calendar days held, date-only diff (matches held_past_max_days)."""
    et = _parse_ts(entry_time)
    if et is None:
        return None
    try:
        return (now.date() - et.date()).days
    except Exception:
        return None


def is_put(trade: dict) -> bool:
    """True iff this trade is a PUT. direction/option_type first, then OCC tail."""
    try:
        d = str(trade.get("direction") or "").strip().lower()
        if d in ("put", "down"):
            return True
        if d in ("call", "up"):
            return False
        ot = str(trade.get("option_type") or trade.get("type") or "").strip().lower()
        if ot.startswith("p"):
            return True
        if ot.startswith("c"):
            return False
        m = _OCC_TAIL.search(str(trade.get("symbol") or ""))
        if m:
            return m.group(2) == "P"
    except Exception:
        pass
    return False


def _stable_key(trade: dict) -> str:
    """A stable per-position key so repeated polls fold to one boundary."""
    did = trade.get("decision_id")
    if did:
        return f"did:{did}"
    return f"{trade.get('symbol')}|{trade.get('entry_time')}"


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


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def observe_position(trade: dict, mark_pnl_pct, current_mark, *,
                     cap_days: int = DEFAULT_CAP_DAYS,
                     now: Optional[datetime] = None,
                     path: Optional[str] = None) -> Optional[dict]:
    """Record ONE boundary line the first time a PUT crosses the N-day cap.

    Pure side-effect (append-only JSONL). Returns the record it wrote, or None
    when nothing was recorded (not a put / not yet at the cap / already logged /
    any error). NEVER raises, NEVER alters ``trade``.
    """
    try:
        if not isinstance(trade, dict) or not is_put(trade):
            return None
        n = _now(now)
        hd = _hold_days(trade.get("entry_time"), n)
        if hd is None or hd < int(cap_days):
            return None
        p = _jsonl_path(path)
        key = _stable_key(trade)
        if key in _keys_of_type(p, RECORD_BOUNDARY):
            return None  # idempotent: one boundary per position
        rec = {
            "type": RECORD_BOUNDARY,
            "key": key,
            "recorded_at": n.isoformat(),
            "symbol": trade.get("symbol"),
            "underlying": trade.get("underlying_symbol"),
            "entry_time": trade.get("entry_time"),
            "entry_price": _to_float(trade.get("entry_price")),
            "cap_days": int(cap_days),
            "hold_days_at_boundary": hd,
            "boundary_mark": _to_float(current_mark),
            "boundary_pnl_pct": _to_float(mark_pnl_pct),
        }
        did = trade.get("decision_id")
        if did:
            rec["decision_id"] = did
        _append_jsonl(rec, p)
        return rec
    except Exception:
        return None


def resolve_boundaries(closed_lookup: Dict[str, dict], *,
                       path: Optional[str] = None,
                       now: Optional[datetime] = None) -> List[dict]:
    """Join unresolved boundaries to their realized close; append resolutions.

    ``closed_lookup`` maps a boundary ``key`` (see ``_stable_key``) to a dict with
    ``actual_exit_pnl_pct`` and optional ``hold_days``. ``shadow_delta_pct`` =
    boundary_pnl_pct - actual_exit_pnl_pct (positive => the cap would have helped).
    Fold-by-key: each boundary resolves at most once. Fail-open -> [].
    """
    p = _jsonl_path(path)
    try:
        recs = load_records(p)
        boundaries = [r for r in recs
                      if isinstance(r, dict) and r.get("type") == RECORD_BOUNDARY]
        resolved = {r.get("key") for r in recs
                    if isinstance(r, dict) and r.get("type") == RECORD_RESOLUTION}
        cl = closed_lookup if isinstance(closed_lookup, dict) else {}
        n = _now(now)
        out: List[dict] = []
        for b in boundaries:
            key = b.get("key")
            if not key or key in resolved:
                continue
            info = cl.get(key)
            if not isinstance(info, dict):
                continue
            actual = _to_float(info.get("actual_exit_pnl_pct"))
            if actual is None:
                continue
            bpnl = _to_float(b.get("boundary_pnl_pct"))
            held = _to_float(info.get("hold_days"))
            cap = b.get("cap_days")
            held_extra = (held - cap) if (held is not None and cap is not None) else None
            delta = (bpnl - actual) if (bpnl is not None and actual is not None) else None
            rec = {
                "type": RECORD_RESOLUTION,
                "key": key,
                "recorded_at": n.isoformat(),
                "symbol": b.get("symbol"),
                "boundary_pnl_pct": bpnl,
                "actual_exit_pnl_pct": actual,
                "held_extra_days": held_extra,
                "shadow_delta_pct": delta,
            }
            _append_jsonl(rec, p)
            resolved.add(key)
            out.append(rec)
        return out
    except Exception:
        return []


def summarize(path: Optional[str] = None) -> dict:
    """Aggregate the ledger. Pure; fail-open."""
    recs = load_records(path)
    b = [r for r in recs if isinstance(r, dict) and r.get("type") == RECORD_BOUNDARY]
    res = [r for r in recs if isinstance(r, dict) and r.get("type") == RECORD_RESOLUTION]
    deltas = [d for d in (_to_float(r.get("shadow_delta_pct")) for r in res)
              if d is not None]
    helped = sum(1 for d in deltas if d > 0)
    hurt = sum(1 for d in deltas if d < 0)
    mean = (sum(deltas) / len(deltas)) if deltas else None
    med = None
    if deltas:
        s = sorted(deltas)
        k = len(s)
        med = s[k // 2] if k % 2 else (s[k // 2 - 1] + s[k // 2]) / 2.0
    return {
        "n_boundaries": len(b),
        "n_resolved": len(res),
        "mean_shadow_delta_pct": mean,
        "median_shadow_delta_pct": med,
        "count_helped": helped,
        "count_hurt": hurt,
    }


# --------------------------------------------------------------------------- #
# Offline self-test (no creds / no network; tmp files only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    try:
        now = datetime(2024, 1, 10, tzinfo=timezone.utc)

        def _iso(days_ago):
            return (datetime(2024, 1, 10, tzinfo=timezone.utc)
                    .replace(day=10 - days_ago)).isoformat()

        put6 = {"symbol": "AAPL240119P00150000", "entry_time": _iso(6),
                "entry_price": 2.0, "underlying_symbol": "AAPL"}
        call6 = {"symbol": "AAPL240119C00150000", "entry_time": _iso(6),
                 "entry_price": 2.0}
        put3 = {"symbol": "AAPL240119P00150000", "entry_time": _iso(3),
                "entry_price": 2.0}

        # direction / OCC parsing
        ok &= is_put(put6) and not is_put(call6)
        ok &= is_put({"direction": "put"}) and not is_put({"direction": "call"})

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "shadow.jsonl")

            # 1) put past cap logs exactly one boundary; repeat poll is a no-op
            r1 = observe_position(put6, -20.0, 1.6, cap_days=5, now=now, path=p)
            r2 = observe_position(put6, -25.0, 1.5, cap_days=5, now=now, path=p)
            ok &= (r1 is not None) and (r2 is None)
            bnd = [x for x in load_records(p) if x.get("type") == RECORD_BOUNDARY]
            ok &= len(bnd) == 1
            ok &= bnd[0]["boundary_pnl_pct"] == -20.0

            # 2) a call is ignored; 3) a put short of the cap logs nothing
            ok &= observe_position(call6, -20.0, 1.6, cap_days=5, now=now, path=p) is None
            ok &= observe_position(put3, -20.0, 1.6, cap_days=5, now=now, path=p) is None
            bnd = [x for x in load_records(p) if x.get("type") == RECORD_BOUNDARY]
            ok &= len(bnd) == 1

            # 4) resolution math: boundary -20% vs actual close -35% => delta +15
            key = _stable_key(put6)
            new = resolve_boundaries(
                {key: {"actual_exit_pnl_pct": -35.0, "hold_days": 9}}, path=p, now=now)
            ok &= len(new) == 1
            ok &= abs(new[0]["shadow_delta_pct"] - 15.0) < 1e-9
            ok &= new[0]["held_extra_days"] == 4
            # resolving again is idempotent
            ok &= resolve_boundaries(
                {key: {"actual_exit_pnl_pct": -35.0}}, path=p, now=now) == []

            s = summarize(p)
            ok &= s["n_boundaries"] == 1 and s["n_resolved"] == 1
            ok &= s["count_helped"] == 1 and s["count_hurt"] == 0

        # 5) junk never raises
        ok &= observe_position(None, 0, 0) is None
        ok &= observe_position({}, None, None) is None
        ok &= resolve_boundaries(None) == []
        ok &= isinstance(summarize(os.path.join(tempfile.gettempdir(), "nope.jsonl")), dict)
    except Exception as exc:  # any raise is a hard fail
        print(f"put_time_stop_observer self-test: FAIL ({exc!r})")
        return 1

    print("put_time_stop_observer self-test: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
