"""
Oracle 3.0 — Upgrade 5: Semantic trade memory (ANALYTICS ONLY).

An append-only JSONL ledger of trade *postmortems*: what the thesis expected,
what actually happened, the failure mode, and the one-line lesson. It exists so
a later decision can RETRIEVE context-only lessons ("we've been burned fading
this regime before") — it NEVER overrides a rule, sizes/prices/blocks a trade,
reads creds, or hits the network. Every public entry point is fail-open.

    record_reflection(reflection, *, path, now) -> Optional[dict]
        Append one reflection row (frozen at close). Returns the stored record
        (with id + recorded_at) or None on failure.

    retrieve_lessons(filters, *, path, limit) -> List[dict]
        Read the ledger back, newest-first, filtered by any subset of
        {symbol, sector, regime, catalyst, strategy_mode, failure_mode}.
        Context lookup only — the caller may show these to a human/LLM but the
        quant + risk gates remain authoritative.

Idiom mirrors oracle_prob_recorder.py (``_now`` / ``_jsonl_path`` /
``_append_jsonl``) and candidate_resolution's append-only / fold-by-id design.
"""

import json
import os
import uuid
from datetime import datetime, timezone
from typing import Dict, List, Optional

LOG_TAG = "[TRADE_MEMORY]"

JSONL_FILE_DEFAULT = "oracle_trade_memory.jsonl"
ENV_KEY = "ORACLE_TRADE_MEMORY_JSONL"

RECORD_TYPE = "reflection"

# The reflection fields we recognise (all optional, all nullable). Anything else
# supplied is preserved verbatim under the row too.
_KNOWN_FIELDS = (
    "expected", "actual", "failure_mode", "lesson", "confidence",
    "symbol", "sector", "regime", "catalyst", "strategy_mode",
)

# Filters that ``retrieve_lessons`` matches (case-insensitive exact match).
_FILTER_FIELDS = (
    "symbol", "sector", "regime", "catalyst", "strategy_mode", "failure_mode",
)


# --------------------------------------------------------------------------- #
# Helpers (mirror oracle_prob_recorder.py)
# --------------------------------------------------------------------------- #
def _now(now: Optional[datetime]) -> datetime:
    if isinstance(now, datetime):
        return now if now.tzinfo else now.replace(tzinfo=timezone.utc)
    return datetime.now(timezone.utc)


def _to_float(v) -> Optional[float]:
    try:
        if v is None or isinstance(v, bool):
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
        return ConfigLoader(path=".env").get_str(ENV_KEY, JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


def _norm(value) -> Optional[str]:
    if value is None:
        return None
    return str(value).strip().lower()


# --------------------------------------------------------------------------- #
# record_reflection
# --------------------------------------------------------------------------- #
def record_reflection(reflection: Optional[dict], *,
                      path: Optional[str] = None,
                      now: Optional[datetime] = None) -> Optional[dict]:
    """Append one postmortem row. Returns the stored record, or None on error."""
    try:
        if not isinstance(reflection, dict):
            return None
        ts = _now(now)
        rec: Dict = {
            "type": RECORD_TYPE,
            "id": str(uuid.uuid4()),
            "recorded_at": ts.isoformat(),
        }
        # Known fields first (nullable), then any extras verbatim.
        for f in _KNOWN_FIELDS:
            if f == "confidence":
                rec[f] = _to_float(reflection.get(f))
            else:
                rec[f] = reflection.get(f)
        for k, v in reflection.items():
            if k not in rec:
                rec[k] = v
        _append_jsonl(rec, _jsonl_path(path))
        return rec
    except Exception:  # pragma: no cover - fail-open
        return None


# --------------------------------------------------------------------------- #
# retrieve_lessons
# --------------------------------------------------------------------------- #
def retrieve_lessons(filters: Optional[dict] = None, *,
                     path: Optional[str] = None,
                     limit: Optional[int] = None) -> List[dict]:
    """Return matching reflections, newest-first. Context-only; never raises."""
    try:
        p = _jsonl_path(path)
        if not os.path.exists(p):
            return []
        wanted = {}
        if isinstance(filters, dict):
            for f in _FILTER_FIELDS:
                if f in filters and filters[f] is not None:
                    wanted[f] = _norm(filters[f])

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
                if not isinstance(rec, dict):
                    continue
                if rec.get("type") != RECORD_TYPE:
                    continue
                if all(_norm(rec.get(k)) == v for k, v in wanted.items()):
                    rows.append(rec)

        rows.sort(key=lambda r: str(r.get("recorded_at", "")), reverse=True)
        if isinstance(limit, int) and limit >= 0:
            rows = rows[:limit]
        return rows
    except Exception:  # pragma: no cover - fail-open
        return []


# --------------------------------------------------------------------------- #
# Self-test (writes only to a throwaway temp path; no network, no creds)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    tmp = tempfile.mkdtemp(prefix="trade_memory_")
    path = os.path.join(tmp, "mem.jsonl")

    try:
        # Empty ledger -> [].
        if retrieve_lessons({}, path=path) != []:
            print("FAIL: empty ledger should be []"); ok = False

        base = datetime(2024, 1, 1, tzinfo=timezone.utc)
        r1 = record_reflection({
            "symbol": "AAPL", "sector": "tech", "regime": "uptrend",
            "catalyst": "earnings", "strategy_mode": "intraday",
            "expected": "gap-and-go continuation", "actual": "faded into close",
            "failure_mode": "chased_extension", "lesson": "wait for reclaim",
            "confidence": 0.8,
        }, path=path, now=base)
        if not r1 or r1.get("id") is None:
            print("FAIL: record_reflection returned no record", r1); ok = False
        if r1 and r1.get("confidence") != 0.8:
            print("FAIL: confidence not floated", r1); ok = False

        r2 = record_reflection({
            "symbol": "MSFT", "sector": "tech", "regime": "downtrend",
            "catalyst": "none", "strategy_mode": "swing",
            "failure_mode": "wrong_regime", "lesson": "respect the trend",
        }, path=path, now=datetime(2024, 1, 2, tzinfo=timezone.utc))
        if not r2:
            print("FAIL: second record failed"); ok = False

        # Retrieve all -> newest first (MSFT before AAPL).
        allrows = retrieve_lessons({}, path=path)
        if len(allrows) != 2:
            print("FAIL: expected 2 rows", len(allrows)); ok = False
        if allrows and allrows[0].get("symbol") != "MSFT":
            print("FAIL: not newest-first", allrows); ok = False

        # Filter by symbol (case-insensitive).
        aapl = retrieve_lessons({"symbol": "aapl"}, path=path)
        if len(aapl) != 1 or aapl[0].get("symbol") != "AAPL":
            print("FAIL: symbol filter", aapl); ok = False

        # Filter by sector -> both.
        if len(retrieve_lessons({"sector": "tech"}, path=path)) != 2:
            print("FAIL: sector filter should match both"); ok = False

        # Filter by failure_mode.
        fm = retrieve_lessons({"failure_mode": "wrong_regime"}, path=path)
        if len(fm) != 1 or fm[0].get("symbol") != "MSFT":
            print("FAIL: failure_mode filter", fm); ok = False

        # Compound filter with no match -> [].
        if retrieve_lessons({"symbol": "AAPL", "regime": "downtrend"},
                            path=path) != []:
            print("FAIL: compound no-match should be []"); ok = False

        # limit.
        if len(retrieve_lessons({}, path=path, limit=1)) != 1:
            print("FAIL: limit=1"); ok = False

        # Extra fields are preserved verbatim.
        r3 = record_reflection({"symbol": "NVDA", "note_extra": "custom"},
                               path=path,
                               now=datetime(2024, 1, 3, tzinfo=timezone.utc))
        if not r3 or r3.get("note_extra") != "custom":
            print("FAIL: extra field not preserved", r3); ok = False

        # Fail-open: junk reflection -> None, junk filters never raise.
        if record_reflection(None, path=path) is not None:
            print("FAIL: junk reflection should be None"); ok = False
        if record_reflection(42, path=path) is not None:      # type: ignore[arg-type]
            print("FAIL: junk int should be None"); ok = False
        for junk in (None, 42, "x", []):
            try:
                retrieve_lessons(junk, path=path)             # type: ignore[arg-type]
            except Exception as exc:  # pragma: no cover
                print("FAIL: retrieve raised on junk", junk, exc); ok = False

    finally:
        try:
            if os.path.exists(path):
                os.remove(path)
            os.rmdir(tmp)
        except Exception:
            pass

    print("oracle.trade_memory self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
