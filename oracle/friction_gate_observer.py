"""
Oracle 3.0 — entry FRICTION gate: live SHADOW observer (analytics only).

The daily report shows the live bot bleeding on the ENTRY side: ~11% round-trip
friction on 0.76-day holds (avg spread 7.56%, slippage 373 bps, only 5.74% of
fills at/inside mid). The existing ``executable_ev`` model reasons about at-quote
cost; it does NOT see the realized adverse selection that this friction actually
costs. This observer closes that gap by measuring, per underlying, the round-trip
friction floor from CLOSED episodes (episodes.db) and asking, at each entry
decision, whether the candidate's expected edge clears that MEASURED floor.

    friction floor (per name) = avg_spread_pct + 2 * (avg_slippage_bps / 100)
                                (entry spread crossed once, slippage both legs)

    verdict = would_block  when expected_edge_pct < floor
              would_pass   when expected_edge_pct >= floor

STRICTLY analytics: this module records beliefs only. It NEVER opens, closes,
sizes, blocks or alters any trade — ``observe_entry`` returns the record it wrote
(or None) and NOTHING acts on that return. Every public entry point is fail-open
(returns None / [] / does nothing on error) and never raises. It is wired behind
``ENABLE_FRICTION_GATE_SHADOW`` (default OFF) so the live path is byte-identical
when disabled.
"""

import json
import os
import re
from datetime import datetime, timezone
from typing import Dict, List, Optional

LOG_TAG = "[FRICTION_GATE_SHADOW]"

JSONL_FILE_DEFAULT = "friction_gate_shadow.jsonl"
RECORD_CHECK = "friction_check"
VERDICT_BLOCK = "would_block"
VERDICT_PASS = "would_pass"

# OCC option symbol tail: 6-digit expiry, C/P, 8-digit strike -> root is the head.
_OCC_TAIL = re.compile(r"\d{6}[CP]\d{8}$")


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
            "FRICTION_GATE_SHADOW_JSONL", JSONL_FILE_DEFAULT)
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


def _underlying(row: dict) -> Optional[str]:
    """Best-effort underlying ticker: explicit field, else OCC root, else symbol."""
    try:
        for k in ("underlying_symbol", "underlying", "ticker"):
            v = row.get(k)
            if v:
                return str(v).upper()
        sym = str(row.get("symbol") or "")
        m = _OCC_TAIL.search(sym)
        if m:
            return sym[:m.start()].upper() or None
        return sym.upper() or None
    except Exception:
        return None


def _stable_key(candidate: dict, now: datetime) -> Optional[str]:
    """Stable per-decision key so repeated polls fold to one check.

    ``decision_id`` when present; else symbol + the candidate's ``as_of`` (or, as
    a last resort, today's date so the same name folds within a session).
    """
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


# --------------------------------------------------------------------------- #
# Measured per-name friction
# --------------------------------------------------------------------------- #
def per_name_friction(episodes) -> Dict[str, dict]:
    """Group CLOSED episodes by underlying and measure entry friction per name.

    Returns ``{UNDERLYING: {avg_spread_pct, avg_slippage_bps, samples}}`` reusing
    ``execution_analytics.entry_quality`` per group. Pure over injected rows;
    fail-open -> {}.
    """
    try:
        import execution_analytics as ea
    except Exception:
        return {}
    try:
        groups: Dict[str, List[dict]] = {}
        for e in episodes or []:
            if not isinstance(e, dict):
                continue
            name = _underlying(e)
            if not name:
                continue
            groups.setdefault(name, []).append(e)
        out: Dict[str, dict] = {}
        for name, rows in groups.items():
            eq = ea.entry_quality(rows)
            out[name] = {
                "avg_spread_pct": eq.get("avg_spread_pct"),
                "avg_slippage_bps": eq.get("avg_slippage_bps"),
                "samples": eq.get("samples", 0),
            }
        return out
    except Exception:
        return {}


def load_name_friction(db_path: str = "episodes.db") -> Dict[str, dict]:
    """Convenience loader: read closed episodes and measure per-name friction.

    Fail-open -> {} (missing store, no quotes, any error). Analytics-only.
    """
    try:
        import execution_analytics as ea
        return per_name_friction(ea.load_episodes(db_path))
    except Exception:
        return {}


def friction_floor_pct(name_friction: Optional[dict],
                       live_spread_pct=None) -> Optional[float]:
    """Round-trip friction floor (%): spread + 2 * slippage, from MEASURED name
    stats when available, else falling back to the live quoted spread.

    Fail-open -> None only when nothing at all is known.
    """
    try:
        nf = name_friction if isinstance(name_friction, dict) else None
        if nf and (nf.get("samples") or 0) > 0:
            spread = _to_float(nf.get("avg_spread_pct"))
            slip_bps = _to_float(nf.get("avg_slippage_bps"))
            base = spread if spread is not None else _to_float(live_spread_pct)
            if base is None:
                return None
            slip_pct = (abs(slip_bps) / 100.0) if slip_bps is not None else 0.0
            return base + 2.0 * slip_pct
        # No measured evidence for this name: fall back to the live spread.
        ls = _to_float(live_spread_pct)
        return ls if ls is not None else None
    except Exception:
        return None


# --------------------------------------------------------------------------- #
# Public API
# --------------------------------------------------------------------------- #
def observe_entry(candidate: dict, live_spread_pct, expected_edge_pct,
                  friction_lookup: Optional[dict] = None, *,
                  now: Optional[datetime] = None,
                  path: Optional[str] = None) -> Optional[dict]:
    """Record ONE would-block/would-pass check for an entry candidate.

    Pure side-effect (append-only JSONL). Compares ``expected_edge_pct`` to the
    MEASURED round-trip friction floor for the candidate's underlying (from
    ``friction_lookup`` = per_name_friction output; falls back to the live
    quoted spread when the name has no measured evidence). Returns the record it
    wrote, or None (nothing recorded / already logged / any error). Idempotent
    per stable key. NEVER vetoes, NEVER raises, NEVER alters ``candidate``.
    """
    try:
        if not isinstance(candidate, dict):
            return None
        n = _now(now)
        key = _stable_key(candidate, n)
        if not key:
            return None
        p = _jsonl_path(path)
        if key in _keys_of_type(p, RECORD_CHECK):
            return None  # idempotent: one check per decision
        name = _underlying(candidate)
        lookup = friction_lookup if isinstance(friction_lookup, dict) else {}
        nf = lookup.get(name) if name else None
        floor = friction_floor_pct(nf, live_spread_pct)
        edge = _to_float(expected_edge_pct)
        verdict = None
        if edge is not None and floor is not None:
            verdict = VERDICT_BLOCK if edge < floor else VERDICT_PASS
        rec = {
            "type": RECORD_CHECK,
            "key": key,
            "recorded_at": n.isoformat(),
            "symbol": candidate.get("symbol"),
            "underlying": name,
            "expected_edge_pct": edge,
            "live_spread_pct": _to_float(live_spread_pct),
            "measured_spread_pct": _to_float(nf.get("avg_spread_pct")) if nf else None,
            "measured_slippage_bps": _to_float(nf.get("avg_slippage_bps")) if nf else None,
            "friction_samples": (nf.get("samples") if nf else 0) or 0,
            "friction_floor_pct": floor,
            "verdict": verdict,
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
            if isinstance(r, dict) and r.get("type") == RECORD_CHECK]
    graded = [r for r in recs if r.get("verdict") in (VERDICT_BLOCK, VERDICT_PASS)]
    n_block = sum(1 for r in graded if r.get("verdict") == VERDICT_BLOCK)
    n_pass = sum(1 for r in graded if r.get("verdict") == VERDICT_PASS)
    edges = [e for e in (_to_float(r.get("expected_edge_pct")) for r in graded)
             if e is not None]
    floors = [f for f in (_to_float(r.get("friction_floor_pct")) for r in graded)
              if f is not None]
    return {
        "n_checks": len(recs),
        "n_graded": len(graded),
        "would_block": n_block,
        "would_pass": n_pass,
        "block_rate_pct": (n_block / len(graded) * 100.0) if graded else None,
        "mean_expected_edge_pct": (sum(edges) / len(edges)) if edges else None,
        "mean_friction_floor_pct": (sum(floors) / len(floors)) if floors else None,
    }


# --------------------------------------------------------------------------- #
# Offline self-test (no creds / no network; tmp files only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    try:
        now = datetime(2026, 1, 5, tzinfo=timezone.utc)

        # 1) per_name_friction groups by underlying and reuses entry_quality.
        #    AAPL: bid 1.00/ask 1.20 mid 1.10 fill 1.15; bid 2.00/ask 2.10 mid
        #    2.05 fill 2.05.  TSLA: one quoted row.  A quote-less row is dropped.
        eps = [
            {"underlying_symbol": "AAPL", "quote_bid": 1.00, "quote_ask": 1.20,
             "fill_price": 1.15},
            {"underlying_symbol": "AAPL", "quote_bid": 2.00, "quote_ask": 2.10,
             "fill_price": 2.05},
            {"symbol": "TSLA260116C00300000", "quote_bid": 5.00,
             "quote_ask": 5.50, "fill_price": 5.40},
            {"underlying_symbol": "AAPL", "fill_price": 9.9},  # no quotes -> skip
        ]
        nf = per_name_friction(eps)
        ok &= "AAPL" in nf and "TSLA" in nf
        ok &= nf["AAPL"]["samples"] == 2
        ok &= nf["TSLA"]["samples"] == 1

        # 2) friction floor math: spread + 2*(slippage_bps/100).
        floor = friction_floor_pct({"avg_spread_pct": 7.0,
                                    "avg_slippage_bps": 300.0, "samples": 5})
        ok &= floor is not None and abs(floor - (7.0 + 2.0 * 3.0)) < 1e-9  # 13.0
        # No samples -> fall back to the live quoted spread.
        ok &= friction_floor_pct({"samples": 0}, live_spread_pct=4.0) == 4.0
        # Nothing known -> None.
        ok &= friction_floor_pct(None, None) is None

        lookup = {"XYZ": {"avg_spread_pct": 6.0, "avg_slippage_bps": 200.0,
                          "samples": 10}}  # floor = 6 + 2*2 = 10.0

        with tempfile.TemporaryDirectory() as d:
            p = os.path.join(d, "friction.jsonl")

            # 3) edge below floor -> would_block; repeat poll is a no-op.
            r1 = observe_entry({"symbol": "XYZ260116C00100000",
                                "underlying_symbol": "XYZ", "decision_id": "A1"},
                               live_spread_pct=6.0, expected_edge_pct=5.0,
                               friction_lookup=lookup, now=now, path=p)
            r2 = observe_entry({"symbol": "XYZ260116C00100000",
                                "underlying_symbol": "XYZ", "decision_id": "A1"},
                               live_spread_pct=6.0, expected_edge_pct=5.0,
                               friction_lookup=lookup, now=now, path=p)
            ok &= (r1 is not None) and (r1["verdict"] == VERDICT_BLOCK)
            ok &= abs(r1["friction_floor_pct"] - 10.0) < 1e-9
            ok &= (r2 is None)  # idempotent

            # 4) edge above floor -> would_pass (distinct decision id).
            r3 = observe_entry({"symbol": "XYZ260116C00100000",
                                "underlying_symbol": "XYZ", "decision_id": "A2"},
                               live_spread_pct=6.0, expected_edge_pct=20.0,
                               friction_lookup=lookup, now=now, path=p)
            ok &= (r3 is not None) and (r3["verdict"] == VERDICT_PASS)

            s = summarize(p)
            ok &= s["n_checks"] == 2 and s["n_graded"] == 2
            ok &= s["would_block"] == 1 and s["would_pass"] == 1
            ok &= abs(s["block_rate_pct"] - 50.0) < 1e-9

        # 5) junk / missing inputs never raise.
        ok &= observe_entry(None, 1.0, 1.0) is None
        ok &= observe_entry({}, None, None) is None
        ok &= per_name_friction(None) == {}
        ok &= isinstance(summarize(os.path.join(tempfile.gettempdir(),
                                                "nope_friction.jsonl")), dict)
    except Exception as exc:  # any raise is a hard fail
        print(f"friction_gate_observer self-test: FAIL ({exc!r})")
        return 1

    print("friction_gate_observer self-test: " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
