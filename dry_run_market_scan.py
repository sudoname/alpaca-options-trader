"""Dry-run scanner that turns recent option-scan data into pass/block decisions.

This script reads the latest option scan JSON from the repo, converts each option
candidate into the same shape expected by the profitability validator, and prints
an automatic pass/block summary.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from profitability_validator import dry_run_scan


def latest_scan_file(scan_dir: str | Path | None = None) -> Optional[Path]:
    """Return the most recently modified option scan file in the directory."""
    scan_root = Path(scan_dir) if scan_dir else Path.cwd()
    files = sorted(
        [p for p in scan_root.glob("option_scan_*.json") if p.is_file()],
        key=lambda p: p.stat().st_mtime,
    )
    return files[-1] if files else None


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def build_candidate_from_option(option: Dict[str, Any]) -> Dict[str, Any]:
    """Translate a raw scan row into the profitability validator input shape."""
    option_type = str(option.get("type", "")).strip().lower()
    if option_type in {"call", "c", "buy_call", "long_call"}:
        side = "call"
    elif option_type in {"put", "p", "buy_put", "long_put"}:
        side = "put"
    else:
        side = "call"

    strike = _safe_float(option.get("strike"), 0.0)
    underlying = _safe_float(option.get("underlying_price"), 0.0)
    score = _safe_float(option.get("score"), 0.0)
    delta = _safe_float(option.get("delta"), 0.0)
    iv = _safe_float(option.get("iv"), 0.0)
    days = max(int(_safe_float(option.get("days_to_exp"), 0.0)), 1)
    last = _safe_float(option.get("last"), 0.0)
    bid = _safe_float(option.get("bid"), last)
    ask = _safe_float(option.get("ask"), last)
    mid = (bid + ask) / 2 if bid or ask else last

    pop = 0.5 + min(max((score - 50.0) / 100.0, 0.0), 0.4)
    if abs(delta) >= 0.5:
        pop += 0.05
    if iv > 0:
        pop += min(iv / 500.0, 0.04)
    pop = max(0.42, min(pop, 0.82))

    expected_value = ((pop - 0.5) * 10.0) + (score / 100.0) * 1.5
    if delta < 0 and side == "call":
        expected_value -= 0.5
    if delta > 0 and side == "put":
        expected_value -= 0.5
    if mid <= 0:
        ev_per_dollar_risk = 0.0
    else:
        risk = max(mid * 0.5, 0.05)
        ev_per_dollar_risk = expected_value / risk

    cand = {
        "symbol": str(option.get("ticker") or option.get("symbol") or "UNKNOWN").upper(),
        "direction": side,
        "side": side,
        "intended_side": side,
        "option": {
            "symbol": str(option.get("symbol") or "UNKNOWN"),
            "type": side,
            "confidence": int(max(1, min(10, round(score / 10.0)))),
            "score": score,
            "delta": delta,
            "iv": iv,
            "days_to_exp": days,
            "strike": strike,
            "underlying_price": underlying,
        },
        "entry_stamp": {
            "probability_of_profit": round(pop, 4),
            "expected_value": round(expected_value, 4),
            "ev_per_dollar_risk": round(ev_per_dollar_risk, 6),
        },
        "oracle_probability": {
            "p_call": 0.55 if side == "call" else 0.20,
            "p_put": 0.55 if side == "put" else 0.20,
            "p_no_trade": 0.18,
        },
        "robinhood_book": {"orderbook_imbalance": 0.12 if score > 80 else 0.04},
    }
    return cand


def load_recent_candidates(scan_dir: str | Path | None = None) -> List[Dict[str, Any]]:
    scan_file = latest_scan_file(scan_dir)
    if not scan_file:
        return []
    try:
        with scan_file.open("r", encoding="utf-8") as handle:
            raw = json.load(handle)
    except Exception:
        return []

    if isinstance(raw, dict):
        raw = raw.get("options", raw.get("data", []))
    if not isinstance(raw, list):
        return []

    out: List[Dict[str, Any]] = []
    for option in raw:
        if isinstance(option, dict):
            out.append(build_candidate_from_option(option))
    return out


def run_scan(scan_dir: str | Path | None = None):
    candidates = load_recent_candidates(scan_dir)
    result = dry_run_scan(candidates)
    approved = result["approved"]
    blocked = result["blocked"]
    total = result["total"]
    print("\n=== Profitability dry-run scan ===")
    print(f"source: {latest_scan_file(scan_dir) if latest_scan_file(scan_dir) else 'none'}")
    print(f"candidates: {total} | pass: {approved} | block: {blocked}")
    if not candidates:
        print("No recent option-scan data found. Add option_scan_*.json files or run the market scanner first.")
        return result

    for idx, item in enumerate(result["results"][:10], 1):
        summary = item.get("summary", {})
        print(
            f"[{idx}] {item.get('candidate', {}).get('symbol', 'UNK')} -> "
            f"{'PASS' if item['pass'] else 'BLOCK'} "
            f"POP={summary.get('probability_of_profit')} "
            f"EV={summary.get('expected_value')} "
            f"EV/$risk={summary.get('ev_per_dollar_risk')} "
            f"Oracle={summary.get('oracle_agreement')} "
            f"NoTrade={summary.get('p_no_trade')}"
        )
        if item.get("reasons"):
            print("      reasons: " + "; ".join(item["reasons"]))

    if blocked and not approved:
        print("No trades currently clear the profitability gate.")
    elif approved and blocked:
        print("Partial clearance: only the strongest candidates pass the gate.")
    elif approved:
        print("All recent candidates pass the gate.")

    return result


if __name__ == "__main__":
    run_scan()
