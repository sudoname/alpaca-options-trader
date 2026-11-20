"""Strategy books for Oracle 4.0.1 Stage G (observe-only).

Separates candidates into per-strategy books so evidence is never mixed:

  0dte    expiration == the current ET session date; intraday-only horizon,
          actual remaining time to the 16:00 ET close, near-expiry liquidity
          annotations.
  swing   everything with a later expiration; bucketed by DTE so overnight /
          holiday gaps and realistic holding periods are kept distinct.

Also provides candidate *group* identity (paired call/put legs of the same
underlying+expiration are one strategy group, never two independent wins)
and a hedge-aware order investigation helper that distinguishes true
duplicate submissions from intentional call+put hedge pairs.

Everything here is classification and annotation only. Nothing in this
module is a trading rule, gate, or order pathway.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

from . import valuation as val

UTC = timezone.utc
ET = ZoneInfo("America/New_York")

BOOK_0DTE = "0dte"
BOOK_SWING = "swing"
BOOK_UNKNOWN = "unknown"

# DTE buckets for the swing book (calendar days to expiry, inclusive).
DTE_BUCKETS: Tuple[Tuple[int, int, str], ...] = (
    (1, 5, "1-5"),
    (6, 21, "6-21"),
    (22, 60, "22-60"),
)

# Observe-only annotation thresholds (never gates).
NEAR_EXPIRY_WIDE_SPREAD_FRAC = 0.10
QUOTE_FRESH_MAX_AGE_S = 120.0


def dte_bucket(dte_days: Optional[float]) -> Optional[str]:
    """Bucket label for the swing book; None when DTE is unknown."""
    if dte_days is None:
        return None
    try:
        d = float(dte_days)
    except (TypeError, ValueError):
        return None
    if d != d:  # NaN
        return None
    di = int(d) if d == int(d) else int(d) + (1 if d > 0 else 0)
    for lo, hi, label in DTE_BUCKETS:
        if lo <= di <= hi:
            return label
    if di > DTE_BUCKETS[-1][1]:
        return ">60"
    return "0"  # same-day; the 0dte book, kept for completeness


def _expiration_date(expiration: Optional[str]) -> Optional[datetime]:
    if not expiration:
        return None
    try:
        d = datetime.strptime(str(expiration)[:10], "%Y-%m-%d")
    except ValueError:
        return None
    return d


def remaining_expiry_minutes(expiration: Optional[str],
                             now: datetime) -> Optional[float]:
    """Actual minutes from ``now`` to 16:00 ET on the expiration date.

    Negative values mean the contract is already past the equity-option
    close; this module reports the truth and never clamps.
    """
    d = _expiration_date(expiration)
    if d is None:
        return None
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)
    close_et = datetime(d.year, d.month, d.day, 16, 0, tzinfo=ET)
    return (close_et - now).total_seconds() / 60.0


def classify_book(expiration: Optional[str], now: datetime,
                  dte_days: Optional[float] = None) -> str:
    """0dte / swing / unknown, from expiration date (preferred) or DTE."""
    d = _expiration_date(expiration)
    if d is not None:
        if now.tzinfo is None:
            now = now.replace(tzinfo=UTC)
        session = now.astimezone(ET).date()
        if d.date() == session:
            return BOOK_0DTE
        return BOOK_SWING if d.date() > session else BOOK_UNKNOWN
    if dte_days is not None:
        try:
            dd = float(dte_days)
        except (TypeError, ValueError):
            return BOOK_UNKNOWN
        if dd != dd:
            return BOOK_UNKNOWN
        return BOOK_0DTE if dd < 1.0 else BOOK_SWING
    return BOOK_UNKNOWN


def candidate_group_id(scan_id: str, underlying: Optional[str],
                       expiration: Optional[str]) -> str:
    """One strategy group = one underlying+expiration inside one scan."""
    return "|".join([str(scan_id), str(underlying or "UNKNOWN"),
                     str(expiration or "UNKNOWN")])


def strategy_id(book: str, underlying: Optional[str],
                option_type: Optional[str]) -> str:
    return ":".join([book, str(underlying or "UNKNOWN"),
                     str(option_type or "na")])


def _contract_of(cand: Any) -> Dict[str, Any]:
    contract = getattr(cand, "contract", None)
    if isinstance(cand, dict):
        contract = cand.get("contract") or contract
    if not isinstance(contract, dict):
        sym = (cand.get("symbol") if isinstance(cand, dict)
               else getattr(cand, "symbol", None))
        contract = val.parse_occ_symbol(sym) or {}
    return contract


def _features_of(cand: Any) -> Dict[str, Any]:
    feats = (cand.get("features") if isinstance(cand, dict)
             else getattr(cand, "features", None))
    return feats if isinstance(feats, dict) else {}


def _field(cand: Any, name: str) -> Any:
    if isinstance(cand, dict):
        return cand.get(name)
    return getattr(cand, name, None)


def assign_books(candidates: Sequence[Any], *, now: datetime, scan_id: str,
                 bars_mode: Optional[str] = None) -> Dict[str, Dict[str, Any]]:
    """Per-decision book assignment for frozen shadow candidates.

    Accepts ``valuation.Candidate`` objects or equivalent dicts. Returns
    ``{decision_id: {book, dte_bucket, group_id, group_role, paired,
    strategy_id, remaining_expiry_min, annotations}}``. Annotations are
    observe-only facts (data-quality notes for the 0dte book), never rules.
    """
    if now.tzinfo is None:
        now = now.replace(tzinfo=UTC)

    # First pass: raw classification + group keys.
    rows: List[Tuple[str, Dict[str, Any]]] = []
    group_types: Dict[str, set] = {}
    for cand in candidates:
        decision_id = str(_field(cand, "decision_id") or "")
        if not decision_id:
            continue
        contract = _contract_of(cand)
        feats = _features_of(cand)
        underlying = (contract.get("underlying")
                      or _field(cand, "underlying") or None)
        expiration = contract.get("expiration")
        option_type = contract.get("option_type")
        dte = feats.get("dte_days")
        book = classify_book(expiration, now, dte)
        gid = candidate_group_id(scan_id, underlying, expiration)
        group_types.setdefault(gid, set()).add(option_type)
        rem = remaining_expiry_minutes(expiration, now)
        info: Dict[str, Any] = {
            "book": book,
            "dte_bucket": (dte_bucket(dte) if book == BOOK_SWING else None),
            "group_id": gid,
            "strategy_id": strategy_id(book, underlying, option_type),
            "underlying": underlying,
            "expiration": expiration,
            "option_type": option_type,
            "remaining_expiry_min": rem,
            "annotations": [],
        }
        if book == BOOK_0DTE:
            notes = info["annotations"]
            if bars_mode is not None and bars_mode != "bars":
                notes.append("0dte_requires_minute_bars")
            if rem is not None and rem <= 0:
                notes.append("0dte_past_close")
            mid = feats.get("quote_mid")
            spread = feats.get("spread_abs")
            if (isinstance(mid, (int, float)) and mid > 0
                    and isinstance(spread, (int, float)) and spread >= 0
                    and spread / mid > NEAR_EXPIRY_WIDE_SPREAD_FRAC):
                notes.append("0dte_wide_spread_near_expiry")
            age = feats.get("quote_age_s")
            if isinstance(age, (int, float)) and age > QUOTE_FRESH_MAX_AGE_S:
                notes.append("0dte_stale_quote")
        rows.append((decision_id, info))

    # Second pass: paired call/put legs inside a group are one strategy.
    out: Dict[str, Dict[str, Any]] = {}
    for decision_id, info in rows:
        types = group_types.get(info["group_id"], set())
        paired = ("call" in types and "put" in types)
        info["paired"] = paired
        if paired and info["option_type"] in ("call", "put"):
            info["group_role"] = f"{info['option_type']}_leg"
        else:
            info["group_role"] = "single"
        out[decision_id] = info
    return out


def summarize_books(assignments: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """Aggregate per-book counts; paired groups count once per group."""
    by_book: Dict[str, int] = {}
    groups_by_book: Dict[str, set] = {}
    buckets: Dict[str, int] = {}
    paired_groups: set = set()
    for info in assignments.values():
        book = info.get("book", BOOK_UNKNOWN)
        by_book[book] = by_book.get(book, 0) + 1
        groups_by_book.setdefault(book, set()).add(info.get("group_id"))
        if info.get("paired"):
            paired_groups.add(info.get("group_id"))
        b = info.get("dte_bucket")
        if b:
            buckets[b] = buckets.get(b, 0) + 1
    return {
        "candidates_by_book": by_book,
        "groups_by_book": {k: len(v) for k, v in groups_by_book.items()},
        "paired_groups": len(paired_groups),
        "swing_dte_buckets": buckets,
    }


# --------------------------------------------------------------------------- #
# duplicate-order investigation (hedge-aware)
# --------------------------------------------------------------------------- #

def _parse_at(v: Any) -> Optional[datetime]:
    if isinstance(v, datetime):
        return v if v.tzinfo else v.replace(tzinfo=UTC)
    if isinstance(v, str):
        try:
            dt = datetime.fromisoformat(v.replace("Z", "+00:00"))
        except ValueError:
            return None
        return dt if dt.tzinfo else dt.replace(tzinfo=UTC)
    return None


def investigate_orders(orders: Iterable[Dict[str, Any]],
                       *, window_s: float = 120.0) -> Dict[str, Any]:
    """Classify same-time order activity via strategy identity.

    Input rows: ``{order_id, symbol, side, qty, at}`` (``at`` ISO string or
    datetime; OCC fields derived from ``symbol``). Output distinguishes:

      duplicate_suspects    >=2 distinct order_ids, SAME contract, same side
                            and qty, within ``window_s`` — the real defect.
      hedge_pairs           call+put on the same underlying+expiration at
                            the same time — an intentional paired strategy,
                            explicitly NOT a duplicate.
      concurrent_distinct   same-time, same underlying+type but different
                            contracts (e.g. laddered strikes) — noted only.

    Observe-only: reports, never blocks or cancels anything.
    """
    rows: List[Dict[str, Any]] = []
    n_bad = 0
    for o in orders:
        at = _parse_at(o.get("at"))
        sym = str(o.get("symbol") or "")
        if at is None or not sym:
            n_bad += 1
            continue
        contract = val.parse_occ_symbol(sym) or {}
        rows.append({
            "order_id": str(o.get("order_id") or ""),
            "symbol": sym,
            "side": str(o.get("side") or "buy").lower(),
            "qty": o.get("qty"),
            "at": at,
            "underlying": contract.get("underlying"),
            "expiration": contract.get("expiration"),
            "option_type": contract.get("option_type"),
        })
    rows.sort(key=lambda r: r["at"])

    duplicate_suspects: List[Dict[str, Any]] = []
    hedge_pairs: List[Dict[str, Any]] = []
    concurrent_distinct: List[Dict[str, Any]] = []
    seen_dup: set = set()
    seen_hedge: set = set()
    seen_conc: set = set()

    for i, a in enumerate(rows):
        for b in rows[i + 1:]:
            dt_s = (b["at"] - a["at"]).total_seconds()
            if dt_s > window_s:
                break
            key = tuple(sorted((a["order_id"], b["order_id"])))
            if (a["symbol"] == b["symbol"] and a["side"] == b["side"]
                    and a["qty"] == b["qty"]
                    and a["order_id"] != b["order_id"]):
                if key not in seen_dup:
                    seen_dup.add(key)
                    duplicate_suspects.append({
                        "symbol": a["symbol"], "side": a["side"],
                        "qty": a["qty"], "seconds_apart": dt_s,
                        "order_ids": list(key),
                    })
                continue
            same_group = (a["underlying"] and a["expiration"]
                          and a["underlying"] == b["underlying"]
                          and a["expiration"] == b["expiration"])
            if not same_group:
                continue
            if (a["option_type"] != b["option_type"]
                    and {a["option_type"], b["option_type"]}
                    == {"call", "put"}):
                if key not in seen_hedge:
                    seen_hedge.add(key)
                    hedge_pairs.append({
                        "underlying": a["underlying"],
                        "expiration": a["expiration"],
                        "symbols": [a["symbol"], b["symbol"]],
                        "seconds_apart": dt_s,
                        "order_ids": list(key),
                        "classification": "intentional_hedge_pair",
                    })
            elif a["symbol"] != b["symbol"]:
                if key not in seen_conc:
                    seen_conc.add(key)
                    concurrent_distinct.append({
                        "underlying": a["underlying"],
                        "symbols": [a["symbol"], b["symbol"]],
                        "seconds_apart": dt_s,
                        "order_ids": list(key),
                    })

    return {
        "n_orders": len(rows),
        "n_unparseable": n_bad,
        "window_s": window_s,
        "duplicate_suspects": duplicate_suspects,
        "hedge_pairs": hedge_pairs,
        "concurrent_distinct": concurrent_distinct,
        "n_duplicate_suspects": len(duplicate_suspects),
        "n_hedge_pairs": len(hedge_pairs),
        "n_concurrent_distinct": len(concurrent_distinct),
        "note": ("hedge pairs are intentional paired strategies and are "
                 "never counted as duplicates; observe-only report"),
    }
