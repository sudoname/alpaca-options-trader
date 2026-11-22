"""Unified trade/outcome attribution identities for Oracle 4.0.2 (P0-3).

Small, deterministic identity layer that joins decisions, orders, fills and
realized trades without double counting:

* ``make_client_order_id`` -- deterministic broker ``client_order_id`` from a
  ``decision_id`` (+ retry attempt), so a submitted order traces back to
  exactly one decision and retries are distinct orders (<= 48 chars, within
  the broker limit).
* ``trade_identity`` / ``dedupe_trades`` -- stable identity for realized
  trade records (broker ``order_id`` preferred, content fingerprint as the
  honest fallback) and duplicate-record removal so re-ingested history can
  never inflate P&L.
* ``ExclusiveTradeMatcher`` -- an exclusive-claim join: once an outcome row
  claims a realized trade, no other scan can claim it again.  Claims can be
  released (rolled back) when a row fails cohort pairing, so unresolved rows
  never consume evidence; unknown outcomes stay unresolved rather than being
  coerced to zero.

Research-only: pure functions and in-memory state; no I/O, no trading.
"""
from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional, Tuple

UTC = timezone.utc

MAX_CLIENT_ORDER_ID_LEN = 48

# match_basis labels (truthful provenance of every attribution)
MATCH_ORDER_ID = "order_id"
MATCH_SYMBOL_TIME = "symbol_session_time"
MATCH_SHARED = "shared_with_challenger"

# trade identity kinds
ID_ORDER = "order_id"
ID_FINGERPRINT = "fingerprint"


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


def _num_repr(x: Any) -> str:
    """Stable textual form for a numeric field in a fingerprint."""
    if _finite(x):
        return repr(float(x))
    return str(x or "")


# --------------------------------------------------------------------------- #
# decision -> order identity
# --------------------------------------------------------------------------- #

def make_client_order_id(decision_id: Any, attempt: int = 0) -> str:
    """Deterministic ``client_order_id`` for a decision (+ retry attempt).

    Same ``(decision_id, attempt)`` always yields the same id; a retry with
    ``attempt + 1`` yields a distinct id, so the broker-side idempotency key
    maps every order to exactly one decision attempt.
    """
    text = str(decision_id or "").strip()
    if not text:
        raise ValueError("decision_id must be a non-empty string")
    if not isinstance(attempt, int) or isinstance(attempt, bool) or attempt < 0:
        raise ValueError("attempt must be a non-negative integer")
    digest = hashlib.sha256(text.encode("utf-8")).hexdigest()[:32]
    cid = f"o4-{digest}-a{attempt}"
    if len(cid) > MAX_CLIENT_ORDER_ID_LEN:  # attempt absurdly large
        raise ValueError("client_order_id would exceed broker length limit")
    return cid


# --------------------------------------------------------------------------- #
# realized-trade identity + dedupe
# --------------------------------------------------------------------------- #

def trade_fingerprint(symbol: Any, entry_time: Any, qty: Any,
                      pnl: Any) -> str:
    """Content fingerprint for a trade record that lacks a broker order id."""
    basis = "|".join([
        str(symbol or ""),
        str(entry_time or ""),
        _num_repr(qty),
        _num_repr(pnl),
    ])
    return "tf-" + hashlib.sha256(basis.encode("utf-8")).hexdigest()[:32]


def trade_identity(trade: Dict[str, Any]) -> Tuple[str, str]:
    """Return ``(kind, id)`` for a realized trade record.

    Prefers the broker ``order_id`` (or ``client_order_id``) when present;
    otherwise falls back to a content fingerprint.  Two records with the
    same identity are the same economic event and must not both carry P&L.
    """
    oid = trade.get("order_id") or trade.get("client_order_id")
    if oid:
        return ID_ORDER, str(oid)
    return ID_FINGERPRINT, trade_fingerprint(
        trade.get("symbol"), trade.get("entry_time"),
        trade.get("qty"), trade.get("pnl"))


def dedupe_trades(
        trades: Iterable[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], int]:
    """Drop duplicate trade records by identity; return (unique, n_dropped)."""
    unique: List[Dict[str, Any]] = []
    seen: set = set()
    dropped = 0
    for t in trades:
        if not isinstance(t, dict):
            continue
        key = trade_identity(t)
        if key in seen:
            dropped += 1
            continue
        seen.add(key)
        unique.append(t)
    return unique, dropped


# --------------------------------------------------------------------------- #
# exclusive-claim matching
# --------------------------------------------------------------------------- #

def shared_claim(claim: Dict[str, Any]) -> Dict[str, Any]:
    """Alias of an existing claim for the other arm of the SAME scan.

    Used when challenger and baseline selected the same contract: one
    realized trade legitimately explains both arms of that one comparison,
    while still being consumed only once from the trade pool.
    """
    return {"pnl": claim["pnl"], "trade_id": claim["trade_id"],
            "match_basis": MATCH_SHARED, "_idx": None}


class ExclusiveTradeMatcher:
    """Exclusive-claim join from outcome rows to realized trades.

    Each realized trade can be claimed at most once; claims are released
    (rolled back) when the caller's row fails to pair.  Exact broker/client
    order identity is required by default.  Legacy symbol/time attribution
    must be explicitly enabled and is never used for an unknown order id.
    """

    def __init__(self, trades: Iterable[Dict[str, Any]]):
        self._entries: List[Dict[str, Any]] = []
        self._by_order: Dict[str, int] = {}
        self._by_sym_session: Dict[Tuple[str, str], List[int]] = {}
        for t in trades:
            if not isinstance(t, dict):
                continue
            sym = str(t.get("symbol") or "")
            at = _parse_utc(t.get("entry_time"))
            pnl = t.get("pnl")
            if not sym or at is None or not _finite(pnl):
                continue
            kind, tid = trade_identity(t)
            idx = len(self._entries)
            self._entries.append({
                "trade_id": tid, "id_kind": kind, "symbol": sym,
                "session": at.date().isoformat(), "entry_at": at,
                "pnl": float(pnl), "claimed": False,
            })
            for order_key in ("order_id", "client_order_id"):
                order_id = t.get(order_key)
                if order_id and str(order_id) not in self._by_order:
                    self._by_order[str(order_id)] = idx
            self._by_sym_session.setdefault(
                (sym, at.date().isoformat()), []).append(idx)
        for idxs in self._by_sym_session.values():
            idxs.sort(key=lambda i: self._entries[i]["entry_at"])

    def _take(self, idx: int, basis: str) -> Dict[str, Any]:
        e = self._entries[idx]
        e["claimed"] = True
        return {"pnl": e["pnl"], "trade_id": e["trade_id"],
                "match_basis": basis, "_idx": idx}

    def same_order(self, first_id: Any, second_id: Any) -> bool:
        """Whether two supplied broker/client ids name the same trade."""
        if not first_id or not second_id:
            return False
        first = self._by_order.get(str(first_id))
        return first is not None and first == self._by_order.get(str(second_id))

    def claim(self, symbol: Any, session: str, after: datetime, *,
              order_id: Any = None,
              allow_heuristic: bool = False) -> Optional[Dict[str, Any]]:
        """Claim one unclaimed trade; return claim dict or None.

        An order id is authoritative: missing or already-claimed ids never
        fall back to another trade.  An exact id must also match the contract,
        session and scan time.  The legacy heuristic is opt-in and applies
        only to scans without any order id.
        """
        if order_id:
            idx = self._by_order.get(str(order_id))
            if idx is None:
                return None
            e = self._entries[idx]
            if (e["claimed"] or e["symbol"] != str(symbol)
                    or e["session"] != session or e["entry_at"] < after):
                return None
            return self._take(idx, MATCH_ORDER_ID)
        if not allow_heuristic or not symbol:
            return None
        for idx in self._by_sym_session.get((str(symbol), session), []):
            e = self._entries[idx]
            if e["claimed"] or e["entry_at"] < after:
                continue
            return self._take(idx, MATCH_SYMBOL_TIME)
        return None

    def release(self, claim: Optional[Dict[str, Any]]) -> None:
        """Roll back a claim (no-op for None / shared aliases)."""
        if not claim:
            return
        idx = claim.get("_idx")
        if idx is not None:
            self._entries[idx]["claimed"] = False
