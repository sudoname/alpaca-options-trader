"""Typed records for the Oracle 4.0 feature/label store.

Conventions (per the Oracle 4.0 master prompt):
  * all timestamps are timezone-aware UTC ``datetime`` objects;
  * every record carries both the event timestamp (when the thing happened)
    and the availability timestamp (when we could first have known it);
  * missing data is ``None`` -- never zero, never invented;
  * returns and IV are decimals internally; option P&L is dollars with
    multiplier * quantity applied explicitly where needed.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

SCHEMA_VERSION = 1


# --------------------------------------------------------------------------- #
# time helpers
# --------------------------------------------------------------------------- #

def parse_utc(ts: Optional[str]) -> Optional[datetime]:
    """Parse an ISO-8601 timestamp to an aware UTC datetime; None on failure.

    Naive timestamps are rejected (return None) rather than guessed: the
    ledger writes offset-aware strings, so a naive value is a data defect.
    """
    if not ts or not isinstance(ts, str):
        return None
    try:
        dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    except ValueError:
        return None
    if dt.tzinfo is None:
        return None
    return dt.astimezone(timezone.utc)


def iso_utc(dt: Optional[datetime]) -> Optional[str]:
    if dt is None:
        return None
    return dt.astimezone(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# outcome-status taxonomy
# --------------------------------------------------------------------------- #

class OutcomeStatus(str, Enum):
    """Provenance of a decision's realized outcome.

    Unknown outcomes are never silently treated as losses, wins or nonfills.
    """
    FILLS_ACTUAL = "fills_actual"          # broker fills on both legs, net $
    MARK_DERIVED = "mark_derived"          # realized but P&L from marks only
    CONFIRMED_NONFILL = "confirmed_nonfill"  # submitted + cancel/reject seen
    POLICY_SKIP = "policy_skip"            # gate rejected before submission
    COUNTERFACTUAL = "counterfactual"      # shadow/simulated, never submitted
    MISSING = "missing"                    # no evidence either way


class LabelStatus(str, Enum):
    """Resolution state of a forward-looking label."""
    OK = "ok"                 # label fully resolved from real observations
    CENSORED = "censored"     # horizon end beyond available data
    MISSING = "missing"       # required inputs absent (never imputed)
    SIMULATED = "simulated"   # resolved only by a simulated/counterfactual
                              # outcome -- never trainable as real (4.0.1-A)


# --------------------------------------------------------------------------- #
# records
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class UnderlyingObs:
    """One observation of an underlying price, with availability time."""
    underlying: str
    event_at: datetime
    available_at: datetime
    price: float
    source_snapshot_id: str


@dataclass
class DecisionRecord:
    """A raw ledger decision (candidate or shadow_candidate snapshot)."""
    decision_id: str
    snapshot_id: str
    kind: str                       # 'candidate' | 'shadow_candidate'
    accepted: bool
    reject_code: Optional[str]
    as_of: datetime                 # event_at_utc of the snapshot
    available_at: datetime          # received_at_utc (>= as_of)
    underlying: str
    symbol: Optional[str]
    expiration: Optional[str]
    strike: Optional[float]
    option_type: Optional[str]
    multiplier: Optional[float]
    quote_bid: Optional[float]
    quote_ask: Optional[float]
    underlying_price: Optional[float]
    iv: Optional[float]
    delta: Optional[float]
    direction: Optional[str]
    score: Optional[float]
    p_win: Optional[float]
    ev: Optional[float]
    signals: Dict[str, Any] = field(default_factory=dict)
    prediction: Dict[str, Any] = field(default_factory=dict)
    sizing: Dict[str, Any] = field(default_factory=dict)
    regime: Optional[str] = None
    config_hash: Optional[str] = None
    code_rev: Optional[str] = None
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class FeatureRow:
    """Point-in-time features for one decision (no future information)."""
    decision_id: str
    snapshot_id: str
    as_of: datetime
    available_at: datetime
    underlying: str
    symbol: Optional[str]
    kind: str
    session: str                    # YYYY-MM-DD exchange-session key (UTC date)
    features: Dict[str, Optional[float]]
    quality: Dict[str, Any]


@dataclass
class LabelRow:
    """One forward label for one decision at one horizon."""
    decision_id: str
    horizon: str                    # '15m' | '60m' | '1s' | '5s' | 'event'
    label_type: str                 # 'log_return' | 'direction' | 'future_rv'
                                    # | 'iv_change' | 'first_touch'
    value: Optional[float]          # numeric value; for first_touch see meta
    status: LabelStatus
    label_end: Optional[datetime]   # when the measured quantity ended
    available_at: Optional[datetime]  # earliest time the label was knowable
    meta: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OutcomeRecord:
    """Realized-outcome classification for one decision."""
    decision_id: str
    status: OutcomeStatus
    gross_pnl_pct: Optional[float]
    net_pnl_dollars: Optional[float]
    net_pnl_pct: Optional[float]
    reason: Optional[str]
    fill_price: Optional[float]
    fill_qty: Optional[float]
    outcome_at: Optional[datetime]
    meta: Dict[str, Any] = field(default_factory=dict)


# --------------------------------------------------------------------------- #
# canonical serialization + hashing (reproducibility)
# --------------------------------------------------------------------------- #

def _canon(obj: Any) -> Any:
    if isinstance(obj, datetime):
        return iso_utc(obj)
    if isinstance(obj, Enum):
        return obj.value
    if isinstance(obj, dict):
        return {k: _canon(v) for k, v in sorted(obj.items())}
    if isinstance(obj, (list, tuple)):
        return [_canon(v) for v in obj]
    if isinstance(obj, float):
        # repr round-trip keeps full precision deterministically
        return float(repr(obj)) if obj == obj else None  # NaN -> None
    return obj


def canonical_json(record: Any) -> str:
    """Deterministic JSON for a dataclass/dict (sorted keys, UTC ISO times)."""
    if hasattr(record, "__dataclass_fields__"):
        record = asdict(record)
    return json.dumps(_canon(record), sort_keys=True, separators=(",", ":"))


def rows_hash(rows: List[Any]) -> str:
    """Order-independent sha256 over canonical row serializations."""
    h = hashlib.sha256()
    for line in sorted(canonical_json(r) for r in rows):
        h.update(line.encode("utf-8"))
        h.update(b"\n")
    return h.hexdigest()
