"""Versioned, broker-independent research contracts shared by Oracle variants.

This module intentionally depends on Python's standard library only. It must
never import an options/futures broker, model framework, or live scheduler.

These are *boundary* types, not a replacement for Oracle 4's existing
decision/schema or model forecast formats. Adapters convert at the boundary.
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Mapping, Optional, Protocol, Sequence, Tuple, runtime_checkable

CONTRACT_VERSION = 1
ACTIONS = frozenset({"enter", "abstain", "block"})
SIDES = frozenset({"long", "short"})


def _utc(dt: datetime) -> datetime:
    if not isinstance(dt, datetime) or dt.tzinfo is None or dt.utcoffset() is None:
        raise ValueError("timestamp must be timezone-aware")
    return dt.astimezone(timezone.utc)


def _finite(value: float, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite number")
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _identity(value: str, name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be nonempty")
    return value


@dataclass(frozen=True)
class FeatureValue:
    """One point-in-time measurement. Missing values must be None."""
    name: str
    value: Optional[float]
    available_at: datetime

    def __post_init__(self) -> None:
        _identity(self.name, "feature name")
        object.__setattr__(self, "available_at", _utc(self.available_at))
        if self.value is not None:
            object.__setattr__(self, "value", _finite(self.value, self.name))


@dataclass(frozen=True)
class ResearchSnapshot:
    """Asset-neutral snapshot. Contract symbols are never interpreted here."""
    snapshot_id: str
    asset_class: str
    instrument_id: str
    observed_at: datetime
    decision_at: datetime
    features: Tuple[FeatureValue, ...]
    schema_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "asset_class", "instrument_id"):
            _identity(getattr(self, name), name)
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("unsupported research contract version")
        object.__setattr__(self, "observed_at", _utc(self.observed_at))
        object.__setattr__(self, "decision_at", _utc(self.decision_at))
        if self.observed_at > self.decision_at:
            raise ValueError("observation after decision time")
        values = tuple(self.features)
        if len({f.name for f in values}) != len(values):
            raise ValueError("duplicate feature name")
        if any(f.available_at > self.decision_at for f in values):
            raise ValueError("feature unavailable at decision time")
        object.__setattr__(self, "features", values)

    def feature_dict(self) -> dict[str, Optional[float]]:
        return {f.name: f.value for f in self.features}


@dataclass(frozen=True)
class ResearchForecast:
    """Calibrated output, without options direction names or broker objects."""
    snapshot_id: str
    model_id: str
    model_checksum: str
    produced_at: datetime
    p_down: float
    p_flat: float
    p_up: float
    horizon: str
    schema_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        for name in ("snapshot_id", "model_id", "model_checksum", "horizon"):
            _identity(getattr(self, name), name)
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("unsupported research contract version")
        object.__setattr__(self, "produced_at", _utc(self.produced_at))
        probs = tuple(_finite(getattr(self, n), n) for n in ("p_down", "p_flat", "p_up"))
        if any(p < 0.0 or p > 1.0 for p in probs) or abs(sum(probs) - 1.0) > 1e-6:
            raise ValueError("direction probabilities must sum to one")


@dataclass(frozen=True)
class ResearchCandidate:
    candidate_id: str
    snapshot_id: str
    strategy_id: str
    side: str
    quantity: int

    def __post_init__(self) -> None:
        for name in ("candidate_id", "snapshot_id", "strategy_id"):
            _identity(getattr(self, name), name)
        if self.side not in SIDES:
            raise ValueError("side must be long or short")
        if isinstance(self.quantity, bool) or not isinstance(self.quantity, int) or self.quantity <= 0:
            raise ValueError("quantity must be a positive integer")


@dataclass(frozen=True)
class ResearchDecision:
    candidate_id: str
    snapshot_id: str
    action: str
    reason: str
    forecast_model_id: Optional[str] = None
    expected_net_usd: Optional[float] = None
    schema_version: int = CONTRACT_VERSION

    def __post_init__(self) -> None:
        _identity(self.candidate_id, "candidate_id")
        _identity(self.snapshot_id, "snapshot_id")
        _identity(self.reason, "reason")
        if self.action not in ACTIONS:
            raise ValueError("unsupported action")
        if self.schema_version != CONTRACT_VERSION:
            raise ValueError("unsupported research contract version")
        if self.expected_net_usd is not None:
            _finite(self.expected_net_usd, "expected_net_usd")
        if self.action == "enter" and (
            self.expected_net_usd is None or not self.forecast_model_id
        ):
            raise ValueError("enter requires model provenance and expected net value")

    def fingerprint(self) -> str:
        payload = {
            "v": self.schema_version, "candidate": self.candidate_id,
            "snapshot": self.snapshot_id, "action": self.action,
            "reason": self.reason, "model": self.forecast_model_id,
            "expected_net_usd": self.expected_net_usd,
        }
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


@runtime_checkable
class SnapshotSource(Protocol):
    def build_snapshot(self, instrument_id: str, as_of: datetime) -> ResearchSnapshot: ...


@runtime_checkable
class ResearchModel(Protocol):
    def predict(self, snapshot: ResearchSnapshot) -> ResearchForecast: ...


@runtime_checkable
class CandidatePolicy(Protocol):
    def decide(
        self, snapshot: ResearchSnapshot,
        candidate: ResearchCandidate,
        forecast: ResearchForecast,
    ) -> ResearchDecision: ...


def validate_prediction(
    snapshot: ResearchSnapshot,
    candidate: ResearchCandidate,
    forecast: ResearchForecast,
) -> None:
    """Reject mismatched inputs, stale/future inference, and fabricated joins."""
    if candidate.snapshot_id != snapshot.snapshot_id or forecast.snapshot_id != snapshot.snapshot_id:
        raise ValueError("snapshot identity mismatch")
    if forecast.produced_at < snapshot.decision_at:
        raise ValueError("forecast predates its inputs")
