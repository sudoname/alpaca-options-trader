"""Oracle 4.0.1 Stage B: verified model inference behind a ForecastProvider.

Fixes the Oracle 4.0 defect where ``ORACLE4_MODEL_ID`` only stamped registry
*metadata* onto shadow records without ever loading or invoking the neural
weights.  This module provides:

* ``ForecastUnavailable`` -- structured, machine-readable abstention.  A
  forecast that cannot be produced *verifiably* is refused with a reason;
  it is never fabricated and never silently defaulted.
* ``Forecast`` -- typed forecast result: model identity (id/kind/checksums/
  git SHA), snapshot timestamp, horizon, real direction logits and
  probabilities (p_down/p_neutral/p_up), ordered log-return quantiles,
  RV forecast, barrier cumulative-incidence (CIF) per cause, calibration
  provenance (temperature, train/val sessions), missingness accounting,
  input digest and inference latency.
* ``MLPForecastProvider`` -- loads a *registered* MLP artifact with the full
  verification chain:
    registry entry -> pinned manifest checksum -> manifest file hash ->
    cross-check of the registry-pinned artifact hash against the manifest's
    own ``checksum_sha256`` -> ``mlp.load_model`` (verifies the ``.pt``
    hash, d_in and state dict) -> feature-schema validation -> exact
    standardizer reconstruction from the manifest.
  Any mismatch refuses to load (fail closed) with a reason.
* a module-level provider cache with *versioned refresh*: the registry entry
  is re-read on each ``get_provider`` call and the cached provider is
  discarded when the pinned manifest checksum changes.  There is no online
  learning; a provider is immutable once constructed.

Classical/GBM models currently persist no weights (Stage E adds that), so
``from_registry`` honestly refuses non-neural kinds with KIND_UNSUPPORTED
rather than substituting a bootstrap and calling it a model.

No trading/order modules are imported here; this is research-only code.
"""
from __future__ import annotations

import hashlib
import json
import os
import time
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from oracle.lab.purged_validation import Standardizer
from oracle.ml import registry as reg
from oracle.ml.schema import canonical_json
from oracle.ml.mlp import (
    HAZARD_BIN_EDGES_MIN, HAZARD_CAUSES, TORCH_AVAILABLE, TORCH_WHY,
    ModelIntegrityError, cif_from_hazards, load_model, predict_heads,
)

SCHEMA_VERSION = 1

#: Default window after the last trained session within which the model is
#: considered current.  Beyond it the provider abstains with MODEL_STALE
#: rather than silently extrapolating from an outdated fit.
DEFAULT_MAX_STALENESS_DAYS = 45


# --------------------------------------------------------------------------- #
# structured refusal
# --------------------------------------------------------------------------- #

class ForecastUnavailable(Exception):
    """A forecast was refused.  ``reason`` is machine-readable."""

    REASONS = (
        "MODEL_ID_UNSET",            # no model configured
        "TORCH_UNAVAILABLE",         # torch missing/broken in this env
        "REGISTRY_MISSING",          # registry file does not exist
        "REGISTRY_ENTRY_MISSING",    # model_id not in registry
        "KIND_UNSUPPORTED",          # no persisted weights for this kind yet
        "MANIFEST_NOT_PINNED",       # registry entry lacks manifest/checksum
        "MANIFEST_MISSING",          # pinned manifest file absent
        "REGISTRY_CHECKSUM_MISMATCH",  # manifest hash != registry-pinned hash
        "ARTIFACT_CHECKSUM_MISMATCH",  # registry artifact hash != manifest's
        "MODEL_INTEGRITY",           # mlp.load_model refused (.pt hash, d_in,
                                     # state dict, unreadable manifest, ...)
        "SCHEMA_INVALID",            # feature_columns inconsistent with
                                     # standardizer feature names
        "MODEL_STALE",               # as_of too far past last trained session
        "TRAIN_OVERLAP",             # as_of inside/before the train window
        "FEATURES_ALL_MISSING",      # no usable feature values at all
        "HISTORY_UNTIMESTAMPED",     # inputs built on an assumed time grid
        "INFERENCE_ERROR",           # unexpected failure during inference
    )

    def __init__(self, reason: str, detail: str = "",
                 model_id: Optional[str] = None) -> None:
        if reason not in self.REASONS:
            raise ValueError(f"unknown ForecastUnavailable reason: {reason!r}")
        self.reason = reason
        self.detail = detail
        self.model_id = model_id
        super().__init__(f"{reason}: {detail}" if detail else reason)

    def to_dict(self) -> Dict[str, Any]:
        return {"available": False, "reason": self.reason,
                "detail": self.detail, "model_id": self.model_id}


# --------------------------------------------------------------------------- #
# typed forecast result
# --------------------------------------------------------------------------- #

@dataclass
class Forecast:
    """One verified model forecast for one decision snapshot."""
    schema_version: int
    model_id: str
    model_kind: str
    horizon: Optional[str]
    checksums: Dict[str, str]          # {"manifest": sha, "artifact": sha}
    git_sha_registered: Optional[str]
    temperature: float
    as_of: datetime
    generated_at: datetime
    # direction (real logits + calibrated probabilities)
    direction_logits: List[float]      # [down, neutral, up] raw
    p_down: float
    p_neutral: float
    p_up: float
    # ordered log-return quantiles keyed by probability ("0.05" .. "0.95")
    quantiles: Dict[str, float]
    # realized-volatility forecast (always > 0 by construction)
    rv_forecast: float
    # barrier cumulative incidence per cause over the hazard time bins
    cif: Dict[str, List[float]]        # cause -> per-bin CIF
    cif_bin_edges_min: List[float]
    # calibration / provenance
    train_sessions: List[str]
    val_session: Optional[str]
    # input accounting
    n_features: int
    n_missing: int
    missing_features: List[str]
    input_digest: str
    latency_ms: float
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "available": True,
            "schema_version": self.schema_version,
            "model_id": self.model_id,
            "model_kind": self.model_kind,
            "horizon": self.horizon,
            "checksums": dict(self.checksums),
            "git_sha_registered": self.git_sha_registered,
            "temperature": self.temperature,
            "as_of": self.as_of.astimezone(timezone.utc).isoformat(),
            "generated_at": self.generated_at.astimezone(
                timezone.utc).isoformat(),
            "direction_logits": list(self.direction_logits),
            "p_down": self.p_down,
            "p_neutral": self.p_neutral,
            "p_up": self.p_up,
            "quantiles": dict(self.quantiles),
            "rv_forecast": self.rv_forecast,
            "cif": {k: list(v) for k, v in self.cif.items()},
            "cif_bin_edges_min": list(self.cif_bin_edges_min),
            "train_sessions": list(self.train_sessions),
            "val_session": self.val_session,
            "n_features": self.n_features,
            "n_missing": self.n_missing,
            "missing_features": list(self.missing_features),
            "input_digest": self.input_digest,
            "latency_ms": self.latency_ms,
            "meta": dict(self.meta),
        }


# --------------------------------------------------------------------------- #
# provider
# --------------------------------------------------------------------------- #

def _schema_check(manifest: Dict[str, Any]) -> List[str]:
    """Validate feature_columns == names + [n+'_missing'] in order.

    Returns the standardizer feature names on success, raises
    ForecastUnavailable(SCHEMA_INVALID) otherwise.
    """
    cols = manifest.get("feature_columns")
    ms = manifest.get("standardizer") or {}
    names = ms.get("feature_names")
    if not isinstance(cols, list) or not isinstance(names, list) or not names:
        raise ForecastUnavailable(
            "SCHEMA_INVALID", "manifest lacks feature_columns/standardizer")
    n = len(names)
    if len(cols) != 2 * n:
        raise ForecastUnavailable(
            "SCHEMA_INVALID",
            f"feature_columns has {len(cols)} entries, expected {2 * n}")
    if cols[:n] != names:
        raise ForecastUnavailable(
            "SCHEMA_INVALID",
            "feature_columns prefix does not match standardizer order")
    expect_flags = [f"{name}_missing" for name in names]
    if cols[n:] != expect_flags:
        raise ForecastUnavailable(
            "SCHEMA_INVALID",
            "feature_columns missing-flag suffix does not match names")
    mean, std = ms.get("mean") or {}, ms.get("std") or {}
    for name in names:
        if name not in mean or name not in std:
            raise ForecastUnavailable(
                "SCHEMA_INVALID", f"standardizer lacks stats for {name!r}")
    return names


def _rebuild_standardizer(manifest: Dict[str, Any]) -> Standardizer:
    ms = manifest["standardizer"]
    std = Standardizer()
    std.feature_names = list(ms["feature_names"])
    std.mean = {k: float(v) for k, v in ms["mean"].items()}
    std.std = {k: float(v) for k, v in ms["std"].items()}
    return std


def _last_session_date(manifest: Dict[str, Any]) -> Optional[datetime]:
    """UTC midnight of the last session used in training/validation."""
    sessions = list(manifest.get("train_sessions") or [])
    if manifest.get("val_session"):
        sessions.append(manifest["val_session"])
    best = None
    for s in sessions:
        try:
            d = datetime.strptime(str(s), "%Y-%m-%d").replace(
                tzinfo=timezone.utc)
        except ValueError:
            continue
        if best is None or d > best:
            best = d
    return best


class MLPForecastProvider:
    """Verified inference wrapper around one registered MLP artifact.

    Immutable after construction: the model, standardizer and calibration
    temperature are fixed; there is no online learning.
    """

    def __init__(self, model: Any, manifest: Dict[str, Any], *,
                 model_id: str, model_kind: str,
                 checksums: Dict[str, str],
                 git_sha_registered: Optional[str],
                 device: str = "cpu",
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> None:
        self.model = model
        self.manifest = manifest
        self.model_id = model_id
        self.model_kind = model_kind
        self.checksums = dict(checksums)
        self.git_sha_registered = git_sha_registered
        self.device = device
        self.max_staleness_days = int(max_staleness_days)
        self.standardizer = _rebuild_standardizer(manifest)
        self.feature_names: List[str] = list(self.standardizer.feature_names)
        self.temperature = float(manifest.get("temperature") or 1.0)
        self.horizon: Optional[str] = (
            (manifest.get("extra") or {}).get("horizon"))
        self.quantile_ladder: List[float] = [
            float(q) for q in (manifest.get("config") or {}).get(
                "quantiles", (0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95))]
        self._last_session = _last_session_date(manifest)

    # ------------------------------------------------------------------ #
    # construction with the verification chain
    # ------------------------------------------------------------------ #

    @classmethod
    def from_manifest(cls, manifest_path: str, *,
                      model_id: str = "unregistered",
                      model_kind: str = "neural",
                      checksums: Optional[Dict[str, str]] = None,
                      git_sha_registered: Optional[str] = None,
                      device: str = "cpu",
                      max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                      ) -> "MLPForecastProvider":
        """Load directly from a manifest (used by tests and from_registry).

        Still enforces every integrity check that does not require a
        registry: ``.pt`` content hash, d_in consistency, feature schema.
        """
        if not TORCH_AVAILABLE:
            raise ForecastUnavailable(
                "TORCH_UNAVAILABLE", TORCH_WHY, model_id=model_id)
        try:
            model, manifest = load_model(manifest_path, device=device)
        except ModelIntegrityError as e:
            raise ForecastUnavailable(
                "MODEL_INTEGRITY", str(e), model_id=model_id) from e
        _schema_check(manifest)
        cks = dict(checksums or {})
        cks.setdefault("manifest", reg.sha256_file(manifest_path))
        cks.setdefault("artifact",
                       str(manifest.get("checksum_sha256") or ""))
        return cls(model, manifest, model_id=model_id, model_kind=model_kind,
                   checksums=cks, git_sha_registered=git_sha_registered,
                   device=device, max_staleness_days=max_staleness_days)

    @classmethod
    def from_registry(cls, registry_path: str, model_id: str, *,
                      device: str = "cpu",
                      max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                      ) -> "MLPForecastProvider":
        """Load a registered model, validating content hashes before load."""
        model_id = (model_id or "").strip()
        if not model_id:
            raise ForecastUnavailable("MODEL_ID_UNSET", "no model configured")
        if not TORCH_AVAILABLE:
            raise ForecastUnavailable(
                "TORCH_UNAVAILABLE", TORCH_WHY, model_id=model_id)
        if not os.path.exists(registry_path):
            raise ForecastUnavailable(
                "REGISTRY_MISSING", registry_path, model_id=model_id)
        entry = reg.get_model(registry_path, model_id)
        if entry is None:
            raise ForecastUnavailable(
                "REGISTRY_ENTRY_MISSING", model_id, model_id=model_id)
        kind = entry.get("kind")
        if kind != "neural":
            raise ForecastUnavailable(
                "KIND_UNSUPPORTED",
                f"kind={kind!r}: only neural artifacts persist loadable "
                "weights today (classical/GBM persistence lands in Stage E)",
                model_id=model_id)
        manifest_path = entry.get("manifest_path")
        pinned = (entry.get("checksums") or {}).get("manifest")
        if not manifest_path or not pinned:
            raise ForecastUnavailable(
                "MANIFEST_NOT_PINNED",
                "registry entry lacks manifest_path/manifest checksum",
                model_id=model_id)
        if not os.path.exists(manifest_path):
            raise ForecastUnavailable(
                "MANIFEST_MISSING", manifest_path, model_id=model_id)
        actual = reg.sha256_file(manifest_path)
        if actual != pinned:
            raise ForecastUnavailable(
                "REGISTRY_CHECKSUM_MISMATCH",
                f"manifest sha256 {actual[:12]} != pinned {pinned[:12]}",
                model_id=model_id)
        provider = cls.from_manifest(
            manifest_path, model_id=model_id, model_kind=kind,
            checksums={"manifest": pinned},
            git_sha_registered=entry.get("git_sha"),
            device=device, max_staleness_days=max_staleness_days)
        # cross-check: if the registry also pinned the .pt artifact hash it
        # must agree with the manifest's own checksum_sha256 (which
        # load_model already verified against the file contents).
        pinned_art = (entry.get("checksums") or {}).get("artifact")
        manifest_art = provider.manifest.get("checksum_sha256")
        if pinned_art and manifest_art and pinned_art != manifest_art:
            raise ForecastUnavailable(
                "ARTIFACT_CHECKSUM_MISMATCH",
                f"registry artifact sha {str(pinned_art)[:12]} != manifest "
                f"checksum_sha256 {str(manifest_art)[:12]}",
                model_id=model_id)
        return provider

    # ------------------------------------------------------------------ #
    # inference
    # ------------------------------------------------------------------ #

    def vectorize(self, features: Dict[str, Optional[float]]
                  ) -> Tuple[List[float], List[Optional[float]]]:
        """Exact training-time transformation of one feature row.

        Mirrors ``baselines.build_design`` for a single row: standardize in
        feature-name order, impute missing to 0.0 and append one missing
        indicator per feature (the training indicator set is always the
        full feature list).  Returns (design vector, raw z list).
        """
        z = self.standardizer.transform(features)
        x = [0.0 if v is None else float(v) for v in z]
        x += [1.0 if v is None else 0.0 for v in z]
        return x, z

    def forecast(self, features: Dict[str, Optional[float]], *,
                 as_of: datetime) -> Forecast:
        """Produce one verified forecast or raise ForecastUnavailable."""
        if as_of.tzinfo is None:
            raise ForecastUnavailable(
                "INFERENCE_ERROR", "as_of must be timezone-aware",
                model_id=self.model_id)
        as_of = as_of.astimezone(timezone.utc)
        last = self._last_session
        if last is not None:
            # train window end = last session's UTC day end
            train_end = last + timedelta(days=1)
            if as_of < train_end:
                raise ForecastUnavailable(
                    "TRAIN_OVERLAP",
                    f"as_of {as_of.date()} is inside/before the train window "
                    f"ending {last.date()} (would leak training data)",
                    model_id=self.model_id)
            if as_of > train_end + timedelta(days=self.max_staleness_days):
                raise ForecastUnavailable(
                    "MODEL_STALE",
                    f"as_of {as_of.date()} is more than "
                    f"{self.max_staleness_days}d past last trained session "
                    f"{last.date()}", model_id=self.model_id)
        x, z = self.vectorize(features)
        missing = [n for n, v in zip(self.feature_names, z) if v is None]
        if len(missing) == len(self.feature_names):
            raise ForecastUnavailable(
                "FEATURES_ALL_MISSING",
                "no usable feature values at as_of", model_id=self.model_id)
        digest = hashlib.sha256(
            canonical_json(dict(features)).encode("utf-8")).hexdigest()
        t0 = time.perf_counter()
        try:
            out = predict_heads(
                self.model, np.asarray([x], dtype=np.float64),
                device=self.device, temperature=self.temperature)
            cif = cif_from_hazards(out["hazards"])
        except Exception as e:  # torch runtime faults -> structured refusal
            raise ForecastUnavailable(
                "INFERENCE_ERROR", f"{type(e).__name__}: {e}",
                model_id=self.model_id) from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
        probs = out["direction_probs"][0]
        logits = out["direction_logits"][0]
        qs = out["quantiles"][0]
        quantiles = {f"{q:g}": float(v)
                     for q, v in zip(self.quantile_ladder, qs)}
        cif_by_cause = {cause: [float(v) for v in cif[0, :, j]]
                        for j, cause in enumerate(HAZARD_CAUSES)}
        return Forecast(
            schema_version=SCHEMA_VERSION,
            model_id=self.model_id,
            model_kind=self.model_kind,
            horizon=self.horizon,
            checksums=dict(self.checksums),
            git_sha_registered=self.git_sha_registered,
            temperature=self.temperature,
            as_of=as_of,
            generated_at=datetime.now(timezone.utc),
            direction_logits=[float(v) for v in logits],
            p_down=float(probs[0]),
            p_neutral=float(probs[1]),
            p_up=float(probs[2]),
            quantiles=quantiles,
            rv_forecast=float(out["rv_forecast"][0]),
            cif=cif_by_cause,
            cif_bin_edges_min=[float(v) for v in HAZARD_BIN_EDGES_MIN],
            train_sessions=list(self.manifest.get("train_sessions") or []),
            val_session=self.manifest.get("val_session"),
            n_features=len(self.feature_names),
            n_missing=len(missing),
            missing_features=missing,
            input_digest=digest,
            latency_ms=latency_ms,
        )


# --------------------------------------------------------------------------- #
# provider cache with versioned refresh (loaded once outside scan loops)
# --------------------------------------------------------------------------- #

_PROVIDERS: Dict[Tuple[str, str, str], MLPForecastProvider] = {}


def clear_cache() -> None:
    _PROVIDERS.clear()


def get_provider(registry_path: str, model_id: str, *,
                 device: str = "cpu",
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                 ) -> MLPForecastProvider:
    """Cached, verified provider lookup with versioned refresh.

    The registry entry is re-read on every call (cheap JSON read); if the
    pinned manifest checksum changed since the provider was cached, the
    stale provider is dropped and the new artifact is verified and loaded.
    Failures are never cached.
    """
    key = (os.path.abspath(registry_path), (model_id or "").strip(), device)
    cached = _PROVIDERS.get(key)
    if cached is not None:
        entry = (reg.get_model(registry_path, model_id)
                 if os.path.exists(registry_path) else None)
        pinned = ((entry or {}).get("checksums") or {}).get("manifest")
        if entry is not None and pinned == cached.checksums.get("manifest"):
            return cached
        del _PROVIDERS[key]  # registry changed under us: refresh
    provider = MLPForecastProvider.from_registry(
        registry_path, model_id, device=device,
        max_staleness_days=max_staleness_days)
    _PROVIDERS[key] = provider
    return provider
