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

Oracle 4.0.2 (P1-7) extends this into a *unified provider interface*: the
same verification chain and temporal guards now serve

* ``MLPForecastProvider`` -- kind="neural" (.pt + manifest pair),
* ``ClassicalForecastProvider`` -- kind="classical" with a multinomial-
  logistic self-checksummed JSON artifact,
* ``GBMForecastProvider`` -- kind="classical" with a LightGBM direction
  artifact (abstains with LIGHTGBM_UNAVAILABLE when the dep is absent),
* ``EnsembleForecastProvider`` -- kind="ensemble": a pinned manifest lists
  registered member model_ids; probabilities are the renormalized member
  mean and member dispersion is reported as uncertainty.  Any member
  failure propagates as that member's structured refusal (never averaged
  around), and
* ``BaselineForecastProvider`` -- an explicitly labeled bootstrap baseline
  (``model_kind="baseline"``) built from training-window returns; it is
  constructed programmatically, never passed off as a trained model.

Every provider emits the same ``Forecast`` shape: identity + checksums,
train-window cutoff and staleness gating, direction probabilities that sum
to 1 with neutral as its own class (never folded into bearish), and
quantiles / RV / CIF only where the model family actually produces them
(``None``/empty otherwise -- heads are never fabricated).  Kinds with no
loadable weights (valuation/other) still refuse with KIND_UNSUPPORTED.

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
from oracle.ml import gbm as gbm_mod
from oracle.ml import registry as reg
from oracle.ml.baselines import (
    ClassicalIntegrityError, QUANTILE_LADDER, apply_temperature_np,
    load_classical,
)
from oracle.ml.schema import canonical_json
from oracle.ml.mlp import (
    HAZARD_BIN_EDGES_MIN, HAZARD_CAUSES, TORCH_AVAILABLE, TORCH_WHY,
    ModelIntegrityError, cif_from_hazards, direction_class, load_model,
    predict_heads,
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
        "LIGHTGBM_UNAVAILABLE",      # lightgbm missing/broken in this env
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
        "BARS_DB_UNSET",             # no bars store configured: no genuine
                                     # timestamped history to build features on
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
    # realized-volatility forecast (> 0 where the family produces one;
    # None for direction-only families -- never fabricated)
    rv_forecast: Optional[float]
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
    vs = manifest.get("val_session")
    if vs:
        sessions.extend(vs if isinstance(vs, list) else [vs])
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


def _as_of_guard(as_of: datetime, last: Optional[datetime],
                 max_staleness_days: int,
                 model_id: str) -> datetime:
    """Shared temporal gating: leak + staleness checks; returns UTC as_of."""
    if as_of.tzinfo is None:
        raise ForecastUnavailable(
            "INFERENCE_ERROR", "as_of must be timezone-aware",
            model_id=model_id)
    as_of = as_of.astimezone(timezone.utc)
    if last is not None:
        # train window end = last session's UTC day end
        train_end = last + timedelta(days=1)
        if as_of < train_end:
            raise ForecastUnavailable(
                "TRAIN_OVERLAP",
                f"as_of {as_of.date()} is inside/before the train window "
                f"ending {last.date()} (would leak training data)",
                model_id=model_id)
        if as_of > train_end + timedelta(days=max_staleness_days):
            raise ForecastUnavailable(
                "MODEL_STALE",
                f"as_of {as_of.date()} is more than "
                f"{max_staleness_days}d past last trained session "
                f"{last.date()}", model_id=model_id)
    return as_of


def _input_digest(features: Dict[str, Optional[float]]) -> str:
    return hashlib.sha256(
        canonical_json(dict(features)).encode("utf-8")).hexdigest()


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
                      ) -> Any:
        """Backwards-compatible alias for :func:`provider_from_registry`.

        The verification chain and kind routing live in the module-level
        function; depending on the registered kind the returned provider may
        be an MLP, classical, GBM or ensemble provider.
        """
        return provider_from_registry(
            registry_path, model_id, device=device,
            max_staleness_days=max_staleness_days)

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
        as_of = _as_of_guard(as_of, self._last_session,
                             self.max_staleness_days, self.model_id)
        x, z = self.vectorize(features)
        missing = [n for n, v in zip(self.feature_names, z) if v is None]
        if len(missing) == len(self.feature_names):
            raise ForecastUnavailable(
                "FEATURES_ALL_MISSING",
                "no usable feature values at as_of", model_id=self.model_id)
        digest = _input_digest(features)
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
# classical providers (direction-only; heads never fabricated)
# --------------------------------------------------------------------------- #

class ClassicalForecastProvider:
    """Verified inference wrapper around one saved classical artifact.

    Multinomial-logistic direction model persisted by
    ``baselines.save_classical`` (self-checksummed JSON).  Direction-only:
    quantiles/RV/CIF come back empty/None rather than fabricated.
    """

    def __init__(self, model: Any, feature_columns: List[str],
                 temperature: float, meta: Dict[str, Any], *,
                 model_id: str, model_kind: str = "classical",
                 checksums: Dict[str, str],
                 git_sha_registered: Optional[str] = None,
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> None:
        self.model = model
        self.meta = dict(meta or {})
        pseudo = {"feature_columns": list(feature_columns),
                  "standardizer": self.meta.get("standardizer") or {}}
        _schema_check(pseudo)
        self.standardizer = _rebuild_standardizer(pseudo)
        self.feature_names: List[str] = list(self.standardizer.feature_names)
        self.model_id = model_id
        self.model_kind = model_kind
        self.checksums = dict(checksums)
        self.git_sha_registered = git_sha_registered
        self.max_staleness_days = int(max_staleness_days)
        self.temperature = float(temperature or 1.0)
        self.horizon: Optional[str] = self.meta.get("horizon")
        self.train_sessions = [
            str(s) for s in (self.meta.get("train_sessions") or [])]
        vs = self.meta.get("val_session")
        if isinstance(vs, list):
            self.val_session: Optional[str] = str(vs[-1]) if vs else None
        else:
            self.val_session = str(vs) if vs else None
        self._last_session = _last_session_date(self.meta)

    @classmethod
    def from_artifact(cls, artifact_path: str, *,
                      model_id: str = "unregistered",
                      model_kind: str = "classical",
                      checksums: Optional[Dict[str, str]] = None,
                      git_sha_registered: Optional[str] = None,
                      max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                      ) -> "ClassicalForecastProvider":
        """Load + verify one self-checksummed classical JSON artifact."""
        try:
            model, cols, temperature, meta = load_classical(artifact_path)
        except (ClassicalIntegrityError, OSError, ValueError, KeyError,
                TypeError) as e:
            raise ForecastUnavailable(
                "MODEL_INTEGRITY", f"{type(e).__name__}: {e}",
                model_id=model_id) from e
        cks = dict(checksums or {})
        cks.setdefault("artifact", reg.sha256_file(artifact_path))
        return cls(model, cols, temperature, meta, model_id=model_id,
                   model_kind=model_kind, checksums=cks,
                   git_sha_registered=git_sha_registered,
                   max_staleness_days=max_staleness_days)

    def vectorize(self, features: Dict[str, Optional[float]]
                  ) -> Tuple[List[float], List[Optional[float]]]:
        """Exact training-time transformation (mirrors build_design)."""
        z = self.standardizer.transform(features)
        x = [0.0 if v is None else float(v) for v in z]
        x += [1.0 if v is None else 0.0 for v in z]
        return x, z

    def forecast(self, features: Dict[str, Optional[float]], *,
                 as_of: datetime) -> Forecast:
        """Produce one verified forecast or raise ForecastUnavailable."""
        as_of = _as_of_guard(as_of, self._last_session,
                             self.max_staleness_days, self.model_id)
        x, z = self.vectorize(features)
        missing = [n for n, v in zip(self.feature_names, z) if v is None]
        if len(missing) == len(self.feature_names):
            raise ForecastUnavailable(
                "FEATURES_ALL_MISSING",
                "no usable feature values at as_of", model_id=self.model_id)
        digest = _input_digest(features)
        t0 = time.perf_counter()
        try:
            logits = self.model.decision_scores(
                np.asarray([x], dtype=np.float64))
            probs = apply_temperature_np(logits, self.temperature)[0]
        except Exception as e:
            raise ForecastUnavailable(
                "INFERENCE_ERROR", f"{type(e).__name__}: {e}",
                model_id=self.model_id) from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
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
            direction_logits=[float(v) for v in logits[0]],
            p_down=float(probs[0]),
            p_neutral=float(probs[1]),
            p_up=float(probs[2]),
            quantiles={},
            rv_forecast=None,
            cif={},
            cif_bin_edges_min=[],
            train_sessions=list(self.train_sessions),
            val_session=self.val_session,
            n_features=len(self.feature_names),
            n_missing=len(missing),
            missing_features=missing,
            input_digest=digest,
            latency_ms=latency_ms,
            meta={"payload_kind": "multinomial_logistic"},
        )


class GBMForecastProvider:
    """Verified inference wrapper around one saved LightGBM direction model.

    Direction-only (3-class).  Abstains with LIGHTGBM_UNAVAILABLE when the
    lightgbm runtime is missing; quantiles/RV/CIF are never fabricated.
    """

    def __init__(self, booster: Any, feature_columns: List[str],
                 missingness_mode: str, meta: Dict[str, Any], *,
                 model_id: str, model_kind: str = "classical",
                 checksums: Dict[str, str],
                 git_sha_registered: Optional[str] = None,
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> None:
        cols = [str(c) for c in feature_columns]
        if missingness_mode == "flags":
            n = len(cols) // 2
            if (len(cols) != 2 * n
                    or cols[n:] != [f"{c}_missing" for c in cols[:n]]):
                raise ForecastUnavailable(
                    "SCHEMA_INVALID",
                    "flags-mode feature_columns lack a matching "
                    "missing-flag suffix", model_id=model_id)
            self.feature_names: List[str] = cols[:n]
        elif missingness_mode == "native":
            self.feature_names = cols
        else:
            raise ForecastUnavailable(
                "SCHEMA_INVALID",
                f"unknown missingness_mode {missingness_mode!r}",
                model_id=model_id)
        self.booster = booster
        self.missingness_mode = missingness_mode
        self.meta = dict(meta or {})
        self.model_id = model_id
        self.model_kind = model_kind
        self.checksums = dict(checksums)
        self.git_sha_registered = git_sha_registered
        self.max_staleness_days = int(max_staleness_days)
        self.temperature = 1.0
        self.horizon: Optional[str] = self.meta.get("horizon")
        self.train_sessions = [
            str(s) for s in (self.meta.get("train_sessions") or [])]
        vs = self.meta.get("val_session")
        if isinstance(vs, list):
            self.val_session: Optional[str] = str(vs[-1]) if vs else None
        else:
            self.val_session = str(vs) if vs else None
        self._last_session = _last_session_date(self.meta)

    @classmethod
    def from_artifact(cls, artifact_path: str, *,
                      model_id: str = "unregistered",
                      model_kind: str = "classical",
                      checksums: Optional[Dict[str, str]] = None,
                      git_sha_registered: Optional[str] = None,
                      max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                      ) -> "GBMForecastProvider":
        """Load + verify one self-checksummed GBM JSON artifact."""
        if not gbm_mod.LGBM_AVAILABLE:
            raise ForecastUnavailable(
                "LIGHTGBM_UNAVAILABLE", gbm_mod.LGBM_WHY, model_id=model_id)
        try:
            booster, cols, mode, meta = gbm_mod.load_gbm_direction(
                artifact_path)
        except (gbm_mod.GBMIntegrityError, OSError, ValueError, KeyError,
                TypeError) as e:
            raise ForecastUnavailable(
                "MODEL_INTEGRITY", f"{type(e).__name__}: {e}",
                model_id=model_id) from e
        cks = dict(checksums or {})
        cks.setdefault("artifact", reg.sha256_file(artifact_path))
        return cls(booster, cols, mode, meta, model_id=model_id,
                   model_kind=model_kind, checksums=cks,
                   git_sha_registered=git_sha_registered,
                   max_staleness_days=max_staleness_days)

    def vectorize(self, features: Dict[str, Optional[float]]
                  ) -> Tuple[List[float], List[str]]:
        """Exact training-time transformation (mirrors gbm.build_matrix)."""
        row: List[float] = []
        missing: List[str] = []
        for name in self.feature_names:
            v = features.get(name)
            is_missing = (v is None or not isinstance(v, (int, float))
                          or v != v)
            if is_missing:
                missing.append(name)
            if self.missingness_mode == "native":
                row.append(float("nan") if is_missing else float(v))
            else:
                row.append(0.0 if is_missing else float(v))
        if self.missingness_mode == "flags":
            miss_set = set(missing)
            row += [1.0 if n in miss_set else 0.0
                    for n in self.feature_names]
        return row, missing

    def forecast(self, features: Dict[str, Optional[float]], *,
                 as_of: datetime) -> Forecast:
        """Produce one verified forecast or raise ForecastUnavailable."""
        as_of = _as_of_guard(as_of, self._last_session,
                             self.max_staleness_days, self.model_id)
        x, missing = self.vectorize(features)
        if len(missing) == len(self.feature_names):
            raise ForecastUnavailable(
                "FEATURES_ALL_MISSING",
                "no usable feature values at as_of", model_id=self.model_id)
        digest = _input_digest(features)
        t0 = time.perf_counter()
        try:
            probs = gbm_mod.predict_direction_proba(
                self.booster, np.asarray([x], dtype=np.float64))[0]
        except Exception as e:
            raise ForecastUnavailable(
                "INFERENCE_ERROR", f"{type(e).__name__}: {e}",
                model_id=self.model_id) from e
        latency_ms = (time.perf_counter() - t0) * 1000.0
        logits = np.log(np.clip(np.asarray(probs, dtype=np.float64),
                                1e-12, 1.0))
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
            quantiles={},
            rv_forecast=None,
            cif={},
            cif_bin_edges_min=[],
            train_sessions=list(self.train_sessions),
            val_session=self.val_session,
            n_features=len(self.feature_names),
            n_missing=len(missing),
            missing_features=missing,
            input_digest=digest,
            latency_ms=latency_ms,
            meta={"payload_kind": "lightgbm_multiclass_direction",
                  "missingness_mode": self.missingness_mode},
        )


# --------------------------------------------------------------------------- #
# baseline + ensemble providers
# --------------------------------------------------------------------------- #

class BaselineForecastProvider:
    """Honest floor: bootstrap/empirical baseline over a train window.

    Constructed programmatically from realized returns (no registry
    artifact): Laplace-smoothed direction frequencies, empirical return
    quantiles on the standard ladder and an EWMA RV forecast.  Constant by
    design -- learned providers must beat it; labelled
    ``model_kind="baseline"`` so it can never masquerade as a model.
    """

    def __init__(self, returns: List[float], train_sessions: List[str], *,
                 model_id: str = "baseline_bootstrap",
                 horizon: Optional[str] = None,
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> None:
        vals = [float(r) for r in (returns or [])
                if isinstance(r, (int, float)) and not isinstance(r, bool)
                and r == r and abs(r) != float("inf")]
        sessions = sorted({str(s) for s in (train_sessions or []) if s})
        if not vals:
            raise ValueError("baseline needs at least one finite return")
        if not sessions:
            raise ValueError(
                "baseline needs its train sessions for cutoff gating")
        counts = [1.0, 1.0, 1.0]  # Laplace prior: no zero-prob directions
        for r in vals:
            counts[direction_class(r)] += 1.0
        tot = sum(counts)
        self._probs = [c / tot for c in counts]
        self._quantiles = {f"{q:g}": float(np.quantile(vals, q))
                           for q in QUANTILE_LADDER}
        v = vals[0] * vals[0]  # EWMA RV (RiskMetrics lambda), inline
        for r in vals[1:]:
            v = 0.94 * v + 0.06 * r * r
        self._rv = float(max(v, 1e-12) ** 0.5)
        digest = hashlib.sha256(canonical_json(
            {"returns": vals, "train_sessions": sessions,
             "horizon": horizon}).encode("utf-8")).hexdigest()
        self.model_id = model_id
        self.model_kind = "baseline"
        self.checksums = {"manifest": digest, "artifact": digest}
        self.git_sha_registered: Optional[str] = None
        self.temperature = 1.0
        self.horizon = horizon
        self.feature_names: List[str] = []
        self.train_sessions = sessions
        self.val_session: Optional[str] = None
        self.max_staleness_days = int(max_staleness_days)
        self._last_session = _last_session_date({"train_sessions": sessions})
        self._n = len(vals)

    def forecast(self, features: Dict[str, Optional[float]], *,
                 as_of: datetime) -> Forecast:
        """Constant (by design) forecast, still cutoff/staleness gated."""
        as_of = _as_of_guard(as_of, self._last_session,
                             self.max_staleness_days, self.model_id)
        t0 = time.perf_counter()
        digest = _input_digest(features)
        probs = self._probs
        return Forecast(
            schema_version=SCHEMA_VERSION,
            model_id=self.model_id,
            model_kind=self.model_kind,
            horizon=self.horizon,
            checksums=dict(self.checksums),
            git_sha_registered=None,
            temperature=1.0,
            as_of=as_of,
            generated_at=datetime.now(timezone.utc),
            direction_logits=[float(np.log(p)) for p in probs],
            p_down=float(probs[0]),
            p_neutral=float(probs[1]),
            p_up=float(probs[2]),
            quantiles=dict(self._quantiles),
            rv_forecast=self._rv,
            cif={},
            cif_bin_edges_min=[],
            train_sessions=list(self.train_sessions),
            val_session=None,
            n_features=0,
            n_missing=0,
            missing_features=[],
            input_digest=digest,
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={"basis": "bootstrap_empirical_train_window",
                  "n_train_returns": self._n},
        )


class EnsembleForecastProvider:
    """Mean-of-members ensemble over already-verified providers.

    The registered manifest pins the member model_ids; every member is
    loaded through the same verification chain.  Probabilities are the
    renormalized member mean, cross-member dispersion is surfaced as
    uncertainty, and any member failure propagates -- an ensemble never
    averages around a broken member.
    """

    def __init__(self, members: List[Any], member_ids: List[str], *,
                 model_id: str, checksums: Dict[str, str],
                 git_sha_registered: Optional[str] = None,
                 horizon: Optional[str] = None,
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS) -> None:
        if not members:
            raise ForecastUnavailable(
                "SCHEMA_INVALID", "ensemble has no members",
                model_id=model_id)
        self.members = list(members)
        self.member_ids = [str(m) for m in member_ids]
        self.model_id = model_id
        self.model_kind = "ensemble"
        self.checksums = dict(checksums)
        self.git_sha_registered = git_sha_registered
        self.temperature = 1.0
        self.horizon = horizon
        self.max_staleness_days = int(max_staleness_days)
        names: set = set()
        sessions: set = set()
        for m in self.members:
            names.update(getattr(m, "feature_names", []) or [])
            sessions.update(getattr(m, "train_sessions", []) or [])
        self.feature_names: List[str] = sorted(names)
        self.train_sessions: List[str] = sorted(sessions)
        self.val_session: Optional[str] = None

    def forecast(self, features: Dict[str, Optional[float]], *,
                 as_of: datetime) -> Forecast:
        """Renormalized member-mean forecast; member failures propagate."""
        t0 = time.perf_counter()
        fcs = [m.forecast(features, as_of=as_of) for m in self.members]
        p = np.asarray([[f.p_down, f.p_neutral, f.p_up] for f in fcs],
                       dtype=np.float64)
        mean = p.mean(axis=0)
        mean = mean / max(float(mean.sum()), 1e-12)
        disp = p.std(axis=0)
        rvs = [f.rv_forecast for f in fcs if f.rv_forecast is not None]
        rv = float(sum(rvs) / len(rvs)) if rvs else None
        # quantiles are averaged only when every member reports the same
        # non-empty ladder; otherwise omitted (never partially fabricated)
        qkeys = [tuple(sorted(f.quantiles)) for f in fcs]
        quantiles: Dict[str, float] = {}
        if qkeys and qkeys[0] and all(k == qkeys[0] for k in qkeys):
            quantiles = {k: float(np.mean([f.quantiles[k] for f in fcs]))
                         for k in qkeys[0]}
        missing = sorted({n for f in fcs for n in f.missing_features})
        return Forecast(
            schema_version=SCHEMA_VERSION,
            model_id=self.model_id,
            model_kind=self.model_kind,
            horizon=self.horizon,
            checksums=dict(self.checksums),
            git_sha_registered=self.git_sha_registered,
            temperature=1.0,
            as_of=fcs[0].as_of,
            generated_at=datetime.now(timezone.utc),
            direction_logits=[float(v)
                              for v in np.log(np.clip(mean, 1e-12, 1.0))],
            p_down=float(mean[0]),
            p_neutral=float(mean[1]),
            p_up=float(mean[2]),
            quantiles=quantiles,
            rv_forecast=rv,
            cif={},
            cif_bin_edges_min=[],
            train_sessions=list(self.train_sessions),
            val_session=None,
            n_features=len(self.feature_names),
            n_missing=len(missing),
            missing_features=missing,
            input_digest=_input_digest(features),
            latency_ms=(time.perf_counter() - t0) * 1000.0,
            meta={"member_model_ids": list(self.member_ids),
                  "prob_dispersion": {"down": float(disp[0]),
                                      "neutral": float(disp[1]),
                                      "up": float(disp[2])},
                  "uncertainty_max_std": float(disp.max())},
        )


# --------------------------------------------------------------------------- #
# unified registry loading: shared verification chain + kind routing
# --------------------------------------------------------------------------- #

def provider_from_registry(registry_path: str, model_id: str, *,
                           device: str = "cpu",
                           max_staleness_days: int =
                           DEFAULT_MAX_STALENESS_DAYS,
                           _seen: frozenset = frozenset()) -> Any:
    """Load any registered forecast provider via the verification chain.

    The shared chain (id -> registry -> entry -> pinned manifest hash) runs
    for every kind; the artifact is then routed on the registered kind:

    * ``neural``    -> :class:`MLPForecastProvider` (torch, full heads)
    * ``classical`` -> :class:`ClassicalForecastProvider` or
      :class:`GBMForecastProvider`, dispatched on the artifact payload's
      own ``kind`` field
    * ``ensemble``  -> :class:`EnsembleForecastProvider` over recursively
      loaded members (membership cycles refuse with SCHEMA_INVALID)
    * ``valuation``/``other`` -> KIND_UNSUPPORTED (no loadable weights)
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ForecastUnavailable("MODEL_ID_UNSET", "no model configured")
    if not os.path.exists(registry_path):
        raise ForecastUnavailable(
            "REGISTRY_MISSING", registry_path, model_id=model_id)
    entry = reg.get_model(registry_path, model_id)
    if entry is None:
        raise ForecastUnavailable(
            "REGISTRY_ENTRY_MISSING", model_id, model_id=model_id)
    kind = entry.get("kind")
    if kind not in ("neural", "classical", "ensemble"):
        raise ForecastUnavailable(
            "KIND_UNSUPPORTED",
            f"kind={kind!r}: no loadable forecast weights for this kind",
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
    pinned_art = (entry.get("checksums") or {}).get("artifact")

    if kind == "neural":
        if not TORCH_AVAILABLE:
            raise ForecastUnavailable(
                "TORCH_UNAVAILABLE", TORCH_WHY, model_id=model_id)
        provider = MLPForecastProvider.from_manifest(
            manifest_path, model_id=model_id, model_kind=kind,
            checksums={"manifest": pinned},
            git_sha_registered=entry.get("git_sha"),
            device=device, max_staleness_days=max_staleness_days)
        # cross-check: if the registry also pinned the .pt artifact hash it
        # must agree with the manifest's own checksum_sha256 (which
        # load_model already verified against the file contents).
        manifest_art = provider.manifest.get("checksum_sha256")
        if pinned_art and manifest_art and pinned_art != manifest_art:
            raise ForecastUnavailable(
                "ARTIFACT_CHECKSUM_MISMATCH",
                f"registry artifact sha {str(pinned_art)[:12]} != manifest "
                f"checksum_sha256 {str(manifest_art)[:12]}",
                model_id=model_id)
        return provider

    if kind == "ensemble":
        if model_id in _seen:
            raise ForecastUnavailable(
                "SCHEMA_INVALID",
                f"ensemble membership cycle via {model_id!r}",
                model_id=model_id)
        try:
            with open(manifest_path, encoding="utf-8") as f:
                em_manifest = json.load(f)
        except (OSError, ValueError) as e:
            raise ForecastUnavailable(
                "MODEL_INTEGRITY", f"{type(e).__name__}: {e}",
                model_id=model_id) from e
        members = (em_manifest or {}).get("members")
        if (not isinstance(members, list) or not members
                or not all(isinstance(m, str) and m.strip()
                           for m in members)):
            raise ForecastUnavailable(
                "SCHEMA_INVALID",
                "ensemble manifest needs a non-empty members list of "
                "model ids", model_id=model_id)
        seen = _seen | {model_id}
        loaded = [provider_from_registry(
            registry_path, m, device=device,
            max_staleness_days=max_staleness_days, _seen=seen)
            for m in members]  # member failures propagate, never averaged
        return EnsembleForecastProvider(
            loaded, members, model_id=model_id,
            checksums={"manifest": pinned},
            git_sha_registered=entry.get("git_sha"),
            horizon=(em_manifest or {}).get("horizon"),
            max_staleness_days=max_staleness_days)

    # classical: dispatch on the artifact payload's own kind field
    artifact_path = entry.get("artifact_path") or manifest_path
    if pinned_art and os.path.exists(artifact_path):
        art_actual = reg.sha256_file(artifact_path)
        if art_actual != pinned_art:
            raise ForecastUnavailable(
                "ARTIFACT_CHECKSUM_MISMATCH",
                f"artifact sha256 {art_actual[:12]} != pinned "
                f"{str(pinned_art)[:12]}", model_id=model_id)
    try:
        with open(artifact_path, encoding="utf-8") as f:
            doc = json.load(f) or {}
        # classical artifacts nest kind under "payload"; gbm artifacts
        # (save_gbm_direction) keep it at the top level.
        pk = (doc.get("payload") or {}).get("kind") or doc.get("kind")
    except (OSError, ValueError) as e:
        raise ForecastUnavailable(
            "MODEL_INTEGRITY", f"{type(e).__name__}: {e}",
            model_id=model_id) from e
    common = dict(model_id=model_id, model_kind=kind,
                  checksums={"manifest": pinned},
                  git_sha_registered=entry.get("git_sha"),
                  max_staleness_days=max_staleness_days)
    if pk == "multinomial_logistic":
        return ClassicalForecastProvider.from_artifact(
            artifact_path, **common)
    if pk == "lightgbm_multiclass_direction":
        return GBMForecastProvider.from_artifact(artifact_path, **common)
    raise ForecastUnavailable(
        "KIND_UNSUPPORTED", f"classical artifact payload kind {pk!r}",
        model_id=model_id)


# --------------------------------------------------------------------------- #
# provider cache with versioned refresh (loaded once outside scan loops)
# --------------------------------------------------------------------------- #

_PROVIDERS: Dict[Tuple[str, str, str], Any] = {}


def clear_cache() -> None:
    _PROVIDERS.clear()


def get_provider(registry_path: str, model_id: str, *,
                 device: str = "cpu",
                 max_staleness_days: int = DEFAULT_MAX_STALENESS_DAYS,
                 ) -> Any:
    """Cached, verified provider lookup with versioned refresh.

    The registry entry is re-read on every call (cheap JSON read); if the
    pinned manifest checksum changed since the provider was cached, the
    stale provider is dropped and the new artifact is verified and loaded.
    Failures are never cached.

    Note: for an ensemble the refresh check covers only the ensemble's own
    pinned manifest; a member re-registered without changing the ensemble
    manifest is not detected until :func:`clear_cache`.
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
    provider = provider_from_registry(
        registry_path, model_id, device=device,
        max_staleness_days=max_staleness_days)
    _PROVIDERS[key] = provider
    return provider
