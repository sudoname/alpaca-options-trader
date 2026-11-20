"""Oracle4 Phase 7a: champion/challenger model registry.

A small, file-based registry (JSON) that records every research model with
an immutable artifact checksum and the git source SHA it was built from,
plus an explicit human approval state.  Governance rules enforced here:

* every registered model starts as ``role="challenger"`` with
  ``approval_state="unapproved"`` — registration alone never changes what
  production would use;
* a model's artifact checksum is immutable: re-registering the same
  ``model_id`` with an identical checksum is an idempotent no-op, while a
  different checksum is rejected (publish a new id instead);
* ``approve_model`` requires a non-empty human ``approved_by`` string —
  there is no code path that approves a model automatically;
* ``set_champion`` only accepts an *approved* model, and demotes the prior
  champion to challenger.  Rollback is therefore just another
  ``set_champion`` call pointing at the previous id — a reference change,
  never a file mutation;
* every state transition is appended to an append-only ``history`` list.

No trading code imports this module; the shadow runner (Phase 7b) reads it
fail-open to stamp records with the configured model's identity.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import tempfile
from datetime import datetime, timezone
from typing import Any, Dict, Optional

SCHEMA_VERSION = 1
DEFAULT_REGISTRY_PATH = os.path.join("artifacts", "oracle4", "registry.json")

VALID_KINDS = ("classical", "neural", "valuation", "ensemble", "other")
APPROVAL_STATES = ("unapproved", "approved")
ROLES = ("champion", "challenger")


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #

def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def current_git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """Best-effort git HEAD SHA; None when git is unavailable."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True, text=True, timeout=10, cwd=cwd)
        if out.returncode == 0:
            sha = out.stdout.strip()
            return sha or None
    except Exception:
        pass
    return None


# --------------------------------------------------------------------------- #
# load / save
# --------------------------------------------------------------------------- #

def _empty_registry() -> Dict[str, Any]:
    return {
        "schema_version": SCHEMA_VERSION,
        "created_utc": _utc_now(),
        "champion_id": None,
        "models": {},
        "history": [],
    }


def load_registry(path: str = DEFAULT_REGISTRY_PATH) -> Dict[str, Any]:
    """Load the registry; a missing file yields a fresh empty registry."""
    if not os.path.exists(path):
        return _empty_registry()
    with open(path, "r", encoding="utf-8") as f:
        reg = json.load(f)
    if not isinstance(reg, dict) or "models" not in reg:
        raise ValueError(f"not a registry file: {path}")
    return reg


def save_registry(reg: Dict[str, Any], path: str = DEFAULT_REGISTRY_PATH) -> None:
    """Atomic write (tmp + replace) so a crash never truncates the registry."""
    d = os.path.dirname(os.path.abspath(path))
    os.makedirs(d, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(reg, f, indent=2, sort_keys=True)
        os.replace(tmp, path)
    finally:
        if os.path.exists(tmp):
            try:
                os.remove(tmp)
            except OSError:
                pass


def _append_history(reg: Dict[str, Any], action: str, model_id: str,
                    by: Optional[str] = None) -> None:
    reg.setdefault("history", []).append({
        "at": _utc_now(), "action": action, "model_id": model_id, "by": by,
    })


# --------------------------------------------------------------------------- #
# operations
# --------------------------------------------------------------------------- #

def register_model(path: str, model_id: str, *, kind: str,
                   artifact_path: Optional[str] = None,
                   manifest_path: Optional[str] = None,
                   metrics: Optional[Dict[str, Any]] = None,
                   notes: str = "",
                   git_sha: Optional[str] = None) -> Dict[str, Any]:
    """Register a model artifact under an immutable id+checksum.

    Returns the registry entry.  Re-registering the identical artifact is a
    no-op; a different artifact under the same id raises ValueError.
    """
    model_id = (model_id or "").strip()
    if not model_id:
        raise ValueError("model_id is required")
    if kind not in VALID_KINDS:
        raise ValueError(f"kind must be one of {VALID_KINDS}, got {kind!r}")
    if not artifact_path and not manifest_path:
        raise ValueError("at least one of artifact_path/manifest_path required")

    checksums: Dict[str, str] = {}
    for label, p in (("artifact", artifact_path), ("manifest", manifest_path)):
        if p:
            if not os.path.exists(p):
                raise FileNotFoundError(f"{label} not found: {p}")
            checksums[label] = sha256_file(p)

    reg = load_registry(path)
    existing = reg["models"].get(model_id)
    if existing is not None:
        if existing.get("checksums") == checksums:
            return existing  # idempotent re-register of the same artifact
        raise ValueError(
            f"model_id {model_id!r} already registered with different "
            f"checksums; model artifacts are immutable — register a new id")

    entry = {
        "model_id": model_id,
        "kind": kind,
        "artifact_path": artifact_path,
        "manifest_path": manifest_path,
        "checksums": checksums,
        "git_sha": git_sha if git_sha is not None else current_git_sha(),
        "registered_utc": _utc_now(),
        "approval_state": "unapproved",
        "approved_by": None,
        "approved_utc": None,
        "role": "challenger",
        "metrics": dict(metrics or {}),
        "notes": notes,
    }
    reg["models"][model_id] = entry
    _append_history(reg, "register", model_id)
    save_registry(reg, path)
    return entry


def approve_model(path: str, model_id: str, *, approved_by: str) -> Dict[str, Any]:
    """Mark a model approved.  Requires a non-empty human reviewer name."""
    approved_by = (approved_by or "").strip()
    if not approved_by:
        raise ValueError(
            "approved_by (a human reviewer) is required — "
            "automatic approval is not supported")
    reg = load_registry(path)
    entry = reg["models"].get(model_id)
    if entry is None:
        raise KeyError(f"unknown model_id: {model_id}")
    entry["approval_state"] = "approved"
    entry["approved_by"] = approved_by
    entry["approved_utc"] = _utc_now()
    _append_history(reg, "approve", model_id, by=approved_by)
    save_registry(reg, path)
    return entry


def set_champion(path: str, model_id: str, *, approved_by: str) -> Dict[str, Any]:
    """Point the champion reference at an *approved* model.

    The prior champion (if any) is demoted to challenger.  Calling this with
    the previous champion's id is the rollback procedure — no files move.
    """
    approved_by = (approved_by or "").strip()
    if not approved_by:
        raise ValueError("approved_by (a human reviewer) is required")
    reg = load_registry(path)
    entry = reg["models"].get(model_id)
    if entry is None:
        raise KeyError(f"unknown model_id: {model_id}")
    if entry.get("approval_state") != "approved":
        raise ValueError(
            f"model {model_id!r} is {entry.get('approval_state')!r}; "
            "only approved models may become champion")
    prior = reg.get("champion_id")
    if prior and prior in reg["models"] and prior != model_id:
        reg["models"][prior]["role"] = "challenger"
        _append_history(reg, "demote", prior, by=approved_by)
    entry["role"] = "champion"
    reg["champion_id"] = model_id
    _append_history(reg, "set_champion", model_id, by=approved_by)
    save_registry(reg, path)
    return entry


def get_model(path: str, model_id: str) -> Optional[Dict[str, Any]]:
    return load_registry(path)["models"].get(model_id)


def get_champion(path: str = DEFAULT_REGISTRY_PATH) -> Optional[Dict[str, Any]]:
    reg = load_registry(path)
    cid = reg.get("champion_id")
    return reg["models"].get(cid) if cid else None


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #

def main(argv: Optional[list] = None) -> int:
    ap = argparse.ArgumentParser(description="Oracle4 model registry")
    ap.add_argument("--registry", default=DEFAULT_REGISTRY_PATH)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register")
    p.add_argument("--model-id", required=True)
    p.add_argument("--kind", required=True, choices=VALID_KINDS)
    p.add_argument("--artifact")
    p.add_argument("--manifest")
    p.add_argument("--notes", default="")

    p = sub.add_parser("approve")
    p.add_argument("--model-id", required=True)
    p.add_argument("--approved-by", required=True)

    p = sub.add_parser("set-champion")
    p.add_argument("--model-id", required=True)
    p.add_argument("--approved-by", required=True)

    sub.add_parser("show")

    args = ap.parse_args(argv)
    if args.cmd == "register":
        entry = register_model(
            args.registry, args.model_id, kind=args.kind,
            artifact_path=args.artifact, manifest_path=args.manifest,
            notes=args.notes)
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif args.cmd == "approve":
        entry = approve_model(args.registry, args.model_id,
                              approved_by=args.approved_by)
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif args.cmd == "set-champion":
        entry = set_champion(args.registry, args.model_id,
                             approved_by=args.approved_by)
        print(json.dumps(entry, indent=2, sort_keys=True))
    elif args.cmd == "show":
        print(json.dumps(load_registry(args.registry), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
