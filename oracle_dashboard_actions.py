"""
Oracle Dashboard — opt-in training actions (research-only).

A SEPARATE module from ``oracle_dashboard`` so the default dashboard app
stays all-GET and free of any action logic (``test_only_get_routes`` and the
execution-token grep in ``test_oracle_dashboard.py`` keep passing verbatim).
``oracle_dashboard.create_app`` registers this blueprint ONLY when
``DASHBOARD_ENABLE_TRAINING`` is truthy.

What the one button does — and all it can ever do:
  snapshot ledger -> build dataset -> train challenger -> register UNAPPROVED.

Governance (hard constraints, pinned by test_oracle_dashboard_actions.py):
  * registration is the ceiling: the produced model lands in the registry as
    an unapproved challenger; approval and champion promotion remain human,
    CLI-only operations (``python -m oracle.ml.registry``) and this module
    never calls them;
  * no execution path: nothing here can open, size, price, gate, or close
    any real or paper position;
  * no user string ever reaches a command line: the request contributes only
    a model name validated against ``ALLOWED_MODELS``; every other argv token
    is a module name, a fixed flag, or a server-side config path.

Operational notes:
  * one job at a time (threading.Lock + 409 on concurrent start);
  * the ledger is snapshotted via the sqlite backup API first, so the live
    scheduler keeps writing the real ledger without contention;
  * job state is persisted to ``job.json`` / ``last_job.json`` — a Waitress
    restart (e.g. a deploy) reports the orphaned job as ``interrupted``,
    never as still ``running``.
"""

import json
import os
import sqlite3
import subprocess
import sys
import threading
import time

from flask import Blueprint, jsonify, request

from config_loader import ConfigLoader

_REPO_DIR = os.path.dirname(os.path.abspath(__file__))

# The ONLY model names the endpoint accepts; the registry "kind" for each.
ALLOWED_MODELS = ("mlp", "gbm", "logistic")
_MODEL_KIND = {"mlp": "neural", "gbm": "classical", "logistic": "classical"}

STEPS = ("snapshot", "dataset", "train", "register")

_LOG_TAIL_BYTES = 8192


class ActionsConfig:
    """Resolved training-action settings (shell env > .env > code default)."""

    def __init__(self, enabled, ledger_path, out_dir, registry_path, device):
        self.enabled = enabled
        self.ledger_path = ledger_path
        self.out_dir = out_dir
        self.registry_path = registry_path
        self.device = device

    @classmethod
    def from_env(cls, path: str = ".env", loader=None) -> "ActionsConfig":
        c = loader if loader is not None else ConfigLoader(path=path)
        return cls(
            enabled=c.get_bool("DASHBOARD_ENABLE_TRAINING", False),
            ledger_path=c.get_str("DASHBOARD_TRAINING_LEDGER", ""),
            out_dir=c.get_str("DASHBOARD_TRAINING_OUT_DIR",
                              os.path.join("artifacts", "oracle4",
                                           "dashboard_jobs")),
            registry_path=c.get_str("DASHBOARD_TRAINING_REGISTRY",
                                    os.path.join("artifacts", "oracle4",
                                                 "registry.json")),
            device=c.get_str("DASHBOARD_TRAINING_DEVICE", "cpu"),
        )


def _utc_ts() -> str:
    return time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())


def _utc_iso() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


class TrainingJobManager:
    """Runs at most one dataset->train->register pipeline at a time.

    All state transitions are persisted (``job.json`` in the job dir plus
    ``last_job.json`` at the out-dir root) so a process restart can report an
    orphaned job as ``interrupted`` instead of silently claiming ``running``.
    """

    def __init__(self, cfg: "ActionsConfig"):
        self.cfg = cfg
        self._lock = threading.Lock()
        self._thread = None
        self._job = self._recover_last_job()

    # -- persistence ------------------------------------------------------ #
    def _last_job_path(self) -> str:
        return os.path.join(self.cfg.out_dir, "last_job.json")

    def _recover_last_job(self):
        """Load the persisted last job; a 'running' job from a previous
        process is, by definition, dead — report it as interrupted."""
        try:
            with open(self._last_job_path(), "r", encoding="utf-8") as f:
                job = json.load(f)
        except Exception:
            return None
        if not isinstance(job, dict):
            return None
        if job.get("state") == "running":
            job["state"] = "interrupted"
            job["error"] = ("server restarted while the job was running; "
                            "the training subprocess did not survive")
            job["finished_at"] = _utc_iso()
            try:
                self._persist(job)
            except Exception:
                pass
        return job

    def _persist(self, job) -> None:
        os.makedirs(self.cfg.out_dir, exist_ok=True)
        blob = json.dumps(job, indent=2, sort_keys=True)
        for path in (os.path.join(job["job_dir"], "job.json"),
                     self._last_job_path()):
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(blob)
            except OSError:
                pass  # persistence is best-effort; status() still serves RAM

    # -- public API -------------------------------------------------------- #
    def start(self, model: str) -> str:
        """Validate + launch; returns job_id. Raises ValueError on a bad
        model name and RuntimeError when a job is already running."""
        if model not in ALLOWED_MODELS:
            raise ValueError(
                f"model must be one of {list(ALLOWED_MODELS)}")
        if not self.cfg.ledger_path:
            raise RuntimeError(
                "DASHBOARD_TRAINING_LEDGER is not configured on the server")
        if not os.path.exists(self.cfg.ledger_path):
            raise RuntimeError(
                f"research ledger not found: {self.cfg.ledger_path}")
        with self._lock:
            if self._job is not None and self._job.get("state") == "running":
                raise RuntimeError("a training job is already running")
            job_id = f"{model}-{_utc_ts()}"
            job_dir = os.path.join(self.cfg.out_dir, job_id)
            os.makedirs(job_dir, exist_ok=True)
            job = {
                "job_id": job_id,
                "model": model,
                "state": "running",
                "step": "snapshot",
                "steps": list(STEPS),
                "started_at": _utc_iso(),
                "finished_at": None,
                "returncode": None,
                "error": None,
                "job_dir": job_dir,
                "artifacts": {},
            }
            self._job = job
            self._persist(job)
            self._thread = threading.Thread(
                target=self._run, args=(job,), daemon=True,
                name=f"training-{job_id}")
            self._thread.start()
            return job_id

    def status(self) -> dict:
        with self._lock:
            job = self._job
            if job is None:
                return {"state": "idle", "job_id": None, "model": None,
                        "step": None, "steps": list(STEPS),
                        "started_at": None, "finished_at": None,
                        "returncode": None, "error": None,
                        "artifacts": {}, "log_tail": ""}
            out = {k: job.get(k) for k in (
                "state", "job_id", "model", "step", "steps", "started_at",
                "finished_at", "returncode", "error", "artifacts")}
        out["log_tail"] = self._read_log_tail(job)
        return out

    def _read_log_tail(self, job) -> str:
        log_path = os.path.join(job.get("job_dir", ""), "job.log")
        try:
            size = os.path.getsize(log_path)
            with open(log_path, "rb") as f:
                if size > _LOG_TAIL_BYTES:
                    f.seek(size - _LOG_TAIL_BYTES)
                data = f.read()
            return data.decode("utf-8", errors="replace")
        except OSError:
            return ""

    # -- pipeline ---------------------------------------------------------- #
    def _set_step(self, job, step: str) -> None:
        with self._lock:
            job["step"] = step
            self._persist(job)

    def _finish(self, job, *, state: str, returncode=None,
                error=None) -> None:
        with self._lock:
            job["state"] = state
            job["returncode"] = returncode
            job["error"] = error
            job["finished_at"] = _utc_iso()
            self._persist(job)

    def _run_subprocess(self, job, argv, log) -> int:
        log.write(f"\n$ {' '.join(argv)}\n")
        log.flush()
        proc = subprocess.run(
            argv, stdout=log, stderr=subprocess.STDOUT, cwd=_REPO_DIR)
        return proc.returncode

    def _snapshot_ledger(self, job) -> str:
        """Consistent copy of the research ledger via the sqlite backup API
        (the live scheduler keeps writing the original untouched)."""
        snap = os.path.join(job["job_dir"], "ledger_snapshot.sqlite")
        src = sqlite3.connect(
            f"file:{self.cfg.ledger_path}?mode=ro", uri=True)
        try:
            dst = sqlite3.connect(snap)
            try:
                with dst:
                    src.backup(dst)
            finally:
                dst.close()
        finally:
            src.close()
        return snap

    def _register(self, job) -> dict:
        """Register the freshly trained model as an UNAPPROVED challenger.

        This is the pipeline ceiling by design: approval and champion
        promotion are separate human CLI steps this module never performs.
        """
        from oracle.ml import registry

        job_dir = job["job_dir"]
        report = os.path.join(job_dir, "report.json")
        stem = os.path.join(job_dir, "report")
        artifact = None
        for cand in (f"{stem}_model.pt", f"{stem}_gbm.json"):
            if os.path.exists(cand):
                artifact = cand
                break
        manifest = f"{stem}_manifest.json"
        if not os.path.exists(manifest):
            manifest = report  # logistic/gbm: the train report is the record

        model_id = f"{job['model']}_60m_dash_{job['job_id'].split('-')[-1]}"
        entry = registry.register_model(
            self.cfg.registry_path, model_id,
            kind=_MODEL_KIND[job["model"]],
            artifact_path=artifact,
            manifest_path=manifest,
            notes=(f"dashboard-triggered training job {job['job_id']}; "
                   "UNAPPROVED challenger — approval/champion promotion "
                   "are human CLI-only"),
        )
        return {"model_id": model_id,
                "approval_state": entry.get("approval_state"),
                "role": entry.get("role")}

    def _run(self, job) -> None:
        job_dir = job["job_dir"]
        log_path = os.path.join(job_dir, "job.log")
        try:
            with open(log_path, "a", encoding="utf-8") as log:
                log.write(f"[{_utc_iso()}] job {job['job_id']} started "
                          f"(model={job['model']}, "
                          f"device={self.cfg.device})\n")

                # 1. snapshot (in-process sqlite backup; no argv involved)
                self._set_step(job, "snapshot")
                snap = self._snapshot_ledger(job)
                log.write(f"[{_utc_iso()}] ledger snapshot -> {snap}\n")
                log.flush()

                dataset = os.path.join(job_dir, "dataset.sqlite")
                ds_manifest = os.path.join(job_dir, "dataset_manifest.json")
                report = os.path.join(job_dir, "report.json")

                # 2. dataset build — argv is module names, fixed flags, and
                #    server-side paths only (model name never appears here).
                self._set_step(job, "dataset")
                rc = self._run_subprocess(job, [
                    sys.executable, "-m", "oracle.ml.dataset",
                    "--ledger", snap,
                    "--output", dataset,
                    "--manifest", ds_manifest,
                ], log)
                if rc != 0:
                    self._finish(job, state="failed", returncode=rc,
                                 error="dataset build failed")
                    return

                # 3. train — the only request-derived token is the model
                #    name, already validated against ALLOWED_MODELS.
                self._set_step(job, "train")
                rc = self._run_subprocess(job, [
                    sys.executable, "-m", "oracle.ml.train",
                    "--model", job["model"],
                    "--dataset", dataset,
                    "--horizon", "60m",
                    "--out", report,
                    "--device", self.cfg.device,
                    "--seed", "42",
                ], log)
                if rc != 0:
                    self._finish(job, state="failed", returncode=rc,
                                 error="training failed")
                    return

                # 4. register as UNAPPROVED challenger (in-process)
                self._set_step(job, "register")
                reg = self._register(job)
                with self._lock:
                    job["artifacts"] = {
                        "report": report,
                        "dataset_manifest": ds_manifest,
                        "registered": reg,
                    }
                log.write(f"[{_utc_iso()}] registered "
                          f"{reg['model_id']} as "
                          f"{reg['approval_state']}/{reg['role']}\n")
                self._finish(job, state="done", returncode=0)
        except Exception as e:
            self._finish(job, state="failed", error=f"{type(e).__name__}: {e}")
        finally:
            # the ledger snapshot is large and single-use
            try:
                snap = os.path.join(job_dir, "ledger_snapshot.sqlite")
                if os.path.exists(snap):
                    os.remove(snap)
            except OSError:
                pass


_manager = None
_manager_lock = threading.Lock()


def get_job_manager(cfg: "ActionsConfig") -> "TrainingJobManager":
    """Process-wide singleton so concurrent requests share the one-job lock."""
    global _manager
    with _manager_lock:
        if _manager is None:
            _manager = TrainingJobManager(cfg)
        return _manager


def create_actions_blueprint(cfg: "ActionsConfig" = None,
                             job_manager=None) -> Blueprint:
    """Build the /api/actions blueprint. ``job_manager`` is injectable so
    tests run with a fake manager (no real subprocesses)."""
    cfg = cfg if cfg is not None else ActionsConfig.from_env()
    mgr = job_manager if job_manager is not None else get_job_manager(cfg)

    bp = Blueprint("actions", __name__, url_prefix="/api/actions")

    @bp.route("/train", methods=["POST"])
    def train_start():
        body = request.get_json(silent=True) or {}
        model = body.get("model")
        if not isinstance(model, str) or model not in ALLOWED_MODELS:
            return jsonify({
                "error": "model must be one of "
                         f"{list(ALLOWED_MODELS)}"}), 400
        try:
            job_id = mgr.start(model)
        except ValueError as e:
            return jsonify({"error": str(e)}), 400
        except RuntimeError as e:
            msg = str(e)
            code = 409 if "already running" in msg else 503
            return jsonify({"error": msg}), code
        return jsonify({"job_id": job_id, "state": "running"}), 202

    @bp.route("/train/status", methods=["GET"])
    def train_status():
        # Deliberately NOT routed through the dashboard _TTLCache: a stale
        # "running"/"idle" here would mislead the operator.
        out = mgr.status()
        out["enabled"] = True
        out["allowed_models"] = list(ALLOWED_MODELS)
        return jsonify(out)

    return bp
