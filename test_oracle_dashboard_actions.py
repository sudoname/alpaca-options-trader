"""
Offline tests for the opt-in dashboard training actions.

No creds, no network, no broker, no real training subprocess: route tests
drive Flask through ``app.test_client()`` with a FAKE job manager injected
via ``create_actions_blueprint(cfg, job_manager=...)``; the one pipeline
test patches the subprocess runner and the register step on a real
``TrainingJobManager``. The contract pinned here:

  1. OPT-IN ONLY. With the default config the dashboard app has ZERO
     non-GET routes and no ``/api/actions/*`` rule at all.
  2. SCOPED WRITES. With the flag on, POST exists ONLY under
     ``/api/actions/``; every other route stays GET.
  3. WHITELIST. The request contributes only a model name validated
     against ``ALLOWED_MODELS``; bad/missing -> 400, valid -> 202+job_id.
  4. ONE JOB AT A TIME. A concurrent start -> 409.
  5. AUTH COVERS ACTIONS. Configured basic auth 401s the POST without
     creds (the dashboard's before_request hook spans /api/actions/*).
  6. GOVERNANCE BY CONSTRUCTION. The actions module source contains no
     execution tokens and never calls approve/set-champion — the pipeline
     ceiling is an UNAPPROVED challenger registration.
  7. NO INJECTION. Every subprocess argv token is sys.executable, a fixed
     flag/module name, a job-dir path, or the validated model enum.
  8. RESTART SAFETY. A persisted 'running' job from a dead process is
     reported as 'interrupted', never as still running.
"""

import json
import os
import sqlite3
import sys
import tempfile
import unittest

import oracle_dashboard_actions as oda
from oracle_dashboard import DashboardConfig, create_app
from oracle_dashboard_actions import (ActionsConfig, ALLOWED_MODELS, STEPS,
                                      TrainingJobManager,
                                      create_actions_blueprint)


def _cfg(user="", password="", ttl=0, training=False):
    return DashboardConfig(host="127.0.0.1", port=0, cache_ttl=ttl,
                           basic_auth_user=user, basic_auth_pass=password,
                           training_enabled=training)


def _actions_cfg(tmp):
    return ActionsConfig(enabled=True,
                         ledger_path=os.path.join(tmp, "ledger.sqlite"),
                         out_dir=os.path.join(tmp, "jobs"),
                         registry_path=os.path.join(tmp, "registry.json"),
                         device="cpu")


class FakeManager:
    """Route-level stand-in: validates like the real manager, runs nothing."""

    def __init__(self, running=False):
        self.running = running
        self.started = []

    def start(self, model):
        if model not in ALLOWED_MODELS:
            raise ValueError(f"model must be one of {list(ALLOWED_MODELS)}")
        if self.running:
            raise RuntimeError("a training job is already running")
        self.running = True
        self.started.append(model)
        return f"{model}-20260101T000000Z"

    def status(self):
        return {"state": "running" if self.running else "idle",
                "job_id": self.started[-1] if self.started else None,
                "model": None, "step": None, "steps": list(STEPS),
                "started_at": None, "finished_at": None, "returncode": None,
                "error": None, "artifacts": {}, "log_tail": ""}


def _app(user="", password="", manager=None):
    """Dashboard app + actions blueprint with an injected fake manager.

    Building the blueprint directly (instead of flipping the env flag)
    keeps the process-wide ``get_job_manager`` singleton untouched and
    guarantees no real subprocess can ever start from a route test.
    """
    app = create_app(_cfg(user=user, password=password))
    app.config["EXPLAIN_CTX_BUILDER"] = lambda s: {}
    app.config["REGIME_CTX_BUILDER"] = lambda: {}
    app.config["SENTIMENT_REPORT"] = lambda: {"verdict": "INSUFFICIENT_DATA"}
    mgr = manager if manager is not None else FakeManager()
    with tempfile.TemporaryDirectory() as tmp:
        app.register_blueprint(
            create_actions_blueprint(_actions_cfg(tmp), job_manager=mgr))
    return app, mgr


class TestOptInOnly(unittest.TestCase):
    def test_default_config_has_no_actions_routes(self):
        app = create_app(_cfg())
        for rule in app.url_map.iter_rules():
            self.assertNotIn("POST", rule.methods, msg=str(rule))
            self.assertNotIn("PUT", rule.methods, msg=str(rule))
            self.assertNotIn("DELETE", rule.methods, msg=str(rule))
            self.assertFalse(str(rule).startswith("/api/actions"),
                             msg=str(rule))

    def test_flag_on_post_exists_only_under_actions_prefix(self):
        singleton_before = oda._manager
        oda._manager = FakeManager()  # keep create_app from building a real one
        try:
            app = create_app(_cfg(training=True))
        finally:
            oda._manager = singleton_before
        post_rules = [str(r) for r in app.url_map.iter_rules()
                      if "POST" in r.methods]
        self.assertEqual(post_rules, ["/api/actions/train"])
        for rule in app.url_map.iter_rules():
            if not str(rule).startswith("/api/actions"):
                self.assertNotIn("POST", rule.methods, msg=str(rule))
                self.assertNotIn("PUT", rule.methods, msg=str(rule))
                self.assertNotIn("DELETE", rule.methods, msg=str(rule))


class TestTrainEndpoint(unittest.TestCase):
    def setUp(self):
        self.app, self.mgr = _app()
        self.client = self.app.test_client()

    def test_missing_model_400(self):
        r = self.client.post("/api/actions/train", json={})
        self.assertEqual(r.status_code, 400)
        self.assertIn("model must be one of", r.get_json()["error"])

    def test_bad_model_400(self):
        for bad in ("rm -rf /", "mlp; ls", "naive", "", 42, None, ["mlp"]):
            r = self.client.post("/api/actions/train", json={"model": bad})
            self.assertEqual(r.status_code, 400, msg=repr(bad))
        self.assertEqual(self.mgr.started, [])

    def test_no_body_400_not_500(self):
        r = self.client.post("/api/actions/train", data="not json",
                             content_type="text/plain")
        self.assertEqual(r.status_code, 400)

    def test_valid_model_202_with_job_id(self):
        r = self.client.post("/api/actions/train", json={"model": "mlp"})
        self.assertEqual(r.status_code, 202)
        body = r.get_json()
        self.assertTrue(body["job_id"].startswith("mlp-"))
        self.assertEqual(body["state"], "running")
        self.assertEqual(self.mgr.started, ["mlp"])

    def test_concurrent_start_409(self):
        self.assertEqual(
            self.client.post("/api/actions/train",
                             json={"model": "mlp"}).status_code, 202)
        r = self.client.post("/api/actions/train", json={"model": "gbm"})
        self.assertEqual(r.status_code, 409)
        self.assertIn("already running", r.get_json()["error"])

    def test_unconfigured_ledger_503(self):
        class Boom(FakeManager):
            def start(self, model):
                raise RuntimeError(
                    "DASHBOARD_TRAINING_LEDGER is not configured on the server")
        app, _ = _app(manager=Boom())
        r = app.test_client().post("/api/actions/train", json={"model": "mlp"})
        self.assertEqual(r.status_code, 503)


class TestStatusEndpoint(unittest.TestCase):
    def test_status_shape_and_freshness(self):
        app, mgr = _app()
        client = app.test_client()
        body = client.get("/api/actions/train/status").get_json()
        for key in ("state", "job_id", "model", "step", "steps",
                    "started_at", "finished_at", "returncode", "error",
                    "artifacts", "log_tail", "enabled", "allowed_models"):
            self.assertIn(key, body, msg=key)
        self.assertEqual(body["state"], "idle")
        self.assertTrue(body["enabled"])
        self.assertEqual(body["allowed_models"], list(ALLOWED_MODELS))
        self.assertEqual(body["steps"], list(STEPS))
        # not TTL-cached: a state change shows up on the very next GET
        client.post("/api/actions/train", json={"model": "logistic"})
        body = client.get("/api/actions/train/status").get_json()
        self.assertEqual(body["state"], "running")


class TestAuthCoversActions(unittest.TestCase):
    def test_post_401_without_creds_202_with(self):
        import base64
        app, _ = _app(user="u", password="p")
        client = app.test_client()
        r = client.post("/api/actions/train", json={"model": "mlp"})
        self.assertEqual(r.status_code, 401)
        self.assertIn("WWW-Authenticate", r.headers)
        tok = base64.b64encode(b"u:p").decode()
        r = client.post("/api/actions/train", json={"model": "mlp"},
                        headers={"Authorization": f"Basic {tok}"})
        self.assertEqual(r.status_code, 202)

    def test_status_401_without_creds(self):
        app, _ = _app(user="u", password="p")
        r = app.test_client().get("/api/actions/train/status")
        self.assertEqual(r.status_code, 401)


class TestGovernanceByConstruction(unittest.TestCase):
    def test_no_execution_or_promotion_symbols_in_source(self):
        import inspect
        src = inspect.getsource(oda)
        forbidden = (
            # execution call-forms (same list the dashboard test pins)
            "place_option_order(", "submit_order(", "execute_trade(",
            "open_position(", "close_position(", "record_outcome(",
            # promotion call-forms: registration must be the ceiling
            "approve_model(", "set_champion(",
        )
        for token in forbidden:
            self.assertNotIn(token, src, msg=f"forbidden token {token!r}")

    def test_whitelist_is_the_enum_the_pipeline_uses(self):
        self.assertEqual(set(ALLOWED_MODELS), {"mlp", "gbm", "logistic"})
        self.assertEqual(set(oda._MODEL_KIND), set(ALLOWED_MODELS))


class TestPipelineArgvAndState(unittest.TestCase):
    """Real TrainingJobManager; subprocess runner + register step patched."""

    def _run_to_completion(self, tmp, model="mlp"):
        cfg = _actions_cfg(tmp)
        con = sqlite3.connect(cfg.ledger_path)
        con.execute("CREATE TABLE t (x INTEGER)")
        con.commit()
        con.close()
        mgr = TrainingJobManager(cfg)
        argvs = []

        def fake_subprocess(job, argv, log):
            argvs.append(list(argv))
            log.write("ok\n")
            return 0

        mgr._run_subprocess = fake_subprocess
        mgr._register = lambda job: {"model_id": "x", "approval_state":
                                     "unapproved", "role": "challenger"}
        job_id = mgr.start(model)
        mgr._thread.join(timeout=30)
        self.assertFalse(mgr._thread.is_alive())
        return mgr, job_id, argvs

    def test_argv_tokens_are_whitelisted(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, job_id, argvs = self._run_to_completion(tmp, model="mlp")
            self.assertEqual(len(argvs), 2)  # dataset, then train
            job_dir = os.path.join(mgr.cfg.out_dir, job_id)
            fixed = {sys.executable, "-m",
                     "oracle.ml.dataset", "oracle.ml.train",
                     "--ledger", "--output", "--manifest",
                     "--model", "--dataset", "--horizon", "60m",
                     "--out", "--device", "cpu", "--seed", "42"}
            for argv in argvs:
                for token in argv:
                    ok = (token in fixed or token in ALLOWED_MODELS
                          or os.path.abspath(token).startswith(
                              os.path.abspath(job_dir)))
                    self.assertTrue(ok, msg=f"unexpected argv token {token!r}"
                                            f" in {argv}")
            # the only request-derived token is the validated model name
            self.assertIn("mlp", argvs[1])

    def test_job_reaches_done_and_persists(self):
        with tempfile.TemporaryDirectory() as tmp:
            mgr, job_id, _ = self._run_to_completion(tmp)
            st = mgr.status()
            self.assertEqual(st["state"], "done")
            self.assertEqual(st["returncode"], 0)
            self.assertEqual(st["artifacts"]["registered"]["approval_state"],
                             "unapproved")
            self.assertIn("ok", st["log_tail"])
            with open(os.path.join(mgr.cfg.out_dir, "last_job.json"),
                      encoding="utf-8") as f:
                self.assertEqual(json.load(f)["state"], "done")
            # snapshot is cleaned up after the run
            self.assertFalse(os.path.exists(os.path.join(
                mgr.cfg.out_dir, job_id, "ledger_snapshot.sqlite")))

    def test_failed_subprocess_marks_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _actions_cfg(tmp)
            con = sqlite3.connect(cfg.ledger_path)
            con.execute("CREATE TABLE t (x INTEGER)")
            con.commit()
            con.close()
            mgr = TrainingJobManager(cfg)
            mgr._run_subprocess = lambda job, argv, log: 1
            mgr.start("gbm")
            mgr._thread.join(timeout=30)
            st = mgr.status()
            self.assertEqual(st["state"], "failed")
            self.assertEqual(st["error"], "dataset build failed")

    def test_missing_ledger_raises_before_any_job(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _actions_cfg(tmp)  # ledger file never created
            mgr = TrainingJobManager(cfg)
            with self.assertRaises(RuntimeError):
                mgr.start("mlp")
            self.assertEqual(mgr.status()["state"], "idle")


class TestRestartSafety(unittest.TestCase):
    def test_persisted_running_job_recovers_as_interrupted(self):
        with tempfile.TemporaryDirectory() as tmp:
            cfg = _actions_cfg(tmp)
            job_dir = os.path.join(cfg.out_dir, "mlp-20260101T000000Z")
            os.makedirs(job_dir)
            job = {"job_id": "mlp-20260101T000000Z", "model": "mlp",
                   "state": "running", "step": "train",
                   "steps": list(STEPS), "started_at": "2026-01-01T00:00:00Z",
                   "finished_at": None, "returncode": None, "error": None,
                   "job_dir": job_dir, "artifacts": {}}
            with open(os.path.join(cfg.out_dir, "last_job.json"), "w",
                      encoding="utf-8") as f:
                json.dump(job, f)
            mgr = TrainingJobManager(cfg)  # fresh process
            st = mgr.status()
            self.assertEqual(st["state"], "interrupted")
            self.assertIn("restarted", st["error"])
            # an interrupted job does not block a new start
            con = sqlite3.connect(cfg.ledger_path)
            con.execute("CREATE TABLE t (x INTEGER)")
            con.commit()
            con.close()
            mgr._run_subprocess = lambda job, argv, log: 0
            mgr._register = lambda job: {"model_id": "x", "approval_state":
                                         "unapproved", "role": "challenger"}
            mgr.start("mlp")
            mgr._thread.join(timeout=30)
            self.assertEqual(mgr.status()["state"], "done")


if __name__ == "__main__":
    unittest.main()
