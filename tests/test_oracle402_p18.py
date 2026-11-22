"""Oracle 4.0.2 P1-8 tests: asynchronous research worker.

Covers:
  * tasks run and metrics count them;
  * payload snapshot immutability (deepcopy at submit time);
  * bounded queue backpressure: submit never blocks, overflow is dropped
    and counted while the production side continues instantly;
  * failure isolation: a raising task never kills the worker;
  * soft-timeout overrun detection + live stalled flag;
  * graceful shutdown; submit-after-stop rejected;
  * uncopyable payload rejected (never shared by reference);
  * atomic status file written for the dashboard;
  * env-configured process singleton + test reset hook;
  * dashboard /api/research-worker endpoint fail-open behavior.

All synchronization is Event-based with bounded polling (no bare sleeps)
to avoid Windows timing flakes.
"""
import json
import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from oracle.ml.research_worker import (  # noqa: E402
    ResearchWorker, _reset_global_for_tests, global_worker, submit_research)

WAIT_S = 10.0


def _poll(pred, timeout=WAIT_S, interval=0.01):
    """Poll ``pred`` until truthy or timeout; returns final value."""
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        v = pred()
        if v:
            return v
        time.sleep(interval)
    return pred()


class TestWorkerBasics(unittest.TestCase):
    def setUp(self):
        self.w = ResearchWorker(maxsize=4, task_timeout_s=60.0,
                                status_path=None).start()

    def tearDown(self):
        self.assertTrue(self.w.stop(timeout=5.0))

    def test_runs_submitted_task(self):
        done = threading.Event()
        seen = {}

        def task(p):
            seen["payload"] = p
            done.set()

        self.assertTrue(self.w.submit("t1", task, {"a": 1}))
        self.assertTrue(done.wait(WAIT_S))
        self.assertEqual(seen["payload"], {"a": 1})
        m = _poll(lambda: (self.w.metrics()
                           if self.w.metrics()["completed"] == 1 else None))
        self.assertIsNotNone(m)
        self.assertEqual(m["submitted"], 1)
        self.assertEqual(m["completed"], 1)
        self.assertEqual(m["failed"], 0)
        self.assertEqual(m["last_task"], "t1")
        self.assertIsNotNone(m["last_duration_s"])
        self.assertTrue(m["alive"])

    def test_payload_snapshot_immutable(self):
        gate = threading.Event()
        seen = {}
        done = threading.Event()

        def task(p):
            gate.wait(WAIT_S)
            seen["payload"] = p
            done.set()

        payload = {"candidates": [{"sym": "SPY", "score": 1.0}]}
        self.assertTrue(self.w.submit("snap", task, payload))
        # mutate AFTER submit, BEFORE the task reads its copy
        payload["candidates"][0]["score"] = -999.0
        payload["candidates"].append({"sym": "EVIL"})
        gate.set()
        self.assertTrue(done.wait(WAIT_S))
        self.assertEqual(seen["payload"],
                         {"candidates": [{"sym": "SPY", "score": 1.0}]})

    def test_failure_isolation(self):
        done = threading.Event()

        def bad(_p):
            raise ValueError("boom")

        def good(_p):
            done.set()

        self.assertTrue(self.w.submit("bad", bad, None))
        self.assertTrue(self.w.submit("good", good, None))
        self.assertTrue(done.wait(WAIT_S))
        m = _poll(lambda: (self.w.metrics()
                           if self.w.metrics()["completed"] >= 1 else None))
        self.assertEqual(m["failed"], 1)
        self.assertIn("boom", m["last_error"])
        self.assertTrue(m["alive"])

    def test_uncopyable_payload_rejected(self):
        ran = threading.Event()
        # a thread lock is not deep-copyable
        ok = self.w.submit("nocopy", lambda p: ran.set(),
                           {"lock": threading.Lock()})
        self.assertFalse(ok)
        m = self.w.metrics()
        self.assertEqual(m["dropped_snapshot"], 1)
        self.assertEqual(m["submitted"], 0)
        self.assertIn("not deep-copyable", m["last_error"])
        self.assertFalse(ran.is_set())


class TestBackpressureAndTimeout(unittest.TestCase):
    def test_bounded_queue_drops_and_submit_is_nonblocking(self):
        w = ResearchWorker(maxsize=2, status_path=None).start()
        try:
            started = threading.Event()
            release = threading.Event()

            def blocker(_p):
                started.set()
                release.wait(WAIT_S)

            self.assertTrue(w.submit("blocker", blocker, None))
            self.assertTrue(started.wait(WAIT_S))  # worker busy, queue empty
            # fill the queue
            self.assertTrue(w.submit("q1", lambda p: None, None))
            self.assertTrue(w.submit("q2", lambda p: None, None))
            # overflow must return instantly with False (production side
            # never blocks on research)
            t0 = time.monotonic()
            self.assertFalse(w.submit("overflow", lambda p: None, None))
            self.assertLess(time.monotonic() - t0, 1.0)
            m = w.metrics()
            self.assertEqual(m["dropped_queue_full"], 1)
            self.assertEqual(m["submitted"], 3)
            self.assertEqual(m["queue_maxsize"], 2)
            release.set()
            m = _poll(lambda: (w.metrics()
                               if w.metrics()["completed"] == 3 else None))
            self.assertIsNotNone(m, "queued tasks should drain after release")
        finally:
            release.set()
            self.assertTrue(w.stop(timeout=5.0))

    def test_overrun_detected_and_stalled_flag(self):
        w = ResearchWorker(maxsize=2, task_timeout_s=0.05,
                           status_path=None).start()
        try:
            started = threading.Event()
            release = threading.Event()

            def slow(_p):
                started.set()
                release.wait(WAIT_S)

            self.assertTrue(w.submit("slow", slow, None))
            self.assertTrue(started.wait(WAIT_S))
            m = _poll(lambda: (w.metrics()
                               if w.metrics()["stalled"] else None))
            self.assertIsNotNone(m, "in-flight overrun should flag stalled")
            self.assertEqual(m["current_task"], "slow")
            self.assertGreater(m["current_task_age_s"], 0.05)
            release.set()
            m = _poll(lambda: (w.metrics()
                               if w.metrics()["completed"] == 1 else None))
            self.assertEqual(m["overruns"], 1)
            self.assertFalse(m["stalled"])
            self.assertIsNone(m["current_task"])
        finally:
            release.set()
            self.assertTrue(w.stop(timeout=5.0))


class TestShutdown(unittest.TestCase):
    def test_graceful_stop_drains_queue(self):
        w = ResearchWorker(maxsize=8, status_path=None).start()
        ran = []
        for i in range(3):
            self.assertTrue(w.submit(f"t{i}",
                                     lambda p: ran.append(p), i))
        self.assertTrue(w.stop(timeout=5.0))
        self.assertEqual(sorted(ran), [0, 1, 2])
        m = w.metrics()
        self.assertEqual(m["completed"], 3)
        self.assertFalse(m["alive"])
        self.assertTrue(m["stopping"])

    def test_submit_after_stop_rejected(self):
        w = ResearchWorker(status_path=None).start()
        self.assertTrue(w.stop(timeout=5.0))
        self.assertFalse(w.submit("late", lambda p: None, None))
        self.assertEqual(w.metrics()["rejected_stopped"], 1)

    def test_stop_before_start_is_safe(self):
        w = ResearchWorker(status_path=None)
        self.assertTrue(w.stop(timeout=1.0))
        self.assertFalse(w.submit("never", lambda p: None, None))


class TestStatusFile(unittest.TestCase):
    def test_status_written_atomically_after_task(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "sub", "status.json")
            w = ResearchWorker(maxsize=4, status_path=path).start()
            try:
                done = threading.Event()
                self.assertTrue(w.submit("s1", lambda p: done.set(), None))
                self.assertTrue(done.wait(WAIT_S))
                self.assertTrue(_poll(lambda: os.path.isfile(path)))
                with open(path, "r", encoding="utf-8") as f:
                    doc = json.load(f)
                self.assertEqual(doc["stage"],
                                 "oracle4.0.2-research-worker")
                self.assertEqual(doc["completed"], 1)
                self.assertEqual(doc["last_task"], "s1")
                self.assertIn("generated_at_utc", doc)
                self.assertIn("queue_depth", doc)
                # no tmp file left behind
                self.assertFalse(os.path.exists(path + ".tmp"))
            finally:
                self.assertTrue(w.stop(timeout=5.0))

    def test_status_path_none_writes_nothing(self):
        w = ResearchWorker(status_path=None).start()
        try:
            done = threading.Event()
            self.assertTrue(w.submit("x", lambda p: done.set(), None))
            self.assertTrue(done.wait(WAIT_S))
        finally:
            self.assertTrue(w.stop(timeout=5.0))


class TestGlobalSingleton(unittest.TestCase):
    def setUp(self):
        _reset_global_for_tests()
        self._env = {}
        for k in ("ORACLE4_RESEARCH_QUEUE_MAX",
                  "ORACLE4_RESEARCH_TASK_TIMEOUT_S",
                  "ORACLE4_RESEARCH_STATUS"):
            self._env[k] = os.environ.pop(k, None)

    def tearDown(self):
        _reset_global_for_tests()
        for k, v in self._env.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v

    def test_env_configuration(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rw.json")
            os.environ["ORACLE4_RESEARCH_QUEUE_MAX"] = "3"
            os.environ["ORACLE4_RESEARCH_TASK_TIMEOUT_S"] = "7.5"
            os.environ["ORACLE4_RESEARCH_STATUS"] = path
            w = global_worker()
            self.assertIs(global_worker(), w)  # singleton
            m = w.metrics()
            self.assertEqual(m["queue_maxsize"], 3)
            self.assertEqual(m["task_timeout_s"], 7.5)
            self.assertEqual(w.status_path, path)
            done = threading.Event()
            self.assertTrue(
                submit_research("g1", lambda p: done.set(), None))
            self.assertTrue(done.wait(WAIT_S))
            self.assertTrue(_poll(lambda: os.path.isfile(path)))
            _reset_global_for_tests()

    def test_bad_env_falls_back_to_defaults(self):
        os.environ["ORACLE4_RESEARCH_QUEUE_MAX"] = "not-a-number"
        os.environ["ORACLE4_RESEARCH_TASK_TIMEOUT_S"] = ""
        w = global_worker()
        m = w.metrics()
        self.assertEqual(m["queue_maxsize"], 8)
        self.assertEqual(m["task_timeout_s"], 120.0)

    def test_submit_research_never_raises(self):
        # even a hostile payload just returns False
        self.assertFalse(submit_research(
            "hostile", lambda p: None, {"lock": threading.Lock()}))


class TestDashboardEndpoint(unittest.TestCase):
    """/api/research-worker: fail-open file read, GET-only app invariant."""

    def _app(self):
        import oracle_dashboard as od
        cfg = od.DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                 basic_auth_user="", basic_auth_pass="")
        return od.create_app(cfg)

    def setUp(self):
        self._old = os.environ.pop("ORACLE4_RESEARCH_STATUS", None)
        self._cwd = os.getcwd()

    def tearDown(self):
        os.chdir(self._cwd)
        if self._old is None:
            os.environ.pop("ORACLE4_RESEARCH_STATUS", None)
        else:
            os.environ["ORACLE4_RESEARCH_STATUS"] = self._old

    def test_missing_status_file_fail_open(self):
        with tempfile.TemporaryDirectory() as td:
            os.environ["ORACLE4_RESEARCH_STATUS"] = os.path.join(
                td, "absent.json")
            app = self._app()
            with app.test_client() as c:
                r = c.get("/api/research-worker")
                self.assertEqual(r.status_code, 200)
                doc = r.get_json()
                self.assertEqual(doc["verdict"], "INSUFFICIENT_DATA")
                self.assertIn("hint", doc)

    def test_reads_status_file(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rw.json")
            with open(path, "w", encoding="utf-8") as f:
                json.dump({"stage": "oracle4.0.2-research-worker",
                           "completed": 5, "failed": 0,
                           "queue_depth": 0, "alive": True}, f)
            os.environ["ORACLE4_RESEARCH_STATUS"] = path
            app = self._app()
            with app.test_client() as c:
                r = c.get("/api/research-worker")
                self.assertEqual(r.status_code, 200)
                doc = r.get_json()
                self.assertEqual(doc["verdict"], "OK")
                self.assertEqual(doc["completed"], 5)
                self.assertTrue(doc["alive"])

    def test_corrupt_status_file_fail_open(self):
        with tempfile.TemporaryDirectory() as td:
            path = os.path.join(td, "rw.json")
            with open(path, "w", encoding="utf-8") as f:
                f.write("{not json")
            os.environ["ORACLE4_RESEARCH_STATUS"] = path
            app = self._app()
            with app.test_client() as c:
                r = c.get("/api/research-worker")
                self.assertEqual(r.status_code, 200)
                self.assertEqual(r.get_json()["verdict"], "ERROR")

    def test_route_is_get_only(self):
        app = self._app()
        for rule in app.url_map.iter_rules():
            if str(rule) == "/api/research-worker":
                methods = rule.methods - {"HEAD", "OPTIONS"}
                self.assertEqual(methods, {"GET"})
                break
        else:
            self.fail("/api/research-worker route not registered")


class TestSchedulerGate(unittest.TestCase):
    """The async gate must default OFF so production behavior is
    byte-for-byte unchanged without ENABLE_ORACLE4_ASYNC_RESEARCH."""

    def test_flag_default_off(self):
        os.environ.pop("ENABLE_ORACLE4_ASYNC_RESEARCH", None)
        self.assertNotEqual(
            os.getenv("ENABLE_ORACLE4_ASYNC_RESEARCH",
                      "").strip().lower(), "true")

    def test_scheduler_source_gates_async_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        src_path = os.path.join(root, "run_alpaca_intraday.py")
        with open(src_path, "r", encoding="utf-8") as f:
            src = f.read()
        self.assertIn("ENABLE_ORACLE4_ASYNC_RESEARCH", src)
        self.assertIn("submit_research", src)
        # sync paths must survive untouched for the default-off flag
        self.assertIn("maybe_shadow_scan(", src)
        self.assertIn("feed_bars(", src)


if __name__ == "__main__":
    unittest.main()
