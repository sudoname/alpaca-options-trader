"""Asynchronous research worker (Oracle 4.0.2 P1-8).

Decouples research/shadow work (bars feed, challenger shadow scans) from
the production scheduler loop so a slow or failing research task can
never delay exits, ranking, or order placement.

Design:
  * **bounded queue** -- ``queue.Queue(maxsize=N)``; when full the NEW
    task is dropped and counted (backpressure), ``submit`` never blocks;
  * **immutable snapshots** -- the payload is ``copy.deepcopy``-ed at
    submit time, so later mutation of scheduler state cannot change what
    the research task sees; an uncopyable payload is rejected, never
    shared by reference;
  * **failure isolation** -- a raising task increments ``failed`` and
    records ``last_error``; the worker thread continues with the next
    task;
  * **soft timeouts** -- Python threads cannot be preempted, so an
    overrunning task is *detected* (``overruns`` counter and a live
    ``stalled`` flag) rather than killed.  The production scheduler is
    still never blocked: ``submit`` is non-blocking and the worker is a
    daemon thread; if the worker wedges, the queue fills and new tasks
    are dropped fail-open;
  * **graceful shutdown** -- ``stop()`` sends a sentinel, lets queued
    tasks finish, and joins with a timeout;
  * **health** -- ``metrics()`` returns a thread-safe snapshot and the
    worker writes an atomic JSON status file after every task for the
    dashboard (``/api/research-worker``).

Production stance: nothing here runs unless the scheduler is started
with ``ENABLE_ORACLE4_ASYNC_RESEARCH=true``; the flag default keeps the
existing synchronous, env-gated research path byte-for-byte unchanged.
"""
from __future__ import annotations

import copy
import json
import os
import queue
import threading
import time
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Optional

UTC = timezone.utc

DEFAULT_MAXSIZE = 8
DEFAULT_TASK_TIMEOUT_S = 120.0
DEFAULT_STATUS_PATH = os.path.join(
    "artifacts", "oracle4", "research_worker_status.json")

_SENTINEL = object()


def _utc_now_iso() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ResearchWorker:
    """Single background daemon thread consuming a bounded task queue."""

    def __init__(
        self,
        *,
        maxsize: int = DEFAULT_MAXSIZE,
        task_timeout_s: float = DEFAULT_TASK_TIMEOUT_S,
        status_path: Optional[str] = DEFAULT_STATUS_PATH,
        name: str = "oracle4-research",
    ) -> None:
        self.task_timeout_s = float(task_timeout_s)
        self.status_path = status_path
        self._q: "queue.Queue" = queue.Queue(maxsize=int(maxsize))
        self._lock = threading.Lock()
        self._stopping = threading.Event()
        self._thread = threading.Thread(
            target=self._run, name=name, daemon=True)
        self._started = False
        self._m: Dict[str, Any] = {
            "submitted": 0,
            "completed": 0,
            "failed": 0,
            "dropped_queue_full": 0,
            "dropped_snapshot": 0,
            "rejected_stopped": 0,
            "overruns": 0,
            "last_task": None,
            "last_duration_s": None,
            "last_finished_utc": None,
            "last_error": None,
            "started_utc": None,
        }
        # live view of the in-flight task (for stall detection)
        self._current_task: Optional[str] = None
        self._current_started_mono: Optional[float] = None

    # -- lifecycle --------------------------------------------------------- #
    def start(self) -> "ResearchWorker":
        """Start the worker thread (idempotent)."""
        with self._lock:
            if not self._started:
                self._m["started_utc"] = _utc_now_iso()
                self._started = True
                self._thread.start()
        return self

    def stop(self, timeout: float = 5.0) -> bool:
        """Graceful shutdown: finish queued tasks, then join.

        Returns True when the thread exited within ``timeout``.  After
        ``stop`` every ``submit`` is rejected.
        """
        self._stopping.set()
        if not self._started:
            return True
        try:
            self._q.put_nowait(_SENTINEL)
        except queue.Full:
            # queue full of real tasks; the drain loop in _run will see
            # the stopping flag once the queue empties.
            pass
        self._thread.join(timeout=float(timeout))
        return not self._thread.is_alive()

    # -- producer side ------------------------------------------------------ #
    def submit(self, task_name: str,
               fn: Callable[[Any], Any],
               payload: Any = None) -> bool:
        """Queue ``fn(payload_snapshot)``; never blocks, never raises.

        Returns True when queued; False when rejected (stopped, payload
        not snapshot-able, or queue full -> backpressure drop).
        """
        if self._stopping.is_set() or (
                self._started and not self._thread.is_alive()):
            with self._lock:
                self._m["rejected_stopped"] += 1
            return False
        try:
            snap = copy.deepcopy(payload)
        except Exception as exc:
            with self._lock:
                self._m["dropped_snapshot"] += 1
                self._m["last_error"] = (
                    f"{task_name}: payload not deep-copyable: {exc!r}")
            return False
        try:
            self._q.put_nowait((str(task_name), fn, snap))
        except queue.Full:
            with self._lock:
                self._m["dropped_queue_full"] += 1
            return False
        with self._lock:
            self._m["submitted"] += 1
        return True

    # -- consumer side ------------------------------------------------------ #
    def _run(self) -> None:
        while True:
            try:
                item = self._q.get(timeout=0.25)
            except queue.Empty:
                if self._stopping.is_set():
                    return
                continue
            if item is _SENTINEL:
                # drain remaining real tasks before exiting
                if self._stopping.is_set() and self._q.empty():
                    return
                continue
            task_name, fn, payload = item
            with self._lock:
                self._current_task = task_name
                self._current_started_mono = time.monotonic()
            t0 = time.monotonic()
            err: Optional[str] = None
            try:
                fn(payload)
            except Exception as exc:  # failure isolation
                err = f"{task_name}: {exc!r}"
            duration = time.monotonic() - t0
            with self._lock:
                self._current_task = None
                self._current_started_mono = None
                self._m["last_task"] = task_name
                self._m["last_duration_s"] = round(duration, 4)
                self._m["last_finished_utc"] = _utc_now_iso()
                if duration > self.task_timeout_s:
                    self._m["overruns"] += 1
                if err is None:
                    self._m["completed"] += 1
                else:
                    self._m["failed"] += 1
                    self._m["last_error"] = err
            self._write_status()

    # -- health ------------------------------------------------------------- #
    def metrics(self) -> Dict[str, Any]:
        """Thread-safe snapshot of counters + live queue/stall state."""
        with self._lock:
            out = dict(self._m)
            cur = self._current_task
            cur_started = self._current_started_mono
        out["queue_depth"] = self._q.qsize()
        out["queue_maxsize"] = self._q.maxsize
        out["alive"] = self._thread.is_alive()
        out["stopping"] = self._stopping.is_set()
        out["task_timeout_s"] = self.task_timeout_s
        out["current_task"] = cur
        age = (time.monotonic() - cur_started) if cur_started is not None \
            else None
        out["current_task_age_s"] = round(age, 4) if age is not None else None
        out["stalled"] = bool(age is not None and age > self.task_timeout_s)
        return out

    def _write_status(self) -> None:
        """Atomic (tmp + replace) status JSON for the dashboard; fail-open."""
        path = self.status_path
        if not path:
            return
        try:
            doc = {
                "stage": "oracle4.0.2-research-worker",
                "generated_at_utc": _utc_now_iso(),
                **self.metrics(),
            }
            d = os.path.dirname(os.path.abspath(path))
            if d:
                os.makedirs(d, exist_ok=True)
            tmp = f"{path}.tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(doc, f, indent=2, sort_keys=True)
            os.replace(tmp, path)
        except Exception:
            pass  # health reporting must never break research work


# -- process-wide singleton for the scheduler ------------------------------- #
_GLOBAL: Optional[ResearchWorker] = None
_GLOBAL_LOCK = threading.Lock()


def global_worker() -> ResearchWorker:
    """Lazily create+start the process singleton (env-configured)."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is None:
            def _f(name: str, default: float) -> float:
                try:
                    return float(os.getenv(name, "") or default)
                except (TypeError, ValueError):
                    return default
            _GLOBAL = ResearchWorker(
                maxsize=int(_f("ORACLE4_RESEARCH_QUEUE_MAX",
                               DEFAULT_MAXSIZE)),
                task_timeout_s=_f("ORACLE4_RESEARCH_TASK_TIMEOUT_S",
                                  DEFAULT_TASK_TIMEOUT_S),
                status_path=os.getenv("ORACLE4_RESEARCH_STATUS",
                                      DEFAULT_STATUS_PATH),
            ).start()
        return _GLOBAL


def submit_research(task_name: str, fn: Callable[[Any], Any],
                    payload: Any = None) -> bool:
    """Convenience: submit onto the process-wide worker; never raises."""
    try:
        return global_worker().submit(task_name, fn, payload)
    except Exception:
        return False


def _reset_global_for_tests() -> None:
    """Stop and clear the singleton (test hook only)."""
    global _GLOBAL
    with _GLOBAL_LOCK:
        if _GLOBAL is not None:
            _GLOBAL.stop(timeout=2.0)
        _GLOBAL = None
