"""
Cross-process locked + atomic JSON state store.

Why this exists
---------------
Several JSON state files (``active_trades.json``, ``telegram_trades.json``,
``trading_history.json``, ``realized_pnl_log.json``, ``day_trades_log.json``)
were read-modify-written with **no locking** by two long-running processes
(``alps-scheduler`` and ``alps-bot``). ``monitor_positions`` in particular read
the whole list, ran a slow loop with broker network calls, then blindly
overwrote the file -- clobbering any append another process made during the
loop. That lost-update race silently dropped freshly-opened positions from
tracking, leaving them with no stops/take-profit/EOD management.

This module serializes every write behind an advisory cross-process lock and
makes writes atomic (temp file + ``os.replace``). The mutation helpers re-read
the **current on-disk** state under the lock and apply a delta, so concurrent
appends are never clobbered.

Design contract
----------------
* Fail-open everywhere: locking that can't be acquired (or isn't available on
  the platform) degrades to a best-effort no-op rather than raising; corrupt /
  missing files read as the supplied default. A failure here must never crash a
  monitor cycle or block a trade.
* Cross-platform: ``fcntl.flock`` on posix (the Linux server), ``msvcrt`` on
  win32 (offline tests), no-op elsewhere.
* Atomic: writes go to a temp file in the same directory, then ``os.replace``
  (atomic on the same filesystem) -- a crash mid-write can't truncate the file.

It is intentionally side-effect-only and never raises out of its public API.
"""

import contextlib
import json
import os
import sys
import tempfile
import time

# Advisory-lock backends, guarded by platform. Either may be unavailable.
try:
    import fcntl  # posix
except ImportError:
    fcntl = None
try:
    import msvcrt  # win32
except ImportError:
    msvcrt = None


def _lock_path(path):
    """Sidecar lock file. We lock this, never the data file itself, so the data
    file can be atomically ``os.replace``d while the lock is held."""
    return f"{path}.lock"


@contextlib.contextmanager
def locked(path, timeout=10.0):
    """Advisory exclusive lock on ``<path>.lock``, fail-open.

    Acquires an exclusive lock via ``fcntl`` (posix) or ``msvcrt`` (win32),
    retrying for up to ``timeout`` seconds. If no backend is available, the lock
    can't be opened, or the timeout elapses, it logs once and yields anyway
    (best-effort) so the caller is never blocked or crashed.

    Not reentrant: do not nest ``locked(path)`` for the same path in one thread.
    """
    if fcntl is None and msvcrt is None:
        yield  # no backend -> best effort
        return

    lock_file = _lock_path(path)
    fd = None
    acquired = False
    try:
        try:
            fd = open(lock_file, "a+")
        except OSError as e:
            print(f"[JSON_STORE] cannot open lock {lock_file}: {e}")
            yield
            return

        deadline = time.time() + max(0.0, timeout)
        while True:
            try:
                if fcntl is not None:
                    fcntl.flock(fd.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                else:  # win32
                    fd.seek(0)
                    msvcrt.locking(fd.fileno(), msvcrt.LK_NBLCK, 1)
                acquired = True
                break
            except OSError:
                if time.time() >= deadline:
                    print(f"[JSON_STORE] lock timeout on {lock_file}; "
                          f"proceeding best-effort")
                    break
                time.sleep(0.05)
        yield
    finally:
        if fd is not None:
            try:
                if acquired:
                    if fcntl is not None:
                        fcntl.flock(fd.fileno(), fcntl.LOCK_UN)
                    else:
                        fd.seek(0)
                        msvcrt.locking(fd.fileno(), msvcrt.LK_UNLCK, 1)
            except OSError:
                pass
            try:
                fd.close()
            except OSError:
                pass


def read_json(path, default=None):
    """Parsed JSON from ``path``, or ``default`` on missing/corrupt. Never raises."""
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception as e:
        print(f"[JSON_STORE] read failed for {path}: {e}")
    return default


def atomic_write_json(path, data):
    """Write ``data`` as JSON atomically (temp file + ``os.replace``).

    Returns True on success, False on failure (never raises).
    """
    directory = os.path.dirname(os.path.abspath(path)) or "."
    tmp_fd = None
    tmp_path = None
    try:
        tmp_fd, tmp_path = tempfile.mkstemp(dir=directory, prefix=".tmp_", suffix=".json")
        with os.fdopen(tmp_fd, "w") as f:
            tmp_fd = None  # now owned by the file object
            json.dump(data, f, indent=2, default=str)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, path)
        return True
    except Exception as e:
        print(f"[JSON_STORE] atomic write failed for {path}: {e}")
        # Clean up the temp file if the replace never happened.
        try:
            if tmp_fd is not None:
                os.close(tmp_fd)
        except OSError:
            pass
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except OSError:
            pass
        return False


def append_item(path, item, *, default=None):
    """Under the lock, re-read the list, append ``item``, atomic-write."""
    with locked(path):
        cur = read_json(path, [] if default is None else default)
        if not isinstance(cur, list):
            cur = []
        cur.append(item)
        return atomic_write_json(path, cur)


def update_items(path, key, mutate_fn, *, default=None):
    """Under the lock, re-read the list and apply ``mutate_fn`` to each row.

    ``mutate_fn(row)`` returning ``None`` DROPS that row; any other return value
    replaces it. ``key`` is accepted for symmetry / future keyed dispatch; the
    callback decides which rows to touch.
    """
    with locked(path):
        cur = read_json(path, [] if default is None else default)
        if not isinstance(cur, list):
            return False
        out = []
        for row in cur:
            new = mutate_fn(row)
            if new is not None:
                out.append(new)
        return atomic_write_json(path, out)


def merge_list(path, *, updates=None, removals=None, key="symbol",
               appends=None, default=None):
    """Merge a delta into the on-disk list under the lock.

    Re-reads the **current** on-disk list (so concurrent appends survive), then:
      * replaces any row whose ``row[key]`` is in ``updates`` (a {key: row} map),
      * drops any row whose ``row[key]`` is in ``removals`` (a set/iterable),
      * appends every item in ``appends``.
    Rows that are neither updated nor removed are preserved unchanged -- this is
    what stops ``monitor_positions`` from clobbering a freshly-appended trade.
    """
    updates = dict(updates or {})
    removals = set(removals or ())
    appends = list(appends or [])
    with locked(path):
        cur = read_json(path, [] if default is None else default)
        if not isinstance(cur, list):
            cur = []
        out = []
        for row in cur:
            k = row.get(key) if isinstance(row, dict) else None
            if k in removals:
                continue
            if k in updates:
                out.append(updates[k])
            else:
                out.append(row)
        out.extend(appends)
        return atomic_write_json(path, out)


def replace_items(path, fn, *, default=None):
    """Under the lock, atomic-write ``fn(current_on_disk)``.

    Escape hatch for dict-shaped stores (e.g. ``trading_history.json``) where a
    list-keyed merge doesn't apply.
    """
    with locked(path):
        cur = read_json(path, default)
        return atomic_write_json(path, fn(cur))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses temp files)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    d = tempfile.mkdtemp()
    path = os.path.join(d, "store.json")

    # 1) round-trip
    atomic_write_json(path, [{"symbol": "A", "v": 1}])
    if read_json(path) != [{"symbol": "A", "v": 1}]:
        print("FAIL: round-trip"); ok = False

    # 2) default on missing + corrupt
    if read_json(os.path.join(d, "nope.json"), "DEF") != "DEF":
        print("FAIL: missing -> default"); ok = False
    bad = os.path.join(d, "bad.json")
    with open(bad, "w") as f:
        f.write("{not valid json")
    if read_json(bad, []) != []:
        print("FAIL: corrupt -> default"); ok = False

    # 3) append preserves order
    ap = os.path.join(d, "ap.json")
    for s in ("A", "B", "C"):
        append_item(ap, {"symbol": s})
    if [r["symbol"] for r in read_json(ap, [])] != ["A", "B", "C"]:
        print("FAIL: append order"); ok = False

    # 4) merge-preserves-unseen (the 67-orphan regression). Seed [A,B]; another
    #    "process" appends C; a caller holding the STALE [A,B] snapshot updates A
    #    and removes B. C must survive, B must go, A must update.
    mg = os.path.join(d, "mg.json")
    atomic_write_json(mg, [{"symbol": "A", "v": 1}, {"symbol": "B", "v": 2}])
    append_item(mg, {"symbol": "C", "v": 3})  # concurrent writer
    merge_list(mg, updates={"A": {"symbol": "A", "v": 99}}, removals={"B"})
    res = {r["symbol"]: r["v"] for r in read_json(mg, [])}
    if res != {"A": 99, "C": 3}:
        print("FAIL: merge preserve-unseen", res); ok = False

    # 5) update_items drop (mutate_fn -> None drops the row)
    ui = os.path.join(d, "ui.json")
    atomic_write_json(ui, [{"symbol": "A"}, {"symbol": "B"}])
    update_items(ui, "symbol", lambda r: None if r["symbol"] == "B" else r)
    if [r["symbol"] for r in read_json(ui, [])] != ["A"]:
        print("FAIL: update_items drop"); ok = False

    # 6) atomicity: a write of unserializable data fails without truncating the
    #    existing file (the temp file is discarded, original stays intact).
    at = os.path.join(d, "at.json")
    atomic_write_json(at, [{"symbol": "A"}])

    class _Boom:
        def __repr__(self):  # default=str will call this and blow up mid-dump
            raise RuntimeError("boom")
    wrote = atomic_write_json(at, _Boom())
    if wrote is not False:
        print("FAIL: bad write should return False"); ok = False
    if read_json(at) != [{"symbol": "A"}]:
        print("FAIL: original file truncated by failed write"); ok = False

    # 7) lock is a context manager that re-acquires cleanly after release
    lk = os.path.join(d, "lk.json")
    atomic_write_json(lk, [])
    try:
        with locked(lk):
            pass
        with locked(lk):
            pass
    except Exception as e:
        print("FAIL: lock re-acquire", e); ok = False

    # 8) replace_items for dict-shaped store
    rp = os.path.join(d, "rp.json")
    atomic_write_json(rp, {"trades": [], "n": 0})
    replace_items(rp, lambda cur: {"trades": [1], "n": (cur or {}).get("n", 0) + 1})
    if read_json(rp) != {"trades": [1], "n": 1}:
        print("FAIL: replace_items"); ok = False

    print("json_store self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
