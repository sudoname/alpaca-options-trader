#!/usr/bin/env python3
"""
Oracle 3.0 — Phase 1 CI-style self-test runner (offline; no creds / no network).

Runs three layers and exits non-zero if ANY of them fail:

  1. Module self-tests  — imports every new Phase-1 module and calls its
     ``_self_test() -> int`` (0 == PASS). These are pure, offline, fail-open.
  2. Compile checks      — byte-compiles the live entry points
     (``smart_trader.py``, ``oracle_intelligence_reports.py``) so a syntax
     regression in the flag-gated wiring is caught here, not in production.
  3. Test suite          — discovers and runs ``tests/`` via stdlib unittest
     (pytest not required).

Usage:
    python run_selftests.py            # everything
    python run_selftests.py --quick    # module self-tests only (skip tests/)
"""

import importlib
import os
import py_compile
import sys
import unittest

ROOT = os.path.dirname(os.path.abspath(__file__))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# New Phase-1 modules, in dependency order (each defines _self_test()).
_MODULES = [
    # Upgrade 1 — unified decision kernel + swappable execution clients.
    "oracle.decision.schema",
    "oracle.decision.direction",
    "oracle.decision.kernel",
    "oracle.execution.client",
    "oracle.execution.alpaca",
    # Upgrade 2 — fill/slippage, latency/partial-fill, executable EV, telemetry.
    "oracle.execution.fill_model",
    "oracle.execution.fill_simulator",
    "oracle.execution.executable_ev",
    "oracle.execution.calibration",
    # Upgrade 3 — Oracle Lab robustness harness.
    "oracle.lab.metrics",
    "oracle.lab.dataset",
    "oracle.lab.experiment",
    "oracle.lab.parameter_sweep",
    "oracle.lab.walk_forward",
    "oracle.lab.parameter_stability",
    # Upgrade 4 — temporal integrity.
    "oracle.temporal",
    # Upgrade 5 — adversarial thesis + semantic trade memory.
    "oracle.thesis_debate",
    "oracle.trade_memory",
    # Phase-2 Upgrade A — unified event-driven simulation engine.
    "oracle.engine.events",
    "oracle.engine.clock",
    "oracle.engine.bus",
    "oracle.engine.strategy",
    "oracle.engine.sim_broker",
    "oracle.engine.backtest_driver",
    # Phase-2 Upgrade B — run the Lab for out-of-sample evidence.
    "oracle.lab.reports",
    "oracle.lab.run_phase2_study",
    # Realized-episode study — point the Lab metric engine at episodes.db.
    "oracle.lab.episodes_study",
    # Phase-2 Upgrade G — automated promotion gates (offline advisory).
    "oracle.promotion",
    "run_promotion_check",
    # Phase-2 Upgrade H — promotion audit ledger (offline audit).
    "oracle.promotion_audit",
    # Phase-2 Upgrade I — promotion regression monitor (offline audit).
    "oracle.promotion_monitor",
]

# Live entry points to byte-compile (flag-gated wiring must still parse).
_COMPILE_TARGETS = [
    "smart_trader.py",
    "oracle_intelligence_reports.py",
]


def _run_module_selftests() -> list:
    """Return a list of (name, ok, detail) for every module self-test."""
    results = []
    for name in _MODULES:
        try:
            mod = importlib.import_module(name)
        except Exception as exc:  # import failure is a hard failure
            results.append((name, False, f"import error: {exc!r}"))
            continue
        fn = getattr(mod, "_self_test", None)
        if not callable(fn):
            results.append((name, False, "no _self_test()"))
            continue
        try:
            rc = fn()
            ok = (rc == 0)
            results.append((name, ok, "PASS" if ok else f"rc={rc}"))
        except Exception as exc:
            results.append((name, False, f"raised: {exc!r}"))
    return results


def _run_compile_checks() -> list:
    results = []
    for rel in _COMPILE_TARGETS:
        path = os.path.join(ROOT, rel)
        if not os.path.exists(path):
            results.append((rel, True, "absent (skipped)"))
            continue
        try:
            py_compile.compile(path, doraise=True)
            results.append((rel, True, "compiled"))
        except Exception as exc:
            results.append((rel, False, f"compile error: {exc!r}"))
    return results


def _run_test_suite() -> bool:
    loader = unittest.TestLoader()
    try:
        suite = loader.discover(os.path.join(ROOT, "tests"), pattern="test_*.py")
    except Exception as exc:
        print(f"[selftests] test discovery failed: {exc!r}")
        return False
    result = unittest.TextTestRunner(verbosity=1).run(suite)
    return result.wasSuccessful()


def main(argv=None) -> int:
    argv = argv if argv is not None else sys.argv[1:]
    quick = "--quick" in argv

    print("=" * 70)
    print("Oracle 3.0 Phase-1 self-tests")
    print("=" * 70)

    failures = 0

    print("\n[1/3] Module self-tests")
    print("-" * 70)
    for name, ok, detail in _run_module_selftests():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4}  {name:38}  {detail}")
        if not ok:
            failures += 1

    print("\n[2/3] Compile checks (live entry points)")
    print("-" * 70)
    for name, ok, detail in _run_compile_checks():
        status = "PASS" if ok else "FAIL"
        print(f"  {status:4}  {name:38}  {detail}")
        if not ok:
            failures += 1

    if quick:
        print("\n[3/3] Test suite  SKIPPED (--quick)")
    else:
        print("\n[3/3] Test suite (tests/)")
        print("-" * 70)
        if not _run_test_suite():
            failures += 1

    print("\n" + "=" * 70)
    if failures == 0:
        print("run_selftests: ALL PASS")
    else:
        print(f"run_selftests: {failures} FAILURE(S)")
    print("=" * 70)
    return 0 if failures == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
