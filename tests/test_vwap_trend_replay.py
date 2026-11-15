"""Hermetic tests for vwap_trend_replay.py (research-only, no network).

The module selftest covers the regression surface the protocol requires:
same opportunity set across variants, $0 for filter skips vs excluded+
disclosed unknown outcomes, missing-signal skips reported separately,
paired per-session deltas, deterministic bootstrap, dev/holdout split,
and the zero-broker-import guarantee. A few direct unit checks follow.
"""
import os
import subprocess
import sys

import vwap_trend_replay as vtr

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def test_selftest_passes_in_fresh_interpreter():
    """Run the selftest in a subprocess: the zero-broker-import guarantee
    scans sys.modules, which the shared pytest process legitimately pollutes
    (other tests import smart_trader / alpaca.trading)."""
    proc = subprocess.run(
        [sys.executable, "vwap_trend_replay.py", "--selftest"],
        cwd=_ROOT, capture_output=True, text=True, timeout=120)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    assert "SELFTEST PASS" in proc.stdout
    assert "FAIL" not in proc.stdout.replace("SELFTEST PASS", "")


def test_takes_semantics():
    o = {"align_vwap": False, "align_full": None}
    assert vtr.takes(o, "baseline") is True
    assert vtr.takes(o, "vwap") is False
    assert vtr.takes(o, "full") is None
    # contra: inverted on defined signals, None stays None
    assert vtr.takes(o, "vwap_contra") is True
    assert vtr.takes(o, "full_contra") is None
    o2 = {"align_vwap": True, "align_full": True}
    assert vtr.takes(o2, "vwap_contra") is False
    assert vtr.takes(o2, "full_contra") is False


def test_variant_metrics_empty_population():
    m = vtr.variant_metrics([], "baseline")
    assert m["opportunities"] == 0
    assert m["net_per_opportunity"] is None
    assert m["win_rate_taken"] is None


def test_paired_delta_zero_when_variants_identical():
    opps = [{"session": "2026-11-04", "net": 5.0, "outcome_known": True,
             "align_vwap": True, "align_full": True, "direction": "CALL"}]
    pd = vtr.paired_delta(opps, "vwap", "baseline")
    assert pd["delta_per_opportunity"] == 0.0


def test_bootstrap_requires_two_sessions():
    assert vtr.bootstrap_ci([("2026-11-04", 1.0, 0.0, 2)]) is None
    bs = vtr.bootstrap_ci([("2026-11-04", 1.0, 0.0, 2),
                           ("2026-11-05", -1.0, 0.0, 2)])
    assert bs is not None and bs["ci95"][0] <= bs["ci95"][1]
