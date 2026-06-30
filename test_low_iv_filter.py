"""Unit tests for low_iv_filter — pure regime-throttle helpers.

No network, no broker, no clock. Mirrors the boundary cases in the module's
own _self_test and adds property-style checks on the floors.
"""
import math

import low_iv_filter as lf


# --------------------------------------------------------------------------- #
# is_low_iv
# --------------------------------------------------------------------------- #
def test_is_low_iv_below_and_at_threshold():
    assert lf.is_low_iv(0.10) is True
    assert lf.is_low_iv(0.149) is True
    assert lf.is_low_iv(0.15) is True  # inclusive boundary


def test_is_low_iv_above_threshold():
    assert lf.is_low_iv(0.16) is False
    assert lf.is_low_iv(0.80) is False


def test_is_low_iv_custom_threshold():
    assert lf.is_low_iv(0.25, threshold=0.30) is True
    assert lf.is_low_iv(0.35, threshold=0.30) is False


def test_is_low_iv_fail_open():
    # Missing / garbage / NaN must never report low-IV (would wrongly throttle).
    assert lf.is_low_iv(None) is False
    assert lf.is_low_iv("nope") is False
    assert lf.is_low_iv(float("nan")) is False


# --------------------------------------------------------------------------- #
# effective_cap
# --------------------------------------------------------------------------- #
def test_effective_cap_noop_when_not_low_iv():
    assert lf.effective_cap(2, low_iv=False) == 2
    assert lf.effective_cap(5, low_iv=False, cap_delta=3) == 5


def test_effective_cap_drops_by_delta():
    assert lf.effective_cap(2, low_iv=True) == 1
    assert lf.effective_cap(5, low_iv=True, cap_delta=2) == 3


def test_effective_cap_never_below_one():
    assert lf.effective_cap(1, low_iv=True) == 1
    assert lf.effective_cap(3, low_iv=True, cap_delta=10) == 1


def test_effective_cap_zero_delta_is_noop():
    assert lf.effective_cap(2, low_iv=True, cap_delta=0) == 2
    assert lf.effective_cap(2, low_iv=True, cap_delta=-1) == 2


# --------------------------------------------------------------------------- #
# adjusted_quantity
# --------------------------------------------------------------------------- #
def test_adjusted_quantity_noop_for_qty_one():
    # The current bot trades qty=1 -> throttle must be a pure no-op there.
    assert lf.adjusted_quantity(1, low_iv=True) == 1


def test_adjusted_quantity_noop_when_not_low_iv():
    assert lf.adjusted_quantity(4, low_iv=False) == 4


def test_adjusted_quantity_scales_and_rounds():
    assert lf.adjusted_quantity(4, low_iv=True) == 2
    assert lf.adjusted_quantity(3, low_iv=True) == 2  # round(1.5) -> 2
    assert lf.adjusted_quantity(2, low_iv=True) == 1


def test_adjusted_quantity_never_below_one():
    assert lf.adjusted_quantity(2, low_iv=True, size_factor=0.1) == 1


def test_adjusted_quantity_factor_edge_cases_fail_open():
    assert lf.adjusted_quantity(10, low_iv=True, size_factor=0.0) == 10
    assert lf.adjusted_quantity(10, low_iv=True, size_factor=1.0) == 10
    assert lf.adjusted_quantity(10, low_iv=True, size_factor=2.0) == 10


def test_adjusted_quantity_custom_factor():
    assert lf.adjusted_quantity(10, low_iv=True, size_factor=0.3) == 3


# --------------------------------------------------------------------------- #
# Self-test entry point
# --------------------------------------------------------------------------- #
def test_self_test_passes():
    assert lf._self_test() == 0
