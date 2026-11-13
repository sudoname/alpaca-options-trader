"""Gate exception policy: fail-closed on evaluation error (Phase 0, Patch A).

Reproduces the previously fail-open behavior offline and locks in the new
default: an evaluation error inside the EV entry gate or the profitability
gate BLOCKS the new entry (returns False) instead of silently proceeding.

Escape hatches restore the legacy behavior:
  - EV_GATE_FAIL_OPEN=1            -> ev_gate_fail_open = True
  - PROFITABILITY_GATE_FAIL_OPEN=1 -> profitability_gate_fail_open = True
Dry-run mode (PROFITABILITY_GATE_DRYRUN=1) still never blocks, including on
evaluation errors, preserving observe-only semantics.

Exit management is untouched by construction: both gates run only in the
pre-submission entry path (place_order_with_stops), never in exit code.

Hermetic: no network, no broker, no files. The trader instance is built with
__new__ and explicit attributes; collaborators are monkeypatched at module
level (the gates import them at call time via `from X import y`).
"""
import sys
import types

import pytest

import entry_ev_stamp
import profitability_validator
from profitability_validator import DEFAULT_RULES
from smart_trader import SmartOptionsTrader


# --------------------------------------------------------------------------- #
# fixtures                                                                     #
# --------------------------------------------------------------------------- #

GOOD_STAMP = {
    'expected_value': 2.5,
    'probability_of_profit': 0.68,
    'ev_per_dollar_risk': 0.02,
}

WEAK_STAMP = {
    'expected_value': -0.5,
    'probability_of_profit': 0.45,
    'ev_per_dollar_risk': -0.004,
}

OPTION = {
    'symbol': 'SPY261219C00600000',
    'type': 'call',
    'confidence': 5,
    'score': 12.0,
    'oracle_probability': {'p_call': 0.60, 'p_put': 0.20, 'p_no_trade': 0.20},
}

LEVELS = {'take_profit_percent': 0.25, 'stop_loss_percent': 0.15}
QUOTE = {'ask': 1.30}


def make_trader(**overrides):
    """Bare SmartOptionsTrader with only the attributes the gates read."""
    t = SmartOptionsTrader.__new__(SmartOptionsTrader)
    t.last_block_reason = None
    t.research_capture = None
    t.last_signal_strength = 3
    # EV gate
    t.use_ev_entry_gate = True
    t.min_ev_per_dollar_risk = 0.0
    t.ev_gate_fail_open = False
    # Profitability gate
    t.use_profitability_gate = True
    t.profitability_gate_dryrun = False
    t.profitability_gate_fail_open = False
    t.profitability_gate_rules = dict(DEFAULT_RULES)
    t.use_oracle_trade_gate = False
    t.enable_orderbook_shadow = False
    for k, v in overrides.items():
        setattr(t, k, v)
    return t


class CaptureSpy:
    """Stands in for research_capture; records on_candidate kwargs."""

    def __init__(self, raise_on_call=False):
        self.calls = []
        self.raise_on_call = raise_on_call

    def on_candidate(self, **kwargs):
        self.calls.append(kwargs)
        if self.raise_on_call:
            raise RuntimeError('ledger write failed')


@pytest.fixture
def quiet_book(monkeypatch):
    """Offline rh_price_book: strong imbalance so orderbook_veto never fires."""
    fake = types.SimpleNamespace(
        get_order_book_imbalance=lambda sym: {'orderbook_imbalance': 0.18})
    monkeypatch.setitem(sys.modules, 'rh_price_book', fake)
    return fake


@pytest.fixture
def good_stamp(monkeypatch):
    monkeypatch.setattr(entry_ev_stamp, 'compute_entry_stamp',
                        lambda *a, **k: dict(GOOD_STAMP))


def call_ev_gate(t, option=None):
    return t._ev_entry_gate_allows(
        dict(option or OPTION), 'SPY', dict(LEVELS), 1.25, 1, 1.20,
        dict(QUOTE))


def call_profit_gate(t, option=None):
    return t._profitability_gate_allows(
        dict(option or OPTION), 'SPY', dict(LEVELS), 1.25, 1, 1.20,
        dict(QUOTE))


# --------------------------------------------------------------------------- #
# EV gate exception policy                                                     #
# --------------------------------------------------------------------------- #

class TestEvGateExceptionPolicy:
    def test_error_blocks_by_default(self, monkeypatch):
        monkeypatch.setattr(
            entry_ev_stamp, 'compute_entry_stamp',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        t = make_trader()
        assert call_ev_gate(t) is False
        assert 'EV gate evaluation error' in t.last_block_reason

    def test_fail_open_flag_reproduces_legacy_proceed(self, monkeypatch):
        monkeypatch.setattr(
            entry_ev_stamp, 'compute_entry_stamp',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        t = make_trader(ev_gate_fail_open=True)
        assert call_ev_gate(t) is True
        assert t.last_block_reason is None

    def test_error_is_captured_with_reject_code(self, monkeypatch):
        monkeypatch.setattr(
            entry_ev_stamp, 'compute_entry_stamp',
            lambda *a, **k: (_ for _ in ()).throw(RuntimeError('boom')))
        spy = CaptureSpy()
        t = make_trader(research_capture=spy)
        assert call_ev_gate(t) is False
        assert len(spy.calls) == 1
        call = spy.calls[0]
        assert call['reject_code'] == 'ev_gate:evaluation_error'
        assert call['accepted'] is False
        assert call['symbol'] == OPTION['symbol']
        assert 'boom' in call['extra']['error']


class TestEvGateBehaviorPreserved:
    def test_above_floor_proceeds(self, good_stamp):
        t = make_trader()
        assert call_ev_gate(t) is True
        assert t.last_block_reason is None

    def test_below_floor_blocks_with_reason_and_capture(self, monkeypatch):
        low = dict(GOOD_STAMP, ev_per_dollar_risk=-0.01)
        monkeypatch.setattr(entry_ev_stamp, 'compute_entry_stamp',
                            lambda *a, **k: low)
        spy = CaptureSpy()
        t = make_trader(research_capture=spy)
        assert call_ev_gate(t) is False
        assert t.last_block_reason.startswith('EV gate:')
        assert spy.calls[0]['reject_code'] == \
            'ev_gate:ev_per_dollar_risk_below_floor'


# --------------------------------------------------------------------------- #
# Profitability gate exception policy                                          #
# --------------------------------------------------------------------------- #

def _raise_validate(monkeypatch):
    monkeypatch.setattr(
        profitability_validator, 'validate_trade',
        lambda *a, **k: (_ for _ in ()).throw(RuntimeError('rules exploded')))


class TestProfitGateExceptionPolicy:
    def test_error_blocks_by_default(self, monkeypatch, quiet_book, good_stamp):
        _raise_validate(monkeypatch)
        t = make_trader()
        assert call_profit_gate(t) is False
        assert 'profitability gate evaluation error' in t.last_block_reason

    def test_fail_open_flag_reproduces_legacy_proceed(
            self, monkeypatch, quiet_book, good_stamp):
        _raise_validate(monkeypatch)
        t = make_trader(profitability_gate_fail_open=True)
        assert call_profit_gate(t) is True
        assert t.last_block_reason is None

    def test_dry_run_never_blocks_even_on_error(
            self, monkeypatch, quiet_book, good_stamp):
        _raise_validate(monkeypatch)
        t = make_trader(profitability_gate_dryrun=True)
        assert call_profit_gate(t) is True
        assert t.last_block_reason is None

    def test_error_is_captured_with_reject_code(
            self, monkeypatch, quiet_book, good_stamp):
        _raise_validate(monkeypatch)
        spy = CaptureSpy()
        t = make_trader(research_capture=spy)
        assert call_profit_gate(t) is False
        call = spy.calls[0]
        assert call['reject_code'] == 'profit_gate:evaluation_error'
        assert call['accepted'] is False
        assert 'rules exploded' in call['extra']['error']

    def test_capture_failure_never_changes_the_block(
            self, monkeypatch, quiet_book, good_stamp):
        _raise_validate(monkeypatch)
        spy = CaptureSpy(raise_on_call=True)
        t = make_trader(research_capture=spy)
        assert call_profit_gate(t) is False  # block stands despite ledger error
        assert len(spy.calls) == 1


class TestProfitGateBehaviorPreserved:
    """Real validate_trade + DEFAULT_RULES: allow/block/dry-run unchanged."""

    def test_strong_candidate_allowed(self, quiet_book, good_stamp):
        t = make_trader()
        assert call_profit_gate(t) is True
        assert t.last_block_reason is None

    def _weak_option(self):
        return dict(
            OPTION, confidence=2,
            oracle_probability={'p_call': 0.20, 'p_put': 0.30,
                                'p_no_trade': 0.50})

    def test_weak_candidate_blocked_with_reason(
            self, monkeypatch, quiet_book):
        monkeypatch.setattr(entry_ev_stamp, 'compute_entry_stamp',
                            lambda *a, **k: dict(WEAK_STAMP))
        spy = CaptureSpy()
        t = make_trader(research_capture=spy)
        assert call_profit_gate(t, self._weak_option()) is False
        assert t.last_block_reason.startswith('profitability gate:')
        assert spy.calls[0]['reject_code'].startswith('profit_gate:')

    def test_dry_run_would_block_proceeds(self, monkeypatch, quiet_book):
        monkeypatch.setattr(entry_ev_stamp, 'compute_entry_stamp',
                            lambda *a, **k: dict(WEAK_STAMP))
        t = make_trader(profitability_gate_dryrun=True)
        assert call_profit_gate(t, self._weak_option()) is True
        assert t.last_block_reason is None
