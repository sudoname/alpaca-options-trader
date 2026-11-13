"""Phase 1 candidate-group capture: hermetic tests.

Locks in the observe-only group capture wired into
SmartOptionsTrader.select_best_option:

  - OFF by default and a strict no-op when disabled (pick parity, no ledger
    rows, no id stamps on the winner).
  - When enabled, the full candidate group (scored rows + in-selector skips)
    lands in the research ledger as `group_candidate` snapshots sharing
    decision_id = deterministic group id; the winner is stamped with
    scan_id/group_id/candidate_id so _candidate_extra joins gate verdicts and
    the accepted-candidate snapshot back to the group.
  - Retries are idempotent (deterministic snapshot ids + INSERT OR IGNORE).
  - Capture is bounded and always keeps the selected row.

Hermetic: no network, no broker, no files. The trader is built via __new__
with only the attributes select_best_option reads; all contracts are
mock=True so the batched quote prefetch never fires (chain_symbols == []).
The ledger is an in-memory SQLite database.
"""
import json
from datetime import date, timedelta

from research_capture import ResearchCapture, candidate_id_for, group_id_for
from research_ledger import ResearchLedger
from smart_trader import SmartOptionsTrader


EXP_NEAR = (date.today() + timedelta(days=30)).isoformat()
EXP_FAR = (date.today() + timedelta(days=200)).isoformat()
PRICE = 600.0


def mk(symbol, strike, bid, ask, typ='call', exp=EXP_NEAR):
    """Mock contract: the mock=True path never touches market data."""
    return {'symbol': symbol, 'strike_price': strike, 'expiration_date': exp,
            'type': typ, 'mock': True, 'mock_bid': bid, 'mock_ask': ask}


def contracts():
    return [
        mk('SPY_C595', 595, 2.0, 2.2),                   # winner (score 22.0)
        mk('SPY_C590', 590, 1.0, 1.2),                   # runner-up (11.0)
        mk('SPY_C598', 598, 11.0, 12.0),                 # $1200 > $1000 budget
        mk('SPY_P605', 605, 1.0, 1.1, typ='put'),        # wrong type -> counter
        mk('SPY_C640', 640, 0.3, 0.4),                   # far OTM -> counter
        mk('SPY_C596F', 596, 1.5, 1.7, exp=EXP_FAR),     # phase2 DTE reject
    ]


def make_trader(enabled=True, scan_id='scan-A', max_rows=60):
    """Bare trader with only what select_best_option + capture read."""
    t = SmartOptionsTrader.__new__(SmartOptionsTrader)
    t.ticker = 'SPY'
    t.news_service = None
    t.use_real_greeks = False
    t.use_delta_targeting = False
    t.max_spread_pct = 15.0
    t.max_budget_per_trade = 1000.0
    t.last_signal_strength = 3
    t.use_normalized_confidence = False
    t.conf_high_signals = 2
    t.conf_very_high_signals = 4
    # Deterministic scoring stub: monotone in bid, so SPY_C595 always wins.
    t.calculate_option_score = lambda od: od['bid'] * 10.0
    # Phase 2 stub: reject long-dated contracts, pass everything else.
    t.evaluate_contract_phase2 = (
        lambda **k: (('dte_out_of_range', 1.0)
                     if (k.get('dte') or 0) > 100 else (None, 1.0)))
    t.enable_group_capture = enabled
    t.group_capture_max = max_rows
    t.current_scan_id = scan_id
    t.research_capture = ResearchCapture(
        ledger=ResearchLedger(':memory:'), enabled=True)
    return t


def snaps_for(t, gid):
    return t.research_capture._ledger.get_decision(gid)['snapshots']


# --------------------------------------------------------------------------- #
# disabled: strict no-op                                                      #
# --------------------------------------------------------------------------- #

def test_disabled_capture_is_a_noop():
    t = make_trader(enabled=False)
    best = t.select_best_option(contracts(), PRICE, 'call')
    assert best['symbol'] == 'SPY_C595'
    assert t.research_capture._ledger.stats()['snapshots'] == 0
    for key in ('scan_id', 'group_id', 'candidate_id'):
        assert key not in best


def test_enabled_preserves_the_baseline_pick():
    off = make_trader(enabled=False).select_best_option(
        contracts(), PRICE, 'call')
    on = make_trader(enabled=True).select_best_option(
        contracts(), PRICE, 'call')
    assert on['symbol'] == off['symbol'] == 'SPY_C595'
    assert on['score'] == off['score']
    assert on['strategy_type'] == off['strategy_type']


# --------------------------------------------------------------------------- #
# enabled: group contents                                                     #
# --------------------------------------------------------------------------- #

def test_group_rows_capture_skips_and_scores():
    t = make_trader()
    best = t.select_best_option(contracts(), PRICE, 'call')
    gid = best['group_id']
    assert gid == group_id_for(
        'scan-A', 'SPY', 'call',
        ['SPY_C598', 'SPY_C596F', 'SPY_C595', 'SPY_C590'])
    snaps = snaps_for(t, gid)
    # 2 scored rows + over_budget + phase2 skip; moneyness/type skips are
    # counters, not rows.
    assert len(snaps) == 4
    assert all(s['kind'] == 'group_candidate' for s in snaps)

    accepted = [s for s in snaps if s['accepted'] == 1]
    assert len(accepted) == 1
    sel = accepted[0]
    assert sel['symbol'] == 'SPY_C595'
    assert sel['reject_code'] is None
    assert sel['quote_bid'] == 2.0 and sel['quote_ask'] == 2.2
    assert sel['score'] == best['score']
    assert sel['direction'] == 'CALL'
    assert sel['underlying_price'] == PRICE

    codes = {s['symbol']: s['reject_code'] for s in snaps
             if s['accepted'] == 0}
    assert codes == {
        'SPY_C590': 'not_selected',
        'SPY_C598': 'over_budget',
        'SPY_C596F': 'phase2:dte_out_of_range',
    }

    extra = json.loads(sel['extra_json'])
    assert extra['scan_id'] == 'scan-A'
    assert extra['group_id'] == gid
    assert extra['candidate_id'] == candidate_id_for(gid, 'SPY_C595')
    assert extra['candidate_id'] == best['candidate_id']
    assert extra['rank'] == 0                       # selected sorts first
    assert extra['dte'] == 30
    assert extra['n_considered'] == 4
    assert extra['n_recorded'] == 4
    assert extra['skip_counts'] == {'wrong_type_or_moneyness': 2}

    sizing = json.loads(sel['sizing_json'])
    assert sizing == {'intended_qty': 2}            # strength 3 -> high -> 2
    risk = json.loads(sel['risk_json'])
    assert risk == {'max_budget_per_trade': 1000.0}


# --------------------------------------------------------------------------- #
# idempotency and bounding                                                    #
# --------------------------------------------------------------------------- #

def test_retry_with_same_scan_id_is_idempotent():
    t = make_trader(scan_id='scan-A')
    b1 = t.select_best_option(contracts(), PRICE, 'call')
    b2 = t.select_best_option(contracts(), PRICE, 'call')
    assert b1['group_id'] == b2['group_id']
    assert b1['candidate_id'] == b2['candidate_id']
    assert t.research_capture._ledger.stats()['snapshots'] == 4
    assert t.research_capture.dropped == {}


def test_different_scan_id_makes_a_new_group():
    t1 = make_trader(scan_id='scan-A')
    t2 = make_trader(scan_id='scan-B')
    g1 = t1.select_best_option(contracts(), PRICE, 'call')['group_id']
    g2 = t2.select_best_option(contracts(), PRICE, 'call')['group_id']
    assert g1 != g2


def test_bounded_capture_always_keeps_the_selected_row():
    t = make_trader(max_rows=2)
    best = t.select_best_option(contracts(), PRICE, 'call')
    snaps = snaps_for(t, best['group_id'])
    assert len(snaps) == 2
    by_sym = {s['symbol']: s for s in snaps}
    # selected kept, then best-scored non-selected row; skips dropped first.
    assert set(by_sym) == {'SPY_C595', 'SPY_C590'}
    assert by_sym['SPY_C595']['accepted'] == 1
    extra = json.loads(by_sym['SPY_C595']['extra_json'])
    assert extra['n_considered'] == 4
    assert extra['n_recorded'] == 2


# --------------------------------------------------------------------------- #
# join path: gate verdicts / accepted candidate link back to the group        #
# --------------------------------------------------------------------------- #

def test_candidate_extra_carries_the_group_link():
    t = make_trader()
    best = t.select_best_option(contracts(), PRICE, 'call')
    extra = SmartOptionsTrader._candidate_extra(best)
    assert extra['scan_id'] == 'scan-A'
    assert extra['group_id'] == best['group_id']
    assert extra['candidate_id'] == best['candidate_id']
