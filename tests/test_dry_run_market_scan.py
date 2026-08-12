import json
from pathlib import Path

from dry_run_market_scan import build_candidate_from_option, latest_scan_file


def test_latest_scan_file_prefers_most_recent(tmp_path):
    old = tmp_path / 'option_scan_20251001.json'
    new = tmp_path / 'option_scan_20251010.json'
    old.write_text('[]')
    new.write_text('[]')
    import os
    os.utime(old, (1000, 1000))
    os.utime(new, (2000, 2000))
    assert latest_scan_file(tmp_path) == new


def test_build_candidate_from_option_uses_real_scan_data():
    option = {
        'symbol': 'AAPL  251121C00245000',
        'ticker': 'AAPL',
        'type': 'CALL',
        'score': 88,
        'delta': 0.54,
        'iv': 29.0,
        'days_to_exp': 42,
        'last': 1.44,
    }
    candidate = build_candidate_from_option(option)
    assert candidate['symbol'] == 'AAPL'
    assert candidate['option']['type'] == 'call'
    assert candidate['entry_stamp']['probability_of_profit'] > 0.5
    assert candidate['entry_stamp']['ev_per_dollar_risk'] > -0.05
