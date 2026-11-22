"""Matched candidates include limit misses and stop overshoot."""
from datetime import datetime, timedelta, timezone

from exit_replay import QuoteTick
from oracle.ml.paired_execution import compare_candidate, run_study


START = datetime(2026, 11, 20, 14, 30, tzinfo=timezone.utc)


def tick(sec, bid, ask, size=10, age=0):
    return QuoteTick(ts=(START + timedelta(seconds=sec)).isoformat(),
                     bid=bid, ask=ask, bid_size=size, ask_size=size,
                     quote_age_s=age)


def candidate(**kwargs):
    result = {"decision_id": "d", "symbol": "SPY_OPTION",
              "decision_time": START.isoformat(), "qty": 1,
              "gross_expected_value": 10.0,
              "limit_expiry_s": 3.0,
              "levels": {"take_profit_percent": 20,
                         "stop_loss_percent": 10}}
    result.update(kwargs)
    return result


def test_limit_quote_touch_and_identical_exit_rules():
    result = compare_candidate(candidate(), [tick(0, 1, 1.10),
                                            tick(1, 1, 1.10),
                                            tick(2, .99, 1.05),
                                            tick(3, 1.5, 1.6)])
    assert result["market"]["entry"]["status"] == "filled"
    assert result["limit"]["entry"]["status"] == "quote_touch_scenario"
    assert result["paired_after_cost_delta"] is not None
    assert result["limit"]["after_cost_pnl"] > result["market"]["after_cost_pnl"]


def test_unfilled_limit_is_a_missed_trade_not_a_winner():
    result = compare_candidate(candidate(gross_expected_value=0),
                               [tick(0, 1, 1.1), tick(1, 1, 1.1),
                                tick(2, 1, 1.1), tick(3, 1, 1.1),
                                tick(4, 1, 1.1), tick(5, .5, .6)])
    assert result["limit"]["entry"]["status"] == "missed_trade"
    assert result["limit"]["after_cost_pnl"] == 0
    assert result["market"]["exit"]["gap_through"] is True
    assert result["market"]["after_cost_pnl"] < 0


def test_sparse_quotes_cannot_certify_limit_miss():
    result = compare_candidate(candidate(), [tick(0, 1, 1.1),
                                            tick(1, 1, 1.1), tick(40, 1.5, 1.6)],
                               max_gap_s=1.0)
    assert result["limit"]["entry"]["status"] == "unresolved_quote_coverage"
    assert result["paired_after_cost_delta"] is None


def test_no_inputs_do_not_fabricate_study(tmp_path):
    result = run_study(str(tmp_path / "missing.jsonl"), str(tmp_path / "quotes"))
    assert result["status"] == "INSUFFICIENT_DATA"
