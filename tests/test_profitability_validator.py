from profitability_validator import DEFAULT_RULES, dry_run_scan, validate_trade


def make_candidate(pass_trade: bool = True):
    if pass_trade:
        return {
            "symbol": "AAPL",
            "entry_price": 1.25,
            "qty": 2,
            "option": {"type": "call", "confidence": 5, "delta": 0.55},
            "dynamic_levels": {"take_profit_percent": 0.25, "stop_loss_percent": 0.15},
            "entry_stamp": {
                "expected_value": 2.5,
                "probability_of_profit": 0.68,
                "ev_per_dollar_risk": 0.02,
            },
            "oracle_probability": {
                "p_call": 0.60,
                "p_put": 0.20,
                "p_no_trade": 0.20,
            },
            "robinhood_book": {"orderbook_imbalance": 0.18},
        }
    return {
        "symbol": "AAPL",
        "entry_price": 1.25,
        "qty": 2,
        "option": {"type": "call", "confidence": 2, "delta": 0.30},
        "dynamic_levels": {"take_profit_percent": 0.25, "stop_loss_percent": 0.15},
        "entry_stamp": {
            "expected_value": -0.5,
            "probability_of_profit": 0.45,
            "ev_per_dollar_risk": -0.004,
        },
        "oracle_probability": {
            "p_call": 0.20,
            "p_put": 0.30,
            "p_no_trade": 0.50,
        },
        "robinhood_book": {"orderbook_imbalance": 0.03},
    }


def test_validator_accepts_strong_trade():
    decision = validate_trade(make_candidate(True), DEFAULT_RULES)
    assert decision["pass"] is True
    assert not decision["reasons"]


def test_validator_blocks_weak_trade():
    decision = validate_trade(make_candidate(False), DEFAULT_RULES)
    assert decision["pass"] is False
    assert decision["reasons"]


def test_dry_run_scan_reports_counts():
    results = dry_run_scan([make_candidate(True), make_candidate(False)])
    assert results["approved"] == 1
    assert results["blocked"] == 1
    assert len(results["results"]) == 2
