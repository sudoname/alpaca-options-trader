import json

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


# --------------------------------------------------------------------------- #
# session_bias rule (veto-only, default OFF)                                   #
# --------------------------------------------------------------------------- #

# Shape produced by SmartOptionsTrader._load_session_bias from the live
# measured artifact (session_signal_eval_live.json, n=2,775 decisions).
LIVE_SESSION_RULE = {
    "call_bp": -12.55, "call_t": -4.05,   # CALL rest-of-day: negative, |t|>2
    "put_bp": 0.60, "put_t": 0.17,        # PUT rest-of-day: flat, insignificant
    "leverage": 20.0, "min_abs_t": 2.0,
}


def make_put_candidate():
    c = make_candidate(True)
    c["option"] = {**c["option"], "type": "put", "delta": -0.55}
    c["oracle_probability"] = {"p_call": 0.20, "p_put": 0.60, "p_no_trade": 0.20}
    return c


class TestSessionBiasDefaultOff:
    def test_default_rules_have_no_session_bias(self):
        assert DEFAULT_RULES["session_bias"] is None

    def test_default_verdict_and_zero_drag(self):
        d = validate_trade(make_candidate(True))
        assert d["pass"] is True
        assert d["summary"]["session_bias_drag"] == 0.0


class TestSessionBiasVeto:
    def test_call_blocked_by_raised_bar(self):
        # drag = 12.55/1e4*20 = 0.0251 -> bar 0.008+0.0251=0.0331 > ev/$ 0.02
        d = validate_trade(make_candidate(True), {"session_bias": LIVE_SESSION_RULE})
        assert d["pass"] is False
        assert d["summary"]["session_bias_drag"] == abs(
            LIVE_SESSION_RULE["call_bp"]) / 1e4 * 20.0
        assert any(r.startswith("session_bias:") for r in d["reasons"])

    def test_high_ev_call_clears_raised_bar(self):
        c = make_candidate(True)
        c["entry_stamp"]["ev_per_dollar_risk"] = 0.05
        d = validate_trade(c, {"session_bias": LIVE_SESSION_RULE})
        assert d["pass"] is True

    def test_put_not_penalized_insignificant_drift(self):
        d = validate_trade(make_put_candidate(), {"session_bias": LIVE_SESSION_RULE})
        assert d["summary"]["session_bias_drag"] == 0.0
        assert not any(r.startswith("session_bias:") for r in d["reasons"])

    def test_positive_drift_never_penalized(self):
        rule = {**LIVE_SESSION_RULE, "call_bp": 12.55, "call_t": 4.05}
        d = validate_trade(make_candidate(True), {"session_bias": rule})
        assert d["pass"] is True
        assert d["summary"]["session_bias_drag"] == 0.0

    def test_insignificant_negative_drift_not_penalized(self):
        rule = {**LIVE_SESSION_RULE, "call_t": -1.5}
        d = validate_trade(make_candidate(True), {"session_bias": rule})
        assert d["pass"] is True

    def test_veto_only_never_rescues_failed_trade(self):
        d = validate_trade(make_candidate(False), {"session_bias": LIVE_SESSION_RULE})
        assert d["pass"] is False


class TestSessionBiasFailOpen:
    def test_malformed_configs_leave_legacy_verdict(self):
        for junk in ("garbage", 42, [1], {"call_bp": "x", "call_t": None}, {}):
            d = validate_trade(make_candidate(True), {"session_bias": junk})
            assert d["pass"] is True
            assert d["summary"]["session_bias_drag"] == 0.0

    def test_missing_ev_per_dollar_risk_no_crash(self):
        c = make_candidate(True)
        del c["entry_stamp"]["ev_per_dollar_risk"]
        d = validate_trade(c, {"session_bias": LIVE_SESSION_RULE})
        assert not any(r.startswith("session_bias:") for r in d["reasons"])


class TestSessionBiasLoader:
    """_load_session_bias builds the rule from the eval artifact (fail-open)."""

    def _artifact(self, tmp_path, call_mean=-0.001255, call_se=0.00031,
                  put_mean=0.00006, put_se=0.000349):
        obj = {
            "by_direction": {
                "long (CALL)": {"intraday_net": {
                    "n": 1639, "mean": call_mean, "se_mean": call_se,
                    "pos_freq": 0.45}},
                "short (PUT)": {"intraday_net": {
                    "n": 1136, "mean": put_mean, "se_mean": put_se,
                    "pos_freq": 0.50}},
            }
        }
        p = tmp_path / "eval.json"
        p.write_text(json.dumps(obj))
        return str(p)

    def test_loader_builds_rule(self, tmp_path):
        from smart_trader import SmartOptionsTrader
        rule = SmartOptionsTrader._load_session_bias(
            self._artifact(tmp_path), leverage=20.0, min_abs_t=2.0)
        assert rule is not None
        assert rule["leverage"] == 20.0 and rule["min_abs_t"] == 2.0
        assert abs(rule["call_bp"] - (-12.55)) < 1e-6
        assert abs(rule["call_t"] - (-0.001255 / 0.00031)) < 1e-9
        assert rule["put_bp"] > 0
        # End-to-end: the loaded rule blocks the marginal CALL candidate.
        d = validate_trade(make_candidate(True), {"session_bias": rule})
        assert d["pass"] is False

    def test_loader_fail_open_missing_file(self, tmp_path):
        from smart_trader import SmartOptionsTrader
        assert SmartOptionsTrader._load_session_bias(
            str(tmp_path / "nope.json")) is None

    def test_loader_fail_open_malformed(self, tmp_path):
        from smart_trader import SmartOptionsTrader
        p = tmp_path / "bad.json"
        p.write_text("{not json")
        assert SmartOptionsTrader._load_session_bias(str(p)) is None
        p2 = tmp_path / "empty.json"
        p2.write_text(json.dumps({"by_direction": {}}))
        assert SmartOptionsTrader._load_session_bias(str(p2)) is None
