"""
Offline tests for Phase 12 — EV Model Error.

No creds, no network, no broker. Covers:
  1. The signed error math: bias = realized - expected, plus MAE.
  2. All four breakdowns (strategy / CALL-vs-PUT / exit reason / EV bucket).
  3. CALL/PUT/SPREAD classification (explicit fields + OCC symbol parsing).
  4. The verdict thresholds (CALIBRATED / OVER / UNDER / INSUFFICIENT).
  5. Empty store -> clean INSUFFICIENT report carrying the analytics footer.

ev_model_error is ANALYTICS ONLY: it only reads closed records and never opens,
closes, sizes, prices, blocks or alters any trade, and never reaches the network.
Records are injected, so the whole module is deterministic.
"""

import unittest

import ev_model_error as eme
from ev_model_error import (
    compute_ev_model_error, format_ev_model_error, generate_ev_model_error_text,
    classify_bias, load_ev_error_records,
    VERDICT_CALIBRATED, VERDICT_OVERPREDICTS, VERDICT_UNDERPREDICTS,
    VERDICT_INSUFFICIENT, MIN_TRADES,
)
from ev_attribution import ANALYTICS_FOOTER, EV_BUCKETS


def _rec(strategy, ev, pnl, exit_reason="take_profit", symbol="SPY",
         max_loss=100.0, **extra):
    r = {"strategy": strategy, "expected_value": ev, "pnl": pnl,
         "max_loss": max_loss, "exit_reason": exit_reason, "symbol": symbol}
    r.update(extra)
    return r


class TestErrorMath(unittest.TestCase):
    def test_bias_is_realized_minus_expected(self):
        # expected 10 each, realized 4 and 16 -> avg realized 10, bias 0.
        recs = [_rec("s", 10.0, 4.0), _rec("s", 10.0, 16.0)]
        rep = compute_ev_model_error(records=recs)
        o = rep["overall"]
        self.assertEqual(o["trades"], 2)
        self.assertEqual(o["avg_expected_ev"], 10.0)
        self.assertEqual(o["avg_realized_pnl"], 10.0)
        self.assertEqual(o["bias"], 0.0)

    def test_mean_abs_error(self):
        recs = [_rec("s", 10.0, 4.0), _rec("s", 10.0, 16.0)]  # errors -6, +6
        rep = compute_ev_model_error(records=recs)
        self.assertEqual(rep["overall"]["mean_abs_error"], 6.0)

    def test_negative_bias_means_overprediction(self):
        # model promised 20 each, realized far less -> strongly negative bias.
        recs = [_rec("s", 20.0, -10.0) for _ in range(MIN_TRADES)]
        rep = compute_ev_model_error(records=recs)
        self.assertEqual(rep["verdict"], VERDICT_OVERPREDICTS)

    def test_positive_bias_means_underprediction(self):
        recs = [_rec("s", 5.0, 60.0) for _ in range(MIN_TRADES)]
        rep = compute_ev_model_error(records=recs)
        self.assertEqual(rep["verdict"], VERDICT_UNDERPREDICTS)

    def test_calibrated_when_within_tolerance(self):
        recs = [_rec("s", 40.0, 44.0) for _ in range(MIN_TRADES)]  # bias +4
        rep = compute_ev_model_error(records=recs)
        self.assertEqual(rep["verdict"], VERDICT_CALIBRATED)


class TestVerdictEdges(unittest.TestCase):
    def test_classify_bias_tolerance(self):
        self.assertEqual(classify_bias(40.0, 5.0), VERDICT_CALIBRATED)
        self.assertEqual(classify_bias(40.0, -30.0), VERDICT_OVERPREDICTS)
        self.assertEqual(classify_bias(40.0, 30.0), VERDICT_UNDERPREDICTS)

    def test_classify_bias_none_is_insufficient(self):
        self.assertEqual(classify_bias(None, 5.0), VERDICT_INSUFFICIENT)
        self.assertEqual(classify_bias(40.0, None), VERDICT_INSUFFICIENT)

    def test_small_sample_is_insufficient(self):
        recs = [_rec("s", 10.0, 20.0)]  # 1 < MIN_TRADES
        rep = compute_ev_model_error(records=recs)
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)


class TestBreakdowns(unittest.TestCase):
    def setUp(self):
        self.recs = [
            _rec("bullish_put_credit_spread", 12.0, 20.0, "take_profit"),
            _rec("bullish_put_credit_spread", 8.0, -50.0, "stop_loss"),
            _rec("single_leg", 5.0, 5.0, "take_profit",
                 symbol="COST260717C00930000"),
            _rec("single_leg", 5.0, -10.0, "stop_loss",
                 symbol="AVGO260717P00410000"),
        ]
        self.rep = compute_ev_model_error(records=self.recs)

    def test_four_breakdowns_present(self):
        for key in ("by_strategy", "by_call_put", "by_exit_reason",
                    "by_ev_bucket"):
            self.assertIn(key, self.rep)

    def test_call_put_spread_classification(self):
        cp = self.rep["by_call_put"]
        self.assertIn("CALL", cp)   # from OCC ...C00930000
        self.assertIn("PUT", cp)    # from OCC ...P00410000
        self.assertIn("SPREAD", cp)  # strategy name contains "spread"

    def test_call_put_from_explicit_field(self):
        recs = [_rec("single_leg", 5.0, 7.0, option_type="call"),
                _rec("single_leg", 5.0, -3.0, option_type="put")]
        cp = compute_ev_model_error(records=recs)["by_call_put"]
        self.assertIn("CALL", cp)
        self.assertIn("PUT", cp)

    def test_strategy_breakdown_groups(self):
        bs = self.rep["by_strategy"]
        self.assertIn("bullish_put_credit_spread", bs)
        self.assertIn("single_leg", bs)
        self.assertEqual(bs["bullish_put_credit_spread"]["trades"], 2)

    def test_exit_reason_breakdown(self):
        er = self.rep["by_exit_reason"]
        self.assertIn("take_profit", er)
        self.assertIn("stop_loss", er)

    def test_ev_bucket_breakdown_uses_shared_buckets(self):
        eb = self.rep["by_ev_bucket"]
        for label, _, _ in EV_BUCKETS:
            self.assertIn(label, eb)


class TestLoadAndFormat(unittest.TestCase):
    def test_load_filters_records_without_ev_or_pnl(self):
        recs = [
            _rec("s", 10.0, 5.0),                 # kept
            {"strategy": "s", "pnl": 5.0},        # no EV -> dropped
            {"strategy": "s", "expected_value": 10.0},  # no PnL -> dropped
        ]
        kept = load_ev_error_records(records=recs)
        self.assertEqual(len(kept), 1)

    def test_empty_store_clean_report(self):
        rep = compute_ev_model_error(records=[])
        self.assertEqual(rep["sample_size"], 0)
        self.assertEqual(rep["verdict"], VERDICT_INSUFFICIENT)
        text = format_ev_model_error(rep)
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn(VERDICT_INSUFFICIENT, text)

    def test_format_never_raises_with_data(self):
        recs = [_rec("s", 10.0, 20.0) for _ in range(MIN_TRADES)]
        text = format_ev_model_error(compute_ev_model_error(records=recs))
        self.assertIn(ANALYTICS_FOOTER, text)
        self.assertIn("EV Model Error", text)

    def test_generate_text_smoke(self):
        # No injected records -> reads disk fail-open; must not raise.
        text = generate_ev_model_error_text()
        self.assertIn(ANALYTICS_FOOTER, text)


class TestSelfTest(unittest.TestCase):
    def test_module_self_test_passes(self):
        self.assertEqual(eme._self_test(), 0)


if __name__ == "__main__":
    unittest.main()
