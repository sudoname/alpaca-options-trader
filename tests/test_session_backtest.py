"""Offline tests for session_backtest (synthetic fixtures only)."""

import math

import pandas as pd
import pytest

from session_backtest import (
    ANN_SESSIONS,
    chrono_splits,
    equal_weight,
    fit_threshold,
    metrics,
    net_leg,
    select_symbols,
)


class TestSplits:
    def test_chronological_with_embargo(self):
        s = chrono_splits(100, 0.6, 0.2, embargo=1)
        assert s["train"] == (0, 60)
        assert s["val"] == (61, 80)   # one purged session after train
        assert s["test"] == (81, 100)  # one purged session after val
        # no overlap anywhere
        assert s["train"][1] < s["val"][0] < s["val"][1] < s["test"][0]

    def test_session_labels_span_one_session_so_embargo_one_purges(self):
        # documented invariant: overnight/intraday labels never span >1 session
        s = chrono_splits(10, embargo=1)
        assert s["val"][0] - s["train"][1] == 1


class TestCostsAndMetrics:
    def test_net_leg_two_sides(self):
        gross = pd.Series([0.001, 0.002])
        net = net_leg(gross, cost_bps=5.0)  # 2 x 5bp = 10bp
        assert net.iloc[0] == pytest.approx(0.001 - 0.001)
        assert net.iloc[1] == pytest.approx(0.002 - 0.001)

    def test_metrics_known_values(self):
        s = pd.Series([0.01, -0.01, 0.02, 0.0])
        m = metrics(s)
        assert m["n_sessions"] == 4
        assert m["mean_per_session"] == pytest.approx(0.005)
        assert m["ann_mean"] == pytest.approx(0.005 * ANN_SESSIONS)
        eq = (1 + s).cumprod()
        assert m["cum_net_return"] == pytest.approx(eq.iloc[-1] - 1)
        assert m["max_drawdown"] == pytest.approx(0.99 * 1.01 / 1.01 - 1)  # -1%
        assert m["sharpe_net"] == pytest.approx(
            0.005 / s.std(ddof=1) * math.sqrt(ANN_SESSIONS)
        )

    def test_empty_metrics(self):
        assert metrics(pd.Series(dtype=float)) == {"n": 0}


class TestSelectionLeakage:
    def _overnight(self):
        idx = pd.RangeIndex(100)
        good = pd.Series([0.002] * 50 + [0.001] * 50, index=idx)   # strong +
        bad = pd.Series([-0.002] * 50 + [-0.001] * 50, index=idx)  # strong -
        # add tiny noise so std > 0
        good = good + pd.Series([1e-5 * ((i % 3) - 1) for i in idx], index=idx)
        bad = bad + pd.Series([1e-5 * ((i % 3) - 1) for i in idx], index=idx)
        return {"GOOD": good, "BAD": bad}

    def test_select_reads_only_train_range(self):
        on = self._overnight()
        base = select_symbols(on, (0, 60), threshold=1.0)
        assert base == ["GOOD"]
        # mutate everything OUTSIDE train: selection must not change
        on["BAD"].iloc[60:] = 10.0   # would make BAD look amazing if leaked
        on["GOOD"].iloc[60:] = -10.0
        assert select_symbols(on, (0, 60), threshold=1.0) == base

    def test_fit_threshold_never_reads_test_segment(self):
        on = self._overnight()
        splits = chrono_splits(100, 0.6, 0.2, embargo=1)
        th1, syms1, _ = fit_threshold(on, splits)
        # mutate ONLY the test segment
        lo = splits["test"][0]
        for s in on.values():
            s.iloc[lo:] = 99.0
        th2, syms2, _ = fit_threshold(on, splits)
        assert th1 == th2
        assert syms1 == syms2


class TestPortfolio:
    def test_equal_weight_skips_nan(self):
        a = pd.Series([0.01, float("nan")], index=[0, 1])
        b = pd.Series([0.03, 0.02], index=[0, 1])
        port = equal_weight({"A": a, "B": b}, ["A", "B"])
        assert port.iloc[0] == pytest.approx(0.02)
        assert port.iloc[1] == pytest.approx(0.02)  # only B contributes

    def test_equal_weight_empty(self):
        assert len(equal_weight({}, ["A"])) == 0
