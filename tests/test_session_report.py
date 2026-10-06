"""Offline tests for session_report + daily-report integration.

Proves: default OFF -> report output identical to pre-feature behavior;
fail-open on missing artifacts; section renders from synthetic artifacts.
"""

import json
import os
from datetime import datetime

import pytest

import session_report as sr


class TestFlagDefaultOff:
    def test_disabled_by_default(self):
        assert sr.session_report_enabled(env={}) is False
        assert sr.compute_session_summary(env={}) == {}

    def test_enabled_values(self):
        for v in ("1", "true", "YES", "on"):
            assert sr.session_report_enabled(env={"ENABLE_SESSION_REPORT": v})
        for v in ("0", "false", "", "off"):
            assert not sr.session_report_enabled(env={"ENABLE_SESSION_REPORT": v})

    def test_daily_report_unchanged_when_disabled(self, monkeypatch):
        """With the flag unset the daily report text must be byte-identical
        to a report built with session explicitly empty (pre-feature shape)."""
        monkeypatch.delenv("ENABLE_SESSION_REPORT", raising=False)
        import oracle_daily_report as odr
        from oracle_analytics import AnalyticsConfig

        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/t.json",
            spread_positions_file="/nonexistent/p.json",
            expected_move_file="/nonexistent/em.csv",
            training_dataset_file="/nonexistent/ds.csv",
        )
        now = datetime(2026, 1, 2, 16, 30)
        rep = odr.build_daily_report(cfg, now=now, candlestick={})
        assert rep["session"] == {}
        txt_default = odr.format_daily_report(rep)
        rep_pre = odr.build_daily_report(cfg, now=now, candlestick={}, session={})
        assert odr.format_daily_report(rep_pre) == txt_default
        assert "Session Analysis" not in txt_default


class TestFailOpen:
    def test_missing_artifacts_listed_not_raised(self, tmp_path):
        env = {
            "ENABLE_SESSION_REPORT": "1",
            "SESSION_STATS_JSON": str(tmp_path / "none1.json"),
            "SESSION_EVAL_JSON": str(tmp_path / "none2.json"),
            "SESSION_BACKTEST_JSON": str(tmp_path / "none3.json"),
        }
        s = sr.compute_session_summary(env=env)
        assert len(s["missing"]) == 3
        txt = sr.format_session_section(s)
        assert "missing" in txt

    def test_malformed_artifact_fails_open(self, tmp_path):
        bad = tmp_path / "bad.json"
        bad.write_text("{not json")
        env = {
            "ENABLE_SESSION_REPORT": "1",
            "SESSION_STATS_JSON": str(bad),
            "SESSION_EVAL_JSON": str(bad),
            "SESSION_BACKTEST_JSON": str(bad),
        }
        s = sr.compute_session_summary(env=env)
        assert len(s["missing"]) == 3


class TestRendering:
    def _artifacts(self, tmp_path):
        ev = {
            "meta": {"decisions": 10, "date_range": ["2026-06-10", "2026-10-12"],
                     "cost_bps_per_side": 1.0, "generated_at": "x"},
            "by_direction": {
                "long (CALL)": {
                    "intraday_net": {"n": 6, "mean": -0.001, "se_mean": 0.0003, "pos_freq": 0.4},
                    "overnight_net": {"n": 6, "mean": 0.0002, "se_mean": 0.0004, "pos_freq": 0.5},
                },
                "short (PUT)": {
                    "intraday_net": {"n": 4, "mean": 0.0, "se_mean": 0.0003, "pos_freq": 0.5},
                    "overnight_net": {"n": 4, "mean": -0.002, "se_mean": 0.0004, "pos_freq": 0.3},
                },
            },
            "flag_counts": {},
        }
        bt = {
            "meta": {"test_dates": ["2026-05-12", "2026-10-02"], "cost_bps_per_side": 1.0,
                     "options_status": "UNTESTED - underlying prices only"},
            "test_results_net": {
                "SPY_overnight_only": {"ann_mean": 0.15, "sharpe_net": 1.9,
                                       "max_drawdown": -0.02, "n_sessions": 100},
            },
        }
        stats = {
            "meta": {"start": "2024-10-01", "end": "2026-10-03"},
            "symbols": {
                "AAA": {"full_sample": {"overnight": {"n": 5, "mean": 0.002},
                                        "intraday": {"n": 5, "mean": 0.001}}},
                "BBB": {"full_sample": {"overnight": {"n": 5, "mean": -0.002},
                                        "intraday": {"n": 5, "mean": 0.001}}},
            },
        }
        paths = {}
        for name, obj in (("stats", stats), ("eval", ev), ("bt", bt)):
            p = tmp_path / f"{name}.json"
            p.write_text(json.dumps(obj))
            paths[name] = str(p)
        return paths

    def test_section_renders_real_shapes(self, tmp_path):
        paths = self._artifacts(tmp_path)
        env = {
            "ENABLE_SESSION_REPORT": "1",
            "SESSION_STATS_JSON": paths["stats"],
            "SESSION_EVAL_JSON": paths["eval"],
            "SESSION_BACKTEST_JSON": paths["bt"],
        }
        s = sr.compute_session_summary(env=env)
        assert s["universe"]["overnight_gt_intraday"] == 1
        assert s["universe"]["n_symbols"] == 2
        assert s["signal_sessions"]["PUT (short)"]["overnight_net"]["mean_bp"] == pytest.approx(-20.0)
        txt = sr.format_session_section(s)
        assert "Session Analysis" in txt
        assert "SPY_overnight_only" in txt
        assert "UNTESTED" in txt
        assert "No decisions are changed" in txt

    def test_enabled_section_appears_in_daily_report(self, tmp_path, monkeypatch):
        paths = self._artifacts(tmp_path)
        monkeypatch.setenv("ENABLE_SESSION_REPORT", "1")
        monkeypatch.setenv("SESSION_STATS_JSON", paths["stats"])
        monkeypatch.setenv("SESSION_EVAL_JSON", paths["eval"])
        monkeypatch.setenv("SESSION_BACKTEST_JSON", paths["bt"])
        import oracle_daily_report as odr
        from oracle_analytics import AnalyticsConfig

        cfg = AnalyticsConfig(
            spread_trades_file="/nonexistent/t.json",
            spread_positions_file="/nonexistent/p.json",
            expected_move_file="/nonexistent/em.csv",
            training_dataset_file="/nonexistent/ds.csv",
        )
        txt = odr.generate_daily_report_text(cfg, now=datetime(2026, 1, 2), candlestick={})
        assert "Session Analysis" in txt
