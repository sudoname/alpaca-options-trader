"""Unit tests for daily_report_v2 -- the consolidated 5-section daily report.

Pure aggregation/formatting tests over injected rows; no network, no broker.
"""
import json
import os
import tempfile
from datetime import datetime

import pytest

import daily_report_v2 as drv2
import evidence_attribution as ea


# --------------------------------------------------------------------------- #
# 1) Trading stats
# --------------------------------------------------------------------------- #
def _rows():
    return [
        {"pnl": 100.0, "pnl_percent": 20.0, "direction": "up", "outcome": "win"},
        {"pnl": -40.0, "pnl_percent": -8.0, "direction": "down", "outcome": "loss"},
        {"pnl": 60.0, "pnl_percent": 12.0, "direction": "up", "outcome": "win"},
    ]


def test_trading_stats_core_math():
    t = drv2.trading_stats(_rows())
    assert t["trades"] == 3
    assert abs(t["win_rate"] - 2 / 3) < 1e-9
    assert t["total_realized"] == 120.0
    assert abs(t["ev_per_trade"] - 40.0) < 1e-9
    assert t["largest_winner"] == 100.0
    assert t["largest_loser"] == -40.0
    # profit factor = gross_win(160) / gross_loss(40) = 4.0
    assert abs(t["profit_factor"] - 4.0) < 1e-9
    assert abs(t["avg_gain"] - 80.0) < 1e-9
    assert abs(t["avg_loss"] - (-40.0)) < 1e-9


def test_trading_stats_empty_never_raises():
    t = drv2.trading_stats([])
    assert t["trades"] == 0
    assert t["win_rate"] is None
    assert t["profit_factor"] is None
    assert t["total_realized"] is None


def test_trading_stats_no_losses_profit_factor_none():
    t = drv2.trading_stats([{"pnl": 10.0, "pnl_percent": 1.0}])
    # gross_loss == 0 -> profit_factor None (avoid div-by-zero)
    assert t["profit_factor"] is None
    assert t["avg_loss"] is None


# --------------------------------------------------------------------------- #
# 4/5) Best / worst evidence
# --------------------------------------------------------------------------- #
def _row(feature, trades, ev, verdict="High"):
    return ea.EvidenceRow(feature=feature, trades=trades, avg_return_pct=ev,
                          win_rate=0.5, ev=ev, verdict=verdict)


def test_best_and_worst_evidence_picks_extremes():
    tables = {
        "agent": [_row("TrendAgent", 50, 7.5), _row("MeanRev", 40, -3.0)],
        "pattern": [_row("BullFlag", 30, 9.0)],
        # below MIN_SAMPLES -> ignored even though EV is extreme
        "regime": [_row("Tiny", 1, 99.0), _row("Bad", 1, -99.0)],
    }
    res = drv2._best_and_worst_evidence(tables)
    assert res["lean_into"]["feature"] == "BullFlag"
    assert res["lean_into"]["ev"] == 9.0
    assert res["avoid"]["feature"] == "MeanRev"
    assert res["avoid"]["ev"] == -3.0


def test_best_and_worst_evidence_empty():
    res = drv2._best_and_worst_evidence({})
    assert res["lean_into"] is None
    assert res["avoid"] is None


def test_ev_signals_flags_confident_signs():
    tables = {
        "agent": [
            ea.EvidenceRow("StrongAgent", 40, 8.0, 0.62, 8.0, ea.V_STRONG),
            ea.EvidenceRow("HighAgent", 40, 3.0, 0.55, 3.0, ea.V_HIGH),
            ea.EvidenceRow("BadAgent", 40, -3.0, 0.40, -3.0, ea.V_NEGATIVE),
            # Low / Weak verdicts are not confident enough -> excluded.
            ea.EvidenceRow("MehAgent", 40, 0.1, 0.50, 0.1, ea.V_LOW),
            ea.EvidenceRow("TinyAgent", 2, 9.0, 0.99, 9.0, ea.V_STRONG),
        ],
    }
    sigs = drv2.ev_signals(tables)
    assert sigs["n_positive"] == 2
    assert sigs["n_negative"] == 1
    # positive sorted by EV desc
    assert sigs["positive"][0]["feature"] == "StrongAgent"
    assert sigs["positive"][1]["feature"] == "HighAgent"
    assert sigs["negative"][0]["feature"] == "BadAgent"
    # below MIN_SAMPLES excluded despite Strong verdict
    feats = {s["feature"] for s in sigs["positive"]}
    assert "TinyAgent" not in feats


def test_ev_signals_empty():
    sigs = drv2.ev_signals({})
    assert sigs == {"positive": [], "negative": [],
                    "n_positive": 0, "n_negative": 0}


# --------------------------------------------------------------------------- #
# Report assembly (injected -> no network)
# --------------------------------------------------------------------------- #
def test_build_consolidated_report_injected():
    now = datetime(2026, 6, 26, 16, 15, 0)
    report = drv2.build_consolidated_report(
        now=now, rows=_rows(), portfolio={}, execution={})
    assert report["date"] == "2026-06-26"
    for section in ("trading", "portfolio", "execution", "learning", "confidence"):
        assert section in report
    assert report["trading"]["trades"] == 3
    assert "_tables" in report


def test_format_consolidated_report_has_five_sections():
    report = drv2.build_consolidated_report(
        rows=_rows(), portfolio={}, execution={})
    md = drv2.format_consolidated_report(report)
    assert "### 1. Trading" in md
    assert "### 2. Portfolio" in md
    assert "### 3. Execution" in md
    assert "### 4. Learning" in md
    assert "### 5. Confidence" in md
    assert drv2.ANALYTICS_FOOTER in md


def test_format_never_raises_on_empty():
    report = drv2.build_consolidated_report(rows=[], portfolio={}, execution={})
    md = drv2.format_consolidated_report(report)
    assert isinstance(md, str) and len(md) > 0


# --------------------------------------------------------------------------- #
# Serialization + artifact
# --------------------------------------------------------------------------- #
def test_json_safe_drops_tables():
    report = drv2.build_consolidated_report(
        rows=_rows(), portfolio={}, execution={})
    safe = drv2._json_safe(report)
    assert "_tables" not in safe
    # round-trips through json
    json.dumps(safe, default=str)


def test_write_dated_artifact(tmp_path):
    now = datetime(2026, 6, 26, 16, 15, 0)
    report = drv2.build_consolidated_report(
        now=now, rows=_rows(), portfolio={}, execution={})
    paths = drv2.write_dated_artifact(report, out_dir=str(tmp_path))
    assert os.path.exists(paths["markdown"])
    assert os.path.exists(paths["json"])
    assert paths["markdown"].endswith("daily_report_20260626.md")
    with open(paths["json"], encoding="utf-8") as fh:
        data = json.load(fh)
    assert data["trading"]["trades"] == 3
    assert "_tables" not in data


# --------------------------------------------------------------------------- #
# Self-test entry point
# --------------------------------------------------------------------------- #
def test_self_test_passes():
    assert drv2._self_test() == 0
