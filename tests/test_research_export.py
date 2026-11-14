"""Hermetic tests for the read-only research export (research_export.py).

The module's own --selftest builds a temp-dir ResearchLedger with a
3-candidate group, a filled+resolved decision, a gate block, a cancelled
no-fill, and a production-shaped unresolved close recovered via the
trading-history mark join, then round-trips both JSONL outputs and checks
ranking_eval compatibility. We invoke it here so pytest covers it, plus a
few direct unit checks on the source-label and history-join helpers.
"""
import json

import pytest

import research_export


def test_selftest_passes(capsys):
    assert research_export._selftest() == 0
    out = capsys.readouterr().out
    assert "SELFTEST PASS" in out
    assert "FAIL" not in out.replace("SELFTEST PASS", "")


@pytest.mark.parametrize("source,env,expected", [
    ("test", "paper", "synthetic"),
    ("test", "live", "synthetic"),
    ("live_hook", "paper", "paper_broker_fill"),
    ("reconciler", "paper", "paper_broker_fill"),
    ("resolver", "live", "live_broker_fill"),
    ("", "paper", "paper_unknown"),
    (None, "live", "live_unknown"),
])
def test_fill_source_label(source, env, expected):
    assert research_export.fill_source_label(source, env) == expected


def test_broker_env_default_tracks_alpaca_paper(monkeypatch):
    monkeypatch.setenv("ALPACA_PAPER", "true")
    assert research_export._broker_env_default() == "paper"
    monkeypatch.setenv("ALPACA_PAPER", "false")
    assert research_export._broker_env_default() == "live"
    monkeypatch.delenv("ALPACA_PAPER", raising=False)
    assert research_export._broker_env_default() == "paper"


def test_load_history_index_first_wins_and_skips_unkeyed(tmp_path):
    p = tmp_path / "trading_history.json"
    p.write_text(json.dumps({"trades": [
        {"order_id": "a", "pnl": 1.0},
        {"order_id": "a", "pnl": 2.0},   # duplicate: first wins
        {"pnl": 3.0},                    # no order_id: skipped
        "not-a-dict",
    ]}), encoding="utf-8")
    idx = research_export.load_history_index(str(p))
    assert list(idx) == ["a"]
    assert idx["a"]["pnl"] == 1.0


def test_load_history_index_accepts_bare_list(tmp_path):
    p = tmp_path / "hist.json"
    p.write_text(json.dumps([{"order_id": "x", "pnl": -5.0}]),
                 encoding="utf-8")
    assert research_export.load_history_index(str(p))["x"]["pnl"] == -5.0


def test_resolve_via_history_upgrades_unresolved_with_mark_label():
    base = {"state": "unresolved", "reason": "open_residual_qty",
            "net_pnl_dollars": None, "pnl_source": None}
    hist = {"o1": {"order_id": "o1", "pnl": -10.0, "pnl_percent": -5.0}}
    out = research_export._resolve_via_history(base, ["o1"], hist)
    assert out["state"] == "filled_resolved"
    assert out["net_pnl_dollars"] == -10.0
    assert out["pnl_source"] == "mark"
    assert out["resolved_via"] == "history_order_id_join"
    assert out["history_order_id"] == "o1"
    assert out["ledger_state"] == "unresolved"
    assert out["ledger_reason"] == "open_residual_qty"
    # original outcome dict untouched (merge copies)
    assert base["state"] == "unresolved"


def test_resolve_via_history_leaves_resolved_and_unjoined_alone():
    hist = {"o1": {"order_id": "o1", "pnl": -10.0}}
    resolved = {"state": "filled_resolved", "reason": "reported_outcome",
                "net_pnl_dollars": 5.0, "pnl_source": "reported"}
    assert research_export._resolve_via_history(
        resolved, ["o1"], hist) is resolved
    unjoined = {"state": "unresolved", "reason": "open_residual_qty",
                "net_pnl_dollars": None, "pnl_source": None}
    assert research_export._resolve_via_history(
        unjoined, ["missing"], hist) is unjoined
    assert research_export._resolve_via_history(
        unjoined, ["o1"], None) is unjoined


def test_resolve_via_history_requires_legacy_provenance():
    # a joined history row with null pnl and no terminal no-fill marker
    # must NOT resolve (classify_legacy_record -> unresolved)
    hist = {"o1": {"order_id": "o1", "pnl": None, "status": "whatever"}}
    base = {"state": "unresolved", "reason": "open_residual_qty",
            "net_pnl_dollars": None, "pnl_source": None}
    assert research_export._resolve_via_history(base, ["o1"], hist) is base
