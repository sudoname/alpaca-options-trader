"""Hermetic tests for the read-only research export (research_export.py).

The module's own --selftest builds a temp-dir ResearchLedger with a
3-candidate group, a filled+resolved decision, a gate block, and a
cancelled no-fill, then round-trips both JSONL outputs and checks
ranking_eval compatibility. We invoke it here so pytest covers it, plus a
few direct unit checks on the source-label helpers.
"""
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
