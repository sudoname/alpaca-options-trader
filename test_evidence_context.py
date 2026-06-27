"""Unit tests for evidence_context.compute_evidence (pure, no network)."""

import evidence_context as ec
import oracle_agents
import oracle_regime
from oracle.signals import candlestick_patterns as cs


def test_self_test_passes():
    assert ec._self_test() == 0


def test_iv_bucket_thresholds():
    assert ec.iv_bucket(0.10) == "low"
    assert ec.iv_bucket(0.29) == "low"
    assert ec.iv_bucket(0.30) == "medium"
    assert ec.iv_bucket(0.59) == "medium"
    assert ec.iv_bucket(0.60) == "high"
    assert ec.iv_bucket(40.0) == "medium"      # 0-100 scale tolerated
    assert ec.iv_bucket(85) == "high"
    assert ec.iv_bucket(None) is None
    assert ec.iv_bucket("garbage") is None


def test_full_context_shape():
    downtrend = [(c + 1, c + 1.2, c - 1, c, 100) for c in (110, 108, 106, 104)]
    hammer = (100.0, 100.5, 95.0, 100.2, 120)
    ev = ec.compute_evidence({
        "regime": "trending", "trend": "up", "momentum": 0.08,
        "realized_vol": 0.18, "vix": 16.0, "news_score": 0.5, "breadth": 0.4,
        "iv_rank": 40.0, "signal_strength": 3, "dte": 30, "delta": 0.45,
        "candles": downtrend + [hammer], "iv": 0.35, "hv": 0.20,
    })
    assert len(ev["agent_votes"]) == len(oracle_agents.AGENTS)
    assert ev["regime_label"] == oracle_regime.TRENDING_BULL
    assert ev["candlestick_pattern"] == cs.HAMMER
    assert ev["pattern"] == "hammer"
    assert ev["direction"] == "up"
    assert ev["iv_bucket"] == "medium"
    assert ev["iv_vs_hv"] == "overpriced"


def test_empty_and_garbage_never_raise():
    for junk in (None, {}, 42, "x", [], {"weird": object()}):
        ev = ec.compute_evidence(junk)
        assert "agent_votes" in ev
        assert "regime_label" in ev
        assert "iv_bucket" in ev


def test_deterministic():
    ctx = {"trend": "up", "momentum": 0.05, "iv_rank": 0.7}
    assert ec.compute_evidence(ctx) == ec.compute_evidence(ctx)
