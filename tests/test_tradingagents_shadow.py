"""Offline tests for tradingagents_shadow (no network, no tradingagents).

The graph is injected (duck-typed FakeGraph), mirroring the trader_factory
convention: tests exercise the ledger, fail-open, label normalization, the
forward-return join, and the pre-registered evidence-gate math.
"""

import json
import os

import tradingagents_shadow as tas
from tradingagents_shadow import (
    BUY,
    GATE_MIN_N,
    HOLD,
    ROW_TYPE,
    SELL,
    TAShadowConfig,
    UNKNOWN,
    api_key_missing_reason,
    evaluate,
    forward_returns_for_rows,
    load_rows,
    normalize_label,
    scan_once,
)


# --------------------------------------------------------------------------- #
# Fakes
# --------------------------------------------------------------------------- #
class FakeGraph:
    def __init__(self, decision="BUY", fail_symbols=()):
        self.calls = []
        self.decision = decision
        self.fail_symbols = set(fail_symbols)

    def propagate(self, symbol, day):
        self.calls.append((symbol, day))
        if symbol in self.fail_symbols:
            raise RuntimeError("llm exploded")
        state = {"final_trade_decision": f"verdict: {self.decision}"}
        return state, self.decision


def make_cfg(tmp_path, **kw):
    defaults = dict(
        enabled=True,
        universe=["SPY", "QQQ", "AAPL"],
        ledger_file=str(tmp_path / "ta_ledger.jsonl"),
        eval_file=str(tmp_path / "ta_eval.json"),
    )
    defaults.update(kw)
    return TAShadowConfig(**defaults)


# --------------------------------------------------------------------------- #
# Label normalization
# --------------------------------------------------------------------------- #
class TestNormalizeLabel:
    def test_plain_labels(self):
        assert normalize_label("BUY") == BUY
        assert normalize_label("sell") == SELL
        assert normalize_label("Hold") == HOLD

    def test_embedded_label(self):
        assert normalize_label("FINAL TRANSACTION PROPOSAL: **SELL**") == SELL

    def test_first_occurrence_wins(self):
        # Pre-registered simple rule: earliest label in the text wins.
        assert normalize_label("HOLD for now, maybe BUY later") == HOLD

    def test_garbage_and_none(self):
        assert normalize_label("no idea") == UNKNOWN
        assert normalize_label(None) == UNKNOWN
        assert normalize_label(123) == UNKNOWN


# --------------------------------------------------------------------------- #
# Scan: flag-gated, append-only, fail-open
# --------------------------------------------------------------------------- #
class TestScanOnce:
    def test_disabled_writes_nothing(self, tmp_path):
        cfg = make_cfg(tmp_path, enabled=False)
        graph = FakeGraph()
        rows = scan_once(cfg, graph_factory=lambda: graph)
        assert rows == []
        assert graph.calls == []
        assert not os.path.exists(cfg.ledger_file)

    def test_scan_appends_one_row_per_symbol(self, tmp_path):
        cfg = make_cfg(tmp_path)
        graph = FakeGraph(decision="BUY")
        rows = scan_once(cfg, graph_factory=lambda: graph)
        assert [r["symbol"] for r in rows] == ["SPY", "QQQ", "AAPL"]
        assert all(r["decision"] == BUY for r in rows)
        assert all(r["type"] == ROW_TYPE for r in rows)
        on_disk = load_rows(cfg.ledger_file)
        assert len(on_disk) == 3
        assert on_disk[0]["raw"].startswith("verdict:") or on_disk[0]["raw"]

    def test_per_symbol_error_fails_open(self, tmp_path):
        cfg = make_cfg(tmp_path)
        graph = FakeGraph(decision="SELL", fail_symbols={"QQQ"})
        rows = scan_once(cfg, graph_factory=lambda: graph)
        assert len(rows) == 3                      # QQQ error did not abort
        bad = next(r for r in rows if r["symbol"] == "QQQ")
        assert bad["decision"] is None
        assert "llm exploded" in bad["error"]
        good = [r for r in rows if r["symbol"] != "QQQ"]
        assert all(r["decision"] == SELL for r in good)

    def test_max_symbols_per_run_caps(self, tmp_path):
        cfg = make_cfg(tmp_path, max_symbols_per_run=2)
        graph = FakeGraph()
        rows = scan_once(cfg, graph_factory=lambda: graph)
        assert [r["symbol"] for r in rows] == ["SPY", "QQQ"]

    def test_load_rows_skips_garbage(self, tmp_path):
        path = tmp_path / "ledger.jsonl"
        path.write_text(
            'not json\n'
            + json.dumps({"type": "other", "symbol": "X"}) + "\n"
            + json.dumps({"type": ROW_TYPE, "symbol": "SPY",
                          "decision": "BUY"}) + "\n",
            encoding="utf-8")
        rows = load_rows(str(path))
        assert len(rows) == 1 and rows[0]["symbol"] == "SPY"


# --------------------------------------------------------------------------- #
# API-key guard
# --------------------------------------------------------------------------- #
class _EmptyLoader:
    def get_str(self, key, default=""):
        return default


class TestApiKeyGuard:
    def test_missing_key_reason(self, monkeypatch):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(tas, "ConfigLoader", lambda *a, **k: _EmptyLoader())
        cfg = TAShadowConfig(enabled=True, llm_provider="openai")
        reason = api_key_missing_reason(cfg)
        assert reason and "OPENAI_API_KEY" in reason

    def test_env_key_clears_reason(self, monkeypatch):
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-test")
        cfg = TAShadowConfig(enabled=True, llm_provider="anthropic")
        assert api_key_missing_reason(cfg) is None

    def test_ollama_needs_no_key(self):
        cfg = TAShadowConfig(enabled=True, llm_provider="ollama")
        assert api_key_missing_reason(cfg) is None

    def test_scan_without_key_and_without_factory_noops(self, monkeypatch,
                                                        tmp_path):
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.setattr(tas, "ConfigLoader", lambda *a, **k: _EmptyLoader())
        cfg = make_cfg(tmp_path)
        rows = scan_once(cfg)        # no factory -> key guard applies
        assert rows == []
        assert not os.path.exists(cfg.ledger_file)


# --------------------------------------------------------------------------- #
# Forward-return join (pure)
# --------------------------------------------------------------------------- #
def closes(*pairs):
    return [(d, c) for d, c in pairs]


class TestForwardReturns:
    SERIES = closes(("2026-01-02", 100.0), ("2026-01-05", 101.0),
                    ("2026-01-06", 102.0), ("2026-01-07", 103.0),
                    ("2026-01-08", 104.0), ("2026-01-09", 105.0),
                    ("2026-01-12", 106.0))

    def test_base_is_decision_day_close(self):
        rows = [{"symbol": "SPY", "date": "2026-01-02", "decision": BUY}]
        out = forward_returns_for_rows(rows, {"SPY": self.SERIES})
        assert len(out) == 1
        rec = out[0]
        assert rec["base_date"] == "2026-01-02"
        assert abs(rec["fwd_1d"] - 0.01) < 1e-12          # 101/100-1
        assert abs(rec["fwd_5d"] - 0.05) < 1e-12          # 105/100-1

    def test_weekend_decision_uses_next_trading_day(self):
        rows = [{"symbol": "SPY", "date": "2026-01-03", "decision": SELL}]
        out = forward_returns_for_rows(rows, {"SPY": self.SERIES})
        assert out[0]["base_date"] == "2026-01-05"
        assert abs(out[0]["fwd_1d"] - (102.0 / 101.0 - 1)) < 1e-12

    def test_incomplete_window_is_pending_none(self):
        rows = [{"symbol": "SPY", "date": "2026-01-09", "decision": BUY}]
        out = forward_returns_for_rows(rows, {"SPY": self.SERIES})
        assert abs(out[0]["fwd_1d"] - (106.0 / 105.0 - 1)) < 1e-12
        assert out[0]["fwd_5d"] is None

    def test_unknown_symbol_or_label_skipped(self):
        rows = [{"symbol": "ZZZ", "date": "2026-01-02", "decision": BUY},
                {"symbol": "SPY", "date": "2026-01-02", "decision": UNKNOWN}]
        assert forward_returns_for_rows(rows, {"SPY": self.SERIES}) == []


# --------------------------------------------------------------------------- #
# Evidence gate (pure)
# --------------------------------------------------------------------------- #
def flat_series(start_price, rets, start_day=2):
    """Build a synthetic close series realizing given forward returns."""
    out = [(f"2026-02-{start_day:02d}", start_price)]
    price = start_price
    day = start_day
    for r in rets:
        day += 1
        price *= (1.0 + r)
        out.append((f"2026-02-{day:02d}", price))
    return out


class TestEvaluate:
    def test_small_sample_never_passes_gate(self):
        series = closes(("2026-01-02", 100.0), ("2026-01-05", 101.0),
                        ("2026-01-06", 102.0), ("2026-01-07", 103.0),
                        ("2026-01-08", 104.0), ("2026-01-09", 105.0),
                        ("2026-01-12", 106.0))
        rows = [{"symbol": "SPY", "date": "2026-01-02", "decision": BUY}]
        rep = evaluate(rows, {"SPY": series})
        assert rep["primary"]["n"] == 1
        assert rep["primary"]["gate_passed"] is False
        assert rep["per_label"][BUY]["5d"]["n"] == 1

    def test_gate_passes_with_n40_positive_t(self):
        # 40 BUY decisions, each with a known positive 5d forward return
        # (alternating 0.9% / 1.1% -> mean 1%, tiny variance, huge t).
        rows = []
        closes_by_symbol = {}
        for i in range(GATE_MIN_N):
            sym = f"S{i:02d}"
            r5 = 0.009 if i % 2 == 0 else 0.011
            per_day = (1.0 + r5) ** (1.0 / 5.0) - 1.0
            closes_by_symbol[sym] = flat_series(100.0, [per_day] * 5)
            rows.append({"symbol": sym, "date": "2026-02-02",
                         "decision": BUY})
        rep = evaluate(rows, closes_by_symbol)
        p = rep["primary"]
        assert p["n"] == GATE_MIN_N
        assert abs(p["mean_bp"] - 100.0) < 1.0
        assert p["t"] >= 2.0
        assert p["gate_passed"] is True

    def test_sell_decisions_enter_signed_metric_negated(self):
        # SELL before a falling market scores positive signed return.
        down = flat_series(100.0, [-0.002] * 5)
        rows = [{"symbol": "SPY", "date": "2026-02-02", "decision": SELL}]
        rep = evaluate(rows, {"SPY": down})
        assert rep["primary"]["n"] == 1
        assert rep["primary"]["mean_bp"] > 0

    def test_hold_excluded_from_primary(self):
        up = flat_series(100.0, [0.01] * 5)
        rows = [{"symbol": "SPY", "date": "2026-02-02", "decision": HOLD}]
        rep = evaluate(rows, {"SPY": up})
        assert rep["primary"]["n"] == 0
        assert rep["per_label"][HOLD]["5d"]["n"] == 1

    def test_never_raises_on_garbage(self):
        assert evaluate(None, None)["primary"]["n"] == 0
        assert evaluate([{"bad": 1}], {"SPY": [("x", "y")]})["primary"]["n"] == 0
