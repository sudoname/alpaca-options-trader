"""Phase 7b tests: observe-only shadow — default OFF, fail-open, no orders."""
import inspect
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import shadow as sh

NOW = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)


def _occ(underlying="SPY", exp="261219", cp="C", strike_milli="00600000"):
    return f"{underlying}{exp}{cp}{strike_milli}"


def _cand(sym="SPY", score=80.0, price=600.0, bid=5.0, ask=5.2, dte=31.0,
          symbol=None):
    """Production-shaped candidate: _rank_and_enter's _evaluate output."""
    return {
        "sym": sym,
        "direction": "call",
        "score": score,
        "underlying_price": price,
        "opt": {
            "symbol": symbol or _occ(sym),
            "underlying": sym,
            "bid": bid, "ask": ask, "dte": dte,
            "expiration": "2026-12-19", "type": "call",
        },
    }


def _history(n=60, s0=600.0):
    # gentle deterministic wiggle; > min_history_returns (30) observations
    return [s0 * (1.0 + 0.001 * ((k * 37) % 7 - 3)) for k in range(n)]


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("ENABLE_ORACLE4_SHADOW", "ORACLE4_MODEL_ID",
              "ORACLE4_MAX_CANDIDATES", "ORACLE4_MAX_SCENARIOS",
              "ORACLE4_SHADOW_JSONL", "ORACLE4_REGISTRY"):
        monkeypatch.delenv(k, raising=False)


# --------------------------------------------------------------------------- #
# flags & safety
# --------------------------------------------------------------------------- #

def test_default_off_no_record_no_file(tmp_path, monkeypatch):
    out = tmp_path / "shadow.jsonl"
    monkeypatch.setenv("ORACLE4_SHADOW_JSONL", str(out))
    assert sh.ShadowConfig.from_env().enabled is False
    assert sh.maybe_shadow_scan([_cand()]) is None
    assert not out.exists()


def test_fail_open_never_raises(monkeypatch, tmp_path):
    monkeypatch.setenv("ENABLE_ORACLE4_SHADOW", "true")
    monkeypatch.setenv("ORACLE4_SHADOW_JSONL", str(tmp_path / "s.jsonl"))
    # candidates=None forces a TypeError inside shadow_scan; must be swallowed
    assert sh.maybe_shadow_scan(None) is None


def test_no_broker_imports_in_module_source():
    src = inspect.getsource(sh)
    for forbidden in ("alpaca", "place_order", "submit_order", "replace_order",
                      "cancel_order", "import requests", "TradingClient"):
        assert forbidden not in src, f"shadow module must not reference {forbidden!r}"


def test_env_parsing(monkeypatch):
    monkeypatch.setenv("ENABLE_ORACLE4_SHADOW", "YES")
    monkeypatch.setenv("ORACLE4_MODEL_ID", " mlp_v1 ")
    monkeypatch.setenv("ORACLE4_MAX_CANDIDATES", "0")   # clamped to >=1
    monkeypatch.setenv("ORACLE4_MAX_SCENARIOS", "bogus")  # falls back
    cfg = sh.ShadowConfig.from_env()
    assert cfg.enabled is True
    assert cfg.model_id == "mlp_v1"
    assert cfg.max_candidates == 1
    assert cfg.max_scenarios == 512


# --------------------------------------------------------------------------- #
# record shape & semantics
# --------------------------------------------------------------------------- #

def test_record_shape_and_baseline_order(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    cands = [_cand("SPY", score=50.0), _cand("QQQ", score=90.0, price=500.0,
                                             symbol=_occ("QQQ"))]
    rec = sh.shadow_scan(cands, scan_id="scan-1", now=NOW,
                         price_history_fn=lambda u, d=90: _history(
                             s0=600.0 if u == "SPY" else 500.0),
                         cfg=sh.ShadowConfig(enabled=True, max_scenarios=256),
                         out_path=out)
    for key in ("schema_version", "type", "at", "scan_id", "model", "flags",
                "inference_ms", "history_grid_minutes", "history_grid_assumed",
                "input", "scenarios", "challenger", "baseline", "note"):
        assert key in rec
    assert rec["type"] == sh.RECORD_TYPE
    assert rec["history_grid_assumed"] is True
    # baseline view preserves production's score ordering (QQQ first)
    assert rec["baseline"]["top_sym"] == "QQQ"
    assert rec["baseline"]["order"][0]["score"] == 90.0
    assert rec["challenger"]["decision"] in ("trade", "abstain")
    assert rec["input"]["n_valued"] == 2
    # one parseable JSON line on disk
    lines = [l for l in open(out, encoding="utf-8").read().splitlines() if l]
    assert len(lines) == 1
    assert json.loads(lines[0])["scan_id"] == "scan-1"
    assert "no order was submitted" in rec["note"]


def test_truncation_counted(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    cands = [_cand("SPY", score=90.0), _cand("QQQ", score=80.0),
             _cand("IWM", score=70.0)]
    rec = sh.shadow_scan(cands, now=NOW,
                         price_history_fn=lambda u, d=90: _history(),
                         cfg=sh.ShadowConfig(enabled=True, max_candidates=1,
                                             max_scenarios=256),
                         out_path=out)
    assert rec["input"]["n_candidates_in"] == 3
    assert rec["input"]["n_candidates_truncated"] == 2
    assert rec["input"]["n_valued"] == 1


def test_missing_quote_lands_in_missingness(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    bad = _cand("SPY")
    bad["opt"]["bid"] = None
    bad["opt"]["ask"] = None
    rec = sh.shadow_scan([bad], now=NOW,
                         price_history_fn=lambda u, d=90: _history(),
                         cfg=sh.ShadowConfig(enabled=True, max_scenarios=256),
                         out_path=out)
    assert sum(rec["input"]["missingness"].values()) == 1
    assert any("quote" in r for r in rec["input"]["missingness"])


def test_no_history_means_abstain_not_crash(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    rec = sh.shadow_scan([_cand("SPY")], now=NOW,
                         price_history_fn=lambda u, d=90: [],
                         cfg=sh.ShadowConfig(enabled=True, max_scenarios=256),
                         out_path=out)
    assert rec["challenger"]["decision"] == "abstain"
    assert any(r.startswith("no_scenarios") for r in rec["input"]["missingness"])


def test_freeze_candidate_maps_production_keys():
    c = sh.freeze_candidate(_cand("SPY", score=77.0), 0, "scanX", NOW)
    assert c.underlying == "SPY"
    assert c.features["baseline_score"] == 77.0
    assert c.features["underlying_price"] == 600.0
    assert c.features["quote_mid"] == pytest.approx(5.1)
    assert c.features["spread_abs"] == pytest.approx(0.2, abs=1e-9)
    assert c.features["dte_days"] == 31.0
    assert c.contract and c.contract["underlying"] == "SPY"


def test_history_obs_synthetic_grid():
    obs = sh._history_obs([1.0, 2.0, 3.0], NOW, step_minutes=5.0)
    assert [p for _, p in obs] == [1.0, 2.0, 3.0]
    assert obs[-1][0] == NOW
    assert obs[1][0] == NOW - timedelta(minutes=5)


def test_model_identity_fail_open_and_registry_lookup(tmp_path):
    # unknown registry path -> no entry, no exception
    cfg = sh.ShadowConfig(enabled=True, model_id="ghost",
                          registry_path=str(tmp_path / "nope.json"))
    ident = sh._model_identity(cfg)
    assert ident["model_id"] == "ghost"
    assert ident["registry_entry_found"] is False
    # real registry entry is surfaced
    from oracle.ml import registry as reg
    art = tmp_path / "a.bin"; art.write_bytes(b"w")
    rp = str(tmp_path / "registry.json")
    reg.register_model(rp, "mlp_v1", kind="neural", artifact_path=str(art))
    ident2 = sh._model_identity(sh.ShadowConfig(enabled=True, model_id="mlp_v1",
                                                registry_path=rp))
    assert ident2["registry_entry_found"] is True
    assert ident2["approval_state"] == "unapproved"
    assert ident2["role"] == "challenger"


# --------------------------------------------------------------------------- #
# log summary
# --------------------------------------------------------------------------- #

def test_summarize_log(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    hist = lambda u, d=90: _history()
    cfg = sh.ShadowConfig(enabled=True, max_scenarios=256)
    sh.shadow_scan([_cand("SPY", score=90.0)], now=NOW,
                   price_history_fn=hist, cfg=cfg, out_path=out)
    sh.shadow_scan([_cand("QQQ", score=80.0, price=500.0,
                          symbol=_occ("QQQ"))], now=NOW,
                   price_history_fn=hist, cfg=cfg, out_path=out)
    with open(out, "a", encoding="utf-8") as f:
        f.write("not json\n")
    rep = sh.summarize_log(out)
    assert rep["n_scans"] == 2
    assert rep["n_bad_lines"] == 1
    assert rep["inference_ms"]["min"] >= 0
    assert sum(rep["decisions"].values()) == 2
    assert rep["scans_truncated"] == 0


def test_summarize_missing_log_is_empty():
    rep = sh.summarize_log(os.path.join("definitely", "missing.jsonl"))
    assert rep["n_scans"] == 0
    assert rep["challenger_vs_baseline_top_agreement"] is None
