"""Phase 7b tests: observe-only shadow — default OFF, fail-open, no orders.

4.0.2 P0-2: history comes exclusively from a real bars store.  No bars store
configured means structured abstention (``BARS_DB_UNSET``); daily closes are
never placed on an assumed intraday grid.
"""
import inspect
import json
import os
from datetime import datetime, timedelta, timezone

import pytest

from oracle.ml import bars as barsmod
from oracle.ml import shadow as sh
from oracle.ml.bars import Bar, BarStore

# 2026-11-18 is a Wednesday under EST: RTH = 14:30-21:00 UTC.
OPEN = datetime(2026, 11, 18, 14, 30, tzinfo=timezone.utc)
# 41 one-minute bars from the open end 14:30+41m = 15:11; at NOW the newest
# completed bar is 9 minutes old -- inside the 15-minute staleness default.
NOW = OPEN + timedelta(minutes=50)


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


def _bar(ts, close, sym="SPY", res=60):
    return Bar(symbol=sym, resolution_s=res, ts_utc=ts, open=close,
               high=close, low=close, close=close, volume=1.0,
               session_date=barsmod.session_date_of(ts),
               rth=barsmod.is_rth(ts), source="test")


def _bars_db(tmp_path, syms=("SPY",), n=41, start=OPEN):
    """Real minute-bar store: n completed bars per symbol from the open."""
    path = str(tmp_path / "bars.sqlite")
    st = BarStore(path, readonly=False)
    for sym in syms:
        base = {"SPY": 600.0, "QQQ": 500.0}.get(sym, 100.0)
        st.insert_bars([_bar(start + timedelta(minutes=i),
                             base + (i % 5) * 0.2, sym=sym)
                        for i in range(n)])
    st.close()
    return path


def _cfg(bars_db="", **kw):
    kw.setdefault("enabled", True)
    kw.setdefault("max_scenarios", 256)
    return sh.ShadowConfig(bars_db=bars_db, **kw)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("ENABLE_ORACLE4_SHADOW", "ORACLE4_MODEL_ID",
              "ORACLE4_MAX_CANDIDATES", "ORACLE4_MAX_SCENARIOS",
              "ORACLE4_SHADOW_JSONL", "ORACLE4_REGISTRY",
              "ORACLE4_BARS_DB", "ORACLE4_BARS_MAX_STALENESS_MIN"):
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
    assert cfg.bars_max_staleness_min == 15.0  # default


def test_staleness_env_parsing(monkeypatch):
    monkeypatch.setenv("ORACLE4_BARS_MAX_STALENESS_MIN", "7.5")
    assert sh.ShadowConfig.from_env().bars_max_staleness_min == 7.5
    monkeypatch.setenv("ORACLE4_BARS_MAX_STALENESS_MIN", "-3")
    assert sh.ShadowConfig.from_env().bars_max_staleness_min == 15.0
    monkeypatch.setenv("ORACLE4_BARS_MAX_STALENESS_MIN", "bogus")
    assert sh.ShadowConfig.from_env().bars_max_staleness_min == 15.0


# --------------------------------------------------------------------------- #
# record shape & semantics (real bars store)
# --------------------------------------------------------------------------- #

def test_record_shape_and_baseline_order(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    db = _bars_db(tmp_path, syms=("SPY", "QQQ"))
    cands = [_cand("SPY", score=50.0), _cand("QQQ", score=90.0, price=500.0,
                                             symbol=_occ("QQQ"))]
    rec = sh.shadow_scan(cands, scan_id="scan-1", now=NOW, step_minutes=1.0,
                         cfg=_cfg(bars_db=db), out_path=out)
    for key in ("schema_version", "type", "at", "scan_id", "model", "flags",
                "inference_ms", "bars_db", "input", "scenarios", "challenger",
                "baseline", "note"):
        assert key in rec
    assert rec["type"] == sh.RECORD_TYPE
    # the synthetic-grid record keys were removed in 4.0.2 (P0-2)
    assert "history_grid_assumed" not in rec
    assert "history_grid_minutes" not in rec
    assert rec["bars_db"] == db
    assert rec["flags"]["ORACLE4_BARS_MAX_STALENESS_MIN"] == 15.0
    for u in ("SPY", "QQQ"):
        assert rec["scenarios"][u]["history_source"] == "bars"
        assert rec["scenarios"][u]["status"] == "ok"
        assert rec["scenarios"][u]["n_history_points"] == 41
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
    db = _bars_db(tmp_path, syms=("SPY",))
    cands = [_cand("SPY", score=90.0), _cand("QQQ", score=80.0),
             _cand("IWM", score=70.0)]
    rec = sh.shadow_scan(cands, now=NOW, step_minutes=1.0,
                         cfg=_cfg(bars_db=db, max_candidates=1),
                         out_path=out)
    assert rec["input"]["n_candidates_in"] == 3
    assert rec["input"]["n_candidates_truncated"] == 2
    assert rec["input"]["n_valued"] == 1


def test_missing_quote_lands_in_missingness(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    db = _bars_db(tmp_path, syms=("SPY",))
    bad = _cand("SPY")
    bad["opt"]["bid"] = None
    bad["opt"]["ask"] = None
    rec = sh.shadow_scan([bad], now=NOW, step_minutes=1.0,
                         cfg=_cfg(bars_db=db), out_path=out)
    assert sum(rec["input"]["missingness"].values()) == 1
    assert any("quote" in r for r in rec["input"]["missingness"])


# --------------------------------------------------------------------------- #
# bars-only history: structured abstentions, never a synthetic grid
# --------------------------------------------------------------------------- #

def test_no_bars_db_abstains_with_bars_db_unset(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    rec = sh.shadow_scan([_cand("SPY")], now=NOW,
                         cfg=_cfg(bars_db=""), out_path=out)
    assert rec["challenger"]["decision"] == "abstain"
    meta = rec["scenarios"]["SPY"]
    assert meta["history_source"] == "bars"
    assert meta["bars_unavailable"]["reason"] == "BARS_DB_UNSET"
    assert any("BARS_DB_UNSET" in r for r in rec["input"]["missingness"])
    assert rec["flags"]["ORACLE4_BARS_DB"] is None


def test_missing_store_abstains_not_crash(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    rec = sh.shadow_scan(
        [_cand("SPY")], now=NOW,
        cfg=_cfg(bars_db=str(tmp_path / "nope.sqlite")), out_path=out)
    assert rec["challenger"]["decision"] == "abstain"
    assert any(r.startswith("no_scenarios") for r in rec["input"]["missingness"])


def test_stale_bars_abstain(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    db = _bars_db(tmp_path, syms=("SPY",))
    # newest completed bar ends 15:11; two hours later it is far beyond the
    # 15-minute freshness bound -> structured STALE abstention
    late = OPEN + timedelta(minutes=170)
    rec = sh.shadow_scan([_cand("SPY")], now=late, step_minutes=1.0,
                         cfg=_cfg(bars_db=db), out_path=out)
    assert rec["challenger"]["decision"] == "abstain"
    assert rec["scenarios"]["SPY"]["bars_unavailable"]["reason"] == "STALE"
    assert any("STALE" in r for r in rec["input"]["missingness"])


def test_freeze_candidate_maps_production_keys():
    c = sh.freeze_candidate(_cand("SPY", score=77.0), 0, "scanX", NOW)
    assert c.underlying == "SPY"
    assert c.features["baseline_score"] == 77.0
    assert c.features["underlying_price"] == 600.0
    assert c.features["quote_mid"] == pytest.approx(5.1)
    assert c.features["spread_abs"] == pytest.approx(0.2, abs=1e-9)
    assert c.features["dte_days"] == 31.0
    assert c.contract and c.contract["underlying"] == "SPY"


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
    db = _bars_db(tmp_path, syms=("SPY", "QQQ"))
    cfg = _cfg(bars_db=db)
    sh.shadow_scan([_cand("SPY", score=90.0)], now=NOW, step_minutes=1.0,
                   cfg=cfg, out_path=out)
    sh.shadow_scan([_cand("QQQ", score=80.0, price=500.0,
                          symbol=_occ("QQQ"))], now=NOW, step_minutes=1.0,
                   cfg=cfg, out_path=out)
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
