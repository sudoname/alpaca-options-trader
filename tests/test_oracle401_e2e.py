"""Stage G end-to-end offline fixture: shadow scan -> books/outcome fields ->
outcome resolution -> per-book forward evidence -> dashboard forward section.

The "November 18" 0DTE rows are MODELED fixtures (server data for 2026-11-18
contained no actual 0DTE entries); they exercise reconciliation logic only and
encode no trading rule fitted to one day.
"""
import json
from datetime import datetime, timezone

import pytest

from oracle.ml import evidence as ev
from oracle.ml import forward as fwd
from oracle.ml import shadow as sh

NOW = datetime(2026, 11, 18, 15, 0, tzinfo=timezone.utc)


@pytest.fixture(autouse=True)
def _clean_env(monkeypatch):
    for k in ("ENABLE_ORACLE4_SHADOW", "ORACLE4_MODEL_ID",
              "ORACLE4_MAX_CANDIDATES", "ORACLE4_MAX_SCENARIOS",
              "ORACLE4_SHADOW_JSONL", "ORACLE4_REGISTRY",
              "ORACLE4_BARS_DB", "ORACLE4_BARS_MAX_STALENESS_MIN"):
        monkeypatch.delenv(k, raising=False)


def _occ(underlying="SPY", exp="261219", cp="C", strike_milli="00600000"):
    return f"{underlying}{exp}{cp}{strike_milli}"


def _cand(sym="SPY", score=80.0, price=600.0, bid=5.0, ask=5.2, dte=31.0,
          symbol=None):
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


# --------------------------------------------------------------------------- #
# scan / trade fixtures for forward evidence
# --------------------------------------------------------------------------- #

CH = "SPY261219C00600000"   # challenger pick (swing)
BL = "QQQ261219C00500000"   # baseline pick (swing)


def _scan(at, scan_id, sel=CH, top=BL):
    return {
        "at": at,
        "scan_id": scan_id,
        "challenger": {"decision": "trade", "selected_symbol": sel},
        "baseline": {"top_symbol": top},
    }


def _trade(symbol, entry_time, pnl):
    return {"symbol": symbol, "entry_time": entry_time, "pnl": pnl}


def _swing_fixture():
    scans = [
        _scan("2026-11-17T15:00:00Z", "e2e-1"),
        _scan("2026-11-18T15:00:00Z", "e2e-2"),
        # after all trades of the session -> unresolved on both arms
        _scan("2026-11-18T16:00:00Z", "e2e-3"),
    ]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0),
        _trade(CH, "2026-11-18T15:05:00Z", 12.5),
        _trade(BL, "2026-11-18T15:05:00Z", 4.0),
    ]
    return scans, trades


def _build(rows, book="swing", **over):
    kw = dict(
        book=book,
        model_id="mlp_v1",
        model_hash="h" * 12,
        baseline_id="production_score_rank",
        policy_id="oracle4_shadow_rank_v1",
        dataset_fingerprint="fp-e2e",
        code_git_sha="deadbeef",
        cost_assumptions={"friction_multiplier": 1.0,
                          "basis": "fixture net P&L as recorded"},
        capital_base=100000.0,
        generated_utc="2026-11-19T00:00:00Z",
    )
    kw.update(over)
    return fwd.build_forward_evidence(rows, **kw)


# --------------------------------------------------------------------------- #
# shadow record carries books + outcome state
# --------------------------------------------------------------------------- #

def test_shadow_record_has_books_and_outcome(tmp_path):
    out = str(tmp_path / "shadow.jsonl")
    cfg = sh.ShadowConfig(enabled=True, max_scenarios=256)
    rec = sh.shadow_scan([_cand("SPY")], scan_id="g-1", now=NOW,
                         cfg=cfg, out_path=out)
    assert rec["outcome"] == {"state": "unresolved", "resolved_at": None,
                              "resolver": None}
    by_dec = rec["books"]["by_decision"]
    assert by_dec and all(v["book"] == "swing" for v in by_dec.values())
    assert rec["books"]["summary"]["candidates_by_book"] == {"swing": 1}

    sh.shadow_scan([_cand("QQQ", score=70.0, price=500.0,
                          symbol=_occ("QQQ"))], scan_id="g-2", now=NOW,
                   cfg=cfg, out_path=out)
    rep = sh.summarize_log(out)
    assert rep["n_scans"] == 2
    assert rep["outcome_states"] == {"unresolved": 2}
    assert rep["books"]["candidates_by_book"] == {"swing": 2}


# --------------------------------------------------------------------------- #
# outcome resolution -> forward evidence (swing book)
# --------------------------------------------------------------------------- #

def test_resolve_then_build_swing_forward_evidence():
    scans, trades = _swing_fixture()
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    assert len(rows) == 3
    resolved = [r for r in rows if r["resolution_state"] == fwd.RESOLVED]
    unresolved = [r for r in rows if r["resolution_state"] == fwd.UNRESOLVED]
    assert len(resolved) == 2 and len(unresolved) == 1
    assert all(r["book"] == "swing" for r in rows)
    assert resolved[0]["challenger_net_pnl"] == 25.0
    assert resolved[0]["baseline_net_pnl"] == -10.0
    assert "no realized trade for: challenger,baseline" in unresolved[0]["reason"]

    doc = _build(rows)
    assert ev.validate_forward_evidence(doc) == []
    assert doc["book"] == "swing"
    assert doc["forward_sessions"] == 2
    assert doc["resolved_opportunities"] == 2
    assert doc["live_fill_based"] is False
    assert doc["coverage"]["n_unresolved"] == 1
    assert doc["coverage"]["resolution_rate"] == pytest.approx(2.0 / 3.0)
    assert sum(g["n_opportunities"]
               for g in doc["coverage"]["groups"].values()) == 3
    assert [s["session"] for s in doc["sessions"]] == \
        ["2026-11-17", "2026-11-18"]
    assert doc["sessions"][0]["challenger_net_pnl"] == 25.0
    assert doc["sessions"][0]["baseline_net_pnl"] == -10.0
    assert len(doc["fills"]) == 2
    assert all(f["source"] == "mark_derived" for f in doc["fills"])
    assert doc["unresolved"][0]["session"] == "2026-11-18"
    assert "no realized trade" in doc["unresolved"][0]["reason"]


def test_build_rejects_book_mixing_and_bad_inputs():
    scans, trades = _swing_fixture()
    rows = fwd.resolve_outcomes_from_history(scans, trades)

    mixed = [dict(r) for r in rows]
    mixed[0]["book"] = "0dte"
    with pytest.raises(ValueError, match="book mixing"):
        _build(mixed)

    only_unresolved = [r for r in rows
                       if r["resolution_state"] == fwd.UNRESOLVED]
    with pytest.raises(ValueError):
        _build(only_unresolved)

    with pytest.raises(ValueError):
        fwd.resolve_outcomes_from_history(scans, trades, source="bogus")

    rows_nofee = fwd.resolve_outcomes_from_history(scans, trades,
                                                   fees_known=False)
    with pytest.raises(ValueError):
        _build([r for r in rows_nofee
                if r["resolution_state"] == fwd.RESOLVED])


# --------------------------------------------------------------------------- #
# MODELED Nov-18 0DTE reconciliation fixture (separate 0dte book)
# --------------------------------------------------------------------------- #

def test_modeled_nov18_0dte_book_evidence():
    # MODELED: paired call/put straddle legs on the 0DTE book — resolution
    # and evidence stay in the 0dte book, never mixed into swing.
    call = "SPY261118C00600000"
    put = "SPY261118P00600000"
    scans = [_scan("2026-11-18T15:00:00Z", "z-1", sel=call, top=put)]
    trades = [_trade(call, "2026-11-18T15:05:00Z", 18.0),
              _trade(put, "2026-11-18T15:05:00Z", -6.0)]
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    assert len(rows) == 1
    assert rows[0]["book"] == "0dte"
    assert rows[0]["resolution_state"] == fwd.RESOLVED

    doc = _build(rows, book="0dte", generated_utc="2026-11-18T23:00:00Z")
    assert ev.validate_forward_evidence(doc) == []
    assert doc["book"] == "0dte"
    assert doc["forward_sessions"] == 1
    # the 0dte rows cannot be pushed into the swing builder
    with pytest.raises(ValueError, match="book mixing"):
        _build(rows, book="swing")


# --------------------------------------------------------------------------- #
# artifact write + dashboard forward section (read-only)
# --------------------------------------------------------------------------- #

def test_write_and_dashboard_forward_section(tmp_path, monkeypatch):
    scans, trades = _swing_fixture()
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    doc = _build(rows)

    monkeypatch.chdir(tmp_path)
    art = tmp_path / "artifacts" / "oracle4"
    art.mkdir(parents=True)
    path = fwd.write_forward_evidence(doc, str(art))
    assert path.endswith("forward_evidence_swing.json")
    with open(path, encoding="utf-8") as f:
        reloaded = json.load(f)
    assert ev.validate_forward_evidence(reloaded) == []

    from oracle_dashboard import DashboardConfig, create_app
    app = create_app(DashboardConfig(host="127.0.0.1", port=0, cache_ttl=0,
                                     basic_auth_user="", basic_auth_pass=""))
    j = app.test_client().get("/api/oracle4?fresh=1").get_json()
    assert j["forward"]["status"] == "online"
    books = j["forward"]["books"]
    assert len(books) == 1
    entry = books[0]
    assert entry["book"] == "swing"
    assert entry["validation"] == "valid"
    assert entry["forward_sessions"] == 2
    assert entry["resolved_opportunities"] == 2
    assert entry["live_fill_based"] is False
