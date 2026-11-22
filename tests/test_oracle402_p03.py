"""Oracle 4.0.2 P0-3: exact trade/outcome attribution.

Acceptance covered here:
* deterministic decision -> client_order_id (retries distinct, broker-safe length)
* duplicate realized-trade records never double-count P&L
* repeated scans cannot claim the same realized trade (exclusive claims)
* order IDs are required by default; unknown IDs never fall back
* one-sided claims roll back so unresolved rows never consume evidence
* unknown outcomes stay unresolved -- never coerced to zero profit
"""
import pytest

from oracle.ml import attribution as att
from oracle.ml import forward as fwd

CH = "SPY261219C00600000"
BL = "QQQ261219C00500000"


def _scan(at, scan_id, sel=CH, top=BL, ch_order_id=None, bl_order_id=None):
    ch = {"decision": "trade", "selected_symbol": sel}
    if ch_order_id:
        ch["order_id"] = ch_order_id
    base = {"top_symbol": top}
    if bl_order_id:
        base["order_id"] = bl_order_id
    return {"at": at, "scan_id": scan_id, "challenger": ch, "baseline": base}


def _trade(symbol, entry_time, pnl, order_id=None, qty=None):
    t = {"symbol": symbol, "entry_time": entry_time, "pnl": pnl}
    if order_id:
        t["order_id"] = order_id
    if qty is not None:
        t["qty"] = qty
    return t


# --------------------------------------------------------------------------- #
# decision -> order identity
# --------------------------------------------------------------------------- #

def test_client_order_id_deterministic_and_retry_distinct():
    a = att.make_client_order_id("dec-123")
    b = att.make_client_order_id("dec-123")
    r1 = att.make_client_order_id("dec-123", attempt=1)
    other = att.make_client_order_id("dec-124")
    assert a == b
    assert a != r1 and a != other and r1 != other
    assert a == att.make_client_order_id("dec-123", attempt=0)
    for cid in (a, r1, other):
        assert len(cid) <= att.MAX_CLIENT_ORDER_ID_LEN
        assert cid.startswith("o4-")


def test_client_order_id_rejects_bad_inputs():
    with pytest.raises(ValueError):
        att.make_client_order_id("")
    with pytest.raises(ValueError):
        att.make_client_order_id("   ")
    with pytest.raises(ValueError):
        att.make_client_order_id(None)
    with pytest.raises(ValueError):
        att.make_client_order_id("dec-1", attempt=-1)
    with pytest.raises(ValueError):
        att.make_client_order_id("dec-1", attempt=True)
    with pytest.raises(ValueError):
        att.make_client_order_id("dec-1", attempt=1.5)


# --------------------------------------------------------------------------- #
# trade identity + dedupe
# --------------------------------------------------------------------------- #

def test_trade_identity_prefers_order_id_then_fingerprint():
    kind, tid = att.trade_identity(
        _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="ord-1"))
    assert (kind, tid) == (att.ID_ORDER, "ord-1")

    t = _trade(CH, "2026-11-17T15:05:00Z", 25.0)
    kind2, fp1 = att.trade_identity(t)
    _, fp2 = att.trade_identity(dict(t))
    assert kind2 == att.ID_FINGERPRINT
    assert fp1 == fp2 and fp1.startswith("tf-")
    # any content change changes the fingerprint
    _, fp3 = att.trade_identity(_trade(CH, "2026-11-17T15:05:00Z", 25.5))
    assert fp3 != fp1


def test_dedupe_trades_drops_exact_duplicates():
    t1 = _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="ord-1")
    t2 = _trade(BL, "2026-11-17T15:05:00Z", -10.0)
    unique, dropped = att.dedupe_trades([t1, dict(t1), t2, dict(t2), dict(t1)])
    assert dropped == 3
    assert unique == [t1, t2]
    # distinct fills on the same symbol/session are NOT duplicates
    t3 = _trade(CH, "2026-11-17T15:30:00Z", 5.0)
    unique2, dropped2 = att.dedupe_trades([t2, t3])
    assert dropped2 == 0 and len(unique2) == 2


# --------------------------------------------------------------------------- #
# exclusive claims via resolve_outcomes_from_history
# --------------------------------------------------------------------------- #

def test_repeated_scans_cannot_claim_same_trade():
    scans = [
        _scan("2026-11-17T15:00:00Z", "s-1"),
        _scan("2026-11-17T15:01:00Z", "s-2"),  # same picks, same session
    ]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0),
    ]
    rows = fwd.resolve_outcomes_from_history(scans, trades,
                                             allow_heuristic=True)
    assert len(rows) == 2
    states = [r["resolution_state"] for r in rows]
    assert states == [fwd.RESOLVED, fwd.UNRESOLVED]
    # the single trade's P&L appears exactly once in aggregate
    total = sum(r.get("challenger_net_pnl", 0.0) for r in rows)
    assert total == 25.0
    assert "challenger" in rows[1]["reason"]
    assert "baseline" in rows[1]["reason"]


def test_duplicate_trade_records_do_not_double_count():
    scans = [
        _scan("2026-11-17T15:00:00Z", "s-1"),
        _scan("2026-11-17T15:01:00Z", "s-2"),
    ]
    one_ch = _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="ord-ch")
    one_bl = _trade(BL, "2026-11-17T15:05:00Z", -10.0, order_id="ord-bl")
    # history re-ingested twice: duplicates must collapse to one event each
    trades = [one_ch, one_bl, dict(one_ch), dict(one_bl)]
    rows = fwd.resolve_outcomes_from_history(scans, trades,
                                             allow_heuristic=True)
    resolved = [r for r in rows if r["resolution_state"] == fwd.RESOLVED]
    assert len(resolved) == 1
    assert resolved[0]["challenger_net_pnl"] == 25.0
    assert rows[1]["resolution_state"] == fwd.UNRESOLVED


def test_order_id_join_preferred_over_time_heuristic():
    scans = [_scan("2026-11-17T15:00:00Z", "s-1",
                   ch_order_id="ord-late", bl_order_id="ord-bl")]
    trades = [
        # earlier fill the time heuristic would pick first
        _trade(CH, "2026-11-17T15:01:00Z", 1.0, order_id="ord-early"),
        _trade(CH, "2026-11-17T15:30:00Z", 99.0, order_id="ord-late"),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0, order_id="ord-bl"),
    ]
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    assert rows[0]["resolution_state"] == fwd.RESOLVED
    assert rows[0]["challenger_net_pnl"] == 99.0
    attr = rows[0]["attribution"]
    assert attr["challenger"] == {"trade_id": "ord-late",
                                  "match_basis": att.MATCH_ORDER_ID}
    assert attr["baseline"]["match_basis"] == att.MATCH_ORDER_ID


def test_one_sided_claim_rolls_back_for_later_scan():
    scans = [
        # s-1: challenger has a trade but baseline never traded
        _scan("2026-11-17T15:00:00Z", "s-1", top="IWM261219C00250000"),
        _scan("2026-11-17T15:01:00Z", "s-2"),
    ]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0),
    ]
    rows = fwd.resolve_outcomes_from_history(scans, trades,
                                             allow_heuristic=True)
    assert rows[0]["resolution_state"] == fwd.UNRESOLVED
    assert rows[0]["reason"] == "no realized trade for: baseline"
    # s-1's tentative challenger claim was released, so s-2 resolves fully
    assert rows[1]["resolution_state"] == fwd.RESOLVED
    assert rows[1]["challenger_net_pnl"] == 25.0
    assert rows[1]["baseline_net_pnl"] == -10.0
    basis = rows[1]["attribution"]["challenger"]["match_basis"]
    assert basis == att.MATCH_SYMBOL_TIME


def test_same_symbol_both_arms_shares_one_trade_once():
    scans = [
        _scan("2026-11-17T15:00:00Z", "s-1", sel=CH, top=CH),
        _scan("2026-11-17T15:01:00Z", "s-2", sel=CH, top=CH),
    ]
    trades = [_trade(CH, "2026-11-17T15:05:00Z", 25.0)]
    rows = fwd.resolve_outcomes_from_history(scans, trades,
                                             allow_heuristic=True)
    assert rows[0]["resolution_state"] == fwd.RESOLVED
    assert rows[0]["challenger_net_pnl"] == 25.0
    assert rows[0]["baseline_net_pnl"] == 25.0
    assert rows[0]["attribution"]["baseline"]["match_basis"] == \
        att.MATCH_SHARED
    # the trade pool is consumed once: the second scan stays unresolved
    assert rows[1]["resolution_state"] == fwd.UNRESOLVED


def test_unresolved_rows_never_carry_zero_pnl():
    scans = [_scan("2026-11-18T16:00:00Z", "s-1")]
    rows = fwd.resolve_outcomes_from_history(scans, [])
    assert len(rows) == 1
    r = rows[0]
    assert r["resolution_state"] == fwd.UNRESOLVED
    assert "challenger_net_pnl" not in r
    assert "baseline_net_pnl" not in r
    assert "attribution" not in r
    # and the artifact builder refuses to synthesize evidence from them
    with pytest.raises(ValueError):
        fwd.build_forward_evidence(
            rows, book="swing", model_id="m", model_hash="h" * 12,
            baseline_id="b", policy_id="p", dataset_fingerprint="fp",
            code_git_sha="sha", cost_assumptions={"friction_multiplier": 1.0},
            capital_base=100000.0)


def test_claimed_order_id_is_not_silently_reattributed():
    # two scans both naming the SAME broker order id: the second must not
    # fall back to the time heuristic and grab a different trade
    scans = [
        _scan("2026-11-17T15:00:00Z", "s-1",
              ch_order_id="ord-1", bl_order_id="ord-bl"),
        _scan("2026-11-17T15:00:30Z", "s-2",
              ch_order_id="ord-1", bl_order_id="ord-bl2"),
    ]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="ord-1"),
        _trade(CH, "2026-11-17T15:06:00Z", 7.0, order_id="ord-2"),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0, order_id="ord-bl"),
        _trade(BL, "2026-11-17T15:06:00Z", -4.0, order_id="ord-bl2"),
    ]
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    assert rows[0]["resolution_state"] == fwd.RESOLVED
    assert rows[0]["challenger_net_pnl"] == 25.0
    assert rows[1]["resolution_state"] == fwd.UNRESOLVED
    assert rows[1]["reason"] == "no realized trade for: challenger"


def test_default_requires_ids_on_both_arms():
    scans = [_scan("2026-11-17T15:00:00Z", "s-1",
                   ch_order_id="ord-ch")]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="ord-ch"),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0, order_id="ord-bl"),
    ]
    rows = fwd.resolve_outcomes_from_history(scans, trades)
    assert rows[0]["resolution_state"] == fwd.UNRESOLVED
    assert rows[0]["reason"] == "no realized trade for: baseline"
    assert "challenger_net_pnl" not in rows[0]


def test_unknown_id_never_falls_back_even_in_legacy_mode():
    scans = [_scan("2026-11-17T15:00:00Z", "s-1",
                   ch_order_id="absent", bl_order_id="ord-bl")]
    trades = [
        _trade(CH, "2026-11-17T15:05:00Z", 25.0, order_id="other"),
        _trade(BL, "2026-11-17T15:05:00Z", -10.0, order_id="ord-bl"),
    ]
    for legacy in (False, True):
        row = fwd.resolve_outcomes_from_history(
            scans, trades, allow_heuristic=legacy)[0]
        assert row["resolution_state"] == fwd.UNRESOLVED
        assert row["reason"] == "no realized trade for: challenger"


def test_client_order_id_alias_matches_with_broker_id_present():
    client_id = att.make_client_order_id("dec-1")
    scans = [_scan("2026-11-17T15:00:00Z", "s-1",
                   ch_order_id=client_id, bl_order_id="ord-bl")]
    trade = _trade(CH, "2026-11-17T15:05:00Z", 25.0,
                   order_id="broker-ch")
    trade["client_order_id"] = client_id
    rows = fwd.resolve_outcomes_from_history(scans, [
        trade, _trade(BL, "2026-11-17T15:05:00Z", -10.0,
                      order_id="ord-bl")])
    assert rows[0]["resolution_state"] == fwd.RESOLVED
    assert rows[0]["attribution"]["challenger"] == {
        "trade_id": "broker-ch", "match_basis": att.MATCH_ORDER_ID}


def test_exact_id_must_match_contract_session_and_scan_time():
    from datetime import datetime, timezone
    trade = _trade(CH, "2026-11-17T15:05:00Z", 25.0,
                   order_id="ord-ch")
    matcher = att.ExclusiveTradeMatcher([trade])
    after = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
    assert matcher.claim(BL, "2026-11-17", after,
                         order_id="ord-ch") is None
    assert matcher.claim(CH, "2026-11-18", after,
                         order_id="ord-ch") is None
    assert matcher.claim(CH, "2026-11-17", datetime(
        2026, 11, 17, 16, tzinfo=timezone.utc),
        order_id="ord-ch") is None
    assert matcher.claim(CH, "2026-11-17", after,
                         order_id="ord-ch") is not None


def test_same_symbol_different_order_ids_do_not_share_pnl():
    scans = [_scan("2026-11-17T15:00:00Z", "s-1", sel=CH, top=CH,
                   ch_order_id="ord-1", bl_order_id="ord-2")]
    trades = [_trade(CH, "2026-11-17T15:05:00Z", 25.0,
                     order_id="ord-1")]
    row = fwd.resolve_outcomes_from_history(scans, trades)[0]
    assert row["resolution_state"] == fwd.UNRESOLVED


def test_same_trade_broker_and_client_ids_share_exactly_once():
    scans = [_scan("2026-11-17T15:00:00Z", "s-1", sel=CH, top=CH,
                   ch_order_id="ord-1", bl_order_id="client-1")]
    trade = _trade(CH, "2026-11-17T15:05:00Z", 25.0,
                   order_id="ord-1")
    trade["client_order_id"] = "client-1"
    row = fwd.resolve_outcomes_from_history(scans, [trade])[0]
    assert row["resolution_state"] == fwd.RESOLVED
    assert row["challenger_net_pnl"] == row["baseline_net_pnl"] == 25.0
    assert row["attribution"]["baseline"]["match_basis"] == att.MATCH_SHARED


def test_legacy_heuristic_cannot_be_labeled_live():
    with pytest.raises(ValueError, match="exact order attribution"):
        fwd.resolve_outcomes_from_history([], [], source="live_order",
                                          allow_heuristic=True)


# --------------------------------------------------------------------------- #
# matcher unit behaviour
# --------------------------------------------------------------------------- #

def test_matcher_release_makes_trade_claimable_again():
    from datetime import datetime, timezone
    m = att.ExclusiveTradeMatcher(
        [_trade(CH, "2026-11-17T15:05:00Z", 25.0)])
    after = datetime(2026, 11, 17, 15, 0, tzinfo=timezone.utc)
    c1 = m.claim(CH, "2026-11-17", after, allow_heuristic=True)
    assert c1 is not None and c1["pnl"] == 25.0
    assert m.claim(CH, "2026-11-17", after, allow_heuristic=True) is None
    m.release(c1)
    c2 = m.claim(CH, "2026-11-17", after, allow_heuristic=True)
    assert c2 is not None and c2["trade_id"] == c1["trade_id"]
    # releasing None / shared aliases is a no-op
    m.release(None)
    m.release(att.shared_claim(c2))
    assert m.claim(CH, "2026-11-17", after, allow_heuristic=True) is None
