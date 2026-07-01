"""Unit tests for capskip_cf — the option-repriced per-underlying cap-skip
counterfactual resolver. No network, no broker, no clock; the resolver takes an
injected option_price_fn and an in-memory EpisodeStore.
"""
from episode_store import EpisodeStore
import capskip_cf as cc


# --------------------------------------------------------------------------- #
# option_cf_return
# --------------------------------------------------------------------------- #
def test_option_cf_return_gain():
    assert cc.option_cf_return(1.00, 1.50) == 50.0


def test_option_cf_return_loss():
    assert cc.option_cf_return(2.00, 1.00) == -50.0


def test_option_cf_return_worthless():
    assert cc.option_cf_return(1.00, 0.0) == -100.0


def test_option_cf_return_sign_is_direction_independent():
    # We always buy-to-open; a call and a put with the same ask/exit score alike.
    assert cc.option_cf_return(1.0, 1.2) == cc.option_cf_return(1.0, 1.2)


def test_option_cf_return_bad_input():
    assert cc.option_cf_return(None, 1.5) is None
    assert cc.option_cf_return(0.0, 1.5) is None
    assert cc.option_cf_return(-1.0, 1.5) is None
    assert cc.option_cf_return(1.0, None) is None
    assert cc.option_cf_return(1.0, -0.5) is None
    assert cc.option_cf_return("x", "y") is None


# --------------------------------------------------------------------------- #
# _exit_mid_from_quote
# --------------------------------------------------------------------------- #
def test_exit_mid_prefers_mid():
    assert cc._exit_mid_from_quote({"bid": 1.0, "ask": 2.0, "mid": 1.5}) == 1.5


def test_exit_mid_falls_back_to_ask_then_bid():
    assert cc._exit_mid_from_quote({"bid": 1.0, "ask": 2.0}) == 2.0
    assert cc._exit_mid_from_quote({"bid": 1.0}) == 1.0


def test_exit_mid_bare_number():
    assert cc._exit_mid_from_quote(1.25) == 1.25


def test_exit_mid_none_and_empty():
    assert cc._exit_mid_from_quote(None) is None
    assert cc._exit_mid_from_quote({}) is None
    assert cc._exit_mid_from_quote({"bid": 0, "ask": 0}) is None


# --------------------------------------------------------------------------- #
# Helpers to build a store
# --------------------------------------------------------------------------- #
def _feats(entry_ask, low_iv, contract):
    return {
        "raw": {"underlying_price": 100.0},
        "capskip": {"entry_ask": entry_ask, "low_iv": low_iv, "base_cap": 2,
                    "eff_cap": 1, "realized_vol": 0.12, "direction": "call",
                    "contract": contract},
    }


def _log_capskip(store, occ, underlying, entry_ask, low_iv):
    return store.log_decision(
        symbol=occ, underlying=underlying, strat="t",
        features=_feats(entry_ask, low_iv, occ), quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="cap-skip-cf",
    )


# --------------------------------------------------------------------------- #
# resolve_due_capskips
# --------------------------------------------------------------------------- #
def test_resolver_books_option_return():
    store = EpisodeStore(":memory:")
    did = _log_capskip(store, "SPY_C", "SPY", 1.00, True)
    resolved = cc.resolve_due_capskips(store, lambda s: {"mid": 1.50})
    assert resolved == 1
    row = {r["decision_id"]: r for r in store._rows("SELECT * FROM episodes")}[did]
    assert row["outcome"] == "capskip_resolved"
    assert abs(row["net_pnl_pct"] - 50.0) < 1e-9
    store.close()


def test_resolver_ignores_non_capskip_skips():
    store = EpisodeStore(":memory:")
    other = store.log_decision(
        symbol="IWM_C", underlying="IWM", strat="t",
        features=_feats(1.0, False, "IWM_C"), quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="live-paper-blocked",
    )
    resolved = cc.resolve_due_capskips(store, lambda s: {"mid": 5.0})
    assert resolved == 0
    row = {r["decision_id"]: r for r in store._rows("SELECT * FROM episodes")}[other]
    assert row["outcome"] is None
    store.close()


def test_resolver_skips_missing_entry_ask():
    store = EpisodeStore(":memory:")
    store.log_decision(
        symbol="DIA_C", underlying="DIA", strat="t",
        features={"capskip": {}}, quote=None, modeled_cost=None,
        rule_action="CALL", rule_confidence=0.0, gate=None,
        chosen_action="SKIP", qty=1, mode="cap-skip-cf",
    )
    assert cc.resolve_due_capskips(store, lambda s: {"mid": 1.0}) == 0
    store.close()


def test_resolver_skips_missing_quote():
    store = EpisodeStore(":memory:")
    _log_capskip(store, "SPY_C", "SPY", 1.00, True)
    assert cc.resolve_due_capskips(store, lambda s: None) == 0
    store.close()


def test_resolver_is_idempotent_via_open_filter():
    store = EpisodeStore(":memory:")
    _log_capskip(store, "SPY_C", "SPY", 1.00, True)
    assert cc.resolve_due_capskips(store, lambda s: {"mid": 1.50}) == 1
    # Already resolved -> open_capskips returns nothing -> no re-resolve.
    assert cc.resolve_due_capskips(store, lambda s: {"mid": 9.99}) == 0
    store.close()


# --------------------------------------------------------------------------- #
# summarize
# --------------------------------------------------------------------------- #
def test_summarize_splits_low_iv_vs_regular():
    store = EpisodeStore(":memory:")
    _log_capskip(store, "SPY_C", "SPY", 1.00, True)   # -> +50% (would-win)
    _log_capskip(store, "QQQ_P", "QQQ", 2.00, False)  # -> -50% (throttle helped)
    quotes = {"SPY_C": {"mid": 1.50}, "QQQ_P": {"mid": 1.00}}
    cc.resolve_due_capskips(store, lambda s: quotes.get(s))
    summary = cc.summarize(store)
    assert summary["all"]["trades"] == 2
    assert summary["low_iv"]["trades"] == 1
    assert summary["regular"]["trades"] == 1
    assert summary["low_iv"]["win_rate"] == 1.0
    assert summary["regular"]["win_rate"] == 0.0
    assert abs(summary["all"]["avg_return"] - 0.0) < 1e-9
    store.close()


def test_summarize_empty_store():
    store = EpisodeStore(":memory:")
    summary = cc.summarize(store)
    assert summary["all"]["trades"] == 0
    assert "no resolved cap-skips" in cc.format_summary(summary)
    store.close()


# --------------------------------------------------------------------------- #
# Self-test entry point
# --------------------------------------------------------------------------- #
def test_self_test_passes():
    assert cc._self_test() == 0
