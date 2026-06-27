"""Unit tests for evidence_attribution (pure, in-memory, no network)."""

import json

import evidence_attribution as ea


def _raw(net_pct, evidence, mode="backfill"):
    return {
        "features_json": json.dumps({"evidence": evidence}),
        "net_pnl_pct": net_pct,
        "net_pnl_dollars": net_pct,
        "outcome": "win" if net_pct > 0 else "loss",
        "mode": mode,
        "decision_id": None,
    }


def test_self_test_passes():
    assert ea._self_test() == 0


def test_normalize_flattens_evidence_and_pnl():
    rec = ea._normalize(_raw(5.0, {"direction": "up", "iv_bucket": "low"}))
    assert rec["direction"] == "up"
    assert rec["iv_bucket"] == "low"
    assert rec["pnl"] == 5.0
    assert rec["pnl_percent"] == 5.0


def test_normalize_drops_rows_without_pnl():
    row = {"features_json": "{}", "net_pnl_pct": None, "net_pnl_dollars": None}
    assert ea._normalize(row) is None


def test_normalize_never_raises_on_garbage():
    for junk in (None, 42, "x", {}, {"features_json": "{bad"}):
        ea._normalize(junk)  # must not raise


def test_leaderboard_groups_and_sorts_by_ev():
    rows = [ea._normalize(_raw(9.0, {"direction": "up"})) for _ in range(12)]
    rows += [ea._normalize(_raw(-7.0, {"direction": "down"})) for _ in range(15)]
    board = ea.leaderboard(rows, "direction")
    feats = [r.feature for r in board]
    assert set(feats) == {"up", "down"}
    # EV-sorted: 'up' (positive) precedes 'down' (negative).
    assert board[0].feature == "up"
    down = next(r for r in board if r.feature == "down")
    assert down.trades == 15
    assert down.verdict == ea.V_NEGATIVE


def test_small_cohort_never_better_than_weak():
    rows = [ea._normalize(_raw(25.0, {"direction": "up"})) for _ in range(3)]
    board = ea.leaderboard(rows, "direction")
    assert board[0].verdict == ea.V_WEAK


def test_agent_dimension_only_convicted():
    rows = []
    for _ in range(8):
        rows.append(ea._normalize(_raw(6.0, {
            "agent_votes": {
                "TrendAgent": {"net": 0.7, "conf": 0.9},
                "FlatAgent": {"net": 0.0, "conf": 0.0},
            }})))
    board = ea.leaderboard(rows, "agent")
    names = {r.feature for r in board}
    assert "TrendAgent" in names
    assert "FlatAgent" not in names


def test_format_markdown_empty_and_populated():
    assert "No closed episodes" in ea.format_markdown({})
    rows = [ea._normalize(_raw(8.0, {"direction": "up"})) for _ in range(6)]
    tables = ea.compute_all(rows=rows)
    md = ea.format_markdown(tables)
    assert "Evidence-EV Leaderboards" in md
    assert "| up |" in md
