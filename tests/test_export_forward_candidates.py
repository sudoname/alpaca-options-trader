import json
from datetime import datetime, timezone

from oracle.ml.export_forward_candidates import export_candidates
from oracle.ml.paired_execution import compare_candidate
from exit_replay import QuoteTick
from research_ledger import ResearchLedger


def test_frozen_snapshot_exports_gross_ev_and_exit_levels(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = ResearchLedger(str(db))
    did = ledger.record_snapshot(
        decision_id="d1", accepted=True, symbol="SPY261120C00700000",
        quote={"bid": 1.00, "ask": 1.10, "bid_size": 20,
               "ask_size": 20, "age_s": 1},
        sizing={"qty": 1},
        prediction={"exit_policy": {"stop_loss_pct": 10,
                                    "take_profit_pct": 20}},
        extra={"gross_expected_value": 10.0},
        event_at="2026-11-20T14:30:00+00:00")
    ledger.close()
    out = tmp_path / "candidates.jsonl"
    stats = export_candidates(str(db), str(out))
    assert stats == {"decisions": 1, "missing_gross_ev": 0,
                     "missing_qty": 0, "missing_exit_policy": 0}
    row = json.loads(out.read_text())
    assert row["decision_id"] == did
    assert row["levels"] == {"stop_loss_percent": 10,
                              "take_profit_percent": 20}
    assert row["decision_quote"]["ask"] == 1.1
    # A quote recorded at the decision can arm the policy before the first poll.
    rows = [QuoteTick("2026-11-20T14:30:01+00:00", 1.00, 1.10, 20, 20),
            QuoteTick("2026-11-20T14:30:02+00:00", 1.30, 1.35, 20, 20)]
    result = compare_candidate(row, rows)
    assert result["market"]["entry"]["status"] == "filled"


def test_unknown_ev_and_quantity_are_explicit(tmp_path):
    db = tmp_path / "ledger.db"
    ledger = ResearchLedger(str(db))
    ledger.record_snapshot(accepted=True, symbol="SPY261120C00700000",
                           event_at="2026-11-20T14:30:00+00:00")
    ledger.close()
    out = tmp_path / "candidates.jsonl"
    stats = export_candidates(str(db), str(out))
    assert stats["missing_gross_ev"] == stats["missing_qty"] == 1
    candidate = json.loads(out.read_text())
    assert compare_candidate(candidate, [])["status"] == "UNRESOLVED_QUANTITY"
