"""Export frozen ledger decisions into paired_execution candidate JSONL.

Missing EV and exit policy remain null/empty; never infer them from an outcome.
The decision-time quote is the immutable ledger snapshot, while subsequent
quotes come from the independent forward poller.
"""
from __future__ import annotations

import argparse
import json
import os
from typing import Optional

from oracle.ml.ingest import connect


def _json(value):
    try:
        obj = json.loads(value or "{}")
        return obj if isinstance(obj, dict) else {}
    except (TypeError, ValueError):
        return {}


def export_candidates(ledger_path: str, out_path: str) -> dict:
    conn = connect(ledger_path)
    try:
        rows = conn.execute("""SELECT snapshot_id, decision_id, kind, event_at_utc,
                              symbol, quote_bid, quote_ask, quote_bid_size,
                              quote_ask_size, quote_age_s, sizing_json,
                              prediction_json, extra_json, ev
                              FROM snapshots WHERE accepted=1
                              AND kind IN ('candidate','shadow_candidate')
                              AND symbol IS NOT NULL
                              ORDER BY event_at_utc, snapshot_id""").fetchall()
        output = []
        seen = set()
        for row in rows:
            did = row["decision_id"]
            if did in seen:  # retries yield multiple snapshots; first immutable one wins
                continue
            seen.add(did)
            sizing, pred, extra = (_json(row[k]) for k in
                                   ("sizing_json", "prediction_json", "extra_json"))
            raw_levels = pred.get("exit_policy") or {}
            levels = {dst: raw_levels[src] for src, dst in (
                ("stop_loss_pct", "stop_loss_percent"),
                ("take_profit_pct", "take_profit_percent"),
                ("trailing_stop_distance", "trailing_stop_distance"))
                if raw_levels.get(src) is not None}
            gross = extra.get("gross_expected_value")
            entry_limit = extra.get("entry_limit") or {}
            if gross is None:
                gross = entry_limit.get("gross_ev")
            if gross is None and row["ev"] is not None and extra.get("round_trip_costs") is not None:
                gross = float(row["ev"]) + float(extra["round_trip_costs"])
            qty = sizing.get("qty")
            try:
                qty = int(qty)
            except (TypeError, ValueError):
                qty = None
            output.append({
                "decision_id": did, "kind": row["kind"],
                "decision_time": row["event_at_utc"], "symbol": row["symbol"],
                "qty": qty, "gross_expected_value": gross,
                "levels": levels,
                "limit_expiry_s": entry_limit.get("expiry_s", 30),
                "decision_quote": {"bid": row["quote_bid"], "ask": row["quote_ask"],
                                   "bid_size": row["quote_bid_size"],
                                   "ask_size": row["quote_ask_size"],
                                   "quote_age_s": row["quote_age_s"]},
            })
    finally:
        conn.close()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    temp = out_path + ".tmp"
    with open(temp, "w", encoding="utf-8") as fh:
        for row in output:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(temp, out_path)
    return {"decisions": len(output),
            "missing_gross_ev": sum(r["gross_expected_value"] is None for r in output),
            "missing_qty": sum(r["qty"] is None for r in output),
            "missing_exit_policy": sum(not r["levels"] for r in output)}


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--ledger", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    print(json.dumps(export_candidates(args.ledger, args.out), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
