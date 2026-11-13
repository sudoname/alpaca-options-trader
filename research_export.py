"""Read-only export: join intent, quotes, broker events, and outcomes.

Phase 1 export tool (master prompt). Walks the research ledger and emits
JSONL that downstream studies (ranking_eval.py, skeptic_impact.py, paired
reports) can consume without touching SQLite or the live trading path.

Two outputs, both joined on STABLE IDs (decision_id / group_id /
candidate_id / broker_order_id):

  decisions JSONL  one line per decision_id: the intent snapshot(s), the
                   entry quote, every broker lifecycle event, and the
                   explicit outcome classification from outcome_states
                   (filled_resolved / confirmed_no_fill / policy_skip /
                   unresolved / missing_coverage). Nothing here guesses:
                   a missing outcome is exported as missing, never as $0.

  groups JSONL     one line per candidate group (kind=group_candidate,
                   commit 080483c): every contract the selector considered,
                   with the SELECTED candidate's realized net outcome
                   attached by following extra_json.group_id from the
                   accepted candidate snapshot. Non-selected candidates get
                   net_outcome=null + outcome_state=counterfactual_required
                   (their hypothetical P&L needs quote replay, which the
                   ledger cannot provide — it stores one quote per decision).
                   Format is ranking_eval.py --groups compatible.

Source labels are preserved end to end: every event keeps its ledger
`source` ('live_hook' | 'reconciler' | 'resolver' | 'test'), every row is
stamped with `broker_env` ('paper' | 'live', from --broker-env or
ALPACA_PAPER), and fills get a combined `fill_source_label` such as
'paper_broker_fill' or 'synthetic'. Quote provenance (observed / estimated /
stale / missing) passes through from the snapshot's provenance_json.

Usage:
    python research_export.py --db research_ledger.db \
        --out export_decisions.jsonl --groups-out export_groups.jsonl
    python research_export.py --selftest

Read-only by construction: only SELECTs are issued against the ledger.
"""
import argparse
import json
import os
import sys

from outcome_states import (
    classify_decision, FILLED_RESOLVED, CONFIRMED_NO_FILL, POLICY_SKIP,
)

_EVENT_FIELDS = ("event_type", "event_at_utc", "received_at_utc",
                 "broker_order_id", "qty", "price", "fees", "status",
                 "source")
_FILL_EVENTS = ("fill", "partial_fill")


def _jload(text):
    """Tolerant JSON column read: dict or {}."""
    if isinstance(text, dict):
        return text
    if isinstance(text, str) and text:
        try:
            parsed = json.loads(text)
            if isinstance(parsed, dict):
                return parsed
        except (ValueError, TypeError):
            pass
    return {}


def _broker_env_default():
    paper = os.environ.get("ALPACA_PAPER", "true").strip().lower()
    return "paper" if paper == "true" else "live"


def fill_source_label(event_source, broker_env):
    """Combined provenance label for a fill event.

    'test' ledger rows are synthetic fixtures; everything written by the
    live hook or the reconciler is an actual broker fill on whichever
    account mode the process ran under (paper vs live).
    """
    src = str(event_source or "").lower()
    if src == "test":
        return "synthetic"
    if src in ("live_hook", "reconciler", "resolver"):
        return f"{broker_env}_broker_fill"
    return f"{broker_env}_{src or 'unknown'}"


def _events_by_decision(ledger):
    """One bulk read of the events table, grouped by decision_id."""
    rows = ledger._rows(
        "SELECT * FROM events "
        "ORDER BY COALESCE(event_at_utc, received_at_utc), received_at_utc")
    grouped = {}
    for r in rows:
        grouped.setdefault(r.get("decision_id"), []).append(r)
    return grouped


def _snapshot_intent(snap):
    """Compact intent view of one snapshot row (raw columns -> export dict)."""
    extra = _jload(snap.get("extra_json"))
    return {
        "snapshot_id": snap.get("snapshot_id"),
        "kind": snap.get("kind"),
        "accepted": snap.get("accepted"),
        "reject_code": snap.get("reject_code"),
        "event_at_utc": snap.get("event_at_utc"),
        "symbol": snap.get("symbol"),
        "underlying": snap.get("underlying"),
        "expiration": snap.get("expiration"),
        "strike": snap.get("strike"),
        "option_type": snap.get("option_type"),
        "multiplier": snap.get("multiplier"),
        "direction": snap.get("direction"),
        "score": snap.get("score"),
        "p_win": snap.get("p_win"),
        "ev": snap.get("ev"),
        "underlying_price": snap.get("underlying_price"),
        "iv": snap.get("iv"),
        "delta": snap.get("delta"),
        "quote": {
            "bid": snap.get("quote_bid"),
            "ask": snap.get("quote_ask"),
            "bid_size": snap.get("quote_bid_size"),
            "ask_size": snap.get("quote_ask_size"),
            "ts_utc": snap.get("quote_ts_utc"),
            "age_s": snap.get("quote_age_s"),
        },
        "provenance": _jload(snap.get("provenance_json")) or None,
        "sizing": _jload(snap.get("sizing_json")) or None,
        "risk": _jload(snap.get("risk_json")) or None,
        "strategy_version": snap.get("strategy_version"),
        "config_hash": snap.get("config_hash"),
        "code_rev": snap.get("code_rev"),
        "scan_id": extra.get("scan_id"),
        "group_id": extra.get("group_id"),
        "candidate_id": extra.get("candidate_id"),
        "extra": extra or None,
    }


def decision_rows(ledger, since=None, until=None, broker_env=None):
    """Yield one joined export row per decision_id (intent decisions only).

    group_candidate snapshots are exported by group_rows(); here we join the
    candidate / shadow_candidate intent snapshots with their lifecycle
    events and classify the outcome explicitly.
    """
    broker_env = broker_env or _broker_env_default()
    events_map = _events_by_decision(ledger)
    seen = set()
    for snap in ledger.snapshots(since=since, until=until):
        if snap.get("kind") == "group_candidate":
            continue
        did = snap.get("decision_id")
        if not did or did in seen:
            continue
        seen.add(did)
        snaps = [s for s in ledger.snapshots()
                 if s.get("decision_id") == did
                 and s.get("kind") != "group_candidate"]
        events = events_map.get(did, [])
        outcome = classify_decision(snaps, events)
        fill_sources = sorted({
            fill_source_label(e.get("source"), broker_env)
            for e in events if e.get("event_type") in _FILL_EVENTS})
        yield {
            "decision_id": did,
            "broker_env": broker_env,
            "snapshots": [_snapshot_intent(s) for s in snaps],
            "events": [
                {**{k: e.get(k) for k in _EVENT_FIELDS},
                 "payload": _jload(e.get("payload_json")) or None}
                for e in events],
            "fill_sources": fill_sources,
            "broker_order_ids": sorted({
                e.get("broker_order_id") for e in events
                if e.get("broker_order_id")}),
            "outcome": outcome,
        }


def _selected_outcomes_by_group(ledger, events_map):
    """group_id -> (selected decision_id, classified outcome).

    Follows the stable-ID link written by _record_candidate_group: the
    accepted candidate snapshot carries extra_json.group_id.
    """
    by_group = {}
    for snap in ledger.snapshots(accepted=True, kind="candidate"):
        gid = _jload(snap.get("extra_json")).get("group_id")
        did = snap.get("decision_id")
        if not gid or not did or gid in by_group:
            continue
        rec = ledger.get_decision(did)
        snaps = [s for s in rec["snapshots"]
                 if s.get("kind") != "group_candidate"]
        by_group[gid] = (did, classify_decision(snaps, rec["events"]))
    return by_group


def group_rows(ledger, since=None, until=None, broker_env=None):
    """Yield one ranking_eval-compatible row per candidate group."""
    broker_env = broker_env or _broker_env_default()
    events_map = _events_by_decision(ledger)
    selected_map = _selected_outcomes_by_group(ledger, events_map)
    groups = {}
    order = []
    for snap in ledger.snapshots(since=since, until=until,
                                 kind="group_candidate"):
        gid = snap.get("decision_id")
        if gid not in groups:
            groups[gid] = []
            order.append(gid)
        groups[gid].append(snap)
    for gid in order:
        snaps = groups[gid]
        first_extra = _jload(snaps[0].get("extra_json"))
        selected = selected_map.get(gid)
        candidates = []
        for s in snaps:
            extra = _jload(s.get("extra_json"))
            is_selected = bool(s.get("accepted"))
            cand = {
                "candidate_id": s.get("snapshot_id"),
                "symbol": s.get("symbol"),
                "selected": is_selected,
                "skip_reason": (None if is_selected
                                else s.get("reject_code")),
                "score": s.get("score"),
                "bid": s.get("quote_bid"),
                "ask": s.get("quote_ask"),
                "quote_ts": s.get("quote_ts_utc"),
                "iv": s.get("iv"),
                "delta": s.get("delta"),
                "dte": extra.get("dte"),
                "rank": extra.get("rank"),
                "net_outcome": None,
                "outcome_state": "counterfactual_required",
                "outcome_source": None,
            }
            if is_selected and selected is not None:
                did, outcome = selected
                cand["decision_id"] = did
                cand["outcome_state"] = outcome["state"]
                cand["outcome_source"] = outcome["pnl_source"]
                if outcome["state"] == FILLED_RESOLVED:
                    cand["net_outcome"] = outcome["net_pnl_dollars"]
                elif outcome["state"] in (CONFIRMED_NO_FILL, POLICY_SKIP):
                    cand["net_outcome"] = 0.0
            elif is_selected:
                cand["outcome_state"] = "selected_unlinked"
            candidates.append(cand)
        yield {
            "group_id": gid,
            "scan_id": first_extra.get("scan_id"),
            "underlying": snaps[0].get("underlying"),
            "direction": snaps[0].get("direction"),
            "event_at_utc": snaps[0].get("event_at_utc"),
            "broker_env": broker_env,
            "n_considered": first_extra.get("n_considered"),
            "skip_counts": first_extra.get("skip_counts"),
            "n_candidates": len(candidates),
            "n_multi": int(len(candidates) > 1),
            "candidates": candidates,
        }


def _write_jsonl(rows, path):
    n = 0
    with open(path, "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, sort_keys=True) + "\n")
            n += 1
    return n


# --------------------------------------------------------------------------- #
# self-test (hermetic: temp DB, no network, no live files)                     #
# --------------------------------------------------------------------------- #

def _selftest():
    import tempfile
    from research_ledger import ResearchLedger
    from research_capture import group_id_for, candidate_id_for

    ok = True

    def check(name, cond):
        nonlocal ok
        print(f"  {'PASS' if cond else 'FAIL'}  {name}")
        ok = ok and bool(cond)

    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ledger.db")
        led = ResearchLedger(db)

        # --- group of three candidates, one selected --------------------- #
        scan_id = "scan-0001"
        syms = ("SPY260117C00600000", "SPY260117C00605000",
                "SPY260117C00610000")
        gid = group_id_for(scan_id, "SPY", "call", list(syms))
        for i, sym in enumerate(syms):
            led.record_snapshot(
                decision_id=gid, kind="group_candidate",
                accepted=(i == 0),
                reject_code=None if i == 0 else "not_selected",
                symbol=sym, underlying="SPY", direction="call",
                score=90 - i, quote={"bid": 1.0 + i, "ask": 1.1 + i},
                event_at="2026-11-13T15:00:00Z",
                snapshot_id=candidate_id_for(gid, sym),
                extra={"scan_id": scan_id, "group_id": gid,
                       "candidate_id": candidate_id_for(gid, sym),
                       "n_considered": 3, "rank": i, "dte": 14})

        # selected candidate's own accepted decision, filled + resolved
        did = led.record_snapshot(
            kind="candidate", accepted=True, symbol=syms[0],
            underlying="SPY", direction="call", score=90,
            quote={"bid": 1.0, "ask": 1.1},
            event_at="2026-11-13T15:00:01Z",
            extra={"scan_id": scan_id, "group_id": gid,
                   "candidate_id": candidate_id_for(gid, syms[0])})
        led.record_event(decision_id=did, event_type="submitted",
                         broker_order_id="ord-1", source="live_hook")
        led.record_event(decision_id=did, event_type="fill",
                         broker_order_id="ord-1", qty=2, price=1.05,
                         source="live_hook", payload={"side": "buy"})
        led.record_event(decision_id=did, event_type="exit",
                         broker_order_id="ord-2", qty=2, price=1.30,
                         source="reconciler")
        led.record_event(decision_id=did, event_type="outcome",
                         broker_order_id="ord-2", source="reconciler",
                         payload={"class": "realized",
                                  "net_pnl_dollars": 50.0,
                                  "gross_pnl_pct": 23.8})

        # a gate-blocked candidate (policy skip) and a cancelled no-fill
        did_skip = led.record_snapshot(
            kind="candidate", accepted=False,
            reject_code="profit_gate:pop_too_low", symbol=syms[1],
            underlying="SPY", event_at="2026-11-13T15:05:00Z")
        did_nofill = led.record_snapshot(
            kind="candidate", accepted=True, symbol=syms[2],
            underlying="SPY", event_at="2026-11-13T15:06:00Z")
        led.record_event(decision_id=did_nofill, event_type="submitted",
                         broker_order_id="ord-3", source="live_hook")
        led.record_event(decision_id=did_nofill, event_type="cancel",
                         broker_order_id="ord-3", source="live_hook",
                         payload={"reason": "broker_canceled_no_fill"})

        # ----------------------------- decisions export ------------------ #
        rows = list(decision_rows(led, broker_env="paper"))
        by_id = {r["decision_id"]: r for r in rows}
        check("exports 3 intent decisions (groups excluded)",
              len(rows) == 3 and gid not in by_id)
        filled = by_id.get(did, {})
        check("filled decision: outcome reported, net=50, source=reported",
              filled.get("outcome", {}).get("state") == FILLED_RESOLVED
              and filled["outcome"]["net_pnl_dollars"] == 50.0
              and filled["outcome"]["pnl_source"] == "reported")
        check("fill source label = paper_broker_fill",
              filled.get("fill_sources") == ["paper_broker_fill"])
        check("broker order ids joined",
              filled.get("broker_order_ids") == ["ord-1", "ord-2"])
        check("intent snapshot carries group link + quote",
              filled["snapshots"][0]["group_id"] == gid
              and filled["snapshots"][0]["quote"]["bid"] == 1.0)
        check("gate block exports as policy_skip with reason",
              by_id[did_skip]["outcome"]["state"] == POLICY_SKIP
              and by_id[did_skip]["outcome"]["reason"]
              == "profit_gate:pop_too_low")
        check("cancelled order exports as confirmed_no_fill (0, broker)",
              by_id[did_nofill]["outcome"]["state"] == CONFIRMED_NO_FILL
              and by_id[did_nofill]["outcome"]["net_pnl_dollars"] == 0.0)

        # ----------------------------- groups export --------------------- #
        groups = list(group_rows(led, broker_env="paper"))
        check("one group exported with 3 candidates, multi flag set",
              len(groups) == 1 and groups[0]["n_candidates"] == 3
              and groups[0]["n_multi"] == 1
              and groups[0]["scan_id"] == scan_id)
        g = groups[0]
        sel = [c for c in g["candidates"] if c["selected"]]
        alt = [c for c in g["candidates"] if not c["selected"]]
        check("selected candidate linked to decision + net_outcome 50",
              len(sel) == 1 and sel[0]["decision_id"] == did
              and sel[0]["net_outcome"] == 50.0
              and sel[0]["outcome_state"] == FILLED_RESOLVED)
        check("alternatives carry counterfactual_required, net null",
              len(alt) == 2 and all(
                  c["net_outcome"] is None
                  and c["outcome_state"] == "counterfactual_required"
                  for c in alt))

        # ----------------------------- file round-trip ------------------- #
        dec_path = os.path.join(tmp, "decisions.jsonl")
        grp_path = os.path.join(tmp, "groups.jsonl")
        n_dec = _write_jsonl(decision_rows(led, broker_env="paper"),
                             dec_path)
        n_grp = _write_jsonl(group_rows(led, broker_env="paper"), grp_path)
        with open(grp_path, encoding="utf-8") as fh:
            reparsed = [json.loads(line) for line in fh]
        check("JSONL round-trip (3 decisions, 1 group)",
              n_dec == 3 and n_grp == 1
              and reparsed[0]["group_id"] == gid)

        # ranking_eval compatibility: its loader wants group_id+candidates
        try:
            from ranking_eval import load_groups_jsonl
            loaded = load_groups_jsonl(grp_path)
            check("groups file loads in ranking_eval.load_groups_jsonl",
                  len(loaded) == 1
                  and len(loaded[0].get("candidates") or []) == 3)
        except ImportError:
            check("groups file loads in ranking_eval.load_groups_jsonl "
                  "(skipped)", True)

        led.close()

    print("SELFTEST", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--db", default="research_ledger.db")
    ap.add_argument("--out", default=None,
                    help="decisions JSONL output path")
    ap.add_argument("--groups-out", default=None,
                    help="candidate-groups JSONL output path "
                         "(ranking_eval --groups compatible)")
    ap.add_argument("--since", default=None)
    ap.add_argument("--until", default=None)
    ap.add_argument("--broker-env", default=None,
                    choices=("paper", "live"),
                    help="account-mode label stamped on every row "
                         "(default: from ALPACA_PAPER)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.out and not args.groups_out:
        ap.error("provide --out and/or --groups-out (or --selftest)")
    if not os.path.exists(args.db):
        print(f"ledger not found: {args.db}", file=sys.stderr)
        return 2

    from research_ledger import ResearchLedger
    led = ResearchLedger(args.db)
    env = args.broker_env or _broker_env_default()
    try:
        if args.out:
            n = _write_jsonl(
                decision_rows(led, since=args.since, until=args.until,
                              broker_env=env), args.out)
            print(f"wrote {n} decisions -> {args.out}")
        if args.groups_out:
            n = _write_jsonl(
                group_rows(led, since=args.since, until=args.until,
                           broker_env=env), args.groups_out)
            print(f"wrote {n} groups -> {args.groups_out}")
    finally:
        led.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
