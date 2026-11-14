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

Optional --history trading_history.json join: the live hook keeps exit
prices and net dollars honest-None (never estimated), so closed trades
classify as unresolved in the ledger. The bot's own history file carries a
mark-derived P&L per close, keyed by the same broker order_id. With
--history, unresolved/missing_coverage outcomes whose order_id joins a
history row are upgraded through outcome_states.classify_legacy_record —
explicitly labeled pnl_source='mark' (never 'fills'), with the ledger's own
state preserved in ledger_state/ledger_reason and
resolved_via='history_order_id_join'.

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
    classify_decision, classify_legacy_record,
    FILLED_RESOLVED, CONFIRMED_NO_FILL, POLICY_SKIP,
    UNRESOLVED, MISSING_COVERAGE,
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


def load_history_index(path):
    """order_id -> legacy trading_history record (read-only mark-P&L join).

    trading_history.json rows carry the mark-derived P&L stamped at close
    time by smart_trader.record_trade_outcome, keyed by broker order_id.
    The ledger deliberately keeps net_pnl_dollars honest-None unless it was
    resolved from broker fills, so this index lets the export recover a
    *labeled* mark net for unresolved rows — never presented as fills-based.
    First record wins on duplicate order_ids (history contains >=1 dup).
    """
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    trades = data.get("trades", []) if isinstance(data, dict) else data
    index = {}
    for rec in trades or []:
        if not isinstance(rec, dict):
            continue
        oid = rec.get("order_id")
        if oid and oid not in index:
            index[oid] = rec
    return index


def _resolve_via_history(outcome, broker_order_ids, history_index):
    """Upgrade an unresolved outcome via the trading-history order_id join.

    Only UNRESOLVED / MISSING_COVERAGE outcomes are touched; states the
    classifier already resolved pass through untouched. The recovered net
    keeps classify_legacy_record's provenance (pnl_source='mark' for
    mark-derived history P&L), and the ledger's own verdict is preserved in
    ledger_state / ledger_reason with resolved_via='history_order_id_join'.
    """
    if not history_index or outcome.get("state") not in (
            UNRESOLVED, MISSING_COVERAGE):
        return outcome
    for oid in broker_order_ids:
        rec = history_index.get(oid)
        if rec is None:
            continue
        legacy = classify_legacy_record(rec)
        if legacy["state"] in (FILLED_RESOLVED, CONFIRMED_NO_FILL):
            merged = dict(outcome)
            merged.update(
                state=legacy["state"],
                reason=legacy["reason"],
                net_pnl_dollars=legacy["net_pnl_dollars"],
                pnl_source=legacy["pnl_source"],
                resolved_via="history_order_id_join",
                history_order_id=oid,
                ledger_state=outcome.get("state"),
                ledger_reason=outcome.get("reason"))
            return merged
    return outcome


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


def decision_rows(ledger, since=None, until=None, broker_env=None,
                  history_index=None):
    """Yield one joined export row per decision_id (intent decisions only).

    group_candidate snapshots are exported by group_rows(); here we join the
    candidate / shadow_candidate intent snapshots with their lifecycle
    events and classify the outcome explicitly. With history_index (from
    load_history_index), unresolved outcomes are additionally joined to the
    legacy mark P&L by broker order_id (labeled, see _resolve_via_history).
    """
    broker_env = broker_env or _broker_env_default()
    events_map = _events_by_decision(ledger)
    # One bulk read of intent snapshots grouped by decision_id (a per-decision
    # ledger.snapshots() rescan here was O(n^2) across ~14k decisions).
    snaps_map = {}
    for s in ledger.snapshots():
        if s.get("kind") != "group_candidate":
            snaps_map.setdefault(s.get("decision_id"), []).append(s)
    seen = set()
    for snap in ledger.snapshots(since=since, until=until):
        if snap.get("kind") == "group_candidate":
            continue
        did = snap.get("decision_id")
        if not did or did in seen:
            continue
        seen.add(did)
        snaps = snaps_map.get(did, [])
        events = events_map.get(did, [])
        broker_order_ids = sorted({
            e.get("broker_order_id") for e in events
            if e.get("broker_order_id")})
        outcome = _resolve_via_history(
            classify_decision(snaps, events), broker_order_ids,
            history_index)
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
            "broker_order_ids": broker_order_ids,
            "outcome": outcome,
        }


def _selected_outcomes_by_group(ledger, events_map, history_index=None):
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
        events = rec["events"]
        broker_order_ids = sorted({
            e.get("broker_order_id") for e in events
            if e.get("broker_order_id")})
        by_group[gid] = (did, _resolve_via_history(
            classify_decision(snaps, events), broker_order_ids,
            history_index))
    return by_group


def group_rows(ledger, since=None, until=None, broker_env=None,
               history_index=None):
    """Yield one ranking_eval-compatible row per candidate group."""
    broker_env = broker_env or _broker_env_default()
    events_map = _events_by_decision(ledger)
    selected_map = _selected_outcomes_by_group(ledger, events_map,
                                               history_index)
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

        # production-shaped close: the live hook keeps the exit price and
        # net dollars honest-None, so the ledger classifies unresolved; the
        # mark P&L lives in trading_history under the same order_id.
        did_mark = led.record_snapshot(
            kind="candidate", accepted=True, symbol="UPS261218P00100000",
            underlying="UPS", event_at="2026-11-13T15:07:00Z")
        led.record_event(decision_id=did_mark, event_type="submitted",
                         broker_order_id="ord-4", source="live_hook")
        led.record_event(decision_id=did_mark, event_type="fill",
                         broker_order_id="ord-4", qty=2, price=8.25,
                         source="live_hook", payload={"side": "buy"})
        led.record_event(decision_id=did_mark, event_type="exit",
                         broker_order_id="ord-4", qty=2, price=None,
                         source="live_hook")
        led.record_event(decision_id=did_mark, event_type="outcome",
                         broker_order_id="ord-4", source="live_hook",
                         payload={"class": "realized",
                                  "gross_pnl_pct": -17.4545,
                                  "net_pnl_dollars": None})
        history = {"ord-4": {"order_id": "ord-4", "pnl": -288.0,
                             "pnl_percent": -17.4545}}

        # ----------------------------- decisions export ------------------ #
        rows = list(decision_rows(led, broker_env="paper"))
        by_id = {r["decision_id"]: r for r in rows}
        check("exports 4 intent decisions (groups excluded)",
              len(rows) == 4 and gid not in by_id)
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

        # ------------------- history join (mark-net recovery) ------------ #
        check("price-None exit stays unresolved without history",
              by_id[did_mark]["outcome"]["state"] == UNRESOLVED
              and by_id[did_mark]["outcome"]["net_pnl_dollars"] is None)
        by_id_h = {r["decision_id"]: r
                   for r in decision_rows(led, broker_env="paper",
                                          history_index=history)}
        o = by_id_h[did_mark]["outcome"]
        check("history join resolves mark net (-288, source=mark, labeled)",
              o["state"] == FILLED_RESOLVED
              and o["net_pnl_dollars"] == -288.0
              and o["pnl_source"] == "mark"
              and o["reason"] == "legacy_mark"
              and o["resolved_via"] == "history_order_id_join"
              and o["history_order_id"] == "ord-4"
              and o["ledger_state"] == UNRESOLVED
              and o["ledger_reason"] == "open_residual_qty")
        check("history join never touches ledger-resolved outcomes",
              by_id_h[did]["outcome"] == by_id[did]["outcome"]
              and by_id_h[did_nofill]["outcome"]
              == by_id[did_nofill]["outcome"])

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
        check("JSONL round-trip (4 decisions, 1 group)",
              n_dec == 4 and n_grp == 1
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
    ap.add_argument("--history", default=None,
                    help="trading_history.json path: join unresolved rows "
                         "to the legacy mark P&L by broker order_id "
                         "(labeled pnl_source='mark'; read-only)")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)

    if args.selftest:
        return _selftest()

    if not args.out and not args.groups_out:
        ap.error("provide --out and/or --groups-out (or --selftest)")
    if not os.path.exists(args.db):
        print(f"ledger not found: {args.db}", file=sys.stderr)
        return 2
    history_index = None
    if args.history:
        if not os.path.exists(args.history):
            print(f"history not found: {args.history}", file=sys.stderr)
            return 2
        history_index = load_history_index(args.history)
        print(f"history join: {len(history_index)} order_ids "
              f"from {args.history}")

    from research_ledger import ResearchLedger
    led = ResearchLedger(args.db)
    env = args.broker_env or _broker_env_default()
    try:
        if args.out:
            n = _write_jsonl(
                decision_rows(led, since=args.since, until=args.until,
                              broker_env=env, history_index=history_index),
                args.out)
            print(f"wrote {n} decisions -> {args.out}")
        if args.groups_out:
            n = _write_jsonl(
                group_rows(led, since=args.since, until=args.until,
                           broker_env=env, history_index=history_index),
                args.groups_out)
            print(f"wrote {n} groups -> {args.groups_out}")
    finally:
        led.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
