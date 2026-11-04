"""
Operator CLI for the Oracle research & validation layer.

Subcommands (all read-only except ``reconcile --apply``):

  inspect    show one decision (snapshots + lifecycle events) or ledger stats
  reconcile  reconcile outcomes against the broker (dry-run by default)
  replay     deterministic replay of recorded decisions under an execution model
  compare    experiments registry / shadow-vs-baseline / component report
  report     chronological validation report (purged splits, walk-forward, ...)

Examples:
  python research_cli.py inspect --stats
  python research_cli.py inspect 7f3a... --db research_ledger.db
  python research_cli.py reconcile --no-network
  python research_cli.py reconcile --apply --activities-after 2026-11-01T00:00:00Z
  python research_cli.py replay --scenario all --since 2026-10-01
  python research_cli.py compare experiments
  python research_cli.py compare shadow --variant v2
  python research_cli.py compare components
  python research_cli.py report --embargo-min 390 --folds 5

Replay/compare/report go through ReplayBroker where possible: a read-only
wrapper that structurally cannot submit orders. This CLI never talks to a
broker except through research_outcomes' GET-only reconciler fetchers.
"""

import argparse
import json
import sys

DEFAULT_DB = "research_ledger.db"
DEFAULT_XDB = "research_experiments.db"


def _open_ledger(path):
    from research_ledger import ResearchLedger
    return ResearchLedger(path)


def _print(obj):
    print(json.dumps(obj, indent=2, default=str))


def _shadow_outcome_summary(led, variant, since=None, until=None):
    """Counterfactual outcome summary for one shadow variant.

    Joins accepted ``shadow_candidate`` snapshots for the variant to their
    ``class: simulated`` outcome events (written by
    research_outcomes.resolve_shadow). Underlying-move counterfactuals only —
    not option P/L.
    """
    marker = f'"shadow_variant": "{variant}"'
    entries = {}  # decision_id -> direction
    for s in led.snapshots(since=since, until=until, kind="shadow_candidate"):
        if s.get("accepted") and marker in (s.get("extra_json") or ""):
            entries[s["decision_id"]] = s.get("direction") or "?"
    out = {"n_entries": len(entries), "n_resolved": 0, "hit_rate": None,
           "avg_pnl_pct": None, "median_pnl_pct": None, "by_direction": {}}
    if not entries:
        return out
    pnls, by_dir = [], {}
    rows = led._rows("SELECT decision_id, payload_json FROM events "
                     "WHERE event_type = 'outcome'", ())
    for r in rows:
        did = r["decision_id"]
        if did not in entries:
            continue
        try:
            payload = json.loads(r.get("payload_json") or "{}") or {}
        except ValueError:
            continue
        if payload.get("class") != "simulated":
            continue
        pnl = payload.get("counterfactual_pnl_pct")
        if not isinstance(pnl, (int, float)):
            continue
        pnls.append(float(pnl))
        by_dir.setdefault(entries[did], []).append(float(pnl))
    if pnls:
        pnls.sort()
        n = len(pnls)
        mid = n // 2
        med = pnls[mid] if n % 2 else (pnls[mid - 1] + pnls[mid]) / 2.0
        out["n_resolved"] = n
        out["hit_rate"] = round(sum(1 for p in pnls if p > 0) / n, 4)
        out["avg_pnl_pct"] = round(sum(pnls) / n, 4)
        out["median_pnl_pct"] = round(med, 4)
        for d, vals in sorted(by_dir.items()):
            out["by_direction"][d] = {
                "n": len(vals),
                "hit_rate": round(sum(1 for p in vals if p > 0) / len(vals), 4),
                "avg_pnl_pct": round(sum(vals) / len(vals), 4)}
    return out


# --------------------------------------------------------------------------- #
# Commands
# --------------------------------------------------------------------------- #
def cmd_inspect(args) -> int:
    led = _open_ledger(args.db)
    try:
        if args.stats or not args.decision_id:
            _print(led.stats())
            return 0
        rec = led.get_decision(args.decision_id)
        if not rec.get("snapshots") and not rec.get("events"):
            print(f"no snapshots or events for decision_id={args.decision_id}")
            return 1
        _print(rec)
        return 0
    finally:
        led.close()


def cmd_reconcile(args) -> int:
    """Delegate to research_outcomes.main — the single reconciliation entry
    point (dry-run default, --apply to write, GET-only broker calls)."""
    import research_outcomes
    argv = ["--db", args.db, "--skip-horizon-min", str(args.skip_horizon_min)]
    if args.apply:
        argv.append("--apply")
    if args.activities_after:
        argv += ["--activities-after", args.activities_after]
    if args.no_network:
        argv.append("--no-network")
    return research_outcomes.main(argv)


def cmd_replay(args) -> int:
    from research_replay import (ADVERSE_SCENARIOS, ReplayBroker,
                                 replay_cohort, run_scenarios)
    led = _open_ledger(args.db)
    broker = ReplayBroker(led)   # structurally cannot submit orders
    try:
        if args.scenario == "all":
            _print(run_scenarios(broker, since=args.since, until=args.until))
            return 0
        model = ADVERSE_SCENARIOS.get(args.scenario)
        if model is None:
            print(f"unknown scenario '{args.scenario}'; "
                  f"choices: {sorted(ADVERSE_SCENARIOS)} or 'all'")
            return 2
        res = replay_cohort(broker, model, since=args.since, until=args.until)
        _print(res if args.full else res["summary"])
        return 0
    finally:
        led.close()


def cmd_compare(args) -> int:
    if args.what == "experiments":
        from research_experiments import ExperimentRegistry
        reg = ExperimentRegistry(args.xdb)
        try:
            if args.id:
                exp = reg.get(args.id)
                if not exp:
                    print(f"no experiment with id {args.id}")
                    return 1
                _print({"experiment": exp, "history": reg.history(args.id)})
            else:
                _print(reg.list(status=args.status))
            return 0
        finally:
            reg.close()

    if args.what == "shadow":
        if not args.variant:
            print("compare shadow requires --variant")
            return 2
        from research_experiments import compare_shadow_to_baseline
        led = _open_ledger(args.db)
        try:
            out = compare_shadow_to_baseline(
                led, args.variant, since=args.since, until=args.until)
            out["counterfactual_outcomes"] = _shadow_outcome_summary(
                led, args.variant, since=args.since, until=args.until)
            _print(out)
            return 0
        finally:
            led.close()

    if args.what == "components":
        from research_experiments import component_report
        from research_replay import ReplayBroker
        led = _open_ledger(args.db)
        try:
            _print(component_report(ReplayBroker(led),
                                    since=args.since, until=args.until))
            return 0
        finally:
            led.close()

    print(f"unknown compare target '{args.what}'")
    return 2


def cmd_report(args) -> int:
    from research_validation import observations_from_ledger, validation_report
    led = _open_ledger(args.db)
    try:
        obs = observations_from_ledger(
            led, since=args.since, until=args.until,
            include_simulated=args.include_simulated)
    finally:
        led.close()
    if not obs:
        _print({"observations": 0,
                "note": "no resolved outcomes in window; run "
                        "`research_cli.py reconcile` first"})
        return 0
    _print(validation_report(obs, embargo_minutes=args.embargo_min,
                             n_folds=args.folds,
                             n_experiments=args.n_experiments,
                             seed=args.seed))
    return 0


# --------------------------------------------------------------------------- #
# Parser / entry point
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(
        prog="research_cli.py",
        description="Inspect, reconcile, replay, compare and report on the "
                    "Oracle research ledger (read-only except reconcile "
                    "--apply).")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("inspect",
                       help="show one decision or overall ledger stats")
    p.add_argument("decision_id", nargs="?", default=None,
                   help="decision_id to dump; omit for --stats")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--stats", action="store_true",
                   help="print ledger-wide counts instead of one decision")
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("reconcile",
                       help="reconcile outcomes against the broker "
                            "(dry-run unless --apply)")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--apply", action="store_true",
                   help="write events (default: print plan only)")
    p.add_argument("--activities-after", default=None,
                   help="ISO timestamp; also sync the FILL activity stream")
    p.add_argument("--skip-horizon-min", type=float, default=390.0)
    p.add_argument("--no-network", action="store_true",
                   help="expiry + classification only (no broker calls)")
    p.set_defaults(fn=cmd_reconcile)

    p = sub.add_parser("replay",
                       help="deterministic replay under an execution model")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--scenario", default="base",
                   help="base|wide_spread|heavy_slippage|stale_quotes|"
                        "thin_book|all (default: base)")
    p.add_argument("--full", action="store_true",
                   help="include per-decision results, not just the summary")
    p.set_defaults(fn=cmd_replay)

    p = sub.add_parser("compare",
                       help="experiments registry / shadow variant / "
                            "component attribution")
    p.add_argument("what", choices=("experiments", "shadow", "components"))
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--xdb", default=DEFAULT_XDB,
                   help="experiments registry database")
    p.add_argument("--id", default=None, help="experiment id to show in full")
    p.add_argument("--status", default=None,
                   help="filter experiment list by status")
    p.add_argument("--variant", default=None, help="shadow variant name")
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.set_defaults(fn=cmd_compare)

    p = sub.add_parser("report", help="chronological validation report")
    p.add_argument("--db", default=DEFAULT_DB)
    p.add_argument("--since", default=None)
    p.add_argument("--until", default=None)
    p.add_argument("--include-simulated", action="store_true",
                   help="include policy-based simulated outcomes")
    p.add_argument("--embargo-min", type=float, default=390.0)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--n-experiments", type=int, default=1,
                   help="Bonferroni divisor for significance thresholds")
    p.add_argument("--seed", type=int, default=7)
    p.set_defaults(fn=cmd_report)

    return ap


def run(argv=None) -> int:
    args = build_parser().parse_args(argv)
    return args.fn(args)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; temp databases)
# --------------------------------------------------------------------------- #
def _seed_demo_ledger(path: str, n: int = 8) -> None:
    from research_ledger import ResearchLedger
    led = ResearchLedger(path)
    for i in range(n):
        day = 1 + i
        t0 = f"2026-10-{day:02d}T14:30:00+00:00"
        t1 = f"2026-10-{day:02d}T18:30:00+00:00"
        did = f"demo-{i}"
        led.record_snapshot(
            decision_id=did, accepted=True,
            symbol="SPY261218C00500000", underlying="SPY",
            quote={"bid": 1.00, "ask": 1.10, "bid_size": 50, "ask_size": 50,
                   "ts": t0, "age_s": 1.0},
            underlying_price=500.0,
            direction="CALL" if i % 2 == 0 else "PUT",
            score=5.0 + i, p_win=0.55, ev=0.4,
            sizing={"qty": 1}, regime="trend", event_at=t0)
        led.record_event(decision_id=did, event_type="submitted",
                         broker_order_id=f"o{i}", qty=1, price=1.10,
                         event_at=t0)
        led.record_event(decision_id=did, event_type="fill",
                         broker_order_id=f"o{i}", qty=1, price=1.10,
                         status="filled", event_at=t0)
        net = 10.0 if i % 2 == 0 else -5.0
        led.record_event(decision_id=did, event_type="exit",
                         broker_order_id=f"o{i}x", qty=1, price=1.21,
                         event_at=t1)
        led.record_event(decision_id=did, event_type="outcome",
                         qty=1, price=1.21, event_at=t1,
                         payload={"class": "realized", "reason": "take_profit",
                                  "gross_pnl_pct": net + 2.0,
                                  "net_pnl_pct": net,
                                  "net_pnl_dollars": net})
    # One shadow-variant snapshot sharing a baseline decision_id.
    led.record_snapshot(
        decision_id="demo-0", kind="shadow_candidate", accepted=True,
        symbol="SPY261218C00500000", direction="CALL", score=6.0,
        strategy_version="shadow:v2", extra={"shadow_variant": "v2"},
        event_at="2026-10-01T14:30:00+00:00")
    # One shadow-only snapshot with a resolved simulated counterfactual.
    led.record_snapshot(
        decision_id="shadow-cf-0", kind="shadow_candidate", accepted=True,
        symbol="QQQ", underlying="QQQ", underlying_price=400.0,
        direction="CALL", score=7.0,
        strategy_version="shadow:v2", extra={"shadow_variant": "v2"},
        event_at="2026-10-02T14:30:00+00:00")
    led.record_event(
        decision_id="shadow-cf-0", event_type="outcome", price=410.0,
        event_at="2026-10-02T21:00:00+00:00",
        payload={"class": "simulated",
                 "policy": "underlying_counterfactual_v1",
                 "entry_underlying_price": 400.0,
                 "counterfactual_pnl_pct": 2.5,
                 "shadow_variant": "v2"})
    led.close()


def _run_json(argv) -> tuple:
    """Run the CLI capturing stdout; return (exit_code, parsed_json_or_text)."""
    import contextlib
    import io
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        code = run(argv)
    text = buf.getvalue()
    try:
        return code, json.loads(text)
    except ValueError:
        return code, text


def _self_test() -> int:
    import os
    import tempfile

    ok = True
    with tempfile.TemporaryDirectory() as tmp:
        db = os.path.join(tmp, "ledger.db")
        xdb = os.path.join(tmp, "experiments.db")
        _seed_demo_ledger(db)

        # inspect --stats
        code, out = _run_json(["inspect", "--stats", "--db", db])
        if code != 0 or not isinstance(out, dict):
            print("FAIL: inspect --stats", code, out); ok = False

        # inspect one decision
        code, out = _run_json(["inspect", "demo-0", "--db", db])
        if code != 0 or not out.get("snapshots") or not out.get("events"):
            print("FAIL: inspect decision", code); ok = False
        types = {e["event_type"] for e in out.get("events", [])}
        if not {"submitted", "fill", "exit", "outcome"} <= types:
            print("FAIL: inspect missing lifecycle events", types); ok = False

        # inspect unknown decision -> exit 1
        code, _ = _run_json(["inspect", "nope", "--db", db])
        if code != 1:
            print("FAIL: inspect unknown should exit 1, got", code); ok = False

        # reconcile dry-run, no network: must not write anything
        code, _ = _run_json(["reconcile", "--db", db, "--no-network"])
        if code != 0:
            print("FAIL: reconcile dry-run exit", code); ok = False
        from research_ledger import ResearchLedger
        led = ResearchLedger(db)
        n_events = led.stats()["events"]
        led.close()
        if n_events != 8 * 4 + 1:   # +1 seeded shadow counterfactual outcome
            print("FAIL: dry-run reconcile wrote events", n_events); ok = False

        # replay single scenario
        code, out = _run_json(["replay", "--db", db, "--scenario", "base"])
        if code != 0 or out.get("decisions") != 8:
            print("FAIL: replay base", code, out); ok = False
        if out.get("by_status", {}).get("filled") != 8:
            print("FAIL: replay should fill all seeded decisions", out); ok = False

        # replay all scenarios
        code, out = _run_json(["replay", "--db", db, "--scenario", "all"])
        if code != 0 or "wide_spread" not in out:
            print("FAIL: replay all scenarios", code); ok = False

        # replay unknown scenario -> exit 2
        code, _ = _run_json(["replay", "--db", db, "--scenario", "bogus"])
        if code != 2:
            print("FAIL: unknown scenario should exit 2, got", code); ok = False

        # compare experiments (empty registry, then one registered)
        code, out = _run_json(["compare", "experiments", "--xdb", xdb])
        if code != 0 or out != []:
            print("FAIL: empty experiments list", code, out); ok = False
        from research_experiments import ExperimentRegistry
        reg = ExperimentRegistry(xdb)
        eid = reg.register(
            name="demo", hypothesis="h", component="direction",
            baseline={"a": 1}, candidate={"a": 2}, dataset_id="d1",
            rejection_criteria={"min_decisions": 1})
        reg.close()
        code, out = _run_json(["compare", "experiments", "--xdb", xdb,
                               "--id", eid])
        if code != 0 or out["experiment"]["experiment_id"] != eid:
            print("FAIL: compare experiments --id", code); ok = False

        # compare shadow
        code, out = _run_json(["compare", "shadow", "--db", db,
                               "--variant", "v2"])
        if code != 0 or out.get("shadow", {}).get("n") != 2:
            print("FAIL: compare shadow", code, out); ok = False
        if out.get("shared_decision_ids") != 1:
            print("FAIL: shadow overlap", out); ok = False
        cf = out.get("counterfactual_outcomes", {})
        if cf.get("n_entries") != 2 or cf.get("n_resolved") != 1:
            print("FAIL: shadow counterfactual counts", cf); ok = False
        if cf.get("hit_rate") != 1.0 or cf.get("avg_pnl_pct") != 2.5:
            print("FAIL: shadow counterfactual stats", cf); ok = False
        if cf.get("by_direction", {}).get("CALL", {}).get("n") != 1:
            print("FAIL: shadow counterfactual by_direction", cf); ok = False
        code, _ = _run_json(["compare", "shadow", "--db", db])
        if code != 2:
            print("FAIL: shadow without --variant should exit 2"); ok = False

        # compare components
        code, out = _run_json(["compare", "components", "--db", db])
        if code != 0 or "direction" not in out:
            print("FAIL: compare components", code); ok = False

        # report over seeded outcomes
        code, out = _run_json(["report", "--db", db, "--folds", "3",
                               "--embargo-min", "0"])
        if code != 0 or "walk_forward" not in out:
            print("FAIL: report", code); ok = False
        if "test_metrics" in out:
            print("FAIL: report must not evaluate the final test slice"); ok = False

        # report on an empty ledger
        empty = os.path.join(tmp, "empty.db")
        ResearchLedger(empty).close()
        code, out = _run_json(["report", "--db", empty])
        if code != 0 or out.get("observations") != 0:
            print("FAIL: empty report", code, out); ok = False

    print("research_cli self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1] == "--selftest":
        sys.exit(_self_test())
    sys.exit(run())
