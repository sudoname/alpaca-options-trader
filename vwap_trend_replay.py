"""VWAP/EMA/RSI trend-filter replay over Oracle's recorded decisions.

Research question: would gating Oracle's recorded directional option entries
on underlying trend/momentum (session VWAP, EMA(20), RSI(14) on 5-minute
candles) have improved NET options P&L per decision opportunity?

Three variants on the SAME opportunity universe:
  baseline   recorded production selection (every submitted decision).
             The research ledger holds no candidate groups (group capture is
             not yet deployed), so re-ranking alternatives is impossible;
             "strongest available existing ranking" degenerates to the
             recorded decisions and conclusions are limited to the recorded
             selected/executed population.
  vwap       baseline, but only opportunities whose direction aligns with
             the VWAP-only state (CALL->bullish, PUT->bearish).
  full       baseline, but aligned on VWAP + EMA(20) + RSI(14) vs 50.
  vwap_contra / full_contra
             the INVERSE gates: take exactly what the aligned filter skips
             (misaligned or neutral); signal-unavailable stays excluded, so
             on common coverage taken(filter) and taken(contra) partition
             the baseline and their nets sum to it. POST-HOC HYPOTHESIS:
             added after the aligned filters were observed to select
             worse-per-trade entries on these same sessions, so the
             dev/holdout split no longer protects against overfitting here.
             Results can motivate forward shadow validation only.

Accounting (per the research protocol):
  * taken opportunity  -> its actual recorded mark net P&L (dollars). Entry
    prices are the real fills, so no counterfactual option repricing or fill
    modeling is needed for removals; there is no unfilled-order leg to
    stress, which is disclosed in the report.
  * filter skip        -> $0 trading P&L for that opportunity.
  * missing signal     -> excluded from the PRIMARY common-coverage
    comparison; in the full-sample view the filter variant skips it ($0) and
    the skip is reported separately from deliberate filter skips.
  * unknown outcome    -> excluded from P&L metrics and disclosed (these are
    still-open positions with no mark).

Splits (recorded before the holdout was examined): development sessions
2026-11-04..2026-11-10, holdout 2026-11-11..2026-11-13. Uncertainty via
paired session-level bootstrap of the per-opportunity delta.

Read-only: consumes export_decisions.jsonl + research_vwap/bars_5min.jsonl,
writes only under research_vwap/. Imports no broker/trading client; zero
order submissions.

Usage:
    python vwap_trend_replay.py                      # real-data replay
    python vwap_trend_replay.py --selftest           # hermetic
    python vwap_trend_replay.py --warmup continuation  # sensitivity run
"""
import argparse
import hashlib
import json
import os
import random
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timezone

import vwap_trend_signal as vts

SESSION_START = "14:30"   # UTC, regular session (EST window, Nov 2026)
SESSION_END = "21:00"
DEV_LAST_SESSION = "2026-11-10"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 1337
VARIANTS = ("baseline", "vwap", "full", "vwap_contra", "full_contra")


# ------------------------------------------------------------------ inputs

def load_bars(path):
    """(symbol, session_date) -> time-sorted regular-session bars."""
    by_key = defaultdict(list)
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            b = json.loads(line)
            hhmm = b["ts"][11:16]
            if not (SESSION_START <= hhmm < SESSION_END):
                continue
            by_key[(b["symbol"], b["ts"][:10])].append(b)
    for rows in by_key.values():
        rows.sort(key=lambda r: r["ts"])
    return by_key


def prior_session_closes(bars_by_key, symbol, session_date):
    """Closes of the latest earlier session for this symbol, or None."""
    prior = [d for (s, d) in bars_by_key
             if s == symbol and d < session_date]
    if not prior:
        return None
    return [b["c"] for b in bars_by_key[(symbol, max(prior))]]


def load_opportunities(path):
    """Directional decision opportunities from the research export.

    One row per decision that submitted an order: decision ts = the first
    ``submitted`` event (point-in-time safe: at or after the actual decision,
    so the signal never uses information unavailable to Oracle).
    """
    opps = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
            except json.JSONDecodeError:
                continue
            subs = [e for e in r.get("events") or []
                    if e.get("event_type") == "submitted"
                    and e.get("event_at_utc")]
            if not subs:
                continue
            snaps = r.get("snapshots") or []
            snap = snaps[0] if snaps else {}
            direction = (snap.get("direction") or "").upper()
            if direction not in ("CALL", "PUT"):
                continue  # non-directional exposure: out of scope
            ts = min(e["event_at_utc"] for e in subs)
            outcome = r.get("outcome") or {}
            resolved = outcome.get("state") == "filled_resolved"
            net = outcome.get("net_pnl_dollars") if resolved else None
            if resolved and net is None:
                resolved = False
            opps.append({
                "decision_id": r.get("decision_id"),
                "ts": ts,
                "session": ts[:10],
                "underlying": (snap.get("underlying") or "").upper(),
                "direction": direction,
                "net": net,                   # None == unknown outcome
                "outcome_known": resolved,
            })
    opps.sort(key=lambda o: o["ts"])
    return opps


# ----------------------------------------------------------------- signals

def attach_signals(opps, bars_by_key, warmup="session"):
    """Annotate each opportunity with alignment under both filter variants.

    align_vwap / align_full: True, False, or None (signal unavailable).
    """
    for o in opps:
        bars = bars_by_key.get((o["underlying"], o["session"]), [])
        prior = (prior_session_closes(bars_by_key, o["underlying"],
                                      o["session"])
                 if warmup == "continuation" else None)
        sig = vts.signal_at(bars, o["ts"], prior_closes=prior)
        o["signal"] = sig
        o["align_vwap"] = vts.aligned(sig["state_vwap_only"], o["direction"])
        o["align_full"] = vts.aligned(sig["state_full"], o["direction"])
    return opps


def takes(o, variant):
    """True/False = variant takes/skips; None = signal missing for filter.

    Contra variants invert the aligned gate on DEFINED signals only: they
    take exactly the opportunities the aligned filter skips (misaligned or
    neutral) and remain None when the signal is unavailable, preserving the
    partition of the baseline on common coverage.
    """
    if variant == "baseline":
        return True
    key = "align_vwap" if variant.startswith("vwap") else "align_full"
    a = o[key]
    if a is None:
        return None
    return (not a) if variant.endswith("_contra") else a


# ----------------------------------------------------------------- metrics

def variant_metrics(opps, variant):
    """Metrics over opportunities with KNOWN outcomes.

    Missing-signal opportunities are skipped by the filter variant ($0) and
    counted in ``missing_signal_skips`` (full-sample behavior).
    """
    known = [o for o in opps if o["outcome_known"]]
    unknown = len(opps) - len(known)
    taken, fskip, mskip = [], [], 0
    for o in known:
        t = takes(o, variant)
        if t is True:
            taken.append(o)
        elif t is False:
            fskip.append(o)
        else:
            mskip += 1
    nets = [o["net"] for o in taken]
    wins = sum(1 for x in nets if x > 0)
    n_opp = len(known)
    total = sum(nets)
    return {
        "opportunities": n_opp,
        "unknown_outcomes": unknown,
        "taken": len(taken),
        "filter_skips": len(fskip),
        "missing_signal_skips": mskip,
        "net_total": round(total, 2),
        "net_per_opportunity": round(total / n_opp, 4) if n_opp else None,
        "net_per_taken": round(total / len(taken), 4) if taken else None,
        "win_rate_taken": round(wins / len(taken), 4) if taken else None,
        "winners_skipped": sum(1 for o in fskip if o["net"] > 0),
        "losers_skipped": sum(1 for o in fskip if o["net"] < 0),
    }


def common_coverage(opps):
    """Primary population: known outcome AND both filter signals defined."""
    return [o for o in opps
            if o["outcome_known"]
            and o["align_vwap"] is not None
            and o["align_full"] is not None]


def session_totals(opps, variant):
    """session -> (net_total_taken, n_opportunities)."""
    out = defaultdict(lambda: [0.0, 0])
    for o in opps:
        s = out[o["session"]]
        s[1] += 1
        if takes(o, variant) is True:
            s[0] += o["net"]
    return {k: tuple(v) for k, v in out.items()}


def paired_delta(opps, variant_a, variant_b):
    """Pooled per-opportunity delta (A - B) and the per-session pairs."""
    ta = session_totals(opps, variant_a)
    tb = session_totals(opps, variant_b)
    sessions = sorted(ta)
    pairs = [(s, ta[s][0], tb[s][0], ta[s][1]) for s in sessions]
    n = sum(p[3] for p in pairs)
    delta = (sum(p[1] - p[2] for p in pairs) / n) if n else None
    return {"delta_per_opportunity": round(delta, 4) if delta is not None
            else None,
            "sessions": sessions, "pairs": pairs}


def bootstrap_ci(pairs, n_boot=BOOTSTRAP_N, seed=BOOTSTRAP_SEED):
    """Paired session-resampling CI of the per-opportunity delta."""
    if len(pairs) < 2:
        return None  # underpowered: no resampling distribution
    rng = random.Random(seed)
    stats = []
    k = len(pairs)
    for _ in range(n_boot):
        sample = [pairs[rng.randrange(k)] for _ in range(k)]
        n = sum(p[3] for p in sample)
        if n == 0:
            continue
        stats.append(sum(p[1] - p[2] for p in sample) / n)
    stats.sort()
    lo = stats[int(0.025 * len(stats))]
    hi = stats[int(0.975 * len(stats)) - 1]
    frac_pos = sum(1 for x in stats if x > 0) / len(stats)
    return {"ci95": [round(lo, 4), round(hi, 4)],
            "frac_positive": round(frac_pos, 4), "n_boot": len(stats)}


def by_direction(opps):
    out = {}
    for d in ("CALL", "PUT"):
        sub = [o for o in opps if o["direction"] == d]
        out[d] = {v: variant_metrics(sub, v) for v in VARIANTS}
    return out


# -------------------------------------------------------------- provenance

def _sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def provenance(args, counts):
    def _git(*cmd):
        try:
            return subprocess.run(["git"] + list(cmd), capture_output=True,
                                  text=True, timeout=10).stdout.strip()
        except Exception:
            return None
    return {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "git_rev": _git("rev-parse", "HEAD"),
        "git_dirty": bool(_git("status", "--porcelain")),
        "inputs": {p: _sha256(p) for p in (args.decisions, args.bars)
                   if os.path.exists(p)},
        "params": {"bar_minutes": vts.BAR_MINUTES_DEFAULT,
                   "ema_period": vts.EMA_PERIOD_DEFAULT,
                   "rsi_period": vts.RSI_PERIOD_DEFAULT,
                   "rsi_midpoint": vts.RSI_MIDPOINT,
                   "max_staleness_min": vts.MAX_STALENESS_MIN_DEFAULT,
                   "warmup": args.warmup,
                   "session_utc": [SESSION_START, SESSION_END],
                   "dev_last_session": DEV_LAST_SESSION,
                   "bootstrap": {"n": BOOTSTRAP_N, "seed": BOOTSTRAP_SEED}},
        "command": "python " + " ".join(sys.argv),
        "counts": counts,
        "pnl_basis": "recorded paper mark nets (research_export history join)",
        "execution_note": ("filters only REMOVE recorded fills; no new "
                           "orders are modeled, so no fill-probability or "
                           "slippage stress applies to the removals"),
    }


# ------------------------------------------------------------------ report

def evaluate(opps):
    """Full evaluation structure over an annotated opportunity list."""
    dev = [o for o in opps if o["session"] <= DEV_LAST_SESSION]
    hold = [o for o in opps if o["session"] > DEV_LAST_SESSION]
    out = {}
    for name, pop in (("dev", dev), ("holdout", hold), ("all", opps)):
        cc = common_coverage(pop)
        block = {
            "sessions": sorted({o["session"] for o in pop}),
            "primary_common_coverage": {
                "n": len(cc),
                "variants": {v: variant_metrics(cc, v) for v in VARIANTS},
                "deltas": {},
            },
            "full_sample": {
                "n_known": sum(1 for o in pop if o["outcome_known"]),
                "n_unknown": sum(1 for o in pop if not o["outcome_known"]),
                "variants": {v: variant_metrics(pop, v) for v in VARIANTS},
            },
        }
        for a, b in (("vwap", "baseline"), ("full", "baseline"),
                     ("full", "vwap"),
                     ("vwap_contra", "baseline"),
                     ("full_contra", "baseline")):
            pd = paired_delta(cc, a, b)
            pd["bootstrap"] = bootstrap_ci(pd.pop("pairs"))
            block["primary_common_coverage"]["deltas"][f"{a}_minus_{b}"] = pd
        block["by_direction_common_coverage"] = by_direction(cc)
        out[name] = block
    return out


def _fmt_variant_table(variants):
    hdr = (f"{'variant':<10}{'opps':>6}{'taken':>7}{'fskip':>7}{'mskip':>7}"
           f"{'net_total':>12}{'net/opp':>10}{'net/taken':>11}{'win%':>7}"
           f"{'Wskip':>7}{'Lskip':>7}")
    lines = [hdr, "-" * len(hdr)]
    for v in VARIANTS:
        m = variants[v]
        wr = f"{100*m['win_rate_taken']:.1f}" if m["win_rate_taken"] \
            is not None else "-"
        lines.append(
            f"{v:<10}{m['opportunities']:>6}{m['taken']:>7}"
            f"{m['filter_skips']:>7}{m['missing_signal_skips']:>7}"
            f"{m['net_total']:>12.2f}"
            f"{(m['net_per_opportunity'] if m['net_per_opportunity'] is not None else 0):>10.2f}"
            f"{(m['net_per_taken'] if m['net_per_taken'] is not None else 0):>11.2f}"
            f"{wr:>7}{m['winners_skipped']:>7}{m['losers_skipped']:>7}")
    return "\n".join(lines)


def print_report(report):
    for split in ("dev", "holdout", "all"):
        blk = report[split]
        cc = blk["primary_common_coverage"]
        print(f"\n=== {split.upper()} | sessions {blk['sessions']} ===")
        print(f"primary (common coverage, n={cc['n']}):")
        print(_fmt_variant_table(cc["variants"]))
        for k, d in cc["deltas"].items():
            bs = d.get("bootstrap")
            ci = (f" ci95={bs['ci95']} frac_pos={bs['frac_positive']}"
                  if bs else " [underpowered: <2 sessions, no CI]")
            print(f"  delta {k}: {d['delta_per_opportunity']}/opp"
                  f" over {len(d['sessions'])} sessions{ci}")
        fs = blk["full_sample"]
        print(f"full sample: known={fs['n_known']} "
              f"unknown(excluded)={fs['n_unknown']}")
        print(_fmt_variant_table(fs["variants"]))


# --------------------------------------------------------------------- CLI

def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--decisions", default="export_decisions.jsonl")
    ap.add_argument("--bars", default="research_vwap/bars_5min.jsonl")
    ap.add_argument("--out", default="research_vwap/replay_report.json")
    ap.add_argument("--warmup", choices=("session", "continuation"),
                    default="session")
    ap.add_argument("--selftest", action="store_true")
    args = ap.parse_args(argv)
    if args.selftest:
        return _selftest()

    bars = load_bars(args.bars)
    opps = load_opportunities(args.decisions)
    attach_signals(opps, bars, warmup=args.warmup)
    report = evaluate(opps)
    counts = {"opportunities": len(opps),
              "known_outcomes": sum(1 for o in opps if o["outcome_known"]),
              "unknown_outcomes": sum(1 for o in opps
                                      if not o["outcome_known"]),
              "signal_unavailable_vwap": sum(1 for o in opps
                                             if o["align_vwap"] is None),
              "signal_unavailable_full": sum(1 for o in opps
                                             if o["align_full"] is None),
              "bar_keys": len(bars)}
    payload = {"provenance": provenance(args, counts), "report": report}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    out_path = args.out
    if args.warmup != "session":
        root, ext = os.path.splitext(out_path)
        out_path = f"{root}_{args.warmup}{ext}"
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, indent=1)
    print(f"warmup={args.warmup} counts={counts}")
    print_report(report)
    print(f"\nreport -> {out_path}")
    return 0


# ----------------------------------------------------------------- selftest

def _mk_bars(symbol, date, closes, start_hhmm="14:30"):
    h, m = int(start_hhmm[:2]), int(start_hhmm[3:])
    t0 = datetime.fromisoformat(f"{date}T00:00:00+00:00").replace(
        hour=h, minute=m)
    rows = []
    for i, c in enumerate(closes):
        ts = t0.timestamp() + i * 300
        iso = datetime.fromtimestamp(ts, tz=timezone.utc).isoformat()
        rows.append({"symbol": symbol, "ts": iso, "o": c, "h": c + 0.5,
                     "l": c - 0.5, "c": c, "v": 100, "vwap": c})
    return rows


def _mk_decision(did, ts, underlying, direction, net, resolved=True):
    outcome = ({"state": "filled_resolved", "net_pnl_dollars": net}
               if resolved else {"state": "unresolved",
                                 "net_pnl_dollars": None})
    return {"decision_id": did,
            "events": [{"event_type": "submitted", "event_at_utc": ts}],
            "snapshots": [{"underlying": underlying, "direction": direction}],
            "outcome": outcome}


def _selftest():
    import tempfile
    ok = True

    def check(name, cond):
        nonlocal ok
        print(("PASS" if cond else "FAIL") + ": " + name)
        ok = ok and cond

    with tempfile.TemporaryDirectory() as td:
        bars_path = os.path.join(td, "bars.jsonl")
        dec_path = os.path.join(td, "dec.jsonl")
        # UP: 25 rising bars (warm for EMA20 by bar 20). DOWN: falling.
        up = _mk_bars("UP", "2026-11-04", [100 + i for i in range(25)])
        up2 = _mk_bars("UP", "2026-11-05", [100 + i for i in range(25)])
        down = _mk_bars("DOWN", "2026-11-04", [100 - i for i in range(25)])
        with open(bars_path, "w", encoding="utf-8") as fh:
            for b in up + up2 + down:
                fh.write(json.dumps(b) + "\n")
            # out-of-session bar must be ignored
            fh.write(json.dumps({"symbol": "UP",
                                 "ts": "2026-11-04T13:00:00+00:00",
                                 "o": 1, "h": 1, "l": 1, "c": 1, "v": 1,
                                 "vwap": 1}) + "\n")

        late = "2026-11-04T16:45:00+00:00"   # after EMA20 warm-up
        early = "2026-11-04T14:40:00+00:00"  # vwap OK, full warming up
        d2 = "2026-11-05T16:45:00+00:00"
        decisions = [
            _mk_decision("a", late, "UP", "CALL", 100.0),    # aligned both
            _mk_decision("b", late, "UP", "PUT", -50.0),     # misaligned
            _mk_decision("c", late, "DOWN", "PUT", 40.0),    # aligned both
            _mk_decision("d", early, "UP", "CALL", -30.0),   # full missing
            _mk_decision("e", late, "NOBARS", "CALL", 10.0),  # all missing
            _mk_decision("f", late, "UP", "CALL", 0.0, resolved=False),
            _mk_decision("g", d2, "UP", "CALL", -20.0),      # 2nd session
        ]
        with open(dec_path, "w", encoding="utf-8") as fh:
            for d in decisions:
                fh.write(json.dumps(d) + "\n")
            fh.write(json.dumps(  # non-directional: out of scope
                _mk_decision("h", late, "UP", "IRON_CONDOR", 5.0)) + "\n")
            fh.write(json.dumps(  # never submitted: not an opportunity
                {"events": [], "snapshots": [{"underlying": "UP",
                                              "direction": "CALL"}],
                 "outcome": {"state": "policy_skip"}}) + "\n")

        bars = load_bars(bars_path)
        check("out-of-session bar excluded",
              all(b["ts"][11:16] >= "14:30" for b in
                  bars[("UP", "2026-11-04")]))
        opps = load_opportunities(dec_path)
        check("universe: submitted+directional only",
              sorted(o["decision_id"] for o in opps) ==
              ["a", "b", "c", "d", "e", "f", "g"]
              and opps[0]["decision_id"] == "d")  # time-sorted
        attach_signals(opps, bars)
        o = {x["decision_id"]: x for x in opps}
        check("same opportunity set for all variants",
              all(takes(o["a"], v) is not None or v != "baseline"
                  for v in VARIANTS) and takes(o["a"], "baseline") is True
              and takes(o["b"], "baseline") is True)
        check("aligned CALL taken by both filters",
              o["a"]["align_vwap"] is True and o["a"]["align_full"] is True)
        check("misaligned PUT skipped by both filters",
              o["b"]["align_vwap"] is False and o["b"]["align_full"] is False)
        check("aligned PUT on falling tape taken",
              o["c"]["align_vwap"] is True and o["c"]["align_full"] is True)
        check("early decision: vwap defined, full missing",
              o["d"]["align_vwap"] is True and o["d"]["align_full"] is None)
        check("no bars -> both signals missing",
              o["e"]["align_vwap"] is None and o["e"]["align_full"] is None)

        # Common coverage drops d (full missing), e (both missing),
        # f (unknown outcome) -> {a, b, c, g}
        cc = common_coverage(opps)
        check("common coverage = known + both signals",
              sorted(x["decision_id"] for x in cc) == ["a", "b", "c", "g"])

        m_base = variant_metrics(cc, "baseline")
        m_vwap = variant_metrics(cc, "vwap")
        check("baseline takes everything",
              m_base["taken"] == 4 and m_base["net_total"] == 70.0)
        # vwap skips b (net -50 -> loser skipped, counted as $0)
        check("filter skip counts as $0",
              m_vwap["taken"] == 3 and m_vwap["net_total"] == 120.0
              and m_vwap["filter_skips"] == 1
              and m_vwap["losers_skipped"] == 1
              and m_vwap["winners_skipped"] == 0)
        check("net per opportunity uses full denominator",
              m_vwap["net_per_opportunity"] == round(120.0 / 4, 4))

        # Contra variants: exact inversion of the aligned gate on defined
        # signals; signal-unavailable stays None (never flipped to taken).
        check("contra inverts aligned gate",
              takes(o["a"], "vwap_contra") is False
              and takes(o["b"], "vwap_contra") is True
              and takes(o["c"], "full_contra") is False)
        check("contra preserves missing-signal None",
              takes(o["d"], "full_contra") is None
              and takes(o["e"], "vwap_contra") is None)
        # Complementarity on common coverage: filter + contra partition the
        # baseline, so their nets sum to the baseline net.
        m_vc = variant_metrics(cc, "vwap_contra")
        m_fc = variant_metrics(cc, "full_contra")
        m_full = variant_metrics(cc, "full")
        check("contra complements filter (vwap partition)",
              m_vc["taken"] + m_vwap["taken"] == m_base["taken"]
              and round(m_vc["net_total"] + m_vwap["net_total"], 2)
              == m_base["net_total"])
        check("contra complements filter (full partition)",
              m_fc["taken"] + m_full["taken"] == m_base["taken"]
              and round(m_fc["net_total"] + m_full["net_total"], 2)
              == m_base["net_total"])

        # Full-sample view: d rejoins vwap variant (aligned) but is a
        # missing-signal skip for full; e missing for both filters;
        # f stays excluded as unknown.
        fs_full = variant_metrics(opps, "full")
        check("unknown outcome excluded + disclosed",
              fs_full["unknown_outcomes"] == 1
              and fs_full["opportunities"] == 6)
        check("missing-signal skips reported separately",
              fs_full["missing_signal_skips"] == 2)

        pd = paired_delta(cc, "vwap", "baseline")
        check("paired delta per opportunity",
              pd["delta_per_opportunity"] == round(50.0 / 4, 4))
        bs = bootstrap_ci(pd["pairs"])
        check("bootstrap runs with 2 sessions",
              bs is not None and bs["n_boot"] > 9000
              and bs["ci95"][0] <= bs["ci95"][1])
        check("bootstrap underpowered with 1 session",
              bootstrap_ci(pd["pairs"][:1]) is None)
        check("bootstrap deterministic",
              bootstrap_ci(pd["pairs"]) == bootstrap_ci(pd["pairs"]))

        report = evaluate(opps)
        check("dev/holdout split by session",
              report["dev"]["sessions"] == ["2026-11-04", "2026-11-05"]
              and report["holdout"]["sessions"] == [])
        check("report serializes", bool(json.dumps(report)))

        # zero-broker guarantee: no trading/broker modules imported
        banned = [m for m in sys.modules
                  if "alpaca.trading" in m or m == "smart_trader"]
        check("no trading modules imported", banned == [])

    print("SELFTEST " + ("PASS" if ok else "FAIL"))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
