"""Incremental-impact measurement for the bounded skeptic (Phase 5e).

The bull/bear/no-trade reviewer (``oracle/thesis_debate.adversarial_review``)
stamps ``option['thesis_review']`` on every scanned candidate BEFORE the
deterministic gates run (EV floor, profit gate, spread/freshness checks).
Phase 5b wires that stamp into the research ledger ``extra_json`` on both
accepted candidates and gate rejections, so logged decisions let us ask:

    Does skeptic doubt carry information BEYOND the deterministic checks?

Method
------
* Split logged candidates by ``deterministic_pass`` (ledger ``accepted``).
* Among deterministic-pass candidates, compare the realized net outcomes of
  the "quiet" group (no flags, zero no_trade_boost) against the "doubt"
  group (any flag or boost > 0). If the doubt group underperforms, the
  skeptic adds value beyond the deterministic gates; if not, it is redundant.
* Among deterministic rejections, count how often the skeptic also doubted
  (redundant doubt) versus stayed quiet.
* Always report sample sizes; verdict is ``insufficient_evidence`` until both
  groups have a minimum number of RESOLVED outcomes. Unresolved outcomes
  (``net_outcome is None``) are counted but never imputed.

Risk enforcement remains deterministic and independent of the reviewer:
this module only *measures*; it changes no decision path.

No credentials, no network. Ledger access is read-only and fail-open.

CLI:
    python skeptic_impact.py --selftest
    python skeptic_impact.py --fixture
    python skeptic_impact.py --ledger research_ledger.db
"""
from __future__ import annotations

import json
import math
import os
import sqlite3
import sys
from typing import Dict, List, Optional

MIN_RESOLVED_PER_GROUP = 5   # below this per group -> insufficient_evidence

# Payload keys tried (in order) for a dollar net outcome on 'outcome' events.
_OUTCOME_KEYS = ("net_pnl", "net_pnl_usd", "realized_pnl", "pnl")


# --------------------------------------------------------------------------- #
# Row model
# --------------------------------------------------------------------------- #
def make_row(decision_id: str, *, deterministic_pass: bool,
             skeptic_flags: Optional[List[str]] = None,
             no_trade_boost: float = 0.0,
             net_outcome: Optional[float] = None) -> Dict:
    """Normalized evaluation row (also the fixture/JSONL schema)."""
    return {
        "decision_id": str(decision_id),
        "deterministic_pass": bool(deterministic_pass),
        "skeptic_flags": list(skeptic_flags or []),
        "no_trade_boost": float(no_trade_boost or 0.0),
        "net_outcome": (float(net_outcome)
                        if isinstance(net_outcome, (int, float))
                        and not isinstance(net_outcome, bool)
                        and math.isfinite(float(net_outcome)) else None),
    }


def _has_doubt(row: Dict) -> bool:
    return bool(row.get("skeptic_flags")) or float(row.get("no_trade_boost") or 0.0) > 0.0


# --------------------------------------------------------------------------- #
# Core evaluation
# --------------------------------------------------------------------------- #
def _group_stats(rows: List[Dict]) -> Dict:
    resolved = [r["net_outcome"] for r in rows if r["net_outcome"] is not None]
    n_res = len(resolved)
    total = sum(resolved) if resolved else 0.0
    wins = sum(1 for v in resolved if v > 0)
    return {
        "n": len(rows),
        "n_resolved": n_res,
        "n_unresolved": len(rows) - n_res,
        "total_net": round(total, 6),
        "avg_net": round(total / n_res, 6) if n_res else None,
        "win_rate": round(wins / n_res, 6) if n_res else None,
    }


def evaluate_skeptic_impact(rows: List[Dict],
                            min_resolved: int = MIN_RESOLVED_PER_GROUP) -> Dict:
    """Measure the skeptic's incremental impact on logged decisions.

    ``rows`` are ``make_row``-shaped dicts. Returns a report dict; never
    raises on odd inputs (rows missing keys are normalized defensively).
    """
    norm = []
    for r in rows or []:
        try:
            norm.append(make_row(
                r.get("decision_id", ""),
                deterministic_pass=r.get("deterministic_pass", False),
                skeptic_flags=r.get("skeptic_flags"),
                no_trade_boost=r.get("no_trade_boost", 0.0),
                net_outcome=r.get("net_outcome"),
            ))
        except Exception:
            continue

    passed = [r for r in norm if r["deterministic_pass"]]
    rejected = [r for r in norm if not r["deterministic_pass"]]

    quiet = [r for r in passed if not _has_doubt(r)]
    doubt = [r for r in passed if _has_doubt(r)]

    quiet_s = _group_stats(quiet)
    doubt_s = _group_stats(doubt)

    # Counterfactual: had the skeptic's doubt been a hard veto, the doubt
    # group's realized P&L would have been avoided. Positive avoided_pnl
    # means the veto would have HELPED (it would have skipped net losses).
    avoided_pnl = round(-doubt_s["total_net"], 6)

    flag_counts: Dict[str, int] = {}
    for r in norm:
        for f in r["skeptic_flags"]:
            flag_counts[f] = flag_counts.get(f, 0) + 1

    redundant_doubt = sum(1 for r in rejected if _has_doubt(r))

    enough = (quiet_s["n_resolved"] >= min_resolved
              and doubt_s["n_resolved"] >= min_resolved)
    if not enough:
        verdict = "insufficient_evidence"
        incremental = None
    else:
        # Incremental value: doubt group underperforms quiet group.
        incremental = round(quiet_s["avg_net"] - doubt_s["avg_net"], 6)
        verdict = ("skeptic_adds_value" if incremental > 0
                   else "skeptic_redundant")

    return {
        "verdict": verdict,
        "n_rows": len(norm),
        "n_deterministic_pass": len(passed),
        "n_deterministic_reject": len(rejected),
        "redundant_doubt_on_rejects": redundant_doubt,
        "quiet": quiet_s,
        "doubt": doubt_s,
        "avg_net_quiet_minus_doubt": incremental,
        "doubt_veto_avoided_pnl": avoided_pnl,
        "flag_counts": flag_counts,
        "min_resolved_per_group": min_resolved,
    }


# --------------------------------------------------------------------------- #
# Ledger adapter (read-only, fail-open)
# --------------------------------------------------------------------------- #
def _extract_net_outcome(payload: Dict, qty: Optional[float],
                         price: Optional[float]) -> Optional[float]:
    for k in _OUTCOME_KEYS:
        v = payload.get(k)
        if isinstance(v, (int, float)) and not isinstance(v, bool) \
                and math.isfinite(float(v)):
            return float(v)
    return None


def rows_from_ledger(db_path: str) -> List[Dict]:
    """Best-effort read of skeptic-labeled snapshots joined to outcomes.

    Only snapshots whose ``extra_json`` carries a ``thesis_review`` dict are
    usable (the skeptic was run and logged). Outcomes join by decision_id on
    ``events.event_type='outcome'``; decisions without one stay unresolved
    (``net_outcome=None``). Missing db / tables / bad JSON -> [] (fail-open).
    """
    if not db_path or not os.path.exists(db_path):
        return []
    rows: List[Dict] = []
    try:
        con = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
        con.row_factory = sqlite3.Row
        try:
            outcomes: Dict[str, Optional[float]] = {}
            for ev in con.execute(
                    "SELECT decision_id, qty, price, payload_json FROM events "
                    "WHERE event_type='outcome' ORDER BY received_at_utc"):
                try:
                    payload = json.loads(ev["payload_json"] or "{}")
                except Exception:
                    payload = {}
                net = _extract_net_outcome(payload, ev["qty"], ev["price"])
                if net is not None:
                    outcomes[ev["decision_id"]] = net
            for sn in con.execute(
                    "SELECT decision_id, accepted, extra_json FROM snapshots "
                    "ORDER BY received_at_utc"):
                try:
                    extra = json.loads(sn["extra_json"] or "{}")
                except Exception:
                    continue
                review = extra.get("thesis_review")
                if not isinstance(review, dict):
                    continue
                rows.append(make_row(
                    sn["decision_id"],
                    deterministic_pass=bool(sn["accepted"]),
                    skeptic_flags=review.get("flags"),
                    no_trade_boost=review.get("no_trade_boost", 0.0),
                    net_outcome=outcomes.get(sn["decision_id"]),
                ))
        finally:
            con.close()
    except Exception:
        return []
    return rows


# --------------------------------------------------------------------------- #
# Fixtures + report
# --------------------------------------------------------------------------- #
def fixture_rows() -> List[Dict]:
    """Deterministic fixture: skeptic doubt is (by construction) predictive."""
    rows = []
    # Quiet winners/losers (deterministic pass, no doubt): avg +12
    for i, pnl in enumerate((40.0, -20.0, 30.0, 10.0, 0.0, 12.0)):
        rows.append(make_row(f"Q{i}", deterministic_pass=True,
                             net_outcome=pnl))
    # Doubted trades (deterministic pass, flags/boost): avg -16
    doubted = ((-60.0, ["stale_catalyst"], 0.05),
               (20.0, ["one_sided_evidence"], 0.03),
               (-40.0, [], 0.08),
               (-10.0, ["stale_catalyst", "pile_on"], 0.10),
               (10.0, ["pile_on"], 0.02))
    for i, (pnl, flags, boost) in enumerate(doubted):
        rows.append(make_row(f"D{i}", deterministic_pass=True,
                             skeptic_flags=flags, no_trade_boost=boost,
                             net_outcome=pnl))
    # Deterministic rejections: 2 with redundant doubt, 1 quiet, all unresolved
    rows.append(make_row("R0", deterministic_pass=False,
                         skeptic_flags=["stale_catalyst"], no_trade_boost=0.05))
    rows.append(make_row("R1", deterministic_pass=False, no_trade_boost=0.02))
    rows.append(make_row("R2", deterministic_pass=False))
    # One unresolved pass row (fill never reconciled)
    rows.append(make_row("U0", deterministic_pass=True,
                         skeptic_flags=["pile_on"], no_trade_boost=0.02))
    return rows


def format_report(report: Dict) -> str:
    L = [f"skeptic_impact: verdict={report['verdict']}  "
         f"rows={report['n_rows']} (pass={report['n_deterministic_pass']}, "
         f"reject={report['n_deterministic_reject']}, "
         f"redundant_doubt_on_rejects={report['redundant_doubt_on_rejects']})"]
    for name in ("quiet", "doubt"):
        g = report[name]
        avg = "n/a" if g["avg_net"] is None else f"{g['avg_net']:+.2f}"
        win = "n/a" if g["win_rate"] is None else f"{g['win_rate']:.0%}"
        L.append(f"  {name:<6} n={g['n']:<3} resolved={g['n_resolved']:<3} "
                 f"total={g['total_net']:+.2f} avg={avg} win={win}")
    inc = report["avg_net_quiet_minus_doubt"]
    L.append(f"  incremental (quiet avg - doubt avg): "
             f"{'n/a' if inc is None else f'{inc:+.2f}'}")
    L.append(f"  doubt-as-veto avoided P&L: "
             f"{report['doubt_veto_avoided_pnl']:+.2f}")
    if report["flag_counts"]:
        flags = ", ".join(f"{k}={v}" for k, v in
                          sorted(report["flag_counts"].items()))
        L.append(f"  flags: {flags}")
    return "\n".join(L)


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def check(cond, msg):
        nonlocal ok
        if not cond:
            print("FAIL:", msg)
            ok = False

    # --- fixture evaluation -------------------------------------------------
    rep = evaluate_skeptic_impact(fixture_rows())
    check(rep["n_rows"] == 15, f"n_rows {rep['n_rows']}")
    check(rep["n_deterministic_pass"] == 12, "pass count")
    check(rep["n_deterministic_reject"] == 3, "reject count")
    check(rep["redundant_doubt_on_rejects"] == 2, "redundant doubt")
    check(rep["quiet"]["n_resolved"] == 6, "quiet resolved")
    check(rep["doubt"]["n_resolved"] == 5, "doubt resolved")
    check(rep["doubt"]["n_unresolved"] == 1, "doubt unresolved (U0)")
    check(abs(rep["quiet"]["avg_net"] - 12.0) < 1e-9, "quiet avg")
    check(abs(rep["doubt"]["avg_net"] - (-16.0)) < 1e-9, "doubt avg")
    check(rep["verdict"] == "skeptic_adds_value", f"verdict {rep['verdict']}")
    check(abs(rep["avg_net_quiet_minus_doubt"] - 28.0) < 1e-9, "incremental")
    check(abs(rep["doubt_veto_avoided_pnl"] - 80.0) < 1e-9, "avoided pnl")
    check(rep["flag_counts"].get("stale_catalyst") == 3, "flag counts")

    # --- insufficient evidence ---------------------------------------------
    small = evaluate_skeptic_impact(fixture_rows()[:4])
    check(small["verdict"] == "insufficient_evidence", "small sample verdict")
    check(small["avg_net_quiet_minus_doubt"] is None, "small incremental None")
    empty = evaluate_skeptic_impact([])
    check(empty["verdict"] == "insufficient_evidence", "empty verdict")
    check(empty["n_rows"] == 0, "empty n_rows")

    # --- redundant skeptic case ---------------------------------------------
    red = []
    for i in range(6):
        red.append(make_row(f"q{i}", deterministic_pass=True, net_outcome=5.0))
        red.append(make_row(f"d{i}", deterministic_pass=True,
                            skeptic_flags=["pile_on"], net_outcome=5.0))
    rep2 = evaluate_skeptic_impact(red)
    check(rep2["verdict"] == "skeptic_redundant", f"verdict {rep2['verdict']}")
    check(rep2["avg_net_quiet_minus_doubt"] == 0.0, "zero incremental")

    # --- defensive normalization --------------------------------------------
    junk = evaluate_skeptic_impact([{"decision_id": "x"},
                                    {"net_outcome": float("nan"),
                                     "deterministic_pass": True},
                                    {"deterministic_pass": True,
                                     "no_trade_boost": None}])
    check(junk["n_rows"] == 3, "junk rows normalized")
    check(junk["quiet"]["n_resolved"] == 0, "nan outcome unresolved")

    # --- ledger round-trip (temp db using research_ledger) -------------------
    import tempfile
    try:
        sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
        from research_ledger import ResearchLedger
        with tempfile.TemporaryDirectory() as td:
            db = os.path.join(td, "led.db")
            led = ResearchLedger(db)
            review = {"adjusted_probability": {"p_call": 0.5},
                      "flags": ["stale_catalyst"], "invalidation": [],
                      "no_trade_boost": 0.05, "notes": "t"}
            did1 = led.record_snapshot(kind="candidate", accepted=True,
                                       symbol="SPY261231C00450000",
                                       extra={"thesis_review": review})
            did2 = led.record_snapshot(kind="candidate", accepted=False,
                                       reject_code="ev_gate",
                                       symbol="SPY261231C00455000",
                                       extra={"thesis_review":
                                              {"flags": [],
                                               "no_trade_boost": 0.0}})
            led.record_snapshot(kind="candidate", accepted=True,
                                symbol="SPY261231P00440000")  # no review
            led.record_event(decision_id=did1, event_type="outcome",
                             payload={"net_pnl": -42.5, "class": "realized"})
            led.close()
            rows = rows_from_ledger(db)
            check(len(rows) == 2, f"ledger rows {len(rows)} (review-only)")
            by_id = {r["decision_id"]: r for r in rows}
            check(by_id[did1]["net_outcome"] == -42.5, "joined outcome")
            check(by_id[did1]["skeptic_flags"] == ["stale_catalyst"], "flags")
            check(by_id[did1]["deterministic_pass"] is True, "pass flag")
            check(by_id[did2]["deterministic_pass"] is False, "reject flag")
            check(by_id[did2]["net_outcome"] is None, "unresolved reject")
    except ImportError:
        print("note: research_ledger unavailable; ledger round-trip skipped")
    check(rows_from_ledger("/nonexistent/led.db") == [], "missing db fail-open")

    print("skeptic_impact self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main(argv: List[str]) -> int:
    if "--selftest" in argv:
        return _self_test()
    if "--fixture" in argv:
        print(format_report(evaluate_skeptic_impact(fixture_rows())))
        return 0
    if "--ledger" in argv:
        i = argv.index("--ledger")
        path = argv[i + 1] if i + 1 < len(argv) else "research_ledger.db"
        rows = rows_from_ledger(path)
        if not rows:
            print(f"skeptic_impact: no skeptic-labeled rows in {path} "
                  "(missing db, or extra_json lacks thesis_review)")
            return 0
        print(format_report(evaluate_skeptic_impact(rows)))
        return 0
    print(__doc__)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
