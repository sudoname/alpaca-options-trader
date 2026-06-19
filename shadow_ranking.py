"""
P13D — Shadow EV-portfolio replay (standalone, zero live-code touch).

Replays stored historical decisions and asks a counterfactual question: if the
screener had ranked each day's candidates by a DIFFERENT key, which would it have
picked, and how would that pick have actually done? Three ranking systems are
compared on the same frozen history:

    RANK_ORACLE   — the current LIVE sort key: oracle_score (v1).
    RANK_BEST_EV  — EV/Risk first, then EV, then oracle_score.
    RANK_LEARNED  — a v2-style blend of the (already-composed) oracle_score with
                    the Bayesian learned edge of each candidate's setup.

For each decision set the system picks its top-1; we then read that pick's
REALIZED outcome and aggregate win rate, average return, profit factor,
expectancy, max drawdown and total P/L per system.

Strictly ANALYTICS / OFFLINE: it never opens, sizes, prices, blocks or alters any
real or paper trade, never mutates a Q-table and never reaches the network beyond
the fail-open loaders. Ranking is a STABLE sort with explicit deterministic
tie-breakers, so a given record set always yields the same replay. Records can be
injected for testing.

Limitation: legacy stores keep only CLOSED rows with no scan/day batch field. Such
rows degrade to SINGLETON decision sets (one candidate each), so every system
"picks" the same trade and the comparison is degenerate for that slice. Forward
stamped rows that carry a batch/date group properly.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional

import ev_attribution as eva
import learned_edge as le
import oracle_analytics as oa

RANK_ORACLE = "oracle"
RANK_BEST_EV = "best_ev"
RANK_LEARNED = "learned"
RANKING_SYSTEMS = (RANK_ORACLE, RANK_BEST_EV, RANK_LEARNED)

# v2-style weighting used by the shadow learned ranker: the v1 composite
# (oracle_score) keeps 0.70 of the weight, the learned edge takes 0.30 — the same
# split as oracle_score_v2's learned_edge weight.
_LEARNED_ORACLE_W = 0.70
_LEARNED_EDGE_W = 0.30

_NEG_INF = float("-inf")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _f(value, default: float = _NEG_INF) -> float:
    v = oa._to_float(value)
    return v if v is not None else default


def _stable_id(rec: dict) -> str:
    return str(eva._rid(rec) or oa._get(rec, "symbol") or "")


# --------------------------------------------------------------------------- #
# Per-system sort keys (higher = preferred). All deterministic & None-tolerant.
# --------------------------------------------------------------------------- #
def _oracle_key(rec: dict, edge: Optional[dict]) -> tuple:
    return (_f(oa._trade_oracle(rec)),)


def _best_ev_key(rec: dict, edge: Optional[dict]) -> tuple:
    return (_f(eva._ev_risk(rec)), _f(eva._ev(rec)), _f(oa._trade_oracle(rec)))


def _learned_key(rec: dict, edge: Optional[dict]) -> tuple:
    o = oa._trade_oracle(rec)
    o_norm = (o / 100.0) if o is not None else 0.5
    le_score = (edge or {}).get("learned_edge_score", 0.5)
    blended = _LEARNED_ORACLE_W * o_norm + _LEARNED_EDGE_W * le_score
    return (round(blended, 6), _f(eva._ev_risk(rec)), _f(oa._trade_oracle(rec)))


_SORT_KEYS = {
    RANK_ORACLE: _oracle_key,
    RANK_BEST_EV: _best_ev_key,
    RANK_LEARNED: _learned_key,
}


# --------------------------------------------------------------------------- #
# Loading & grouping
# --------------------------------------------------------------------------- #
def load_replay_records(config: Optional[le.LearnedEdgeConfig] = None,
                        **kwargs) -> List[dict]:
    """Reuse the learned-edge loader so the replay sees the same history."""
    return le.load_edge_records(config, **kwargs)


def _group_key(rec: dict) -> Optional[str]:
    batch = oa._get(rec, "scan_id", "batch_id", "decision_batch")
    if batch is not None:
        return f"batch={batch}"
    ts = oa._get(rec, "entry_time", "opened_at", "open_time", "timestamp",
                 "as_of", "created_at", "date")
    if ts:
        day = str(ts)[:10]
        if day:
            return f"date={day}"
    return None


def group_decision_sets(records: List[dict]) -> List[List[dict]]:
    """Group records into decision sets by scan/day batch, preserving order.

    Records with no batch/date field fall into their own SINGLETON set (the
    documented legacy limitation)."""
    groups: Dict[str, List[dict]] = {}
    order: List[str] = []
    singleton = 0
    for r in records or []:
        if not isinstance(r, dict):
            continue
        gk = _group_key(r)
        if gk is None:
            gk = f"__singleton__{singleton}"
            singleton += 1
        if gk not in groups:
            groups[gk] = []
            order.append(gk)
        groups[gk].append(r)
    return [groups[k] for k in order]


# --------------------------------------------------------------------------- #
# Choosing & aggregating
# --------------------------------------------------------------------------- #
def _choose(decision_set: List[dict], system: str,
            edges: Dict[int, dict]) -> Optional[dict]:
    if not decision_set:
        return None
    keyfn = _SORT_KEYS[system]
    ordered = sorted(decision_set, key=_stable_id)
    return max(ordered, key=lambda r: keyfn(r, edges.get(id(r))))


def _choice_row(rec: dict, edge: Optional[dict]) -> dict:
    return {
        "symbol": oa._get(rec, "symbol"),
        "strategy": oa._get(rec, "strategy", "strategy_name"),
        "oracle_score": oa._trade_oracle(rec),
        "ev_per_dollar_risk": eva._ev_risk(rec),
        "learned_edge_score": (edge or {}).get("learned_edge_score"),
        "pnl": oa._trade_pnl(rec),
        "pnl_percent": oa._trade_pnl_pct(rec),
        "win": oa._is_win(rec),
    }


def _aggregate(chosen: List[dict]) -> dict:
    stats = eva.bucket_stats(chosen)
    rets = le._mean([oa._trade_pnl_pct(r) for r in chosen])
    pnls = [oa._trade_pnl(r) or 0.0 for r in chosen]
    expectancy = (sum(pnls) / len(pnls)) if pnls else 0.0
    return {
        "decisions": len(chosen),
        "win_rate": stats["win_rate"],
        "avg_return": round(rets, 4) if rets is not None else None,
        "profit_factor": stats["profit_factor"],
        "expectancy": round(expectancy, 2),
        "max_drawdown": stats["max_loss_observed"],
        "total_pnl": stats["total_pnl"],
    }


def replay(records: Optional[List[dict]] = None,
           config: Optional[le.LearnedEdgeConfig] = None) -> dict:
    """Replay the history under all three ranking systems. Never raises."""
    try:
        cfg = config or le.LearnedEdgeConfig.from_env()
        if records is None:
            records = load_replay_records(cfg)
        records = records or []

        # Pre-compute the learned edge for every record once (read-only).
        edges: Dict[int, dict] = {}
        for r in records:
            try:
                edges[id(r)] = le.estimate_edge(r, cfg, records)
            except Exception:
                edges[id(r)] = {}

        sets = group_decision_sets(records)
        out_systems: Dict[str, dict] = {}
        for system in RANKING_SYSTEMS:
            chosen: List[dict] = []
            choice_rows: List[dict] = []
            for ds in sets:
                pick = _choose(ds, system, edges)
                if pick is not None:
                    chosen.append(pick)
                    choice_rows.append(_choice_row(pick, edges.get(id(pick))))
            out_systems[system] = {
                "stats": _aggregate(chosen),
                "choices": choice_rows,
            }

        return {
            "generated_at": _now_iso(),
            "num_records": len(records),
            "num_decision_sets": len(sets),
            "systems": out_systems,
        }
    except Exception:  # pragma: no cover - fail-open
        empty = {s: {"stats": _aggregate([]), "choices": []}
                 for s in RANKING_SYSTEMS}
        return {"generated_at": _now_iso(), "num_records": 0,
                "num_decision_sets": 0, "systems": empty}


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — fully injected records)
# --------------------------------------------------------------------------- #
def _cand(symbol, oracle, ev_risk, ev, pnl, *, date="2025-01-02", rid=None):
    return {
        "id": rid or symbol, "symbol": symbol, "date": date,
        "oracle_score": oracle, "ev_per_dollar_risk": ev_risk,
        "expected_value": ev, "pnl": pnl,
        "pnl_percent": 20.0 if pnl > 0 else -20.0, "max_loss": 100.0,
        "regime": "trending", "trend": "up", "realized_vol": 0.20,
        "signal_strength": 2, "dte": 30, "entry_delta": 0.4,
    }


def _self_test() -> int:
    ok = True

    # One decision set (same date) with two candidates:
    #   A: high oracle (90) but low EV/Risk (0.05) -> LOSES.
    #   B: low oracle (60) but high EV/Risk (0.30) -> WINS.
    day1 = [
        _cand("AAA", 90.0, 0.05, 3.0, -40.0, date="2025-01-02", rid="a1"),
        _cand("BBB", 60.0, 0.30, 18.0, +50.0, date="2025-01-02", rid="b1"),
    ]
    # A second day where oracle and EV agree on the winner.
    day2 = [
        _cand("CCC", 80.0, 0.25, 15.0, +30.0, date="2025-01-03", rid="c1"),
        _cand("DDD", 40.0, 0.02, 1.0, -20.0, date="2025-01-03", rid="d1"),
    ]
    records = day1 + day2

    rep = replay(records=records, config=le.LearnedEdgeConfig())

    if rep["num_decision_sets"] != 2:
        print("FAIL: expected 2 decision sets", rep["num_decision_sets"]); ok = False

    oracle_picks = [c["symbol"] for c in rep["systems"][RANK_ORACLE]["choices"]]
    ev_picks = [c["symbol"] for c in rep["systems"][RANK_BEST_EV]["choices"]]

    # Oracle ranker picks the high-score loser on day 1.
    if oracle_picks[0] != "AAA":
        print("FAIL: oracle should pick AAA on day1", oracle_picks); ok = False
    # EV ranker picks the high-EV winner on day 1.
    if ev_picks[0] != "BBB":
        print("FAIL: best_ev should pick BBB on day1", ev_picks); ok = False
    # Both agree on day 2.
    if oracle_picks[1] != "CCC" or ev_picks[1] != "CCC":
        print("FAIL: both should pick CCC on day2", oracle_picks, ev_picks)
        ok = False

    # Hand-checked aggregates: oracle picks AAA(-40)+CCC(+30) -> 1 win / 2.
    o_stats = rep["systems"][RANK_ORACLE]["stats"]
    if o_stats["decisions"] != 2 or abs(o_stats["total_pnl"] - (-10.0)) > 1e-6:
        print("FAIL: oracle aggregate", o_stats); ok = False
    if abs(o_stats["win_rate"] - 0.5) > 1e-6:
        print("FAIL: oracle win rate", o_stats["win_rate"]); ok = False
    # EV picks BBB(+50)+CCC(+30) -> 2 wins, total 80.
    e_stats = rep["systems"][RANK_BEST_EV]["stats"]
    if abs(e_stats["total_pnl"] - 80.0) > 1e-6 or e_stats["win_rate"] != 1.0:
        print("FAIL: best_ev aggregate", e_stats); ok = False

    # Learned system runs and produces a valid pick per set.
    learned_picks = [c["symbol"] for c in rep["systems"][RANK_LEARNED]["choices"]]
    if len(learned_picks) != 2 or any(p not in
                                      ("AAA", "BBB", "CCC", "DDD")
                                      for p in learned_picks):
        print("FAIL: learned picks invalid", learned_picks); ok = False

    # Singleton fallback: rows without a date each become their own set.
    legacy = [
        {"id": "x1", "symbol": "X", "oracle_score": 70.0, "pnl": 5.0,
         "max_loss": 100.0},
        {"id": "y1", "symbol": "Y", "oracle_score": 30.0, "pnl": -5.0,
         "max_loss": 100.0},
    ]
    rep2 = replay(records=legacy, config=le.LearnedEdgeConfig())
    if rep2["num_decision_sets"] != 2:
        print("FAIL: legacy rows should be singleton sets",
              rep2["num_decision_sets"]); ok = False
    # Every system picks the same trade from a singleton set -> identical stats.
    if (rep2["systems"][RANK_ORACLE]["stats"]["total_pnl"]
            != rep2["systems"][RANK_BEST_EV]["stats"]["total_pnl"]):
        print("FAIL: singleton sets should give identical totals"); ok = False

    # Empty + garbage never raise.
    if replay(records=[], config=le.LearnedEdgeConfig())["num_decision_sets"] != 0:
        print("FAIL: empty replay"); ok = False
    replay(records=[None, 42, "x", {"junk": 1}], config=le.LearnedEdgeConfig())

    print("shadow_ranking self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
