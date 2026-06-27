"""
P1 — Evidence-EV Leaderboards (analytics only, additive, fail-open).

The headline learning tables. For every piece of *evidence* that was frozen at
entry — which agents convicted, the regime, the candlestick pattern, and the
feature buckets (IV / DTE / delta / direction / strength / strategy) — this
module answers one question per cohort:

    Feature            | Trades | Avg Return | Win% | EV
    Trend Agent        |   610  |   +7.5%    | 60%  | Strong
    Bull Flag          |   218  |   +8.3%    | 61%  | High
    Low IV             |   381  |   -1.2%    | 48%  | Negative

EV here is the **Bayesian-smoothed expected return per trade** (percent),
shrunk toward the global base rate so a cohort seen three times never looks like
a sure thing. The verdict label (Strong / High / Low / Weak / Negative) folds EV
sign together with sample size.

Source of truth: closed rows in ``episodes.db`` (broker round-trips + advisory
backfill today; live evidence-stamped trades as they close). Reuses
``ev_attribution.bucket_stats`` for the per-cohort stat block.

STRICTLY analytics: never opens / closes / sizes / blocks a trade, never imports
the live trader, never touches the network. Every reader fails open: a missing
or malformed store yields empty tables, never an exception.
"""

import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional

import ev_attribution as eva
import oracle_analytics as oa

# Bayesian shrink: a cohort of size n is blended with the global prior with
# weight n / (n + PRIOR_K). MIN_FULL is the sample size at which a cohort is
# considered fully trusted for the verdict's confidence component.
PRIOR_K = 20.0
MIN_FULL = 30.0

# A cohort smaller than this is never called better than "Weak".
MIN_SAMPLES = 5

# An agent is counted as having "convicted" on a trade when the magnitude of its
# net (bull - bear) vote clears this floor.
CONVICTION_NET = 0.05

# Verdict labels.
V_STRONG = "Strong"
V_HIGH = "High"
V_LOW = "Low"
V_WEAK = "Weak"
V_NEGATIVE = "Negative"

# dimension -> evidence key (categorical dims read straight from evidence).
# 'agent' is special-cased (multi-valued via agent_votes).
CATEGORICAL_DIMS = {
    "pattern": "pattern",
    "regime": "regime_label",
    "iv_bucket": "iv_bucket",
    "dte_bucket": "dte_bucket",
    "delta_bucket": "delta_bucket",
    "direction": "direction",
    "strength": "strength",
    "strategy": "strategy",
}

# All dimensions in report order.
DIMENSIONS = ("agent",) + tuple(CATEGORICAL_DIMS)


@dataclass
class EvidenceRow:
    feature: str
    trades: int
    avg_return_pct: Optional[float]
    win_rate: float
    ev: Optional[float]
    verdict: str

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Loading & normalization (episode row -> trade-like record)
# --------------------------------------------------------------------------- #
def _normalize(raw: dict) -> Optional[dict]:
    """Flatten a completed episode row into a trade-like evidence record.

    Reads ``features_json.evidence`` onto the top level, maps the episode's
    net P/L onto the ``pnl`` / ``pnl_percent`` keys the analytics helpers expect,
    and keeps ``agent_votes`` for the agent dimension. Returns None for rows with
    no realized P/L (nothing to attribute)."""
    if not isinstance(raw, dict):
        return None
    pnl_pct = raw.get("net_pnl_pct")
    pnl_dollars = raw.get("net_pnl_dollars")
    if pnl_pct is None and pnl_dollars is None:
        return None

    evidence: dict = {}
    try:
        feats = json.loads(raw.get("features_json") or "{}")
        if isinstance(feats, dict):
            ev = feats.get("evidence")
            if isinstance(ev, dict):
                evidence = ev
    except (TypeError, ValueError):
        evidence = {}

    rec = dict(evidence)  # flatten categorical evidence to top level
    rec["agent_votes"] = evidence.get("agent_votes")
    rec["pnl"] = pnl_dollars
    rec["pnl_percent"] = pnl_pct
    rec["net_pnl_pct"] = pnl_pct
    rec["outcome"] = raw.get("outcome")
    rec["mode"] = raw.get("mode")
    rec["decision_id"] = raw.get("decision_id")
    return rec


def load_completed(db_path: str = "episodes.db") -> List[dict]:
    """Normalized closed-episode records from the store. Fail-open to []."""
    try:
        from episode_store import EpisodeStore
        store = EpisodeStore(db_path)
        try:
            raw = store.completed()
        finally:
            store.close()
    except Exception:
        return []
    out = []
    for r in raw or []:
        norm = _normalize(r)
        if norm is not None:
            out.append(norm)
    return out


# --------------------------------------------------------------------------- #
# Cohort math
# --------------------------------------------------------------------------- #
def _avg_return(rows: List[dict]) -> Optional[float]:
    vals = [oa._trade_pnl_pct(r) for r in rows]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _prior_return(rows: List[dict]) -> float:
    return _avg_return(rows) or 0.0


def _smoothed_ev(cohort: List[dict], prior_return: float) -> Optional[float]:
    """Sample-size-weighted blend of cohort avg return toward the global prior."""
    n = len(cohort)
    if n == 0:
        return None
    cohort_avg = _avg_return(cohort)
    if cohort_avg is None:
        return None
    w = n / (n + PRIOR_K)
    return round(w * cohort_avg + (1.0 - w) * prior_return, 4)


def _verdict(ev: Optional[float], n: int) -> str:
    """Fold EV sign and sample size into a single label."""
    if n < MIN_SAMPLES or ev is None:
        return V_WEAK
    conf = min(1.0, n / MIN_FULL)
    if ev <= -0.5:
        return V_NEGATIVE
    if ev >= 5.0 and conf >= 0.6:
        return V_STRONG
    if ev >= 2.0 and conf >= 0.3:
        return V_HIGH
    if ev > 0.0:
        return V_LOW
    return V_WEAK


def _row_for(feature: str, cohort: List[dict], prior_return: float
             ) -> EvidenceRow:
    stats = eva.bucket_stats(cohort)
    ev = _smoothed_ev(cohort, prior_return)
    return EvidenceRow(
        feature=feature,
        trades=stats["trades"],
        avg_return_pct=_avg_return(cohort),
        win_rate=round(stats["win_rate"], 4),
        ev=ev,
        verdict=_verdict(ev, stats["trades"]),
    )


# --------------------------------------------------------------------------- #
# Per-dimension grouping
# --------------------------------------------------------------------------- #
def _group_categorical(rows: List[dict], key: str) -> Dict[str, List[dict]]:
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        val = r.get(key)
        if val is None or val == "":
            continue
        groups.setdefault(str(val), []).append(r)
    return groups


def _group_agents(rows: List[dict]) -> Dict[str, List[dict]]:
    """One cohort per agent: the trades on which the agent convicted.

    An agent is in a trade's cohort when its net (bull-bear) vote magnitude
    clears ``CONVICTION_NET`` — i.e. it expressed a real directional opinion."""
    groups: Dict[str, List[dict]] = {}
    for r in rows:
        votes = r.get("agent_votes")
        if not isinstance(votes, dict):
            continue
        for name, vote in votes.items():
            if not isinstance(vote, dict):
                continue
            net = oa._to_float(vote.get("net"))
            if net is None:
                bull = oa._to_float(vote.get("bull")) or 0.0
                bear = oa._to_float(vote.get("bear")) or 0.0
                net = bull - bear
            if abs(net) >= CONVICTION_NET:
                groups.setdefault(str(name), []).append(r)
    return groups


def leaderboard(rows: List[dict], dimension: str) -> List[EvidenceRow]:
    """EvidenceRows for one dimension, sorted by EV desc then trades desc."""
    if dimension == "agent":
        groups = _group_agents(rows)
    else:
        key = CATEGORICAL_DIMS.get(dimension)
        if key is None:
            return []
        groups = _group_categorical(rows, key)
    prior = _prior_return(rows)
    out = [_row_for(feat, cohort, prior) for feat, cohort in groups.items()]
    out.sort(key=lambda r: ((r.ev if r.ev is not None else -1e9), r.trades),
             reverse=True)
    return out


def compute_all(rows: Optional[List[dict]] = None,
                db_path: str = "episodes.db") -> Dict[str, List[EvidenceRow]]:
    """Every dimension's leaderboard. Loads the store when rows is None."""
    if rows is None:
        rows = load_completed(db_path)
    return {dim: leaderboard(rows, dim) for dim in DIMENSIONS}


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{'+' if value >= 0 else ''}{value:.1f}%"


def _table_md(title: str, table: List[EvidenceRow]) -> List[str]:
    lines = [f"### {title}", "", "| Feature | Trades | Avg Return | Win% | EV |",
             "|---|---:|---:|---:|---|"]
    if not table:
        lines.append("| _no data_ |  |  |  |  |")
        return lines
    for r in table:
        lines.append(
            f"| {r.feature} | {r.trades} | {_pct(r.avg_return_pct)} | "
            f"{r.win_rate * 100:.0f}% | {r.verdict} |")
    return lines


_TITLES = {
    "agent": "Agents", "pattern": "Candlestick Patterns", "regime": "Regimes",
    "iv_bucket": "IV Buckets", "dte_bucket": "DTE Buckets",
    "delta_bucket": "Delta Buckets", "direction": "Direction",
    "strength": "Signal Strength", "strategy": "Strategy",
}


def format_markdown(tables: Dict[str, List[EvidenceRow]]) -> str:
    """Markdown report of every populated dimension (skips empty ones)."""
    out: List[str] = ["## Evidence-EV Leaderboards", ""]
    any_data = False
    for dim in DIMENSIONS:
        table = tables.get(dim) or []
        if not table:
            continue
        any_data = True
        out += _table_md(_TITLES.get(dim, dim), table)
        out.append("")
    if not any_data:
        out.append("_No closed episodes with evidence yet._")
    return "\n".join(out)


def to_json(tables: Dict[str, List[EvidenceRow]]) -> dict:
    return {dim: [r.to_dict() for r in rows] for dim, rows in tables.items()}


def generate_report_text(db_path: str = "episodes.db") -> str:
    return format_markdown(compute_all(db_path=db_path))


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds — synthetic in-memory episode rows)
# --------------------------------------------------------------------------- #
def _raw(net_pct, *, evidence, mode="backfill"):
    """A minimal completed-episode row as the store would return it."""
    return {
        "features_json": json.dumps({"evidence": evidence}),
        "net_pnl_pct": net_pct,
        "net_pnl_dollars": net_pct,  # 1:1 for the test
        "outcome": "win" if net_pct > 0 else "loss",
        "mode": mode,
        "decision_id": None,
    }


def _self_test() -> int:
    ok = True

    # 12 'up' winners (~+9%), 15 'down' losers (~-7%) -> direction should split.
    rows_raw = []
    for i in range(12):
        rows_raw.append(_raw(8.0 + (i % 3),
                             evidence={"direction": "up", "pattern": "hammer",
                                       "iv_bucket": "low"}))
    for i in range(15):
        rows_raw.append(_raw(-6.0 - (i % 3),
                             evidence={"direction": "down", "iv_bucket": "high"}))
    rows = [_normalize(r) for r in rows_raw]
    rows = [r for r in rows if r is not None]
    if len(rows) != 27:
        print("FAIL: normalize dropped rows", len(rows)); ok = False

    tables = compute_all(rows=rows)

    # Direction leaderboard: 'up' positive EV, 'down' negative.
    dir_rows = {r.feature: r for r in tables["direction"]}
    if "up" not in dir_rows or "down" not in dir_rows:
        print("FAIL: direction groups missing", dir_rows); ok = False
    else:
        if dir_rows["up"].trades != 12 or dir_rows["down"].trades != 15:
            print("FAIL: direction trade counts",
                  dir_rows["up"].trades, dir_rows["down"].trades); ok = False
        if not (dir_rows["up"].ev and dir_rows["up"].ev > 0):
            print("FAIL: up EV should be positive", dir_rows["up"].ev); ok = False
        if dir_rows["down"].verdict != V_NEGATIVE:
            print("FAIL: down verdict should be Negative",
                  dir_rows["down"].verdict); ok = False
        # Sorted by EV desc -> 'up' first.
        if tables["direction"][0].feature != "up":
            print("FAIL: leaderboard not EV-sorted"); ok = False

    # Win% math: up cohort is all winners.
    if abs(dir_rows["up"].win_rate - 1.0) > 1e-9:
        print("FAIL: up win rate", dir_rows["up"].win_rate); ok = False

    # Categorical pattern present only on winners -> hammer should appear.
    pat = {r.feature: r for r in tables["pattern"]}
    if "hammer" not in pat or pat["hammer"].trades != 12:
        print("FAIL: pattern grouping", pat); ok = False

    # Agent dimension: inject agent_votes, only convicted agents grouped.
    ar = []
    for i in range(10):
        ar.append(_raw(5.0, evidence={
            "direction": "up",
            "agent_votes": {
                "TrendAgent": {"net": 0.6, "conf": 0.8},
                "QuietAgent": {"net": 0.0, "conf": 0.0},  # no conviction
            }}))
    arows = [_normalize(r) for r in ar]
    atables = compute_all(rows=arows)
    anames = {r.feature for r in atables["agent"]}
    if "TrendAgent" not in anames:
        print("FAIL: convicted agent missing", anames); ok = False
    if "QuietAgent" in anames:
        print("FAIL: unconvicted agent should be excluded", anames); ok = False

    # Small cohort -> never better than Weak.
    small = [_normalize(_raw(20.0, evidence={"direction": "up"}))
             for _ in range(3)]
    st = compute_all(rows=small)
    if st["direction"] and st["direction"][0].verdict not in (V_WEAK,):
        print("FAIL: tiny cohort should be Weak",
              st["direction"][0].verdict); ok = False

    # Empty input -> empty tables, formatting never raises.
    empty = compute_all(rows=[])
    md = format_markdown(empty)
    if "No closed episodes" not in md:
        print("FAIL: empty markdown", md[:80]); ok = False

    # Garbage rows never raise.
    for junk in (None, 42, "x", {}, {"features_json": "{bad"}):
        if _normalize(junk) is not None and not isinstance(_normalize(junk), dict):
            print("FAIL: normalize garbage", junk); ok = False

    # Full markdown renders for populated tables.
    full_md = format_markdown(tables)
    if "Evidence-EV Leaderboards" not in full_md or "| up |" not in full_md:
        print("FAIL: full markdown render", full_md[:120]); ok = False

    print("evidence_attribution self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--report" in sys.argv:
        print(generate_report_text())
        sys.exit(0)
    sys.exit(_self_test())
