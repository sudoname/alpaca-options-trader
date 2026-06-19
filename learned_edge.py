"""
P13A — Learned Edge Engine (pure, offline, injectable, never raises).

Given a CANDIDATE setup, this module answers: *how has the same kind of setup
done historically?* It does NOT predict price. It estimates the Bayesian-smoothed
historical edge of the candidate's setup cohort:

    win_rate, avg_return, avg_ev, avg_holding_time, max_drawdown, sample_size,
    confidence_score, ci_low/ci_high, and a single learned_edge_score in [0, 1].

Two ideas keep sparse cohorts honest:

  * BAYESIAN SMOOTHING — a Beta-Binomial prior on the win rate and a
    shrink-to-prior blend on the means. Small samples regress toward the global
    base rate; large samples trust their own data. The learned_edge_score is
    additionally pulled toward the neutral 0.5 when confidence is low, so a setup
    seen twice never looks like a sure thing.

  * HIERARCHICAL BACKOFF — if the full setup key has too few samples, drop the
    least-important dimension and re-aggregate, repeating until a cohort is large
    enough (finally the global prior). Importance order, kept-longest first:
    regime > volatility > direction > strength > dte_bucket > delta_bucket > pattern.

This module is ANALYTICS / SHADOW ONLY. It never opens, sizes, prices, blocks or
alters any real or paper trade, never mutates a Q-table, and reaches the network
only via the read-only, fail-open loaders. The live ranker does NOT import it by
default; ``estimate_edge`` is invoked by offline tooling, reports and tests.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional

import ev_attribution as eva
import feature_buckets as fb
import oracle_analytics as oa
from config_loader import ConfigLoader
from oracle_analytics import AnalyticsConfig

# How aggressively a strong EV/Risk ratio nudges the edge above neutral. A
# smoothed EV/Risk of EV_RISK_NORM maps the volatility component to its ceiling.
EV_RISK_NORM = 0.5

# Backoff levels (kept-longest first). 'pattern' is dropped first; the empty
# tuple is the global prior. A candidate without a pattern collapses L0->L1.
BACKOFF_LEVELS = (
    ("regime", "volatility", "direction", "strength", "dte_bucket", "delta_bucket", "pattern"),
    ("regime", "volatility", "direction", "strength", "dte_bucket", "delta_bucket"),
    ("regime", "volatility", "direction", "strength", "dte_bucket"),
    ("regime", "volatility", "direction", "strength"),
    ("regime", "volatility", "direction"),
    ("regime", "volatility"),
    ("regime",),
    (),
)


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class LearnedEdgeConfig:
    enabled: bool = False
    prior_strength_k: float = 20.0
    min_samples_full: int = 30
    backoff_min_samples: int = 8
    spread_trades_file: str = "spread_paper_trades.json"
    trade_history_file: str = "trading_history.json"
    episode_db_file: str = "episodes.db"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "LearnedEdgeConfig":
        try:
            cfg = loader if loader is not None else ConfigLoader(path=path)
            return LearnedEdgeConfig(
                enabled=cfg.get_bool("LEARNED_EDGE_ENABLED", False),
                prior_strength_k=cfg.get_float("LEARNED_EDGE_PRIOR_K", 20.0),
                min_samples_full=cfg.get_int("LEARNED_EDGE_MIN_SAMPLES_FULL", 30),
                backoff_min_samples=cfg.get_int(
                    "LEARNED_EDGE_BACKOFF_MIN_SAMPLES", 8),
                spread_trades_file=cfg.get_str("SPREAD_PAPER_TRADES_FILE",
                                               "spread_paper_trades.json"),
                trade_history_file=cfg.get_str("TRADE_HISTORY_FILE",
                                               "trading_history.json"),
                episode_db_file=cfg.get_str("EPISODE_DB_FILE", "episodes.db"),
            )
        except Exception:  # pragma: no cover - fail-open
            return LearnedEdgeConfig()


# --------------------------------------------------------------------------- #
# Record loading (read-only, fail-open, dedup)
# --------------------------------------------------------------------------- #
def _hold_days(rec: dict) -> Optional[float]:
    return oa._to_float(
        oa._get(rec, "hold_days", "holding_days", "days_held", "days_in_trade"))


def _normalize_episode(row: dict) -> Optional[dict]:
    """Map an episode_store completed row to a trade-like record.

    Keeps the original ``features_json``/``state_key`` so feature extraction can
    read the point-in-time market context. Returns None for rows with neither a
    dollar nor a percent P/L (nothing to learn from)."""
    if not isinstance(row, dict):
        return None
    pnl = row.get("net_pnl_dollars")
    pnl_pct = row.get("net_pnl_pct")
    if pnl is None and pnl_pct is None:
        return None
    out = dict(row)
    out["pnl"] = pnl
    out["pnl_percent"] = pnl_pct
    out["hold_days"] = row.get("hold_days")
    out["id"] = row.get("decision_id") or row.get("id")
    act = str(row.get("chosen_action") or "").upper()
    if act == "CALL":
        out.setdefault("direction", "up")
    elif act == "PUT":
        out.setdefault("direction", "down")
    return out


def load_edge_records(config: Optional[LearnedEdgeConfig] = None, *,
                      spread_trades: Optional[List[dict]] = None,
                      history_trades: Optional[List[dict]] = None,
                      snapshots: Optional[List[dict]] = None,
                      episodes: Optional[List[dict]] = None,
                      shadow_rows: Optional[List[dict]] = None) -> List[dict]:
    """All closed setups across every store, deduped by trade id. Fail-open.

    Sources: spread/history/attribution (via ``ev_attribution.load_closed_records``)
    and the RL episode store. Any source that errors contributes nothing."""
    cfg = config or LearnedEdgeConfig.from_env()
    records: List[dict] = []
    seen = set()

    def _add(rec: dict, fallback_id: str) -> None:
        rid = str(eva._rid(rec) or rec.get("id") or fallback_id)
        if rid in seen:
            return
        seen.add(rid)
        records.append(rec)

    # 1) Closed spread + single-leg history + attribution snapshots.
    try:
        an_cfg = AnalyticsConfig(spread_trades_file=cfg.spread_trades_file,
                                 trade_history_file=cfg.trade_history_file)
        closed = eva.load_closed_records(config=an_cfg, trades=spread_trades,
                                         snapshots=snapshots,
                                         history_trades=history_trades)
    except Exception:
        closed = []
    for i, rec in enumerate(closed or []):
        if isinstance(rec, dict):
            _add(rec, f"closed#{i}")

    # 2) RL episode store (read-only).
    eps = episodes
    if eps is None:
        try:
            from episode_store import EpisodeStore
            eps = EpisodeStore(cfg.episode_db_file).completed()
        except Exception:
            eps = []
    for i, row in enumerate(eps or []):
        norm = _normalize_episode(row)
        if norm is not None:
            _add(norm, f"episode#{i}")

    # 3) Optional pre-loaded shadow rows (already trade-like).
    for i, rec in enumerate(shadow_rows or []):
        if isinstance(rec, dict):
            _add(rec, f"shadow#{i}")

    return records


# --------------------------------------------------------------------------- #
# Cohort statistics & global prior
# --------------------------------------------------------------------------- #
def _mean(values: List[float]) -> Optional[float]:
    vals = [v for v in values if v is not None]
    return sum(vals) / len(vals) if vals else None


def _cohort_stats(rows: List[dict]) -> dict:
    """Plain (un-smoothed) statistics over a cohort of closed records."""
    n = len(rows)
    wins = sum(1 for r in rows if oa._is_win(r))
    returns = _mean([oa._trade_pnl_pct(r) for r in rows])
    evs = _mean([eva._ev(r) for r in rows])
    ev_risks = _mean([eva._ev_risk(r) for r in rows])
    holds = _mean([_hold_days(r) for r in rows])
    bstats = eva.bucket_stats(rows)
    return {
        "trades": n,
        "wins": wins,
        "win_rate": (wins / n) if n else 0.0,
        "avg_return": returns,
        "avg_ev": evs,
        "avg_ev_risk": ev_risks,
        "avg_holding_time": holds,
        "max_drawdown": bstats.get("max_loss_observed", 0.0),
        "profit_factor": bstats.get("profit_factor"),
        "total_pnl": bstats.get("total_pnl", 0.0),
    }


def compute_prior(records: List[dict]) -> dict:
    """Global base rate across every record — the shrink target."""
    stats = _cohort_stats(records or [])
    return {
        "n": stats["trades"],
        "win_rate": stats["win_rate"],
        "avg_return": stats["avg_return"] or 0.0,
        "avg_ev": stats["avg_ev"] or 0.0,
        "avg_ev_risk": stats["avg_ev_risk"] or 0.0,
    }


# --------------------------------------------------------------------------- #
# Setup-key matching & backoff
# --------------------------------------------------------------------------- #
def _record_key(record: dict) -> Dict[str, Optional[str]]:
    """Full feature dict (every dimension present, value may be None)."""
    return fb.extract_features(record)


def _candidate_levels(cand_key: Dict[str, Optional[str]]):
    """Per-candidate backoff order, collapsing the pattern level when absent."""
    has_pattern = bool(cand_key.get("pattern"))
    levels = []
    for dims in BACKOFF_LEVELS:
        d = tuple(x for x in dims if x != "pattern" or has_pattern)
        if not levels or levels[-1] != d:
            levels.append(d)
    return levels


def _matches(rec_key: Dict[str, Optional[str]],
             cand_key: Dict[str, Optional[str]], dims) -> bool:
    for dim in dims:
        if rec_key.get(dim) != cand_key.get(dim):
            return False
    return True


# --------------------------------------------------------------------------- #
# Bayesian smoothing
# --------------------------------------------------------------------------- #
def _smooth(stats: dict, prior: dict, config: LearnedEdgeConfig) -> dict:
    """Beta-Binomial win rate + shrink-to-prior means + neutral-pulled edge."""
    k = max(0.0, float(config.prior_strength_k))
    n = stats["trades"]
    wins = stats["wins"]

    p_prior = _clamp01(prior["win_rate"])
    alpha0 = p_prior * k
    denom = n + k
    smoothed_win_rate = (wins + alpha0) / denom if denom > 0 else p_prior

    w = (n / denom) if denom > 0 else 0.0
    s_return = w * (stats["avg_return"] or prior["avg_return"]) \
        + (1.0 - w) * prior["avg_return"]
    s_ev = w * (stats["avg_ev"] or prior["avg_ev"]) + (1.0 - w) * prior["avg_ev"]
    s_ev_risk = w * (stats["avg_ev_risk"] or prior["avg_ev_risk"]) \
        + (1.0 - w) * prior["avg_ev_risk"]

    min_full = max(1, int(config.min_samples_full))
    confidence_score = min(1.0, n / min_full)

    p = _clamp01(smoothed_win_rate)
    se = math.sqrt(p * (1.0 - p) / denom) if denom > 0 else 0.0
    ci_low = _clamp01(p - 1.96 * se)
    ci_high = _clamp01(p + 1.96 * se)

    normalized = s_ev_risk / EV_RISK_NORM if EV_RISK_NORM else 0.0
    raw_edge = 0.5 * p + 0.5 * _clamp01(0.5 + normalized)
    learned_edge_score = confidence_score * raw_edge \
        + (1.0 - confidence_score) * 0.5

    return {
        "learned_edge_score": round(_clamp01(learned_edge_score), 4),
        "win_rate": round(p, 4),
        "avg_return": round(s_return, 4),
        "avg_ev": round(s_ev, 4),
        "avg_ev_risk": round(s_ev_risk, 4),
        "confidence_score": round(confidence_score, 4),
        "ci_low": round(ci_low, 4),
        "ci_high": round(ci_high, 4),
    }


def _neutral_estimate(setup_key: Dict[str, Optional[str]]) -> dict:
    return {
        "learned_edge_score": 0.5, "win_rate": 0.5, "avg_return": 0.0,
        "avg_ev": 0.0, "avg_ev_risk": 0.0, "avg_holding_time": None,
        "max_drawdown": 0.0, "sample_size": 0, "confidence_score": 0.0,
        "ci_low": 0.0, "ci_high": 1.0, "backoff_level": None,
        "matched_dims": None, "setup_key": setup_key,
    }


# --------------------------------------------------------------------------- #
# Public: estimate a candidate's learned edge
# --------------------------------------------------------------------------- #
def estimate_edge(candidate: dict,
                  config: Optional[LearnedEdgeConfig] = None,
                  records: Optional[List[dict]] = None) -> dict:
    """Bayesian-smoothed historical edge for ``candidate``'s setup. Never raises.

    Pass ``records`` to stay pure/offline; otherwise they are loaded fail-open.
    A candidate with no matching history (or no records at all) returns a neutral
    0.5 edge with zero confidence — never false confidence.
    """
    try:
        cfg = config or LearnedEdgeConfig.from_env()
        if records is None:
            records = load_edge_records(cfg)
        cand_key = fb.make_setup_key(candidate)
        if not records:
            return _neutral_estimate(cand_key)

        prior = compute_prior(records)
        rec_keys = [(_record_key(r), r) for r in records]

        chosen_rows: List[dict] = []
        chosen_dims = ()
        chosen_level = len(BACKOFF_LEVELS) - 1
        levels = _candidate_levels(cand_key)
        for idx, dims in enumerate(levels):
            cohort = [r for rk, r in rec_keys if _matches(rk, cand_key, dims)]
            chosen_rows, chosen_dims, chosen_level = cohort, dims, idx
            if len(cohort) >= cfg.backoff_min_samples:
                break
        # If even the global level is sparse, we still use it (chosen_rows set).

        stats = _cohort_stats(chosen_rows)
        smoothed = _smooth(stats, prior, cfg)
        out = dict(smoothed)
        out.update({
            "avg_holding_time": stats["avg_holding_time"],
            "max_drawdown": stats["max_drawdown"],
            "sample_size": stats["trades"],
            "backoff_level": chosen_level,
            "matched_dims": list(chosen_dims),
            "setup_key": cand_key,
        })
        return out
    except Exception:  # pragma: no cover - fail-open to neutral
        try:
            return _neutral_estimate(fb.make_setup_key(candidate))
        except Exception:
            return _neutral_estimate({})


def build_edge_index(records: List[dict]) -> Dict[tuple, dict]:
    """Aggregate every observed FULL setup key -> cohort stats (for reports)."""
    groups: Dict[tuple, List[dict]] = {}
    for r in records or []:
        try:
            key = fb.make_setup_key(r)
            tup = fb.setup_key_tuple(key)
        except Exception:
            continue
        groups.setdefault(tup, []).append(r)
    index = {}
    for tup, rows in groups.items():
        stats = _cohort_stats(rows)
        stats["key_str"] = fb.setup_key_str(dict(tup))
        index[tup] = stats
    return index


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — fully injected records)
# --------------------------------------------------------------------------- #
def _rec(regime, direction, vol, win, *, ev=10.0, ev_risk=0.10, max_loss=100.0,
         pnl_pct=None, strength=2, dte=30, delta=0.4, pattern=None, rid=None):
    pnl = 50.0 if win else -50.0
    return {
        "id": rid, "regime": regime, "trend": direction, "realized_vol": vol,
        "signal_strength": strength, "dte": dte, "entry_delta": delta,
        "candlestick_pattern": pattern, "pnl": pnl,
        "pnl_percent": pnl_pct if pnl_pct is not None else (20.0 if win else -20.0),
        "expected_value": ev, "ev_per_dollar_risk": ev_risk, "max_loss": max_loss,
        "hold_days": 3,
    }


def _self_test() -> int:
    ok = True
    cfg = LearnedEdgeConfig(prior_strength_k=20.0, min_samples_full=30,
                            backoff_min_samples=8)

    # A large, strongly-winning cohort: trending/up/normal.
    big = [_rec("trending", "up", 0.20, i % 5 != 0, rid=f"b{i}")
           for i in range(50)]  # 80% win rate
    # A handful of mixed records elsewhere to give the prior some spread.
    other = [_rec("ranging", "down", 0.10, i % 2 == 0, rid=f"o{i}")
             for i in range(10)]
    records = big + other

    cand = {"regime": "trending", "trend": "up", "realized_vol": 0.20,
            "signal_strength": 2, "dte": 30, "entry_delta": 0.4}
    est = estimate_edge(cand, cfg, records)
    if est["sample_size"] < 8:
        print("FAIL: big cohort should match many", est["sample_size"]); ok = False
    if est["win_rate"] < 0.65:
        print("FAIL: smoothed win rate should track the 80% cohort",
              est["win_rate"]); ok = False
    if est["learned_edge_score"] <= 0.5:
        print("FAIL: winning cohort edge should exceed neutral",
              est["learned_edge_score"]); ok = False
    if not (0.0 <= est["ci_low"] <= est["ci_high"] <= 1.0):
        print("FAIL: CI must be ordered within [0,1]",
              est["ci_low"], est["ci_high"]); ok = False

    # Small, sparse cohort regresses toward the neutral prior with low confidence.
    sparse = [_rec("volatile", "flat", 0.60, True, rid="s0"),
              _rec("volatile", "flat", 0.60, True, rid="s1")]
    cand2 = {"regime": "volatile", "trend": "flat", "realized_vol": 0.60}
    est2 = estimate_edge(cand2, cfg, sparse + other)
    if est2["confidence_score"] >= 0.5:
        print("FAIL: 2-sample cohort should be low confidence",
              est2["confidence_score"]); ok = False
    # Edge pulled toward neutral despite 100% sample win rate.
    if est2["learned_edge_score"] > 0.85:
        print("FAIL: low-confidence edge should be pulled toward 0.5",
              est2["learned_edge_score"]); ok = False

    # Backoff: a candidate whose full key is unseen falls back to a coarser
    # cohort rather than returning nothing.
    cand3 = {"regime": "trending", "trend": "up", "realized_vol": 0.20,
             "signal_strength": 2, "dte": 30, "entry_delta": 0.4,
             "candlestick_pattern": "never_seen_pattern"}
    est3 = estimate_edge(cand3, cfg, records)
    if est3["sample_size"] < 8:
        print("FAIL: backoff should recover a large cohort", est3); ok = False
    if "pattern" in (est3["matched_dims"] or []):
        print("FAIL: unseen pattern should be dropped by backoff",
              est3["matched_dims"]); ok = False

    # No records -> neutral, zero confidence.
    est4 = estimate_edge(cand, cfg, [])
    if est4["learned_edge_score"] != 0.5 or est4["confidence_score"] != 0.0:
        print("FAIL: empty records must be neutral", est4); ok = False

    # Never raises on garbage candidate.
    for junk in (None, 42, "x", [], {"weird": object()}):
        try:
            estimate_edge(junk, cfg, records)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk candidate", junk, exc); ok = False

    # Index builds and carries readable keys.
    idx = build_edge_index(records)
    if not idx or not any("regime=" in v["key_str"] for v in idx.values()):
        print("FAIL: edge index missing readable keys"); ok = False

    print("learned_edge self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
