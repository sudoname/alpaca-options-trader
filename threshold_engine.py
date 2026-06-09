"""
Phase 8C — Threshold recommendation engine (advisory, read-only, offline-pure).

This module answers: *which Oracle Score, Volatility Edge, DTE, IV Rank and
Strategy thresholds have actually produced the best results in the simulated
paper book?* It reads only historical artifacts —

    * spread_paper_trades.json     (CLOSED simulated spread trades)
    * oracle_training_dataset.csv  (features / predictions / outcomes)
    * expected_move_history.csv    (expected-move predictions + vol edge)

and emits performance tables + recommendations. It is STRICTLY advisory: it
contains no order placement, no spread execution, no live-trading or gating
logic — nothing here can open, modify, gate, or close any real or paper
position. It only describes what the historical data says and suggests
thresholds a human could later choose to apply. Every reader fails open
(missing / empty / malformed → "no data") and every public function returns a
plain dict that is safe to format even when inputs are empty.

It builds on the Phase 8B analytics layer (:mod:`oracle_analytics`), reusing its
robust readers, trade accessors and bucket helpers so the two layers stay
consistent.

Public API (all accept an optional ``config`` and optional pre-loaded ``trades``
/ ``em_rows`` so they are trivially unit-testable):

    analyze_oracle_score_thresholds()
    analyze_vol_edge_thresholds()
    analyze_dte_buckets()
    analyze_iv_rank_buckets()
    analyze_strategy_performance()
    compute_confidence()
    compute_data_coverage()
    compute_recommendations()
"""

import logging
from collections import OrderedDict
from datetime import timedelta
from typing import Dict, List, Optional

import oracle_analytics as oa
from oracle_analytics import AnalyticsConfig

try:  # canonical strategy names (best-effort; data-driven either way)
    from spread_builder import (
        BEARISH_CALL_CREDIT_SPREAD, BULLISH_PUT_CREDIT_SPREAD,
        DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR,
    )
    CANONICAL_STRATEGIES = (
        BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD, IRON_CONDOR,
        DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD,
    )
except Exception:  # pragma: no cover - defensive
    CANONICAL_STRATEGIES = (
        "bullish_put_credit_spread", "bearish_call_credit_spread",
        "iron_condor", "debit_call_spread", "debit_put_spread",
    )

logger = logging.getLogger(__name__)

# Thresholds tested (Req 2 / Req 3). Edges are fractions (0.01 == 1%).
ORACLE_SCORE_THRESHOLDS = (40.0, 50.0, 60.0, 70.0, 80.0, 90.0)
VOL_EDGE_THRESHOLDS = (0.0, 0.01, 0.02, 0.03, 0.04)

# Confidence boundaries (Req 8): <50 Low, 50-200 Medium, >200 High.
CONFIDENCE_LOW_MAX = 50
CONFIDENCE_HIGH_MIN = 200

# Coverage horizons surfaced in DATA_COVERAGE (Req 7 example uses 1D/7D/30D).
COVERAGE_HORIZONS = OrderedDict([("1d", 1), ("3d", 3), ("7d", 7), ("30d", 30)])


# --------------------------------------------------------------------------- #
# Core metric: profit factor
# --------------------------------------------------------------------------- #
def _profit_factor(trades: List[dict]):
    """gross_profit / gross_loss over a list of closed trades.

    Returns a float, ``float('inf')`` when there are wins but no losses, and
    ``None`` when there are no trades at all (undefined). ``0.0`` when there is
    no profit. Never raises.
    """
    if not trades:
        return None
    gross_profit = 0.0
    gross_loss = 0.0
    for t in trades:
        pnl = oa._trade_pnl(t)
        if pnl is None:
            continue
        if pnl > 0:
            gross_profit += pnl
        elif pnl < 0:
            gross_loss += -pnl
    if gross_loss > 0:
        return round(gross_profit / gross_loss, 2)
    return float("inf") if gross_profit > 0 else 0.0


def _row(trades: List[dict], label) -> dict:
    """A performance row: label + count / win-rate / pnl / avg / profit-factor."""
    agg = oa._aggregate(trades)
    n = agg["trades"]
    return {
        "label": label,
        "trades": n,
        "win_rate": agg["win_rate"],
        "pnl": agg["pnl"],
        "avg_pnl": round(agg["pnl"] / n, 2) if n else 0.0,
        "profit_factor": _profit_factor(trades),
    }


def _pf_sort_key(row: dict):
    """Sort key that ranks rows by profit factor, then avg PnL.

    ``None`` profit factor (no trades) ranks lowest; ``inf`` ranks highest.
    """
    pf = row["profit_factor"]
    if pf is None:
        pf = float("-inf")
    return (pf, row["avg_pnl"], row["pnl"])


# --------------------------------------------------------------------------- #
# Req 2 — Oracle score thresholds
# --------------------------------------------------------------------------- #
def analyze_oracle_score_thresholds(config: Optional[AnalyticsConfig] = None,
                                    trades: Optional[List[dict]] = None) -> dict:
    """Performance at each ``score >= T`` cut + a recommended minimum score."""
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)
    scored = [t for t in closed if oa._trade_oracle(t) is not None]

    rows = []
    for thr in ORACLE_SCORE_THRESHOLDS:
        subset = [t for t in scored if oa._trade_oracle(t) >= thr]
        r = _row(subset, ">= %g" % thr)
        r["threshold"] = thr
        rows.append(r)

    recommended = _recommend_threshold(rows)
    return {"rows": rows, "recommended_min_oracle_score": recommended}


# --------------------------------------------------------------------------- #
# Req 3 — Volatility-edge thresholds
# --------------------------------------------------------------------------- #
def analyze_vol_edge_thresholds(config: Optional[AnalyticsConfig] = None,
                                trades: Optional[List[dict]] = None) -> dict:
    """Performance at each ``edge >= T`` cut + a recommended minimum edge."""
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)
    edged = [t for t in closed if oa._trade_edge(t) is not None]

    rows = []
    for thr in VOL_EDGE_THRESHOLDS:
        subset = [t for t in edged if oa._trade_edge(t) >= thr]
        r = _row(subset, ">= %.0f%%" % (thr * 100))
        r["threshold"] = thr
        rows.append(r)

    recommended = _recommend_threshold(rows)
    return {"rows": rows, "recommended_min_volatility_edge": recommended}


def _recommend_threshold(rows: List[dict]):
    """Pick the threshold whose subset has the best profit factor / avg PnL.

    Only rows with at least one trade are eligible. Rows are evaluated from the
    lowest threshold up, so on a tie ``max`` returns the *least restrictive*
    (lowest) threshold that still achieves top performance — i.e. the smallest
    minimum cut that retains the most trades. Returns the winning row's
    ``threshold`` (a number) or ``None`` when no row has any trades.
    """
    eligible = [r for r in rows if r["trades"] > 0]
    if not eligible:
        return None
    best = max(eligible, key=_pf_sort_key)
    return best["threshold"]


# --------------------------------------------------------------------------- #
# Req 4 / Req 5 — DTE & IV-rank buckets
# --------------------------------------------------------------------------- #
def _bucket_table(config, trades, value_fn, buckets) -> dict:
    closed = oa.load_closed_spread_trades(config, trades)
    table = oa._bucket(closed, value_fn, buckets)  # OrderedDict label->aggregate
    rows = []
    for label, agg in table.items():
        rows.append({
            "label": label,
            "trades": agg["trades"],
            "win_rate": agg["win_rate"],
            "pnl": agg["pnl"],
        })
    # recommend the populated bucket with the highest PnL (tie: win rate).
    eligible = [r for r in rows if r["trades"] > 0]
    recommended = None
    if eligible:
        recommended = max(eligible,
                          key=lambda r: (r["pnl"], r["win_rate"]))["label"]
    return {"rows": rows, "recommended": recommended}


def analyze_dte_buckets(config: Optional[AnalyticsConfig] = None,
                        trades: Optional[List[dict]] = None) -> dict:
    """Win rate / PnL by DTE bucket + a recommended DTE range."""
    config = config or AnalyticsConfig.from_env()
    t = _bucket_table(config, trades, oa._trade_dte, oa._DTE_BUCKETS)
    return {"rows": t["rows"], "recommended_dte_range": t["recommended"]}


def analyze_iv_rank_buckets(config: Optional[AnalyticsConfig] = None,
                            trades: Optional[List[dict]] = None) -> dict:
    """Win rate / PnL by IV-rank bucket + a recommended IV-rank range."""
    config = config or AnalyticsConfig.from_env()
    t = _bucket_table(config, trades, oa._trade_iv_rank, oa._IV_RANK_BUCKETS)
    return {"rows": t["rows"], "recommended_iv_rank_range": t["recommended"]}


# --------------------------------------------------------------------------- #
# Req 6 — Strategy performance
# --------------------------------------------------------------------------- #
def analyze_strategy_performance(config: Optional[AnalyticsConfig] = None,
                                 trades: Optional[List[dict]] = None) -> dict:
    """Per-strategy performance + best / worst strategy by profit factor."""
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)

    groups: "OrderedDict[str, list]" = OrderedDict()
    # seed the canonical strategies so they always appear (even with 0 trades).
    for name in CANONICAL_STRATEGIES:
        groups[name] = []
    for t in closed:
        strat = (t.get("strategy") or "unknown")
        groups.setdefault(strat, []).append(t)

    rows = [_row(rows_, strat) for strat, rows_ in groups.items()]

    eligible = [r for r in rows if r["trades"] > 0]
    best = max(eligible, key=_pf_sort_key)["label"] if eligible else None
    worst = min(eligible, key=_pf_sort_key)["label"] if eligible else None
    return {"rows": rows, "best_strategy": best, "worst_strategy": worst}


# --------------------------------------------------------------------------- #
# Req 8 — Confidence scoring
# --------------------------------------------------------------------------- #
def compute_confidence(n_trades: int) -> str:
    """'Low' (<50), 'Medium' (50-200), or 'High' (>200) from sample size."""
    n = n_trades or 0
    if n < CONFIDENCE_LOW_MAX:
        return "Low"
    if n <= CONFIDENCE_HIGH_MIN:
        return "Medium"
    return "High"


# --------------------------------------------------------------------------- #
# Req 7 (DATA_COVERAGE) — data coverage + prediction coverage
# --------------------------------------------------------------------------- #
def compute_data_coverage(config: Optional[AnalyticsConfig] = None,
                          trades: Optional[List[dict]] = None,
                          em_rows: Optional[List[dict]] = None,
                          dataset_rows: Optional[List[dict]] = None) -> dict:
    """Trades / symbols analyzed + per-horizon prediction coverage.

    Prediction coverage for a horizon = fraction of prediction rows (that have
    a value in ``expected_move_<h>`` plus a usable timestamp+price) for which a
    later same-symbol observation at least ``h`` days out exists, so the
    prediction could actually be evaluated. Coverage naturally falls for longer
    horizons (less future data). Returns 0.0 coverage when there is no data.
    """
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)
    rows = em_rows if em_rows is not None else oa.read_csv_rows(config.expected_move_file)
    ds = dataset_rows if dataset_rows is not None else oa.read_csv_rows(config.training_dataset_file)

    # symbols seen across all sources.
    symbols = set()
    for t in closed:
        s = str(t.get("symbol") or "").strip().upper()
        if s:
            symbols.add(s)
    for r in rows:
        s = str(r.get("symbol") or "").strip().upper()
        if s:
            symbols.add(s)
    for r in ds:
        s = str(r.get("symbol") or "").strip().upper()
        if s:
            symbols.add(s)

    # group EM rows by symbol with parsed ts + price, sorted by time.
    by_symbol: Dict[str, List[dict]] = {}
    for r in rows:
        sym = str(r.get("symbol") or "").strip().upper()
        ts = oa._parse_ts(r.get("timestamp"))
        price = oa._to_float(oa._get(r, "in_price", "price"))
        if not sym or ts is None or price is None or price <= 0:
            continue
        by_symbol.setdefault(sym, []).append({"ts": ts, "row": r})
    for seq in by_symbol.values():
        seq.sort(key=lambda x: x["ts"])

    coverage = OrderedDict()
    for hname, hdays in COVERAGE_HORIZONS.items():
        col = "expected_move_" + hname
        candidates = 0
        matched = 0
        for seq in by_symbol.values():
            for i, base in enumerate(seq):
                if oa._to_float(base["row"].get(col)) is None:
                    continue
                candidates += 1
                target_ts = base["ts"] + timedelta(days=hdays)
                if any(later["ts"] >= target_ts for later in seq[i + 1:]):
                    matched += 1
        coverage[hname] = round(matched / candidates, 4) if candidates else 0.0

    return {
        "trades_analyzed": len(closed),
        "symbols_analyzed": len(symbols),
        "prediction_coverage": coverage,
    }


# --------------------------------------------------------------------------- #
# Aggregate recommendations
# --------------------------------------------------------------------------- #
def compute_recommendations(config: Optional[AnalyticsConfig] = None,
                            trades: Optional[List[dict]] = None,
                            em_rows: Optional[List[dict]] = None,
                            dataset_rows: Optional[List[dict]] = None) -> dict:
    """One-shot bundle of every analysis + headline recommendations."""
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)

    score = analyze_oracle_score_thresholds(config, closed)
    edge = analyze_vol_edge_thresholds(config, closed)
    dte = analyze_dte_buckets(config, closed)
    ivr = analyze_iv_rank_buckets(config, closed)
    strat = analyze_strategy_performance(config, closed)
    coverage = compute_data_coverage(config, closed, em_rows, dataset_rows)

    return {
        "n_trades": len(closed),
        "confidence": compute_confidence(len(closed)),
        "oracle_score": score,
        "volatility_edge": edge,
        "dte": dte,
        "iv_rank": ivr,
        "strategy": strat,
        "coverage": coverage,
        "recommended_min_oracle_score": score["recommended_min_oracle_score"],
        "recommended_min_volatility_edge": edge["recommended_min_volatility_edge"],
        "recommended_dte_range": dte["recommended_dte_range"],
        "recommended_iv_rank_range": ivr["recommended_iv_rank_range"],
        "best_strategy": strat["best_strategy"],
        "worst_strategy": strat["worst_strategy"],
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; synthetic data only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/threshold_st.json",
                          expected_move_file="/nonexistent/threshold_st.csv",
                          training_dataset_file="/nonexistent/threshold_ds.csv")

    # --- empty everything is safe ---
    rec = compute_recommendations(cfg)
    if rec["n_trades"] != 0 or rec["confidence"] != "Low":
        print("FAIL: empty recommendations", rec); ok = False
    if rec["recommended_min_oracle_score"] is not None:
        print("FAIL: empty score reco should be None"); ok = False

    # --- profit factor edge cases ---
    if _profit_factor([]) is not None:
        print("FAIL: pf empty"); ok = False
    if _profit_factor([{"pnl": 10}, {"pnl": 5}]) != float("inf"):
        print("FAIL: pf no-loss should be inf"); ok = False
    if _profit_factor([{"pnl": 30}, {"pnl": -10}]) != 3.0:
        print("FAIL: pf 3.0", _profit_factor([{"pnl": 30}, {"pnl": -10}])); ok = False

    # --- confidence levels ---
    if (compute_confidence(10), compute_confidence(120), compute_confidence(500)) \
            != ("Low", "Medium", "High"):
        print("FAIL: confidence levels"); ok = False

    # --- synthetic trades: high scores/edges win, low ones lose ---
    trades = [
        {"strategy": "bullish_put_credit_spread", "status": "closed",
         "oracle_score": 85, "volatility_edge": 0.035, "pnl": 120.0,
         "dte": 35, "iv_rank": 60},
        {"strategy": "bullish_put_credit_spread", "status": "closed",
         "oracle_score": 82, "volatility_edge": 0.03, "pnl": 90.0,
         "dte": 40, "iv_rank": 55},
        {"strategy": "iron_condor", "status": "closed",
         "oracle_score": 45, "volatility_edge": 0.005, "pnl": -100.0,
         "dte": 10, "iv_rank": 20},
    ]
    score = analyze_oracle_score_thresholds(cfg, trades)
    # The loser scored 45, so any cut >= 50 keeps only the two winners (inf PF).
    # The recommendation is the LEAST restrictive winning cut: >= 50.
    if score["recommended_min_oracle_score"] != 50.0:
        print("FAIL: score reco", score["recommended_min_oracle_score"]); ok = False
    row80 = next(r for r in score["rows"] if r["threshold"] == 80.0)
    if row80["trades"] != 2 or row80["profit_factor"] != float("inf"):
        print("FAIL: score row80", row80); ok = False

    edge = analyze_vol_edge_thresholds(cfg, trades)
    # The loser's edge was 0.5%, so any cut >= 1% keeps only winners; the least
    # restrictive winning cut is >= 1% (0.01).
    if edge["recommended_min_volatility_edge"] != 0.01:
        print("FAIL: edge reco", edge["recommended_min_volatility_edge"]); ok = False

    strat = analyze_strategy_performance(cfg, trades)
    if strat["best_strategy"] != "bullish_put_credit_spread":
        print("FAIL: best strategy", strat["best_strategy"]); ok = False
    if strat["worst_strategy"] != "iron_condor":
        print("FAIL: worst strategy", strat["worst_strategy"]); ok = False
    # canonical strategies all present even with no trades.
    if len(strat["rows"]) < len(CANONICAL_STRATEGIES):
        print("FAIL: canonical strategies missing"); ok = False

    dte = analyze_dte_buckets(cfg, trades)
    if dte["recommended_dte_range"] != "31-60":
        print("FAIL: dte reco", dte["recommended_dte_range"]); ok = False
    ivr = analyze_iv_rank_buckets(cfg, trades)
    if ivr["recommended_iv_rank_range"] != "50-75":
        print("FAIL: iv reco", ivr["recommended_iv_rank_range"]); ok = False

    # --- data coverage from a small EM history ---
    em = [
        {"timestamp": "2025-01-01T00:00:00", "symbol": "SPY", "in_price": "500",
         "expected_move_1d": "5", "expected_move_30d": "27"},
        {"timestamp": "2025-01-02T00:00:00", "symbol": "SPY", "in_price": "503",
         "expected_move_1d": "5", "expected_move_30d": "27"},
    ]
    cov = compute_data_coverage(cfg, trades=trades, em_rows=em)
    # 1d: base row 1 finds a >=1d-later row -> 1 of 2 candidates matched = 0.5.
    if cov["prediction_coverage"]["1d"] != 0.5:
        print("FAIL: coverage 1d", cov["prediction_coverage"]); ok = False
    # 30d: no row 30 days later -> 0 coverage.
    if cov["prediction_coverage"]["30d"] != 0.0:
        print("FAIL: coverage 30d", cov["prediction_coverage"]); ok = False
    if cov["trades_analyzed"] != 3:
        print("FAIL: coverage trades", cov); ok = False

    print("threshold_engine self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
