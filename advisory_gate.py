"""
Phase 9A — Advisory trade gate (advisory, read-only, offline-pure).

Given a *proposed* setup (oracle_score, volatility_edge, DTE, IV rank and
strategy) this module compares it against the data-driven thresholds produced
by :mod:`threshold_engine` and emits an ADVISORY recommendation —

    STRONG_ACCEPT  | ACCEPT | NEUTRAL | WEAK_SETUP | REJECT_CANDIDATE

— together with the historical win-rate / profit-factor of comparable closed
trades and a per-check breakdown.

It is STRICTLY advisory. It contains no order placement, no spread execution,
no live/paper trading and no gating: nothing here can open, modify, block or
close any real or paper position, and it never changes strategy selection. It
only describes what the historical data says about a proposed setup. Every
reader fails open (missing / empty / malformed inputs -> "no data") and every
public function returns a plain dict / string that is safe to format even when
the inputs are empty.

It builds on the Phase 8B analytics layer (:mod:`oracle_analytics`) and the
Phase 8C recommendation engine (:mod:`threshold_engine`) so the three layers
stay consistent.

Public API (all accept an optional ``config`` / pre-loaded data so they are
trivially unit-testable):

    evaluate_setup()              -> recommendation dict
    log_advisory_gate()           -> the [ADVISORY_GATE] log line (advisory)
    gather_symbol_features()      -> latest known features for a symbol
    advisory_check_for_symbol()   -> (features, result)
    generate_advisory_check_text()-> Telegram-ready string
"""

import logging
from datetime import datetime
from typing import Optional, List

import oracle_analytics as oa
import threshold_engine as te
from oracle_analytics import AnalyticsConfig

logger = logging.getLogger(__name__)

# Recommendation labels (Req 1 / Req 2).
STRONG_ACCEPT = "STRONG_ACCEPT"
ACCEPT = "ACCEPT"
NEUTRAL = "NEUTRAL"
WEAK_SETUP = "WEAK_SETUP"
REJECT_CANDIDATE = "REJECT_CANDIDATE"

ADVISORY_FOOTER = "Advisory only — nothing is blocked."

# Fallbacks used only when the data-driven recommendation is undefined (no data).
DEFAULT_MIN_ORACLE_SCORE = 60.0

# A setup is flagged "historically poor" (-> REJECT_CANDIDATE) when comparable
# trades have a profit factor below this with at least this many samples.
POOR_PF = 1.0
POOR_MIN_TRADES = 5

# Reuse the analytics bucket predicates so DTE / IV-rank membership checks match
# exactly what threshold_engine recommends.
_DTE_PRED = {label: pred for label, pred in oa._DTE_BUCKETS}
_IV_PRED = {label: pred for label, pred in oa._IV_RANK_BUCKETS}


# --------------------------------------------------------------------------- #
# Individual threshold checks (each returns True / False)
# --------------------------------------------------------------------------- #
def _check_min(value, threshold) -> bool:
    """True when ``value`` meets/exceeds ``threshold``.

    An undefined threshold (no data) can't be failed -> True. A missing value
    cannot satisfy a real threshold -> False.
    """
    if threshold is None:
        return True
    v = oa._to_float(value)
    if v is None:
        return False
    return v >= float(threshold)


def _check_bucket(value, label, pred_map) -> bool:
    """True when ``value`` falls in the recommended bucket ``label``.

    No recommended bucket (no data) -> True; missing / out-of-bucket value ->
    False.
    """
    if not label:
        return True
    pred = pred_map.get(label)
    if pred is None:
        return True
    v = oa._to_float(value)
    if v is None:
        return False
    try:
        return bool(pred(v))
    except Exception:  # pragma: no cover - predicate safety
        return False


def _check_strategy(strategy, rec) -> bool:
    """True unless ``strategy`` is the historically worst strategy."""
    if not strategy:
        return True
    worst = rec.get("worst_strategy")
    return not (worst and strategy == worst)


# --------------------------------------------------------------------------- #
# Historical performance of comparable trades
# --------------------------------------------------------------------------- #
def _historical(closed: List[dict], strategy) -> dict:
    """Win-rate / profit-factor of closed trades for ``strategy`` (or all)."""
    if strategy:
        subset = [t for t in closed if (t.get("strategy") or "") == strategy]
    else:
        subset = list(closed)
    agg = oa._aggregate(subset)
    return {
        "trades": agg["trades"],
        "win_rate": agg["win_rate"],
        "profit_factor": te._profit_factor(subset),
    }


# --------------------------------------------------------------------------- #
# Recommendation classification (Req 2)
# --------------------------------------------------------------------------- #
def _classify(passed: int, strategy, rec: dict, hist: dict, n_trades: int) -> str:
    """Map passed-check count + history to one of the five labels."""
    if n_trades == 0:
        # Nothing to learn from yet -> stay neutral (advisory).
        return NEUTRAL

    pf = hist["profit_factor"]
    poor_pf = (hist["trades"] >= POOR_MIN_TRADES and pf is not None
               and pf != float("inf") and pf < POOR_PF)
    worst = rec.get("worst_strategy")
    worst_strat = (strategy and worst and strategy == worst
                   and hist["trades"] >= POOR_MIN_TRADES)
    if poor_pf or worst_strat:
        return REJECT_CANDIDATE

    if passed >= 5:
        return STRONG_ACCEPT
    if passed == 4:
        return ACCEPT
    if passed == 3:
        return NEUTRAL
    if passed == 2:
        return WEAK_SETUP
    return REJECT_CANDIDATE


def _confidence(n_trades: int) -> str:
    """LOW / MEDIUM / HIGH from sample size (reuses threshold_engine)."""
    return te.compute_confidence(n_trades).upper()


# --------------------------------------------------------------------------- #
# Core evaluation (Req 1 / Req 2)
# --------------------------------------------------------------------------- #
def evaluate_setup(oracle_score=None, volatility_edge=None, dte=None,
                   iv_rank=None, strategy=None, *,
                   config: Optional[AnalyticsConfig] = None,
                   recommendations: Optional[dict] = None,
                   trades: Optional[List[dict]] = None) -> dict:
    """Advisory verdict for a proposed setup.

    Returns a dict with ``recommendation``, ``confidence``, per-check booleans,
    ``historical_win_rate`` and ``historical_profit_factor`` (plus a few extras
    that are handy for logging / formatting). Never raises.
    """
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)
    rec = recommendations if recommendations is not None else \
        te.compute_recommendations(config, closed)

    n_trades = rec.get("n_trades", len(closed))

    checks = {
        "oracle_score": _check_min(oracle_score,
                                   rec.get("recommended_min_oracle_score")),
        "vol_edge": _check_min(volatility_edge,
                               rec.get("recommended_min_volatility_edge")),
        "dte": _check_bucket(dte, rec.get("recommended_dte_range"), _DTE_PRED),
        "iv_rank": _check_bucket(iv_rank, rec.get("recommended_iv_rank_range"),
                                 _IV_PRED),
        "strategy": _check_strategy(strategy, rec),
    }
    passed = sum(1 for v in checks.values() if v)

    hist = _historical(closed, strategy)
    recommendation = _classify(passed, strategy, rec, hist, n_trades)

    return {
        "recommendation": recommendation,
        "confidence": _confidence(n_trades),
        "checks": checks,
        "historical_win_rate": hist["win_rate"],
        "historical_profit_factor": hist["profit_factor"],
        "passed_checks": passed,
        "n_trades": n_trades,
        "strategy": strategy,
        "oracle_score": oracle_score,
        "volatility_edge": volatility_edge,
        "dte": dte,
        "iv_rank": iv_rank,
    }


# --------------------------------------------------------------------------- #
# Req 3 — structured log line (advisory; does not gate anything)
# --------------------------------------------------------------------------- #
def log_advisory_gate(symbol, strategy, oracle_score, volatility_edge,
                      result: dict, logger_=None) -> str:
    """Emit (and return) the ``[ADVISORY_GATE]`` line for a proposed trade.

    Advisory only — calling this never blocks, gates or alters a trade.
    """
    line = (
        "[ADVISORY_GATE] "
        f"symbol={symbol} "
        f"strategy={strategy} "
        f"oracle_score={oracle_score} "
        f"volatility_edge={volatility_edge} "
        f"recommendation={result.get('recommendation')} "
        f"historical_win_rate={result.get('historical_win_rate')} "
        f"historical_profit_factor={_pf_str(result.get('historical_profit_factor'))}"
    )
    (logger_ or logger).info(line)
    return line


# --------------------------------------------------------------------------- #
# Req 4 — symbol feature lookup + Telegram formatting
# --------------------------------------------------------------------------- #
def _latest_trade(trades: List[dict]) -> dict:
    def key(t):
        ts = oa._parse_ts(t.get("timestamp") or t.get("closed_at") or t.get("date"))
        return ts or datetime.min
    return max(trades, key=key)


def gather_symbol_features(symbol, config: Optional[AnalyticsConfig] = None,
                           em_rows: Optional[List[dict]] = None,
                           dataset_rows: Optional[List[dict]] = None,
                           trades: Optional[List[dict]] = None,
                           positions: Optional[List[dict]] = None) -> dict:
    """Best-effort latest known features for ``symbol`` from analytics files.

    Pulls volatility_edge / oracle_score from the volatility-edge leaderboard
    (latest expected-move row per symbol) and strategy / DTE / IV-rank from the
    most recent same-symbol trade (open or closed). Missing values stay ``None``.
    """
    config = config or AnalyticsConfig.from_env()
    sym = str(symbol or "").strip().upper()
    features = {"symbol": sym, "oracle_score": None, "volatility_edge": None,
                "dte": None, "iv_rank": None, "strategy": None}
    if not sym:
        return features

    board = oa.compute_vol_edge_leaderboard(config, em_rows=em_rows,
                                            dataset_rows=dataset_rows, top_n=0)
    for row in board:
        if row.get("symbol") == sym:
            features["volatility_edge"] = row.get("volatility_edge")
            features["oracle_score"] = row.get("oracle_score")
            break

    recs: List[dict] = []
    recs.extend(oa.load_open_spread_positions(config, positions))
    recs.extend(oa.load_closed_spread_trades(config, trades))
    sym_trades = [t for t in recs
                  if str(t.get("symbol") or "").strip().upper() == sym]
    if sym_trades:
        latest = _latest_trade(sym_trades)
        features["strategy"] = latest.get("strategy") or features["strategy"]
        if features["dte"] is None:
            features["dte"] = oa._trade_dte(latest)
        if features["iv_rank"] is None:
            features["iv_rank"] = oa._trade_iv_rank(latest)
        if features["oracle_score"] is None:
            features["oracle_score"] = oa._trade_oracle(latest)
        if features["volatility_edge"] is None:
            features["volatility_edge"] = oa._trade_edge(latest)
    return features


def advisory_check_for_symbol(symbol,
                              config: Optional[AnalyticsConfig] = None):
    """(features, result) for ``symbol`` — gather features then evaluate."""
    config = config or AnalyticsConfig.from_env()
    features = gather_symbol_features(symbol, config)
    result = evaluate_setup(
        oracle_score=features["oracle_score"],
        volatility_edge=features["volatility_edge"],
        dte=features["dte"], iv_rank=features["iv_rank"],
        strategy=features["strategy"], config=config)
    return features, result


def _pf_str(pf) -> str:
    if pf is None:
        return "n/a"
    if pf == float("inf"):
        return "∞"
    return "%.2f" % pf


def _pct_str(value) -> str:
    v = oa._to_float(value)
    return "n/a" if v is None else "%.1f%%" % (v * 100.0)


def _num_str(value) -> str:
    v = oa._to_float(value)
    return "n/a" if v is None else ("%g" % v)


def _check_line(label: str, ok: bool) -> str:
    return f"{'✅' if ok else '❌'} {label}"


def format_advisory_check(symbol, features: dict, result: dict) -> str:
    """Telegram-ready advisory summary for one symbol."""
    checks = result["checks"]
    lines = [
        f"*Advisory Check — {symbol}*",
        "",
        f"Recommendation: *{result['recommendation']}*",
        f"Confidence: *{result['confidence']}*",
        "",
        "*Threshold checks:*",
        _check_line(f"Oracle score ({_num_str(features.get('oracle_score'))})",
                    checks["oracle_score"]),
        _check_line(f"Vol edge ({_pct_str(features.get('volatility_edge'))})",
                    checks["vol_edge"]),
        _check_line(f"DTE ({_num_str(features.get('dte'))})", checks["dte"]),
        _check_line(f"IV rank ({_num_str(features.get('iv_rank'))})",
                    checks["iv_rank"]),
        _check_line(f"Strategy ({features.get('strategy') or 'n/a'})",
                    checks["strategy"]),
        "",
        f"Historical win rate: *{result['historical_win_rate'] * 100:.1f}%*",
        f"Historical profit factor: *{_pf_str(result['historical_profit_factor'])}*",
        "",
        f"_({ADVISORY_FOOTER})_",
    ]
    return "\n".join(lines)


def generate_advisory_check_text(symbol,
                                 config: Optional[AnalyticsConfig] = None) -> str:
    """Top-level entry for the ADVISORY_CHECK SYMBOL Telegram command."""
    sym = str(symbol or "").strip().upper()
    if not sym:
        return "Usage: `ADVISORY_CHECK SYMBOL`"
    features, result = advisory_check_for_symbol(sym, config)
    return format_advisory_check(sym, features, result)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; synthetic data only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/ag_st.json",
                          spread_positions_file="/nonexistent/ag_pos.json",
                          expected_move_file="/nonexistent/ag_em.csv",
                          training_dataset_file="/nonexistent/ag_ds.csv")

    # --- empty data -> NEUTRAL / LOW, never raises ---
    res = evaluate_setup(oracle_score=85, volatility_edge=0.04, dte=40,
                         iv_rank=60, strategy="bullish_put_credit_spread",
                         config=cfg)
    if res["recommendation"] != NEUTRAL or res["confidence"] != "LOW":
        print("FAIL: empty -> neutral/low", res); ok = False
    if set(res["checks"]) != {"oracle_score", "vol_edge", "dte", "iv_rank", "strategy"}:
        print("FAIL: checks keys", res["checks"]); ok = False

    # --- synthetic book: high score/edge winners, low loser ---
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
    rec = te.compute_recommendations(cfg, trades)

    # a setup matching the winners should pass most/all checks.
    strong = evaluate_setup(oracle_score=88, volatility_edge=0.04, dte=38,
                            iv_rank=60, strategy="bullish_put_credit_spread",
                            config=cfg, recommendations=rec, trades=trades)
    if strong["recommendation"] not in (STRONG_ACCEPT, ACCEPT):
        print("FAIL: strong setup", strong); ok = False
    if strong["historical_profit_factor"] != float("inf"):
        print("FAIL: winners PF inf", strong["historical_profit_factor"]); ok = False
    if abs(strong["historical_win_rate"] - 1.0) > 1e-9:
        print("FAIL: winners win rate", strong["historical_win_rate"]); ok = False

    # a weak setup failing the cuts should be downgraded.
    weak = evaluate_setup(oracle_score=30, volatility_edge=0.0, dte=5,
                          iv_rank=10, strategy="iron_condor",
                          config=cfg, recommendations=rec, trades=trades)
    if weak["recommendation"] not in (WEAK_SETUP, REJECT_CANDIDATE, NEUTRAL):
        print("FAIL: weak setup", weak); ok = False
    if weak["checks"]["oracle_score"] or weak["checks"]["vol_edge"]:
        print("FAIL: weak checks should fail", weak["checks"]); ok = False

    # --- check helpers ---
    if not _check_min(5, None):  # unknown threshold can't fail
        print("FAIL: _check_min None thr"); ok = False
    if _check_min(None, 60):  # missing value fails a real threshold
        print("FAIL: _check_min missing value"); ok = False
    if not _check_bucket(40, "31-60", _DTE_PRED):
        print("FAIL: _check_bucket dte hit"); ok = False
    if _check_bucket(5, "31-60", _DTE_PRED):
        print("FAIL: _check_bucket dte miss"); ok = False

    # --- log line shape ---
    line = log_advisory_gate("SPY", "bullish_put_credit_spread", 88, 0.04, strong)
    if not line.startswith("[ADVISORY_GATE] ") or "recommendation=" not in line:
        print("FAIL: log line", line); ok = False

    # --- formatting never raises ---
    txt = format_advisory_check("SPY", {"oracle_score": 88, "volatility_edge": 0.04,
                                        "dte": 38, "iv_rank": 60,
                                        "strategy": "bullish_put_credit_spread"},
                                strong)
    if "Advisory Check — SPY" not in txt or ADVISORY_FOOTER not in txt:
        print("FAIL: format", txt); ok = False

    # --- empty symbol usage hint ---
    if "Usage" not in generate_advisory_check_text("", config=cfg):
        print("FAIL: empty symbol usage"); ok = False

    print("advisory_gate self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
