"""
Phase 8B — Oracle analytics & validation layer (read-only, offline-pure).

This module answers one question: *do Oracle's predictions, volatility edge,
and score actually have predictive value?* It reads only historical artifacts —

    * oracle_training_dataset.csv      (features / predictions / outcomes)
    * expected_move_history.csv        (expected-move predictions + vol edge)
    * spread_paper_trades.json         (CLOSED simulated spread trades)
    * spread_paper_positions.json      (OPEN simulated spread positions)
    * trading_history.json             (existing single-leg trade history)

and computes summary statistics. It is STRICTLY analytics: no trading logic, no
order placement, no spread execution — nothing here can open, modify, or close a
real or paper position. Every reader fails open: a missing, empty, or malformed
file is treated as "no data", and every public function returns a plain dict /
list that is safe to format even when the inputs are empty.

Public API (all accept an optional ``config`` and optional pre-loaded data so
they are trivially unit-testable):

    compute_oracle_stats()
    compute_vol_edge_leaderboard()
    compute_spread_performance()
    compute_learning_performance()
    compute_prediction_accuracy()
"""

import csv
import json
import logging
import os
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from config_loader import ConfigLoader

logger = logging.getLogger(__name__)

HORIZONS = OrderedDict([("1d", 1), ("3d", 3), ("7d", 7), ("30d", 30)])


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class AnalyticsConfig:
    training_dataset_file: str = "oracle_training_dataset.csv"
    expected_move_file: str = "expected_move_history.csv"
    spread_trades_file: str = "spread_paper_trades.json"
    spread_positions_file: str = "spread_paper_positions.json"
    trade_history_file: str = "trading_history.json"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "AnalyticsConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return AnalyticsConfig(
            training_dataset_file=cfg.get_str("ORACLE_DATASET_FILE",
                                              "oracle_training_dataset.csv"),
            expected_move_file=cfg.get_str("EXPECTED_MOVE_HISTORY_FILE",
                                           "expected_move_history.csv"),
            spread_trades_file=cfg.get_str("SPREAD_PAPER_TRADES_FILE",
                                           "spread_paper_trades.json"),
            spread_positions_file=cfg.get_str("SPREAD_PAPER_POSITIONS_FILE",
                                              "spread_paper_positions.json"),
            trade_history_file=cfg.get_str("TRADE_HISTORY_FILE",
                                           "trading_history.json"),
        )


# --------------------------------------------------------------------------- #
# Robust low-level readers / coercion (never raise)
# --------------------------------------------------------------------------- #
def _to_float(value) -> Optional[float]:
    """Coerce to float; None for missing/blank/non-numeric (e.g. 'n/a')."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        s = str(value).strip()
        if not s or s.lower() in ("n/a", "na", "none", "null", "nan"):
            return None
        return float(s)
    except (TypeError, ValueError):
        return None


def _get(row: dict, *keys):
    """First present, non-empty value among ``keys`` (case-insensitive)."""
    if not isinstance(row, dict):
        return None
    lower = None
    for k in keys:
        if k in row and row[k] not in (None, ""):
            return row[k]
    # case-insensitive fallback (handles PnL vs pnl, etc.)
    lower = {str(k).lower(): v for k, v in row.items()}
    for k in keys:
        v = lower.get(str(k).lower())
        if v not in (None, ""):
            return v
    return None


def read_csv_rows(path: str) -> List[dict]:
    """Read a CSV into a list of dicts. Missing/corrupt -> []."""
    if not path or not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", newline="") as fh:
            return [dict(r) for r in csv.DictReader(fh)]
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("analytics CSV read failed (%s): %s", path, exc)
        return []


def read_json(path: str):
    """Read a JSON file. Missing/corrupt -> None."""
    if not path or not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception as exc:
        logger.warning("analytics JSON read failed (%s): %s", path, exc)
        return None


def load_closed_spread_trades(config: AnalyticsConfig,
                              trades: Optional[List[dict]] = None) -> List[dict]:
    """Closed simulated spread trades (status == 'closed' or in trades file)."""
    if trades is not None:
        rows = trades
    else:
        rows = read_json(config.spread_trades_file)
    if not isinstance(rows, list):
        return []
    out = []
    for r in rows:
        if not isinstance(r, dict):
            continue
        status = str(r.get("status", "closed")).lower()
        if status in ("", "closed"):
            out.append(r)
    return out


def load_open_spread_positions(config: AnalyticsConfig,
                               positions: Optional[List[dict]] = None) -> List[dict]:
    if positions is not None:
        rows = positions
    else:
        rows = read_json(config.spread_positions_file)
    if not isinstance(rows, list):
        return []
    return [r for r in rows
            if isinstance(r, dict) and str(r.get("status", "open")).lower() == "open"]


# --------------------------------------------------------------------------- #
# Trade-level accessors (tolerant of label variants)
# --------------------------------------------------------------------------- #
def _trade_pnl(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "pnl", "PnL", "net_pnl", "profit_loss"))


def _trade_pnl_pct(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "pnl_percent", "PnL_percent", "pnl_pct"))


def _trade_oracle(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "oracle_score"))


def _trade_edge(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "volatility_edge", "vol_edge"))


def _trade_dte(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "dte", "DTE"))


def _trade_iv_rank(trade: dict) -> Optional[float]:
    return _to_float(_get(trade, "iv_rank", "IV_rank", "iv_rank_pct"))


def _is_win(trade: dict) -> bool:
    pnl = _trade_pnl(trade)
    if pnl is not None:
        return pnl > 0
    pct = _trade_pnl_pct(trade)
    return pct is not None and pct > 0


def _aggregate(trades: List[dict]) -> Dict[str, float]:
    """{'trades', 'wins', 'win_rate', 'pnl'} over a list of closed trades."""
    n = len(trades)
    wins = sum(1 for t in trades if _is_win(t))
    pnl = sum((_trade_pnl(t) or 0.0) for t in trades)
    return {
        "trades": n,
        "wins": wins,
        "win_rate": (wins / n) if n else 0.0,
        "pnl": round(pnl, 2),
    }


# --------------------------------------------------------------------------- #
# Generic bucketing
# --------------------------------------------------------------------------- #
def _bucket(trades: List[dict], value_fn, buckets) -> "OrderedDict":
    """Group trades into ordered ``buckets`` = [(label, predicate), ...].

    A trade is placed in the first bucket whose predicate(value) is True; trades
    whose value is None are skipped. Returns an OrderedDict label -> aggregate.
    """
    out = OrderedDict((label, []) for label, _ in buckets)
    for t in trades:
        v = value_fn(t)
        if v is None:
            continue
        for label, pred in buckets:
            try:
                if pred(v):
                    out[label].append(t)
                    break
            except Exception:  # pragma: no cover - predicate safety
                continue
    return OrderedDict((label, _aggregate(rows)) for label, rows in out.items())


_ORACLE_BUCKETS = [
    ("80-100", lambda v: 80 <= v <= 100),
    ("60-79", lambda v: 60 <= v < 80),
    ("40-59", lambda v: 40 <= v < 60),
    ("0-39", lambda v: v < 40),
]

# volatility_edge is stored as a FRACTION (0.03 == 3%); buckets are by percent.
_EDGE_BUCKETS = [
    ("3%+", lambda v: v >= 0.03),
    ("2%-3%", lambda v: 0.02 <= v < 0.03),
    ("1%-2%", lambda v: 0.01 <= v < 0.02),
    ("0%-1%", lambda v: 0.0 <= v < 0.01),
    ("<0%", lambda v: v < 0.0),
]

_DTE_BUCKETS = [
    ("0-14", lambda v: 0 <= v <= 14),
    ("15-30", lambda v: 15 <= v <= 30),
    ("31-60", lambda v: 31 <= v <= 60),
    ("60+", lambda v: v > 60),
]

_IV_RANK_BUCKETS = [
    ("0-25", lambda v: 0 <= v < 25),
    ("25-50", lambda v: 25 <= v < 50),
    ("50-75", lambda v: 50 <= v < 75),
    ("75-100", lambda v: 75 <= v <= 100),
]


# --------------------------------------------------------------------------- #
# Timestamp helper for the prediction-accuracy self-join
# --------------------------------------------------------------------------- #
def _parse_ts(value) -> Optional[datetime]:
    if not value:
        return None
    s = str(value).strip().replace("Z", "+00:00")
    try:
        dt = datetime.fromisoformat(s)
    except ValueError:
        # try a couple of common fallbacks
        for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%d"):
            try:
                dt = datetime.strptime(s[:len(fmt) + 2], fmt)
                break
            except ValueError:
                dt = None
        if dt is None:
            return None
    # normalize to naive UTC for safe subtraction
    if dt.tzinfo is not None:
        dt = dt.replace(tzinfo=None)
    return dt


# --------------------------------------------------------------------------- #
# 1) Oracle stats
# --------------------------------------------------------------------------- #
def compute_oracle_stats(config: Optional[AnalyticsConfig] = None,
                         trades: Optional[List[dict]] = None,
                         positions: Optional[List[dict]] = None,
                         em_rows: Optional[List[dict]] = None) -> dict:
    """Headline Oracle stats from the simulated spread book + expected moves."""
    config = config or AnalyticsConfig.from_env()
    closed = load_closed_spread_trades(config, trades)
    opens = load_open_spread_positions(config, positions)

    agg = _aggregate(closed)
    oracle_vals = [v for v in (_trade_oracle(t) for t in closed) if v is not None]
    edge_vals = [v for v in (_trade_edge(t) for t in closed) if v is not None]
    open_pnl = round(sum((_to_float(p.get("pnl")) or 0.0) for p in opens), 2)

    accuracy = compute_prediction_accuracy(config, em_rows=em_rows)
    em_error = OrderedDict(
        (h, accuracy["horizons"][h]["mae_pct"]) for h in HORIZONS)

    return {
        "trades": agg["trades"],
        "wins": agg["wins"],
        "win_rate": agg["win_rate"],
        "total_pnl": agg["pnl"],
        "avg_oracle_score": (sum(oracle_vals) / len(oracle_vals)
                             if oracle_vals else None),
        "avg_volatility_edge": (sum(edge_vals) / len(edge_vals)
                                if edge_vals else None),
        "expected_move_error": em_error,          # fractional MAE per horizon
        "open_pnl": open_pnl,
        "closed_pnl": agg["pnl"],
        "open_positions": len(opens),
    }


# --------------------------------------------------------------------------- #
# 2) Volatility-edge leaderboard
# --------------------------------------------------------------------------- #
def compute_vol_edge_leaderboard(config: Optional[AnalyticsConfig] = None,
                                 em_rows: Optional[List[dict]] = None,
                                 dataset_rows: Optional[List[dict]] = None,
                                 top_n: int = 10) -> List[dict]:
    """Top symbols by current volatility edge (latest prediction per symbol)."""
    config = config or AnalyticsConfig.from_env()
    rows = em_rows if em_rows is not None else read_csv_rows(config.expected_move_file)

    # latest expected-move row per symbol (by timestamp, else file order).
    latest: "OrderedDict[str, dict]" = OrderedDict()
    for r in rows:
        sym = (r.get("symbol") or "").strip().upper()
        if not sym:
            continue
        prev = latest.get(sym)
        if prev is None:
            latest[sym] = r
            continue
        ts_new, ts_old = _parse_ts(r.get("timestamp")), _parse_ts(prev.get("timestamp"))
        if ts_new is not None and ts_old is not None:
            if ts_new >= ts_old:
                latest[sym] = r
        else:
            latest[sym] = r  # no usable ts -> keep most recent in file order

    # oracle score per symbol from the training dataset, when available.
    oracle_by_symbol: Dict[str, float] = {}
    ds = dataset_rows if dataset_rows is not None else read_csv_rows(config.training_dataset_file)
    for r in ds:
        sym = (r.get("symbol") or "").strip().upper()
        score = _to_float(_get(r, "pred_oracle_score", "oracle_score"))
        if sym and score is not None:
            oracle_by_symbol[sym] = score

    board = []
    for sym, r in latest.items():
        edge = _to_float(r.get("volatility_edge"))
        if edge is None:
            continue
        board.append({
            "symbol": sym,
            "volatility_edge": edge,
            "expected_move": _to_float(_get(r, "expected_move_30d", "expected_move_7d",
                                            "expected_move_1d")),
            "market_expected_move": _to_float(r.get("market_expected_move")),
            "oracle_score": oracle_by_symbol.get(sym),
        })
    board.sort(key=lambda e: e["volatility_edge"], reverse=True)
    return board[:top_n] if top_n else board


# --------------------------------------------------------------------------- #
# 3) Spread performance by strategy
# --------------------------------------------------------------------------- #
def compute_spread_performance(config: Optional[AnalyticsConfig] = None,
                               trades: Optional[List[dict]] = None) -> "OrderedDict":
    """Win rate + PnL grouped by strategy name (closed simulated trades)."""
    config = config or AnalyticsConfig.from_env()
    closed = load_closed_spread_trades(config, trades)
    groups: "OrderedDict[str, list]" = OrderedDict()
    for t in closed:
        strat = (t.get("strategy") or "unknown")
        groups.setdefault(strat, []).append(t)
    out = OrderedDict((strat, _aggregate(rows)) for strat, rows in groups.items())
    # stable, useful ordering: most PnL first
    return OrderedDict(sorted(out.items(), key=lambda kv: kv[1]["pnl"], reverse=True))


# --------------------------------------------------------------------------- #
# 4) Learning performance (bucketed)
# --------------------------------------------------------------------------- #
def compute_learning_performance(config: Optional[AnalyticsConfig] = None,
                                 trades: Optional[List[dict]] = None) -> dict:
    """PnL / win-rate bucketed by oracle score, vol edge, DTE and IV rank."""
    config = config or AnalyticsConfig.from_env()
    closed = load_closed_spread_trades(config, trades)
    return {
        "by_oracle_score": _bucket(closed, _trade_oracle, _ORACLE_BUCKETS),
        "by_vol_edge": _bucket(closed, _trade_edge, _EDGE_BUCKETS),
        "by_dte": _bucket(closed, _trade_dte, _DTE_BUCKETS),
        "by_iv_rank": _bucket(closed, _trade_iv_rank, _IV_RANK_BUCKETS),
        "n_trades": len(closed),
    }


# --------------------------------------------------------------------------- #
# 5) Prediction accuracy (expected-move error)
# --------------------------------------------------------------------------- #
def compute_prediction_accuracy(config: Optional[AnalyticsConfig] = None,
                                em_rows: Optional[List[dict]] = None) -> dict:
    """Expected-move accuracy via a self-join on expected_move_history.csv.

    For each prediction row with a usable timestamp + price, we find the earliest
    LATER row for the same symbol whose timestamp is at least ``horizon`` calendar
    days ahead and use its price as the realized price. Then, per horizon::

        realized_move = |later_price - base_price|              (dollars)
        predicted     = expected_move_h  (converted to dollars if fractional)
        error         = |realized_move - predicted|
        mae           = mean(error)            over all matched pairs
        mae_pct       = mean(error / base_price)
        coverage      = fraction with realized_move <= predicted  (calibration)

    Horizons with no matched pairs report n=0 and None metrics. Never raises.
    """
    config = config or AnalyticsConfig.from_env()
    rows = em_rows if em_rows is not None else read_csv_rows(config.expected_move_file)

    # group rows by symbol with parsed ts + price, sorted by time
    by_symbol: Dict[str, List[dict]] = {}
    for r in rows:
        sym = (r.get("symbol") or "").strip().upper()
        ts = _parse_ts(r.get("timestamp"))
        price = _to_float(_get(r, "in_price", "price"))
        if not sym or ts is None or price is None or price <= 0:
            continue
        by_symbol.setdefault(sym, []).append({"ts": ts, "price": price, "row": r})
    for seq in by_symbol.values():
        seq.sort(key=lambda x: x["ts"])

    horizons = OrderedDict()
    for hname, hdays in HORIZONS.items():
        errors_abs: List[float] = []
        errors_pct: List[float] = []
        covered = 0
        matched = 0
        col = "expected_move_" + hname
        for seq in by_symbol.values():
            for i, base in enumerate(seq):
                pred = _to_float(base["row"].get(col))
                if pred is None:
                    continue
                in_dollars = str(base["row"].get("in_dollars", "")).strip().lower() in (
                    "true", "1", "yes")
                predicted = pred if in_dollars else pred * base["price"]
                target_ts = base["ts"] + timedelta(days=hdays)
                realized = None
                for later in seq[i + 1:]:
                    if later["ts"] >= target_ts:
                        realized = abs(later["price"] - base["price"])
                        break
                if realized is None:
                    continue
                matched += 1
                err = abs(realized - predicted)
                errors_abs.append(err)
                if base["price"] > 0:
                    errors_pct.append(err / base["price"])
                if realized <= predicted:
                    covered += 1
        horizons[hname] = {
            "n": matched,
            "mae": round(sum(errors_abs) / len(errors_abs), 4) if errors_abs else None,
            "mae_pct": round(sum(errors_pct) / len(errors_pct), 4) if errors_pct else None,
            "coverage": round(covered / matched, 4) if matched else None,
        }

    return {"horizons": horizons,
            "n_rows": sum(len(v) for v in by_symbol.values())}


# --------------------------------------------------------------------------- #
# Convenience
# --------------------------------------------------------------------------- #
def compute_all(config: Optional[AnalyticsConfig] = None) -> dict:
    config = config or AnalyticsConfig.from_env()
    return {
        "oracle_stats": compute_oracle_stats(config),
        "vol_edge_leaderboard": compute_vol_edge_leaderboard(config),
        "spread_performance": compute_spread_performance(config),
        "learning_performance": compute_learning_performance(config),
        "prediction_accuracy": compute_prediction_accuracy(config),
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses temp files + synthetic data)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    d = tempfile.mkdtemp()
    trades_path = os.path.join(d, "trades.json")
    pos_path = os.path.join(d, "pos.json")
    em_path = os.path.join(d, "em.csv")
    ds_path = os.path.join(d, "ds.csv")
    cfg = AnalyticsConfig(spread_trades_file=trades_path,
                          spread_positions_file=pos_path,
                          expected_move_file=em_path,
                          training_dataset_file=ds_path)

    # --- empty / missing everything is safe ---
    s = compute_oracle_stats(cfg)
    if s["trades"] != 0 or s["total_pnl"] != 0.0:
        print("FAIL: empty oracle_stats", s); ok = False
    if compute_vol_edge_leaderboard(cfg) != []:
        print("FAIL: empty leaderboard"); ok = False
    if compute_spread_performance(cfg) != OrderedDict():
        print("FAIL: empty spread_perf"); ok = False

    # --- malformed files are treated as empty ---
    with open(trades_path, "w", encoding="utf-8") as fh:
        fh.write("{not valid json")
    with open(em_path, "w", encoding="utf-8") as fh:
        fh.write("garbage,,,\n\x00\x01")
    if compute_oracle_stats(cfg)["trades"] != 0:
        print("FAIL: malformed not treated as empty"); ok = False

    # --- synthetic closed trades ---
    closed = [
        {"symbol": "SPY", "strategy": "bull_put_credit_spread", "status": "closed",
         "oracle_score": 85, "volatility_edge": 0.035, "pnl": 120.0,
         "pnl_percent": 25, "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "bull_put_credit_spread", "status": "closed",
         "oracle_score": 72, "volatility_edge": 0.015, "pnl": -80.0,
         "pnl_percent": -20, "dte": 10, "iv_rank": 30},
        {"symbol": "META", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 55, "volatility_edge": -0.01, "pnl": 40.0,
         "pnl_percent": 8, "dte": 70, "iv_rank": 80},
    ]
    stats = compute_oracle_stats(cfg, trades=closed)
    if stats["trades"] != 3 or round(stats["total_pnl"], 2) != 80.0:
        print("FAIL: stats totals", stats); ok = False
    if abs(stats["win_rate"] - (2 / 3)) > 1e-6:
        print("FAIL: win_rate", stats["win_rate"]); ok = False

    perf = compute_spread_performance(cfg, trades=closed)
    if perf["bull_put_credit_spread"]["trades"] != 2:
        print("FAIL: spread_perf grouping", perf); ok = False
    if round(perf["bull_put_credit_spread"]["pnl"], 2) != 40.0:
        print("FAIL: spread_perf pnl", perf); ok = False

    learn = compute_learning_performance(cfg, trades=closed)
    if learn["by_oracle_score"]["80-100"]["trades"] != 1:
        print("FAIL: oracle bucket", learn["by_oracle_score"]); ok = False
    if learn["by_dte"]["0-14"]["trades"] != 1:
        print("FAIL: dte bucket", learn["by_dte"]); ok = False
    if learn["by_iv_rank"]["75-100"]["trades"] != 1:
        print("FAIL: iv bucket", learn["by_iv_rank"]); ok = False
    if learn["by_vol_edge"]["3%+"]["trades"] != 1:
        print("FAIL: edge bucket", learn["by_vol_edge"]); ok = False

    # --- leaderboard + prediction accuracy from an EM history ---
    with open(em_path, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "symbol", "in_price", "expected_move_1d",
                    "expected_move_3d", "expected_move_7d", "expected_move_30d",
                    "market_expected_move", "volatility_edge", "in_dollars"])
        # SPY: predict $5 1d move; price goes 500 -> 503 next day (err |3-5|=2)
        w.writerow(["2025-01-01T16:00:00+00:00", "SPY", "500", "5", "9", "13",
                    "27", "30", "0.12", "True"])
        w.writerow(["2025-01-02T16:00:00+00:00", "SPY", "503", "5", "9", "13",
                    "27", "30", "0.10", "True"])
        w.writerow(["2025-01-01T16:00:00+00:00", "NVDA", "100", "2", "3", "5",
                    "10", "11", "0.30", "True"])

    board = compute_vol_edge_leaderboard(cfg)
    if not board or board[0]["symbol"] != "NVDA":
        print("FAIL: leaderboard order", board); ok = False

    acc = compute_prediction_accuracy(cfg)
    if acc["horizons"]["1d"]["n"] != 1:
        print("FAIL: accuracy match count", acc["horizons"]["1d"]); ok = False
    if acc["horizons"]["1d"]["mae"] is None or abs(acc["horizons"]["1d"]["mae"] - 2.0) > 1e-6:
        print("FAIL: accuracy mae", acc["horizons"]["1d"]); ok = False

    print("oracle_analytics self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
