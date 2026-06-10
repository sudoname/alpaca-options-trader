"""
Phase 10G-A — Daily Vol Forecast Scorecard (analytics only).

Answers the most fundamental question underneath every Oracle signal:

    *Does Oracle's forecast_vol predict realized volatility better than the
    market's implied volatility does?*

Every other belief (volatility edge, PoP, EV) is derived from forecast_vol,
so if the forecast does not beat IV at predicting realized vol, the whole
stack has no edge over the null hypothesis of "just sell premium".

For every snapshot in expected_move_history.csv / oracle_training_dataset.csv
that carries both forecast_vol and implied_vol, and for every horizon
(1d/3d/7d/30d) where later price snapshots exist, we build a comparison row:

    symbol, date, horizon, forecast_vol, market_iv, realized_vol,
    forecast_error, iv_error, plus absolute and squared errors.

Aggregates: MAE and RMSE for both predictors, the relative improvement of
the forecast over IV, and Mincer-Zarnowitz regressions
(realized = alpha + beta * prediction; unbiased means alpha=0, beta=1).

Verdict: FORECAST_BEATS_IV / IV_BEATS_FORECAST / INCONCLUSIVE — the forecast
must win on BOTH MAE and RMSE to claim a win (and vice versa for IV).
Confidence is row-based: Low <100, Medium 100-1000, High >1000.

STRICTLY analytics: never opens, closes, sizes, blocks or alters any real or
paper trade; never touches the network. All readers fail open.
"""

import math
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Sequence, Tuple

import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER
from oracle_analytics import AnalyticsConfig, HORIZONS

# Scorecard verdicts.
VERDICT_FORECAST_BEATS_IV = "FORECAST_BEATS_IV"
VERDICT_IV_BEATS_FORECAST = "IV_BEATS_FORECAST"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

# Realized-vol estimation guards: at least this many log returns inside the
# horizon window, and the window must cover at least this fraction of the
# horizon (otherwise "realized vol" over 30d would be measured on 1 day).
MIN_RETURNS = 2
MIN_COVERAGE = 0.6

YEAR_DAYS = 365.25
YEAR_SECONDS = YEAR_DAYS * 86400.0

SCORECARD_QUESTION = "Does forecast_vol beat market IV at predicting realized vol?"


# ---------------------------------------------------------------------------
# Snapshot extraction (tolerant of both CSV shapes)
# ---------------------------------------------------------------------------
def _snap(row: dict) -> Optional[dict]:
    """Normalize an expected-move-history or training-dataset row to
    {ts, symbol, price, forecast, iv}; None when timestamp/price missing."""
    if not isinstance(row, dict):
        return None
    ts = oa._parse_ts(oa._get(row, "timestamp", "time", "date"))
    symbol = oa._get(row, "symbol", "ticker")
    price = oa._to_float(oa._get(row, "in_price", "feat_price", "price",
                                 "feat_in_price", "underlying_price"))
    if ts is None or not symbol or price is None or price <= 0:
        return None
    forecast = oa._to_float(oa._get(row, "forecast_vol", "pred_forecast_vol",
                                    "feat_forecast_vol"))
    iv = oa._to_float(oa._get(row, "implied_vol", "pred_implied_vol",
                              "feat_implied_vol", "market_iv"))
    if iv is None:
        vix = oa._to_float(oa._get(row, "in_vix", "feat_vix", "vix"))
        iv = vix / 100.0 if vix is not None and vix > 0 else None
    return {"ts": ts, "symbol": str(symbol).upper(), "price": price,
            "forecast": forecast, "iv": iv}


def _load_snaps(config: Optional[AnalyticsConfig],
                em_rows: Optional[List[dict]],
                dataset_rows: Optional[List[dict]]) -> List[dict]:
    cfg = config or AnalyticsConfig.from_env()
    if em_rows is None:
        em_rows = oa.read_csv_rows(cfg.expected_move_file)
    if dataset_rows is None:
        dataset_rows = oa.read_csv_rows(cfg.training_dataset_file)
    snaps = []
    for row in list(em_rows or []) + list(dataset_rows or []):
        s = _snap(row)
        if s is not None:
            snaps.append(s)
    return snaps


# ---------------------------------------------------------------------------
# Realized volatility over irregularly spaced snapshots
# ---------------------------------------------------------------------------
def realized_vol(points: Sequence[Tuple[datetime, float]],
                 start_ts: datetime, start_price: float,
                 horizon_days: float) -> Optional[float]:
    """Annualized realized vol from price snapshots within the horizon.

    realized = sqrt( sum(ln(p_i / p_{i-1})^2) / T_years ) over the points in
    (start_ts, start_ts + horizon], anchored at the start snapshot. Requires
    MIN_RETURNS returns and MIN_COVERAGE of the horizon to be spanned.
    """
    if start_price is None or start_price <= 0:
        return None
    end_ts = start_ts + timedelta(days=horizon_days)
    window: List[Tuple[datetime, float]] = []
    last_ts = start_ts
    for ts, price in sorted(points or []):
        if ts <= start_ts or ts > end_ts or price is None or price <= 0:
            continue
        if ts == last_ts:  # duplicate snapshot timestamp
            continue
        window.append((ts, price))
        last_ts = ts
    if len(window) < MIN_RETURNS:
        return None
    elapsed = (window[-1][0] - start_ts).total_seconds()
    if elapsed < MIN_COVERAGE * horizon_days * 86400.0:
        return None
    sum_sq = 0.0
    prev = start_price
    for _, price in window:
        sum_sq += math.log(price / prev) ** 2
        prev = price
    t_years = elapsed / YEAR_SECONDS
    if t_years <= 0:
        return None
    return math.sqrt(sum_sq / t_years)


# ---------------------------------------------------------------------------
# Comparison-row construction
# ---------------------------------------------------------------------------
def build_rows(config: Optional[AnalyticsConfig] = None,
               em_rows: Optional[List[dict]] = None,
               dataset_rows: Optional[List[dict]] = None) -> List[dict]:
    """One row per (symbol, snapshot, horizon) with resolvable realized vol."""
    snaps = _load_snaps(config, em_rows, dataset_rows)
    by_symbol: Dict[str, List[Tuple[datetime, float]]] = {}
    for s in snaps:
        by_symbol.setdefault(s["symbol"], []).append((s["ts"], s["price"]))
    for sym in by_symbol:
        by_symbol[sym].sort()

    rows = []
    for s in snaps:
        if s["forecast"] is None or s["iv"] is None:
            continue
        points = by_symbol.get(s["symbol"], [])
        for horizon, days in HORIZONS.items():
            realized = realized_vol(points, s["ts"], s["price"], days)
            if realized is None:
                continue
            f_err = s["forecast"] - realized
            iv_err = s["iv"] - realized
            rows.append({
                "symbol": s["symbol"],
                "date": s["ts"].strftime("%Y-%m-%d"),
                "horizon": horizon,
                "forecast_vol": round(s["forecast"], 6),
                "market_iv": round(s["iv"], 6),
                "realized_vol": round(realized, 6),
                "forecast_error": round(f_err, 6),
                "iv_error": round(iv_err, 6),
                "abs_forecast_error": round(abs(f_err), 6),
                "abs_iv_error": round(abs(iv_err), 6),
                "sq_forecast_error": round(f_err ** 2, 8),
                "sq_iv_error": round(iv_err ** 2, 8),
            })
    return rows


# ---------------------------------------------------------------------------
# Regression + aggregate metrics
# ---------------------------------------------------------------------------
def linear_regression(xs: Sequence[float], ys: Sequence[float]) -> dict:
    """OLS y = alpha + beta*x -> {alpha, beta, r_squared, n}. Never raises;
    returns None fields when the fit is undefined (n<2 or zero x-variance)."""
    pairs = [(oa._to_float(x), oa._to_float(y)) for x, y in zip(xs, ys)]
    pairs = [(x, y) for x, y in pairs if x is not None and y is not None]
    n = len(pairs)
    out = {"alpha": None, "beta": None, "r_squared": None, "n": n}
    if n < 2:
        return out
    mean_x = sum(x for x, _ in pairs) / n
    mean_y = sum(y for _, y in pairs) / n
    sxx = sum((x - mean_x) ** 2 for x, _ in pairs)
    sxy = sum((x - mean_x) * (y - mean_y) for x, y in pairs)
    sst = sum((y - mean_y) ** 2 for _, y in pairs)
    if sxx <= 0:
        return out
    beta = sxy / sxx
    alpha = mean_y - beta * mean_x
    if sst > 0:
        ssr = sum((y - (alpha + beta * x)) ** 2 for x, y in pairs)
        r_squared = max(0.0, 1.0 - ssr / sst)
    else:
        r_squared = None
    out.update({"alpha": round(alpha, 6), "beta": round(beta, 6),
                "r_squared": round(r_squared, 4) if r_squared is not None
                else None})
    return out


def _mae(rows: List[dict], key: str) -> Optional[float]:
    vals = [oa._to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 6) if vals else None


def _rmse(rows: List[dict], key: str) -> Optional[float]:
    vals = [oa._to_float(r.get(key)) for r in rows]
    vals = [v for v in vals if v is not None]
    return round(math.sqrt(sum(vals) / len(vals)), 6) if vals else None


def scorecard_confidence(n_rows: int) -> str:
    """'Low' (<100), 'Medium' (100-1000), 'High' (>1000) comparison rows."""
    n = n_rows or 0
    if n < 100:
        return "Low"
    if n <= 1000:
        return "Medium"
    return "High"


def _improvement(iv_metric, forecast_metric) -> Optional[float]:
    if iv_metric is None or forecast_metric is None or iv_metric == 0:
        return None
    return round((iv_metric - forecast_metric) / iv_metric, 4)


def compute_scorecard(config: Optional[AnalyticsConfig] = None,
                      em_rows: Optional[List[dict]] = None,
                      dataset_rows: Optional[List[dict]] = None,
                      rows: Optional[List[dict]] = None) -> dict:
    """Full vol-forecast scorecard. Never raises; empty data fails open."""
    if rows is None:
        rows = build_rows(config=config, em_rows=em_rows,
                          dataset_rows=dataset_rows)
    forecast_mae = _mae(rows, "abs_forecast_error")
    iv_mae = _mae(rows, "abs_iv_error")
    forecast_rmse = _rmse(rows, "sq_forecast_error")
    iv_rmse = _rmse(rows, "sq_iv_error")

    verdict = VERDICT_INCONCLUSIVE
    if rows and None not in (forecast_mae, iv_mae, forecast_rmse, iv_rmse):
        if forecast_mae < iv_mae and forecast_rmse < iv_rmse:
            verdict = VERDICT_FORECAST_BEATS_IV
        elif forecast_mae > iv_mae and forecast_rmse > iv_rmse:
            verdict = VERDICT_IV_BEATS_FORECAST

    realized = [r.get("realized_vol") for r in rows]
    by_horizon = {}
    for horizon in HORIZONS:
        sub = [r for r in rows if r.get("horizon") == horizon]
        by_horizon[horizon] = {
            "rows": len(sub),
            "forecast_mae": _mae(sub, "abs_forecast_error"),
            "iv_mae": _mae(sub, "abs_iv_error"),
        }

    return {
        "question": SCORECARD_QUESTION,
        "rows": len(rows),
        "confidence": scorecard_confidence(len(rows)),
        "forecast_mae": forecast_mae,
        "iv_mae": iv_mae,
        "forecast_rmse": forecast_rmse,
        "iv_rmse": iv_rmse,
        "forecast_vs_iv_improvement": _improvement(iv_mae, forecast_mae),
        "forecast_vs_iv_rmse_improvement": _improvement(iv_rmse,
                                                        forecast_rmse),
        "mz_forecast": linear_regression(
            [r.get("forecast_vol") for r in rows], realized),
        "mz_iv": linear_regression(
            [r.get("market_iv") for r in rows], realized),
        "by_horizon": by_horizon,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------
def _pct(value) -> str:
    return f"{value * 100:.2f}%" if value is not None else "n/a"


def _num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def _mz_line(title: str, mz: dict) -> str:
    if mz.get("beta") is None:
        return f"*{title}:* insufficient data (n=`{mz.get('n', 0)}`)"
    return (f"*{title}:* alpha `{_num(mz['alpha'])}`, beta `{_num(mz['beta'])}`,"
            f" R² `{_num(mz['r_squared'], 4)}`, n=`{mz['n']}`")


def format_scorecard(card: dict) -> str:
    """Telegram-ready VOL_FORECAST_SCORECARD. Pure formatting."""
    header = "📏 *Vol Forecast Scorecard* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if card.get("rows", 0) == 0:
        return "\n".join([
            header, "",
            f"_{SCORECARD_QUESTION}_",
            "",
            "No resolvable forecast/realized-vol pairs yet.",
            "They accrue as expected-move snapshots age past their horizons.",
            f"*Verdict:* {VERDICT_INCONCLUSIVE}",
            "", footer,
        ])
    lines = [
        header, "",
        f"_{SCORECARD_QUESTION}_",
        "",
        "*Forecast vs realized vol:*",
        f"MAE — forecast `{_num(card['forecast_mae'])}` vs "
        f"IV `{_num(card['iv_mae'])}`",
        f"RMSE — forecast `{_num(card['forecast_rmse'])}` vs "
        f"IV `{_num(card['iv_rmse'])}`",
        f"Forecast improvement over IV (MAE): "
        f"`{_pct(card['forecast_vs_iv_improvement'])}`",
        "",
        _mz_line("MZ forecast_vol", card["mz_forecast"]),
        _mz_line("MZ market IV", card["mz_iv"]),
        "",
        "*By horizon (MAE forecast | IV):*",
    ]
    for horizon, stats in card.get("by_horizon", {}).items():
        if stats.get("rows", 0) > 0:
            lines.append(f"`{horizon}`: `{_num(stats['forecast_mae'])}` | "
                         f"`{_num(stats['iv_mae'])}` "
                         f"(`{stats['rows']}` rows)")
    lines += [
        "",
        f"*Verdict:* {card['verdict']}",
        f"Rows: `{card['rows']}` · Confidence: *{card['confidence']}*",
        "", footer,
    ]
    return "\n".join(lines)


def generate_vol_forecast_scorecard_text(
        config: Optional[AnalyticsConfig] = None) -> str:
    """Top-level entry for the VOL_FORECAST_SCORECARD Telegram command."""
    return format_scorecard(compute_scorecard(config=config))
