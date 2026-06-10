"""
Phase 10G-C — EV Calibration (analytics only).

Expected value is only useful if it predicts realized PnL. Two separate
claims are tested over closed paper spreads that carry the EV stamped at
entry:

  1. RANKING — do higher-EV trades make more money? Measured by the OLS
     slope of realized_pnl on expected_value and by the Phase 10E
     bucket-predictiveness check over the EV buckets (<0 / 0-10 / 10-20 /
     20-50 / 50+).
  2. MAGNITUDE — does $X of predicted EV deliver about $X of realized PnL?
     Measured by calibration_error = avg_realized_pnl - avg_expected_ev,
     overall and per bucket.

Verdicts:
    EV_CALIBRATED            slope > 0, buckets rank, magnitude within tol
    EV_RANKS_BUT_MISPRICES   ranking evidence (slope or buckets), bad sizing
    EV_NOT_PREDICTIVE        no ranking evidence at all
    INSUFFICIENT_DATA        fewer than 10 resolved trades

STRICTLY analytics: never opens, closes, sizes, blocks or alters any real or
paper trade; never touches the network. All readers fail open.
"""

from typing import List, Optional

import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER, EV_BUCKETS, VERDICT_YES
from oracle_analytics import AnalyticsConfig
from vol_forecast_scorecard import linear_regression

# Calibration verdicts.
VERDICT_EV_CALIBRATED = "EV_CALIBRATED"
VERDICT_EV_RANKS = "EV_RANKS_BUT_MISPRICES"
VERDICT_EV_NOT_PREDICTIVE = "EV_NOT_PREDICTIVE"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# Need at least this many resolved trades before any calibration verdict.
MIN_TRADES = 10
# Magnitude tolerance: realized must land within max($10, 50% of predicted)
# of the predicted EV to count as calibrated.
ABS_TOLERANCE = 10.0
REL_TOLERANCE = 0.5

EV_CAL_QUESTION = "Does $1 of predicted EV deliver $1 of realized PnL?"


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------
def load_ev_records(records: Optional[List[dict]] = None,
                    config: Optional[AnalyticsConfig] = None,
                    attribution_path: Optional[str] = None) -> List[dict]:
    """Closed records carrying both a PnL and an entry-time EV stamp."""
    if records is None:
        records = eva.load_closed_records(config=config,
                                          attribution_path=attribution_path)
    return [r for r in records if isinstance(r, dict)
            and eva._ev(r) is not None and oa._trade_pnl(r) is not None]


# ---------------------------------------------------------------------------
# Calibration computation
# ---------------------------------------------------------------------------
def _avg(values) -> Optional[float]:
    vals = [oa._to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _ev_block(rows: List[dict]) -> dict:
    """trades / avg_expected_ev / avg_realized_pnl / profit_factor /
    calibration_error for one EV bucket (or overall)."""
    stats = eva.bucket_stats(rows)
    expected = _avg(eva._ev(r) for r in rows)
    realized = stats["average_pnl"] if rows else None
    error = (round(realized - expected, 4)
             if expected is not None and realized is not None else None)
    return {
        "trades": stats["trades"],
        "avg_expected_ev": expected,
        "avg_realized_pnl": realized,
        "profit_factor": stats["profit_factor"],
        "calibration_error": error,
    }


def within_tolerance(expected, realized) -> Optional[bool]:
    """Is realized PnL within max($10, 50% of |expected|) of expected EV?"""
    if expected is None or realized is None:
        return None
    tol = max(ABS_TOLERANCE, REL_TOLERANCE * abs(expected))
    return abs(realized - expected) <= tol


def compute_ev_calibration(records: Optional[List[dict]] = None,
                           config: Optional[AnalyticsConfig] = None,
                           attribution_path: Optional[str] = None) -> dict:
    """EV regression + bucket calibration report. Never raises."""
    rows = load_ev_records(records=records, config=config,
                           attribution_path=attribution_path)
    regression = linear_regression([eva._ev(r) for r in rows],
                                   [oa._trade_pnl(r) for r in rows])
    regression = {"alpha": regression["alpha"], "beta": regression["beta"],
                  "r_squared": regression["r_squared"],
                  "sample_size": regression["n"]}

    buckets = {}
    for label, lo, hi in EV_BUCKETS:
        sub = [r for r in rows
               if eva.bucket_label(eva._ev(r), EV_BUCKETS) == label]
        buckets[label] = _ev_block(sub)
    ranking = eva.compute_predictiveness(buckets, [b[0] for b in EV_BUCKETS])

    overall = _ev_block(rows)
    slope = regression["beta"]
    slope_positive = slope is not None and slope > 0
    ranks = ranking["verdict"] == VERDICT_YES
    calibrated = within_tolerance(overall["avg_expected_ev"],
                                  overall["avg_realized_pnl"])
    if overall["trades"] < MIN_TRADES:
        verdict = VERDICT_INSUFFICIENT
    elif slope_positive and ranks and bool(calibrated):
        verdict = VERDICT_EV_CALIBRATED
    elif slope_positive or ranks:
        verdict = VERDICT_EV_RANKS
    else:
        verdict = VERDICT_EV_NOT_PREDICTIVE

    return {
        "question": EV_CAL_QUESTION,
        "sample_size": overall["trades"],
        "regression": regression,
        "buckets": buckets,
        "ranking": ranking,
        "overall": overall,
        "slope_positive": slope_positive,
        "magnitude_calibrated": calibrated,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------
def _num(value, digits: int = 4) -> str:
    return f"{value:.{digits}f}" if value is not None else "n/a"


def format_ev_calibration(report: dict) -> str:
    """Telegram-ready EV_CALIBRATION. Pure formatting."""
    header = "⚖️ *EV Calibration* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            f"_{EV_CAL_QUESTION}_",
            "",
            "No closed trades carrying an entry EV stamp yet.",
            f"*Verdict:* {VERDICT_INSUFFICIENT}",
            "", footer,
        ])
    reg = report["regression"]
    if reg.get("beta") is not None:
        reg_line = (f"realized = `{_num(reg['alpha'], 2)}` + "
                    f"`{_num(reg['beta'], 4)}` x EV, "
                    f"R² `{_num(reg['r_squared'], 4)}`, "
                    f"n=`{reg['sample_size']}`")
    else:
        reg_line = f"insufficient data (n=`{reg.get('sample_size', 0)}`)"
    lines = [
        header, "",
        f"_{EV_CAL_QUESTION}_",
        "",
        f"*Regression:* {reg_line}",
        "",
        "*EV buckets (expected -> realized):*",
    ]
    for label, _, _ in EV_BUCKETS:
        b = report["buckets"].get(label) or {}
        if b.get("trades", 0) == 0:
            lines.append(f"`{label}`: no trades")
            continue
        lines.append(
            f"`{label}`: `{b['trades']}` trades, expected "
            f"`{eva._money(b['avg_expected_ev'])}` -> realized "
            f"`{eva._money(b['avg_realized_pnl'])}` "
            f"(err `{eva._money(b['calibration_error'])}`), "
            f"PF `{eva._pf_str(b['profit_factor'])}`")
    o = report["overall"]
    ranking = report["ranking"]
    lines += [
        "",
        f"*Ranking:* higher EV buckets outperform: *{ranking['verdict']}*",
        f"*Magnitude:* expected `{eva._money(o['avg_expected_ev'])}` -> "
        f"realized `{eva._money(o['avg_realized_pnl'])}` "
        f"(err `{eva._money(o['calibration_error'])}`)",
        f"*Verdict:* {report['verdict']}",
        f"Sample size: `{report['sample_size']}`",
        "", footer,
    ]
    return "\n".join(lines)


def generate_ev_calibration_text(config: Optional[AnalyticsConfig] = None,
                                 attribution_path: Optional[str] = None
                                 ) -> str:
    """Top-level entry for the EV_CALIBRATION Telegram command."""
    return format_ev_calibration(
        compute_ev_calibration(config=config,
                               attribution_path=attribution_path))
