"""
Phase 10G-B — PoP Calibration (analytics only).

The null hypothesis killer for a premium-selling system: a strategy with NO
edge still wins at roughly its probability-of-profit. So "78% win rate" is
meaningless until it is compared to the PoP the model itself predicted.

For every closed paper spread that carries the probability_of_profit stamped
at entry (Phase 10E attribution records), bucket by predicted PoP
(90-100 / 80-90 / 70-80 / 60-70 / 50-60 / <50%) and compare:

    predicted_avg_pop  vs  actual_win_rate  ->  calibration_error

calibration_error = actual_win_rate - predicted_avg_pop (positive = the
model wins MORE often than it promised = underconfident; negative =
overconfident). Verdict:

    WELL_CALIBRATED    |overall error| <= 5pp
    OVERCONFIDENT      overall error <= -5pp
    UNDERCONFIDENT     overall error >= +5pp
    INSUFFICIENT_DATA  fewer than 10 resolved trades

STRICTLY analytics: never opens, closes, sizes, blocks or alters any real or
paper trade; never touches the network. All readers fail open.
"""

from typing import List, Optional

import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER
from oracle_analytics import AnalyticsConfig

# Calibration verdicts.
VERDICT_WELL_CALIBRATED = "WELL_CALIBRATED"
VERDICT_OVERCONFIDENT = "OVERCONFIDENT"
VERDICT_UNDERCONFIDENT = "UNDERCONFIDENT"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# Need at least this many resolved trades before any calibration verdict.
MIN_TRADES = 10
# Within +/- this win-rate gap, predicted PoP is considered calibrated.
CALIBRATION_TOLERANCE = 0.05

POP_QUESTION = "Do trades win as often as the model's PoP promised?"

# Display order: highest predicted PoP first (per spec).
POP_CAL_BUCKETS = (
    ("PoP 90-100%", 0.90, None),
    ("PoP 80-90%", 0.80, 0.90),
    ("PoP 70-80%", 0.70, 0.80),
    ("PoP 60-70%", 0.60, 0.70),
    ("PoP 50-60%", 0.50, 0.60),
    ("PoP <50%", None, 0.50),
)


# ---------------------------------------------------------------------------
# Record loading
# ---------------------------------------------------------------------------
def load_pop_records(records: Optional[List[dict]] = None,
                     config: Optional[AnalyticsConfig] = None,
                     attribution_path: Optional[str] = None) -> List[dict]:
    """Closed records that carry both a PnL and an entry-time PoP stamp."""
    if records is None:
        records = eva.load_closed_records(config=config,
                                          attribution_path=attribution_path)
    return [r for r in records if isinstance(r, dict)
            and eva._pop(r) is not None and oa._trade_pnl(r) is not None]


# ---------------------------------------------------------------------------
# Calibration computation
# ---------------------------------------------------------------------------
def _avg(values) -> Optional[float]:
    vals = [oa._to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _calibration_block(rows: List[dict]) -> dict:
    """trades / predicted_avg_pop / actual_win_rate / calibration_error /
    profit_factor / avg_pnl for one PoP bucket (or overall)."""
    stats = eva.bucket_stats(rows)
    predicted = _avg(eva._pop(r) for r in rows)
    actual = round(stats["win_rate"], 4) if rows else None
    error = (round(actual - predicted, 4)
             if predicted is not None and actual is not None else None)
    return {
        "trades": stats["trades"],
        "predicted_avg_pop": predicted,
        "actual_win_rate": actual,
        "calibration_error": error,
        "profit_factor": stats["profit_factor"],
        "avg_pnl": stats["average_pnl"] if rows else None,
    }


def compute_pop_calibration(records: Optional[List[dict]] = None,
                            config: Optional[AnalyticsConfig] = None,
                            attribution_path: Optional[str] = None) -> dict:
    """Bucketed PoP-vs-realized calibration report. Never raises."""
    rows = load_pop_records(records=records, config=config,
                            attribution_path=attribution_path)
    buckets = {}
    for label, lo, hi in POP_CAL_BUCKETS:
        sub = [r for r in rows
               if eva.bucket_label(eva._pop(r), POP_CAL_BUCKETS) == label]
        buckets[label] = _calibration_block(sub)

    overall = _calibration_block(rows)
    error = overall.get("calibration_error")
    if overall["trades"] < MIN_TRADES or error is None:
        verdict = VERDICT_INSUFFICIENT
    elif error <= -CALIBRATION_TOLERANCE:
        verdict = VERDICT_OVERCONFIDENT
    elif error >= CALIBRATION_TOLERANCE:
        verdict = VERDICT_UNDERCONFIDENT
    else:
        verdict = VERDICT_WELL_CALIBRATED

    return {
        "question": POP_QUESTION,
        "sample_size": overall["trades"],
        "overall": overall,
        "buckets": buckets,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------
def _pp(value) -> str:
    """Win-rate style percentage, n/a-safe."""
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _err_pp(value) -> str:
    return f"{value * 100:+.1f}pp" if value is not None else "n/a"


def format_pop_calibration(report: dict) -> str:
    """Telegram-ready POP_CALIBRATION. Pure formatting."""
    header = "🎯 *PoP Calibration* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            f"_{POP_QUESTION}_",
            "",
            "No closed trades carrying an entry PoP stamp yet.",
            f"*Verdict:* {VERDICT_INSUFFICIENT}",
            "", footer,
        ])
    lines = [
        header, "",
        f"_{POP_QUESTION}_",
        "",
        "*Predicted PoP -> actual win rate:*",
    ]
    for label, _, _ in POP_CAL_BUCKETS:
        b = report["buckets"].get(label) or {}
        if b.get("trades", 0) == 0:
            lines.append(f"`{label}`: no trades")
            continue
        lines.append(
            f"`{label}`: `{b['trades']}` trades, predicted "
            f"`{_pp(b['predicted_avg_pop'])}` -> actual "
            f"`{_pp(b['actual_win_rate'])}` ({_err_pp(b['calibration_error'])}),"
            f" PF `{eva._pf_str(b['profit_factor'])}`, "
            f"avg `{eva._money(b['avg_pnl'])}`")
    o = report["overall"]
    lines += [
        "",
        f"*Overall:* predicted `{_pp(o['predicted_avg_pop'])}` -> actual "
        f"`{_pp(o['actual_win_rate'])}` ({_err_pp(o['calibration_error'])})",
        f"*Verdict:* {report['verdict']}",
        f"Sample size: `{report['sample_size']}`",
        "", footer,
    ]
    return "\n".join(lines)


def generate_pop_calibration_text(config: Optional[AnalyticsConfig] = None,
                                  attribution_path: Optional[str] = None
                                  ) -> str:
    """Top-level entry for the POP_CALIBRATION Telegram command."""
    return format_pop_calibration(
        compute_pop_calibration(config=config,
                                attribution_path=attribution_path))
