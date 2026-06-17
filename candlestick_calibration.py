"""
Phase 11B-3 — Candlestick pattern calibration (ANALYTICS ONLY).

Reads the append-only candidate ledger (``candidate_resolution.jsonl``, folded
by candidate_id) and asks, per candlestick pattern: *did it help or hurt?*
Outcome statistics are computed over RESOLVED candidates only, reusing the same
trade-like adapter + shared ``ev_attribution`` bucket helpers as
``calibration_reports`` so the numbers are consistent across the analytics
suite.

Candlestick patterns are market-behaviour features only. This module never
opens, closes, sizes, blocks or alters any real or paper trade, and never
touches the network. Every reader fails open (missing / malformed -> a clean
"no data" report). Patterns below ``MIN_PATTERN_SAMPLE_SIZE`` resolved trades
are flagged ``low_sample`` so a thin sample is never mistaken for an edge.
"""

from typing import Dict, List, Optional

import calibration_reports as calib
import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER

# Default minimum resolved trades before a pattern's verdict is trusted.
DEFAULT_MIN_PATTERN_SAMPLE_SIZE = 20
LOW_SAMPLE_WARNING = "LOW_SAMPLE"

# EV-impact verdicts.
EV_IMPACT_POSITIVE = "Positive"
EV_IMPACT_NEUTRAL = "Neutral"
EV_IMPACT_NEGATIVE = "Negative"

# Bias values we skip for directional calibration (indecision patterns).
_NEUTRAL_BIAS = "neutral"

REPORT_TITLE = "🕯️ Candlestick Pattern Calibration"
NO_DATA_MSG = "No candlestick patterns detected yet."


# ---------------------------------------------------------------------------
# Config + small helpers
# ---------------------------------------------------------------------------
def min_pattern_sample_size() -> int:
    """``MIN_PATTERN_SAMPLE_SIZE`` (shell > .env > default). Fail-open."""
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_int(
            "MIN_PATTERN_SAMPLE_SIZE", DEFAULT_MIN_PATTERN_SAMPLE_SIZE)
    except Exception:
        return DEFAULT_MIN_PATTERN_SAMPLE_SIZE


def _pf_num(stats: dict) -> Optional[float]:
    """Numeric profit factor (inf capped at PF_CAP) or None."""
    return eva._pf_measure(stats)


def _ev_impact(stats: dict) -> str:
    """Positive when the pattern both wins money and clears a >1 profit
    factor; Negative when it loses on both; Neutral otherwise."""
    pf = _pf_num(stats)
    avg = stats.get("average_pnl")
    if pf is not None and avg is not None:
        if pf > 1.0 and avg > 0:
            return EV_IMPACT_POSITIVE
        if pf < 1.0 and avg < 0:
            return EV_IMPACT_NEGATIVE
    return EV_IMPACT_NEUTRAL


def _avg(values) -> Optional[float]:
    return calib._avg(values)


def _mfe_mae(resolved: List[dict]):
    """Best-effort favorable/adverse excursion proxies from ``actual_move``:
    mean of positive moves and mean of negative moves."""
    moves = [oa._to_float(r.get("actual_move")) for r in resolved]
    moves = [m for m in moves if m is not None]
    ups = [m for m in moves if m > 0]
    downs = [m for m in moves if m < 0]
    mfe = round(sum(ups) / len(ups), 4) if ups else None
    mae = round(sum(downs) / len(downs), 4) if downs else None
    return mfe, mae


# ---------------------------------------------------------------------------
# Per-pattern computation
# ---------------------------------------------------------------------------
def _pattern_block(records: List[dict], min_sample: int) -> dict:
    """Stats for the candidates carrying ONE candlestick pattern."""
    resolved = [r for r in records if calib._is_resolved(r)]
    trades = [t for t in (calib._to_trade_like(r) for r in resolved)
              if oa._trade_pnl(t) is not None]
    stats = eva.bucket_stats(trades)
    mfe, mae = _mfe_mae([r for r in resolved
                         if oa._trade_pnl(calib._to_trade_like(r)) is not None])

    advisory_distribution: Dict[str, int] = {}
    for r in records:
        adv = r.get("advisory_recommendation")
        if adv:
            advisory_distribution[adv] = advisory_distribution.get(adv, 0) + 1

    n_trades = len(trades)
    low_sample = n_trades < min_sample
    return {
        "pattern_name": records[0].get("candlestick_pattern"),
        "bias": records[0].get("candlestick_bias"),
        "occurrences": len(records),
        "resolved_trades": n_trades,
        "win_rate": round(stats["win_rate"], 4) if trades else None,
        "profit_factor": stats["profit_factor"] if trades else None,
        "average_return": _avg(r.get("actual_move") for r in resolved),
        "average_pnl": stats["average_pnl"] if trades else None,
        "average_max_favorable_excursion": mfe,
        "average_max_adverse_excursion": mae,
        "average_pop": _avg(r.get("probability_of_profit") for r in records),
        "average_ev": _avg(r.get("expected_value") for r in records),
        "average_triple_gap": _avg(r.get("triple_gap_score") for r in records),
        "average_oracle_score": _avg(r.get("oracle_score") for r in records),
        "average_volatility_edge": _avg(r.get("volatility_edge")
                                        for r in records),
        "average_confidence": _avg(r.get("candlestick_confidence")
                                   for r in records),
        "advisory_distribution": advisory_distribution,
        "options_outcomes": {
            "selected": sum(1 for r in records
                            if r.get("selected_for_paper_trade")),
            "resolved": n_trades,
        },
        "ev_impact": _ev_impact(stats) if trades else EV_IMPACT_NEUTRAL,
        "low_sample": low_sample,
        "warning": LOW_SAMPLE_WARNING if low_sample else None,
    }


def _overall(candidates: List[dict], min_sample: int) -> dict:
    """Did pattern-tagged trades out-earn untagged ones? Resolved trades only."""
    tagged, untagged = [], []
    for rec in candidates:
        if not calib._is_resolved(rec):
            continue
        trade = calib._to_trade_like(rec)
        if oa._trade_pnl(trade) is None:
            continue
        if rec.get("candlestick_pattern"):
            tagged.append(trade)
        else:
            untagged.append(trade)

    t_stats = eva.bucket_stats(tagged)
    u_stats = eva.bucket_stats(untagged)
    improved = None
    if tagged and untagged:
        t_pf = _pf_num(t_stats) or 0.0
        u_pf = _pf_num(u_stats) or 0.0
        improved = bool(t_pf > u_pf
                        and t_stats["average_pnl"] > u_stats["average_pnl"])
    return {
        "tagged": {
            "trades": len(tagged),
            "win_rate": round(t_stats["win_rate"], 4) if tagged else None,
            "profit_factor": t_stats["profit_factor"] if tagged else None,
            "average_pnl": t_stats["average_pnl"] if tagged else None,
        },
        "untagged": {
            "trades": len(untagged),
            "win_rate": round(u_stats["win_rate"], 4) if untagged else None,
            "profit_factor": u_stats["profit_factor"] if untagged else None,
            "average_pnl": u_stats["average_pnl"] if untagged else None,
        },
        "improved_ev": improved,
        "enough_sample": len(tagged) >= min_sample,
    }


def _leaderboards(patterns: Dict[str, dict]) -> dict:
    """Rank patterns by occurrences, win rate and profit factor."""
    items = list(patterns.values())

    def _rank(key, only_resolved=True):
        rows = [p for p in items
                if (not only_resolved or p["resolved_trades"] > 0)]
        rows = [p for p in rows if p.get(key) is not None]
        rows.sort(key=lambda p: (_pf_num(p) if key == "profit_factor"
                                 else p[key]), reverse=True)
        return [{"pattern_name": p["pattern_name"], key: p[key],
                 "resolved_trades": p["resolved_trades"],
                 "low_sample": p["low_sample"]} for p in rows]

    by_occ = sorted(items, key=lambda p: p["occurrences"], reverse=True)
    return {
        "by_occurrences": [
            {"pattern_name": p["pattern_name"],
             "occurrences": p["occurrences"],
             "resolved_trades": p["resolved_trades"]} for p in by_occ],
        "by_win_rate": _rank("win_rate"),
        "by_profit_factor": _rank("profit_factor"),
    }


def compute_candlestick_calibration(records: Optional[List[dict]] = None,
                                    jsonl_path: Optional[str] = None) -> dict:
    """Per-pattern calibration + leaderboards + overall verdict. Never raises.

    Directional patterns only (neutral/indecision patterns are skipped). Pure
    when ``records`` is supplied; otherwise reads the folded JSONL ledger.
    """
    try:
        candidates = calib.load_candidates(records, jsonl_path)
    except Exception:
        candidates = []
    min_sample = min_pattern_sample_size()

    groups: Dict[str, List[dict]] = {}
    for rec in candidates:
        name = rec.get("candlestick_pattern")
        if not name or rec.get("candlestick_bias") == _NEUTRAL_BIAS:
            continue
        groups.setdefault(name, []).append(rec)

    patterns = {name: _pattern_block(recs, min_sample)
                for name, recs in groups.items()}
    sample_size = sum(p["resolved_trades"] for p in patterns.values())
    return {
        "sample_size": sample_size,
        "patterns_detected": len(patterns),
        "min_sample_size": min_sample,
        "patterns": patterns,
        "leaderboards": _leaderboards(patterns),
        "overall": _overall(candidates, min_sample),
    }


def compute_daily_candlestick_summary(records: Optional[List[dict]] = None,
                                      jsonl_path: Optional[str] = None) -> dict:
    """Compact summary for the daily report. Fail-open to ``{}``."""
    try:
        rep = compute_candlestick_calibration(records, jsonl_path)
        top = []
        for row in rep["leaderboards"]["by_occurrences"][:3]:
            p = rep["patterns"].get(row["pattern_name"], {})
            top.append({
                "pattern_name": row["pattern_name"],
                "occurrences": row["occurrences"],
                "win_rate": p.get("win_rate"),
                "ev_impact": p.get("ev_impact"),
                "low_sample": p.get("low_sample"),
            })
        return {
            "patterns_detected": rep["patterns_detected"],
            "sample_size": rep["sample_size"],
            "top_patterns": top,
            "improved_ev": rep["overall"].get("improved_ev"),
        }
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# Formatters
# ---------------------------------------------------------------------------
def _pct(value) -> str:
    return f"{value * 100:.1f}%" if value is not None else "n/a"


def format_candlestick_report(report: dict) -> str:
    """Render the calibration report as Telegram-friendly text. Never raises."""
    lines = [REPORT_TITLE, ""]
    try:
        patterns = report.get("patterns") or {}
        if not patterns:
            lines.append(NO_DATA_MSG)
            lines.append("")
            lines.append(ANALYTICS_FOOTER)
            return "\n".join(lines)

        lines.append(f"Resolved trades: {report.get('sample_size', 0)}  |  "
                     f"Patterns: {report.get('patterns_detected', 0)}  |  "
                     f"Min sample: {report.get('min_sample_size')}")
        lines.append("")

        boards = report.get("leaderboards") or {}
        occ = boards.get("by_occurrences") or []
        if occ:
            lines.append("Most detected:")
            for row in occ[:5]:
                lines.append(
                    f"  • {row['pattern_name']}: {row['occurrences']} seen, "
                    f"{row['resolved_trades']} resolved")
            lines.append("")

        wr = boards.get("by_win_rate") or []
        if wr:
            lines.append("Top win rate:")
            for row in wr[:5]:
                flag = "  ⚠️ low sample" if row["low_sample"] else ""
                lines.append(
                    f"  • {row['pattern_name']}: {_pct(row['win_rate'])} "
                    f"({row['resolved_trades']} trades){flag}")
            lines.append("")

        pf = boards.get("by_profit_factor") or []
        if pf:
            lines.append("Top profit factor:")
            for row in pf[:5]:
                flag = "  ⚠️ low sample" if row["low_sample"] else ""
                lines.append(
                    f"  • {row['pattern_name']}: "
                    f"{eva._pf_str(row['profit_factor'])}{flag}")
            lines.append("")

        lines.append("Per-pattern EV impact:")
        for name, p in patterns.items():
            flag = " ⚠️ LOW SAMPLE" if p["low_sample"] else ""
            lines.append(
                f"  • {name} ({p['bias']}): {p['ev_impact']} | "
                f"WR {_pct(p['win_rate'])} | "
                f"PF {eva._pf_str(p['profit_factor'])} | "
                f"avgPnL {eva._money(p['average_pnl'])} | "
                f"avgEV {eva._money(p['average_ev'])} | "
                f"PoP {_pct(p['average_pop'])} | "
                f"TG {p['average_triple_gap']}{flag}")
        lines.append("")

        overall = report.get("overall") or {}
        improved = overall.get("improved_ev")
        verdict = ("patterns improved EV" if improved
                   else "no clear EV improvement" if improved is False
                   else "insufficient data")
        lines.append(f"Did patterns help? {verdict}")
        tagged = overall.get("tagged", {})
        untagged = overall.get("untagged", {})
        lines.append(
            f"  tagged: {tagged.get('trades', 0)} trades, "
            f"WR {_pct(tagged.get('win_rate'))}, "
            f"PF {eva._pf_str(tagged.get('profit_factor'))}")
        lines.append(
            f"  untagged: {untagged.get('trades', 0)} trades, "
            f"WR {_pct(untagged.get('win_rate'))}, "
            f"PF {eva._pf_str(untagged.get('profit_factor'))}")
        lines.append("")
    except Exception as exc:  # pragma: no cover - defensive
        lines.append(f"(report error: {exc})")
    lines.append(ANALYTICS_FOOTER)
    return "\n".join(lines)


def generate_candlestick_report_text(records: Optional[List[dict]] = None,
                                     jsonl_path: Optional[str] = None) -> str:
    """Top-level entry: compute + format. Fail-open to a clean report."""
    try:
        report = compute_candlestick_calibration(records, jsonl_path)
    except Exception:
        report = {"patterns": {}}
    return format_candlestick_report(report)


if __name__ == "__main__":
    print(generate_candlestick_report_text())
