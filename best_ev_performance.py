"""
Phase 10F — Best EV Performance + Oracle Proof Report (analytics only).

Measures the trades opened by the Phase 10D Best-EV paper runner (identified
by the EV belief stamped on the position at open) and answers the headline
question for the BEST_EV_PERFORMANCE and ORACLE_PROOF_REPORT Telegram
commands:

    *Is Oracle's scoring system predictive?*

The proof report combines the Phase 10E EV-attribution bucket evidence
(Oracle Score, Volatility Edge, Expected Value, Advisory Recommendation) with
the Phase 10G null-anchored checks — vol forecast vs IV (vol_forecast_
scorecard), excess win rate over predicted PoP (pop_calibration), and EV
calibration (ev_calibration) — and emits a single verdict:
PREDICTIVE / PROMISING_BUT_INCONCLUSIVE / NOT_PREDICTIVE_YET /
INSUFFICIENT_DATA, with a sample-size confidence (Low <50 trades,
Medium 50-200, High >200). Null-anchored means winning *at* the model's own
predicted PoP rate counts as zero evidence: a zero-edge premium seller does
exactly that.

STRICTLY analytics: never opens, closes, sizes, blocks or alters any real or
paper trade; never imports the live trader; never touches the network. All
readers fail open.
"""

from typing import List, Optional, Sequence, Tuple

import ev_attribution as eva
import ev_engine
import oracle_analytics as oa
import threshold_engine as te
from ev_attribution import (
    ANALYTICS_FOOTER, EV_BUCKETS, EV_RISK_BUCKETS, ADVISORY_ORDER,
    ORACLE_BUCKETS, VOL_EDGE_BUCKETS,
    VERDICT_YES, VERDICT_NO, VERDICT_INCONCLUSIVE,
)
from oracle_analytics import AnalyticsConfig

# Overall proof-report conclusions (Phase 10G-D null-anchored).
CONCLUSION_PREDICTIVE = "PREDICTIVE"
CONCLUSION_PROMISING = "PROMISING_BUT_INCONCLUSIVE"
CONCLUSION_NOT_PREDICTIVE = "NOT_PREDICTIVE_YET"
CONCLUSION_INSUFFICIENT = "INSUFFICIENT_DATA"

# A win rate within +/- this margin of the predicted PoP is exactly what a
# zero-edge premium seller gets, so it counts as NO evidence either way.
POP_EXCESS_MARGIN = 0.02

PROOF_QUESTION = "Is Oracle's scoring system predictive?"


# ---------------------------------------------------------------------------
# Best-EV paper trade selection
# ---------------------------------------------------------------------------
def is_best_ev_trade(row) -> bool:
    """A closed paper spread opened by the Best-EV runner: it carries the EV
    belief stamped at open (plain spread-paper trades do not)."""
    return (isinstance(row, dict)
            and oa._to_float(row.get("expected_value")) is not None
            and oa._trade_pnl(row) is not None)


def load_best_ev_trades(config: Optional[AnalyticsConfig] = None,
                        trades: Optional[List[dict]] = None) -> List[dict]:
    """Closed Best-EV runner trades from the spread paper trades file."""
    if trades is None:
        cfg = config or AnalyticsConfig.from_env()
        data = oa.read_json(cfg.spread_trades_file)
        trades = data if isinstance(data, list) else []
    return [row for row in trades if is_best_ev_trade(row)]


# ---------------------------------------------------------------------------
# Performance metrics
# ---------------------------------------------------------------------------
def _mean(values) -> Optional[float]:
    vals = [oa._to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


def _best_worst(table: dict) -> Tuple[Optional[str], Optional[str]]:
    """(best, worst) keys of a {key: bucket_stats} table by profit factor."""
    occupied = [(key, eva._pf_measure(stats))
                for key, stats in table.items()
                if stats.get("trades", 0) > 0
                and eva._pf_measure(stats) is not None]
    if not occupied:
        return None, None
    best = max(occupied, key=lambda x: x[1])[0]
    worst = min(occupied, key=lambda x: x[1])[0]
    return best, worst


def compute_best_ev_performance(trades: Optional[List[dict]] = None,
                                config: Optional[AnalyticsConfig] = None
                                ) -> dict:
    """Headline stats + breakdowns for Best-EV runner trades. Never raises."""
    rows = load_best_ev_trades(config=config, trades=trades)
    overall = eva.bucket_stats(rows)

    strategies = []
    for row in rows:
        strategy = row.get("strategy")
        if strategy and strategy not in strategies:
            strategies.append(strategy)
    by_strategy = {s: eva.bucket_stats([r for r in rows
                                        if r.get("strategy") == s])
                   for s in strategies}
    best_strategy, worst_strategy = _best_worst(by_strategy)

    ev_table = eva.compute_bucket_table(rows, eva._ev, EV_BUCKETS)
    return {
        "sample_size": len(rows),
        "confidence": te.compute_confidence(len(rows)),
        "overall": overall,
        "avg_expected_value": _mean(r.get("expected_value") for r in rows),
        "avg_ev_per_risk": _mean(r.get("ev_per_dollar_risk") for r in rows),
        "avg_oracle_score": _mean(oa._trade_oracle(r) for r in rows),
        "avg_volatility_edge": _mean(oa._trade_edge(r) for r in rows),
        "by_strategy": by_strategy,
        "best_strategy": best_strategy,
        "worst_strategy": worst_strategy,
        "by_recommendation": eva.compute_category_table(
            rows, "ev_recommendation", ADVISORY_ORDER),
        "by_advisory": eva.compute_category_table(
            rows, "advisory_recommendation", ADVISORY_ORDER),
        "ev_buckets": ev_table,
        "ev_risk_buckets": eva.compute_bucket_table(rows, eva._ev_risk,
                                                    EV_RISK_BUCKETS),
        "ev_predictiveness": eva.compute_predictiveness(
            ev_table, [b[0] for b in EV_BUCKETS]),
    }


# ---------------------------------------------------------------------------
# BEST_EV_PERFORMANCE formatting
# ---------------------------------------------------------------------------
def _strategy_str(name) -> str:
    return ev_engine.display_strategy_name(name) if name else "n/a"


def format_best_ev_performance(perf: dict) -> str:
    """Telegram-ready BEST_EV_PERFORMANCE summary. Pure formatting."""
    header = "🏆 *Best EV Paper Performance* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if perf.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            "No closed Best-EV paper trades yet.",
            "They appear after `BEST_EV_PAPER_RUN` positions are closed.",
            "", footer,
        ])

    m = perf["overall"]
    avg_ratio = perf.get("avg_ev_per_risk")
    ratio_s = f"{avg_ratio:.2f}" if avg_ratio is not None else "n/a"
    lines = [
        header, "",
        "*Best EV Paper Trades:*",
        f"Trades: `{m['trades']}`",
        f"Win Rate: `{m['win_rate'] * 100:.0f}%`",
        f"Total PnL: `{eva._money(m['total_pnl'])}`",
        f"Profit Factor: `{eva._pf_str(m['profit_factor'])}`",
        f"Avg EV: `{eva._money(perf.get('avg_expected_value'))}`",
        f"Avg EV/Risk: `{ratio_s}`",
        "",
        f"Best Strategy: `{_strategy_str(perf.get('best_strategy'))}`",
        f"Worst Strategy: `{_strategy_str(perf.get('worst_strategy'))}`",
        "",
        "*EV Predictiveness:*",
        f"Higher EV buckets outperform lower EV buckets: "
        f"`{perf['ev_predictiveness']['verdict']}`",
        "",
        f"Sample size: `{perf['sample_size']}` · "
        f"Confidence: *{perf['confidence']}*",
        "", footer,
    ]
    return "\n".join(lines)


def generate_best_ev_performance_text(config: Optional[AnalyticsConfig] = None
                                      ) -> str:
    """Top-level entry for the BEST_EV_PERFORMANCE Telegram command."""
    return format_best_ev_performance(compute_best_ev_performance(config=config))


# ---------------------------------------------------------------------------
# ORACLE_PROOF_REPORT — the weekly "is it predictive?" rollup
# ---------------------------------------------------------------------------
def _vote_from_verdict(verdict: str) -> int:
    """+1 supportive / -1 opposing / 0 neutral from a YES/NO verdict."""
    if verdict == VERDICT_YES:
        return 1
    if verdict == VERDICT_NO:
        return -1
    return 0


def _null_checks(records: List[dict],
                 config: Optional[AnalyticsConfig],
                 scorecard: Optional[dict],
                 pop_cal: Optional[dict],
                 ev_cal: Optional[dict]) -> dict:
    """The Phase 10G null-anchored evidence. Each check carries a vote:
    +1 supportive, -1 opposing, 0 neutral/insufficient. Never raises."""
    import ev_calibration as evc
    import pop_calibration as pc
    import vol_forecast_scorecard as vfs

    if scorecard is None:
        try:
            scorecard = vfs.compute_scorecard(config=config)
        except Exception:
            scorecard = {"verdict": vfs.VERDICT_INCONCLUSIVE, "rows": 0}
    if pop_cal is None:
        try:
            pop_cal = pc.compute_pop_calibration(records=records,
                                                 config=config)
        except Exception:
            pop_cal = {"verdict": pc.VERDICT_INSUFFICIENT,
                       "overall": {}, "sample_size": 0}
    if ev_cal is None:
        try:
            ev_cal = evc.compute_ev_calibration(records=records,
                                                config=config)
        except Exception:
            ev_cal = {"verdict": evc.VERDICT_INSUFFICIENT, "sample_size": 0}

    vol_verdict = scorecard.get("verdict")
    vol_vote = (1 if vol_verdict == vfs.VERDICT_FORECAST_BEATS_IV
                else -1 if vol_verdict == vfs.VERDICT_IV_BEATS_FORECAST
                else 0)

    # Excess win rate over the model's own predicted PoP — the null line.
    overall = pop_cal.get("overall") or {}
    excess = overall.get("calibration_error")
    if (excess is None
            or overall.get("trades", 0) < pc.MIN_TRADES):
        pop_vote = 0
    elif excess >= POP_EXCESS_MARGIN:
        pop_vote = 1
    elif excess <= -POP_EXCESS_MARGIN:
        pop_vote = -1
    else:
        pop_vote = 0

    ev_verdict = ev_cal.get("verdict")
    ev_vote = (1 if ev_verdict in (evc.VERDICT_EV_CALIBRATED,
                                   evc.VERDICT_EV_RANKS)
               else -1 if ev_verdict == evc.VERDICT_EV_NOT_PREDICTIVE
               else 0)

    return {
        "vol_forecast": {"verdict": vol_verdict,
                         "rows": scorecard.get("rows", 0),
                         "vote": vol_vote},
        "pop_excess": {"excess_win_rate": excess,
                       "predicted_avg_pop": overall.get("predicted_avg_pop"),
                       "actual_win_rate": overall.get("actual_win_rate"),
                       "trades": overall.get("trades", 0),
                       "vote": pop_vote},
        "ev_calibration": {"verdict": ev_verdict,
                           "sample_size": ev_cal.get("sample_size", 0),
                           "vote": ev_vote},
    }


def compute_proof_report(records: Optional[List[dict]] = None,
                         config: Optional[AnalyticsConfig] = None,
                         attribution_path: Optional[str] = None,
                         best_ev_trades: Optional[List[dict]] = None,
                         scorecard: Optional[dict] = None,
                         pop_cal: Optional[dict] = None,
                         ev_cal: Optional[dict] = None) -> dict:
    """Combine all predictiveness evidence into one null-anchored verdict.

    Seven evidence items: the four Phase 10E separation dimensions plus the
    three Phase 10G null-anchored checks (vol forecast vs IV, excess win
    rate over predicted PoP, EV calibration). Never raises.
    """
    if records is None:
        records = eva.load_closed_records(config=config,
                                          attribution_path=attribution_path)
    evidence = {
        "oracle_score": eva.compute_predictiveness(
            eva.compute_bucket_table(records, eva._oracle, ORACLE_BUCKETS),
            [b[0] for b in ORACLE_BUCKETS]),
        "volatility_edge": eva.compute_predictiveness(
            eva.compute_bucket_table(records, eva._edge_pct, VOL_EDGE_BUCKETS),
            [b[0] for b in VOL_EDGE_BUCKETS]),
        "expected_value": eva.compute_predictiveness(
            eva.compute_bucket_table(records, eva._ev, EV_BUCKETS),
            [b[0] for b in EV_BUCKETS]),
        "advisory_recommendation": eva.compute_predictiveness(
            eva.compute_category_table(records, "advisory_recommendation",
                                       ADVISORY_ORDER),
            ADVISORY_ORDER),
    }
    null_checks = _null_checks(records, config, scorecard, pop_cal, ev_cal)

    votes = ([_vote_from_verdict(e["verdict"]) for e in evidence.values()]
             + [c["vote"] for c in null_checks.values()])
    supportive = votes.count(1)
    opposing = votes.count(-1)
    confidence = te.compute_confidence(len(records))

    if not records:
        conclusion = CONCLUSION_INSUFFICIENT
    elif opposing > supportive:
        conclusion = CONCLUSION_NOT_PREDICTIVE
    elif (supportive >= 5 and opposing == 0 and confidence != "Low"
          and null_checks["pop_excess"]["vote"] == 1):
        # PREDICTIVE is only claimable when the system beats its own null:
        # winning meaningfully MORE often than the PoP it promised.
        conclusion = CONCLUSION_PREDICTIVE
    else:
        conclusion = CONCLUSION_PROMISING

    return {
        "question": PROOF_QUESTION,
        "sample_size": len(records),
        "confidence": confidence,
        "evidence": evidence,
        "null_checks": null_checks,
        "best_ev": compute_best_ev_performance(trades=best_ev_trades,
                                               config=config),
        "supportive": supportive,
        "opposing": opposing,
        "conclusion": conclusion,
    }


def _evidence_line(title: str, p: dict) -> str:
    # Verdicts go in backticks: bare underscores (INSUFFICIENT_DATA) are
    # unbalanced Markdown italics and make Telegram reject the message.
    if p.get("buckets_with_data", 0) < 2:
        return f"*{title}:* insufficient data — `{VERDICT_INCONCLUSIVE}`"
    return (f"*{title}:* separation `{p['separation']:+.2f}`, "
            f"monotonicity `{p['monotonicity'] * 100:.0f}%`, "
            f"best `{p['best_bucket']}` → `{p['verdict']}`")


def _vote_str(vote: int) -> str:
    return ("supportive" if vote > 0
            else "opposing" if vote < 0 else "neutral")


def _null_check_lines(checks: dict) -> List[str]:
    vol = checks.get("vol_forecast") or {}
    pop = checks.get("pop_excess") or {}
    evc = checks.get("ev_calibration") or {}
    lines = ["*Null-anchored checks:*"]
    lines.append(f"*Vol forecast vs IV:* `{vol.get('verdict', 'n/a')}` "
                 f"(`{vol.get('rows', 0)}` rows) — "
                 f"{_vote_str(vol.get('vote', 0))}")
    excess = pop.get("excess_win_rate")
    if excess is None:
        lines.append(f"*Excess win rate vs PoP:* insufficient data — "
                     f"{_vote_str(pop.get('vote', 0))}")
    else:
        lines.append(
            f"*Excess win rate vs PoP:* `{excess * 100:+.1f}pp` "
            f"(actual `{(pop.get('actual_win_rate') or 0) * 100:.0f}%` vs "
            f"promised `{(pop.get('predicted_avg_pop') or 0) * 100:.0f}%`) — "
            f"{_vote_str(pop.get('vote', 0))}")
    lines.append(f"*EV calibration:* `{evc.get('verdict', 'n/a')}` "
                 f"(n=`{evc.get('sample_size', 0)}`) — "
                 f"{_vote_str(evc.get('vote', 0))}")
    return lines


def format_proof_report(report: dict) -> str:
    """Telegram-ready ORACLE_PROOF_REPORT. Pure formatting."""
    header = "🔬 *Oracle Proof Report* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            f"_{PROOF_QUESTION}_",
            "",
            "No closed paper spread trades yet — no evidence either way.",
            f"*Overall conclusion:* `{CONCLUSION_INSUFFICIENT}`",
            "", footer,
        ])

    ev = report["evidence"]
    best = report["best_ev"]
    lines = [
        header, "",
        f"_{PROOF_QUESTION}_",
        "",
        _evidence_line("Oracle Score", ev["oracle_score"]),
        _evidence_line("Volatility Edge", ev["volatility_edge"]),
        _evidence_line("Expected Value", ev["expected_value"]),
        _evidence_line("Advisory Recommendation",
                       ev["advisory_recommendation"]),
        "",
    ]
    lines += _null_check_lines(report.get("null_checks") or {})
    lines.append("")
    bm = best.get("overall") or {}
    if best.get("sample_size", 0) > 0:
        lines.append(
            f"*Best-EV paper trades:* `{bm['trades']}` trades, "
            f"WR `{bm['win_rate'] * 100:.0f}%`, "
            f"PF `{eva._pf_str(bm['profit_factor'])}`, "
            f"PnL `{eva._money(bm['total_pnl'])}`")
    else:
        lines.append("*Best-EV paper trades:* none closed yet")
    lines += [
        "",
        f"*Overall conclusion:* `{report['conclusion']}` "
        f"({report['supportive']} supportive / {report['opposing']} opposing)",
        f"Sample size: `{report['sample_size']}` · "
        f"Confidence: *{report['confidence']}*",
        "", footer,
    ]
    return "\n".join(lines)


def generate_oracle_proof_report_text(config: Optional[AnalyticsConfig] = None,
                                      attribution_path: Optional[str] = None
                                      ) -> str:
    """Top-level entry for the ORACLE_PROOF_REPORT Telegram command."""
    report = compute_proof_report(config=config,
                                  attribution_path=attribution_path)
    return format_proof_report(report)
