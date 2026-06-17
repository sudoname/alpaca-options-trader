"""
Phase 11A-4 — Calibration Reports (analytics only).

Reads the append-only candidate ledger (``candidate_resolution.jsonl``, folded
by candidate_id) and asks four questions over EVERY evaluated candidate —
selected AND rejected — so the analysis is free of the survivorship bias that
plagues closed-trade-only reports:

  1. Triple Gap calibration — do higher Triple-Gap (model-vs-market
     disagreement) buckets actually win and earn more?
  2. PoP calibration       — do trades win as often as the model's PoP promised?
  3. EV calibration        — does $1 of predicted EV deliver $1 of realized PnL?
  4. Signal separation     — which entry signal best separates winners from
     losers (by profit-factor separation)?

Outcome statistics are computed over RESOLVED candidates only, using a trade-
like adapter so the shared Phase 10E helpers in ``ev_attribution`` can be reused
verbatim. Realized PnL per candidate = ``actual_paper_pnl`` when known, else the
``hypothetical_hold_to_expiry_pnl`` payoff.

STRICTLY analytics: this module only reads and reports. It never opens, closes,
sizes, blocks or alters any real or paper trade, and never touches the network.
Every reader fails open (missing / malformed -> insufficient-data report).
"""

from typing import List, Optional, Sequence

import candidate_resolution as cr
import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import (
    ANALYTICS_FOOTER, EV_BUCKETS, EV_RISK_BUCKETS, ORACLE_BUCKETS,
    VOL_EDGE_BUCKETS, ADVISORY_ORDER,
)

# Minimum resolved candidates before any verdict beyond INSUFFICIENT_DATA.
MIN_RESOLVED = 10

# Overall verdicts shared by the Triple-Gap and signal-separation reports.
VERDICT_PREDICTIVE = "PREDICTIVE"
VERDICT_PROMISING = "PROMISING_BUT_INCONCLUSIVE"
VERDICT_NOT_YET = "NOT_PREDICTIVE_YET"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# A candidate counts toward outcome stats once it carries a resolved status.
RESOLVED_STATUSES = (cr.RESOLUTION_EXPIRY, cr.RESOLUTION_PARTIAL)

TRIPLE_GAP_QUESTION = "Does model-vs-market disagreement predict profit?"
SIGNAL_SEP_QUESTION = "Which entry signal best separates winners from losers?"

# Triple Gap score buckets (ascending so predictiveness ordering is meaningful;
# displayed best-first). Half-open [lo, hi); None = unbounded.
TRIPLE_GAP_BUCKETS = (
    ("TG <60", None, 60.0),
    ("TG 60-69", 60.0, 70.0),
    ("TG 70-79", 70.0, 80.0),
    ("TG 80-89", 80.0, 90.0),
    ("TG 90-100", 90.0, None),
)

# PoP calibration buckets (ascending; displayed high-first per spec).
POP_CAL_BUCKETS = (
    ("PoP <50%", None, 0.50),
    ("PoP 50-60%", 0.50, 0.60),
    ("PoP 60-70%", 0.60, 0.70),
    ("PoP 70-80%", 0.70, 0.80),
    ("PoP 80-90%", 0.80, 0.90),
    ("PoP 90-100%", 0.90, None),
)

# Numeric signals for the separation report: (name, value_fn, buckets).
_SEPARATION_SIGNALS = (
    ("Triple Gap", lambda r: r.get("triple_gap_score"), TRIPLE_GAP_BUCKETS),
    ("Expected Value", eva._ev, EV_BUCKETS),
    ("EV/Risk", eva._ev_risk, EV_RISK_BUCKETS),
    ("Oracle Score", eva._oracle, ORACLE_BUCKETS),
    ("Volatility Edge", eva._edge_pct, VOL_EDGE_BUCKETS),
)
_ADVISORY_SIGNAL = "Advisory Recommendation"


# ---------------------------------------------------------------------------
# Record loading + trade-like adapter
# ---------------------------------------------------------------------------
def _to_trade_like(rec: dict) -> dict:
    """Map a resolved candidate to a Phase-10E trade-like dict so the shared
    ``ev_attribution`` bucket helpers can consume it directly. Realized PnL is
    the actual paper PnL when known, else the hold-to-expiry payoff."""
    pnl = rec.get("actual_paper_pnl")
    if pnl is None:
        pnl = rec.get("hypothetical_hold_to_expiry_pnl")
    return {
        "pnl": pnl,
        "max_loss": rec.get("max_loss"),
        "probability_of_profit": rec.get("probability_of_profit"),
        "expected_value": rec.get("expected_value"),
        "ev_per_dollar_risk": rec.get("ev_per_dollar_risk"),
        "triple_gap_score": rec.get("triple_gap_score"),
        "oracle_score": rec.get("oracle_score"),
        "volatility_edge": rec.get("volatility_edge"),
        "advisory_recommendation": rec.get("advisory_recommendation"),
        "actual_move": rec.get("actual_move"),
        "selected_for_paper_trade": rec.get("selected_for_paper_trade"),
        "resolution_status": rec.get("resolution_status"),
    }


def _is_resolved(rec: dict) -> bool:
    return rec.get("resolution_status") in RESOLVED_STATUSES


def load_candidates(records: Optional[List[dict]] = None,
                    jsonl_path: Optional[str] = None) -> List[dict]:
    """Folded candidate ledger (all candidates, resolved or not). Fail-open."""
    if records is not None:
        return [r for r in records if isinstance(r, dict)]
    return cr.load_jsonl_records(jsonl_path)


def load_resolved_trades(records: Optional[List[dict]] = None,
                         jsonl_path: Optional[str] = None) -> List[dict]:
    """Resolved candidates mapped to trade-like dicts that carry a PnL."""
    out = []
    for rec in load_candidates(records, jsonl_path):
        if not _is_resolved(rec):
            continue
        trade = _to_trade_like(rec)
        if oa._trade_pnl(trade) is not None:
            out.append(trade)
    return out


def _avg(values) -> Optional[float]:
    vals = [oa._to_float(v) for v in values]
    vals = [v for v in vals if v is not None]
    return round(sum(vals) / len(vals), 4) if vals else None


# ---------------------------------------------------------------------------
# 1. Triple Gap calibration
# ---------------------------------------------------------------------------
def _triple_gap_block(records: List[dict]) -> dict:
    """Per-bucket Triple-Gap calibration over the candidates in one bucket."""
    resolved = [r for r in records if _is_resolved(r)]
    trades = [t for t in (_to_trade_like(r) for r in resolved)
              if oa._trade_pnl(t) is not None]
    stats = eva.bucket_stats(trades)
    return {
        "candidates": len(records),
        "selected": sum(1 for r in records
                        if r.get("selected_for_paper_trade")),
        "resolved": len(trades),
        "win_rate": round(stats["win_rate"], 4) if trades else None,
        "profit_factor": stats["profit_factor"],
        "avg_pnl": stats["average_pnl"] if trades else None,
        "avg_actual_move": _avg(r.get("actual_move") for r in resolved),
        "avg_ev": _avg(r.get("expected_value") for r in records),
        "avg_pop": _avg(r.get("probability_of_profit") for r in records),
    }


def compute_triple_gap_report(records: Optional[List[dict]] = None,
                              jsonl_path: Optional[str] = None) -> dict:
    """Triple-Gap bucket calibration + predictiveness verdict. Never raises."""
    candidates = load_candidates(records, jsonl_path)
    scored = [r for r in candidates
              if oa._to_float(r.get("triple_gap_score")) is not None]

    buckets = {}
    pred_table = {}
    for label, _, _ in TRIPLE_GAP_BUCKETS:
        sub = [r for r in scored
               if eva.bucket_label(r.get("triple_gap_score"),
                                   TRIPLE_GAP_BUCKETS) == label]
        block = _triple_gap_block(sub)
        buckets[label] = block
        pred_table[label] = eva.bucket_stats(
            [t for t in (_to_trade_like(r) for r in sub if _is_resolved(r))
             if oa._trade_pnl(t) is not None])

    ranking = eva.compute_predictiveness(
        pred_table, [b[0] for b in TRIPLE_GAP_BUCKETS])
    resolved_total = sum(b["resolved"] for b in buckets.values())

    if resolved_total < MIN_RESOLVED:
        verdict = VERDICT_INSUFFICIENT
    elif ranking["verdict"] == eva.VERDICT_YES:
        verdict = VERDICT_PREDICTIVE
    elif ranking["verdict"] == eva.VERDICT_NO:
        verdict = VERDICT_NOT_YET
    else:
        verdict = VERDICT_PROMISING

    return {
        "question": TRIPLE_GAP_QUESTION,
        "candidates": len(candidates),
        "scored": len(scored),
        "resolved": resolved_total,
        "buckets": buckets,
        "ranking": ranking,
        "separation_score": ranking.get("separation"),
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# 2. PoP calibration  /  3. EV calibration (over the resolved candidate set)
# ---------------------------------------------------------------------------
def _pop_block(trades: List[dict]) -> dict:
    stats = eva.bucket_stats(trades)
    predicted = _avg(t.get("probability_of_profit") for t in trades)
    actual = round(stats["win_rate"], 4) if trades else None
    error = (round(actual - predicted, 4)
             if predicted is not None and actual is not None else None)
    return {
        "trades": stats["trades"],
        "predicted_avg_pop": predicted,
        "actual_win_rate": actual,
        "calibration_error": error,
        "profit_factor": stats["profit_factor"],
        "avg_pnl": stats["average_pnl"] if trades else None,
    }


def compute_pop_calibration(records: Optional[List[dict]] = None,
                            jsonl_path: Optional[str] = None) -> dict:
    """PoP-vs-realized calibration over resolved candidates. Never raises."""
    trades = load_resolved_trades(records, jsonl_path)
    buckets = {}
    for label, _, _ in POP_CAL_BUCKETS:
        sub = [t for t in trades
               if eva.bucket_label(t.get("probability_of_profit"),
                                   POP_CAL_BUCKETS) == label]
        buckets[label] = _pop_block(sub)
    overall = _pop_block(trades)
    return {
        "sample_size": len(trades),
        "buckets": buckets,
        "overall": overall,
    }


def _ev_block(trades: List[dict]) -> dict:
    stats = eva.bucket_stats(trades)
    expected = _avg(eva._ev(t) for t in trades)
    realized = stats["average_pnl"] if trades else None
    error = (round(realized - expected, 4)
             if expected is not None and realized is not None else None)
    return {
        "trades": stats["trades"],
        "avg_expected_ev": expected,
        "avg_realized_pnl": realized,
        "calibration_error": error,
        "profit_factor": stats["profit_factor"],
    }


def compute_ev_calibration(records: Optional[List[dict]] = None,
                           jsonl_path: Optional[str] = None) -> dict:
    """EV-vs-realized calibration over resolved candidates. Never raises."""
    trades = load_resolved_trades(records, jsonl_path)
    buckets = {}
    for label, _, _ in EV_BUCKETS:
        sub = [t for t in trades
               if eva.bucket_label(eva._ev(t), EV_BUCKETS) == label]
        buckets[label] = _ev_block(sub)
    overall = _ev_block(trades)
    return {
        "sample_size": len(trades),
        "buckets": buckets,
        "overall": overall,
    }


# ---------------------------------------------------------------------------
# 4. Signal separation
# ---------------------------------------------------------------------------
def compute_signal_separation(records: Optional[List[dict]] = None,
                              jsonl_path: Optional[str] = None) -> dict:
    """Profit-factor separation per entry signal over resolved candidates.

    Higher separation = the signal pulls winners and losers further apart.
    Never raises.
    """
    trades = load_resolved_trades(records, jsonl_path)
    rankings = {}
    separation = {}
    for name, value_fn, buckets in _SEPARATION_SIGNALS:
        table = eva.compute_bucket_table(trades, value_fn, buckets)
        pred = eva.compute_predictiveness(table, [b[0] for b in buckets])
        rankings[name] = pred
        separation[name] = pred.get("separation")
    adv_table = eva.compute_category_table(
        trades, "advisory_recommendation", ADVISORY_ORDER)
    adv_pred = eva.compute_predictiveness(adv_table, list(ADVISORY_ORDER))
    rankings[_ADVISORY_SIGNAL] = adv_pred
    separation[_ADVISORY_SIGNAL] = adv_pred.get("separation")

    present = {k: v for k, v in separation.items() if v is not None}
    best = max(present, key=present.get) if present else None
    weakest = min(present, key=present.get) if present else None

    if len(trades) < MIN_RESOLVED:
        verdict = VERDICT_INSUFFICIENT
    elif not present:
        verdict = VERDICT_NOT_YET
    else:
        any_yes = any(rankings[k]["verdict"] == eva.VERDICT_YES
                      for k in present)
        best_sep = max(present.values())
        if any_yes and best_sep > 0:
            verdict = VERDICT_PREDICTIVE
        elif best_sep > 0:
            verdict = VERDICT_PROMISING
        else:
            verdict = VERDICT_NOT_YET

    return {
        "question": SIGNAL_SEP_QUESTION,
        "sample_size": len(trades),
        "separation_score_by_signal": separation,
        "rankings": rankings,
        "best_predictive_signal": best,
        "weakest_predictive_signal": weakest,
        "overall_verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting (pure)
# ---------------------------------------------------------------------------
def _pct(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _err_pp(value) -> str:
    return f"{value * 100:+.1f}pp" if value is not None else "n/a"


def _sep_str(value) -> str:
    return f"{value:+.2f}" if value is not None else "n/a"


def _move_str(value) -> str:
    return f"{value * 100:+.2f}%" if value is not None else "n/a"


def format_triple_gap_report(report: dict) -> str:
    """Telegram-ready TRIPLE_GAP_REPORT. Pure formatting."""
    header = "🔺 *Triple Gap Report* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("candidates", 0) == 0:
        return "\n".join([
            header, "",
            f"_{TRIPLE_GAP_QUESTION}_",
            "",
            "No candidates stamped yet.",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`",
            "", footer,
        ])
    lines = [
        header, "",
        f"_{TRIPLE_GAP_QUESTION}_",
        "",
        "*Triple Gap buckets (candidates / resolved -> WR, PF):*",
    ]
    for label, _, _ in reversed(TRIPLE_GAP_BUCKETS):  # best-first display
        b = report["buckets"].get(label) or {}
        if b.get("candidates", 0) == 0:
            lines.append(f"`{label}`: no candidates")
            continue
        if b.get("resolved", 0) == 0:
            lines.append(
                f"`{label}`: `{b['candidates']}` cand "
                f"(`{b['selected']}` sel), 0 resolved")
            continue
        lines.append(
            f"`{label}`: `{b['candidates']}` cand "
            f"(`{b['selected']}` sel, `{b['resolved']}` res), "
            f"WR `{_pct(b['win_rate'])}`, PF `{eva._pf_str(b['profit_factor'])}`, "
            f"avg `{eva._money(b['avg_pnl'])}`, move `{_move_str(b['avg_actual_move'])}`")
    lines += [
        "",
        f"*Separation score:* `{_sep_str(report.get('separation_score'))}`",
        f"*Verdict:* `{report['verdict']}`",
        f"Candidates: `{report['candidates']}` · "
        f"scored: `{report['scored']}` · resolved: `{report['resolved']}`",
        "", footer,
    ]
    return "\n".join(lines)


def format_signal_separation(report: dict) -> str:
    """Telegram-ready SIGNAL_SEPARATION. Pure formatting."""
    header = "🧭 *Signal Separation* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            f"_{SIGNAL_SEP_QUESTION}_",
            "",
            "No resolved candidates yet.",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`",
            "", footer,
        ])
    lines = [
        header, "",
        f"_{SIGNAL_SEP_QUESTION}_",
        "",
        "*PF separation by signal (higher = better):*",
    ]
    sep = report["separation_score_by_signal"]
    rankings = report["rankings"]
    ordered = sorted(
        sep.items(),
        key=lambda kv: (kv[1] is not None, kv[1] if kv[1] is not None else 0.0),
        reverse=True)
    for name, value in ordered:
        pred = rankings.get(name) or {}
        lines.append(
            f"`{name}`: separation `{_sep_str(value)}` "
            f"(`{pred.get('verdict', eva.VERDICT_INCONCLUSIVE)}`, "
            f"{pred.get('buckets_with_data', 0)} buckets)")
    lines += [
        "",
        f"*Best signal:* `{report.get('best_predictive_signal') or 'n/a'}`",
        f"*Weakest signal:* `{report.get('weakest_predictive_signal') or 'n/a'}`",
        f"*Verdict:* `{report['overall_verdict']}`",
        f"Sample size: `{report['sample_size']}`",
        "", footer,
    ]
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level Telegram entry points
# ---------------------------------------------------------------------------
def generate_triple_gap_report_text(jsonl_path: Optional[str] = None) -> str:
    """Top-level entry for the TRIPLE_GAP_REPORT Telegram command."""
    return format_triple_gap_report(
        compute_triple_gap_report(jsonl_path=jsonl_path))


def generate_signal_separation_text(jsonl_path: Optional[str] = None) -> str:
    """Top-level entry for the SIGNAL_SEPARATION Telegram command."""
    return format_signal_separation(
        compute_signal_separation(jsonl_path=jsonl_path))
