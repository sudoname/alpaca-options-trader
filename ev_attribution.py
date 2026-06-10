"""
Phase 10E — EV Attribution (analytics only, additive, fail-open).

Answers: *do the EV beliefs Oracle held at entry actually predict outcomes?*

Reads CLOSED paper spread trades and buckets them by the beliefs that were
frozen at open (Phase 10C/10D — never recomputed after close):

    A. Expected Value          D. Oracle Score
    B. EV per dollar of risk   E. Volatility Edge
    C. Probability of Profit   F. Advisory Recommendation

and reports, per bucket: trades, wins, losses, win rate, total / average PnL,
profit factor, max loss observed and average return on risk. For the EV and
EV/Risk dimensions it adds predictiveness checks (monotonicity, best / worst
bucket, separation score) so the EV_ATTRIBUTION Telegram command can answer
"does higher EV produce better outcomes?" directly.

Data sources (merged by trade id, fail-open):
  * ``spread_paper_trades.json``   — closed simulated spreads (has max_loss,
    plus the EV stamp when opened by the Phase 10D Best-EV paper runner).
  * ``advisory_attribution.json``  — entry-time advisory + EV snapshots with
    realized outcomes appended at close (Phase 9C/10C).
  ``learning_validation.csv`` is NOT consumed: it holds shadow
  recommendations without realized-EV linkage, so it adds no closed-trade
  outcome rows to this analysis.

STRICTLY analytics: this module never opens, closes, sizes, blocks or alters
any real or paper trade, never imports the live trader and never touches the
network. Every reader fails open (missing / malformed -> empty report).
"""

from typing import Callable, List, Optional, Sequence, Tuple

import advisory_attribution as aa
import advisory_gate as ag
import oracle_analytics as oa
import threshold_engine as te
from oracle_analytics import AnalyticsConfig

ANALYTICS_FOOTER = "Analytics only — no trading decisions changed."

# Predictiveness verdicts.
VERDICT_YES = "YES"
VERDICT_NO = "NO"
VERDICT_INCONCLUSIVE = "INCONCLUSIVE"

# Profit factor of inf (wins, no losses) is capped for scoring arithmetic.
PF_CAP = 99.99

# ---------------------------------------------------------------------------
# Bucket definitions: (label, lo, hi) over half-open intervals [lo, hi);
# None = unbounded. Listed in ascending order so monotonicity is meaningful.
# ---------------------------------------------------------------------------
EV_BUCKETS = (
    ("EV < 0", None, 0.0),
    ("EV 0-10", 0.0, 10.0),
    ("EV 10-20", 10.0, 20.0),
    ("EV 20-50", 20.0, 50.0),
    ("EV 50+", 50.0, None),
)

EV_RISK_BUCKETS = (
    ("EV/Risk < 0", None, 0.0),
    ("EV/Risk 0-0.05", 0.0, 0.05),
    ("EV/Risk 0.05-0.10", 0.05, 0.10),
    ("EV/Risk 0.10-0.20", 0.10, 0.20),
    ("EV/Risk 0.20+", 0.20, None),
)

POP_BUCKETS = (
    ("PoP <50%", None, 0.50),
    ("PoP 50-60%", 0.50, 0.60),
    ("PoP 60-70%", 0.60, 0.70),
    ("PoP 70-80%", 0.70, 0.80),
    ("PoP 80%+", 0.80, None),
)

ORACLE_BUCKETS = (
    ("Oracle 0-39", None, 40.0),
    ("Oracle 40-59", 40.0, 60.0),
    ("Oracle 60-79", 60.0, 80.0),
    ("Oracle 80-100", 80.0, None),
)

# Volatility edge is stored as a fraction (0.012 = 1.2%); bucket in percent.
VOL_EDGE_BUCKETS = (
    ("Edge <0%", None, 0.0),
    ("Edge 0-1%", 0.0, 1.0),
    ("Edge 1-2%", 1.0, 2.0),
    ("Edge 2-3%", 2.0, 3.0),
    ("Edge 3%+", 3.0, None),
)

# Advisory recommendation: categorical; ascending order (worst -> best) used
# for predictiveness, display order per spec is best -> worst.
ADVISORY_ORDER = (
    ag.REJECT_CANDIDATE, ag.WEAK_SETUP, ag.NEUTRAL, ag.ACCEPT, ag.STRONG_ACCEPT,
)


# ---------------------------------------------------------------------------
# Value extractors (None when the record doesn't carry the feature)
# ---------------------------------------------------------------------------
def _ev(row: dict):
    return oa._to_float(row.get("expected_value"))


def _ev_risk(row: dict):
    return oa._to_float(row.get("ev_per_dollar_risk"))


def _pop(row: dict):
    return oa._to_float(row.get("probability_of_profit"))


def _oracle(row: dict):
    return oa._trade_oracle(row)


def _edge_pct(row: dict):
    edge = oa._trade_edge(row)
    return edge * 100.0 if edge is not None else None


# ---------------------------------------------------------------------------
# Closed-record loading (merge trades file + attribution snapshots by id)
# ---------------------------------------------------------------------------
def _rid(row: dict):
    return row.get("trade_id") or row.get("id") or row.get("order_id")


def load_closed_records(config: Optional[AnalyticsConfig] = None,
                        attribution_path: Optional[str] = None,
                        trades: Optional[List[dict]] = None,
                        snapshots: Optional[List[dict]] = None) -> List[dict]:
    """Closed paper-spread records with their frozen entry-time beliefs.

    Trade rows (have max_loss, and the EV stamp when runner-opened) are merged
    with attribution snapshots (have advisory_recommendation + EV at entry) by
    trade id: the snapshot fills any field the trade row is missing, and the
    snapshot's advisory_recommendation wins (it is the frozen entry belief).
    Snapshot-only records (e.g. rotated trades file) still count. Fail-open.
    """
    cfg = config or AnalyticsConfig.from_env()
    if trades is None:
        data = oa.read_json(cfg.spread_trades_file)
        trades = data if isinstance(data, list) else []
    if snapshots is None:
        snapshots = aa.load_snapshots(attribution_path)

    merged = {}
    order: List[str] = []
    for row in trades:
        if not isinstance(row, dict) or oa._trade_pnl(row) is None:
            continue
        key = str(_rid(row) or f"trade#{len(order)}")
        merged[key] = dict(row)
        order.append(key)
    for i, snap in enumerate(snapshots or []):
        if not isinstance(snap, dict) or oa._trade_pnl(snap) is None:
            continue
        key = str(_rid(snap) or f"snap#{i}")
        if key in merged:
            rec = merged[key]
            for field, value in snap.items():
                if rec.get(field) is None and value is not None:
                    rec[field] = value
            if snap.get("advisory_recommendation") is not None:
                rec["advisory_recommendation"] = snap["advisory_recommendation"]
        else:
            merged[key] = dict(snap)
            order.append(key)
    return [merged[k] for k in order]


# ---------------------------------------------------------------------------
# Per-bucket statistics
# ---------------------------------------------------------------------------
def bucket_label(value, buckets) -> Optional[str]:
    """Label of the half-open [lo, hi) bucket containing value, else None."""
    v = oa._to_float(value)
    if v is None:
        return None
    for label, lo, hi in buckets:
        if (lo is None or v >= lo) and (hi is None or v < hi):
            return label
    return None


def bucket_stats(rows: List[dict]) -> dict:
    """The Phase 10E stat block over a list of closed records."""
    agg = oa._aggregate(rows)
    n, wins, total = agg["trades"], agg["wins"], agg["pnl"]
    worst = 0.0
    rors = []
    for row in rows:
        pnl = oa._trade_pnl(row)
        if pnl is not None and pnl < 0:
            worst = min(worst, pnl)
        risk = oa._to_float(row.get("max_loss"))
        if pnl is not None and risk is not None and risk > 0:
            rors.append(pnl / risk)
    return {
        "trades": n,
        "wins": wins,
        "losses": n - wins,
        "win_rate": agg["win_rate"],
        "total_pnl": total,
        "average_pnl": round(total / n, 2) if n else 0.0,
        "profit_factor": te._profit_factor(rows),
        "max_loss_observed": round(abs(worst), 2),
        "average_return_on_risk":
            round(sum(rors) / len(rors), 4) if rors else None,
    }


def compute_bucket_table(rows: List[dict], value_fn: Callable[[dict], object],
                         buckets) -> dict:
    """{label: bucket_stats} in bucket order; rows without the feature drop."""
    groups = {label: [] for label, _, _ in buckets}
    for row in rows:
        label = bucket_label(value_fn(row), buckets)
        if label is not None:
            groups[label].append(row)
    return {label: bucket_stats(group) for label, group in groups.items()}


def compute_category_table(rows: List[dict], key: str,
                           categories: Sequence[str]) -> dict:
    """{category: bucket_stats} for a categorical feature (e.g. advisory)."""
    return {cat: bucket_stats([r for r in rows if r.get(key) == cat])
            for cat in categories}


# ---------------------------------------------------------------------------
# Predictiveness checks (monotonicity / best / worst / separation)
# ---------------------------------------------------------------------------
def _pf_measure(stats: dict) -> Optional[float]:
    pf = stats.get("profit_factor")
    if pf is None:
        return None
    if pf == float("inf"):
        return PF_CAP
    return float(pf)


def compute_predictiveness(table: dict, order: Sequence[str]) -> dict:
    """Does the higher-ordered bucket outperform the lower-ordered one?

    Measured on profit factor (inf capped at PF_CAP) over OCCUPIED buckets:
      * monotonicity — fraction of adjacent occupied pairs where the higher
        bucket's PF >= the lower bucket's PF (1.0 = perfectly ordered).
      * best_bucket / worst_bucket — highest / lowest PF among occupied.
      * separation — PF of the top-ordered occupied bucket minus PF of the
        bottom-ordered occupied bucket (e.g. 2.4 - 0.6 = 1.8).
      * verdict — YES (separation > 0 and monotonicity >= 0.5),
        NO (separation < 0), else INCONCLUSIVE (incl. <2 occupied buckets).
    """
    occupied: List[Tuple[str, float]] = []
    for label in order:
        stats = table.get(label) or {}
        measure = _pf_measure(stats)
        if stats.get("trades", 0) > 0 and measure is not None:
            occupied.append((label, measure))

    out = {
        "buckets_with_data": len(occupied),
        "monotonicity": None,
        "best_bucket": None,
        "worst_bucket": None,
        "separation": None,
        "verdict": VERDICT_INCONCLUSIVE,
    }
    if not occupied:
        return out
    out["best_bucket"] = max(occupied, key=lambda x: x[1])[0]
    out["worst_bucket"] = min(occupied, key=lambda x: x[1])[0]
    if len(occupied) < 2:
        return out

    pairs = list(zip(occupied, occupied[1:]))
    rising = sum(1 for lo, hi in pairs if hi[1] >= lo[1])
    out["monotonicity"] = round(rising / len(pairs), 2)
    out["separation"] = round(occupied[-1][1] - occupied[0][1], 2)
    if out["separation"] > 0 and out["monotonicity"] >= 0.5:
        out["verdict"] = VERDICT_YES
    elif out["separation"] < 0:
        out["verdict"] = VERDICT_NO
    return out


# ---------------------------------------------------------------------------
# Full report
# ---------------------------------------------------------------------------
def compute_ev_attribution(records: Optional[List[dict]] = None,
                           config: Optional[AnalyticsConfig] = None,
                           attribution_path: Optional[str] = None,
                           trades: Optional[List[dict]] = None,
                           snapshots: Optional[List[dict]] = None) -> dict:
    """All six bucket tables + EV / EV-Risk predictiveness. Never raises."""
    if records is None:
        records = load_closed_records(config=config,
                                      attribution_path=attribution_path,
                                      trades=trades, snapshots=snapshots)
    ev_table = compute_bucket_table(records, _ev, EV_BUCKETS)
    ev_risk_table = compute_bucket_table(records, _ev_risk, EV_RISK_BUCKETS)
    return {
        "sample_size": len(records),
        "confidence": te.compute_confidence(len(records)),
        "ev_buckets": ev_table,
        "ev_risk_buckets": ev_risk_table,
        "pop_buckets": compute_bucket_table(records, _pop, POP_BUCKETS),
        "oracle_buckets": compute_bucket_table(records, _oracle,
                                               ORACLE_BUCKETS),
        "vol_edge_buckets": compute_bucket_table(records, _edge_pct,
                                                 VOL_EDGE_BUCKETS),
        "advisory_buckets": compute_category_table(
            records, "advisory_recommendation", ADVISORY_ORDER),
        "ev_predictiveness": compute_predictiveness(
            ev_table, [b[0] for b in EV_BUCKETS]),
        "ev_risk_predictiveness": compute_predictiveness(
            ev_risk_table, [b[0] for b in EV_RISK_BUCKETS]),
    }


# ---------------------------------------------------------------------------
# Telegram formatting
# ---------------------------------------------------------------------------
def _money(value) -> str:
    v = oa._to_float(value)
    if v is None:
        return "n/a"
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _pf_str(pf) -> str:
    if pf is None:
        return "n/a"
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def _bucket_line(label: str, m: dict) -> str:
    return (f"`{label}` — {m['trades']} trades, "
            f"WR {m['win_rate'] * 100:.0f}%, "
            f"PF {_pf_str(m['profit_factor'])}, "
            f"PnL {_money(m['total_pnl'])}")


def _table_lines(title: str, table: dict, order: Sequence[str]) -> List[str]:
    lines = [f"*{title}*"]
    seen = False
    for label in order:
        m = table.get(label) or {}
        if m.get("trades", 0) == 0:
            continue
        lines.append(_bucket_line(label, m))
        seen = True
    if not seen:
        lines.append("_no data_")
    return lines


def _predictiveness_lines(title: str, p: dict) -> List[str]:
    if p.get("buckets_with_data", 0) < 2:
        return [f"{title}: insufficient bucket coverage — "
                f"*{VERDICT_INCONCLUSIVE}*"]
    mono = p.get("monotonicity")
    return [
        f"{title}:",
        f"Best bucket: `{p['best_bucket']}` · "
        f"Worst bucket: `{p['worst_bucket']}`",
        f"Monotonicity: `{mono * 100:.0f}%` · "
        f"Separation: `{p['separation']:+.2f}`",
        f"Higher buckets outperform lower buckets: `{p['verdict']}`",
    ]


def format_ev_attribution(report: dict) -> str:
    """Telegram-ready EV_ATTRIBUTION summary. Pure formatting."""
    header = "📐 *EV Attribution* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if report.get("sample_size", 0) == 0:
        return "\n".join([
            header, "",
            "No closed paper spread trades with EV/advisory context yet.",
            "", footer,
        ])

    lines = [header, ""]
    lines += _table_lines("Expected Value buckets:", report["ev_buckets"],
                          [b[0] for b in EV_BUCKETS])
    lines += [""]
    lines += _table_lines("EV/Risk buckets:", report["ev_risk_buckets"],
                          [b[0] for b in EV_RISK_BUCKETS])
    lines += ["", "*Predictiveness:*"]
    lines += _predictiveness_lines("EV", report["ev_predictiveness"])
    lines += _predictiveness_lines("EV/Risk", report["ev_risk_predictiveness"])
    lines += [
        "",
        f"Sample size: `{report['sample_size']}` · "
        f"Confidence: *{report['confidence']}*",
        "", footer,
    ]
    return "\n".join(lines)


def generate_ev_attribution_text(config: Optional[AnalyticsConfig] = None,
                                 attribution_path: Optional[str] = None) -> str:
    """Top-level entry for the EV_ATTRIBUTION Telegram command."""
    report = compute_ev_attribution(config=config,
                                    attribution_path=attribution_path)
    return format_ev_attribution(report)
