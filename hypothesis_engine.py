"""
Phase 8E — Automatic hypothesis testing (advisory, read-only, offline-pure).

This module answers: *which Oracle Score / Volatility Edge / DTE / IV Rank /
Strategy choices actually produced better simulated results?* It does so by
splitting the CLOSED simulated spread book into A/B groups and comparing their
performance for a fixed catalogue of hypotheses (Phase 8E Req 4):

    A. Oracle Score   — >=80 vs 60-79, >=70 vs <70, >=80 vs <80
    B. Volatility Edge — >=3% vs 1-3%, >=2% vs <2%
    C. DTE            — 31-60 vs 15-30, 31-60 vs all others
    D. IV Rank        — 50-75 vs 75-100, 50-75 vs all others
    E. Strategy       — Bull Put vs Iron Condor, Bear Call vs Iron Condor,
                        Credit Spreads vs Debit Spreads

It reads only ``spread_paper_trades.json`` (via :mod:`oracle_analytics`) and
reuses :func:`threshold_engine._profit_factor` so the metrics stay consistent
with the rest of the analytics stack. It is STRICTLY advisory: it contains no
order placement, no spread execution, no live-trading or gating logic — nothing
here can open, modify, gate, or close any real or paper position. It only
describes what the historical data says. Every reader fails open (missing /
empty / malformed → "no data") and every public function returns a plain dict /
list that is safe to format even when inputs are empty.

Public API (all accept an optional ``config`` and optional pre-loaded ``trades``
so they are trivially unit-testable, with no network or credentials):

    evaluate_hypothesis(spec, trades)   -> result dict
    compute_all_hypotheses()            -> list[result dict]
    rank_hypotheses(results)            -> list sorted by strongest improvement
    format_hypothesis_report(results)   -> Telegram-formatted string
    generate_hypothesis_report_text()   -> compute + rank + format in one call
    hypothesis_confidence(n_a, n_b)     -> 'Low' / 'Medium' / 'High'
"""

import logging
from typing import List, Optional

import oracle_analytics as oa
import threshold_engine as te
from oracle_analytics import AnalyticsConfig

try:  # canonical strategy names (best-effort; data-driven either way)
    from spread_builder import (
        BEARISH_CALL_CREDIT_SPREAD, BULLISH_PUT_CREDIT_SPREAD,
        DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR,
    )
except Exception:  # pragma: no cover - defensive
    BULLISH_PUT_CREDIT_SPREAD = "bullish_put_credit_spread"
    BEARISH_CALL_CREDIT_SPREAD = "bearish_call_credit_spread"
    IRON_CONDOR = "iron_condor"
    DEBIT_CALL_SPREAD = "debit_call_spread"
    DEBIT_PUT_SPREAD = "debit_put_spread"

logger = logging.getLogger(__name__)

CREDIT_SPREADS = (BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD)
DEBIT_SPREADS = (DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD)

ADVISORY_FOOTER = "Advisory only — no thresholds changed."

# Sample-size confidence boundaries (Req 4):
#   Low    if either group has < 30 trades
#   High   if both groups have > 100 trades
#   Medium otherwise (both >= 30, not both > 100)
CONFIDENCE_MIN = 30
CONFIDENCE_HIGH = 100


# --------------------------------------------------------------------------- #
# Hypothesis catalogue
# --------------------------------------------------------------------------- #
def _strategy_of(t: dict) -> str:
    return str(t.get("strategy") or "unknown")


# Each spec: (name, group_a_label, group_b_label, value_fn, pred_a, pred_b).
# value_fn extracts the comparison value; a trade joins group A if pred_a(value)
# is True, else group B if pred_b(value) is True, else it is ignored (skipped
# when the value is missing). Edges are fractions (0.03 == 3%).
HYPOTHESES = [
    # A. Oracle Score
    ("Oracle Score >= 80 vs 60-79", "Score >= 80", "Score 60-79",
     oa._trade_oracle, lambda v: v >= 80, lambda v: 60 <= v < 80),
    ("Oracle Score >= 70 vs < 70", "Score >= 70", "Score < 70",
     oa._trade_oracle, lambda v: v >= 70, lambda v: v < 70),
    ("Oracle Score >= 80 vs < 80", "Score >= 80", "Score < 80",
     oa._trade_oracle, lambda v: v >= 80, lambda v: v < 80),
    # B. Volatility Edge
    ("Vol Edge >= 3% vs 1-3%", "Edge >= 3%", "Edge 1-3%",
     oa._trade_edge, lambda v: v >= 0.03, lambda v: 0.01 <= v < 0.03),
    ("Vol Edge >= 2% vs < 2%", "Edge >= 2%", "Edge < 2%",
     oa._trade_edge, lambda v: v >= 0.02, lambda v: v < 0.02),
    # C. DTE
    ("DTE 31-60 vs 15-30", "DTE 31-60", "DTE 15-30",
     oa._trade_dte, lambda v: 31 <= v <= 60, lambda v: 15 <= v <= 30),
    ("DTE 31-60 vs all others", "DTE 31-60", "DTE other",
     oa._trade_dte, lambda v: 31 <= v <= 60, lambda v: not (31 <= v <= 60)),
    # D. IV Rank
    ("IV Rank 50-75 vs 75-100", "IV 50-75", "IV 75-100",
     oa._trade_iv_rank, lambda v: 50 <= v < 75, lambda v: 75 <= v <= 100),
    ("IV Rank 50-75 vs all others", "IV 50-75", "IV other",
     oa._trade_iv_rank, lambda v: 50 <= v < 75, lambda v: not (50 <= v < 75)),
    # E. Strategy
    ("Bull Put Credit Spread vs Iron Condor", "Bull Put", "Iron Condor",
     _strategy_of, lambda v: v == BULLISH_PUT_CREDIT_SPREAD, lambda v: v == IRON_CONDOR),
    ("Bear Call Credit Spread vs Iron Condor", "Bear Call", "Iron Condor",
     _strategy_of, lambda v: v == BEARISH_CALL_CREDIT_SPREAD, lambda v: v == IRON_CONDOR),
    ("Credit Spreads vs Debit Spreads", "Credit Spreads", "Debit Spreads",
     _strategy_of, lambda v: v in CREDIT_SPREADS, lambda v: v in DEBIT_SPREADS),
]


# --------------------------------------------------------------------------- #
# Confidence + comparison
# --------------------------------------------------------------------------- #
def hypothesis_confidence(n_a: int, n_b: int) -> str:
    """'Low' (<30 either side), 'High' (>100 both), else 'Medium'."""
    n_a = n_a or 0
    n_b = n_b or 0
    if n_a < CONFIDENCE_MIN or n_b < CONFIDENCE_MIN:
        return "Low"
    if n_a > CONFIDENCE_HIGH and n_b > CONFIDENCE_HIGH:
        return "High"
    return "Medium"


def _perf_key(pf, avg_pnl, pnl):
    """Ranking tuple: profit factor (None -> -inf), then avg PnL, then PnL."""
    p = pf if pf is not None else float("-inf")
    return (p, avg_pnl, pnl)


def _conclude(trades_a, trades_b, key_a, key_b) -> str:
    """'A outperformed B' / 'B outperformed A' / 'Inconclusive'.

    Inconclusive when either group is empty or the two perform identically.
    """
    if trades_a == 0 or trades_b == 0:
        return "Inconclusive"
    if key_a > key_b:
        return "A outperformed B"
    if key_b > key_a:
        return "B outperformed A"
    return "Inconclusive"


# --------------------------------------------------------------------------- #
# Evaluate a single hypothesis
# --------------------------------------------------------------------------- #
def _group_stats(trades: List[dict]) -> dict:
    agg = oa._aggregate(trades)
    n = agg["trades"]
    return {
        "trades": n,
        "win_rate": agg["win_rate"],
        "pnl": agg["pnl"],
        "avg_pnl": round(agg["pnl"] / n, 2) if n else 0.0,
        "profit_factor": te._profit_factor(trades),
    }


def evaluate_hypothesis(spec, trades: List[dict]) -> dict:
    """Split ``trades`` into A/B per ``spec`` and compare their performance."""
    name, label_a, label_b, value_fn, pred_a, pred_b = spec
    group_a: List[dict] = []
    group_b: List[dict] = []
    for t in trades:
        v = value_fn(t)
        if v is None:
            continue
        try:
            if pred_a(v):
                group_a.append(t)
            elif pred_b(v):
                group_b.append(t)
        except Exception:  # pragma: no cover - predicate safety
            continue

    a = _group_stats(group_a)
    b = _group_stats(group_b)
    key_a = _perf_key(a["profit_factor"], a["avg_pnl"], a["pnl"])
    key_b = _perf_key(b["profit_factor"], b["avg_pnl"], b["pnl"])
    conclusion = _conclude(a["trades"], b["trades"], key_a, key_b)

    return {
        "hypothesis_name": name,
        "group_a": label_a,
        "group_b": label_b,
        "trades_a": a["trades"],
        "trades_b": b["trades"],
        "win_rate_a": a["win_rate"],
        "win_rate_b": b["win_rate"],
        "pnl_a": a["pnl"],
        "pnl_b": b["pnl"],
        "avg_pnl_a": a["avg_pnl"],
        "avg_pnl_b": b["avg_pnl"],
        "profit_factor_a": a["profit_factor"],
        "profit_factor_b": b["profit_factor"],
        "conclusion": conclusion,
        "confidence": hypothesis_confidence(a["trades"], b["trades"]),
        # signed effect on average PnL (A minus B); magnitude drives ranking.
        "effect_size": round(a["avg_pnl"] - b["avg_pnl"], 2),
    }


def compute_all_hypotheses(config: Optional[AnalyticsConfig] = None,
                           trades: Optional[List[dict]] = None) -> List[dict]:
    """Evaluate every catalogued hypothesis over the closed simulated book."""
    config = config or AnalyticsConfig.from_env()
    closed = oa.load_closed_spread_trades(config, trades)
    return [evaluate_hypothesis(spec, closed) for spec in HYPOTHESES]


def rank_hypotheses(results: List[dict]) -> List[dict]:
    """Sort findings strongest-first: conclusive ones, by |effect size| desc.

    Inconclusive hypotheses (empty groups or ties) sort to the bottom.
    """
    def key(r):
        conclusive = 1 if r["conclusion"] != "Inconclusive" else 0
        return (conclusive, abs(r.get("effect_size") or 0.0))

    return sorted(results, key=key, reverse=True)


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _fmt_pf(pf) -> str:
    if pf is None:
        return "n/a"
    if pf == float("inf"):
        return "∞"
    return f"{pf:.2f}"


def format_hypothesis_report(results: List[dict], top_n: int = 5) -> str:
    """Render ranked hypotheses into Telegram markdown (top findings first)."""
    ranked = rank_hypotheses(results)
    conclusive = [r for r in ranked if r["conclusion"] != "Inconclusive"]
    shown = (conclusive or ranked)[:top_n]

    lines = ["🔬 *Hypothesis Report* _(advisory)_", ""]
    if not any(r["trades_a"] or r["trades_b"] for r in results):
        lines.append("📭 No closed paper spreads yet — nothing to test.")
        lines.append("")
        lines.append(f"_({ADVISORY_FOOTER})_")
        return "\n".join(lines)

    if not conclusive:
        lines.append("No conclusive findings yet (groups too thin or tied).")
        lines.append("")

    for r in shown:
        lines.append(f"*{r['hypothesis_name']}*")
        lines.append(f"  Result: `{r['conclusion']}` · confidence `{r['confidence']}`")
        lines.append(
            f"  {r['group_a']}: `{r['trades_a']}` trades · "
            f"`{r['win_rate_a'] * 100:.0f}%` win · `${r['pnl_a']:+.2f}` "
            f"(PF {_fmt_pf(r['profit_factor_a'])})")
        lines.append(
            f"  {r['group_b']}: `{r['trades_b']}` trades · "
            f"`{r['win_rate_b'] * 100:.0f}%` win · `${r['pnl_b']:+.2f}` "
            f"(PF {_fmt_pf(r['profit_factor_b'])})")
        lines.append("")

    lines.append(f"_({ADVISORY_FOOTER})_")
    return "\n".join(lines)


def generate_hypothesis_report_text(config: Optional[AnalyticsConfig] = None,
                                    trades: Optional[List[dict]] = None,
                                    top_n: int = 5) -> str:
    """Compute + rank + format the hypothesis report in one call."""
    return format_hypothesis_report(
        compute_all_hypotheses(config=config, trades=trades), top_n=top_n)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; synthetic data only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/hypo_trades.json")

    # --- empty data: every hypothesis is Inconclusive, footer present ---
    results = compute_all_hypotheses(cfg)
    if len(results) != len(HYPOTHESES):
        print("FAIL: hypothesis count", len(results)); ok = False
    if any(r["conclusion"] != "Inconclusive" for r in results):
        print("FAIL: empty should be inconclusive"); ok = False
    txt = format_hypothesis_report(results)
    if ADVISORY_FOOTER not in txt or "Hypothesis Report" not in txt:
        print("FAIL: empty report text"); ok = False

    # --- confidence levels ---
    if hypothesis_confidence(10, 200) != "Low":
        print("FAIL: confidence low"); ok = False
    if hypothesis_confidence(50, 60) != "Medium":
        print("FAIL: confidence medium"); ok = False
    if hypothesis_confidence(150, 200) != "High":
        print("FAIL: confidence high"); ok = False

    # --- synthetic: high scores win, low scores lose ---
    trades = [
        {"strategy": "bullish_put_credit_spread", "status": "closed",
         "oracle_score": 85, "volatility_edge": 0.035, "pnl": 120.0,
         "dte": 35, "iv_rank": 60},
        {"strategy": "bullish_put_credit_spread", "status": "closed",
         "oracle_score": 82, "volatility_edge": 0.03, "pnl": 90.0,
         "dte": 40, "iv_rank": 55},
        {"strategy": "iron_condor", "status": "closed",
         "oracle_score": 65, "volatility_edge": 0.015, "pnl": -100.0,
         "dte": 20, "iv_rank": 80},
    ]
    results = compute_all_hypotheses(cfg, trades=trades)
    by_name = {r["hypothesis_name"]: r for r in results}

    h = by_name["Oracle Score >= 80 vs 60-79"]
    if h["trades_a"] != 2 or h["trades_b"] != 1:
        print("FAIL: score split", h); ok = False
    if h["conclusion"] != "A outperformed B":
        print("FAIL: score conclusion", h["conclusion"]); ok = False

    h = by_name["Bull Put Credit Spread vs Iron Condor"]
    if h["trades_a"] != 2 or h["trades_b"] != 1:
        print("FAIL: strategy split", h); ok = False
    if h["conclusion"] != "A outperformed B":
        print("FAIL: strategy conclusion", h["conclusion"]); ok = False

    # ranking puts a conclusive finding first
    ranked = rank_hypotheses(results)
    if ranked[0]["conclusion"] == "Inconclusive":
        print("FAIL: ranking conclusive-first"); ok = False

    txt = format_hypothesis_report(results)
    if "outperformed" not in txt:
        print("FAIL: sample report text"); ok = False

    print("hypothesis_engine self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
