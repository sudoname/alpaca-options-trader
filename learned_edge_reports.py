"""
P13E — Learned-edge calibration analytics (compute / format / generate trios).

Three Telegram-ready, analytics-only reports built on the P13 engine:

  * LEARNED_EDGE_REPORT      — the global prior, how many setup keys carry enough
                               evidence, and the strongest / weakest setups by
                               Bayesian-smoothed edge.
  * ORACLE_SCORE_COMPARISON  — replays the frozen history under the live Oracle
                               (v1), EV-first and learned rankings and tabulates
                               win rate, avg return, profit factor, max drawdown
                               and expectancy (delegates to ``shadow_ranking``).
  * LEARNED_EDGE_LEADERBOARD — setups ranked by sample-size-adjusted expected
                               profitability (confidence x smoothed avg return).

STRICTLY analytics: these only read and report. Nothing here opens, closes, sizes,
prices, blocks or alters any real or paper trade, mutates a Q-table, or reaches the
network beyond the fail-open loaders. Every reader degrades to INSUFFICIENT_DATA on
empty/malformed input and every formatter ends with the shared analytics footer.
Records are injectable for deterministic testing.
"""

from typing import Dict, List, Optional

import learned_edge as le
import shadow_ranking as sr
import ev_attribution as eva
from ev_attribution import ANALYTICS_FOOTER

# Minimum closed records before any verdict beyond INSUFFICIENT_DATA.
MIN_SAMPLES = 10
VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# How many setups to surface in the top/bottom and leaderboard lists.
TOP_N = 8


def _pct(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _ret(value) -> str:
    return f"{value:+.1f}%" if value is not None else "n/a"


# --------------------------------------------------------------------------- #
# LEARNED_EDGE_REPORT
# --------------------------------------------------------------------------- #
def compute_learned_edge_report(records: Optional[List[dict]] = None,
                                config: Optional[le.LearnedEdgeConfig] = None
                                ) -> dict:
    """Global prior + per-setup smoothed edges. Never raises."""
    try:
        cfg = config or le.LearnedEdgeConfig.from_env()
        if records is None:
            records = le.load_edge_records(cfg)
        records = records or []
        n_total = len(records)
        prior = le.compute_prior(records)
        index = le.build_edge_index(records)

        rows = []
        confident = 0
        full = 0
        for tup, stats in index.items():
            n = stats["trades"]
            if n >= cfg.backoff_min_samples:
                confident += 1
            if n >= cfg.min_samples_full:
                full += 1
            if n < cfg.backoff_min_samples:
                continue
            smoothed = le._smooth(stats, prior, cfg)
            rows.append({
                "key_str": stats["key_str"],
                "sample_size": n,
                "learned_edge_score": smoothed["learned_edge_score"],
                "win_rate": smoothed["win_rate"],
                "avg_return": smoothed["avg_return"],
                "ci_low": smoothed["ci_low"],
                "ci_high": smoothed["ci_high"],
                "confidence_score": smoothed["confidence_score"],
            })
        rows.sort(key=lambda r: r["learned_edge_score"], reverse=True)

        return {
            "sample_size": n_total,
            "num_keys": len(index),
            "num_confident_keys": confident,
            "num_full_keys": full,
            "prior": prior,
            "top_setups": rows[:TOP_N],
            "bottom_setups": list(reversed(rows[-TOP_N:])) if rows else [],
            "verdict": VERDICT_OK if n_total >= MIN_SAMPLES
            else VERDICT_INSUFFICIENT,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "num_keys": 0, "num_confident_keys": 0,
                "num_full_keys": 0, "prior": le.compute_prior([]),
                "top_setups": [], "bottom_setups": [],
                "verdict": VERDICT_INSUFFICIENT}


def format_learned_edge_report(report: dict) -> str:
    header = "🧠 *Learned Edge Report* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if not report or report.get("sample_size", 0) == 0:
        return "\n".join([header, "", "No closed setups recorded yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", footer])
    prior = report.get("prior", {})
    lines = [
        header, "",
        f"_Historical edge of each setup (Bayesian-smoothed)._", "",
        f"*Global prior:* WR `{_pct(prior.get('win_rate'))}`, "
        f"avg `{_ret(prior.get('avg_return'))}`, "
        f"n `{prior.get('n', 0)}`",
        f"*Setup keys:* `{report['num_keys']}` "
        f"(`{report['num_confident_keys']}` with enough evidence, "
        f"`{report['num_full_keys']}` full)",
        "",
        "*Strongest setups (smoothed edge / WR / n):*",
    ]
    if report["top_setups"]:
        for r in report["top_setups"]:
            lines.append(
                f"`{r['key_str']}` -> edge `{r['learned_edge_score']:.2f}`, "
                f"WR `{_pct(r['win_rate'])}`, n `{r['sample_size']}`")
    else:
        lines.append("_No setup has enough evidence yet._")
    if report["bottom_setups"]:
        lines += ["", "*Weakest setups:*"]
        for r in report["bottom_setups"]:
            lines.append(
                f"`{r['key_str']}` -> edge `{r['learned_edge_score']:.2f}`, "
                f"WR `{_pct(r['win_rate'])}`, n `{r['sample_size']}`")
    lines += ["",
              f"*Verdict:* `{report['verdict']}` · "
              f"sample `{report['sample_size']}`",
              "", footer]
    return "\n".join(lines)


def generate_learned_edge_report_text(
        config: Optional[le.LearnedEdgeConfig] = None,
        records: Optional[List[dict]] = None) -> str:
    return format_learned_edge_report(
        compute_learned_edge_report(records=records, config=config))


# --------------------------------------------------------------------------- #
# ORACLE_SCORE_COMPARISON
# --------------------------------------------------------------------------- #
def compute_oracle_score_comparison(records: Optional[List[dict]] = None,
                                    config: Optional[le.LearnedEdgeConfig] = None
                                    ) -> dict:
    """Replay the history under all three ranking systems. Never raises."""
    try:
        cfg = config or le.LearnedEdgeConfig.from_env()
        rep = sr.replay(records=records, config=cfg)
        # Pick the best system by total P/L (ties -> win rate) for the verdict.
        best = None
        best_key = (float("-inf"), float("-inf"))
        for system, blob in rep.get("systems", {}).items():
            st = blob.get("stats", {})
            key = (st.get("total_pnl", 0.0) or 0.0, st.get("win_rate", 0.0) or 0.0)
            if key > best_key:
                best_key = key
                best = system
        rep["best_system"] = best
        rep["verdict"] = (VERDICT_OK if rep.get("num_records", 0) >= MIN_SAMPLES
                          else VERDICT_INSUFFICIENT)
        return rep
    except Exception:  # pragma: no cover - fail-open
        return {"num_records": 0, "num_decision_sets": 0, "systems": {},
                "best_system": None, "verdict": VERDICT_INSUFFICIENT}


_SYSTEM_LABELS = {
    sr.RANK_ORACLE: "Oracle v1 (live)",
    sr.RANK_BEST_EV: "Best EV/Risk",
    sr.RANK_LEARNED: "Learned edge",
}


def format_oracle_score_comparison(report: dict) -> str:
    header = "⚖️ *Oracle Score Comparison* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if not report or report.get("num_records", 0) == 0:
        return "\n".join([header, "", "No replayable history yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", footer])
    lines = [
        header, "",
        f"_If each day's pick had been ranked differently…_", "",
        f"Decision sets: `{report.get('num_decision_sets', 0)}` "
        f"(records `{report.get('num_records', 0)}`)", "",
    ]
    for system in sr.RANKING_SYSTEMS:
        blob = report.get("systems", {}).get(system, {})
        st = blob.get("stats", {})
        lines.append(
            f"*{_SYSTEM_LABELS.get(system, system)}*: "
            f"WR `{_pct(st.get('win_rate'))}`, "
            f"avg `{_ret(st.get('avg_return'))}`, "
            f"PF `{eva._pf_str(st.get('profit_factor'))}`, "
            f"exp `{eva._money(st.get('expectancy'))}`, "
            f"maxDD `{eva._money(st.get('max_drawdown'))}`, "
            f"total `{eva._money(st.get('total_pnl'))}`")
    best = report.get("best_system")
    lines += ["",
              f"*Best by total P/L:* `{_SYSTEM_LABELS.get(best, best) or 'n/a'}`",
              f"*Verdict:* `{report.get('verdict')}`",
              "", footer]
    return "\n".join(lines)


def generate_oracle_score_comparison_text(
        config: Optional[le.LearnedEdgeConfig] = None,
        records: Optional[List[dict]] = None) -> str:
    return format_oracle_score_comparison(
        compute_oracle_score_comparison(records=records, config=config))


# --------------------------------------------------------------------------- #
# LEARNED_EDGE_LEADERBOARD
# --------------------------------------------------------------------------- #
def compute_learned_edge_leaderboard(records: Optional[List[dict]] = None,
                                     config: Optional[le.LearnedEdgeConfig] = None
                                     ) -> dict:
    """Rank setups by confidence x smoothed avg return. Never raises."""
    try:
        cfg = config or le.LearnedEdgeConfig.from_env()
        if records is None:
            records = le.load_edge_records(cfg)
        records = records or []
        prior = le.compute_prior(records)
        index = le.build_edge_index(records)

        rows = []
        for tup, stats in index.items():
            n = stats["trades"]
            if n < cfg.backoff_min_samples:
                continue
            smoothed = le._smooth(stats, prior, cfg)
            adj = smoothed["confidence_score"] * smoothed["avg_return"]
            rows.append({
                "key_str": stats["key_str"],
                "sample_size": n,
                "adjusted_score": round(adj, 4),
                "avg_return": smoothed["avg_return"],
                "win_rate": smoothed["win_rate"],
                "confidence_score": smoothed["confidence_score"],
            })
        rows.sort(key=lambda r: r["adjusted_score"], reverse=True)
        return {
            "sample_size": len(records),
            "leaderboard": rows[:TOP_N],
            "verdict": VERDICT_OK if len(records) >= MIN_SAMPLES
            else VERDICT_INSUFFICIENT,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "leaderboard": [],
                "verdict": VERDICT_INSUFFICIENT}


def format_learned_edge_leaderboard(report: dict) -> str:
    header = "🏆 *Learned Edge Leaderboard* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    if not report or report.get("sample_size", 0) == 0 \
            or not report.get("leaderboard"):
        return "\n".join([header, "",
                          "Not enough evidence to rank setups yet.",
                          f"*Verdict:* `{report.get('verdict', VERDICT_INSUFFICIENT)}`",
                          "", footer])
    lines = [
        header, "",
        f"_Setups by sample-size-adjusted expected profitability._", "",
    ]
    for i, r in enumerate(report["leaderboard"], 1):
        lines.append(
            f"`{i}.` `{r['key_str']}` -> "
            f"score `{r['adjusted_score']:+.2f}`, "
            f"avg `{_ret(r['avg_return'])}`, "
            f"WR `{_pct(r['win_rate'])}`, n `{r['sample_size']}`")
    lines += ["",
              f"*Verdict:* `{report['verdict']}` · "
              f"sample `{report['sample_size']}`",
              "", footer]
    return "\n".join(lines)


def generate_learned_edge_leaderboard_text(
        config: Optional[le.LearnedEdgeConfig] = None,
        records: Optional[List[dict]] = None) -> str:
    return format_learned_edge_leaderboard(
        compute_learned_edge_leaderboard(records=records, config=config))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — fully injected records)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = le.LearnedEdgeConfig(backoff_min_samples=8, min_samples_full=20)

    records = (
        [le._rec("trending", "up", 0.20, i % 5 != 0, ev_risk=0.25, rid=f"w{i}")
         for i in range(20)]          # strong cohort, 80% WR
        + [le._rec("ranging", "down", 0.10, i % 3 == 0, ev_risk=0.02,
                   rid=f"l{i}") for i in range(12)]  # weak cohort
    )

    # LEARNED_EDGE_REPORT.
    rep = compute_learned_edge_report(records=records, config=cfg)
    if rep["verdict"] != VERDICT_OK:
        print("FAIL: report verdict", rep["verdict"]); ok = False
    if rep["num_confident_keys"] < 2:
        print("FAIL: expected >=2 confident keys", rep["num_confident_keys"])
        ok = False
    txt = format_learned_edge_report(rep)
    if ANALYTICS_FOOTER not in txt or "Learned Edge Report" not in txt:
        print("FAIL: report text"); ok = False
    # The strong cohort should out-rank the weak one.
    if rep["top_setups"] and rep["bottom_setups"]:
        if rep["top_setups"][0]["learned_edge_score"] < \
                rep["bottom_setups"][0]["learned_edge_score"]:
            print("FAIL: ordering of setups"); ok = False

    # ORACLE_SCORE_COMPARISON.
    cmp = compute_oracle_score_comparison(records=records, config=cfg)
    if cmp["verdict"] != VERDICT_OK:
        print("FAIL: comparison verdict", cmp["verdict"]); ok = False
    ctxt = format_oracle_score_comparison(cmp)
    if ANALYTICS_FOOTER not in ctxt or "Oracle Score Comparison" not in ctxt:
        print("FAIL: comparison text"); ok = False

    # LEARNED_EDGE_LEADERBOARD.
    lb = compute_learned_edge_leaderboard(records=records, config=cfg)
    if not lb["leaderboard"]:
        print("FAIL: empty leaderboard"); ok = False
    ltxt = format_learned_edge_leaderboard(lb)
    if ANALYTICS_FOOTER not in ltxt or "Leaderboard" not in ltxt:
        print("FAIL: leaderboard text"); ok = False

    # Empty input -> INSUFFICIENT_DATA, clean text, never raises.
    for compute, fmt in (
        (compute_learned_edge_report, format_learned_edge_report),
        (compute_oracle_score_comparison, format_oracle_score_comparison),
        (compute_learned_edge_leaderboard, format_learned_edge_leaderboard),
    ):
        r = compute(records=[], config=cfg)
        if r["verdict"] != VERDICT_INSUFFICIENT:
            print("FAIL: empty verdict", compute.__name__, r["verdict"]); ok = False
        t = fmt(r)
        if ANALYTICS_FOOTER not in t or VERDICT_INSUFFICIENT not in t:
            print("FAIL: empty text", compute.__name__); ok = False

    # generate_*_text never raise even with no injected records (disk fail-open).
    for gen in (generate_learned_edge_report_text,
                generate_oracle_score_comparison_text,
                generate_learned_edge_leaderboard_text):
        if ANALYTICS_FOOTER not in gen(config=cfg, records=[]):
            print("FAIL: generate text", gen.__name__); ok = False

    print("learned_edge_reports self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
