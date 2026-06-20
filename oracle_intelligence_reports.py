"""
Oracle 3.0 — Intelligence-layer reports (compute / format / generate trios).

Eight Telegram-ready, analytics-only reports over the Intelligence Layer. Each
follows the repo's trio convention and ends with the shared ``ANALYTICS_FOOTER``;
each ``compute_*`` fails open to ``INSUFFICIENT_DATA`` on empty / malformed input
and never raises. Records and contexts are injectable for deterministic testing.

  * ORACLE_REGIME             — the current 8-label regime + confidence + reasons.
  * ORACLE_EXPLAIN <ticker>   — agent votes -> probability -> attribution for one
                                ticker's evidence context.
  * ORACLE_AGENT_REPORT       — per-agent hit-rate (win rate of trades on which an
                                agent expressed conviction) vs the global base rate.
  * ORACLE_PROBABILITY_REPORT — calibration (Brier score) of the model's P(call)
                                against realized wins, vs a 0.5 baseline.
  * ORACLE_FEATURE_IMPORTANCE — mean agent contribution share across closed trades.
  * ORACLE_WEIGHT_CHANGES     — adaptive voting-weight history + drift (Phase 4).
  * ORACLE_HYPOTHESIS_REPORT  — delegates to the existing hypothesis engine.
  * ORACLE_REGIME_PERFORMANCE — per-regime WR / avg P/L / PF / total via bucket_stats.

STRICTLY analytics: nothing here opens, sizes, prices, blocks or alters any real
or paper trade, mutates a Q-table, or reaches the network beyond fail-open loaders.
"""

import json
from typing import Dict, List, Optional

import ev_attribution as eva
import oracle_analytics as oa
from ev_attribution import ANALYTICS_FOOTER

import oracle_regime as orr
import oracle_agents as oag
import oracle_voting as ovo
import oracle_explain as oex

MIN_SAMPLES = 10
TOP_N = 8
VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"


# --------------------------------------------------------------------------- #
# Shared helpers
# --------------------------------------------------------------------------- #
def _footer() -> str:
    return f"_{ANALYTICS_FOOTER}_"


def _pct(value) -> str:
    return f"{value * 100:.0f}%" if value is not None else "n/a"


def _load_records(records, config) -> List[dict]:
    if records is not None:
        return [r for r in records if isinstance(r, dict)]
    try:
        return eva.load_closed_records(config)
    except Exception:  # pragma: no cover - fail-open
        return []


def _parse_json_field(value):
    if value is None:
        return None
    if isinstance(value, (dict, list)):
        return value
    if isinstance(value, str):
        try:
            return json.loads(value)
        except Exception:
            return None
    return None


def _to_float(value, default=None):
    try:
        f = float(value)
        return default if f != f else f
    except (TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# 1) ORACLE_REGIME
# --------------------------------------------------------------------------- #
def compute_oracle_regime_report(market_view=None, *, regime_raw=None,
                                 vix=None, breadth=None, news_score=None,
                                 symbol: str = "SPY", config=None) -> dict:
    try:
        est = orr.classify_regime(market_view, regime_raw=regime_raw, vix=vix,
                                  breadth=breadth, news_score=news_score,
                                  symbol=symbol, config=config)
        has_ctx = regime_raw is not None or market_view is not None
        est = dict(est)
        est["verdict"] = VERDICT_OK if has_ctx else VERDICT_INSUFFICIENT
        return est
    except Exception:  # pragma: no cover - fail-open
        return {"label": orr.RANGE_BOUND, "confidence": 0.0, "components": {},
                "reasons": [], "verdict": VERDICT_INSUFFICIENT}


def format_oracle_regime_report(report: dict) -> str:
    header = "🧭 *Oracle Regime* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "", "No market context available.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    comp = report.get("components", {}) or {}
    lines = [
        header, "",
        f"*Regime:* `{report.get('label')}` "
        f"(confidence `{report.get('confidence', 0.0):.2f}`)", "",
        f"realized_vol `{comp.get('realized_vol')}` · "
        f"momentum `{comp.get('momentum')}` · trend `{comp.get('trend')}`",
        f"vix `{comp.get('vix')}` · breadth `{comp.get('breadth')}` · "
        f"news `{comp.get('news_score')}`", "",
    ]
    if report.get("reasons"):
        lines.append("*Why:*")
        for r in report["reasons"]:
            lines.append(f"• {r}")
        lines.append("")
    lines += [f"*Verdict:* `{report.get('verdict')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_regime_report_text(market_view=None, *, regime_raw=None,
                                       vix=None, breadth=None, news_score=None,
                                       symbol: str = "SPY", config=None) -> str:
    return format_oracle_regime_report(compute_oracle_regime_report(
        market_view, regime_raw=regime_raw, vix=vix, breadth=breadth,
        news_score=news_score, symbol=symbol, config=config))


# --------------------------------------------------------------------------- #
# 2) ORACLE_EXPLAIN <ticker>
# --------------------------------------------------------------------------- #
def compute_oracle_explain(ticker: str, ctx: Optional[dict] = None,
                           weights: Optional[dict] = None, prior: float = 0.5,
                           regime=None, agents_config=None) -> dict:
    try:
        ctx = ctx if isinstance(ctx, dict) else {}
        votes = oag.run_agents(ctx, agents_config)
        tally = ovo.tally_votes(votes, weights)
        prob = ovo.bayesian_probability(votes, prior, weights)
        expl = oex.explain(votes, weights=weights, probability=prob,
                           regime=regime)
        has_ctx = bool(ctx)
        return {
            "ticker": str(ticker or "").upper() or "?",
            "votes": [v.to_dict() for v in votes],
            "tally": tally,
            "probability": prob,
            "explanation": expl,
            "verdict": VERDICT_OK if has_ctx else VERDICT_INSUFFICIENT,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"ticker": str(ticker or "?"), "votes": [], "tally": {},
                "probability": {}, "explanation": {},
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_explain(report: dict) -> str:
    tkr = report.get("ticker", "?") if report else "?"
    header = f"🔎 *Oracle Explain — {tkr}* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "", "No evidence context for this ticker.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    prob = report.get("probability", {}) or {}
    expl = report.get("explanation", {}) or {}
    lines = [
        header, "",
        f"*P(call)* `{_pct(prob.get('p_call'))}` · "
        f"*P(put)* `{_pct(prob.get('p_put'))}` · "
        f"*P(no-trade)* `{_pct(prob.get('p_no_trade'))}`", "",
    ]
    contrib = expl.get("agent_contributions", {}) or {}
    if contrib:
        lines.append("*Agent contributions:*")
        for name, share in sorted(contrib.items(), key=lambda kv: kv[1],
                                  reverse=True)[:TOP_N]:
            lines.append(f"`{name}` {_pct(share)}")
        lines.append("")
    if expl.get("top_reasons"):
        lines.append("*Top reasons:*")
        for r in expl["top_reasons"]:
            lines.append(f"• {r}")
        lines.append("")
    if expl.get("summary_str"):
        lines += [f"_{expl['summary_str']}_", ""]
    lines += [f"*Verdict:* `{report.get('verdict')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_explain_text(ticker: str, ctx: Optional[dict] = None,
                                 weights: Optional[dict] = None,
                                 prior: float = 0.5, regime=None,
                                 agents_config=None) -> str:
    return format_oracle_explain(compute_oracle_explain(
        ticker, ctx=ctx, weights=weights, prior=prior, regime=regime,
        agents_config=agents_config))


# --------------------------------------------------------------------------- #
# 3) ORACLE_AGENT_REPORT
# --------------------------------------------------------------------------- #
def compute_oracle_agent_report(records: Optional[List[dict]] = None,
                                config=None) -> dict:
    """Per-agent hit-rate: win rate of closed trades on which the agent
    expressed directional conviction, vs the global base win rate. Never raises.
    """
    try:
        rows = _load_records(records, config)
        scored = []          # rows carrying agent_votes
        base_wins = 0
        per_agent: Dict[str, Dict[str, float]] = {}
        for row in rows:
            votes = _parse_json_field(row.get("agent_votes"))
            if not isinstance(votes, dict):
                continue
            win = 1 if oa._is_win(row) else 0
            scored.append(row)
            base_wins += win
            for name, v in votes.items():
                if not isinstance(v, dict):
                    continue
                bull = _to_float(v.get("bullish_score"), 0.0) or 0.0
                bear = _to_float(v.get("bearish_score"), 0.0) or 0.0
                if abs(bull - bear) <= 0.0:
                    continue                       # neutral -> no conviction
                a = per_agent.setdefault(name, {"votes": 0, "wins": 0,
                                                "conf": 0.0})
                a["votes"] += 1
                a["wins"] += win
                a["conf"] += _to_float(v.get("confidence"), 0.0) or 0.0

        n = len(scored)
        base_wr = base_wins / n if n else None
        agents = []
        for name, a in per_agent.items():
            vts = a["votes"]
            if vts == 0:
                continue
            hr = a["wins"] / vts
            agents.append({
                "agent": name, "votes": vts, "hit_rate": hr,
                "avg_confidence": a["conf"] / vts,
                "lift": (hr - base_wr) if base_wr is not None else None,
            })
        agents.sort(key=lambda r: r["hit_rate"], reverse=True)
        return {
            "sample_size": n, "base_win_rate": base_wr, "agents": agents,
            "verdict": VERDICT_OK if n >= MIN_SAMPLES else VERDICT_INSUFFICIENT,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "base_win_rate": None, "agents": [],
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_agent_report(report: dict) -> str:
    header = "🤖 *Oracle Agent Report* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "",
                          "Not enough closed trades carry agent votes yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    lines = [header, "",
             f"_Win rate of trades where each agent had conviction "
             f"(base WR `{_pct(report.get('base_win_rate'))}`)._", ""]
    for a in report.get("agents", [])[:TOP_N]:
        lift = a.get("lift")
        lift_s = f"{lift * 100:+.0f}pp" if lift is not None else "n/a"
        lines.append(
            f"`{a['agent']}` -> WR `{_pct(a['hit_rate'])}` "
            f"(lift `{lift_s}`), n `{a['votes']}`, "
            f"conf `{a['avg_confidence']:.2f}`")
    lines += ["", f"*Verdict:* `{report.get('verdict')}` · "
              f"sample `{report.get('sample_size')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_agent_report_text(records: Optional[List[dict]] = None,
                                      config=None) -> str:
    return format_oracle_agent_report(
        compute_oracle_agent_report(records=records, config=config))


# --------------------------------------------------------------------------- #
# 4) ORACLE_PROBABILITY_REPORT
# --------------------------------------------------------------------------- #
def compute_oracle_probability_report(records: Optional[List[dict]] = None,
                                      config=None) -> dict:
    """Brier score of the model's P(call) vs realized wins, vs a 0.5 baseline."""
    try:
        rows = _load_records(records, config)
        pairs = []           # (p_call, win)
        for row in rows:
            p = _to_float(row.get("model_p_call"))
            if p is None:
                continue
            win = 1.0 if oa._is_win(row) else 0.0
            pairs.append((max(0.0, min(1.0, p)), win))
        n = len(pairs)
        if n == 0:
            return {"sample_size": 0, "brier": None, "baseline_brier": None,
                    "skill": None, "avg_p_call": None,
                    "verdict": VERDICT_INSUFFICIENT}
        brier = sum((p - w) ** 2 for p, w in pairs) / n
        baseline = sum((0.5 - w) ** 2 for _, w in pairs) / n
        skill = (baseline - brier) / baseline if baseline > 0 else None
        return {
            "sample_size": n,
            "brier": brier,
            "baseline_brier": baseline,
            "skill": skill,
            "avg_p_call": sum(p for p, _ in pairs) / n,
            "realized_win_rate": sum(w for _, w in pairs) / n,
            "verdict": VERDICT_OK if n >= MIN_SAMPLES else VERDICT_INSUFFICIENT,
        }
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "brier": None, "baseline_brier": None,
                "skill": None, "avg_p_call": None,
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_probability_report(report: dict) -> str:
    header = "🎯 *Oracle Probability Report* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "",
                          "Not enough trades carry a model probability yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    skill = report.get("skill")
    skill_s = f"{skill * 100:+.0f}%" if skill is not None else "n/a"
    lines = [
        header, "",
        "_Calibration of P(call) against realized wins (lower Brier is "
        "better)._", "",
        f"*Brier:* `{report['brier']:.4f}` "
        f"(baseline `{report['baseline_brier']:.4f}`)",
        f"*Skill vs 0.5:* `{skill_s}`",
        f"avg P(call) `{_pct(report.get('avg_p_call'))}` · "
        f"realized WR `{_pct(report.get('realized_win_rate'))}`", "",
        f"*Verdict:* `{report.get('verdict')}` · "
        f"sample `{report.get('sample_size')}`", "", _footer(),
    ]
    return "\n".join(lines)


def generate_oracle_probability_report_text(records: Optional[List[dict]] = None,
                                            config=None) -> str:
    return format_oracle_probability_report(
        compute_oracle_probability_report(records=records, config=config))


# --------------------------------------------------------------------------- #
# 5) ORACLE_FEATURE_IMPORTANCE
# --------------------------------------------------------------------------- #
def compute_oracle_feature_importance(records: Optional[List[dict]] = None,
                                      config=None) -> dict:
    """Mean agent contribution share across closed trades that stamped one."""
    try:
        rows = _load_records(records, config)
        totals: Dict[str, float] = {}
        n = 0
        for row in rows:
            contrib = _parse_json_field(row.get("agent_contributions"))
            if not isinstance(contrib, dict) or not contrib:
                continue
            n += 1
            for name, share in contrib.items():
                s = _to_float(share, 0.0) or 0.0
                totals[name] = totals.get(name, 0.0) + s
        if n == 0:
            return {"sample_size": 0, "features": [],
                    "verdict": VERDICT_INSUFFICIENT}
        features = [{"agent": k, "importance": v / n}
                    for k, v in totals.items()]
        features.sort(key=lambda r: r["importance"], reverse=True)
        return {"sample_size": n, "features": features,
                "verdict": VERDICT_OK if n >= MIN_SAMPLES
                else VERDICT_INSUFFICIENT}
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "features": [],
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_feature_importance(report: dict) -> str:
    header = "📊 *Oracle Feature Importance* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "",
                          "Not enough trades carry agent contributions yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    lines = [header, "",
             "_Average share of each agent in the decision._", ""]
    for f in report.get("features", [])[:TOP_N]:
        lines.append(f"`{f['agent']}` -> `{_pct(f['importance'])}`")
    lines += ["", f"*Verdict:* `{report.get('verdict')}` · "
              f"sample `{report.get('sample_size')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_feature_importance_text(records: Optional[List[dict]] = None,
                                            config=None) -> str:
    return format_oracle_feature_importance(
        compute_oracle_feature_importance(records=records, config=config))


# --------------------------------------------------------------------------- #
# 6) ORACLE_WEIGHT_CHANGES
# --------------------------------------------------------------------------- #
def compute_oracle_weight_changes(history: Optional[List[dict]] = None,
                                  config=None) -> dict:
    """Adaptive voting-weight history + drift. Delegates to oracle_weights when
    present; fails open to INSUFFICIENT_DATA when no history is available."""
    try:
        if history is None:
            try:
                import oracle_weights as ow
                history = ow.weight_history(config=config)
            except Exception:
                history = None
        history = [h for h in (history or []) if isinstance(h, dict)]
        if not history:
            return {"snapshots": 0, "current": {}, "drift": None,
                    "verdict": VERDICT_INSUFFICIENT}
        current = history[-1].get("weights", {}) or {}
        first = history[0].get("weights", {}) or {}
        drift = None
        if current and first:
            keys = set(current) | set(first)
            drift = sum(abs((_to_float(current.get(k), 0.0) or 0.0)
                            - (_to_float(first.get(k), 0.0) or 0.0))
                        for k in keys)
        return {"snapshots": len(history), "current": current, "drift": drift,
                "verdict": VERDICT_OK}
    except Exception:  # pragma: no cover - fail-open
        return {"snapshots": 0, "current": {}, "drift": None,
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_weight_changes(report: dict) -> str:
    header = "⚖️ *Oracle Weight Changes* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "",
                          "No adaptive weight history yet (uniform weights).",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    lines = [header, "",
             f"Weight snapshots: `{report.get('snapshots', 0)}`"]
    drift = report.get("drift")
    if drift is not None:
        lines.append(f"Total drift since start: `{drift:.3f}`")
    lines.append("")
    current = report.get("current", {}) or {}
    if current:
        lines.append("*Current weights:*")
        for name, w in sorted(current.items(), key=lambda kv: kv[1],
                              reverse=True):
            lines.append(f"`{name}` -> `{_to_float(w, 0.0):.2f}`")
    lines += ["", f"*Verdict:* `{report.get('verdict')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_weight_changes_text(history: Optional[List[dict]] = None,
                                        config=None) -> str:
    return format_oracle_weight_changes(
        compute_oracle_weight_changes(history=history, config=config))


# --------------------------------------------------------------------------- #
# 7) ORACLE_HYPOTHESIS_REPORT (delegates to hypothesis_engine)
# --------------------------------------------------------------------------- #
def generate_oracle_hypothesis_report_text(config=None,
                                           trades: Optional[List[dict]] = None
                                           ) -> str:
    try:
        import hypothesis_engine as he
        body = he.generate_hypothesis_report_text(config=config, trades=trades)
    except Exception:  # pragma: no cover - fail-open
        body = "🔬 *Hypothesis Report*\n\nUnavailable."
    if ANALYTICS_FOOTER not in body:
        body = body.rstrip() + "\n\n" + _footer()
    return body


# --------------------------------------------------------------------------- #
# 8) ORACLE_REGIME_PERFORMANCE
# --------------------------------------------------------------------------- #
def compute_oracle_regime_performance(records: Optional[List[dict]] = None,
                                      config=None) -> dict:
    """Per-regime WR / avg P/L / PF / total over closed records stamped with a
    ``regime_label`` (Phase 3). Never raises."""
    try:
        rows = _load_records(records, config)
        groups: Dict[str, List[dict]] = {}
        for row in rows:
            label = row.get("regime_label")
            if not label:
                continue
            groups.setdefault(str(label), []).append(row)
        n = sum(len(g) for g in groups.values())
        regimes = []
        for label, group in groups.items():
            st = eva.bucket_stats(group)
            regimes.append({"regime": label, **st})
        regimes.sort(key=lambda r: r.get("total_pnl", 0.0) or 0.0, reverse=True)
        return {"sample_size": n, "regimes": regimes,
                "verdict": VERDICT_OK if n >= MIN_SAMPLES
                else VERDICT_INSUFFICIENT}
    except Exception:  # pragma: no cover - fail-open
        return {"sample_size": 0, "regimes": [],
                "verdict": VERDICT_INSUFFICIENT}


def format_oracle_regime_performance(report: dict) -> str:
    header = "🌐 *Oracle Regime Performance* _(analytics)_"
    if not report or report.get("verdict") == VERDICT_INSUFFICIENT:
        return "\n".join([header, "",
                          "Not enough regime-stamped trades yet.",
                          f"*Verdict:* `{VERDICT_INSUFFICIENT}`", "", _footer()])
    lines = [header, "",
             "_Realized performance by market regime._", ""]
    for r in report.get("regimes", []):
        lines.append(
            f"`{r['regime']}` -> WR `{_pct(r.get('win_rate'))}`, "
            f"avg `{eva._money(r.get('average_pnl'))}`, "
            f"PF `{eva._pf_str(r.get('profit_factor'))}`, "
            f"total `{eva._money(r.get('total_pnl'))}`, "
            f"n `{r.get('trades')}`")
    lines += ["", f"*Verdict:* `{report.get('verdict')}` · "
              f"sample `{report.get('sample_size')}`", "", _footer()]
    return "\n".join(lines)


def generate_oracle_regime_performance_text(records: Optional[List[dict]] = None,
                                            config=None) -> str:
    return format_oracle_regime_performance(
        compute_oracle_regime_performance(records=records, config=config))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — injected records / contexts)
# --------------------------------------------------------------------------- #
def _synthetic_records(n: int = 14) -> List[dict]:
    recs = []
    for i in range(n):
        win = i % 5 != 0                        # ~80% WR
        pnl = 30.0 if win else -40.0
        votes = {
            "trend": {"bullish_score": 0.8 if win else 0.0,
                      "bearish_score": 0.0 if win else 0.6,
                      "confidence": 0.9},
            "news": {"bullish_score": 0.5, "bearish_score": 0.0,
                     "confidence": 0.6},
            "liquidity": {"bullish_score": 0.0, "bearish_score": 0.0,
                          "confidence": 0.7},
        }
        contrib = {"trend": 0.6, "news": 0.3, "liquidity": 0.1}
        recs.append({
            "id": f"t{i}", "pnl": pnl, "max_loss": 100.0,
            "regime_label": "TRENDING_BULL" if win else "HIGH_VOLATILITY",
            "model_p_call": 0.7 if win else 0.4,
            "model_p_put": 0.2 if win else 0.5,
            "agent_votes": votes, "agent_contributions": contrib,
        })
    return recs


def _self_test() -> int:
    ok = True
    recs = _synthetic_records()

    # Each format ends with the footer; OK verdict on sufficient data.
    checks = [
        ("regime",
         generate_oracle_regime_report_text(
             regime_raw={"regime": "trending", "trend": "up",
                         "realized_vol": 0.2, "momentum": 0.06})),
        ("explain",
         generate_oracle_explain_text(
             "SPY", ctx={"trend": "up", "momentum": 0.08, "news_score": 0.5})),
        ("agent", generate_oracle_agent_report_text(records=recs)),
        ("probability", generate_oracle_probability_report_text(records=recs)),
        ("feature", generate_oracle_feature_importance_text(records=recs)),
        ("weights", generate_oracle_weight_changes_text(
            history=[{"weights": {"trend": 1.0}},
                     {"weights": {"trend": 1.5}}])),
        ("hypothesis", generate_oracle_hypothesis_report_text(trades=[])),
        ("regime_perf", generate_oracle_regime_performance_text(records=recs)),
    ]
    for name, txt in checks:
        if not isinstance(txt, str):
            print("FAIL: not a string", name); ok = False
            continue
        if ANALYTICS_FOOTER not in txt:
            print("FAIL: missing footer", name); ok = False

    # Empty -> INSUFFICIENT_DATA + footer for the records-based reports.
    for gen in (generate_oracle_agent_report_text,
                generate_oracle_probability_report_text,
                generate_oracle_feature_importance_text,
                generate_oracle_regime_performance_text):
        txt = gen(records=[])
        if VERDICT_INSUFFICIENT not in txt or ANALYTICS_FOOTER not in txt:
            print("FAIL: empty report", gen.__name__); ok = False

    # Substantive checks.
    agent = compute_oracle_agent_report(records=recs)
    if agent["verdict"] != VERDICT_OK:
        print("FAIL: agent verdict", agent["verdict"]); ok = False
    prob = compute_oracle_probability_report(records=recs)
    if prob["verdict"] != VERDICT_OK or prob["brier"] is None:
        print("FAIL: probability verdict", prob); ok = False
    perf = compute_oracle_regime_performance(records=recs)
    if perf["verdict"] != VERDICT_OK or not perf["regimes"]:
        print("FAIL: regime perf", perf); ok = False

    # Never raises on garbage.
    for junk in (None, 42, "x", [None, 42], {"weird": object()}):
        try:
            compute_oracle_agent_report(records=junk)  # type: ignore[arg-type]
            compute_oracle_probability_report(records=junk)  # type: ignore
            compute_oracle_regime_performance(records=junk)  # type: ignore
            compute_oracle_feature_importance(records=junk)  # type: ignore
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("oracle_intelligence_reports self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
