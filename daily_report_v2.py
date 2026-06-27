"""
Consolidated daily report (v2) -- the 5-section Oracle report over the LIVE
single-leg episode store (source of truth), distinct from the spread/advisory
``oracle_daily_report``.

Sections:
  1. **Trading**    -- win rate, profit factor, EV/trade, avg gain/loss,
                       largest winner/loser, total realized P/L.
  2. **Portfolio**  -- net delta/gamma, sector exposure, beta, correlation
                       (``portfolio_analytics``).
  3. **Execution**  -- spread paid, slippage, holding time, fill quality
                       (``execution_analytics``).
  4. **Learning**   -- evidence-EV leaderboards (``evidence_attribution``),
                       RL trend (``episode_store.stats``), calibration error
                       (``pop_calibration``).
  5. **Confidence** -- predicted-vs-actual win%, and the single biggest
                       EV-positive evidence to lean into / EV-negative to avoid.

STRICTLY analytics: read-only over ``episodes.db`` + the broker export. No order
placement, no broker writes. Every section fails open to "no data" so the report
never raises. Pure aggregation is unit-testable with injected rows (no network).
"""

import json
import os
from datetime import datetime
from typing import Dict, List, Optional

import evidence_attribution as ea

DEFAULT_DB = "episodes.db"
EXPORT_DIR = "trade_export"
ANALYTICS_FOOTER = "Analytics only -- no trades placed."

# Dimensions surfaced in the Learning section (the headline tables).
LEARNING_DIMS = ("agent", "pattern", "regime", "iv_bucket", "dte_bucket",
                 "delta_bucket", "direction", "strength", "strategy")


# --------------------------------------------------------------------------- #
# 1) Trading
# --------------------------------------------------------------------------- #
def trading_stats(rows: List[dict]) -> dict:
    """Realized-trade summary from normalized episode rows. Never raises."""
    pnls = [r.get("pnl") for r in rows if r.get("pnl") is not None]
    pcts = [r.get("pnl_percent") for r in rows if r.get("pnl_percent") is not None]
    n = len(pnls)
    wins = [p for p in pnls if p > 0]
    losses = [p for p in pnls if p <= 0]
    gross_win = sum(wins)
    gross_loss = -sum(losses)
    return {
        "trades": n,
        "win_rate": (len(wins) / n) if n else None,
        "total_realized": sum(pnls) if n else None,
        "ev_per_trade": (sum(pnls) / n) if n else None,
        "ev_per_trade_pct": (sum(pcts) / len(pcts)) if pcts else None,
        "avg_gain": (gross_win / len(wins)) if wins else None,
        "avg_loss": (sum(losses) / len(losses)) if losses else None,
        "largest_winner": max(pnls) if n else None,
        "largest_loser": min(pnls) if n else None,
        "profit_factor": (gross_win / gross_loss) if gross_loss > 0 else None,
    }


# --------------------------------------------------------------------------- #
# 4/5) Learning + Confidence helpers
# --------------------------------------------------------------------------- #
def _episode_store_stats(db_path: str) -> dict:
    try:
        from episode_store import EpisodeStore
        return EpisodeStore(db_path).stats()
    except Exception:
        return {}


def _pop_calibration() -> dict:
    try:
        import pop_calibration as pc
        return pc.compute_pop_calibration()
    except Exception:
        return {}


def _best_and_worst_evidence(tables: Dict[str, list]) -> dict:
    """Highest-EV evidence to lean into and lowest-EV to avoid, across dims."""
    best = None
    worst = None
    for dim in LEARNING_DIMS:
        for r in tables.get(dim, []) or []:
            if r.trades < ea.MIN_SAMPLES:
                continue
            tagged = {"dimension": dim, "feature": r.feature, "ev": r.ev,
                      "trades": r.trades, "verdict": r.verdict,
                      "avg_return_pct": r.avg_return_pct, "win_rate": r.win_rate}
            if best is None or r.ev > best["ev"]:
                best = tagged
            if worst is None or r.ev < worst["ev"]:
                worst = tagged
    return {"lean_into": best, "avoid": worst}


# Verdicts that mean the EV sign is established with enough confidence to act on.
_POSITIVE_VERDICTS = (ea.V_STRONG, ea.V_HIGH)
_NEGATIVE_VERDICTS = (ea.V_NEGATIVE,)


def ev_signals(tables: Dict[str, list]) -> dict:
    """Evidence whose EV is confidently signed with enough samples -- the
    actionable learning signals. ``positive`` = lean-into edges (Strong/High),
    ``negative`` = avoid these (Negative). Sorted by |EV| desc. Never raises."""
    positive: List[dict] = []
    negative: List[dict] = []
    for dim in LEARNING_DIMS:
        for r in tables.get(dim, []) or []:
            if r.trades < ea.MIN_SAMPLES or r.ev is None:
                continue
            tagged = {"dimension": dim, "feature": r.feature, "ev": r.ev,
                      "trades": r.trades, "verdict": r.verdict,
                      "avg_return_pct": r.avg_return_pct, "win_rate": r.win_rate}
            if r.verdict in _POSITIVE_VERDICTS:
                positive.append(tagged)
            elif r.verdict in _NEGATIVE_VERDICTS:
                negative.append(tagged)
    positive.sort(key=lambda d: d["ev"], reverse=True)
    negative.sort(key=lambda d: d["ev"])
    return {"positive": positive, "negative": negative,
            "n_positive": len(positive), "n_negative": len(negative)}


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def build_consolidated_report(
        now: Optional[datetime] = None,
        db_path: str = DEFAULT_DB,
        rows: Optional[List[dict]] = None,
        portfolio: Optional[dict] = None,
        execution: Optional[dict] = None,
        export_csv: Optional[str] = None,
        fills_path: Optional[str] = None) -> dict:
    """Assemble the 5-section consolidated report dict. Read-only, never raises.

    Injectable args (``rows``, ``portfolio``, ``execution``) keep this pure and
    unit-testable; when absent each section fails open to live/empty data.
    """
    now = now or datetime.now()

    if rows is None:
        rows = ea.load_completed(db_path)

    # 1) Trading
    trading = trading_stats(rows)

    # 2) Portfolio
    if portfolio is None:
        try:
            import portfolio_analytics as pa
            portfolio = pa.generate_live_report(csv_path=export_csv)
        except Exception:
            portfolio = {}

    # 3) Execution
    if execution is None:
        try:
            import execution_analytics as ex
            execution = ex.generate_live_report(db_path=db_path,
                                                fills_path=fills_path)
        except Exception:
            execution = {}

    # 4) Learning
    tables = ea.compute_all(rows=rows)
    learning = {
        "leaderboards": ea.to_json(tables),
        "rl_stats": _episode_store_stats(db_path),
    }

    # 5) Confidence
    calibration = _pop_calibration()
    confidence = {
        "calibration": {
            "verdict": calibration.get("verdict"),
            "sample_size": calibration.get("sample_size"),
            "calibration_error":
                (calibration.get("overall") or {}).get("calibration_error"),
        },
        "evidence": _best_and_worst_evidence(tables),
        "signals": ev_signals(tables),
    }

    return {
        "date": now.strftime("%Y-%m-%d"),
        "generated_at": now.isoformat(timespec="seconds"),
        "trading": trading,
        "portfolio": portfolio,
        "execution": execution,
        "learning": learning,
        "confidence": confidence,
        "_tables": tables,  # in-memory EvidenceRow tables for markdown reuse
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _d(v, prefix="$", nd=2):
    if v is None:
        return "n/a"
    return f"{prefix}{v:,.{nd}f}"


def _p(v, nd=1):
    if v is None:
        return "n/a"
    return f"{v * 100:.{nd}f}%"


def _x(v, nd=2):
    if v is None:
        return "n/a"
    return f"{v:.{nd}f}"


def _format_trading(t: dict) -> List[str]:
    return [
        "### 1. Trading", "",
        f"- Trades: **{t.get('trades', 0)}** | "
        f"Win rate: **{_p(t.get('win_rate'))}** | "
        f"Profit factor: **{_x(t.get('profit_factor'))}**",
        f"- EV/trade: **{_d(t.get('ev_per_trade'))}** "
        f"({_x(t.get('ev_per_trade_pct'))}%) | "
        f"Total realized: **{_d(t.get('total_realized'))}**",
        f"- Avg gain: **{_d(t.get('avg_gain'))}** | "
        f"Avg loss: **{_d(t.get('avg_loss'))}**",
        f"- Largest winner: **{_d(t.get('largest_winner'))}** | "
        f"Largest loser: **{_d(t.get('largest_loser'))}**",
        "",
    ]


def _format_portfolio(p: dict) -> List[str]:
    try:
        import portfolio_analytics as pa
        body = pa.format_markdown(p) if p else "_no position data_"
    except Exception:
        body = "_no position data_"
    return ["### 2. Portfolio", "", body, ""]


def _format_execution(e: dict) -> List[str]:
    try:
        import execution_analytics as ex
        body = ex.format_markdown(e) if e else "_no execution data_"
    except Exception:
        body = "_no execution data_"
    return ["### 3. Execution", "", body, ""]


def _format_learning(report: dict) -> List[str]:
    out = ["### 4. Learning", ""]
    tables = report.get("_tables") or {}
    out.append(ea.format_markdown(tables))
    rl = report.get("learning", {}).get("rl_stats") or {}
    if rl:
        out += [
            "",
            f"- Episodes: **{rl.get('completed', 0)}/{rl.get('total', 0)}** "
            f"closed | RL win rate: **{_p(rl.get('win_rate'))}** | "
            f"mean net: **{_x(rl.get('mean_net_pnl_pct'))}%**",
        ]
    out.append("")
    return out


def _format_confidence(c: dict) -> List[str]:
    cal = c.get("calibration", {})
    ev = c.get("evidence", {})
    out = [
        "### 5. Confidence", "",
        f"- Calibration: **{cal.get('verdict') or 'n/a'}** "
        f"(n={cal.get('sample_size', 0)}, "
        f"error={_err(cal.get('calibration_error'))})",
    ]
    lean = ev.get("lean_into")
    avoid = ev.get("avoid")
    if lean:
        out.append(
            f"- Lean into: **{lean['feature']}** ({lean['dimension']}, "
            f"{lean['trades']} trades, EV {lean['ev']:+.2f}, {lean['verdict']})")
    if avoid:
        out.append(
            f"- Avoid: **{avoid['feature']}** ({avoid['dimension']}, "
            f"{avoid['trades']} trades, EV {avoid['ev']:+.2f}, {avoid['verdict']})")

    sig = c.get("signals") or {}
    pos = sig.get("positive") or []
    neg = sig.get("negative") or []
    if pos or neg:
        out += ["", f"**EV signals learned** "
                f"(+{len(pos)} positive / -{len(neg)} negative):"]
        for s in pos[:5]:
            out.append(
                f"  - + {s['feature']} ({s['dimension']}, {s['trades']} trades, "
                f"EV {s['ev']:+.2f}, {s['verdict']})")
        for s in neg[:5]:
            out.append(
                f"  - - {s['feature']} ({s['dimension']}, {s['trades']} trades, "
                f"EV {s['ev']:+.2f}, {s['verdict']})")
    out.append("")
    return out


def _err(v):
    if v is None:
        return "n/a"
    return f"{v * 100:+.1f}pp"


def format_consolidated_report(report: dict) -> str:
    """Full 5-section markdown. Pure formatting."""
    lines = [f"# Oracle Daily Report -- {report.get('date', '')}", ""]
    lines += _format_trading(report.get("trading", {}))
    lines += _format_portfolio(report.get("portfolio", {}))
    lines += _format_execution(report.get("execution", {}))
    lines += _format_learning(report)
    lines += _format_confidence(report.get("confidence", {}))
    lines.append(f"_{ANALYTICS_FOOTER}_")
    return "\n".join(lines)


def _json_safe(report: dict) -> dict:
    """Drop the in-memory EvidenceRow tables before JSON serialization."""
    out = {k: v for k, v in report.items() if k != "_tables"}
    return out


# --------------------------------------------------------------------------- #
# Delivery: dated artifact
# --------------------------------------------------------------------------- #
def write_dated_artifact(report: dict, out_dir: str = EXPORT_DIR) -> Dict[str, str]:
    """Write ``daily_report_YYYYMMDD.md`` + ``.json``. Returns the paths."""
    paths: Dict[str, str] = {}
    try:
        os.makedirs(out_dir, exist_ok=True)
        stamp = report.get("date", datetime.now().strftime("%Y-%m-%d"))
        stamp = stamp.replace("-", "")
        md_path = os.path.join(out_dir, f"daily_report_{stamp}.md")
        json_path = os.path.join(out_dir, f"daily_report_{stamp}.json")
        with open(md_path, "w", encoding="utf-8") as fh:
            fh.write(format_consolidated_report(report))
        with open(json_path, "w", encoding="utf-8") as fh:
            json.dump(_json_safe(report), fh, indent=2, default=str)
        paths = {"markdown": md_path, "json": json_path}
    except Exception:
        pass
    return paths


def generate_consolidated_report_text(db_path: str = DEFAULT_DB) -> str:
    return format_consolidated_report(build_consolidated_report(db_path=db_path))


# --------------------------------------------------------------------------- #
# Self-test (no network/disk writes outside tmp)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    rows = [
        {"pnl": 100.0, "pnl_percent": 20.0, "direction": "up", "outcome": "win"},
        {"pnl": -40.0, "pnl_percent": -8.0, "direction": "down", "outcome": "loss"},
        {"pnl": 60.0, "pnl_percent": 12.0, "direction": "up", "outcome": "win"},
    ]
    t = trading_stats(rows)
    if t["trades"] != 3:
        print("FAIL: trades", t["trades"]); ok = False
    if abs(t["win_rate"] - 2 / 3) > 1e-9:
        print("FAIL: win_rate", t["win_rate"]); ok = False
    if abs(t["total_realized"] - 120.0) > 1e-9:
        print("FAIL: total_realized", t["total_realized"]); ok = False
    # profit factor = 160 / 40 = 4.0
    if abs(t["profit_factor"] - 4.0) > 1e-9:
        print("FAIL: profit_factor", t["profit_factor"]); ok = False
    if t["largest_winner"] != 100.0 or t["largest_loser"] != -40.0:
        print("FAIL: extremes", t); ok = False

    # empty never raises
    if trading_stats([])["trades"] != 0:
        print("FAIL: empty trades"); ok = False

    # build + format never raise on injected empty sections
    rep = build_consolidated_report(rows=rows, portfolio={}, execution={})
    md = format_consolidated_report(rep)
    for needed in ("1. Trading", "2. Portfolio", "3. Execution",
                   "4. Learning", "5. Confidence"):
        if needed not in md:
            print("FAIL: missing section", needed); ok = False
    # JSON-safe drops the in-memory tables
    if "_tables" in _json_safe(rep):
        print("FAIL: _tables leaked into json"); ok = False
    json.dumps(_json_safe(rep), default=str)  # must serialize

    # EV signals: confidently-signed evidence is flagged with the right sign.
    sig_tables = {
        "agent": [ea.EvidenceRow("Strong", 40, 8.0, 0.62, 8.0, ea.V_STRONG),
                  ea.EvidenceRow("Bad", 40, -3.0, 0.40, -3.0, ea.V_NEGATIVE),
                  ea.EvidenceRow("Meh", 40, 0.1, 0.50, 0.1, ea.V_LOW)],
    }
    sigs = ev_signals(sig_tables)
    if sigs["n_positive"] != 1 or sigs["positive"][0]["feature"] != "Strong":
        print("FAIL: ev_signals positive", sigs); ok = False
    if sigs["n_negative"] != 1 or sigs["negative"][0]["feature"] != "Bad":
        print("FAIL: ev_signals negative", sigs); ok = False

    print("daily_report_v2 self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--live" in sys.argv:
        rep = build_consolidated_report()
        print(format_consolidated_report(rep))
        if "--write" in sys.argv:
            print()
            print("[artifact]", write_dated_artifact(rep))
        sys.exit(0)
    sys.exit(_self_test())
