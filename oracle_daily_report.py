"""
Phase 8D — Daily Oracle report (read-only, offline-pure).

This module assembles a single human-readable daily report from the existing
analytics layers. It reads only historical artifacts via :mod:`oracle_analytics`
and :mod:`threshold_engine` —

    * spread_paper_positions.json  (OPEN simulated spread positions)
    * spread_paper_trades.json     (CLOSED simulated spread trades)
    * expected_move_history.csv    (expected-move predictions + vol edge)
    * oracle_training_dataset.csv  (features / predictions / outcomes)

and emits a structured report dict plus a Telegram-formatted string. It is
STRICTLY analytics: it contains no order placement, no spread execution, no
live-trading or gating logic — nothing here can open, modify, gate, or close
any real or paper position. The "paper account summary" is derived purely from
the local simulated spread book, never from a broker. Every reader fails open
(missing / empty / malformed → "no data") and every public function returns a
plain dict / string that is safe to format even when inputs are empty.

Public API (all accept optional ``config`` and pre-loaded data so they are
trivially unit-testable, with no network or credentials):

    build_daily_report()            -> structured dict
    format_daily_report(report)     -> Telegram-formatted string
    generate_daily_report_text()    -> build + format in one call
    should_send_daily_report(...)   -> pure scheduling predicate (once/day)
    read_last_sent_date() / write_last_sent_date()
"""

import json
import logging
import os
from collections import OrderedDict
from datetime import datetime
from typing import List, Optional

import oracle_analytics as oa
import threshold_engine as te
from oracle_analytics import AnalyticsConfig

logger = logging.getLogger(__name__)

# Persisted "last sent" marker so the scheduled sender fires once per day and
# never duplicates after a restart (Phase 8D Req 3).
DEFAULT_STATE_FILE = "oracle_daily_report_state.json"

ANALYTICS_FOOTER = "Analytics only — no trades placed."

# How many top vol-edge opportunities to surface in the report.
TOP_VOL_EDGE = 5


# --------------------------------------------------------------------------- #
# Report assembly
# --------------------------------------------------------------------------- #
def _candlestick_summary() -> dict:
    """Fail-open candlestick daily summary (analytics only). ``{}`` on any
    failure so the daily report never breaks when the feature is unused."""
    try:
        import candlestick_calibration as cc
        return cc.compute_daily_candlestick_summary() or {}
    except Exception:
        return {}


def build_daily_report(config: Optional[AnalyticsConfig] = None,
                       now: Optional[datetime] = None,
                       trades: Optional[List[dict]] = None,
                       positions: Optional[List[dict]] = None,
                       em_rows: Optional[List[dict]] = None,
                       dataset_rows: Optional[List[dict]] = None,
                       candlestick: Optional[dict] = None) -> dict:
    """Assemble the daily report dict from the analytics + threshold layers.

    Pure read-only aggregation; never raises on missing/empty data. The
    ``candlestick`` summary may be injected (tests) or is derived fail-open
    from the candlestick calibration layer.
    """
    config = config or AnalyticsConfig.from_env()
    now = now or datetime.now()

    closed = oa.load_closed_spread_trades(config, trades)
    opens = oa.load_open_spread_positions(config, positions)

    stats = oa.compute_oracle_stats(config, trades=closed, positions=opens,
                                    em_rows=em_rows)
    accuracy = oa.compute_prediction_accuracy(config, em_rows=em_rows)
    leaderboard = oa.compute_vol_edge_leaderboard(
        config, em_rows=em_rows, dataset_rows=dataset_rows, top_n=TOP_VOL_EDGE)
    spread_perf = oa.compute_spread_performance(config, trades=closed)
    strat = te.analyze_strategy_performance(config, closed)
    recs = te.compute_recommendations(config, trades=closed, em_rows=em_rows,
                                      dataset_rows=dataset_rows)

    return {
        "date": now.strftime("%Y-%m-%d"),
        "account": {
            "open_positions": stats["open_positions"],
            "open_pnl": stats["open_pnl"],
            "closed_pnl": stats["closed_pnl"],
            "total_pnl": stats["total_pnl"],
            "total_trades": stats["trades"],
            "win_rate": stats["win_rate"],
        },
        "open_positions": opens,
        "prediction_accuracy": accuracy["horizons"],
        "top_vol_edge": leaderboard,
        "spread_performance": spread_perf,
        "thresholds": {
            "recommended_min_oracle_score": recs["recommended_min_oracle_score"],
            "recommended_min_volatility_edge": recs["recommended_min_volatility_edge"],
            "recommended_dte_range": recs["recommended_dte_range"],
            "recommended_iv_rank_range": recs["recommended_iv_rank_range"],
        },
        "best_strategy": strat["best_strategy"],
        "worst_strategy": strat["worst_strategy"],
        "confidence": recs["confidence"],
        "n_trades": recs["n_trades"],
        "candlestick": (candlestick if candlestick is not None
                        else _candlestick_summary()),
    }


# --------------------------------------------------------------------------- #
# Formatting helpers
# --------------------------------------------------------------------------- #
def _pretty_strategy(name) -> str:
    return name.replace("_", " ").title() if name else "n/a"


def _fmt_score(v) -> str:
    return ">= %.0f" % v if isinstance(v, (int, float)) else "n/a"


def _fmt_edge(v) -> str:
    return ">= %.1f%%" % (v * 100) if isinstance(v, (int, float)) else "n/a"


def _fmt_pct(v) -> str:
    return "%.0f%%" % (v * 100) if isinstance(v, (int, float)) else "n/a"


def format_daily_report(report: dict) -> str:
    """Render a :func:`build_daily_report` dict into Telegram markdown."""
    acct = report.get("account", {})
    lines = [f"📊 *Oracle Daily Report* — {report.get('date', '')}", ""]

    # 1) Paper account summary (derived from the simulated spread book).
    lines.append("*Paper Account (simulated):*")
    lines.append(f"  Open positions: `{acct.get('open_positions', 0)}`")
    lines.append(f"  Open P/L: `${acct.get('open_pnl', 0.0):+.2f}`")
    lines.append(f"  Closed P/L: `${acct.get('closed_pnl', 0.0):+.2f}`")
    lines.append(f"  Total trades: `{acct.get('total_trades', 0)}`")
    lines.append(f"  Win rate: `{acct.get('win_rate', 0.0) * 100:.0f}%`")
    lines.append("")

    # 2) Open positions detail (up to a handful).
    opens = report.get("open_positions") or []
    if opens:
        lines.append(f"*Open Positions ({len(opens)}):*")
        for p in opens[:5]:
            sym = str(p.get("symbol") or "?")
            strat = _pretty_strategy(p.get("strategy"))
            pnl = oa._to_float(p.get("pnl")) or 0.0
            lines.append(f"  • `{sym}` {strat} `${pnl:+.2f}`")
    else:
        lines.append("*Open Positions:* none")
    lines.append("")

    # 3) Prediction accuracy (expected-move MAE per horizon).
    lines.append("*Prediction Accuracy (expected-move error):*")
    horizons = report.get("prediction_accuracy", {}) or {}
    if horizons:
        for h in ("1d", "3d", "7d", "30d"):
            hd = horizons.get(h)
            if not hd:
                continue
            mae = hd.get("mae_pct")
            n = hd.get("n", 0)
            mae_s = f"{mae * 100:.1f}%" if isinstance(mae, (int, float)) else "n/a"
            lines.append(f"  {h.upper()}: MAE `{mae_s}` (`{n}` matched)")
    else:
        lines.append("  no prediction data yet")
    lines.append("")

    # 4) Top volatility-edge opportunities.
    board = report.get("top_vol_edge") or []
    if board:
        lines.append("*Top Volatility Edge:*")
        for e in board:
            edge = e.get("volatility_edge")
            edge_s = f"{edge * 100:+.1f}%" if isinstance(edge, (int, float)) else "n/a"
            score = e.get("oracle_score")
            score_s = f" · score `{score:.0f}`" if isinstance(score, (int, float)) else ""
            lines.append(f"  • `{e.get('symbol', '?')}` edge `{edge_s}`{score_s}")
    else:
        lines.append("*Top Volatility Edge:* no data")
    lines.append("")

    # 5) Spread performance by strategy.
    perf = report.get("spread_performance") or {}
    if perf:
        lines.append("*Spread Performance by Strategy:*")
        for strat, a in perf.items():
            lines.append(
                f"  • {_pretty_strategy(strat)}: `{a['trades']}` trades · "
                f"`{a['win_rate'] * 100:.0f}%` win · `${a['pnl']:+.2f}`")
    else:
        lines.append("*Spread Performance:* no closed trades yet")
    lines.append("")

    # 6) Threshold recommendations (advisory).
    thr = report.get("thresholds", {})
    lines.append("*Threshold Recommendations (advisory):*")
    lines.append(f"  Oracle Score: `{_fmt_score(thr.get('recommended_min_oracle_score'))}`")
    lines.append(f"  Vol Edge: `{_fmt_edge(thr.get('recommended_min_volatility_edge'))}`")
    lines.append(f"  DTE: `{thr.get('recommended_dte_range') or 'n/a'}`")
    lines.append(f"  IV Rank: `{thr.get('recommended_iv_rank_range') or 'n/a'}`")
    lines.append("")

    # 7) Best / worst strategy + data confidence.
    lines.append(f"*Best Strategy:* `{_pretty_strategy(report.get('best_strategy'))}`")
    lines.append(f"*Worst Strategy:* `{_pretty_strategy(report.get('worst_strategy'))}`")
    lines.append(f"*Data Confidence:* `{report.get('confidence', 'Low')}` "
                 f"({report.get('n_trades', 0)} trades)")
    lines.append("")

    # 8) Candlestick patterns (analytics only — never alters any decision).
    cs = report.get("candlestick") or {}
    tops = cs.get("top_patterns") or []
    if tops:
        lines.append("*Candlestick Patterns (analytics):*")
        for p in tops:
            wr = p.get("win_rate")
            wr_s = f"{wr * 100:.0f}%" if isinstance(wr, (int, float)) else "n/a"
            flag = " ⚠️ low sample" if p.get("low_sample") else ""
            lines.append(
                f"  • `{p.get('pattern_name', '?')}`: "
                f"{p.get('occurrences', 0)} seen · win `{wr_s}` · "
                f"{p.get('ev_impact', 'Neutral')}{flag}")
        improved = cs.get("improved_ev")
        verdict = ("improved EV" if improved
                   else "no clear EV gain" if improved is False
                   else "insufficient data")
        lines.append(f"  Did patterns help? `{verdict}` "
                     f"({cs.get('sample_size', 0)} resolved)")
        lines.append("")

    lines.append(f"_({ANALYTICS_FOOTER})_")
    return "\n".join(lines)


def generate_daily_report_text(config: Optional[AnalyticsConfig] = None,
                               now: Optional[datetime] = None,
                               **kwargs) -> str:
    """Build + format the daily report in one call (used by the bot)."""
    return format_daily_report(build_daily_report(config=config, now=now, **kwargs))


# --------------------------------------------------------------------------- #
# Scheduling (Req 3) — pure predicate + tiny JSON state file
# --------------------------------------------------------------------------- #
def read_last_sent_date(path: str = DEFAULT_STATE_FILE) -> Optional[str]:
    """Last date (YYYY-MM-DD) the daily report was sent, or None. Fails open."""
    data = oa.read_json(path)
    if isinstance(data, dict):
        v = data.get("last_sent_date")
        return str(v) if v else None
    return None


def write_last_sent_date(date_str: str, path: str = DEFAULT_STATE_FILE) -> bool:
    """Persist the last-sent date so restarts don't re-send. Never raises."""
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump({"last_sent_date": str(date_str)}, fh)
        return True
    except Exception as exc:  # pragma: no cover - defensive
        logger.warning("daily report state write failed (%s): %s", path, exc)
        return False


def should_send_daily_report(now: datetime, hour: int, minute: int,
                             last_sent_date: Optional[str]) -> bool:
    """True iff ``now`` is at/after ``hour:minute`` and we haven't sent today.

    Pure and deterministic. ``last_sent_date`` is the persisted YYYY-MM-DD; when
    it equals today the function returns False (so a restart never duplicates
    the day's report, and the report fires at most once per calendar day).
    """
    today = now.strftime("%Y-%m-%d")
    if last_sent_date == today:
        return False
    return (now.hour, now.minute) >= (int(hour), int(minute))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; synthetic data only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/dr_trades.json",
                          spread_positions_file="/nonexistent/dr_pos.json",
                          expected_move_file="/nonexistent/dr_em.csv",
                          training_dataset_file="/nonexistent/dr_ds.csv")

    # --- empty data is safe + footer present ---
    rep = build_daily_report(cfg, now=datetime(2025, 1, 2, 16, 30))
    if rep["date"] != "2025-01-02" or rep["n_trades"] != 0:
        print("FAIL: empty report header", rep); ok = False
    txt = format_daily_report(rep)
    if ANALYTICS_FOOTER not in txt or "Oracle Daily Report" not in txt:
        print("FAIL: empty report text"); ok = False

    # --- sample data flows into the report ---
    trades = [
        {"symbol": "SPY", "strategy": "bullish_put_credit_spread", "status": "closed",
         "oracle_score": 85, "volatility_edge": 0.035, "pnl": 120.0,
         "dte": 35, "iv_rank": 60},
        {"symbol": "QQQ", "strategy": "iron_condor", "status": "closed",
         "oracle_score": 45, "volatility_edge": 0.005, "pnl": -50.0,
         "dte": 12, "iv_rank": 20},
    ]
    opens = [{"symbol": "AAPL", "strategy": "debit_call_spread",
              "status": "open", "pnl": 25.0}]
    rep = build_daily_report(cfg, now=datetime(2025, 1, 2),
                             trades=trades, positions=opens)
    if rep["account"]["total_trades"] != 2:
        print("FAIL: sample trade count", rep["account"]); ok = False
    if rep["account"]["open_positions"] != 1:
        print("FAIL: sample open count", rep["account"]); ok = False
    if round(rep["account"]["closed_pnl"], 2) != 70.0:
        print("FAIL: sample closed pnl", rep["account"]); ok = False
    if abs(rep["account"]["win_rate"] - 0.5) > 1e-9:
        print("FAIL: sample win rate", rep["account"]); ok = False
    if rep["best_strategy"] != "bullish_put_credit_spread":
        print("FAIL: best strategy", rep["best_strategy"]); ok = False
    txt = format_daily_report(rep)
    if "AAPL" not in txt or "Bullish Put Credit Spread" not in txt:
        print("FAIL: sample report text"); ok = False

    # --- scheduling predicate ---
    now = datetime(2025, 1, 2, 16, 20)
    if not should_send_daily_report(now, 16, 15, None):
        print("FAIL: should send after time"); ok = False
    if should_send_daily_report(now, 16, 15, "2025-01-02"):
        print("FAIL: should NOT send when already sent today (restart-safe)"); ok = False
    if should_send_daily_report(datetime(2025, 1, 2, 16, 10), 16, 15, None):
        print("FAIL: should NOT send before time"); ok = False

    # --- state file round-trip ---
    d = tempfile.mkdtemp()
    state = os.path.join(d, "state.json")
    if read_last_sent_date(state) is not None:
        print("FAIL: missing state should be None"); ok = False
    write_last_sent_date("2025-01-02", state)
    if read_last_sent_date(state) != "2025-01-02":
        print("FAIL: state round-trip"); ok = False

    print("oracle_daily_report self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
