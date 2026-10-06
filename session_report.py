"""Session-analysis section for the daily Oracle report (flag-gated, read-only).

Surfaces the overnight-vs-intraday evidence produced by the research scripts

    session_returns.py       -> session_stats_universe.json
    session_signal_eval.py   -> session_signal_eval_live.json
    session_backtest.py      -> session_backtest.json

in the existing daily report.  STRICTLY analytics: nothing here changes any
decision, gate, risk limit, or order.  The section is rendered ONLY when
``ENABLE_SESSION_REPORT`` is truthy (default OFF -> the daily report output is
byte-identical to before this feature existed).

Every reader fails open: a missing/malformed artifact yields a
"missing data" line, never an exception.
"""

from __future__ import annotations

import json
import os
from typing import Optional

DEFAULT_STATS_JSON = "session_stats_universe.json"
DEFAULT_EVAL_JSON = "session_signal_eval_live.json"
DEFAULT_BACKTEST_JSON = "session_backtest.json"

_TRUTHY = {"1", "true", "yes", "on"}


def session_report_enabled(env: Optional[dict] = None) -> bool:
    """Default OFF; enable with ENABLE_SESSION_REPORT=1."""
    e = env if env is not None else os.environ
    return str(e.get("ENABLE_SESSION_REPORT", "")).strip().lower() in _TRUTHY


def _read_json(path: str) -> Optional[dict]:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            data = json.load(fh)
        return data if isinstance(data, dict) else None
    except Exception:
        return None


def compute_session_summary(stats_path: Optional[str] = None,
                            eval_path: Optional[str] = None,
                            backtest_path: Optional[str] = None,
                            env: Optional[dict] = None) -> dict:
    """Assemble the session section dict from research artifacts (fail-open).

    Returns ``{}`` when the feature flag is off, so callers can gate on
    truthiness and existing behavior is unchanged by default.
    """
    e = env if env is not None else os.environ
    if not session_report_enabled(e):
        return {}

    stats = _read_json(stats_path or e.get("SESSION_STATS_JSON", DEFAULT_STATS_JSON))
    ev = _read_json(eval_path or e.get("SESSION_EVAL_JSON", DEFAULT_EVAL_JSON))
    bt = _read_json(backtest_path or e.get("SESSION_BACKTEST_JSON", DEFAULT_BACKTEST_JSON))

    out: dict = {"missing": [], "signal_sessions": {}, "oos_backtest": {}, "universe": {}}

    if ev:
        meta = ev.get("meta", {})
        out["eval_meta"] = {
            "decisions": meta.get("decisions"),
            "date_range": meta.get("date_range"),
            "cost_bps_per_side": meta.get("cost_bps_per_side"),
            "generated_at": meta.get("generated_at"),
        }
        for label, key in (("CALL (long)", "long (CALL)"), ("PUT (short)", "short (PUT)")):
            d = (ev.get("by_direction") or {}).get(key) or {}
            row = {}
            for leg in ("intraday_net", "overnight_net"):
                s = d.get(leg) or {}
                if s.get("n"):
                    row[leg] = {
                        "n": s["n"],
                        "mean_bp": s["mean"] * 1e4,
                        "se_bp": s["se_mean"] * 1e4,
                        "pos_freq": s["pos_freq"],
                    }
            if row:
                out["signal_sessions"][label] = row
        out["flag_counts"] = ev.get("flag_counts") or {}
    else:
        out["missing"].append("signal-session evaluation (run session_signal_eval.py)")

    if bt:
        meta = bt.get("meta", {})
        res = bt.get("test_results_net") or {}
        keep = {}
        for name in ("SPY_overnight_only", "SPY_intraday_only", "SPY_buy_hold",
                     "universe_overnight_only", "universe_intraday_only",
                     "session_model_overnight"):
            m = res.get(name)
            if m and m.get("n_sessions"):
                keep[name] = {
                    "ann_mean": m["ann_mean"],
                    "sharpe_net": m["sharpe_net"],
                    "max_drawdown": m["max_drawdown"],
                    "n_sessions": m["n_sessions"],
                }
        out["oos_backtest"] = {
            "test_dates": meta.get("test_dates"),
            "cost_bps_per_side": meta.get("cost_bps_per_side"),
            "options_status": meta.get("options_status"),
            "results": keep,
        }
    else:
        out["missing"].append("OOS session backtest (run session_backtest.py)")

    if stats:
        syms = stats.get("symbols") or {}
        n_on_gt = n_tot = 0
        for v in syms.values():
            fs = v.get("full_sample") or {}
            on, idl = fs.get("overnight") or {}, fs.get("intraday") or {}
            if on.get("n") and idl.get("n"):
                n_tot += 1
                if on["mean"] > idl["mean"]:
                    n_on_gt += 1
        out["universe"] = {
            "n_symbols": n_tot,
            "overnight_gt_intraday": n_on_gt,
            "window": [stats.get("meta", {}).get("start"), stats.get("meta", {}).get("end")],
        }
    else:
        out["missing"].append("universe session stats (run session_returns.py)")

    return out


def format_session_section(summary: dict) -> str:
    """Render the session summary as Telegram-markdown lines ('' when empty)."""
    if not summary:
        return ""
    lines = ["*Session Analysis (overnight vs intraday — analytics only):*"]

    uni = summary.get("universe") or {}
    if uni.get("n_symbols"):
        lines.append(
            f"  Universe: overnight mean > intraday in "
            f"`{uni['overnight_gt_intraday']}/{uni['n_symbols']}` symbols "
            f"({uni['window'][0]} -> {uni['window'][1]})")

    sig = summary.get("signal_sessions") or {}
    if sig:
        em = summary.get("eval_meta") or {}
        lines.append(
            f"  Signals evaluated: `{em.get('decisions', '?')}` decisions "
            f"{em.get('date_range', '')}, underlying proxy net of "
            f"`{em.get('cost_bps_per_side', '?')}`bp/side:")
        for label, row in sig.items():
            for leg_key, leg_name in (("intraday_net", "rest-of-day"),
                                      ("overnight_net", "next overnight")):
                s = row.get(leg_key)
                if s:
                    lines.append(
                        f"  • {label} {leg_name}: `{s['mean_bp']:+.1f}bp` "
                        f"(se `{s['se_bp']:.1f}`, n `{s['n']}`)")

    bt = summary.get("oos_backtest") or {}
    res = bt.get("results") or {}
    if res:
        lines.append(
            f"  OOS test {bt.get('test_dates')} net of "
            f"`{bt.get('cost_bps_per_side')}`bp/side "
            f"[{bt.get('options_status', '')}]:")
        for name, m in res.items():
            lines.append(
                f"  • `{name}`: ann `{m['ann_mean']:+.1%}` · "
                f"Sharpe `{m['sharpe_net']:.2f}` · MDD `{m['max_drawdown']:.1%}` "
                f"(n `{m['n_sessions']}`)")

    for miss in summary.get("missing") or []:
        lines.append(f"  ⚠️ missing: {miss}")

    lines.append("  _No decisions are changed by this section._")
    return "\n".join(lines)
