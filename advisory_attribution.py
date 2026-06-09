"""
Phase 9C — Advisory recommendation attribution (advisory, additive, fail-open).

This module answers the single most important validation question Oracle has:

    *Does the advisory gate have predictive value BEFORE the outcome is known?*

To avoid hindsight contamination it snapshots the advisory recommendation at the
moment a trade is OPENED — exactly as it existed at entry, while the outcome is
still unknown — and persists it. When the trade CLOSES, only the realized
outcome (PnL, win/loss, exit reason) is appended. The advisory fields are NEVER
recomputed after open.

That lets :func:`compute_advisory_performance` group every completed trade by the
recommendation it received at entry (STRONG_ACCEPT / ACCEPT / NEUTRAL /
WEAK_SETUP / REJECT_CANDIDATE) and report, per category:

    Trades · Win Rate · Profit Factor · Total PnL · Average PnL

plus an overall sample-size confidence (LOW / MEDIUM / HIGH).

STRICTLY advisory / additive: this module never opens, modifies, blocks, sizes or
closes any real or paper position, and never changes strategy selection. The
``record_open`` / ``record_close`` hooks are pure observers — they only persist a
JSON snapshot and can never alter the trade that triggered them. Every reader and
writer fails open (missing / empty / malformed -> "no data"); no public function
raises.

Persistence: an append/upsert JSON list at ``advisory_attribution.json`` keyed by
``trade_id`` (regenerated locally, never versioned).

Public API:

    record_open(trade)                  -> snapshot dict | None   (open-time)
    record_close(trade)                 -> snapshot dict | None   (close-time)
    build_open_snapshot(trade, ...)     -> snapshot dict          (pure)
    compute_advisory_performance(...)   -> per-category metrics
    format_advisory_performance(...)    -> Telegram-ready string
    generate_advisory_performance_text()-> Telegram command entry
"""

import json
import logging
import os
from datetime import datetime
from typing import List, Optional

import oracle_analytics as oa
import threshold_engine as te
import advisory_gate as ag
from oracle_analytics import AnalyticsConfig

logger = logging.getLogger(__name__)

# Snapshot store (regenerated locally; see .gitignore). A JSON list of records
# keyed by ``trade_id``.
DEFAULT_ATTRIBUTION_FILE = "advisory_attribution.json"

# Recommendation categories, in display order (matches advisory_gate labels).
CATEGORIES = (
    ag.STRONG_ACCEPT, ag.ACCEPT, ag.NEUTRAL, ag.WEAK_SETUP, ag.REJECT_CANDIDATE,
)

# Fields captured exactly at OPEN time (belief before the outcome is known).
OPEN_FIELDS = (
    "trade_id", "symbol", "date_opened", "strategy",
    "oracle_score", "volatility_edge", "dte", "iv_rank",
    "advisory_recommendation", "advisory_confidence",
    "historical_win_rate", "historical_profit_factor",
    "threshold_checks",
)

# Fields appended at CLOSE time (the realized outcome only).
CLOSE_FIELDS = (
    "date_closed", "pnl", "pnl_percent", "win_loss", "exit_reason",
)


# --------------------------------------------------------------------------- #
# Persistence helpers (JSON list; fail-open)
# --------------------------------------------------------------------------- #
def load_snapshots(path: Optional[str] = None) -> List[dict]:
    """Read the attribution store. Missing / corrupt / non-list -> []."""
    path = path or DEFAULT_ATTRIBUTION_FILE
    data = oa.read_json(path)
    return data if isinstance(data, list) else []


def save_snapshots(rows: List[dict], path: Optional[str] = None) -> bool:
    """Write the attribution store. Returns True on success (never raises)."""
    path = path or DEFAULT_ATTRIBUTION_FILE
    try:
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
        return True
    except Exception as exc:  # pragma: no cover - disk safety
        logger.warning("attribution store write failed (%s): %s", path, exc)
        return False


def _find_index(rows: List[dict], trade_id) -> int:
    tid = str(trade_id or "")
    for i, r in enumerate(rows):
        if str(r.get("trade_id") or "") == tid:
            return i
    return -1


# --------------------------------------------------------------------------- #
# Field extraction from a trade / position record
# --------------------------------------------------------------------------- #
def _trade_id(trade: dict):
    return trade.get("trade_id") or trade.get("id") or trade.get("order_id")


def _date_opened(trade: dict) -> Optional[str]:
    stamp = (trade.get("date_opened") or trade.get("timestamp")
             or trade.get("entry_time") or trade.get("date"))
    return str(stamp)[:10] if stamp else None


def _win_loss(pnl) -> Optional[str]:
    v = oa._to_float(pnl)
    if v is None:
        return None
    return "win" if v > 0 else "loss"


# --------------------------------------------------------------------------- #
# OPEN-time snapshot (the advisory belief, captured before the outcome)
# --------------------------------------------------------------------------- #
def build_open_snapshot(trade: dict, *,
                        config: Optional[AnalyticsConfig] = None,
                        trades: Optional[List[dict]] = None,
                        recommendations: Optional[dict] = None) -> dict:
    """Build the entry-time advisory snapshot for ``trade``.

    Reads the features present on the trade record at open (oracle_score,
    volatility_edge, dte, iv_rank, strategy) and runs the advisory gate ONCE to
    capture the recommendation / confidence / threshold checks and the
    historical win-rate / profit-factor *as they stood at entry*. Pure — does
    not persist. Never raises.
    """
    config = config or AnalyticsConfig.from_env()
    oracle_score = oa._trade_oracle(trade)
    volatility_edge = oa._trade_edge(trade)
    dte = oa._trade_dte(trade)
    iv_rank = oa._trade_iv_rank(trade)
    strategy = trade.get("strategy")

    result = ag.evaluate_setup(
        oracle_score=oracle_score, volatility_edge=volatility_edge,
        dte=dte, iv_rank=iv_rank, strategy=strategy,
        config=config, recommendations=recommendations, trades=trades)

    return {
        "trade_id": _trade_id(trade),
        "symbol": trade.get("symbol"),
        "date_opened": _date_opened(trade),
        "strategy": strategy,
        "oracle_score": oracle_score,
        "volatility_edge": volatility_edge,
        "dte": dte,
        "iv_rank": iv_rank,
        "advisory_recommendation": result.get("recommendation"),
        "advisory_confidence": result.get("confidence"),
        "historical_win_rate": result.get("historical_win_rate"),
        "historical_profit_factor": result.get("historical_profit_factor"),
        "threshold_checks": result.get("checks"),
        # Close-time fields are filled in later by record_close (None until then).
        "date_closed": None,
        "pnl": None,
        "pnl_percent": None,
        "win_loss": None,
        "exit_reason": None,
    }


def record_open(trade: dict, *,
                config: Optional[AnalyticsConfig] = None,
                path: Optional[str] = None,
                trades: Optional[List[dict]] = None,
                recommendations: Optional[dict] = None) -> Optional[dict]:
    """Observer hook: persist the entry-time advisory snapshot for ``trade``.

    Upserts by ``trade_id``. Advisory / additive only — this never affects the
    trade that triggered it and never raises. Returns the snapshot (or None on
    failure / when the trade has no id).
    """
    try:
        tid = _trade_id(trade)
        if not tid:
            return None
        snap = build_open_snapshot(trade, config=config, trades=trades,
                                   recommendations=recommendations)
        path = path or DEFAULT_ATTRIBUTION_FILE
        rows = load_snapshots(path)
        idx = _find_index(rows, tid)
        if idx >= 0:
            rows[idx] = snap  # re-open / overwrite stale snapshot for same id
        else:
            rows.append(snap)
        save_snapshots(rows, path)
        logger.info("[ADVISORY_ATTRIBUTION] open id=%s sym=%s rec=%s conf=%s",
                    tid, snap.get("symbol"), snap.get("advisory_recommendation"),
                    snap.get("advisory_confidence"))
        return snap
    except Exception as exc:  # pragma: no cover - observer must never raise
        logger.warning("[ADVISORY_ATTRIBUTION] record_open ignored: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# CLOSE-time append (the realized outcome only — never recomputes advisory)
# --------------------------------------------------------------------------- #
def record_close(trade: dict, *, path: Optional[str] = None) -> Optional[dict]:
    """Observer hook: append the realized outcome to an existing snapshot.

    Looks up the snapshot by ``trade_id`` and fills in ``date_closed``, ``pnl``,
    ``pnl_percent``, ``win_loss`` and ``exit_reason``. The advisory fields are
    left exactly as captured at open (no recomputation). Advisory / additive
    only; never raises. Returns the updated snapshot, or None if no matching
    open snapshot exists (e.g. trade opened before this feature existed).
    """
    try:
        tid = _trade_id(trade)
        if not tid:
            return None
        path = path or DEFAULT_ATTRIBUTION_FILE
        rows = load_snapshots(path)
        idx = _find_index(rows, tid)
        if idx < 0:
            logger.info("[ADVISORY_ATTRIBUTION] close id=%s has no open snapshot "
                        "(ignored)", tid)
            return None
        snap = rows[idx]
        pnl = oa._trade_pnl(trade)
        closed_at = (trade.get("date_closed") or trade.get("closed_at")
                     or trade.get("exit_time") or trade.get("date"))
        snap["date_closed"] = str(closed_at)[:10] if closed_at else None
        snap["pnl"] = pnl
        snap["pnl_percent"] = oa._trade_pnl_pct(trade)
        snap["win_loss"] = _win_loss(pnl)
        snap["exit_reason"] = trade.get("exit_reason")
        rows[idx] = snap
        save_snapshots(rows, path)
        logger.info("[ADVISORY_ATTRIBUTION] close id=%s rec=%s pnl=%s win_loss=%s",
                    tid, snap.get("advisory_recommendation"), snap.get("pnl"),
                    snap.get("win_loss"))
        return snap
    except Exception as exc:  # pragma: no cover - observer must never raise
        logger.warning("[ADVISORY_ATTRIBUTION] record_close ignored: %s", exc)
        return None


# --------------------------------------------------------------------------- #
# ADVISORY_PERFORMANCE — per-category predictive-value metrics
# --------------------------------------------------------------------------- #
def _is_closed(snap: dict) -> bool:
    """A snapshot counts once its outcome is known (PnL present)."""
    return oa._trade_pnl(snap) is not None


def _category_metrics(snaps: List[dict]) -> dict:
    """{trades, win_rate, profit_factor, total_pnl, avg_pnl} over closed snaps."""
    agg = oa._aggregate(snaps)
    n = agg["trades"]
    total_pnl = agg["pnl"]
    return {
        "trades": n,
        "win_rate": agg["win_rate"],
        "profit_factor": te._profit_factor(snaps),
        "total_pnl": total_pnl,
        "avg_pnl": round(total_pnl / n, 2) if n else 0.0,
    }


def compute_advisory_performance(config: Optional[AnalyticsConfig] = None,
                                 path: Optional[str] = None,
                                 snapshots: Optional[List[dict]] = None) -> dict:
    """Group completed trades by entry-time recommendation and measure each.

    Returns ``{"categories": {LABEL: metrics, ...}, "sample_size": int,
    "confidence": "LOW|MEDIUM|HIGH", "uncategorized": metrics}``. ``categories``
    always contains all five labels (zero-filled when unseen) so formatting is
    stable. Never raises.
    """
    rows = snapshots if snapshots is not None else load_snapshots(path)
    closed = [s for s in rows if _is_closed(s)]

    categories = {}
    for label in CATEGORIES:
        subset = [s for s in closed
                  if s.get("advisory_recommendation") == label]
        categories[label] = _category_metrics(subset)

    known = {s.get("advisory_recommendation") for s in closed}
    other = [s for s in closed
             if s.get("advisory_recommendation") not in CATEGORIES]

    return {
        "categories": categories,
        "uncategorized": _category_metrics(other),
        "sample_size": len(closed),
        "confidence": te.compute_confidence(len(closed)).upper(),
    }


# --------------------------------------------------------------------------- #
# Formatting
# --------------------------------------------------------------------------- #
def _pf_str(pf) -> str:
    if pf is None:
        return "n/a"
    if pf == float("inf"):
        return "∞"
    return "%.2f" % pf


def _pnl_str(value) -> str:
    v = oa._to_float(value)
    if v is None:
        return "n/a"
    sign = "-" if v < 0 else "+"
    return f"{sign}${abs(v):,.0f}"


def _category_block(label: str, m: dict) -> str:
    return (
        f"*{label}*\n"
        f"Trades: `{m['trades']}`\n"
        f"Win Rate: `{m['win_rate'] * 100:.0f}%`\n"
        f"Profit Factor: `{_pf_str(m['profit_factor'])}`\n"
        f"PnL: `{_pnl_str(m['total_pnl'])}`\n"
        f"Avg PnL: `{_pnl_str(m['avg_pnl'])}`"
    )


def format_advisory_performance(metrics: dict) -> str:
    """Telegram-ready ADVISORY_PERFORMANCE summary."""
    if metrics.get("sample_size", 0) == 0:
        return ("📊 *Advisory Performance* _(advisory)_\n\n"
                "No completed trades with an entry-time advisory snapshot yet.\n"
                "_(Advisory only — measures predictive value, trades nothing.)_")

    lines = ["📊 *Advisory Performance* _(advisory)_", ""]
    cats = metrics["categories"]
    for label in CATEGORIES:
        m = cats[label]
        if m["trades"] == 0:
            continue
        lines.append(_category_block(label, m))
        lines.append("")

    other = metrics.get("uncategorized", {})
    if other.get("trades", 0):
        lines.append(_category_block("UNCATEGORIZED", other))
        lines.append("")

    lines.append(f"Sample size: `{metrics['sample_size']}`")
    lines.append(f"Confidence: *{metrics['confidence']}*")
    lines.append("")
    lines.append("_(Advisory only — measures predictive value, trades nothing.)_")
    return "\n".join(lines)


def generate_advisory_performance_text(config: Optional[AnalyticsConfig] = None,
                                       path: Optional[str] = None) -> str:
    """Top-level entry for the ADVISORY_PERFORMANCE Telegram command."""
    metrics = compute_advisory_performance(config=config, path=path)
    return format_advisory_performance(metrics)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; temp files + synthetic snapshots)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    cfg = AnalyticsConfig(spread_trades_file="/nonexistent/aa_t.json",
                          spread_positions_file="/nonexistent/aa_p.json",
                          expected_move_file="/nonexistent/aa_e.csv",
                          training_dataset_file="/nonexistent/aa_d.csv")
    d = tempfile.mkdtemp()
    store = os.path.join(d, "advisory_attribution.json")

    # --- open snapshot captures advisory fields, no outcome yet ---
    pos = {"id": "t1", "symbol": "SPY",
           "strategy": "bullish_put_credit_spread", "oracle_score": 85,
           "volatility_edge": 0.035, "dte": 35, "iv_rank": 60,
           "timestamp": "2026-01-02T10:00:00"}
    snap = record_open(pos, config=cfg, path=store)
    if not snap or snap["pnl"] is not None or snap["advisory_recommendation"] is None:
        print("FAIL: open snapshot", snap); ok = False
    for key in OPEN_FIELDS:
        if key not in snap:
            print("FAIL: missing open field", key); ok = False

    # --- re-open same id overwrites (no duplicate) ---
    record_open(pos, config=cfg, path=store)
    if len(load_snapshots(store)) != 1:
        print("FAIL: duplicate open"); ok = False

    # --- close appends outcome only; advisory unchanged ---
    rec_before = snap["advisory_recommendation"]
    closed = record_close({"id": "t1", "pnl": 120.0, "pnl_percent": 40.0,
                           "exit_reason": "take_profit",
                           "closed_at": "2026-01-05T15:00:00"}, path=store)
    if not closed or closed["win_loss"] != "win" or closed["pnl"] != 120.0:
        print("FAIL: close append", closed); ok = False
    if closed["advisory_recommendation"] != rec_before:
        print("FAIL: advisory recomputed at close"); ok = False
    if closed["date_closed"] != "2026-01-05":
        print("FAIL: date_closed", closed.get("date_closed")); ok = False

    # --- close with no open snapshot is a no-op ---
    if record_close({"id": "ghost", "pnl": 10.0}, path=store) is not None:
        print("FAIL: close without open should be None"); ok = False

    # --- performance metrics group by entry recommendation ---
    snaps = [
        {"trade_id": "a", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": 100.0},
        {"trade_id": "b", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": 50.0},
        {"trade_id": "c", "advisory_recommendation": ag.STRONG_ACCEPT, "pnl": -40.0},
        {"trade_id": "d", "advisory_recommendation": ag.WEAK_SETUP, "pnl": -60.0},
        {"trade_id": "e", "advisory_recommendation": ag.WEAK_SETUP, "pnl": 20.0},
        {"trade_id": "f", "advisory_recommendation": ag.NEUTRAL, "pnl": None},  # open
    ]
    m = compute_advisory_performance(snapshots=snaps)
    sa = m["categories"][ag.STRONG_ACCEPT]
    if sa["trades"] != 3 or abs(sa["win_rate"] - 2 / 3) > 1e-9:
        print("FAIL: STRONG_ACCEPT metrics", sa); ok = False
    if sa["total_pnl"] != 110.0 or sa["avg_pnl"] != round(110.0 / 3, 2):
        print("FAIL: STRONG_ACCEPT pnl", sa); ok = False
    if abs(sa["profit_factor"] - (150.0 / 40.0)) > 1e-9:
        print("FAIL: STRONG_ACCEPT PF", sa["profit_factor"]); ok = False
    if m["sample_size"] != 5:  # the open (pnl=None) one is excluded
        print("FAIL: sample size", m["sample_size"]); ok = False
    if m["categories"][ag.ACCEPT]["trades"] != 0:
        print("FAIL: empty category not zero-filled"); ok = False

    # --- formatting never raises, includes seen categories + confidence ---
    txt = format_advisory_performance(m)
    if "Advisory Performance" not in txt or "STRONG_ACCEPT" not in txt \
            or "Confidence" not in txt:
        print("FAIL: format", txt); ok = False
    if ("*%s*" % ag.ACCEPT) in txt:  # zero-trade category should be hidden
        print("FAIL: empty category shown"); ok = False

    # --- empty store formats safely ---
    empty = format_advisory_performance(compute_advisory_performance(snapshots=[]))
    if "No completed trades" not in empty:
        print("FAIL: empty format", empty); ok = False

    print("advisory_attribution self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
