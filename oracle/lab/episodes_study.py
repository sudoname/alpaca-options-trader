"""
Oracle Lab — realized-episode study (ANALYTICS ONLY, offline).

The Phase-2 study harness (``run_phase2_study``) synthesizes forward-return
labels from a ``MarketView`` — that is the right tool for *hypothetical*
research. This module is the complement: it points the Lab's metric engine at
the REAL realized-trade history the live system already recorded in
``episodes.db`` (via ``episode_store.EpisodeStore``), so the strategy's actual
edge (or lack of it) can be measured offline with the same statistics the Lab
uses everywhere else.

Pipeline:
    EpisodeStore.completed()                 (closed rows w/ realized PnL)
        -> episode_to_trade  (pure map)      (Lab metrics trade-dict shape)
        -> oracle.lab.metrics.compute_metrics + breakdowns
        -> a readable, JSON-safe report

Nothing here trades, sizes, prices, blocks, or alters a real/paper order, reads
creds, or hits the network. It only READS a SQLite file and computes pure
statistics. The live path in ``smart_trader.py`` is byte-identical whether or
not this module is imported.

Determinism: ``build_report`` is a pure function of its trade list. The only
I/O is the read-only ``load_trades`` DB fetch.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

from oracle.lab import metrics as _metrics

_DB_DEFAULT = "episodes.db"


# --------------------------------------------------------------------------- #
# Pure mapping: an episodes.db row -> a Lab metrics trade record
# --------------------------------------------------------------------------- #
def _direction_from_row(row: dict) -> Optional[str]:
    """Map a closed decision to the Lab's 'call'/'put' direction label.

    Prefer the explicit ``chosen_action`` (CALL/PUT); fall back to a substring
    of ``strat`` (e.g. 'long_call'). Returns None for anything non-directional.
    """
    action = str(row.get("chosen_action") or "").strip().upper()
    if action == "CALL":
        return _metrics.DIR_CALL
    if action == "PUT":
        return _metrics.DIR_PUT
    strat = str(row.get("strat") or "").lower()
    if "call" in strat:
        return _metrics.DIR_CALL
    if "put" in strat:
        return _metrics.DIR_PUT
    return None


def _regime_from_features(features_json: Any) -> Optional[str]:
    """Best-effort regime label from the free-form ``features_json`` blob.

    Looks for a regime at the top level, then under the Oracle namespace, then
    under a ``ctx`` context dict. A regime stored as a nested object is reduced
    to its ``label``/``regime`` field. Fail-open: any parse issue -> None.
    """
    if not features_json:
        return None
    try:
        d = json.loads(features_json) if isinstance(features_json, str) \
            else features_json
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    for path in (("regime",), ("oracle", "regime"),
                 ("oracle", "regime_label"), ("ctx", "regime")):
        cur: Any = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if not ok or cur in (None, ""):
            continue
        if isinstance(cur, dict):
            lbl = cur.get("label") or cur.get("regime")
            if lbl:
                return str(lbl)
            continue
        return str(cur)
    return None


def _dte_from_features(features_json: Any) -> Optional[float]:
    """Best-effort days-to-expiry AT ENTRY from ``features_json``.

    ``dte_entry`` is the one genuine entry-time attribute the broker-roundtrip
    backfill preserves. Looks top-level, then under the Oracle / ctx namespaces.
    Fail-open: any parse issue -> None.
    """
    if not features_json:
        return None
    try:
        d = json.loads(features_json) if isinstance(features_json, str) \
            else features_json
    except Exception:
        return None
    if not isinstance(d, dict):
        return None
    for path in (("dte_entry",), ("oracle", "dte_entry"), ("ctx", "dte_entry")):
        cur: Any = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if not ok or cur in (None, ""):
            continue
        try:
            return float(cur)
        except (TypeError, ValueError):
            continue
    return None


def episode_to_trade(row: dict) -> Optional[dict]:
    """Map one closed episodes.db row to a Lab metrics trade record.

    Returns None when the row carries no realized dollar PnL (nothing to
    measure). Pure: no I/O, never raises on a malformed row.
    """
    if not isinstance(row, dict):
        return None
    pnl = row.get("net_pnl_dollars")
    if pnl is None:
        return None
    try:
        pnl = float(pnl)
    except (TypeError, ValueError):
        return None
    trade: Dict[str, Any] = {"pnl": pnl}
    ret = row.get("net_pnl_pct")
    if ret is not None:
        try:
            trade["return_pct"] = float(ret)
        except (TypeError, ValueError):
            pass
    direction = _direction_from_row(row)
    if direction:
        trade["direction"] = direction
    mode = row.get("mode")
    if mode not in (None, ""):
        trade["strategy_mode"] = str(mode)
    regime = _regime_from_features(row.get("features_json"))
    if regime:
        trade["regime"] = regime
    dte = _dte_from_features(row.get("features_json"))
    if dte is not None:
        trade["dte_entry"] = dte
    hd = row.get("hold_days")
    if hd is not None:
        try:
            trade["hold_days"] = float(hd)
        except (TypeError, ValueError):
            pass
    # Carry a couple of identity/context fields for downstream inspection.
    for k in ("symbol", "underlying", "strat", "outcome", "as_of", "closed_at"):
        v = row.get(k)
        if v not in (None, ""):
            trade[k] = v
    return trade


def rows_to_trades(rows: Sequence[dict]) -> List[dict]:
    """Map a list of closed rows to trade records, dropping unmeasurable ones."""
    out: List[dict] = []
    for r in rows or []:
        t = episode_to_trade(r)
        if t is not None:
            out.append(t)
    return out


# --------------------------------------------------------------------------- #
# Read (the only I/O; read-only, offline)
# --------------------------------------------------------------------------- #
def load_trades(db_path: str = _DB_DEFAULT, *, strat: Optional[str] = None,
                since: Optional[str] = None, until: Optional[str] = None
                ) -> List[dict]:
    """Load closed episodes from ``db_path`` and map them to trade records.

    Read-only. Fail-open: a missing DB or read error yields an empty list
    rather than raising, so the study degrades to an empty report.
    """
    if db_path != ":memory:" and not os.path.exists(db_path):
        print(f"[episodes_study] no database at {db_path!r}; empty study.")
        return []
    try:
        from episode_store import EpisodeStore
        store = EpisodeStore(db_path)
        try:
            rows = store.completed(strat=strat, since=since, until=until)
        finally:
            store.close()
    except Exception as exc:  # pragma: no cover - defensive
        print(f"[episodes_study] load failed: {exc}")
        return []
    return rows_to_trades(rows)


# --------------------------------------------------------------------------- #
# Report (pure)
# --------------------------------------------------------------------------- #
def _period_key(as_of: Any, period: str = "month") -> str:
    """Cohort label for an ISO ``as_of`` timestamp.

    ``day`` -> 'YYYY-MM-DD', ``week`` -> 'YYYY-Www' (ISO week), ``month`` ->
    'YYYY-MM' (default). Fail-open: an unparseable/empty stamp -> 'unknown'.
    """
    s = str(as_of or "")
    if not s:
        return "unknown"
    if period == "day":
        return s[:10] or "unknown"
    if period == "week":
        try:
            from datetime import datetime
            clean = s.replace("Z", "").split("+")[0].split(".")[0]
            dt = datetime.fromisoformat(clean)
            y, w, _ = dt.isocalendar()
            return f"{y}-W{w:02d}"
        except Exception:
            return (s[:10] or "unknown")
    return s[:7] or "unknown"  # month


def cohort_breakdown(trades: Sequence[dict], *, period: str = "month"
                     ) -> Dict[str, Dict[str, Any]]:
    """Group trades into entry-date cohorts, then split CALL vs PUT within each.

    Answers "is the CALL bleed cohort-specific or persistent?" Returns
    ``{cohort_label: {all, call, put}}`` with per-slice ``compute_metrics``.
    Deterministic cohort order.
    """
    groups: Dict[str, List[dict]] = {}
    for t in trades or []:
        if not isinstance(t, dict):
            continue
        groups.setdefault(_period_key(t.get("as_of"), period), []).append(t)
    out: Dict[str, Dict[str, Any]] = {}
    for label in sorted(groups.keys()):
        sub = groups[label]
        calls = [t for t in sub
                 if str(t.get("direction") or "").lower() == _metrics.DIR_CALL]
        puts = [t for t in sub
                if str(t.get("direction") or "").lower() == _metrics.DIR_PUT]
        out[label] = {
            "all": _metrics.compute_metrics(sub),
            "call": _metrics.compute_metrics(calls),
            "put": _metrics.compute_metrics(puts),
        }
    return out


_HOLD_LABELS = ("0d", "1d", "2-3d", "4-7d", "8d+")


def _hold_bucket(hd: float) -> str:
    if hd < 1.0:
        return "0d"
    if hd < 2.0:
        return "1d"
    if hd < 4.0:
        return "2-3d"
    if hd < 8.0:
        return "4-7d"
    return "8d+"


def hold_days_breakdown(trades: Sequence[dict], *,
                        direction: Optional[str] = None
                        ) -> Dict[str, Dict[str, Any]]:
    """Bucket trades by holding time (0d / 1d / 2-3d / 4-7d / 8d+), optionally
    restricted to one ``direction`` (call|put). Answers whether a leg's edge is
    entry-driven or degrades the longer it is held. Rows with no ``hold_days``
    land in 'unknown'. Deterministic bucket order.
    """
    sub = [t for t in (trades or []) if isinstance(t, dict)
           and (direction is None
                or str(t.get("direction") or "").lower() == direction)]
    groups: Dict[str, List[dict]] = {}
    for t in sub:
        hd = t.get("hold_days")
        try:
            hd = float(hd)
        except (TypeError, ValueError):
            groups.setdefault("unknown", []).append(t)
            continue
        groups.setdefault(_hold_bucket(hd), []).append(t)
    order = [lbl for lbl in _HOLD_LABELS if lbl in groups]
    if "unknown" in groups:
        order.append("unknown")
    return {label: _metrics.compute_metrics(groups[label]) for label in order}


_DTE_LABELS = ("0-7", "8-21", "22-45", "46+")


def _dte_bucket(dte: float) -> str:
    if dte < 8.0:
        return "0-7"
    if dte < 22.0:
        return "8-21"
    if dte < 46.0:
        return "22-45"
    return "46+"


def dte_breakdown(trades: Sequence[dict], *,
                  direction: Optional[str] = None
                  ) -> Dict[str, Dict[str, Any]]:
    """Bucket trades by days-to-expiry AT ENTRY (0-7 / 8-21 / 22-45 / 46+),
    optionally restricted to one ``direction`` (call|put).

    ``dte_entry`` is the sole genuine entry-time knob preserved in the
    broker-roundtrip backfill, so this is the closest available proxy for an
    "entry filter would have helped" question when no conviction/EV was
    recorded. Rows with no ``dte_entry`` land in 'unknown'. Deterministic order.
    """
    sub = [t for t in (trades or []) if isinstance(t, dict)
           and (direction is None
                or str(t.get("direction") or "").lower() == direction)]
    groups: Dict[str, List[dict]] = {}
    for t in sub:
        dte = t.get("dte_entry")
        try:
            dte = float(dte)
        except (TypeError, ValueError):
            groups.setdefault("unknown", []).append(t)
            continue
        groups.setdefault(_dte_bucket(dte), []).append(t)
    order = [lbl for lbl in _DTE_LABELS if lbl in groups]
    if "unknown" in groups:
        order.append("unknown")
    return {label: _metrics.compute_metrics(groups[label]) for label in order}


def build_report(trades: Sequence[dict], *,
                 window_minutes: Optional[float] = None,
                 cohort_period: Optional[str] = None,
                 hold_split: Optional[str] = None,
                 dte_split: Optional[str] = None) -> Dict[str, Any]:
    """Compute the overall metrics + categorical breakdowns for realized
    trades. Pure, deterministic, JSON-safe.

    ``cohort_period`` (day|week|month), when set, adds a ``cohorts`` section
    slicing CALL vs PUT within each entry-date cohort.
    """
    trades = [t for t in (trades or []) if isinstance(t, dict)]
    report: Dict[str, Any] = {
        "n_trades": len(trades),
        "overall": _metrics.compute_metrics(trades, window_minutes=window_minutes),
        "direction_calibration": _metrics.direction_calibration(trades),
        "breakdowns": {
            "by_direction": _metrics.breakdown_by_direction(trades),
            "by_mode": _metrics.breakdown_by_mode(trades),
            "by_regime": _metrics.breakdown_by_regime(trades),
        },
    }
    if cohort_period:
        report["cohort_period"] = cohort_period
        report["cohorts"] = cohort_breakdown(trades, period=cohort_period)
    if hold_split:
        direction = None if hold_split == "all" else hold_split
        report["hold_split"] = {
            "direction": hold_split,
            "buckets": hold_days_breakdown(trades, direction=direction),
        }
    if dte_split:
        direction = None if dte_split == "all" else dte_split
        report["dte_split"] = {
            "direction": dte_split,
            "buckets": dte_breakdown(trades, direction=direction),
        }
    return report


def _fmt_num(v: Any, nd: int = 2) -> str:
    if v is None:
        return "n/a"
    try:
        return f"{float(v):.{nd}f}"
    except (TypeError, ValueError):
        return str(v)


def _fmt_metrics_line(label: str, m: Dict[str, Any]) -> str:
    pf = m.get("profit_factor")
    return (f"  {label:<18} n={m.get('trade_count', 0):<5} "
            f"win={_fmt_num(m.get('win_rate'), 3):<6} "
            f"exp=${_fmt_num(m.get('expectancy')):<9} "
            f"pnl=${_fmt_num(m.get('total_pnl')):<11} "
            f"pf={_fmt_num(pf) if pf is not None else 'n/a'}")


def format_report(report: Dict[str, Any]) -> str:
    """Render ``build_report`` output as a plain-text (ASCII) block."""
    if not isinstance(report, dict):
        return "Oracle Lab episodes study: (no data)"
    lines: List[str] = []
    lines.append("=" * 70)
    lines.append("Oracle Lab — realized episodes study (offline analytics)")
    lines.append("=" * 70)
    n = report.get("n_trades", 0)
    lines.append(f"closed trades with realized PnL: {n}")
    if not n:
        lines.append("(no measurable trades)")
        return "\n".join(lines)

    overall = report.get("overall", {})
    lines.append("")
    lines.append("OVERALL")
    lines.append(_fmt_metrics_line("all", overall))
    lines.append(f"    avg_win=${_fmt_num(overall.get('avg_win'))}  "
                 f"avg_loss=${_fmt_num(overall.get('avg_loss'))}  "
                 f"max_dd=${_fmt_num(overall.get('max_drawdown'))}  "
                 f"sharpe={_fmt_num(overall.get('sharpe'), 3)}")

    dc = report.get("direction_calibration", {})
    if dc:
        lines.append("")
        lines.append("DIRECTION CALIBRATION")
        for side in ("call", "put"):
            if side in dc:
                s = dc[side]
                lines.append(f"  {side:<6} n={s.get('n', 0):<5} "
                             f"win={_fmt_num(s.get('win_rate'), 3):<6} "
                             f"exp=${_fmt_num(s.get('expectancy'))}")

    bds = report.get("breakdowns", {})
    for title, key in (("BY DIRECTION", "by_direction"),
                       ("BY STRATEGY MODE", "by_mode"),
                       ("BY REGIME", "by_regime")):
        grp = bds.get(key, {})
        if not grp:
            continue
        lines.append("")
        lines.append(title)
        for label in sorted(grp.keys()):
            lines.append(_fmt_metrics_line(label, grp[label]))

    cohorts = report.get("cohorts")
    if cohorts:
        lines.append("")
        lines.append(f"BY COHORT ({report.get('cohort_period', 'month')}) "
                     f"— CALL vs PUT")
        for label in sorted(cohorts.keys()):
            c = cohorts[label]["call"]
            p = cohorts[label]["put"]
            lines.append(
                f"  {label:<10} "
                f"CALL n={c.get('trade_count', 0):<4} "
                f"exp=${_fmt_num(c.get('expectancy')):>8} "
                f"pnl=${_fmt_num(c.get('total_pnl')):>10}  |  "
                f"PUT n={p.get('trade_count', 0):<4} "
                f"exp=${_fmt_num(p.get('expectancy')):>8} "
                f"pnl=${_fmt_num(p.get('total_pnl')):>10}")

    hs = report.get("hold_split")
    if hs and hs.get("buckets"):
        lines.append("")
        lines.append(f"BY HOLD-DAYS ({hs.get('direction', 'all')})")
        for label, m in hs["buckets"].items():
            lines.append(_fmt_metrics_line(label, m))

    ds = report.get("dte_split")
    if ds and ds.get("buckets"):
        lines.append("")
        lines.append(f"BY DTE-AT-ENTRY ({ds.get('direction', 'all')})")
        for label, m in ds["buckets"].items():
            lines.append(_fmt_metrics_line(label, m))
    return "\n".join(lines)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _safe_print(text: str) -> None:
    """Print tolerant of a narrow console encoding (Windows cp1252)."""
    try:
        print(text)
    except UnicodeEncodeError:
        enc = getattr(sys.stdout, "encoding", None) or "utf-8"
        print(text.encode(enc, "replace").decode(enc, "replace"))


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    p = argparse.ArgumentParser(
        description="Oracle Lab: study realized trades from episodes.db")
    p.add_argument("--db", default=_DB_DEFAULT, help="path to episodes.db")
    p.add_argument("--strat", default=None, help="filter to one strat")
    p.add_argument("--since", default=None, help="as_of >= (ISO)")
    p.add_argument("--until", default=None, help="as_of <= (ISO)")
    p.add_argument("--cohort", choices=("day", "week", "month"), default=None,
                   help="slice CALL vs PUT by entry-date cohort")
    p.add_argument("--hold-split", choices=("call", "put", "all"), default=None,
                   help="bucket by holding time (optionally one direction)")
    p.add_argument("--dte-split", choices=("call", "put", "all"), default=None,
                   help="bucket by days-to-expiry at entry (optionally one "
                        "direction)")
    p.add_argument("--json", action="store_true",
                   help="emit the JSON report instead of text")
    args = p.parse_args(argv)

    trades = load_trades(args.db, strat=args.strat, since=args.since,
                         until=args.until)
    report = build_report(trades, cohort_period=args.cohort,
                          hold_split=args.hold_split,
                          dte_split=args.dte_split)
    if args.json:
        _safe_print(json.dumps(report, indent=2, sort_keys=True, default=str))
    else:
        _safe_print(format_report(report))
    return 0


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes — in-memory DB only)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # --- Pure mapping ------------------------------------------------------- #
    t = episode_to_trade({
        "net_pnl_dollars": 120.0, "net_pnl_pct": 12.0, "chosen_action": "CALL",
        "mode": "backfill", "hold_days": 2,
        "features_json": json.dumps({"oracle": {"regime": "trending"}}),
        "symbol": "SPY", "outcome": "take_profit",
    })
    if t is None or t["pnl"] != 120.0 or t["direction"] != "call":
        print("FAIL: episode_to_trade basic", t); ok = False
    if t.get("regime") != "trending" or t.get("strategy_mode") != "backfill":
        print("FAIL: episode_to_trade context", t); ok = False

    # No realized PnL -> unmeasurable -> None.
    if episode_to_trade({"chosen_action": "CALL"}) is not None:
        print("FAIL: row without pnl should map to None"); ok = False

    # Direction falls back to strat substring.
    if _direction_from_row({"strat": "long_put"}) != "put":
        print("FAIL: strat-substring direction"); ok = False

    # Regime extraction variants + junk tolerance.
    if _regime_from_features('{"regime": "choppy"}') != "choppy":
        print("FAIL: top-level regime"); ok = False
    if _regime_from_features("not json") is not None:
        print("FAIL: junk features -> None"); ok = False

    # --- Report over hand-built trades -------------------------------------- #
    trades = [
        {"pnl": 100.0, "return_pct": 10.0, "direction": "call",
         "strategy_mode": "backfill", "regime": "trending", "hold_days": 1},
        {"pnl": -50.0, "return_pct": -5.0, "direction": "put",
         "strategy_mode": "backfill", "regime": "choppy", "hold_days": 3},
    ]
    rep = build_report(trades)
    if rep["n_trades"] != 2 or rep["overall"]["trade_count"] != 2:
        print("FAIL: report n_trades", rep); ok = False
    if "call" not in rep["breakdowns"]["by_direction"]:
        print("FAIL: direction breakdown missing", rep["breakdowns"]); ok = False
    if "PROFIT" in format_report(rep):  # sanity: no crash, returns text
        pass
    if "realized episodes study" not in format_report(rep):
        print("FAIL: format_report header"); ok = False

    # --- Cohort slice (CALL vs PUT by entry-date) --------------------------- #
    cohort_trades = [
        {"pnl": -100.0, "direction": "call", "as_of": "2024-06-11T16:00:00"},
        {"pnl": 20.0, "direction": "put", "as_of": "2024-06-11T15:00:00"},
        {"pnl": -30.0, "direction": "call", "as_of": "2024-07-02T16:00:00"},
    ]
    crep = build_report(cohort_trades, cohort_period="month")
    coh = crep.get("cohorts", {})
    if set(coh.keys()) != {"2024-06", "2024-07"}:
        print("FAIL: cohort keys", list(coh.keys())); ok = False
    jun = coh.get("2024-06", {})
    if jun.get("call", {}).get("trade_count") != 1 or \
            jun.get("put", {}).get("trade_count") != 1:
        print("FAIL: cohort call/put split", jun); ok = False
    if abs(jun.get("call", {}).get("total_pnl", 0.0) - (-100.0)) > 1e-9:
        print("FAIL: cohort call pnl", jun); ok = False
    if "BY COHORT" not in format_report(crep):
        print("FAIL: cohort not rendered"); ok = False
    if _period_key("2024-06-11T16:00:00", "day") != "2024-06-11":
        print("FAIL: day period key"); ok = False
    if _period_key("2024-06-11T16:00:00", "week") != "2024-W24":
        print("FAIL: week period key",
              _period_key("2024-06-11T16:00:00", "week")); ok = False
    if _period_key(None, "month") != "unknown":
        print("FAIL: empty period key"); ok = False

    # --- Hold-days split (direction-filtered) ------------------------------- #
    hold_trades = [
        {"pnl": 50.0, "direction": "call", "hold_days": 0},   # 0d call win
        {"pnl": -200.0, "direction": "call", "hold_days": 11},  # 8d+ call loss
        {"pnl": 30.0, "direction": "put", "hold_days": 1},    # excluded (put)
    ]
    hrep = build_report(hold_trades, hold_split="call")
    hb = hrep.get("hold_split", {}).get("buckets", {})
    if list(hb.keys()) != ["0d", "8d+"]:
        print("FAIL: hold buckets/order", list(hb.keys())); ok = False
    if hb.get("0d", {}).get("trade_count") != 1 or \
            hb.get("8d+", {}).get("trade_count") != 1:
        print("FAIL: hold bucket counts", hb); ok = False
    if abs(hb.get("8d+", {}).get("total_pnl", 0.0) - (-200.0)) > 1e-9:
        print("FAIL: hold bucket pnl", hb); ok = False
    if _hold_bucket(0.0) != "0d" or _hold_bucket(3.0) != "2-3d" or \
            _hold_bucket(7.0) != "4-7d" or _hold_bucket(8.0) != "8d+":
        print("FAIL: hold bucket edges"); ok = False
    if "BY HOLD-DAYS (call)" not in format_report(hrep):
        print("FAIL: hold split not rendered"); ok = False
    # Missing hold_days -> 'unknown'.
    urep = build_report([{"pnl": 1.0, "direction": "call"}], hold_split="all")
    if "unknown" not in urep["hold_split"]["buckets"]:
        print("FAIL: missing hold_days -> unknown"); ok = False

    # --- DTE-at-entry split (direction-filtered) ---------------------------- #
    if _dte_from_features('{"dte_entry": 30}') != 30.0:
        print("FAIL: top-level dte_entry"); ok = False
    if _dte_from_features('{"oracle": {"dte_entry": 14}}') != 14.0:
        print("FAIL: nested dte_entry"); ok = False
    if _dte_from_features("not json") is not None:
        print("FAIL: junk features -> dte None"); ok = False
    # episode_to_trade carries dte_entry.
    dt_row = episode_to_trade({
        "net_pnl_dollars": 5.0, "chosen_action": "CALL",
        "features_json": json.dumps({"dte_entry": 40})})
    if dt_row is None or dt_row.get("dte_entry") != 40.0:
        print("FAIL: episode_to_trade dte carry", dt_row); ok = False
    dte_trades = [
        {"pnl": -100.0, "direction": "call", "dte_entry": 3},    # 0-7
        {"pnl": -50.0, "direction": "call", "dte_entry": 30},    # 22-45
        {"pnl": 200.0, "direction": "put", "dte_entry": 30},     # excluded (put)
    ]
    drep = build_report(dte_trades, dte_split="call")
    db = drep.get("dte_split", {}).get("buckets", {})
    if list(db.keys()) != ["0-7", "22-45"]:
        print("FAIL: dte buckets/order", list(db.keys())); ok = False
    if db.get("22-45", {}).get("trade_count") != 1 or \
            abs(db.get("22-45", {}).get("total_pnl", 0.0) - (-50.0)) > 1e-9:
        print("FAIL: dte bucket 22-45", db); ok = False
    if _dte_bucket(7.0) != "0-7" or _dte_bucket(8.0) != "8-21" or \
            _dte_bucket(21.0) != "8-21" or _dte_bucket(22.0) != "22-45" or \
            _dte_bucket(45.0) != "22-45" or _dte_bucket(46.0) != "46+":
        print("FAIL: dte bucket edges"); ok = False
    if "BY DTE-AT-ENTRY (call)" not in format_report(drep):
        print("FAIL: dte split not rendered"); ok = False
    # Missing dte_entry -> 'unknown'.
    udrep = build_report([{"pnl": 1.0, "direction": "call"}], dte_split="all")
    if "unknown" not in udrep["dte_split"]["buckets"]:
        print("FAIL: missing dte_entry -> unknown"); ok = False

    # Empty report never raises.
    empty = build_report([])
    if empty["n_trades"] != 0:
        print("FAIL: empty report", empty); ok = False
    if "no measurable trades" not in format_report(empty):
        print("FAIL: empty format", format_report(empty)); ok = False

    # --- End-to-end via an in-memory EpisodeStore (no file, no creds) ------- #
    try:
        from episode_store import EpisodeStore
        store = EpisodeStore(":memory:")
        did = store.log_decision(
            symbol="SPY250101C00500000", underlying="SPY", strat="long_call",
            features={"oracle": {"regime": "trending"}}, quote=None,
            modeled_cost=None, rule_action="CALL", rule_confidence=70.0,
            gate=None, chosen_action="CALL", qty=1, mode="backfill")
        store.record_outcome(did, net_pnl_pct=10.0, net_pnl_dollars=90.0,
                             hold_days=1, outcome="take_profit")
        # log an OPEN decision too -> must be excluded (no outcome).
        store.log_decision(
            symbol="QQQ", underlying="QQQ", strat="long_put",
            features={}, quote=None, modeled_cost=None, rule_action="PUT",
            rule_confidence=60.0, gate=None, chosen_action="PUT", qty=1,
            mode="backfill")
        mapped = rows_to_trades(store.completed())
        store.close()
        if len(mapped) != 1 or mapped[0]["pnl"] != 90.0:
            print("FAIL: store round-trip mapping", mapped); ok = False
        if mapped[0].get("regime") != "trending":
            print("FAIL: store round-trip regime", mapped); ok = False
    except Exception as exc:  # pragma: no cover
        print("FAIL: in-memory store path raised", exc); ok = False

    # load_trades on a nonexistent DB is fail-open (empty, no raise).
    if load_trades("does_not_exist_xyz.db") != []:
        print("FAIL: missing DB should be empty"); ok = False

    print("lab.episodes_study self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    if "--self-test" in sys.argv[1:]:
        sys.exit(_self_test())
    sys.exit(main())
