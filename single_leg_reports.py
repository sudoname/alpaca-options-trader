"""
single_leg_reports.py — READ-ONLY analytics over the single-leg intraday bot's
on-disk stores.

This deployment runs the single-leg intraday bot (``smart_trader`` /
``run_alpaca_intraday``), whose live activity lands in:

  * ``active_trades.json``     — open single-leg option positions (JSON list)
  * ``trading_history.json``   — closed trades under a ``"trades"`` key (JSON dict)
  * ``realized_pnl_log.json``  — realized dollar P/L entries (JSON list)
  * ``episodes.db``            — RL decision episodes (SQLite ``episodes`` table)

The Oracle 3.0 / spread-paper analytics functions the dashboard normally wires
up read a *different* lineage (``spread_paper_*.json`` / ``oracle_training_dataset``)
which is dormant here, so those widgets correctly report INSUFFICIENT_DATA. This
module surfaces the data that actually exists on a single-leg box.

It is IMPORT-ONLY over read-only stores: it opens files for reading and queries
the episode store with SELECTs. It never writes trading state and has no path to
open, size, price, gate, or close any position. Every function fails open: a
missing/corrupt source yields a verdict-carrying dict, never an exception.
"""

import json
import os
from datetime import datetime
from typing import Any, Dict, List

VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# Canonical filenames (relative to cwd — matches the trading modules; the
# dashboard runs with WorkingDirectory at the deploy root).
ACTIVE_TRADES_FILE = "active_trades.json"
TRADE_HISTORY_FILE = "trading_history.json"
REALIZED_PNL_FILE = "realized_pnl_log.json"
EPISODE_DB_FILE = "episodes.db"


# --------------------------------------------------------------------------- #
# fail-open loaders (read only)
# --------------------------------------------------------------------------- #
def _read_json(path: str) -> Any:
    try:
        if os.path.exists(path):
            with open(path, "r") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _load_active_trades(path: str = ACTIVE_TRADES_FILE) -> List[Dict]:
    """active_trades.json is a JSON list (see run_alpaca_intraday._read_active_trades)."""
    data = _read_json(path)
    return data if isinstance(data, list) else []


def _load_history_trades(path: str = TRADE_HISTORY_FILE) -> List[Dict]:
    """trading_history.json wraps closed trades under a ``"trades"`` key."""
    data = _read_json(path)
    if isinstance(data, dict):
        trades = data.get("trades")
        return trades if isinstance(trades, list) else []
    return []


def _load_realized(path: str = REALIZED_PNL_FILE) -> List[Dict]:
    """realized_pnl_log.json is a JSON list of {date, timestamp, amount, symbol}."""
    data = _read_json(path)
    return data if isinstance(data, list) else []


def _coerce_float(v):
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


# --------------------------------------------------------------------------- #
# KPI summary
# --------------------------------------------------------------------------- #
def compute_single_leg_kpis(
    *,
    active_path: str = ACTIVE_TRADES_FILE,
    history_path: str = TRADE_HISTORY_FILE,
    realized_path: str = REALIZED_PNL_FILE,
    today: str = None,
) -> Dict:
    """Headline numbers for the single-leg deployment.

    Returns a dict with ``verdict`` plus realized dollar P/L (all-time + today),
    open-position count, closed-trade count, win rate and average return. The
    dollar figures come from ``realized_pnl_log.json`` (the bot's canonical
    realized-P/L ledger); win rate / average return come from closed trades'
    ``pnl_percent`` (stored in percent units, e.g. 15.0 == +15%).
    """
    active = _load_active_trades(active_path)
    history = _load_history_trades(history_path)
    realized = _load_realized(realized_path)

    today = today or datetime.now().date().isoformat()
    realized_total = 0.0
    today_realized = 0.0
    for r in realized:
        amt = _coerce_float(r.get("amount"))
        if amt is None:
            continue
        realized_total += amt
        if r.get("date") == today:
            today_realized += amt

    pcts = [p for p in (_coerce_float(t.get("pnl_percent")) for t in history)
            if p is not None]
    wins = sum(1 for p in pcts if p > 0)
    win_rate = (wins / len(pcts)) if pcts else 0.0
    avg_return_pct = (sum(pcts) / len(pcts)) if pcts else 0.0

    has_data = bool(active or history or realized)
    return {
        "verdict": VERDICT_OK if has_data else VERDICT_INSUFFICIENT,
        "realized_total": round(realized_total, 2),
        "today_realized": round(today_realized, 2),
        "open_positions": len(active),
        "closed_trades": len(history),
        "win_rate": win_rate,
        "wins": wins,
        "losses": len(pcts) - wins,
        "avg_return_pct": avg_return_pct,
    }


# --------------------------------------------------------------------------- #
# Open positions
# --------------------------------------------------------------------------- #
def compute_single_leg_positions(*, active_path: str = ACTIVE_TRADES_FILE) -> Dict:
    """The bot's currently-open single-leg option positions (display shape)."""
    active = _load_active_trades(active_path)
    if not active:
        return {"verdict": VERDICT_INSUFFICIENT, "positions": [], "count": 0}

    positions = []
    for t in active:
        if not isinstance(t, dict):
            continue
        metrics = t.get("metrics") if isinstance(t.get("metrics"), dict) else {}
        positions.append({
            "symbol": t.get("symbol"),
            "underlying": t.get("underlying_symbol"),
            "quantity": t.get("quantity"),
            "entry_price": t.get("entry_price"),
            "entry_time": t.get("entry_time"),
            "stop_loss_trigger": t.get("stop_loss_trigger"),
            "take_profit_trigger": t.get("take_profit_trigger"),
            "highest_price": t.get("highest_price"),
            "source": t.get("source"),
            "expected_value": metrics.get("expected_value"),
            "probability_of_profit": metrics.get("probability_of_profit"),
        })
    return {"verdict": VERDICT_OK, "positions": positions, "count": len(positions)}


# --------------------------------------------------------------------------- #
# RL episodes (episodes.db)
# --------------------------------------------------------------------------- #
def compute_single_leg_episodes(*, episode_db: str = EPISODE_DB_FILE) -> Dict:
    """Aggregate the RL episode store: completion/win stats + action/outcome mix.

    Opens the SQLite store read-only via ``EpisodeStore`` (SELECT queries only).
    Guarded so we never *create* an empty DB on a box that has none.
    """
    if not os.path.exists(episode_db):
        return {"verdict": VERDICT_INSUFFICIENT, "stats": {},
                "chosen_action_counts": {}, "outcome_counts": {}}

    try:
        from episode_store import EpisodeStore
    except Exception as e:  # pragma: no cover - import guarded for fail-open
        return {"verdict": "ERROR", "error": str(e)}

    store = None
    try:
        store = EpisodeStore(episode_db)
        stats = store.stats()
        completed = store.completed()
    except Exception as e:  # pragma: no cover - exercised via fail-open path
        return {"verdict": "ERROR", "error": str(e)}
    finally:
        if store is not None:
            try:
                store.close()
            except Exception:
                pass

    chosen: Dict[str, int] = {}
    outcomes: Dict[str, int] = {}
    for row in completed:
        ca = str(row.get("chosen_action") or "?")
        chosen[ca] = chosen.get(ca, 0) + 1
        oc = str(row.get("outcome") or "?")
        outcomes[oc] = outcomes.get(oc, 0) + 1

    verdict = VERDICT_OK if stats.get("total") else VERDICT_INSUFFICIENT
    return {
        "verdict": verdict,
        "stats": stats,
        "chosen_action_counts": chosen,
        "outcome_counts": outcomes,
    }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses temp files / an in-memory DB)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    d = tempfile.mkdtemp()

    # 1. Empty / missing sources -> INSUFFICIENT_DATA, never raise.
    k = compute_single_leg_kpis(
        active_path=os.path.join(d, "none.json"),
        history_path=os.path.join(d, "none2.json"),
        realized_path=os.path.join(d, "none3.json"))
    if k.get("verdict") != VERDICT_INSUFFICIENT or k.get("open_positions") != 0:
        print("FAIL: empty kpis should be INSUFFICIENT_DATA:", k); ok = False
    if compute_single_leg_positions(
            active_path=os.path.join(d, "none.json")).get("verdict") \
            != VERDICT_INSUFFICIENT:
        print("FAIL: empty positions should be INSUFFICIENT_DATA"); ok = False
    if compute_single_leg_episodes(
            episode_db=os.path.join(d, "none.db")).get("verdict") \
            != VERDICT_INSUFFICIENT:
        print("FAIL: missing episodes.db should be INSUFFICIENT_DATA"); ok = False

    # 2. Populated sources -> OK with correct aggregates.
    today = datetime.now().date().isoformat()
    active_p = os.path.join(d, "active.json")
    hist_p = os.path.join(d, "history.json")
    real_p = os.path.join(d, "realized.json")
    with open(active_p, "w") as f:
        json.dump([
            {"symbol": "SPY260101C00500000", "underlying_symbol": "SPY",
             "quantity": 2, "entry_price": 1.50, "entry_time": "2026-01-01T10:00:00",
             "metrics": {"expected_value": 0.2, "probability_of_profit": 0.55}},
            {"symbol": "QQQ260101P00400000", "underlying_symbol": "QQQ",
             "quantity": 1, "entry_price": 2.0},
        ], f)
    with open(hist_p, "w") as f:
        json.dump({"trades": [
            {"symbol": "A", "pnl_percent": 20.0},
            {"symbol": "B", "pnl_percent": -10.0},
            {"symbol": "C", "pnl_percent": 5.0},
        ]}, f)
    with open(real_p, "w") as f:
        json.dump([
            {"date": today, "amount": 120.0, "symbol": "A"},
            {"date": today, "amount": -30.0, "symbol": "B"},
            {"date": "2000-01-01", "amount": 999.0, "symbol": "OLD"},
        ], f)

    k = compute_single_leg_kpis(active_path=active_p, history_path=hist_p,
                                realized_path=real_p, today=today)
    if k.get("verdict") != VERDICT_OK:
        print("FAIL: populated kpis verdict:", k); ok = False
    if k.get("open_positions") != 2 or k.get("closed_trades") != 3:
        print("FAIL: kpi counts:", k); ok = False
    if abs(k.get("realized_total") - 1089.0) > 1e-6:
        print("FAIL: realized_total should sum all entries:", k); ok = False
    if abs(k.get("today_realized") - 90.0) > 1e-6:
        print("FAIL: today_realized should be 90:", k); ok = False
    if k.get("wins") != 2 or k.get("losses") != 1:
        print("FAIL: win/loss split:", k); ok = False
    if abs(k.get("win_rate") - (2.0 / 3.0)) > 1e-6:
        print("FAIL: win_rate:", k); ok = False

    p = compute_single_leg_positions(active_path=active_p)
    if p.get("verdict") != VERDICT_OK or p.get("count") != 2:
        print("FAIL: positions:", p); ok = False
    if p["positions"][0].get("underlying") != "SPY":
        print("FAIL: position underlying mapping:", p["positions"][0]); ok = False

    # 3. Episodes against a real in-temp SQLite store with one closed episode.
    db_p = os.path.join(d, "episodes.db")
    try:
        from episode_store import EpisodeStore
        store = EpisodeStore(db_p)
        did = store.log_decision(
            symbol="SPY", underlying="SPY", strat="intraday",
            features={"x": 1}, quote={"bid": 1.0, "ask": 1.1},
            modeled_cost=None, rule_action="CALL", rule_confidence=0.6,
            gate=None, risk=None, chosen_action="CALL",
            qty=1, mode="0DTE")
        store.record_outcome(decision_id=did, fill_price=1.0, exit_price=1.3,
                             gross_pnl_pct=30.0, net_pnl_pct=28.0,
                             net_pnl_dollars=28.0, hold_days=0,
                             outcome="take_profit")
        store.close()
        e = compute_single_leg_episodes(episode_db=db_p)
        if e.get("verdict") != VERDICT_OK:
            print("FAIL: episodes verdict:", e); ok = False
        if e["stats"].get("completed") != 1:
            print("FAIL: episodes completed count:", e["stats"]); ok = False
        if e["chosen_action_counts"].get("CALL") != 1:
            print("FAIL: chosen action count:", e["chosen_action_counts"]); ok = False
        if e["outcome_counts"].get("take_profit") != 1:
            print("FAIL: outcome count:", e["outcome_counts"]); ok = False
    except Exception as ex:
        print("FAIL: episode store integration:", ex); ok = False

    print("single_leg_reports self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
