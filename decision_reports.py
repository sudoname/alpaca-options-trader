"""
decision_reports.py — READ-ONLY analytics over the RL episode store
(``episodes.db``, table ``episodes``).

Surfaces the per-decision record the bot already writes on every scan:

  * decision log   — one row per decision (traded or blocked), including the
                     per-agent votes stored at ``features_json.evidence.agent_votes``
  * decision funnel — how many decisions become trades vs. gate blocks vs.
                     cap skips, plus the top block reasons
  * agent contributions — per-agent agreement and hit rates over real trades

It is IMPORT-ONLY over a read-only store: every query is a SELECT and the
module never creates ``episodes.db`` when it is missing (the existence guard
runs before any ``sqlite3.connect``). Every provider fails open: a missing or
corrupt store yields a verdict-carrying dict, never an exception.

NOTE on P/L semantics: SKIP rows carry a *counterfactual* ``net_pnl_pct``
(what the skipped trade would have done). The decision log flags those rows
with ``pnl_is_counterfactual`` so the UI never presents them as realized money.
"""

import json
import os
import sqlite3
from typing import Any, Dict, List, Optional

VERDICT_OK = "OK"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# Canonical filename (relative to cwd — matches single_leg_reports.py; the
# dashboard runs with WorkingDirectory at the deploy root).
EPISODE_DB_FILE = "episodes.db"

_SCOPES = ("traded", "blocked", "all")


# --------------------------------------------------------------------------- #
# fail-open helpers (read only)
# --------------------------------------------------------------------------- #
def _open_db(path: str) -> Optional[sqlite3.Connection]:
    """Open an EXISTING episodes.db; None when absent (never creates one)."""
    if not os.path.exists(path):
        return None
    try:
        con = sqlite3.connect(path)
        con.row_factory = sqlite3.Row
        return con
    except Exception:
        return None


def _close(con) -> None:
    try:
        if con is not None:
            con.close()
    except Exception:
        pass


def _loads(s) -> Optional[Dict]:
    """json.loads that fails open to None and only admits dicts."""
    try:
        v = json.loads(s)
        return v if isinstance(v, dict) else None
    except Exception:
        return None


def _coerce_float(v) -> Optional[float]:
    try:
        return float(v)
    except (TypeError, ValueError):
        return None


def _reason_prefix(reason) -> Optional[str]:
    """Leading phrase of a block reason, e.g. 'profitability gate: pop 0.30 …'
    -> 'profitability gate'. Splits on the first ':' or '('; fails open."""
    try:
        s = str(reason or "").strip()
        if not s:
            return None
        for sep in (":", "("):
            i = s.find(sep)
            if i > 0:
                s = s[:i]
        s = s.strip()
        return s or None
    except Exception:
        return None


def _votes_list(features) -> Optional[List[Dict]]:
    """Per-agent votes from ``features_json.evidence.agent_votes``.

    Returns a name-sorted list of ``{agent, bull, bear, conf, net}`` rows or
    None when the decision carries no votes (older rows / SKIPs without
    evidence). Fails open to None.
    """
    try:
        ev = features.get("evidence") if isinstance(features, dict) else None
        votes = ev.get("agent_votes") if isinstance(ev, dict) else None
        if not isinstance(votes, dict) or not votes:
            return None
        out = []
        for agent in sorted(votes):
            v = votes.get(agent)
            if not isinstance(v, dict):
                continue
            out.append({
                "agent": str(agent),
                "bull": _coerce_float(v.get("bull")),
                "bear": _coerce_float(v.get("bear")),
                "conf": _coerce_float(v.get("conf")),
                "net": _coerce_float(v.get("net")),
            })
        return out or None
    except Exception:
        return None


def _scope_where(scope: str) -> str:
    if scope == "traded":
        return "WHERE chosen_action IN ('CALL','PUT')"
    if scope == "blocked":
        return "WHERE chosen_action = 'SKIP'"
    return ""


# --------------------------------------------------------------------------- #
# Decision log
# --------------------------------------------------------------------------- #
def compute_decision_log(
    *,
    episode_db: str = EPISODE_DB_FILE,
    limit: int = 50,
    scope: str = "all",
) -> Dict:
    """Newest-first per-decision rows from the episode store (SELECT only).

    scope: traded (CALL/PUT) | blocked (SKIP) | all. limit clamped 1-200.
    Each row carries the gate verdict, the risk block reason (truncated +
    full + normalized prefix), the outcome/P-L fields, the regime evidence,
    and the per-agent votes parsed from evidence.agent_votes. SKIP rows set
    pnl_is_counterfactual so their P/L is never read as realized.
    """
    try:
        limit = int(limit)
    except (TypeError, ValueError):
        limit = 50
    limit = max(1, min(limit, 200))
    if scope not in _SCOPES:
        scope = "all"

    empty = {"verdict": VERDICT_INSUFFICIENT, "scope": scope, "limit": limit,
             "count": 0, "total": 0, "rows": []}
    con = _open_db(episode_db)
    if con is None:
        return empty
    where = _scope_where(scope)
    try:
        total = con.execute(
            f"SELECT COUNT(*) FROM episodes {where}").fetchone()[0]
        raw = con.execute(
            "SELECT decision_id, as_of, symbol, underlying, strat, "
            "rule_action, rule_confidence, chosen_action, qty, mode, "
            "gate_json, risk_json, outcome, net_pnl_pct, net_pnl_dollars, "
            "hold_days, features_json "
            f"FROM episodes {where} ORDER BY rowid DESC LIMIT ?",
            (limit,)).fetchall()
    except Exception:
        return empty
    finally:
        _close(con)

    rows: List[Dict] = []
    for r in raw:
        feats = _loads(r["features_json"])
        ev = feats.get("evidence") if isinstance(feats, dict) else None
        ev = ev if isinstance(ev, dict) else {}
        gate = _loads(r["gate_json"])
        risk = _loads(r["risk_json"]) or {}
        reason = risk.get("block_reason")
        reason = str(reason) if reason is not None else None
        chosen = r["chosen_action"]
        rows.append({
            "decision_id": r["decision_id"],
            "as_of": r["as_of"],
            "symbol": r["symbol"],
            "underlying": r["underlying"],
            "strat": r["strat"],
            "rule_action": r["rule_action"],
            "rule_confidence": _coerce_float(r["rule_confidence"]),
            "chosen_action": chosen,
            "qty": r["qty"],
            "mode": r["mode"],
            "gate": ({
                "veto": gate.get("veto"),
                "reason": gate.get("reason"),
                "q": _coerce_float(gate.get("q")),
                "visits": gate.get("visits"),
                "confidence": _coerce_float(gate.get("confidence")),
            } if isinstance(gate, dict) else None),
            "block_reason": (reason[:200] if reason else None),
            "block_reason_full": reason,
            "block_reason_prefix": _reason_prefix(reason),
            "outcome": r["outcome"],
            "net_pnl_pct": _coerce_float(r["net_pnl_pct"]),
            "net_pnl_dollars": _coerce_float(r["net_pnl_dollars"]),
            "hold_days": r["hold_days"],
            "pnl_is_counterfactual": chosen == "SKIP",
            "regime_label": ev.get("regime_label"),
            "regime_confidence": _coerce_float(ev.get("regime_confidence")),
            "conviction": _coerce_float(ev.get("conviction")),
            "direction": ev.get("direction"),
            "agent_votes": _votes_list(feats),
        })

    # rowid is near-chronological; re-sort the small page by as_of for safety.
    rows.sort(key=lambda x: str(x.get("as_of") or ""), reverse=True)
    return {"verdict": VERDICT_OK if rows else VERDICT_INSUFFICIENT,
            "scope": scope, "limit": limit, "count": len(rows),
            "total": total, "rows": rows}


# --------------------------------------------------------------------------- #
# Decision funnel
# --------------------------------------------------------------------------- #
def compute_decision_funnel(*, episode_db: str = EPISODE_DB_FILE) -> Dict:
    """Where decisions go: traded vs. gate-blocked vs. cap-skipped (SELECT only).

    Pure SQL GROUP BYs over chosen_action / mode / outcome (never touches
    features_json), plus the top block-reason prefixes parsed from the
    risk_json of SKIP rows.
    """
    empty = {"verdict": VERDICT_INSUFFICIENT, "total": 0, "completed": 0,
             "open": 0, "by_chosen_action": {}, "by_mode": {},
             "by_outcome": {}, "block_reasons": [], "funnel": []}
    con = _open_db(episode_db)
    if con is None:
        return empty
    try:
        total = con.execute("SELECT COUNT(*) FROM episodes").fetchone()[0]
        completed = con.execute(
            "SELECT COUNT(*) FROM episodes WHERE outcome IS NOT NULL"
        ).fetchone()[0]
        by_action = {str(k if k is not None else "?"): n for k, n in con.execute(
            "SELECT chosen_action, COUNT(*) FROM episodes "
            "GROUP BY chosen_action")}
        by_mode = {str(k if k is not None else "?"): n for k, n in con.execute(
            "SELECT mode, COUNT(*) FROM episodes GROUP BY mode")}
        by_outcome = {str(k if k is not None else "open"): n for k, n in con.execute(
            "SELECT outcome, COUNT(*) FROM episodes GROUP BY outcome")}
        reasons: Dict[str, int] = {}
        for (rj,) in con.execute(
                "SELECT risk_json FROM episodes "
                "WHERE chosen_action = 'SKIP'"):
            risk = _loads(rj) or {}
            pref = _reason_prefix(risk.get("block_reason"))
            if pref:
                reasons[pref] = reasons.get(pref, 0) + 1
    except Exception:
        return empty
    finally:
        _close(con)

    traded = by_action.get("CALL", 0) + by_action.get("PUT", 0)
    blocked = by_action.get("SKIP", 0)
    cap_skipped = by_mode.get("cap-skip-cf", 0)
    top_reasons = sorted(reasons.items(), key=lambda kv: (-kv[1], kv[0]))[:10]

    return {
        "verdict": VERDICT_OK if total else VERDICT_INSUFFICIENT,
        "total": total,
        "completed": completed,
        "open": total - completed,
        "by_chosen_action": by_action,
        "by_mode": by_mode,
        "by_outcome": by_outcome,
        "block_reasons": [{"prefix": p, "count": n} for p, n in top_reasons],
        "funnel": [
            {"stage": "decisions", "count": total},
            {"stage": "traded", "count": traded},
            {"stage": "blocked", "count": blocked},
            {"stage": "cap_skipped", "count": cap_skipped},
        ],
    }


# --------------------------------------------------------------------------- #
# Agent contributions
# --------------------------------------------------------------------------- #
def compute_agent_contributions(
    *,
    episode_db: str = EPISODE_DB_FILE,
    cap: int = 5000,
) -> Dict:
    """Per-agent vote quality over REAL trades (CALL/PUT with resolved P/L).

    For each agent found in evidence.agent_votes (discovered, not hardcoded):
    coverage n, agreement_rate (sign of the agent's net vote matches the trade
    direction, +CALL / -PUT; net == 0 counts as not agreeing), hit rate when
    agreeing, hit rate + counts when bullish (net > 0) / bearish (net < 0),
    mean trade P/L split into winning and losing trades the agent covered, and
    mean confidence. Rates are None when their denominator is 0. SELECT only,
    scanning the newest ``cap`` trades.
    """
    try:
        cap = int(cap)
    except (TypeError, ValueError):
        cap = 5000
    cap = max(1, min(cap, 20000))

    empty = {"verdict": VERDICT_INSUFFICIENT, "cap": cap, "n_trades": 0,
             "n_with_votes": 0, "agents": []}
    con = _open_db(episode_db)
    if con is None:
        return empty
    try:
        raw = con.execute(
            "SELECT chosen_action, net_pnl_pct, features_json FROM episodes "
            "WHERE chosen_action IN ('CALL','PUT') "
            "AND net_pnl_pct IS NOT NULL "
            "ORDER BY rowid DESC LIMIT ?", (cap,)).fetchall()
    except Exception:
        return empty
    finally:
        _close(con)

    n_trades = len(raw)
    n_with_votes = 0
    acc: Dict[str, Dict[str, Any]] = {}
    for r in raw:
        votes = _votes_list(_loads(r["features_json"]))
        if not votes:
            continue
        pnl = _coerce_float(r["net_pnl_pct"])
        if pnl is None:
            continue
        n_with_votes += 1
        is_win = pnl > 0
        direction = 1.0 if r["chosen_action"] == "CALL" else -1.0
        for v in votes:
            a = acc.setdefault(v["agent"], {
                "n": 0, "win_sum": 0.0, "win_n": 0,
                "loss_sum": 0.0, "loss_n": 0,
                "agree_n": 0, "agree_hit": 0,
                "bull_n": 0, "bull_hit": 0,
                "bear_n": 0, "bear_hit": 0,
                "conf_sum": 0.0, "conf_n": 0,
            })
            a["n"] += 1
            if is_win:
                a["win_sum"] += pnl
                a["win_n"] += 1
            else:
                a["loss_sum"] += pnl
                a["loss_n"] += 1
            net = v.get("net")
            if net is not None:
                if net * direction > 0:
                    a["agree_n"] += 1
                    if is_win:
                        a["agree_hit"] += 1
                if net > 0:
                    a["bull_n"] += 1
                    if is_win:
                        a["bull_hit"] += 1
                elif net < 0:
                    a["bear_n"] += 1
                    if is_win:
                        a["bear_hit"] += 1
            conf = v.get("conf")
            if conf is not None:
                a["conf_sum"] += conf
                a["conf_n"] += 1

    def _rate(num, den):
        return round(num / den, 4) if den else None

    agents = []
    for name in sorted(acc):
        a = acc[name]
        agents.append({
            "agent": name,
            "n": a["n"],
            "agreement_rate": _rate(a["agree_n"], a["n"]),
            "hit_rate_when_agreeing": _rate(a["agree_hit"], a["agree_n"]),
            "hit_rate_when_bullish": _rate(a["bull_hit"], a["bull_n"]),
            "n_bullish": a["bull_n"],
            "hit_rate_when_bearish": _rate(a["bear_hit"], a["bear_n"]),
            "n_bearish": a["bear_n"],
            "mean_net_win": _rate(a["win_sum"], a["win_n"]),
            "mean_net_loss": _rate(a["loss_sum"], a["loss_n"]),
            "mean_conf": _rate(a["conf_sum"], a["conf_n"]),
        })

    return {"verdict": VERDICT_OK if agents else VERDICT_INSUFFICIENT,
            "cap": cap, "n_trades": n_trades,
            "n_with_votes": n_with_votes, "agents": agents}
