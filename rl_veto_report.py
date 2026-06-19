"""
Phase 12 — RL Veto Report (analytics only, read-only, fail-open).

The tabular Q-learner (``rl_agent.QLearningAgent``) runs in SHADOW mode: it never
blocks and never places a trade. This report surfaces where the learned policy
would *disagree* with the rule engine, so a future, human-approved veto layer has
evidence to start from. It changes nothing.

Reads two read-only sources:
  * the persisted Q-table (``QLearningAgent.policy_report()``), and
  * closed episodes (``EpisodeStore.completed()``) for the rule action that was
    actually taken in each state and how it realized.

It then flags three patterns over states with enough visits:
  1. CALL states with a NEGATIVE Q     — RL learned buying calls here loses.
  2. PUT states with a POSITIVE Q       — RL learned buying puts here wins.
  3. Rule/RL disagreements              — greedy best_action != the dominant
     rule action, annotated with the rule's realized average net PnL %.

and proposes ADVISORY veto thresholds. Nothing here writes a Q-table, opens,
sizes or blocks any trade, or touches the network. Both sources are optional and
INJECTABLE so the whole module is offline-testable. Every path fails open.
"""

import os
from typing import Dict, List, Optional

RL_VETO_FOOTER = ("Shadow RL — never blocks or places trades; "
                  "suggested vetoes are advisory only.")

VERDICT_DISAGREES = "RL_DISAGREES_WITH_RULES"
VERDICT_ALIGNED = "RL_ALIGNED_WITH_RULES"
VERDICT_INSUFFICIENT = "INSUFFICIENT_DATA"

# A (state, action) cell needs at least this many visits before it is trusted.
DEFAULT_MIN_VISITS = 5

CALL, PUT, SKIP = "CALL", "PUT", "SKIP"


# ---------------------------------------------------------------------------
# Source acquisition (both optional + injectable; fail-open)
# ---------------------------------------------------------------------------
def _min_visits_env(override: Optional[int]) -> int:
    if override is not None:
        try:
            return max(1, int(override))
        except (TypeError, ValueError):
            return DEFAULT_MIN_VISITS
    try:
        return max(1, int(float(os.environ.get("RL_VETO_MIN_VISITS",
                                               DEFAULT_MIN_VISITS))))
    except (TypeError, ValueError):
        return DEFAULT_MIN_VISITS


def _load_policy(agent, qtable_file: Optional[str]) -> dict:
    """policy_report()['policy'] from an injected agent or the persisted table."""
    a = agent
    if a is None:
        try:
            from rl_agent import QLearningAgent
            a = QLearningAgent(qtable_file=qtable_file or "rl_qtable.json")
        except Exception:
            return {}
    try:
        return a.policy_report().get("policy", {}) or {}
    except Exception:
        return {}


def _episode_index(store, db_path: Optional[str]) -> Dict[str, dict]:
    """state_key -> {rule_actions: {action: n}, pnls: [net_pnl_pct], n}."""
    obj = store
    created = False
    if obj is None:
        try:
            from episode_store import EpisodeStore
            obj = EpisodeStore(db_path or "episodes.db")
            created = True
        except Exception:
            return {}
    idx: Dict[str, dict] = {}
    try:
        for row in obj.completed():
            sk = row.get("state_key")
            if not sk:
                continue
            d = idx.setdefault(sk, {"rule_actions": {}, "pnls": [], "n": 0})
            ra = str(row.get("rule_action") or "").upper()
            if ra:
                d["rule_actions"][ra] = d["rule_actions"].get(ra, 0) + 1
            pnl = row.get("net_pnl_pct")
            if pnl is not None:
                try:
                    d["pnls"].append(float(pnl))
                except (TypeError, ValueError):
                    pass
            d["n"] += 1
    except Exception:
        pass
    finally:
        if created:
            try:
                obj.close()
            except Exception:
                pass
    return idx


def _dominant_rule_action(entry: Optional[dict]) -> Optional[str]:
    if not entry or not entry.get("rule_actions"):
        return None
    return max(entry["rule_actions"], key=entry["rule_actions"].get)


def _avg(values) -> Optional[float]:
    return round(sum(values) / len(values), 4) if values else None


# ---------------------------------------------------------------------------
# Computation
# ---------------------------------------------------------------------------
def compute_rl_veto_report(agent=None, store=None, *,
                           qtable_file: Optional[str] = None,
                           db_path: Optional[str] = None,
                           min_visits: Optional[int] = None) -> dict:
    """RL/rule disagreement + veto-candidate analysis. Never raises."""
    mv = _min_visits_env(min_visits)
    policy = _load_policy(agent, qtable_file)
    episodes = _episode_index(store, db_path)

    call_negative: List[dict] = []
    put_positive: List[dict] = []
    disagreements: List[dict] = []

    for sk, info in policy.items():
        qv = info.get("q_values", {}) or {}
        vis = info.get("visits", {}) or {}
        best = info.get("best_action")
        q_call, q_put = qv.get(CALL), qv.get(PUT)
        v_call, v_put = vis.get(CALL, 0), vis.get(PUT, 0)

        if q_call is not None and q_call < 0 and v_call >= mv:
            call_negative.append({
                "state_key": sk, "q_call": round(float(q_call), 4),
                "visits": int(v_call)})
        if q_put is not None and q_put > 0 and v_put >= mv:
            put_positive.append({
                "state_key": sk, "q_put": round(float(q_put), 4),
                "visits": int(v_put)})

        rule_action = _dominant_rule_action(episodes.get(sk))
        total_visits = sum(int(v) for v in vis.values())
        if (rule_action and best and best != rule_action
                and total_visits >= mv):
            rule_pnl = _avg(episodes.get(sk, {}).get("pnls", []))
            disagreements.append({
                "state_key": sk,
                "rule_action": rule_action,
                "rl_best_action": best,
                "q_values": {k: round(float(v), 4) for k, v in qv.items()},
                "visits": total_visits,
                "rule_realized_avg_pnl_pct": rule_pnl})

    call_negative.sort(key=lambda d: (d["q_call"], -d["visits"]))
    put_positive.sort(key=lambda d: (-d["q_put"], -d["visits"]))
    disagreements.sort(key=lambda d: -d["visits"])

    num_states = len(policy)
    if num_states == 0:
        verdict = VERDICT_INSUFFICIENT
    elif disagreements:
        verdict = VERDICT_DISAGREES
    else:
        verdict = VERDICT_ALIGNED

    suggested = {
        "min_visits": mv,
        "veto_call_when_q_below": 0.0,
        "favor_put_when_q_above": 0.0,
        "call_veto_candidates": len(call_negative),
        "put_favor_candidates": len(put_positive),
        # The most extreme observed Q values are evidence for where a future,
        # human-approved threshold could sit. Advisory only.
        "strongest_negative_call_q": call_negative[0]["q_call"] if call_negative else None,
        "strongest_positive_put_q": put_positive[0]["q_put"] if put_positive else None,
        "note": RL_VETO_FOOTER,
    }

    return {
        "num_states": num_states,
        "min_visits": mv,
        "call_negative_states": call_negative,
        "put_positive_states": put_positive,
        "disagreements": disagreements,
        "suggested_thresholds": suggested,
        "verdict": verdict,
    }


# ---------------------------------------------------------------------------
# Telegram formatting (pure)
# ---------------------------------------------------------------------------
def _q(value) -> str:
    return f"{value:+.4f}" if isinstance(value, (int, float)) else "n/a"


def _pct(value) -> str:
    return f"{value:+.2f}%" if isinstance(value, (int, float)) else "n/a"


def format_rl_veto_report(report: dict, limit: int = 8) -> str:
    """Telegram-ready RL_VETO_REPORT. Pure formatting."""
    header = "🛑 *RL Veto Report* _(analytics)_"
    footer = f"_{RL_VETO_FOOTER}_"
    if report.get("num_states", 0) == 0:
        return "\n".join([
            header, "",
            "No learned RL states yet (empty Q-table).",
            f"*Verdict:* `{VERDICT_INSUFFICIENT}`",
            "", footer,
        ])

    s = report["suggested_thresholds"]
    lines = [
        header, "",
        f"States: `{report['num_states']}` · "
        f"min visits: `{report['min_visits']}`",
        "",
        "*CALL states with negative Q (RL: calls lose here):*",
    ]
    cn = report["call_negative_states"]
    if not cn:
        lines.append("_none_")
    for d in cn[:limit]:
        lines.append(f"`{d['state_key']}` — Q(CALL) `{_q(d['q_call'])}` "
                     f"({d['visits']} visits)")

    lines += ["", "*PUT states with positive Q (RL: puts win here):*"]
    pp = report["put_positive_states"]
    if not pp:
        lines.append("_none_")
    for d in pp[:limit]:
        lines.append(f"`{d['state_key']}` — Q(PUT) `{_q(d['q_put'])}` "
                     f"({d['visits']} visits)")

    lines += ["", "*Rule / RL disagreements (high visits):*"]
    dis = report["disagreements"]
    if not dis:
        lines.append("_none_")
    for d in dis[:limit]:
        lines.append(
            f"`{d['state_key']}` — rule `{d['rule_action']}` vs RL "
            f"`{d['rl_best_action']}` ({d['visits']} visits, rule realized "
            f"`{_pct(d['rule_realized_avg_pnl_pct'])}`)")

    lines += [
        "",
        "*Suggested veto thresholds (advisory):*",
        f"Veto CALL when Q(CALL) < `{s['veto_call_when_q_below']:+.2f}` "
        f"(`{s['call_veto_candidates']}` states qualify; strongest "
        f"`{_q(s['strongest_negative_call_q'])}`)",
        f"Favor PUT when Q(PUT) > `{s['favor_put_when_q_above']:+.2f}` "
        f"(`{s['put_favor_candidates']}` states qualify; strongest "
        f"`{_q(s['strongest_positive_put_q'])}`)",
        "",
        f"*Verdict:* `{report['verdict']}`",
        "", footer,
    ]
    return "\n".join(lines)


def generate_rl_veto_report_text(qtable_file: Optional[str] = None,
                                 db_path: Optional[str] = None) -> str:
    """Top-level entry for the RL_VETO_REPORT Telegram command."""
    return format_rl_veto_report(
        compute_rl_veto_report(qtable_file=qtable_file, db_path=db_path))


# ---------------------------------------------------------------------------
# Self-test (no creds, no network)
# ---------------------------------------------------------------------------
class _FakeAgent:
    """Minimal stand-in exposing policy_report() like QLearningAgent."""

    def __init__(self, policy):
        self._policy = policy

    def policy_report(self):
        return {"num_states": len(self._policy), "policy": self._policy}


class _FakeStore:
    def __init__(self, rows):
        self._rows = rows

    def completed(self):
        return self._rows


def _self_test() -> int:
    ok = True
    policy = {
        # bullish state: RL likes CALL — aligned with a CALL rule.
        "strat=t|change=up_strong": {
            "best_action": CALL,
            "q_values": {CALL: 0.30, PUT: -0.20, SKIP: 0.0},
            "visits": {CALL: 20, PUT: 8, SKIP: 5}},
        # RL learned CALL LOSES here but the rule still fired CALL -> disagreement
        # (RL best is SKIP) AND a negative-Q call candidate.
        "strat=t|change=dn_strong": {
            "best_action": SKIP,
            "q_values": {CALL: -0.25, PUT: 0.18, SKIP: 0.0},
            "visits": {CALL: 15, PUT: 12, SKIP: 6}},
    }
    store = _FakeStore([
        {"state_key": "strat=t|change=up_strong", "rule_action": "CALL",
         "net_pnl_pct": 12.0},
        {"state_key": "strat=t|change=dn_strong", "rule_action": "CALL",
         "net_pnl_pct": -18.0},
        {"state_key": "strat=t|change=dn_strong", "rule_action": "CALL",
         "net_pnl_pct": -9.0},
    ])
    rep = compute_rl_veto_report(agent=_FakeAgent(policy), store=store,
                                 min_visits=5)

    # Negative-Q CALL state detected.
    if not any(d["state_key"] == "strat=t|change=dn_strong"
               for d in rep["call_negative_states"]):
        print("FAIL: negative CALL Q not flagged"); ok = False
    # Positive-Q PUT state detected.
    if not any(d["state_key"] == "strat=t|change=dn_strong"
               for d in rep["put_positive_states"]):
        print("FAIL: positive PUT Q not flagged"); ok = False
    # Disagreement detected (rule CALL vs RL SKIP) with realized loss attached.
    dis = [d for d in rep["disagreements"]
           if d["state_key"] == "strat=t|change=dn_strong"]
    if not dis or dis[0]["rule_action"] != "CALL" or dis[0]["rl_best_action"] != SKIP:
        print("FAIL: disagreement not flagged", rep["disagreements"]); ok = False
    if dis and dis[0]["rule_realized_avg_pnl_pct"] is None:
        print("FAIL: rule realized pnl not attached"); ok = False
    if rep["verdict"] != VERDICT_DISAGREES:
        print("FAIL: verdict should be DISAGREES", rep["verdict"]); ok = False

    # Visit floor: raise it above all visits -> nothing qualifies.
    high = compute_rl_veto_report(agent=_FakeAgent(policy), store=store,
                                  min_visits=999)
    if high["call_negative_states"] or high["disagreements"]:
        print("FAIL: high visit floor should suppress flags"); ok = False

    # Empty Q-table -> clean insufficient report + footer.
    empty = format_rl_veto_report(compute_rl_veto_report(agent=_FakeAgent({})))
    if RL_VETO_FOOTER not in empty or VERDICT_INSUFFICIENT not in empty:
        print("FAIL: empty report malformed"); ok = False

    _ = format_rl_veto_report(rep)  # never raises

    print("rl_veto_report self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
