"""
RLAdvisor - shadow/advisory reinforcement-learning wrapper.

This module is the integration point between the existing rule-based strategies
and the Q-learning agent. In the default SHADOW mode it:

  1. Observes the strategy's `analysis` output and recommends an action.
  2. Logs a "pending" experience keyed to the trade (order id / symbol).
  3. When the trade closes, matches the outcome to its pending experience,
     computes a reward, and updates the agent.

It NEVER changes what the strategy actually trades in shadow mode. Hooks that
call into this wrapper are wrapped in try/except by the strategies so a failure
here can never block a live trade.

Modes (env RL_MODE):
  shadow  - recommend + learn only (default, implemented)
  gate    - reserved: allow agent to veto a trade (NOT wired yet)
  control - reserved: agent decides direction (NOT wired yet)
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Optional

from rl_env import (
    extract_features,
    state_key,
    valid_actions,
    compute_reward,
    SKIP,
)
from rl_agent import QLearningAgent


def _env(name: str, default: str = "") -> str:
    val = os.getenv(name)
    if val is not None:
        return val
    # Fall back to a manual .env scan (strategies may not have loaded dotenv).
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        if k.strip() == name:
                            return v.strip()
        except OSError:
            pass
    return default


def rl_enabled() -> bool:
    return _env("RL_ENABLED", "true").lower() in ("1", "true", "yes", "on")


def rl_mode() -> str:
    return _env("RL_MODE", "shadow").lower()


def _gate_config() -> Dict:
    """Conservative veto-only gate thresholds (env-overridable)."""

    def _f(name, default):
        try:
            return float(_env(name, str(default)))
        except (TypeError, ValueError):
            return default

    return {
        "min_visits": int(_f("RL_GATE_MIN_VISITS", 8)),
        "max_q": _f("RL_GATE_MAX_Q", -0.10),
        "min_confidence": _f("RL_GATE_MIN_CONFIDENCE", 0.75),
    }


class RLAdvisor:
    def __init__(
        self,
        strat_name: str = "generic",
        experience_file: str = "rl_experience.json",
        qtable_file: str = "rl_qtable.json",
    ):
        self.strat_name = strat_name
        self.experience_file = experience_file
        self.agent = QLearningAgent(qtable_file=qtable_file)
        self.mode = rl_mode()

    # --------------------------------------------------------------- advising
    def advise(
        self,
        analysis: Dict,
        pdt_remaining: Optional[int] = None,
        day_of_week: Optional[int] = None,
    ) -> Dict:
        """
        Return a recommendation for the current context.

        In shadow mode this is purely informational - callers should NOT use it
        to change the trade. The rule action is whatever the strategy decided
        (direction if it intends to trade, else SKIP).
        """
        features = extract_features(
            analysis, pdt_remaining, day_of_week, self.strat_name
        )
        skey = state_key(features)
        valid = valid_actions(analysis)

        rule_action = (
            (analysis.get("direction") or SKIP).upper()
            if analysis.get("should_trade", True)
            else SKIP
        )

        recommended, rec_q = self.agent.best_action(skey, valid)
        qvals = self.agent.q_values(skey, valid)

        return {
            "state_key": skey,
            "features": features,
            "valid_actions": valid,
            "rule_action": rule_action,
            "recommended_action": recommended,
            "recommended_q": rec_q,
            "q_values": qvals,
            "agreement": recommended == rule_action,
            "mode": self.mode,
        }

    # ----------------------------------------------------------------- gating
    def gate_decision(
        self,
        analysis: Dict,
        pdt_remaining: Optional[int] = None,
        day_of_week: Optional[int] = None,
        overrides: Optional[Dict] = None,
    ) -> Dict:
        """
        Conservative, veto-only gate. Pure / read-only / fail-open.

        Returns a dict describing whether the agent would VETO the rule's
        directional trade (turn CALL/PUT into SKIP). It NEVER flips direction,
        never chooses CALL vs PUT, and never mutates the Q-table.

        A veto fires only when ALL hold:
          1. RL_MODE == 'gate'
          2. the rule wants to trade a direction (CALL/PUT, not SKIP)
          3. visits(state, rule_action) >= min_visits
          4. Q(state, rule_action) <= max_q  (materially negative)
          5. confidence (= min(1, visits/min_visits)) >= min_confidence

        Any exception results in veto=False so a live trade is never blocked.
        """
        result = {
            "veto": False,
            "rule_action": SKIP,
            "q": 0.0,
            "visits": 0,
            "confidence": 0.0,
            "reason": "",
            "state_key": "",
        }
        try:
            cfg = dict(_gate_config())
            if overrides:
                cfg.update(overrides)
            min_visits = max(1, int(cfg["min_visits"]))
            max_q = float(cfg["max_q"])
            min_confidence = float(cfg["min_confidence"])

            features = extract_features(
                analysis, pdt_remaining, day_of_week, self.strat_name
            )
            skey = state_key(features)
            result["state_key"] = skey

            rule_action = (
                (analysis.get("direction") or SKIP).upper()
                if analysis.get("should_trade", True)
                else SKIP
            )
            result["rule_action"] = rule_action

            q = self.agent.get_q(skey, rule_action)
            visits = self.agent.visits(skey, rule_action)
            confidence = min(1.0, visits / float(min_visits))
            result["q"] = q
            result["visits"] = visits
            result["confidence"] = confidence

            if rl_mode() != "gate":
                result["reason"] = "mode!=gate"
                return result
            if rule_action not in ("CALL", "PUT"):
                result["reason"] = "no directional trade to veto"
                return result
            if visits < min_visits:
                result["reason"] = f"insufficient visits ({visits}<{min_visits})"
                return result
            if q > max_q:
                result["reason"] = f"q {q:.4f} not <= {max_q:.4f}"
                return result
            if confidence < min_confidence:
                result["reason"] = (
                    f"confidence {confidence:.2f}<{min_confidence:.2f}"
                )
                return result

            result["veto"] = True
            result["reason"] = (
                f"VETO {rule_action}: q={q:.4f} visits={visits} "
                f"conf={confidence:.2f}"
            )
            return result
        except Exception as exc:  # fail-open: never block a trade
            result["veto"] = False
            result["reason"] = f"gate error (fail-open): {exc}"
            return result

    # ----------------------------------------------------------- experiences
    def _load_experiences(self) -> List[Dict]:
        if os.path.exists(self.experience_file):
            try:
                with open(self.experience_file, "r") as f:
                    return json.load(f)
            except (OSError, json.JSONDecodeError):
                return []
        return []

    def _save_experiences(self, experiences: List[Dict]) -> None:
        with open(self.experience_file, "w") as f:
            json.dump(experiences, f, indent=2, default=str)

    def log_pending(
        self,
        key_id: str,
        state_key_str: str,
        action: str,
        context: Optional[Dict] = None,
    ) -> None:
        """Record a decision whose outcome is not yet known."""
        experiences = self._load_experiences()
        experiences.append(
            {
                "key_id": str(key_id),
                "strat": self.strat_name,
                "state_key": state_key_str,
                "action": action,
                "context": context or {},
                "status": "PENDING",
                "logged_at": datetime.now().isoformat(),
            }
        )
        self._save_experiences(experiences)

    def record_outcome(
        self,
        key_id: str,
        pnl_pct: Optional[float],
        took_day_trade: bool = False,
        pdt_remaining_before: Optional[int] = None,
        next_state_key: Optional[str] = None,
        done: bool = True,
    ) -> Optional[float]:
        """
        Match a closed trade to its pending experience, compute the reward, and
        update the agent. Returns the new Q-value (or None if no match).
        """
        experiences = self._load_experiences()
        target = None
        for exp in experiences:
            if exp.get("key_id") == str(key_id) and exp.get("status") == "PENDING":
                target = exp
                break

        if target is None:
            return None

        action = target["action"]
        reward = compute_reward(
            pnl_pct, action, pdt_remaining_before, took_day_trade
        )
        new_q = self.agent.update(
            target["state_key"],
            action,
            reward,
            next_state_key=next_state_key,
            done=done,
        )
        self.agent.save()

        target["status"] = "COMPLETED"
        target["pnl_pct"] = pnl_pct
        target["reward"] = reward
        target["new_q"] = new_q
        target["completed_at"] = datetime.now().isoformat()
        self._save_experiences(experiences)

        return new_q

    # ----------------------------------------------------- one-shot convenience
    def observe_and_log(
        self,
        analysis: Dict,
        key_id: str,
        action: str,
        pdt_remaining: Optional[int] = None,
        day_of_week: Optional[int] = None,
        context: Optional[Dict] = None,
    ) -> Dict:
        """
        Convenience used by strategy hooks: compute the advice and immediately
        log a pending experience for the action the strategy is taking.
        """
        advice = self.advise(analysis, pdt_remaining, day_of_week)
        self.log_pending(key_id, advice["state_key"], action, context)
        return advice


def _demo() -> int:
    """Round-trip demo with no creds: advise -> log_pending -> record_outcome."""
    advisor = RLAdvisor(
        strat_name="demo",
        experience_file="rl_experience_demo.json",
        qtable_file="rl_qtable_demo.json",
    )
    advisor.agent.reset()

    analysis = {
        "direction": "CALL",
        "confidence": 80.0,
        "spy_change": 0.45,
        "vix_level": 14.0,
        "vix_change": -6.0,
        "gap": 0.4,
        "intraday_position": 0.8,
        "should_trade": True,
    }

    advice = advisor.advise(analysis, pdt_remaining=2, day_of_week=0)
    print("=" * 50)
    print("RL ADVISOR DEMO")
    print("=" * 50)
    print("State:", advice["state_key"])
    print("Rule action:", advice["rule_action"])
    print("Recommended:", advice["recommended_action"], advice["q_values"])

    advisor.log_pending("ORDER123", advice["state_key"], "CALL")
    new_q = advisor.record_outcome("ORDER123", pnl_pct=22.0, took_day_trade=True,
                                   pdt_remaining_before=2)
    print(f"Recorded +22% outcome -> new Q(CALL) = {new_q:.4f}")

    advice2 = advisor.advise(analysis, pdt_remaining=2, day_of_week=0)
    print("Recommended after learning:", advice2["recommended_action"],
          advice2["q_values"])

    ok = new_q is not None and new_q > 0

    for fn in ("rl_qtable_demo.json", "rl_experience_demo.json"):
        try:
            os.remove(fn)
        except OSError:
            pass

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_demo())
