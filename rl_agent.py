"""
Tabular Q-learning agent for the options-trading RL layer.

Pure-Python, dependency-free. The Q-table is a flat dict keyed by
"<state_key>|<action>" and persisted as JSON so it is human-inspectable.

Hyperparameters are read from .env (with sane defaults) so the agent can be
tuned without code changes:

    RL_ALPHA          learning rate            (default 0.10)
    RL_GAMMA          discount factor          (default 0.50)
    RL_EPSILON        exploration rate         (default 0.20)
    RL_EPSILON_DECAY  per-update decay         (default 0.995)
    RL_EPSILON_MIN    floor for epsilon        (default 0.02)
"""

import os
import json
import random
from typing import Dict, List, Optional, Tuple

from rl_env import ACTIONS


def _load_env() -> Dict[str, str]:
    """Lightweight .env reader (avoids importing dotenv just for this)."""
    env = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
        except OSError:
            pass
    return env


class QLearningAgent:
    def __init__(
        self,
        actions: Optional[List[str]] = None,
        qtable_file: str = "rl_qtable.json",
        alpha: Optional[float] = None,
        gamma: Optional[float] = None,
        epsilon: Optional[float] = None,
        epsilon_decay: Optional[float] = None,
        epsilon_min: Optional[float] = None,
        seed: Optional[int] = None,
    ):
        env = _load_env()

        def _f(name, override, default):
            if override is not None:
                return override
            try:
                return float(env.get(name, default))
            except (TypeError, ValueError):
                return default

        self.actions = list(actions) if actions else list(ACTIONS)
        self.qtable_file = qtable_file
        self.alpha = _f("RL_ALPHA", alpha, 0.10)
        self.gamma = _f("RL_GAMMA", gamma, 0.50)
        self.epsilon = _f("RL_EPSILON", epsilon, 0.20)
        self.epsilon_decay = _f("RL_EPSILON_DECAY", epsilon_decay, 0.995)
        self.epsilon_min = _f("RL_EPSILON_MIN", epsilon_min, 0.02)

        if seed is not None:
            random.seed(seed)

        self.q: Dict[str, float] = {}
        self.counts: Dict[str, int] = {}
        self.updates = 0
        self.load()

    # ------------------------------------------------------------------ keys
    @staticmethod
    def _qk(state_key: str, action: str) -> str:
        return f"{state_key}|{action}"

    # --------------------------------------------------------------- queries
    def get_q(self, state_key: str, action: str) -> float:
        return self.q.get(self._qk(state_key, action), 0.0)

    def best_action(
        self, state_key: str, valid: Optional[List[str]] = None
    ) -> Tuple[str, float]:
        """Greedy action (ties broken randomly) and its Q-value."""
        candidates = valid if valid else self.actions
        best_val = None
        best: List[str] = []
        for a in candidates:
            v = self.get_q(state_key, a)
            if best_val is None or v > best_val:
                best_val, best = v, [a]
            elif v == best_val:
                best.append(a)
        if not best:
            return (candidates[0] if candidates else self.actions[0], 0.0)
        return random.choice(best), best_val

    def select_action(
        self,
        state_key: str,
        valid: Optional[List[str]] = None,
        explore: bool = True,
    ) -> str:
        """Epsilon-greedy action selection."""
        candidates = valid if valid else self.actions
        if explore and random.random() < self.epsilon:
            return random.choice(candidates)
        return self.best_action(state_key, candidates)[0]

    def q_values(
        self, state_key: str, valid: Optional[List[str]] = None
    ) -> Dict[str, float]:
        candidates = valid if valid else self.actions
        return {a: self.get_q(state_key, a) for a in candidates}

    # ---------------------------------------------------------------- update
    def update(
        self,
        state_key: str,
        action: str,
        reward: float,
        next_state_key: Optional[str] = None,
        done: bool = True,
        next_valid: Optional[List[str]] = None,
    ) -> float:
        """
        Q-learning TD update:
            Q(s,a) <- Q(s,a) + alpha * (r + gamma * max_a' Q(s',a') - Q(s,a))

        Returns the new Q-value. For a terminal step (done or no next state)
        the bootstrap term is zero, reducing the update to a bandit-style step.
        """
        qk = self._qk(state_key, action)
        current = self.q.get(qk, 0.0)

        future = 0.0
        if not done and next_state_key:
            future = self.best_action(next_state_key, next_valid)[1]

        target = reward + self.gamma * future
        new_val = current + self.alpha * (target - current)

        self.q[qk] = new_val
        self.counts[qk] = self.counts.get(qk, 0) + 1
        self.updates += 1

        # Decay exploration toward the floor.
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return new_val

    # ------------------------------------------------------------ persistence
    def save(self) -> None:
        payload = {
            "q": self.q,
            "counts": self.counts,
            "meta": {
                "updates": self.updates,
                "epsilon": self.epsilon,
                "alpha": self.alpha,
                "gamma": self.gamma,
                "actions": self.actions,
            },
        }
        with open(self.qtable_file, "w") as f:
            json.dump(payload, f, indent=2)

    def load(self) -> None:
        if not os.path.exists(self.qtable_file):
            return
        try:
            with open(self.qtable_file, "r") as f:
                data = json.load(f)
            self.q = data.get("q", {})
            self.counts = data.get("counts", {})
            meta = data.get("meta", {})
            self.updates = meta.get("updates", 0)
            # Restore decayed epsilon so training resumes where it left off.
            if "epsilon" in meta:
                self.epsilon = meta["epsilon"]
        except (OSError, json.JSONDecodeError):
            # Corrupt/empty file: start fresh rather than crash a trading run.
            self.q, self.counts, self.updates = {}, {}, 0

    def reset(self) -> None:
        self.q, self.counts, self.updates = {}, {}, 0
        self.save()

    # --------------------------------------------------------------- reporting
    def policy_report(self) -> Dict:
        """Summarize the learned greedy policy per state."""
        states: Dict[str, Dict[str, float]] = {}
        for qk, val in self.q.items():
            state_key, action = qk.rsplit("|", 1)
            states.setdefault(state_key, {})[action] = val

        policy = {}
        for state_key, avals in states.items():
            best_action = max(avals, key=avals.get)
            policy[state_key] = {
                "best_action": best_action,
                "q_values": avals,
                "visits": {
                    a: self.counts.get(self._qk(state_key, a), 0) for a in avals
                },
            }
        return {
            "num_states": len(states),
            "num_updates": self.updates,
            "epsilon": round(self.epsilon, 4),
            "policy": policy,
        }


def _self_test() -> int:
    """
    Synthetic convergence check (no credentials, no network).

    Two states with a clearly correct action. After training the greedy policy
    should pick the rewarding action in each.
    """
    agent = QLearningAgent(
        actions=["SKIP", "CALL", "PUT"],
        qtable_file="rl_qtable_selftest.json",
        epsilon=0.3,
        epsilon_decay=0.99,
        seed=42,
    )
    agent.reset()

    bull = "strat=test|change=up_strong"
    bear = "strat=test|change=dn_strong"

    for _ in range(500):
        # In a bullish state CALL pays, PUT/SKIP do not.
        a = agent.select_action(bull, valid=["SKIP", "CALL", "PUT"])
        r = 0.25 if a == "CALL" else (-0.20 if a == "PUT" else 0.0)
        agent.update(bull, a, r, done=True)

        # In a bearish state PUT pays.
        a = agent.select_action(bear, valid=["SKIP", "CALL", "PUT"])
        r = 0.25 if a == "PUT" else (-0.20 if a == "CALL" else 0.0)
        agent.update(bear, a, r, done=True)

    bull_best = agent.best_action(bull)[0]
    bear_best = agent.best_action(bear)[0]

    print("=" * 50)
    print("Q-LEARNING SELF-TEST")
    print("=" * 50)
    print(f"Bullish state -> {bull_best} (expected CALL)")
    print(f"Bearish state -> {bear_best} (expected PUT)")
    print(f"Updates: {agent.updates} | epsilon: {agent.epsilon:.4f}")

    ok = bull_best == "CALL" and bear_best == "PUT"

    # Clean up the self-test artifact.
    try:
        os.remove("rl_qtable_selftest.json")
    except OSError:
        pass

    print("RESULT:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
