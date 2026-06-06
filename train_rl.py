"""
Offline trainer / reporter for the RL trading layer.

Replays historical trade logs through the Q-learning agent to bootstrap the
Q-table, and prints the learned policy. This lets the agent start with some
knowledge instead of a blank table when shadow mode begins.

Usage:
    python train_rl.py --replay              # train from logs
    python train_rl.py --replay --epochs 50  # multiple passes
    python train_rl.py --report              # print learned policy
    python train_rl.py --reset               # wipe the Q-table
    python train_rl.py --replay --report     # train then report

Only records that carry both an `analysis` block and a realized P/L
(`profit_pct` / `pnl_percent`) are usable. Others are counted and skipped.
"""

import os
import json
import argparse
from datetime import datetime
from typing import Dict, List, Optional

from rl_env import extract_features, state_key, valid_actions, compute_reward, SKIP
from rl_agent import QLearningAgent


# Trade-log file -> strategy name used for the state key.
LOG_FILES = {
    "spy_1dte_trades.json": "spy_1dte",
    "spy_hybrid_trades.json": "spy_hybrid",
    "spy_qqq_hybrid_trades.json": "spy_qqq_hybrid",
    "schwab_trades.json": "schwab",
}


def _read_json_list(path: str) -> List[Dict]:
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return data if isinstance(data, list) else []
    except (OSError, json.JSONDecodeError):
        return []


def _get_pnl(record: Dict) -> Optional[float]:
    """Realized P/L percent from a trade record, if present."""
    for key in ("profit_pct", "pnl_percent", "pnl_pct"):
        if key in record and record[key] is not None:
            try:
                return float(record[key])
            except (TypeError, ValueError):
                continue
    return None


def _record_timestamp(record: Dict) -> str:
    return (
        record.get("exit_time")
        or record.get("timestamp")
        or record.get("entry_time")
        or ""
    )


def build_experiences() -> Dict:
    """
    Convert raw trade logs into RL experiences.

    Returns dict with `experiences` (usable) and counters for reporting.
    """
    usable: List[Dict] = []
    total = 0
    skipped_no_outcome = 0
    skipped_no_analysis = 0

    for fname, strat in LOG_FILES.items():
        for record in _read_json_list(fname):
            total += 1

            analysis = record.get("analysis")
            if not isinstance(analysis, dict):
                # smart_trader / schwab records may store market context inline
                analysis = {
                    "direction": record.get("type"),
                    "confidence": record.get("confidence", 0.0),
                }
                if "type" not in record:
                    skipped_no_analysis += 1
                    continue

            pnl = _get_pnl(record)
            if pnl is None:
                skipped_no_outcome += 1
                continue

            pdt = None
            if isinstance(record.get("pdt_status"), dict):
                pdt = record["pdt_status"].get("remaining")

            direction = (record.get("type") or analysis.get("direction") or "").upper()
            action = direction if direction in ("CALL", "PUT") else SKIP

            features = extract_features(analysis, pdt, None, strat)
            took_day_trade = record.get("mode") == "1DTE"

            usable.append(
                {
                    "strat": strat,
                    "state_key": state_key(features),
                    "action": action,
                    "pnl_pct": pnl,
                    "pdt_remaining": pdt,
                    "took_day_trade": took_day_trade,
                    "valid": valid_actions(
                        {"direction": direction, "should_trade": True}
                    ),
                    "ts": _record_timestamp(record),
                }
            )

    usable.sort(key=lambda e: e["ts"])

    return {
        "experiences": usable,
        "total": total,
        "skipped_no_outcome": skipped_no_outcome,
        "skipped_no_analysis": skipped_no_analysis,
    }


def replay_experiences(agent: QLearningAgent, experiences: List[Dict], epochs: int = 1) -> int:
    """Train the agent over a list of experiences. Returns total updates applied."""
    for _ in range(max(1, epochs)):
        for i, exp in enumerate(experiences):
            # Bootstrap from the next experience of the same strategy (weak
            # cross-day coupling); terminal otherwise.
            next_state = None
            done = True
            for j in range(i + 1, len(experiences)):
                if experiences[j]["strat"] == exp["strat"]:
                    next_state = experiences[j]["state_key"]
                    done = False
                    break

            reward = compute_reward(
                exp["pnl_pct"],
                exp["action"],
                exp["pdt_remaining"],
                exp["took_day_trade"],
            )
            agent.update(
                exp["state_key"],
                exp["action"],
                reward,
                next_state_key=next_state,
                done=done,
            )

    agent.save()
    return len(experiences) * max(1, epochs)


def build_episode_experiences(db_path: str = "episodes.db") -> Dict:
    """Load experiences from the SQLite episode store (net-of-cost P/L)."""
    from episode_store import EpisodeStore

    store = EpisodeStore(db_path)
    try:
        experiences = store.to_rl_experiences()
    finally:
        store.close()
    return {
        "experiences": experiences,
        "total": len(experiences),
        "skipped_no_outcome": 0,
        "skipped_no_analysis": 0,
    }


def replay(agent: QLearningAgent, epochs: int = 1, source: str = "logs",
           db_path: str = "episodes.db") -> Dict:
    data = build_episode_experiences(db_path) if source == "episodes" else build_experiences()
    data["trained"] = replay_experiences(agent, data["experiences"], epochs)
    return data


def print_report(agent: QLearningAgent) -> None:
    report = agent.policy_report()
    print("=" * 70)
    print("LEARNED POLICY REPORT")
    print("=" * 70)
    print(f"States learned : {report['num_states']}")
    print(f"Total updates  : {report['num_updates']}")
    print(f"Epsilon (now)  : {report['epsilon']}")
    print("-" * 70)

    if not report["policy"]:
        print("(empty - run with --replay first, or no usable trade data yet)")
        return

    for skey, info in sorted(report["policy"].items()):
        qstr = ", ".join(
            f"{a}={v:+.3f}({info['visits'].get(a, 0)})"
            for a, v in info["q_values"].items()
        )
        print(f"-> best={info['best_action']:4s} | {qstr}")
        print(f"   state: {skey}")


def main() -> int:
    parser = argparse.ArgumentParser(description="RL offline trainer / reporter")
    parser.add_argument("--replay", action="store_true", help="train from trade logs")
    parser.add_argument("--epochs", type=int, default=1, help="passes over the data")
    parser.add_argument("--report", action="store_true", help="print learned policy")
    parser.add_argument("--reset", action="store_true", help="wipe the Q-table first")
    parser.add_argument(
        "--qtable", default="rl_qtable.json", help="Q-table file path"
    )
    parser.add_argument(
        "--source", choices=["logs", "episodes"], default="logs",
        help="training source: JSON trade logs (default) or the SQLite episode store",
    )
    parser.add_argument(
        "--db", default="episodes.db", help="episode store path (when --source episodes)"
    )
    args = parser.parse_args()

    if not (args.replay or args.report or args.reset):
        parser.print_help()
        return 0

    agent = QLearningAgent(qtable_file=args.qtable)

    if args.reset:
        agent.reset()
        print(f"[RESET] Q-table cleared: {args.qtable}")

    if args.replay:
        print(f"[REPLAY] Training (epochs={args.epochs}, source={args.source})...")
        data = replay(agent, epochs=args.epochs, source=args.source, db_path=args.db)
        print(f"[REPLAY] Trade records scanned : {data['total']}")
        print(f"[REPLAY] Usable experiences    : {len(data['experiences'])}")
        print(f"[REPLAY] Skipped (no outcome)  : {data['skipped_no_outcome']}")
        print(f"[REPLAY] Skipped (no analysis) : {data['skipped_no_analysis']}")
        print(f"[REPLAY] Updates applied       : {data['trained']}")
        if not data["experiences"]:
            print(
                "[REPLAY] NOTE: No closed trades with realized P/L yet. The agent "
                "will learn online as trades close in shadow mode."
            )
        print(f"[REPLAY] Saved -> {args.qtable}")

    if args.report:
        print()
        print_report(agent)

    return 0


if __name__ == "__main__":
    import sys

    sys.exit(main())
