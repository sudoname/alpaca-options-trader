"""
Oracle 3.0 — Adaptive Feature Weighting (SHADOW credit-assignment over agents).

Learns a per-agent *voting weight* from stored trade outcomes. This is credit
assignment over the evidence agents — NOT a buy/sell classifier, NOT a price
predictor, NOT a direction model. The only thing it produces is a bounded,
mean-normalized weight per agent that ``oracle_voting.tally_votes`` /
``bayesian_probability`` may consume. Every weight is clamped to
``[w_min, w_max]`` so, exactly as the plan requires, **no single agent can flip
a decision** no matter how it scores historically.

Design (mirrors the repo's analytics conventions):
  * PURE + fail-open: ``compute_weights`` never raises and never persists; on
    empty / malformed / insufficient data it returns *uniform* weights (1.0 each)
    with an ``INSUFFICIENT_DATA`` verdict. Until validated, weights stay uniform.
  * Credit signal = the existing RL reward (realized win / pnl) folded per agent
    via a hit-rate lift over the global base rate, shrunk toward neutral by
    sample size so a handful of trades can't swing a weight.
  * The live path is untouched: nothing here opens, sizes, prices, blocks, or
    alters a trade, and nothing imports this on the hot path. The voting layer
    only reads these weights when the Intelligence Layer is explicitly enabled.

API:
  ``compute_weights(records, config=None) -> {weights, per_agent, sample_size,
      base_win_rate, verdict}``     (pure)
  ``current_weights(config=None) -> {name: weight}``     (reads the weights file)
  ``weight_history(config=None) -> [ {weights, ...}, ... ]``     (reads history)
  ``save_weights(result, config=None) -> bool``     (persist + append history)
  ``update_weights(records, config=None, persist=True) -> dict``     (compute+save)
  ``weight_drift(history) -> float``     (sum |last - first| across agents)
"""

import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, List, Optional

import oracle_analytics as oa
from config_loader import ConfigLoader
from oracle_agents import AGENT_NAMES

# Tuning constants (intentionally gentle — this is shadow credit assignment).
WEIGHT_NEUTRAL = 1.0
DEFAULT_W_MIN = 0.25
DEFAULT_W_MAX = 3.0
MIN_SAMPLES = 10
# How hard a unit of (hit-rate lift) bends a weight away from neutral.
LIFT_GAIN = 4.0
# Shrinkage prior: an agent needs ~this many convicted trades for its lift to
# count at full strength (n / (n + K)).
SHRINK_K = 20.0
# Below this directional magnitude an agent is "neutral" on a trade (no credit).
CONVICTION_EPS = 1e-9


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class OracleWeightsConfig:
    weights_file: str = "oracle_agent_weights.json"
    w_min: float = DEFAULT_W_MIN
    w_max: float = DEFAULT_W_MAX
    min_samples: int = MIN_SAMPLES

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "OracleWeightsConfig":
        try:
            cfg = loader if loader is not None else ConfigLoader(path=path)
            return OracleWeightsConfig(
                weights_file=cfg.get_str("ORACLE_AGENT_WEIGHTS_FILE",
                                         "oracle_agent_weights.json"),
                w_min=cfg.get_float("ORACLE_WEIGHT_MIN", DEFAULT_W_MIN),
                w_max=cfg.get_float("ORACLE_WEIGHT_MAX", DEFAULT_W_MAX),
            )
        except Exception:  # pragma: no cover - fail-open
            return OracleWeightsConfig()


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_float(value, default: Optional[float] = None):
    try:
        f = float(value)
        if math.isnan(f) or math.isinf(f):
            return default
        return f
    except (TypeError, ValueError):
        return default


def _parse_votes(value):
    """Coerce a stored ``agent_votes`` field into a {name: vote-dict} mapping."""
    if value is None:
        return None
    if isinstance(value, str):
        try:
            value = json.loads(value)
        except Exception:
            return None
    return value if isinstance(value, dict) else None


def _clamp(x: float, lo: float, hi: float) -> float:
    return lo if x < lo else hi if x > hi else x


def uniform_weights(names=AGENT_NAMES) -> Dict[str, float]:
    """Every agent at the neutral weight (the validated-off default)."""
    return {n: WEIGHT_NEUTRAL for n in names}


# --------------------------------------------------------------------------- #
# Core: credit assignment -> bounded, mean-normalized weights (PURE)
# --------------------------------------------------------------------------- #
def compute_weights(records: Optional[List[dict]] = None,
                    config: Optional[OracleWeightsConfig] = None) -> dict:
    """Learn per-agent voting weights from closed-trade outcomes. Never raises.

    For each closed record that carries ``agent_votes``, an agent is "convicted"
    when ``|bullish - bearish| > CONVICTION_EPS``. Its weight grows with the win
    rate of the trades it was convicted on, relative to the global base win rate,
    shrunk by how many such trades exist. Weights are clamped to
    ``[w_min, w_max]`` and mean-normalized to ~1.0 so the calibrated voting scale
    is preserved and no agent can dominate. Insufficient data -> uniform.
    """
    config = config or OracleWeightsConfig()
    names = list(AGENT_NAMES)
    try:
        rows = [r for r in (records or []) if isinstance(r, dict)]
        scored = []
        base_wins = 0
        per_agent: Dict[str, Dict[str, float]] = {}
        for row in rows:
            votes = _parse_votes(row.get("agent_votes"))
            if not votes:
                continue
            win = 1 if oa._is_win(row) else 0
            scored.append(row)
            base_wins += win
            for name, v in votes.items():
                if not isinstance(v, dict):
                    continue
                bull = _to_float(v.get("bullish_score"), 0.0) or 0.0
                bear = _to_float(v.get("bearish_score"), 0.0) or 0.0
                if abs(bull - bear) <= CONVICTION_EPS:
                    continue                      # neutral -> no credit either way
                a = per_agent.setdefault(name, {"votes": 0, "wins": 0})
                a["votes"] += 1
                a["wins"] += win

        n = len(scored)
        if n < config.min_samples:
            return {"weights": uniform_weights(names), "per_agent": {},
                    "sample_size": n, "base_win_rate": None,
                    "verdict": "INSUFFICIENT_DATA"}

        base_wr = base_wins / n
        raw: Dict[str, float] = {}
        per_agent_out: Dict[str, dict] = {}
        for name in names:
            a = per_agent.get(name)
            if not a or a["votes"] == 0:
                raw[name] = WEIGHT_NEUTRAL
                continue
            vts = a["votes"]
            hr = a["wins"] / vts
            lift = hr - base_wr
            shrink = vts / (vts + SHRINK_K)
            w = _clamp(WEIGHT_NEUTRAL + LIFT_GAIN * lift * shrink,
                       config.w_min, config.w_max)
            raw[name] = w
            per_agent_out[name] = {"votes": vts, "hit_rate": hr, "lift": lift,
                                   "shrink": shrink, "raw_weight": w}

        # Mean-normalize to ~1.0, then re-clamp (keeps the voting scale stable
        # and the bounds hard).
        mean = sum(raw.values()) / len(raw) if raw else WEIGHT_NEUTRAL
        weights = {}
        for name, w in raw.items():
            nw = w / mean if mean > 0 else WEIGHT_NEUTRAL
            weights[name] = round(_clamp(nw, config.w_min, config.w_max), 4)

        return {"weights": weights, "per_agent": per_agent_out,
                "sample_size": n, "base_win_rate": base_wr, "verdict": "OK"}
    except Exception:  # pragma: no cover - fail-open
        return {"weights": uniform_weights(names), "per_agent": {},
                "sample_size": 0, "base_win_rate": None,
                "verdict": "INSUFFICIENT_DATA"}


# --------------------------------------------------------------------------- #
# Drift
# --------------------------------------------------------------------------- #
def weight_drift(history: Optional[List[dict]]) -> Optional[float]:
    """Sum of |last - first| across agents over a weight-history list."""
    hist = [h for h in (history or []) if isinstance(h, dict)]
    if len(hist) < 2:
        return None
    first = hist[0].get("weights", {}) or {}
    last = hist[-1].get("weights", {}) or {}
    keys = set(first) | set(last)
    if not keys:
        return None
    return round(sum(abs((_to_float(last.get(k), 0.0) or 0.0)
                         - (_to_float(first.get(k), 0.0) or 0.0))
                     for k in keys), 6)


# --------------------------------------------------------------------------- #
# Persistence (JSON; fail-open). Store shape:
#   {"current": {name: w}, "updated_at": iso, "history": [ {weights, ...}, ... ]}
# --------------------------------------------------------------------------- #
def _read_store(config: OracleWeightsConfig) -> dict:
    data = oa.read_json(config.weights_file)
    return data if isinstance(data, dict) else {}


def current_weights(config: Optional[OracleWeightsConfig] = None) -> Dict[str, float]:
    """The persisted current weights, or uniform when none exist. Never raises."""
    config = config or OracleWeightsConfig()
    try:
        store = _read_store(config)
        cur = store.get("current")
        if isinstance(cur, dict) and cur:
            return {str(k): _to_float(v, WEIGHT_NEUTRAL) or WEIGHT_NEUTRAL
                    for k, v in cur.items()}
    except Exception:  # pragma: no cover - fail-open
        pass
    return uniform_weights()


def weight_history(config: Optional[OracleWeightsConfig] = None) -> List[dict]:
    """The persisted weight-snapshot history (oldest first). Never raises."""
    config = config or OracleWeightsConfig()
    try:
        store = _read_store(config)
        hist = store.get("history")
        return [h for h in hist if isinstance(h, dict)] if isinstance(hist, list) \
            else []
    except Exception:  # pragma: no cover - fail-open
        return []


def save_weights(result: dict,
                 config: Optional[OracleWeightsConfig] = None) -> bool:
    """Persist ``result['weights']`` as current and append a history snapshot.

    Additive: appends to the existing history, never rewrites it. Returns True on
    success. Never raises.
    """
    config = config or OracleWeightsConfig()
    try:
        weights = (result or {}).get("weights") or {}
        if not isinstance(weights, dict) or not weights:
            return False
        store = _read_store(config)
        history = store.get("history")
        if not isinstance(history, list):
            history = []
        snapshot = {
            "weights": weights,
            "sample_size": (result or {}).get("sample_size"),
            "base_win_rate": (result or {}).get("base_win_rate"),
            "verdict": (result or {}).get("verdict"),
            "updated_at": datetime.now(timezone.utc).isoformat(),
        }
        history.append(snapshot)
        store["current"] = weights
        store["updated_at"] = snapshot["updated_at"]
        store["history"] = history
        with open(config.weights_file, "w", encoding="utf-8") as fh:
            json.dump(store, fh, indent=2, default=str)
        return True
    except Exception:  # pragma: no cover - disk safety
        return False


def update_weights(records: Optional[List[dict]] = None,
                   config: Optional[OracleWeightsConfig] = None,
                   persist: bool = True) -> dict:
    """Compute weights from ``records`` and (optionally) persist them. Pure when
    ``persist=False``. Never raises."""
    config = config or OracleWeightsConfig()
    result = compute_weights(records, config)
    if persist and result.get("verdict") == "OK":
        save_weights(result, config)
    return result


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network, no disk side effects outside a temp file)
# --------------------------------------------------------------------------- #
def _synthetic_records(n: int = 30) -> List[dict]:
    """trend is *selectively* predictive — it only takes a directional stand on
    trades that go on to win, so its hit-rate beats the base rate and it earns a
    higher weight. volume convicts indiscriminately (no edge -> ~neutral) and
    liquidity never convicts (stays exactly neutral)."""
    recs = []
    for i in range(n):
        win = i % 4 != 0                                   # ~75% WR
        trend_vote = ({"bullish_score": 0.8, "bearish_score": 0.0,
                       "confidence": 0.9} if win else
                      {"bullish_score": 0.0, "bearish_score": 0.0,
                       "confidence": 0.4})                 # neutral on losers
        recs.append({
            "id": f"t{i}", "pnl": 25.0 if win else -30.0, "max_loss": 100.0,
            "agent_votes": {
                "trend": trend_vote,
                # convicted on every other trade regardless of outcome -> no edge.
                "volume": {"bullish_score": 0.6 if i % 2 else 0.0,
                           "bearish_score": 0.0 if i % 2 else 0.6,
                           "confidence": 0.5},
                # liquidity never convicts -> remains exactly neutral.
                "liquidity": {"bullish_score": 0.0, "bearish_score": 0.0,
                              "confidence": 0.7},
            },
        })
    return recs


def _self_test() -> int:
    ok = True

    # 1) Empty / garbage -> uniform + INSUFFICIENT_DATA, never raises.
    for junk in (None, [], [None, 42, "x"], "x", 7, [{"weird": object()}]):
        r = compute_weights(junk)  # type: ignore[arg-type]
        if r["verdict"] != "INSUFFICIENT_DATA":
            print("FAIL: junk should be insufficient", junk); ok = False
        if set(r["weights"]) != set(AGENT_NAMES):
            print("FAIL: weights must cover all agents"); ok = False
        if any(abs(w - 1.0) > 1e-9 for w in r["weights"].values()):
            print("FAIL: insufficient data must be uniform"); ok = False

    # 2) With predictive history -> OK, all weights bounded, trend out-weighs.
    cfg = OracleWeightsConfig()
    res = compute_weights(_synthetic_records(), cfg)
    if res["verdict"] != "OK":
        print("FAIL: sufficient data should be OK", res["verdict"]); ok = False
    for name, w in res["weights"].items():
        if not (cfg.w_min - 1e-9 <= w <= cfg.w_max + 1e-9):
            print("FAIL: weight out of bounds", name, w); ok = False
    if not (res["weights"]["trend"] > res["weights"]["liquidity"]):
        print("FAIL: predictive agent should out-weigh the neutral one"); ok = False
    if res["weights"]["liquidity"] <= 0:
        print("FAIL: neutral agent should keep a positive weight"); ok = False

    # 3) Determinism.
    if compute_weights(_synthetic_records(), cfg)["weights"] != res["weights"]:
        print("FAIL: compute_weights should be deterministic"); ok = False

    # 4) Persistence round-trip + drift, via a temp file (cleaned up).
    import tempfile
    import uuid
    tmp = os.path.join(tempfile.gettempdir(), f"ow_selftest_{uuid.uuid4().hex}.json")
    try:
        tcfg = OracleWeightsConfig(weights_file=tmp)
        if current_weights(tcfg) != uniform_weights():
            print("FAIL: missing file should read uniform"); ok = False
        if weight_history(tcfg) != []:
            print("FAIL: missing file should have empty history"); ok = False
        if not save_weights(res, tcfg):
            print("FAIL: save_weights should succeed"); ok = False
        if current_weights(tcfg) != {k: v for k, v in res["weights"].items()}:
            print("FAIL: persisted current weights mismatch"); ok = False
        # second snapshot -> history of 2, drift computable.
        save_weights(compute_weights(_synthetic_records(40), tcfg), tcfg)
        hist = weight_history(tcfg)
        if len(hist) != 2:
            print("FAIL: history should hold two snapshots", len(hist)); ok = False
        if weight_drift(hist) is None:
            print("FAIL: drift should be computable over two snapshots"); ok = False
        if weight_drift(hist[:1]) is not None:
            print("FAIL: drift needs >=2 snapshots"); ok = False
    finally:
        try:
            os.remove(tmp)
        except OSError:
            pass

    print("oracle_weights self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
