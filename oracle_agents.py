"""
Oracle 3.0 — Evidence Agents (uniform, context-driven, fail-open protocol).

Ten independent "agents" each look at ONE slice of the tape and return a small,
comparable :class:`AgentVote`::

    AgentVote{name, bullish_score[0..1], bearish_score[0..1],
              confidence[0..1], reasons:[str], data:{...}}

with the invariant ``bullish_score + bearish_score <= 1`` (the remainder is the
agent's *neutral* mass). No agent ever opens, sizes, prices, blocks or alters a
trade; agents only emit evidence. The voting/probability layer combines the
votes — by design **no single agent can trigger a decision** (see the plan's
constraint), and the candlestick agent in particular is confidence-capped so a
lone pattern can never dominate.

Everything is PURE and OFFLINE: each agent reads pre-fetched, normalized fields
from a context ``ctx`` dict (so tests inject data instead of touching services).
``run_agents`` wraps every agent in try/except, so a single broken agent degrades
to a neutral vote rather than taking the slate down (fail-open).

Standard ``ctx`` fields (all optional; agents tolerate missing/garbage values):
    regime/trend/momentum/realized_vol/vix, volume_ratio, news_score, breadth,
    candlestick (PatternStamp.to_dict) or candles, iv_rank, skew, rel_strength,
    rl_preference, plus option liquidity (spread_pct/open_interest/option_volume).
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from config_loader import ConfigLoader

import regime as rg

TREND_MOM = rg.TRENDING_MOMENTUM          # 0.05
HIGH_VOL = rg.VOLATILE_VOL                 # 0.30


# --------------------------------------------------------------------------- #
# Vote container + helpers
# --------------------------------------------------------------------------- #
@dataclass
class AgentVote:
    name: str
    bullish_score: float = 0.0
    bearish_score: float = 0.0
    confidence: float = 0.0
    reasons: List[str] = field(default_factory=list)
    data: Dict = field(default_factory=dict)

    @property
    def neutral_score(self) -> float:
        return _clamp01(1.0 - self.bullish_score - self.bearish_score)

    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "bullish_score": round(self.bullish_score, 6),
            "bearish_score": round(self.bearish_score, 6),
            "neutral_score": round(self.neutral_score, 6),
            "confidence": round(self.confidence, 6),
            "reasons": list(self.reasons),
            "data": dict(self.data),
        }


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _to_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _get(ctx, *keys):
    """First present (non-None) value among ``keys`` in a dict-like ctx."""
    if not isinstance(ctx, dict):
        return None
    for k in keys:
        if k in ctx and ctx[k] is not None:
            return ctx[k]
    return None


def _gf(ctx, *keys) -> Optional[float]:
    return _to_float(_get(ctx, *keys))


def _neutral(name: str, reason: str = "no usable signal",
             data: Optional[dict] = None) -> AgentVote:
    return AgentVote(name=name, bullish_score=0.0, bearish_score=0.0,
                     confidence=0.0, reasons=[reason], data=data or {})


def _directional(name: str, signal: float, confidence: float,
                 reasons: List[str], data: Optional[dict] = None) -> AgentVote:
    """Map a signed ``signal`` in [-1, 1] to a one-sided directional vote.

    Magnitude becomes the directional score; the opposite side stays 0 and the
    remainder is neutral. ``confidence`` is clamped independently.
    """
    mag = _clamp01(abs(signal))
    bull = mag if signal > 0 else 0.0
    bear = mag if signal < 0 else 0.0
    return AgentVote(name=name, bullish_score=bull, bearish_score=bear,
                     confidence=_clamp01(confidence), reasons=reasons,
                     data=data or {})


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class OracleAgentsConfig:
    # Candlestick can NEVER dominate: its confidence is capped here so the
    # voting layer can't let a lone pattern flip a decision.
    candlestick_max_confidence: float = 0.50

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "OracleAgentsConfig":
        try:
            cfg = loader if loader is not None else ConfigLoader(path=path)
            return OracleAgentsConfig(
                candlestick_max_confidence=cfg.get_float(
                    "ORACLE_CANDLESTICK_MAX_CONFIDENCE", 0.50))
        except Exception:  # pragma: no cover - fail-open
            return OracleAgentsConfig()


# --------------------------------------------------------------------------- #
# The ten agents. Each exposes ``name`` and ``evaluate(ctx, config) -> AgentVote``.
# --------------------------------------------------------------------------- #
class TrendAgent:
    name = "trend"

    def evaluate(self, ctx, config) -> AgentVote:
        trend = str(_get(ctx, "trend") or "").strip().lower()
        mom = _gf(ctx, "momentum")
        if mom is None and trend not in ("up", "down"):
            return _neutral(self.name, "no trend/momentum context")
        amom = abs(mom) if mom is not None else TREND_MOM
        # Direction from momentum sign if present, else from the trend label.
        if mom is not None and mom != 0.0:
            sign = 1.0 if mom > 0 else -1.0
        elif trend == "up":
            sign = 1.0
        elif trend == "down":
            sign = -1.0
        else:
            return _neutral(self.name, "flat trend")
        strength = _clamp01(amom / (2.0 * TREND_MOM))
        conf = _clamp01(amom / (2.0 * TREND_MOM))
        return _directional(self.name, sign * strength, conf,
                            [f"{('up' if sign > 0 else 'down')} trend, "
                             f"momentum {mom if mom is not None else 0.0:+.3f}"],
                            {"trend": trend, "momentum": mom})


class VolatilityAgent:
    name = "volatility"

    def evaluate(self, ctx, config) -> AgentVote:
        rv = _gf(ctx, "realized_vol")
        vix = _gf(ctx, "vix")
        if rv is None and vix is None:
            return _neutral(self.name, "no volatility context")
        rvv = rv if rv is not None else 0.20
        # Volatility is non-directional; extreme vol expresses a mild risk-off
        # (bearish) tilt and otherwise stays neutral. It mostly contributes
        # confidence about the *environment*, never a strong directional call.
        if rvv >= HIGH_VOL:
            lean = -_clamp01((rvv - HIGH_VOL) / HIGH_VOL) * 0.5  # capped tilt
            conf = _clamp01((rvv - HIGH_VOL) / HIGH_VOL)
            return _directional(self.name, lean, conf,
                                [f"elevated realized vol {rvv:.2f} (risk-off)"],
                                {"realized_vol": rvv, "vix": vix})
        conf = _clamp01((HIGH_VOL - rvv) / HIGH_VOL) * 0.6
        return AgentVote(self.name, 0.0, 0.0, conf,
                         [f"calm vol {rvv:.2f}"], {"realized_vol": rvv,
                                                   "vix": vix})


class VolumeAgent:
    name = "volume"

    def evaluate(self, ctx, config) -> AgentVote:
        vr = _gf(ctx, "volume_ratio", "rel_volume")
        mom = _gf(ctx, "momentum")
        if vr is None:
            return _neutral(self.name, "no volume context")
        # Volume confirms direction: a >1 ratio amplifies whatever momentum says.
        excess = _clamp01((vr - 1.0))           # 0 at avg, 1 at 2x avg
        if mom is None or mom == 0.0:
            return AgentVote(self.name, 0.0, 0.0, _clamp01(excess * 0.5),
                             [f"volume {vr:.2f}x avg, no direction"],
                             {"volume_ratio": vr})
        sign = 1.0 if mom > 0 else -1.0
        return _directional(self.name, sign * excess, _clamp01(excess),
                            [f"volume {vr:.2f}x avg confirms "
                             f"{'up' if sign > 0 else 'down'} move"],
                            {"volume_ratio": vr, "momentum": mom})


class LiquidityAgent:
    name = "liquidity"

    def evaluate(self, ctx, config) -> AgentVote:
        spread = _gf(ctx, "spread_pct", "bid_ask_spread_pct")
        oi = _gf(ctx, "open_interest")
        ov = _gf(ctx, "option_volume")
        if spread is None and oi is None and ov is None:
            return _neutral(self.name, "no liquidity context")
        # Liquidity is a *tradability* signal, not directional: it only sets
        # confidence (good liquidity -> high, wide spreads -> low). Always
        # neutral on direction.
        quality = 1.0
        reasons = []
        if spread is not None:
            quality *= _clamp01(1.0 - spread / 0.10)   # 10% spread -> 0
            reasons.append(f"spread {spread*100:.1f}%")
        if oi is not None:
            quality *= _clamp01(oi / 1000.0)
            reasons.append(f"OI {oi:.0f}")
        if ov is not None:
            quality *= _clamp01(ov / 500.0)
            reasons.append(f"opt vol {ov:.0f}")
        return AgentVote(self.name, 0.0, 0.0, _clamp01(quality),
                         reasons or ["liquidity"],
                         {"spread_pct": spread, "open_interest": oi,
                          "option_volume": ov})


class NewsAgent:
    name = "news"

    def evaluate(self, ctx, config) -> AgentVote:
        ns = _gf(ctx, "news_score", "sentiment_score")
        if ns is None:
            return _neutral(self.name, "no news/sentiment")
        ns = max(-1.0, min(1.0, ns))
        conf = _clamp01(abs(ns))
        return _directional(self.name, ns, conf,
                            [f"news/sentiment {ns:+.2f}"],
                            {"news_score": ns})


class BreadthAgent:
    name = "breadth"

    def evaluate(self, ctx, config) -> AgentVote:
        b = _gf(ctx, "breadth", "net_advancers")
        if b is None:
            return _neutral(self.name, "no market breadth")
        b = max(-1.0, min(1.0, b))
        conf = _clamp01(abs(b))
        return _directional(self.name, b, conf,
                            [f"market breadth {b:+.2f}"], {"breadth": b})


class CandlestickAgent:
    name = "candlestick"

    def evaluate(self, ctx, config) -> AgentVote:
        cap = getattr(config, "candlestick_max_confidence", 0.50)
        stamp = _get(ctx, "candlestick")
        if not isinstance(stamp, dict):
            candles = _get(ctx, "candles")
            if candles:
                try:  # lazy import; analytics-only, offline-safe
                    from oracle.signals.candlestick_patterns import (
                        detect_primary, CandlestickConfig)
                    s = detect_primary(candles, CandlestickConfig())
                    stamp = s.to_dict() if s is not None else None
                except Exception:
                    stamp = None
        if not isinstance(stamp, dict):
            return _neutral(self.name, "no candlestick pattern")
        bias = str(stamp.get("bias") or "neutral").strip().lower()
        pconf = _to_float(stamp.get("confidence")) or 0.0
        # HARD CAP: a candlestick alone can never carry full conviction.
        conf = _clamp01(min(pconf, cap))
        name = stamp.get("pattern_name", "pattern")
        if bias == "bullish":
            return _directional(self.name, conf, conf,
                                [f"{name} (bullish, capped)"], {"stamp": stamp})
        if bias == "bearish":
            return _directional(self.name, -conf, conf,
                                [f"{name} (bearish, capped)"], {"stamp": stamp})
        return AgentVote(self.name, 0.0, 0.0, 0.0,
                         [f"{name} (neutral)"], {"stamp": stamp})


class OptionsStructureAgent:
    name = "options_structure"

    def evaluate(self, ctx, config) -> AgentVote:
        skew = _gf(ctx, "skew")
        iv_rank = _gf(ctx, "iv_rank")
        if skew is None and iv_rank is None:
            return _neutral(self.name, "no options structure")
        # Skew sign is the directional read (call skew bullish, put skew
        # bearish); iv_rank scales conviction. Bounded and modest.
        if skew is None:
            return AgentVote(self.name, 0.0, 0.0,
                             _clamp01((iv_rank or 0.0) / 100.0
                                      if (iv_rank or 0) > 1 else (iv_rank or 0.0)),
                             [f"iv_rank {iv_rank}"], {"iv_rank": iv_rank})
        s = max(-1.0, min(1.0, skew))
        ivr = iv_rank if iv_rank is not None else 0.0
        ivr = ivr / 100.0 if ivr > 1.0 else ivr
        conf = _clamp01(abs(s) * (0.5 + 0.5 * _clamp01(ivr)))
        return _directional(self.name, s, conf,
                            [f"skew {s:+.2f}, iv_rank {iv_rank}"],
                            {"skew": s, "iv_rank": iv_rank})


class RelativeStrengthAgent:
    name = "relative_strength"

    def evaluate(self, ctx, config) -> AgentVote:
        rs = _gf(ctx, "rel_strength", "relative_strength")
        if rs is None:
            return _neutral(self.name, "no relative strength")
        # rs is the symbol's excess return vs a benchmark; clamp the signal.
        sig = max(-1.0, min(1.0, rs * 10.0))   # 10% excess -> full
        conf = _clamp01(abs(sig))
        return _directional(self.name, sig, conf,
                            [f"relative strength {rs:+.3f}"],
                            {"rel_strength": rs})


class RLPreferenceAgent:
    name = "rl_preference"

    def evaluate(self, ctx, config) -> AgentVote:
        # ADVISORY ONLY: a shadow RL preference in [-1, 1]. Capped conviction.
        pref = _gf(ctx, "rl_preference", "rl_score")
        if pref is None:
            return _neutral(self.name, "no RL preference")
        p = max(-1.0, min(1.0, pref))
        conf = _clamp01(abs(p)) * 0.6           # advisory -> de-weighted conf
        return _directional(self.name, p, conf,
                            [f"RL preference {p:+.2f} (advisory)"],
                            {"rl_preference": p})


# Registry — order is stable for deterministic reports.
AGENTS = [
    TrendAgent(), VolatilityAgent(), VolumeAgent(), LiquidityAgent(),
    NewsAgent(), BreadthAgent(), CandlestickAgent(), OptionsStructureAgent(),
    RelativeStrengthAgent(), RLPreferenceAgent(),
]

AGENT_NAMES = tuple(a.name for a in AGENTS)


def run_agents(ctx, config: Optional[OracleAgentsConfig] = None
               ) -> List[AgentVote]:
    """Evaluate every agent against ``ctx``. Fail-open: a broken agent yields a
    neutral vote and never aborts the slate. Always returns one vote per agent.
    """
    cfg = config if config is not None else OracleAgentsConfig()
    votes: List[AgentVote] = []
    for agent in AGENTS:
        try:
            v = agent.evaluate(ctx, cfg)
            if not isinstance(v, AgentVote):
                v = _neutral(agent.name, "agent returned non-vote")
            # Enforce the invariant defensively.
            v.bullish_score = _clamp01(v.bullish_score)
            v.bearish_score = _clamp01(v.bearish_score)
            if v.bullish_score + v.bearish_score > 1.0:
                total = v.bullish_score + v.bearish_score
                v.bullish_score /= total
                v.bearish_score /= total
            v.confidence = _clamp01(v.confidence)
            votes.append(v)
        except Exception:  # pragma: no cover - fail-open
            votes.append(_neutral(agent.name, "agent error"))
    return votes


# --------------------------------------------------------------------------- #
# Self-test
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = OracleAgentsConfig()

    bull_ctx = {
        "trend": "up", "momentum": 0.08, "realized_vol": 0.18, "vix": 16.0,
        "volume_ratio": 1.6, "news_score": 0.5, "breadth": 0.4,
        "candlestick": {"pattern_name": "hammer", "bias": "bullish",
                        "confidence": 0.9, "requires_confirmation": True},
        "skew": 0.3, "iv_rank": 40.0, "rel_strength": 0.04,
        "rl_preference": 0.6, "spread_pct": 0.01, "open_interest": 5000,
        "option_volume": 2000,
    }
    votes = run_agents(bull_ctx, cfg)
    if len(votes) != len(AGENTS):
        print("FAIL: vote count", len(votes)); ok = False
    for v in votes:
        if not (0.0 <= v.bullish_score <= 1.0 and 0.0 <= v.bearish_score <= 1.0):
            print("FAIL: score range", v); ok = False
        if v.bullish_score + v.bearish_score > 1.0 + 1e-9:
            print("FAIL: bull+bear > 1", v); ok = False
        if not (0.0 <= v.confidence <= 1.0):
            print("FAIL: confidence range", v); ok = False

    by = {v.name: v for v in votes}
    if not (by["trend"].bullish_score > by["trend"].bearish_score):
        print("FAIL: trend should be bullish", by["trend"]); ok = False
    if not (by["news"].bullish_score > 0):
        print("FAIL: news should be bullish", by["news"]); ok = False
    # Candlestick is confidence-capped.
    if by["candlestick"].confidence > cfg.candlestick_max_confidence + 1e-9:
        print("FAIL: candlestick not capped", by["candlestick"]); ok = False

    bear_ctx = {"trend": "down", "momentum": -0.08, "news_score": -0.6,
                "breadth": -0.5, "rel_strength": -0.05, "rl_preference": -0.7}
    bvotes = {v.name: v for v in run_agents(bear_ctx, cfg)}
    if not (bvotes["trend"].bearish_score > bvotes["trend"].bullish_score):
        print("FAIL: trend should be bearish", bvotes["trend"]); ok = False
    if not (bvotes["news"].bearish_score > 0):
        print("FAIL: news should be bearish", bvotes["news"]); ok = False

    # Empty ctx -> all neutral, zero confidence, no raise.
    empty = run_agents({}, cfg)
    for v in empty:
        if v.bullish_score != 0.0 or v.bearish_score != 0.0:
            print("FAIL: empty not neutral", v); ok = False

    # Determinism.
    if [v.to_dict() for v in run_agents(bull_ctx, cfg)] != \
            [v.to_dict() for v in run_agents(bull_ctx, cfg)]:
        print("FAIL: non-deterministic"); ok = False

    # Garbage ctx never raises.
    for junk in (None, 42, "x", [], {"weird": object()}):
        try:
            run_agents(junk, cfg)  # type: ignore[arg-type]
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("oracle_agents self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
