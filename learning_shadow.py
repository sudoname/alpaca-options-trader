"""
Phase 8A — learning shadow layer (ADVISORY ONLY).

Observes signals that the engine already produced for a spread idea and emits a
*shadow recommendation* with a confidence score. It is strictly advisory: it
never alters trade decisions, never places orders, and never touches the broker.
Every observation is logged to ``learning_shadow_log.csv`` so the recommendations
can later be scored against actual outcomes (see oracle_dataset_builder).

Observed signals:
    volatility_edge   normalized edge in [-1, 1] (+ = options overpriced)
    oracle_score      0-100 proposal confidence (from spread_builder)
    spread_type       a spread_builder strategy_name (credit/debit/iron/no_trade)
    dte               days to expiration
    trend             'bullish' | 'bearish' | 'neutral'
    vix               index level in points (mapped to a regime)

Output (:class:`ShadowRecommendation`):
    recommendation    'STRONG_TAKE' | 'TAKE' | 'NEUTRAL' | 'AVOID'
    confidence        float in [0, 1]
    rationale         short human-readable explanation

The heuristic is deterministic and unit-testable. It blends the oracle score
with edge/trend/DTE/VIX alignment — purely to *describe* conviction, not to gate
anything.
"""

import csv
import logging
import os
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from typing import Optional

from config_loader import ConfigLoader

try:
    from spread_builder import (
        CREDIT_STRATEGIES, DEBIT_STRATEGIES, NO_TRADE, STRATEGY_PROFILE,
    )
except Exception:  # pragma: no cover - keep importable in isolation
    NO_TRADE = "no_trade"
    CREDIT_STRATEGIES = {"bullish_put_credit_spread", "bearish_call_credit_spread",
                         "iron_condor"}
    DEBIT_STRATEGIES = {"debit_call_spread", "debit_put_spread"}
    STRATEGY_PROFILE = {}

logger = logging.getLogger(__name__)

REC_STRONG_TAKE = "STRONG_TAKE"
REC_TAKE = "TAKE"
REC_NEUTRAL = "NEUTRAL"
REC_AVOID = "AVOID"

REGIME_LOW = "low"
REGIME_NORMAL = "normal"
REGIME_ELEVATED = "elevated"
REGIME_HIGH = "high"
REGIME_UNKNOWN = "unknown"


def vix_regime(vix: Optional[float]) -> str:
    if not isinstance(vix, (int, float)) or vix <= 0:
        return REGIME_UNKNOWN
    if vix < 13:
        return REGIME_LOW
    if vix < 20:
        return REGIME_NORMAL
    if vix < 30:
        return REGIME_ELEVATED
    return REGIME_HIGH


def _clamp01(x: float) -> float:
    return max(0.0, min(1.0, x))


@dataclass
class LearningShadowConfig:
    enabled: bool = True
    log_file: str = "learning_shadow_log.csv"
    take_threshold: float = 0.55
    strong_threshold: float = 0.75
    avoid_threshold: float = 0.40

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "LearningShadowConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return LearningShadowConfig(
            enabled=cfg.get_bool("LEARNING_SHADOW_ENABLED", True),
            log_file=cfg.get_str("LEARNING_SHADOW_LOG_FILE",
                                 "learning_shadow_log.csv"),
            take_threshold=cfg.get_float("LEARNING_SHADOW_TAKE_THRESHOLD", 0.55),
            strong_threshold=cfg.get_float("LEARNING_SHADOW_STRONG_THRESHOLD", 0.75),
            avoid_threshold=cfg.get_float("LEARNING_SHADOW_AVOID_THRESHOLD", 0.40),
        )


@dataclass
class ShadowObservation:
    symbol: str = ""
    volatility_edge: Optional[float] = None
    oracle_score: Optional[float] = None
    spread_type: str = ""
    dte: Optional[int] = None
    trend: Optional[str] = None
    vix: Optional[float] = None

    def as_dict(self) -> dict:
        return asdict(self)


@dataclass
class ShadowRecommendation:
    recommendation: str = REC_NEUTRAL
    confidence: float = 0.0
    rationale: str = ""
    vix_regime: str = REGIME_UNKNOWN

    def to_dict(self) -> dict:
        return asdict(self)


class LearningShadow:
    def __init__(self, config: Optional[LearningShadowConfig] = None):
        self.config = config or LearningShadowConfig.from_env()

    def evaluate(self, obs: ShadowObservation) -> ShadowRecommendation:
        """Produce an advisory recommendation + confidence (no side effects)."""
        regime = vix_regime(obs.vix)
        strategy = obs.spread_type or ""
        notes = []

        # Hard advisory: a no_trade idea is never a take.
        if strategy == NO_TRADE or not strategy:
            return ShadowRecommendation(
                recommendation=REC_AVOID, confidence=0.0,
                rationale="no tradeable structure", vix_regime=regime)

        is_credit = strategy in CREDIT_STRATEGIES
        is_debit = strategy in DEBIT_STRATEGIES

        base = (_clamp01(obs.oracle_score / 100.0)
                if isinstance(obs.oracle_score, (int, float)) else 0.5)
        adjust = 0.0

        # Volatility-edge alignment: credit likes overpriced (edge>0); debit
        # likes underpriced (edge<0). Misalignment penalizes.
        edge = obs.volatility_edge
        if isinstance(edge, (int, float)):
            if is_credit and edge > 0:
                adjust += min(edge, 0.5) * 0.4; notes.append("edge+credit")
            elif is_debit and edge < 0:
                adjust += min(-edge, 0.5) * 0.4; notes.append("edge+debit")
            elif is_credit and edge < 0:
                adjust -= min(-edge, 0.5) * 0.4; notes.append("edge-credit")
            elif is_debit and edge > 0:
                adjust -= min(edge, 0.5) * 0.4; notes.append("edge-debit")

        # DTE sweet spot.
        if isinstance(obs.dte, (int, float)):
            if 30 <= obs.dte <= 45:
                adjust += 0.05; notes.append("dte_sweet")
            elif obs.dte < 14 or obs.dte > 90:
                adjust -= 0.10; notes.append("dte_poor")

        # Trend alignment vs the strategy's desired trend.
        desired_trend = STRATEGY_PROFILE.get(strategy, (None, None))[1]
        if desired_trend and obs.trend:
            if desired_trend == obs.trend:
                adjust += 0.05; notes.append("trend_match")
            elif desired_trend != "neutral" and obs.trend not in (desired_trend, "neutral"):
                adjust -= 0.10; notes.append("trend_mismatch")

        # VIX regime: rich premium favors credit; cheap premium favors debit.
        if is_credit and regime in (REGIME_ELEVATED, REGIME_HIGH):
            adjust += 0.05; notes.append("vix_favors_credit")
        elif is_debit and regime == REGIME_LOW:
            adjust += 0.05; notes.append("vix_favors_debit")
        elif is_credit and regime == REGIME_LOW:
            adjust -= 0.05; notes.append("vix_hurts_credit")
        elif is_debit and regime in (REGIME_ELEVATED, REGIME_HIGH):
            adjust -= 0.05; notes.append("vix_hurts_debit")

        confidence = round(_clamp01(base + adjust), 4)
        if confidence >= self.config.strong_threshold:
            rec = REC_STRONG_TAKE
        elif confidence >= self.config.take_threshold:
            rec = REC_TAKE
        elif confidence >= self.config.avoid_threshold:
            rec = REC_NEUTRAL
        else:
            rec = REC_AVOID

        return ShadowRecommendation(
            recommendation=rec, confidence=confidence,
            rationale=", ".join(notes) or "baseline", vix_regime=regime)

    def observe_and_log(self, obs: ShadowObservation,
                        path: Optional[str] = None) -> ShadowRecommendation:
        """Evaluate ``obs`` and append the observation + recommendation to CSV."""
        rec = self.evaluate(obs)
        path = path or self.config.log_file
        row = {"timestamp": datetime.now(timezone.utc).isoformat()}
        row.update(obs.as_dict())
        row.update(rec.to_dict())
        _csv_append_row(path, row)
        print(f"[LEARNING_SHADOW] ADVISORY sym={obs.symbol or '?'} "
              f"rec={rec.recommendation} conf={rec.confidence:.2f} "
              f"regime={rec.vix_regime} ({rec.rationale}) "
              f"(advisory only — does NOT alter trades)")
        return rec


# --------------------------------------------------------------------------- #
# CSV append helper (header auto-managed; never raises)
# --------------------------------------------------------------------------- #
def _csv_append_row(path: str, row: dict) -> None:
    try:
        existing = []
        fieldnames = []
        if os.path.exists(path) and os.path.getsize(path) > 0:
            with open(path, "r", newline="", encoding="utf-8") as f:
                reader = csv.DictReader(f)
                fieldnames = list(reader.fieldnames or [])
                existing = list(reader)
        new_fields = [k for k in row.keys() if k not in fieldnames]
        if new_fields:
            fieldnames = fieldnames + new_fields
            with open(path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                for e in existing:
                    writer.writerow(e)
                writer.writerow(row)
        else:
            with open(path, "a", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writerow(row)
    except Exception as exc:
        logger.warning("learning_shadow log write failed (%s): %s", path, exc)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile

    ok = True
    shadow = LearningShadow(LearningShadowConfig(enabled=True))

    if vix_regime(10) != REGIME_LOW or vix_regime(18) != REGIME_NORMAL:
        print("FAIL: regime low/normal"); ok = False
    if vix_regime(25) != REGIME_ELEVATED or vix_regime(40) != REGIME_HIGH:
        print("FAIL: regime elevated/high"); ok = False
    if vix_regime(None) != REGIME_UNKNOWN:
        print("FAIL: regime unknown"); ok = False

    # Strong aligned credit idea -> high confidence take.
    good = shadow.evaluate(ShadowObservation(
        symbol="SPY", volatility_edge=0.4, oracle_score=80.0,
        spread_type="bullish_put_credit_spread", dte=40, trend="bullish", vix=28.0))
    if good.recommendation not in (REC_TAKE, REC_STRONG_TAKE) or good.confidence < 0.7:
        print("FAIL: good idea", good.to_dict()); ok = False

    # Misaligned (credit but underpriced + trend mismatch) -> lower.
    bad = shadow.evaluate(ShadowObservation(
        symbol="SPY", volatility_edge=-0.4, oracle_score=45.0,
        spread_type="bullish_put_credit_spread", dte=5, trend="bearish", vix=11.0))
    if bad.confidence >= good.confidence:
        print("FAIL: bad >= good", bad.to_dict(), good.to_dict()); ok = False

    # no_trade -> AVOID.
    nt = shadow.evaluate(ShadowObservation(spread_type=NO_TRADE, oracle_score=99.0))
    if nt.recommendation != REC_AVOID or nt.confidence != 0.0:
        print("FAIL: no_trade avoid", nt.to_dict()); ok = False

    # Logging round-trips.
    d = tempfile.mkdtemp()
    path = os.path.join(d, "shadow.csv")
    shadow.observe_and_log(ShadowObservation(
        symbol="QQQ", volatility_edge=0.2, oracle_score=70.0,
        spread_type="debit_call_spread", dte=35, trend="bullish", vix=12.0), path=path)
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if len(rows) != 1 or "recommendation" not in rows[0] or "confidence" not in rows[0]:
        print("FAIL: shadow log", rows); ok = False

    print("learning_shadow self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
