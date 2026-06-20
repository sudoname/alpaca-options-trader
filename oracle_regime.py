"""
Oracle 3.0 — Market Regime Agent (8-label taxonomy + confidence).

A thin, deterministic LABELLING layer on top of ``regime.detect_regime``. It does
NOT predict price and it does NOT pick or place trades. It maps the existing
3x3 (regime x trend) classification — plus realized vol, momentum and optional
VIX / market-breadth / news corroboration — onto a richer 8-label taxonomy that
the Intelligence Layer's reports and agents can reason about:

    TRENDING_BULL, TRENDING_BEAR, RANGE_BOUND, HIGH_VOLATILITY,
    LOW_VOLATILITY, NEWS_DRIVEN, BREAKOUT, PANIC_SELLING

Two design rules keep this honest:

  * NO single axis triggers a "dramatic" label. NEWS_DRIVEN and PANIC_SELLING
    each require corroboration from at least two independent axes (e.g. extreme
    realized vol AND a strong directional move, optionally reinforced by news);
    a lone news headline or a lone vol spike cannot, by itself, produce them.
  * CONFIDENCE is the normalized margin by which the deciding thresholds were
    cleared, in [0, 1]. Sparse / borderline tape yields low confidence rather
    than a falsely decisive label.

This module is ANALYTICS / SHADOW ONLY: it reads a ``MarketView`` (or an injected
regime dict) and returns a label. It never opens, sizes, prices, blocks or alters
any real or paper trade, and every public function fails open (never raises).
"""

from dataclasses import dataclass
from typing import Dict, List, Optional

import regime as rg
from config_loader import ConfigLoader

# 8-label taxonomy.
TRENDING_BULL = "TRENDING_BULL"
TRENDING_BEAR = "TRENDING_BEAR"
RANGE_BOUND = "RANGE_BOUND"
HIGH_VOLATILITY = "HIGH_VOLATILITY"
LOW_VOLATILITY = "LOW_VOLATILITY"
NEWS_DRIVEN = "NEWS_DRIVEN"
BREAKOUT = "BREAKOUT"
PANIC_SELLING = "PANIC_SELLING"

REGIME_LABELS = (
    TRENDING_BULL, TRENDING_BEAR, RANGE_BOUND, HIGH_VOLATILITY,
    LOW_VOLATILITY, NEWS_DRIVEN, BREAKOUT, PANIC_SELLING,
)

# Derived thresholds, anchored on regime.py so the labels stay consistent with
# the live classification.
LOW_VOL_MAX = 0.15                 # below this realized vol -> "calm"
HIGH_VOL_MIN = rg.VOLATILE_VOL     # 0.30 -> volatile regime
EXTREME_VOL_MIN = 0.50             # panic territory
BREAKOUT_MOMENTUM = 2.0 * rg.TRENDING_MOMENTUM   # 0.10 — a strong directional thrust
PANIC_MOMENTUM = rg.TRENDING_MOMENTUM            # 0.05 — strong down move for panic
NEWS_STRONG = 0.30                 # |news_score| beyond this is a strong signal
VIX_ELEVATED = 25.0
VIX_PANIC = 35.0


def _clamp01(x: float) -> float:
    return 0.0 if x < 0.0 else 1.0 if x > 1.0 else x


def _to_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


@dataclass
class OracleRegimeConfig:
    vix_symbol: str = "^VIX"

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "OracleRegimeConfig":
        try:
            cfg = loader if loader is not None else ConfigLoader(path=path)
            return OracleRegimeConfig(
                vix_symbol=cfg.get_str("ORACLE_REGIME_VIX_SYMBOL", "^VIX"))
        except Exception:  # pragma: no cover - fail-open
            return OracleRegimeConfig()


def _neutral(reasons: Optional[List[str]] = None) -> dict:
    return {
        "label": RANGE_BOUND,
        "confidence": 0.0,
        "components": {"regime": "ranging", "trend": "flat",
                       "realized_vol": None, "momentum": None,
                       "vix": None, "breadth": None, "news_score": None},
        "reasons": reasons or ["insufficient market context"],
    }


def classify_regime(market_view=None, *, regime_raw: Optional[dict] = None,
                    vix=None, breadth=None, news_score=None,
                    symbol: str = "SPY",
                    config: Optional[OracleRegimeConfig] = None) -> dict:
    """Classify the current market into one of the 8 labels. Never raises.

    Pass ``regime_raw`` (a ``regime.detect_regime`` dict) to stay pure/offline,
    or a ``market_view`` to derive it. ``vix`` / ``breadth`` / ``news_score`` are
    optional corroborating axes; when absent the dramatic labels simply cannot
    fire from a single source.

    Returns ``{label, confidence[0..1], components{...}, reasons[]}``.
    """
    try:
        raw = regime_raw
        if raw is None and market_view is not None:
            raw = rg.detect_regime(market_view, symbol)
        if not isinstance(raw, dict):
            return _neutral()

        regime = str(raw.get("regime") or "ranging").strip().lower()
        trend = str(raw.get("trend") or "flat").strip().lower()
        rvol = _to_float(raw.get("realized_vol"))
        mom = _to_float(raw.get("momentum"))
        vixf = _to_float(vix)
        breadthf = _to_float(breadth)
        newsf = _to_float(news_score)

        components = {
            "regime": regime, "trend": trend, "realized_vol": rvol,
            "momentum": mom, "vix": vixf, "breadth": breadthf,
            "news_score": newsf,
        }

        amom = abs(mom) if mom is not None else 0.0
        rv = rvol if rvol is not None else 0.20
        reasons: List[str] = []

        # --- PANIC_SELLING: extreme vol AND a strong down move (>=2 axes). --- #
        panic_vol = rv >= EXTREME_VOL_MIN or (vixf is not None and vixf >= VIX_PANIC)
        panic_dir = (trend == "down" and mom is not None and mom <= -PANIC_MOMENTUM)
        if panic_vol and panic_dir:
            vol_margin = _clamp01((rv - EXTREME_VOL_MIN) / EXTREME_VOL_MIN)
            dir_margin = _clamp01((amom - PANIC_MOMENTUM) / PANIC_MOMENTUM)
            conf = _clamp01(0.55 + 0.30 * max(vol_margin, dir_margin))
            if newsf is not None and newsf <= -NEWS_STRONG:
                conf = _clamp01(conf + 0.10)
                reasons.append("negative news corroborates the sell-off")
            reasons.append(f"extreme vol ({rv:.2f}) with a strong down move")
            return {"label": PANIC_SELLING, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # --- NEWS_DRIVEN: strong news AND elevated vol (>=2 axes). --- #
        news_strong = newsf is not None and abs(newsf) >= NEWS_STRONG
        vol_elevated = rv >= HIGH_VOL_MIN or (vixf is not None and vixf >= VIX_ELEVATED)
        if news_strong and vol_elevated:
            news_margin = _clamp01((abs(newsf) - NEWS_STRONG) / (1.0 - NEWS_STRONG))
            conf = _clamp01(0.50 + 0.35 * news_margin)
            reasons.append(f"strong news ({newsf:+.2f}) into an elevated-vol tape")
            return {"label": NEWS_DRIVEN, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # --- BREAKOUT: a strong directional thrust clearing 2x trend momentum,
        #     while NOT in a high-vol/panic regime (a controlled expansion). --- #
        if (trend in ("up", "down") and mom is not None
                and amom >= BREAKOUT_MOMENTUM and rv < EXTREME_VOL_MIN):
            margin = _clamp01((amom - BREAKOUT_MOMENTUM) / BREAKOUT_MOMENTUM)
            conf = _clamp01(0.50 + 0.35 * margin)
            reasons.append(f"momentum thrust ({mom:+.2f}) beyond breakout band")
            return {"label": BREAKOUT, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # --- Volatile regime (not panic) -> HIGH_VOLATILITY. --- #
        if regime == "volatile":
            margin = _clamp01((rv - HIGH_VOL_MIN) / HIGH_VOL_MIN)
            conf = _clamp01(0.45 + 0.40 * margin)
            reasons.append(f"realized vol {rv:.2f} above {HIGH_VOL_MIN:.2f}")
            return {"label": HIGH_VOLATILITY, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # --- Trending regime -> directional label. --- #
        if regime == "trending" and trend in ("up", "down"):
            margin = _clamp01((amom - rg.TRENDING_MOMENTUM) / rg.TRENDING_MOMENTUM)
            conf = _clamp01(0.45 + 0.40 * margin)
            label = TRENDING_BULL if trend == "up" else TRENDING_BEAR
            reasons.append(f"{trend} trend, momentum {mom:+.2f}")
            return {"label": label, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # --- Quiet tape: LOW_VOLATILITY when genuinely calm, else RANGE_BOUND. --- #
        if rv < LOW_VOL_MAX:
            margin = _clamp01((LOW_VOL_MAX - rv) / LOW_VOL_MAX)
            conf = _clamp01(0.45 + 0.35 * margin)
            reasons.append(f"calm tape, realized vol {rv:.2f} below {LOW_VOL_MAX:.2f}")
            return {"label": LOW_VOLATILITY, "confidence": round(conf, 4),
                    "components": components, "reasons": reasons}

        # Default: range-bound. Confidence rises as momentum sits near flat.
        flat_margin = _clamp01((rg.TRENDING_MOMENTUM - amom) / rg.TRENDING_MOMENTUM)
        conf = _clamp01(0.40 + 0.30 * flat_margin)
        reasons.append("no directional or volatility edge; range-bound")
        return {"label": RANGE_BOUND, "confidence": round(conf, 4),
                "components": components, "reasons": reasons}
    except Exception:  # pragma: no cover - fail-open
        return _neutral(["classification error — defaulted to range-bound"])


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network — injected regime dicts)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    def _raw(regime, trend, rvol, mom):
        return {"regime": regime, "trend": trend, "realized_vol": rvol,
                "momentum": mom}

    cases = [
        # (raw, kwargs, expected_label)
        (_raw("trending", "up", 0.20, 0.06), {}, TRENDING_BULL),
        (_raw("trending", "down", 0.20, -0.06), {}, TRENDING_BEAR),
        (_raw("ranging", "flat", 0.20, 0.0), {}, RANGE_BOUND),
        (_raw("ranging", "flat", 0.10, 0.0), {}, LOW_VOLATILITY),
        (_raw("volatile", "flat", 0.40, 0.0), {}, HIGH_VOLATILITY),
        # Breakout: strong thrust, controlled vol.
        (_raw("trending", "up", 0.25, 0.14), {}, BREAKOUT),
        # Panic: extreme vol + strong down move.
        (_raw("volatile", "down", 0.60, -0.12), {}, PANIC_SELLING),
        # News-driven: strong news into elevated vol; modest move (no breakout).
        (_raw("volatile", "flat", 0.35, 0.0), {"news_score": -0.6}, NEWS_DRIVEN),
    ]
    for raw, kw, want in cases:
        got = classify_regime(regime_raw=raw, **kw)
        if got["label"] != want:
            print("FAIL: label", raw, kw, "->", got["label"], "want", want)
            ok = False
        if not (0.0 <= got["confidence"] <= 1.0):
            print("FAIL: confidence out of range", got); ok = False

    # No single axis triggers PANIC: extreme vol alone (no down move) must not.
    solo_vol = classify_regime(regime_raw=_raw("volatile", "flat", 0.65, 0.0))
    if solo_vol["label"] == PANIC_SELLING:
        print("FAIL: lone vol spike should not be PANIC", solo_vol); ok = False

    # No single axis triggers NEWS_DRIVEN: strong news but calm vol must not.
    solo_news = classify_regime(regime_raw=_raw("ranging", "flat", 0.12, 0.0),
                                news_score=-0.8)
    if solo_news["label"] == NEWS_DRIVEN:
        print("FAIL: lone news should not be NEWS_DRIVEN", solo_news); ok = False

    # Determinism.
    a = classify_regime(regime_raw=_raw("trending", "up", 0.2, 0.06))
    b = classify_regime(regime_raw=_raw("trending", "up", 0.2, 0.06))
    if a != b:
        print("FAIL: non-deterministic"); ok = False

    # Never raises on garbage / empty.
    for junk in (None, 42, "x", [], {"weird": object()}):
        try:
            r = classify_regime(regime_raw=junk)  # type: ignore[arg-type]
            if r["label"] not in REGIME_LABELS:
                print("FAIL: junk label", r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("oracle_regime self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
