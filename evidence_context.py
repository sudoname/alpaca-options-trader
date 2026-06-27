"""
P0b — Evidence context: stamp the full evidence slate onto one trade decision.

This is the single chokepoint that turns a point-in-time market ``ctx`` dict into
a flat ``evidence`` dict combining EVERY orphaned evidence producer:

  * ``oracle_agents.run_agents``          -> 10 agent votes + signed contributions
  * ``oracle_regime.classify_regime``     -> 8-label regime + confidence
  * ``candlestick_patterns.stamp_...``    -> the 6 frozen candlestick fields
  * ``feature_buckets.extract_features``  -> setup-key dims (regime/vol/dir/...)
  * IV bucket (iv_rank thresholds) + iv-vs-hv (``spread_builder``)

The result is what the shadow recorder persists under ``features_json.evidence``
so the learning engine can later build ``Feature | Trades | Avg Return | Win% |
EV`` leaderboards keyed on each evidence dimension.

Design contract (matches every producer it wraps):
  * PURE + DETERMINISTIC + OFFLINE — no network, no creds, no file writes. All
    market data must already live in ``ctx`` (network lives upstream in the
    live entry path's ``_build_evidence_ctx``).
  * FAIL-OPEN — every producer is wrapped; a broken one degrades that slice to
    ``None``/neutral and never aborts the slate or raises.
  * ADVISORY ONLY — nothing here opens, sizes, prices, blocks or alters a trade.

Standard ``ctx`` fields (all optional; missing -> None/neutral):
    regime ({trending,ranging,volatile}), trend ({up,down,flat}), momentum,
    realized_vol, vix, volume_ratio, news_score, breadth, candles (OHLCV oldest
    -> newest) or candlestick (a PatternStamp dict), skew, iv_rank, iv, hv,
    rel_strength, rl_preference, spread_pct, open_interest, option_volume,
    signal_strength, dte, delta.
"""

from typing import Dict, Optional

import oracle_agents
import oracle_regime
import feature_buckets
from oracle.signals import candlestick_patterns as cs


# --------------------------------------------------------------------------- #
# Helpers
# --------------------------------------------------------------------------- #
def _to_float(value) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _norm_iv_rank(iv_rank) -> Optional[float]:
    """Normalize an IV rank to [0, 1]; tolerates a 0-100 percentage scale."""
    v = _to_float(iv_rank)
    if v is None:
        return None
    if v > 1.0:
        v = v / 100.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def iv_bucket(iv_rank) -> Optional[str]:
    """low / medium / high from a (normalized) IV rank. None when missing.

    Thresholds: low < 0.30, medium 0.30-0.60, high >= 0.60 — matches the
    user's 'Low IV' / 'High IV' leaderboard rows.
    """
    v = _norm_iv_rank(iv_rank)
    if v is None:
        return None
    if v < 0.30:
        return "low"
    if v < 0.60:
        return "medium"
    return "high"


def _iv_vs_hv(ctx: dict) -> str:
    """overpriced / fair / underpriced / unknown from IV vs HV. Fail-open."""
    try:
        from spread_builder import classify_volatility, SpreadConfig
        iv = _to_float(ctx.get("iv"))
        hv = _to_float(ctx.get("hv"))
        if hv is None:
            hv = _to_float(ctx.get("realized_vol"))  # HV proxy
        return classify_volatility(iv, hv, SpreadConfig())
    except Exception:
        return "unknown"


def _candle_stamp(ctx: dict) -> Dict[str, Optional[object]]:
    """Run candlestick detection from ctx candles (or pass through a stamp).

    Returns the 6 frozen ``candlestick_*`` fields (all None when nothing
    detected). Never raises.
    """
    # A pre-computed PatternStamp dict short-circuits detection.
    pre = ctx.get("candlestick")
    if isinstance(pre, dict) and pre.get("pattern_name"):
        return {
            cs.FIELD_PATTERN: pre.get("pattern_name"),
            cs.FIELD_BIAS: pre.get("bias"),
            cs.FIELD_STRENGTH: pre.get("strength"),
            cs.FIELD_CONFIDENCE: _to_float(pre.get("confidence")),
            cs.FIELD_REASON: pre.get("reason"),
            cs.FIELD_REQUIRES_CONFIRMATION: pre.get("requires_confirmation"),
        }
    candles = ctx.get("candles")
    stamped = cs.stamp_candlestick_patterns({}, candles)
    return {f: stamped.get(f) for f in cs.STAMP_FIELDS}


def _agent_evidence(ctx: dict) -> Dict[str, dict]:
    """Per-agent votes + signed directional contributions + consensus.

    ``contribution`` = (bull - bear) * confidence in [-1, 1]; ``consensus`` is
    the mean contribution across the convicted (confidence > 0) agents.
    """
    votes = oracle_agents.run_agents(ctx)
    agent_votes: Dict[str, dict] = {}
    contributions: Dict[str, float] = {}
    convicted = []
    for v in votes:
        net = round(v.bullish_score - v.bearish_score, 6)
        contrib = round(net * v.confidence, 6)
        agent_votes[v.name] = {
            "bull": round(v.bullish_score, 6),
            "bear": round(v.bearish_score, 6),
            "conf": round(v.confidence, 6),
            "net": net,
        }
        contributions[v.name] = contrib
        if v.confidence > 0:
            convicted.append(contrib)
    consensus = round(sum(convicted) / len(convicted), 6) if convicted else 0.0
    return {
        "agent_votes": agent_votes,
        "agent_contributions": contributions,
        "agent_consensus": consensus,
    }


# --------------------------------------------------------------------------- #
# Public entry
# --------------------------------------------------------------------------- #
def compute_evidence(ctx: Optional[dict]) -> dict:
    """Assemble the full flat evidence dict from a market ``ctx``. Never raises.

    Every dimension fails open to ``None`` (or a neutral default) so a missing
    or malformed slice never blocks the others. The shape is stable and
    JSON-serializable for persistence under ``features_json.evidence``.
    """
    if not isinstance(ctx, dict):
        ctx = {}

    evidence: dict = {}

    # 1) Candlestick stamp first — the agent slate reuses the same detection.
    try:
        candle_fields = _candle_stamp(ctx)
    except Exception:
        candle_fields = {f: None for f in cs.STAMP_FIELDS}
    evidence.update(candle_fields)

    # Feed the detected stamp back so CandlestickAgent reads the same pattern.
    agent_ctx = dict(ctx)
    pat = candle_fields.get(cs.FIELD_PATTERN)
    if pat and not isinstance(agent_ctx.get("candlestick"), dict):
        agent_ctx["candlestick"] = {
            "pattern_name": pat,
            "bias": candle_fields.get(cs.FIELD_BIAS),
            "confidence": candle_fields.get(cs.FIELD_CONFIDENCE),
        }

    # 2) Agents.
    try:
        evidence.update(_agent_evidence(agent_ctx))
    except Exception:
        evidence.update({"agent_votes": {}, "agent_contributions": {},
                         "agent_consensus": 0.0})

    # 3) Regime (8-label taxonomy).
    try:
        regime_raw = {
            "regime": ctx.get("regime"),
            "trend": ctx.get("trend"),
            "realized_vol": ctx.get("realized_vol"),
            "momentum": ctx.get("momentum"),
        }
        rc = oracle_regime.classify_regime(
            regime_raw=regime_raw,
            vix=ctx.get("vix"),
            breadth=ctx.get("breadth"),
            news_score=ctx.get("news_score"),
        )
        evidence["regime_label"] = rc.get("label")
        evidence["regime_confidence"] = rc.get("confidence")
    except Exception:
        evidence["regime_label"] = None
        evidence["regime_confidence"] = None

    # 4) Feature buckets (setup-key dims). Build a record from ctx + stamp.
    try:
        record = {
            "regime": ctx.get("regime"),
            "trend": ctx.get("trend"),
            "direction": ctx.get("direction"),
            "momentum": ctx.get("momentum"),
            "realized_vol": ctx.get("realized_vol"),
            "signal_strength": ctx.get("signal_strength"),
            "dte": ctx.get("dte"),
            "days_to_expiration": ctx.get("dte"),
            "delta": ctx.get("delta"),
            "candlestick_pattern": candle_fields.get(cs.FIELD_PATTERN),
        }
        feats = feature_buckets.extract_features(record)
        evidence["regime_bucket"] = feats.get("regime")
        evidence["volatility_bucket"] = feats.get("volatility")
        evidence["direction"] = feats.get("direction")
        evidence["strength"] = feats.get("strength")
        evidence["dte_bucket"] = feats.get("dte_bucket")
        evidence["delta_bucket"] = feats.get("delta_bucket")
        evidence["pattern"] = feats.get("pattern")
    except Exception:
        for k in ("regime_bucket", "volatility_bucket", "direction",
                  "strength", "dte_bucket", "delta_bucket", "pattern"):
            evidence.setdefault(k, None)

    # 5) IV dimensions.
    evidence["iv_rank"] = _norm_iv_rank(ctx.get("iv_rank"))
    evidence["iv_bucket"] = iv_bucket(ctx.get("iv_rank"))
    evidence["iv_vs_hv"] = _iv_vs_hv(ctx)

    return evidence


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # A fully-populated bullish context with a hammer after a downtrend.
    downtrend = [(c + 1, c + 1.2, c - 1, c, 100) for c in (110, 108, 106, 104)]
    hammer = (100.0, 100.5, 95.0, 100.2, 120)
    ctx = {
        "regime": "trending", "trend": "up", "momentum": 0.08,
        "realized_vol": 0.18, "vix": 16.0, "volume_ratio": 1.6,
        "news_score": 0.5, "breadth": 0.4, "skew": 0.3, "iv_rank": 40.0,
        "rel_strength": 0.04, "rl_preference": 0.6, "spread_pct": 0.01,
        "open_interest": 5000, "option_volume": 2000,
        "signal_strength": 3, "dte": 30, "delta": 0.45,
        "candles": downtrend + [hammer],
        "iv": 0.35, "hv": 0.20,
    }
    ev = compute_evidence(ctx)

    # Agents present, all 10, every contribution in range.
    if len(ev["agent_votes"]) != len(oracle_agents.AGENTS):
        print("FAIL: agent count", len(ev["agent_votes"])); ok = False
    for name, c in ev["agent_contributions"].items():
        if not (-1.0 <= c <= 1.0):
            print("FAIL: contribution range", name, c); ok = False
    if ev["agent_votes"]["trend"]["net"] <= 0:
        print("FAIL: trend should be bullish", ev["agent_votes"]["trend"])
        ok = False

    # Regime: trending + up momentum -> TRENDING_BULL.
    if ev["regime_label"] != oracle_regime.TRENDING_BULL:
        print("FAIL: regime label", ev["regime_label"]); ok = False
    if not (0.0 <= (ev["regime_confidence"] or 0) <= 1.0):
        print("FAIL: regime confidence", ev["regime_confidence"]); ok = False

    # Candlestick: hammer detected and flowed into pattern bucket.
    if ev["candlestick_pattern"] != cs.HAMMER:
        print("FAIL: candlestick pattern", ev["candlestick_pattern"]); ok = False
    if ev["pattern"] != "hammer":
        print("FAIL: pattern bucket", ev["pattern"]); ok = False

    # Feature buckets.
    if ev["regime_bucket"] != "trending":
        print("FAIL: regime bucket", ev["regime_bucket"]); ok = False
    if ev["direction"] != "up":
        print("FAIL: direction", ev["direction"]); ok = False
    if ev["volatility_bucket"] != "normal":
        print("FAIL: volatility bucket", ev["volatility_bucket"]); ok = False
    if ev["strength"] != "strong":
        print("FAIL: strength", ev["strength"]); ok = False
    if ev["dte_bucket"] != "22-45":
        print("FAIL: dte bucket", ev["dte_bucket"]); ok = False
    if ev["delta_bucket"] != "0.30-0.50":
        print("FAIL: delta bucket", ev["delta_bucket"]); ok = False

    # IV: rank 40 -> medium; iv 0.35 vs hv 0.20 (ratio 1.75) -> overpriced.
    if ev["iv_bucket"] != "medium":
        print("FAIL: iv bucket", ev["iv_bucket"]); ok = False
    if ev["iv_vs_hv"] != "overpriced":
        print("FAIL: iv_vs_hv", ev["iv_vs_hv"]); ok = False

    # Determinism.
    if compute_evidence(ctx) != compute_evidence(ctx):
        print("FAIL: non-deterministic"); ok = False

    # Empty / garbage ctx -> stable shape, never raises.
    for junk in (None, {}, 42, "x", [], {"weird": object()}):
        try:
            e = compute_evidence(junk)  # type: ignore[arg-type]
            for key in ("agent_votes", "regime_label", "candlestick_pattern",
                        "iv_bucket", "pattern", "direction"):
                if key not in e:
                    print("FAIL: missing key on junk", junk, key); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    # Empty ctx -> neutral agents, no pattern, range-bound regime.
    e0 = compute_evidence({})
    if e0["candlestick_pattern"] is not None:
        print("FAIL: empty should have no pattern", e0["candlestick_pattern"])
        ok = False
    if e0["agent_consensus"] != 0.0:
        print("FAIL: empty consensus should be 0", e0["agent_consensus"]); ok = False

    print("evidence_context self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
