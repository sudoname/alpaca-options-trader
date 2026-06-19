"""
P13 — Shared feature-bucketing primitives (pure, no I/O, never raises).

Single source of truth for the *setup key* dimensions used by the Learned Edge
engine (``learned_edge``), the historical setup profiler (``historical_profiler``)
and the offline shadow-ranking replay (``shadow_ranking``). Keeping the bucketing
in one place guarantees the same trade lands in the same setup key everywhere.

Design (confirmed):
  * HYBRID buckets — regime and direction use the REAL labels emitted by
    ``regime.py`` ({trending, ranging, volatile} + {up, down, flat}); the
    volatility bucket {low, normal, elevated, extreme} is derived from realized
    vol thresholds anchored on ``regime.py`` constants. The IV/HV
    overpriced/fair/underpriced axis (``spread_builder.classify_volatility``) is
    a DIFFERENT axis and is NOT part of the primary setup key.
  * PATTERN is an OPTIONAL dimension — the candlestick pattern enters the setup
    key only when a record carries one. Legacy trades (no pattern) simply omit
    that dimension; aggregations degrade gracefully.

This module only LABELS trades. It never reads files, never touches the network,
and never opens, sizes, blocks or alters any real or paper trade. Every function
is None-tolerant and returns a label (or None) without raising.
"""

from typing import Dict, Optional, Tuple

import oracle_analytics as oa
import regime as rg

# --------------------------------------------------------------------------- #
# Label vocabularies
# --------------------------------------------------------------------------- #
REGIME_LABELS = ("trending", "ranging", "volatile")
DIRECTION_LABELS = ("up", "down", "flat")
VOL_LABELS = ("low", "normal", "elevated", "extreme")
STRENGTH_LABELS = ("weak", "medium", "strong")
DTE_LABELS = ("0-7", "8-21", "22-45", "45+")
DELTA_LABELS = ("<0.30", "0.30-0.50", ">0.50")

# Volatility-bucket thresholds (anchored on regime.py). Half-open [lo, hi).
VOL_LOW_MAX = 0.15
VOL_NORMAL_MAX = rg.VOLATILE_VOL          # 0.30
VOL_ELEVATED_MAX = 0.50

# Setup-key dimensions in importance order (kept-longest first for backoff).
# 'pattern' is last so it is dropped first when a cohort is sparse.
SETUP_DIMENSIONS = (
    "regime", "volatility", "direction", "strength",
    "dte_bucket", "delta_bucket", "pattern",
)

# Token used to render an explicitly-present-but-None dimension in a key string.
NONE_TOKEN = "none"


# --------------------------------------------------------------------------- #
# Single-dimension bucketers (all None-tolerant)
# --------------------------------------------------------------------------- #
def regime_bucket(regime) -> Optional[str]:
    """Normalize a regime label to one of REGIME_LABELS, else None."""
    if regime is None:
        return None
    s = str(regime).strip().lower()
    return s if s in REGIME_LABELS else None


def direction_bucket(trend) -> Optional[str]:
    """Normalize a trend/direction label to one of DIRECTION_LABELS, else None.

    Tolerates a few common synonyms (bullish/up, bearish/down, neutral/flat).
    """
    if trend is None:
        return None
    s = str(trend).strip().lower()
    if s in DIRECTION_LABELS:
        return s
    synonyms = {
        "bullish": "up", "bull": "up", "long": "up",
        "bearish": "down", "bear": "down", "short": "down",
        "neutral": "flat", "sideways": "flat", "none": None,
    }
    return synonyms.get(s)


def volatility_bucket(realized_vol) -> Optional[str]:
    """Bucket a realized-vol fraction into {low, normal, elevated, extreme}.

    Anchored on regime.py thresholds: low <0.15, normal 0.15-0.30,
    elevated 0.30-0.50, extreme >=0.50. None when the value is missing.
    """
    rv = oa._to_float(realized_vol)
    if rv is None:
        return None
    if rv < VOL_LOW_MAX:
        return "low"
    if rv < VOL_NORMAL_MAX:
        return "normal"
    if rv < VOL_ELEVATED_MAX:
        return "elevated"
    return "extreme"


def strength_bucket(signal_strength) -> Optional[str]:
    """Map an integer signal-strength count to weak/medium/strong.

    0 -> weak, 1-2 -> medium, 3+ -> strong (mirrors smart_trader's winning-side
    signal count). None when missing/non-numeric.
    """
    v = oa._to_float(signal_strength)
    if v is None:
        return None
    n = int(v)
    if n <= 0:
        return "weak"
    if n <= 2:
        return "medium"
    return "strong"


def dte_bucket(dte) -> Optional[str]:
    """Bucket days-to-expiry: 0-7 / 8-21 / 22-45 / 45+ (half-open). None when
    missing/non-numeric."""
    v = oa._to_float(dte)
    if v is None:
        return None
    if v <= 7:
        return "0-7"
    if v <= 21:
        return "8-21"
    if v <= 45:
        return "22-45"
    return "45+"


def delta_bucket(delta) -> Optional[str]:
    """Bucket |delta|: <0.30 / 0.30-0.50 / >0.50. None when missing/non-numeric."""
    v = oa._to_float(delta)
    if v is None:
        return None
    a = abs(v)
    if a < 0.30:
        return "<0.30"
    if a <= 0.50:
        return "0.30-0.50"
    return ">0.50"


def pattern_bucket(pattern_name) -> Optional[str]:
    """Passthrough a candlestick pattern name (lower-cased), else None.

    Neutral / empty pattern labels are treated as 'no pattern' (None) so the
    OPTIONAL pattern dimension is only present for an actual directional stamp.
    """
    if pattern_name is None:
        return None
    s = str(pattern_name).strip().lower()
    if not s or s in ("none", "neutral", "n/a", "na"):
        return None
    return s


# --------------------------------------------------------------------------- #
# Tolerant feature extraction across the several historical schemas
# --------------------------------------------------------------------------- #
def _metrics(record: dict) -> dict:
    m = record.get("metrics") if isinstance(record, dict) else None
    return m if isinstance(m, dict) else {}


def _raw_features(record: dict) -> dict:
    """episode_store rows carry a features_json blob whose ``raw`` holds the
    point-in-time market context. Tolerate a parsed dict or a JSON string."""
    fj = record.get("features_json") if isinstance(record, dict) else None
    if isinstance(fj, str):
        try:
            import json
            fj = json.loads(fj)
        except Exception:
            fj = None
    if isinstance(fj, dict):
        raw = fj.get("raw")
        if isinstance(raw, dict):
            return raw
    return {}


def extract_features(record: dict) -> Dict[str, Optional[str]]:
    """Pull every setup dimension from a trade-like record across ALL schemas.

    Tolerant of trading_history.json (metrics.signal_strength / entry_delta),
    spread_paper_trades.json (dte, iv_rank, volatility_edge), episode_store rows
    (features_json.raw momentum/spy_change) and forward stamped candidates
    (regime, realized_vol, candlestick_pattern). Missing dimensions resolve to
    None and are dropped by the engine's backoff. Never raises.
    """
    if not isinstance(record, dict):
        return {d: None for d in SETUP_DIMENSIONS}

    metrics = _metrics(record)
    raw = _raw_features(record)

    # Regime.
    regime = oa._get(record, "regime") or raw.get("regime")

    # Direction / trend: explicit label, else sign of momentum/spy_change.
    trend = (oa._get(record, "trend", "direction")
             or raw.get("trend") or raw.get("direction"))
    direction = direction_bucket(trend)
    if direction is None:
        mom = oa._to_float(
            oa._get(record, "momentum") if isinstance(record, dict) else None)
        if mom is None:
            mom = oa._to_float(raw.get("momentum"))
        if mom is None:
            mom = oa._to_float(raw.get("spy_change"))
        if mom is not None:
            if mom > rg.TREND_FLAT_BAND:
                direction = "up"
            elif mom < -rg.TREND_FLAT_BAND:
                direction = "down"
            else:
                direction = "flat"

    # Realized vol.
    rv = (oa._to_float(oa._get(record, "realized_vol", "volatility"))
          or oa._to_float(raw.get("realized_vol")))

    # Signal strength.
    strength = oa._get(record, "signal_strength")
    if strength is None:
        strength = metrics.get("signal_strength")
    if strength is None:
        strength = raw.get("signal_strength")

    # DTE.
    dte = oa._trade_dte(record)

    # Delta.
    delta = oa._get(record, "delta", "entry_delta")
    if delta is None:
        delta = metrics.get("entry_delta")

    # Candlestick pattern (OPTIONAL dimension).
    pattern = oa._get(record, "candlestick_pattern", "pattern")

    return {
        "regime": regime_bucket(regime),
        "volatility": volatility_bucket(rv),
        "direction": direction,
        "strength": strength_bucket(strength),
        "dte_bucket": dte_bucket(dte),
        "delta_bucket": delta_bucket(delta),
        "pattern": pattern_bucket(pattern),
    }


# --------------------------------------------------------------------------- #
# Setup-key construction
# --------------------------------------------------------------------------- #
def make_setup_key(record: dict) -> Dict[str, str]:
    """Build the setup key for a record.

    The PATTERN dimension is included ONLY when the record carries a pattern
    (decision: pattern is optional). All other dimensions are always present
    (their value may be None, which the engine's backoff then drops).
    """
    feats = extract_features(record)
    key: Dict[str, str] = {}
    for dim in SETUP_DIMENSIONS:
        val = feats.get(dim)
        if dim == "pattern":
            if val is not None:           # optional dimension
                key[dim] = val
        else:
            key[dim] = val
    return key


def setup_key_tuple(key: Dict[str, Optional[str]], dims=SETUP_DIMENSIONS) -> Tuple:
    """Canonical hashable tuple of (dim, value) over the chosen ``dims``.

    Dimensions absent from ``key`` are skipped entirely (e.g. an omitted optional
    pattern); a present-but-None value is preserved so distinct setups stay
    distinct. Order follows ``dims`` for determinism.
    """
    out = []
    for dim in dims:
        if dim in key:
            out.append((dim, key[dim]))
    return tuple(out)


def setup_key_str(key: Dict[str, Optional[str]], dims=SETUP_DIMENSIONS) -> str:
    """Stable human-readable key string: ``dim=value`` joined by ``|``.

    A present-but-None value renders as ``dim=none``; an omitted dimension (e.g.
    a missing optional pattern) is left out of the string entirely.
    """
    parts = []
    for dim in dims:
        if dim in key:
            v = key[dim]
            parts.append(f"{dim}={NONE_TOKEN if v is None else v}")
    return "|".join(parts) if parts else "(empty)"


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Single-dimension boundaries.
    checks = [
        (volatility_bucket(0.10), "low"),
        (volatility_bucket(0.149), "low"),
        (volatility_bucket(0.15), "normal"),
        (volatility_bucket(0.29), "normal"),
        (volatility_bucket(0.30), "elevated"),
        (volatility_bucket(0.49), "elevated"),
        (volatility_bucket(0.50), "extreme"),
        (volatility_bucket(None), None),
        (strength_bucket(0), "weak"),
        (strength_bucket(1), "medium"),
        (strength_bucket(2), "medium"),
        (strength_bucket(3), "strong"),
        (strength_bucket(None), None),
        (dte_bucket(7), "0-7"),
        (dte_bucket(8), "8-21"),
        (dte_bucket(21), "8-21"),
        (dte_bucket(22), "22-45"),
        (dte_bucket(45), "22-45"),
        (dte_bucket(46), "45+"),
        (delta_bucket(0.29), "<0.30"),
        (delta_bucket(0.30), "0.30-0.50"),
        (delta_bucket(0.50), "0.30-0.50"),
        (delta_bucket(0.51), ">0.50"),
        (delta_bucket(-0.55), ">0.50"),      # abs()
        (regime_bucket("TRENDING"), "trending"),
        (regime_bucket("garbage"), None),
        (direction_bucket("bullish"), "up"),
        (direction_bucket("flat"), "flat"),
        (pattern_bucket("Hammer"), "hammer"),
        (pattern_bucket("neutral"), None),
        (pattern_bucket(None), None),
    ]
    for got, want in checks:
        if got != want:
            print("FAIL bucket:", got, "!=", want); ok = False

    # Pattern is optional: absent -> not in key; present -> in key.
    k_no = make_setup_key({"regime": "trending", "dte": 30, "entry_delta": 0.4})
    if "pattern" in k_no:
        print("FAIL: pattern should be omitted when absent", k_no); ok = False
    k_yes = make_setup_key({"regime": "trending", "candlestick_pattern": "hammer"})
    if k_yes.get("pattern") != "hammer":
        print("FAIL: pattern should be present when stamped", k_yes); ok = False

    # Legacy row (history metrics) extracts strength/delta, leaves rest None.
    feats = extract_features(
        {"metrics": {"signal_strength": 3, "entry_delta": 0.25}})
    if feats["strength"] != "strong" or feats["delta_bucket"] != "<0.30":
        print("FAIL: legacy metrics extraction", feats); ok = False
    if feats["regime"] is not None:
        print("FAIL: legacy row should have no regime", feats); ok = False

    # episode_store-style raw momentum -> direction.
    feats2 = extract_features(
        {"features_json": {"raw": {"momentum": 0.08, "realized_vol": 0.4}}})
    if feats2["direction"] != "up" or feats2["volatility"] != "elevated":
        print("FAIL: raw-features extraction", feats2); ok = False

    # Key string stability + none-token.
    ks = setup_key_str({"regime": "trending", "volatility": None})
    if ks != "regime=trending|volatility=none":
        print("FAIL: key string", ks); ok = False

    # Tuple drops omitted dims, preserves present-None.
    kt = setup_key_tuple({"regime": "ranging", "direction": None})
    if kt != (("regime", "ranging"), ("direction", None)):
        print("FAIL: key tuple", kt); ok = False

    # Never raises on garbage.
    for junk in (None, 42, "nonsense", [], {"x": object()}):
        try:
            extract_features(junk)  # type: ignore[arg-type]
            make_setup_key(junk if isinstance(junk, dict) else {})
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("feature_buckets self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
