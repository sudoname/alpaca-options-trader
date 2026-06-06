"""
Single, shared, no-lookahead feature path.

`compute_features(as_of, market_view, ...)` is the ONLY place features are
derived. It takes a point-in-time `MarketView` and never calls
`datetime.now()` itself, so the exact same code produces identical features in
a backtest and live (train-serve skew = 0). The continuous ("raw") block is
computed from completed, known data; the discrete block delegates to the
existing `rl_env` discretization so the Q-table keying is unchanged.

Decision timing convention used here: features describe the most recent
COMPLETED daily bar known at `as_of` (i.e. decide at/after that bar's close;
the trade outcome belongs to a later bar). Because the bar is complete, using
its own open/high/low/close is not lookahead.
"""

from datetime import datetime
from typing import Dict, Optional

from rl_env import extract_features, state_key

FEATURE_VERSION = "1.0.0"


def compute_features(
    as_of: datetime,
    market_view,
    *,
    symbol: str = "SPY",
    strat_name: str = "generic",
    lookback: int = 30,
    pdt_remaining: Optional[int] = None,
    day_of_week: Optional[int] = None,
    extra: Optional[Dict] = None,
) -> Dict:
    """
    Build features for `symbol` as of `as_of` using only data the
    `market_view` says was knowable by then.

    Returns:
      {
        "feature_version": str,
        "as_of": ISO str,
        "symbol": str,
        "strat": str,
        "raw": {continuous inputs},
        "discrete": {rl_env discretized features},
        "state_key": str,
        "warmup": bool,        # True when there was not enough history
      }
    """
    extra = extra or {}
    bars = market_view.daily_bars(symbol, lookback)

    raw: Dict[str, float] = {}
    warmup = len(bars) < 2

    if not warmup:
        today = bars[-1]
        prev = bars[-2]
        if today.o:
            raw["spy_change"] = (today.c - today.o) / today.o * 100.0
        if prev.c:
            raw["gap"] = (today.o - prev.c) / prev.c * 100.0
        rng = today.h - today.l
        if rng > 0:
            raw["intraday_position"] = (today.c - today.l) / rng

    vix_level = market_view.vix(extra.get("vix_symbol", "^VIX"))
    if vix_level is not None:
        raw["vix_level"] = vix_level
        vbars = market_view.vix_bars(extra.get("vix_symbol", "^VIX"), 2)
        if len(vbars) >= 2 and vbars[-2].c:
            raw["vix_change"] = (vbars[-1].c - vbars[-2].c) / vbars[-2].c * 100.0

    # Confidence is the rule strategy's own signal; it is an input, not derived.
    if "confidence" in extra:
        raw["confidence"] = extra["confidence"]

    # Discrete block: hand the raw inputs to the existing rl_env discretizer so
    # the discrete state (and therefore Q-table keys) stays identical.
    analysis_like = {
        "spy_change": raw.get("spy_change", 0.0),
        "vix_level": raw.get("vix_level", 15.0),
        "vix_change": raw.get("vix_change", 0.0),
        "gap": raw.get("gap", 0.0),
        "intraday_position": raw.get("intraday_position", 0.5),
        "confidence": raw.get("confidence", 0.0),
    }
    discrete = extract_features(
        analysis_like,
        pdt_remaining=pdt_remaining,
        day_of_week=day_of_week,
        strat_name=strat_name,
    )

    return {
        "feature_version": FEATURE_VERSION,
        "as_of": as_of.isoformat(),
        "symbol": symbol,
        "strat": strat_name,
        "raw": raw,
        "discrete": discrete,
        "state_key": state_key(discrete),
        "warmup": warmup,
    }


def feature_state_key(features: Dict) -> str:
    """State key for a features dict (delegates to rl_env on the discrete block)."""
    return features.get("state_key") or state_key(features.get("discrete", {}))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from market_view import HistoricalMarketView, make_bar

    ok = True

    daily = {
        "SPY": [
            make_bar("2026-01-02", 470, 472, 469, 471, 1e6),
            make_bar("2026-01-05", 471, 474, 470, 473, 1e6),
            make_bar("2026-01-06", 472, 476, 471, 475, 1e6),
        ]
    }
    vix = {
        "^VIX": [
            make_bar("2026-01-05", 16, 16, 16, 16, 0),
            make_bar("2026-01-06", 15, 15, 15, 15, 0),
        ]
    }
    as_of = datetime(2026, 1, 6, 16, 0)
    mv = HistoricalMarketView(as_of, daily=daily, vix_series=vix)

    f = compute_features(as_of, mv, symbol="SPY", strat_name="spy_1dte",
                         extra={"confidence": 80.0})

    if f["feature_version"] != FEATURE_VERSION:
        print("FAIL: missing/incorrect feature_version"); ok = False
    if f["warmup"]:
        print("FAIL: should not be warmup with 3 bars"); ok = False

    # spy_change = (475-472)/472*100 ~= 0.6356
    if abs(f["raw"]["spy_change"] - (3 / 472 * 100)) > 1e-6:
        print("FAIL: spy_change wrong", f["raw"].get("spy_change")); ok = False
    # gap = (472-473)/473*100 ~= -0.2114
    if abs(f["raw"]["gap"] - (-1 / 473 * 100)) > 1e-6:
        print("FAIL: gap wrong", f["raw"].get("gap")); ok = False
    # intraday_position = (475-471)/(476-471) = 0.8
    if abs(f["raw"]["intraday_position"] - 0.8) > 1e-9:
        print("FAIL: intraday_position wrong", f["raw"].get("intraday_position")); ok = False
    # vix_change = (15-16)/16*100 = -6.25
    if abs(f["raw"]["vix_change"] - (-6.25)) > 1e-9:
        print("FAIL: vix_change wrong", f["raw"].get("vix_change")); ok = False

    # determinism: same inputs -> same key.
    mv2 = HistoricalMarketView(as_of, daily=daily, vix_series=vix)
    f2 = compute_features(as_of, mv2, symbol="SPY", strat_name="spy_1dte",
                          extra={"confidence": 80.0})
    if f["state_key"] != f2["state_key"]:
        print("FAIL: non-deterministic state_key"); ok = False

    # warmup with a single bar.
    mv_warm = HistoricalMarketView(
        datetime(2026, 1, 2, 16, 0),
        daily={"SPY": [make_bar("2026-01-02", 470, 472, 469, 471, 1e6)]},
    )
    fw = compute_features(datetime(2026, 1, 2, 16, 0), mv_warm, symbol="SPY")
    if not fw["warmup"]:
        print("FAIL: single bar should be warmup"); ok = False

    print("features self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
