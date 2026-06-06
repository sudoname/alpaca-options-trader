"""
Cross-cutting integrity harness (no creds, no network).

These are the guardrails that keep the measurement honest. They are deliberately
independent of any single module's own self-test and assert the properties the
whole pipeline depends on:

  1. POINT-IN-TIME AUDIT — no datum stamped after `as_of` ever escapes a
     MarketView, even when the underlying series contains future bars.
  2. REPLAY-EQUALS-LIVE — features computed from a "historical" view and from a
     "live" view pinned to the SAME `as_of` are identical, so there is zero
     train/serve skew. The live view is fed extra future bars to prove the
     as_of filter (not the data) drives equality.
  3. FUTURE-SHIFT — shifting labels forward one step materially degrades the
     walk-forward OOS expectancy (a model with real edge must rely on alignment).
  4. HONESTY REPORT — a single net-of-cost summary (expectancy, win rate, max
     drawdown, coverage, effective sample size) for any episode set.

`python tests_integrity.py` exits non-zero if ANY check fails.
"""

from datetime import datetime, timedelta
from typing import Dict, List, Optional

from market_view import HistoricalMarketView, LiveMarketView, make_bar, Bar
from features import compute_features
from walk_forward import walk_forward_eval, shift_labels_forward, _max_drawdown
from model import ExpectedReturnModel


# --------------------------------------------------------------------------- #
# A network-free "live-like" view: inherits the base point-in-time filter from
# MarketView but, like LiveMarketView, is conceptually a live snapshot. We feed
# it fixture candidates (including FUTURE bars) without touching the network.
# --------------------------------------------------------------------------- #
class StubLiveView(LiveMarketView):
    def __init__(self, as_of, *, daily, vix_series=None, **kw):
        super().__init__(headers={}, as_of=as_of, **kw)
        self._stub_daily = daily
        self._stub_vix = vix_series or {}

    def _candidate_daily_bars(self, symbol):
        return list(self._stub_daily.get(symbol, []))

    def _candidate_vix(self, symbol):
        return list(self._stub_vix.get(symbol, []))


# --------------------------------------------------------------------------- #
# Fixtures
# --------------------------------------------------------------------------- #
def _spy_fixture() -> Dict[str, List[Bar]]:
    return {
        "SPY": [
            make_bar("2026-01-02", 470, 472, 469, 471, 1e6),
            make_bar("2026-01-05", 471, 474, 470, 473, 1e6),
            make_bar("2026-01-06", 472, 476, 471, 475, 1e6),
        ]
    }


def _vix_fixture() -> Dict[str, List[Bar]]:
    return {
        "^VIX": [
            make_bar("2026-01-05", 16, 16, 16, 16, 0),
            make_bar("2026-01-06", 15, 15, 15, 15, 0),
        ]
    }


# --------------------------------------------------------------------------- #
# 1. Point-in-time audit
# --------------------------------------------------------------------------- #
def test_point_in_time_audit() -> bool:
    as_of = datetime(2026, 1, 6, 16, 0)
    daily = _spy_fixture()
    vix = _vix_fixture()
    # Inject FUTURE bars that must never be returned.
    daily["SPY"].append(make_bar("2026-01-07", 475, 480, 474, 479, 1e6))
    vix["^VIX"].append(make_bar("2026-01-07", 14, 14, 14, 14, 0))

    mv = HistoricalMarketView(as_of, daily=daily, vix_series=vix)
    feats = compute_features(as_of, mv, symbol="SPY", strat_name="spy_1dte")

    leaks = [rec for rec in mv.audit if rec["ts"] > as_of]
    if leaks:
        print("FAIL[1]: point-in-time leak", leaks); return False
    # The future close (479) must not have driven spy_change.
    if abs(feats["raw"]["spy_change"] - (3 / 472 * 100)) > 1e-6:
        print("FAIL[1]: future bar contaminated features", feats["raw"]); return False
    print("PASS[1]: point-in-time audit (no ts > as_of; future bar excluded)")
    return True


# --------------------------------------------------------------------------- #
# 2. Replay-equals-live
# --------------------------------------------------------------------------- #
def test_replay_equals_live() -> bool:
    as_of = datetime(2026, 1, 6, 16, 0)

    hist = HistoricalMarketView(as_of, daily=_spy_fixture(), vix_series=_vix_fixture())

    # The "live" view sees the same history PLUS future bars (as a live feed
    # would, since 'now' data exists). The as_of filter must erase the difference.
    live_daily = _spy_fixture()
    live_daily["SPY"].append(make_bar("2026-01-07", 475, 480, 474, 479, 1e6))
    live_vix = _vix_fixture()
    live_vix["^VIX"].append(make_bar("2026-01-07", 14, 14, 14, 14, 0))
    live = StubLiveView(as_of, daily=live_daily, vix_series=live_vix)

    fh = compute_features(as_of, hist, symbol="SPY", strat_name="spy_1dte",
                          extra={"confidence": 80.0})
    fl = compute_features(as_of, live, symbol="SPY", strat_name="spy_1dte",
                          extra={"confidence": 80.0})

    if fh["state_key"] != fl["state_key"]:
        print("FAIL[2]: state_key differs historical vs live",
              fh["state_key"], fl["state_key"]); return False
    if fh["raw"] != fl["raw"]:
        print("FAIL[2]: raw features differ historical vs live",
              fh["raw"], fl["raw"]); return False
    print("PASS[2]: replay-equals-live (identical features at same as_of)")
    return True


# --------------------------------------------------------------------------- #
# 3. Future-shift degrades OOS
# --------------------------------------------------------------------------- #
def _nonperiodic_episodes(n: int = 60, seed: int = 99) -> List[Dict]:
    import random
    rng = random.Random(seed)
    day = datetime(2026, 1, 1)
    out = []
    for i in range(n):
        skey = "good" if rng.random() < 0.5 else "bad"
        out.append({
            "as_of": (day + timedelta(days=i)).isoformat(),
            "state_key": skey,
            "net_pnl_pct": 20.0 if skey == "good" else -15.0,
            "feature_version": "1.0.0",
        })
    return out


def test_future_shift_degrades() -> bool:
    eps = _nonperiodic_episodes()
    factory = lambda: ExpectedReturnModel(min_total=5, min_state_samples=2)

    aligned = walk_forward_eval(eps, model_factory=factory)
    shifted = walk_forward_eval(shift_labels_forward(eps), model_factory=factory)
    a = aligned["oos_aggregate"]["expectancy_pct"]
    s = shifted["oos_aggregate"]["expectancy_pct"]
    if not (s < a):
        print("FAIL[3]: shifted OOS not worse than aligned", s, a); return False
    print(f"PASS[3]: future-shift degrades OOS (aligned={a:.2f} > shifted={s:.2f})")
    return True


# --------------------------------------------------------------------------- #
# 4. Honesty report
# --------------------------------------------------------------------------- #
def honesty_report(episodes: List[Dict]) -> Dict:
    """Net-of-cost summary over a set of completed episodes (time-ordered)."""
    ordered = sorted(episodes, key=lambda e: (e.get("as_of") or e.get("closed_at") or ""))
    taken = [float(e["net_pnl_pct"]) for e in ordered if e.get("net_pnl_pct") is not None]
    n = len(taken)
    covered = sum(1 for e in ordered if e.get("state_key"))
    wins = sum(1 for p in taken if p > 0)
    return {
        "n_episodes": len(ordered),
        "expectancy_pct": (sum(taken) / n) if n else 0.0,
        "win_rate": (wins / n) if n else 0.0,
        "max_drawdown": _max_drawdown(taken),
        "coverage": (covered / len(ordered)) if ordered else 0.0,
        "effective_sample_size": float(n),
    }


def test_honesty_report() -> bool:
    # 3 wins (+10) and 2 losses (-20): expectancy = (30-40)/5 = -2.0.
    eps = []
    day = datetime(2026, 2, 1)
    pnls = [10.0, -20.0, 10.0, -20.0, 10.0]
    for i, p in enumerate(pnls):
        eps.append({"as_of": (day + timedelta(days=i)).isoformat(),
                    "state_key": "s", "net_pnl_pct": p})
    rep = honesty_report(eps)
    if abs(rep["expectancy_pct"] - (-2.0)) > 1e-9:
        print("FAIL[4]: expectancy wrong", rep); return False
    if abs(rep["win_rate"] - 0.6) > 1e-9:
        print("FAIL[4]: win_rate wrong", rep); return False
    # Worst peak-to-trough: equity path 10,-10,0,-20,-10 -> peak 10, trough -20 -> 30.
    if abs(rep["max_drawdown"] - 30.0) > 1e-9:
        print("FAIL[4]: max_drawdown wrong", rep); return False
    if abs(rep["coverage"] - 1.0) > 1e-9 or rep["effective_sample_size"] != 5.0:
        print("FAIL[4]: coverage/ess wrong", rep); return False
    print("PASS[4]: honesty report (expectancy/win/mdd/coverage/ess)")
    return True


# --------------------------------------------------------------------------- #
# Runner
# --------------------------------------------------------------------------- #
def main() -> int:
    print("=" * 50)
    print("INTEGRITY HARNESS")
    print("=" * 50)
    checks = [
        test_point_in_time_audit,
        test_replay_equals_live,
        test_future_shift_degrades,
        test_honesty_report,
    ]
    results = []
    for c in checks:
        try:
            results.append(bool(c()))
        except Exception as e:
            print(f"FAIL: {c.__name__} raised {e!r}")
            results.append(False)
    ok = all(results)
    print("=" * 50)
    print("RESULT:", "PASS" if ok else "FAIL", f"({sum(results)}/{len(results)})")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(main())
