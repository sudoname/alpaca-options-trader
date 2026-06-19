"""
Offline regression lock for Phase 13B — the ONLY live-touching change.

No creds, no network, no broker. Proves the additive ``compute_oracle_score``
edit is backward compatible:
  1. The DEFAULT call (no version / no learned edge) is byte-identical to the
     pinned v1 arithmetic — the live ranking is unchanged.
  2. version="v1" explicitly equals the default.
  3. version="v2" WITH a learned edge differs predictably (matches blend_v2).
  4. version="v2" with NO learned edge falls back to v1.
  5. A v2 blend failure falls open to v1 (never raises, never returns garbage).

compute_oracle_score is PROPOSAL-ONLY: there is no spread execution, so this
never opens, sizes or alters any trade.
"""

import unittest

from spread_builder import (
    SpreadConfig, SpreadLeg, compute_oracle_score,
    build_bull_put_credit_spread,
    _vol_edge_subscore, _liquidity_subscore, _risk_reward_subscore,
    _cost_subscore, _trend_subscore, _clamp01,
)
import oracle_score_v2 as v2

CFG = SpreadConfig(enabled=True, max_loss_limit=1000.0, max_leg_spread_pct=50.0,
                   min_open_interest=0.0, min_volume=0.0)


def _leg(action, otype, strike, bid, ask, oi=500, vol=500):
    return SpreadLeg(action=action, option_type=otype, strike=strike,
                     bid=bid, ask=ask, open_interest=oi, volume=vol)


def _proposal():
    return build_bull_put_credit_spread(
        _leg("sell", "put", 100, 1.20, 1.25),
        _leg("buy", "put", 95, 0.40, 0.45), CFG, "SPY")


def _pinned_v1(proposal, vol_state, trend):
    """Re-implements the pinned v1 arithmetic independently of the module."""
    vol_edge = _vol_edge_subscore(proposal.strategy_name, vol_state)
    liquidity = _liquidity_subscore(proposal.legs)
    risk_reward = _risk_reward_subscore(proposal)
    cost = _cost_subscore(proposal.legs, CFG)
    trend_align = _trend_subscore(proposal.strategy_name, trend)
    blended = (0.25 * vol_edge + 0.20 * liquidity + 0.25 * risk_reward +
               0.15 * cost + 0.15 * trend_align)
    return round(_clamp01(blended) * 100.0, 1)


class TestDefaultUnchanged(unittest.TestCase):
    def test_default_equals_pinned_v1(self):
        p = _proposal()
        for vol_state, trend in [(None, None), ("underpriced", "up"),
                                 ("overpriced", "down"), ("fair", "flat")]:
            self.assertEqual(compute_oracle_score(p, CFG, vol_state, trend),
                             _pinned_v1(p, vol_state, trend))

    def test_explicit_v1_equals_default(self):
        p = _proposal()
        self.assertEqual(
            compute_oracle_score(p, CFG, "underpriced", "up"),
            compute_oracle_score(p, CFG, "underpriced", "up", version="v1"))

    def test_no_trade_scores_zero(self):
        # An empty/no-trade proposal scores 0.0 under both versions.
        p = _proposal()
        p.legs = []
        self.assertEqual(compute_oracle_score(p, CFG), 0.0)
        self.assertEqual(
            compute_oracle_score(p, CFG, version="v2", learned_edge_score=0.9),
            0.0)


class TestV2Branch(unittest.TestCase):
    def test_v2_matches_blend_v2(self):
        p = _proposal()
        vol_state, trend = "underpriced", "up"
        subs = {
            "vol_edge": _vol_edge_subscore(p.strategy_name, vol_state),
            "liquidity": _liquidity_subscore(p.legs),
            "risk_reward": _risk_reward_subscore(p),
            "cost": _cost_subscore(p.legs, CFG),
            "trend_align": _trend_subscore(p.strategy_name, trend),
        }
        expected = v2.blend_v2(subs, 0.8)
        got = compute_oracle_score(p, CFG, vol_state, trend,
                                   version="v2", learned_edge_score=0.8)
        self.assertEqual(got, expected)

    def test_v2_differs_from_v1_with_edge(self):
        p = _proposal()
        v1 = compute_oracle_score(p, CFG, "underpriced", "up")
        # A strong edge should lift the score above v1 (sub-scores < perfect).
        v2_high = compute_oracle_score(p, CFG, "underpriced", "up",
                                       version="v2", learned_edge_score=1.0)
        self.assertNotEqual(v1, v2_high)

    def test_v2_without_edge_falls_back_to_v1(self):
        p = _proposal()
        self.assertEqual(
            compute_oracle_score(p, CFG, "underpriced", "up", version="v2"),
            compute_oracle_score(p, CFG, "underpriced", "up"))

    def test_v2_blend_failure_falls_open_to_v1(self):
        p = _proposal()
        v1 = compute_oracle_score(p, CFG, "underpriced", "up")
        import unittest.mock as mock
        with mock.patch.object(v2, "blend_v2", side_effect=RuntimeError("boom")):
            got = compute_oracle_score(p, CFG, "underpriced", "up",
                                       version="v2", learned_edge_score=0.8)
        self.assertEqual(got, v1)


if __name__ == "__main__":
    unittest.main()
