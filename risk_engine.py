"""
Risk engine + kill-switch — hard limits that live OUTSIDE the learner.

Deliberate asymmetry vs the RL gate:
  * the RL gate fails OPEN (a broken/empty model never blocks a trade), because
    it is advisory and must never silently strangle the strategy;
  * this risk engine fails CLOSED (any exception, missing input, or breached
    limit -> allowed=False), because it is the last line of capital protection.

It enforces, all net-of-nothing/raw-dollar caps:
  * per-trade budget        (max_budget_per_trade)
  * daily realized loss      (daily_loss_limit)
  * max concurrent positions (max_concurrent)
  * PDT day-trade headroom   (pdt_remaining, supplied by pdt_tracker)
and a global kill switch on the day's realized P/L.

In SHADOW mode the verdict is only RECORDED (ShadowRecorder.on_decision passes it
through to the episode's `risk_json` column); it never blocks a manual trade.
This module computes the verdict; it does not place or cancel orders.
"""

import math
from dataclasses import dataclass
from typing import Dict, List, Optional


# --------------------------------------------------------------------------- #
# Limits
# --------------------------------------------------------------------------- #
@dataclass
class RiskLimits:
    max_budget_per_trade: float = 500.0   # $ per single position
    daily_loss_limit: float = 300.0       # $ realized loss that blocks new trades
    max_concurrent: int = 3               # open positions allowed at once
    min_pdt_remaining: int = 1            # required day-trade headroom for a day trade
    kill_switch_loss: float = 500.0       # $ realized daily loss that trips the switch
    # Concentration cap: max open positions allowed on a SINGLE underlying.
    # Default is intentionally high (no-op) so existing behavior is unchanged
    # until an operator opts in via MAX_POSITIONS_PER_UNDERLYING. A value <= 0
    # also means "no limit" (disabled).
    max_per_underlying: int = 1000


def _env_float(env: Dict[str, str], key: str, default: float) -> float:
    try:
        return float(env.get(key, default))
    except (TypeError, ValueError):
        return default


def _env_int(env: Dict[str, str], key: str, default: int) -> int:
    try:
        return int(float(env.get(key, default)))
    except (TypeError, ValueError):
        return default


def load_risk_limits_from_env(path: str = ".env") -> RiskLimits:
    """Resolve limits via the shared loader (shell env > .env > default).

    Phase 4.5: ``ConfigLoader`` is a drop-in for the parsed-``.env`` dict this
    used to build, so ``_env_float``/``_env_int`` work unchanged while a shell
    ``KEY=... python ...`` now overrides ``.env``.
    """
    from config_loader import ConfigLoader
    env = ConfigLoader(path)
    return RiskLimits(
        max_budget_per_trade=_env_float(env, "MAX_BUDGET_PER_TRADE", 500.0),
        daily_loss_limit=_env_float(env, "DAILY_LOSS_LIMIT", 300.0),
        max_concurrent=_env_int(env, "MAX_CONCURRENT_POSITIONS", 3),
        min_pdt_remaining=_env_int(env, "MIN_PDT_REMAINING", 1),
        kill_switch_loss=_env_float(env, "KILL_SWITCH_LOSS", 500.0),
        max_per_underlying=_env_int(env, "MAX_POSITIONS_PER_UNDERLYING", 1000),
    )


# --------------------------------------------------------------------------- #
# Engine
# --------------------------------------------------------------------------- #
class RiskEngine:
    def __init__(self, limits: Optional[RiskLimits] = None):
        self.limits = limits or RiskLimits()

    def kill_switch_tripped(self, realized_pnl_today: Optional[float]) -> bool:
        """True when the day's realized loss has reached the kill-switch level.

        Fails CLOSED: a missing/garbage input is treated as tripped.
        """
        try:
            val = float(realized_pnl_today)
            if math.isnan(val):  # a NaN P/L is unsafe -> treat as tripped
                return True
            return val <= -abs(self.limits.kill_switch_loss)
        except (TypeError, ValueError):
            return True

    def check(
        self,
        *,
        trade_cost: Optional[float] = None,
        realized_pnl_today: Optional[float] = None,
        open_positions: Optional[int] = None,
        pdt_remaining: Optional[int] = None,
        may_day_trade: bool = False,
        positions_for_underlying: Optional[int] = None,
    ) -> Dict:
        """Return {allowed, reason, breaches}. FAIL-CLOSED on any problem.

        Required inputs: trade_cost, realized_pnl_today, open_positions. A None
        for any of these (or any exception) yields allowed=False so a missing
        signal can never be read as permission.

        Optional input: positions_for_underlying = how many positions are
        already open on the underlying this trade targets. The per-underlying
        concentration cap is only evaluated when this is supplied AND the limit
        is active (max_per_underlying > 0). When the count is omitted the cap is
        a no-op, so callers that don't pass it keep their existing behavior.
        """
        try:
            breaches: List[str] = []
            lim = self.limits

            # Hard requirement: these must be present and numeric.
            if trade_cost is None or realized_pnl_today is None or open_positions is None:
                return {
                    "allowed": False,
                    "reason": "missing_required_input",
                    "breaches": ["missing_required_input"],
                }

            trade_cost = float(trade_cost)
            realized_pnl_today = float(realized_pnl_today)
            open_positions = int(open_positions)

            if self.kill_switch_tripped(realized_pnl_today):
                breaches.append("kill_switch")

            if realized_pnl_today <= -abs(lim.daily_loss_limit):
                breaches.append("daily_loss_limit")

            if trade_cost > lim.max_budget_per_trade:
                breaches.append("over_budget")

            if trade_cost <= 0:
                breaches.append("nonpositive_cost")

            if open_positions >= lim.max_concurrent:
                breaches.append("max_concurrent")

            # Per-underlying concentration cap (opt-in). Only evaluated when the
            # caller supplies the current per-underlying count AND a positive
            # limit is configured; otherwise it's a no-op (default behavior).
            if positions_for_underlying is not None and lim.max_per_underlying > 0:
                if int(positions_for_underlying) >= lim.max_per_underlying:
                    breaches.append("max_per_underlying")

            if may_day_trade:
                if pdt_remaining is None:
                    breaches.append("pdt_unknown")
                elif int(pdt_remaining) < lim.min_pdt_remaining:
                    breaches.append("pdt_block")

            allowed = not breaches
            return {
                "allowed": allowed,
                "reason": "; ".join(breaches) if breaches else "ok",
                "breaches": breaches,
            }
        except Exception as e:  # fail closed on anything unexpected
            return {
                "allowed": False,
                "reason": f"exception:{type(e).__name__}",
                "breaches": ["exception"],
            }


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    eng = RiskEngine(RiskLimits(
        max_budget_per_trade=500.0,
        daily_loss_limit=300.0,
        max_concurrent=3,
        min_pdt_remaining=1,
        kill_switch_loss=500.0,
    ))

    # Clean trade -> allowed.
    r = eng.check(trade_cost=200.0, realized_pnl_today=-50.0, open_positions=1,
                  pdt_remaining=2, may_day_trade=True)
    if not r["allowed"] or r["reason"] != "ok":
        print("FAIL: clean trade should be allowed", r); ok = False

    # Over budget -> blocked.
    r = eng.check(trade_cost=600.0, realized_pnl_today=0.0, open_positions=0)
    if r["allowed"] or "over_budget" not in r["breaches"]:
        print("FAIL: over-budget should be blocked", r); ok = False

    # Daily loss limit -> blocked.
    r = eng.check(trade_cost=100.0, realized_pnl_today=-350.0, open_positions=0)
    if r["allowed"] or "daily_loss_limit" not in r["breaches"]:
        print("FAIL: daily loss limit should block", r); ok = False

    # Kill switch -> tripped + blocked.
    if not eng.kill_switch_tripped(-500.0):
        print("FAIL: kill switch should trip at -500"); ok = False
    r = eng.check(trade_cost=100.0, realized_pnl_today=-600.0, open_positions=0)
    if r["allowed"] or "kill_switch" not in r["breaches"]:
        print("FAIL: kill switch should block", r); ok = False

    # Max concurrent -> blocked.
    r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=3)
    if r["allowed"] or "max_concurrent" not in r["breaches"]:
        print("FAIL: max concurrent should block", r); ok = False

    # Per-underlying cap is a no-op by default (high default limit), even when
    # a count is supplied.
    r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                  positions_for_underlying=50)
    if not r["allowed"]:
        print("FAIL: default per-underlying cap should be no-op", r); ok = False

    # When enabled, the cap blocks once same-symbol exposure reaches the limit.
    eng_cap = RiskEngine(RiskLimits(max_per_underlying=2))
    r = eng_cap.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=2)
    if r["allowed"] or "max_per_underlying" not in r["breaches"]:
        print("FAIL: per-underlying cap should block at the limit", r); ok = False
    # ...but still allows when under the cap.
    r = eng_cap.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1,
                      positions_for_underlying=1)
    if not r["allowed"]:
        print("FAIL: per-underlying cap should allow under the limit", r); ok = False
    # ...and omitting the count leaves the cap a no-op even when configured.
    r = eng_cap.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=1)
    if not r["allowed"]:
        print("FAIL: per-underlying cap should be skipped when count omitted", r); ok = False

    # PDT headroom -> blocked when it's a day trade with no remaining.
    r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=0,
                  pdt_remaining=0, may_day_trade=True)
    if r["allowed"] or "pdt_block" not in r["breaches"]:
        print("FAIL: PDT block should fire", r); ok = False
    # ...but a non-day-trade with no PDT headroom is fine.
    r = eng.check(trade_cost=100.0, realized_pnl_today=0.0, open_positions=0,
                  pdt_remaining=0, may_day_trade=False)
    if not r["allowed"]:
        print("FAIL: non-day-trade should ignore PDT", r); ok = False

    # FAIL-CLOSED: missing required input -> blocked.
    r = eng.check(trade_cost=None, realized_pnl_today=0.0, open_positions=0)
    if r["allowed"] or "missing_required_input" not in r["breaches"]:
        print("FAIL: missing input should fail closed", r); ok = False

    # FAIL-CLOSED: garbage input -> blocked, no raise.
    r = eng.check(trade_cost="oops", realized_pnl_today=0.0, open_positions=0)
    if r["allowed"]:
        print("FAIL: garbage input should fail closed", r); ok = False
    # Kill switch on garbage -> treated as tripped.
    if not eng.kill_switch_tripped("nan"):
        print("FAIL: garbage kill-switch input should fail closed"); ok = False

    # Multiple simultaneous breaches are all reported.
    r = eng.check(trade_cost=600.0, realized_pnl_today=-600.0, open_positions=5)
    for b in ("over_budget", "daily_loss_limit", "kill_switch", "max_concurrent"):
        if b not in r["breaches"]:
            print(f"FAIL: expected breach '{b}'", r); ok = False

    print("risk_engine self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
