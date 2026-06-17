"""
Shadow recorder: closes the broken RL learning loop.

On each (manual / paper) trade DECISION, this logs what WOULD be learned under a
single `decision_id`: the discrete state key, the modeled round-trip cost, the
read-only RL gate verdict, and an episode-store row. When the trade CLOSES, the
realized NET-of-cost outcome is attached to that SAME `decision_id` and fed to
the RL agent.

This fixes two live-path bugs (see smart_trader):
  * the decision was logged under `order['id']` but the position-closed path
    called `record_trade_outcome(trade, 'closed')` with the default
    `pnl_percent=0`, and the RL hook was guarded by `and pnl_percent`, so the
    update never fired; and
  * P/L was gross, not net of execution cost.

The recorder is READ-ONLY with respect to trading: it never changes what the
strategy trades, and every method is defensive so a failure can never block a
live trade (callers also guard with try/except).
"""

from datetime import datetime
from typing import Dict, Optional

from rl_env import extract_features, state_key, SKIP

FEATURE_VERSION = "1.0.0"


class ShadowRecorder:
    def __init__(
        self,
        store,
        cost_model,
        advisor=None,
        mode: str = "shadow",
        strat_name: str = "smart_trader",
    ):
        self.store = store
        self.cost_model = cost_model
        self.advisor = advisor
        self.mode = mode
        self.strat_name = strat_name

    # ------------------------------------------------------------- features
    def _features_from_analysis(
        self, analysis, as_of, symbol, strat, pdt_remaining=None, day_of_week=None
    ) -> Dict:
        """Build a features dict from a live `analysis`, delegating discretization
        to the SAME rl_env path the backtest uses (so state keys are identical)."""
        disc = extract_features(analysis, pdt_remaining, day_of_week, strat)
        skey = state_key(disc)
        return {
            "feature_version": FEATURE_VERSION,
            "as_of": as_of,
            "symbol": symbol,
            "strat": strat,
            "raw": {
                "spy_change": analysis.get("spy_change"),
                "vix_level": analysis.get("vix_level"),
                "vix_change": analysis.get("vix_change"),
                "gap": analysis.get("gap"),
                "intraday_position": analysis.get("intraday_position"),
                "confidence": analysis.get("confidence"),
                "momentum": analysis.get("momentum"),
                # Underlying price at decision time: the entry reference the
                # SKIP counterfactual resolver compares the forward price against.
                "underlying_price": analysis.get("underlying_price"),
            },
            "discrete": disc,
            "state_key": skey,
        }

    # ------------------------------------------------------------- decision
    def on_decision(
        self,
        *,
        symbol: str,
        underlying: str,
        analysis: Dict,
        quote: Optional[Dict] = None,
        entry_premium: Optional[float] = None,
        qty: int = 1,
        mode: Optional[str] = None,
        strat: Optional[str] = None,
        as_of: Optional[str] = None,
        day_of_week: Optional[int] = None,
        pdt_remaining: Optional[int] = None,
        gate: Optional[Dict] = None,
        gate_overrides: Optional[Dict] = None,
        risk: Optional[Dict] = None,
        features: Optional[Dict] = None,
    ) -> Optional[str]:
        """Record a decision and return its decision_id. Never raises."""
        try:
            strat = strat or self.strat_name
            as_of = as_of or datetime.now().isoformat()
            mode = mode or self.mode

            feats = features or self._features_from_analysis(
                analysis, as_of, symbol, strat, pdt_remaining, day_of_week
            )

            # Modeled round-trip cost from the entry quote (if any).
            modeled_cost = None
            if quote and quote.get("bid") is not None and quote.get("ask") is not None:
                try:
                    modeled_cost = self.cost_model.round_trip_cost(
                        float(quote["bid"]), float(quote["ask"]), qty=qty
                    )
                except Exception:
                    modeled_cost = None

            # Read-only RL gate verdict (informational; never blocks).
            if gate is None and self.advisor is not None:
                try:
                    gate = self.advisor.gate_decision(analysis, overrides=gate_overrides)
                except Exception:
                    gate = None

            raw_dir = analysis.get("direction") or SKIP
            rule_action = raw_dir.upper() if isinstance(raw_dir, str) else SKIP
            chosen_action = rule_action if analysis.get("should_trade", True) else SKIP

            decision_id = self.store.log_decision(
                symbol=symbol,
                underlying=underlying,
                strat=strat,
                features=feats,
                quote=quote,
                modeled_cost=modeled_cost,
                rule_action=rule_action,
                rule_confidence=float(analysis.get("confidence", 0.0) or 0.0),
                gate=gate,
                chosen_action=chosen_action,
                qty=qty,
                mode=mode,
                as_of=as_of,
                risk=risk,
            )

            # Log a PENDING RL experience under the SAME id so on_close can match.
            if self.advisor is not None and chosen_action in ("CALL", "PUT"):
                try:
                    self.advisor.observe_and_log(
                        analysis, decision_id, chosen_action,
                        pdt_remaining=pdt_remaining, day_of_week=day_of_week,
                    )
                except Exception:
                    pass

            return decision_id
        except Exception:
            return None

    # ---------------------------------------------------------------- close
    def on_close(
        self,
        decision_id: Optional[str],
        *,
        entry_bid: Optional[float] = None,
        entry_ask: Optional[float] = None,
        exit_bid: Optional[float] = None,
        exit_ask: Optional[float] = None,
        entry_price: Optional[float] = None,
        exit_price: Optional[float] = None,
        gross_pnl_pct: Optional[float] = None,
        qty: int = 1,
        hold_days: int = 0,
        outcome: str = "closed",
        took_day_trade: bool = False,
        closed_at: Optional[str] = None,
    ) -> Optional[float]:
        """Attach the realized NET outcome to `decision_id` and update the agent.

        No-op (returns None) for legacy trades without a decision_id. Never raises.
        """
        if not decision_id:
            return None

        # Prefer explicit bid/ask; otherwise treat the single price as a mid and
        # let the cost model apply its spread floor + slippage + fees.
        eb = entry_bid if entry_bid is not None else entry_price
        ea = entry_ask if entry_ask is not None else entry_price
        xb = exit_bid if exit_bid is not None else exit_price
        xa = exit_ask if exit_ask is not None else exit_price

        net_pct = None
        net_dollars = None
        if None not in (eb, ea, xb, xa):
            try:
                res = self.cost_model.net_pnl(
                    float(eb), float(ea), float(xb), float(xa),
                    qty=qty, hold_days=hold_days,
                )
                net_pct = res["net_pnl_pct"]
                net_dollars = res["net_pnl_dollars"]
            except Exception:
                net_pct = None

        # If we couldn't model net, still close the loop using gross.
        if net_pct is None:
            net_pct = gross_pnl_pct

        try:
            self.store.record_outcome(
                decision_id,
                fill_price=entry_price,
                exit_price=exit_price,
                gross_pnl_pct=gross_pnl_pct,
                net_pnl_pct=net_pct,
                net_pnl_dollars=net_dollars,
                hold_days=hold_days,
                outcome=outcome,
                closed_at=closed_at or datetime.now().isoformat(),
            )
        except Exception:
            pass

        if self.advisor is not None and net_pct is not None:
            try:
                self.advisor.record_outcome(
                    decision_id, net_pct, took_day_trade=took_day_trade
                )
            except Exception:
                pass

        return net_pct


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import os
    import tempfile
    import uuid

    from episode_store import EpisodeStore
    from cost_model import CostModel, CostConfig
    from rl_wrapper import RLAdvisor

    ok = True
    os.environ["RL_MODE"] = "shadow"

    store = EpisodeStore(":memory:")
    cm = CostModel(CostConfig())
    qf = os.path.join(tempfile.gettempdir(), f"sr_q_{uuid.uuid4().hex}.json")
    ef = os.path.join(tempfile.gettempdir(), f"sr_e_{uuid.uuid4().hex}.json")
    advisor = RLAdvisor(strat_name="spy_1dte", experience_file=ef, qtable_file=qf)
    advisor.agent.reset()

    rec = ShadowRecorder(store, cm, advisor=advisor, strat_name="spy_1dte")

    analysis = {
        "direction": "CALL",
        "confidence": 80.0,
        "spy_change": 0.45,
        "vix_level": 14.0,
        "vix_change": -6.0,
        "gap": 0.4,
        "intraday_position": 0.8,
        "should_trade": True,
    }

    did = rec.on_decision(
        symbol="SPY260108C00475000",
        underlying="SPY",
        analysis=analysis,
        quote={"bid": 1.00, "ask": 1.06, "ts": "2026-01-07T16:00:00"},
        entry_premium=1.03,
        qty=1,
        mode="shadow",
        as_of="2026-01-07T16:00:00",
        day_of_week=2,
    )
    if not did:
        print("FAIL: on_decision returned no id"); ok = False
    if len(store.open_decisions()) != 1:
        print("FAIL: expected one open decision"); ok = False

    skey = state_key(extract_features(analysis, None, 2, "spy_1dte"))
    q_before = advisor.agent.get_q(skey, "CALL")

    # Big favorable exit -> net positive, agent learns.
    net = rec.on_close(
        did,
        entry_bid=1.00, entry_ask=1.06,
        exit_bid=1.60, exit_ask=1.66,
        entry_price=1.03, exit_price=1.63,
        gross_pnl_pct=58.0, hold_days=1, outcome="take_profit",
    )
    if net is None or net <= 0:
        print("FAIL: net P/L should be positive on a big winner", net); ok = False
    if store.open_decisions():
        print("FAIL: decision should no longer be open"); ok = False

    completed = store.completed()
    if len(completed) != 1 or completed[0]["net_pnl_pct"] is None:
        print("FAIL: outcome not attached / net missing"); ok = False
    # NET must be below the gross we passed in (costs always subtract).
    if completed and completed[0]["net_pnl_pct"] >= 58.0:
        print("FAIL: net should be below gross", completed[0]["net_pnl_pct"]); ok = False

    q_after = advisor.agent.get_q(skey, "CALL")
    if not (q_after > q_before):
        print("FAIL: agent Q did not increase after a winning outcome",
              q_before, q_after); ok = False

    # to_rl_experiences should carry the NET pnl.
    exps = store.to_rl_experiences()
    if len(exps) != 1 or exps[0]["action"] != "CALL":
        print("FAIL: episode->experience adaptation wrong", exps); ok = False

    # Legacy trade without a decision_id -> no-op.
    if rec.on_close(None, entry_price=1.0, exit_price=1.2) is not None:
        print("FAIL: on_close(None) should be a no-op"); ok = False

    # Determinism: same analysis/as_of -> identical state key (no skew).
    f1 = rec._features_from_analysis(analysis, "2026-01-07T16:00:00", "SPY", "spy_1dte", None, 2)
    f2 = rec._features_from_analysis(analysis, "2026-01-07T16:00:00", "SPY", "spy_1dte", None, 2)
    if f1["state_key"] != f2["state_key"]:
        print("FAIL: non-deterministic state key"); ok = False

    store.close()
    for fn in (qf, ef):
        try:
            os.remove(fn)
        except OSError:
            pass

    print("shadow_recorder self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
