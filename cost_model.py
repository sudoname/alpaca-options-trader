"""
Realistic execution / cost model.

One place that defines how a fill is priced and what a round trip costs, so that
NET (not gross) P/L is used everywhere: backtests, the episode store, and the
supervised model's target. Options are priced off the bid/ask, not the mid:

    buy  fills at ask + slippage
    sell fills at bid - slippage

and a round trip pays the spread once, slippage twice, per-contract fees twice,
and any overnight carry. Alpaca charges $0 options commission, but OCC / regulatory
fees (~$0.01-0.04/contract) are real, so they default to a small non-zero value.

All parameters are configurable and meant to be CALIBRATED from real fills over
time (backtest -> paper -> live). The decision policy should gate on
cost-adjusted expectancy, i.e. gross expected return minus the round-trip cost.
"""

import os
from dataclasses import dataclass
from typing import Dict

CONTRACT_MULTIPLIER = 100.0  # one option contract = 100 shares


@dataclass
class CostConfig:
    slippage_per_contract: float = 0.02        # $ per contract, each side, vs quoted price
    occ_fee_per_contract: float = 0.02         # OCC/reg fee, $ per contract, each side
    commission_per_contract: float = 0.0       # Alpaca options commission = $0
    overnight_carry_per_contract_per_day: float = 0.0
    min_spread_floor: float = 0.01             # assume at least a 1-cent spread


class CostModel:
    def __init__(self, config: CostConfig = None):
        self.config = config or CostConfig()

    # --------------------------------------------------------------- fills
    def estimate_fill(self, side: str, bid: float, ask: float, qty: int = 1) -> Dict:
        """
        Model the per-contract fill price and the dollar fees for one side.

        side: 'buy' or 'sell'. Buys lift the ask (+slippage); sells hit the bid
        (-slippage). Returns price, total fees, and signed cash flow (negative =
        cash out for a buy, positive = cash in for a sell), all for `qty`.
        """
        cfg = self.config
        bid = max(0.0, float(bid))
        ask = max(0.0, float(ask))
        # enforce a minimal spread so a degenerate bid==ask quote still costs something
        if ask - bid < cfg.min_spread_floor:
            mid = (ask + bid) / 2.0 if (ask or bid) else 0.0
            half = cfg.min_spread_floor / 2.0
            bid, ask = max(0.0, mid - half), mid + half

        side = side.lower()
        if side == "buy":
            price = ask + cfg.slippage_per_contract
        elif side == "sell":
            price = max(0.0, bid - cfg.slippage_per_contract)
        else:
            raise ValueError("side must be 'buy' or 'sell'")

        fees = (cfg.occ_fee_per_contract + cfg.commission_per_contract) * qty
        notional = price * CONTRACT_MULTIPLIER * qty
        cash = -(notional + fees) if side == "buy" else (notional - fees)
        return {"side": side, "price": price, "fees": fees,
                "notional": notional, "cash": cash}

    # ----------------------------------------------------------- round trip
    def round_trip_cost(self, bid: float, ask: float, qty: int = 1, hold_days: int = 0) -> Dict:
        """
        Total cost (in $ and as a % of entry notional) to buy then later sell at
        the same quote: spread + 2*slippage + 2*fees + carry.
        """
        cfg = self.config
        entry = self.estimate_fill("buy", bid, ask, qty)
        exit_ = self.estimate_fill("sell", bid, ask, qty)
        carry = cfg.overnight_carry_per_contract_per_day * max(0, hold_days) * qty
        cost_dollars = (-entry["cash"]) - exit_["cash"] + carry  # cash out minus cash in
        entry_notional = entry["notional"] or 1.0
        return {
            "cost_dollars": cost_dollars,
            "cost_pct": cost_dollars / entry_notional * 100.0,
            "entry_price": entry["price"],
            "exit_price": exit_["price"],
            "carry": carry,
        }

    # --------------------------------------------------------------- net P/L
    def net_pnl(
        self,
        entry_bid: float,
        entry_ask: float,
        exit_bid: float,
        exit_ask: float,
        qty: int = 1,
        hold_days: int = 0,
        side: str = "long",
    ) -> Dict:
        """
        THE net P/L definition consumed by backtests, episodes, and the model.

        For a long option: buy at entry (lift ask), sell at exit (hit bid),
        paying fees both sides plus carry. Percentages are relative to the entry
        cash outlay (so they are directly comparable to gross premium %).
        """
        if side.lower() != "long":
            raise ValueError("only 'long' options are modeled")
        cfg = self.config
        entry = self.estimate_fill("buy", entry_bid, entry_ask, qty)
        exit_ = self.estimate_fill("sell", exit_bid, exit_ask, qty)
        carry = cfg.overnight_carry_per_contract_per_day * max(0, hold_days) * qty

        entry_outlay = -entry["cash"]          # positive cash spent
        proceeds = exit_["cash"]               # positive cash received
        net_dollars = proceeds - entry_outlay - carry
        net_pct = (net_dollars / entry_outlay * 100.0) if entry_outlay else 0.0
        return {
            "net_pnl_dollars": net_dollars,
            "net_pnl_pct": net_pct,
            "entry_outlay": entry_outlay,
            "proceeds": proceeds,
            "entry_price": entry["price"],
            "exit_price": exit_["price"],
            "fees": entry["fees"] + exit_["fees"],
            "carry": carry,
        }

    # ----------------------------------------------------- adjusted expectancy
    def adjusted_expectancy(
        self, gross_pnl_pct: float, bid: float, ask: float, qty: int = 1, hold_days: int = 0
    ) -> float:
        """
        Cost-adjusted expected return %: gross % minus round-trip cost %. The
        decision policy rejects candidates whose adjusted expectancy is <= 0.
        """
        rt = self.round_trip_cost(bid, ask, qty, hold_days)
        return float(gross_pnl_pct) - rt["cost_pct"]


# --------------------------------------------------------------------------- #
# Config from .env (manual parse, matching the rest of the project)
# --------------------------------------------------------------------------- #
def _load_env() -> Dict[str, str]:
    env = {}
    if os.path.exists(".env"):
        try:
            with open(".env", "r") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#") and "=" in line:
                        k, v = line.split("=", 1)
                        env[k.strip()] = v.strip()
        except OSError:
            pass
    return env


def load_cost_config_from_env() -> CostConfig:
    env = _load_env()

    def _f(name, default):
        try:
            return float(env.get(name, str(default)))
        except (TypeError, ValueError):
            return default

    return CostConfig(
        slippage_per_contract=_f("COST_SLIPPAGE_PER_CONTRACT", 0.02),
        occ_fee_per_contract=_f("COST_OCC_FEE_PER_CONTRACT", 0.02),
        commission_per_contract=_f("COST_COMMISSION_PER_CONTRACT", 0.0),
        overnight_carry_per_contract_per_day=_f("COST_CARRY_PER_CONTRACT_PER_DAY", 0.0),
        min_spread_floor=_f("COST_MIN_SPREAD_FLOOR", 0.01),
    )


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cm = CostModel(CostConfig(slippage_per_contract=0.02, occ_fee_per_contract=0.02,
                              commission_per_contract=0.0))

    # Tight market 1.00/1.02. Buy fills 1.04, sell fills 0.98.
    buy = cm.estimate_fill("buy", 1.00, 1.02, qty=1)
    if abs(buy["price"] - 1.04) > 1e-9:
        print("FAIL: buy fill price", buy["price"]); ok = False
    sell = cm.estimate_fill("sell", 1.00, 1.02, qty=1)
    if abs(sell["price"] - 0.98) > 1e-9:
        print("FAIL: sell fill price", sell["price"]); ok = False

    # Flat round trip (same quote) must cost > 0.
    rt = cm.round_trip_cost(1.00, 1.02, qty=1)
    if rt["cost_dollars"] <= 0:
        print("FAIL: round trip should cost money", rt); ok = False

    # Net P/L: a price move that looks like +5% gross can go negative net on a
    # wide spread. Entry quote 1.00/1.10 (buy ~1.12), exit quote 1.13/1.15
    # (sell ~1.11) -> net negative even though mid rose.
    res = cm.net_pnl(1.00, 1.10, 1.13, 1.15, qty=1)
    if res["net_pnl_pct"] >= 0:
        print("FAIL: wide-spread trade should be net negative", res["net_pnl_pct"]); ok = False

    # A large favorable move should be net positive.
    res2 = cm.net_pnl(1.00, 1.02, 1.50, 1.52, qty=1)
    if res2["net_pnl_pct"] <= 0:
        print("FAIL: big winner should be net positive", res2["net_pnl_pct"]); ok = False

    # adjusted_expectancy reduces a gross figure by the round-trip cost.
    adj = cm.adjusted_expectancy(5.0, 1.00, 1.02, qty=1)
    if adj >= 5.0:
        print("FAIL: adjusted expectancy should be below gross", adj); ok = False

    print("cost_model self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
