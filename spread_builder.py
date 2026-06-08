"""
Defined-risk spread PROPOSAL builder (Phase 6A) — simulation only, never trades.

This module is intentionally PURE and side-effect free: it builds proposal
objects from already-resolved leg quotes and applies hard safety rejections. It
NEVER fetches data, places orders, or mutates account state. The orchestration
that fetches a chain + quotes and calls these builders lives in
`smart_trader.propose_spread`; the Telegram `/spread_proposal` command surfaces
the result. Both are gated behind `USE_SPREAD_PROPOSALS` (default OFF).

Supported strategy names (Requirement 1):
  bullish_put_credit_spread, bearish_call_credit_spread,
  debit_call_spread, debit_put_spread, iron_condor, no_trade.

A proposal carries: legs, max_profit, max_loss, breakeven, net_credit_or_debit,
width, estimated_probability, strategy_name, reason (Requirement 3). All dollar
figures are per 1-contract structure (x100 multiplier).

Safety (Requirement 5): reject undefined-risk structures (any unprotected short
leg), missing bid/ask, wide per-leg quotes, max_loss above the risk limit, and
illiquid legs. On any rejection the builder returns a `no_trade` proposal whose
`reason` names the failed check — it NEVER returns an order or raises.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

# Per-contract share multiplier for US equity options.
CONTRACT_MULTIPLIER = 100.0

# --------------------------------------------------------------------------- #
# Strategy name constants (Requirement 1)
# --------------------------------------------------------------------------- #
BULLISH_PUT_CREDIT_SPREAD = "bullish_put_credit_spread"
BEARISH_CALL_CREDIT_SPREAD = "bearish_call_credit_spread"
DEBIT_CALL_SPREAD = "debit_call_spread"
DEBIT_PUT_SPREAD = "debit_put_spread"
IRON_CONDOR = "iron_condor"
NO_TRADE = "no_trade"

STRATEGY_NAMES = {
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
}

CREDIT_STRATEGIES = {BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD, IRON_CONDOR}
DEBIT_STRATEGIES = {DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class SpreadConfig:
    enabled: bool = False                 # USE_SPREAD_PROPOSALS
    max_loss_limit: float = 500.0         # $ ceiling for a structure's max loss
    max_leg_spread_pct: float = 15.0      # reject a leg whose bid/ask is wider, %
    min_open_interest: float = 0.0        # reject a leg below this OI (0 = off)
    min_volume: float = 0.0               # reject a leg below this volume (0 = off)
    iv_overpriced_ratio: float = 1.20     # IV/HV >= this -> "overpriced"
    iv_underpriced_ratio: float = 0.80    # IV/HV <= this -> "underpriced"
    wing_width: float = 5.0               # target strike width (orchestration hint)
    min_trend_momentum: float = 0.01      # |momentum| below this -> trend neutral

    @staticmethod
    def from_env(path: str = ".env") -> "SpreadConfig":
        from config_loader import ConfigLoader
        env = ConfigLoader(path)
        # The risk ceiling defaults to the existing per-trade budget so spread
        # max-loss is held to the same dollar cap as a long single-leg trade,
        # with an optional SPREAD_MAX_LOSS override.
        budget = env.get_float("MAX_BUDGET_PER_TRADE", 500.0)
        return SpreadConfig(
            enabled=env.get_bool("USE_SPREAD_PROPOSALS", False),
            max_loss_limit=env.get_float("SPREAD_MAX_LOSS", budget),
            max_leg_spread_pct=env.get_float("SPREAD_MAX_LEG_SPREAD_PCT", 15.0),
            min_open_interest=env.get_float("MIN_OPTION_OPEN_INTEREST", 0.0),
            min_volume=env.get_float("MIN_OPTION_VOLUME", 0.0),
            iv_overpriced_ratio=env.get_float("SPREAD_IV_OVERPRICED_RATIO", 1.20),
            iv_underpriced_ratio=env.get_float("SPREAD_IV_UNDERPRICED_RATIO", 0.80),
            wing_width=env.get_float("SPREAD_WING_WIDTH", 5.0),
            min_trend_momentum=env.get_float("SPREAD_MIN_TREND_MOMENTUM", 0.01),
        )


# --------------------------------------------------------------------------- #
# Leg + proposal data
# --------------------------------------------------------------------------- #
@dataclass
class SpreadLeg:
    action: str                 # 'buy' | 'sell'
    option_type: str            # 'call' | 'put'
    strike: float
    bid: Optional[float] = None
    ask: Optional[float] = None
    symbol: str = ""
    expiration: str = ""
    open_interest: Optional[float] = None
    volume: Optional[float] = None

    def as_dict(self) -> Dict:
        return {
            "action": self.action, "type": self.option_type,
            "strike": self.strike, "bid": self.bid, "ask": self.ask,
            "symbol": self.symbol, "expiration": self.expiration,
            "open_interest": self.open_interest, "volume": self.volume,
        }

    def label(self) -> str:
        return f"{self.action.upper()} {self.option_type.upper()} {self.strike:g}"


@dataclass
class SpreadProposal:
    strategy_name: str
    symbol: str = ""
    legs: List[SpreadLeg] = field(default_factory=list)
    net_credit_or_debit: float = 0.0     # signed: + = net credit, - = net debit
    max_profit: float = 0.0              # dollars (per 1-contract structure)
    max_loss: float = 0.0                # dollars (positive number)
    breakeven: Union[float, List[float], None] = None
    width: float = 0.0
    estimated_probability: float = 0.0
    reason: str = ""

    @property
    def is_credit(self) -> bool:
        return self.net_credit_or_debit > 0

    @property
    def is_tradeable(self) -> bool:
        return self.strategy_name != NO_TRADE

    def to_dict(self) -> Dict:
        return {
            "strategy_name": self.strategy_name,
            "symbol": self.symbol,
            "legs": [l.as_dict() for l in self.legs],
            "net_credit_or_debit": self.net_credit_or_debit,
            "max_profit": self.max_profit,
            "max_loss": self.max_loss,
            "breakeven": self.breakeven,
            "width": self.width,
            "estimated_probability": self.estimated_probability,
            "reason": self.reason,
        }


def _no_trade(reason: str, symbol: str = "", legs: Optional[List[SpreadLeg]] = None) -> SpreadProposal:
    return SpreadProposal(strategy_name=NO_TRADE, symbol=symbol,
                          legs=legs or [], reason=reason)


def no_trade_proposal(reason: str, symbol: str = "",
                      legs: Optional[List[SpreadLeg]] = None) -> SpreadProposal:
    """Public constructor for a no_trade proposal carrying a custom reason.

    Used by orchestration (`smart_trader.propose_spread`) to surface a specific
    no-edge / no-strikes / error reason without reaching into the private helper.
    """
    return _no_trade(reason, symbol, legs)


# --------------------------------------------------------------------------- #
# Selection rules (Requirement 4) — pure mapping
# --------------------------------------------------------------------------- #
def classify_volatility(iv: Optional[float], hv: Optional[float],
                        config: SpreadConfig) -> str:
    """'overpriced' | 'underpriced' | 'fair' | 'unknown' from IV vs HV."""
    try:
        iv = float(iv); hv = float(hv)
    except (TypeError, ValueError):
        return "unknown"
    if hv <= 0 or iv <= 0:
        return "unknown"
    ratio = iv / hv
    if ratio >= config.iv_overpriced_ratio:
        return "overpriced"
    if ratio <= config.iv_underpriced_ratio:
        return "underpriced"
    return "fair"


def classify_trend(momentum: Optional[float], config: SpreadConfig) -> str:
    """'bullish' | 'bearish' | 'neutral' from signed momentum."""
    try:
        m = float(momentum)
    except (TypeError, ValueError):
        return "neutral"
    if m >= config.min_trend_momentum:
        return "bullish"
    if m <= -config.min_trend_momentum:
        return "bearish"
    return "neutral"


def select_spread_strategy(vol_state: str, trend: str, edge_ok: bool = True) -> str:
    """Map (volatility state, trend) -> strategy name, exactly per Requirement 4.

    Weak edge, fair/unknown IV, or an unsupported (state, trend) combination all
    resolve to ``no_trade``.
    """
    if not edge_ok:
        return NO_TRADE
    if vol_state == "overpriced":
        if trend == "neutral":
            return IRON_CONDOR
        if trend == "bullish":
            return BULLISH_PUT_CREDIT_SPREAD
        if trend == "bearish":
            return BEARISH_CALL_CREDIT_SPREAD
    elif vol_state == "underpriced":
        if trend == "bullish":
            return DEBIT_CALL_SPREAD
        if trend == "bearish":
            return DEBIT_PUT_SPREAD
    return NO_TRADE


# --------------------------------------------------------------------------- #
# Safety validation (Requirement 5)
# --------------------------------------------------------------------------- #
def validate_defined_risk(legs: List[SpreadLeg]) -> Optional[str]:
    """Return 'undefined_risk' if any SHORT leg is not paired with a LONG leg of
    the SAME option type that caps its loss; else None.

    A short call's unlimited upside is capped by ANY long call (net payoff as
    price -> inf is K_short - K_long, bounded regardless of strike order), and a
    short put's downside is capped by ANY long put. So definition of risk is a
    count-based pairing: at least as many long calls as short calls, and at
    least as many long puts as short puts. Strike order only distinguishes a
    credit from a debit structure (checked per-builder), not risk-definition.
    This correctly admits debit spreads (long leg deeper, short leg further OTM)
    while still rejecting naked shorts and short-heavy ratio spreads.
    """
    short_calls = sum(1 for l in legs if l.action == "sell" and l.option_type == "call")
    long_calls = sum(1 for l in legs if l.action == "buy" and l.option_type == "call")
    short_puts = sum(1 for l in legs if l.action == "sell" and l.option_type == "put")
    long_puts = sum(1 for l in legs if l.action == "buy" and l.option_type == "put")
    if short_calls > long_calls or short_puts > long_puts:
        return "undefined_risk"
    return None


def _leg_spread_pct(leg: SpreadLeg) -> Optional[float]:
    if leg.bid is None or leg.ask is None:
        return None
    if leg.ask <= 0:
        return 100.0
    return (leg.ask - leg.bid) / leg.ask * 100.0


def validate_legs(legs: List[SpreadLeg], max_loss_dollars: float,
                  config: SpreadConfig) -> Optional[str]:
    """Run every hard rejection in order. Return a reason string on the first
    failure, else None. NEVER raises.
    """
    # Undefined risk (unprotected short).
    reason = validate_defined_risk(legs)
    if reason:
        return reason

    # Missing bid/ask.
    for leg in legs:
        if leg.bid is None or leg.ask is None or leg.bid <= 0 or leg.ask <= 0:
            return "missing_quote"

    # Wide per-leg quote.
    for leg in legs:
        sp = _leg_spread_pct(leg)
        if sp is None or sp > config.max_leg_spread_pct:
            return "wide_spread"

    # Illiquid legs (only when a floor is configured AND the data is present;
    # fail-open when the data is missing, matching the single-leg liquidity gate).
    for leg in legs:
        if config.min_open_interest > 0 and leg.open_interest is not None \
                and leg.open_interest < config.min_open_interest:
            return "illiquid_leg"
        if config.min_volume > 0 and leg.volume is not None \
                and leg.volume < config.min_volume:
            return "illiquid_leg"

    # Max loss above the risk limit.
    if max_loss_dollars > config.max_loss_limit:
        return "max_loss_exceeds_limit"

    return None


def _clamp_prob(p: float) -> float:
    return max(0.01, min(0.99, p))


# --------------------------------------------------------------------------- #
# Builders — each returns a SpreadProposal (valid) or a no_trade proposal whose
# reason names the failed safety check. Quotes are filled conservatively: you
# BUY at the ask and SELL at the bid.
# --------------------------------------------------------------------------- #
def build_bull_put_credit_spread(short_put: SpreadLeg, long_put: SpreadLeg,
                                 config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Sell higher-strike put, buy lower-strike put. Bullish, defined risk."""
    short_put.action, short_put.option_type = "sell", "put"
    long_put.action, long_put.option_type = "buy", "put"
    legs = [short_put, long_put]

    if short_put.strike <= long_put.strike:
        return _no_trade("undefined_risk", symbol, legs)

    width = short_put.strike - long_put.strike
    reason = validate_legs(legs, 0.0, config)  # quote checks first
    if reason in ("missing_quote", "wide_spread", "illiquid_leg", "undefined_risk"):
        return _no_trade(reason, symbol, legs)

    net_credit = short_put.bid - long_put.ask
    if net_credit <= 0:
        return _no_trade("non_positive_credit", symbol, legs)
    max_profit = net_credit * CONTRACT_MULTIPLIER
    max_loss = (width - net_credit) * CONTRACT_MULTIPLIER
    breakeven = short_put.strike - net_credit

    reason = validate_legs(legs, max_loss, config)
    if reason:
        return _no_trade(reason, symbol, legs)

    return SpreadProposal(
        strategy_name=BULLISH_PUT_CREDIT_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=net_credit, max_profit=max_profit, max_loss=max_loss,
        breakeven=breakeven, width=width,
        estimated_probability=_clamp_prob(1 - net_credit / width),
        reason="vol overpriced + bullish -> sell put spread below price")


def build_bear_call_credit_spread(short_call: SpreadLeg, long_call: SpreadLeg,
                                  config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Sell lower-strike call, buy higher-strike call. Bearish, defined risk."""
    short_call.action, short_call.option_type = "sell", "call"
    long_call.action, long_call.option_type = "buy", "call"
    legs = [short_call, long_call]

    if long_call.strike <= short_call.strike:
        return _no_trade("undefined_risk", symbol, legs)

    width = long_call.strike - short_call.strike
    reason = validate_legs(legs, 0.0, config)
    if reason in ("missing_quote", "wide_spread", "illiquid_leg", "undefined_risk"):
        return _no_trade(reason, symbol, legs)

    net_credit = short_call.bid - long_call.ask
    if net_credit <= 0:
        return _no_trade("non_positive_credit", symbol, legs)
    max_profit = net_credit * CONTRACT_MULTIPLIER
    max_loss = (width - net_credit) * CONTRACT_MULTIPLIER
    breakeven = short_call.strike + net_credit

    reason = validate_legs(legs, max_loss, config)
    if reason:
        return _no_trade(reason, symbol, legs)

    return SpreadProposal(
        strategy_name=BEARISH_CALL_CREDIT_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=net_credit, max_profit=max_profit, max_loss=max_loss,
        breakeven=breakeven, width=width,
        estimated_probability=_clamp_prob(1 - net_credit / width),
        reason="vol overpriced + bearish -> sell call spread above price")


def build_debit_call_spread(long_call: SpreadLeg, short_call: SpreadLeg,
                            config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Buy lower-strike call, sell higher-strike call. Bullish, defined risk."""
    long_call.action, long_call.option_type = "buy", "call"
    short_call.action, short_call.option_type = "sell", "call"
    legs = [long_call, short_call]

    if short_call.strike <= long_call.strike:
        return _no_trade("undefined_risk", symbol, legs)

    width = short_call.strike - long_call.strike
    reason = validate_legs(legs, 0.0, config)
    if reason in ("missing_quote", "wide_spread", "illiquid_leg", "undefined_risk"):
        return _no_trade(reason, symbol, legs)

    net_debit = long_call.ask - short_call.bid
    if net_debit <= 0:
        return _no_trade("non_positive_debit", symbol, legs)
    max_loss = net_debit * CONTRACT_MULTIPLIER
    max_profit = (width - net_debit) * CONTRACT_MULTIPLIER
    breakeven = long_call.strike + net_debit

    reason = validate_legs(legs, max_loss, config)
    if reason:
        return _no_trade(reason, symbol, legs)

    return SpreadProposal(
        strategy_name=DEBIT_CALL_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=-net_debit, max_profit=max_profit, max_loss=max_loss,
        breakeven=breakeven, width=width,
        estimated_probability=_clamp_prob(net_debit / width),
        reason="vol underpriced + bullish -> buy call spread")


def build_debit_put_spread(long_put: SpreadLeg, short_put: SpreadLeg,
                           config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Buy higher-strike put, sell lower-strike put. Bearish, defined risk."""
    long_put.action, long_put.option_type = "buy", "put"
    short_put.action, short_put.option_type = "sell", "put"
    legs = [long_put, short_put]

    if long_put.strike <= short_put.strike:
        return _no_trade("undefined_risk", symbol, legs)

    width = long_put.strike - short_put.strike
    reason = validate_legs(legs, 0.0, config)
    if reason in ("missing_quote", "wide_spread", "illiquid_leg", "undefined_risk"):
        return _no_trade(reason, symbol, legs)

    net_debit = long_put.ask - short_put.bid
    if net_debit <= 0:
        return _no_trade("non_positive_debit", symbol, legs)
    max_loss = net_debit * CONTRACT_MULTIPLIER
    max_profit = (width - net_debit) * CONTRACT_MULTIPLIER
    breakeven = long_put.strike - net_debit

    reason = validate_legs(legs, max_loss, config)
    if reason:
        return _no_trade(reason, symbol, legs)

    return SpreadProposal(
        strategy_name=DEBIT_PUT_SPREAD, symbol=symbol, legs=legs,
        net_credit_or_debit=-net_debit, max_profit=max_profit, max_loss=max_loss,
        breakeven=breakeven, width=width,
        estimated_probability=_clamp_prob(net_debit / width),
        reason="vol underpriced + bearish -> buy put spread")


def build_iron_condor(long_put: SpreadLeg, short_put: SpreadLeg,
                      short_call: SpreadLeg, long_call: SpreadLeg,
                      config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Sell an OTM put spread + an OTM call spread. Neutral, defined risk.

    Legs: long_put (lowest) < short_put < short_call < long_call (highest).
    """
    long_put.action, long_put.option_type = "buy", "put"
    short_put.action, short_put.option_type = "sell", "put"
    short_call.action, short_call.option_type = "sell", "call"
    long_call.action, long_call.option_type = "buy", "call"
    legs = [long_put, short_put, short_call, long_call]

    # Structural ordering must keep both shorts protected.
    if not (long_put.strike < short_put.strike <= short_call.strike < long_call.strike):
        return _no_trade("undefined_risk", symbol, legs)

    reason = validate_legs(legs, 0.0, config)
    if reason in ("missing_quote", "wide_spread", "illiquid_leg", "undefined_risk"):
        return _no_trade(reason, symbol, legs)

    put_width = short_put.strike - long_put.strike
    call_width = long_call.strike - short_call.strike
    width = max(put_width, call_width)  # conservative max-loss wing

    net_credit = ((short_put.bid - long_put.ask) +
                  (short_call.bid - long_call.ask))
    if net_credit <= 0:
        return _no_trade("non_positive_credit", symbol, legs)
    max_profit = net_credit * CONTRACT_MULTIPLIER
    max_loss = (width - net_credit) * CONTRACT_MULTIPLIER
    breakeven = [short_put.strike - net_credit, short_call.strike + net_credit]

    reason = validate_legs(legs, max_loss, config)
    if reason:
        return _no_trade(reason, symbol, legs)

    return SpreadProposal(
        strategy_name=IRON_CONDOR, symbol=symbol, legs=legs,
        net_credit_or_debit=net_credit, max_profit=max_profit, max_loss=max_loss,
        breakeven=breakeven, width=width,
        estimated_probability=_clamp_prob(1 - net_credit / width),
        reason="vol overpriced + neutral -> iron condor")


def build_spread(strategy_name: str, legs: List[SpreadLeg],
                 config: SpreadConfig, symbol: str = "") -> SpreadProposal:
    """Generic dispatcher used by orchestration and the undefined-risk path.

    ``legs`` may be supplied in any order; roles are inferred from action +
    option_type + strike. Returns a no_trade proposal (reason='undefined_risk'
    or 'bad_leg_count') when the legs do not form the expected structure.
    """
    if strategy_name == NO_TRADE:
        return _no_trade("no_trade", symbol, legs)

    # Reject obviously undefined-risk leg sets up front (e.g. a lone short).
    reason = validate_defined_risk(legs)
    if reason:
        return _no_trade(reason, symbol, legs)

    puts = sorted([l for l in legs if l.option_type == "put"], key=lambda l: l.strike)
    calls = sorted([l for l in legs if l.option_type == "call"], key=lambda l: l.strike)

    try:
        if strategy_name == BULLISH_PUT_CREDIT_SPREAD:
            if len(puts) != 2:
                return _no_trade("bad_leg_count", symbol, legs)
            return build_bull_put_credit_spread(puts[1], puts[0], config, symbol)
        if strategy_name == BEARISH_CALL_CREDIT_SPREAD:
            if len(calls) != 2:
                return _no_trade("bad_leg_count", symbol, legs)
            return build_bear_call_credit_spread(calls[0], calls[1], config, symbol)
        if strategy_name == DEBIT_CALL_SPREAD:
            if len(calls) != 2:
                return _no_trade("bad_leg_count", symbol, legs)
            return build_debit_call_spread(calls[0], calls[1], config, symbol)
        if strategy_name == DEBIT_PUT_SPREAD:
            if len(puts) != 2:
                return _no_trade("bad_leg_count", symbol, legs)
            return build_debit_put_spread(puts[1], puts[0], config, symbol)
        if strategy_name == IRON_CONDOR:
            if len(puts) != 2 or len(calls) != 2:
                return _no_trade("bad_leg_count", symbol, legs)
            return build_iron_condor(puts[0], puts[1], calls[0], calls[1], config, symbol)
    except Exception as e:  # never raise out of a proposal build
        return _no_trade(f"build_error:{type(e).__name__}", symbol, legs)

    return _no_trade("unknown_strategy", symbol, legs)


# --------------------------------------------------------------------------- #
# Logging (Requirement 6)
# --------------------------------------------------------------------------- #
def _fmt_breakeven(be) -> str:
    if be is None:
        return "n/a"
    if isinstance(be, (list, tuple)):
        return "/".join(f"{x:.2f}" for x in be)
    return f"{be:.2f}"


def format_proposal_log(proposal: SpreadProposal) -> str:
    """Multi-line [SPREAD_PROPOSAL] block with every required field."""
    legs = "; ".join(l.label() for l in proposal.legs) if proposal.legs else "none"
    return (
        "[SPREAD_PROPOSAL]\n"
        f"strategy={proposal.strategy_name}\n"
        f"symbol={proposal.symbol}\n"
        f"legs={legs}\n"
        f"net_credit_or_debit={proposal.net_credit_or_debit:.2f}\n"
        f"max_profit={proposal.max_profit:.2f}\n"
        f"max_loss={proposal.max_loss:.2f}\n"
        f"breakeven={_fmt_breakeven(proposal.breakeven)}\n"
        f"reason={proposal.reason}"
    )


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    cfg = SpreadConfig(enabled=True, max_loss_limit=1000.0, max_leg_spread_pct=50.0)

    def leg(action, otype, strike, bid, ask, oi=500, vol=500):
        return SpreadLeg(action=action, option_type=otype, strike=strike,
                         bid=bid, ask=ask, open_interest=oi, volume=vol)

    # Bull put credit: sell 100P @1.20, buy 95P @0.40 -> credit 0.80, width 5.
    p = build_bull_put_credit_spread(leg("sell", "put", 100, 1.20, 1.25),
                                     leg("buy", "put", 95, 0.40, 0.45), cfg, "SPY")
    if p.strategy_name != BULLISH_PUT_CREDIT_SPREAD or round(p.max_profit, 2) != 75.0:
        print("FAIL bull put", p); ok = False

    # Bear call credit.
    p = build_bear_call_credit_spread(leg("sell", "call", 100, 1.20, 1.25),
                                      leg("buy", "call", 105, 0.40, 0.45), cfg, "SPY")
    if p.strategy_name != BEARISH_CALL_CREDIT_SPREAD or p.max_loss <= 0:
        print("FAIL bear call", p); ok = False

    # Debit call.
    p = build_debit_call_spread(leg("buy", "call", 100, 1.95, 2.00),
                                leg("sell", "call", 105, 0.50, 0.55), cfg, "SPY")
    if p.strategy_name != DEBIT_CALL_SPREAD or p.net_credit_or_debit >= 0:
        print("FAIL debit call", p); ok = False

    # Iron condor.
    p = build_iron_condor(leg("buy", "put", 90, 0.30, 0.35),
                          leg("sell", "put", 95, 0.90, 0.95),
                          leg("sell", "call", 105, 0.90, 0.95),
                          leg("buy", "call", 110, 0.30, 0.35), cfg, "SPY")
    if p.strategy_name != IRON_CONDOR or not isinstance(p.breakeven, list):
        print("FAIL iron condor", p); ok = False

    # Undefined risk: a lone short put.
    p = build_spread(BULLISH_PUT_CREDIT_SPREAD, [leg("sell", "put", 100, 1.0, 1.1)], cfg, "SPY")
    if p.strategy_name != NO_TRADE or p.reason != "undefined_risk":
        print("FAIL undefined risk", p); ok = False

    # Selection rules.
    checks = [
        (("overpriced", "neutral"), IRON_CONDOR),
        (("overpriced", "bullish"), BULLISH_PUT_CREDIT_SPREAD),
        (("overpriced", "bearish"), BEARISH_CALL_CREDIT_SPREAD),
        (("underpriced", "bullish"), DEBIT_CALL_SPREAD),
        (("underpriced", "bearish"), DEBIT_PUT_SPREAD),
        (("fair", "bullish"), NO_TRADE),
        (("underpriced", "neutral"), NO_TRADE),
    ]
    for (vs, tr), exp in checks:
        if select_spread_strategy(vs, tr) != exp:
            print("FAIL selection", vs, tr, "->", select_spread_strategy(vs, tr)); ok = False
    if select_spread_strategy("overpriced", "bullish", edge_ok=False) != NO_TRADE:
        print("FAIL weak edge -> no_trade"); ok = False

    # Log block has the required fields.
    log = format_proposal_log(p)
    for tok in ("[SPREAD_PROPOSAL]", "strategy=", "symbol=", "legs=",
                "net_credit_or_debit=", "max_profit=", "max_loss=",
                "breakeven=", "reason="):
        if tok not in log:
            print("FAIL log missing", tok); ok = False

    print("spread_builder self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
