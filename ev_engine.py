"""
Phase 10A — Expected Value (EV) Engine.  ADVISORY ANALYTICS ONLY.

Estimates, for a candidate defined-risk options structure (a
``spread_builder.SpreadProposal``):

  1. Probability of Profit (PoP)  — terminal price distribution vs breakeven(s)
  2. Expected Value (EV)          — probability-weighted payoff minus costs
  3. EV / Max Loss                — risk-adjusted ranking number
  4. Recommendation               — STRONG_ACCEPT .. REJECT_CANDIDATE

The objective is NOT perfect option pricing — it is a *consistent relative
ranking* across candidates, so the simple closed forms below are intentional:

  Credit spread:   EV = PoP*max_profit - (1-PoP)*max_loss - costs
  Debit spread:    EV = P_max*max_profit + P_partial*partial_payout
                        - P_loss*max_loss - costs
                   (partial_payout approximated as max_profit/2 — the midpoint
                    of the payoff ramp between breakeven and the short strike)
  Iron condor:     EV = range_prob*max_profit - tail_prob*max_loss - costs

Probabilities come from ``barrier_engine.prob_close_beyond`` (terminal
log-normal tail, driftless by default). Costs come from ``cost_model``
(spread crossing + slippage + fees + carry, per leg, round trip). Nothing
here is duplicated from those modules.

HARD SCOPE RULE (Phase 10A): no execution path may consume this module.
It must never place, size, gate, filter or alter a trade — it only computes.
``smart_trader`` and ``run_alpaca_intraday`` must NOT import it (guarded by
test_ev_engine.TestNoExecutionPathTouched). The only surface is the Telegram
``EV_ANALYSIS`` analytics command.

Everything fails open: bad/missing inputs produce an EVResult with
``status='insufficient_data'`` and a reason — never an exception.
"""

import math
from dataclasses import dataclass, asdict
from datetime import date, datetime
from typing import Optional

from barrier_engine import prob_close_beyond
from cost_model import CONTRACT_MULTIPLIER, CostModel
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR, NO_TRADE,
)

# --------------------------------------------------------------------------- #
# Recommendation vocabulary — same strings as advisory_gate's tiers so the two
# advisory surfaces speak one language (string constants only; no logic shared).
# --------------------------------------------------------------------------- #
STRONG_ACCEPT = "STRONG_ACCEPT"
ACCEPT = "ACCEPT"
NEUTRAL = "NEUTRAL"
WEAK_SETUP = "WEAK_SETUP"
REJECT_CANDIDATE = "REJECT_CANDIDATE"

STATUS_OK = "ok"
STATUS_INSUFFICIENT = "insufficient_data"

# Human display names for Telegram output.
_DISPLAY_NAMES = {
    BULLISH_PUT_CREDIT_SPREAD: "Bull Put Credit Spread",
    BEARISH_CALL_CREDIT_SPREAD: "Bear Call Credit Spread",
    DEBIT_CALL_SPREAD: "Debit Call Spread",
    DEBIT_PUT_SPREAD: "Debit Put Spread",
    IRON_CONDOR: "Iron Condor",
}

_CREDIT_SPREADS = {BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD}
_DEBIT_SPREADS = {DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD}


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class EVConfig:
    """Recommendation thresholds on EV / max_loss (ev_per_dollar_risk)."""
    strong_accept_min: float = 0.15   # ratio >= this -> STRONG_ACCEPT
    accept_min: float = 0.05          # ratio >= this -> ACCEPT
    weak_min: float = -0.05           # ratio >= this (but < 0) -> WEAK_SETUP
    default_days: int = 30            # horizon when DTE can't be derived

    @staticmethod
    def from_env(path: str = ".env", loader=None) -> "EVConfig":
        from config_loader import ConfigLoader
        cfg = loader if loader is not None else ConfigLoader(path=path)
        return EVConfig(
            strong_accept_min=cfg.get_float("EV_STRONG_ACCEPT_MIN", 0.15),
            accept_min=cfg.get_float("EV_ACCEPT_MIN", 0.05),
            weak_min=cfg.get_float("EV_WEAK_MIN", -0.05),
            default_days=cfg.get_int("EV_DEFAULT_DAYS", 30),
        )


# --------------------------------------------------------------------------- #
# Result
# --------------------------------------------------------------------------- #
@dataclass
class EVResult:
    symbol: str = ""
    strategy: str = ""
    expected_value: Optional[float] = None
    probability_of_profit: Optional[float] = None
    ev_per_dollar_risk: Optional[float] = None
    max_profit: Optional[float] = None
    max_loss: Optional[float] = None
    estimated_costs: Optional[float] = None
    oracle_score: Optional[float] = None
    volatility_edge: Optional[float] = None
    days: Optional[int] = None
    recommendation: str = NEUTRAL
    status: str = STATUS_OK
    reason: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


# --------------------------------------------------------------------------- #
# Validation helpers
# --------------------------------------------------------------------------- #
def _is_num(x) -> bool:
    return isinstance(x, (int, float)) and not isinstance(x, bool) and math.isfinite(x)


def _valid_prob(p) -> bool:
    """True only for a finite number in [0, 1]."""
    return _is_num(p) and 0.0 <= p <= 1.0


def _insufficient(symbol, strategy, reason, **kw) -> EVResult:
    return EVResult(symbol=symbol or "", strategy=strategy or "",
                    status=STATUS_INSUFFICIENT, reason=reason,
                    recommendation=NEUTRAL, **kw)


def _get(proposal, name, default=None):
    """Field access that works for both SpreadProposal and plain dicts."""
    if isinstance(proposal, dict):
        return proposal.get(name, default)
    return getattr(proposal, name, default)


# --------------------------------------------------------------------------- #
# Terminal-distribution probabilities (delegated to barrier_engine)
# --------------------------------------------------------------------------- #
def _p_terminal_above(spot: float, level: float, sigma: float,
                      mu: float = 0.0, days: float = 30.0) -> float:
    """P(S_T >= level) from barrier_engine's terminal log-normal tail."""
    p = prob_close_beyond(spot, level, sigma, mu, days)
    # prob_close_beyond returns P(S_T >= level) for an upper barrier and
    # P(S_T <= level) for a lower (or equal) one — flip the lower case.
    return p if level > spot else 1.0 - p


def _p_terminal_below(spot: float, level: float, sigma: float,
                      mu: float = 0.0, days: float = 30.0) -> float:
    return 1.0 - _p_terminal_above(spot, level, sigma, mu, days)


def _short_strike(legs, option_type: str) -> Optional[float]:
    """Strike of the SELL leg of the given option type (debit-spread cap)."""
    for leg in legs or []:
        action = _get(leg, "action", "")
        otype = _get(leg, "option_type", _get(leg, "type", ""))
        if str(action).lower() == "sell" and str(otype).lower() == option_type:
            strike = _get(leg, "strike")
            return float(strike) if _is_num(strike) else None
    return None


# --------------------------------------------------------------------------- #
# Pure EV formulas (unit-testable; return None on invalid probabilities)
# --------------------------------------------------------------------------- #
def credit_spread_ev(pop, max_profit, max_loss, costs=0.0) -> Optional[float]:
    """EV = PoP*max_profit - (1-PoP)*max_loss - costs."""
    if not _valid_prob(pop):
        return None
    if not (_is_num(max_profit) and _is_num(max_loss) and _is_num(costs)):
        return None
    return pop * max_profit - (1.0 - pop) * max_loss - costs


def debit_spread_ev(p_max, p_partial, p_loss, max_profit, partial_payout,
                    max_loss, costs=0.0) -> Optional[float]:
    """EV = P_max*max_profit + P_partial*partial_payout - P_loss*max_loss - costs.

    The three region probabilities must each be valid and sum to <= 1 (within
    numerical tolerance); otherwise the inputs are inconsistent -> None.
    """
    for p in (p_max, p_partial, p_loss):
        if not _valid_prob(p):
            return None
    if p_max + p_partial + p_loss > 1.0 + 1e-6:
        return None
    if not all(_is_num(x) for x in (max_profit, partial_payout, max_loss, costs)):
        return None
    return (p_max * max_profit + p_partial * partial_payout
            - p_loss * max_loss - costs)


def iron_condor_ev(range_prob, tail_prob, max_profit, max_loss,
                   costs=0.0) -> Optional[float]:
    """EV = range_prob*max_profit - tail_prob*max_loss - costs."""
    if not (_valid_prob(range_prob) and _valid_prob(tail_prob)):
        return None
    if range_prob + tail_prob > 1.0 + 1e-6:
        return None
    if not (_is_num(max_profit) and _is_num(max_loss) and _is_num(costs)):
        return None
    return range_prob * max_profit - tail_prob * max_loss - costs


def ev_per_dollar_risk(ev, max_loss) -> Optional[float]:
    """EV normalized by the dollars at risk; None when undefined."""
    if not (_is_num(ev) and _is_num(max_loss)) or max_loss <= 0:
        return None
    return ev / max_loss


def classify_ev(ratio, config: Optional[EVConfig] = None) -> str:
    """Map an EV/max_loss ratio to a recommendation tier (None -> NEUTRAL)."""
    cfg = config or EVConfig()
    if not _is_num(ratio):
        return NEUTRAL
    if ratio >= cfg.strong_accept_min:
        return STRONG_ACCEPT
    if ratio >= cfg.accept_min:
        return ACCEPT
    if ratio >= 0.0:
        return NEUTRAL
    if ratio >= cfg.weak_min:
        return WEAK_SETUP
    return REJECT_CANDIDATE


# --------------------------------------------------------------------------- #
# Execution-cost estimate (delegated to cost_model, per leg, round trip)
# --------------------------------------------------------------------------- #
def estimate_structure_costs(legs, days: int = 0,
                             model: Optional[CostModel] = None) -> float:
    """Round-trip execution cost in dollars for the whole structure (qty=1/leg).

    Legs with a usable bid/ask use ``CostModel.round_trip_cost`` directly.
    Legs missing quotes use a conservative floor: the model's minimum spread
    plus slippage both ways plus fees both ways plus carry.
    """
    model = model or CostModel()
    cfg = model.config
    hold = max(0, int(days)) if _is_num(days) else 0
    total = 0.0
    for leg in legs or []:
        bid = _get(leg, "bid")
        ask = _get(leg, "ask")
        if _is_num(bid) and _is_num(ask) and ask > 0:
            total += model.round_trip_cost(float(bid), float(ask),
                                           qty=1, hold_days=hold)["cost_dollars"]
        else:
            total += ((cfg.min_spread_floor + 2.0 * cfg.slippage_per_contract)
                      * CONTRACT_MULTIPLIER
                      + 2.0 * (cfg.occ_fee_per_contract + cfg.commission_per_contract)
                      + cfg.overnight_carry_per_contract_per_day * hold)
    return total


# --------------------------------------------------------------------------- #
# Core evaluation (pure: proposal + spot + sigma in, EVResult out)
# --------------------------------------------------------------------------- #
def evaluate_proposal(proposal, spot, sigma, days=None, mu: float = 0.0,
                      volatility_edge=None, oracle_score=None,
                      config: Optional[EVConfig] = None,
                      cost_model: Optional[CostModel] = None) -> EVResult:
    """Estimate PoP / EV / EV-per-risk for one SpreadProposal. Never raises.

    ``spot``  — current underlying price.
    ``sigma`` — annualized volatility (decimal, e.g. 0.20).
    ``days``  — horizon to expiry (defaults to ``config.default_days``).
    ``mu``    — annualized drift (0.0 = neutral terminal distribution).
    """
    cfg = config or EVConfig()
    symbol = _get(proposal, "symbol", "") if proposal is not None else ""
    strategy = _get(proposal, "strategy_name", "") if proposal is not None else ""

    # ---- guards (fail open to insufficient_data) ------------------------- #
    if proposal is None:
        return _insufficient(symbol, strategy, "no proposal")
    if strategy == NO_TRADE or strategy not in _DISPLAY_NAMES:
        reason = _get(proposal, "reason", "") or "no tradeable structure"
        return _insufficient(symbol, strategy, reason)
    if not (_is_num(spot) and spot > 0):
        return _insufficient(symbol, strategy, "missing spot price")
    if not (_is_num(sigma) and sigma > 0):
        return _insufficient(symbol, strategy, "missing volatility")

    if not (_is_num(days) and days > 0):
        days = cfg.default_days
    days_i = max(1, int(days))

    max_profit = _get(proposal, "max_profit")
    max_loss = _get(proposal, "max_loss")
    breakeven = _get(proposal, "breakeven")
    legs = _get(proposal, "legs") or []
    if not (_is_num(max_profit) and _is_num(max_loss)) or max_loss <= 0:
        return _insufficient(symbol, strategy, "missing max_profit/max_loss")

    if oracle_score is None:
        oracle_score = _get(proposal, "oracle_score")

    costs = estimate_structure_costs(legs, days=days_i, model=cost_model)

    # ---- per-strategy probabilities + EV --------------------------------- #
    ev = pop = None
    if strategy in _CREDIT_SPREADS:
        if not _is_num(breakeven):
            return _insufficient(symbol, strategy, "missing breakeven")
        if strategy == BULLISH_PUT_CREDIT_SPREAD:
            pop = _p_terminal_above(spot, float(breakeven), sigma, mu, days_i)
        else:  # bear call credit: profit when price stays below breakeven
            pop = _p_terminal_below(spot, float(breakeven), sigma, mu, days_i)
        ev = credit_spread_ev(pop, max_profit, max_loss, costs)

    elif strategy in _DEBIT_SPREADS:
        if not _is_num(breakeven):
            return _insufficient(symbol, strategy, "missing breakeven")
        otype = "call" if strategy == DEBIT_CALL_SPREAD else "put"
        short_k = _short_strike(legs, otype)
        if short_k is None:
            return _insufficient(symbol, strategy, "missing short strike")
        if strategy == DEBIT_CALL_SPREAD:
            p_max = _p_terminal_above(spot, short_k, sigma, mu, days_i)
            p_loss = _p_terminal_below(spot, float(breakeven), sigma, mu, days_i)
        else:
            p_max = _p_terminal_below(spot, short_k, sigma, mu, days_i)
            p_loss = _p_terminal_above(spot, float(breakeven), sigma, mu, days_i)
        p_partial = min(1.0, max(0.0, 1.0 - p_max - p_loss))
        pop = min(1.0, max(0.0, p_max + p_partial))
        ev = debit_spread_ev(p_max, p_partial, p_loss, max_profit,
                             max_profit / 2.0, max_loss, costs)

    else:  # IRON_CONDOR
        bes = breakeven if isinstance(breakeven, (list, tuple)) else None
        if not bes or len(bes) != 2 or not all(_is_num(b) for b in bes):
            return _insufficient(symbol, strategy, "missing condor breakevens")
        be_low, be_high = sorted(float(b) for b in bes)
        range_prob = min(1.0, max(0.0,
            _p_terminal_below(spot, be_high, sigma, mu, days_i)
            - _p_terminal_below(spot, be_low, sigma, mu, days_i)))
        tail_prob = 1.0 - range_prob
        pop = range_prob
        ev = iron_condor_ev(range_prob, tail_prob, max_profit, max_loss, costs)

    if ev is None or not _valid_prob(pop):
        return _insufficient(symbol, strategy, "invalid probability inputs",
                             max_profit=max_profit, max_loss=max_loss,
                             estimated_costs=round(costs, 2), days=days_i)

    ratio = ev_per_dollar_risk(ev, max_loss)
    return EVResult(
        symbol=symbol, strategy=strategy,
        expected_value=round(ev, 2),
        probability_of_profit=round(pop, 4),
        ev_per_dollar_risk=round(ratio, 4) if ratio is not None else None,
        max_profit=round(float(max_profit), 2),
        max_loss=round(float(max_loss), 2),
        estimated_costs=round(costs, 2),
        oracle_score=oracle_score if _is_num(oracle_score) else None,
        volatility_edge=volatility_edge if _is_num(volatility_edge) else None,
        days=days_i,
        recommendation=classify_ev(ratio, cfg),
        status=STATUS_OK,
        reason="",
    )


# --------------------------------------------------------------------------- #
# Symbol-level convenience (used by the Telegram EV_ANALYSIS command only).
# Takes a duck-typed trader; this module never imports smart_trader.
# --------------------------------------------------------------------------- #
def _dte_from_legs(legs, today: Optional[date] = None) -> Optional[int]:
    """Days to expiry from the first leg's YYYY-MM-DD expiration (>=1)."""
    today = today or date.today()
    for leg in legs or []:
        exp = _get(leg, "expiration", "")
        try:
            d = datetime.strptime(str(exp)[:10], "%Y-%m-%d").date()
            return max(1, (d - today).days)
        except (TypeError, ValueError):
            continue
    return None


def evaluate_for_symbol(trader, symbol: str,
                        config: Optional[EVConfig] = None) -> EVResult:
    """Propose the symbol's best spread and EV-score it. Advisory; never raises."""
    cfg = config or EVConfig()
    symbol = (symbol or "").strip().upper()
    try:
        proposal = trader.propose_spread(symbol)
    except Exception as e:
        return _insufficient(symbol, "", f"proposal failed: {e}")
    if proposal is None or not getattr(proposal, "is_tradeable", False):
        reason = _get(proposal, "reason", "") if proposal is not None else "no proposal"
        return _insufficient(symbol, _get(proposal, "strategy_name", "") or "",
                             reason or "no tradeable structure")

    spot = sigma = vol_edge = None
    try:
        from expected_move_engine import (
            ExpectedMoveConfig, ExpectedMoveEngine, gather_inputs_from_trader,
        )
        inputs = gather_inputs_from_trader(trader, symbol)
        em = ExpectedMoveEngine(ExpectedMoveConfig.from_env()).compute(
            inputs, symbol=symbol)
        spot = getattr(inputs, "price", None)
        sigma = getattr(em, "forecast_vol", None)
        vol_edge = getattr(em, "volatility_edge", None)
    except Exception as e:
        return _insufficient(symbol, _get(proposal, "strategy_name", ""),
                             f"vol/spot unavailable: {e}")

    days = _dte_from_legs(_get(proposal, "legs")) or cfg.default_days
    return evaluate_proposal(proposal, spot, sigma, days=days,
                             volatility_edge=vol_edge, config=cfg)


# --------------------------------------------------------------------------- #
# Telegram formatting (analytics text only)
# --------------------------------------------------------------------------- #
def display_strategy_name(strategy: str) -> str:
    return _DISPLAY_NAMES.get(strategy, (strategy or "Unknown").replace("_", " ").title())


def format_ev_report(result: EVResult) -> str:
    """Markdown EV report for Telegram. Pure formatting; no side effects."""
    sym = result.symbol or "?"
    if result.status != STATUS_OK or result.expected_value is None:
        return (f"📭 *No EV analysis for {sym}*\n"
                f"Reason: {result.reason or 'insufficient data'}\n"
                f"_(Advisory analytics only — nothing was traded.)_")

    pop_pct = round((result.probability_of_profit or 0.0) * 100.0)
    ratio = result.ev_per_dollar_risk
    lines = [
        f"📐 *EV ANALYSIS — {sym}*",
        "",
        f"Strategy: {display_strategy_name(result.strategy)}",
        "",
        f"Expected Value: {result.expected_value:+.2f}",
        f"Probability of Profit: {pop_pct}%",
        "",
        f"EV / Risk: {ratio:.2f}" if ratio is not None else "EV / Risk: n/a",
        "",
        f"Max Profit: ${result.max_profit:.2f}  |  Max Loss: ${result.max_loss:.2f}",
        f"Est. Costs: ${result.estimated_costs:.2f}  |  Horizon: {result.days}d",
    ]
    extras = []
    if result.oracle_score is not None:
        extras.append(f"Oracle Score: {result.oracle_score:.1f}")
    if result.volatility_edge is not None:
        extras.append(f"Vol Edge: {result.volatility_edge:+.2f}")
    if extras:
        lines.append("  |  ".join(extras))
    lines += [
        "",
        f"Recommendation: {result.recommendation}",
        "",
        "_(Advisory analytics only — no orders placed or altered.)_",
    ]
    return "\n".join(lines)
