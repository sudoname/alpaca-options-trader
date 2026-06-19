"""
Phase 12 — Strategy EV Matrix (analytics only, fail-open, no execution).

The live Oracle decides *direction -> CALL/PUT -> contract -> (optional) EV*:
``determine_option_strategy`` picks call/put/skip from signals BEFORE any EV is
computed (smart_trader.py:1258), ``select_best_option`` ranks contracts by an ML
heuristic rather than EV (smart_trader.py:1499), and the spread path commits to a
structure via ``select_spread_strategy`` and only then scores it.

This module answers the inverse, EV-FIRST question for one underlying:

    Given the market state, build EVERY candidate structure, estimate the
    expected value of each, and rank them — so the highest-EV structure (or
    SKIP, when every structure is negative) is visible.

It builds the eight candidates — long call, long put, debit call spread, debit
put spread, bull-put credit spread, bear-call credit spread, iron condor, and a
synthetic SKIP — scores spreads through ``ev_engine.evaluate_proposal`` and the
long single legs through ``entry_ev_stamp.compute_entry_stamp``, annotates each
with an advisory recommendation (``advisory_gate.evaluate_setup``) and the
read-only RL Q-value for the matching action, then ranks by EV-per-dollar-risk
then EV. SKIP is a zero-EV anchor, so when every real structure has negative
EV/Risk the matrix ranks SKIP first.

STRICTLY analytics: this module never opens, closes, sizes, prices, blocks or
triggers any real or paper trade, never mutates a Q-table, and never reaches the
network. The option chain is INJECTED (a list of quote dicts), so the whole
module is deterministic and offline-testable. Every path fails open: a structure
that can't be built or scored becomes an ``insufficient_data`` row, never an
exception.
"""

from typing import Callable, Dict, List, Optional

import ev_engine
import spread_builder as sb
from entry_ev_stamp import compute_entry_stamp

# Synthetic "do nothing" candidate. Its EV and EV/Risk are exactly zero, so it
# wins the ranking whenever every real structure is negative.
SKIP = "skip"

# Long single-leg pseudo-strategies (ev_engine only scores spreads).
LONG_CALL = "long_call"
LONG_PUT = "long_put"

# Default take-profit / stop-loss barriers for the single-leg gambler's-ruin EV
# model (mirrors entry_ev_stamp's own defaults).
DEFAULT_TP = 0.25
DEFAULT_SL = 0.15

ANALYTICS_FOOTER = "Analytics only — no trades placed, sized or blocked."

# Direction bias per candidate (used for display + RL action mapping).
BULLISH = "bullish"
BEARISH = "bearish"
NEUTRAL = "neutral"
NONE_BIAS = "none"

_BIAS = {
    LONG_CALL: BULLISH,
    LONG_PUT: BEARISH,
    sb.DEBIT_CALL_SPREAD: BULLISH,
    sb.DEBIT_PUT_SPREAD: BEARISH,
    sb.BULLISH_PUT_CREDIT_SPREAD: BULLISH,
    sb.BEARISH_CALL_CREDIT_SPREAD: BEARISH,
    sb.IRON_CONDOR: NEUTRAL,
    SKIP: NONE_BIAS,
}

# RL action that corresponds to each candidate's directional commitment.
_RL_ACTION = {
    BULLISH: "CALL",
    BEARISH: "PUT",
    NEUTRAL: "SKIP",
    NONE_BIAS: "SKIP",
}


# ---------------------------------------------------------------------------
# Quote normalization / chain helpers (pure)
# ---------------------------------------------------------------------------
def _f(value, default: Optional[float] = None) -> Optional[float]:
    try:
        if value is None or isinstance(value, bool):
            return default
        return float(value)
    except (TypeError, ValueError):
        return default


def _mid(bid, ask) -> Optional[float]:
    b, a = _f(bid), _f(ask)
    if b is not None and a is not None and a >= b >= 0 and a > 0:
        return (a + b) / 2.0
    return a if a and a > 0 else (b if b and b > 0 else None)


def _norm_quote(q) -> Optional[dict]:
    """A chain entry -> normalized quote dict, or None if unusable."""
    if not isinstance(q, dict):
        return None
    otype = str(q.get("option_type") or q.get("type") or "").lower()
    if otype in ("c", "call"):
        otype = "call"
    elif otype in ("p", "put"):
        otype = "put"
    else:
        return None
    strike = _f(q.get("strike"))
    if strike is None or strike <= 0:
        return None
    return {
        "option_type": otype,
        "strike": strike,
        "bid": _f(q.get("bid")),
        "ask": _f(q.get("ask")),
        "delta": _f(q.get("delta")),
        "open_interest": _f(q.get("open_interest")),
        "volume": _f(q.get("volume")),
        "symbol": str(q.get("symbol") or ""),
        "expiration": str(q.get("expiration") or ""),
        "confidence": q.get("confidence"),
    }


def _split_chain(chain) -> Dict[str, List[dict]]:
    """Normalized calls/puts sorted ascending by strike. Fail-open to empties."""
    calls: List[dict] = []
    puts: List[dict] = []
    for q in chain or []:
        nq = _norm_quote(q)
        if nq is None:
            continue
        (calls if nq["option_type"] == "call" else puts).append(nq)
    calls.sort(key=lambda x: x["strike"])
    puts.sort(key=lambda x: x["strike"])
    return {"call": calls, "put": puts}


def _nearest_idx(quotes: List[dict], spot: float) -> Optional[int]:
    if not quotes:
        return None
    return min(range(len(quotes)),
               key=lambda i: abs(quotes[i]["strike"] - spot))


def _pick(quotes: List[dict], idx: Optional[int], offset: int) -> Optional[dict]:
    if idx is None:
        return None
    j = idx + offset
    if 0 <= j < len(quotes):
        return quotes[j]
    return None


def _liquidity_score(quotes: List[dict]) -> Optional[float]:
    """0-1 tightness score = 1 - widest relative bid/ask spread across legs.

    Returns None when no leg carries a usable two-sided quote (fail-open: the
    caller treats None as 'unknown', never as illiquid)."""
    worst = None
    for q in quotes:
        b, a = q.get("bid"), q.get("ask")
        if b is None or a is None or a <= 0 or b < 0 or a < b:
            continue
        rel = (a - b) / a
        worst = rel if worst is None else max(worst, rel)
    if worst is None:
        return None
    return round(max(0.0, 1.0 - worst), 4)


def _to_leg(q: dict) -> sb.SpreadLeg:
    return sb.SpreadLeg(
        action="buy", option_type=q["option_type"], strike=q["strike"],
        bid=q["bid"], ask=q["ask"], symbol=q["symbol"],
        expiration=q["expiration"], open_interest=q["open_interest"],
        volume=q["volume"])


# ---------------------------------------------------------------------------
# Candidate rows
# ---------------------------------------------------------------------------
def _blank_row(symbol: str, strategy: str) -> dict:
    return {
        "symbol": symbol,
        "strategy": strategy,
        "direction_bias": _BIAS.get(strategy, NONE_BIAS),
        "expected_value": None,
        "ev_per_dollar_risk": None,
        "probability_of_profit": None,
        "max_profit": None,
        "max_loss": None,
        "liquidity_score": None,
        "oracle_score": None,
        "volatility_edge": None,
        "advisory_recommendation": None,
        "rl_state_q": None,
        "rl_action": _RL_ACTION.get(_BIAS.get(strategy, NONE_BIAS), "SKIP"),
        "final_rank": None,
        "status": "insufficient_data",
        "reason": "",
    }


def _skip_row(symbol: str) -> dict:
    row = _blank_row(symbol, SKIP)
    row.update({
        "expected_value": 0.0,
        "ev_per_dollar_risk": 0.0,
        "probability_of_profit": None,
        "max_profit": 0.0,
        "max_loss": 0.0,
        "status": ev_engine.STATUS_OK,
        "reason": "do nothing (zero-EV anchor)",
    })
    return row


def _single_leg_row(symbol: str, strategy: str, quote: Optional[dict],
                    volatility_edge: Optional[float],
                    tp: float, sl: float) -> dict:
    row = _blank_row(symbol, strategy)
    row["volatility_edge"] = volatility_edge
    if quote is None:
        row["reason"] = "no contract in chain"
        return row
    entry = _mid(quote.get("bid"), quote.get("ask"))
    if entry is None or entry <= 0:
        row["reason"] = "no usable premium"
        return row
    option = {"delta": quote.get("delta"), "confidence": quote.get("confidence")}
    levels = {"take_profit_percent": tp, "stop_loss_percent": sl}
    stamp = compute_entry_stamp(option, levels, entry, 1,
                                bid=quote.get("bid"), ask=quote.get("ask"))
    if not stamp:
        row["reason"] = "stamp unavailable"
        return row
    row.update({
        "expected_value": stamp.get("expected_value"),
        "ev_per_dollar_risk": stamp.get("ev_per_dollar_risk"),
        "probability_of_profit": stamp.get("probability_of_profit"),
        # A long option's upside is unbounded; max_profit stays None.
        "max_profit": None,
        "max_loss": stamp.get("max_loss"),
        "liquidity_score": _liquidity_score([quote]),
        "status": ev_engine.STATUS_OK,
        "reason": "single-leg gambler's-ruin EV",
    })
    return row


def _spread_row(symbol: str, strategy: str, proposal,
                spot: float, sigma: float, days: Optional[int], mu: float,
                volatility_edge: Optional[float],
                ev_config, cost_model) -> dict:
    row = _blank_row(symbol, strategy)
    row["volatility_edge"] = volatility_edge
    if proposal is None or getattr(proposal, "strategy_name", sb.NO_TRADE) == sb.NO_TRADE:
        row["reason"] = getattr(proposal, "reason", "") or "structure not buildable"
        return row
    result = ev_engine.evaluate_proposal(
        proposal, spot, sigma, days=days, mu=mu,
        volatility_edge=volatility_edge, config=ev_config,
        cost_model=cost_model)
    d = result.to_dict()
    row.update({
        "expected_value": d.get("expected_value"),
        "ev_per_dollar_risk": d.get("ev_per_dollar_risk"),
        "probability_of_profit": d.get("probability_of_profit"),
        "max_profit": d.get("max_profit"),
        "max_loss": d.get("max_loss"),
        "oracle_score": d.get("oracle_score"),
        "liquidity_score": _liquidity_score(
            [{"bid": l.bid, "ask": l.ask} for l in proposal.legs]),
        "status": d.get("status"),
        "reason": d.get("reason") or "",
        "ev_recommendation": d.get("recommendation"),
        "estimated_costs": d.get("estimated_costs"),
    })
    return row


# ---------------------------------------------------------------------------
# Annotation: advisory + RL (both fail-open, both read-only)
# ---------------------------------------------------------------------------
def _annotate_advisory(row: dict, dte: Optional[int], iv_rank: Optional[float],
                       advisory_fn: Optional[Callable]) -> None:
    if advisory_fn is None or row.get("status") != ev_engine.STATUS_OK:
        return
    if row["strategy"] == SKIP:
        return
    try:
        verdict = advisory_fn(
            oracle_score=row.get("oracle_score"),
            volatility_edge=row.get("volatility_edge"),
            dte=dte, iv_rank=iv_rank, strategy=row["strategy"],
            trades=[],
            expected_value=row.get("expected_value"),
            ev_per_dollar_risk=row.get("ev_per_dollar_risk"),
            probability_of_profit=row.get("probability_of_profit"),
            estimated_costs=row.get("estimated_costs"),
            ev_recommendation=row.get("ev_recommendation"))
        if isinstance(verdict, dict):
            row["advisory_recommendation"] = verdict.get("recommendation")
    except Exception:
        row["advisory_recommendation"] = None


def _annotate_rl(row: dict, rl_agent, state_key: Optional[str]) -> None:
    if rl_agent is None or not state_key:
        return
    try:
        row["rl_state_q"] = round(
            float(rl_agent.get_q(state_key, row["rl_action"])), 4)
    except Exception:
        row["rl_state_q"] = None


# ---------------------------------------------------------------------------
# Ranking core (pure, offline-testable)
# ---------------------------------------------------------------------------
def _rank_key(row: dict):
    """Sort key: rankable rows first, then EV/Risk desc, then EV desc.

    Rows without numeric EV/Risk (un-buildable / insufficient) sort last but are
    still returned, so the matrix never silently drops a candidate."""
    evr = row.get("ev_per_dollar_risk")
    ev = row.get("expected_value")
    rankable = evr is not None and ev is not None
    return (
        1 if rankable else 0,
        evr if evr is not None else float("-inf"),
        ev if ev is not None else float("-inf"),
    )


def rank_candidates(rows: List[dict]) -> List[dict]:
    """Rank candidate rows by EV/Risk then EV (descending) and stamp
    ``final_rank`` (1 = best). Pure: mutates only ``final_rank`` and returns a
    new ordered list. SKIP's zero EV makes it win when all else is negative."""
    ordered = sorted(rows, key=_rank_key, reverse=True)
    for i, row in enumerate(ordered, start=1):
        row["final_rank"] = i
    return ordered


# ---------------------------------------------------------------------------
# Builder
# ---------------------------------------------------------------------------
def build_ev_matrix(symbol: str, spot, sigma, chain, *,
                    days: Optional[int] = None, mu: float = 0.0,
                    volatility_edge: Optional[float] = None,
                    iv_rank: Optional[float] = None,
                    dte: Optional[int] = None,
                    take_profit_pct: float = DEFAULT_TP,
                    stop_loss_pct: float = DEFAULT_SL,
                    spread_config=None, ev_config=None, cost_model=None,
                    rl_agent=None, rl_state_key=None,
                    advisory: bool = True,
                    advisory_fn: Optional[Callable] = None) -> dict:
    """Build + EV-score + rank every candidate structure for one underlying.

    ``chain`` is an INJECTED list of option-quote dicts (option_type/strike/bid/
    ask/delta/open_interest/volume). Nothing here touches the network; the
    Telegram handler is responsible for supplying a live chain. Never raises.
    """
    symbol = str(symbol or "")
    try:
        s = _f(spot)
        vol = _f(sigma)
        cfg = spread_config or sb.SpreadConfig.from_env()
        rows: List[dict] = [_skip_row(symbol)]

        if s is None or s <= 0 or vol is None or vol <= 0:
            ranked = rank_candidates(rows)
            return _report(symbol, s, vol, days, ranked,
                           reason="missing spot/volatility")

        split = _split_chain(chain)
        calls, puts = split["call"], split["put"]
        ci = _nearest_idx(calls, s)
        pi = _nearest_idx(puts, s)

        # --- single legs -------------------------------------------------- #
        rows.append(_single_leg_row(symbol, LONG_CALL, _pick(calls, ci, 0),
                                    volatility_edge, take_profit_pct, stop_loss_pct))
        rows.append(_single_leg_row(symbol, LONG_PUT, _pick(puts, pi, 0),
                                    volatility_edge, take_profit_pct, stop_loss_pct))

        # --- spreads (build then EV-score each) --------------------------- #
        builds = []
        lc0, lc1, lc2 = _pick(calls, ci, 0), _pick(calls, ci, 1), _pick(calls, ci, 2)
        lp0, lpm1, lpm2 = _pick(puts, pi, 0), _pick(puts, pi, -1), _pick(puts, pi, -2)

        if lc0 and lc1:
            builds.append((sb.DEBIT_CALL_SPREAD,
                           sb.build_debit_call_spread(_to_leg(lc0), _to_leg(lc1), cfg, symbol)))
        else:
            builds.append((sb.DEBIT_CALL_SPREAD, None))

        if lp0 and lpm1:
            builds.append((sb.DEBIT_PUT_SPREAD,
                           sb.build_debit_put_spread(_to_leg(lp0), _to_leg(lpm1), cfg, symbol)))
        else:
            builds.append((sb.DEBIT_PUT_SPREAD, None))

        if lpm1 and lpm2:
            builds.append((sb.BULLISH_PUT_CREDIT_SPREAD,
                           sb.build_bull_put_credit_spread(_to_leg(lpm1), _to_leg(lpm2), cfg, symbol)))
        else:
            builds.append((sb.BULLISH_PUT_CREDIT_SPREAD, None))

        if lc1 and lc2:
            builds.append((sb.BEARISH_CALL_CREDIT_SPREAD,
                           sb.build_bear_call_credit_spread(_to_leg(lc1), _to_leg(lc2), cfg, symbol)))
        else:
            builds.append((sb.BEARISH_CALL_CREDIT_SPREAD, None))

        if lpm2 and lpm1 and lc1 and lc2:
            builds.append((sb.IRON_CONDOR,
                           sb.build_iron_condor(_to_leg(lpm2), _to_leg(lpm1),
                                                _to_leg(lc1), _to_leg(lc2), cfg, symbol)))
        else:
            builds.append((sb.IRON_CONDOR, None))

        for strat, proposal in builds:
            rows.append(_spread_row(symbol, strat, proposal, s, vol, days, mu,
                                    volatility_edge, ev_config, cost_model))

        # --- annotate (advisory + RL), then rank -------------------------- #
        afn = None
        if advisory:
            afn = advisory_fn
            if afn is None:
                try:
                    from advisory_gate import evaluate_setup as afn  # type: ignore
                except Exception:
                    afn = None
        for row in rows:
            _annotate_advisory(row, dte, iv_rank, afn)
            _annotate_rl(row, rl_agent, rl_state_key)

        ranked = rank_candidates(rows)
        return _report(symbol, s, vol, days, ranked)
    except Exception as exc:  # absolute fail-open
        return _report(symbol, _f(spot), _f(sigma), days,
                       rank_candidates([_skip_row(symbol)]),
                       reason=f"error: {exc}")


def _report(symbol, spot, sigma, days, ranked, reason: str = "") -> dict:
    best = ranked[0] if ranked else None
    return {
        "symbol": symbol,
        "spot": spot,
        "sigma": sigma,
        "days": days,
        "candidates": ranked,
        "best": best,
        "best_strategy": best["strategy"] if best else None,
        "recommend_skip": bool(best and best["strategy"] == SKIP),
        "note": reason,
    }


# ---------------------------------------------------------------------------
# Telegram formatting (pure)
# ---------------------------------------------------------------------------
def _money(value) -> str:
    v = _f(value)
    if v is None:
        return "n/a"
    return f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"


def _ratio(value) -> str:
    v = _f(value)
    return f"{v:+.4f}" if v is not None else "n/a"


def _pct(value) -> str:
    v = _f(value)
    return f"{v * 100:.0f}%" if v is not None else "n/a"


def format_ev_matrix(report: dict) -> str:
    """Telegram-ready STRATEGY_EV_MATRIX. Pure formatting."""
    header = "🧮 *Strategy EV Matrix* _(analytics)_"
    footer = f"_{ANALYTICS_FOOTER}_"
    symbol = report.get("symbol") or "?"
    cands = report.get("candidates") or []
    if not cands:
        return "\n".join([header, "", f"No candidates for `{symbol}`.",
                          "", footer])

    spot, sigma = report.get("spot"), report.get("sigma")
    ctx = (f"`{symbol}` spot `{_f(spot):.2f}`" if _f(spot) is not None
           else f"`{symbol}`")
    if _f(sigma) is not None:
        ctx += f" · σ `{_f(sigma):.4f}`"
    if report.get("days") is not None:
        ctx += f" · {report['days']}d"

    lines = [header, "", ctx, "", "*Ranked by EV/Risk then EV:*"]
    for row in cands:
        rank = row.get("final_rank")
        evr = _ratio(row.get("ev_per_dollar_risk"))
        ev = _money(row.get("expected_value"))
        pop = _pct(row.get("probability_of_profit"))
        adv = row.get("advisory_recommendation") or "n/a"
        rlq = row.get("rl_state_q")
        rl_str = f"{rlq:+.4f}" if isinstance(rlq, (int, float)) else "n/a"
        tag = " ⟵ SKIP" if row.get("strategy") == SKIP else ""
        lines.append(
            f"`{rank}.` `{row.get('strategy')}` "
            f"[{row.get('direction_bias')}] — EV/Risk `{evr}`, "
            f"EV `{ev}`, PoP `{pop}`, adv `{adv}`, "
            f"RL Q({row.get('rl_action')}) `{rl_str}`{tag}")

    best = report.get("best") or {}
    if report.get("recommend_skip"):
        verdict = "SKIP — every structure is non-positive EV"
    else:
        verdict = (f"`{best.get('strategy')}` "
                   f"(EV/Risk `{_ratio(best.get('ev_per_dollar_risk'))}`)")
    lines += ["", f"*EV-first pick:* {verdict}"]
    if report.get("note"):
        lines.append(f"_note: {report['note']}_")
    lines += ["", footer]
    return "\n".join(lines)


def generate_strategy_ev_matrix_text(symbol: str,
                                     chain_provider: Optional[Callable] = None,
                                     **kwargs) -> str:
    """Top-level entry for the STRATEGY_EV_MATRIX Telegram command.

    ``chain_provider(symbol) -> dict`` supplies live market context
    ``{spot, sigma, chain, days?, volatility_edge?, iv_rank?, dte?,
    rl_agent?, rl_state_key?}``. It is INJECTED so this module never reaches the
    network itself. With no provider (or on any failure) a clean advisory
    message is returned — never an exception.
    """
    symbol = str(symbol or "").upper()
    if not symbol:
        return format_ev_matrix({"symbol": "", "candidates": []})
    if chain_provider is None:
        return "\n".join([
            "🧮 *Strategy EV Matrix* _(analytics)_", "",
            f"No live chain provider wired for `{symbol}`.",
            "", f"_{ANALYTICS_FOOTER}_"])
    try:
        ctx = chain_provider(symbol) or {}
        report = build_ev_matrix(
            symbol, ctx.get("spot"), ctx.get("sigma"), ctx.get("chain"),
            days=ctx.get("days"), volatility_edge=ctx.get("volatility_edge"),
            iv_rank=ctx.get("iv_rank"), dte=ctx.get("dte"),
            rl_agent=ctx.get("rl_agent"), rl_state_key=ctx.get("rl_state_key"),
            **kwargs)
        return format_ev_matrix(report)
    except Exception as exc:
        return (f"🧮 *Strategy EV Matrix* _(analytics)_\n\n"
                f"Could not build matrix for `{symbol}`: {exc}\n\n"
                f"_{ANALYTICS_FOOTER}_")


# ---------------------------------------------------------------------------
# Self-test (no creds, no network)
# ---------------------------------------------------------------------------
def _synthetic_chain(spot: float) -> List[dict]:
    """A small symmetric chain around spot with tight quotes."""
    chain = []
    for k in range(-3, 4):
        strike = round(spot + k * 5.0, 2)
        # crude intrinsic + time value so quotes are monotone & two-sided
        call_mid = max(0.5, (spot - strike) * 0.5 + 6.0 - abs(k))
        put_mid = max(0.5, (strike - spot) * 0.5 + 6.0 - abs(k))
        chain.append({"option_type": "call", "strike": strike,
                      "bid": round(call_mid - 0.1, 2), "ask": round(call_mid + 0.1, 2),
                      "delta": 0.5, "open_interest": 500, "volume": 200})
        chain.append({"option_type": "put", "strike": strike,
                      "bid": round(put_mid - 0.1, 2), "ask": round(put_mid + 0.1, 2),
                      "delta": -0.5, "open_interest": 500, "volume": 200})
    return chain


def _self_test() -> int:
    ok = True
    spot = 100.0
    chain = _synthetic_chain(spot)

    rep = build_ev_matrix("SPY", spot, 0.25, chain, days=30,
                          volatility_edge=0.01, advisory=False)
    cands = rep["candidates"]

    # Ranks are dense 1..N.
    ranks = sorted(r["final_rank"] for r in cands)
    if ranks != list(range(1, len(cands) + 1)):
        print("FAIL: ranks not dense", ranks); ok = False

    str016 = {r["strategy"] for r in cands}
    # Both single legs evaluated before any choice is made.
    if LONG_CALL not in str016 or LONG_PUT not in str016:
        print("FAIL: call & put not both present"); ok = False
    if SKIP not in str016:
        print("FAIL: SKIP candidate missing"); ok = False

    # Ranking core: higher EV/Risk must rank ahead of lower.
    synth = [
        {"strategy": "a", "expected_value": 10.0, "ev_per_dollar_risk": 0.20},
        {"strategy": "b", "expected_value": 5.0, "ev_per_dollar_risk": 0.05},
        {"strategy": SKIP, "expected_value": 0.0, "ev_per_dollar_risk": 0.0},
    ]
    ranked = rank_candidates(synth)
    if ranked[0]["strategy"] != "a" or ranked[-1]["strategy"] != SKIP:
        print("FAIL: ranking order wrong", [r["strategy"] for r in ranked]); ok = False

    # All-negative -> SKIP wins.
    neg = [
        {"strategy": "a", "expected_value": -5.0, "ev_per_dollar_risk": -0.10},
        {"strategy": "b", "expected_value": -2.0, "ev_per_dollar_risk": -0.04},
        {"strategy": SKIP, "expected_value": 0.0, "ev_per_dollar_risk": 0.0},
    ]
    if rank_candidates(neg)[0]["strategy"] != SKIP:
        print("FAIL: SKIP should win when all negative"); ok = False

    # Empty / bad inputs fail open (still returns SKIP).
    empty = build_ev_matrix("X", None, None, None)
    if not empty["candidates"] or empty["best_strategy"] != SKIP:
        print("FAIL: bad input should fail open to SKIP"); ok = False

    # Formatter never raises.
    _ = format_ev_matrix(rep)
    _ = generate_strategy_ev_matrix_text("SPY")  # no provider -> clean message

    print("strategy_ev_matrix self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
