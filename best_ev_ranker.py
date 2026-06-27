"""
Phase 10B — Best EV Ranking.  ADVISORY ANALYTICS ONLY.

Ranks candidate defined-risk spread structures across a symbol universe by the
Phase 10A EV engine's numbers. The unit of ranking is the *structure*, not the
symbol: NVDA may rank #1 as a bull put credit spread while ranking nowhere as
anything else.

Pipeline (one pass):
  symbols -> trader_factory(symbol).propose_spread()   (existing proposal logic)
          -> ev_engine.evaluate_for_symbol()           (PoP / EV / EV-per-risk)
          -> filter (recommendation tier, EV-positive unless include-negative)
          -> sort  (ev_per_dollar_risk, expected_value, oracle_score, costs)

Ranked objects are plain ``ev_engine.EVResult`` rows, which already carry every
required output field: symbol, strategy, expected_value, probability_of_profit,
ev_per_dollar_risk, max_profit, max_loss, estimated_costs, oracle_score,
volatility_edge, days (DTE), recommendation, reason.

HARD SCOPE RULE (Phase 10B): nothing here may touch execution. This module
never imports smart_trader (the trader is duck-typed via ``trader_factory``),
never places/sizes/gates a trade, and is consumed ONLY by the Telegram
``BEST_EV_TRADES`` analytics command. ``run_alpaca_intraday`` and
``smart_trader`` must NOT import it (statically guarded by
test_best_ev_ranker.TestNoExecutionPathTouched).

Everything fails open: a symbol that errors is skipped; bad config falls back
to defaults; an empty universe produces an explanatory message, never a raise.
"""

import re
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import ev_engine
from ev_engine import (
    EVConfig, EVResult,
    STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE,
    STATUS_OK,
)

# Tier ordering for the min-recommendation filter (higher = better).
_TIER_RANK = {
    REJECT_CANDIDATE: 0,
    WEAK_SETUP: 1,
    NEUTRAL: 2,
    ACCEPT: 3,
    STRONG_ACCEPT: 4,
}

_SYMBOL_RE = re.compile(r"^[A-Z]{1,5}$")

NO_CANDIDATES_MESSAGE = "No EV-positive candidates found."
FOOTER = "_Advisory only — no orders placed._"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class BestEVConfig:
    max_symbols: int = 25          # BEST_EV_MAX_SYMBOLS — universe scan cap
    min_recommendation: str = NEUTRAL  # BEST_EV_MIN_RECOMMENDATION — tier floor
    include_negative: bool = False     # BEST_EV_INCLUDE_NEGATIVE
    top_n: int = 5                 # entries shown in the Telegram report

    @staticmethod
    def from_env(path: str = ".env", loader=None) -> "BestEVConfig":
        from config_loader import ConfigLoader
        cfg = loader if loader is not None else ConfigLoader(path=path)
        min_rec = cfg.get_str("BEST_EV_MIN_RECOMMENDATION", NEUTRAL).strip().upper()
        if min_rec not in _TIER_RANK:
            min_rec = NEUTRAL
        return BestEVConfig(
            max_symbols=max(1, cfg.get_int("BEST_EV_MAX_SYMBOLS", 25)),
            min_recommendation=min_rec,
            include_negative=cfg.get_bool("BEST_EV_INCLUDE_NEGATIVE", False),
            top_n=max(1, cfg.get_int("BEST_EV_TOP_N", 5)),
        )


# --------------------------------------------------------------------------- #
# Symbol parsing / default universe
# --------------------------------------------------------------------------- #
def parse_symbols(raw: Union[str, Sequence[str], None]) -> List[str]:
    """Normalize a symbol universe: split on commas/whitespace, uppercase,
    drop invalid tokens, dedupe preserving order. Accepts a string or a list.
    """
    if raw is None:
        return []
    tokens: List[str] = []
    if isinstance(raw, str):
        tokens = raw.replace(",", " ").split()
    else:
        for item in raw:
            tokens.extend(str(item).replace(",", " ").split())
    out: List[str] = []
    seen = set()
    for tok in tokens:
        sym = tok.strip().upper()
        if _SYMBOL_RE.fullmatch(sym) and sym not in seen:
            seen.add(sym)
            out.append(sym)
    return out


def default_universe(path: str = ".env", loader=None) -> List[str]:
    """The Oracle universe: SCHEDULER_SYMBOLS from config, SPY/QQQ fallback."""
    try:
        from config_loader import ConfigLoader
        cfg = loader if loader is not None else ConfigLoader(path=path)
        syms = parse_symbols(cfg.get_str("SCHEDULER_SYMBOLS", ""))
        return syms or ["SPY", "QQQ"]
    except Exception:
        return ["SPY", "QQQ"]


# --------------------------------------------------------------------------- #
# Scan + rank (pure given a trader_factory; per-symbol failures are skipped)
# --------------------------------------------------------------------------- #
def scan_universe(symbols: Sequence[str],
                  trader_factory: Callable[[str], object],
                  ev_config: Optional[EVConfig] = None) -> List[EVResult]:
    """EV-evaluate each symbol's best spread proposal. Never raises."""
    results: List[EVResult] = []
    for symbol in symbols:
        try:
            trader = trader_factory(symbol)
            results.append(ev_engine.evaluate_for_symbol(trader, symbol,
                                                         config=ev_config))
        except Exception as e:
            print(f"[BEST_EV] {symbol} skipped: {e}")
    return results


def _tier(recommendation: str) -> int:
    return _TIER_RANK.get((recommendation or "").strip().upper(), -1)


def _neg_inf_if_none(x) -> float:
    return x if isinstance(x, (int, float)) else float("-inf")


def _sort_key(r: EVResult) -> Tuple:
    """Best first: EV/risk, then EV, then oracle score desc; then cost asc
    (cheaper execution = better liquidity/cost quality)."""
    return (
        -_neg_inf_if_none(r.ev_per_dollar_risk),
        -_neg_inf_if_none(r.expected_value),
        -_neg_inf_if_none(r.oracle_score),
        r.estimated_costs if isinstance(r.estimated_costs, (int, float))
        else float("inf"),
    )


def rank_candidates(results: Sequence[EVResult],
                    config: Optional[BestEVConfig] = None) -> List[EVResult]:
    """Filter to acceptable candidates and sort best-first.

    Keeps only rows with status='ok' and a computed EV, at or above the
    configured recommendation tier. Unless ``include_negative`` is set, only
    EV-positive candidates survive.
    """
    cfg = config or BestEVConfig()
    floor = _tier(cfg.min_recommendation)
    kept = []
    for r in results or []:
        if r is None or r.status != STATUS_OK or r.expected_value is None:
            continue
        if _tier(r.recommendation) < floor:
            continue
        if not cfg.include_negative and r.expected_value <= 0:
            continue
        kept.append(r)
    return sorted(kept, key=_sort_key)


# --------------------------------------------------------------------------- #
# Phase 11B — candlestick stamping (ANALYTICS ONLY).
#
# Opt-in: when enabled, the ranker fetches recent daily bars per unique ranked
# symbol, runs the pure candlestick detector, and freezes the 6 derived
# ``candlestick_*`` fields onto each candidate via ``extras``. Raw candles are
# NEVER persisted. This is a market-behaviour annotation only — it can never
# change ranking, EV, PoP, risk, advisory, gates, sizing, or any order. Fully
# fail-open: missing creds / network errors / detector errors -> {}.
# --------------------------------------------------------------------------- #
def _to_candle(bar) -> dict:
    """Coerce a bar (``Bar`` namedtuple / object / dict) to an OHLCV dict.

    ``Bar`` is a NamedTuple, so it would mis-parse via the detector's tuple
    branch (its first field is the date string). Converting to a dict here
    keeps the shared detector untouched and parses by name.
    """
    if isinstance(bar, dict):
        return bar
    return {
        "o": getattr(bar, "o", getattr(bar, "open", None)),
        "h": getattr(bar, "h", getattr(bar, "high", None)),
        "l": getattr(bar, "l", getattr(bar, "low", None)),
        "c": getattr(bar, "c", getattr(bar, "close", None)),
        "v": getattr(bar, "v", getattr(bar, "volume", None)),
    }


def _default_bar_provider(config=None):
    """Build a daily-bar fetcher ``(symbol, lookback) -> bars`` from env creds.

    Returns None when credentials are absent or anything fails (offline-safe).
    Never raises.
    """
    try:
        from config_loader import ConfigLoader
        c = ConfigLoader()
        key = c.get_str("ALPACA_API_KEY", "")
        secret = c.get_str("ALPACA_SECRET_KEY", "")
        if not key or not secret:
            return None
        from market_view import LiveMarketView
        mv = LiveMarketView(headers={
            "APCA-API-KEY-ID": key,
            "APCA-API-SECRET-KEY": secret,
        })
        return lambda symbol, lookback: mv.daily_bars(symbol, lookback)
    except Exception:
        return None


def _candlestick_extras(ranked: Sequence[EVResult], *,
                        bar_provider=None, config=None, now=None) -> dict:
    """Map ``candidate_key -> {6 frozen candlestick fields}``. Analytics only.

    Fetches recent daily bars once per unique ranked symbol and runs the pure
    detector. Returns ``{}`` when disabled, creds/provider missing, or nothing
    detected. Never raises and never affects ranking/EV/orders.
    """
    try:
        from oracle.signals import candlestick_patterns as csp
        cfg = config or csp.CandlestickConfig.from_env()
        if not (getattr(cfg, "enabled", False)
                and getattr(cfg, "fetch_in_ranker", False)):
            return {}
        provider = bar_provider or _default_bar_provider(cfg)
        if provider is None:
            return {}
        import candidate_resolution as cr
        from datetime import datetime, timezone
        ts = now or datetime.now(timezone.utc)
        day = ts.strftime("%Y-%m-%d")
        lookback = getattr(cfg, "ranker_lookback", 15) or 15

        extras: dict = {}
        stamp_by_symbol: dict = {}
        for r in ranked or []:
            symbol = getattr(r, "symbol", None)
            strategy = getattr(r, "strategy", None)
            if not symbol or not strategy:
                continue
            sym = str(symbol).upper()
            if sym not in stamp_by_symbol:
                stamp = None
                try:
                    bars = provider(sym, lookback)
                    candles = [_to_candle(b) for b in (bars or [])]
                    stamp = csp.detect_primary(candles, cfg)
                except Exception:
                    stamp = None
                stamp_by_symbol[sym] = stamp
            stamp = stamp_by_symbol.get(sym)
            if stamp is None:
                continue
            key = cr.candidate_key(sym, strategy, day)
            extras[key] = {
                "candlestick_pattern": stamp.pattern_name,
                "candlestick_bias": stamp.bias,
                "candlestick_strength": stamp.strength,
                "candlestick_confidence": stamp.confidence,
                "candlestick_reason": stamp.reason,
                "candlestick_requires_confirmation": stamp.requires_confirmation,
            }
        return extras
    except Exception:
        return {}


def run_best_ev(symbols: Union[str, Sequence[str], None],
                trader_factory: Callable[[str], object],
                config: Optional[BestEVConfig] = None,
                ev_config: Optional[EVConfig] = None,
                ) -> Tuple[List[EVResult], int]:
    """Parse -> cap -> scan -> rank. Returns (ranked, symbols_scanned)."""
    cfg = config or BestEVConfig()
    universe = parse_symbols(symbols)[: cfg.max_symbols]
    results = scan_universe(universe, trader_factory, ev_config=ev_config)
    ranked = rank_candidates(results, cfg)
    # Phase 10G-E: persist every evaluated candidate for later resolution.
    # Recording only — cannot affect the ranking or any trade. Fail-open.
    try:
        import candidate_resolution as cr
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc)
        # Phase 11B: freeze candlestick annotations onto the candidates. Opt-in,
        # analytics-only, fail-open — {} when disabled or candles unavailable.
        extras = _candlestick_extras(ranked, now=now)
        cr.record_candidates(ranked, source="best_ev_ranker",
                             extras=extras, now=now)
    except Exception as exc:
        print(f"[BEST_EV] candidate recording skipped: {exc}")
    return ranked, len(universe)


# --------------------------------------------------------------------------- #
# Telegram formatting (analytics text only)
# --------------------------------------------------------------------------- #
def _format_entry(rank: int, r: EVResult) -> str:
    name = ev_engine.display_strategy_name(r.strategy)
    ev_val = r.expected_value if isinstance(r.expected_value, (int, float)) else 0.0
    title = f"{rank}. {r.symbol} {name}"
    if ev_val < 0:
        title += "  ⚠️ NEGATIVE EV"
    lines = [
        title,
        f"EV: {'+' if ev_val >= 0 else '-'}${abs(ev_val):.2f}",
        f"PoP: {round((r.probability_of_profit or 0.0) * 100)}%",
        (f"EV/Risk: {r.ev_per_dollar_risk:.2f}"
         if r.ev_per_dollar_risk is not None else "EV/Risk: n/a"),
    ]
    if r.oracle_score is not None:
        lines.append(f"Score: {r.oracle_score:.0f}")
    if r.days is not None:
        lines.append(f"DTE: {r.days}")
    lines.append(f"Recommendation: {r.recommendation}")
    return "\n".join(lines)


def format_best_ev_report(ranked: Sequence[EVResult], scanned: int = 0,
                          config: Optional[BestEVConfig] = None) -> str:
    """Markdown leaderboard for Telegram. Pure formatting; no side effects."""
    cfg = config or BestEVConfig()
    header = "🏆 *Best EV Trades*"
    scanned_line = f"_Scanned {scanned} symbol(s)._" if scanned else ""

    if not ranked:
        parts = [header, "", NO_CANDIDATES_MESSAGE]
        if scanned_line:
            parts += ["", scanned_line]
        parts += ["", FOOTER]
        return "\n".join(parts)

    entries = [_format_entry(i + 1, r) for i, r in enumerate(ranked[: cfg.top_n])]
    parts = [header, ""]
    parts.append("\n\n".join(entries))
    if scanned_line:
        parts += ["", scanned_line]
    parts += ["", FOOTER]
    return "\n".join(parts)
