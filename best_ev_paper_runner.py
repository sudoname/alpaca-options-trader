"""
Phase 10D — Paper trade the highest-EV structures.  SIMULATION ONLY.

Bridges the Phase 10B Best-EV ranking into the Phase 6C spread paper trader:

    best_ev_ranker.run_best_ev()  ->  paper thresholds  ->
    spread_paper_trader.SpreadPaperTrader.open_position()

Every trade opened here is a SIMULATED spread position written to a local JSON
file. There is no broker client in this module or anywhere below it: the
spread paper trader never submits orders, and this module never imports the
live trader (the proposal source is an injected ``trader_factory``, exactly
like best_ev_ranker). Long call/put execution is untouched. Nothing here can
block, size or alter a real trade.

Feature flag: ``ENABLE_BEST_EV_PAPER_TRADING`` (default OFF). When off,
``run_paper_from_best_ev`` does nothing and reports itself disabled. When on,
the underlying simulator is driven directly by this flag for the run (the
legacy ``USE_SPREAD_PAPER_TRADING`` flag governs the older flow and is not
consulted here) — still simulation only.

Paper thresholds (soft selection of WHICH candidates to simulate — they gate
nothing outside this simulation):
    BEST_EV_PAPER_MAX_TRADES_PER_RUN  (3)
    BEST_EV_PAPER_MIN_RECOMMENDATION  (ACCEPT)
    BEST_EV_PAPER_MIN_EV_PER_RISK     (0.05)
    BEST_EV_PAPER_MIN_EV              (0.00)

Each opened simulated position carries the EV belief at entry
(expected_value / ev_per_dollar_risk / probability_of_profit /
ev_recommendation / estimated_costs) plus the advisory snapshot fields, so
Phase 10C attribution can later answer "did the EV ranking predict outcomes?".

Everything fails open; per-candidate problems are logged and skipped.
"""

import dataclasses
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple, Union

import best_ev_ranker as ber
import ev_engine
from ev_engine import STRONG_ACCEPT, ACCEPT, NEUTRAL, WEAK_SETUP, REJECT_CANDIDATE
from spread_paper_trader import (
    SpreadPaperTrader, SpreadPaperConfig, REASON_OPENED,
)

LOG_TAG = "[BEST_EV_PAPER]"
PAPER_FOOTER = "Simulated paper only — no broker orders placed."

# Skip reasons produced by THIS module (open_position adds its own, e.g.
# duplicate_position / low_oracle_score / invalid_max_loss).
SKIP_DISABLED = "disabled"
SKIP_BELOW_RECOMMENDATION = "below_min_recommendation"
SKIP_EV_BELOW_MIN = "ev_below_min"
SKIP_EV_PER_RISK_BELOW_MIN = "ev_per_risk_below_min"
SKIP_MAX_TRADES = "max_trades_reached"
SKIP_PROPOSAL_CHANGED = "proposal_changed"
SKIP_PROPOSAL_ERROR = "proposal_error"

# EV belief fields stamped onto every opened simulated position.
EV_CONTEXT_FIELDS = (
    "expected_value", "ev_per_dollar_risk", "probability_of_profit",
    "ev_recommendation", "estimated_costs",
)

# Advisory snapshot fields copied (best-effort) onto the position record.
ADVISORY_CONTEXT_FIELDS = (
    "advisory_recommendation", "advisory_confidence", "threshold_checks",
    "historical_win_rate", "historical_profit_factor",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class BestEVPaperConfig:
    enabled: bool = False              # ENABLE_BEST_EV_PAPER_TRADING
    max_trades_per_run: int = 3        # BEST_EV_PAPER_MAX_TRADES_PER_RUN
    min_recommendation: str = ACCEPT   # BEST_EV_PAPER_MIN_RECOMMENDATION
    min_ev_per_risk: float = 0.05      # BEST_EV_PAPER_MIN_EV_PER_RISK
    min_ev: float = 0.00               # BEST_EV_PAPER_MIN_EV

    @staticmethod
    def from_env(path: str = ".env", loader=None) -> "BestEVPaperConfig":
        from config_loader import ConfigLoader
        cfg = loader if loader is not None else ConfigLoader(path=path)
        min_rec = cfg.get_str("BEST_EV_PAPER_MIN_RECOMMENDATION",
                              ACCEPT).strip().upper()
        if min_rec not in ber._TIER_RANK:
            min_rec = ACCEPT
        return BestEVPaperConfig(
            enabled=cfg.get_bool("ENABLE_BEST_EV_PAPER_TRADING", False),
            max_trades_per_run=max(
                0, cfg.get_int("BEST_EV_PAPER_MAX_TRADES_PER_RUN", 3)),
            min_recommendation=min_rec,
            min_ev_per_risk=cfg.get_float("BEST_EV_PAPER_MIN_EV_PER_RISK", 0.05),
            min_ev=cfg.get_float("BEST_EV_PAPER_MIN_EV", 0.00),
        )


def _default_paper_trader() -> SpreadPaperTrader:
    """Simulator instance for this run. ENABLE_BEST_EV_PAPER_TRADING is the
    single gate for this flow, so the simulator itself is enabled for the run
    (file locations and the oracle-score floor still come from env).
    """
    cfg = dataclasses.replace(SpreadPaperConfig.from_env(), enabled=True)
    return SpreadPaperTrader(cfg)


# --------------------------------------------------------------------------- #
# Logging (Req 4) — one line per candidate decision
# --------------------------------------------------------------------------- #
def log_decision(result: "ev_engine.EVResult", action: str, reason: str) -> str:
    line = (
        f"{LOG_TAG} "
        f"symbol={result.symbol} "
        f"strategy={result.strategy} "
        f"expected_value={result.expected_value} "
        f"ev_per_dollar_risk={result.ev_per_dollar_risk} "
        f"recommendation={result.recommendation} "
        f"action={action} "
        f"reason={reason}"
    )
    print(line)
    return line


def _skip_row(r: "ev_engine.EVResult", reason: str) -> dict:
    return {"symbol": r.symbol, "strategy": r.strategy, "reason": reason}


# --------------------------------------------------------------------------- #
# Paper thresholds (select WHICH candidates to simulate; gates nothing real)
# --------------------------------------------------------------------------- #
def passes_paper_thresholds(r: "ev_engine.EVResult",
                            config: BestEVPaperConfig) -> Optional[str]:
    """None when the candidate qualifies, else the skip reason."""
    if ber._tier(r.recommendation) < ber._tier(config.min_recommendation):
        return SKIP_BELOW_RECOMMENDATION
    ev = r.expected_value
    if not isinstance(ev, (int, float)) or ev < config.min_ev:
        return SKIP_EV_BELOW_MIN
    ratio = r.ev_per_dollar_risk
    if not isinstance(ratio, (int, float)) or ratio < config.min_ev_per_risk:
        return SKIP_EV_PER_RISK_BELOW_MIN
    return None


# --------------------------------------------------------------------------- #
# Open path (simulation only)
# --------------------------------------------------------------------------- #
def _persist_position(position: dict, paper_trader: SpreadPaperTrader) -> None:
    """Re-save the (context-enriched) position row by id. Fail-open."""
    try:
        rows = paper_trader.load_positions()
        for i, row in enumerate(rows):
            if row.get("id") == position.get("id"):
                rows[i] = position
                paper_trader.save_positions(rows)
                return
    except Exception as exc:  # pragma: no cover - disk safety
        print(f"{LOG_TAG} persist ignored: {exc}")


def _attach_ev_context(position: dict, r: "ev_engine.EVResult",
                       paper_trader: SpreadPaperTrader) -> None:
    """Stamp the EV belief at entry onto the simulated position, refresh the
    advisory attribution snapshot so it is EV-aware (Phase 10C upserts by
    trade id), and copy the advisory fields back onto the record. Fail-open —
    nothing here can affect whether the position opened.
    """
    position["expected_value"] = r.expected_value
    position["ev_per_dollar_risk"] = r.ev_per_dollar_risk
    position["probability_of_profit"] = r.probability_of_profit
    position["ev_recommendation"] = r.recommendation
    position["estimated_costs"] = r.estimated_costs
    try:
        import advisory_attribution
        snap = advisory_attribution.record_open(position)
        if snap:
            for field_name in ADVISORY_CONTEXT_FIELDS:
                position[field_name] = snap.get(field_name)
    except Exception as exc:
        print(f"{LOG_TAG} attribution hook ignored: {exc}")
    _persist_position(position, paper_trader)


def _open_one(r: "ev_engine.EVResult",
              trader_factory: Callable[[str], object],
              paper_trader: SpreadPaperTrader):
    """(position, reason): position is None when skipped/rejected."""
    try:
        trader = trader_factory(r.symbol)
        proposal = trader.propose_spread(r.symbol)
    except Exception as exc:
        return None, f"{SKIP_PROPOSAL_ERROR}: {exc}"
    if getattr(proposal, "strategy_name", None) != r.strategy:
        # Market moved between ranking and opening — don't open something
        # other than what was ranked.
        return None, SKIP_PROPOSAL_CHANGED

    context = {}
    if r.days is not None:
        context["dte"] = r.days
    if r.volatility_edge is not None:
        context["volatility_edge"] = r.volatility_edge

    result = paper_trader.open_position(proposal, context=context)
    if not result.get("allowed"):
        return None, result.get("reason") or "rejected"
    position = result["position"]
    _attach_ev_context(position, r, paper_trader)
    return position, REASON_OPENED


def open_candidates(ranked: Sequence["ev_engine.EVResult"],
                    trader_factory: Callable[[str], object],
                    paper_trader: SpreadPaperTrader,
                    config: BestEVPaperConfig) -> Tuple[List[dict], List[dict]]:
    """Walk the ranked list best-first, simulating up to max_trades_per_run.

    Returns ``(opened_positions, skipped_rows)``. Never raises.
    """
    opened: List[dict] = []
    skipped: List[dict] = []
    for r in ranked or []:
        if r is None:
            continue
        if len(opened) >= config.max_trades_per_run:
            log_decision(r, "skipped", SKIP_MAX_TRADES)
            skipped.append(_skip_row(r, SKIP_MAX_TRADES))
            continue
        reason = passes_paper_thresholds(r, config)
        if reason:
            log_decision(r, "skipped", reason)
            skipped.append(_skip_row(r, reason))
            continue
        position, why = _open_one(r, trader_factory, paper_trader)
        if position is not None:
            log_decision(r, "opened", REASON_OPENED)
            opened.append(position)
        else:
            log_decision(r, "skipped", why)
            skipped.append(_skip_row(r, why))
    # Phase 10G-E: mark which evaluated candidates became paper trades and
    # enrich them with strikes/expiry/entry price. Recording only; fail-open.
    try:
        import candidate_resolution as cr
        selected, extras = cr.selection_context(opened)
        cr.record_candidates(ranked, selected_keys=selected, extras=extras,
                             source="best_ev_paper_runner")
    except Exception as exc:
        print(f"{LOG_TAG} candidate recording skipped: {exc}")
    return opened, skipped


# --------------------------------------------------------------------------- #
# Top-level run
# --------------------------------------------------------------------------- #
def run_paper_from_best_ev(symbols: Union[str, Sequence[str], None],
                           trader_factory: Optional[Callable[[str], object]],
                           *,
                           config: Optional[BestEVPaperConfig] = None,
                           ranker_config: Optional["ber.BestEVConfig"] = None,
                           ev_config=None,
                           paper_trader: Optional[SpreadPaperTrader] = None,
                           ) -> dict:
    """Rank by EV, then SIMULATE the qualifying top candidates.

    Returns a summary dict: ``{enabled, scanned, candidates, opened, skipped}``
    where ``opened`` is the list of simulated position dicts and ``skipped``
    is ``[{symbol, strategy, reason}, ...]``. Default OFF; never raises.
    """
    cfg = config or BestEVPaperConfig.from_env()
    summary = {"enabled": cfg.enabled, "scanned": 0, "candidates": 0,
               "opened": [], "skipped": []}
    if not cfg.enabled:
        print(f"{LOG_TAG} action=skipped reason={SKIP_DISABLED} "
              f"(set ENABLE_BEST_EV_PAPER_TRADING=true to enable)")
        return summary
    if trader_factory is None:
        summary["skipped"].append({"symbol": "*", "strategy": None,
                                   "reason": "no_trader_factory"})
        return summary

    universe = ber.parse_symbols(symbols) or ber.default_universe()
    ranked, scanned = ber.run_best_ev(universe, trader_factory,
                                      config=ranker_config,
                                      ev_config=ev_config)
    summary["scanned"] = scanned
    summary["candidates"] = len(ranked)

    pt = paper_trader or _default_paper_trader()
    opened, skipped = open_candidates(ranked, trader_factory, pt, cfg)
    summary["opened"] = opened
    summary["skipped"] = skipped
    return summary


# --------------------------------------------------------------------------- #
# Telegram formatting (text only)
# --------------------------------------------------------------------------- #
def _money(value) -> str:
    if not isinstance(value, (int, float)):
        return "n/a"
    return f"{'+' if value >= 0 else '-'}${abs(value):.2f}"


def _opened_line(i: int, p: dict) -> str:
    name = ev_engine.display_strategy_name(p.get("strategy") or "")
    ev = p.get("expected_value")
    ratio = p.get("ev_per_dollar_risk")
    ratio_s = f"{ratio:.2f}" if isinstance(ratio, (int, float)) else "n/a"
    return f"{i}. {p.get('symbol')} {name} — EV {_money(ev)}, EV/Risk {ratio_s}"


def format_paper_run_report(summary: dict) -> str:
    """Markdown summary for Telegram. Pure formatting; no side effects."""
    header = "🧪 *Best EV Paper Run*"
    footer = f"_{PAPER_FOOTER}_"
    if not summary.get("enabled"):
        return "\n".join([
            header, "",
            "Best-EV paper trading is disabled.",
            "Set `ENABLE_BEST_EV_PAPER_TRADING=true` to enable.",
            "", footer,
        ])

    opened = summary.get("opened") or []
    skipped = summary.get("skipped") or []
    lines = [
        header, "",
        f"Scanned: {summary.get('scanned', 0)} symbol(s)",
        f"Candidates: {summary.get('candidates', 0)}",
        f"Opened: {len(opened)} simulated trade(s)",
    ]
    if opened:
        lines += ["", "*Opened:*"]
        lines += [_opened_line(i + 1, p) for i, p in enumerate(opened)]
    if skipped:
        lines += ["", "*Skipped:*"]
        lines += [f"{row.get('symbol')} — {row.get('reason')}"
                  for row in skipped]
    lines += ["", footer]
    return "\n".join(lines)
