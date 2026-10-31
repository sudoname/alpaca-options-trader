"""
TradingAgents shadow recorder + observability.  SHADOW ONLY — no trade impact.

Hypothesis under test (evidence gate in TRADINGAGENTS_SHADOW.md):
    Daily BUY/SELL/HOLD decisions produced by the TradingAgents multi-agent
    LLM framework (Tauric Research, arXiv 2412.20138) carry directional
    information about forward underlying returns.  Until the pre-registered
    gate passes, these decisions influence NOTHING.

Why forward-only: LLMs may "remember" historical price paths from
pretraining, so a historical backtest of LLM decisions is contaminated by
construction.  The only honest evaluation is decisions recorded BEFORE the
outcome, which is exactly what this ledger provides.

What this module does:
    * ``--scan``: for each symbol in the universe, asks TradingAgents for a
      decision as-of today and appends one JSONL row per symbol (decision or
      error) to the shadow ledger.  The full candidate set is recorded —
      no survivorship.
    * ``--eval``: joins recorded decisions with LATER official daily closes
      (Alpaca, 'all'-adjusted via session_returns.fetch_daily_bars) and
      reports per-label forward returns plus the pre-registered primary
      metric: mean signed (BUY=+1, SELL=-1) 5-day forward return,
      gate = n>=40 non-HOLD decisions AND mean>0 AND t>=2.
    * ``--status``: ledger counts by label, errors, last run.
    * ``--research SYMBOL``: one interactive run printing the agent reports
      (for the Telegram RESEARCH command / manual reading). Decision support
      for the human only.

Honesty / safety invariants (same conventions as the session/IV-RV work):
    * Feature flag ``ENABLE_TRADINGAGENTS_SHADOW`` defaults OFF — when off,
      ``scan_once`` does nothing and writes nothing.
    * No imports from the live order path.  The graph is an injected factory
      (duck-typed) so tests never touch the network or the real package.
    * ``tradingagents`` is imported lazily; package or API key absent ->
      clean no-op with a logged reason.
    * Everything fails open: per-symbol errors are recorded and skipped.
    * Decisions are recorded BEFORE outcomes are known; evaluation uses only
      closes strictly after the decision date.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence, Tuple

from config_loader import ConfigLoader

LOG_TAG = "[TA_SHADOW]"

DEFAULT_UNIVERSE = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD"
ROW_TYPE = "tradingagents_decision"

BUY, SELL, HOLD, UNKNOWN = "BUY", "SELL", "HOLD", "UNKNOWN"
LABELS = (BUY, SELL, HOLD)
HORIZONS = (1, 5)          # forward trading-day horizons (5d = primary)
PRIMARY_HORIZON = 5
GATE_MIN_N = 40            # non-HOLD decisions required
GATE_MIN_T = 2.0

# llm_provider -> env var that must exist for a scan to be attempted.
PROVIDER_KEY_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GOOGLE_API_KEY",
    "deepseek": "DEEPSEEK_API_KEY",
    "groq": "GROQ_API_KEY",
    "ollama": None,        # local; no key needed
}

# Report fields printed by --research, in order (best-effort).
RESEARCH_FIELDS = (
    "market_report",
    "sentiment_report",
    "news_report",
    "fundamentals_report",
    "trader_investment_plan",
    "final_trade_decision",
)


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class TAShadowConfig:
    enabled: bool = False                  # ENABLE_TRADINGAGENTS_SHADOW
    universe: List[str] = field(default_factory=list)   # TA_UNIVERSE
    llm_provider: str = "openai"           # TA_LLM_PROVIDER
    deep_model: str = "gpt-4o-mini"        # TA_DEEP_MODEL (cheapest viable)
    quick_model: str = "gpt-4o-mini"       # TA_QUICK_MODEL
    max_debate_rounds: int = 1             # TA_MAX_DEBATE_ROUNDS
    max_symbols_per_run: int = 10          # TA_MAX_SYMBOLS_PER_RUN
    ledger_file: str = "tradingagents_shadow.jsonl"     # TA_LEDGER_FILE
    eval_file: str = "tradingagents_eval.json"          # TA_EVAL_FILE

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "TAShadowConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        uni = cfg.get_str("TA_UNIVERSE", DEFAULT_UNIVERSE)
        symbols = [s.strip().upper() for s in uni.split(",") if s.strip()]
        return TAShadowConfig(
            enabled=cfg.get_bool("ENABLE_TRADINGAGENTS_SHADOW", False),
            universe=symbols,
            llm_provider=cfg.get_str("TA_LLM_PROVIDER", "openai").lower(),
            deep_model=cfg.get_str("TA_DEEP_MODEL", "gpt-4o-mini"),
            quick_model=cfg.get_str("TA_QUICK_MODEL", "gpt-4o-mini"),
            max_debate_rounds=max(1, cfg.get_int("TA_MAX_DEBATE_ROUNDS", 1)),
            max_symbols_per_run=max(1, cfg.get_int("TA_MAX_SYMBOLS_PER_RUN",
                                                   10)),
            ledger_file=cfg.get_str("TA_LEDGER_FILE",
                                    "tradingagents_shadow.jsonl"),
            eval_file=cfg.get_str("TA_EVAL_FILE", "tradingagents_eval.json"),
        )


def api_key_missing_reason(cfg: TAShadowConfig) -> Optional[str]:
    """Human-readable reason when the provider's API key is absent."""
    env_name = PROVIDER_KEY_ENV.get(cfg.llm_provider, "OPENAI_API_KEY")
    if env_name is None:
        return None
    if os.environ.get(env_name, "").strip():
        return None
    # ConfigLoader-style .env fallback so cron works without exported vars.
    try:
        if ConfigLoader().get_str(env_name, "").strip():
            return None
    except Exception:
        pass
    return (f"no {env_name} configured for provider "
            f"'{cfg.llm_provider}' — scan skipped")


# --------------------------------------------------------------------------- #
# Pure functions (offline-testable)
# --------------------------------------------------------------------------- #
def normalize_label(decision) -> str:
    """Map a raw TradingAgents decision to BUY/SELL/HOLD/UNKNOWN."""
    try:
        text = str(decision).upper()
    except Exception:
        return UNKNOWN
    # Longest-first so e.g. "do not BUY, HOLD" resolves by first occurrence.
    hits = [(text.find(lbl), lbl) for lbl in LABELS if text.find(lbl) >= 0]
    if not hits:
        return UNKNOWN
    return min(hits)[1]


def _mean_t(values: Sequence[float]) -> Tuple[int, Optional[float],
                                              Optional[float]]:
    """(n, mean, t) with t = mean / (sd/sqrt(n)); None when undefined."""
    vals = [float(v) for v in values if v is not None and math.isfinite(v)]
    n = len(vals)
    if n == 0:
        return 0, None, None
    mean = sum(vals) / n
    if n < 2:
        return n, mean, None
    var = sum((v - mean) ** 2 for v in vals) / (n - 1)
    sd = math.sqrt(var)
    if sd <= 0:
        return n, mean, None
    return n, mean, mean / (sd / math.sqrt(n))


def forward_returns_for_rows(
    rows: Sequence[Mapping],
    closes_by_symbol: Mapping[str, Sequence[Tuple[str, float]]],
    horizons: Sequence[int] = HORIZONS,
) -> List[Dict]:
    """Attach forward returns to decision rows.  Pure; never raises.

    ``closes_by_symbol``: symbol -> sequence of (iso_date, close), which is
    sorted here before use.  Base price is the official close ON the decision
    date (the scan runs pre-market, so the first attainable price at or after
    the decision is that day's close); forward return over h trading days is
    close[i+h]/close[i] - 1.  Rows whose forward window is not yet complete
    are returned with that horizon set to None (pending).
    """
    out: List[Dict] = []
    index: Dict[str, Tuple[List[str], List[float]]] = {}
    for sym, series in (closes_by_symbol or {}).items():
        try:
            pairs = sorted((str(d), float(c)) for d, c in series
                           if c is not None and float(c) > 0)
        except Exception:
            continue
        index[sym.upper()] = ([d for d, _ in pairs], [c for _, c in pairs])
    for row in rows or []:
        try:
            label = row.get("decision")
            sym = str(row.get("symbol", "")).upper()
            day = str(row.get("date", ""))
            if label not in LABELS or sym not in index or not day:
                continue
            dates, closes = index[sym]
            # First trading day at/after the decision date.
            i = next((k for k, d in enumerate(dates) if d >= day), None)
            if i is None:
                continue
            rec = {"symbol": sym, "date": day, "decision": label,
                   "base_date": dates[i], "base_close": closes[i]}
            for h in horizons:
                j = i + int(h)
                rec[f"fwd_{h}d"] = (closes[j] / closes[i] - 1.0
                                    if j < len(closes) else None)
            out.append(rec)
        except Exception:
            continue
    return out


def evaluate(rows: Sequence[Mapping],
             closes_by_symbol: Mapping[str, Sequence[Tuple[str, float]]],
             horizons: Sequence[int] = HORIZONS) -> Dict:
    """Pre-registered evaluation.  Pure; never raises.

    Primary: mean signed forward return at ``PRIMARY_HORIZON`` over non-HOLD
    decisions (BUY=+1, SELL=-1).  Gate: n>=GATE_MIN_N AND mean>0 AND
    t>=GATE_MIN_T.  Secondary: per-label per-horizon table.
    """
    joined = forward_returns_for_rows(rows, closes_by_symbol, horizons)
    per_label: Dict[str, Dict] = {}
    for lbl in LABELS:
        per_label[lbl] = {}
        for h in horizons:
            vals = [r[f"fwd_{h}d"] for r in joined
                    if r["decision"] == lbl and r.get(f"fwd_{h}d") is not None]
            n, mean, t = _mean_t(vals)
            per_label[lbl][f"{h}d"] = {
                "n": n,
                "mean_bp": None if mean is None else round(mean * 1e4, 2),
                "t": None if t is None else round(t, 2),
            }
    key = f"fwd_{PRIMARY_HORIZON}d"
    signed = [r[key] if r["decision"] == BUY else -r[key]
              for r in joined
              if r["decision"] in (BUY, SELL) and r.get(key) is not None]
    n, mean, t = _mean_t(signed)
    gate_passed = bool(n >= GATE_MIN_N and mean is not None and mean > 0
                       and t is not None and t >= GATE_MIN_T)
    pending = sum(1 for r in joined if r.get(key) is None)
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "primary": {
            "metric": (f"mean signed {PRIMARY_HORIZON}d forward return "
                       "(BUY=+1, SELL=-1), non-HOLD only"),
            "n": n,
            "mean_bp": None if mean is None else round(mean * 1e4, 2),
            "t": None if t is None else round(t, 2),
            "gate": (f"n>={GATE_MIN_N} and mean>0 and t>={GATE_MIN_T}"),
            "gate_passed": gate_passed,
        },
        "per_label": per_label,
        "n_joined": len(joined),
        "n_pending_primary": pending,
        "verdict": ("PASSED — promote only via a new, separately gated step"
                    if gate_passed else
                    "NOT PASSED — decisions remain shadow-only (no trade "
                    "influence)"),
    }


# --------------------------------------------------------------------------- #
# Ledger I/O (append-only JSONL; fail-open)
# --------------------------------------------------------------------------- #
def _append_row(path: str, row: Mapping) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} ledger append ignored: {exc}")


def load_rows(path: str) -> List[Dict]:
    rows: List[Dict] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                if isinstance(obj, dict) and obj.get("type") == ROW_TYPE:
                    rows.append(obj)
    except FileNotFoundError:
        pass
    except Exception as exc:  # pragma: no cover - fail-open
        print(f"{LOG_TAG} ledger read ignored: {exc}")
    return rows


# --------------------------------------------------------------------------- #
# TradingAgents adapter (lazy; injected in tests)
# --------------------------------------------------------------------------- #
def build_graph(cfg: TAShadowConfig):
    """Build a TradingAgentsGraph, or return None with a logged reason."""
    try:
        from tradingagents.default_config import DEFAULT_CONFIG
        from tradingagents.graph.trading_graph import TradingAgentsGraph
    except Exception as exc:
        print(f"{LOG_TAG} tradingagents not importable ({exc}) — scan skipped")
        return None
    try:
        conf = dict(DEFAULT_CONFIG)
        conf["llm_provider"] = cfg.llm_provider
        conf["deep_think_llm"] = cfg.deep_model
        conf["quick_think_llm"] = cfg.quick_model
        conf["max_debate_rounds"] = cfg.max_debate_rounds
        return TradingAgentsGraph(debug=False, config=conf)
    except Exception as exc:
        print(f"{LOG_TAG} graph construction failed ({exc}) — scan skipped")
        return None


def scan_once(cfg: TAShadowConfig,
              graph_factory: Optional[Callable[[], object]] = None,
              today: Optional[date] = None) -> List[Dict]:
    """One shadow scan over the universe.  Returns the appended rows."""
    if not cfg.enabled:
        print(f"{LOG_TAG} disabled (ENABLE_TRADINGAGENTS_SHADOW off) — no-op")
        return []
    reason = api_key_missing_reason(cfg)
    if graph_factory is None and reason:
        print(f"{LOG_TAG} {reason}")
        return []
    graph = graph_factory() if graph_factory else build_graph(cfg)
    if graph is None:
        return []
    day = (today or datetime.now(timezone.utc).date()).isoformat()
    rows: List[Dict] = []
    for sym in cfg.universe[: cfg.max_symbols_per_run]:
        started = time.time()
        row: Dict = {
            "type": ROW_TYPE,
            "recorded_at": datetime.now(timezone.utc).isoformat(),
            "date": day,
            "symbol": sym,
            "provider": cfg.llm_provider,
            "deep_model": cfg.deep_model,
            "quick_model": cfg.quick_model,
        }
        try:
            _state, decision = graph.propagate(sym, day)
            row["decision"] = normalize_label(decision)
            row["raw"] = str(decision)[:300]
        except Exception as exc:
            row["decision"] = None
            row["error"] = f"{type(exc).__name__}: {exc}"[:300]
        finally:
            row["elapsed_sec"] = round(time.time() - started, 1)
            _append_row(cfg.ledger_file, row)
            rows.append(row)
            print(f"{LOG_TAG} {day} {sym}: "
                  f"{row.get('decision') or row.get('error')} "
                  f"({row['elapsed_sec']}s)")
    return rows


# --------------------------------------------------------------------------- #
# CLI plumbing (network only here)
# --------------------------------------------------------------------------- #
def _closes_from_ledger(rows: Sequence[Mapping]) -> Dict[str,
                                                         List[Tuple[str,
                                                                    float]]]:
    """Fetch official daily closes covering the ledger via Alpaca."""
    from session_returns import fetch_daily_bars

    symbols = sorted({str(r.get("symbol", "")).upper()
                      for r in rows if r.get("symbol")})
    days = [str(r.get("date")) for r in rows if r.get("date")]
    if not symbols or not days:
        return {}
    start = date.fromisoformat(min(days)) - timedelta(days=5)
    end = datetime.now(timezone.utc).date()
    bars = fetch_daily_bars(symbols, start, end, adjustment="all")
    out: Dict[str, List[Tuple[str, float]]] = {}
    for sym, df in bars.items():
        try:
            out[sym] = [(d.isoformat(), float(c))
                        for d, c in zip(df["date"], df["close"])]
        except Exception:
            continue
    return out


def run_eval(cfg: TAShadowConfig) -> int:
    rows = load_rows(cfg.ledger_file)
    decided = [r for r in rows if r.get("decision") in LABELS]
    if not decided:
        print(f"{LOG_TAG} no decisions in {cfg.ledger_file} yet — nothing "
              "to evaluate")
        return 0
    closes = _closes_from_ledger(decided)
    report = evaluate(decided, closes)
    try:
        with open(cfg.eval_file, "w", encoding="utf-8") as fh:
            json.dump(report, fh, indent=2)
    except Exception as exc:
        print(f"{LOG_TAG} eval write ignored: {exc}")
    p = report["primary"]
    print(f"{LOG_TAG} decisions joined: {report['n_joined']} "
          f"(pending {PRIMARY_HORIZON}d window: "
          f"{report['n_pending_primary']})")
    for lbl in LABELS:
        cells = "  ".join(
            f"{h}: n={report['per_label'][lbl][h]['n']} "
            f"mean={report['per_label'][lbl][h]['mean_bp']}bp "
            f"t={report['per_label'][lbl][h]['t']}"
            for h in (f"{x}d" for x in HORIZONS))
        print(f"{LOG_TAG}   {lbl:<4} {cells}")
    print(f"{LOG_TAG} PRIMARY {p['metric']}: n={p['n']} "
          f"mean={p['mean_bp']}bp t={p['t']} -> "
          f"{'GATE PASSED' if p['gate_passed'] else 'gate not passed'}")
    print(f"{LOG_TAG} {report['verdict']}")
    return 0


def run_status(cfg: TAShadowConfig) -> int:
    rows = load_rows(cfg.ledger_file)
    by_label: Dict[str, int] = {}
    errors = 0
    for r in rows:
        if r.get("error"):
            errors += 1
        lbl = r.get("decision") or "ERROR"
        by_label[lbl] = by_label.get(lbl, 0) + 1
    last = rows[-1].get("recorded_at") if rows else None
    print(f"{LOG_TAG} ledger={cfg.ledger_file} rows={len(rows)} "
          f"errors={errors} last={last}")
    for lbl, n in sorted(by_label.items()):
        print(f"{LOG_TAG}   {lbl}: {n}")
    non_hold = sum(n for lbl, n in by_label.items() if lbl in (BUY, SELL))
    print(f"{LOG_TAG} evidence-gate progress: {non_hold}/{GATE_MIN_N} "
          "non-HOLD decisions")
    return 0


def run_research(cfg: TAShadowConfig, symbol: str) -> int:
    """One interactive research run; prints agent reports.  Human-only."""
    reason = api_key_missing_reason(cfg)
    if reason:
        print(f"{LOG_TAG} {reason}")
        return 1
    graph = build_graph(cfg)
    if graph is None:
        return 1
    day = datetime.now(timezone.utc).date().isoformat()
    sym = symbol.strip().upper()
    print(f"{LOG_TAG} research {sym} as of {day} "
          f"(provider={cfg.llm_provider})")
    try:
        state, decision = graph.propagate(sym, day)
    except Exception as exc:
        print(f"{LOG_TAG} research failed: {type(exc).__name__}: {exc}")
        return 1
    for fieldname in RESEARCH_FIELDS:
        try:
            text = (state or {}).get(fieldname)
        except Exception:
            text = None
        if text:
            print(f"\n=== {fieldname} ===\n{str(text).strip()}")
    print(f"\n=== decision ===\n{normalize_label(decision)} "
          f"(raw: {str(decision)[:200]})")
    print("\nShadow-only research. NOT a validated signal; the evidence "
          "gate has not passed.")
    return 0


def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(
        description="TradingAgents shadow recorder (no trade impact)")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--scan", action="store_true",
                       help="record one decision per universe symbol")
    group.add_argument("--eval", action="store_true",
                       help="join decisions with later closes; report gate")
    group.add_argument("--status", action="store_true",
                       help="ledger counts and gate progress")
    group.add_argument("--research", metavar="SYMBOL",
                       help="one verbose run for human reading")
    args = parser.parse_args(argv)
    cfg = TAShadowConfig.from_env()
    if args.scan:
        scan_once(cfg)
        return 0
    if args.eval:
        return run_eval(cfg)
    if args.research:
        return run_research(cfg, args.research)
    return run_status(cfg)


if __name__ == "__main__":
    raise SystemExit(main())
