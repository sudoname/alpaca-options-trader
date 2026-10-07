"""
IV-RV credit-spread paper experiment.  SIMULATION ONLY — no broker orders.

Hypothesis under test (evidence gate in IVRV_EXPERIMENT.md):
    Defined-risk CREDIT spreads opened only when implied volatility is RICH
    versus realized volatility (Goyal & Saretto, JFE 2009: the cross-section
    of RV-IV predicts option returns) have positive expected net P&L, unlike
    the single-leg long-premium flow (live measured mean: about -15%/trade).

What this module does:
    * Computes trailing annualized realized vol (RV) from 'all'-adjusted daily
      closes and reads ATM implied vol (IV) from the option snapshot.
    * Gates on IV/RV >= min ratio ("rich"); only then asks the EXISTING
      ``smart_trader.propose_spread`` for a structure, and accepts it only if
      it is a CREDIT strategy (bull put / bear call / iron condor).
    * Opens the position in a dedicated :class:`SpreadPaperTrader` store
      (separate files from the legacy spread paper flow) using CONSERVATIVE
      crossed entry quotes: SELL legs fill at BID, BUY legs fill at ASK — the
      simulated entry pays the full quoted spread (Muravyev & Pearson show
      real effective spreads are tighter, so this is a lower bound on edge).
    * At expiration, settles every leg at INTRINSIC value from the official
      daily close of the underlying (no spread at settlement).
    * Appends one JSONL row per SCANNED symbol (opened or not) so the later
      evaluation is computed over the full candidate set, not just winners.

Honesty / safety invariants (same conventions as the session-analysis work):
    * Feature flag ``ENABLE_IVRV_CREDIT_PAPER`` defaults OFF — when off,
      ``scan_once``/``settle_expired`` do nothing and write nothing.
    * No imports from the live order path; the proposal source is an injected
      ``trader_factory`` (duck-typed), exactly like best_ev_paper_runner.
    * Everything fails open: per-symbol errors are recorded and skipped.
    * Results stay UNTESTED until the evidence gate passes:
      >= 40 settled trades AND mean net P&L per $ max-loss > 0 with |t| >= 2.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, List, Mapping, Optional, Sequence

from config_loader import ConfigLoader
from spread_builder import CREDIT_STRATEGIES, NO_TRADE
from spread_paper_trader import (
    SpreadPaperConfig,
    SpreadPaperTrader,
    _leg_key,
)

LOG_TAG = "[IVRV_PAPER]"
EXIT_EXPIRED_INTRINSIC = "expired_intrinsic"

DEFAULT_UNIVERSE = "SPY,QQQ,AAPL,MSFT,NVDA,AMZN,META,GOOGL,TSLA,AMD"

# Verdicts from the richness classifier.
RICH = "rich"
NOT_RICH = "not_rich"
UNKNOWN = "unknown"


# --------------------------------------------------------------------------- #
# Config
# --------------------------------------------------------------------------- #
@dataclass
class IVRVConfig:
    enabled: bool = False                 # ENABLE_IVRV_CREDIT_PAPER
    min_iv_rv_ratio: float = 1.25         # IVRV_MIN_RATIO (Goyal-Saretto gate)
    rv_window: int = 252                  # IVRV_RV_WINDOW trailing sessions
    min_rv_obs: int = 60                  # IVRV_MIN_RV_OBS
    max_opens_per_run: int = 3            # IVRV_MAX_OPENS_PER_RUN
    min_oracle_score: float = 0.0         # IVRV_MIN_ORACLE_SCORE (0 = richness
                                          # gate is the ONLY experiment filter)
    universe: List[str] = field(default_factory=list)   # IVRV_UNIVERSE
    scan_ledger: str = "iv_rv_scan_ledger.jsonl"        # IVRV_SCAN_LEDGER
    positions_file: str = "iv_rv_paper_positions.json"  # IVRV_POSITIONS_FILE
    trades_file: str = "iv_rv_paper_trades.json"        # IVRV_TRADES_FILE

    @staticmethod
    def from_env(path: str = ".env",
                 loader: Optional[ConfigLoader] = None) -> "IVRVConfig":
        cfg = loader if loader is not None else ConfigLoader(path=path)
        uni = cfg.get_str("IVRV_UNIVERSE", DEFAULT_UNIVERSE)
        symbols = [s.strip().upper() for s in uni.split(",") if s.strip()]
        return IVRVConfig(
            enabled=cfg.get_bool("ENABLE_IVRV_CREDIT_PAPER", False),
            min_iv_rv_ratio=cfg.get_float("IVRV_MIN_RATIO", 1.25),
            rv_window=max(2, cfg.get_int("IVRV_RV_WINDOW", 252)),
            min_rv_obs=max(2, cfg.get_int("IVRV_MIN_RV_OBS", 60)),
            max_opens_per_run=max(0, cfg.get_int("IVRV_MAX_OPENS_PER_RUN", 3)),
            min_oracle_score=cfg.get_float("IVRV_MIN_ORACLE_SCORE", 0.0),
            universe=symbols,
            scan_ledger=cfg.get_str("IVRV_SCAN_LEDGER",
                                    "iv_rv_scan_ledger.jsonl"),
            positions_file=cfg.get_str("IVRV_POSITIONS_FILE",
                                       "iv_rv_paper_positions.json"),
            trades_file=cfg.get_str("IVRV_TRADES_FILE",
                                    "iv_rv_paper_trades.json"),
        )


def build_paper_trader(cfg: IVRVConfig) -> SpreadPaperTrader:
    """Dedicated simulator store for this experiment (separate files)."""
    return SpreadPaperTrader(SpreadPaperConfig(
        enabled=cfg.enabled,
        min_oracle_score=cfg.min_oracle_score,
        positions_file=cfg.positions_file,
        trades_file=cfg.trades_file,
    ))


# --------------------------------------------------------------------------- #
# Pure signal functions
# --------------------------------------------------------------------------- #
def annualized_realized_vol(closes: Sequence[float],
                            window: int = 252,
                            min_obs: int = 60) -> Optional[float]:
    """Trailing annualized realized vol from daily closes (log returns).

    Uses up to the last ``window`` returns; returns None when fewer than
    ``min_obs`` returns are available or inputs are unusable.
    """
    try:
        px = [float(c) for c in closes if c is not None and float(c) > 0]
    except (TypeError, ValueError):
        return None
    if len(px) < min_obs + 1:
        return None
    rets = [math.log(px[i] / px[i - 1]) for i in range(1, len(px))]
    rets = rets[-window:]
    n = len(rets)
    if n < min_obs:
        return None
    mean = sum(rets) / n
    var = sum((r - mean) ** 2 for r in rets) / (n - 1)
    return math.sqrt(var) * math.sqrt(252.0)


def iv_rv_ratio(iv: Optional[float], rv: Optional[float]) -> Optional[float]:
    try:
        iv_f, rv_f = float(iv), float(rv)
    except (TypeError, ValueError):
        return None
    if iv_f <= 0 or rv_f <= 0:
        return None
    return iv_f / rv_f


def classify_richness(ratio: Optional[float], min_ratio: float) -> str:
    """'rich' | 'not_rich' | 'unknown' — the single experiment gate."""
    if not isinstance(ratio, (int, float)):
        return UNKNOWN
    return RICH if ratio >= min_ratio else NOT_RICH


# --------------------------------------------------------------------------- #
# Pure execution-convention functions (honesty-critical)
# --------------------------------------------------------------------------- #
def conservative_entry_quotes(legs: Sequence[Mapping]) -> Optional[Dict[str, float]]:
    """Crossed entry fills: SELL legs at BID, BUY legs at ASK.

    Returns a quotes mapping for ``SpreadPaperTrader`` keyed by leg key, or
    None when any leg lacks the needed crossed-side quote (caller must skip —
    we never fall back to optimistic mid fills at entry).
    """
    out: Dict[str, float] = {}
    for leg in legs:
        action = str(leg.get("action", "")).lower()
        px = leg.get("bid") if action == "sell" else leg.get("ask")
        if not isinstance(px, (int, float)) or px <= 0:
            return None
        out[_leg_key(leg)] = float(px)
    return out


def leg_intrinsic(leg: Mapping, underlying_close: float) -> Optional[float]:
    """Option intrinsic value at expiration; None on unusable inputs."""
    try:
        k = float(leg.get("strike"))
        s = float(underlying_close)
    except (TypeError, ValueError):
        return None
    opt_type = str(leg.get("type", "")).lower()
    if opt_type == "call":
        return max(0.0, s - k)
    if opt_type == "put":
        return max(0.0, k - s)
    return None


def intrinsic_settlement_quotes(legs: Sequence[Mapping],
                                underlying_close: float) -> Optional[Dict[str, float]]:
    """Settlement quotes: every leg marked at intrinsic (no spread at expiry).

    A worthless leg maps to 0.0, which ``compute_mark`` correctly treats as a
    zero contribution. Returns None if any leg is unparseable (caller skips).
    """
    out: Dict[str, float] = {}
    for leg in legs:
        v = leg_intrinsic(leg, underlying_close)
        if v is None:
            return None
        out[_leg_key(leg)] = round(v, 4)
    return out


def earliest_expiration(legs: Sequence[Mapping]) -> Optional[date]:
    """Earliest leg expiration as a date; None when unparseable."""
    dates = []
    for leg in legs:
        raw = str(leg.get("expiration") or "")[:10]
        try:
            dates.append(date.fromisoformat(raw))
        except ValueError:
            return None
    return min(dates) if dates else None


# --------------------------------------------------------------------------- #
# Ledger (append-only JSONL, fail-open)
# --------------------------------------------------------------------------- #
def _append_ledger(path: str, row: dict) -> None:
    try:
        with open(path, "a", encoding="utf-8") as fh:
            fh.write(json.dumps(row, default=str) + "\n")
    except Exception as exc:  # pragma: no cover - disk safety
        print(f"{LOG_TAG} ledger append ignored: {exc}")


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --------------------------------------------------------------------------- #
# Market-data helpers (network at the edges; injectable for tests)
# --------------------------------------------------------------------------- #
def fetch_close_series(symbols: Sequence[str], window: int,
                       as_of: Optional[date] = None) -> Dict[str, List[float]]:
    """'all'-adjusted daily closes per symbol for the trailing RV window."""
    from session_returns import fetch_daily_bars
    end = as_of or datetime.now(timezone.utc).date()
    start = end - timedelta(days=int(window * 1.6) + 30)  # calendar padding
    bars = fetch_daily_bars(list(symbols), start, end, adjustment="all")
    out: Dict[str, List[float]] = {}
    for sym in symbols:
        df = bars.get(sym)
        out[sym] = [] if df is None or df.empty else [float(c) for c in df["close"]]
    return out


def fetch_expiry_close(symbol: str, expiry: date) -> Optional[float]:
    """Official (raw) daily close of ``symbol`` on ``expiry``; None if absent."""
    from session_returns import fetch_daily_bars
    try:
        bars = fetch_daily_bars([symbol], expiry, expiry, adjustment="raw")
        df = bars.get(symbol)
        if df is None or df.empty:
            return None
        rows = df[df["date"] == expiry]
        if rows.empty:
            return None
        return float(rows.iloc[-1]["close"])
    except Exception as exc:
        print(f"{LOG_TAG} expiry close fetch failed {symbol} {expiry}: {exc}")
        return None


def _atm_iv(trader, symbol: str, price: float) -> Optional[float]:
    """ATM-call implied vol via the nearest expiration snapshot (fail-open).

    Mirrors the IV source used by ``smart_trader.propose_spread`` so the gate
    and the structure are judged on the same IV.
    """
    try:
        contracts = trader.get_option_contracts(symbol) or []
        expirations = sorted({c.get("expiration_date") for c in contracts
                              if c.get("expiration_date")})
        if not expirations:
            return None
        calls = {}
        for c in contracts:
            if c.get("expiration_date") != expirations[0] or c.get("type") != "call":
                continue
            try:
                calls[float(c.get("strike_price"))] = c
            except (TypeError, ValueError):
                continue
        if not calls:
            return None
        atm = min(calls, key=lambda s: abs(s - price))
        snap = trader.get_option_snapshot(calls[atm].get("symbol", "")) or {}
        return snap.get("iv")
    except Exception as exc:
        print(f"{LOG_TAG} atm iv failed {symbol}: {exc}")
        return None


# --------------------------------------------------------------------------- #
# Scan (open candidates)
# --------------------------------------------------------------------------- #
def _persist_position(position: dict, paper_trader: SpreadPaperTrader) -> None:
    """Re-save the enriched position row by id. Fail-open."""
    try:
        rows = paper_trader.load_positions()
        for i, row in enumerate(rows):
            if row.get("id") == position.get("id"):
                rows[i] = position
                paper_trader.save_positions(rows)
                return
    except Exception as exc:  # pragma: no cover - disk safety
        print(f"{LOG_TAG} persist ignored: {exc}")


def scan_once(cfg: Optional[IVRVConfig] = None,
              trader_factory: Optional[Callable[[str], object]] = None,
              paper_trader: Optional[SpreadPaperTrader] = None,
              closes_by_symbol: Optional[Mapping[str, Sequence[float]]] = None,
              today: Optional[date] = None) -> dict:
    """One scan pass over the universe. Default OFF; never raises.

    Returns ``{enabled, scanned, rich, opened, rows}`` where ``rows`` are the
    per-symbol ledger rows (also appended to the JSONL scan ledger).
    """
    cfg = cfg or IVRVConfig.from_env()
    summary = {"enabled": cfg.enabled, "scanned": 0, "rich": 0,
               "opened": 0, "rows": []}
    if not cfg.enabled:
        print(f"{LOG_TAG} action=skipped reason=disabled "
              f"(set ENABLE_IVRV_CREDIT_PAPER=true to enable)")
        return summary
    if trader_factory is None:
        print(f"{LOG_TAG} action=skipped reason=no_trader_factory")
        return summary

    today = today or datetime.now(timezone.utc).date()
    pt = paper_trader or build_paper_trader(cfg)
    if closes_by_symbol is None:
        try:
            closes_by_symbol = fetch_close_series(cfg.universe, cfg.rv_window,
                                                  as_of=today)
        except Exception as exc:
            print(f"{LOG_TAG} bars fetch failed; aborting scan: {exc}")
            return summary

    opened = 0
    for symbol in cfg.universe:
        row = {"type": "ivrv_scan", "recorded_at": _now_iso(),
               "symbol": symbol, "rv": None, "atm_iv": None, "ratio": None,
               "verdict": UNKNOWN, "action": "skipped", "reason": None,
               "position_id": None, "strategy": None, "oracle_score": None,
               "net_credit": None, "max_loss": None, "entry_mark": None}
        try:
            trader = trader_factory(symbol)

            rv = annualized_realized_vol(
                closes_by_symbol.get(symbol) or [],
                window=cfg.rv_window, min_obs=cfg.min_rv_obs)
            row["rv"] = rv

            price = trader.get_current_price(symbol)
            if not price or price <= 0:
                row["reason"] = "no_underlying_price"
                continue

            iv = _atm_iv(trader, symbol, float(price))
            row["atm_iv"] = iv
            ratio = iv_rv_ratio(iv, rv)
            row["ratio"] = round(ratio, 4) if ratio is not None else None
            verdict = classify_richness(ratio, cfg.min_iv_rv_ratio)
            row["verdict"] = verdict

            if verdict != RICH:
                row["reason"] = f"iv_not_rich ratio={row['ratio']}"
                continue
            summary["rich"] += 1

            if opened >= cfg.max_opens_per_run:
                row["reason"] = "max_opens_reached"
                continue

            proposal = trader.propose_spread(symbol)
            strategy = getattr(proposal, "strategy_name", NO_TRADE)
            row["strategy"] = strategy
            if strategy == NO_TRADE:
                row["reason"] = (f"builder_no_trade: "
                                 f"{getattr(proposal, 'reason', '')}")
                continue
            if strategy not in CREDIT_STRATEGIES:
                # Debit structures are the documented loser — out of scope.
                row["reason"] = f"not_credit_strategy: {strategy}"
                continue

            legs = [l.as_dict() for l in getattr(proposal, "legs", [])]
            quotes = conservative_entry_quotes(legs)
            if quotes is None:
                row["reason"] = "missing_crossed_quote"
                continue

            expiry = earliest_expiration(legs)
            context = {
                "entry_underlying_price": float(price),
                "dte": (expiry - today).days if expiry else None,
            }
            result = pt.open_position(proposal, quotes=quotes, context=context)
            if not result.get("allowed"):
                row["reason"] = result.get("reason") or "rejected"
                continue

            position = result["position"]
            # Enrich the stored position with the experiment signal at entry.
            position["atm_iv"] = iv
            position["realized_vol"] = rv
            position["iv_rv_ratio"] = row["ratio"]
            _persist_position(position, pt)

            opened += 1
            row.update({
                "action": "opened", "reason": "opened",
                "position_id": position["id"],
                "oracle_score": position.get("oracle_score"),
                "net_credit": position.get("net_credit_or_debit"),
                "max_loss": position.get("max_loss"),
                "entry_mark": position.get("entry_mark"),
            })
        except Exception as exc:
            row["reason"] = f"error: {exc}"
        finally:
            summary["scanned"] += 1
            print(f"{LOG_TAG} symbol={symbol} verdict={row['verdict']} "
                  f"ratio={row['ratio']} action={row['action']} "
                  f"reason={row['reason']}")
            _append_ledger(cfg.scan_ledger, row)
            summary["rows"].append(row)

    summary["opened"] = opened
    return summary


# --------------------------------------------------------------------------- #
# Settle (close expired positions at intrinsic)
# --------------------------------------------------------------------------- #
def settle_expired(cfg: Optional[IVRVConfig] = None,
                   paper_trader: Optional[SpreadPaperTrader] = None,
                   price_lookup: Optional[Callable[[str, date], Optional[float]]] = None,
                   today: Optional[date] = None) -> dict:
    """Settle every OPEN position whose expiration has passed, at intrinsic.

    ``price_lookup(symbol, expiry_date)`` must return the official close of
    the underlying on the expiration date (default: raw Alpaca daily bar).
    Positions whose close is not yet available are left open and retried on
    the next run. Default OFF; never raises.
    """
    cfg = cfg or IVRVConfig.from_env()
    summary = {"enabled": cfg.enabled, "checked": 0, "settled": 0,
               "pending": 0, "trades": []}
    if not cfg.enabled:
        print(f"{LOG_TAG} settle skipped reason=disabled")
        return summary

    today = today or datetime.now(timezone.utc).date()
    pt = paper_trader or build_paper_trader(cfg)
    lookup = price_lookup or fetch_expiry_close

    for pos in pt.get_open_positions():
        summary["checked"] += 1
        try:
            legs = pos.get("legs") or []
            expiry = earliest_expiration(legs)
            if expiry is None or expiry > today:
                summary["pending"] += 1
                continue
            close_px = lookup(pos.get("symbol", ""), expiry)
            if close_px is None:
                print(f"{LOG_TAG} settle pending id={pos.get('id')} "
                      f"sym={pos.get('symbol')} expiry={expiry} "
                      f"(close not available yet)")
                summary["pending"] += 1
                continue
            quotes = intrinsic_settlement_quotes(legs, close_px)
            if quotes is None:
                print(f"{LOG_TAG} settle skipped id={pos.get('id')} "
                      f"reason=unparseable_legs")
                summary["pending"] += 1
                continue
            trade = pt.close_position(
                pos["id"], quotes=quotes,
                exit_reason=EXIT_EXPIRED_INTRINSIC,
                context={"exit_underlying_price": close_px})
            if trade is not None:
                summary["settled"] += 1
                summary["trades"].append(trade)
        except Exception as exc:
            print(f"{LOG_TAG} settle error id={pos.get('id')}: {exc}")
            summary["pending"] += 1
    return summary


# --------------------------------------------------------------------------- #
# Evaluation (evidence gate math)
# --------------------------------------------------------------------------- #
def summarize_results(cfg: Optional[IVRVConfig] = None,
                      paper_trader: Optional[SpreadPaperTrader] = None) -> dict:
    """Mean net P&L per $ max-loss over settled trades, with se and t-stat.

    This is the number the evidence gate in IVRV_EXPERIMENT.md is written
    against: n >= 40 settled trades AND mean > 0 with |t| >= 2.
    """
    cfg = cfg or IVRVConfig.from_env()
    pt = paper_trader or build_paper_trader(cfg)
    rets = []
    for t in pt.load_trades():
        pnl = t.get("pnl")
        max_loss = t.get("max_loss")
        if isinstance(pnl, (int, float)) and isinstance(max_loss, (int, float)) \
                and max_loss > 0:
            rets.append(pnl / max_loss)
    out = {"n_settled": len(rets), "n_open": len(pt.get_open_positions()),
           "mean_ret_per_maxloss": None, "se": None, "t_stat": None,
           "win_rate": None, "gate_passed": False}
    n = len(rets)
    if n >= 2:
        mean = sum(rets) / n
        var = sum((r - mean) ** 2 for r in rets) / (n - 1)
        se = math.sqrt(var / n) if var > 0 else 0.0
        out["mean_ret_per_maxloss"] = round(mean, 6)
        out["se"] = round(se, 6)
        out["t_stat"] = round(mean / se, 3) if se > 0 else None
        out["win_rate"] = round(sum(1 for r in rets if r > 0) / n, 4)
        out["gate_passed"] = bool(
            n >= 40 and mean > 0 and out["t_stat"] is not None
            and out["t_stat"] >= 2.0)
    return out


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main(argv: Optional[Sequence[str]] = None) -> int:
    import argparse

    ap = argparse.ArgumentParser(
        description="IV-RV credit-spread paper experiment (simulation only)")
    ap.add_argument("--scan", action="store_true",
                    help="scan universe, open qualifying paper positions")
    ap.add_argument("--settle", action="store_true",
                    help="settle expired paper positions at intrinsic")
    ap.add_argument("--status", action="store_true",
                    help="print evidence-gate summary")
    args = ap.parse_args(argv)

    cfg = IVRVConfig.from_env()
    rc = 0
    if args.scan:
        from smart_trader import SmartOptionsTrader
        summary = scan_once(
            cfg, trader_factory=lambda s: SmartOptionsTrader(ticker=s))
        print(f"{LOG_TAG} scan done: scanned={summary['scanned']} "
              f"rich={summary['rich']} opened={summary['opened']}")
    if args.settle:
        summary = settle_expired(cfg)
        print(f"{LOG_TAG} settle done: checked={summary['checked']} "
              f"settled={summary['settled']} pending={summary['pending']}")
    if args.status or not (args.scan or args.settle):
        s = summarize_results(cfg)
        print(f"{LOG_TAG} status: open={s['n_open']} settled={s['n_settled']} "
              f"mean_ret/maxloss={s['mean_ret_per_maxloss']} "
              f"se={s['se']} t={s['t_stat']} win_rate={s['win_rate']} "
              f"gate_passed={s['gate_passed']} "
              f"(gate: n>=40, mean>0, t>=2 — see IVRV_EXPERIMENT.md)")
    return rc


if __name__ == "__main__":
    import sys

    sys.exit(main())
