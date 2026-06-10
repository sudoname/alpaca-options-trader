"""
Phase 10G-E — Candidate Resolution Store (analytics only).

The Best-EV surfaces (BEST_EV_TRADES, BEST_EV_PAPER_RUN) evaluate many
candidates but only a handful become paper trades, so today's evidence is
survivorship-biased: we learn nothing about the trades Oracle scored and
did NOT take. This module records EVERY ranked candidate at evaluation time
and resolves it later against what actually happened:

    * selected_for_paper_trade — was it opened as a paper spread?
    * underlying_price_at_entry / _at_resolution
    * hypothetical_hold_to_expiry_pnl — payoff if held naively to expiry
    * hypothetical_policy_pnl / actual_paper_pnl — filled when known

Records are upserted by (symbol, strategy, day) so the ranker pass and the
paper-runner pass on the same day enrich one row instead of duplicating it.
Resolution only happens when the data exists (price snapshots after expiry);
otherwise rows stay pending. Everything fails open.

STRICTLY analytics: this module records and resolves beliefs. It never
opens, closes, sizes, blocks or alters any real or paper trade, never
touches the network, and is only ever invoked behind try/except from its
callers, so a failure here cannot affect a trading path.
"""

import json
import os
import uuid
from datetime import date, datetime, timedelta, timezone
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import oracle_analytics as oa
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR,
)

LOG_TAG = "[CANDIDATE_RESOLUTION]"

# Payoff rises with the underlying (low strike = -max_loss, high = +profit).
_RISING_PAYOFF = {BULLISH_PUT_CREDIT_SPREAD, DEBIT_CALL_SPREAD}
# Payoff falls with the underlying (low strike = +profit, high = -max_loss).
_FALLING_PAYOFF = {BEARISH_CALL_CREDIT_SPREAD, DEBIT_PUT_SPREAD}


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
class CandidateResolutionConfig:
    def __init__(self, enabled: bool = True,
                 file: str = "candidate_resolutions.json"):
        self.enabled = enabled
        self.file = file

    @staticmethod
    def from_env(path: str = ".env") -> "CandidateResolutionConfig":
        try:
            from config_loader import ConfigLoader
            cfg = ConfigLoader(path=path)
            return CandidateResolutionConfig(
                enabled=cfg.get_bool("CANDIDATE_RESOLUTION_ENABLED", True),
                file=cfg.get_str("CANDIDATE_RESOLUTION_FILE",
                                 "candidate_resolutions.json"),
            )
        except Exception:
            return CandidateResolutionConfig()


# ---------------------------------------------------------------------------
# Storage (fail-open JSON list)
# ---------------------------------------------------------------------------
def load_records(config: Optional[CandidateResolutionConfig] = None
                 ) -> List[dict]:
    cfg = config or CandidateResolutionConfig.from_env()
    data = oa.read_json(cfg.file)
    return [r for r in data if isinstance(r, dict)] \
        if isinstance(data, list) else []


def _save_records(rows: List[dict],
                  config: Optional[CandidateResolutionConfig]) -> bool:
    cfg = config or CandidateResolutionConfig.from_env()
    try:
        tmp = cfg.file + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(rows, fh, indent=2, default=str)
        os.replace(tmp, cfg.file)
        return True
    except Exception as exc:  # pragma: no cover - disk safety
        print(f"{LOG_TAG} save ignored: {exc}")
        return False


# ---------------------------------------------------------------------------
# Keys / coercion
# ---------------------------------------------------------------------------
def candidate_key(symbol, strategy, day) -> str:
    """Stable upsert key: one row per symbol+strategy+calendar day."""
    return f"{str(symbol or '').upper()}|{strategy or ''}|{day or ''}"


def _now(now: Optional[datetime]) -> datetime:
    return now or datetime.now(timezone.utc)


def _as_dict(result) -> Optional[dict]:
    if isinstance(result, dict):
        return result
    to_dict = getattr(result, "to_dict", None)
    if callable(to_dict):
        try:
            return to_dict()
        except Exception:
            return None
    return None


# ---------------------------------------------------------------------------
# Recording (called fail-open from the ranker and the paper runner)
# ---------------------------------------------------------------------------
_CANDIDATE_FIELDS = (
    "symbol", "strategy", "strikes", "expiry", "dte", "oracle_score",
    "volatility_edge", "forecast_vol", "market_iv", "expected_move",
    "market_expected_move", "probability_of_profit", "expected_value",
    "ev_per_dollar_risk", "max_profit", "max_loss", "recommendation",
)

_RESOLUTION_FIELDS = (
    "underlying_price_at_entry", "underlying_price_at_resolution",
    "hypothetical_hold_to_expiry_pnl", "hypothetical_policy_pnl",
    "actual_paper_pnl", "paper_position_id",
)


def _fresh_record(key: str, ts: datetime) -> dict:
    rec = {
        "candidate_id": uuid.uuid4().hex[:12],
        "candidate_key": key,
        "timestamp": ts.isoformat(),
        "selected_for_paper_trade": False,
        "resolved": False,
        "sources": [],
    }
    for f in _CANDIDATE_FIELDS + _RESOLUTION_FIELDS:
        rec[f] = None
    return rec


def record_candidates(results: Sequence,
                      selected_keys: Iterable[str] = (),
                      source: str = "",
                      config: Optional[CandidateResolutionConfig] = None,
                      extras: Optional[Dict[str, dict]] = None,
                      now: Optional[datetime] = None) -> int:
    """Upsert every evaluated candidate. Returns rows written; never raises.

    ``extras`` maps candidate_key -> extra fields (strikes, expiry,
    underlying_price_at_entry, ...) known only to the caller.
    """
    try:
        cfg = config or CandidateResolutionConfig.from_env()
        if not cfg.enabled:
            return 0
        ts = _now(now)
        day = ts.strftime("%Y-%m-%d")
        selected = {str(k) for k in (selected_keys or ())}
        extras = extras or {}

        rows = load_records(cfg)
        index = {r.get("candidate_key"): r for r in rows
                 if r.get("candidate_key")}
        written = 0
        for result in results or []:
            d = _as_dict(result)
            if not d or not d.get("symbol") or not d.get("strategy"):
                continue
            key = candidate_key(d.get("symbol"), d.get("strategy"), day)
            rec = index.get(key)
            if rec is None:
                rec = _fresh_record(key, ts)
                rows.append(rec)
                index[key] = rec

            incoming = {
                "symbol": str(d.get("symbol")).upper(),
                "strategy": d.get("strategy"),
                "dte": d.get("days") if d.get("days") is not None
                else d.get("dte"),
                "oracle_score": d.get("oracle_score"),
                "volatility_edge": d.get("volatility_edge"),
                "forecast_vol": d.get("forecast_vol"),
                "market_iv": d.get("market_iv"),
                "expected_move": d.get("expected_move"),
                "market_expected_move": d.get("market_expected_move"),
                "probability_of_profit": d.get("probability_of_profit"),
                "expected_value": d.get("expected_value"),
                "ev_per_dollar_risk": d.get("ev_per_dollar_risk"),
                "max_profit": d.get("max_profit"),
                "max_loss": d.get("max_loss"),
                "recommendation": d.get("recommendation"),
            }
            incoming.update(extras.get(key) or {})
            for field_name, value in incoming.items():
                if value is not None and rec.get(field_name) is None:
                    rec[field_name] = value
            if key in selected:
                rec["selected_for_paper_trade"] = True  # sticky
            if source and source not in rec.get("sources", []):
                rec.setdefault("sources", []).append(source)
            written += 1
        if written:
            _save_records(rows, cfg)
        return written
    except Exception as exc:
        print(f"{LOG_TAG} record ignored: {exc}")
        return 0


def selection_context(positions: Sequence[dict],
                      now: Optional[datetime] = None
                      ) -> Tuple[List[str], Dict[str, dict]]:
    """(selected_keys, extras) from the paper positions opened this run.

    Extras carry the execution-side facts the EVResult does not know:
    strikes, expiry, entry underlying price, expected moves, position id.
    """
    ts = _now(now)
    day = ts.strftime("%Y-%m-%d")
    selected: List[str] = []
    extras: Dict[str, dict] = {}
    for pos in positions or []:
        if not isinstance(pos, dict):
            continue
        key = candidate_key(pos.get("symbol"), pos.get("strategy"), day)
        legs = pos.get("legs") or []
        strikes = sorted(s for s in
                         (oa._to_float(leg.get("strike")) for leg in legs
                          if isinstance(leg, dict)) if s is not None)
        expiry = None
        for leg in legs:
            if isinstance(leg, dict):
                expiry = leg.get("expiration") or leg.get("expiry")
                if expiry:
                    break
        selected.append(key)
        extras[key] = {
            "strikes": strikes or None,
            "expiry": expiry,
            "dte": pos.get("dte"),
            "underlying_price_at_entry": pos.get("entry_underlying_price"),
            "expected_move": pos.get("expected_move"),
            "market_expected_move": pos.get("market_expected_move"),
            "paper_position_id": pos.get("id"),
        }
    return selected, extras


# ---------------------------------------------------------------------------
# Resolution (only when the data exists; pending rows are left alone)
# ---------------------------------------------------------------------------
def hold_to_expiry_pnl(strategy: str, strikes, price,
                       max_profit, max_loss) -> Optional[float]:
    """Piecewise-linear expiry payoff of the spread at underlying ``price``.

    Vertical spreads interpolate between -max_loss and +max_profit across
    their two strikes; iron condors pay max_profit inside the short strikes
    and -max_loss beyond the long wings. None when inputs are unknown.
    """
    p = oa._to_float(price)
    profit = oa._to_float(max_profit)
    loss = oa._to_float(max_loss)
    ks = sorted(s for s in (oa._to_float(k) for k in (strikes or []))
                if s is not None)
    if p is None or profit is None or loss is None or not ks:
        return None

    def _rising(lo: float, hi: float) -> float:
        if p <= lo:
            return -loss
        if p >= hi:
            return profit
        return -loss + (profit + loss) * (p - lo) / (hi - lo)

    def _falling(lo: float, hi: float) -> float:
        if p <= lo:
            return profit
        if p >= hi:
            return -loss
        return profit - (profit + loss) * (p - lo) / (hi - lo)

    if strategy in _RISING_PAYOFF and len(ks) >= 2:
        return round(_rising(ks[0], ks[-1]), 2)
    if strategy in _FALLING_PAYOFF and len(ks) >= 2:
        return round(_falling(ks[0], ks[-1]), 2)
    if strategy == IRON_CONDOR and len(ks) >= 4:
        pl, ps, cs, cl = ks[0], ks[1], ks[-2], ks[-1]
        if ps <= p <= cs:
            return round(profit, 2)
        if p < ps:
            return round(_rising(pl, ps), 2)
        return round(_falling(cs, cl), 2)
    return None


def _effective_expiry(rec: dict) -> Optional[date]:
    """Stated expiry, else entry day + DTE."""
    exp = oa._parse_ts(rec.get("expiry"))
    if exp is not None:
        return exp.date()
    entry = oa._parse_ts(rec.get("timestamp"))
    dte = oa._to_float(rec.get("dte"))
    if entry is not None and dte is not None:
        return (entry + timedelta(days=dte)).date()
    return None


def resolve_pending(price_lookup: Callable[[str], Optional[float]],
                    today: Optional[date] = None,
                    config: Optional[CandidateResolutionConfig] = None) -> int:
    """Resolve expired candidates against ``price_lookup(symbol)``.

    Stamps underlying_price_at_resolution and the hold-to-expiry payoff.
    Rows without an expiry or a price stay pending. Never raises.
    """
    try:
        cfg = config or CandidateResolutionConfig.from_env()
        if not cfg.enabled:
            return 0
        ref_day = today or datetime.now(timezone.utc).date()
        rows = load_records(cfg)
        resolved = 0
        for rec in rows:
            if rec.get("resolved"):
                continue
            expiry = _effective_expiry(rec)
            if expiry is None or expiry > ref_day:
                continue
            try:
                price = price_lookup(rec.get("symbol"))
            except Exception:
                price = None
            price = oa._to_float(price)
            if price is None:
                continue
            rec["underlying_price_at_resolution"] = price
            rec["hypothetical_hold_to_expiry_pnl"] = hold_to_expiry_pnl(
                rec.get("strategy"), rec.get("strikes"), price,
                rec.get("max_profit"), rec.get("max_loss"))
            rec["resolved"] = True
            rec["resolved_at"] = datetime.now(timezone.utc).isoformat()
            resolved += 1
        if resolved:
            _save_records(rows, cfg)
        return resolved
    except Exception as exc:
        print(f"{LOG_TAG} resolve ignored: {exc}")
        return 0


def record_paper_outcome(candidate_id: str, pnl,
                         policy_pnl=None,
                         config: Optional[CandidateResolutionConfig] = None
                         ) -> bool:
    """Attach the realized paper PnL to a candidate (by candidate id or
    paper position id) once its simulated trade is closed. Never raises."""
    try:
        cfg = config or CandidateResolutionConfig.from_env()
        if not cfg.enabled:
            return False
        rows = load_records(cfg)
        for rec in rows:
            if candidate_id in (rec.get("candidate_id"),
                                rec.get("paper_position_id")):
                rec["actual_paper_pnl"] = oa._to_float(pnl)
                if policy_pnl is not None:
                    rec["hypothetical_policy_pnl"] = oa._to_float(policy_pnl)
                return _save_records(rows, cfg)
        return False
    except Exception as exc:
        print(f"{LOG_TAG} outcome ignored: {exc}")
        return False


# ---------------------------------------------------------------------------
# Summary (for future analytics surfaces)
# ---------------------------------------------------------------------------
def summarize(config: Optional[CandidateResolutionConfig] = None) -> dict:
    rows = load_records(config)
    selected = [r for r in rows if r.get("selected_for_paper_trade")]
    resolved = [r for r in rows if r.get("resolved")]
    return {
        "candidates": len(rows),
        "selected": len(selected),
        "not_selected": len(rows) - len(selected),
        "resolved": len(resolved),
        "pending": len(rows) - len(resolved),
    }
