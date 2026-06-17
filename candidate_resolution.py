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
import triple_gap as tg
from spread_builder import (
    BULLISH_PUT_CREDIT_SPREAD, BEARISH_CALL_CREDIT_SPREAD,
    DEBIT_CALL_SPREAD, DEBIT_PUT_SPREAD, IRON_CONDOR,
)

LOG_TAG = "[CANDIDATE_RESOLUTION]"

# --- Phase 11A: append-only JSONL candidate layer (additive; the JSON-list
# store above is left fully intact). Stamps append ``record_type="candidate"``
# lines; resolution appends a newer full snapshot per candidate_id. Readers
# fold by candidate_id (last non-null wins) so resolution overrides the stamp.
JSONL_FILE_DEFAULT = "candidate_resolution.jsonl"
RECORD_TYPE_CANDIDATE = "candidate"
RECORD_TYPE_RESOLUTION = "resolution"

RESOLUTION_UNRESOLVED = "unresolved"
RESOLUTION_PARTIAL = "resolved_partial"
RESOLUTION_EXPIRY = "resolved_expiry"
RESOLUTION_MISSING_PRICE = "missing_price_data"

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
        # Phase 11A: additive, fail-open stamp into the append-only JSONL layer
        # (auto-covers BEST_EV_TRADES + BEST_EV_PAPER_RUN). Cannot affect the
        # JSON-list result above or any trading path.
        try:
            stamp_candidates(results, source_command=source,
                             selected_keys=selected, extras=extras, now=ts)
        except Exception:
            pass
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
# Phase 11A — append-only JSONL candidate layer (stamp + fold + resolve)
# ---------------------------------------------------------------------------
def _jsonl_path(jsonl_path: Optional[str] = None) -> str:
    """Resolve the JSONL store path (arg > env > default). Fail-open."""
    if jsonl_path:
        return jsonl_path
    try:
        from config_loader import ConfigLoader
        return ConfigLoader(path=".env").get_str(
            "CANDIDATE_RESOLUTION_JSONL", JSONL_FILE_DEFAULT)
    except Exception:
        return JSONL_FILE_DEFAULT


def _append_jsonl(rec: dict, path: str) -> None:
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(rec, default=str) + "\n")


_CANDLESTICK_KEYS = (
    "candlestick_pattern", "candlestick_bias", "candlestick_strength",
    "candlestick_confidence", "candlestick_reason",
    "candlestick_requires_confirmation",
)


def _candlestick_fields(f: dict) -> dict:
    """Derive the 6 frozen candlestick fields for a candidate. Fail-open.

    Precomputed ``candlestick_*`` on the incoming fields win (already frozen
    upstream). Otherwise, if OHLCV ``candles`` were injected, run the pure
    detector. With neither, all six are None. Never raises — pattern detection
    is analytics only and must never break candidate stamping.
    """
    out = {k: None for k in _CANDLESTICK_KEYS}
    try:
        if f.get("candlestick_pattern") is not None:
            for k in _CANDLESTICK_KEYS:
                out[k] = f.get(k)
            return out
        candles = f.get("candles")
        if not candles:
            return out
        from oracle.signals import candlestick_patterns as csp
        stamp = csp.detect_primary(candles)
        if stamp is not None:
            out.update({
                "candlestick_pattern": stamp.pattern_name,
                "candlestick_bias": stamp.bias,
                "candlestick_strength": stamp.strength,
                "candlestick_confidence": stamp.confidence,
                "candlestick_reason": stamp.reason,
                "candlestick_requires_confirmation": stamp.requires_confirmation,
            })
    except Exception:
        return {k: None for k in _CANDLESTICK_KEYS}
    return out


def stamp_candidate(fields: dict, *, jsonl_path: Optional[str] = None,
                    now: Optional[datetime] = None) -> Optional[str]:
    """Append ONE candidate line, freezing the signals + Triple Gap computed
    from them. Returns the candidate_id, or None (fail-open). Never raises.

    Values are frozen at candidate time and never recomputed on later passes.
    """
    try:
        f = dict(fields or {})
        symbol = str(f.get("symbol") or "").upper()
        strategy = f.get("strategy")
        if not symbol or not strategy:
            return None
        ts = _now(now)
        cid = f.get("candidate_id") or uuid.uuid4().hex[:12]

        market_iv = f.get("market_iv")
        forecast_vol = f.get("forecast_vol")
        market_expected_move = f.get("market_expected_move")
        oracle_expected_move = (f.get("oracle_expected_move")
                                if f.get("oracle_expected_move") is not None
                                else f.get("expected_move"))
        oracle_ev = f.get("expected_value")
        market_neutral_ev = f.get("market_neutral_expected_value")
        gap = tg.compute_triple_gap(
            symbol=symbol, strategy=strategy,
            market_iv=market_iv, forecast_vol=forecast_vol,
            market_expected_move=market_expected_move,
            oracle_expected_move=oracle_expected_move,
            oracle_expected_value=oracle_ev,
            market_neutral_expected_value=market_neutral_ev)

        # Phase 11B — freeze candlestick pattern fields (analytics only).
        # Patterns NEVER alter strategy/EV/PoP/risk/advisory/approval; they are
        # market-behaviour features stored alongside the other signals. Opt-in:
        # detection runs only when candles are injected. Fail-open to None.
        cs = _candlestick_fields(f)

        rec = {
            "record_type": RECORD_TYPE_CANDIDATE,
            "candidate_id": cid,
            "candidate_key": candidate_key(symbol, strategy,
                                           ts.strftime("%Y-%m-%d")),
            "timestamp": ts.isoformat(),
            "symbol": symbol,
            "strategy": strategy,
            "strikes": f.get("strikes"),
            "expiry": f.get("expiry"),
            "dte": f.get("dte") if f.get("dte") is not None else f.get("days"),
            "oracle_score": f.get("oracle_score"),
            "volatility_edge": f.get("volatility_edge"),
            "forecast_vol": forecast_vol,
            "market_iv": market_iv,
            "oracle_expected_move": oracle_expected_move,
            "market_expected_move": market_expected_move,
            "probability_of_profit": f.get("probability_of_profit"),
            "expected_value": oracle_ev,
            "ev_per_dollar_risk": f.get("ev_per_dollar_risk"),
            "ev_recommendation": (f.get("ev_recommendation")
                                  if f.get("ev_recommendation") is not None
                                  else f.get("recommendation")),
            "max_profit": f.get("max_profit"),
            "max_loss": f.get("max_loss"),
            "vol_gap": gap.vol_gap,
            "move_gap": gap.move_gap,
            "ev_gap": gap.ev_gap,
            "triple_gap_score": gap.triple_gap_score,
            "ev_gap_source": gap.ev_gap_source,
            "advisory_recommendation": f.get("advisory_recommendation"),
            "advisory_confidence": f.get("advisory_confidence"),
            "underlying_price_at_entry": f.get("underlying_price_at_entry"),
            "selected_for_paper_trade": bool(
                f.get("selected_for_paper_trade", False)),
            "source_command": (f.get("source_command")
                               or f.get("source") or ""),
            "candlestick_pattern": cs["candlestick_pattern"],
            "candlestick_bias": cs["candlestick_bias"],
            "candlestick_strength": cs["candlestick_strength"],
            "candlestick_confidence": cs["candlestick_confidence"],
            "candlestick_reason": cs["candlestick_reason"],
            "candlestick_requires_confirmation":
                cs["candlestick_requires_confirmation"],
        }
        _append_jsonl(rec, _jsonl_path(jsonl_path))
        return cid
    except Exception as exc:
        print(f"{LOG_TAG} stamp ignored: {exc}")
        return None


def stamp_candidates(results: Sequence, *, source_command: str = "",
                     selected_keys: Iterable[str] = (),
                     extras: Optional[Dict[str, dict]] = None,
                     jsonl_path: Optional[str] = None,
                     now: Optional[datetime] = None) -> int:
    """Stamp every EVResult/SpreadProposal-like result. Returns stamped count.

    ``extras`` maps candidate_key -> execution-side fields (strikes, expiry,
    entry price, ...). ``selected_keys`` flags rows opened as paper trades.
    Never raises.
    """
    try:
        ts = _now(now)
        day = ts.strftime("%Y-%m-%d")
        selected = {str(k) for k in (selected_keys or ())}
        extras = extras or {}
        path = _jsonl_path(jsonl_path)
        count = 0
        for result in results or []:
            d = _as_dict(result)
            if not d or not d.get("symbol") or not d.get("strategy"):
                continue
            key = candidate_key(d.get("symbol"), d.get("strategy"), day)
            fields = dict(d)
            fields.update(extras.get(key) or {})
            fields["source_command"] = source_command
            fields["selected_for_paper_trade"] = (
                key in selected or bool(fields.get("selected_for_paper_trade")))
            if stamp_candidate(fields, jsonl_path=path, now=ts) is not None:
                count += 1
        return count
    except Exception as exc:
        print(f"{LOG_TAG} stamp_candidates ignored: {exc}")
        return 0


def load_jsonl_records(jsonl_path: Optional[str] = None) -> List[dict]:
    """Fold the append-only JSONL store by candidate_id (last non-null wins so
    resolution snapshots override the original stamp). Tolerates a missing
    file, malformed lines and partial records. Never raises."""
    path = _jsonl_path(jsonl_path)
    folded: Dict[str, dict] = {}
    order: List[str] = []
    try:
        with open(path, "r", encoding="utf-8") as fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except Exception:
                    continue
                if not isinstance(rec, dict):
                    continue
                cid = rec.get("candidate_id")
                if not cid:
                    continue
                if cid not in folded:
                    folded[cid] = {}
                    order.append(cid)
                merged = folded[cid]
                for k, v in rec.items():
                    if v is not None or k not in merged:
                        merged[k] = v
    except FileNotFoundError:
        return []
    except Exception as exc:
        print(f"{LOG_TAG} jsonl load ignored: {exc}")
        return []
    return [folded[cid] for cid in order]


def resolve_jsonl_candidates(price_fn: Callable[[str], Optional[float]], *,
                             today: Optional[date] = None,
                             horizon_days: Optional[int] = None,
                             jsonl_path: Optional[str] = None,
                             now: Optional[datetime] = None) -> int:
    """Resolve folded candidates against ``price_fn(symbol)`` and append a new
    snapshot line per resolved candidate (append-only; never rewrites).

    Status per candidate::

        missing_price_data  expiry due but price_fn returned None
        resolved_expiry     effective expiry <= today AND price available
        resolved_partial    price available before expiry (interim mark)
        unresolved          no price and not yet due -> left as-is, no line

    Returns the count newly marked resolved (expiry or partial). ``price_fn``
    is injected (not wired into any live loop). Never raises.
    """
    try:
        path = _jsonl_path(jsonl_path)
        recs = load_jsonl_records(path)
        ref_now = _now(now)
        ref_day = today or ref_now.date()
        resolved = 0
        for rec in recs:
            if rec.get("resolution_status") == RESOLUTION_EXPIRY:
                continue  # already finalised; never re-resolve
            symbol = rec.get("symbol")
            try:
                price = price_fn(symbol)
            except Exception:
                price = None
            price = oa._to_float(price)

            expiry = _effective_expiry(rec)
            if expiry is None and horizon_days is not None:
                entry = oa._parse_ts(rec.get("timestamp"))
                if entry is not None:
                    expiry = (entry + timedelta(days=horizon_days)).date()
            due = expiry is not None and expiry <= ref_day

            if price is None:
                if due:
                    status = RESOLUTION_MISSING_PRICE
                else:
                    continue  # unresolved: leave as-is, append nothing
            elif due:
                status = RESOLUTION_EXPIRY
            else:
                status = RESOLUTION_PARTIAL

            entry_price = oa._to_float(rec.get("underlying_price_at_entry"))
            actual_move = None
            if entry_price not in (None, 0) and price is not None:
                actual_move = round((price - entry_price) / entry_price, 6)
            hold_pnl = None
            if price is not None:
                hold_pnl = hold_to_expiry_pnl(
                    rec.get("strategy"), rec.get("strikes"), price,
                    rec.get("max_profit"), rec.get("max_loss"))

            snap = dict(rec)
            snap["record_type"] = RECORD_TYPE_RESOLUTION
            snap["underlying_price_at_resolution"] = price
            snap["actual_move"] = actual_move
            snap["hypothetical_hold_to_expiry_pnl"] = hold_pnl
            snap.setdefault("hypothetical_policy_pnl", None)
            snap.setdefault("actual_paper_pnl", None)
            snap["resolved_at"] = ref_now.isoformat()
            snap["resolution_status"] = status
            _append_jsonl(snap, path)
            if status in (RESOLUTION_EXPIRY, RESOLUTION_PARTIAL):
                resolved += 1
        return resolved
    except Exception as exc:
        print(f"{LOG_TAG} resolve_jsonl ignored: {exc}")
        return 0


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
