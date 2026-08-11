"""
Oracle Lab — point-in-time feature/label dataset builder.

``build_dataset`` walks a set of (symbol, as_of) decision points and, for each
one, extracts FEATURES from a ``MarketView(as_of)`` (which the project already
guarantees is point-in-time: only data with close_dt <= as_of escapes) and a
forward-return LABEL from a ``MarketView(horizon_end)`` where
``horizon_end > as_of``. Features and labels therefore live on opposite sides of
the decision instant and cannot leak into each other.

The builder never touches the network itself: the caller supplies a
``market_view_factory(as_of) -> MarketView``. In tests / offline research this
is a closure over pre-fetched ``HistoricalMarketView`` series; in a live-adjacent
context it could return a ``LiveMarketView`` (still offline-safe here because we
only read what the view exposes).

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Deterministic given (symbols, as_ofs, factory, horizon, feature_fn).
  * Fail-open on a single bad point (skip it, keep going); never raise in the
    research path. A leak, however, is a correctness bug — the self-test asserts
    features never see a bar stamped after as_of.
  * Pure-ish: the only I/O is optional JSONL persistence (reuses the
    oracle_prob_recorder fold-by-id ledger idiom).
"""

from __future__ import annotations

import json
import math
import uuid
from dataclasses import asdict, dataclass, field
from datetime import datetime, timedelta
from typing import Any, Callable, Dict, List, Optional, Sequence

_LABEL_UP = "up"
_LABEL_DOWN = "down"
_LABEL_FLAT = "flat"


def _to_float(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        f = float(value)
    except (TypeError, ValueError):
        return None
    if math.isnan(f) or math.isinf(f):
        return None
    return f


# --------------------------------------------------------------------------- #
# Snapshot record
# --------------------------------------------------------------------------- #
@dataclass
class FeatureSnapshot:
    """One point-in-time decision row: features known at ``as_of`` + a forward
    label observed at ``horizon_end`` (strictly later)."""

    snapshot_id: str
    symbol: str
    as_of: str                       # ISO, the decision instant
    horizon_end: str                 # ISO, when the label is observed (> as_of)
    mode: str
    ctx: Dict[str, Any]              # point-in-time features
    label_return_pct: Optional[float]
    label_direction: Optional[str]   # up / down / flat / None
    entry_price: Optional[float]
    exit_price: Optional[float]
    event_timestamp: Optional[str] = None       # temporal-integrity (nullable)
    available_timestamp: Optional[str] = None    # temporal-integrity (nullable)
    meta: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict:
        d = asdict(self)
        d["record_type"] = "feature_snapshot"
        return d


# --------------------------------------------------------------------------- #
# Default point-in-time feature extractor
# --------------------------------------------------------------------------- #
def _sma(values: Sequence[float], n: int) -> Optional[float]:
    vals = [v for v in values if v is not None]
    if len(vals) < n or n <= 0:
        return None
    return sum(vals[-n:]) / n


def _realized_vol_pct(closes: Sequence[float]) -> Optional[float]:
    xs = [c for c in closes if c is not None and c > 0]
    if len(xs) < 3:
        return None
    rets = [math.log(xs[i] / xs[i - 1]) for i in range(1, len(xs))]
    n = len(rets)
    mu = sum(rets) / n
    var = sum((r - mu) ** 2 for r in rets) / (n - 1) if n > 1 else 0.0
    return round(math.sqrt(var) * 100.0, 6)


def _default_features(mv, symbol: str) -> Dict[str, Any]:
    """A small, deterministic, point-in-time feature vector read purely through
    the ``MarketView`` API (so the no-lookahead audit covers it). Fail-open: any
    missing series yields a None field rather than an error."""
    try:
        bars = mv.daily_bars(symbol, lookback=30)
    except Exception:
        bars = []
    closes = [b.c for b in bars] if bars else []
    last = closes[-1] if closes else None
    sma5 = _sma(closes, 5)
    sma10 = _sma(closes, 10)
    momentum_pct = None
    if last is not None and len(closes) >= 6 and closes[-6] > 0:
        momentum_pct = round((last / closes[-6] - 1.0) * 100.0, 6)
    try:
        vix = mv.vix()
    except Exception:
        vix = None
    return {
        "symbol": symbol,
        "last_close": last,
        "sma5": sma5,
        "sma10": sma10,
        "sma5_over_sma10": (round(sma5 / sma10, 6)
                            if sma5 and sma10 and sma10 != 0 else None),
        "momentum_5d_pct": momentum_pct,
        "realized_vol_pct": _realized_vol_pct(closes),
        "vix": _to_float(vix),
        "n_bars": len(closes),
    }


# --------------------------------------------------------------------------- #
# Builder
# --------------------------------------------------------------------------- #
def daily_as_ofs(start: datetime, end: datetime, *, hour: int = 16,
                 minute: int = 0) -> List[datetime]:
    """Convenience: one decision instant per calendar day in [start, end] at the
    given wall-clock time (default 16:00, the session close). Deterministic."""
    out: List[datetime] = []
    if not isinstance(start, datetime) or not isinstance(end, datetime):
        return out
    day = start.date()
    last = end.date()
    while day <= last:
        out.append(datetime(day.year, day.month, day.day, hour, minute))
        day = day + timedelta(days=1)
    return out


def _label(entry: Optional[float], exit_: Optional[float],
           flat_band_pct: float) -> (Optional[float], Optional[str]):
    if entry is None or exit_ is None or entry <= 0:
        return None, None
    ret_pct = round((exit_ / entry - 1.0) * 100.0, 6)
    if ret_pct > flat_band_pct:
        return ret_pct, _LABEL_UP
    if ret_pct < -flat_band_pct:
        return ret_pct, _LABEL_DOWN
    return ret_pct, _LABEL_FLAT


def build_dataset(symbols: Sequence[str], as_ofs: Sequence[datetime], *,
                  market_view_factory: Callable[[datetime], Any],
                  horizon: timedelta,
                  mode: str = "intraday",
                  feature_fn: Optional[Callable[[Any, str], Dict[str, Any]]] = None,
                  price_fn: Optional[Callable[[Any, str], Optional[float]]] = None,
                  flat_band_pct: float = 0.0) -> List[FeatureSnapshot]:
    """Produce point-in-time FeatureSnapshot rows.

    For each (symbol, as_of): build ``mv = factory(as_of)``, extract features via
    ``feature_fn(mv, symbol)`` (default ``_default_features``); build
    ``mv_future = factory(as_of + horizon)`` and read the forward price via
    ``price_fn`` (default ``last_close``). The label is the forward return.

    Determinism: rows are emitted in (as_of, symbol) sorted order. ``snapshot_id``
    is a stable hash of (symbol, as_of, horizon) so re-running folds cleanly.
    """
    feature_fn = feature_fn or _default_features
    price_fn = price_fn or (lambda mv, s: mv.last_close(s))
    rows: List[FeatureSnapshot] = []
    if not isinstance(horizon, timedelta) or horizon <= timedelta(0):
        return rows

    pairs = []
    for a in (as_ofs or []):
        if not isinstance(a, datetime):
            continue
        for s in (symbols or []):
            if s:
                pairs.append((a, str(s).upper()))
    pairs.sort(key=lambda p: (p[0], p[1]))

    for as_of, symbol in pairs:
        try:
            mv = market_view_factory(as_of)
            ctx = feature_fn(mv, symbol) or {}
            entry = price_fn(mv, symbol)
            horizon_end = as_of + horizon
            mv_future = market_view_factory(horizon_end)
            exit_ = price_fn(mv_future, symbol)
            ret_pct, direction = _label(_to_float(entry), _to_float(exit_),
                                        flat_band_pct)
            sid = uuid.uuid5(
                uuid.NAMESPACE_URL,
                f"{symbol}|{as_of.isoformat()}|{horizon.total_seconds()}"
            ).hex[:12]
            rows.append(FeatureSnapshot(
                snapshot_id=sid,
                symbol=symbol,
                as_of=as_of.isoformat(),
                horizon_end=horizon_end.isoformat(),
                mode=str(mode),
                ctx=ctx,
                label_return_pct=ret_pct,
                label_direction=direction,
                entry_price=_to_float(entry),
                exit_price=_to_float(exit_),
                available_timestamp=as_of.isoformat(),
            ))
        except Exception as exc:  # pragma: no cover - fail-open per point
            print(f"[lab.dataset] skipped {symbol}@{as_of}: {exc}")
            continue
    return rows


# --------------------------------------------------------------------------- #
# JSONL persistence (fold-by-id, last-write-wins) — mirrors oracle_prob_recorder
# --------------------------------------------------------------------------- #
def save_dataset(rows: Sequence[FeatureSnapshot], path: str) -> int:
    """Append snapshot rows to a JSONL ledger. Returns count written."""
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for r in rows or []:
            rec = r.to_dict() if isinstance(r, FeatureSnapshot) else r
            fh.write(json.dumps(rec, default=str) + "\n")
            n += 1
    return n


def load_dataset(path: str) -> List[dict]:
    """Load a JSONL dataset, folding by snapshot_id (last line wins)."""
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
                sid = rec.get("snapshot_id")
                if not sid:
                    continue
                if sid not in folded:
                    order.append(sid)
                folded[sid] = rec
    except FileNotFoundError:
        return []
    return [folded[s] for s in order]


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds; asserts no look-ahead in features)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    from market_view import HistoricalMarketView, make_bar

    # Build a rising synthetic daily series for SPY over 8 sessions.
    dates = [f"2024-01-0{i}" for i in range(1, 9)]  # 2024-01-01 .. 2024-01-08
    prices = [100.0, 101.0, 102.0, 103.0, 104.0, 105.0, 106.0, 107.0]
    bars = [make_bar(d, p, p + 0.5, p - 0.5, p, 1_000_000)
            for d, p in zip(dates, prices)]

    def factory(as_of):
        return HistoricalMarketView(as_of, daily={"SPY": list(bars)})

    # Decide at the close of 2024-01-04 (price 103), 1-day horizon -> 01-05
    # (price 104). Forward return = +0.9709%.
    as_of = datetime(2024, 1, 4, 16, 0)
    rows = build_dataset(["SPY"], [as_of], market_view_factory=factory,
                         horizon=timedelta(days=1), mode="intraday")
    if len(rows) != 1:
        print("FAIL: expected 1 row", rows); return 1
    r = rows[0]

    # Feature entry price must be 103 (the as_of close), NOT a future bar.
    if r.entry_price != 103.0:
        print("FAIL: entry not point-in-time", r.entry_price); ok = False
    if r.exit_price != 104.0:
        print("FAIL: exit not the horizon close", r.exit_price); ok = False
    if r.label_direction != _LABEL_UP:
        print("FAIL: label direction", r.label_direction); ok = False
    if r.label_return_pct is None or abs(r.label_return_pct - 0.970874) > 1e-4:
        print("FAIL: label return", r.label_return_pct); ok = False

    # CRITICAL leakage check: the feature view's audit must contain NO datum
    # stamped after as_of. Re-run the feature extraction on the same view and
    # inspect its audit log.
    mv = factory(as_of)
    _default_features(mv, "SPY")
    leaked = [rec for rec in mv.audit if rec["ts"] > as_of]
    if leaked:
        print("FAIL: LOOK-AHEAD LEAK in features", leaked); ok = False
    # The n_bars feature must equal the count of bars with close <= as_of (4).
    if r.ctx.get("n_bars") != 4:
        print("FAIL: feature saw future bars", r.ctx.get("n_bars")); ok = False

    # Determinism: same snapshot_id + identical row on re-run.
    rows2 = build_dataset(["SPY"], [as_of], market_view_factory=factory,
                          horizon=timedelta(days=1), mode="intraday")
    if rows2[0].snapshot_id != r.snapshot_id:
        print("FAIL: non-deterministic id"); ok = False
    if rows2[0].to_dict() != r.to_dict():
        print("FAIL: non-deterministic row"); ok = False

    # Flat band: a tiny move inside the band labels 'flat'.
    flat_rows = build_dataset(["SPY"], [as_of], market_view_factory=factory,
                              horizon=timedelta(days=1), flat_band_pct=5.0)
    if flat_rows[0].label_direction != _LABEL_FLAT:
        print("FAIL: flat band", flat_rows[0].label_direction); ok = False

    # Junk tolerance: bad horizon / non-datetime as_of -> empty, no raise.
    if build_dataset(["SPY"], [as_of], market_view_factory=factory,
                     horizon=timedelta(0)) != []:
        print("FAIL: bad horizon should be empty"); ok = False
    if build_dataset(["SPY"], ["not-a-date"], market_view_factory=factory,
                     horizon=timedelta(days=1)) != []:
        print("FAIL: bad as_of should be empty"); ok = False

    print("lab.dataset self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
