"""Point-in-time Kalshi-derived features (Oracle 4.0.2 P1-6).

Builds a small, fixed set of ``kalshi_*`` features for one Oracle
underlying (SPY/QQQ) at a decision time ``as_of`` from the snapshot
store written by ``oracle.ml.kalshi.collector``.

Rules:
  * strictly point-in-time: only snapshot rows with
    ``observed_at <= as_of`` are read (enforced in SQL), and only within
    ``max_staleness_min`` of ``as_of``;
  * the snapshot store has no settlement/result column, so settled
    outcomes structurally cannot appear here;
  * missing/uncomputable values are ``None`` -- never fabricated,
    repaired, or defaulted; the downstream ``Standardizer`` is None-safe;
  * Kalshi strikes are on the cash index (SPX/NDX).  ``kalshi_p_above_spot``
    therefore requires an *index-level* spot supplied by the caller
    (``index_spot``); no index<->ETF ratio is ever guessed.

Features (all ``Optional[float]``):
  * ``kalshi_p_above_spot``       implied P(index settles above index_spot)
  * ``kalshi_implied_dispersion`` (q25 - q75 strike) / implied median
  * ``kalshi_book_spread_mean``   mean yes_ask - yes_bid (probability units)
  * ``kalshi_staleness_min``      minutes between snapshot and as_of
  * ``kalshi_n_markets``          usable CDF points in the chosen event
"""
from __future__ import annotations

import math
import os
import sqlite3
from datetime import datetime, timedelta, timezone
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .market_mapping import series_for_underlying

UTC = timezone.utc

#: fixed feature-name tuple -- every build returns exactly these keys.
KALSHI_FEATURES: Tuple[str, ...] = (
    "kalshi_p_above_spot",
    "kalshi_implied_dispersion",
    "kalshi_book_spread_mean",
    "kalshi_staleness_min",
    "kalshi_n_markets",
)

DEFAULT_MAX_STALENESS_MIN = 60.0


def _empty() -> Dict[str, Optional[float]]:
    return {name: None for name in KALSHI_FEATURES}


def _iso_z(dt: datetime) -> str:
    return dt.astimezone(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def _parse_utc(s) -> Optional[datetime]:
    if not s:
        return None
    try:
        txt = str(s).strip().replace("Z", "+00:00")
        dt = datetime.fromisoformat(txt)
    except ValueError:
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _mid(yes_bid: Optional[float], yes_ask: Optional[float],
         last_price: Optional[float]) -> Optional[float]:
    """Best available probability for a market; None when unquoted."""
    if yes_bid is not None and yes_ask is not None:
        return (yes_bid + yes_ask) / 2.0
    if last_price is not None:
        return last_price
    return yes_bid if yes_bid is not None else yes_ask


def _cdf_points(rows: Sequence[sqlite3.Row]
                ) -> List[Tuple[float, float]]:
    """(strike, P(index above strike)) from greater/less markets.

    ``between`` (range) markets are skipped -- they do not pin a single
    CDF point.  Duplicate strikes are averaged; monotonicity (P above is
    non-increasing in strike) is enforced with a running minimum.
    """
    raw: Dict[float, List[float]] = {}
    for r in rows:
        p = _mid(r["yes_bid"], r["yes_ask"], r["last_price"])
        if p is None or not (0.0 <= p <= 1.0):
            continue
        st = str(r["strike_type"] or "").lower()
        if st.startswith("greater"):
            strike = r["floor_strike"]
            p_above = p
        elif st.startswith("less"):
            strike = r["cap_strike"]
            p_above = 1.0 - p
        else:
            continue
        if strike is None or not math.isfinite(float(strike)):
            continue
        raw.setdefault(float(strike), []).append(p_above)
    pts = sorted((k, sum(v) / len(v)) for k, v in raw.items())
    out: List[Tuple[float, float]] = []
    running = 1.0
    for strike, p in pts:
        running = min(running, p)
        out.append((strike, running))
    return out


def _p_above(points: Sequence[Tuple[float, float]], x: float
             ) -> Optional[float]:
    """Linear interpolation of P(above x); None outside strike range."""
    if len(points) < 2 or x < points[0][0] or x > points[-1][0]:
        return None
    for (s0, p0), (s1, p1) in zip(points, points[1:]):
        if s0 <= x <= s1:
            if s1 == s0:
                return p0
            w = (x - s0) / (s1 - s0)
            return p0 + w * (p1 - p0)
    return None


def _strike_at(points: Sequence[Tuple[float, float]], p_target: float
               ) -> Optional[float]:
    """Strike where P(above) crosses p_target; None if not bracketed."""
    if len(points) < 2:
        return None
    for (s0, p0), (s1, p1) in zip(points, points[1:]):
        lo, hi = min(p0, p1), max(p0, p1)
        if lo <= p_target <= hi:
            if p0 == p1:
                return (s0 + s1) / 2.0
            w = (p0 - p_target) / (p0 - p1)
            return s0 + w * (s1 - s0)
    return None


def build_kalshi_features(
    db_path: str,
    underlying: str,
    as_of: datetime,
    *,
    index_spot: Optional[float] = None,
    max_staleness_min: float = DEFAULT_MAX_STALENESS_MIN,
) -> Dict[str, Optional[float]]:
    """PIT kalshi_* features for ``underlying`` at ``as_of``.

    Fail-open: missing db / no fresh snapshot / unknown underlying all
    return the all-None feature dict (keys always present).
    """
    out = _empty()
    if not series_for_underlying(underlying):
        return out
    if not os.path.exists(db_path):
        return out
    as_of = (as_of if as_of.tzinfo else as_of.replace(tzinfo=UTC)
             ).astimezone(UTC)
    lo = as_of - timedelta(minutes=float(max_staleness_min))

    conn = sqlite3.connect(f"file:{os.path.abspath(db_path)}?mode=ro",
                           uri=True)
    conn.row_factory = sqlite3.Row
    try:
        u = str(underlying or "").strip().upper()
        row = conn.execute(
            "SELECT MAX(observed_at) AS t FROM snapshots "
            "WHERE underlying = ? AND observed_at <= ? "
            "AND observed_at >= ?",
            (u, _iso_z(as_of), _iso_z(lo))).fetchone()
        observed_at = row["t"] if row else None
        if not observed_at:
            return out
        rows = conn.execute(
            "SELECT ticker, event_ticker, yes_bid, yes_ask, last_price, "
            "floor_strike, cap_strike, strike_type, close_time "
            "FROM snapshots WHERE underlying = ? AND observed_at = ?",
            (u, observed_at)).fetchall()
    except sqlite3.Error:
        return out
    finally:
        conn.close()
    if not rows:
        return out

    # choose the event closing soonest after as_of (nearest-expiry
    # distribution); if no close_time parses, use all rows.
    by_event: Dict[str, List[sqlite3.Row]] = {}
    closes: Dict[str, datetime] = {}
    for r in rows:
        ev = str(r["event_ticker"] or r["ticker"] or "")
        by_event.setdefault(ev, []).append(r)
        ct = _parse_utc(r["close_time"])
        if ct is not None and ct > as_of:
            prev = closes.get(ev)
            closes[ev] = ct if prev is None else min(prev, ct)
    if closes:
        best = min(closes, key=lambda e: closes[e])
        use_rows: Sequence[sqlite3.Row] = by_event[best]
    else:
        use_rows = rows

    snap_t = _parse_utc(observed_at)
    if snap_t is not None:
        out["kalshi_staleness_min"] = max(
            0.0, (as_of - snap_t).total_seconds() / 60.0)

    spreads = [r["yes_ask"] - r["yes_bid"] for r in use_rows
               if r["yes_ask"] is not None and r["yes_bid"] is not None]
    if spreads:
        out["kalshi_book_spread_mean"] = sum(spreads) / len(spreads)

    points = _cdf_points(use_rows)
    out["kalshi_n_markets"] = float(len(points))
    if len(points) >= 2:
        if index_spot is not None and math.isfinite(float(index_spot)):
            out["kalshi_p_above_spot"] = _p_above(points, float(index_spot))
        median = _strike_at(points, 0.5)
        s25 = _strike_at(points, 0.25)   # higher strike
        s75 = _strike_at(points, 0.75)   # lower strike
        if (median is not None and median > 0.0
                and s25 is not None and s75 is not None):
            out["kalshi_implied_dispersion"] = (s25 - s75) / median
    return out


def kalshi_features_for_samples(
    samples: Sequence,
    db_path: str,
    underlying: str,
    *,
    max_staleness_min: float = DEFAULT_MAX_STALENESS_MIN,
    lag_min: float = 0.0,
    index_spot_fn: Optional[Callable] = None,
) -> List[Dict[str, Optional[float]]]:
    """One kalshi feature dict per sample, computed at ``as_of - lag``.

    ``samples`` need only expose ``.as_of`` (datetime); matches
    ``oracle.lab.purged_validation.Sample``.  ``index_spot_fn(sample)``
    may supply an index-level spot per sample (None -> feature None).
    ``lag_min > 0`` deliberately uses *older* information (experiment E).
    """
    lag = timedelta(minutes=float(lag_min))
    out: List[Dict[str, Optional[float]]] = []
    for s in samples:
        spot = index_spot_fn(s) if index_spot_fn is not None else None
        out.append(build_kalshi_features(
            db_path, underlying, s.as_of - lag,
            index_spot=spot, max_staleness_min=max_staleness_min))
    return out
