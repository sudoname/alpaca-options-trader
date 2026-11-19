"""Point-in-time feature construction for Oracle 4.0 decisions.

Every feature uses only observations whose *event* happened at or before the
decision's ``as_of`` AND whose *availability* timestamp is at or before
``as_of`` (no future joins, no look-ahead through late-arriving data).

Missing inputs produce ``None`` features -- never zero, never imputed.
"""
from __future__ import annotations

import math
from bisect import bisect_right
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional

from .schema import DecisionRecord, FeatureRow, UnderlyingObs, parse_utc

__all__ = ["FeatureConfig", "ObsIndex", "build_feature_row"]


@dataclass(frozen=True)
class FeatureConfig:
    """Knobs for feature construction (recorded in the dataset manifest)."""
    return_windows_min: tuple = (15, 30, 60)
    stale_factor: float = 1.0        # base obs must lie within [t-2w, t-w]
    rsi_period: int = 14
    ema_span: int = 20
    rv_min_returns: int = 3
    session_min_obs: int = 2


def _session_key(dt: datetime) -> str:
    """Exchange-session key = UTC date (ledger spans US regular hours only)."""
    return dt.strftime("%Y-%m-%d")


class ObsIndex:
    """Per-underlying sorted price series with availability-aware lookups."""

    def __init__(self, obs_by_underlying: Dict[str, List[UnderlyingObs]]):
        self._series = obs_by_underlying
        self._times: Dict[str, List[datetime]] = {
            u: [o.event_at for o in series]
            for u, series in obs_by_underlying.items()
        }

    def underlyings(self) -> List[str]:
        return sorted(self._series.keys())

    def asof(self, underlying: str, t: datetime,
             available_by: Optional[datetime] = None) -> Optional[UnderlyingObs]:
        """Latest obs with event_at <= t and available_at <= available_by.

        ``available_by`` defaults to ``t`` (strict point-in-time semantics).
        """
        series = self._series.get(underlying)
        if not series:
            return None
        cutoff = available_by if available_by is not None else t
        i = bisect_right(self._times[underlying], t) - 1
        while i >= 0:
            obs = series[i]
            if obs.available_at <= cutoff:
                return obs
            i -= 1
        return None

    def window(self, underlying: str, start: datetime, end: datetime,
               available_by: Optional[datetime] = None) -> List[UnderlyingObs]:
        """All obs with start < event_at <= end, available by ``available_by``
        (default ``end``), in event-time order."""
        series = self._series.get(underlying)
        if not series:
            return []
        cutoff = available_by if available_by is not None else end
        times = self._times[underlying]
        lo = bisect_right(times, start)
        hi = bisect_right(times, end)
        return [o for o in series[lo:hi] if o.available_at <= cutoff]

    def first_at_or_after(self, underlying: str, t: datetime,
                          not_after: Optional[datetime] = None
                          ) -> Optional[UnderlyingObs]:
        """Earliest obs with event_at >= t (and <= not_after if given).

        Used by the *label* builder, where future observations are the point;
        availability is reported on the label, not filtered here.
        """
        series = self._series.get(underlying)
        if not series:
            return None
        times = self._times[underlying]
        i = bisect_right(times, t)
        if i > 0 and times[i - 1] == t:
            i -= 1          # inclusive lower bound
        if i >= len(series):
            return None
        obs = series[i]
        if not_after is not None and obs.event_at > not_after:
            return None
        return obs

    def session_obs(self, underlying: str, t: datetime,
                    available_by: Optional[datetime] = None
                    ) -> List[UnderlyingObs]:
        """Same-session obs with event_at <= t, availability-filtered."""
        day_start = t.replace(hour=0, minute=0, second=0, microsecond=0)
        return self.window(underlying, day_start - timedelta(seconds=1), t,
                           available_by=available_by)

    def last_of_session(self, underlying: str, session: str
                        ) -> Optional[UnderlyingObs]:
        """Final obs of a session (label use; availability reported on label)."""
        series = self._series.get(underlying)
        if not series:
            return None
        for obs in reversed(series):
            key = _session_key(obs.event_at)
            if key == session:
                return obs
            if key < session:
                return None
        return None

    def sessions(self, underlying: str) -> List[str]:
        """Sorted distinct session keys with at least one obs."""
        series = self._series.get(underlying)
        if not series:
            return []
        seen: List[str] = []
        for obs in series:
            key = _session_key(obs.event_at)
            if not seen or seen[-1] != key:
                seen.append(key)
        return seen


# --------------------------------------------------------------------------- #
# indicator helpers (None when insufficient data)
# --------------------------------------------------------------------------- #

def _log_return(p_now: float, p_then: float) -> Optional[float]:
    if p_now <= 0 or p_then <= 0:
        return None
    return math.log(p_now / p_then)


def _rsi(prices: List[float], period: int) -> Optional[float]:
    if len(prices) < period + 1:
        return None
    gains = losses = 0.0
    for a, b in zip(prices[-period - 1:-1], prices[-period:]):
        d = b - a
        if d >= 0:
            gains += d
        else:
            losses -= d
    if gains + losses == 0:
        return 50.0
    return 100.0 * gains / (gains + losses)


def _ema(prices: List[float], span: int) -> Optional[float]:
    if len(prices) < 2:
        return None
    k = 2.0 / (span + 1.0)
    ema = prices[0]
    for p in prices[1:]:
        ema = p * k + ema * (1.0 - k)
    return ema


def _rv(prices: List[float], min_returns: int) -> Optional[float]:
    rets = []
    for a, b in zip(prices, prices[1:]):
        r = _log_return(b, a)
        if r is not None:
            rets.append(r)
    if len(rets) < min_returns:
        return None
    mean = sum(rets) / len(rets)
    var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
    return math.sqrt(var)


# --------------------------------------------------------------------------- #
# feature row
# --------------------------------------------------------------------------- #

def build_feature_row(decision: DecisionRecord, index: ObsIndex,
                      config: FeatureConfig = FeatureConfig()) -> FeatureRow:
    """Construct the point-in-time feature row for one decision."""
    t = decision.as_of
    u = decision.underlying
    feats: Dict[str, Optional[float]] = {}
    quality: Dict[str, object] = {}

    # --- underlying price at decision time (ledger value wins; index backs up)
    spot = decision.underlying_price
    asof_obs = index.asof(u, t)
    if spot is None and asof_obs is not None:
        spot = asof_obs.price
    feats["underlying_price"] = spot
    quality["spot_from_ledger"] = decision.underlying_price is not None
    quality["asof_gap_s"] = (
        (t - asof_obs.event_at).total_seconds() if asof_obs else None
    )

    # --- trailing returns over fixed windows
    for w_min in config.return_windows_min:
        name = f"ret_{w_min}m"
        if spot is None:
            feats[name] = None
            continue
        w = timedelta(minutes=w_min)
        base = index.asof(u, t - w)
        max_stale = w * (1.0 + config.stale_factor)
        if base is None or (t - base.event_at) > max_stale:
            feats[name] = None
        else:
            feats[name] = _log_return(spot, base.price)

    # --- session-based series indicators
    sess = index.session_obs(u, t)
    prices = [o.price for o in sess]
    if spot is not None and (not prices or prices[-1] != spot):
        prices = prices + [spot]
    quality["session_obs_used"] = len(sess)

    if len(prices) >= config.session_min_obs and prices[0] > 0 and spot:
        feats["ret_session"] = _log_return(spot, prices[0])
    else:
        feats["ret_session"] = None

    feats["rsi"] = _rsi(prices, config.rsi_period)
    ema = _ema(prices, config.ema_span)
    feats["ema_ratio"] = (
        (spot / ema - 1.0) if (ema and spot and ema > 0) else None
    )
    feats["rv_session"] = _rv(prices, config.rv_min_returns)

    # --- time-of-day / calendar
    first = sess[0] if sess else None
    feats["minutes_since_first_obs"] = (
        (t - first.event_at).total_seconds() / 60.0 if first else None
    )
    feats["hour_utc"] = float(t.hour) + t.minute / 60.0
    feats["day_of_week"] = float(t.weekday())

    # --- ledger signals (point-in-time by construction)
    for key in ("momentum", "volatility"):
        v = decision.signals.get(key)
        feats[f"sig_{key}"] = float(v) if isinstance(v, (int, float)) else None
    for key in ("sma", "std", "z", "window"):
        v = decision.signals.get(key)
        feats[f"shadow_{key}"] = float(v) if isinstance(v, (int, float)) else None

    # --- option contract features
    bid, ask = decision.quote_bid, decision.quote_ask
    mid = (bid + ask) / 2.0 if (bid is not None and ask is not None) else None
    feats["quote_mid"] = mid
    feats["spread_abs"] = (
        (ask - bid) if (bid is not None and ask is not None) else None
    )
    feats["spread_pct"] = (
        (ask - bid) / mid if (mid not in (None, 0.0) and bid is not None
                              and ask is not None and mid > 0) else None
    )
    feats["iv"] = decision.iv
    feats["delta"] = decision.delta

    if (spot and decision.strike and spot > 0 and decision.strike > 0
            and decision.option_type in ("call", "put")):
        sign = 1.0 if decision.option_type == "call" else -1.0
        feats["moneyness_log"] = sign * math.log(spot / decision.strike)
    else:
        feats["moneyness_log"] = None

    exp = parse_utc(f"{decision.expiration}T21:00:00+00:00") \
        if decision.expiration else None
    feats["dte_days"] = (
        (exp - t).total_seconds() / 86400.0 if exp else None
    )

    # --- baseline opinions (prefixed so experiments can exclude them)
    feats["baseline_score"] = decision.score
    feats["baseline_p_win"] = decision.p_win
    feats["baseline_ev"] = decision.ev

    quality["has_quote"] = bid is not None and ask is not None
    quality["missing_features"] = sorted(
        k for k, v in feats.items() if v is None
    )

    return FeatureRow(
        decision_id=decision.decision_id,
        snapshot_id=decision.snapshot_id,
        as_of=t,
        available_at=decision.available_at,
        underlying=u,
        symbol=decision.symbol,
        kind=decision.kind,
        session=_session_key(t),
        features=feats,
        quality=quality,
    )
