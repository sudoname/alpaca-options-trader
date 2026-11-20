"""Forward labels and realized-outcome classification for Oracle 4.0.

Labels (per decision):
  * ``log_return`` / ``direction`` at fixed minute horizons (15m, 60m) and
    session horizons (rest-of-session ``1s``, five-session ``5s``);
  * ``future_rv`` realized variance proxy over the horizon window;
  * ``iv_change`` where a same-contract IV series exists (rare -- reported);
  * ``first_touch`` from the decision's own exit/outcome events.

Every label records its end time and the earliest time it became knowable.
Censoring (horizon beyond data coverage) is explicit -- status CENSORED --
and gaps inside coverage are status MISSING. Values are never imputed.

Outcome classification maps ledger evidence onto the OutcomeStatus taxonomy;
unknown outcomes stay MISSING, never silently treated as losses or nonfills.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import math

from .features import ObsIndex, _log_return, _session_key
from .ingest import LedgerEvent
from .schema import (
    DecisionRecord,
    LabelRow,
    LabelStatus,
    OutcomeRecord,
    OutcomeStatus,
)

__all__ = ["LabelConfig", "build_labels", "classify_outcome", "build_iv_index"]


@dataclass(frozen=True)
class LabelConfig:
    """Label-construction knobs (recorded in the dataset manifest)."""
    minute_horizons: tuple = (15, 60)
    tolerance_frac: float = 0.5      # obs accepted in [t+h, t+h*(1+frac)]
    neutral_band: float = 0.001      # |log return| below this => neutral (0)
    session_ahead: int = 4           # '5s' = close of the 4th later session
    rv_min_returns: int = 3


# --------------------------------------------------------------------------- #
# IV series (per contract symbol) for iv_change labels
# --------------------------------------------------------------------------- #

def build_iv_index(decisions: List[DecisionRecord]
                   ) -> Dict[str, List[Tuple[datetime, datetime, float]]]:
    """(event_at, available_at, iv) per option symbol, sorted by event time."""
    out: Dict[str, List[Tuple[datetime, datetime, float]]] = {}
    for d in decisions:
        if d.symbol and d.iv is not None:
            out.setdefault(d.symbol, []).append(
                (d.as_of, d.available_at, float(d.iv)))
    for series in out.values():
        series.sort(key=lambda x: x[0])
    return out


def _iv_at_or_after(series: List[Tuple[datetime, datetime, float]],
                    t: datetime, not_after: datetime
                    ) -> Optional[Tuple[datetime, datetime, float]]:
    for ev_at, av_at, iv in series:
        if ev_at >= t:
            return (ev_at, av_at, iv) if ev_at <= not_after else None
    return None


# --------------------------------------------------------------------------- #
# label builder
# --------------------------------------------------------------------------- #

def _direction(log_ret: float, band: float) -> float:
    if log_ret > band:
        return 1.0
    if log_ret < -band:
        return -1.0
    return 0.0


def build_labels(
    decision: DecisionRecord,
    index: ObsIndex,
    coverage_end: datetime,
    events: Optional[List[LedgerEvent]] = None,
    iv_index: Optional[Dict[str, List[Tuple[datetime, datetime, float]]]] = None,
    config: LabelConfig = LabelConfig(),
) -> List[LabelRow]:
    """All labels for one decision. ``coverage_end`` = last obs event time in
    the dataset window; horizons past it are CENSORED, gaps inside it MISSING.
    """
    t = decision.as_of
    u = decision.underlying
    labels: List[LabelRow] = []

    p_start = decision.underlying_price
    if p_start is None:
        start_obs = index.asof(u, t)
        p_start = start_obs.price if start_obs else None

    def _ret_labels(horizon: str, end_obs, target_t: datetime,
                    censored: bool, meta: Dict) -> None:
        """Append log_return + direction for one horizon."""
        if end_obs is not None and p_start is not None:
            value = _log_return(end_obs.price, p_start)
            status = LabelStatus.OK if value is not None else LabelStatus.MISSING
            labels.append(LabelRow(
                decision_id=decision.decision_id, horizon=horizon,
                label_type="log_return", value=value, status=status,
                label_end=end_obs.event_at, available_at=end_obs.available_at,
                meta=meta))
            labels.append(LabelRow(
                decision_id=decision.decision_id, horizon=horizon,
                label_type="direction",
                value=(_direction(value, config.neutral_band)
                       if value is not None else None),
                status=status,
                label_end=end_obs.event_at, available_at=end_obs.available_at,
                meta=meta))
        else:
            status = LabelStatus.CENSORED if censored else LabelStatus.MISSING
            why = dict(meta)
            why["why"] = ("start_price_missing" if (end_obs is not None)
                          else ("beyond_coverage" if censored else "obs_gap"))
            for label_type in ("log_return", "direction"):
                labels.append(LabelRow(
                    decision_id=decision.decision_id, horizon=horizon,
                    label_type=label_type, value=None, status=status,
                    label_end=target_t, available_at=None, meta=why))

    # ---- fixed minute horizons --------------------------------------------
    for h_min in config.minute_horizons:
        horizon = f"{h_min}m"
        target_t = t + timedelta(minutes=h_min)
        not_after = t + timedelta(minutes=h_min * (1.0 + config.tolerance_frac))
        end_obs = index.first_at_or_after(u, target_t, not_after=not_after)
        censored = target_t > coverage_end
        meta = {"target_t": target_t.isoformat()}
        _ret_labels(horizon, end_obs, target_t, censored, meta)

        # future realized variance over (t, label_end]
        rv_end = end_obs.event_at if end_obs else min(target_t, coverage_end)
        win = index.window(u, t, rv_end, available_by=rv_end)
        prices = [o.price for o in win]
        rets = [r for r in
                (_log_return(b, a) for a, b in zip(prices, prices[1:]))
                if r is not None]
        if len(rets) >= config.rv_min_returns:
            mean = sum(rets) / len(rets)
            var = sum((r - mean) ** 2 for r in rets) / (len(rets) - 1)
            labels.append(LabelRow(
                decision_id=decision.decision_id, horizon=horizon,
                label_type="future_rv", value=math.sqrt(var),
                status=LabelStatus.OK,
                label_end=win[-1].event_at,
                available_at=max(o.available_at for o in win),
                meta={"n_returns": len(rets)}))
        else:
            labels.append(LabelRow(
                decision_id=decision.decision_id, horizon=horizon,
                label_type="future_rv", value=None,
                status=(LabelStatus.CENSORED if censored else LabelStatus.MISSING),
                label_end=target_t, available_at=None,
                meta={"n_returns": len(rets)}))

        # IV change where a same-contract IV series exists
        iv_series = (iv_index or {}).get(decision.symbol or "", [])
        if decision.iv is not None and iv_series:
            hit = _iv_at_or_after(iv_series, target_t, not_after)
            if hit is not None:
                ev_at, av_at, iv_end = hit
                labels.append(LabelRow(
                    decision_id=decision.decision_id, horizon=horizon,
                    label_type="iv_change", value=iv_end - decision.iv,
                    status=LabelStatus.OK, label_end=ev_at,
                    available_at=av_at, meta={}))
                continue
        labels.append(LabelRow(
            decision_id=decision.decision_id, horizon=horizon,
            label_type="iv_change", value=None,
            status=(LabelStatus.CENSORED if censored else LabelStatus.MISSING),
            label_end=target_t, available_at=None,
            meta={"why": ("no_start_iv" if decision.iv is None
                          else "no_future_iv_obs")}))

    # ---- session horizons --------------------------------------------------
    sessions = index.sessions(u)
    this_session = _session_key(t)
    coverage_session = _session_key(coverage_end)

    # '1s' = rest of the decision's own session
    end_obs = index.last_of_session(u, this_session)
    # the final session in coverage may be truncated mid-session -> censored
    censored_1s = this_session >= coverage_session
    if censored_1s:
        _ret_labels("1s", None, t, True, {"session": this_session})
    else:
        _ret_labels("1s", end_obs, t, False, {"session": this_session})

    # '5s' = close of the Nth later observed session
    try:
        i = sessions.index(this_session)
    except ValueError:
        i = -1
    target_idx = i + config.session_ahead
    if i >= 0 and target_idx < len(sessions) \
            and sessions[target_idx] < coverage_session:
        target_session = sessions[target_idx]
        end_obs = index.last_of_session(u, target_session)
        _ret_labels("5s", end_obs, t, False, {"session": target_session})
    else:
        _ret_labels("5s", None, t, True,
                    {"why": "insufficient_future_sessions"})

    # ---- first touch (target/stop) from the decision's own events ----------
    labels.append(_first_touch_label(decision, events or [], coverage_end))
    return labels


# Exit reason -> (hazard cause, fine-grained mechanism). Keys are the live
# bot's actual outcome vocabulary (matched lowercased, EXACT -- 4.0.1-A fix:
# the old substring rules classified TIME_STOP and dynamic_trailing_stop as
# price stops and roll_failed_take_profit as a profit-target touch). The
# coarse cause stays within HAZARD_CAUSES = ('stop','target','other_exit');
# the detail preserves the true mechanism for finer models (Stage E).
_TOUCH_BY_REASON: Dict[str, Tuple[str, str]] = {
    # configured price stop-loss
    "dynamic_stop_loss": ("stop", "stop"),
    "stop_loss": ("stop", "stop"),
    "stop": ("stop", "stop"),
    # trailing stop locks in gains; it is NOT the configured loss barrier
    "dynamic_trailing_stop": ("other_exit", "trailing_stop"),
    "trailing_stop": ("other_exit", "trailing_stop"),
    # time-based exits
    "time_stop": ("other_exit", "time_stop"),
    "eod_close": ("other_exit", "eod_close"),
    # profit target
    "take_profit": ("target", "target"),
    "profit_target": ("target", "target"),
    "roll_take_profit": ("target", "roll_target"),
    "target": ("target", "target"),
    # explicit non-barrier exits
    "roll_failed_take_profit": ("other_exit", "roll_failed_target"),
    "dynamic_exit": ("other_exit", "dynamic_exit"),
    "stale_carryover": ("other_exit", "stale_carryover"),
    "closed": ("other_exit", "manual_close"),
    "reconciled_closed": ("other_exit", "reconciled"),
    "reconciled_fill_close": ("other_exit", "reconciled"),
    "expired": ("other_exit", "expiry"),
}


def _first_touch_label(decision: DecisionRecord, events: List[LedgerEvent],
                       coverage_end: datetime) -> LabelRow:
    """Which barrier resolved the trade first, and after how long (minutes).

    value = minutes from decision to exit/outcome; meta['touch'] in
    {'target','stop','other_exit'} via the exact-match reason map
    (meta['touch_detail'] keeps the finer mechanism; unmapped reasons get
    meta['reason_unmapped']). Simulated-class outcomes resolve to status
    SIMULATED, never OK. Censored when a filled trade has no exit inside
    coverage; MISSING when the decision never traded (no fill path).
    """
    by_type: Dict[str, LedgerEvent] = {}
    for e in events:
        by_type.setdefault(e.event_type, e)

    outcome = by_type.get("outcome")
    exit_ev = by_type.get("exit")
    fill = by_type.get("fill")

    resolved = outcome or exit_ev
    if resolved is not None:
        reason = ""
        if outcome is not None:
            reason = str(outcome.payload.get("reason") or "")
        mapped = _TOUCH_BY_REASON.get(reason.strip().lower())
        touch, detail = mapped if mapped else ("other_exit", "unknown")
        simulated = (str(outcome.payload.get("class")) == "simulated"
                     if outcome is not None else False)
        minutes = (resolved.event_at - decision.as_of).total_seconds() / 60.0
        meta: Dict[str, object] = {
            "touch": touch, "touch_detail": detail,
            "reason": reason or None, "simulated": simulated}
        if mapped is None and reason:
            meta["reason_unmapped"] = True
        return LabelRow(
            decision_id=decision.decision_id, horizon="event",
            label_type="first_touch", value=minutes,
            status=LabelStatus.SIMULATED if simulated else LabelStatus.OK,
            label_end=resolved.event_at, available_at=resolved.available_at,
            meta=meta)

    if fill is not None:
        # traded but unresolved inside coverage -> censored at coverage end
        return LabelRow(
            decision_id=decision.decision_id, horizon="event",
            label_type="first_touch", value=None, status=LabelStatus.CENSORED,
            label_end=coverage_end, available_at=None,
            meta={"why": "filled_no_exit_in_coverage"})

    return LabelRow(
        decision_id=decision.decision_id, horizon="event",
        label_type="first_touch", value=None, status=LabelStatus.MISSING,
        label_end=None, available_at=None,
        meta={"why": "never_filled"})


# --------------------------------------------------------------------------- #
# outcome classification
# --------------------------------------------------------------------------- #

def classify_outcome(decision: DecisionRecord,
                     events: List[LedgerEvent]) -> OutcomeRecord:
    """Map ledger evidence for one decision onto the OutcomeStatus taxonomy."""
    by_type: Dict[str, List[LedgerEvent]] = {}
    for e in events:
        by_type.setdefault(e.event_type, []).append(e)

    def _first(kind: str) -> Optional[LedgerEvent]:
        evs = by_type.get(kind)
        return evs[0] if evs else None

    fill = _first("fill")
    outcome = _first("outcome")

    # shadow decisions are counterfactual by construction
    if decision.kind != "candidate":
        meta: Dict[str, object] = {"kind": decision.kind}
        if outcome is not None:
            meta["simulated"] = outcome.payload
        return OutcomeRecord(
            decision_id=decision.decision_id,
            status=OutcomeStatus.COUNTERFACTUAL,
            gross_pnl_pct=None, net_pnl_dollars=None, net_pnl_pct=None,
            reason=None, fill_price=None, fill_qty=None,
            outcome_at=outcome.event_at if outcome else None, meta=meta)

    # gate rejected before submission
    if not decision.accepted:
        return OutcomeRecord(
            decision_id=decision.decision_id,
            status=OutcomeStatus.POLICY_SKIP,
            gross_pnl_pct=None, net_pnl_dollars=None, net_pnl_pct=None,
            reason=decision.reject_code, fill_price=None, fill_qty=None,
            outcome_at=None, meta={})

    submitted = _first("submitted")
    cancel = _first("cancel") or _first("reject")

    if outcome is not None and str(outcome.payload.get("class")) == "realized":
        p = outcome.payload

        def _num(key: str) -> Optional[float]:
            v = p.get(key)
            return float(v) if isinstance(v, (int, float)) else None

        net_dollars = _num("net_pnl_dollars")
        status = (OutcomeStatus.FILLS_ACTUAL if net_dollars is not None
                  else OutcomeStatus.MARK_DERIVED)
        return OutcomeRecord(
            decision_id=decision.decision_id, status=status,
            gross_pnl_pct=_num("gross_pnl_pct"),
            net_pnl_dollars=net_dollars,
            net_pnl_pct=_num("net_pnl_pct"),
            reason=str(p.get("reason")) if p.get("reason") else None,
            fill_price=fill.price if fill else None,
            fill_qty=fill.qty if fill else None,
            outcome_at=outcome.event_at, meta={})

    if submitted is not None and fill is None and cancel is not None:
        return OutcomeRecord(
            decision_id=decision.decision_id,
            status=OutcomeStatus.CONFIRMED_NONFILL,
            gross_pnl_pct=None, net_pnl_dollars=None, net_pnl_pct=None,
            reason=cancel.event_type, fill_price=None, fill_qty=None,
            outcome_at=cancel.event_at, meta={})

    # accepted but unresolved -- stays MISSING, with the stage recorded
    if fill is not None:
        stage = "filled_no_outcome"
    elif submitted is not None:
        stage = "submitted_unresolved"
    else:
        stage = "no_submission"
    return OutcomeRecord(
        decision_id=decision.decision_id, status=OutcomeStatus.MISSING,
        gross_pnl_pct=None, net_pnl_dollars=None, net_pnl_pct=None,
        reason=None,
        fill_price=fill.price if fill else None,
        fill_qty=fill.qty if fill else None,
        outcome_at=None, meta={"stage": stage})
