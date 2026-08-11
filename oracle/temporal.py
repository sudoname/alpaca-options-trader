"""
Temporal integrity / look-ahead protection (formalized contract + guard).

``market_view.MarketView`` already enforces point-in-time access (every accessor
filters to ``close_dt <= as_of`` and appends to ``self.audit``). This module
turns that implicit guarantee into an explicit, testable contract that the Lab,
the sim driver and (in strict mode) any offline replay can assert against.

Three ideas:

  * ``TemporalStamp`` — the three timestamps that define whether a datum was
    legally usable at a decision point:
        event_timestamp      when the thing happened (a bar's window, an
                             earnings release, a quote print).
        available_timestamp  the earliest wall-clock time a decider could have
                             KNOWN it (a daily bar is not available until its
                             session close; a quote is available at its print).
        decision_timestamp   the ``as_of`` of the decision consuming it.
    A feature is valid iff ``event <= available <= decision``.

  * ``conservative_available_ts`` — the availability policy. When only the event
    time is known, it returns the *latest-plausible* moment the datum became
    usable (bar -> session close, intraday bar -> window end, earnings/news ->
    release + a conservative delay), so the check is strict, never optimistic.

  * ``assert_no_lookahead`` / ``TemporalGuard`` — fold a ``MarketView.audit``
    (or a stream of stamps) and confirm nothing stamped after ``as_of`` ever
    escaped. Strict mode (flag ``ENABLE_TEMPORAL_INTEGRITY``) raises
    ``LookAheadError``; default (OFF) logs and returns False (fail-open) so the
    live path is never destabilized by this check.

Live is unaffected: ``LiveMarketView`` uses ``as_of = now`` so every real datum
is trivially ``<= now`` — the guard is a research/replay safety net, not a live
gate.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, time, timedelta
from typing import Any, Dict, List, Optional, Tuple

LOG_TAG = "[TEMPORAL]"

# Conservative availability lags (seconds) added to an event time when we must
# infer availability. Deliberately pessimistic — better to reject a borderline
# datum in research than to let a leak through.
_AVAILABILITY_LAG_SEC: Dict[str, float] = {
    "earnings": 0.0,          # known at the release timestamp itself
    "news": 0.0,              # known at publish time
    "analyst_rating": 0.0,    # known at publish time
    "catalyst": 0.0,
    "quote": 0.0,             # a quote is usable at its print timestamp
    "option_quote": 0.0,
    "fundamental": 24 * 3600.0,  # filings often usable next session
}


class LookAheadError(Exception):
    """Raised (strict mode only) when a datum stamped after the decision's
    ``as_of`` was used to build a feature."""


# --------------------------------------------------------------------------- #
# Timestamp coercion
# --------------------------------------------------------------------------- #
def to_dt(value: Any) -> Optional[datetime]:
    """Best-effort parse to a naive datetime (drops tz to match MarketView's
    single naive frame). Returns None on failure — never raises."""
    if value is None or isinstance(value, bool):
        return None
    if isinstance(value, datetime):
        return value
    if isinstance(value, (int, float)):
        try:
            return datetime.fromtimestamp(float(value))
        except (ValueError, OverflowError, OSError):
            return None
    try:
        s = str(value).strip().replace("Z", "")
        if "T" in s or ":" in s:
            # Drop a tz offset if present (keep wall-clock components).
            head = s[:11]
            tail = s[11:]
            for sep in ("+",):
                if sep in tail:
                    tail = tail.split(sep)[0]
            s = (head + tail)[:19]
            return datetime.fromisoformat(s)
        return datetime.fromisoformat(s[:10])
    except (ValueError, AttributeError, TypeError):
        return None


def conservative_available_ts(event_ts: Any, kind: str, *,
                              close_time: time = time(16, 0),
                              interval_minutes: Optional[int] = None
                              ) -> Optional[datetime]:
    """Latest-plausible moment a datum of ``kind`` became usable.

      * ``daily_bar`` / ``bar`` / ``vix_bar`` -> that calendar date's session
        close (a daily bar is NOT knowable until the close).
      * ``intraday_bar`` / ``intraday_bars`` -> event + ``interval_minutes``
        (the window must fully elapse). Defaults to 1 minute.
      * everything else -> event + a conservative per-kind lag.
    """
    ev = to_dt(event_ts)
    if ev is None:
        return None
    k = (kind or "").strip().lower()
    if k in ("daily_bar", "bar", "vix_bar", "vix"):
        return datetime.combine(ev.date(), close_time)
    if k in ("intraday_bar", "intraday_bars"):
        return ev + timedelta(minutes=max(1, int(interval_minutes or 1)))
    lag = _AVAILABILITY_LAG_SEC.get(k, 0.0)
    return ev + timedelta(seconds=lag)


# --------------------------------------------------------------------------- #
# TemporalStamp
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class TemporalStamp:
    """The three timestamps that decide whether a datum was legally usable."""

    event_timestamp: Optional[datetime]
    available_timestamp: Optional[datetime]
    decision_timestamp: Optional[datetime]
    kind: str = "feature"
    ident: str = ""

    @staticmethod
    def make(event_ts: Any, decision_ts: Any, *, kind: str = "feature",
             ident: str = "", available_ts: Any = None,
             interval_minutes: Optional[int] = None) -> "TemporalStamp":
        ev = to_dt(event_ts)
        dec = to_dt(decision_ts)
        av = to_dt(available_ts)
        if av is None:
            av = conservative_available_ts(ev, kind,
                                           interval_minutes=interval_minutes)
        return TemporalStamp(event_timestamp=ev, available_timestamp=av,
                             decision_timestamp=dec, kind=str(kind),
                             ident=str(ident))

    def is_valid(self) -> bool:
        """Primary contract: available_timestamp <= decision_timestamp."""
        ok, _ = validate_feature(self)
        return ok

    def to_dict(self) -> dict:
        return {
            "kind": self.kind,
            "ident": self.ident,
            "event_timestamp": (self.event_timestamp.isoformat()
                                if self.event_timestamp else None),
            "available_timestamp": (self.available_timestamp.isoformat()
                                    if self.available_timestamp else None),
            "decision_timestamp": (self.decision_timestamp.isoformat()
                                   if self.decision_timestamp else None),
        }


def validate_feature(stamp: TemporalStamp) -> Tuple[bool, str]:
    """Return ``(ok, reason)``. A feature is valid iff it was *available* at or
    before the decision and its event did not post-date its availability
    (causality). Missing timestamps fail closed for the availability check but
    are reported distinctly so callers can decide."""
    if not isinstance(stamp, TemporalStamp):
        return (False, "not_a_stamp")
    dec = stamp.decision_timestamp
    av = stamp.available_timestamp
    ev = stamp.event_timestamp
    if dec is None:
        return (False, "missing_decision_ts")
    if av is None:
        return (False, "missing_available_ts")
    if av > dec:
        return (False, "available_after_decision")   # the look-ahead leak
    if ev is not None and ev > av:
        return (False, "event_after_available")       # impossible causality
    return (True, "ok")


# --------------------------------------------------------------------------- #
# Flag resolution
# --------------------------------------------------------------------------- #
def _strict_enabled(config: Any = None) -> bool:
    """Resolve ``ENABLE_TEMPORAL_INTEGRITY`` (default OFF). Accepts a ConfigLoader
    (or anything with ``get_bool``); otherwise falls back to a fresh loader, then
    to ``os.environ``. Fail-open to False on any error."""
    try:
        if config is not None and hasattr(config, "get_bool"):
            return bool(config.get_bool("ENABLE_TEMPORAL_INTEGRITY", False))
        from config_loader import ConfigLoader
        return bool(ConfigLoader().get_bool("ENABLE_TEMPORAL_INTEGRITY", False))
    except Exception:
        import os
        return str(os.environ.get("ENABLE_TEMPORAL_INTEGRITY", "")
                   ).strip().lower() in ("1", "true", "yes", "on")


# --------------------------------------------------------------------------- #
# Audit folding
# --------------------------------------------------------------------------- #
def audit_violations(market_view: Any) -> List[Dict[str, Any]]:
    """Fold a ``MarketView.audit`` and return every record whose ``ts`` is after
    the view's ``as_of`` (i.e. leaked). Empty list == clean. Never raises."""
    out: List[Dict[str, Any]] = []
    try:
        as_of = market_view.as_of
        audit = getattr(market_view, "audit", None) or []
    except Exception:
        return out
    for rec in audit:
        try:
            ts = rec.get("ts")
        except AttributeError:
            continue
        if ts is not None and as_of is not None and ts > as_of:
            out.append(dict(rec))
    return out


def assert_no_lookahead(market_view: Any, *, strict: Optional[bool] = None,
                        config: Any = None) -> bool:
    """Assert that a ``MarketView`` handed out nothing stamped after ``as_of``.

    Returns True when clean. On a violation: strict -> raise ``LookAheadError``;
    non-strict -> log and return False (fail-open). ``strict`` defaults to the
    ``ENABLE_TEMPORAL_INTEGRITY`` flag.
    """
    if strict is None:
        strict = _strict_enabled(config)
    violations = audit_violations(market_view)
    if not violations:
        return True
    msg = (f"{LOG_TAG} look-ahead: {len(violations)} datum(s) stamped after "
           f"as_of={getattr(market_view, 'as_of', None)}; "
           f"first={violations[0]}")
    if strict:
        raise LookAheadError(msg)
    print(msg)
    return False


# --------------------------------------------------------------------------- #
# TemporalGuard
# --------------------------------------------------------------------------- #
class TemporalGuard:
    """Context wrapper for one decision. Records ``decision_timestamp`` and lets
    the caller ``check(kind, event_ts, ...)`` each datum before it enters a
    feature. Collects violations; in strict mode ``check`` raises on the first
    invalid datum. Fail-open when not strict."""

    def __init__(self, decision_ts: Any, *, strict: Optional[bool] = None,
                 config: Any = None):
        self.decision_ts = to_dt(decision_ts)
        self.strict = _strict_enabled(config) if strict is None else bool(strict)
        self.checked: List[TemporalStamp] = []
        self.violations: List[TemporalStamp] = []

    def __enter__(self) -> "TemporalGuard":
        return self

    def __exit__(self, exc_type, exc, tb) -> bool:
        return False  # never swallow exceptions

    def check(self, kind: str, event_ts: Any, *, ident: str = "",
              available_ts: Any = None,
              interval_minutes: Optional[int] = None) -> bool:
        stamp = TemporalStamp.make(event_ts, self.decision_ts, kind=kind,
                                   ident=ident, available_ts=available_ts,
                                   interval_minutes=interval_minutes)
        self.checked.append(stamp)
        ok, reason = validate_feature(stamp)
        if not ok:
            self.violations.append(stamp)
            msg = (f"{LOG_TAG} invalid feature ({reason}) kind={kind} "
                   f"id={ident} {stamp.to_dict()}")
            if self.strict:
                raise LookAheadError(msg)
            print(msg)
        return ok

    def ok(self) -> bool:
        """True iff no violation has been recorded."""
        return not self.violations


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    dec = datetime(2024, 1, 5, 16, 0)   # decision at Friday close

    # --- availability policy ---------------------------------------------- #
    # A daily bar for the decision date is available at that date's close.
    av = conservative_available_ts("2024-01-05", "daily_bar")
    if av != datetime(2024, 1, 5, 16, 0):
        print("FAIL: daily bar availability", av); ok = False
    # An intraday 5-min bar starting 09:30 is available at 09:35.
    av_i = conservative_available_ts("2024-01-05T09:30:00", "intraday_bar",
                                     interval_minutes=5)
    if av_i != datetime(2024, 1, 5, 9, 35):
        print("FAIL: intraday availability", av_i); ok = False

    # --- valid: a prior-day bar known before the decision ----------------- #
    s_ok = TemporalStamp.make("2024-01-04", dec, kind="daily_bar", ident="AAA")
    if not s_ok.is_valid():
        print("FAIL: prior-day bar should be valid"); ok = False

    # --- future bar rejected ---------------------------------------------- #
    s_future_bar = TemporalStamp.make("2024-01-08", dec, kind="daily_bar")
    v, reason = validate_feature(s_future_bar)
    if v or reason != "available_after_decision":
        print("FAIL: future bar not rejected", v, reason); ok = False

    # --- future earnings rejected ----------------------------------------- #
    s_earn = TemporalStamp.make("2024-01-06T08:00:00", dec, kind="earnings")
    if validate_feature(s_earn)[0]:
        print("FAIL: future earnings not rejected"); ok = False

    # --- future analyst rating rejected ----------------------------------- #
    s_rating = TemporalStamp.make("2024-01-05T16:30:00", dec,
                                  kind="analyst_rating")
    if validate_feature(s_rating)[0]:
        print("FAIL: post-close analyst rating not rejected"); ok = False

    # --- future option quote rejected ------------------------------------- #
    s_q = TemporalStamp.make("2024-01-05T16:00:01", dec, kind="option_quote")
    if validate_feature(s_q)[0]:
        print("FAIL: future option quote not rejected"); ok = False
    # A quote one second before the decision is fine.
    s_q_ok = TemporalStamp.make("2024-01-05T15:59:59", dec, kind="option_quote")
    if not s_q_ok.is_valid():
        print("FAIL: pre-decision quote should be valid"); ok = False

    # --- realized outcome cannot enter features --------------------------- #
    # An "outcome" measured at the horizon end (after the decision) is a future
    # event -> rejected, which is exactly the label-leakage guard.
    s_outcome = TemporalStamp.make("2024-01-12T16:00:00", dec, kind="daily_bar",
                                   ident="AAA:forward_return")
    if s_outcome.is_valid():
        print("FAIL: realized outcome leaked into features"); ok = False

    # --- assert_no_lookahead over a MarketView.audit ---------------------- #
    class _FakeMV:
        def __init__(self, as_of, audit):
            self.as_of = as_of
            self.audit = audit

    clean = _FakeMV(dec, [{"kind": "daily_bar", "ts": datetime(2024, 1, 4, 16),
                           "id": "AAA"}])
    if not assert_no_lookahead(clean, strict=True):
        print("FAIL: clean audit should pass"); ok = False

    leaky = _FakeMV(dec, [{"kind": "daily_bar", "ts": datetime(2024, 1, 9, 16),
                           "id": "AAA"}])
    # Non-strict: logs + returns False (fail-open).
    if assert_no_lookahead(leaky, strict=False) is not False:
        print("FAIL: leaky audit non-strict should return False"); ok = False
    # Strict: raises.
    try:
        assert_no_lookahead(leaky, strict=True)
        print("FAIL: leaky audit strict should raise"); ok = False
    except LookAheadError:
        pass

    # --- TemporalGuard ----------------------------------------------------- #
    with TemporalGuard(dec, strict=False) as g:
        g.check("daily_bar", "2024-01-04", ident="AAA")     # valid
        g.check("daily_bar", "2024-01-09", ident="AAA")     # leak
    if g.ok():
        print("FAIL: guard should have recorded a violation"); ok = False
    if len(g.checked) != 2 or len(g.violations) != 1:
        print("FAIL: guard counts", len(g.checked), len(g.violations)); ok = False

    # Strict guard raises on the first bad datum.
    try:
        with TemporalGuard(dec, strict=True) as g2:
            g2.check("daily_bar", "2024-01-09")
        print("FAIL: strict guard should raise"); ok = False
    except LookAheadError:
        pass

    # Junk tolerance (fail-open, no raise in non-strict paths).
    if validate_feature("not a stamp")[0]:          # type: ignore[arg-type]
        print("FAIL: junk stamp validated true"); ok = False
    if to_dt("garbage") is not None:
        print("FAIL: garbage ts parsed"); ok = False

    print("oracle.temporal self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
