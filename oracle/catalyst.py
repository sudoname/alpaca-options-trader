"""
Oracle 2.1 — Catalyst detector (ANALYTICS ONLY, pure, no I/O).

A *catalyst* is an exogenous event that can drive an outsized move: a news
shock (sudden, one-sided headline flow) or a scheduled earnings window. This
module RECORDS the catalyst as evidence — it is a FEATURE, never a trade
trigger. The ``direction_hint`` it emits is copied into the shadow slate for
later leaderboard analysis; it NEVER opens, sizes, prices, blocks, or alters a
real/paper trade, and it NEVER decides direction (that stays an OUTPUT of the
first-principles rule engine).

Two detectors, both fail-open:
  * News shock (default): a strong, one-sided news_score backed by enough
    headlines. severity ~ |news_score|; direction_hint = call if score>0 else
    put. Requires ``news_count`` >= NEWS_MIN_COUNT to avoid single-headline
    noise.
  * Earnings window: an optional ``earnings_days`` (calendar days until the
    next report). Inside EARNINGS_WINDOW_DAYS it flags a scheduled catalyst.
    Earnings are direction-agnostic, so direction_hint stays None; severity
    grows as the report approaches.

When both fire, the higher-severity catalyst wins (ties -> earnings, the known
scheduled event). When neither fires -> type "none", severity 0.

Design rules (mirror oracle/signals/candlestick_patterns.py):
  * Pure functions. No network, no creds, no file writes, no env reads.
  * Prefer None over a guess; never raise on malformed input.
  * Offline-testable with synthetic inputs.
"""

from dataclasses import dataclass
from typing import Optional

# Catalyst type labels.
CATALYST_NONE = "none"
CATALYST_NEWS = "news_shock"
CATALYST_EARNINGS = "earnings"

# Direction labels (recorded only, never a trigger).
DIR_CALL = "call"
DIR_PUT = "put"

# News shock: |news_score| at/above this, backed by >= NEWS_MIN_COUNT
# headlines, is a one-sided shock rather than routine flow.
NEWS_SHOCK_SCORE = 0.5
NEWS_MIN_COUNT = 3

# Earnings window: a report within this many calendar days is a catalyst.
EARNINGS_WINDOW_DAYS = 5.0


def _to_float(value) -> Optional[float]:
    """Coerce to float; bools and junk -> None (never raises)."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _clamp01(v: float) -> float:
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else round(v, 6)


@dataclass
class CatalystStamp:
    """One catalyst reading. All fields analytics-only."""

    catalyst_type: str            # none / news_shock / earnings
    severity: float               # 0.0-1.0 magnitude
    direction_hint: Optional[str]  # 'call'/'put'/None (recorded, not a trigger)
    source: Optional[str]         # short label of the driving input
    reason: str                   # human-readable rationale

    def to_dict(self) -> dict:
        return {
            "catalyst_type": self.catalyst_type,
            "severity": self.severity,
            "direction_hint": self.direction_hint,
            "source": self.source,
            "reason": self.reason,
        }


def _none_stamp(reason: str = "no catalyst") -> dict:
    return CatalystStamp(
        catalyst_type=CATALYST_NONE,
        severity=0.0,
        direction_hint=None,
        source=None,
        reason=reason,
    ).to_dict()


def _news_shock(ctx: dict) -> Optional[CatalystStamp]:
    """News shock from a strong, well-backed one-sided news_score. Else None."""
    score = _to_float(ctx.get("news_score"))
    if score is None:
        return None
    count = _to_float(ctx.get("news_count"))
    # A shock needs conviction (magnitude) AND corroboration (headline count).
    if abs(score) < NEWS_SHOCK_SCORE:
        return None
    if count is not None and count < NEWS_MIN_COUNT:
        return None
    severity = _clamp01(abs(score))
    direction = DIR_CALL if score > 0 else DIR_PUT
    n_txt = f"{int(count)}" if count is not None else "n/a"
    return CatalystStamp(
        catalyst_type=CATALYST_NEWS,
        severity=severity,
        direction_hint=direction,
        source="news",
        reason=f"news_score={score:+.2f} count={n_txt} -> {direction}",
    )


def _earnings(ctx: dict) -> Optional[CatalystStamp]:
    """Scheduled earnings catalyst from optional ``earnings_days``. Else None."""
    days = _to_float(ctx.get("earnings_days"))
    if days is None or days < 0.0 or days > EARNINGS_WINDOW_DAYS:
        return None
    # Severity grows as the report approaches: days=0 -> 1.0, at the window edge
    # -> a small positive floor (still a catalyst, just distant).
    severity = _clamp01((EARNINGS_WINDOW_DAYS - days + 1.0)
                        / (EARNINGS_WINDOW_DAYS + 1.0))
    return CatalystStamp(
        catalyst_type=CATALYST_EARNINGS,
        severity=severity,
        direction_hint=None,  # earnings is direction-agnostic
        source="earnings",
        reason=f"earnings in {days:.0f}d (<= {EARNINGS_WINDOW_DAYS:.0f}d window)",
    )


def detect_catalyst(ctx: Optional[dict]) -> dict:
    """Return a catalyst stamp dict from a market ``ctx``. Never raises.

    Consumes (all optional): ``news_score`` [-1,1], ``news_count``,
    ``earnings_days``. Returns a ``type "none"`` stamp when nothing fires.
    """
    if not isinstance(ctx, dict):
        return _none_stamp()

    try:
        news = _news_shock(ctx)
    except Exception:
        news = None
    try:
        earn = _earnings(ctx)
    except Exception:
        earn = None

    if news is None and earn is None:
        return _none_stamp()
    if news is None:
        return earn.to_dict()
    if earn is None:
        return news.to_dict()
    # Both fired: stronger severity wins; earnings breaks ties (known event).
    if earn.severity >= news.severity:
        return earn.to_dict()
    return news.to_dict()


# --------------------------------------------------------------------------- #
# Self-test (no network, no creds, no file writes)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True

    # Strong positive news, enough headlines -> call news_shock.
    t = detect_catalyst({"news_score": 0.8, "news_count": 5})
    if t["catalyst_type"] != CATALYST_NEWS:
        print("FAIL: news shock type", t); ok = False
    if t["direction_hint"] != DIR_CALL:
        print("FAIL: news shock -> call", t); ok = False
    if abs(t["severity"] - 0.8) > 1e-9:
        print("FAIL: news severity = |score|", t); ok = False

    # Strong negative news -> put.
    tp = detect_catalyst({"news_score": -0.7, "news_count": 4})
    if tp["catalyst_type"] != CATALYST_NEWS or tp["direction_hint"] != DIR_PUT:
        print("FAIL: negative news -> put", tp); ok = False

    # Strong score but too few headlines -> not a shock.
    tn = detect_catalyst({"news_score": 0.9, "news_count": 1})
    if tn["catalyst_type"] != CATALYST_NONE:
        print("FAIL: single-headline should not shock", tn); ok = False

    # Weak score, plenty of headlines -> routine flow, no shock.
    tw = detect_catalyst({"news_score": 0.2, "news_count": 10})
    if tw["catalyst_type"] != CATALYST_NONE:
        print("FAIL: weak score should not shock", tw); ok = False

    # Missing count is tolerated (score alone can shock).
    tc = detect_catalyst({"news_score": 0.6})
    if tc["catalyst_type"] != CATALYST_NEWS:
        print("FAIL: score alone should shock when count absent", tc); ok = False

    # Earnings tomorrow -> earnings catalyst, no direction hint, high severity.
    te = detect_catalyst({"earnings_days": 0})
    if te["catalyst_type"] != CATALYST_EARNINGS:
        print("FAIL: earnings type", te); ok = False
    if te["direction_hint"] is not None:
        print("FAIL: earnings is direction-agnostic", te); ok = False
    if te["severity"] <= 0.9:
        print("FAIL: earnings day-of severity ~1", te); ok = False

    # Earnings well outside the window -> nothing.
    tf = detect_catalyst({"earnings_days": 20})
    if tf["catalyst_type"] != CATALYST_NONE:
        print("FAIL: far earnings should be none", tf); ok = False

    # Nearer earnings is more severe than distant (monotonic in proximity).
    near = detect_catalyst({"earnings_days": 1})
    far = detect_catalyst({"earnings_days": 4})
    if not (near["severity"] > far["severity"]):
        print("FAIL: earnings severity monotonic in proximity", near, far)
        ok = False

    # Both fire: earnings-day (sev ~1) beats a moderate news shock.
    tb = detect_catalyst({"news_score": 0.6, "news_count": 5, "earnings_days": 0})
    if tb["catalyst_type"] != CATALYST_EARNINGS:
        print("FAIL: stronger earnings should win", tb); ok = False

    # Both fire: a very strong news shock beats a distant earnings window.
    tb2 = detect_catalyst({"news_score": 0.95, "news_count": 8,
                           "earnings_days": 5})
    if tb2["catalyst_type"] != CATALYST_NEWS:
        print("FAIL: stronger news should win", tb2); ok = False

    # Determinism + never-raise on junk.
    if detect_catalyst({"news_score": 0.8, "news_count": 5}) != \
            detect_catalyst({"news_score": 0.8, "news_count": 5}):
        print("FAIL: non-deterministic"); ok = False
    for junk in (None, 42, "x", [], {"weird": object()},
                 {"news_score": "bad"}, {"earnings_days": "soon"}):
        try:
            r = detect_catalyst(junk)  # type: ignore[arg-type]
            if "catalyst_type" not in r or "severity" not in r:
                print("FAIL: junk shape", junk, r); ok = False
        except Exception as exc:  # pragma: no cover
            print("FAIL: raised on junk", junk, exc); ok = False

    print("catalyst self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    sys.exit(_self_test())
