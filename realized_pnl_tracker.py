"""
Phase 4 — realized daily P/L tracker for the kill-switch.

Why this exists
---------------
The kill-switch previously read ``account.equity - account.last_equity`` as
"today's P/L". That figure includes the *unrealized* mark of every open option,
so a deep but temporary intraday drawdown on open contracts could trip the
switch even though nothing was actually lost. Per Phase 4 requirement 4 the
kill-switch must consider **realized** P/L only and **reset daily**.

This tracker (modeled on ``pdt_tracker.py``) accumulates realized dollar P/L from
*closed* trades into ``realized_pnl_log.json`` and exposes today's realized total.
"Reset daily" is implicit: ``get_today_realized`` only sums entries whose date is
today (local date), so a new calendar day starts from $0 with no cron needed.

It is intentionally side-effect-only (it never blocks a trade) and never raises;
the consumer (``smart_trader._risk_check``) decides whether to feed this number
to the risk engine, gated behind ``USE_REALIZED_PNL_KILLSWITCH``.
"""

import json
import os
from datetime import datetime
from typing import List, Dict, Optional


class RealizedPnLTracker:
    def __init__(self, log_file: str = "realized_pnl_log.json"):
        self.log_file = log_file

    # -- storage ---------------------------------------------------------- #
    def _load(self) -> List[Dict]:
        try:
            if os.path.exists(self.log_file):
                with open(self.log_file, "r") as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except Exception:
            pass
        return []

    def _save(self, rows: List[Dict]) -> None:
        try:
            with open(self.log_file, "w") as f:
                json.dump(rows, f, indent=2)
        except Exception:
            pass

    # -- API -------------------------------------------------------------- #
    @staticmethod
    def _today() -> str:
        return datetime.now().date().isoformat()

    def add_realized(self, amount: float, symbol: Optional[str] = None,
                     when: Optional[datetime] = None) -> None:
        """Record a realized dollar P/L for a closed trade. Never raises."""
        try:
            amt = float(amount)
        except (TypeError, ValueError):
            return
        ts = (when or datetime.now())
        row = {
            "date": ts.date().isoformat(),
            "timestamp": ts.isoformat(),
            "amount": amt,
            "symbol": symbol or "",
        }
        rows = self._load()
        rows.append(row)
        self._save(rows)

    def get_today_realized(self) -> float:
        """Sum of realized dollar P/L recorded for *today* (local date).

        Returns 0.0 when there are no entries today (a fresh day starts flat),
        so the kill-switch never trips purely on stale or unrealized figures.
        """
        today = self._today()
        total = 0.0
        for r in self._load():
            if r.get("date") == today:
                try:
                    total += float(r.get("amount", 0.0))
                except (TypeError, ValueError):
                    continue
        return total

    def reset_today(self) -> None:
        """Drop today's entries (operator tool; not used on the trade path)."""
        today = self._today()
        self._save([r for r in self._load() if r.get("date") != today])


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network; uses a temp file)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    import tempfile
    from datetime import timedelta

    ok = True
    path = os.path.join(tempfile.mkdtemp(), "realized_pnl_log.json")
    t = RealizedPnLTracker(path)

    if t.get_today_realized() != 0.0:
        print("FAIL: empty tracker should be 0.0"); ok = False

    t.add_realized(-120.0, "SPY")
    t.add_realized(30.0, "QQQ")
    if abs(t.get_today_realized() - (-90.0)) > 1e-9:
        print("FAIL: today's realized should net to -90", t.get_today_realized()); ok = False

    # An entry dated yesterday must NOT count toward today (daily reset).
    t.add_realized(-999.0, "OLD", when=datetime.now() - timedelta(days=1))
    if abs(t.get_today_realized() - (-90.0)) > 1e-9:
        print("FAIL: yesterday's loss must not count today", t.get_today_realized()); ok = False

    # Garbage amount is ignored, not fatal.
    t.add_realized("oops", "BAD")
    if abs(t.get_today_realized() - (-90.0)) > 1e-9:
        print("FAIL: garbage amount should be ignored", t.get_today_realized()); ok = False

    print("realized_pnl_tracker self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys

    if "--selftest" in sys.argv or len(sys.argv) == 1:
        sys.exit(_self_test())
