#!/usr/bin/env python3
"""
run_alpaca_intraday.py — Alpaca-native intraday auto-entry scheduler.

Autonomously enters and manages SPY/QQQ options on the ALPACA PAPER account by
orchestrating the existing SmartOptionsTrader pipeline. It adds NO new trading
logic: direction, option selection, budget/sentiment/risk/PDT gating, fill
readback and shadow/RL recording all live in smart_trader.py and are reused
verbatim.

SAFETY
------
  * Paper-only (SmartOptionsTrader talks to the paper endpoint when ALPACA_PAPER
    is true, which is the default).
  * DRY-RUN BY DEFAULT: it runs the full selection pipeline and logs the trade it
    WOULD place, but never touches an order endpoint. Set SCHEDULER_ARMED=1 (or
    pass --armed) to actually place paper orders.
  * The broker clock (/v2/clock) is the source of truth for market hours,
    holidays and the close time, so there is no hand-rolled calendar to get wrong.

Usage
-----
  python run_alpaca_intraday.py            # continuous loop (dry-run)
  python run_alpaca_intraday.py --once     # single scan then exit (dry-run)
  SCHEDULER_ARMED=1 python run_alpaca_intraday.py        # arm (place paper orders)
  python run_alpaca_intraday.py --armed --once           # arm, single scan
  python run_alpaca_intraday.py --selftest # no creds/network; exits non-zero on fail

Env overrides (read from .env via the same manual parse smart_trader uses):
  SCHEDULER_SYMBOLS=SPY,QQQ   SCHEDULER_QTY=1
  SCAN_INTERVAL_MIN=15        ENTRY_START_ET=09:45
  ENTRY_CUTOFF_MIN_BEFORE_CLOSE=60   EOD_CLOSE_MIN_BEFORE_CLOSE=15
  MAX_NEW_TRADES_PER_DAY=4    SCHEDULER_ARMED=0
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, date, time as dtime

ACTIVE_TRADES_FILE = "active_trades.json"
STATUS_FILE = "scheduler_status.json"
SCHEDULER_SOURCE = "alpaca_scheduler"


# --------------------------------------------------------------------------- #
# Config (manual .env parse — mirrors SmartOptionsTrader.load_credentials so a
# value set in .env is honoured even though python-dotenv is not on the path).
# --------------------------------------------------------------------------- #
def _load_env_file(path=".env"):
    env = {}
    if os.path.exists(path):
        with open(path, "r") as f:
            for line in f:
                if "=" in line and not line.strip().startswith("#"):
                    k, v = line.strip().split("=", 1)
                    env[k] = v
    return env


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


class Config:
    """Resolved scheduler configuration (env first, then .env file, then default)."""

    def __init__(self, env=None, armed_override=None):
        env = env if env is not None else {}

        def get(key, default):
            # os.environ wins so a shell `KEY=... python ...` overrides .env.
            return os.environ.get(key, env.get(key, default))

        self.symbols = [s.strip().upper() for s in get("SCHEDULER_SYMBOLS", "SPY,QQQ").split(",") if s.strip()]
        self.qty = int(get("SCHEDULER_QTY", "1"))
        self.scan_interval_min = float(get("SCAN_INTERVAL_MIN", "15"))
        self.entry_start_et = _parse_hhmm(get("ENTRY_START_ET", "09:45"))
        self.entry_cutoff_min = float(get("ENTRY_CUTOFF_MIN_BEFORE_CLOSE", "60"))
        self.eod_close_min = float(get("EOD_CLOSE_MIN_BEFORE_CLOSE", "15"))
        self.max_new_trades_per_day = int(get("MAX_NEW_TRADES_PER_DAY", "4"))
        if armed_override is None:
            self.armed = _truthy(get("SCHEDULER_ARMED", "0"))
        else:
            self.armed = bool(armed_override)

    @property
    def mode(self):
        return "armed" if self.armed else "dry-run"


# --------------------------------------------------------------------------- #
# Pure helpers (deterministic, no creds/network — covered by --selftest).
# All time/window logic lives here so it can be unit-tested in isolation.
# --------------------------------------------------------------------------- #
def _parse_hhmm(s):
    """'09:45' -> datetime.time(9, 45)."""
    parts = str(s).strip().split(":")
    return dtime(int(parts[0]), int(parts[1]))


def _parse_iso(ts):
    """Parse an Alpaca ISO timestamp into a tz-aware datetime.

    Alpaca returns offsets like '...-04:00'; tolerate a trailing 'Z' too.
    """
    if ts is None:
        return None
    ts = ts.strip()
    if ts.endswith("Z"):
        ts = ts[:-1] + "+00:00"
    return datetime.fromisoformat(ts)


def parse_clock(clock):
    """Normalise an Alpaca /v2/clock payload -> {is_open, now, next_close, next_open}."""
    return {
        "is_open": bool(clock.get("is_open", False)),
        "now": _parse_iso(clock.get("timestamp")),
        "next_close": _parse_iso(clock.get("next_close")),
        "next_open": _parse_iso(clock.get("next_open")),
    }


def minutes_to_close(now, next_close):
    """Minutes from `now` until `next_close` (negative if already past)."""
    return (next_close - now).total_seconds() / 60.0


def _local_time_of_day(now, ref):
    """Time-of-day of `now` in the exchange's timezone (taken from `ref`'s offset)."""
    return now.astimezone(ref.tzinfo).time()


def in_entry_window(now, next_close, *, start_et, cutoff_min):
    """True when new entries are allowed: at/after the ET start time and more than
    `cutoff_min` minutes before the close."""
    if now is None or next_close is None:
        return False
    if minutes_to_close(now, next_close) <= cutoff_min:
        return False
    return _local_time_of_day(now, next_close) >= start_et


def in_eod_window(now, next_close, *, eod_min):
    """True in the final `eod_min` minutes before the close (and not past it)."""
    if now is None or next_close is None:
        return False
    m = minutes_to_close(now, next_close)
    return 0 < m <= eod_min


def session_key(next_close):
    """Date used to detect a new session for daily counter resets."""
    return next_close.date() if next_close is not None else None


def underlying_of(occ_symbol):
    """OCC option symbol -> underlying root, e.g. 'SPY240705C00540000' -> 'SPY'."""
    m = re.match(r"^([A-Za-z]+)", occ_symbol or "")
    return m.group(1).upper() if m else ""


# --------------------------------------------------------------------------- #
# Logging
# --------------------------------------------------------------------------- #
def log(msg):
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] {msg}", flush=True)


# --------------------------------------------------------------------------- #
# Active-trades file helpers (stamp scheduler-owned rows for EOD targeting)
# --------------------------------------------------------------------------- #
def _read_active_trades():
    if not os.path.exists(ACTIVE_TRADES_FILE):
        return []
    try:
        with open(ACTIVE_TRADES_FILE, "r") as f:
            return json.load(f)
    except Exception:
        return []


def _stamp_scheduler_trade(option_symbol):
    """Mark the most-recent active trade for `option_symbol` as scheduler-owned so
    the EOD force-close only ever touches positions this scheduler opened."""
    trades = _read_active_trades()
    stamped = False
    for trade in reversed(trades):
        if trade.get("symbol") == option_symbol:
            trade["source"] = SCHEDULER_SOURCE
            stamped = True
            break
    if stamped:
        try:
            with open(ACTIVE_TRADES_FILE, "w") as f:
                json.dump(trades, f, indent=2)
        except Exception as e:
            log(f"[WARN] could not stamp active trade {option_symbol}: {e}")
    return stamped


# --------------------------------------------------------------------------- #
# Scheduler
# --------------------------------------------------------------------------- #
class IntradayScheduler:
    def __init__(self, config, trader, pdt):
        self.cfg = config
        self.trader = trader
        self.pdt = pdt
        self.current_session = None
        self.trades_today = 0
        self.entered_today = set()

    # -- per-session counters ------------------------------------------------ #
    def _roll_session(self, sk):
        if sk is not None and sk != self.current_session:
            self.current_session = sk
            self.trades_today = 0
            self.entered_today = set()
            log(f"[SESSION] new session {sk}: counters reset")

    # -- exits --------------------------------------------------------------- #
    def force_close_scheduler_positions(self):
        """Force-close (market sell) only positions this scheduler opened."""
        trades = _read_active_trades()
        if not trades:
            return
        positions = self.trader.get_positions() or []
        by_symbol = {p.get("symbol"): p for p in positions}
        for trade in trades:
            if trade.get("source") != SCHEDULER_SOURCE:
                continue
            pos = by_symbol.get(trade.get("symbol"))
            if pos:
                log(f"[EOD] closing scheduler position {trade.get('symbol')}")
                try:
                    self.trader.close_position(trade, pos, "EOD_CLOSE")
                except Exception as e:
                    log(f"[WARN] EOD close failed for {trade.get('symbol')}: {e}")

    def _has_live_position(self, sym):
        positions = self.trader.get_positions() or []
        return any(underlying_of(p.get("symbol")) == sym for p in positions)

    # -- entries ------------------------------------------------------------- #
    def _try_enter(self, sym):
        if sym in self.entered_today:
            return
        if self._has_live_position(sym):
            log(f"[SKIP] {sym}: already holding a position")
            self.entered_today.add(sym)
            return

        self.trader.ticker = sym  # keep direction/underlying defaults correct

        price = self.trader.get_current_price(sym)
        if not price:
            log(f"[SKIP] {sym}: no current price")
            return

        contracts = self.trader.get_option_contracts(sym)
        if not contracts:
            log(f"[SKIP] {sym}: no option contracts")
            return

        direction = self.trader.determine_option_strategy(sym)  # 'call' / 'put'
        opt = self.trader.select_best_option(contracts, price, strategy=direction)
        if not opt:
            log(f"[SKIP] {sym}: no suitable option ({direction.upper()})")
            return

        desc = (f"{sym} {direction.upper()} {opt.get('symbol')} x{self.cfg.qty} "
                f"(ask={opt.get('ask')}, score={opt.get('score')})")

        if not self.cfg.armed:
            log(f"[DRY-RUN] WOULD ENTER {desc}")
            self.entered_today.add(sym)
            return

        log(f"[ARMED] entering {desc}")
        order = self.trader.place_order_with_stops(opt, quantity=self.cfg.qty)
        if order:
            self.trades_today += 1
            self.entered_today.add(sym)
            _stamp_scheduler_trade(opt.get("symbol"))
            log(f"[ARMED] order placed for {opt.get('symbol')} "
                f"(trades_today={self.trades_today})")
        else:
            log(f"[ARMED] order NOT placed for {opt.get('symbol')} "
                f"(blocked by risk/budget/duplicate or broker rejection)")

    # -- heartbeat ----------------------------------------------------------- #
    def _write_status(self, parsed):
        pdt_status = {}
        try:
            s = self.pdt.get_status_message()
            pdt_status = {"count": s.get("count"), "remaining": s.get("remaining"),
                          "can_trade": s.get("can_trade")}
        except Exception:
            pass
        status = {
            "last_heartbeat": datetime.now().isoformat(),
            "mode": self.cfg.mode,
            "is_open": parsed.get("is_open"),
            "next_close": parsed["next_close"].isoformat() if parsed.get("next_close") else None,
            "symbols": self.cfg.symbols,
            "scan_interval_min": self.cfg.scan_interval_min,
            "trades_today": self.trades_today,
            "max_new_trades_per_day": self.cfg.max_new_trades_per_day,
            "entered_today": sorted(self.entered_today),
            "pdt": pdt_status,
        }
        try:
            with open(STATUS_FILE, "w") as f:
                json.dump(status, f, indent=2)
        except Exception as e:
            log(f"[WARN] could not write {STATUS_FILE}: {e}")

    # -- one scan ------------------------------------------------------------ #
    def scan_once(self):
        """Run a single scan. Returns the parsed clock (for sleep decisions)."""
        clock = self.trader.get_market_status()
        parsed = parse_clock(clock)

        if not parsed["is_open"]:
            log("[CLOSED] market is closed")
            self._write_status(parsed)
            return parsed

        self._roll_session(session_key(parsed["next_close"]))

        # 1) manage exits first (threshold/trailing on all tracked trades)
        try:
            self.trader.monitor_positions()
        except Exception as e:
            log(f"[WARN] monitor_positions failed: {e}")

        # 2) EOD force-close window — close ours, do not open new
        if in_eod_window(parsed["now"], parsed["next_close"], eod_min=self.cfg.eod_close_min):
            log(f"[EOD] within {self.cfg.eod_close_min:g} min of close")
            self.force_close_scheduler_positions()
            self._write_status(parsed)
            return parsed

        # 3) entries
        if in_entry_window(parsed["now"], parsed["next_close"],
                           start_et=self.cfg.entry_start_et,
                           cutoff_min=self.cfg.entry_cutoff_min):
            if self.trades_today >= self.cfg.max_new_trades_per_day:
                log(f"[CAP] reached max_new_trades_per_day={self.cfg.max_new_trades_per_day}")
            elif not self._pdt_ok():
                log("[PDT] no day-trade headroom; skipping entries")
            else:
                for sym in self.cfg.symbols:
                    try:
                        self._try_enter(sym)
                    except Exception as e:
                        log(f"[WARN] entry attempt failed for {sym}: {e}")
        else:
            mtc = minutes_to_close(parsed["now"], parsed["next_close"])
            log(f"[WAIT] outside entry window (min_to_close={mtc:.0f})")

        self._write_status(parsed)
        return parsed

    def _pdt_ok(self):
        # EOD close turns entries into same-day round trips, so respect PDT.
        try:
            return self.pdt.can_day_trade()
        except Exception:
            return True  # fail-open here; the risk engine enforces PDT on the order

    # -- main loop ----------------------------------------------------------- #
    def run(self, once=False):
        log(f"[START] Alpaca intraday scheduler — mode={self.cfg.mode}, "
            f"symbols={self.cfg.symbols}, qty={self.cfg.qty}, "
            f"scan={self.cfg.scan_interval_min:g}m")
        if not self.cfg.armed:
            log("[DRY-RUN] no orders will be placed (set SCHEDULER_ARMED=1 to arm)")

        try:
            self.trader.reconcile_open_trades()
        except Exception as e:
            log(f"[WARN] reconcile_open_trades failed: {e}")

        while True:
            parsed = self.scan_once()
            if once:
                return
            # Sleep less aggressively when the market is closed.
            if parsed.get("is_open"):
                time.sleep(self.cfg.scan_interval_min * 60)
            else:
                time.sleep(min(self.cfg.scan_interval_min * 60, 300))


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _selftest():
    failures = []

    def check(name, cond):
        if cond:
            print(f"  ok  - {name}")
        else:
            print(f"  FAIL- {name}")
            failures.append(name)

    # Build a synthetic ET session: open 09:30, close 16:00, offset -04:00.
    def clk(hhmm, *, is_open=True, close="16:00"):
        d = "2026-04-01"
        return {
            "is_open": is_open,
            "timestamp": f"{d}T{hhmm}:00-04:00",
            "next_close": f"{d}T{close}:00-04:00",
            "next_open": f"{d}T09:30:00-04:00",
        }

    start = _parse_hhmm("09:45")

    p_open = parse_clock(clk("12:00"))
    check("parse_clock tz-aware now", p_open["now"].utcoffset() is not None)
    check("minutes_to_close ~240 at noon",
          abs(minutes_to_close(p_open["now"], p_open["next_close"]) - 240) < 0.5)

    # entry window
    check("entry: open midday is in window",
          in_entry_window(parse_clock(clk("12:00"))["now"],
                          parse_clock(clk("12:00"))["next_close"],
                          start_et=start, cutoff_min=60) is True)
    check("entry: 09:30 (before start) NOT in window",
          in_entry_window(parse_clock(clk("09:30"))["now"],
                          parse_clock(clk("09:30"))["next_close"],
                          start_et=start, cutoff_min=60) is False)
    check("entry: 15:30 (inside 60m cutoff) NOT in window",
          in_entry_window(parse_clock(clk("15:30"))["now"],
                          parse_clock(clk("15:30"))["next_close"],
                          start_et=start, cutoff_min=60) is False)
    check("entry: 09:45 exactly is in window",
          in_entry_window(parse_clock(clk("09:45"))["now"],
                          parse_clock(clk("09:45"))["next_close"],
                          start_et=start, cutoff_min=60) is True)

    # eod window
    check("eod: 15:50 within 15m of close",
          in_eod_window(parse_clock(clk("15:50"))["now"],
                        parse_clock(clk("15:50"))["next_close"], eod_min=15) is True)
    check("eod: 12:00 not in eod window",
          in_eod_window(parse_clock(clk("12:00"))["now"],
                        parse_clock(clk("12:00"))["next_close"], eod_min=15) is False)
    check("eod: 16:00 (at close) not in eod window",
          in_eod_window(parse_clock(clk("16:00"))["now"],
                        parse_clock(clk("16:00"))["next_close"], eod_min=15) is False)

    # entry and eod are mutually exclusive late in the day
    late = parse_clock(clk("15:50"))
    check("late: not entry while in eod",
          in_entry_window(late["now"], late["next_close"], start_et=start, cutoff_min=60) is False)

    # session_key / underlying_of
    check("session_key is the close date",
          session_key(parse_clock(clk("12:00"))["next_close"]) == date(2026, 4, 1))
    check("underlying_of SPY option", underlying_of("SPY240705C00540000") == "SPY")
    check("underlying_of QQQ option", underlying_of("QQQ260101P00400000") == "QQQ")
    check("underlying_of empty", underlying_of("") == "")

    # 'Z' suffix tolerated
    check("parse_iso tolerates Z", _parse_iso("2026-04-01T12:00:00Z").utcoffset() is not None)

    # day-rollover reset on the scheduler object (no creds needed)
    cfg = Config(env={}, armed_override=False)
    sched = IntradayScheduler(cfg, trader=None, pdt=None)
    sched.trades_today = 3
    sched.entered_today = {"SPY"}
    sched._roll_session(date(2026, 4, 1))
    check("rollover resets trades_today", sched.trades_today == 0)
    check("rollover resets entered_today", sched.entered_today == set())
    sched.trades_today = 2
    sched._roll_session(date(2026, 4, 1))  # same session -> no reset
    check("same session keeps counters", sched.trades_today == 2)

    # config dry-run default
    check("config defaults to dry-run", Config(env={}).armed is False)
    check("config arms via env", Config(env={"SCHEDULER_ARMED": "1"}).armed is True)
    check("config symbols default SPY,QQQ", Config(env={}).symbols == ["SPY", "QQQ"])

    if failures:
        print(f"\nSELFTEST FAILED ({len(failures)} failed)")
        return 1
    print("\nSELFTEST PASSED")
    return 0


# --------------------------------------------------------------------------- #
# Entry point
# --------------------------------------------------------------------------- #
def main(argv=None):
    parser = argparse.ArgumentParser(description="Alpaca-native intraday auto-entry scheduler")
    parser.add_argument("--once", action="store_true", help="run a single scan then exit")
    parser.add_argument("--armed", action="store_true", help="place paper orders (overrides env)")
    parser.add_argument("--dry-run", dest="dry_run", action="store_true",
                        help="force dry-run (overrides env)")
    parser.add_argument("--selftest", action="store_true",
                        help="run deterministic self-tests (no creds/network)")
    args = parser.parse_args(argv)

    if args.selftest:
        return _selftest()

    armed_override = None
    if args.armed:
        armed_override = True
    if args.dry_run:
        armed_override = False

    cfg = Config(env=_load_env_file(), armed_override=armed_override)

    from smart_trader import SmartOptionsTrader
    from pdt_tracker import PDTTracker

    trader = SmartOptionsTrader(ticker=cfg.symbols[0] if cfg.symbols else None, quantity=cfg.qty)
    pdt = PDTTracker()

    scheduler = IntradayScheduler(cfg, trader, pdt)
    try:
        scheduler.run(once=args.once)
    except KeyboardInterrupt:
        log("[STOP] interrupted; exiting")
    return 0


if __name__ == "__main__":
    sys.exit(main())
