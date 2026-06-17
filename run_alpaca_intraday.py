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

The universe is the base SCHEDULER_SYMBOLS (SPY,QQQ) plus, when
SCHEDULER_INCLUDE_SCREENER=1 (default), the screener's picks from
supported_tickers.json (refresh them with `./run.sh screen`). The merged list is
deduped and capped at SCHEDULER_MAX_SYMBOLS; over-budget/illiquid names self-skip
in select_best_option, and MAX_NEW_TRADES_PER_DAY still bounds actual entries.

Env overrides (read from .env via the same manual parse smart_trader uses):
  SCHEDULER_SYMBOLS=SPY,QQQ   SCHEDULER_QTY=1
  SCHEDULER_INCLUDE_SCREENER=1   SCHEDULER_MAX_SYMBOLS=12
  SCAN_INTERVAL_MIN=15        ENTRY_START_ET=09:45
  ENTRY_CUTOFF_MIN_BEFORE_CLOSE=60   EOD_CLOSE_MIN_BEFORE_CLOSE=15
  EOD_CLOSE_ENABLED=1         MAX_HOLD_DAYS=0
  MAX_NEW_TRADES_PER_DAY=4    SCHEDULER_ARMED=0

EOD_CLOSE_ENABLED=0 switches to hold-overnight mode: the final-minutes window
still blocks new entries but leaves positions running to their TP/SL across
sessions; pair it with MAX_HOLD_DAYS=N so anything unresolved after N calendar
days is force-closed (TIME_STOP) instead of rotting.
"""

import os
import re
import sys
import json
import time
import argparse
from datetime import datetime, date, time as dtime

from exit_manager import enforce_exit

ACTIVE_TRADES_FILE = "active_trades.json"
STATUS_FILE = "scheduler_status.json"
SCHEDULER_SOURCE = "alpaca_scheduler"


# --------------------------------------------------------------------------- #
# Config (manual .env parse — mirrors SmartOptionsTrader.load_credentials so a
# value set in .env is honoured even though python-dotenv is not on the path).
# --------------------------------------------------------------------------- #
def _load_env_file(path=".env"):
    # Phase 4.5: one parser for the whole project (shared config_loader).
    from config_loader import parse_env_file
    return parse_env_file(path)


def _truthy(val):
    return str(val).strip().lower() in ("1", "true", "yes", "on")


def _load_screener_symbols(path="supported_tickers.json"):
    """Symbols picked by stock_screener.py (`./run.sh screen --write-tickers`).

    Returns [] if the file is missing/unreadable so the scheduler simply
    degrades to its base symbols. Local file read only (no network); it is
    refreshed once per session so fresh screener picks are honoured without a
    restart.
    """
    try:
        with open(path, "r") as f:
            data = json.load(f)
        return [str(s).strip().upper() for s in data.get("tickers", []) if str(s).strip()]
    except Exception:
        return []


class Config:
    """Resolved scheduler configuration (env first, then .env file, then default)."""

    def __init__(self, env=None, armed_override=None):
        env = env if env is not None else {}

        # Phase 4.5: resolve through the shared loader. os.environ wins, then the
        # passed `env` (.env or a selftest dict), then the default — identical to
        # the previous `os.environ.get(key, env.get(key, default))` semantics.
        from config_loader import ConfigLoader
        get = ConfigLoader(file_values=env).get

        self.symbols = [s.strip().upper() for s in get("SCHEDULER_SYMBOLS", "SPY,QQQ").split(",") if s.strip()]
        # Optionally fold the screener's picks (supported_tickers.json, written by
        # `./run.sh screen`) into the trading universe alongside the base symbols.
        # The merge happens at runtime in IntradayScheduler._refresh_universe so
        # this Config stays deterministic for --selftest (no file read here).
        self.include_screener = _truthy(get("SCHEDULER_INCLUDE_SCREENER", "1"))
        self.max_symbols = int(get("SCHEDULER_MAX_SYMBOLS", "12"))
        self.qty = int(get("SCHEDULER_QTY", "1"))
        self.scan_interval_min = float(get("SCAN_INTERVAL_MIN", "15"))
        self.entry_start_et = _parse_hhmm(get("ENTRY_START_ET", "09:45"))
        self.entry_cutoff_min = float(get("ENTRY_CUTOFF_MIN_BEFORE_CLOSE", "60"))
        self.eod_close_min = float(get("EOD_CLOSE_MIN_BEFORE_CLOSE", "15"))
        # Hold-overnight mode: when EOD_CLOSE_ENABLED=0 the end-of-day window
        # stops opening new trades but leaves positions running to their
        # TP/SL (monitored every scan, across sessions). MAX_HOLD_DAYS is the
        # companion time stop: scheduler positions older than this many
        # CALENDAR days are force-closed on the next scan (0 = no time stop).
        self.eod_close_enabled = _truthy(get("EOD_CLOSE_ENABLED", "1"))
        self.max_hold_days = float(get("MAX_HOLD_DAYS", "0"))
        self.max_new_trades_per_day = int(get("MAX_NEW_TRADES_PER_DAY", "4"))
        # Per-underlying position cap. The scheduler used to allow exactly one
        # open position per underlying; this lets a name accrue up to
        # MAX_POSITIONS_PER_UNDERLYING positions (the same cap the risk engine
        # enforces). Default 1 preserves the old one-per-underlying behavior
        # when the env is unset (keeps --selftest deterministic).
        self.max_per_underlying = int(get("MAX_POSITIONS_PER_UNDERLYING", "1"))
        # SKIP-counterfactual horizon: how many wall-clock minutes after a
        # declined setup is logged before its forward-underlying-return outcome
        # is resolved. ~390 = one trading day, so skips resolve next session.
        self.skip_cf_horizon_min = int(get("SKIP_CF_HORIZON_MIN", "390"))
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
    # Alpaca can return nanosecond precision (9 fractional digits); fromisoformat
    # only accepts 3 or 6. Clamp the fractional-seconds field to 6 digits.
    ts = re.sub(r"(\.\d{6})\d+", r"\1", ts)
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


def held_past_max_days(entry_time, now, max_days):
    """True when a tracked trade's entry is more than `max_days` CALENDAR days
    old — the hold-overnight time stop. Date-only comparison so timezone
    offsets can't shave a day. Fail-open False on a missing/garbled stamp or a
    disabled (<= 0) limit, so a bad row is never force-closed by accident."""
    if not max_days or max_days <= 0 or now is None:
        return False
    try:
        entry = datetime.fromisoformat(str(entry_time)[:19])
        return (now.date() - entry.date()).days > max_days
    except (TypeError, ValueError):
        return False


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


def _write_active_trades(trades):
    """Persist the active-trades list (the survivors left after closes). Pairs
    with `_read_active_trades`; both callers (EOD close + the fill-driven
    reconcile) prune closed rows here so a sold position can't linger as
    'tracked open'. Fail-open: a write error is logged, never raised."""
    try:
        with open(ACTIVE_TRADES_FILE, "w") as f:
            json.dump(trades, f, indent=2, default=str)
    except Exception as e:
        log(f"[WARN] could not write {ACTIVE_TRADES_FILE}: {e}")


def realized_from_fills(trade, fills):
    """PURE: compute (exit_price, pnl_percent) for `trade` from its SELL `fills`.

    `fills` is a list of activity dicts ({'qty','price','transaction_time'}) for
    the trade's symbol. Uses a quantity-weighted average over the most-recent
    sells that cover the open quantity, so a position closed in several fills
    (e.g. 1+1) books at its true average exit. Returns (None, None) when there
    is no usable entry price / quantity / fill — the caller then leaves the row
    tracked rather than booking a bogus outcome. No network, unit-testable."""
    try:
        entry = float(trade.get("entry_price") or 0)
        want = float(trade.get("quantity") or 0)
    except (TypeError, ValueError):
        return (None, None)
    if entry <= 0 or want <= 0 or not fills:
        return (None, None)
    ordered = sorted(fills, key=lambda a: str(a.get("transaction_time") or ""),
                     reverse=True)
    filled_qty = 0.0
    proceeds = 0.0
    for a in ordered:
        try:
            q = float(a.get("qty") or 0)
            p = float(a.get("price") or 0)
        except (TypeError, ValueError):
            continue
        if q <= 0 or p <= 0:
            continue
        take = min(q, want - filled_qty)
        if take <= 0:
            break
        filled_qty += take
        proceeds += take * p
        if filled_qty >= want:
            break
    if filled_qty <= 0:
        return (None, None)
    exit_price = proceeds / filled_qty
    pnl_percent = (exit_price - entry) / entry * 100.0
    return (exit_price, pnl_percent)


def _fetch_sell_fills_by_symbol(trader):
    """Pull recent FILL activities from the broker and group the SELL fills by
    symbol. Fail-open: returns {} on any HTTP / parse error so the reconcile
    simply does nothing rather than raising on the live path."""
    import requests
    try:
        r = requests.get(
            f"{trader.base_url}/v2/account/activities",
            headers=trader.headers,
            params={"activity_types": "FILL", "direction": "desc",
                    "page_size": 100},
            timeout=30,
        )
        if r.status_code != 200:
            return {}
        acts = r.json()
    except Exception:
        return {}
    out = {}
    for a in acts:
        if a.get("side") in ("sell", "sell_to_close", "sell_to_open"):
            out.setdefault(a.get("symbol"), []).append(a)
    return out


def _already_booked(trader, trade):
    """True if `trade` (matched on symbol + entry_time) is already in the
    trader's trading_history — guards the reconcile against double-counting a
    close that the EOD/monitor path already recorded."""
    sym = trade.get("symbol")
    et = str(trade.get("entry_time") or "")
    try:
        for r in trader.trading_history.get("trades", []):
            if r.get("symbol") == sym and str(r.get("entry_time") or "") == et:
                return True
    except Exception:
        pass
    return False


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
        # Active trading universe: base symbols now; screener picks are folded in
        # at runtime (refreshed each session). Held separately from cfg so the
        # pure-helper --selftest stays deterministic.
        self.base_symbols = list(config.symbols)
        self.symbols = list(config.symbols)

    # -- trading universe ---------------------------------------------------- #
    def _refresh_universe(self):
        """Rebuild the active universe = base symbols + screener picks, deduped,
        order-preserving, capped at cfg.max_symbols. Over-budget or illiquid
        names self-skip later in select_best_option, so a wide list is safe."""
        universe = []
        seen = set()
        sources = list(self.base_symbols)
        if self.cfg.include_screener:
            sources += _load_screener_symbols()
        for s in sources:
            if s and s not in seen:
                seen.add(s)
                universe.append(s)
        if self.cfg.max_symbols and self.cfg.max_symbols > 0:
            universe = universe[: self.cfg.max_symbols]
        if universe != self.symbols:
            self.symbols = universe
            log(f"[UNIVERSE] {len(self.symbols)} symbols: {', '.join(self.symbols)}")

    # -- per-session counters ------------------------------------------------ #
    def _roll_session(self, sk):
        if sk is not None and sk != self.current_session:
            self.current_session = sk
            self.trades_today = 0
            self.entered_today = set()
            if self.trader is not None:  # skip the file read during --selftest
                self._refresh_universe()
            log(f"[SESSION] new session {sk}: counters reset")

    # -- exits --------------------------------------------------------------- #
    def force_close_scheduler_positions(self, reason="EOD_CLOSE", only=None):
        """Force-close (market sell) only positions this scheduler opened.

        Routes each close through the shared `enforce_exit` path so the outcome
        is recorded EXACTLY like every other exit (trading_history + realized
        P/L + RL/shadow), then prunes the closed rows from active_trades.json.
        Previously this called `close_position` directly, which fired the sell
        but never recorded the outcome or pruned the row — leaving sold
        positions stuck as 'tracked open' and missing from realized P/L.

        `only` optionally narrows the sweep to trades matching a predicate
        (e.g. the MAX_HOLD_DAYS time stop); `reason` is recorded as the exit
        outcome ('EOD_CLOSE', 'TIME_STOP')."""
        trades = _read_active_trades()
        if not trades:
            return
        positions = self.trader.get_positions() or []
        by_symbol = {p.get("symbol"): p for p in positions}
        closed_syms = set()
        for trade in trades:
            if trade.get("source") != SCHEDULER_SOURCE:
                continue
            if only is not None and not only(trade):
                continue
            pos = by_symbol.get(trade.get("symbol"))
            if pos:
                log(f"[{reason}] closing scheduler position {trade.get('symbol')}")
                try:
                    current_price = float(pos.get("current_price") or 0)
                    entry_price = float(trade.get("entry_price") or 0)
                    pnl_percent = (((current_price - entry_price) / entry_price
                                    * 100.0) if entry_price else 0.0)
                    enforce_exit(self.trader, trade, pos, reason,
                                 pnl_percent, "scheduler", current_price)
                    closed_syms.add(trade.get("symbol"))
                except Exception as e:
                    log(f"[WARN] {reason} failed for {trade.get('symbol')}: {e}")
        if closed_syms:
            survivors = [t for t in trades
                         if t.get("symbol") not in closed_syms]
            _write_active_trades(survivors)

    def reconcile_closed_from_fills(self):
        """Fill-driven safety net: book any scheduler-owned tracked trade that is
        no longer held at the broker, using its ACTUAL sell fill(s), then prune
        it from active_trades.json.

        This is the catch-all behind the EOD/monitor exit paths: it books closes
        that happened through ANY route we don't own (a force-close that errored
        mid-batch, a manual liquidation, expiry/auto-exercise), so realized P/L
        and the kill-switch can't silently miss a close again. Idempotent via
        `_already_booked`; fail-open — never raises on the live path."""
        try:
            trades = _read_active_trades()
            if not trades:
                return
            positions = self.trader.get_positions() or []
            held = {p.get("symbol") for p in positions}
            orphans = [t for t in trades
                       if t.get("source") == SCHEDULER_SOURCE
                       and t.get("symbol") not in held]
            if not orphans:
                return
            fills_by_symbol = _fetch_sell_fills_by_symbol(self.trader)
            closed_syms = set()
            for t in orphans:
                sym = t.get("symbol")
                if _already_booked(self.trader, t):
                    closed_syms.add(sym)  # booked elsewhere; just prune it
                    continue
                exit_price, pnl_percent = realized_from_fills(
                    t, fills_by_symbol.get(sym, []))
                if exit_price is None:
                    log(f"[RECONCILE] no sell fill for {sym}; leaving tracked")
                    continue
                log(f"[RECONCILE] booking missed close {sym} "
                    f"exit={exit_price:.2f} pnl={pnl_percent:+.1f}%")
                try:
                    self.trader.record_trade_outcome(
                        t, "reconciled_fill_close", pnl_percent)
                    closed_syms.add(sym)
                except Exception as e:
                    log(f"[RECONCILE] record failed for {sym}: {e}")
            if closed_syms:
                survivors = [t for t in trades
                             if t.get("symbol") not in closed_syms]
                _write_active_trades(survivors)
        except Exception as e:
            log(f"[WARN] reconcile_closed_from_fills failed: {e}")

    def _underlying_position_count(self, sym):
        """Number of open positions whose OCC root matches this underlying."""
        positions = self.trader.get_positions() or []
        return sum(1 for p in positions if underlying_of(p.get("symbol")) == sym)

    def _resolve_skip_counterfactuals(self, now=None):
        """Score any due SKIP episodes with their forward-underlying return.

        No-op unless the shadow recorder's episode store is active. Runs inside
        the scan (market open only), so a skip logged late in a session resolves
        on the next session once its ~390-min horizon has elapsed."""
        store = getattr(self.trader, "_episode_store", None)
        if not store:
            return
        try:
            from skip_counterfactual import resolve_due_skips
            n = resolve_due_skips(
                store, self.trader.get_current_price,
                horizon_min=self.cfg.skip_cf_horizon_min, now=now)
            if n:
                log(f"[SKIP-CF] resolved {n} counterfactual skips")
        except Exception as e:
            log(f"[WARN] skip-counterfactual resolve failed: {e}")

    # -- entries ------------------------------------------------------------- #
    def _evaluate(self, sym):
        """Score a symbol's best contract WITHOUT entering. Returns a candidate
        dict ``{sym, direction, opt, score}`` or None if the symbol skips.

        Selection is split from entry so the entry loop can rank candidates by
        score and let the daily cap fill with the strongest setups (not just
        whatever comes first in universe order)."""
        # Per-underlying capacity gate. A name may accrue up to
        # cfg.max_per_underlying open positions (the contract-level
        # _has_open_or_pending guard still blocks stacking the identical
        # contract, so growth only happens on a different strike/expiry/side).
        # entered_today is no longer a gate -- it stays populated on entry only
        # for the heartbeat/status display -- so a below-cap name is re-evaluated
        # across scans instead of being one-and-done for the day.
        cap = getattr(self.cfg, "max_per_underlying", 1)
        held = self._underlying_position_count(sym)
        if cap and held >= cap:
            log(f"[SKIP] {sym}: at per-underlying cap ({held}/{cap})")
            return None

        self.trader.ticker = sym  # keep direction/underlying defaults correct

        price = self.trader.get_current_price(sym)
        if not price:
            log(f"[SKIP] {sym}: no current price")
            return None

        contracts = self.trader.get_option_contracts(sym)
        if not contracts:
            log(f"[SKIP] {sym}: no option contracts")
            return None

        direction = self.trader.determine_option_strategy(sym)  # 'call' / 'put' / 'skip'
        # Phase 3: a weak/flat-signal NO_TRADE short-circuits before any contract
        # lookup or order. OFF by default (never returns 'skip'), so unchanged.
        if str(direction).lower() in ("skip", "no_trade"):
            reason = getattr(self.trader, "last_skip_reason", None) or "weak/flat signal"
            log(f"[SKIP] {sym}: NO_TRADE ({reason})")
            return None
        opt = self.trader.select_best_option(contracts, price, strategy=direction)
        if not opt:
            log(f"[SKIP] {sym}: no suitable option ({direction.upper()})")
            return None

        # Phase 4: aggregate-exposure preview. When USE_PORTFOLIO_GREEK_LIMITS is
        # on, project this candidate onto the live book and skip it early (before
        # ranking/entry) if it would breach a portfolio delta/vega/theta or
        # same-direction/per-underlying cap. OFF by default -> never runs, so the
        # scheduler is unchanged. place_order_with_stops re-checks authoritatively
        # with the final conviction-sized quantity; this preview uses cfg.qty.
        if getattr(self.trader, "use_portfolio_greek_limits", False):
            try:
                pf = self.trader._portfolio_greek_check(opt, self.cfg.qty)
                if not pf.get("allowed", True):
                    log(f"[SKIP] {sym}: portfolio limit ({pf.get('reason')})")
                    return None
            except Exception as e:
                log(f"[WARN] portfolio preview failed for {sym} (ignored): {e}")

        return {"sym": sym, "direction": direction, "opt": opt,
                "score": float(opt.get("score") or 0),
                "underlying_price": price}

    def _enter_candidate(self, cand):
        """Place (or, in dry-run, log) the entry for a pre-scored candidate."""
        sym, direction, opt = cand["sym"], cand["direction"], cand["opt"]
        self.trader.ticker = sym  # underlying/direction defaults
        desc = (f"{sym} {direction.upper()} {opt.get('symbol')} x{self.cfg.qty} "
                f"(ask={opt.get('ask')}, score={opt.get('score')})")

        if not self.cfg.armed:
            log(f"[DRY-RUN] WOULD ENTER {desc}")
            self.entered_today.add(sym)
            return

        log(f"[ARMED] entering {desc}")
        # quantity=None lets place_order_with_stops size by directional conviction
        # (very high -> 3, high -> 2, regular -> 1 contract).
        order = self.trader.place_order_with_stops(opt, quantity=None)
        if order:
            self.trades_today += 1
            self.entered_today.add(sym)
            _stamp_scheduler_trade(opt.get("symbol"))
            log(f"[ARMED] order placed for {opt.get('symbol')} "
                f"(trades_today={self.trades_today})")
        else:
            log(f"[ARMED] order NOT placed for {opt.get('symbol')} "
                f"(blocked by risk/budget/duplicate or broker rejection)")
            # Capture the declined setup as a SKIP episode (counterfactual class).
            # Single capture point for every non-placement -- EV-gate, portfolio,
            # risk engine, budget, duplicate, or broker rejection -- carrying the
            # underlying price so the resolver can score the forward return later.
            rec = getattr(self.trader, "shadow_recorder", None)
            if rec:
                try:
                    rec.on_decision(
                        symbol=opt.get("symbol"),
                        underlying=sym,
                        analysis={
                            "should_trade": False,
                            "direction": direction,
                            "underlying_price": cand.get("underlying_price"),
                            "confidence": cand.get("score"),
                        },
                        quote=None,
                        mode="live-paper-blocked",
                        risk={"block_reason": getattr(self.trader,
                                                      "last_block_reason", None)},
                    )
                except Exception as e:
                    log(f"[SKIP-CF] capture failed for {sym}: {e}")

    def _rank_and_enter(self):
        """Evaluate the whole universe, rank by score (desc), then enter in that
        order. Armed mode stops at max_new_trades_per_day so the cap fills with
        the highest-scoring setups; dry-run previews all candidates, ranked."""
        candidates = []
        for sym in self.symbols:
            try:
                cand = self._evaluate(sym)
            except Exception as e:
                log(f"[WARN] evaluate failed for {sym}: {e}")
                cand = None
            if cand:
                candidates.append(cand)

        if not candidates:
            log("[ENTRY] no candidates this scan")
            return

        candidates.sort(key=lambda c: c["score"], reverse=True)
        log("[RANK] " + ", ".join(f"{c['sym']}({c['score']:.0f})" for c in candidates))

        for cand in candidates:
            if self.cfg.armed and self.trades_today >= self.cfg.max_new_trades_per_day:
                log(f"[CAP] reached max_new_trades_per_day={self.cfg.max_new_trades_per_day}")
                break
            try:
                self._enter_candidate(cand)
            except Exception as e:
                log(f"[WARN] entry attempt failed for {cand['sym']}: {e}")

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
            "symbols": self.symbols,
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

        # 1b) fill-driven safety net: book + prune any tracked position that
        # has already left the broker (closed via any path) from its real fill.
        self.reconcile_closed_from_fills()

        # 1d) resolve any SKIP episodes whose counterfactual horizon has elapsed
        # (forward underlying return in the would-be direction). No-op unless the
        # shadow recorder is active. Uses naive datetime.now() to match the naive
        # created_at the episode store stamps at decision time (no tz drift).
        self._resolve_skip_counterfactuals()

        # 1c) time stop (hold-overnight mode): positions held past
        # MAX_HOLD_DAYS calendar days are force-closed so nothing rots while
        # waiting on a TP/SL that never resolves. No-op when disabled (0).
        if self.cfg.max_hold_days > 0:
            try:
                self.force_close_scheduler_positions(
                    reason="TIME_STOP",
                    only=lambda t: held_past_max_days(
                        t.get("entry_time"), parsed["now"],
                        self.cfg.max_hold_days))
            except Exception as e:
                log(f"[WARN] time-stop sweep failed: {e}")

        # 2) EOD window — never open new trades this late; close ours only
        # when EOD_CLOSE_ENABLED (hold-overnight mode leaves them running
        # to their TP/SL/time stop, monitored again next session).
        if in_eod_window(parsed["now"], parsed["next_close"], eod_min=self.cfg.eod_close_min):
            log(f"[EOD] within {self.cfg.eod_close_min:g} min of close")
            if self.cfg.eod_close_enabled:
                self.force_close_scheduler_positions()
            else:
                log("[EOD] hold-overnight mode: leaving positions open")
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
                self._rank_and_enter()
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
        self._refresh_universe()
        log(f"[START] Alpaca intraday scheduler — mode={self.cfg.mode}, "
            f"symbols={self.symbols}, qty={self.cfg.qty}, "
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

    # hold-overnight time stop
    now = parse_clock(clk("12:00"))["now"]  # 2026-04-01
    check("timestop: fresh trade not past max days",
          held_past_max_days("2026-04-01T10:00:00", now, 5) is False)
    check("timestop: exactly max days old is kept",
          held_past_max_days("2026-03-27T10:00:00", now, 5) is False)
    check("timestop: older than max days triggers",
          held_past_max_days("2026-03-26T10:00:00", now, 5) is True)
    check("timestop: disabled (0) never triggers",
          held_past_max_days("2020-01-01T10:00:00", now, 0) is False)
    check("timestop: garbage entry_time fails open",
          held_past_max_days("not-a-date", now, 5) is False)
    check("timestop: missing entry_time fails open",
          held_past_max_days(None, now, 5) is False)

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
    check("config eod close enabled by default",
          Config(env={}).eod_close_enabled is True)
    check("config eod close disables via env",
          Config(env={"EOD_CLOSE_ENABLED": "0"}).eod_close_enabled is False)
    check("config max_hold_days default off", Config(env={}).max_hold_days == 0)
    check("config max_hold_days via env",
          Config(env={"MAX_HOLD_DAYS": "7"}).max_hold_days == 7.0)
    check("config symbols default SPY,QQQ", Config(env={}).symbols == ["SPY", "QQQ"])

    # universe: screener-merge logic (deterministic; screener disabled -> no read)
    check("config max_symbols default 12", Config(env={}).max_symbols == 12)
    check("config include_screener default on", Config(env={}).include_screener is True)
    check("config include_screener off via env",
          Config(env={"SCHEDULER_INCLUDE_SCREENER": "0"}).include_screener is False)
    s_off = IntradayScheduler(Config(env={"SCHEDULER_INCLUDE_SCREENER": "0"}),
                              trader=None, pdt=None)
    s_off.base_symbols = ["SPY", "QQQ", "SPY"]
    s_off.symbols = []
    s_off._refresh_universe()
    check("universe dedupes base (screener off)", s_off.symbols == ["SPY", "QQQ"])
    s_cap = IntradayScheduler(Config(env={"SCHEDULER_INCLUDE_SCREENER": "0",
                                          "SCHEDULER_MAX_SYMBOLS": "2"}),
                              trader=None, pdt=None)
    s_cap.base_symbols = ["SPY", "QQQ", "AAPL", "NVDA"]
    s_cap.symbols = []
    s_cap._refresh_universe()
    check("universe respects max_symbols cap", s_cap.symbols == ["SPY", "QQQ"])

    # per-underlying cap: config default + count helper
    check("config max_per_underlying default 1", Config(env={}).max_per_underlying == 1)
    check("config max_per_underlying via env",
          Config(env={"MAX_POSITIONS_PER_UNDERLYING": "10"}).max_per_underlying == 10)

    class _CountTrader:
        def get_positions(self):
            return [{"symbol": "SPY240705C00540000"},
                    {"symbol": "SPY260101P00400000"},
                    {"symbol": "QQQ260101P00400000"}]
    s_cnt = IntradayScheduler(Config(env={}), trader=_CountTrader(), pdt=None)
    check("count helper tallies multiple per underlying",
          s_cnt._underlying_position_count("SPY") == 2)
    check("count helper isolates other underlyings",
          s_cnt._underlying_position_count("QQQ") == 1)
    check("count helper zero when unheld",
          s_cnt._underlying_position_count("AAPL") == 0)

    # SKIP-counterfactual: config knob + resolver end-to-end (no network).
    check("config skip_cf_horizon_min default 390",
          Config(env={}).skip_cf_horizon_min == 390)
    check("config skip_cf_horizon_min via env",
          Config(env={"SKIP_CF_HORIZON_MIN": "120"}).skip_cf_horizon_min == 120)

    from episode_store import EpisodeStore
    from skip_counterfactual import resolve_due_skips
    _store = EpisodeStore(":memory:")
    _did = _store.log_decision(
        symbol="SPY260101C00500000", underlying="SPY", strat="t",
        features={"raw": {"underlying_price": 100.0}, "state_key": "k"},
        quote=None, modeled_cost=None, rule_action="CALL", rule_confidence=0.0,
        gate=None, chosen_action="SKIP", qty=1, mode="live-paper-blocked")
    check("open_skips finds the logged skip",
          len(_store.open_skips()) == 1 and _store.open_skips()[0]["decision_id"] == _did)
    # horizon_min=0 -> immediately due; price 100->90 on a CALL skip = +10%.
    _n = resolve_due_skips(_store, lambda s: 90.0, horizon_min=0)
    _rows = _store._rows("SELECT * FROM episodes WHERE decision_id=?", (_did,))
    check("resolve_due_skips resolves a due skip", _n == 1)
    check("resolve_due_skips sets skip_resolved outcome",
          _rows and _rows[0]["outcome"] == "skip_resolved")
    check("resolve_due_skips computes +10% for CALL 100->90",
          _rows and abs((_rows[0]["net_pnl_pct"] or 0) - 10.0) < 1e-9)
    check("resolved skip no longer open", len(_store.open_skips()) == 0)
    _store.close()

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
