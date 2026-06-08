"""
Unified exit manager — the single source of truth for stop loss, take profit,
trailing stop, expiration, and stale-position exits.

Phase 5: before this module, two code paths decided exits independently:
  * the scheduler (`smart_trader.monitor_positions`) — the canonical, rich path
    that actually closes positions and records outcomes; and
  * the Telegram monitor (`telegram_bot.monitor_positions`) — ALERT-ONLY, with
    hardcoded +20%/-5%/-10% thresholds that never closed or recorded anything.

This module extracts the basic stop/take/trailing/expiration/stale comparison so
both callers share ONE decision function (`evaluate_exit`). The scheduler keeps
its richer roll-on-profit / partial-close / dynamic-exit branches around it; the
Telegram monitor opts into enforcement via `USE_UNIFIED_EXIT_MANAGER`.

Design notes / parity:
  * `evaluate_exit` is PURE (no network, no I/O) and deterministic, so it is
    unit-testable offline. It takes already-resolved level percentages so the
    caller controls how levels are computed.
  * Level units mirror `smart_trader.monitor_positions`: `stop_loss_percent` and
    `take_profit_percent` are in PERCENT (e.g. 10.0, 20.0); `trailing_stop_
    distance` is a FRACTION (e.g. 0.05). `pnl_percent` is in percent.
  * Precedence matches the scheduler exactly: stop loss > take profit > trailing
    stop. Expiration and stale are appended after (the scheduler keeps its own
    expiration handling inside `should_exit_dynamically`, so it passes
    `check_expiration=False`; the Telegram path passes `check_expiration=True`).
  * `enforce_exit` performs the side-effecting close + a SINGLE
    `record_trade_outcome` call, plus the unified one-line exit log. The
    scheduler maps the decision action to its legacy reason codes
    ('dynamic_stop_loss' etc.) so recorded outcomes stay byte-identical.
"""

from dataclasses import dataclass
from datetime import datetime
from typing import Dict, Optional


# --------------------------------------------------------------------------- #
# Decision
# --------------------------------------------------------------------------- #
@dataclass
class ExitDecision:
    action: str        # 'stop_loss'|'take_profit'|'trailing_stop'|'expiration'|'stale'|'hold'
    reason: str        # human-readable explanation
    pnl_percent: float
    should_exit: bool


def _parse_dt(value) -> Optional[datetime]:
    """Best-effort ISO/timestamp parse; None on failure (fail-open)."""
    if not value:
        return None
    try:
        return datetime.fromisoformat(str(value))
    except (ValueError, TypeError):
        pass
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(str(value), fmt)
        except (ValueError, TypeError):
            continue
    return None


def evaluate_exit(
    trade: Dict,
    current_price: float,
    levels: Dict,
    *,
    now: Optional[datetime] = None,
    roll_enabled: bool = False,
    check_expiration: bool = True,
    max_hold_days: Optional[int] = None,
) -> ExitDecision:
    """Decide whether an open option position should be exited.

    Pure / deterministic. Precedence (matches the scheduler): stop loss, then
    take profit (suppressed when ``roll_enabled`` so winners can run to the roll
    trigger), then trailing stop, then expiration, then stale.

    ``levels`` keys (resolved by the caller):
      * ``stop_loss_percent``      — percent, e.g. 10.0
      * ``take_profit_percent``    — percent, e.g. 20.0
      * ``trailing_stop_distance`` — fraction, e.g. 0.05

    Returns ``ExitDecision``. ``action='hold'`` (``should_exit=False``) when no
    rule fires or inputs are unusable (fail-open: never invents an exit).
    """
    now = now or datetime.now()

    try:
        entry_price = float(trade.get('entry_price') or 0)
        current_price = float(current_price or 0)
    except (TypeError, ValueError):
        return ExitDecision('hold', 'bad_price_input', 0.0, False)

    if entry_price <= 0 or current_price <= 0:
        return ExitDecision('hold', 'missing_price', 0.0, False)

    pnl_percent = ((current_price - entry_price) / entry_price) * 100.0

    stop_pct = float(levels.get('stop_loss_percent', 10.0))
    take_pct = float(levels.get('take_profit_percent', 20.0))
    trail_dist = float(levels.get('trailing_stop_distance', 0.05))

    # 1) Stop loss.
    if pnl_percent <= -stop_pct:
        return ExitDecision(
            'stop_loss', f'stop_loss {pnl_percent:.1f}% <= -{stop_pct:.1f}%',
            pnl_percent, True)

    # 2) Take profit (full exit). Suppressed while rolling is enabled so the
    #    scheduler can let winners run to its roll trigger instead.
    if not roll_enabled and pnl_percent >= take_pct:
        return ExitDecision(
            'take_profit', f'take_profit {pnl_percent:.1f}% >= {take_pct:.1f}%',
            pnl_percent, True)

    # 3) Trailing stop (only once armed, i.e. price has exceeded entry).
    if trade.get('trailing_stop_active'):
        try:
            highest = float(trade.get('highest_price') or 0)
        except (TypeError, ValueError):
            highest = 0.0
        if highest > 0:
            trailing_stop_price = highest * (1 - trail_dist)
            if current_price <= trailing_stop_price:
                return ExitDecision(
                    'trailing_stop',
                    f'trailing_stop {trail_dist:.1%} from high {highest:.2f}',
                    pnl_percent, True)

    # 4) Expiration (near expiry). Scheduler keeps this in should_exit_dynamically
    #    and passes check_expiration=False; the Telegram path opts in.
    if check_expiration and trade.get('expiration'):
        try:
            dte = (datetime.strptime(trade['expiration'], '%Y-%m-%d') - now).days
            if dte <= 2:
                return ExitDecision(
                    'expiration', f'near_expiration dte={dte}', pnl_percent, True)
        except (ValueError, TypeError):
            pass

    # 5) Stale position (held too long). Only when a positive cap is supplied.
    if max_hold_days is not None and max_hold_days > 0:
        entry_dt = _parse_dt(trade.get('entry_time'))
        if entry_dt and (now - entry_dt).days >= max_hold_days:
            return ExitDecision(
                'stale', f'stale held>={max_hold_days}d', pnl_percent, True)

    return ExitDecision('hold', 'hold', pnl_percent, False)


# --------------------------------------------------------------------------- #
# Logging + enforcement
# --------------------------------------------------------------------------- #
def format_exit_log(source: str, trade: Dict, current_price: float,
                    reason_code: str, pnl_percent: float) -> str:
    """One-line, parseable exit log with every Phase-5 required field:
    symbol, contract, entry_price, current_price, pnl_percent, exit_reason, source.
    """
    symbol = trade.get('underlying_symbol') or trade.get('ticker') or '?'
    contract = trade.get('symbol', '?')
    try:
        entry_price = float(trade.get('entry_price') or 0)
    except (TypeError, ValueError):
        entry_price = 0.0
    try:
        current_price = float(current_price or 0)
    except (TypeError, ValueError):
        current_price = 0.0
    return (
        f"[EXIT] source={source} symbol={symbol} contract={contract} "
        f"entry_price={entry_price:.2f} current_price={current_price:.2f} "
        f"pnl_percent={pnl_percent:.2f} exit_reason={reason_code}"
    )


def enforce_exit(trader, trade: Dict, position: Dict, reason_code: str,
                 pnl_percent: float, source: str,
                 current_price: Optional[float] = None) -> None:
    """Close the position and record its outcome EXACTLY ONCE, emitting the
    unified exit log first.

    Shared by both callers so the close+record path can never drift. The
    ``reason_code`` is supplied by the caller (the scheduler maps the decision
    action to its legacy 'dynamic_*' codes for byte-identical recorded outcomes;
    the Telegram path uses the decision action directly).
    """
    if current_price is None:
        try:
            current_price = float(position.get('current_price') or 0)
        except (TypeError, ValueError, AttributeError):
            current_price = 0.0
    print(format_exit_log(source, trade, current_price, reason_code, pnl_percent))
    trader.close_position(trade, position, reason_code)
    trader.record_trade_outcome(trade, reason_code, pnl_percent)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    ok = True
    levels = {'stop_loss_percent': 10.0, 'take_profit_percent': 20.0,
              'trailing_stop_distance': 0.05}

    # Stop loss fires when pnl <= -stop.
    d = evaluate_exit({'entry_price': 1.00}, 0.85, levels)
    if d.action != 'stop_loss' or not d.should_exit:
        print("FAIL: stop loss should fire", d); ok = False

    # Take profit fires when pnl >= take and not rolling.
    d = evaluate_exit({'entry_price': 1.00}, 1.25, levels)
    if d.action != 'take_profit':
        print("FAIL: take profit should fire", d); ok = False

    # Take profit suppressed while rolling.
    d = evaluate_exit({'entry_price': 1.00}, 1.25, levels, roll_enabled=True)
    if d.action != 'hold':
        print("FAIL: take profit should be suppressed when rolling", d); ok = False

    # Trailing stop fires once armed and price pulls back past distance, while
    # pnl is still below the take-profit threshold (so take doesn't pre-empt it).
    d = evaluate_exit(
        {'entry_price': 1.00, 'trailing_stop_active': True, 'highest_price': 1.18},
        1.10, levels)
    if d.action != 'trailing_stop':
        print("FAIL: trailing stop should fire", d); ok = False

    # Stop beats take when both could apply (precedence): impossible numerically,
    # but stop must be checked first — a -50% move never reads as take.
    d = evaluate_exit({'entry_price': 1.00}, 0.50, levels)
    if d.action != 'stop_loss':
        print("FAIL: stop precedence", d); ok = False

    # Expiration fires when within 2 days and opted in.
    soon = datetime.now()
    exp = soon.replace(microsecond=0)
    near = (exp).strftime('%Y-%m-%d')
    d = evaluate_exit({'entry_price': 1.00, 'expiration': near}, 1.00, levels,
                      now=exp, check_expiration=True)
    if d.action != 'expiration':
        print("FAIL: expiration should fire", d); ok = False
    # ...but not when the scheduler opts out.
    d = evaluate_exit({'entry_price': 1.00, 'expiration': near}, 1.00, levels,
                      now=exp, check_expiration=False)
    if d.action != 'hold':
        print("FAIL: expiration should be skipped when check_expiration=False", d); ok = False

    # Stale fires when held past the cap.
    old = "2000-01-01T00:00:00"
    d = evaluate_exit({'entry_price': 1.00, 'entry_time': old}, 1.00, levels,
                      max_hold_days=5)
    if d.action != 'stale':
        print("FAIL: stale should fire", d); ok = False

    # Hold when nothing triggers.
    d = evaluate_exit({'entry_price': 1.00}, 1.05, levels)
    if d.should_exit:
        print("FAIL: should hold", d); ok = False

    # Fail-open on bad price.
    d = evaluate_exit({'entry_price': 0}, 1.0, levels)
    if d.should_exit:
        print("FAIL: bad entry price should hold", d); ok = False

    # Log line contains all required fields.
    line = format_exit_log('scheduler',
                           {'underlying_symbol': 'SPY', 'symbol': 'SPY...C',
                            'entry_price': 1.0}, 0.9, 'dynamic_stop_loss', -10.0)
    for token in ('source=scheduler', 'symbol=SPY', 'contract=SPY...C',
                  'entry_price=', 'current_price=', 'pnl_percent=', 'exit_reason=dynamic_stop_loss'):
        if token not in line:
            print(f"FAIL: log missing '{token}'", line); ok = False

    print("exit_manager self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    import sys
    sys.exit(_self_test())
