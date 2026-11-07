"""Replay SPY same-day puts on completed bars and timestamped option BBOs.

Run from the repository root.

    python -m oracle.lab.spy_put_roll session.jsonl

The first JSONL row supplies the exchange session and SPY's previous regular close.

    {"open":"2026-11-05T09:30:00-05:00","close":"2026-11-05T16:00:00-05:00","previous_close":760}

The remaining rows supply completed one-minute SPY bars or option BBO updates.

    {"type":"bar","ts":"2026-11-05T14:31:00Z","o":757,"h":758,"l":756,"c":757,"v":1000}
    {"type":"quote","ts":"2026-11-05T14:41:01Z","symbol":"SPY261105P00750000","bid":1.24,"ask":1.28,"bid_size":20,"ask_size":25}

Bar timestamps are the end of the minute. Optional received_at is when a row
became available to the strategy. Use actual quote sizes in contracts. The
model assumes sale proceeds are reusable intraday unless --cash-account is
given. Each file starts with the configured balance. The three profiles use
the same opening signal. Hold keeps the first put until a stop or cutoff,
fixed_roll reuses a fixed premium budget, and capped_roll reinvests a share
of cumulative realized gains up to the configured premium cap. This is an
offline replay and expects normalized data from a historical quote feed.
"""

import argparse
import json
import math
from dataclasses import dataclass, replace
from datetime import datetime, timedelta, timezone
from functools import lru_cache

from backfill_episodes import parse_occ
from cost_model import CONTRACT_MULTIPLIER, CostConfig
from oracle.engine.bus import EventBus
from oracle.engine.events import BarEvent, OptionQuoteEvent, TimerEvent, TradeIntentEvent
from oracle.engine.sim_broker import SimBroker
from oracle.execution.fill_simulator import FillSimulator, LatencyConfig


MINUTE = timedelta(minutes=1)
SOURCE = "spy_put_roll"
PROFILES = ("hold", "fixed_roll", "capped_roll")


@dataclass(frozen=True)
class Config:
    starting_cash: float = 700.0
    base_premium: float = 175.0
    max_premium: float = 350.0
    daily_loss: float = 175.0
    reinvest_fraction: float = 0.5
    max_rolls: int = 3
    opening_minutes: int = 10
    gap_down: float = 0.0025
    opening_decline: float = 0.002
    roll_gain: float = 0.30
    stop_loss: float = 0.35
    max_spread_fraction: float = 0.12
    quote_age_seconds: float = 5.0
    order_delay_seconds: float = 0.5
    order_timeout_seconds: float = 20.0
    reuse_sale_proceeds: bool = True
    seed: int = 0


@dataclass(frozen=True)
class Session:
    opened: datetime
    closed: datetime
    previous_close: float


@dataclass(frozen=True)
class Order:
    side: str
    symbol: str
    qty: int
    submitted_at: datetime
    reason: str
    limit: float | None = None


class PutReplay:
    def __init__(self, session, config, profile):
        self.session = session
        self.cfg = config
        self.profile = profile
        self.opening_end = session.opened + config.opening_minutes * MINUTE
        self.order_cutoff = session.closed - 30 * MINUTE
        self.flatten_at = session.closed - 40 * MINUTE
        self.roll_until = self.flatten_at - 5 * MINUTE
        self.initial_until = min(session.opened + 60 * MINUTE, self.roll_until)
        self.trade_date = session.opened.date().isoformat()
        self.fee = CostConfig().occ_fee_per_contract
        self.cash = self.settled_budget = config.starting_cash
        self.fees = 0.0
        self.peak_equity = config.starting_cash
        self.max_drawdown = 0.0
        self.opening_bars = []
        self.last_bar = None
        self.turnover = self.volume = 0.0
        self.quotes = {}
        self.held = None
        self.previous_strike = None
        self.entries = 0
        self.can_roll = False
        self.leg_start_pnl = 0.0
        self.pending = None
        self.cancelled_buys = 0
        # Apply observed quote arrival latency before each attempt. A single
        # simulation slice avoids assuming that displayed depth replenishes.
        self.broker = SimBroker(
            emit=lambda event: None, seed=config.seed,
            fill_simulator=FillSimulator(config=LatencyConfig(
                submit_latency_sec=0, ack_latency_sec=0, max_slices=1,
            )),
        )

    @property
    def net_pnl(self):
        return self.broker.realized_pnl - self.fees

    def on_bar(self, event):
        ended = event.data["market_timestamp"]
        now = event.event_timestamp
        if ended <= self.opening_end:
            self.opening_bars.append(event)
        self.last_bar = event
        self.turnover += event.v * (event.h + event.l + event.c) / 3
        self.volume += event.v
        self.decide(now, allow_entry=True)

    def on_quote(self, quote):
        previous = self.quotes.get(quote.symbol)
        if previous and quote.data["market_timestamp"] < previous.data["market_timestamp"]:
            return  # delayed update cannot supersede a newer market snapshot
        self.quotes[quote.symbol] = quote
        order = self.pending
        if order is not None and order.symbol == quote.symbol:
            self.try_fill(order, quote)
        # Update risk intraminute. Entry still requires a completed SPY bar.
        if quote.symbol == self.held:
            self.decide(quote.event_timestamp)
            if quote.bid > 0 and quote.ask >= quote.bid and quote.data["bid_size"] > 0:
                qty = self.broker.books[self.held].qty
                self.mark_equity(self.cash + qty * (quote.bid * CONTRACT_MULTIPLIER - self.fee))

    def mark_equity(self, equity):
        self.peak_equity = max(self.peak_equity, equity)
        self.max_drawdown = max(self.max_drawdown, self.peak_equity - equity)

    def decide(self, now, *, allow_entry=False):
        if self.pending and self.pending.side == "sell":
            if now >= self.flatten_at:
                self.can_roll = False
            return

        bar = self.last_bar
        vwap = self.turnover / self.volume if self.volume else 0.0
        if self.held:
            book = self.broker.books[self.held]
            quote = self.quotes[self.held]
            strike = _contract(self.held)[3]
            reason = None
            if now >= self.flatten_at:
                reason = "time_exit"
            elif bar and vwap and bar.c >= vwap:
                reason = "vwap_reclaim"
            elif self.fresh(quote, now):
                if quote.bid <= book.avg_price * (1 - self.cfg.stop_loss):
                    reason = "premium_stop"
                elif (self.profile != "hold" and self.entries <= self.cfg.max_rolls
                      and now < self.roll_until and bar
                      and now - bar.data["market_timestamp"] <= MINUTE
                      and bar.c - strike <= 2
                      and quote.bid >= book.avg_price * (1 + self.cfg.roll_gain)):
                    reason = "roll_winner"
            if reason:
                self.can_roll = reason == "roll_winner"
                self.pending = Order("sell", self.held, int(book.qty), now, reason)
            return

        cutoff = self.roll_until if self.entries else self.initial_until
        if self.pending:
            if (now >= cutoff or not self.bearish(bar, vwap)
                    or now - self.pending.submitted_at
                    > timedelta(seconds=self.cfg.order_timeout_seconds)):
                self.cancelled_buys += 1
                self.pending = None
            else:
                return

        if (not allow_entry or now >= cutoff or not bar
                or (self.entries and not self.can_roll)):
            return
        if now <= self.opening_end or not self.bearish(bar, vwap):
            return
        self.submit_buy(now, bar.c)

    def bearish(self, bar, vwap):
        if bar is None or len(self.opening_bars) != self.cfg.opening_minutes:
            return False
        first, last = self.opening_bars[0], self.opening_bars[-1]
        if any(item.data["market_timestamp"] != self.session.opened + i * MINUTE
               for i, item in enumerate(self.opening_bars, 1)):
            return False
        return (
            first.o <= self.session.previous_close * (1 - self.cfg.gap_down)
            and last.c <= first.o * (1 - self.cfg.opening_decline)
            and bar.c < min(vwap, min(item.l for item in self.opening_bars))
        )

    def fresh(self, quote, now):
        market_time = quote.data["market_timestamp"]
        return now - market_time <= timedelta(seconds=self.cfg.quote_age_seconds)

    def liquid(self, quote):
        return (0 < quote.bid <= quote.ask and quote.data["ask_size"] > 0
                and (quote.ask - quote.bid) / quote.mid <= self.cfg.max_spread_fraction)

    def submit_buy(self, now, spot):
        risk_floor = max(0.0, self.cfg.starting_cash - self.cfg.daily_loss)
        bonus = (max(0.0, self.net_pnl) * self.cfg.reinvest_fraction
                 if self.profile == "capped_roll" else 0)
        budget = min(self.cfg.base_premium + bonus, self.cfg.max_premium,
                     self.cash - risk_floor)
        if not self.cfg.reuse_sale_proceeds:
            budget = min(budget, self.settled_budget)
        candidates = []
        for symbol, quote in self.quotes.items():
            strike = _contract(symbol)[3]
            distance = spot - strike
            if not 3 <= distance <= 5:
                continue
            if self.previous_strike is not None and strike >= self.previous_strike:
                continue
            if not self.fresh(quote, now) or not self.liquid(quote):
                continue
            limit = round(quote.ask + 0.02, 4)
            qty = min(int(quote.data["ask_size"]),
                      int(budget // (limit * CONTRACT_MULTIPLIER + 2 * self.fee)))
            if qty > 0:
                candidates.append((abs(distance - 4), quote.ask - quote.bid, symbol, qty, limit))
        if candidates:
            _, _, symbol, qty, limit = min(candidates)
            reason = "roll_entry" if self.entries else "opening_breakdown"
            self.pending = Order("buy", symbol, qty, now, reason, limit)

    def try_fill(self, order, quote):
        now = quote.event_timestamp
        market_time = quote.data["market_timestamp"]
        if market_time < order.submitted_at + timedelta(seconds=self.cfg.order_delay_seconds):
            return
        if not self.fresh(quote, now) or now >= self.order_cutoff:
            return
        if order.side == "buy":
            self.decide(now)  # cancel buys that have aged out or crossed a cutoff
            if self.pending is None or not self.liquid(quote) or quote.ask > order.limit:
                return
            qty = min(order.qty, quote.data["ask_size"])
        else:
            if not 0 < quote.bid <= quote.ask or quote.data["bid_size"] <= 0:
                return
            qty = min(order.qty, quote.data["bid_size"])

        self.broker.on_trade_intent(TradeIntentEvent(
            symbol="SPY", contract=order.symbol, side=order.side, direction="PUT",
            qty=qty, order_type="limit" if order.limit else "market",
            limit_price=order.limit, bid=quote.bid, ask=quote.ask, market=quote.data,
            event_timestamp=now, source=SOURCE,
            correlation_id=f"{order.symbol}-{len(self.broker.order_log) + 1}",
        ))
        record = self.broker.order_log[-1]
        record.update(ts=now.isoformat(), submitted_at=order.submitted_at.isoformat(),
                      reason=order.reason, remaining_before_attempt=order.qty)
        filled = int(record["filled_qty"])
        if not filled:
            return

        fee = self.fee * filled
        notional = record["avg_fill_price"] * filled * CONTRACT_MULTIPLIER
        if order.side == "buy":
            self.leg_start_pnl = self.net_pnl
        self.fees += fee
        if order.side == "buy":
            self.cash -= notional + fee
            self.settled_budget -= notional + fee
            self.entries += 1
            self.held = order.symbol
            self.previous_strike = _contract(order.symbol)[3]
            self.pending = None  # keep the fill and cancel any remaining entry size
            self.can_roll = False
        else:
            self.cash += notional - fee
            remaining = int(self.broker.books[order.symbol].qty)
            if remaining:
                self.pending = replace(order, qty=remaining)
            else:
                self.held = self.pending = None
                self.can_roll = self.can_roll and self.net_pnl > self.leg_start_pnl
                self.mark_equity(self.cash)

    def report(self):
        return {
            "profile": self.profile,
            "net_realized_pnl": round(self.net_pnl, 2),
            "ending_cash": round(self.cash, 2),
            "estimated_fees": round(self.fees, 2),
            "max_bid_mark_drawdown": round(self.max_drawdown, 2),
            "entries": self.entries,
            "cancelled_buys": self.cancelled_buys + int(
                self.pending is not None and self.pending.side == "buy"),
            "open_positions": self.broker.summary()["open_positions"],
            "flat_at_end": self.held is None,
            "orders": self.broker.order_log,
        }


@lru_cache(maxsize=8192)
def _contract(symbol):
    return parse_occ(symbol)


def replay(session, events, config=Config(), profile="capped_roll"):
    """Replay one session. Events are ordered by availability; tied bars run first."""
    if profile not in PROFILES:
        raise ValueError(f"Unknown profile {profile!r}")
    model = PutReplay(session, config, profile)
    bus = EventBus()
    bus.subscribe("BAR", model.on_bar)
    bus.subscribe("OPTION_QUOTE", model.on_quote)
    bus.subscribe("TIMER", lambda event: model.decide(event.event_timestamp))
    bus.publish(TimerEvent(event_timestamp=model.flatten_at))
    # At a tie, observe the completed SPY bar before attempting a quote fill.
    available = []
    for event in events:
        if (event.event_timestamp.utcoffset() is None
                or (event.received_timestamp is not None
                    and event.received_timestamp.utcoffset() is None)):
            raise ValueError("Timestamps must include a UTC offset")
        market_time = event.event_timestamp.astimezone(timezone.utc)
        now = (event.received_timestamp or market_time).astimezone(timezone.utc)
        if now < market_time:
            raise ValueError("An event cannot be received before its market timestamp")
        if not session.opened < now < session.closed or market_time < session.opened:
            continue
        if isinstance(event, BarEvent) and event.symbol == "SPY":
            if now - market_time >= MINUTE:
                continue
        elif isinstance(event, OptionQuoteEvent):
            contract = _contract(event.symbol)
            if not contract or contract[:3] != ("SPY", model.trade_date, "P"):
                continue
        else:
            continue
        # The bus orders replay observations by availability. Keep the
        # original market time for completed-bar and quote-freshness checks.
        available.append(replace(
            event, event_timestamp=now, received_timestamp=now,
            data={**event.data, "market_timestamp": market_time},
        ))
    available.sort(key=lambda event: (event.event_timestamp, not isinstance(event, BarEvent)))
    bus.publish_all(available)
    bus.run(max_events=len(available) + 1)
    if bus.errors:
        raise RuntimeError(f"Replay handler failed: {bus.errors[0]}")
    report = model.report()
    bar_count = sum(isinstance(event, BarEvent) for event in available)
    report["data"] = {
        "bars": bar_count,
        "expected_bars": int((session.closed - session.opened) // MINUTE) - 1,
        "same_day_put_quotes": len(available) - bar_count,
    }
    return report


def timestamp(value):
    parsed = datetime.fromisoformat(value)
    if parsed.utcoffset() is None:
        raise ValueError("Timestamps must include a UTC offset")
    return parsed.astimezone(timezone.utc)


def load_feed(path):
    with open(path, encoding="utf-8") as source:
        header = json.loads(next(source))
        session = Session(timestamp(header["open"]), timestamp(header["close"]),
                          float(header["previous_close"]))
        if (not math.isfinite(session.previous_close) or session.previous_close <= 0
                or session.closed.date() != session.opened.date()
                or session.closed - session.opened <= 60 * MINUTE):
            raise ValueError("Invalid session metadata")
        events = []
        bar_times = set()
        for number, line in enumerate(source, 2):
            if not line.strip():
                continue
            row = json.loads(line)
            ts = timestamp(row["ts"])
            received = timestamp(row["received_at"]) if "received_at" in row else None
            if row["type"] == "bar":
                o, h, l, c, v = (float(row[key]) for key in "ohlcv")
                if (ts in bar_times or ts.second or ts.microsecond
                        or not all(math.isfinite(x) for x in (o, h, l, c, v))
                        or not 0 < l <= min(o, c) <= max(o, c) <= h or v < 0):
                    raise ValueError(f"Invalid or duplicate SPY minute bar on line {number}")
                bar_times.add(ts)
                events.append(BarEvent(symbol="SPY", event_timestamp=ts,
                                       received_timestamp=received, timeframe="1Min",
                                       o=o, h=h, l=l, c=c, v=v))
            elif row["type"] == "quote":
                bid, ask = float(row["bid"]), float(row["ask"])
                sizes = {key: row[key] for key in ("bid_size", "ask_size")}
                if (not math.isfinite(bid) or not math.isfinite(ask) or min(bid, ask) < 0
                        or any(type(size) is not int or size < 0 for size in sizes.values())):
                    raise ValueError(f"Invalid BBO on line {number}")
                events.append(OptionQuoteEvent(symbol=row["symbol"], underlying="SPY",
                                               event_timestamp=ts, received_timestamp=received,
                                               bid=bid, ask=ask, data=sizes))
            else:
                raise ValueError(f"Unknown event type on line {number}")
    return session, events


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("session", nargs="+", help="One or more session JSONL files")
    parser.add_argument("--cash-account", action="store_true",
                        help="Do not reuse same-day sales when sizing new purchases")
    parser.add_argument("--starting-cash", type=float, default=Config.starting_cash)
    parser.add_argument("--base-premium", type=float, default=Config.base_premium)
    parser.add_argument("--max-premium", type=float, default=Config.max_premium)
    parser.add_argument("--daily-loss", type=float, default=Config.daily_loss)
    parser.add_argument("--reinvest-fraction", type=float, default=Config.reinvest_fraction)
    parser.add_argument("--profile", choices=PROFILES, help="Run only this profile")
    args = parser.parse_args()
    budgets = (args.starting_cash, args.base_premium, args.max_premium, args.daily_loss)
    if any(not math.isfinite(value) or value <= 0 for value in budgets):
        parser.error("Cash, premium, and loss budgets must be positive and finite")
    if args.max_premium < args.base_premium or not 0 <= args.reinvest_fraction <= 1:
        parser.error("Set max premium at or above base premium and reinvest fraction from 0 to 1")
    config = Config(
        starting_cash=args.starting_cash, base_premium=args.base_premium,
        max_premium=args.max_premium, daily_loss=args.daily_loss,
        reinvest_fraction=args.reinvest_fraction, reuse_sale_proceeds=not args.cash_account,
    )
    for path in args.session:
        session, events = load_feed(path)
        results = [replay(session, events, config, profile)
                   for profile in ((args.profile,) if args.profile else PROFILES)]
        print(json.dumps({"session": session.opened.date().isoformat(), "results": results}))


if __name__ == "__main__":
    main()
