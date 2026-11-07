"""Run SPY 0DTE shadow trading as a separate, read-only process.

    python -m oracle.lab.spy_0dte_shadow
    python -m oracle.lab.spy_0dte_shadow --direction both

The process follows Alpaca's exchange calendar and writes one JSONL journal
per session and direction setting. It uses OPRA and SIP by default. Indicative
and IEX are available for feed smoke tests. Fill estimates from those feeds
are unsuitable as execution evidence. Run from the repository root to use
the existing .env or export ALPACA_API_KEY and ALPACA_SECRET_KEY.

All requests are GETs. Orders, positions, and P&L in the journal are simulated.
A journal already on disk prevents starting a second book for the same session.
"""

import argparse
import json
import math
import os
import time
from dataclasses import asdict, replace
from datetime import datetime, timedelta, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen
from zoneinfo import ZoneInfo

from backfill_episodes import parse_occ
from oracle.engine.events import BarEvent, OptionQuoteEvent
from oracle.lab.spy_put_roll import (
    DIRECTIONS, MINUTE, PROFILES, Config, OptionRoll, Session, timestamp,
)


PAPER = "https://paper-api.alpaca.markets"
DATA = "https://data.alpaca.markets"
NEW_YORK = ZoneInfo("America/New_York")


def utcnow():
    return datetime.now(timezone.utc)


def market_time(day, local_time):
    return datetime.fromisoformat(f"{day}T{local_time}").replace(
        tzinfo=NEW_YORK).astimezone(timezone.utc)


class FeedError(Exception):
    def __init__(self, endpoint, status=None):
        self.retryable = status is None or status == 429 or status >= 500
        super().__init__(f"{endpoint} unavailable ({status or 'network or timeout'})")


class AlpacaMarketData:
    """The runner's only external interface is to fixed read-only endpoints."""

    def __init__(self, key, secret, stock_feed, option_feed):
        self.headers = {"APCA-API-KEY-ID": key, "APCA-API-SECRET-KEY": secret}
        self.stock_feed = stock_feed
        self.option_feed = option_feed

    def get(self, host, endpoint, **params):
        request = Request(f"{host}{endpoint}?{urlencode(params)}", method="GET")
        for name, value in self.headers.items():
            request.add_unredirected_header(name, value)
        try:
            with urlopen(request, timeout=8) as response:
                return json.load(response)
        except HTTPError as error:
            raise FeedError(endpoint, error.code) from None
        except (URLError, TimeoutError, OSError) as error:
            raise FeedError(endpoint) from error

    def calendar(self, now):
        date = now.astimezone(NEW_YORK).date()
        rows = self.get(PAPER, "/v2/calendar", start=date - timedelta(days=10),
                        end=date + timedelta(days=14))
        return sorted(rows, key=lambda row: row["date"])

    def bars(self, start, end):
        response = self.get(DATA, "/v2/stocks/bars", symbols="SPY", timeframe="1Min",
                            start=start.isoformat(), end=end.isoformat(), feed=self.stock_feed,
                            adjustment="raw", limit=1000)
        received = utcnow()
        for row in (response.get("bars") or {}).get("SPY", ()):
            closed = timestamp(row["t"]) + MINUTE
            o, h, l, c, v = (float(row[key]) for key in "ohlcv")
            if (start < closed <= min(end, received) and all(map(math.isfinite, (o, h, l, c, v)))
                    and 0 < l <= min(o, c) <= max(o, c) <= h and v >= 0):
                yield BarEvent(symbol="SPY", event_timestamp=closed, received_timestamp=received,
                               timeframe="1Min", o=o, h=h, l=l, c=c, v=v)

    def contracts(self, date, directions):
        result = {}
        right_codes = {direction[0].upper() for direction in directions}
        params = {"underlying_symbols": "SPY", "root_symbol": "SPY",
                  "expiration_date": date, "limit": 1000}
        while True:
            response = self.get(PAPER, "/v2/options/contracts", **params)
            for row in response.get("option_contracts") or ():
                symbol = row["symbol"]
                contract = parse_occ(symbol)
                if (contract and contract[0:2] == ("SPY", date)
                        and contract[2] in right_codes
                        and str(row.get("size")) == "100"):
                    result[symbol] = contract[3]
            token = response.get("next_page_token")
            if not token:
                return result
            if token == params.get("page_token"):
                raise ValueError("Alpaca returned a repeated options page token")
            params["page_token"] = token

    def quotes(self, symbols):
        response = self.get(DATA, "/v1beta1/options/quotes/latest",
                            symbols=",".join(symbols), feed=self.option_feed)
        received = utcnow()
        for symbol, row in sorted((response.get("quotes") or {}).items()):
            market = timestamp(row["t"])
            bid, ask = float(row["bp"]), float(row["ap"])
            sizes = {"bid_size": row["bs"], "ask_size": row["as"]}
            if (market <= received and math.isfinite(bid) and math.isfinite(ask)
                    and 0 <= bid <= ask and all(type(n) is int and n >= 0 for n in sizes.values())):
                yield OptionQuoteEvent(symbol=symbol, underlying="SPY", event_timestamp=market,
                                       received_timestamp=received, bid=bid, ask=ask, data=sizes)


def quote_symbols(contracts, spot, models):
    # Subscribe to the held and pending symbols even after SPY moves away.
    needed = {model.held for model in models if model.held}
    needed.update(model.pending.symbol for model in models if model.pending)
    near = sorted((abs(strike - spot), symbol) for symbol, strike in contracts.items()
                  if abs(strike - spot) <= 12 and symbol not in needed)
    return sorted(needed) + [symbol for _, symbol in near[:100 - len(needed)]]


def run_session(feed, session, contracts, directions, config, profile, audit, poll_seconds,
                *, now=utcnow, sleep=time.sleep):
    models = [OptionRoll(session, config, profile, direction, audit) for direction in directions]
    cutoff = models[0].order_cutoff
    began = now()
    last_bar = session.opened
    spot = session.previous_close
    bar_count = quote_count = feed_errors = 0
    bars_retry_at = quotes_retry_at = began
    outcome = "completed"

    def retry_at(error):
        nonlocal feed_errors, outcome
        feed_errors += 1
        observed = now()
        audit({"event": "feed_error", "observed_at": observed.isoformat(), "detail": str(error)})
        if not error.retryable:
            outcome = "feed_unavailable"
            raise error
        return observed + timedelta(seconds=15)

    try:
        while (observed := now()) < cutoff:
            # Older bars seed indicators. Entries use only new bars that
            # closed after the process began and are still timely.
            end = observed.replace(second=0, microsecond=0)
            if end > last_bar and observed >= max(bars_retry_at, end + timedelta(seconds=2)):
                try:
                    for event in sorted(feed.bars(last_bar, end), key=lambda event: event.event_timestamp):
                        if event.event_timestamp <= last_bar:
                            continue
                        warmup = (event.event_timestamp <= began
                                  or event.received_timestamp - event.event_timestamp >= MINUTE
                                  or event.event_timestamp != last_bar + MINUTE)
                        last_bar, spot = event.event_timestamp, event.c
                        bar_count += 1
                        for model in models:
                            seen = model.observation(event, warmup=warmup)
                            if seen:
                                model.on_bar(seen, warmup=warmup)
                except FeedError as error:
                    bars_retry_at = retry_at(error)
            tick = now()
            if tick >= cutoff:
                break
            for model in models:
                model.decide(tick)
            symbols = quote_symbols(contracts, spot, models)
            if symbols and tick >= quotes_retry_at:
                try:
                    for event in feed.quotes(symbols):
                        if event.received_timestamp >= cutoff:
                            continue
                        quote_count += 1
                        for model in models:
                            seen = model.observation(event)
                            if seen:
                                model.on_quote(seen)
                except FeedError as error:
                    quotes_retry_at = retry_at(error)
            sleep(max(0.0, poll_seconds - (now() - observed).total_seconds()))
    except KeyboardInterrupt:
        outcome = "interrupted"
        raise
    except Exception:
        if outcome == "completed":
            outcome = "failed"
        raise
    finally:
        audit({"event": "session_summary", "outcome": outcome, "observed_at": now().isoformat(),
               "bars": bar_count, "quote_snapshots": quote_count, "feed_errors": feed_errors,
               "results": [model.report() for model in models]})


def next_session(feed, now, after=None):
    calendar = feed.calendar(now)
    for previous, current in zip(calendar, calendar[1:]):
        date = current["date"]
        if after is not None and date <= after:
            continue
        opened, closed = (market_time(date, current[key]) for key in ("open", "close"))
        if now < closed - 30 * MINUTE:
            return previous, date, opened, closed
    raise ValueError("No upcoming exchange session in the Alpaca calendar")


def serve(feed, config, args):
    after = None
    directions = DIRECTIONS[:2] if args.direction == "both" else (args.direction,)
    args.log_dir.mkdir(parents=True, exist_ok=True)
    while True:
        try:
            previous, date, opened, closed = next_session(feed, utcnow(), after)
            path = args.log_dir / f"{date}-{args.direction}.jsonl"
            if path.exists():
                print(f"{date} already has a shadow journal at {path}", flush=True)
                after = date
                if args.once:
                    return
                continue
            print(f"Shadow session {date} opens at {opened.astimezone(NEW_YORK):%H:%M %Z}", flush=True)
            # Recheck the calendar after long waits before using a session.
            until_open = (opened - utcnow()).total_seconds()
            if until_open > 300:
                time.sleep(min(until_open - 300, 3600))
                continue
            prior_close = market_time(previous["date"], previous["close"])
            previous_bars = list(feed.bars(prior_close - 15 * MINUTE, prior_close))
            if not previous_bars:
                raise ValueError("Missing SPY bars at the previous regular close")
            last = max(previous_bars, key=lambda bar: bar.event_timestamp)
            contracts = feed.contracts(date, directions)
            if not contracts:
                raise ValueError(f"No standard SPY same-day contracts available for {date}")
            session = Session(opened, closed, last.c)
            time.sleep(max(0.0, (opened - utcnow()).total_seconds()))
            try:
                journal = path.open("x", encoding="utf-8", buffering=1)
            except FileExistsError:
                continue
            with journal:
                def audit(record):
                    print(json.dumps(record), file=journal, flush=True)

                audit({"event": "session_start", "observed_at": utcnow().isoformat(),
                       "mode": "shadow", "direction": args.direction, "profile": args.profile,
                       "open": opened.isoformat(), "close": closed.isoformat(),
                       "previous_close": last.c, "previous_bar_closed_at": last.event_timestamp.isoformat(),
                       "stock_feed": feed.stock_feed, "option_feed": feed.option_feed,
                       "poll_seconds": args.poll_seconds, "config": asdict(config)})
                run_session(feed, session, contracts, directions, config, args.profile, audit,
                            args.poll_seconds)
            print(f"Shadow session recorded in {path}", flush=True)
            after = date
            if args.once:
                return
        except FeedError as error:
            if not error.retryable or args.once:
                raise
            print(str(error), flush=True)
            time.sleep(30)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--direction", choices=DIRECTIONS, default="put")
    parser.add_argument("--profile", choices=PROFILES, default="capped_roll")
    parser.add_argument("--stock-feed", choices=("sip", "iex"), default="sip")
    parser.add_argument("--option-feed", choices=("opra", "indicative"), default="opra")
    parser.add_argument("--log-dir", type=Path, default=Path("logs/spy_0dte_shadow"))
    parser.add_argument("--poll-seconds", type=float, default=3.0)
    parser.add_argument("--cash-account", action="store_true")
    parser.add_argument("--once", action="store_true", help="Exit after one exchange session")
    args = parser.parse_args()
    if not math.isfinite(args.poll_seconds) or args.poll_seconds < 2:
        parser.error("--poll-seconds must be at least 2")
    if Path(".env").is_file():
        from dotenv import load_dotenv
        load_dotenv(".env")
    key = os.getenv("ALPACA_API_KEY") or os.getenv("APCA_API_KEY_ID")
    secret = os.getenv("ALPACA_SECRET_KEY") or os.getenv("APCA_API_SECRET_KEY")
    if not key or not secret:
        parser.error("Set ALPACA_API_KEY and ALPACA_SECRET_KEY for read-only market data")
    feed = AlpacaMarketData(key, secret, args.stock_feed, args.option_feed)
    config = replace(Config(), reuse_sale_proceeds=not args.cash_account)
    try:
        serve(feed, config, args)
    except (FeedError, ValueError) as error:
        parser.exit(1, f"{error}\n")
    except KeyboardInterrupt:
        parser.exit(130)


if __name__ == "__main__":
    main()
