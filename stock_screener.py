"""
Stock Screener — Nasdaq Buy/Strong-Buy universe ranked with the project's
existing data source.

Modeled on shirosaidev/stockbot's screening approach (the `moved`,
`lowtomarket` and `lowtohigh` ranking strategies), but it keeps THIS project's
current data source: price/volume data comes from Alpaca market data and the
exchange filter uses Alpaca assets instead of Yahoo Finance.

Workflow:
  1. Pull the "Buy" / "Strong Buy" universe from the Nasdaq.com screener API.
  2. For each candidate, pull recent daily bars from Alpaca and compute:
       - moved %            (open->close move over MOVED_DAYS)
       - low_to_market $    (recovery from the period low to last price)
       - low_to_high $      (period range)
  3. Filter out symbols outside the price range or not on NYSE/Nasdaq.
  4. Rank by the selected strategy and keep the top MAX_NUM_STOCKS.
  5. `confirm_trending_up()` checks that recent closes rose more often than
     they fell — a pre-trade confirmation, mirroring stockbot's "buy when the
     price is going up" rule.

The screener only PICKS underlyings. It does not place trades. The ranked
tickers feed the existing options selection flow (Telegram / smart_trader /
Schwab scanner).
"""

import os
import sys
import json
import time
import argparse
from datetime import datetime, timedelta
from typing import Dict, List, Optional

import requests

# Defaults mirror shirosaidev/stockbot's config.py.sample so behavior is
# familiar; all are overridable via .env.
DEFAULT_NASDAQ_API_URL = (
    "https://api.nasdaq.com/api/screener/stocks"
    "?tableonly=true&limit=100&marketcap=large|mid|small"
    "&recommendation=strong_buy|buy"
)
DEFAULT_STOCK_MIN_PRICE = 20.0
DEFAULT_STOCK_MAX_PRICE = 100.0
DEFAULT_MAX_NUM_STOCKS = 20
DEFAULT_MOVED_DAYS = 5
# Exchanges Alpaca reports for NYSE / Nasdaq listings.
ALLOWED_EXCHANGES = {'NYSE', 'NASDAQ'}

VALID_STRATEGIES = ('moved', 'lowtomarket', 'lowtohigh')


class StockScreener:
    """Screen the Nasdaq Buy/Strong-Buy list using Alpaca price data."""

    def __init__(self):
        self.load_config()
        self.base_url = (
            "https://paper-api.alpaca.markets" if self.paper
            else "https://api.alpaca.markets"
        )
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
        }
        # Cache asset exchange lookups within a run.
        self._exchange_cache: Dict[str, Optional[str]] = {}

    def load_config(self):
        """Load credentials + screener parameters from .env (manual parse to
        match the rest of this project, which does not use python-dotenv)."""
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        self.api_key = env_vars.get('ALPACA_API_KEY', '')
        self.secret_key = env_vars.get('ALPACA_SECRET_KEY', '')
        self.paper = env_vars.get('ALPACA_PAPER', 'true').lower() == 'true'
        self.alpaca_feed = env_vars.get('SCREENER_ALPACA_FEED', 'iex')

        self.nasdaq_api_url = env_vars.get('NASDAQ_API_URL', DEFAULT_NASDAQ_API_URL)
        self.min_price = float(env_vars.get('STOCK_MIN_PRICE', DEFAULT_STOCK_MIN_PRICE))
        self.max_price = float(env_vars.get('STOCK_MAX_PRICE', DEFAULT_STOCK_MAX_PRICE))
        self.max_num_stocks = int(env_vars.get('MAX_NUM_STOCKS', DEFAULT_MAX_NUM_STOCKS))
        self.moved_days = int(env_vars.get('MOVED_DAYS', DEFAULT_MOVED_DAYS))
        self.default_strategy = env_vars.get('SCREEN_STRATEGY', 'moved').lower()

    # ------------------------------------------------------------------ #
    # Universe: Nasdaq Buy / Strong-Buy screener
    # ------------------------------------------------------------------ #
    def get_nasdaq_buystocks(self) -> List[Dict]:
        """Fetch the Buy/Strong-Buy universe from the Nasdaq screener API.

        Returns a list of {'symbol', 'name', 'lastsale'} dicts. Fails open
        (returns []) on any network/parse error.
        """
        # Nasdaq's API rejects non-browser clients, so send browser-like headers.
        headers = {
            'accept': 'application/json, text/plain, */*',
            'accept-language': 'en-US,en;q=0.9',
            'origin': 'https://www.nasdaq.com',
            'referer': 'https://www.nasdaq.com/',
            'user-agent': (
                'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 '
                '(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36'
            ),
        }
        try:
            resp = requests.get(self.nasdaq_api_url, headers=headers, timeout=20)
        except requests.RequestException as e:
            print(f"[SCREENER] Nasdaq request failed: {e}")
            return []

        if resp.status_code != 200:
            print(f"[SCREENER] Nasdaq API status {resp.status_code}")
            return []

        try:
            data = resp.json()
            rows = data['data']['table']['rows']
        except (ValueError, KeyError, TypeError) as e:
            print(f"[SCREENER] Could not parse Nasdaq response: {e}")
            return []

        stocks = []
        for row in rows:
            symbol = (row.get('symbol') or '').strip().upper()
            if not symbol or '^' in symbol or '/' in symbol:
                continue  # skip indices / odd tickers
            stocks.append({
                'symbol': symbol,
                'name': row.get('name', ''),
                'lastsale': self._parse_money(row.get('lastsale')),
            })
        return stocks

    @staticmethod
    def _parse_money(value) -> Optional[float]:
        """Parse Nasdaq '$123.45' style strings into floats."""
        if value is None:
            return None
        try:
            return float(str(value).replace('$', '').replace(',', '').strip())
        except (ValueError, TypeError):
            return None

    # ------------------------------------------------------------------ #
    # Price data: Alpaca (replaces stockbot's Yahoo Finance dependency)
    # ------------------------------------------------------------------ #
    def get_daily_bars(self, symbol: str, days: int) -> List[Dict]:
        """Recent daily OHLCV bars for a symbol from Alpaca market data."""
        end = datetime.now()
        # Pad the window for weekends/holidays so we still get `days` bars.
        start = end - timedelta(days=days * 2 + 10)
        try:
            resp = requests.get(
                f"{self.data_url}/v2/stocks/{symbol}/bars",
                headers=self.headers,
                params={
                    'timeframe': '1Day',
                    'start': start.strftime('%Y-%m-%d'),
                    'end': end.strftime('%Y-%m-%d'),
                    'limit': days + 15,
                    'feed': self.alpaca_feed,
                    'adjustment': 'raw',
                },
                timeout=20,
            )
        except requests.RequestException as e:
            print(f"[SCREENER] Bars request failed for {symbol}: {e}")
            return []

        if resp.status_code != 200:
            return []

        bars = resp.json().get('bars', []) or []
        # Keep only the most recent `days` bars.
        return bars[-days:] if len(bars) > days else bars

    def get_exchange(self, symbol: str) -> Optional[str]:
        """Look up the listing exchange for a symbol via Alpaca assets."""
        if symbol in self._exchange_cache:
            return self._exchange_cache[symbol]
        exchange = None
        try:
            resp = requests.get(
                f"{self.base_url}/v2/assets/{symbol}",
                headers=self.headers,
                timeout=15,
            )
            if resp.status_code == 200:
                exchange = resp.json().get('exchange')
        except requests.RequestException as e:
            print(f"[SCREENER] Asset lookup failed for {symbol}: {e}")
        self._exchange_cache[symbol] = exchange
        return exchange

    # ------------------------------------------------------------------ #
    # Screening
    # ------------------------------------------------------------------ #
    def _evaluate(self, symbol: str, name: str) -> Optional[Dict]:
        """Compute ranking metrics for one candidate, or None to skip."""
        bars = self.get_daily_bars(symbol, self.moved_days)
        if len(bars) < 2:
            return None

        try:
            opens = [float(b['o']) for b in bars]
            highs = [float(b['h']) for b in bars]
            lows = [float(b['l']) for b in bars]
            closes = [float(b['c']) for b in bars]
            volume = float(bars[-1].get('v', 0) or 0)
        except (KeyError, TypeError, ValueError):
            return None

        market_price = closes[-1]
        if market_price < self.min_price or market_price > self.max_price:
            return None

        # Exchange filter (NYSE / Nasdaq only), matching stockbot's NYQ/NMS rule.
        exchange = self.get_exchange(symbol)
        if exchange is not None and exchange not in ALLOWED_EXCHANGES:
            return None

        # moved %: open(first)->close(last) over the window.
        moved = round((closes[-1] - opens[0]) / opens[0] * 100, 3)

        period_low = min(lows)
        period_high = max(highs)
        change_low_to_market = round(market_price - period_low, 3)
        change_low_to_high = round(period_high - period_low, 3)

        return {
            'symbol': symbol,
            'company': name,
            'market_price': round(market_price, 2),
            'low': round(period_low, 2),
            'high': round(period_high, 2),
            'volume': int(volume),
            'moved': moved,
            'change_low_to_market': change_low_to_market,
            'change_low_to_high': change_low_to_high,
            'exchange': exchange,
        }

    def screen(self, strategy: Optional[str] = None,
               limit: Optional[int] = None) -> List[Dict]:
        """Run the full screen and return ranked picks.

        Args:
            strategy: 'moved' | 'lowtomarket' | 'lowtohigh' (defaults to .env).
            limit: max picks to return (defaults to MAX_NUM_STOCKS).
        """
        strategy = (strategy or self.default_strategy).lower()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {VALID_STRATEGIES}."
            )
        limit = limit or self.max_num_stocks

        universe = self.get_nasdaq_buystocks()
        if not universe:
            print("[SCREENER] No candidates returned from Nasdaq screener.")
            return []

        print(f"[SCREENER] Evaluating {len(universe)} Buy/Strong-Buy candidates "
              f"(strategy={strategy}, price ${self.min_price:.0f}-${self.max_price:.0f})...")

        evaluated = []
        for stock in universe:
            info = self._evaluate(stock['symbol'], stock['name'])
            if info:
                evaluated.append(info)

        sort_key = {
            'moved': 'moved',
            'lowtomarket': 'change_low_to_market',
            'lowtohigh': 'change_low_to_high',
        }[strategy]
        ranked = sorted(evaluated, key=lambda i: i[sort_key], reverse=True)

        print(f"[SCREENER] {len(ranked)} passed filters; returning top {limit}.")
        return ranked[:limit]

    def confirm_trending_up(self, symbol: str, lookback: int = 5) -> bool:
        """Confirm the price is rising more than falling over recent closes.

        Mirrors stockbot's pre-buy rule ("buy when it's going up"): counts the
        up vs down day-over-day moves in the last `lookback` closes.
        """
        bars = self.get_daily_bars(symbol, lookback + 1)
        closes = [float(b['c']) for b in bars if 'c' in b]
        if len(closes) < 2:
            return False
        ups = downs = 0
        for prev, cur in zip(closes, closes[1:]):
            if cur > prev:
                ups += 1
            elif cur < prev:
                downs += 1
        return ups > downs


def screen_tickers(strategy: Optional[str] = None,
                   limit: Optional[int] = None) -> List[str]:
    """Convenience helper: return just the ranked ticker symbols."""
    return [pick['symbol'] for pick in StockScreener().screen(strategy, limit)]


def _format_table(picks: List[Dict]) -> str:
    if not picks:
        return "No stocks passed the screen."
    header = (f"{'#':>2}  {'SYMBOL':<8}{'PRICE':>9}{'MOVED%':>9}"
              f"{'LOW>MKT':>9}{'LOW>HIGH':>10}  EXCH")
    lines = [header, '-' * len(header)]
    for i, p in enumerate(picks, 1):
        lines.append(
            f"{i:>2}  {p['symbol']:<8}{p['market_price']:>9.2f}{p['moved']:>9.2f}"
            f"{p['change_low_to_market']:>9.2f}{p['change_low_to_high']:>10.2f}"
            f"  {p['exchange'] or '?'}"
        )
    return '\n'.join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Screen Nasdaq Buy/Strong-Buy stocks using Alpaca price data."
    )
    parser.add_argument('-s', '--strategy', default=None,
                        choices=VALID_STRATEGIES,
                        help="ranking strategy (default from .env SCREEN_STRATEGY or 'moved')")
    parser.add_argument('-n', '--limit', type=int, default=None,
                        help="max number of picks (default from .env MAX_NUM_STOCKS)")
    parser.add_argument('--write-tickers', action='store_true',
                        help="write the picks into supported_tickers.json (used by the Telegram bot)")
    parser.add_argument('--json', action='store_true',
                        help="output full pick details as JSON")
    args = parser.parse_args()

    screener = StockScreener()
    if not screener.api_key or not screener.secret_key:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)

    picks = screener.screen(strategy=args.strategy, limit=args.limit)

    if args.json:
        print(json.dumps(picks, indent=2))
    else:
        print('\n' + _format_table(picks) + '\n')

    if args.write_tickers and picks:
        symbols = [p['symbol'] for p in picks]
        tickers_file = 'supported_tickers.json'
        data = {
            'tickers': sorted(set(symbols)),
            'last_updated': datetime.now().isoformat(),
            'source': f"stock_screener:{args.strategy or screener.default_strategy}",
        }
        with open(tickers_file, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Wrote {len(symbols)} screened tickers to {tickers_file}")


if __name__ == '__main__':
    main()
