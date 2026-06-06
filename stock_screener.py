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
# Optional EV-aware ranking. Off by default; when selected it enriches each
# pick with a point-in-time regime label and an `ev_score`, and ranks by it.
VALID_SCORES = ('ev',)


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
    def _bars_as_dicts(self, symbol: str, market_view=None) -> List[Dict]:
        """Source OHLCV bars either from Alpaca (live) or a MarketView (tests).

        When `market_view` is provided, bars come from its point-in-time
        `daily_bars` (no network, deterministic) and are normalized to the same
        dict shape the live path returns.
        """
        if market_view is not None:
            out = []
            for b in market_view.daily_bars(symbol, self.moved_days):
                out.append({'o': b.o, 'h': b.h, 'l': b.l, 'c': b.c, 'v': b.v})
            return out
        return self.get_daily_bars(symbol, self.moved_days)

    @staticmethod
    def _ev_score(info: Dict, regime_label: Dict) -> float:
        """Heuristic EV-aware ranking score (reports/ranking only).

        Rewards an upward open->close move confirmed by an up trend in a
        non-volatile regime, scaled by where price sits in its recent range.
        This is a transparent ranking heuristic; it never sizes or places a
        trade and is only used when `--score ev` is explicitly chosen.
        """
        moved = info['moved']
        rng = info['change_low_to_high'] or 1e-9
        pos_in_range = max(0.0, min(1.0, info['change_low_to_market'] / rng))
        regime_mult = {'trending': 1.25, 'ranging': 1.0, 'volatile': 0.6}.get(
            regime_label.get('regime'), 1.0)
        trend = regime_label.get('trend')
        trend_mult = 1.15 if trend == 'up' else (0.85 if trend == 'down' else 1.0)
        return round(moved * regime_mult * trend_mult * (0.5 + 0.5 * pos_in_range), 3)

    def _evaluate(self, symbol: str, name: str,
                  score: Optional[str] = None, market_view=None) -> Optional[Dict]:
        """Compute ranking metrics for one candidate, or None to skip.

        With `score='ev'` the result is enriched with a point-in-time `regime`
        label and an `ev_score`; otherwise the output is unchanged.
        """
        bars = self._bars_as_dicts(symbol, market_view)
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
        # Skipped in MarketView (offline/test) mode where there is no asset API.
        exchange = None if market_view is not None else self.get_exchange(symbol)
        if exchange is not None and exchange not in ALLOWED_EXCHANGES:
            return None

        # moved %: open(first)->close(last) over the window.
        moved = round((closes[-1] - opens[0]) / opens[0] * 100, 3)

        period_low = min(lows)
        period_high = max(highs)
        change_low_to_market = round(market_price - period_low, 3)
        change_low_to_high = round(period_high - period_low, 3)

        info = {
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

        if score == 'ev':
            from datetime import datetime, time as _time
            from regime import detect_regime
            from market_view import HistoricalMarketView, make_bar
            mv = market_view
            if mv is None:
                # Build a point-in-time view from the fetched bars (as_of = last
                # known session close) so the regime label cannot peek ahead.
                today = datetime.now().date()
                mv_bars = []
                for i, b in enumerate(bars):
                    d = (today - timedelta(days=len(bars) - 1 - i)).strftime('%Y-%m-%d')
                    mv_bars.append(make_bar(d, b['o'], b['h'], b['l'], b['c'],
                                            b.get('v', 0)))
                as_of = datetime.combine(today, _time(16, 0))
                mv = HistoricalMarketView(as_of, daily={symbol: mv_bars})
            regime_label = detect_regime(mv, symbol, vol_lookback=self.moved_days)
            info['regime'] = regime_label['regime']
            info['trend'] = regime_label['trend']
            info['realized_vol'] = regime_label['realized_vol']
            info['ev_score'] = self._ev_score(info, regime_label)

        return info

    def screen(self, strategy: Optional[str] = None,
               limit: Optional[int] = None,
               score: Optional[str] = None,
               universe: Optional[List[Dict]] = None) -> List[Dict]:
        """Run the full screen and return ranked picks.

        Args:
            strategy: 'moved' | 'lowtomarket' | 'lowtohigh' (defaults to .env).
            limit: max picks to return (defaults to MAX_NUM_STOCKS).
            score: optional 'ev' to enrich picks with regime + ev_score and rank
                by ev_score instead of the base strategy. Default (None) leaves
                rankings byte-identical to the original behavior.
            universe: optional pre-supplied candidate list (used by tests to
                avoid the Nasdaq network call).
        """
        strategy = (strategy or self.default_strategy).lower()
        if strategy not in VALID_STRATEGIES:
            raise ValueError(
                f"Unknown strategy '{strategy}'. Choose from {VALID_STRATEGIES}."
            )
        if score is not None and score not in VALID_SCORES:
            raise ValueError(
                f"Unknown score '{score}'. Choose from {VALID_SCORES}."
            )
        limit = limit or self.max_num_stocks

        if universe is None:
            universe = self.get_nasdaq_buystocks()
        if not universe:
            print("[SCREENER] No candidates returned from Nasdaq screener.")
            return []

        rank_label = score or strategy
        print(f"[SCREENER] Evaluating {len(universe)} Buy/Strong-Buy candidates "
              f"(rank={rank_label}, price ${self.min_price:.0f}-${self.max_price:.0f})...")

        evaluated = []
        for stock in universe:
            info = self._evaluate(stock['symbol'], stock['name'], score=score)
            if info:
                evaluated.append(info)

        if score == 'ev':
            sort_key = 'ev_score'
        else:
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
    has_ev = 'ev_score' in picks[0]
    header = (f"{'#':>2}  {'SYMBOL':<8}{'PRICE':>9}{'MOVED%':>9}"
              f"{'LOW>MKT':>9}{'LOW>HIGH':>10}")
    if has_ev:
        header += f"{'REGIME':>10}{'TREND':>6}{'EV':>9}"
    header += "  EXCH"
    lines = [header, '-' * len(header)]
    for i, p in enumerate(picks, 1):
        row = (
            f"{i:>2}  {p['symbol']:<8}{p['market_price']:>9.2f}{p['moved']:>9.2f}"
            f"{p['change_low_to_market']:>9.2f}{p['change_low_to_high']:>10.2f}"
        )
        if has_ev:
            row += f"{p['regime']:>10}{p['trend']:>6}{p['ev_score']:>9.2f}"
        row += f"  {p['exchange'] or '?'}"
        lines.append(row)
    return '\n'.join(lines)


# --------------------------------------------------------------------------- #
# Self-test (no creds, no network)
# --------------------------------------------------------------------------- #
def _self_test() -> int:
    from datetime import datetime as _dt
    from market_view import HistoricalMarketView, make_bar

    ok = True
    sc = StockScreener()
    sc.min_price, sc.max_price = 1.0, 10_000.0  # don't price-filter the fixtures
    sc.moved_days = 5

    # Fixture: a steady riser (UP) and a choppy/volatile name (CHOP).
    up_bars = [make_bar(f"2026-01-{i+1:02d}", 100 + i, 100.6 + i, 99.5 + i, 100.6 + i)
               for i in range(6)]
    chop_bars = []
    base = 100.0
    for i in range(6):
        c = base * (1.08 if i % 2 == 0 else 0.93)
        chop_bars.append(make_bar(f"2026-01-{i+1:02d}", base, max(base, c), min(base, c), c))
        base = c
    mv_up = HistoricalMarketView(_dt(2026, 1, 6, 16, 0), daily={"UP": up_bars})
    mv_chop = HistoricalMarketView(_dt(2026, 1, 6, 16, 0), daily={"CHOP": chop_bars})

    # Default (no score) output keeps the original keys exactly — no ev fields.
    base_info = sc._evaluate("UP", "Up Co", market_view=mv_up)
    expected_keys = {'symbol', 'company', 'market_price', 'low', 'high', 'volume',
                     'moved', 'change_low_to_market', 'change_low_to_high', 'exchange'}
    if base_info is None or set(base_info.keys()) != expected_keys:
        print("FAIL: default _evaluate keys changed", base_info); ok = False

    # EV scoring enriches with regime/trend/ev_score.
    ev_up = sc._evaluate("UP", "Up Co", score="ev", market_view=mv_up)
    for k in ("regime", "trend", "ev_score"):
        if k not in ev_up:
            print(f"FAIL: ev score missing '{k}'", ev_up); ok = False
    if ev_up and ev_up["trend"] != "up":
        print("FAIL: steady riser should be trend=up", ev_up); ok = False

    ev_chop = sc._evaluate("CHOP", "Chop Co", score="ev", market_view=mv_chop)
    if ev_chop and ev_chop["regime"] != "volatile":
        print("FAIL: choppy name should be volatile", ev_chop); ok = False

    # A clean up-trend should out-score a volatile chop with similar gross move.
    if ev_up and ev_chop and not (ev_up["ev_score"] > ev_chop["ev_score"]):
        print("FAIL: trending name should out-score volatile",
              ev_up["ev_score"], ev_chop["ev_score"]); ok = False

    # Point-in-time: bars sourced through the MarketView never exceed as_of.
    if any(rec["ts"] > mv_up.as_of for rec in mv_up.audit):
        print("FAIL: screener read a bar stamped after as_of"); ok = False

    print("stock_screener self-test:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


def main():
    parser = argparse.ArgumentParser(
        description="Screen Nasdaq Buy/Strong-Buy stocks using Alpaca price data."
    )
    parser.add_argument('-s', '--strategy', default=None,
                        choices=VALID_STRATEGIES,
                        help="ranking strategy (default from .env SCREEN_STRATEGY or 'moved')")
    parser.add_argument('-n', '--limit', type=int, default=None,
                        help="max number of picks (default from .env MAX_NUM_STOCKS)")
    parser.add_argument('--score', default=None, choices=VALID_SCORES,
                        help="optional EV-aware scoring; enriches picks with regime + ev_score "
                             "and ranks by it (default off -> rankings unchanged)")
    parser.add_argument('--write-tickers', action='store_true',
                        help="write the picks into supported_tickers.json (used by the Telegram bot)")
    parser.add_argument('--json', action='store_true',
                        help="output full pick details as JSON")
    parser.add_argument('--selftest', action='store_true',
                        help="run the no-creds self-test and exit")
    args = parser.parse_args()

    if args.selftest:
        sys.exit(_self_test())

    screener = StockScreener()
    if not screener.api_key or not screener.secret_key:
        print("ERROR: ALPACA_API_KEY / ALPACA_SECRET_KEY missing from .env")
        sys.exit(1)

    picks = screener.screen(strategy=args.strategy, limit=args.limit, score=args.score)

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
