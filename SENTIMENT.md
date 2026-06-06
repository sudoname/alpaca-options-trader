# Market Fear & Greed Sentiment Module

A self-contained sentiment layer that measures broad market **Fear & Greed** and
uses it as a **risk filter** on top of the existing trading strategies. It is
designed to *fail open*: any failure to obtain sentiment data results in trades
proceeding **unchanged** — sentiment never crashes or silently blocks the bot.

## What it does

1. Computes a 0–100 Fear & Greed score from two independent sources:
   - **Custom score (PRIMARY)** — built from market data we fetch ourselves.
     Keeps working even when CNN is down.
   - **CNN Fear & Greed Index (comparison/validation)** — scraped from CNN's
     unofficial data-viz endpoint.
2. Classifies the score: `Extreme Fear / Fear / Neutral / Greed / Extreme Greed`.
3. Caches the result (default 15 min) to avoid hammering data sources.
4. Adjusts trade risk (position size / allow-block) before orders are placed.

## Data sources

### CNN unofficial scraper (`cnn_fear_greed.py`)
- Endpoint: `https://production.dataviz.cnn.io/index/fearandgreed/graphdata`
- **Unofficial & undocumented.** Used by CNN's web widget. It can change shape,
  rate-limit, require browser-like headers, or disappear without notice.
- Defensive by design: browser `User-Agent`/`Referer` headers, request timeout,
  HTTP-status check, JSON-validity check, broad exception handling.
- **Never raises.** Always returns a normalized dict:
  ```python
  {"source": "cnn_unofficial", "status": "available"|"error",
   "score": 0-100|None, "classification": "...", "timestamp": "...", ...}
  ```

### Custom score (`custom_fear_greed.py`)
Reimplements CNN's index from data we control. Two providers are included:
`SchwabMarketDataProvider` (Schwab price-history API, used by the SPY strategies)
and `AlpacaMarketDataProvider` (Alpaca bars API, used by `smart_trader.py`).
Seven components:

| # | Component            | Signal                                            | Source            |
|---|----------------------|---------------------------------------------------|-------------------|
| 1 | Market momentum      | SPY vs its 125-day MA (above = greed)             | SPY history       |
| 2 | Market volatility    | VIX percentile (high VIX = fear, **inverse**)     | $VIX.X history    |
| 3 | Put/call ratio       | high ratio = fear (**inverse**)                   | provider (opt.)   |
| 4 | Junk bond demand     | HYG vs LQD relative strength (strong HYG = greed) | HYG, LQD history  |
| 5 | Safe-haven demand    | SPY vs bonds (TLT/IEF); bonds winning = fear      | SPY, TLT/IEF      |
| 6 | Market breadth       | advancers vs decliners                            | provider (opt.)   |
| 7 | New highs / new lows  | 52-week highs vs lows                             | provider (opt.)   |

**Scoring** uses rolling percentiles:

```python
percentile_score(series, current_value, inverse=False)
# percentile = % of historical values below current; inverse => 100 - percentile
```

`inverse=True` is used for VIX and put/call (where *high = fear*).

**Missing data is never faked.** Components 3, 6 and 7 have no readily available
feed from the broker API, so the default provider returns `None` for them and
they are reported under `unavailable_components` and **excluded from the
average**. The final custom score is the simple mean of the *available*
component scores. If fewer than `SENTIMENT_MIN_COMPONENTS` are available, the
custom score reports `status="error"` (and the service falls back to CNN).

To wire real put/call, breadth, or highs/lows feeds later, subclass
`MarketDataProvider` and implement `get_put_call_ratio_series`,
`get_market_breadth`, and/or `get_new_highs_lows`.

### Classification bands
| Score   | Classification |
|---------|----------------|
| 0–25    | Extreme Fear   |
| 26–45   | Fear           |
| 46–55   | Neutral        |
| 56–75   | Greed          |
| 76–100  | Extreme Greed  |

## Sentiment service (`sentiment_service.py`)
The single entry point. Returns:
```python
{"cnn_score": {...} | None,
 "custom_score": {...} | None,
 "primary_score": {...},        # custom by default; CNN as fallback
 "primary_source": "custom"|"cnn"|None,
 "timestamp": "..."}
```
Custom is primary; CNN is only for comparison/validation. If the configured
primary is unavailable, the service falls back to the other source.

### Caching (`sentiment_cache.py`)
- Fresh cache (within `SENTIMENT_CACHE_MINUTES`) → returned immediately.
- Otherwise recompute; on success, cache and return.
- If a refresh fails but a **stale** cache exists, the stale value is returned
  with a warning (`from_stale_cache=True`).

## Trading impact (`sentiment_filter.py`)
Single integration function:

```python
adjust_trade_risk_by_sentiment(trade_candidate, sentiment) -> decision
```

`trade_candidate` carries `size` (contracts) and optional `confidence`/`direction`.
The returned `decision`:

```python
{"allowed": True/False,
 "original_size": 100, "adjusted_size": 75, "size_multiplier": 0.75,
 "reason": "Fear sentiment detected; reducing position size by 25%",
 "classification": "Fear", "score": 38.0, "confidence_floor": None}
```

Policy:

| Sentiment      | Behavior                                                          |
|----------------|-------------------------------------------------------------------|
| Extreme Fear   | Block aggressive longs (confidence < 80%); else cut size by 50%.  |
| Fear           | Reduce size by 25%.                                               |
| Neutral        | Normal sizing.                                                    |
| Greed          | Normal sizing.                                                    |
| Extreme Greed  | Trim size by 25% to avoid over-leveraging into euphoria.          |
| Unavailable    | **Pass through unchanged (fail-open).**                          |

### Where it's wired in
Guarded, fail-open hooks (mirroring the RL shadow hooks):
- `spy_1dte_strategy.py` — after direction analysis; can skip the day's trade.
- `spy_hybrid_strategy.py` — same pattern.
- `smart_trader.py` — scales `order_quantity` in `place_order_with_stops`,
  using an **`AlpacaMarketDataProvider`** so the full custom score is computed
  here too (falls back to CNN if Alpaca data is unavailable).

Each hook is wrapped in `try/except` and only runs when `SENTIMENT_ENABLED=true`.

## Configuration (`.env`)
```ini
SENTIMENT_ENABLED=true          # master on/off
SENTIMENT_CACHE_MINUTES=15      # cache TTL
SENTIMENT_USE_CNN=true          # fetch CNN index
SENTIMENT_USE_CUSTOM=true       # compute custom score (PRIMARY)
SENTIMENT_PRIMARY_SOURCE=custom # custom | cnn
SENTIMENT_MIN_COMPONENTS=3      # min components for a valid custom score
# Optional overrides:
# SENTIMENT_CNN_URL=...
# SENTIMENT_CNN_TIMEOUT=10
# SENTIMENT_HISTORY_DAYS=400
# SENTIMENT_MOMENTUM_WINDOW=125
# SENTIMENT_CACHE_FILE=sentiment_cache.json
# SENTIMENT_ALPACA_FEED=iex       # smart_trader bars feed: iex (free) | sip
```

### Alpaca provider notes (`alpaca_provider.py`)
`smart_trader.py` uses Alpaca, not Schwab, so it builds an
`AlpacaMarketDataProvider(data_url, headers, feed="iex")` for the custom score.
Caveats:
- The free **IEX** feed has shallower history, so the 125-day momentum component
  may be unavailable on the free tier (reported as such, never faked). Set
  `SENTIMENT_ALPACA_FEED=sip` if you have a paid data subscription.
- Alpaca's stock feed has no index data, so the VIX symbol `$VIX.X` is mapped to
  the **VIXY** ETF (a VIX short-term futures proxy) for the volatility component.

## Logging
The module logs (via the stdlib `logging` module) the score, classification,
source, and unavailable components, and the strategies print the sentiment
summary and any trade adjustment (e.g. `[SENTIMENT] Fear sentiment detected ...`).

## Usage
```python
from sentiment import SentimentService, SchwabMarketDataProvider, \
    adjust_trade_risk_by_sentiment, summarize_for_log

service = SentimentService(SchwabMarketDataProvider(client))
sentiment = service.get_sentiment()
print(summarize_for_log(sentiment))

decision = adjust_trade_risk_by_sentiment(
    {"size": 4, "confidence": 72, "direction": "CALL"}, sentiment)
if decision["allowed"]:
    qty = decision["adjusted_size"]
```

## Tests
```bash
python -m unittest test_sentiment -v
```
35 unit tests cover the percentile helper, classification boundaries, CNN
success/failure/timeout/invalid-JSON, custom score (greed/fear/missing
components/insufficient data), cache (fresh/stale/corrupt), service
orchestration (primary selection, CNN fallback, stale-cache fallback), and the
trade filter policy. **All external calls are mocked — no live internet.**

## CNN scraping limitations
The CNN endpoint is unofficial and may break at any time. When it does, the
custom score (the primary signal) continues to operate. CNN is treated purely as
a comparison/validation reference and is never required for the bot to function.
