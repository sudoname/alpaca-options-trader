# Schwab Options Trading System - Setup & Usage Guide

## Overview
This system scans and analyzes stock options using live Schwab API data, with automated scoring, Greeks analysis, and Telegram bot integration.

## Components

### 1. **schwab_option_scanner.py**
Advanced options scanner that:
- Scans multiple tickers for CALL or PUT options
- Filters by Delta (0.35-0.65), IV (max 50%), and days to expiration (20-60)
- Calculates comprehensive scores based on:
  - Delta (30 points max) - prefers 0.4-0.6 ATM range
  - Gamma (15 points max) - higher is better
  - Theta (15 points max) - lower time decay is better
  - IV (15 points max) - prefers 20-40% moderate IV
  - Liquidity (20 points max) - volume & open interest
  - Spread (10 points max) - tighter bid-ask spreads
- Returns sorted list of options with scores 0-100

### 2. **schwab_trader.py**
Main trading bot that:
- Uses scanner to find best trade opportunities
- Performs detailed analysis with confidence scoring
- Executes trades in DRY RUN mode (no real money)
- Logs all trades to `schwab_trades.json`
- Analyzes options based on:
  - Delta analysis (optimal 0.45-0.55)
  - IV analysis (moderate 20-35% preferred)
  - Liquidity (volume >100, OI >1000)
  - Theta/time decay (< 0.15 preferred)
  - Days to expiration (30-45 days optimal)

### 3. **schwab_telegram_bot.py**
Telegram bot interface with commands:
- `/start` - Welcome message and help
- `/scan` - Scan for best CALL options
- `/scan_puts` - Scan for best PUT options
- `/analyze TICKER` - Analyze specific ticker
- `/trade` - Execute best trade (DRY RUN)
- `/status` - Bot status and info
- `/tickers` - Show/update ticker list

## Setup

### 1. Environment Variables (.env)
```
SCHWAB_APP_KEY=egi2ATAEJd8Dhs5RLM0SoPfoNkl9j6ht
SCHWAB_APP_SECRET=k5zcNhvSfAc4NmQG
SCHWAB_CALLBACK_URL=https://127.0.0.1:5000/callback
SCHWAB_TOKEN_FILE=schwab_tokens.json
TELEGRAM_BOT_TOKEN=7931252176:AAGwp2PVlaN7rBtINfzUd8lwaS3Db8iUsi0
```

### 2. Install Dependencies
```bash
pip install schwab-py python-telegram-bot python-dotenv multiprocess
```

### 3. Authenticate Schwab
Run the authentication script (only needed once):
```bash
python auth_schwab_simple.py
```
This will:
- Open browser for Schwab OAuth login
- Save tokens to `schwab_tokens.json`
- Test connection with AAPL quote

## Usage

### Option 1: Command Line Scanner
```bash
python schwab_option_scanner.py
```
Scans AAPL and saves results to timestamped files.

### Option 2: Command Line Trader
```bash
python schwab_trader.py
```
Scans multiple tickers (AAPL, MSFT, GOOGL, NVDA, TSLA), analyzes, and simulates trades.

### Option 3: Telegram Bot
```bash
python schwab_telegram_bot.py
```
Start the Telegram bot and interact via:
- Chat with bot at your Telegram account
- Use commands like `/scan` or `/analyze AAPL`

### Option 4: Integration Test
```bash
python test_integration.py
```
Runs full system test verifying all components.

## Test Results (2025-10-10)

Integration test output:
```
[TEST 1] Scanner initialized - OK
[TEST 2] Trader initialized - OK
[TEST 3] Scanning AAPL
  - Found 19 options
  - Best: AAPL $245 CALL, Score: 89.6/100
  - Greeks: Delta=0.536, IV=27.8%
[TEST 4] Analysis
  - Recommendation: BUY
  - Confidence: 100%
  - Risk Level: MEDIUM
[TEST 5] Multi-ticker scan (AAPL, MSFT, GOOGL)
  - AAPL: 19 options
  - MSFT: 37 options
  - GOOGL: 24 options
  - Best: AAPL $245 CALL, Score: 89.6/100

ALL TESTS PASSED - SYSTEM READY
```

## Key Features

✅ **Live Schwab Data** - Real-time options chains with Greeks
✅ **Automated Scoring** - Weighted algorithm for option ranking
✅ **Greeks Analysis** - Delta, Gamma, Theta, Vega evaluation
✅ **Multi-ticker Support** - Scan multiple stocks simultaneously
✅ **Risk Assessment** - Confidence-based trading decisions
✅ **Telegram Integration** - Remote bot control via messages
✅ **DRY RUN Mode** - Safe testing without real money

## Default Settings

**Scanner Filters:**
- Min Days to Expiration: 20
- Max Days to Expiration: 60
- Min Delta: 0.35
- Max Delta: 0.65
- Max IV: 50%
- Strike Count: 10 (5 above/below ATM)

**Trader Settings:**
- Budget: $2,000 per trade
- Min Confidence: 60% to execute
- Default Mode: DRY RUN (no real trades)

**Default Tickers:**
AAPL, MSFT, GOOGL, NVDA, AMD, TSLA, META, AMZN

## Files Generated

- `schwab_tokens.json` - OAuth tokens (auto-refreshed)
- `schwab_trades.json` - Trade log
- `option_scan_YYYYMMDD_HHMMSS.txt` - Scan reports
- `option_scan_YYYYMMDD_HHMMSS.json` - Scan data

## Important Notes

⚠️ **DRY RUN MODE** - System currently runs in simulation mode only. No real trades are executed.

⚠️ **Token Refresh** - Schwab tokens expire after 7 days. Re-run `auth_schwab_simple.py` if you get authentication errors.

⚠️ **Rate Limits** - Schwab API has rate limits. The scanner paces requests to avoid hitting limits.

⚠️ **Market Hours** - Best results during market hours (9:30 AM - 4:00 PM ET). After-hours data may be limited.

## Next Steps

To enable **LIVE TRADING** (not implemented yet):
1. Implement order placement logic in `schwab_trader.py` (line 220)
2. Set `dry_run=False` when initializing SchwabOptionsTrader
3. Add risk management and position sizing
4. Add order confirmation and monitoring
5. Test thoroughly with small amounts first

## Support

For issues or questions:
- Check logs for error details
- Verify token file exists and is valid
- Test connection with `python auth_schwab_simple.py`
- Run integration test: `python test_integration.py`
