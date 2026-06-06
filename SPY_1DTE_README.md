# SPY 1DTE Options Strategy

Automated daily SPY options trading strategy that:
- Runs at **9:30 AM EST** every weekday
- Trades **1DTE (1 Day To Expiration)** options
- Analyzes market direction using multiple factors
- Executes **CALL or PUT** based on market analysis
- Automatically closes at **20% profit**

## Features

### Market Analysis
The strategy analyzes:
- **SPY momentum** (pre-market and intraday movement)
- **VIX levels** (fear/greed indicator)
- **Gap analysis** (overnight gaps)
- **Futures performance** (ES futures)

### Option Selection Criteria
- **DTE**: 0-1 days to expiration
- **Moneyness**: Out of The Money (OTM)
- **Premium**: Less than $10 per contract
- **Delta**: Target ~0.35 for optimal risk/reward

### Exit Strategy
- **Profit Target**: 20% gain
- **Time-based**: Closes before market close (3:45 PM)
- **Monitoring**: Checks every 30 seconds

## Files

1. **spy_1dte_strategy.py** - Main strategy logic
2. **run_spy_1dte_daily.py** - Scheduler (runs at 9:30 AM daily)
3. **run_spy_1dte_now.py** - Manual execution
4. **spy_1dte_trades.json** - Trade log

## Usage

### Automatic (Recommended)
Run the scheduler to execute daily at 9:30 AM:
```bash
python run_spy_1dte_daily.py
```

This will:
- Start the scheduler
- Run every weekday at 9:30 AM EST
- Skip weekends automatically
- Run in background until stopped (Ctrl+C)

### Manual
Execute the strategy immediately:
```bash
python run_spy_1dte_now.py
```

## Trade Log

All trades are logged to `spy_1dte_trades.json` with:
- Entry time and price
- Market analysis details
- Exit time and price
- Profit/loss
- Order IDs

## Example Trade Flow

**9:30 AM** - Market Opens
```
[ANALYSIS] Analyzing market direction...
[SPY] Price: $450.25
[SPY] Change: +0.35%
[VIX] Level: 14.2
[SIGNAL] Positive momentum (+1)
[SIGNAL] Low VIX - Complacency (+1 bullish)
[DECISION] CALL with 67% confidence

[SCAN] Scanning for 1DTE SPY CALL options...
[FOUND] Best option:
  Strike: $451.00
  Premium: $0.85
  Delta: 0.35
  DTE: 1

[EXECUTE] Placing order...
[SUCCESS] Order placed! Order ID: 123456789
```

**10:15 AM** - Monitoring
```
[MONITOR] Monitoring for 20% profit target...
[TARGET] Entry: $0.85 -> Target: $1.02
[CHECK] Current bid: $0.95 | P/L: 11.8%
```

**11:30 AM** - Exit
```
[CHECK] Current bid: $1.05 | P/L: 23.5%
[TARGET HIT] 20% profit achieved! Closing position...
[SUCCESS] Position closed!
[PROFIT] $20.00 (23.5%)
```

## Risk Management

- **Max Risk**: Premium paid (typically $50-$100 per trade)
- **Max Contracts**: 1 per day
- **Account**: Uses account ...879 ($500 balance)
- **Daily Limit**: 1 trade only

## Configuration

Edit `.env` to customize:
```
# Trading mode
DRY_RUN=false  # Set to true for paper trading

# Account to use
SCHWAB_ACCOUNT_HASH=your_account_hash
```

## Monitoring

View trade history:
```bash
python -m json.tool spy_1dte_trades.json
```

## Requirements

```bash
pip install schedule pytz
```

## Safety Features

1. **Weekday Only**: Automatically skips weekends
2. **One Trade Per Day**: Prevents over-trading
3. **Premium Cap**: Max $10 per contract
4. **Auto-Close**: Closes before market close
5. **Error Handling**: Logs all errors

## Performance Tracking

The strategy logs all trades with full details for analysis:
- Win rate
- Average profit
- Best/worst trades
- Market conditions

Review `spy_1dte_trades.json` regularly to analyze performance.
