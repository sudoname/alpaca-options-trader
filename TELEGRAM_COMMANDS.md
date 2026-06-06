# Schwab Options Trading Bot - Telegram Commands

## Quick Reference

### `/start`
Welcome message with full command list and feature overview.

### `/scan` or `/scan TICKER1 TICKER2 ...`
Scan for best CALL options.

**Examples:**
- `/scan` - Uses default tickers (AAPL, MSFT, GOOGL, NVDA, AMD, TSLA, META, AMZN)
- `/scan AAPL TSLA` - Scans only AAPL and TSLA

**Returns:**
- Best option found with full details
- Strike price, expiration, days to expiry
- Bid/Ask/Last prices
- Complete Greeks (Delta, Gamma, Theta, Vega)
- IV, Volume, Open Interest
- Score (0-100)
- Analysis with recommendation, confidence %, and risk level
- Reasons for recommendation

### `/scan_puts` or `/scan_puts TICKER1 TICKER2 ...`
Scan for best PUT options.

**Examples:**
- `/scan_puts` - Uses default tickers
- `/scan_puts NVDA AMD` - Scans only NVDA and AMD puts

**Returns:**
Same detailed analysis as `/scan` but for PUT options.

### `/analyze TICKER`
Analyze options for a specific ticker and show top 3 opportunities.

**Examples:**
- `/analyze AAPL`
- `/analyze MSFT`

**Returns:**
Top 3 options for the ticker with:
- Score ranking
- Strike and expiration
- Price, Delta, IV

### `/trade`
Execute the best trade found (DRY RUN mode).

Performs full workflow:
1. Scans default tickers
2. Finds best option
3. Analyzes with confidence scoring
4. Simulates trade execution (if confidence >= 60%)

**Returns:**
- Trade execution details (simulated)
- Symbol, ticker, type, strike, expiration
- Quantity and entry price
- Total cost
- Greeks
- Status and timestamp

**Note:** This is SIMULATED only - no real money is used.

### `/status`
Show bot status and configuration.

**Returns:**
- Online status
- Current mode (DRY RUN)
- Current time
- Default tickers list
- Available features

### `/tickers`
Show current ticker list and how to customize scans.

**Returns:**
- Current default ticker list
- Instructions for custom ticker scans

## Example Workflow

1. **Start bot:**
   ```
   /start
   ```

2. **Quick scan for calls:**
   ```
   /scan
   ```

3. **Analyze specific stock:**
   ```
   /analyze AAPL
   ```

4. **Scan for puts:**
   ```
   /scan_puts
   ```

5. **Execute simulated trade:**
   ```
   /trade
   ```

6. **Check status:**
   ```
   /status
   ```

## Sample Output

### `/scan` Example:
```
🎯 Best Option Found

Symbol: AAPL  251121C00245000
Ticker: AAPL | Type: CALL
Strike: $245 | Exp: 2025-11-21
Days to Exp: 42

Pricing:
Bid: $7.25 | Ask: $7.45 | Last: $7.35
Cost: $745.00 per contract

Greeks:
Delta: 0.536
Gamma: 0.018
Theta: -0.082
Vega: 0.124

Metrics:
IV: 27.8%
Volume: 1,245
Open Interest: 8,932
Score: 89.6/100

Analysis:
Recommendation: BUY
Confidence: 100%
Risk Level: MEDIUM

Reasons:
• Optimal delta for ATM trading
• Moderate IV - balanced premium
• Excellent liquidity
• Low time decay
• Optimal time frame

Underlying: $246.18 | Moneyness: ATM
```

## Filters Applied

All scans use these filters:
- **Budget:** $2,000 max per contract
- **Days to Expiration:** 20-60 days
- **Delta Range:** 0.35-0.65
- **Max IV:** 50%
- **Strike Count:** 10 (5 above/below ATM)

## Tips

✅ Use `/scan` during market hours for best real-time data
✅ Use `/analyze` to deep-dive on specific stocks
✅ Check `/status` to verify bot is online
✅ Custom scans: `/scan TICKER1 TICKER2` for specific stocks
✅ `/trade` shows full analysis before simulating execution

⚠️ Remember: All trades are SIMULATED (DRY RUN mode)
⚠️ Bot data is live from Schwab API
⚠️ Scans may take 30-60 seconds for multiple tickers

## Running the Bot

Start the bot from command line:
```bash
cd C:\Users\yomi\alpaca-options-trader
python schwab_telegram_bot.py
```

Bot will log:
```
Starting Schwab Options Telegram Bot...
Monitoring 8 tickers by default
```

Then interact via Telegram on your phone or desktop app!
