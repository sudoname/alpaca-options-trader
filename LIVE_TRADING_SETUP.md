# Live Trading Setup Guide

## ⚠️ IMPORTANT
Live trading uses REAL MONEY. Test thoroughly in DRY RUN mode first!

## Setup Steps

### Step 1: Get Your Schwab Account Hash

Run this command:
```bash
python get_account_hash.py
```

This will output something like:
```
Account Number: 12345678
Hash Value: ABC123XYZ456...

Add this to your .env file:
SCHWAB_ACCOUNT_HASH=ABC123XYZ456...
```

### Step 2: Update .env File

Add the account hash to your `.env` file:
```
SCHWAB_APP_KEY=egi2ATAEJd8Dhs5RLM0SoPfoNkl9j6ht
SCHWAB_APP_SECRET=k5zcNhvSfAc4NmQG
SCHWAB_CALLBACK_URL=https://127.0.0.1:5000/callback
SCHWAB_TOKEN_FILE=schwab_tokens.json
SCHWAB_ACCOUNT_HASH=YOUR_ACCOUNT_HASH_HERE  # ← Add this line
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id
```

### Step 3: Start Bot in Live Mode

**DRY RUN (default - simulated trades):**
```bash
python telegram_trader.py
```

**LIVE TRADING (real money!):**
```bash
python telegram_trader.py --live
```

You'll see a confirmation prompt:
```
============================================================
⚠️  WARNING: LIVE TRADING MODE ENABLED
============================================================
Real money will be used for trades!
Make sure SCHWAB_ACCOUNT_HASH is set in .env
============================================================

Type 'YES' to continue with live trading: 
```

Type `YES` to proceed.

## How It Works

### DRY RUN Mode (Default)
- All trades are SIMULATED
- No real money used
- Perfect for testing
- Status shows: "✅ DRY RUN (Simulated)"

### LIVE TRADING Mode
- Trades execute through Schwab API
- REAL MONEY is used
- Orders placed as BUY TO OPEN limit orders
- Status shows: "🔴 LIVE TRADING"

## Order Execution

When you send "YES AAPL" in live mode:

1. Bot finds best option
2. Creates Schwab limit order
3. Submits to your account
4. You get confirmation with Order ID
5. Trade logged to schwab_trades.json

## Example Live Trade Response

```
✅ TRADE EXECUTED - AAPL

📊 Order Details:
• Symbol: AAPL 251121C00245000
• Type: CALL
• Strike: $245
• Expiration: 2025-11-21
• Quantity: 1 contract(s)
• Entry Price: $8.40
• Total Cost: $840.00
• Order ID: 123456789  ← Real Schwab order ID

🔬 Greeks:
• Delta: 0.523
• Gamma: 0.021
• Theta: -0.184
• Vega: 0.235

📈 Targets:
• Breakeven: $253.40
• Target (+25%): $10.50
• Stop Loss (-10%): $7.56

⏰ Time: 2025-10-10T15:45:23
Status: EXECUTED  ← Actually placed!

✨ Trade logged to schwab_trades.json
```

## Safety Features

1. **Confirmation Required**
   - Must type 'YES' to start live trading
   - Each trade requires "YES TICKER" confirmation

2. **Budget Limits**
   - $2,000 max per contract by default
   - Filters out expensive options

3. **Status Visibility**
   - Send "STATUS" anytime to check mode
   - Clear indicators for live vs dry run

4. **Error Handling**
   - Failed orders don't crash bot
   - Error messages sent via Telegram
   - All errors logged

## Monitoring Live Trades

### Check Status
Send: `STATUS`

Response shows current mode:
```
Schwab Options Bot

Status: Online
Mode: 🔴 LIVE TRADING
Time: 15:45:23

Send ticker to scan options!
```

### View Trade Log
```bash
cat schwab_trades.json
```

Shows all executed trades with order IDs.

## Switching Modes

### To DRY RUN:
Stop bot (Ctrl+C), restart without --live flag:
```bash
python telegram_trader.py
```

### To LIVE:
Stop bot, restart with --live flag:
```bash
python telegram_trader.py --live
```

## Common Issues

### "Account hash not configured"
**Solution:** Run `python get_account_hash.py` and add to .env

### "Order failed with status 400"
**Possible causes:**
- Invalid option symbol
- Insufficient funds
- Market closed
- Order price too far from market

### "Token expired"
**Solution:** Re-run `python auth_schwab_simple.py`

## Best Practices

1. **Start Small**
   - Test with 1 contract
   - Use cheap options first
   - Verify order goes through

2. **Monitor Closely**
   - Watch first few trades
   - Check Schwab app for confirmations
   - Review trade logs

3. **Set Alerts**
   - Enable Telegram notifications
   - Monitor position_monitor.py output
   - Track P&L daily

4. **Use Limits**
   - Only trade during market hours
   - Set max daily loss
   - Don't exceed budget

## Auto-Roll in Live Mode

Position monitor also supports live trading:

```bash
# Dry run (default)
python position_monitor.py

# Live trading
# Edit position_monitor.py line 140: dry_run=False
python position_monitor.py
```

Auto-rolls will execute REAL trades when positions go ITM.

## Emergency Stop

To immediately stop all trading:
1. Press `Ctrl+C` to stop bot
2. Check Schwab app for open orders
3. Cancel any pending orders manually if needed

---

**Remember: Live trading = REAL MONEY. Always test in DRY RUN first!**
