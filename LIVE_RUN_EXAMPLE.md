# Live Trading Example

## Setup for Live Trading

### Option 1: Using .env (Recommended)

Edit `.env` file:
```bash
# Change this line:
DRY_RUN=true

# To this:
DRY_RUN=false
```

Then start normally:
```bash
python telegram_trader.py
```

### Option 2: Using --live Flag

Keep `.env` as `DRY_RUN=true` but override with flag:
```bash
python telegram_trader.py --live
```

---

## Example: Live Trading Session

### 1. Start Bot in Live Mode

```bash
$ python telegram_trader.py --live

============================================================
⚠️  WARNING: LIVE TRADING MODE ENABLED
============================================================
Real money will be used for trades!
Make sure SCHWAB_ACCOUNT_HASH is set in .env
============================================================

Type 'YES' to continue with live trading: YES

⚠️  LIVE TRADING ENABLED - Real money will be used!
   Mode set via: --live flag
============================================================
Schwab Telegram Bot Started
============================================================
Send messages on Telegram:
  - AAPL, NVDA, etc. - Scan ticker options
  - SCAN - Scan multiple tickers
  - STATUS - Bot status
  - HELP - Show commands
============================================================
```

### 2. Check Status on Telegram

**You send:** `STATUS`

**Bot replies:**
```
Schwab Options Bot

Status: Online
Mode: 🔴 LIVE TRADING
Time: 15:30:45

Send ticker to scan options!
```

### 3. Scan for Options

**You send:** `AAPL`

**Bot replies:**
```
AAPL COMPREHENSIVE ANALYSIS

💹 Stock Data:
• Current: $246.18 🟢 0.49 (0.2%)
• Market: 🟢 OPEN
• Volume: 45,234
• Moneyness: ATM

🎯 Selected Option:
• Strike: $245.0 CALL (0.5% ITM)
• Expires: 2025-11-21 (42 days)
• Premium: $8.40 (3.4% of stock)
• ML Score: 96/100 🔥

🔬 Option Greeks:
• Delta: 0.523 (52% stock correlation)
• Gamma: 0.021 (Delta acceleration)
• Theta: -0.184 ($-18.40/day decay)
• Vega: 0.235 (Vol sensitivity)
• IV: 32.3% 📊 Normal

💰 Risk/Reward Analysis:
• Cost: $840 per contract
• Breakeven: $253.40 (3.3% move needed)
• 25% Profit: Stock > $255.50 (4.2% move)
• 50% Profit: Stock > $257.60 (5.0% move)
• Max Loss: $840 (100% premium)

🧠 AI Insights:
• Confidence: 100%
• Risk Level: MEDIUM
• Liquidity: 🟢 Excellent
• Open Interest: 2,418

🛡️ Smart Risk Management:
• Initial Stop: -10% ($7.56)
• Profit Target: +25% close 50%
• Time Stop: Close 5 days before expiry
• Strategy: CALL options

💡 Recommendation: 🟢 BUY
Confidence: 100%

📋 Analysis Reasons:
• Optimal delta for ATM trading
• Moderate IV - balanced premium
• Excellent liquidity
• At-the-money - balanced risk/reward

💬 Reply:
• YES AAPL - Execute this trade
• NO - Cancel
```

### 4. Execute Live Trade

**You send:** `YES AAPL`

**Bot processes (in console):**
```
Received: YES AAPL from 5035226682
LIVE TRADE - Placing order for 1 contract(s) of AAPL  251121C00245000
✅ Order placed successfully. Order ID: 987654321
Sent response (412 chars)
```

**Bot replies to you:**
```
✅ TRADE EXECUTED - AAPL

📊 Order Details:
• Symbol: AAPL  251121C00245000
• Type: CALL
• Strike: $245
• Expiration: 2025-11-21
• Quantity: 1 contract(s)
• Entry Price: $8.40
• Total Cost: $840.00
• Order ID: 987654321  ← REAL SCHWAB ORDER!

🔬 Greeks:
• Delta: 0.523
• Gamma: 0.021
• Theta: -0.184
• Vega: 0.235

📈 Targets:
• Breakeven: $253.40
• Target (+25%): $10.50
• Stop Loss (-10%): $7.56

⏰ Time: 2025-10-10T15:30:52
Status: EXECUTED  ← REAL TRADE!

✨ Trade logged to schwab_trades.json
```

### 5. Verify in Schwab App

Open your Schwab mobile app or website:
- Go to Accounts > Positions
- See: AAPL 251121C00245000 - 1 contract
- Status: Filled
- Cost: $840.00

### 6. Check Trade Log

```bash
$ cat schwab_trades.json
```

```json
[
  {
    "timestamp": "2025-10-10T15:30:52.123456",
    "symbol": "AAPL  251121C00245000",
    "ticker": "AAPL",
    "type": "CALL",
    "strike": 245.0,
    "expiration": "2025-11-21",
    "quantity": 1,
    "entry_price": 8.40,
    "cost": 840.0,
    "underlying_price": 246.18,
    "score": 95.8,
    "days_to_exp": 42,
    "moneyness": "ATM",
    "greeks": {
      "delta": 0.523,
      "gamma": 0.021,
      "theta": -0.184,
      "vega": 0.235
    },
    "iv": 32.3,
    "status": "EXECUTED",
    "order_id": "987654321"
  }
]
```

---

## Live Position Monitor

### Start Position Monitor in Live Mode

Edit `.env`:
```
DRY_RUN=false
```

Then run:
```bash
$ python position_monitor.py

============================================================
SCHWAB POSITION MONITOR - AUTO ROLL ITM POSITIONS
============================================================
Mode: 🔴 LIVE TRADING
Check interval: 300 seconds
Roll logic: 30-60 days based on ML score
Press Ctrl+C to stop
============================================================

[2025-10-10 15:35:00] Checking positions...
  Active positions: 1
  ✓ AAPL: $245 (Current: $246.18)
```

### When Position Goes ITM

```
[2025-10-10 16:05:00] Checking positions...
  Active positions: 1

  🔔 AAPL is ITM!
     Strike: $245, Current: $252.50
  📊 Rolling AAPL based on ML score 95.8 → 60 days
  ✅ Rolled to $255 exp 2025-12-15
  💰 Est. P&L: $750.00 (89.3%)
```

**Telegram notification:**
```
🔄 POSITION ROLLED - AAPL

📊 Closed Position:
• Strike: $245
• Entry: $8.40
• ML Score: 95.8
• Est. P&L: $750.00 (89.3%)

📈 New Position:
• Strike: $255
• Expires: 2025-12-15 (60 days)
• Entry: $9.20
• Cost: $920.00
• ML Score: 92.5/100

💡 Reason: ITM position auto-rolled
⏰ 2025-10-10 16:05:23
```

---

## Comparison: DRY RUN vs LIVE

### DRY RUN Mode (DRY_RUN=true)

**Status shows:**
```
Mode: ✅ DRY RUN (Simulated)
```

**Trade response:**
```
Status: SIMULATED
```

**In Schwab:** No orders appear

### LIVE Mode (DRY_RUN=false)

**Status shows:**
```
Mode: 🔴 LIVE TRADING
```

**Trade response:**
```
Status: EXECUTED
Order ID: 987654321
```

**In Schwab:** Real orders and positions appear!

---

## Quick Switch Guide

### Currently in DRY RUN, want to go LIVE:

1. Stop bot (Ctrl+C)
2. Edit `.env`: `DRY_RUN=false`
3. Restart: `python telegram_trader.py`
4. Verify status shows: `🔴 LIVE TRADING`

### Currently LIVE, want to go DRY RUN:

1. Stop bot (Ctrl+C)
2. Edit `.env`: `DRY_RUN=true`
3. Restart: `python telegram_trader.py`
4. Verify status shows: `✅ DRY RUN (Simulated)`

---

## Safety Checklist Before Going Live

- [ ] Account hash added to .env
- [ ] Tested thoroughly in DRY RUN mode
- [ ] Verified Schwab API connection works
- [ ] Checked account has sufficient funds
- [ ] Only trading during market hours
- [ ] Starting with 1 contract on cheap option
- [ ] Monitoring bot console output
- [ ] Telegram notifications working
- [ ] Ready to check Schwab app after first trade

**Once all checked ✓ → Set DRY_RUN=false and go live!**
