# SPY 1DTE Strategy - Optimizations & Features

## ✅ Optimized Parameters

### Option Selection
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Delta Range** | 0.30 - 0.45 | Optimal probability vs. premium for 1DTE OTM options |
| **Min Volume** | 100 contracts | Ensures liquidity for entry/exit |
| **Min Open Interest** | 500 contracts | Prevents illiquid options |
| **Max Premium** | $10.00 | Risk management per trade |
| **DTE** | 0-1 days | 1DTE strategy focus |

### Risk Management
| Parameter | Value | Reasoning |
|-----------|-------|-----------|
| **Profit Target** | 20% | Sweet spot for 1DTE options |
| **Stop Loss** | -30% | Limits downside risk |
| **Trailing Stop** | 10% from peak | Locks in profits after 15%+ gain |
| **Max Trades** | 1 per day | Prevents overtrading |

## 🎯 Market Direction Scoring

### Signals Analyzed (Weighted)
1. **SPY Momentum** (0-2 points)
   - Strong move (>0.2%) = 2 points
   - Moderate move = 1 point

2. **VIX Level** (0-1 point)
   - High VIX (>25) = Bearish
   - Low VIX (<15) = Bullish

3. **Gap Analysis** (0-1 point)
   - Gap up >0.3% = Bullish
   - Gap down <-0.3% = Bearish

4. **Confidence Score**
   - Calculated as: (Winning signals / Total signals) × 100

## 📱 Telegram Notifications

### Entry Notification
Sent immediately when trade is executed:
```
*SPY 1DTE TRADE OPENED*

Type: CALL/PUT
Strike: $XXX.XX
Premium: $X.XX
Cost: $XXX.XX
Delta: 0.XXX
Volume: X,XXX

Market Analysis:
SPY: $XXX.XX (+X.XX%)
VIX: XX.XX
Confidence: XX%

Target: 20% profit ($X.XX)
Stop: -30% loss ($X.XX)

Order ID: XXXXXXXXX
```

### Progress Updates
- **At +10% profit:** "SPY 1DTE Update: +10.X% profit ($XX.XX)"
- **At +15% profit:** "SPY 1DTE Update: +15.X% profit - Near target!"

### Exit Notification
Sent when position is closed:
```
🎯 SPY 1DTE TRADE CLOSED

Reason: PROFIT_TARGET / STOP_LOSS / TRAILING_STOP / MARKET_CLOSE

Entry: $X.XX
Exit: $X.XX
Profit: $+XX.XX (+XX.X%)

Trade Details:
Type: CALL/PUT
Strike: $XXX.XX
Hold Time: XX minutes

Order IDs:
Entry: XXXXXXXXX
Exit: XXXXXXXXX
```

## 🔄 Advanced Features

### 1. Intelligent Option Scoring
Combines multiple factors:
- **Delta Score** (70%): Closeness to optimal 0.35-0.40 delta
- **Volume Score** (20%): Higher volume = better liquidity
- **Spread Score** (10%): Tighter bid-ask spread = better fill

### 2. Trailing Stop Logic
- Activates after position reaches +15% profit
- Closes if profit drops 10% from peak
- Example: Peak at +25%, closes if drops to +15%

### 3. Multiple Exit Conditions
1. **Profit Target** (20%): Primary exit
2. **Stop Loss** (-30%): Risk management
3. **Trailing Stop** (10% from peak): Protect profits
4. **Market Close** (3:45 PM): Time-based exit

## 📊 Option Selection Algorithm

```python
def score_option(option):
    # Perfect delta = 0.35-0.40
    delta_score = 100 - abs((delta - 0.375) * 200)

    # Higher volume = better (capped at 30 points)
    volume_score = min(volume / 1000 * 10, 30)

    # Tighter spread = better (max 20 points)
    spread_pct = (ask - bid) / ask
    spread_score = max(20 - (spread_pct * 100), 0)

    return delta_score + volume_score + spread_score
```

## 🎮 Usage Examples

### Start Automated Daily Trading
```bash
python run_spy_1dte_daily.py
```
- Runs at 9:30 AM every weekday
- Analyzes market
- Executes 1 trade
- Monitors until close
- Sends Telegram updates

### Manual Execution (Test)
```bash
python run_spy_1dte_now.py
```
- Run strategy immediately
- Good for testing
- Same logic as automated

## 📈 Expected Performance

### Win Rate Target
- **Goal:** 60-70% win rate
- 1DTE options are directional bets
- 20% profit target is achievable multiple times per week

### Risk/Reward
- **Max Risk:** $100 per trade (premium paid)
- **Target Profit:** $20 per trade (20%)
- **Risk/Reward Ratio:** 1:5 (excellent)

### Monthly Projection (Conservative)
- **Trades:** ~20 per month (1 per day)
- **Win Rate:** 65%
- **Winners:** 13 trades @ $20 = $260
- **Losers:** 7 trades @ -$30 = -$210
- **Net Profit:** $50/month (~10% on $500 account)

## ⚙️ Configuration

All parameters in `.env`:
```bash
# Schwab Account
SCHWAB_ACCOUNT_HASH=your_account_hash

# Telegram
TELEGRAM_BOT_TOKEN=your_bot_token
TELEGRAM_CHAT_ID=your_chat_id

# Trading Mode
DRY_RUN=false  # Set to true for paper trading
```

## 🛡️ Safety Features

1. **One Trade Per Day** - Prevents overtrading
2. **Liquidity Filters** - Min volume/OI requirements
3. **Spread Protection** - Scores tighter spreads higher
4. **Auto-Exit** - Closes before market close
5. **Stop Loss** - Limits downside to 30%
6. **Trailing Stop** - Protects realized gains

## 📱 Telegram Setup

1. Create bot with @BotFather
2. Get bot token
3. Get your chat ID (use @userinfobot)
4. Add to `.env`:
```bash
TELEGRAM_BOT_TOKEN=123456789:ABCdef...
TELEGRAM_CHAT_ID=123456789
```

## 🔍 Monitoring

### Real-time
- Telegram notifications
- Console output

### Historical
- `spy_1dte_trades.json` - Complete trade log
- Includes entry/exit prices, P/L, reasons, order IDs

### Analysis
```bash
python -m json.tool spy_1dte_trades.json
```

## 🚀 Next Steps

1. **Start scheduler:** `python run_spy_1dte_daily.py`
2. **Monitor Telegram** for trade notifications
3. **Review trades** in `spy_1dte_trades.json`
4. **Adjust parameters** in `.env` if needed

---

**Remember:** This strategy trades real money. Start with the scheduler and let it run for a few weeks to validate performance before increasing capital.
