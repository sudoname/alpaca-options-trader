# Auto-Roll ITM Positions Feature

## Overview
Automatically rolls In-The-Money (ITM) positions to new expiration dates based on ML confidence scores.

## How It Works

### ITM Detection
- Monitors active positions every 5 minutes (default)
- Checks if position is ITM:
  - **CALL**: Current price > Strike price
  - **PUT**: Current price < Strike price

### Auto-Roll Logic

When an ITM position is detected, the system automatically:

1. **Determines Roll Period** based on ML Score:
   - **High Score (>85)**: Roll 60 days out (high confidence)
   - **Medium Score (70-85)**: Roll 45 days out
   - **Low Score (<70)**: Roll 30 days out (lower confidence, shorter exposure)

2. **Finds New Option** with same criteria:
   - Same ticker & option type
   - Expiration in target range (±10 days)
   - Delta 0.35-0.65
   - IV < 50%
   - Under $2,000 budget

3. **Executes Roll**:
   - Closes current position
   - Opens new position
   - Calculates & logs P&L

4. **Sends Notification** (if Telegram connected)

## Usage

### Run Continuously (Recommended)
```bash
python position_monitor.py
```
Checks every 5 minutes (300 seconds)

### Custom Interval
```bash
python position_monitor.py --interval 600
```
Checks every 10 minutes

### One-Time Check
```bash
python position_monitor.py --once
```
Checks once and exits

## Example Output

```
============================================================
SCHWAB POSITION MONITOR - AUTO ROLL ITM POSITIONS
============================================================
Check interval: 300 seconds
Roll logic: 30-60 days based on ML score
Press Ctrl+C to stop
============================================================

[2025-10-10 15:30:00] Checking positions...
  Active positions: 2
  ✓ MSFT: $510 (Current: $508.50)

  🔔 AAPL is ITM!
     Strike: $245, Current: $248.50
  📊 Rolling AAPL based on ML score 95.8 → 60 days
  ✅ Rolled to $250 exp 2025-12-15
  💰 Est. P&L: $350.00 (41.7%)
```

## Telegram Notification

When a roll happens, you receive:

```
🔄 POSITION ROLLED - AAPL

📊 Closed Position:
• Strike: $245
• Entry: $8.40
• ML Score: 95.8
• Est. P&L: $350.00 (41.7%)

📈 New Position:
• Strike: $250
• Expires: 2025-12-15 (60 days)
• Entry: $9.20
• Cost: $920.00
• ML Score: 92.5/100

💡 Reason: ITM position auto-rolled
⏰ 2025-10-10 15:30:15
```

## Files Generated

- **roll_log.json** - Complete history of all rolls
- **schwab_trades.json** - Updated with new positions

## Integration with Telegram Bot

The position monitor can be integrated with the Telegram bot to send notifications. See telegram_trader.py for integration example.

## Safety Features

- **DRY RUN Mode**: All rolls are simulated by default
- **Budget Limits**: Only rolls within $2,000 budget
- **Score Validation**: Requires ML score data
- **Error Handling**: Continues monitoring even if roll fails

## Best Practices

1. Run position monitor in background
2. Monitor roll_log.json for performance
3. Adjust interval based on market conditions
4. Review roll decisions weekly

## Advanced Configuration

Edit `position_monitor.py` to customize:
- Roll thresholds (ML score ranges)
- Days to expiration targets
- Budget limits
- Delta/IV filters
