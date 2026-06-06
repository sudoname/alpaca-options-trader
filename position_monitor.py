"""
Position Monitor - Runs continuously to manage positions
Automatically rolls ITM positions based on ML score
"""

import time
import json
import os
from datetime import datetime
from schwab_trader import SchwabOptionsTrader
import argparse

def check_if_itm(position, current_price):
    """Check if position is In The Money"""
    strike = position['strike']
    option_type = position['type']

    if option_type == 'CALL':
        return current_price > strike
    else:  # PUT
        return current_price < strike


def determine_roll_days(ml_score):
    """
    Determine roll period based on ML score

    High score (>85): 60 days (more confidence)
    Medium score (70-85): 45 days
    Low score (<70): 30 days (less confidence, shorter exposure)
    """
    if ml_score >= 85:
        return 60
    elif ml_score >= 70:
        return 45
    else:
        return 30


def roll_position(trader, position):
    """
    Roll an ITM position to new expiration

    Args:
        trader: SchwabOptionsTrader instance
        position: Current position dict

    Returns:
        Roll result dict
    """
    ticker = position['ticker']
    option_type = position['type']
    ml_score = position.get('score', 75)  # Use score from position

    # Determine new expiration period
    roll_days = determine_roll_days(ml_score)

    print(f"  📊 Rolling {ticker} based on ML score {ml_score:.1f} → {roll_days} days")

    # Find new option
    new_option = trader.find_best_trade(
        tickers=[ticker],
        option_type=option_type,
        budget=2000.0,
        min_days=roll_days - 10,
        max_days=roll_days + 10,
        min_delta=0.35,
        max_delta=0.65,
        max_iv=50.0
    )

    if not new_option:
        print(f"  ❌ No suitable roll option found for {ticker}")
        return None

    # Analyze new option
    analysis = trader.analyze_option(new_option)

    # Calculate estimated P&L from closing old position
    entry_cost = position['cost']
    current_value = (position['underlying_price'] - position['strike']) * 100  # Simplified
    pnl = current_value - entry_cost
    pnl_pct = (pnl / entry_cost) * 100 if entry_cost > 0 else 0

    # Execute new position
    new_trade = trader.execute_trade(new_option, quantity=1)

    # Log the roll
    roll_record = {
        'timestamp': datetime.now().isoformat(),
        'ticker': ticker,
        'action': 'ROLL',
        'old_position': {
            'strike': position['strike'],
            'expiration': position['expiration'],
            'entry_price': position['entry_price'],
            'cost': entry_cost,
            'ml_score': ml_score
        },
        'new_position': {
            'strike': new_option['strike'],
            'expiration': new_option['expiration'],
            'entry_price': new_option['ask'],
            'cost': new_option['ask'] * 100,
            'ml_score': new_option['score']
        },
        'estimated_pnl': pnl,
        'pnl_pct': pnl_pct,
        'roll_days': roll_days,
        'reason': f'ITM position auto-rolled based on ML score {ml_score:.1f}'
    }

    # Save roll log
    save_roll_log(roll_record)

    print(f"  ✅ Rolled to ${new_option['strike']} exp {new_option['expiration'][:10]}")
    print(f"  💰 Est. P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")

    return roll_record


def save_roll_log(roll_record):
    """Save roll to log file"""
    roll_file = 'roll_log.json'

    if os.path.exists(roll_file):
        with open(roll_file, 'r') as f:
            rolls = json.load(f)
    else:
        rolls = []

    rolls.append(roll_record)

    with open(roll_file, 'w') as f:
        json.dump(rolls, f, indent=2)


def continuous_monitor(interval_seconds=60, telegram_bot=None, chat_id=None):
    """Continuously monitor positions and auto-roll ITM ones"""
    # Read from .env - defaults to respecting DRY_RUN setting
    trader = SchwabOptionsTrader()

    mode = "🔴 LIVE TRADING" if not trader.dry_run else "✅ DRY RUN (Simulated)"

    print("=" * 60)
    print("SCHWAB POSITION MONITOR - AUTO ROLL ITM POSITIONS")
    print("=" * 60)
    print(f"Mode: {mode}")
    print(f"Check interval: {interval_seconds} seconds")
    print(f"Roll logic: 30-60 days based on ML score")
    print("Press Ctrl+C to stop")
    print("=" * 60)

    try:
        while True:
            current_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            print(f"\n[{current_time}] Checking positions...")

            try:
                # Load active positions
                trades_file = 'schwab_trades.json'
                if not os.path.exists(trades_file):
                    print("  No trades file found")
                    time.sleep(interval_seconds)
                    continue

                with open(trades_file, 'r') as f:
                    trades = json.load(f)

                # Filter active positions
                active = [t for t in trades if t.get('status') != 'CLOSED']

                if not active:
                    print("  No active positions")
                else:
                    print(f"  Active positions: {len(active)}")

                    for position in active:
                        ticker = position['ticker']
                        strike = position['strike']

                        # Get current price (from underlying_price in position or re-fetch)
                        current_price = position.get('underlying_price', strike)

                        # Check if ITM
                        if check_if_itm(position, current_price):
                            print(f"\n  🔔 {ticker} is ITM!")
                            print(f"     Strike: ${strike}, Current: ${current_price:.2f}")

                            # Roll the position
                            roll_result = roll_position(trader, position)

                            # Send Telegram notification if available
                            if roll_result and telegram_bot and chat_id:
                                send_roll_notification(telegram_bot, chat_id, roll_result)
                        else:
                            print(f"  ✓ {ticker}: ${strike} (Current: ${current_price:.2f})")

            except Exception as e:
                print(f"  Error during monitoring: {e}")
                import traceback
                traceback.print_exc()

            time.sleep(interval_seconds)

    except KeyboardInterrupt:
        print("\n\nPosition monitor stopped by user")

def send_roll_notification(telegram_bot, chat_id, roll_record):
    """Send Telegram notification about the roll"""
    old = roll_record['old_position']
    new = roll_record['new_position']

    message = f"🔄 POSITION ROLLED - {roll_record['ticker']}\n\n"
    message += f"📊 Closed Position:\n"
    message += f"• Strike: ${old['strike']}\n"
    message += f"• Entry: ${old['entry_price']:.2f}\n"
    message += f"• ML Score: {old['ml_score']:.1f}\n"
    message += f"• Est. P&L: ${roll_record['estimated_pnl']:.2f} ({roll_record['pnl_pct']:.1f}%)\n\n"

    message += f"📈 New Position:\n"
    message += f"• Strike: ${new['strike']}\n"
    message += f"• Expires: {new['expiration'][:10]} ({roll_record['roll_days']} days)\n"
    message += f"• Entry: ${new['entry_price']:.2f}\n"
    message += f"• Cost: ${new['cost']:.2f}\n"
    message += f"• ML Score: {new['ml_score']:.1f}/100\n\n"

    message += f"💡 Reason: ITM position auto-rolled\n"
    message += f"⏰ {roll_record['timestamp'][:19]}"

    telegram_bot.send_message(chat_id, message)


def main():
    parser = argparse.ArgumentParser(description='Schwab Position Monitor with Auto-Roll')
    parser.add_argument('--interval', type=int, default=300,
                       help='Check interval in seconds (default: 300 = 5 min)')
    parser.add_argument('--once', action='store_true',
                       help='Check once and exit')

    args = parser.parse_args()

    if args.once:
        print("Running single position check with auto-roll...")
        # Read from .env - defaults to respecting DRY_RUN setting
        trader = SchwabOptionsTrader()

        # Check for ITM positions
        trades_file = 'schwab_trades.json'
        if os.path.exists(trades_file):
            with open(trades_file, 'r') as f:
                trades = json.load(f)

            active = [t for t in trades if t.get('status') != 'CLOSED']

            for position in active:
                ticker = position['ticker']
                strike = position['strike']
                current_price = position.get('underlying_price', strike)

                if check_if_itm(position, current_price):
                    print(f"ITM detected: {ticker}")
                    roll_position(trader, position)

        print("Position check complete")
    else:
        continuous_monitor(args.interval)


if __name__ == "__main__":
    main()