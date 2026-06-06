"""
Daily Scheduler for SPY+QQQ Hybrid Strategy
Runs every weekday at 9:00 AM CST
Automatically switches between 1DTE and 2DTE based on PDT status
Trades both SPY and QQQ (picks best opportunity)
Max premium: $0.50 | Delta range: 0.25-0.35
"""

import schedule
import time
import json
from datetime import datetime
from spy_qqq_hybrid_strategy import SPYQQQHybridStrategy
from pdt_tracker import PDTTracker


def update_scheduler_status():
    """Write scheduler status to file for Telegram bot to read"""
    try:
        jobs = schedule.get_jobs()
        next_run = jobs[0].next_run if jobs else None

        status = {
            'last_heartbeat': datetime.now().isoformat(),
            'next_run': next_run.isoformat() if next_run else None,
            'schedule': '9:00 AM CST Daily',
            'tickers': ['SPY', 'QQQ'],
            'max_premium': 6.0,
            'delta_range': '0.25-0.35',
            'status': 'running'
        }

        with open('scheduler_status.json', 'w') as f:
            json.dump(status, f, indent=2)
    except Exception as e:
        print(f"[STATUS] Error writing status file: {e}")


def is_market_open_day():
    """Check if today is a weekday (market is open)"""
    now = datetime.now()
    return now.weekday() < 5  # Monday to Friday


def run_strategy():
    """Run the hybrid strategy if market is open"""
    if is_market_open_day():
        print(f"\n[SCHEDULER] Starting SPY+QQQ Hybrid strategy at {datetime.now()}")

        # Show PDT status before trading
        pdt = PDTTracker()
        status = pdt.get_status_message()
        print(f"[PDT] Status: {status['status']}")

        try:
            # Import TelegramNotifier from strategy file
            from spy_qqq_hybrid_strategy import TelegramNotifier
            telegram = TelegramNotifier()

            # Send START notification
            telegram.send(f"""🤖 *SPY+QQQ STRATEGY STARTING*

🕐 Time: `{datetime.now().strftime('%I:%M %p CST')}`
📊 Tickers: `SPY, QQQ`
🎯 Max Premium: `$6.00`
📐 Delta Range: `0.25-0.35`
🛡️ PDT Status: `{status['count']}/3` ({status['remaining']} remaining)

Analyzing market conditions...""")

            strategy = SPYQQQHybridStrategy()
            strategy.run_daily_strategy()

            # Success - strategy completed without crash
            print(f"[SUCCESS] Strategy execution completed")

        except Exception as e:
            print(f"[ERROR] Strategy failed: {e}")
            import traceback
            error_details = traceback.format_exc()
            traceback.print_exc()

            # Send ERROR notification to Telegram
            try:
                from spy_qqq_hybrid_strategy import TelegramNotifier
                telegram = TelegramNotifier()

                # Truncate error for Telegram (max 4000 chars)
                error_msg = str(e)
                if len(error_msg) > 200:
                    error_msg = error_msg[:200] + "..."

                telegram.send(f"""❌ *SPY+QQQ STRATEGY FAILED*

🕐 Time: `{datetime.now().strftime('%I:%M %p CST')}`
🛡️ PDT: `{status['count']}/3`

**Error:**
`{error_msg}`

**Details:**
{error_details[:500] if len(error_details) > 500 else error_details}

⚠️ Strategy will retry tomorrow at 9:00 AM CST""")
            except:
                print("[ERROR] Failed to send Telegram error notification")
    else:
        print(f"[SCHEDULER] Market closed today ({datetime.now().strftime('%A')})")


def main():
    print("=" * 60)
    print("SPY+QQQ HYBRID STRATEGY - DAILY SCHEDULER")
    print("=" * 60)
    print("Schedule: Every weekday at 9:00 AM CST")
    print("Market Hours: 8:30 AM - 3:00 PM CST")
    print("Trading Window: 9:00 AM - 2:45 PM CST")
    print("Features:")
    print("  - Trades both SPY and QQQ (picks best)")
    print("  - Max premium: $6.00 | Delta: 0.25-0.35")
    print("  - Automatic 1DTE/2DTE switching")
    print("  - PDT protection (auto-track day trades)")
    print("  - Trade filtering (70% confidence min)")
    print("  - No profit cap (trailing stop only!)")
    print("  - Telegram notifications")
    print("=" * 60)

    # Show current PDT status
    pdt = PDTTracker()
    status = pdt.get_status_message()
    print(f"\n[PDT] Current Status: {status['status']}")
    print(f"[PDT] Day Trades: {status['count']}/3")
    print(f"[PDT] Remaining: {status['remaining']}")

    # Schedule the strategy to run at 9:00 AM CST every day
    schedule.every().day.at("09:00").do(run_strategy)

    print(f"\n[READY] Scheduler started at {datetime.now()}")
    print("[WAITING] Next run: 9:00 AM CST")
    print("Press Ctrl+C to stop\n")

    # Write initial status
    update_scheduler_status()

    # Keep running
    while True:
        schedule.run_pending()
        update_scheduler_status()  # Update status file every minute
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    main()
