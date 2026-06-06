"""
Daily Scheduler for SPY Hybrid Strategy
Runs every weekday at 10:00 AM EST
Automatically switches between 1DTE and 2DTE based on PDT status
"""

import schedule
import time
from datetime import datetime
from spy_hybrid_strategy import SPYHybridStrategy
from pdt_tracker import PDTTracker


def is_market_open_day():
    """Check if today is a weekday (market is open)"""
    now = datetime.now()
    return now.weekday() < 5  # Monday to Friday


def run_strategy():
    """Run the hybrid strategy if market is open"""
    if is_market_open_day():
        print(f"\n[SCHEDULER] Starting SPY Hybrid strategy at {datetime.now()}")

        # Show PDT status before trading
        pdt = PDTTracker()
        status = pdt.get_status_message()
        print(f"[PDT] Status: {status['status']}")

        try:
            strategy = SPYHybridStrategy()
            strategy.run_daily_strategy()
        except Exception as e:
            print(f"[ERROR] Strategy failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[SCHEDULER] Market closed today ({datetime.now().strftime('%A')})")


def main():
    print("=" * 60)
    print("SPY HYBRID STRATEGY - DAILY SCHEDULER")
    print("=" * 60)
    print("Schedule: Every weekday at 9:00 AM CST")
    print("Market Hours: 8:30 AM - 3:00 PM CST")
    print("Trading Window: 9:00 AM - 2:45 PM CST")
    print("Features:")
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

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    main()
