"""
Daily Scheduler for SPY 1DTE Strategy - ENHANCED
Runs every weekday at 10:00 AM EST
(Waits 30 minutes after market open for direction confirmation)
"""

import schedule
import time
from datetime import datetime
import pytz
from spy_1dte_strategy import SPY1DTEStrategy


def is_market_open_day():
    """Check if today is a weekday (market is open)"""
    now = datetime.now()
    # Monday = 0, Sunday = 6
    return now.weekday() < 5  # Monday to Friday


def run_strategy():
    """Run the 1DTE strategy if market is open"""
    if is_market_open_day():
        print(f"\n[SCHEDULER] Starting SPY 1DTE strategy at {datetime.now()}")
        try:
            strategy = SPY1DTEStrategy()
            strategy.run_daily_strategy()
        except Exception as e:
            print(f"[ERROR] Strategy failed: {e}")
            import traceback
            traceback.print_exc()
    else:
        print(f"[SCHEDULER] Market closed today ({datetime.now().strftime('%A')})")


def main():
    print("=" * 60)
    print("SPY 1DTE DAILY SCHEDULER - ENHANCED")
    print("=" * 60)
    print("Schedule: Every weekday at 10:00 AM EST")
    print("Strategy: 1DTE SPY options with 20% profit target")
    print("Features: Trade filtering, 15-min monitoring, early stop loss")
    print("=" * 60)

    # Schedule the strategy to run at 10:00 AM EST every day
    # (30 minutes after market open for direction confirmation)
    schedule.every().day.at("10:00").do(run_strategy)

    print(f"\n[READY] Scheduler started at {datetime.now()}")
    print("[WAITING] Next run: Tomorrow at 10:00 AM EST")
    print("Press Ctrl+C to stop\n")

    # Keep running
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute


if __name__ == '__main__':
    main()
