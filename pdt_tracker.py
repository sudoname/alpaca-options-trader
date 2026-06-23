"""
Pattern Day Trader (PDT) Tracking System
Monitors day trades in rolling 5-business-day window
Prevents PDT violations for accounts < $25,000
"""

import os
import json
import json_store
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()


class PDTTracker:
    def __init__(self):
        self.log_file = 'day_trades_log.json'
        # PDT self-limiter is opt-out via env. When disabled the tracker still
        # logs day trades for visibility but never blocks a trade.
        self.enabled = os.getenv('PDT_ENABLED', 'true').strip().lower() \
            not in ('0', 'false', 'no', 'off')
        self.pdt_limit = int(os.getenv('PDT_LIMIT', '3'))  # day trades / 5 business days
        self.account_threshold = 25000  # PDT exempt if >= $25k

    def load_day_trades(self):
        """Load day trade history from file"""
        trades = json_store.read_json(self.log_file, [])
        return trades if isinstance(trades, list) else []

    def save_day_trades(self, trades):
        """Save day trade history to file (lock + atomic via json_store)."""
        with json_store.locked(self.log_file):
            json_store.atomic_write_json(self.log_file, trades)

    def get_business_days_ago(self, days=5):
        """Get date N business days ago"""
        current = datetime.now()
        business_days = 0

        while business_days < days:
            current -= timedelta(days=1)
            # Skip weekends
            if current.weekday() < 5:  # Monday = 0, Friday = 4
                business_days += 1

        return current

    def count_recent_day_trades(self):
        """Count day trades in last 5 business days"""
        trades = self.load_day_trades()
        cutoff_date = self.get_business_days_ago(5)

        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t['date']) >= cutoff_date
        ]

        return len(recent_trades)

    def get_oldest_day_trade_date(self):
        """Get date of oldest day trade in 5-day window"""
        trades = self.load_day_trades()
        cutoff_date = self.get_business_days_ago(5)

        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t['date']) >= cutoff_date
        ]

        if not recent_trades:
            return None

        # Sort by date
        recent_trades.sort(key=lambda x: x['date'])
        return datetime.fromisoformat(recent_trades[0]['date'])

    def days_until_reset(self):
        """Calculate business days until oldest day trade drops off"""
        oldest = self.get_oldest_day_trade_date()
        if not oldest:
            return 0

        # Day trade drops off after 5 business days
        reset_date = oldest + timedelta(days=7)  # Conservative estimate
        days = (reset_date - datetime.now()).days

        return max(0, days)

    def can_day_trade(self):
        """Check if we can make another day trade without PDT violation"""
        if not self.enabled:
            return True

        count = self.count_recent_day_trades()

        if count >= self.pdt_limit:
            return False

        return True

    def get_remaining_day_trades(self):
        """Get number of day trades remaining"""
        if not self.enabled:
            return 9999  # PDT self-limiter disabled

        count = self.count_recent_day_trades()
        remaining = self.pdt_limit - count
        return max(0, remaining)

    def log_day_trade(self, trade_info):
        """Record a day trade"""
        # Add new day trade
        day_trade = {
            'date': datetime.now().isoformat(),
            'symbol': trade_info.get('symbol', 'Unknown'),
            'entry_time': trade_info.get('entry_time', ''),
            'exit_time': trade_info.get('exit_time', ''),
            'profit': trade_info.get('profit', 0),
            'order_id': trade_info.get('order_id', '')
        }

        # Locked atomic append: both the scheduler and the bot log day trades,
        # so a blind read-append-write could lose entries.
        json_store.append_item(self.log_file, day_trade)

        print(f"[PDT] Day trade logged: {day_trade['symbol']}")
        print(f"[PDT] Remaining day trades: {self.get_remaining_day_trades()}/3")

    def get_status_message(self):
        """Get PDT status message for display"""
        count = self.count_recent_day_trades()
        remaining = self.get_remaining_day_trades()

        if not self.enabled:
            return {
                'count': count,
                'remaining': remaining,
                'limit': self.pdt_limit,
                'status': "DISABLED - PDT self-limiter off (PDT_ENABLED=false)",
                'can_trade': True,
            }

        if count == 0:
            status = "SAFE - No day trades in last 5 days"
        elif count == 1:
            status = f"CAUTION - 1 day trade used, {remaining} remaining"
        elif count == 2:
            status = f"WARNING - 2 day trades used, {remaining} remaining"
        elif count >= 3:
            days = self.days_until_reset()
            status = f"LIMIT REACHED - Cannot day trade for ~{days} days"

        return {
            'count': count,
            'remaining': remaining,
            'limit': self.pdt_limit,
            'status': status,
            'can_trade': remaining > 0
        }

    def clean_old_trades(self):
        """Remove day trades older than 5 business days"""
        trades = self.load_day_trades()
        cutoff_date = self.get_business_days_ago(5)

        recent_trades = [
            t for t in trades
            if datetime.fromisoformat(t['date']) >= cutoff_date
        ]

        removed = len(trades) - len(recent_trades)
        if removed > 0:
            self.save_day_trades(recent_trades)
            print(f"[PDT] Cleaned {removed} old day trade(s)")

        return removed

    def reset_counter(self):
        """Reset day trade counter (use with caution!)"""
        print("[PDT WARNING] Resetting day trade counter!")
        print("[PDT WARNING] Only do this if you know what you're doing!")

        self.save_day_trades([])
        print("[PDT] Counter reset. All day trades cleared.")


def main():
    """Test PDT tracker"""
    tracker = PDTTracker()

    print("=" * 60)
    print("PDT TRACKER TEST")
    print("=" * 60)

    # Clean old trades
    tracker.clean_old_trades()

    # Get status
    status = tracker.get_status_message()
    print(f"\nStatus: {status['status']}")
    print(f"Day Trades: {status['count']}/{status['limit']}")
    print(f"Remaining: {status['remaining']}")
    print(f"Can Trade: {status['can_trade']}")

    # Show recent trades
    trades = tracker.load_day_trades()
    if trades:
        print(f"\nRecent Day Trades ({len(trades)}):")
        for trade in trades[-5:]:  # Last 5
            date = datetime.fromisoformat(trade['date']).strftime('%Y-%m-%d %H:%M')
            print(f"  {date} | {trade['symbol']} | P/L: ${trade['profit']:.2f}")
    else:
        print("\nNo recent day trades found.")

    print("\n" + "=" * 60)


if __name__ == '__main__':
    main()
