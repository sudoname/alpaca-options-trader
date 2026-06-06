"""
Backtest SPY 1DTE Strategy
Simulates the strategy from January 2025 to present
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from schwab import auth, client
import random

load_dotenv()


class SPY1DTEBacktest:
    def __init__(self):
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.client = auth.client_from_token_file(
            self.token_file, self.app_key, self.app_secret
        )

        # Strategy parameters
        self.profit_target = 0.20  # 20%
        self.stop_loss = -0.30  # -30%
        self.initial_capital = 500
        self.max_premium = 10

    def get_historical_data(self, start_date, end_date):
        """Get historical SPY price data"""
        print(f"[DATA] Fetching SPY data from {start_date} to {end_date}...")

        # Convert to datetime objects
        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Get price history
        response = self.client.get_price_history(
            'SPY',
            period_type=client.Client.PriceHistory.PeriodType.YEAR,
            period=client.Client.PriceHistory.Period.ONE_YEAR,
            frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
            frequency=client.Client.PriceHistory.Frequency.DAILY,
            start_datetime=start_dt,
            end_datetime=end_dt
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to get price history: {response.status_code}")
            return []

        data = response.json()
        candles = data.get('candles', [])

        print(f"[DATA] Retrieved {len(candles)} trading days")

        return candles

    def analyze_market_direction(self, current_candle, prev_candle):
        """
        Analyze market direction for the day
        Returns: 'CALL' or 'PUT' and confidence
        """
        current_open = current_candle['open']
        current_close = prev_candle['close']  # Previous day's close

        # Gap
        gap = ((current_open - current_close) / current_close) * 100

        # Intraday momentum (simulated based on first hour)
        # In reality, we'd wait until 9:30 AM
        # For backtest, we'll use a simplified model
        intraday_move = random.uniform(-0.5, 0.5)  # Random noise

        bullish = 0
        bearish = 0

        # Gap analysis
        if gap > 0.3:
            bullish += 1
        elif gap < -0.3:
            bearish += 1

        # Simulate VIX (inverse correlation with SPY)
        if gap < -0.5:  # Big gap down = high VIX
            bearish += 1
        elif gap > 0.5:  # Big gap up = low VIX
            bullish += 1

        # Default to gap direction
        if gap > 0:
            bullish += 1
        else:
            bearish += 1

        if bullish > bearish:
            direction = 'CALL'
            confidence = (bullish / (bullish + bearish)) * 100
        else:
            direction = 'PUT'
            confidence = (bearish / (bullish + bearish)) * 100

        return direction, confidence, gap

    def simulate_option_trade(self, direction, spy_price, spy_daily_return):
        """
        Simulate a 1DTE option trade

        For simplicity:
        - Entry premium: $0.50 - $1.50 (typical for OTM 1DTE)
        - Delta: ~0.35
        - Option moves ~35% of underlying move
        """
        # Simulate entry premium (OTM option with ~0.35 delta)
        entry_premium = random.uniform(0.50, 1.50)

        # Ensure it's under max premium
        if entry_premium > self.max_premium:
            entry_premium = random.uniform(0.50, 1.00)

        # Simulate strike (slightly OTM)
        if direction == 'CALL':
            strike = spy_price * 1.005  # 0.5% OTM
        else:
            strike = spy_price * 0.995  # 0.5% OTM

        # Simulate intraday price movement
        # For 1DTE options, gamma is high, so they're sensitive to moves
        # Approximate: option moves ~30-40% of underlying move for 0.35 delta

        delta_effect = 0.35  # Our target delta
        gamma_effect = 0.15  # Additional gamma impact for 1DTE

        # Calculate option P/L based on SPY move
        underlying_pct_move = spy_daily_return * 100  # Convert to percentage

        if direction == 'CALL':
            # CALL profits when SPY goes up
            option_pct_move = underlying_pct_move * (delta_effect + gamma_effect)
        else:
            # PUT profits when SPY goes down
            option_pct_move = -underlying_pct_move * (delta_effect + gamma_effect)

        # Add some randomness (bid-ask spread, slippage, volatility)
        option_pct_move += random.uniform(-5, 5)

        # Calculate exit premium
        exit_premium = entry_premium * (1 + option_pct_move / 100)

        # Ensure exit premium is reasonable (can't go negative)
        exit_premium = max(exit_premium, 0.01)

        profit_pct = ((exit_premium - entry_premium) / entry_premium) * 100
        profit_dollars = (exit_premium - entry_premium) * 100  # Per contract

        # Determine exit reason
        if profit_pct >= (self.profit_target * 100):
            exit_reason = 'PROFIT_TARGET'
            exit_premium = entry_premium * 1.20  # Cap at 20%
            profit_pct = 20.0
            profit_dollars = (exit_premium - entry_premium) * 100
        elif profit_pct <= (self.stop_loss * 100):
            exit_reason = 'STOP_LOSS'
            exit_premium = entry_premium * 0.70  # Cap at -30%
            profit_pct = -30.0
            profit_dollars = (exit_premium - entry_premium) * 100
        else:
            exit_reason = 'EOD_CLOSE'

        return {
            'entry_premium': entry_premium,
            'exit_premium': exit_premium,
            'strike': strike,
            'profit_pct': profit_pct,
            'profit_dollars': profit_dollars,
            'exit_reason': exit_reason
        }

    def run_backtest(self, start_date='2025-01-01'):
        """Run the backtest"""
        print("=" * 60)
        print("SPY 1DTE STRATEGY BACKTEST")
        print("=" * 60)

        # Get historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        candles = self.get_historical_data(start_date, end_date)

        if len(candles) < 2:
            print("[ERROR] Not enough data for backtest")
            return

        trades = []
        capital = self.initial_capital

        print(f"\n[BACKTEST] Starting capital: ${capital:.2f}")
        print(f"[BACKTEST] Period: {start_date} to {end_date}")
        print(f"[BACKTEST] Total days: {len(candles)}\n")

        # Skip first day (need previous close)
        for i in range(1, len(candles)):
            prev_candle = candles[i-1]
            current_candle = candles[i]

            date = datetime.fromtimestamp(current_candle['datetime'] / 1000).strftime('%Y-%m-%d')

            # Skip weekends (already filtered by API, but double check)
            day_of_week = datetime.fromtimestamp(current_candle['datetime'] / 1000).weekday()
            if day_of_week >= 5:  # Saturday or Sunday
                continue

            # Analyze market direction at open
            direction, confidence, gap = self.analyze_market_direction(current_candle, prev_candle)

            # Get SPY data for the day
            spy_open = current_candle['open']
            spy_close = current_candle['close']
            spy_daily_return = ((spy_close - spy_open) / spy_open)

            # Simulate option trade
            trade_result = self.simulate_option_trade(direction, spy_open, spy_daily_return)

            # Record trade
            trade = {
                'date': date,
                'direction': direction,
                'confidence': confidence,
                'gap': gap,
                'spy_open': spy_open,
                'spy_close': spy_close,
                'spy_return': spy_daily_return * 100,
                'entry_premium': trade_result['entry_premium'],
                'exit_premium': trade_result['exit_premium'],
                'profit_dollars': trade_result['profit_dollars'],
                'profit_pct': trade_result['profit_pct'],
                'exit_reason': trade_result['exit_reason'],
                'capital_before': capital,
                'capital_after': capital + trade_result['profit_dollars']
            }

            capital += trade_result['profit_dollars']
            trades.append(trade)

            # Print trade
            profit_emoji = "+" if trade_result['profit_dollars'] > 0 else ""
            print(f"{date} | {direction:4} | SPY: ${spy_open:6.2f} -> ${spy_close:6.2f} | "
                  f"P/L: {profit_emoji}${trade_result['profit_dollars']:6.2f} ({profit_emoji}{trade_result['profit_pct']:5.1f}%) | "
                  f"{trade_result['exit_reason']:13} | Capital: ${capital:7.2f}")

        # Generate report
        self.generate_report(trades, capital)

        # Save results
        self.save_backtest(trades)

        return trades

    def generate_report(self, trades, final_capital):
        """Generate performance report"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS")
        print("=" * 60)

        if not trades:
            print("No trades executed")
            return

        # Calculate metrics
        total_trades = len(trades)
        winning_trades = [t for t in trades if t['profit_dollars'] > 0]
        losing_trades = [t for t in trades if t['profit_dollars'] < 0]

        win_rate = (len(winning_trades) / total_trades * 100) if total_trades > 0 else 0

        total_profit = sum(t['profit_dollars'] for t in trades)
        avg_win = sum(t['profit_dollars'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit_dollars'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        profit_factor = abs(sum(t['profit_dollars'] for t in winning_trades) /
                           sum(t['profit_dollars'] for t in losing_trades)) if losing_trades and sum(t['profit_dollars'] for t in losing_trades) != 0 else 0

        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        # Exit reasons
        profit_target_exits = len([t for t in trades if t['exit_reason'] == 'PROFIT_TARGET'])
        stop_loss_exits = len([t for t in trades if t['exit_reason'] == 'STOP_LOSS'])
        eod_exits = len([t for t in trades if t['exit_reason'] == 'EOD_CLOSE'])

        # Print report
        print(f"\nTotal Trades: {total_trades}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")

        print(f"\nExit Reasons:")
        print(f"  Profit Target (20%): {profit_target_exits} ({profit_target_exits/total_trades*100:.1f}%)")
        print(f"  Stop Loss (-30%): {stop_loss_exits} ({stop_loss_exits/total_trades*100:.1f}%)")
        print(f"  End of Day: {eod_exits} ({eod_exits/total_trades*100:.1f}%)")

        print(f"\nCapital:")
        print(f"  Starting: ${self.initial_capital:.2f}")
        print(f"  Ending: ${final_capital:.2f}")
        print(f"  Total Profit: ${total_profit:.2f}")
        print(f"  Total Return: {total_return:+.2f}%")

        # Monthly breakdown
        print(f"\nMonthly Performance:")
        monthly_profits = {}
        for trade in trades:
            month = trade['date'][:7]  # YYYY-MM
            if month not in monthly_profits:
                monthly_profits[month] = 0
            monthly_profits[month] += trade['profit_dollars']

        for month in sorted(monthly_profits.keys()):
            print(f"  {month}: ${monthly_profits[month]:+.2f}")

        print("\n" + "=" * 60)

    def save_backtest(self, trades):
        """Save backtest results to file"""
        filename = f"backtest_spy_1dte_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump({
                'strategy': 'SPY_1DTE',
                'start_date': trades[0]['date'] if trades else None,
                'end_date': trades[-1]['date'] if trades else None,
                'total_trades': len(trades),
                'trades': trades
            }, f, indent=2)

        print(f"\n[SAVED] Backtest results saved to {filename}")


def main():
    backtest = SPY1DTEBacktest()
    backtest.run_backtest(start_date='2025-01-01')


if __name__ == '__main__':
    main()
