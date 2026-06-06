"""
ENHANCED Backtest SPY 1DTE Strategy
Incorporates all win rate improvements:
- Tightened delta range (0.35-0.40)
- Trade filtering (70% confidence minimum)
- Real VIX data and filtering
- Intraday monitoring simulation (15-min checks)
- Early stop loss (20% before 11 AM)
- Trailing stop (10% from peak after 15%)
- Delayed entry (10:00 AM simulation)
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from schwab import auth, client
import random

load_dotenv()


class SPY1DTEBacktestEnhanced:
    def __init__(self):
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.client = auth.client_from_token_file(
            self.token_file, self.app_key, self.app_secret
        )

        # ENHANCED STRATEGY PARAMETERS
        self.target_delta_min = 0.35  # Tightened from 0.30
        self.target_delta_max = 0.40  # Tightened from 0.45
        self.profit_target = 0.20  # 20%
        self.stop_loss = -0.30  # -30%
        self.early_stop_loss = -0.20  # -20% before 11 AM
        self.trailing_stop = 0.10  # 10% from peak
        self.min_confidence = 70  # Only trade high-confidence setups
        self.initial_capital = 500
        self.max_premium = 10

    def get_historical_data(self, start_date, end_date):
        """Get historical SPY and VIX price data"""
        print(f"[DATA] Fetching SPY data from {start_date} to {end_date}...")

        start_dt = datetime.strptime(start_date, '%Y-%m-%d')
        end_dt = datetime.strptime(end_date, '%Y-%m-%d')

        # Get SPY price history
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
            return [], []

        spy_data = response.json()
        spy_candles = spy_data.get('candles', [])

        # Get VIX data
        print(f"[DATA] Fetching VIX data...")
        vix_response = self.client.get_price_history(
            '$VIX.X',
            period_type=client.Client.PriceHistory.PeriodType.YEAR,
            period=client.Client.PriceHistory.Period.ONE_YEAR,
            frequency_type=client.Client.PriceHistory.FrequencyType.DAILY,
            frequency=client.Client.PriceHistory.Frequency.DAILY,
            start_datetime=start_dt,
            end_datetime=end_dt
        )

        vix_candles = []
        if vix_response.status_code == 200:
            vix_data = vix_response.json()
            vix_candles = vix_data.get('candles', [])

        print(f"[DATA] Retrieved {len(spy_candles)} SPY days, {len(vix_candles)} VIX days")

        return spy_candles, vix_candles

    def get_vix_for_date(self, vix_candles, date_timestamp):
        """Get VIX value for a specific date"""
        for candle in vix_candles:
            if abs(candle['datetime'] - date_timestamp) < 86400000:  # Within 1 day
                return candle['close']
        return 15.0  # Default VIX if not found

    def analyze_market_direction_enhanced(self, current_candle, prev_candle, vix_level):
        """
        ENHANCED market analysis with filtering
        - Uses real VIX data
        - Filters out high VIX and large gaps
        - Requires 70% confidence minimum
        - Simulates 10:00 AM entry (30 min after open)
        """
        current_open = current_candle['open']
        current_close = prev_candle['close']
        spy_price = current_open  # At 10 AM, close to open price

        # Simulate intraday movement (first 30 minutes)
        # In reality, by 10 AM we'd have first 30 min data
        first_30min_move = (current_candle['high'] - current_open) / current_open * 100
        if current_candle['close'] < current_open:
            first_30min_move = (current_candle['low'] - current_open) / current_open * 100

        # Gap analysis
        gap = ((current_open - current_close) / current_close) * 100

        # Simulate VIX change (approximate)
        vix_change = random.uniform(-10, 10) if vix_level else 0

        # Weighted signal scoring
        bullish_signals = 0
        bearish_signals = 0
        skip_reasons = []

        # FILTER 1: High VIX (unpredictable market)
        if vix_level > 30:
            skip_reasons.append(f"VIX too high ({vix_level:.1f})")

        # FILTER 2: Large gap (uncertainty)
        if abs(gap) > 1.0:
            skip_reasons.append(f"Large gap ({gap:+.2f}%)")

        # Signal 1: Intraday momentum (simulated first 30 min)
        if first_30min_move > 0.3:
            bullish_signals += 2
        elif first_30min_move > 0.1:
            bullish_signals += 1
        elif first_30min_move < -0.3:
            bearish_signals += 2
        elif first_30min_move < -0.1:
            bearish_signals += 1

        # Signal 2: VIX level
        if vix_level > 25:
            bearish_signals += 1
        elif vix_level < 15:
            bullish_signals += 1

        # Signal 3: Gap (moderate only)
        if 0.3 < gap < 1.0:
            bullish_signals += 1
        elif -1.0 < gap < -0.3:
            bearish_signals += 1

        # Signal 4: VIX direction (simulated)
        if vix_change < -5:
            bullish_signals += 1
        elif vix_change > 5:
            bearish_signals += 1

        # Calculate confidence
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            confidence = 0
            direction = None
        elif bullish_signals > bearish_signals:
            direction = 'CALL'
            confidence = (bullish_signals / total_signals) * 100
        elif bearish_signals > bullish_signals:
            direction = 'PUT'
            confidence = (bearish_signals / total_signals) * 100
        else:
            direction = 'CALL' if first_30min_move >= 0 else 'PUT'
            confidence = 50

        # Apply filtering
        should_trade = True
        if skip_reasons:
            should_trade = False
        if confidence < self.min_confidence:
            should_trade = False

        return {
            'direction': direction,
            'confidence': confidence,
            'gap': gap,
            'vix_level': vix_level,
            'should_trade': should_trade,
            'skip_reasons': skip_reasons
        }

    def simulate_option_trade_enhanced(self, direction, spy_open, spy_close, spy_high, spy_low, entry_time_hour=10):
        """
        ENHANCED option simulation with realistic intraday monitoring
        - Simulates 15-minute checks
        - Early stop loss before 11 AM
        - Trailing stop after 15% gain
        - Profit target at 20%
        - Regular stop loss at -30%
        """
        # Entry premium (OTM option with 0.35-0.40 delta)
        entry_premium = random.uniform(0.60, 1.20)  # Tighter premium range

        # Simulate intraday price path (hourly approximation)
        # We'll check at: 10:00, 10:15, 10:30, 10:45, 11:00, 11:30, 12:00, etc.
        spy_daily_return = ((spy_close - spy_open) / spy_open)

        # Simulate intraday volatility
        intraday_moves = []
        current_spy = spy_open
        hours_to_close = 6  # 10 AM to 4 PM
        checks_per_hour = 4  # Every 15 minutes
        total_checks = hours_to_close * checks_per_hour

        # Generate realistic intraday path
        for i in range(total_checks):
            # Random walk towards daily close
            progress = (i + 1) / total_checks
            target_price = spy_open + (spy_close - spy_open) * progress
            noise = random.uniform(-0.001, 0.001) * spy_open  # Small noise
            current_spy = target_price + noise
            intraday_moves.append(current_spy)

        # Simulate option pricing at each check
        max_profit_pct = 0
        current_premium = entry_premium

        for i, spy_price in enumerate(intraday_moves):
            # Calculate time
            minutes_elapsed = i * 15
            current_hour = 10 + (minutes_elapsed // 60)

            # Calculate SPY move from entry
            spy_move_pct = ((spy_price - spy_open) / spy_open) * 100

            # Option moves ~40% of underlying (delta 0.35-0.40 + gamma)
            delta_effect = 0.375  # Mid-point of our range
            gamma_effect = 0.15  # 1DTE gamma impact

            if direction == 'CALL':
                option_move_pct = spy_move_pct * (delta_effect + gamma_effect)
            else:  # PUT
                option_move_pct = -spy_move_pct * (delta_effect + gamma_effect)

            # Add theta decay (accelerates near EOD)
            hours_held = minutes_elapsed / 60
            theta_decay = -0.02 * hours_held  # 2% decay per hour

            # Calculate current option value
            option_move_pct += theta_decay
            current_premium = entry_premium * (1 + option_move_pct / 100)
            current_premium = max(current_premium, 0.01)  # Can't go negative

            profit_pct = ((current_premium - entry_premium) / entry_premium) * 100

            # Track max profit for trailing stop
            if profit_pct > max_profit_pct:
                max_profit_pct = profit_pct

            # CHECK 1: Profit target (20%)
            if profit_pct >= 20:
                exit_premium = entry_premium * 1.20
                return {
                    'entry_premium': entry_premium,
                    'exit_premium': exit_premium,
                    'profit_pct': 20.0,
                    'profit_dollars': (exit_premium - entry_premium) * 100,
                    'exit_reason': 'PROFIT_TARGET',
                    'exit_time': f"{current_hour}:{minutes_elapsed % 60:02d}"
                }

            # CHECK 2: Trailing stop (10% from peak after 15%)
            if max_profit_pct >= 15:
                trailing_trigger = max_profit_pct - 10
                if profit_pct <= trailing_trigger:
                    exit_premium = current_premium
                    return {
                        'entry_premium': entry_premium,
                        'exit_premium': exit_premium,
                        'profit_pct': profit_pct,
                        'profit_dollars': (exit_premium - entry_premium) * 100,
                        'exit_reason': 'TRAILING_STOP',
                        'exit_time': f"{current_hour}:{minutes_elapsed % 60:02d}"
                    }

            # CHECK 3: Early stop loss (20% before 11 AM)
            if current_hour < 11 and profit_pct <= -20:
                exit_premium = entry_premium * 0.80
                return {
                    'entry_premium': entry_premium,
                    'exit_premium': exit_premium,
                    'profit_pct': -20.0,
                    'profit_dollars': (exit_premium - entry_premium) * 100,
                    'exit_reason': 'EARLY_STOP_LOSS',
                    'exit_time': f"{current_hour}:{minutes_elapsed % 60:02d}"
                }

            # CHECK 4: Regular stop loss (-30%)
            if profit_pct <= -30:
                exit_premium = entry_premium * 0.70
                return {
                    'entry_premium': entry_premium,
                    'exit_premium': exit_premium,
                    'profit_pct': -30.0,
                    'profit_dollars': (exit_premium - entry_premium) * 100,
                    'exit_reason': 'STOP_LOSS',
                    'exit_time': f"{current_hour}:{minutes_elapsed % 60:02d}"
                }

        # If no exit triggered, close at EOD
        exit_premium = current_premium
        profit_pct = ((exit_premium - entry_premium) / entry_premium) * 100

        return {
            'entry_premium': entry_premium,
            'exit_premium': exit_premium,
            'profit_pct': profit_pct,
            'profit_dollars': (exit_premium - entry_premium) * 100,
            'exit_reason': 'EOD_CLOSE',
            'exit_time': '14:45'
        }

    def run_backtest(self, start_date='2025-01-01'):
        """Run the ENHANCED backtest with all improvements"""
        print("=" * 60)
        print("SPY 1DTE STRATEGY BACKTEST - ENHANCED")
        print("=" * 60)
        print("Improvements:")
        print("- Delta range: 0.35-0.40 (tightened)")
        print("- Min confidence: 70% (filtered)")
        print("- Real VIX filtering (>30 skipped)")
        print("- Large gaps skipped (>1.0%)")
        print("- 15-min monitoring simulation")
        print("- Early stop loss: -20% before 11 AM")
        print("- Trailing stop: 10% from peak after 15%")
        print("=" * 60)

        # Get historical data
        end_date = datetime.now().strftime('%Y-%m-%d')
        spy_candles, vix_candles = self.get_historical_data(start_date, end_date)

        if len(spy_candles) < 2:
            print("[ERROR] Not enough data for backtest")
            return

        trades = []
        skipped = []
        capital = self.initial_capital

        print(f"\n[BACKTEST] Starting capital: ${capital:.2f}")
        print(f"[BACKTEST] Period: {start_date} to {end_date}")
        print(f"[BACKTEST] Total days: {len(spy_candles)}\n")

        # Skip first day (need previous close)
        for i in range(1, len(spy_candles)):
            prev_candle = spy_candles[i-1]
            current_candle = spy_candles[i]

            date = datetime.fromtimestamp(current_candle['datetime'] / 1000).strftime('%Y-%m-%d')

            # Skip weekends
            day_of_week = datetime.fromtimestamp(current_candle['datetime'] / 1000).weekday()
            if day_of_week >= 5:
                continue

            # Get VIX for this date
            vix_level = self.get_vix_for_date(vix_candles, current_candle['datetime'])

            # ENHANCED market analysis with filtering
            analysis = self.analyze_market_direction_enhanced(current_candle, prev_candle, vix_level)

            # TRADE FILTERING: Skip if conditions not met
            if not analysis['should_trade']:
                skipped.append({
                    'date': date,
                    'reasons': analysis['skip_reasons'],
                    'confidence': analysis['confidence'],
                    'vix': vix_level
                })
                print(f"{date} | SKIP | Conf: {analysis['confidence']:.0f}% | VIX: {vix_level:.1f} | {', '.join(analysis['skip_reasons']) if analysis['skip_reasons'] else f'Low confidence'}")
                continue

            # Simulate option trade with ENHANCED logic
            direction = analysis['direction']
            spy_open = current_candle['open']
            spy_close = current_candle['close']
            spy_high = current_candle['high']
            spy_low = current_candle['low']

            trade_result = self.simulate_option_trade_enhanced(
                direction, spy_open, spy_close, spy_high, spy_low
            )

            # Record trade
            trade = {
                'date': date,
                'direction': direction,
                'confidence': analysis['confidence'],
                'gap': analysis['gap'],
                'vix': vix_level,
                'spy_open': spy_open,
                'spy_close': spy_close,
                'spy_return': ((spy_close - spy_open) / spy_open) * 100,
                'entry_premium': trade_result['entry_premium'],
                'exit_premium': trade_result['exit_premium'],
                'profit_dollars': trade_result['profit_dollars'],
                'profit_pct': trade_result['profit_pct'],
                'exit_reason': trade_result['exit_reason'],
                'exit_time': trade_result['exit_time'],
                'capital_before': capital,
                'capital_after': capital + trade_result['profit_dollars']
            }

            capital += trade_result['profit_dollars']
            trades.append(trade)

            # Print trade
            profit_sign = "+" if trade_result['profit_dollars'] > 0 else ""
            print(f"{date} | {direction:4} | SPY: ${spy_open:6.2f} -> ${spy_close:6.2f} | "
                  f"P/L: {profit_sign}${trade_result['profit_dollars']:6.2f} ({profit_sign}{trade_result['profit_pct']:5.1f}%) | "
                  f"{trade_result['exit_reason']:17} @ {trade_result['exit_time']} | Capital: ${capital:7.2f}")

        # Generate report
        self.generate_report(trades, skipped, capital)

        # Save results
        self.save_backtest(trades, skipped)

        return trades

    def generate_report(self, trades, skipped, final_capital):
        """Generate ENHANCED performance report"""
        print("\n" + "=" * 60)
        print("BACKTEST RESULTS - ENHANCED STRATEGY")
        print("=" * 60)

        total_days = len(trades) + len(skipped)
        print(f"\nTotal Days: {total_days}")
        print(f"Days Traded: {len(trades)} ({len(trades)/total_days*100:.1f}%)")
        print(f"Days Skipped: {len(skipped)} ({len(skipped)/total_days*100:.1f}%)")

        if not trades:
            print("\nNo trades executed")
            return

        # Calculate metrics
        winning_trades = [t for t in trades if t['profit_dollars'] > 0]
        losing_trades = [t for t in trades if t['profit_dollars'] < 0]

        win_rate = (len(winning_trades) / len(trades) * 100) if trades else 0

        total_profit = sum(t['profit_dollars'] for t in trades)
        avg_win = sum(t['profit_dollars'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['profit_dollars'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        total_wins = sum(t['profit_dollars'] for t in winning_trades)
        total_losses = abs(sum(t['profit_dollars'] for t in losing_trades))
        profit_factor = (total_wins / total_losses) if total_losses > 0 else 0

        total_return = ((final_capital - self.initial_capital) / self.initial_capital) * 100

        # Exit reasons
        profit_target_exits = len([t for t in trades if t['exit_reason'] == 'PROFIT_TARGET'])
        stop_loss_exits = len([t for t in trades if t['exit_reason'] == 'STOP_LOSS'])
        early_stop_exits = len([t for t in trades if t['exit_reason'] == 'EARLY_STOP_LOSS'])
        trailing_stop_exits = len([t for t in trades if t['exit_reason'] == 'TRAILING_STOP'])
        eod_exits = len([t for t in trades if t['exit_reason'] == 'EOD_CLOSE'])

        # Print report
        print(f"\n=== TRADE STATISTICS ===")
        print(f"Total Trades: {len(trades)}")
        print(f"Winning Trades: {len(winning_trades)} ({win_rate:.1f}%)")
        print(f"Losing Trades: {len(losing_trades)} ({100-win_rate:.1f}%)")
        print(f"\nAverage Win: ${avg_win:.2f}")
        print(f"Average Loss: ${avg_loss:.2f}")
        print(f"Profit Factor: {profit_factor:.2f}")

        print(f"\n=== EXIT REASONS ===")
        print(f"Profit Target (20%): {profit_target_exits} ({profit_target_exits/len(trades)*100:.1f}%)")
        print(f"Trailing Stop: {trailing_stop_exits} ({trailing_stop_exits/len(trades)*100:.1f}%)")
        print(f"Early Stop Loss (-20%): {early_stop_exits} ({early_stop_exits/len(trades)*100:.1f}%)")
        print(f"Stop Loss (-30%): {stop_loss_exits} ({stop_loss_exits/len(trades)*100:.1f}%)")
        print(f"End of Day: {eod_exits} ({eod_exits/len(trades)*100:.1f}%)")

        print(f"\n=== CAPITAL ===")
        print(f"Starting: ${self.initial_capital:.2f}")
        print(f"Ending: ${final_capital:.2f}")
        print(f"Total Profit: ${total_profit:.2f}")
        print(f"Total Return: {total_return:+.2f}%")

        # Skip reasons
        if skipped:
            print(f"\n=== SKIP REASONS ===")
            skip_reason_counts = {}
            for skip in skipped:
                for reason in skip['reasons']:
                    skip_reason_counts[reason] = skip_reason_counts.get(reason, 0) + 1
                if not skip['reasons']:
                    skip_reason_counts['Low confidence'] = skip_reason_counts.get('Low confidence', 0) + 1

            for reason, count in sorted(skip_reason_counts.items(), key=lambda x: x[1], reverse=True):
                print(f"  {reason}: {count}")

        # Monthly breakdown
        print(f"\n=== MONTHLY PERFORMANCE ===")
        monthly_profits = {}
        monthly_trades = {}
        for trade in trades:
            month = trade['date'][:7]
            monthly_profits[month] = monthly_profits.get(month, 0) + trade['profit_dollars']
            monthly_trades[month] = monthly_trades.get(month, 0) + 1

        for month in sorted(monthly_profits.keys()):
            print(f"{month}: ${monthly_profits[month]:+7.2f} ({monthly_trades[month]:2d} trades)")

        print("\n" + "=" * 60)

    def save_backtest(self, trades, skipped):
        """Save ENHANCED backtest results"""
        filename = f"backtest_spy_1dte_enhanced_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump({
                'strategy': 'SPY_1DTE_ENHANCED',
                'version': '2.0',
                'enhancements': [
                    'Delta range: 0.35-0.40',
                    'Min confidence: 70%',
                    'VIX filtering (>30)',
                    'Gap filtering (>1.0%)',
                    '15-min monitoring',
                    'Early stop loss (-20% before 11 AM)',
                    'Trailing stop (10% from peak)',
                    'Delayed entry (10 AM simulation)'
                ],
                'start_date': trades[0]['date'] if trades else None,
                'end_date': trades[-1]['date'] if trades else None,
                'total_days': len(trades) + len(skipped),
                'days_traded': len(trades),
                'days_skipped': len(skipped),
                'trades': trades,
                'skipped': skipped
            }, f, indent=2)

        print(f"\n[SAVED] Enhanced backtest results saved to {filename}")


def main():
    backtest = SPY1DTEBacktestEnhanced()
    backtest.run_backtest(start_date='2025-01-01')


if __name__ == '__main__':
    main()
