"""
Simple Options Trading Backtest Simulator
This simulates the strategy without actual API calls for demonstration
"""

from datetime import datetime, timedelta
import random
import json

class SimpleBacktest:
    def __init__(self, start_date, end_date, budget=500):
        self.start_date = datetime.strptime(start_date, '%Y-%m-%d')
        self.end_date = datetime.strptime(end_date, '%Y-%m-%d')
        self.budget = budget
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.trades = []

    def simulate_option_selection(self, ticker, date):
        """Simulate finding an ITM option"""
        # Simulate realistic option parameters
        days_to_expiry = random.randint(30, 60)
        delta = random.uniform(0.4, 0.7)  # ITM options typically have higher delta
        iv = random.uniform(0.15, 0.35)  # Implied volatility

        # Simulate option price (typically 1-5% of stock price for ITM)
        stock_price = self.get_simulated_stock_price(ticker, date)
        option_price = stock_price * random.uniform(0.01, 0.03)

        # Calculate how many contracts we can buy
        contracts = int(self.budget / (option_price * 100))
        if contracts == 0:
            return None

        return {
            'ticker': ticker,
            'strike': stock_price * random.uniform(0.95, 0.99),  # ITM strike
            'expiry_days': days_to_expiry,
            'delta': delta,
            'iv': iv,
            'price': option_price,
            'contracts': contracts,
            'cost': contracts * option_price * 100
        }

    def get_simulated_stock_price(self, ticker, date):
        """Generate realistic stock prices"""
        base_prices = {
            'SPY': 450 + (date - self.start_date).days * 0.1,
            'QQQ': 380 + (date - self.start_date).days * 0.08,
            'AAPL': 180 + (date - self.start_date).days * 0.05
        }
        price = base_prices.get(ticker, 100)
        # Add some volatility
        return price * (1 + random.uniform(-0.02, 0.02))

    def simulate_trade_outcome(self, trade_info):
        """Simulate trade P&L based on Greeks and market conditions"""
        # Probability of profit based on delta
        win_probability = trade_info['delta'] + 0.1  # Slight edge

        if random.random() < win_probability:
            # Winning trade: 20-80% return
            return_pct = random.uniform(0.2, 0.8)
            pnl = trade_info['cost'] * return_pct
            exit_reason = "Target reached"
        else:
            # Losing trade: -30% to -70% loss
            return_pct = random.uniform(-0.7, -0.3)
            pnl = trade_info['cost'] * return_pct
            exit_reason = "Stop loss"

        return pnl, return_pct * 100, exit_reason

    def run(self, tickers):
        """Run the backtest simulation"""
        current_date = self.start_date
        trades_log = []
        equity_curve = []

        print(f"\nRunning backtest from {self.start_date.date()} to {self.end_date.date()}")
        print("="*60)

        while current_date <= self.end_date:
            # Trade weekly on Mondays
            if current_date.weekday() == 0:
                for ticker in tickers:
                    if self.current_capital > self.budget:
                        option = self.simulate_option_selection(ticker, current_date)
                        if option:
                            # Enter trade
                            self.current_capital -= option['cost']

                            # Simulate outcome
                            pnl, return_pct, exit_reason = self.simulate_trade_outcome(option)
                            self.current_capital += option['cost'] + pnl

                            trade_record = {
                                'date': current_date.strftime('%Y-%m-%d'),
                                'ticker': ticker,
                                'contracts': option['contracts'],
                                'cost': option['cost'],
                                'pnl': pnl,
                                'return_pct': return_pct,
                                'exit_reason': exit_reason
                            }
                            trades_log.append(trade_record)

                            status = "WIN" if pnl > 0 else "LOSS"
                            print(f"{current_date.date()} | {ticker} | {status} | PnL: ${pnl:,.2f} ({return_pct:.1f}%)")
                            break  # One trade per week

            equity_curve.append({
                'date': current_date.strftime('%Y-%m-%d'),
                'capital': self.current_capital
            })

            current_date += timedelta(days=1)

        return self.generate_report(trades_log, equity_curve)

    def generate_report(self, trades, equity_curve):
        """Generate performance report"""
        if not trades:
            return {'error': 'No trades executed'}

        total_trades = len(trades)
        winning_trades = [t for t in trades if t['pnl'] > 0]
        losing_trades = [t for t in trades if t['pnl'] < 0]

        total_pnl = sum(t['pnl'] for t in trades)
        win_rate = len(winning_trades) / total_trades * 100 if total_trades > 0 else 0

        avg_win = sum(t['pnl'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            'summary': {
                'start_date': self.start_date.strftime('%Y-%m-%d'),
                'end_date': self.end_date.strftime('%Y-%m-%d'),
                'initial_capital': self.initial_capital,
                'final_capital': self.current_capital,
                'total_pnl': total_pnl,
                'total_return_pct': (total_pnl / self.initial_capital) * 100
            },
            'trades': {
                'total': total_trades,
                'winners': len(winning_trades),
                'losers': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss
            },
            'trades_log': trades[-10:],  # Last 10 trades
            'equity_curve': equity_curve[-30:]  # Last 30 days
        }

def main():
    print("\n" + "="*60)
    print("OPTIONS TRADING STRATEGY BACKTEST (SIMULATION)")
    print("="*60)

    # Run backtest from Jan 1, 2025 to Jan 24, 2025
    backtest = SimpleBacktest(
        start_date='2025-01-01',
        end_date='2025-01-24',
        budget=500
    )

    tickers = ['SPY', 'QQQ', 'AAPL']
    print(f"\nTesting with tickers: {', '.join(tickers)}")
    print(f"Budget per trade: $500")
    print(f"Initial capital: $10,000")

    report = backtest.run(tickers)

    # Print results
    print("\n" + "="*60)
    print("BACKTEST RESULTS")
    print("="*60)

    summary = report['summary']
    print(f"\nPeriod: {summary['start_date']} to {summary['end_date']}")
    print(f"Initial Capital: ${summary['initial_capital']:,.2f}")
    print(f"Final Capital: ${summary['final_capital']:,.2f}")
    print(f"Total P&L: ${summary['total_pnl']:,.2f}")
    print(f"Total Return: {summary['total_return_pct']:.2f}%")

    trades = report['trades']
    print(f"\nTotal Trades: {trades['total']}")
    print(f"Winners: {trades['winners']}")
    print(f"Losers: {trades['losers']}")
    print(f"Win Rate: {trades['win_rate']:.1f}%")
    print(f"Average Win: ${trades['avg_win']:,.2f}")
    print(f"Average Loss: ${trades['avg_loss']:,.2f}")

    # Save report
    with open('backtest_report.json', 'w') as f:
        json.dump(report, f, indent=2)
    print("\n[OK] Full report saved to backtest_report.json")

    # Performance assessment
    if summary['total_pnl'] > 0:
        print(f"\n[OK] Strategy shows positive returns: {summary['total_return_pct']:.2f}%")
    else:
        print(f"\n[X] Strategy shows negative returns: {summary['total_return_pct']:.2f}%")

if __name__ == "__main__":
    main()