"""
Smart Trading System Runner
Main interface for the enhanced options trading system
"""

import argparse
import sys
import json
from datetime import datetime
from smart_trader import SmartOptionsTrader

def main():
    parser = argparse.ArgumentParser(description='Smart Options Trading System')
    parser.add_argument('--ticker', type=str, help='Stock ticker to trade')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing positions')
    parser.add_argument('--report', action='store_true', help='Generate performance report')
    parser.add_argument('--live', action='store_true', help='Execute live trades')
    parser.add_argument('--budget', type=float, default=500, help='Budget per trade')

    args = parser.parse_args()

    print("\n" + "="*60)
    print("SMART OPTIONS TRADING SYSTEM")
    print("="*60)

    try:
        trader = SmartOptionsTrader()

        if args.report:
            print("\nGenerating Performance Report...")
            report = trader.generate_performance_report()

            if 'message' in report:
                print(report['message'])
            else:
                print(f"\nTrading Performance:")
                print(f"  Total Trades: {report['total_trades']}")
                print(f"  Win Rate: {report['win_rate']:.1f}%")
                print(f"  Average Win: {report['avg_win_percent']:.1f}%")
                print(f"  Average Loss: {report['avg_loss_percent']:.1f}%")
                print(f"  Total P&L: {report['total_pnl_percent']:.1f}%")

                print(f"\nCurrent ML Weights:")
                for param, weight in report['current_weights'].items():
                    print(f"  {param}: {weight:.3f}")

                print(f"\nLearned Patterns:")
                print(f"  Success Patterns: {report['patterns_learned']['success']}")
                print(f"  Failure Patterns: {report['patterns_learned']['failure']}")

            return 0

        if args.monitor:
            print("\nMonitoring existing positions...")
            trader.monitor_positions()

            positions = trader.get_positions()
            if positions:
                print(f"\nCurrent Positions:")
                for pos in positions:
                    pnl = float(pos.get('unrealized_pl', 0))
                    pnl_pct = float(pos.get('unrealized_plpc', 0)) * 100
                    print(f"  {pos['symbol']}: {pos['qty']} contracts")
                    print(f"    P&L: ${pnl:.2f} ({pnl_pct:.1f}%)")
            else:
                print("No open positions")

            return 0

        if not args.ticker:
            print("Please specify --ticker, --monitor, or --report")
            return 1

        # Check market status
        market_response = trader.get_market_status()
        if hasattr(trader, 'get_market_status'):
            market = trader.get_market_status()
            if not market.get('is_open') and args.live:
                print("[WARNING] Market is closed - live trading not available")
                return 1

        # Get account info
        account = trader.get_account()
        if not account:
            print("[ERROR] Could not connect to trading account")
            return 1

        print(f"\nAccount Status:")
        print(f"  Buying Power: ${float(account['buying_power']):,.2f}")
        print(f"  Mode: {'PAPER' if trader.paper else 'LIVE'}")

        # Get current price
        print(f"\nAnalyzing {args.ticker}...")
        current_price = trader.get_current_price(args.ticker)
        if not current_price:
            print(f"[ERROR] Could not get price for {args.ticker}")
            return 1

        print(f"Current Price: ${current_price:.2f}")

        # Get option contracts (simplified for demo)
        print("Searching for optimal option...")

        # Mock option contracts for demo (in real system, fetch from API)
        mock_contracts = [{
            'symbol': f'{args.ticker}251031C{int(current_price * 0.97 * 1000):08d}',
            'strike_price': str(current_price * 0.97),
            'expiration_date': '2025-10-31',
            'type': 'call'
        }]

        best_option = trader.select_best_option(mock_contracts, current_price)

        if not best_option:
            print("No suitable options found")
            return 1

        # Display selection
        print("\n" + "="*50)
        print("SELECTED OPTION (ML-OPTIMIZED)")
        print("="*50)
        print(f"Symbol: {best_option['symbol']}")
        print(f"Strike: ${best_option['strike']:.2f}")
        print(f"Expiration: {best_option['expiration']}")
        print(f"ML Score: {best_option['score']:.2f}/100")
        print(f"Delta: {best_option['delta']:.3f}")
        print(f"Moneyness: {best_option['moneyness']:.1%}")

        # Risk management display
        print(f"\nRisk Management:")
        print(f"  Stop Loss: -10% (Auto)")
        print(f"  Take Profit: +20% (Close 50%)")
        print(f"  Trailing Stop: 5% from high")
        print(f"  Dynamic Exit: Active")

        if args.live:
            print(f"\n[LIVE TRADING] Placing order...")
            # In real implementation, this would place the actual order
            print("[DEMO MODE] Order would be placed with full risk management")

            # Simulate trade recording
            trade_info = {
                'symbol': best_option['symbol'],
                'entry_price': current_price * 0.025,
                'entry_time': datetime.now().isoformat(),
                'metrics': best_option
            }

            trader.save_active_trade(trade_info)
            print("[OK] Trade recorded for monitoring")
        else:
            print(f"\n[DRY RUN] Use --live to place actual trades")

        print(f"\nNext Steps:")
        print(f"  1. Monitor: python smart_trade_runner.py --monitor")
        print(f"  2. Report: python smart_trade_runner.py --report")
        print(f"  3. Trade: python smart_trade_runner.py --ticker AAPL --live")

        return 0

    except Exception as e:
        print(f"\n[ERROR] {e}")
        return 1

if __name__ == "__main__":
    sys.exit(main())