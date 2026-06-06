import argparse
import logging
from datetime import datetime
from colorama import init, Fore, Style
from backtest import OptionBacktester
from tabulate import tabulate
import json
import sys

init(autoreset=True)

def setup_logging(verbose=False):
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(f'backtest_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
            logging.StreamHandler()
        ]
    )
    return logging.getLogger(__name__)

def print_report(report):
    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}BACKTEST RESULTS{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}Period:{Style.RESET_ALL}")
    period_data = [
        ["Start Date", report['period']['start']],
        ["End Date", report['period']['end']],
        ["Total Days", report['period']['days']]
    ]
    print(tabulate(period_data, tablefmt="grid"))

    print(f"\n{Fore.YELLOW}Performance Summary:{Style.RESET_ALL}")
    perf_data = [
        ["Initial Capital", f"${report['performance']['initial_capital']:,.2f}"],
        ["Final Capital", f"${report['performance']['final_capital']:,.2f}"],
        ["Total P&L", f"${report['performance']['total_pnl']:,.2f}"],
        ["Total Return", f"{report['performance']['total_return']:.2f}%"],
        ["Max Drawdown", f"{report['performance']['max_drawdown']:.2f}%"],
        ["Sharpe Ratio", f"{report['performance']['sharpe_ratio']:.2f}"]
    ]
    print(tabulate(perf_data, tablefmt="grid"))

    print(f"\n{Fore.YELLOW}Trade Statistics:{Style.RESET_ALL}")
    trade_data = [
        ["Total Trades", report['trades']['total']],
        ["Winners", report['trades']['winners']],
        ["Losers", report['trades']['losers']],
        ["Win Rate", f"{report['trades']['win_rate']:.1f}%"],
        ["Average Win", f"${report['trades']['avg_win']:,.2f}"],
        ["Average Loss", f"${report['trades']['avg_loss']:,.2f}"],
        ["Profit Factor", f"{report['trades']['profit_factor']:.2f}"]
    ]
    print(tabulate(trade_data, tablefmt="grid"))

    if report['trades']['total'] > 0:
        print(f"\n{Fore.YELLOW}Recent Trades:{Style.RESET_ALL}")
        recent_trades = report['trade_details'][-5:]
        trade_table = []
        for trade in recent_trades:
            trade_table.append([
                trade['underlying'],
                f"${trade['strike']:.2f}",
                trade['type'].upper(),
                f"${trade['pnl']:.2f}" if trade['pnl'] else "N/A",
                f"{trade['pnl_percent']:.1f}%" if trade['pnl_percent'] else "N/A",
                trade['exit_reason'] or "Open"
            ])
        headers = ["Symbol", "Strike", "Type", "P&L", "Return", "Exit"]
        print(tabulate(trade_table, headers=headers, tablefmt="grid"))

    total_return = report['performance']['total_return']
    if total_return > 0:
        print(f"\n{Fore.GREEN}✓ Strategy profitable: {total_return:.2f}% return{Style.RESET_ALL}")
    else:
        print(f"\n{Fore.RED}✗ Strategy unprofitable: {total_return:.2f}% return{Style.RESET_ALL}")

def main():
    parser = argparse.ArgumentParser(description='Backtest Options Trading Strategy')
    parser.add_argument('--start', type=str, default='2025-01-01',
                       help='Start date (YYYY-MM-DD)')
    parser.add_argument('--end', type=str, default=datetime.now().strftime('%Y-%m-%d'),
                       help='End date (YYYY-MM-DD)')
    parser.add_argument('--tickers', type=str, nargs='+', default=['SPY', 'QQQ', 'AAPL'],
                       help='List of tickers to trade')
    parser.add_argument('--budget', type=float, default=500,
                       help='Budget per trade (default: $500)')
    parser.add_argument('--capital', type=float, default=10000,
                       help='Initial capital (default: $10,000)')
    parser.add_argument('--verbose', action='store_true',
                       help='Enable verbose logging')
    parser.add_argument('--save', action='store_true',
                       help='Save detailed report to JSON file')

    args = parser.parse_args()
    logger = setup_logging(args.verbose)

    print(f"\n{Fore.CYAN}{'='*80}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}OPTIONS STRATEGY BACKTESTER{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*80}{Style.RESET_ALL}\n")

    print(f"{Fore.YELLOW}Configuration:{Style.RESET_ALL}")
    config_data = [
        ["Start Date", args.start],
        ["End Date", args.end],
        ["Tickers", ', '.join(args.tickers)],
        ["Budget per Trade", f"${args.budget:,.2f}"],
        ["Initial Capital", f"${args.capital:,.2f}"]
    ]
    print(tabulate(config_data, tablefmt="grid"))

    try:
        print(f"\n{Fore.YELLOW}Initializing backtester...{Style.RESET_ALL}")
        backtester = OptionBacktester(
            start_date=args.start,
            end_date=args.end,
            budget_per_trade=args.budget
        )
        backtester.initial_capital = args.capital
        backtester.current_capital = args.capital

        print(f"{Fore.GREEN}✓ Backtester initialized{Style.RESET_ALL}")

        print(f"\n{Fore.YELLOW}Running backtest...{Style.RESET_ALL}")
        print(f"This may take several minutes depending on the date range and number of tickers.\n")

        report = backtester.run_backtest(tickers=args.tickers)

        print(f"\n{Fore.GREEN}✓ Backtest complete{Style.RESET_ALL}")

        print_report(report)

        if args.save:
            filename = backtester.save_report(report)
            print(f"\n{Fore.GREEN}✓ Detailed report saved to: {filename}{Style.RESET_ALL}")

        return 0

    except KeyboardInterrupt:
        print(f"\n{Fore.YELLOW}Backtest interrupted by user{Style.RESET_ALL}")
        return 1
    except Exception as e:
        logger.error(f"Error during backtest: {e}", exc_info=True)
        print(f"\n{Fore.RED}✗ Error during backtest: {e}{Style.RESET_ALL}")
        return 1

if __name__ == "__main__":
    sys.exit(main())