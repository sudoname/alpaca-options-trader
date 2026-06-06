"""
Manual Runner for SPY 1DTE Strategy
Run this anytime to execute the strategy immediately
"""

from spy_1dte_strategy import SPY1DTEStrategy

if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("MANUAL SPY 1DTE STRATEGY EXECUTION")
    print("=" * 60)

    confirmation = input("\nRun SPY 1DTE strategy now? (yes/no): ")

    if confirmation.lower() != 'yes':
        print("Cancelled by user")
        exit(0)

    strategy = SPY1DTEStrategy()
    strategy.run_daily_strategy()
