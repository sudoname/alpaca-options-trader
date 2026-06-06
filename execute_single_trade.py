"""
Execute a single trade on Schwab account
"""

from schwab_trader import SchwabOptionsTrader
import json

print("=" * 60)
print("EXECUTING SINGLE TRADE FROM ACCOUNT ...879")
print("=" * 60)

# Load tickers
with open('supported_tickers.json', 'r') as f:
    data = json.load(f)
    tickers = data['tickers'][:5]  # Top 5 tickers

print(f"\nScanning {len(tickers)} tickers: {', '.join(tickers)}")

# Initialize trader (will use account from .env)
trader = SchwabOptionsTrader(dry_run=False)  # LIVE TRADING

print("\nFinding best trade opportunity...")

# Find best trade across all tickers
best_option = trader.find_best_trade(
    tickers=tickers,
    option_type='PUT',  # Or 'CALL'
    budget=500.0,
    min_days=30,
    max_days=90,
    min_delta=0.30,
    max_delta=0.70,
    max_iv=60.0
)

if not best_option:
    print("\n[X] No suitable trade found")
    exit(1)

print(f"\n[OK] Best option found:")
print(f"   Ticker: {best_option['ticker']}")
print(f"   Strike: ${best_option['strike']}")
print(f"   Type: {best_option['type']}")
print(f"   Expiration: {best_option['expiration'][:10]}")
print(f"   Premium: ${best_option['ask']:.2f}")
print(f"   Cost: ${best_option['ask'] * 100:.2f}")
print(f"   Score: {best_option['score']:.2f}/100")

# Execute the trade
print("\n" + "=" * 60)
print("READY TO EXECUTE TRADE")
print("=" * 60)

print("\n[EXECUTING] Placing order...")
result = trader.execute_trade(best_option, quantity=1)

if result and result.get('success'):
    print("\n[SUCCESS] TRADE EXECUTED!")
    print(f"   Order ID: {result.get('order_id', 'N/A')}")
    print(f"   Symbol: {result.get('symbol', 'N/A')}")
    print(f"   Quantity: 1 contract")
    print(f"   Premium: ${best_option['ask']:.2f}")
    print(f"   Total Cost: ${best_option['ask'] * 100:.2f}")
else:
    print("\n[FAILED] Trade execution failed")
    print(f"   Error: {result.get('error', 'Unknown error')}")

print("\n" + "=" * 60)
