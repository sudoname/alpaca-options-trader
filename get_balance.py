"""
Get Schwab Account Balance
"""

import os
from dotenv import load_dotenv
from schwab import auth, client

load_dotenv()

app_key = os.getenv('SCHWAB_APP_KEY')
app_secret = os.getenv('SCHWAB_APP_SECRET')
token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
account_hash = os.getenv('SCHWAB_ACCOUNT_HASH')

if not account_hash:
    print("❌ SCHWAB_ACCOUNT_HASH not found in .env file")
    print("Run 'python get_account_hash.py' to get your account hash")
    exit(1)

print("Connecting to Schwab...")
schwab_client = auth.client_from_token_file(token_file, app_key, app_secret)

print(f"Fetching account balance for {account_hash[:8]}...")
response = schwab_client.get_account(account_hash, fields=client.Client.Account.Fields.POSITIONS)

if response.status_code == 200:
    account_data = response.json()

    # Extract balance information
    securities_account = account_data.get('securitiesAccount', {})

    print("\n" + "=" * 60)
    print("SCHWAB ACCOUNT BALANCE")
    print("=" * 60)

    # Account type and number
    account_type = securities_account.get('type', 'N/A')
    print(f"\nAccount Type: {account_type}")

    # Current balances
    current_balances = securities_account.get('currentBalances', {})

    if current_balances:
        print(f"\nACCOUNT BALANCES:")
        print(f"  Cash Balance: ${current_balances.get('cashBalance', 0):,.2f}")
        print(f"  Liquidation Value: ${current_balances.get('liquidationValue', 0):,.2f}")
        print(f"  Market Value: ${current_balances.get('longMarketValue', 0):,.2f}")
        print(f"  Buying Power: ${current_balances.get('buyingPower', 0):,.2f}")
        print(f"  Available Funds: ${current_balances.get('availableFunds', 0):,.2f}")

    # Positions
    positions = securities_account.get('positions', [])

    if positions:
        print(f"\nOPEN POSITIONS ({len(positions)}):")
        for pos in positions:
            instrument = pos.get('instrument', {})
            symbol = instrument.get('symbol', 'N/A')
            asset_type = instrument.get('assetType', 'N/A')
            quantity = pos.get('longQuantity', 0)
            market_value = pos.get('marketValue', 0)
            avg_price = pos.get('averagePrice', 0)
            current_price = pos.get('currentDayProfitLoss', 0)

            print(f"\n  {symbol} ({asset_type})")
            print(f"    Quantity: {quantity}")
            print(f"    Market Value: ${market_value:,.2f}")
            print(f"    Avg Price: ${avg_price:,.2f}")
            print(f"    Day P/L: ${current_price:,.2f}")
    else:
        print("\nNo open positions")

    print("\n" + "=" * 60)

else:
    print(f"❌ Error: {response.status_code}")
    print(response.text)
