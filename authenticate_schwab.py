"""
Complete Schwab authentication with redirect URL
"""

import os
from dotenv import load_dotenv
from schwab import auth
import json

# Load environment variables
load_dotenv()

app_key = os.getenv('SCHWAB_APP_KEY')
app_secret = os.getenv('SCHWAB_APP_SECRET')
callback_url = os.getenv('SCHWAB_CALLBACK_URL')
token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')

# The redirect URL from browser
redirect_url = "https://127.0.0.1:5000/callback?code=C0.b2F1dGgyLmJkYy5zY2h3YWIuY29t._HJYuTzg1ytbOtn4pCVUswnxqXk1anF5MVoQs9cH6i0%40&session=dbd1a471-50ce-45d7-ace0-4f49d663502f&state=Ktt54nzlGp2XMJRnILzHLvLJhfyCKf"

# Extract state from URL
import urllib.parse
parsed_url = urllib.parse.urlparse(redirect_url)
query_params = urllib.parse.parse_qs(parsed_url.query)
state = query_params.get('state', [None])[0]

print("Authenticating with Schwab...")
print(f"Callback URL: {callback_url}")
print(f"Token file: {token_file}")
print(f"State: {state}")

try:
    # Create token write function
    def token_write_func(token_dict):
        with open(token_file, 'w') as f:
            json.dump(token_dict, f)
        print(f"Tokens written to {token_file}")

    # Get auth context with the state from the URL
    auth_context = auth.get_auth_context(app_key, callback_url, state=state)

    # Create client from the redirect URL
    client = auth.client_from_received_url(
        api_key=app_key,
        app_secret=app_secret,
        auth_context=auth_context,
        received_url=redirect_url,
        token_write_func=token_write_func
    )

    print("\n[SUCCESS] Authentication successful!")
    print(f"Tokens saved to: {token_file}")
    print("\nYou can now use the Schwab API. Testing connection...")

    # Test the connection
    response = client.get_quote('AAPL')
    if response.status_code == 200:
        quote_data = response.json()
        print(f"\n[SUCCESS] Successfully retrieved AAPL quote!")
        print(f"Symbol: {quote_data.get('AAPL', {}).get('quote', {}).get('symbol', 'N/A')}")
        print(f"Last Price: ${quote_data.get('AAPL', {}).get('quote', {}).get('lastPrice', 'N/A')}")
    else:
        print(f"[ERROR] Failed to get quote: {response.status_code}")

except Exception as e:
    print(f"\n[ERROR] Authentication failed: {str(e)}")
    import traceback
    traceback.print_exc()
