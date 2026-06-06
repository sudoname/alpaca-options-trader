"""
Simple Schwab Authentication - Uses easy_client for automatic auth
"""

import os
from dotenv import load_dotenv
from schwab import auth
from multiprocess import freeze_support

def main():
    # Load environment variables
    load_dotenv()

    app_key = os.getenv('SCHWAB_APP_KEY')
    app_secret = os.getenv('SCHWAB_APP_SECRET')
    callback_url = os.getenv('SCHWAB_CALLBACK_URL')
    token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')

    print("=== Schwab API Authentication ===")
    print("A browser window will open automatically for you to log in.")
    print("After logging in and authorizing, the browser will redirect automatically.")
    print("\nStarting authentication...\n")

    try:
        # Use easy_client with non-interactive mode
        client = auth.easy_client(
            api_key=app_key,
            app_secret=app_secret,
            callback_url=callback_url,
            token_path=token_file,
            interactive=False  # Don't wait for Enter key
        )

        print("\n[SUCCESS] Authentication completed!")
        print(f"Tokens saved to: {token_file}")
        print("\nTesting connection with AAPL quote...")

        # Test the connection
        response = client.get_quote('AAPL')
        if response.status_code == 200:
            quote_data = response.json()
            aapl_data = quote_data.get('AAPL', {})
            quote = aapl_data.get('quote', {})

            print("\n[SUCCESS] Retrieved AAPL quote!")
            print(f"Symbol: {quote.get('symbol', 'N/A')}")
            print(f"Last Price: ${quote.get('lastPrice', 'N/A')}")
            print(f"Change: ${quote.get('netChange', 'N/A')} ({quote.get('netPercentChangeInDouble', 'N/A')}%)")
            print(f"Volume: {quote.get('totalVolume', 'N/A')}")
        else:
            print(f"[ERROR] Failed to get quote: {response.status_code}")

        print("\nYou're all set! The Schwab client is ready to use.")

    except KeyboardInterrupt:
        print("\n\nAuthentication cancelled by user.")
    except Exception as e:
        print(f"\n[ERROR] Authentication failed: {str(e)}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    freeze_support()  # Required for Windows multiprocessing
    main()
