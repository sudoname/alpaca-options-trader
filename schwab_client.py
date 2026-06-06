"""
Schwab API Client Wrapper
Provides a simple interface for Schwab trading operations
"""

import os
import json
from dotenv import load_dotenv
from schwab import auth, client
import logging

# Load environment variables
load_dotenv()

logger = logging.getLogger(__name__)


class SchwabClient:
    """Wrapper for Schwab API client"""

    def __init__(self):
        """Initialize Schwab client"""
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.callback_url = os.getenv('SCHWAB_CALLBACK_URL')
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
        self.client = None

        if not self.app_key or not self.app_secret:
            raise ValueError("Schwab API credentials not found in .env file")

    def authenticate(self):
        """
        Authenticate with Schwab API
        First time will require browser authentication
        Subsequent calls will use saved tokens
        """
        try:
            # Try to load existing tokens
            if os.path.exists(self.token_file):
                logger.info("Loading existing Schwab tokens...")
                self.client = auth.client_from_token_file(
                    self.token_file,
                    self.app_key,
                    self.app_secret
                )
                logger.info("Successfully authenticated with existing tokens")
            else:
                logger.info("No existing tokens found. Starting browser authentication...")
                logger.info("A browser window will open for you to authorize the application.")
                logger.info(f"Make sure your redirect URI in Schwab dev portal is: {self.callback_url}")

                # First time authentication - will open browser
                self.client = auth.client_from_manual_flow(
                    self.app_key,
                    self.app_secret,
                    self.callback_url,
                    self.token_file
                )
                logger.info("Successfully authenticated! Tokens saved for future use.")

            return True

        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def get_quote(self, symbol):
        """Get real-time quote for a symbol"""
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.get_quote(symbol)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get quote: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting quote for {symbol}: {str(e)}")
            return None

    def get_option_chain(self, symbol, contract_type=None, strike_count=None,
                         include_quotes=True, from_date=None, to_date=None):
        """
        Get option chain for a symbol

        Args:
            symbol: Stock ticker symbol
            contract_type: 'CALL', 'PUT', or 'ALL'
            strike_count: Number of strikes above and below ATM
            include_quotes: Include quote data
            from_date: Start date for expiration (YYYY-MM-DD)
            to_date: End date for expiration (YYYY-MM-DD)
        """
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.get_option_chain(
                symbol,
                contract_type=contract_type,
                strike_count=strike_count,
                include_quotes=include_quotes,
                from_date=from_date,
                to_date=to_date
            )

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get option chain: {response.status_code} - {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error getting option chain for {symbol}: {str(e)}")
            return None

    def get_account_info(self, account_hash):
        """Get account information"""
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.get_account(account_hash)
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account info: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting account info: {str(e)}")
            return None

    def get_account_numbers(self):
        """Get all account numbers linked to this API key"""
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.get_account_numbers()
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get account numbers: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting account numbers: {str(e)}")
            return None

    def place_option_order(self, account_hash, order):
        """
        Place an option order

        Args:
            account_hash: Account identifier
            order: Order object (dict) following Schwab API format
        """
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.place_order(account_hash, order)
            if response.status_code in [200, 201]:
                logger.info("Order placed successfully")
                return True
            else:
                logger.error(f"Failed to place order: {response.status_code} - {response.text}")
                return False
        except Exception as e:
            logger.error(f"Error placing order: {str(e)}")
            return False

    def get_orders(self, account_hash, from_date=None, to_date=None, status=None):
        """Get orders for an account"""
        if not self.client:
            raise RuntimeError("Client not authenticated. Call authenticate() first.")

        try:
            response = self.client.get_orders(
                account_hash,
                from_date=from_date,
                to_date=to_date,
                status=status
            )
            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Failed to get orders: {response.status_code} - {response.text}")
                return None
        except Exception as e:
            logger.error(f"Error getting orders: {str(e)}")
            return None


def main():
    """Test the Schwab client"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    try:
        # Initialize and authenticate
        schwab = SchwabClient()
        schwab.authenticate()

        # Test: Get account numbers
        print("\n=== Getting Account Numbers ===")
        accounts = schwab.get_account_numbers()
        if accounts:
            print(json.dumps(accounts, indent=2))

        # Test: Get a quote
        print("\n=== Getting Quote for AAPL ===")
        quote = schwab.get_quote('AAPL')
        if quote:
            print(json.dumps(quote, indent=2))

        # Test: Get option chain
        print("\n=== Getting Option Chain for AAPL ===")
        option_chain = schwab.get_option_chain('AAPL', contract_type='CALL', strike_count=5)
        if option_chain:
            print(f"Found option chain with {len(option_chain.get('callExpDateMap', {}))} expiration dates")

    except Exception as e:
        logger.error(f"Error in main: {str(e)}")
        raise


if __name__ == "__main__":
    main()
