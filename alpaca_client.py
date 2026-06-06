import os
from dotenv import load_dotenv
from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOptionContractsRequest, MarketOrderRequest
from alpaca.trading.enums import AssetClass, OrderSide, TimeInForce
from alpaca.data.historical.option import OptionHistoricalDataClient
from alpaca.data.requests import OptionLatestQuoteRequest, OptionSnapshotRequest
import logging

load_dotenv()

class AlpacaOptionsClient:
    def __init__(self):
        api_key = os.getenv('ALPACA_API_KEY')
        secret_key = os.getenv('ALPACA_SECRET_KEY')
        paper = os.getenv('ALPACA_PAPER', 'true').lower() == 'true'

        if not api_key or not secret_key:
            raise ValueError("Please set ALPACA_API_KEY and ALPACA_SECRET_KEY in .env file")

        self.trading_client = TradingClient(api_key, secret_key, paper=paper)
        self.data_client = OptionHistoricalDataClient(api_key, secret_key)
        self.logger = logging.getLogger(__name__)

    def get_account(self):
        try:
            return self.trading_client.get_account()
        except Exception as e:
            self.logger.error(f"Error getting account info: {e}")
            raise

    def get_option_contracts(self, underlying_symbol, expiration_date_gte=None, expiration_date_lte=None,
                            strike_price_gte=None, strike_price_lte=None, option_type=None):
        try:
            request = GetOptionContractsRequest(
                underlying_symbols=[underlying_symbol],
                expiration_date_gte=expiration_date_gte,
                expiration_date_lte=expiration_date_lte,
                strike_price_gte=strike_price_gte,
                strike_price_lte=strike_price_lte,
                type=option_type
            )
            return self.trading_client.get_option_contracts(request)
        except Exception as e:
            self.logger.error(f"Error getting option contracts: {e}")
            raise

    def get_option_snapshot(self, symbols):
        try:
            request = OptionSnapshotRequest(symbols=symbols)
            snapshots = self.data_client.get_option_snapshot(request)
            return snapshots
        except Exception as e:
            self.logger.error(f"Error getting option snapshots: {e}")
            return None

    def get_latest_quote(self, symbol):
        try:
            request = OptionLatestQuoteRequest(symbols=[symbol])
            quotes = self.data_client.get_option_latest_quote(request)
            return quotes.get(symbol)
        except Exception as e:
            self.logger.error(f"Error getting latest quote: {e}")
            return None

    def place_option_order(self, symbol, qty, side=OrderSide.BUY):
        try:
            request = MarketOrderRequest(
                symbol=symbol,
                qty=qty,
                side=side,
                time_in_force=TimeInForce.DAY,
                asset_class=AssetClass.US_OPTION
            )
            order = self.trading_client.submit_order(request)
            return order
        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            raise