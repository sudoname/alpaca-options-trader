from datetime import datetime, timedelta
import pandas as pd
from typing import List, Dict, Optional
import logging

class OptionChainAnalyzer:
    def __init__(self, alpaca_client):
        self.client = alpaca_client
        self.logger = logging.getLogger(__name__)

    def get_option_chain(self, ticker: str, min_days_to_expiry: int = 30,
                        max_days_to_expiry: int = 90) -> pd.DataFrame:
        try:
            current_date = datetime.now().date()
            min_expiry = current_date + timedelta(days=min_days_to_expiry)
            max_expiry = current_date + timedelta(days=max_days_to_expiry)

            contracts = self.client.get_option_contracts(
                underlying_symbol=ticker,
                expiration_date_gte=min_expiry.strftime('%Y-%m-%d'),
                expiration_date_lte=max_expiry.strftime('%Y-%m-%d')
            )

            if not contracts or not contracts.option_contracts:
                self.logger.warning(f"No option contracts found for {ticker}")
                return pd.DataFrame()

            chain_data = []
            symbols_batch = []

            for contract in contracts.option_contracts:
                symbols_batch.append(contract.symbol)

                if len(symbols_batch) >= 100:
                    snapshots = self.client.get_option_snapshot(symbols_batch)
                    if snapshots:
                        for symbol in symbols_batch:
                            if symbol in snapshots:
                                snapshot = snapshots[symbol]
                                contract_info = self._get_contract_by_symbol(contracts.option_contracts, symbol)
                                if contract_info:
                                    chain_data.append(self._process_contract(contract_info, snapshot))
                    symbols_batch = []

            if symbols_batch:
                snapshots = self.client.get_option_snapshot(symbols_batch)
                if snapshots:
                    for symbol in symbols_batch:
                        if symbol in snapshots:
                            snapshot = snapshots[symbol]
                            contract_info = self._get_contract_by_symbol(contracts.option_contracts, symbol)
                            if contract_info:
                                chain_data.append(self._process_contract(contract_info, snapshot))

            df = pd.DataFrame(chain_data)
            if not df.empty:
                df['days_to_expiry'] = (pd.to_datetime(df['expiration_date']) - pd.Timestamp.now()).dt.days
                df = df[df['days_to_expiry'] >= min_days_to_expiry]

            return df

        except Exception as e:
            self.logger.error(f"Error getting option chain: {e}")
            return pd.DataFrame()

    def _get_contract_by_symbol(self, contracts, symbol):
        for contract in contracts:
            if contract.symbol == symbol:
                return contract
        return None

    def _process_contract(self, contract, snapshot) -> Dict:
        greeks = snapshot.greeks if snapshot and snapshot.greeks else None
        quote = snapshot.latest_quote if snapshot and snapshot.latest_quote else None

        return {
            'symbol': contract.symbol,
            'underlying': contract.underlying_symbol,
            'strike': float(contract.strike_price),
            'expiration_date': contract.expiration_date,
            'type': contract.type,
            'bid': float(quote.bid_price) if quote and quote.bid_price else 0,
            'ask': float(quote.ask_price) if quote and quote.ask_price else 0,
            'mid': (float(quote.bid_price) + float(quote.ask_price)) / 2 if quote and quote.bid_price and quote.ask_price else 0,
            'volume': snapshot.latest_trade.size if snapshot and snapshot.latest_trade else 0,
            'open_interest': contract.open_interest if hasattr(contract, 'open_interest') else 0,
            'delta': float(greeks.delta) if greeks and greeks.delta else None,
            'gamma': float(greeks.gamma) if greeks and greeks.gamma else None,
            'theta': float(greeks.theta) if greeks and greeks.theta else None,
            'vega': float(greeks.vega) if greeks and greeks.vega else None,
            'iv': float(greeks.implied_volatility) if greeks and greeks.implied_volatility else None
        }

    def filter_itm_options(self, df: pd.DataFrame, current_price: float, option_type: str = 'CALL') -> pd.DataFrame:
        if df.empty:
            return df

        if option_type.upper() == 'CALL':
            itm_df = df[(df['type'] == 'call') & (df['strike'] < current_price)]
        else:
            itm_df = df[(df['type'] == 'put') & (df['strike'] > current_price)]

        return itm_df.sort_values('strike', ascending=(option_type.upper() == 'CALL'))