import pandas as pd
from typing import Optional, Dict
import logging

class OptionSelector:
    def __init__(self, budget: float = 500):
        self.budget = budget
        self.logger = logging.getLogger(__name__)

    def select_best_option(self, option_chain: pd.DataFrame, current_price: float) -> Optional[Dict]:
        if option_chain.empty:
            self.logger.warning("Empty option chain provided")
            return None

        eligible_options = option_chain[
            (option_chain['days_to_expiry'] > 30) &
            (option_chain['ask'] > 0) &
            (option_chain['bid'] > 0) &
            (option_chain['delta'].notna())
        ].copy()

        if eligible_options.empty:
            self.logger.warning("No eligible options found with >30 days expiry")
            return None

        eligible_options['contract_cost'] = eligible_options['ask'] * 100
        eligible_options = eligible_options[eligible_options['contract_cost'] <= self.budget]

        if eligible_options.empty:
            self.logger.warning(f"No options found within budget of ${self.budget}")
            return None

        eligible_options['moneyness'] = abs(eligible_options['strike'] - current_price)
        eligible_options['score'] = self._calculate_option_score(eligible_options)

        best_option = eligible_options.nlargest(1, 'score').iloc[0]

        max_contracts = int(self.budget / (best_option['ask'] * 100))

        return {
            'symbol': best_option['symbol'],
            'strike': best_option['strike'],
            'expiration': best_option['expiration_date'],
            'type': best_option['type'],
            'ask': best_option['ask'],
            'bid': best_option['bid'],
            'delta': best_option['delta'],
            'gamma': best_option['gamma'],
            'theta': best_option['theta'],
            'vega': best_option['vega'],
            'iv': best_option['iv'],
            'days_to_expiry': best_option['days_to_expiry'],
            'contracts_to_buy': max_contracts,
            'total_cost': max_contracts * best_option['ask'] * 100,
            'score': best_option['score']
        }

    def _calculate_option_score(self, options: pd.DataFrame) -> pd.Series:
        score = pd.Series(index=options.index, dtype=float)

        if 'delta' in options.columns and options['delta'].notna().any():
            delta_score = options['delta'].abs() * 30
            score += delta_score.fillna(0)

        if 'gamma' in options.columns and options['gamma'].notna().any():
            gamma_norm = (options['gamma'] - options['gamma'].min()) / (options['gamma'].max() - options['gamma'].min() + 0.0001)
            score += gamma_norm.fillna(0) * 10

        if 'theta' in options.columns and options['theta'].notna().any():
            theta_score = (1 - (options['theta'].abs() / options['theta'].abs().max())) * 15
            score += theta_score.fillna(0)

        if 'iv' in options.columns and options['iv'].notna().any():
            iv_median = options['iv'].median()
            iv_score = (1 - abs(options['iv'] - iv_median) / iv_median) * 15
            score += iv_score.fillna(0)

        moneyness_score = (1 - options['moneyness'] / options['moneyness'].max()) * 20

        score += moneyness_score

        days_score = (options['days_to_expiry'] / options['days_to_expiry'].max()) * 10
        score += days_score

        return score