"""
Advanced Options Trading System with ML and Position Management
"""

import os
import sys
import json
import requests
from datetime import datetime, timedelta
import argparse
from typing import Dict, List, Optional, Tuple
import pickle
import math

class SmartOptionsTrader:
    def __init__(self, ticker: str = None, quantity: int = 1):
        self.load_credentials()
        self.base_url = "https://paper-api.alpaca.markets" if self.paper else "https://api.alpaca.markets"
        self.data_url = "https://data.alpaca.markets"
        self.headers = {
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key
        }
        self.max_budget_per_trade = 500
        self.ticker = ticker
        self.quantity = quantity

        # Dynamic stop loss and take profit parameters are now loaded from .env in load_credentials()

        self.load_trading_history()
        self.load_ml_model()

        # RL advisory layer (shadow mode: observes & learns, never overrides)
        self.rl_advisor = None
        try:
            from rl_wrapper import RLAdvisor, rl_enabled
            if rl_enabled():
                self.rl_advisor = RLAdvisor(strat_name='smart_trader')
                print("[RL] Advisor active (shadow mode)")
        except Exception as e:
            print(f"[RL] Advisor unavailable: {e}")

        # Sentiment (Fear & Greed) risk filter. Fail-open. Uses an Alpaca-backed
        # market data provider so the custom score (PRIMARY) can be computed here
        # too; falls back to CNN if Alpaca data is unavailable. Used to scale
        # position size in place_order_with_stops; never blocks/crashes a trade.
        self.sentiment_service = None
        try:
            from sentiment import SentimentService, AlpacaMarketDataProvider, SentimentConfig
            if SentimentConfig.from_env().enabled:
                feed = os.getenv('SENTIMENT_ALPACA_FEED', 'iex')
                provider = AlpacaMarketDataProvider(self.data_url, self.headers, feed=feed)
                self.sentiment_service = SentimentService(provider)
                print("[SENTIMENT] Fear & Greed filter active")
        except Exception as e:
            print(f"[SENTIMENT] Filter unavailable: {e}")

    def load_credentials(self):
        """Load API credentials and trading parameters from .env file"""
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        self.api_key = env_vars.get('ALPACA_API_KEY', '')
        self.secret_key = env_vars.get('ALPACA_SECRET_KEY', '')
        self.paper = env_vars.get('ALPACA_PAPER', 'true').lower() == 'true'

        # Load profit/loss thresholds from .env with defaults
        self.base_stop_loss = float(env_vars.get('BASE_STOP_LOSS', '0.10'))
        self.base_take_profit = float(env_vars.get('BASE_TAKE_PROFIT', '0.20'))
        self.max_stop_loss = float(env_vars.get('MAX_STOP_LOSS', '0.25'))
        self.max_take_profit = float(env_vars.get('MAX_TAKE_PROFIT', '0.50'))
        self.trailing_stop_distance = float(env_vars.get('TRAILING_STOP_DISTANCE', '0.05'))

    def load_trading_history(self):
        """Load historical trades for learning"""
        self.history_file = 'trading_history.json'
        if os.path.exists(self.history_file):
            with open(self.history_file, 'r') as f:
                self.trading_history = json.load(f)
        else:
            self.trading_history = {
                'trades': [],
                'performance_metrics': {},
                'learned_patterns': {}
            }

    def save_trading_history(self):
        """Save trading history for future learning"""
        with open(self.history_file, 'w') as f:
            json.dump(self.trading_history, f, indent=2, default=str)

    def load_ml_model(self):
        """Load or initialize ML model for trade optimization"""
        self.model_file = 'trade_optimizer.pkl'
        if os.path.exists(self.model_file):
            with open(self.model_file, 'rb') as f:
                self.ml_model = pickle.load(f)
        else:
            # Initialize simple scoring model
            self.ml_model = {
                'weights': {
                    'delta': 0.30,
                    'gamma': 0.10,
                    'theta': 0.15,
                    'vega': 0.10,
                    'iv': 0.15,
                    'moneyness': 0.20
                },
                'success_patterns': [],
                'failure_patterns': []
            }

    def save_ml_model(self):
        """Save ML model after updates"""
        with open(self.model_file, 'wb') as f:
            pickle.dump(self.ml_model, f)

    def get_account(self):
        """Get account information"""
        response = requests.get(f"{self.base_url}/v2/account", headers=self.headers)
        return response.json() if response.status_code == 200 else None

    def get_positions(self):
        """Get current positions"""
        response = requests.get(f"{self.base_url}/v2/positions", headers=self.headers)
        return response.json() if response.status_code == 200 else []

    def get_orders(self):
        """Get current orders"""
        response = requests.get(f"{self.base_url}/v2/orders", headers=self.headers)
        return response.json() if response.status_code == 200 else []

    def get_market_status(self):
        """Check if market is open"""
        response = requests.get(f"{self.base_url}/v2/clock", headers=self.headers)
        if response.status_code == 200:
            return response.json()
        else:
            return {'is_open': False}

    def get_current_price(self, ticker=None):
        """Get current stock price using last trade price"""
        symbol = ticker or self.ticker
        if not symbol:
            return None

        # Use last trade price for most accurate current price
        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/trades/latest",
            headers=self.headers,
            params={'feed': 'iex'}  # Use IEX data for free tier
        )
        if response.status_code == 200:
            data = response.json()
            return float(data['trade']['p'])  # 'p' = price
        return None

    def get_price_history(self, ticker: str = None, days: int = 10) -> List[float]:
        """Get historical prices for volatility calculation"""
        symbol = ticker or self.ticker
        if not symbol:
            return []

        end_time = datetime.now()
        start_time = end_time - timedelta(days=days)

        response = requests.get(
            f"{self.data_url}/v2/stocks/{symbol}/bars",
            headers=self.headers,
            params={
                'timeframe': '1Day',
                'start': start_time.strftime('%Y-%m-%d'),
                'end': end_time.strftime('%Y-%m-%d'),
                'limit': days + 5,
                'feed': 'iex'  # Use IEX data for free tier
            }
        )

        if response.status_code == 200:
            data = response.json()
            bars = data.get('bars', [])
            # IEX returns bars as a list, not nested by symbol
            return [float(bar['c']) for bar in bars]
        return []

    def calculate_volatility(self, ticker: str = None) -> float:
        """Calculate historical volatility"""
        prices = self.get_price_history(ticker)
        if len(prices) < 2:
            return 0.20  # Default 20% volatility

        # Calculate daily returns
        returns = []
        for i in range(1, len(prices)):
            daily_return = (prices[i] - prices[i-1]) / prices[i-1]
            returns.append(daily_return)

        if not returns:
            return 0.20

        # Calculate standard deviation of returns (volatility)
        mean_return = sum(returns) / len(returns)
        variance = sum([(r - mean_return) ** 2 for r in returns]) / len(returns)
        volatility = math.sqrt(variance) * math.sqrt(252)  # Annualized volatility

        return min(max(volatility, 0.10), 0.80)  # Cap between 10% and 80%

    def calculate_momentum(self, ticker: str = None) -> float:
        """Calculate momentum indicator"""
        prices = self.get_price_history(ticker, days=5)
        if len(prices) < 3:
            return 0

        # Calculate momentum as recent price change
        recent_momentum = (prices[-1] - prices[-3]) / prices[-3] if len(prices) >= 3 else 0

        # Smooth momentum with moving average
        if len(prices) >= 5:
            recent_avg = sum(prices[-3:]) / 3
            older_avg = sum(prices[-5:-2]) / 3
            trend_momentum = (recent_avg - older_avg) / older_avg
            momentum = (recent_momentum + trend_momentum) / 2
        else:
            momentum = recent_momentum

        return momentum

    def get_market_regime(self, ticker: str = None) -> str:
        """Determine market regime (trending, ranging, volatile)"""
        volatility = self.calculate_volatility(ticker)
        momentum = abs(self.calculate_momentum(ticker))

        if volatility > 0.30:
            return "volatile"
        elif momentum > 0.05:
            return "trending"
        else:
            return "ranging"

    def calculate_dynamic_levels(self, ticker: str = None, current_price: float = None) -> Dict:
        """Calculate dynamic stop loss and take profit levels"""
        volatility = self.calculate_volatility(ticker)
        momentum = self.calculate_momentum(ticker)
        market_regime = self.get_market_regime(ticker)

        # Base adjustments
        vol_multiplier = 1 + (volatility - 0.20) * 2  # Adjust based on volatility
        momentum_multiplier = 1 + abs(momentum) * 5  # Adjust based on momentum

        # Regime-based adjustments
        regime_adjustments = {
            "volatile": {"stop_multiplier": 1.5, "profit_multiplier": 1.8},
            "trending": {"stop_multiplier": 0.8, "profit_multiplier": 1.5},
            "ranging": {"stop_multiplier": 1.2, "profit_multiplier": 1.0}
        }

        regime_adj = regime_adjustments.get(market_regime, {"stop_multiplier": 1.0, "profit_multiplier": 1.0})

        # Calculate dynamic stop loss
        dynamic_stop_loss = self.base_stop_loss * vol_multiplier * regime_adj["stop_multiplier"]
        dynamic_stop_loss = min(max(dynamic_stop_loss, self.base_stop_loss), self.max_stop_loss)

        # Calculate dynamic take profit
        dynamic_take_profit = self.base_take_profit * momentum_multiplier * regime_adj["profit_multiplier"]
        dynamic_take_profit = min(max(dynamic_take_profit, self.base_take_profit), self.max_take_profit)

        # Trailing stop adjustments
        if momentum > 0.03:  # Strong upward momentum
            trailing_distance = self.trailing_stop_distance * 0.7  # Tighter trailing
        elif momentum < -0.03:  # Strong downward momentum
            trailing_distance = self.trailing_stop_distance * 1.5  # Looser trailing
        else:
            trailing_distance = self.trailing_stop_distance

        return {
            "stop_loss_percent": dynamic_stop_loss,
            "take_profit_percent": dynamic_take_profit,
            "trailing_stop_distance": trailing_distance,
            "volatility": volatility,
            "momentum": momentum,
            "market_regime": market_regime,
            "vol_multiplier": vol_multiplier,
            "momentum_multiplier": momentum_multiplier
        }

    def get_option_price(self, symbol):
        """Get current option price using latest quotes endpoint"""
        try:
            response = requests.get(
                f"{self.data_url}/v1/options/quotes/latest",
                headers=self.headers,
                params={'symbols': symbol, 'feed': 'indicative'}
            )

            if response.status_code == 200:
                data = response.json()
                # Response format: {"quotes": {"SYMBOL": {...}}}
                if 'quotes' in data and symbol in data['quotes']:
                    quote = data['quotes'][symbol]
                    bid = float(quote.get('bp', 0))
                    ask = float(quote.get('ap', 0))

                    if bid > 0 or ask > 0:
                        return {
                            'bid': bid,
                            'ask': ask,
                            'mid': (bid + ask) / 2 if (bid > 0 and ask > 0) else (ask if ask > 0 else bid)
                        }

            print(f"[OPTION PRICE] No quote data for {symbol}, status: {response.status_code}")
            return None
        except Exception as e:
            print(f"[OPTION PRICE ERROR] {symbol}: {e}")
            return None

    def calculate_option_score(self, option: Dict) -> float:
        """Calculate option score using ML-enhanced model"""
        base_score = 0

        # Apply learned weights
        weights = self.ml_model['weights']

        # Delta score (prefer 0.5-0.7 for momentum)
        if 'delta' in option:
            delta_optimal = 0.6
            delta_score = (1 - abs(option['delta'] - delta_optimal)) * weights['delta']
            base_score += delta_score * 100

        # Gamma score (moderate gamma for flexibility)
        if 'gamma' in option:
            gamma_score = min(option['gamma'] * 10, 1) * weights['gamma']
            base_score += gamma_score * 100

        # Theta score (minimize time decay)
        if 'theta' in option:
            theta_penalty = abs(option['theta']) * weights['theta']
            base_score += (1 - min(theta_penalty, 1)) * 100

        # IV score (prefer reasonable IV)
        if 'iv' in option:
            iv_optimal = 0.25
            iv_score = (1 - abs(option['iv'] - iv_optimal)) * weights['iv']
            base_score += iv_score * 100

        # Moneyness score
        if 'moneyness' in option:
            moneyness_score = (1 - abs(option['moneyness'])) * weights['moneyness']
            base_score += moneyness_score * 100

        # Apply learned patterns boost/penalty
        pattern_adjustment = self.apply_learned_patterns(option)
        base_score *= (1 + pattern_adjustment)

        return min(max(base_score, 0), 100)

    def apply_learned_patterns(self, option: Dict) -> float:
        """Apply learned patterns from historical trades"""
        adjustment = 0

        # Check success patterns
        for pattern in self.ml_model.get('success_patterns', []):
            if self.matches_pattern(option, pattern):
                adjustment += 0.1

        # Check failure patterns
        for pattern in self.ml_model.get('failure_patterns', []):
            if self.matches_pattern(option, pattern):
                adjustment -= 0.15

        return adjustment

    def matches_pattern(self, option: Dict, pattern: Dict) -> bool:
        """Check if option matches a learned pattern"""
        threshold = 0.1
        matches = 0
        checks = 0

        for key in ['delta', 'gamma', 'theta', 'iv']:
            if key in option and key in pattern:
                checks += 1
                if abs(option[key] - pattern[key]) <= threshold:
                    matches += 1

        return matches / checks > 0.7 if checks > 0 else False

    def determine_option_strategy(self, ticker: str = None) -> str:
        """Determine whether to trade calls or puts based on market analysis"""
        symbol = ticker or self.ticker

        # Get market conditions
        momentum = self.calculate_momentum(symbol)
        volatility = self.calculate_volatility(symbol)
        market_regime = self.get_market_regime(symbol)

        # Get price trends
        prices = self.get_price_history(symbol, days=10)
        if len(prices) < 5:
            return 'call'  # Default to calls if insufficient data

        # Calculate short and medium term trends
        short_trend = (prices[-1] - prices[-3]) / prices[-3]  # 3-day trend
        medium_trend = (prices[-1] - prices[-5]) / prices[-5]  # 5-day trend

        # Decision logic
        bearish_signals = 0
        bullish_signals = 0

        # Momentum analysis
        if momentum < -0.03:  # Strong negative momentum
            bearish_signals += 2
        elif momentum < -0.01:  # Moderate negative momentum
            bearish_signals += 1
        elif momentum > 0.03:  # Strong positive momentum
            bullish_signals += 2
        elif momentum > 0.01:  # Moderate positive momentum
            bullish_signals += 1

        # Trend analysis
        if short_trend < -0.02:  # Short-term downtrend
            bearish_signals += 1
        elif short_trend > 0.02:  # Short-term uptrend
            bullish_signals += 1

        if medium_trend < -0.03:  # Medium-term downtrend
            bearish_signals += 1
        elif medium_trend > 0.03:  # Medium-term uptrend
            bullish_signals += 1

        # Volatility consideration (high vol favors direction plays)
        if volatility > 0.4 and bearish_signals > bullish_signals:
            bearish_signals += 1
        elif volatility > 0.4 and bullish_signals > bearish_signals:
            bullish_signals += 1

        # Market regime consideration
        if market_regime == 'volatile' and bearish_signals > 0:
            bearish_signals += 1  # Volatile markets often favor puts

        # Make decision
        if bearish_signals > bullish_signals and bearish_signals >= 2:
            strategy = 'put'
        else:
            strategy = 'call'  # Default to calls unless strong bearish signals

        print(f"[STRATEGY] Analysis for {symbol}:")
        print(f"[STRATEGY] Momentum: {momentum:.3f}, Volatility: {volatility:.1%}")
        print(f"[STRATEGY] Short trend: {short_trend:.2%}, Medium trend: {medium_trend:.2%}")
        print(f"[STRATEGY] Bearish signals: {bearish_signals}, Bullish signals: {bullish_signals}")
        print(f"[STRATEGY] Decision: {strategy.upper()} options")

        return strategy

    def select_best_option(self, contracts, current_price):
        """Select best option using enhanced ML scoring with call/put intelligence"""
        # Determine optimal strategy (call or put)
        strategy = self.determine_option_strategy()

        print(f"[STRATEGY] Selected strategy: {strategy.upper()}")
        print(f"[CONTRACTS] Analyzing {len(contracts)} contracts")

        best_option = None
        best_score = -1
        validated_count = 0

        for contract in contracts:
            strike = float(contract['strike_price'])
            contract_type = contract.get('type', 'call').lower()
            expiration = contract['expiration_date']

            # Filter by strategy and moneyness
            if strategy == 'call':
                # For calls, prefer ITM or near-the-money
                if strike > current_price * 1.05:  # Skip far OTM calls
                    continue
                if contract_type != 'call':
                    continue

                # Calculate call-specific metrics
                delta = min(0.95, max(0.05, (current_price - strike) / current_price * 0.7 + 0.5))
                moneyness = (current_price - strike) / strike

            else:  # strategy == 'put'
                # For puts, prefer ITM or near-the-money
                if strike < current_price * 0.95:  # Skip far OTM puts
                    continue
                if contract_type != 'put':
                    continue

                # Calculate put-specific metrics
                delta = min(-0.05, max(-0.95, (strike - current_price) / current_price * 0.7 - 0.5))
                moneyness = (strike - current_price) / strike

            # Validate that this option actually exists by checking if we can get a quote
            option_symbol = contract['symbol']

            # Get pricing data - either from API or mock Black-Scholes calculation
            ask_price = 0
            bid_price = 0
            spread = 0
            volume = contract.get('volume', 0)
            open_interest = contract.get('open_interest', 0)

            if contract.get('mock', False):
                # Use mock Black-Scholes prices
                bid_price = contract.get('mock_bid', 0)
                ask_price = contract.get('mock_ask', 0)
                spread = ask_price - bid_price
                validated_count += 1
                print(f"[MOCK] ${strike:.2f} {contract_type.upper()} exp {expiration} - Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f} (Black-Scholes)")
            else:
                # Try to get real quote from API
                option_quote = self.get_option_price(option_symbol)
                if not option_quote or option_quote['ask'] <= 0:
                    print(f"[SKIP] No valid quote for {option_symbol} (Strike: ${strike:.2f}, Exp: {expiration})")
                    continue

                ask_price = option_quote['ask']
                bid_price = option_quote['bid']
                spread = ask_price - bid_price
                spread_pct = (spread / ask_price * 100) if ask_price > 0 else 100

                validated_count += 1
                print(f"[VALIDATED] ${strike:.2f} {contract_type.UPPER()} exp {expiration} - Bid: ${bid_price:.2f}, Ask: ${ask_price:.2f}, Vol: {volume}, OI: {open_interest}")

                # Skip options with very wide spreads (>20%) or no liquidity
                if spread_pct > 20:
                    print(f"[SKIP] Spread too wide: {spread_pct:.1f}%")
                    continue

            # Calculate option metrics
            option_data = {
                'symbol': contract['symbol'],
                'underlying': self.ticker,
                'strike': strike,
                'expiration': contract['expiration_date'],
                'type': strategy,
                'delta': delta,
                'gamma': 0.01,
                'theta': -0.05,
                'iv': 0.25,
                'moneyness': abs(moneyness),
                'mock': contract.get('mock', False),
                'ask': ask_price,
                'bid': bid_price,
                'spread': spread,
                'volume': volume,
                'open_interest': open_interest
            }

            # Calculate base score
            score = self.calculate_option_score(option_data)

            # Boost score for strategy alignment
            if contract_type == strategy:
                score *= 1.1  # 10% boost for matching strategy

            # Add liquidity scoring (critical for real trading)
            if not contract.get('mock', False):
                # Volume scoring (max 15 points)
                if volume > 100:
                    volume_score = min(15, volume / 100)
                else:
                    volume_score = volume / 10  # Penalize low volume

                # Open interest scoring (max 15 points)
                if open_interest > 100:
                    oi_score = min(15, open_interest / 100)
                else:
                    oi_score = open_interest / 10  # Penalize low OI

                # Spread scoring (tighter spread = higher score, max 10 points)
                spread_score = max(0, 10 - spread_pct)

                liquidity_score = volume_score + oi_score + spread_score
                score += liquidity_score

                print(f"[LIQUIDITY] Vol: {volume_score:.1f}, OI: {oi_score:.1f}, Spread: {spread_score:.1f} = Total: {liquidity_score:.1f}")

            if score > best_score:
                best_score = score
                best_option = option_data
                best_option['score'] = score
                best_option['strategy_type'] = strategy

        print(f"[VALIDATION] {validated_count} contracts validated with real quotes")

        if best_option and not best_option.get('mock', False):
            print(f"[BEST OPTION] Strike: ${best_option['strike']:.2f} {best_option['type'].upper()}")
            print(f"[BEST OPTION] Expiration: {best_option['expiration']}")
            print(f"[BEST OPTION] Score: {best_option['score']:.2f}")

        return best_option

    def place_order_with_stops(self, option: Dict, quantity: int = None):
        """Place order with dynamic stop loss and take profit"""
        order_quantity = quantity or self.quantity

        # Calculate dynamic levels based on market conditions
        underlying_symbol = option.get('underlying', self.ticker)
        dynamic_levels = self.calculate_dynamic_levels(underlying_symbol)

        print(f"[DYNAMIC LEVELS] Market Regime: {dynamic_levels['market_regime']}")
        print(f"[DYNAMIC LEVELS] Volatility: {dynamic_levels['volatility']:.2%}")
        print(f"[DYNAMIC LEVELS] Momentum: {dynamic_levels['momentum']:.2%}")
        print(f"[DYNAMIC LEVELS] Stop Loss: {dynamic_levels['stop_loss_percent']:.2%}")
        print(f"[DYNAMIC LEVELS] Take Profit: {dynamic_levels['take_profit_percent']:.2%}")

        # Get current option price before placing order
        option_symbol = option['symbol']
        current_option_price = self.get_option_price(option_symbol)

        if not current_option_price:
            print(f"[ERROR] Cannot get current price for option {option_symbol}")
            return None

        entry_price = current_option_price['ask']
        bid_price = current_option_price['bid']

        print(f"[OPTION PRICE] Bid: ${bid_price:.2f}, Ask: ${entry_price:.2f}")

        # Validate price is reasonable
        if entry_price <= 0:
            print(f"[ERROR] Invalid option price: ${entry_price}")
            return None

        # Calculate total cost
        total_cost = entry_price * 100 * order_quantity
        print(f"[COST] Total cost for {order_quantity} contract(s): ${total_cost:.2f}")

        # Check if within budget
        if total_cost > self.max_budget_per_trade:
            print(f"[WARNING] Cost ${total_cost:.2f} exceeds budget ${self.max_budget_per_trade:.2f}")
            # Adjust quantity to fit budget
            order_quantity = int(self.max_budget_per_trade / (entry_price * 100))
            if order_quantity < 1:
                print(f"[ERROR] Cannot afford even 1 contract at ${entry_price:.2f}")
                return None
            total_cost = entry_price * 100 * order_quantity
            print(f"[ADJUSTED] New quantity: {order_quantity} contract(s), cost: ${total_cost:.2f}")

        # Sentiment risk filter (fail-open): scale size down in fearful/euphoric
        # markets and block aggressive entries during Extreme Fear. Never crashes.
        if self.sentiment_service:
            try:
                from sentiment import adjust_trade_risk_by_sentiment, summarize_for_log
                sentiment = self.sentiment_service.get_sentiment()
                print(f"[SENTIMENT] {summarize_for_log(sentiment)}")
                decision = adjust_trade_risk_by_sentiment(
                    {'size': order_quantity,
                     'confidence': dynamic_levels.get('confidence'),
                     'direction': option.get('type')},
                    sentiment,
                )
                print(f"[SENTIMENT] {decision['reason']}")
                if not decision['allowed']:
                    print("[SENTIMENT] Trade blocked by sentiment filter")
                    return None
                if decision['adjusted_size'] != order_quantity:
                    order_quantity = decision['adjusted_size']
                    total_cost = entry_price * 100 * order_quantity
                    print(f"[SENTIMENT] Adjusted quantity: {order_quantity} "
                          f"contract(s), cost: ${total_cost:.2f}")
            except Exception as e:
                print(f"[SENTIMENT] filter error (ignored): {e}")

        # Place main order
        order_data = {
            'symbol': option['symbol'],
            'qty': order_quantity,
            'side': 'buy',
            'type': 'market',
            'time_in_force': 'day',
            'asset_class': 'us_option'
        }

        response = requests.post(
            f"{self.base_url}/v2/orders",
            headers=self.headers,
            json=order_data
        )

        if response.status_code in [200, 201]:
            order = response.json()

            # Store trade info with dynamic levels
            trade_info = {
                'order_id': order['id'],
                'symbol': option['symbol'],
                'underlying_symbol': underlying_symbol,
                'entry_price': entry_price,
                'quantity': order_quantity,
                'dynamic_stop_loss_percent': dynamic_levels['stop_loss_percent'],
                'dynamic_take_profit_percent': dynamic_levels['take_profit_percent'],
                'trailing_stop_distance': dynamic_levels['trailing_stop_distance'],
                'stop_loss_trigger': entry_price * (1 - dynamic_levels['stop_loss_percent']),
                'take_profit_trigger': entry_price * (1 + dynamic_levels['take_profit_percent']),
                'partial_close_done': False,
                'trailing_stop_active': False,
                'highest_price': entry_price,
                'entry_time': datetime.now().isoformat(),
                'market_conditions': {
                    'volatility': dynamic_levels['volatility'],
                    'momentum': dynamic_levels['momentum'],
                    'market_regime': dynamic_levels['market_regime']
                }
            }

            # Save to active trades file
            self.save_active_trade(trade_info)

            # RL shadow: log the decision so we can learn from its outcome
            if self.rl_advisor:
                try:
                    action = (option.get('type') or 'call').upper()  # CALL/PUT
                    analysis_ctx = {
                        'direction': action,
                        'momentum': dynamic_levels.get('momentum', 0),
                        'confidence': option.get('score', 0),
                    }
                    advice = self.rl_advisor.observe_and_log(
                        analysis_ctx, order['id'], action,
                        day_of_week=datetime.now().weekday()
                    )
                    print(f"[RL] Recommended: {advice['recommended_action']} | "
                          f"Rule: {advice['rule_action']} | "
                          f"Agree: {advice['agreement']}")
                except Exception as e:
                    print(f"[RL] observe failed: {e}")

            return order

        return None

    def save_active_trade(self, trade_info: Dict):
        """Save active trade for monitoring"""
        active_file = 'active_trades.json'

        if os.path.exists(active_file):
            with open(active_file, 'r') as f:
                active_trades = json.load(f)
        else:
            active_trades = []

        active_trades.append(trade_info)

        with open(active_file, 'w') as f:
            json.dump(active_trades, f, indent=2, default=str)

    def monitor_positions(self):
        """Monitor positions for stop loss and take profit"""
        active_file = 'active_trades.json'

        if not os.path.exists(active_file):
            return

        with open(active_file, 'r') as f:
            active_trades = json.load(f)

        positions = self.get_positions()
        updated_trades = []

        for trade in active_trades:
            position = next((p for p in positions if p['symbol'] == trade['symbol']), None)

            if not position:
                # Position closed, record outcome
                self.record_trade_outcome(trade, 'closed')
                continue

            current_price = float(position['current_price']) if position['current_price'] else 0
            entry_price = trade['entry_price']

            if current_price == 0:
                updated_trades.append(trade)
                continue

            pnl_percent = ((current_price - entry_price) / entry_price) * 100

            # Update highest price for trailing stop
            if current_price > trade['highest_price']:
                trade['highest_price'] = current_price
                trade['trailing_stop_active'] = True

            # Use dynamic levels or fallback to static
            stop_loss_percent = trade.get('dynamic_stop_loss_percent', 0.10) * 100
            take_profit_percent = trade.get('dynamic_take_profit_percent', 0.20) * 100
            trailing_distance = trade.get('trailing_stop_distance', 0.05)

            # Re-calculate dynamic levels for current market conditions
            underlying_symbol = trade.get('underlying_symbol', self.ticker)
            if underlying_symbol:
                current_dynamic = self.calculate_dynamic_levels(underlying_symbol)
                # Update levels if market conditions have changed significantly
                old_regime = trade.get('market_conditions', {}).get('market_regime', 'ranging')
                new_regime = current_dynamic['market_regime']

                if old_regime != new_regime:
                    print(f"[REGIME CHANGE] {old_regime} → {new_regime}")
                    stop_loss_percent = current_dynamic['stop_loss_percent'] * 100
                    take_profit_percent = current_dynamic['take_profit_percent'] * 100
                    trailing_distance = current_dynamic['trailing_stop_distance']

            # Check for partial close at dynamic take profit level
            partial_threshold = take_profit_percent * 0.6  # 60% of take profit target
            if pnl_percent >= partial_threshold and not trade['partial_close_done']:
                self.close_partial_position(trade, position, 0.4)  # Close 40%
                trade['partial_close_done'] = True
                print(f"[PARTIAL CLOSE] Closed 40% at {pnl_percent:.1f}% (target: {partial_threshold:.1f}%)")

            # Check for dynamic stop loss
            elif pnl_percent <= -stop_loss_percent:
                print(f"[STOP LOSS] Dynamic stop at {stop_loss_percent:.1f}%")
                self.close_position(trade, position, 'dynamic_stop_loss')
                self.record_trade_outcome(trade, 'dynamic_stop_loss', pnl_percent)
                continue

            # Check for dynamic take profit (full exit)
            elif pnl_percent >= take_profit_percent:
                print(f"[TAKE PROFIT] Dynamic target at {take_profit_percent:.1f}%")
                self.close_position(trade, position, 'dynamic_take_profit')
                self.record_trade_outcome(trade, 'dynamic_take_profit', pnl_percent)
                continue

            # Check dynamic trailing stop
            elif trade['trailing_stop_active']:
                trailing_stop_price = trade['highest_price'] * (1 - trailing_distance)
                if current_price <= trailing_stop_price:
                    print(f"[TRAILING STOP] Dynamic trailing at {trailing_distance:.1%}")
                    self.close_position(trade, position, 'dynamic_trailing_stop')
                    self.record_trade_outcome(trade, 'dynamic_trailing_stop', pnl_percent)
                    continue

            # Dynamic exit based on market conditions
            if self.should_exit_dynamically(trade, position, current_price):
                self.close_position(trade, position, 'dynamic_exit')
                self.record_trade_outcome(trade, 'dynamic_exit', pnl_percent)
                continue

            updated_trades.append(trade)

        # Save updated active trades
        with open(active_file, 'w') as f:
            json.dump(updated_trades, f, indent=2, default=str)

    def close_partial_position(self, trade: Dict, position: Dict, percentage: float):
        """Close partial position"""
        qty_to_close = math.floor(float(position['qty']) * percentage)

        if qty_to_close > 0:
            order_data = {
                'symbol': trade['symbol'],
                'qty': qty_to_close,
                'side': 'sell',
                'type': 'market',
                'time_in_force': 'day',
                'asset_class': 'us_option'
            }

            requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
            print(f"[PARTIAL CLOSE] Closed {qty_to_close} contracts at +20% profit")

    def close_position(self, trade: Dict, position: Dict, reason: str):
        """Close entire position"""
        order_data = {
            'symbol': trade['symbol'],
            'qty': position['qty'],
            'side': 'sell',
            'type': 'market',
            'time_in_force': 'day',
            'asset_class': 'us_option'
        }

        requests.post(f"{self.base_url}/v2/orders", headers=self.headers, json=order_data)
        print(f"[CLOSE] Position closed - Reason: {reason}")

    def should_exit_dynamically(self, trade: Dict, position: Dict, current_price: float) -> bool:
        """Determine if position should be exited based on dynamic conditions"""
        underlying_symbol = trade.get('underlying_symbol', self.ticker)

        # Time-based exit (close if near expiration)
        if 'expiration' in trade:
            days_to_expiry = (datetime.strptime(trade['expiration'], '%Y-%m-%d') - datetime.now()).days
            if days_to_expiry <= 2:
                print("[EXIT] Near expiration")
                return True

        # Get current market conditions
        current_dynamic = self.calculate_dynamic_levels(underlying_symbol)
        entry_momentum = trade.get('market_conditions', {}).get('momentum', 0)
        current_momentum = current_dynamic['momentum']

        # Momentum reversal detection (more sophisticated)
        momentum_change = abs(current_momentum - entry_momentum)
        if momentum_change > 0.08 and current_momentum * entry_momentum < 0:  # Sign reversal
            print(f"[EXIT] Momentum reversal: {entry_momentum:.2%} → {current_momentum:.2%}")
            return True

        # Volatility spike exit
        entry_volatility = trade.get('market_conditions', {}).get('volatility', 0.20)
        current_volatility = current_dynamic['volatility']
        vol_increase = (current_volatility - entry_volatility) / entry_volatility

        if vol_increase > 0.5:  # 50% volatility increase
            print(f"[EXIT] Volatility spike: {entry_volatility:.1%} → {current_volatility:.1%}")
            return True

        # Market regime change exit (if unfavorable)
        entry_regime = trade.get('market_conditions', {}).get('market_regime', 'ranging')
        current_regime = current_dynamic['market_regime']

        # Exit if regime becomes unfavorable for options
        if entry_regime in ['trending', 'ranging'] and current_regime == 'volatile':
            entry_price = trade['entry_price']
            pnl_percent = ((current_price - entry_price) / entry_price) * 100

            # Only exit if we're not in significant profit
            if pnl_percent < 15:
                print(f"[EXIT] Unfavorable regime change: {entry_regime} → {current_regime}")
                return True

        # Price action based exit
        if trade.get('highest_price'):
            entry_price = trade['entry_price']
            highest_price = trade['highest_price']

            # Calculate maximum adverse excursion (MAE) and maximum favorable excursion (MFE)
            mae = min(0, ((current_price - entry_price) / entry_price)) * 100
            mfe = ((highest_price - entry_price) / entry_price) * 100

            # Exit if we've given back too much profit after a good run
            if mfe > 20 and mae < -10:  # Had 20%+ profit but now down 10%+
                print(f"[EXIT] Profit giveback: MFE {mfe:.1f}%, current MAE {mae:.1f}%")
                return True

            # Pullback from high threshold (dynamic based on volatility)
            pullback_threshold = 0.15 + (current_volatility - 0.20) * 0.5  # Adjust for volatility
            pullback = (highest_price - current_price) / highest_price

            if pullback > pullback_threshold:
                print(f"[EXIT] Pullback {pullback:.1%} > threshold {pullback_threshold:.1%}")
                return True

        return False

    def record_trade_outcome(self, trade: Dict, outcome: str, pnl_percent: float = 0):
        """Record trade outcome for learning"""
        trade_record = {
            'symbol': trade['symbol'],
            'entry_time': trade['entry_time'],
            'exit_time': datetime.now().isoformat(),
            'outcome': outcome,
            'pnl_percent': pnl_percent,
            'metrics': trade.get('metrics', {})
        }

        self.trading_history['trades'].append(trade_record)

        # Update ML model based on outcome
        if pnl_percent > 10:
            self.ml_model['success_patterns'].append(trade.get('metrics', {}))
        elif pnl_percent < -5:
            self.ml_model['failure_patterns'].append(trade.get('metrics', {}))

        # Adjust weights based on performance
        self.update_model_weights(pnl_percent)

        self.save_trading_history()
        self.save_ml_model()

        # RL shadow: feed realized outcome back to the agent
        if getattr(self, 'rl_advisor', None) and pnl_percent:
            try:
                self.rl_advisor.record_outcome(trade.get('order_id'), pnl_percent)
                print(f"[RL] Outcome recorded: {pnl_percent:+.1f}%")
            except Exception as e:
                print(f"[RL] record_outcome failed: {e}")

    def update_model_weights(self, pnl_percent: float):
        """Update model weights based on trade performance"""
        learning_rate = 0.01

        if pnl_percent > 0:
            # Successful trade - reinforce current weights slightly
            adjustment = learning_rate * (pnl_percent / 100)
        else:
            # Failed trade - adjust weights
            adjustment = -learning_rate * (abs(pnl_percent) / 100)

        # Apply adjustments with normalization
        total = 0
        for key in self.ml_model['weights']:
            self.ml_model['weights'][key] *= (1 + adjustment)
            total += self.ml_model['weights'][key]

        # Normalize weights to sum to 1
        for key in self.ml_model['weights']:
            self.ml_model['weights'][key] /= total

    def generate_performance_report(self) -> Dict:
        """Generate performance report from trading history"""
        if not self.trading_history['trades']:
            return {'message': 'No trading history available'}

        trades = self.trading_history['trades']

        winning_trades = [t for t in trades if t['pnl_percent'] > 0]
        losing_trades = [t for t in trades if t['pnl_percent'] < 0]

        total_pnl = sum(t['pnl_percent'] for t in trades)
        avg_win = sum(t['pnl_percent'] for t in winning_trades) / len(winning_trades) if winning_trades else 0
        avg_loss = sum(t['pnl_percent'] for t in losing_trades) / len(losing_trades) if losing_trades else 0

        return {
            'total_trades': len(trades),
            'winning_trades': len(winning_trades),
            'losing_trades': len(losing_trades),
            'win_rate': len(winning_trades) / len(trades) * 100 if trades else 0,
            'avg_win_percent': avg_win,
            'avg_loss_percent': avg_loss,
            'total_pnl_percent': total_pnl,
            'current_weights': self.ml_model['weights'],
            'patterns_learned': {
                'success': len(self.ml_model['success_patterns']),
                'failure': len(self.ml_model['failure_patterns'])
            }
        }

    def trade_symbol(self, ticker: str = None, quantity: int = None):
        """Trade specified symbol with quantity"""
        symbol = ticker or self.ticker
        qty = quantity or self.quantity

        if not symbol:
            print("[ERROR] No ticker symbol specified")
            return False

        print(f"[TRADE] Looking for {symbol} options with quantity {qty}")

        # Check options access first
        has_options_access = self.check_options_access()
        if not has_options_access:
            print(f"[WARNING] Alpaca account lacks options trading access")
            print(f"[INFO] Using simulation mode with real market analysis")

        # Get current price
        current_price = self.get_current_price(symbol)
        if not current_price:
            print(f"[ERROR] Cannot get current price for {symbol}")
            return False

        print(f"[PRICE] {symbol} current price: ${current_price:.2f}")

        # Get option chain and select best option
        contracts = self.get_option_contracts(symbol)
        if not contracts:
            print(f"[ERROR] No option contracts found for {symbol}")
            return False

        best_option = self.select_best_option(contracts, current_price)
        if not best_option:
            print(f"[ERROR] No suitable options found for {symbol}")
            return False

        print(f"[SELECTED] {best_option['symbol']} - Score: {best_option.get('score', 0):.2f}")

        # Check if this is a mock contract
        if best_option.get('mock', False):
            print(f"[SIMULATION] Mock option selected - no real order will be placed")
            print(f"[SIMULATION] To enable real options trading, upgrade to Alpaca Pro")
            return True  # Return success for simulation

        # Place real order
        order = self.place_order_with_stops(best_option, qty)
        if order:
            print(f"[SUCCESS] Real order placed for {qty} contracts of {symbol}")
            return True
        else:
            print(f"[ERROR] Failed to place order for {symbol}")
            return False

    def check_options_access(self) -> bool:
        """Check if account has options trading access"""
        try:
            response = requests.get(
                f"{self.data_url}/v2/options/contracts/AAPL",
                headers=self.headers,
                params={'limit': 1}
            )
            return response.status_code == 200
        except:
            return False

    def get_option_contracts(self, ticker: str):
        """Get option contracts for ticker including both calls and puts"""
        # First check if options are available
        if not self.check_options_access():
            print(f"[OPTIONS] Alpaca account does not have options access")
            print(f"[OPTIONS] Generating mock option contracts for {ticker}")
            return self.generate_mock_option_contracts(ticker)

        expiration_start = (datetime.now() + timedelta(days=30)).strftime('%Y-%m-%d')
        expiration_end = (datetime.now() + timedelta(days=90)).strftime('%Y-%m-%d')

        # Get both calls and puts
        all_contracts = []

        for option_type in ['call', 'put']:
            response = requests.get(
                f"{self.data_url}/v2/options/contracts/{ticker}",
                headers=self.headers,
                params={
                    'expiration_date_gte': expiration_start,
                    'expiration_date_lte': expiration_end,
                    'type': option_type
                }
            )

            if response.status_code == 200:
                contracts = response.json().get('option_contracts', [])
                # Add type to each contract
                for contract in contracts:
                    contract['type'] = option_type
                all_contracts.extend(contracts)

        if not all_contracts:
            print(f"[OPTIONS] No real contracts found, using mock contracts for {ticker}")
            return self.generate_mock_option_contracts(ticker)

        return all_contracts

    def calculate_black_scholes_price(self, S: float, K: float, T: float, r: float, sigma: float, option_type: str = 'call') -> float:
        """Calculate option price using Black-Scholes model"""
        from math import log, sqrt, exp
        from scipy.stats import norm

        # Black-Scholes formula
        d1 = (log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * sqrt(T))
        d2 = d1 - sigma * sqrt(T)

        if option_type.lower() == 'call':
            price = S * norm.cdf(d1) - K * exp(-r * T) * norm.cdf(d2)
        else:  # put
            price = K * exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)

        return max(price, 0.01)  # Minimum price of $0.01

    def generate_mock_option_contracts(self, ticker: str):
        """Generate realistic mock option contracts with Black-Scholes pricing"""
        current_price = self.get_current_price(ticker)
        if not current_price:
            return []

        # Calculate volatility for more accurate pricing
        volatility = self.calculate_volatility(ticker)

        # Risk-free rate (approximate current rate)
        risk_free_rate = 0.045  # 4.5% annual rate

        # Generate realistic strikes around current price
        contracts = []
        base_date = datetime.now() + timedelta(days=45)  # ~6 weeks out
        exp_date = base_date.strftime('%Y-%m-%d')
        days_to_exp = 45
        time_to_exp = days_to_exp / 365.0  # Convert to years

        # Generate ITM and OTM strikes
        strikes = [
            current_price * 0.95,  # 5% ITM call
            current_price * 0.97,  # 3% ITM call
            current_price * 1.03,  # 3% OTM call
            current_price * 1.05,  # 5% OTM call
        ]

        for i, strike in enumerate(strikes):
            strike_rounded = round(strike, 2)

            # Calculate realistic prices using Black-Scholes
            call_price = self.calculate_black_scholes_price(
                current_price, strike_rounded, time_to_exp, risk_free_rate, volatility, 'call'
            )
            put_price = self.calculate_black_scholes_price(
                current_price, strike_rounded, time_to_exp, risk_free_rate, volatility, 'put'
            )

            contracts.extend([
                {
                    'symbol': f'{ticker}{base_date.strftime("%y%m%d")}C{int(strike_rounded * 1000):08d}',
                    'strike_price': str(strike_rounded),
                    'expiration_date': exp_date,
                    'type': 'call',
                    'mock': True,
                    'mock_bid': call_price * 0.98,  # Slightly below mid
                    'mock_ask': call_price * 1.02   # Slightly above mid
                },
                {
                    'symbol': f'{ticker}{base_date.strftime("%y%m%d")}P{int(strike_rounded * 1000):08d}',
                    'strike_price': str(strike_rounded),
                    'expiration_date': exp_date,
                    'type': 'put',
                    'mock': True,
                    'mock_bid': put_price * 0.98,
                    'mock_ask': put_price * 1.02
                }
            ])

        return contracts


def main():
    """Command line interface for options trading"""
    parser = argparse.ArgumentParser(description='Smart Options Trader - Trade any symbol with specified quantity')
    parser.add_argument('command', help='Command: SYMBOL QUANTITY (e.g., "IREN 5", "AAPL 3")')
    parser.add_argument('--monitor', action='store_true', help='Monitor existing positions')
    parser.add_argument('--status', action='store_true', help='Show performance status')
    parser.add_argument('--continuous', action='store_true', help='Run continuous monitoring')

    args = parser.parse_args()

    # Parse command for symbol and quantity
    parts = args.command.upper().split()

    if len(parts) == 2:
        symbol, quantity_str = parts
        try:
            quantity = int(quantity_str)
        except ValueError:
            print(f"[ERROR] Invalid quantity: {quantity_str}")
            return 1
    else:
        print(f"[ERROR] Invalid command format. Use: SYMBOL QUANTITY (e.g., 'IREN 5')")
        return 1

    # Initialize trader
    trader = SmartOptionsTrader(ticker=symbol, quantity=quantity)

    if args.status:
        report = trader.generate_performance_report()
        print("\n=== PERFORMANCE REPORT ===")
        for key, value in report.items():
            print(f"{key}: {value}")
        return 0

    if args.monitor:
        print(f"[MONITOR] Checking existing positions...")
        trader.monitor_positions()
        return 0

    if args.continuous:
        print(f"[CONTINUOUS] Starting continuous trading for {symbol} with quantity {quantity}")
        import time
        while True:
            try:
                trader.monitor_positions()
                time.sleep(60)  # Check every minute
            except KeyboardInterrupt:
                print("\n[STOP] Continuous monitoring stopped")
                break
        return 0

    # Execute single trade
    print(f"[START] Smart Options Trader")
    print(f"[TARGET] {symbol} with quantity {quantity}")

    success = trader.trade_symbol()

    if success:
        print(f"[COMPLETE] Trade executed successfully")
        return 0
    else:
        print(f"[FAILED] Trade execution failed")
        return 1


if __name__ == "__main__":
    sys.exit(main())