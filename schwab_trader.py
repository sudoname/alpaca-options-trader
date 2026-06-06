"""
Main Schwab Options Trader
Combines scanner with trading logic
"""

import os
import json
import time
from datetime import datetime
from dotenv import load_dotenv
from schwab import auth
from schwab_option_scanner import SchwabOptionScanner
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchwabOptionsTrader:
    """Main trading bot using Schwab"""

    def __init__(self, dry_run=None):
        """
        Initialize trader

        Args:
            dry_run: If True, no actual trades will be placed.
                    If None, reads from DRY_RUN env variable (default: True)
        """
        load_dotenv(override=True)

        # Read from .env if not specified
        if dry_run is None:
            dry_run_env = os.getenv('DRY_RUN', 'true').lower()
            self.dry_run = dry_run_env in ['true', '1', 'yes']
        else:
            self.dry_run = dry_run
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')

        # Initialize client
        self.client = auth.client_from_token_file(
            self.token_file,
            self.app_key,
            self.app_secret
        )

        # Initialize scanner
        self.scanner = SchwabOptionScanner()

        # Load trading parameters from .env
        self.min_days = int(os.getenv('MIN_DAYS_TO_EXPIRATION', '90'))
        self.max_days = int(os.getenv('MAX_DAYS_TO_EXPIRATION', '180'))
        self.min_delta = float(os.getenv('MIN_DELTA', '0.35'))
        self.max_delta = float(os.getenv('MAX_DELTA', '0.65'))
        self.max_iv = float(os.getenv('MAX_IV', '50.0'))
        self.max_budget = float(os.getenv('MAX_BUDGET_PER_TRADE', '2000.0'))
        self.min_confidence = float(os.getenv('MIN_CONFIDENCE', '60.0'))

        logger.info(f"Schwab Trader initialized (DRY RUN: {self.dry_run})")
        logger.info(f"Trading Parameters: {self.min_days}-{self.max_days} days, Delta: {self.min_delta}-{self.max_delta}, Max IV: {self.max_iv}%, Budget: ${self.max_budget}")

    def find_best_trade(
        self,
        tickers: list,
        option_type: str = 'CALL',
        budget: float = 500.0,
        **scanner_kwargs
    ):
        """
        Find the best trade opportunity

        Args:
            tickers: List of tickers to scan
            option_type: 'CALL' or 'PUT'
            budget: Maximum budget for the trade
            **scanner_kwargs: Additional scanner parameters

        Returns:
            Best option contract or None
        """
        logger.info(f"Scanning {len(tickers)} tickers for best {option_type}...")

        all_options = []

        for ticker in tickers:
            options = self.scanner.scan_ticker(
                ticker,
                option_type=option_type,
                **scanner_kwargs
            )

            # Filter by budget
            affordable_options = [
                opt for opt in options
                if opt['ask'] * 100 <= budget  # Options are per 100 shares
            ]

            all_options.extend(affordable_options)

        if not all_options:
            logger.warning("No affordable options found")
            return None

        # Sort by score
        all_options.sort(key=lambda x: x['score'], reverse=True)

        best = all_options[0]
        logger.info(f"Best option found: {best['ticker']} ${best['strike']} {best['type']} - Score: {best['score']}")

        return best

    def analyze_option(self, option):
        """
        Perform detailed analysis on an option

        Args:
            option: Option dictionary from scanner

        Returns:
            Analysis dictionary
        """
        analysis = {
            'symbol': option['symbol'],
            'ticker': option['ticker'],
            'recommendation': 'HOLD',
            'confidence': 0,
            'reasons': [],
            'risk_level': 'MEDIUM'
        }

        # Delta analysis
        delta = abs(option['delta'])
        if 0.45 <= delta <= 0.55:
            analysis['reasons'].append("Optimal delta for ATM trading")
            analysis['confidence'] += 20
        elif 0.35 <= delta <= 0.65:
            analysis['reasons'].append("Good delta range")
            analysis['confidence'] += 15

        # IV analysis
        iv = option['iv']
        if 20 <= iv <= 35:
            analysis['reasons'].append("Moderate IV - balanced premium")
            analysis['confidence'] += 20
        elif iv > 50:
            analysis['reasons'].append("High IV - expensive premium")
            analysis['confidence'] -= 10
            analysis['risk_level'] = 'HIGH'

        # Liquidity analysis
        if option['volume'] > 100 and option['open_interest'] > 1000:
            analysis['reasons'].append("Excellent liquidity")
            analysis['confidence'] += 20
        elif option['volume'] > 50:
            analysis['reasons'].append("Good liquidity")
            analysis['confidence'] += 10
        else:
            analysis['reasons'].append("Low liquidity - may be hard to exit")
            analysis['confidence'] -= 15

        # Theta analysis (time decay)
        theta = abs(option['theta'])
        if theta < 0.15:
            analysis['reasons'].append("Low time decay")
            analysis['confidence'] += 15
        elif theta > 0.25:
            analysis['reasons'].append("High time decay - risky for long holds")
            analysis['risk_level'] = 'HIGH'

        # Days to expiration
        days = option['days_to_exp']
        if 30 <= days <= 45:
            analysis['reasons'].append("Optimal time frame")
            analysis['confidence'] += 15
        elif days < 21:
            analysis['reasons'].append("Short expiration - high risk")
            analysis['risk_level'] = 'HIGH'

        # Moneyness
        if option['moneyness'] == 'ATM':
            analysis['reasons'].append("At-the-money - balanced risk/reward")
            analysis['confidence'] += 10

        # Final recommendation
        if analysis['confidence'] >= 70:
            analysis['recommendation'] = 'BUY'
        elif analysis['confidence'] >= 50:
            analysis['recommendation'] = 'HOLD'
        else:
            analysis['recommendation'] = 'PASS'

        return analysis

    def execute_trade(self, option, quantity=1):
        """
        Execute a trade (or simulate in dry run mode)

        Args:
            option: Option contract to trade
            quantity: Number of contracts

        Returns:
            Trade result dictionary
        """
        cost = option['ask'] * 100 * quantity

        trade_details = {
            'timestamp': datetime.now().isoformat(),
            'symbol': option['symbol'],
            'ticker': option['ticker'],
            'type': option['type'],
            'strike': option['strike'],
            'expiration': option['expiration'],
            'quantity': quantity,
            'entry_price': option['ask'],
            'cost': cost,
            'underlying_price': option.get('underlying_price', 0),
            'score': option.get('score', 0),  # ML score for auto-roll logic
            'days_to_exp': option.get('days_to_exp', 0),
            'moneyness': option.get('moneyness', 'UNKNOWN'),
            'greeks': {
                'delta': option['delta'],
                'gamma': option['gamma'],
                'theta': option['theta'],
                'vega': option['vega']
            },
            'iv': option['iv'],
            'status': 'SIMULATED' if self.dry_run else 'EXECUTED'
        }

        if self.dry_run:
            logger.info(f"DRY RUN - Would buy {quantity} contract(s) of {option['symbol']} for ${cost:.2f}")
            logger.info(f"Entry: ${option['ask']:.2f} | Strike: ${option['strike']} | Exp: {option['expiration']}")
        else:
            # Execute actual trade via Schwab API
            logger.info(f"LIVE TRADE - Placing order for {quantity} contract(s) of {option['symbol']}")

            try:
                order_result = self._place_option_order(
                    symbol=option['symbol'],
                    quantity=quantity,
                    price=option['ask'],
                    instruction='BUY_TO_OPEN'
                )

                if order_result['success']:
                    trade_details['status'] = 'EXECUTED'
                    trade_details['order_id'] = order_result['order_id']
                    logger.info(f"✅ Order placed successfully. Order ID: {order_result['order_id']}")
                else:
                    trade_details['status'] = 'FAILED'
                    trade_details['error'] = order_result['error']
                    logger.error(f"❌ Order failed: {order_result['error']}")

            except Exception as e:
                logger.error(f"❌ Error placing order: {str(e)}")
                trade_details['status'] = 'ERROR'
                trade_details['error'] = str(e)

        # Save trade log
        self._log_trade(trade_details)

        return trade_details

    def _place_option_order(self, symbol, quantity, price, instruction='BUY_TO_OPEN'):
        """
        Place an option order via Schwab API

        Args:
            symbol: Option symbol
            quantity: Number of contracts
            price: Limit price
            instruction: BUY_TO_OPEN, SELL_TO_CLOSE, etc.

        Returns:
            Dict with success status and order_id or error
        """
        try:
            from schwab.orders.options import option_buy_to_open_limit

            # Build order object
            order = option_buy_to_open_limit(
                symbol=symbol,
                quantity=quantity,
                price=price
            )

            # Get account hash (you may need to get this from account info)
            # For now, using a placeholder - you'll need to fetch this
            account_hash = os.getenv('SCHWAB_ACCOUNT_HASH')

            if not account_hash:
                logger.error("SCHWAB_ACCOUNT_HASH not set in .env")
                return {
                    'success': False,
                    'error': 'Account hash not configured. Add SCHWAB_ACCOUNT_HASH to .env'
                }

            # Place the order
            response = self.client.place_order(account_hash, order)

            if response.status_code == 201:
                # Order placed successfully
                # Extract order ID from response headers
                order_id = response.headers.get('Location', '').split('/')[-1]

                return {
                    'success': True,
                    'order_id': order_id,
                    'status_code': response.status_code
                }
            else:
                return {
                    'success': False,
                    'error': f"Order failed with status {response.status_code}: {response.text}",
                    'status_code': response.status_code
                }

        except Exception as e:
            logger.error(f"Exception placing order: {str(e)}")
            return {
                'success': False,
                'error': str(e)
            }

    def _log_trade(self, trade):
        """Log trade to file"""
        log_file = 'schwab_trades.json'

        try:
            if os.path.exists(log_file):
                with open(log_file, 'r') as f:
                    trades = json.load(f)
            else:
                trades = []

            trades.append(trade)

            with open(log_file, 'w') as f:
                json.dump(trades, f, indent=2)

            logger.info(f"Trade logged to {log_file}")

        except Exception as e:
            logger.error(f"Failed to log trade: {str(e)}")

    def run_scan_and_trade(
        self,
        tickers: list,
        budget: float = None,
        min_confidence: float = None
    ):
        """
        Complete workflow: scan, analyze, and trade

        Args:
            tickers: List of tickers to scan
            budget: Trading budget (uses .env if None)
            min_confidence: Minimum confidence score to trade (uses .env if None)

        Returns:
            Trade result or None
        """
        # Use env values if not provided
        if budget is None:
            budget = self.max_budget
        if min_confidence is None:
            min_confidence = self.min_confidence

        logger.info("=" * 80)
        logger.info("STARTING SCHWAB OPTIONS TRADING SCAN")
        logger.info("=" * 80)

        # Find best option using env parameters
        best_option = self.find_best_trade(
            tickers=tickers,
            budget=budget,
            min_days=self.min_days,
            max_days=self.max_days,
            min_delta=self.min_delta,
            max_delta=self.max_delta,
            max_iv=self.max_iv
        )

        if not best_option:
            logger.warning("No suitable options found")
            return None

        # Analyze
        logger.info("\nAnalyzing best option...")
        analysis = self.analyze_option(best_option)

        logger.info(f"\nAnalysis Results:")
        logger.info(f"Recommendation: {analysis['recommendation']}")
        logger.info(f"Confidence: {analysis['confidence']}%")
        logger.info(f"Risk Level: {analysis['risk_level']}")
        logger.info(f"Reasons:")
        for reason in analysis['reasons']:
            logger.info(f"  - {reason}")

        # Trade if confidence is high enough
        if analysis['confidence'] >= min_confidence and analysis['recommendation'] == 'BUY':
            logger.info(f"\nConfidence {analysis['confidence']}% >= {min_confidence}% - Executing trade...")
            result = self.execute_trade(best_option, quantity=1)
            return result
        else:
            logger.info(f"\nConfidence {analysis['confidence']}% < {min_confidence}% - Skipping trade")
            return None


def main():
    """Run the trader"""

    # Load tickers from supported_tickers.json
    ticker_file = 'supported_tickers.json'
    if os.path.exists(ticker_file):
        with open(ticker_file, 'r') as f:
            ticker_data = json.load(f)
            TICKERS = list(set(ticker_data.get('tickers', [])))  # Remove duplicates
            logger.info(f"Loaded {len(TICKERS)} tickers from {ticker_file}")
    else:
        # Fallback to default tickers
        TICKERS = ['AAPL', 'MSFT', 'GOOGL', 'NVDA', 'TSLA']
        logger.info(f"Using default tickers: {TICKERS}")

    # Initialize trader - reads all config from .env
    trader = SchwabOptionsTrader(dry_run=None)

    # Run scan and trade - uses .env parameters
    result = trader.run_scan_and_trade(tickers=TICKERS)

    if result:
        logger.info("\n" + "=" * 80)
        logger.info("TRADE EXECUTED")
        logger.info("=" * 80)
        logger.info(json.dumps(result, indent=2))
    else:
        logger.info("\n" + "=" * 80)
        logger.info("NO TRADE EXECUTED")
        logger.info("=" * 80)


if __name__ == '__main__':
    main()
