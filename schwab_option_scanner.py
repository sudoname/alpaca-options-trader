"""
Schwab Options Scanner
Scans for best options based on Greeks, IV, and other criteria
"""

import os
import json
from datetime import datetime, timedelta
from dotenv import load_dotenv
from schwab import auth, client
import logging
from typing import List, Dict, Optional

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SchwabOptionScanner:
    """Scanner for finding optimal options using Schwab data"""

    def __init__(self):
        """Initialize the scanner"""
        load_dotenv()

        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')

        # Load client from tokens
        self.client = auth.client_from_token_file(
            self.token_file,
            self.app_key,
            self.app_secret
        )
        logger.info("Schwab client initialized")

    def scan_ticker(
        self,
        ticker: str,
        option_type: str = 'CALL',
        min_days: int = 20,
        max_days: int = 60,
        strike_count: int = 10,
        min_delta: float = 0.3,
        max_delta: float = 0.7,
        max_iv: float = 100.0
    ) -> List[Dict]:
        """
        Scan options for a single ticker

        Args:
            ticker: Stock symbol
            option_type: 'CALL' or 'PUT'
            min_days: Minimum days to expiration
            max_days: Maximum days to expiration
            strike_count: Number of strikes to retrieve
            min_delta: Minimum delta filter
            max_delta: Maximum delta filter
            max_iv: Maximum implied volatility

        Returns:
            List of option contracts with scores
        """
        logger.info(f"Scanning {ticker} for {option_type}s...")

        try:
            # Calculate date range
            from_date = datetime.now() + timedelta(days=min_days)
            to_date = datetime.now() + timedelta(days=max_days)

            # Get option chain
            contract_type = (client.Client.Options.ContractType.CALL
                           if option_type.upper() == 'CALL'
                           else client.Client.Options.ContractType.PUT)

            response = self.client.get_option_chain(
                ticker,
                contract_type=contract_type,
                strike_count=strike_count,
                include_underlying_quote=True,
                from_date=from_date,
                to_date=to_date
            )

            if response.status_code != 200:
                logger.error(f"Failed to get option chain for {ticker}: {response.status_code}")
                return []

            data = response.json()

            # Extract underlying price
            underlying_price = data.get('underlyingPrice', 0)
            if underlying_price == 0:
                logger.warning(f"No underlying price for {ticker}")
                return []

            logger.info(f"{ticker} underlying price: ${underlying_price}")

            # Parse options
            options_map = (data.get('callExpDateMap', {})
                          if option_type.upper() == 'CALL'
                          else data.get('putExpDateMap', {}))

            options = []

            for exp_date, strikes in options_map.items():
                for strike_price, contracts in strikes.items():
                    if not contracts:
                        continue

                    contract = contracts[0]  # First contract at this strike

                    # Apply filters
                    delta = abs(contract.get('delta', 0))
                    iv = contract.get('volatility', 0)

                    if delta < min_delta or delta > max_delta:
                        continue

                    if iv > max_iv:
                        continue

                    # Calculate score
                    score = self._calculate_score(contract, underlying_price)

                    options.append({
                        'symbol': contract.get('symbol'),
                        'ticker': ticker,
                        'strike': float(strike_price),
                        'expiration': exp_date,
                        'type': option_type,
                        'bid': contract.get('bid', 0),
                        'ask': contract.get('ask', 0),
                        'last': contract.get('last', 0),
                        'volume': contract.get('totalVolume', 0),
                        'open_interest': contract.get('openInterest', 0),
                        'delta': contract.get('delta', 0),
                        'gamma': contract.get('gamma', 0),
                        'theta': contract.get('theta', 0),
                        'vega': contract.get('vega', 0),
                        'iv': contract.get('volatility', 0),
                        'days_to_exp': contract.get('daysToExpiration', 0),
                        'underlying_price': underlying_price,
                        'moneyness': self._calculate_moneyness(
                            float(strike_price),
                            underlying_price,
                            option_type
                        ),
                        'score': score
                    })

            # Sort by score (highest first)
            options.sort(key=lambda x: x['score'], reverse=True)

            logger.info(f"Found {len(options)} options for {ticker} after filtering")
            return options

        except Exception as e:
            logger.error(f"Error scanning {ticker}: {str(e)}")
            return []

    def scan_multiple_tickers(
        self,
        tickers: List[str],
        top_n: int = 5,
        **kwargs
    ) -> List[Dict]:
        """
        Scan multiple tickers and return top options

        Args:
            tickers: List of stock symbols
            top_n: Number of top options to return per ticker
            **kwargs: Additional arguments for scan_ticker

        Returns:
            List of best options across all tickers
        """
        all_options = []

        for ticker in tickers:
            options = self.scan_ticker(ticker, **kwargs)
            all_options.extend(options[:top_n])

        # Sort all options by score
        all_options.sort(key=lambda x: x['score'], reverse=True)

        return all_options

    def _calculate_score(self, option: Dict, underlying_price: float) -> float:
        """
        Calculate a score for an option based on multiple factors

        Higher score = better option
        """
        try:
            delta = abs(option.get('delta', 0))
            gamma = abs(option.get('gamma', 0))
            theta = abs(option.get('theta', 0))
            vega = abs(option.get('vega', 0))
            iv = option.get('volatility', 0)
            volume = option.get('totalVolume', 0)
            open_interest = option.get('openInterest', 0)

            # Bid-ask spread (tighter is better)
            bid = option.get('bid', 0)
            ask = option.get('ask', 0)
            spread = ask - bid if ask > bid else 0
            spread_pct = (spread / ask * 100) if ask > 0 else 100

            # Scoring weights
            score = 0

            # Delta: prefer 0.4-0.6 range (ATM options)
            if 0.4 <= delta <= 0.6:
                score += 30
            elif 0.3 <= delta < 0.4 or 0.6 < delta <= 0.7:
                score += 20
            else:
                score += 10

            # Gamma: higher is better (more responsive to price changes)
            score += min(gamma * 1000, 15)

            # Theta: lower absolute value is better (less time decay)
            score += max(0, 15 - theta * 50)

            # IV: prefer moderate IV (20-40%)
            if 20 <= iv <= 40:
                score += 15
            elif 15 <= iv < 20 or 40 < iv <= 50:
                score += 10
            else:
                score += 5

            # Liquidity: volume and open interest
            if volume > 100:
                score += 10
            elif volume > 50:
                score += 5

            if open_interest > 1000:
                score += 10
            elif open_interest > 500:
                score += 5

            # Spread: tighter is better
            if spread_pct < 2:
                score += 10
            elif spread_pct < 5:
                score += 5

            return round(score, 2)

        except Exception as e:
            logger.error(f"Error calculating score: {str(e)}")
            return 0

    def _calculate_moneyness(self, strike: float, underlying: float, option_type: str) -> str:
        """Calculate if option is ITM, ATM, or OTM"""
        diff_pct = abs(strike - underlying) / underlying * 100

        if option_type.upper() == 'CALL':
            if strike < underlying - underlying * 0.02:
                return 'ITM'
            elif strike > underlying + underlying * 0.02:
                return 'OTM'
            else:
                return 'ATM'
        else:  # PUT
            if strike > underlying + underlying * 0.02:
                return 'ITM'
            elif strike < underlying - underlying * 0.02:
                return 'OTM'
            else:
                return 'ATM'

    def format_option_report(self, options: List[Dict], max_results: int = 10) -> str:
        """Format options into a readable report"""
        if not options:
            return "No options found matching criteria."

        report = []
        report.append("=" * 80)
        report.append("SCHWAB OPTIONS SCANNER RESULTS")
        report.append("=" * 80)
        report.append(f"Scanned at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append(f"Total options found: {len(options)}")
        report.append("\n" + "=" * 80)
        report.append(f"TOP {min(max_results, len(options))} OPTIONS")
        report.append("=" * 80)

        for i, opt in enumerate(options[:max_results], 1):
            report.append(f"\n#{i} - Score: {opt['score']:.1f}/100")
            report.append("-" * 80)
            report.append(f"Ticker: {opt['ticker']} | Type: {opt['type']} | Strike: ${opt['strike']}")
            report.append(f"Expiration: {opt['expiration']} | Days: {opt['days_to_exp']}")
            report.append(f"Underlying: ${opt['underlying_price']:.2f} | Moneyness: {opt['moneyness']}")
            report.append(f"Bid: ${opt['bid']:.2f} | Ask: ${opt['ask']:.2f} | Last: ${opt['last']:.2f}")
            report.append(f"Volume: {opt['volume']} | OI: {opt['open_interest']}")
            report.append(f"Greeks: Delta={opt['delta']:.3f}, Gamma={opt['gamma']:.3f}, Theta={opt['theta']:.3f}, Vega={opt['vega']:.3f}")
            report.append(f"IV: {opt['iv']:.1f}%")
            report.append(f"Symbol: {opt['symbol']}")

        report.append("\n" + "=" * 80)
        return "\n".join(report)


def main():
    """Test the scanner"""
    scanner = SchwabOptionScanner()

    # Scan AAPL for calls
    print("\nScanning AAPL for high-probability calls...")
    print("=" * 80)

    options = scanner.scan_ticker(
        'AAPL',
        option_type='CALL',
        min_days=20,
        max_days=60,
        min_delta=0.35,
        max_delta=0.65,
        max_iv=50.0
    )

    # Print report
    report = scanner.format_option_report(options, max_results=5)
    print(report)

    # Save to file
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    filename = f'option_scan_{timestamp}.txt'
    with open(filename, 'w') as f:
        f.write(report)
    print(f"\nReport saved to: {filename}")

    # Save JSON data
    json_file = f'option_scan_{timestamp}.json'
    with open(json_file, 'w') as f:
        json.dump(options[:10], f, indent=2)
    print(f"Data saved to: {json_file}")


if __name__ == '__main__':
    main()
