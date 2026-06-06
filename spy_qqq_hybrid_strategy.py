"""
SPY + QQQ Hybrid Strategy - FIXED VERSION
- Trades both SPY and QQQ
- Max premium: $0.50 (affordable contracts)
- Delta range: 0.25-0.35 (moderate risk)
- Automatically switches between 1DTE and 2DTE
- PDT protection built-in
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from schwab_trader import SchwabOptionsTrader
from schwab import auth, client
from schwab.client import Client
from pdt_tracker import PDTTracker
import time

load_dotenv()


class TelegramNotifier:
    """Send Telegram notifications"""
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)

    def send(self, message):
        if not self.enabled:
            print("[TELEGRAM] Not configured")
            return

        try:
            url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
            data = {
                'chat_id': self.chat_id,
                'text': message,
                'parse_mode': 'Markdown'
            }
            response = requests.post(url, data=data, timeout=10)
            if response.status_code == 200:
                print("[TELEGRAM] Sent!")
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")


class SPYQQQHybridStrategy:
    def __init__(self):
        self.trader = SchwabOptionsTrader(dry_run=False)
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.client = auth.client_from_token_file(
            self.token_file, self.app_key, self.app_secret
        )
        self.log_file = 'spy_qqq_hybrid_trades.json'
        self.telegram = TelegramNotifier()
        self.pdt = PDTTracker()

        # TRADE BOTH SPY AND QQQ
        self.tickers = ['SPY', 'QQQ']

        # ENHANCED PARAMETERS
        self.target_delta_min = 0.25  # Updated: Lower risk tolerance
        self.target_delta_max = 0.35  # Updated: Lower risk tolerance
        self.min_volume = 100
        self.min_open_interest = 500
        self.min_confidence = 70

        # MAX PREMIUM: $6.00 (REALISTIC FOR SPY/QQQ)
        self.max_premium = 6.00  # Updated from $1.00

        # 1DTE PARAMETERS
        self.dte_1_profit_target = 0.20
        self.dte_1_stop_loss = -0.30
        self.dte_1_early_stop = -0.20
        self.dte_1_trailing_stop = 0.10
        self.dte_1_monitor_interval = 900  # 15 min

        # 2DTE PARAMETERS
        self.dte_2_profit_target = 0.25
        self.dte_2_stop_loss = -0.40
        self.dte_2_early_stop = -0.25
        self.dte_2_trailing_stop = 0.12
        self.dte_2_monitor_interval = 1800  # 30 min

        self.mode = 'AUTO'

    def determine_trading_mode(self):
        """Determine 1DTE or 2DTE based on PDT status"""
        print("\n[HYBRID] Determining trading mode...")

        pdt_status = self.pdt.get_status_message()
        remaining = pdt_status['remaining']
        count = pdt_status['count']

        print(f"[PDT] Day trades: {count}/3 | Remaining: {remaining}")

        if self.mode == 'FORCE_1DTE':
            mode = '1DTE'
            reason = 'Forced by user'
        elif self.mode == 'FORCE_2DTE':
            mode = '2DTE'
            reason = 'Forced by user'
        elif remaining == 0:
            mode = '2DTE'
            reason = 'PDT limit reached'
        elif remaining == 1:
            day_of_week = datetime.now().weekday()
            if day_of_week == 4:  # Friday
                mode = '1DTE'
                reason = 'Last day trade - Friday'
            else:
                mode = '2DTE'
                reason = 'Saving last day trade'
        else:
            day_of_week = datetime.now().weekday()
            if day_of_week in [0, 2, 4]:  # Mon, Wed, Fri
                mode = '1DTE'
                reason = f'{remaining} day trades - preferred day'
            else:
                mode = '2DTE'
                reason = f'{remaining} day trades - non-preferred day'

        print(f"[HYBRID] Mode: {mode} | Reason: {reason}")

        return {
            'mode': mode,
            'reason': reason,
            'pdt_status': pdt_status
        }

    def analyze_ticker(self, ticker):
        """Analyze market direction for a ticker"""
        print(f"\n[ANALYSIS] Analyzing {ticker}...")

        # Get current price
        response = self.client.get_quote(ticker)
        if response.status_code != 200:
            print(f"[ERROR] Failed to get {ticker} quote")
            return None

        data = response.json().get(ticker, {}).get('quote', {})

        price = data.get('lastPrice', 0)
        open_price = data.get('openPrice', price)
        close_prev = data.get('closePrice', price)
        change = data.get('netPercentChangeInDouble', 0)
        volume = data.get('totalVolume', 0)
        high = data.get('highPrice', price)
        low = data.get('lowPrice', price)

        print(f"[{ticker}] Price: ${price:.2f}")
        print(f"[{ticker}] Change: {change:.2f}%")

        # Get VIX (with error handling)
        vix_level = 15  # Default
        vix_change = 0
        try:
            vix_response = self.client.get_quote('VIX')
            if vix_response.status_code == 200:
                vix_data = vix_response.json().get('VIX', {}).get('quote', {})
                vix_level = vix_data.get('lastPrice', 15)
                vix_change = vix_data.get('netPercentChangeInDouble', 0)
                print(f"[VIX] Level: {vix_level:.2f}")
        except:
            print(f"[VIX] Using default: 15")

        # Signal analysis
        bullish_signals = 0
        bearish_signals = 0
        skip_reasons = []

        # Filter: High VIX
        if vix_level > 30:
            skip_reasons.append(f"VIX too high ({vix_level:.1f})")

        # Filter: Large gap
        gap = ((open_price - close_prev) / close_prev) * 100
        if abs(gap) > 1.0:
            skip_reasons.append(f"Large gap ({gap:+.2f}%)")

        # Signal 1: Momentum
        if change > 0.3:
            bullish_signals += 2
        elif change > 0.1:
            bullish_signals += 1
        elif change < -0.3:
            bearish_signals += 2
        elif change < -0.1:
            bearish_signals += 1

        # Signal 2: Price position
        intraday_position = ((price - low) / (high - low)) if high > low else 0.5
        if intraday_position > 0.7:
            bullish_signals += 1
        elif intraday_position < 0.3:
            bearish_signals += 1

        # Signal 3: VIX direction
        if vix_change < -5:
            bullish_signals += 1
        elif vix_change > 5:
            bearish_signals += 1

        # Signal 4: VIX level
        if vix_level > 25:
            bearish_signals += 1
        elif vix_level < 15:
            bullish_signals += 1

        # Signal 5: Gap
        if 0.3 < gap < 1.0:
            bullish_signals += 1
        elif -1.0 < gap < -0.3:
            bearish_signals += 1

        # Calculate confidence
        total_signals = bullish_signals + bearish_signals
        if total_signals == 0:
            confidence = 0
            direction = None
        elif bullish_signals > bearish_signals:
            direction = 'CALL'
            confidence = (bullish_signals / total_signals) * 100
        elif bearish_signals > bullish_signals:
            direction = 'PUT'
            confidence = (bearish_signals / total_signals) * 100
        else:
            direction = 'CALL' if change >= 0 else 'PUT'
            confidence = 50

        print(f"[{ticker}] Bullish: {bullish_signals} | Bearish: {bearish_signals}")
        print(f"[{ticker}] {direction} with {confidence:.0f}% confidence")

        should_trade = True
        if skip_reasons or confidence < self.min_confidence:
            should_trade = False

        return {
            'ticker': ticker,
            'direction': direction,
            'confidence': confidence,
            'price': price,
            'change': change,
            'vix_level': vix_level,
            'gap': gap,
            'should_trade': should_trade,
            'skip_reasons': skip_reasons
        }

    def get_next_friday_expiration(self, dte_mode):
        """Get next Friday expiration date (options expire on Fridays)"""
        today = datetime.now()
        current_weekday = today.weekday()  # 0=Mon, 4=Fri

        if dte_mode == 1:
            # 1DTE: This week's Friday (or next Friday if today is Fri/Sat/Sun)
            if current_weekday < 4:  # Mon-Thu
                days_until_friday = 4 - current_weekday
            else:  # Fri-Sun
                days_until_friday = 7 - current_weekday + 4
        else:
            # 2DTE: Next week's Friday (give more time to hold)
            if current_weekday < 4:  # Mon-Thu
                days_until_friday = 7 + (4 - current_weekday)  # Next week's Friday
            else:  # Fri-Sun
                days_until_friday = 14 - current_weekday + 4  # Week after next Friday

        expiration_date = today + timedelta(days=days_until_friday)
        return expiration_date.date()

    def find_option(self, ticker, direction, dte_target):
        """Find option with max premium $0.50"""
        print(f"\n[SCAN] Scanning {ticker} {dte_target}DTE {direction} options...")
        print(f"[FILTER] Max premium: ${self.max_premium:.2f}")

        # Get next Friday expiration (FIX: Options expire on Fridays, not Saturdays!)
        target_date = self.get_next_friday_expiration(dte_target)
        from_date = target_date
        to_date = target_date  # Search for exact expiration date

        print(f"[EXPIRATION] Targeting {target_date.strftime('%A %B %d, %Y')}")

        # Convert direction to enum (FIX FOR ERROR)
        if direction == 'CALL':
            contract_type = Client.Options.ContractType.CALL
        else:
            contract_type = Client.Options.ContractType.PUT

        # Get option chain (expanded strike range to find cheaper options)
        response = self.client.get_option_chain(
            ticker,
            contract_type=contract_type,
            strike_count=60,  # Increased from 30 to find more affordable options
            include_underlying_quote=True,
            from_date=from_date,
            to_date=to_date
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to get option chain: {response.status_code}")
            return None

        chain_data = response.json()
        underlying_price = chain_data.get('underlyingPrice', 0)

        print(f"[{ticker}] Current price: ${underlying_price:.2f}")

        # Extract options
        option_map = chain_data.get('putExpDateMap' if direction == 'PUT' else 'callExpDateMap', {})

        all_options = []
        for exp_date, strikes in option_map.items():
            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    # NOTE: We already specify exact expiration date to API
                    # DTE filter is not needed and causes issues when Friday logic
                    # returns 7-day expiration for end-of-week trading

                    ask = contract.get('ask', 999)
                    bid = contract.get('bid', 0)
                    strike = contract.get('strikePrice', 0)
                    delta = contract.get('delta', 0)
                    days_to_exp = contract.get('daysToExpiration', 0)

                    # Check if OTM
                    is_otm = False
                    if direction == 'CALL':
                        is_otm = strike > underlying_price
                    else:
                        is_otm = strike < underlying_price

                    volume = contract.get('totalVolume', 0)
                    open_interest = contract.get('openInterest', 0)

                    # Apply filters
                    if (is_otm and
                        ask > 0 and ask <= self.max_premium and
                        abs(delta) >= self.target_delta_min and
                        abs(delta) <= self.target_delta_max and
                        volume >= self.min_volume and
                        open_interest >= self.min_open_interest):

                        all_options.append({
                            'symbol': contract.get('symbol'),
                            'strike': strike,
                            'ask': ask,
                            'bid': bid,
                            'delta': abs(delta),
                            'volume': volume,
                            'open_interest': open_interest,
                            'dte': days_to_exp
                        })

        if not all_options:
            print(f"[ERROR] No {ticker} options found with premium <= ${self.max_premium:.2f}")
            return None

        # Score options (optimize for MAX PROFIT!)
        def score_option(opt):
            # Delta score: HIGHEST WEIGHT - prefer middle of range (0.30)
            # Better delta = higher win probability = more profit
            delta_score = (100 - abs((opt['delta'] - 0.30) * 200)) * 2  # 2x weight!

            # Volume score: high liquidity = better fills and exits
            volume_score = min(opt['volume'] / 1000 * 20, 60)  # Increased weight

            # Open Interest score: more OI = better liquidity
            oi_score = min(opt['open_interest'] / 1000 * 15, 40)

            # Spread score: tighter spread = better entry/exit
            spread = opt['ask'] - opt['bid']
            spread_pct = spread / opt['ask'] if opt['ask'] > 0 else 1
            spread_score = max(40 - (spread_pct * 100), 0)  # Increased weight

            # Risk/Reward score: premium vs profit potential
            # Lower premium = more % gain potential on same move
            risk_reward_score = max(30 - (opt['ask'] * 3), 0)

            return delta_score + volume_score + oi_score + spread_score + risk_reward_score

        all_options.sort(key=score_option, reverse=True)

        best = all_options[0]
        print(f"\n[FOUND] Best {ticker} option:")
        print(f"  Symbol: {best['symbol']}")
        print(f"  Strike: ${best['strike']:.2f}")
        print(f"  Premium: ${best['ask']:.2f}")
        print(f"  Delta: {best['delta']:.3f}")
        print(f"  DTE: {best['dte']}")

        return best

    def execute_trade(self, ticker, option, analysis, mode_info):
        """Execute trade"""
        print(f"\n[EXECUTE] Placing {mode_info['mode']} order for {option['symbol']}...")

        from schwab.orders.options import option_buy_to_open_limit

        order = option_buy_to_open_limit(
            option['symbol'],
            1,  # 1 contract
            str(option['ask'])
        )

        account_hash = os.getenv('SCHWAB_ACCOUNT_HASH')
        response = self.client.place_order(account_hash, order)

        if response.status_code == 201:
            order_id = response.headers.get('Location', '').split('/')[-1]
            print(f"[SUCCESS] Order placed! Order ID: {order_id}")

            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order_id,
                'ticker': ticker,
                'symbol': option['symbol'],
                'type': analysis['direction'],
                'strike': option['strike'],
                'premium': option['ask'],
                'cost': option['ask'] * 100,
                'delta': option['delta'],
                'dte': option['dte'],
                'mode': mode_info['mode'],
                'pdt_status': mode_info['pdt_status'],
                'analysis': analysis,
                'status': 'OPEN',
                'entry_time': datetime.now().isoformat()
            }

            self.log_trade(trade_record)

            # TELEGRAM
            pdt_status = mode_info['pdt_status']
            msg = f"""
*{ticker} {mode_info['mode']} TRADE OPENED*

*Mode:* {mode_info['mode']}
*Reason:* {mode_info['reason']}

*PDT Status:*
Day Trades: {pdt_status['count']}/3
Remaining: {pdt_status['remaining']}

*Trade:*
Type: {analysis['direction']}
Strike: ${option['strike']:.2f}
Premium: ${option['ask']:.2f}
Cost: ${option['ask'] * 100:.2f}
Delta: {option['delta']:.3f}

*Market:*
{ticker}: ${analysis['price']:.2f} ({analysis['change']:+.2f}%)
VIX: {analysis['vix_level']:.2f}
Confidence: {analysis['confidence']:.0f}%

{'Will close today (day trade)' if mode_info['mode'] == '1DTE' else 'Will hold overnight'}

Order: `{order_id}`
"""
            self.telegram.send(msg)

            return trade_record
        else:
            print(f"[FAILED] Order placement failed: {response.status_code}")

            # Send failure notification
            pdt_status = mode_info['pdt_status']
            self.telegram.send(f"""❌ *{ticker} {mode_info['mode']} ORDER FAILED*

*Mode:* {mode_info['mode']}
*PDT Status:* {pdt_status['count']}/3

*Attempted Trade:*
Type: {analysis['direction']}
Strike: ${option['strike']:.2f}
Premium: ${option['ask']:.2f}

*Error:*
HTTP {response.status_code} - Order placement failed

⚠️ Will retry tomorrow at 9:00 AM CST""")

            return None

    def log_trade(self, trade):
        """Log trade"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                trades = json.load(f)
        else:
            trades = []

        trades.append(trade)

        with open(self.log_file, 'w') as f:
            json.dump(trades, f, indent=2)

        print(f"[LOG] Logged to {self.log_file}")

    def run_daily_strategy(self):
        """Run complete strategy"""
        print("=" * 60)
        print("SPY + QQQ HYBRID STRATEGY")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Clean PDT records
        self.pdt.clean_old_trades()

        # Determine mode
        mode_info = self.determine_trading_mode()

        # Analyze both SPY and QQQ
        best_trade = None
        best_option = None

        for ticker in self.tickers:
            analysis = self.analyze_ticker(ticker)

            if not analysis:
                continue

            if not analysis['should_trade']:
                print(f"[{ticker}] Skipped - conditions not met")
                continue

            # Find option
            dte_target = 1 if mode_info['mode'] == '1DTE' else 2
            option = self.find_option(ticker, analysis['direction'], dte_target)

            if option:
                # Score this trade
                score = analysis['confidence'] * (1 - option['ask'] / self.max_premium)
                print(f"[{ticker}] Trade score: {score:.1f}")

                if best_trade is None or score > best_trade['score']:
                    best_trade = {
                        'ticker': ticker,
                        'analysis': analysis,
                        'option': option,
                        'score': score
                    }

        if not best_trade:
            print("\n[ABORT] No suitable trades found")
            self.telegram.send(f"""
*SPY/QQQ HYBRID - NO TRADE*

No options found meeting criteria:
- Max premium: ${self.max_premium:.2f}
- Min confidence: {self.min_confidence}%

PDT: {mode_info['pdt_status']['count']}/3
""")
            return

        # Execute best trade
        print(f"\n[BEST] {best_trade['ticker']} with score {best_trade['score']:.1f}")

        trade = self.execute_trade(
            best_trade['ticker'],
            best_trade['option'],
            best_trade['analysis'],
            mode_info
        )

        if trade and mode_info['mode'] == '1DTE':
            # Log day trade
            self.pdt.log_day_trade({
                'symbol': trade['symbol'],
                'entry_time': trade['entry_time'],
                'exit_time': datetime.now().isoformat(),
                'profit': 0,
                'order_id': trade['order_id']
            })

        print("\n" + "=" * 60)
        print("STRATEGY COMPLETE")
        print("=" * 60)


def main():
    strategy = SPYQQQHybridStrategy()
    strategy.run_daily_strategy()


if __name__ == '__main__':
    main()
