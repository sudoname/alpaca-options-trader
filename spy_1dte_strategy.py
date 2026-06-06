"""
SPY 1DTE Options Strategy - ENHANCED WIN RATE VERSION
- Runs daily at 10:00 AM (waits 30min after market open for direction confirmation)
- Scans for OTM options with premium < $10 and delta 0.35-0.40
- Executes CALL or PUT based on multi-factor market analysis
- Trade filtering: Only high-confidence setups (70%+)
- Closes at 20% profit with trailing stop and Telegram notifications
- Intraday monitoring every 15 minutes with early stop loss
"""

import os
import json
import requests
from datetime import datetime, timedelta
from dotenv import load_dotenv
from schwab_trader import SchwabOptionsTrader
from schwab import auth
import time

load_dotenv()


class TelegramNotifier:
    """Send Telegram notifications for trades"""
    def __init__(self):
        self.bot_token = os.getenv('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = os.getenv('TELEGRAM_CHAT_ID', '')
        self.enabled = bool(self.bot_token and self.chat_id)

    def send(self, message):
        """Send message to Telegram"""
        if not self.enabled:
            print("[TELEGRAM] Not configured, skipping notification")
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
                print("[TELEGRAM] Notification sent!")
            else:
                print(f"[TELEGRAM] Failed: {response.status_code}")
        except Exception as e:
            print(f"[TELEGRAM] Error: {e}")


class SPY1DTEStrategy:
    def __init__(self):
        self.trader = SchwabOptionsTrader(dry_run=False)
        self.token_file = os.getenv('SCHWAB_TOKEN_FILE', 'schwab_tokens.json')
        self.app_key = os.getenv('SCHWAB_APP_KEY')
        self.app_secret = os.getenv('SCHWAB_APP_SECRET')
        self.client = auth.client_from_token_file(
            self.token_file, self.app_key, self.app_secret
        )
        self.log_file = 'spy_1dte_trades.json'
        self.telegram = TelegramNotifier()

        # RL advisory layer (shadow mode: observes & learns, never overrides)
        self.rl_advisor = None
        try:
            from rl_wrapper import RLAdvisor, rl_enabled
            if rl_enabled():
                self.rl_advisor = RLAdvisor(strat_name='spy_1dte')
                print("[RL] Advisor active (shadow mode)")
        except Exception as e:
            print(f"[RL] Advisor unavailable: {e}")

        # Sentiment (Fear & Greed) risk filter. Fail-open: any failure here must
        # never block a trade.
        self.sentiment_service = None
        try:
            from sentiment import SentimentService, SchwabMarketDataProvider, SentimentConfig
            if SentimentConfig.from_env().enabled:
                self.sentiment_service = SentimentService(
                    SchwabMarketDataProvider(self.client)
                )
                print("[SENTIMENT] Fear & Greed filter active")
        except Exception as e:
            print(f"[SENTIMENT] Filter unavailable: {e}")

        # ENHANCED PARAMETERS FOR WIN RATE IMPROVEMENT
        self.target_delta_min = 0.35  # Tightened delta range
        self.target_delta_max = 0.40  # More predictable P/L
        self.min_volume = 100  # Minimum option volume
        self.min_open_interest = 500  # Minimum open interest
        self.profit_target = 0.20  # 20% profit target
        self.stop_loss = -0.30  # 30% stop loss
        self.trailing_stop = 0.10  # 10% trailing stop from peak
        self.early_stop_loss = -0.20  # 20% early stop by 11 AM
        self.min_confidence = 70  # Minimum confidence to trade
        self.monitor_interval = 900  # Check every 15 minutes (900 seconds)

    def analyze_market_direction(self):
        """
        ENHANCED: Analyze market direction with better technical indicators
        - Wait until 10:00 AM for first 30min confirmation
        - Use real VIX data
        - Add volume analysis
        - Add intraday momentum
        - Filter out low-confidence setups
        """
        print("\n[ANALYSIS] Analyzing market direction at 10:00 AM...")

        # Get SPY current price and intraday data
        spy_response = self.client.get_quote('SPY')
        spy_data = spy_response.json().get('SPY', {}).get('quote', {})

        spy_price = spy_data.get('lastPrice', 0)
        spy_open = spy_data.get('openPrice', spy_price)
        spy_close_prev = spy_data.get('closePrice', spy_price)
        spy_change = spy_data.get('netPercentChangeInDouble', 0)
        spy_volume = spy_data.get('totalVolume', 0)
        spy_high = spy_data.get('highPrice', spy_price)
        spy_low = spy_data.get('lowPrice', spy_price)

        print(f"[SPY] Price: ${spy_price:.2f}")
        print(f"[SPY] Open: ${spy_open:.2f}")
        print(f"[SPY] Prev Close: ${spy_close_prev:.2f}")
        print(f"[SPY] Change: {spy_change:.2f}%")
        print(f"[SPY] Volume: {spy_volume:,}")

        # Get VIX for volatility assessment (REAL DATA)
        vix_response = self.client.get_quote('$VIX.X')
        vix_data = vix_response.json().get('$VIX.X', {}).get('quote', {})
        vix_level = vix_data.get('lastPrice', 15)
        vix_change = vix_data.get('netPercentChangeInDouble', 0)

        print(f"[VIX] Level: {vix_level:.2f} ({vix_change:+.2f}%)")

        # Decision logic with weighted signals
        bullish_signals = 0
        bearish_signals = 0
        skip_reasons = []

        # FILTER 1: Skip if VIX too high (unpredictable)
        if vix_level > 30:
            skip_reasons.append(f"VIX too high ({vix_level:.1f} > 30)")

        # FILTER 2: Skip if gap too large (uncertain)
        gap = ((spy_open - spy_close_prev) / spy_close_prev) * 100
        if abs(gap) > 1.0:
            skip_reasons.append(f"Large gap ({gap:+.2f}% > 1.0%)")

        # Signal 1: Intraday momentum (first 30 minutes 9:30-10:00)
        # Strong signal if price is moving decisively
        intraday_range = ((spy_high - spy_low) / spy_low) * 100
        intraday_position = ((spy_price - spy_low) / (spy_high - spy_low)) if spy_high > spy_low else 0.5

        if spy_change > 0.3:
            bullish_signals += 2
            print("[SIGNAL] Strong intraday momentum (+2 bullish)")
        elif spy_change > 0.1:
            bullish_signals += 1
            print("[SIGNAL] Positive intraday momentum (+1 bullish)")
        elif spy_change < -0.3:
            bearish_signals += 2
            print("[SIGNAL] Strong bearish momentum (+2 bearish)")
        elif spy_change < -0.1:
            bearish_signals += 1
            print("[SIGNAL] Negative momentum (+1 bearish)")

        # Signal 2: Price position in range (buying/selling pressure)
        if intraday_position > 0.7:
            bullish_signals += 1
            print(f"[SIGNAL] Trading near highs ({intraday_position*100:.0f}% of range, +1 bullish)")
        elif intraday_position < 0.3:
            bearish_signals += 1
            print(f"[SIGNAL] Trading near lows ({intraday_position*100:.0f}% of range, +1 bearish)")

        # Signal 3: VIX direction (inverse correlation with SPY)
        if vix_change < -5:  # VIX dropping significantly
            bullish_signals += 1
            print(f"[SIGNAL] VIX falling sharply ({vix_change:.1f}%, +1 bullish)")
        elif vix_change > 5:  # VIX spiking
            bearish_signals += 1
            print(f"[SIGNAL] VIX spiking ({vix_change:+.1f}%, +1 bearish)")

        # Signal 4: VIX absolute level
        if vix_level > 25:
            bearish_signals += 1
            print(f"[SIGNAL] High VIX - elevated fear (+1 bearish)")
        elif vix_level < 15:
            bullish_signals += 1
            print(f"[SIGNAL] Low VIX - complacency (+1 bullish)")

        # Signal 5: Gap analysis (moderate gaps only)
        if 0.3 < gap < 1.0:
            bullish_signals += 1
            print(f"[SIGNAL] Moderate gap up ({gap:.2f}%, +1 bullish)")
        elif -1.0 < gap < -0.3:
            bearish_signals += 1
            print(f"[SIGNAL] Moderate gap down ({gap:.2f}%, +1 bearish)")

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
            # Tie - default to intraday momentum
            direction = 'CALL' if spy_change >= 0 else 'PUT'
            confidence = 50

        print(f"\n[SCORE] Bullish: {bullish_signals} | Bearish: {bearish_signals}")
        print(f"[DECISION] {direction} with {confidence:.0f}% confidence")

        # TRADE FILTERING: Check if confidence is high enough
        should_trade = True
        if skip_reasons:
            print(f"\n[FILTER] Skip reasons:")
            for reason in skip_reasons:
                print(f"  - {reason}")
            should_trade = False

        if confidence < self.min_confidence:
            print(f"\n[FILTER] Confidence too low ({confidence:.0f}% < {self.min_confidence}%)")
            should_trade = False

        return {
            'direction': direction,
            'confidence': confidence,
            'spy_price': spy_price,
            'spy_change': spy_change,
            'spy_open': spy_open,
            'spy_close_prev': spy_close_prev,
            'vix_level': vix_level,
            'vix_change': vix_change,
            'gap': gap,
            'intraday_range': intraday_range,
            'should_trade': should_trade,
            'skip_reasons': skip_reasons
        }

    def find_1dte_option(self, direction):
        """
        Find 1DTE SPY options:
        - Out of the money (OTM)
        - Premium < $10
        - Expires tomorrow
        """
        print(f"\n[SCAN] Scanning for 1DTE SPY {direction} options...")

        # Calculate tomorrow's date for 1DTE
        tomorrow = datetime.now() + timedelta(days=1)
        from_date = tomorrow.strftime('%Y-%m-%d')
        to_date = (tomorrow + timedelta(days=2)).strftime('%Y-%m-%d')

        # Get SPY option chain
        response = self.client.get_option_chain(
            'SPY',
            contract_type=direction,
            strike_count=20,
            include_underlying_quote=True,
            from_date=from_date,
            to_date=to_date
        )

        if response.status_code != 200:
            print(f"[ERROR] Failed to get option chain: {response.status_code}")
            return None

        chain_data = response.json()
        spy_price = chain_data.get('underlyingPrice', 0)

        print(f"[SPY] Current price: ${spy_price:.2f}")

        # Extract options
        option_map = chain_data.get('putExpDateMap' if direction == 'PUT' else 'callExpDateMap', {})

        all_options = []
        for exp_date, strikes in option_map.items():
            for strike_price, contracts in strikes.items():
                for contract in contracts:
                    # Check if 1DTE
                    exp_str = contract.get('expirationDate', '')
                    days_to_exp = contract.get('daysToExpiration', 999)

                    if days_to_exp <= 1:  # 0DTE or 1DTE
                        ask = contract.get('ask', 999)
                        bid = contract.get('bid', 0)
                        strike = contract.get('strikePrice', 0)
                        delta = contract.get('delta', 0)

                        # Filter: OTM and premium < $10
                        is_otm = False
                        if direction == 'CALL':
                            is_otm = strike > spy_price
                        else:  # PUT
                            is_otm = strike < spy_price

                        volume = contract.get('totalVolume', 0)
                        open_interest = contract.get('openInterest', 0)

                        # OPTIMIZED FILTERS
                        if (is_otm and
                            ask > 0 and ask < 10 and
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
                                'iv': contract.get('volatility', 0),
                                'dte': days_to_exp,
                                'expiration': exp_str
                            })

        if not all_options:
            print("[ERROR] No suitable 1DTE options found")
            return None

        # OPTIMIZED SCORING: Consider delta, volume, and bid-ask spread
        def score_option(opt):
            # Perfect delta = 0.35-0.40
            delta_score = 100 - abs((opt['delta'] - 0.375) * 200)

            # Higher volume = better
            volume_score = min(opt['volume'] / 1000 * 10, 30)

            # Tighter spread = better
            spread = opt['ask'] - opt['bid']
            spread_pct = spread / opt['ask'] if opt['ask'] > 0 else 1
            spread_score = max(20 - (spread_pct * 100), 0)

            return delta_score + volume_score + spread_score

        all_options.sort(key=score_option, reverse=True)

        best = all_options[0]
        print(f"\n[FOUND] Best option:")
        print(f"  Symbol: {best['symbol']}")
        print(f"  Strike: ${best['strike']:.2f}")
        print(f"  Premium: ${best['ask']:.2f}")
        print(f"  Delta: {best['delta']:.3f}")
        print(f"  DTE: {best['dte']}")
        print(f"  Volume: {best['volume']}")

        return best

    def execute_trade(self, option, analysis):
        """Execute the 1DTE trade"""
        print(f"\n[EXECUTE] Placing order for {option['symbol']}...")

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

            # Log the trade
            trade_record = {
                'timestamp': datetime.now().isoformat(),
                'order_id': order_id,
                'symbol': option['symbol'],
                'type': analysis['direction'],
                'strike': option['strike'],
                'premium': option['ask'],
                'cost': option['ask'] * 100,
                'delta': option['delta'],
                'dte': option['dte'],
                'spy_price': analysis['spy_price'],
                'analysis': analysis,
                'status': 'OPEN',
                'entry_time': datetime.now().isoformat()
            }

            self.log_trade(trade_record)

            # RL shadow: log the decision so we can learn from its outcome
            if self.rl_advisor:
                try:
                    advice = self.rl_advisor.observe_and_log(
                        analysis, order_id, analysis['direction'],
                        day_of_week=datetime.now().weekday()
                    )
                    print(f"[RL] Recommended: {advice['recommended_action']} | "
                          f"Rule: {advice['rule_action']} | "
                          f"Agree: {advice['agreement']}")
                except Exception as e:
                    print(f"[RL] observe failed: {e}")

            # TELEGRAM NOTIFICATION - ENTRY
            telegram_msg = f"""
*SPY 1DTE TRADE OPENED*

*Type:* {analysis['direction']}
*Strike:* ${option['strike']:.2f}
*Premium:* ${option['ask']:.2f}
*Cost:* ${option['ask'] * 100:.2f}
*Delta:* {option['delta']:.3f}
*Volume:* {option['volume']:,}

*Market Analysis:*
SPY: ${analysis['spy_price']:.2f} ({analysis['spy_change']:+.2f}%)
VIX: {analysis['vix_level']:.2f}
Confidence: {analysis['confidence']:.0f}%

*Target:* 20% profit (${option['ask'] * 1.20:.2f})
*Stop:* -30% loss (${option['ask'] * 0.70:.2f})

Order ID: `{order_id}`
"""
            self.telegram.send(telegram_msg)

            return trade_record
        else:
            print(f"[FAILED] Order failed: {response.status_code}")
            print(response.text)
            return None

    def log_trade(self, trade):
        """Log trade to file"""
        if os.path.exists(self.log_file):
            with open(self.log_file, 'r') as f:
                trades = json.load(f)
        else:
            trades = []

        trades.append(trade)

        with open(self.log_file, 'w') as f:
            json.dump(trades, f, indent=2)

        print(f"[LOG] Trade logged to {self.log_file}")

    def monitor_and_close(self, trade, target_profit=0.20):
        """
        ENHANCED: Monitor position with 15-minute checks and early stop loss
        - Check every 15 minutes (not every 30 seconds)
        - Early stop loss: Exit if down 20% by 11:00 AM
        - Trailing stop after +15%
        - Telegram updates at milestones
        """
        print(f"\n[MONITOR] Monitoring position (15-min intervals)...")
        print(f"[TARGET] Entry: ${trade['premium']:.2f}")
        print(f"[TARGET] Profit: ${trade['premium'] * 1.20:.2f} (+20%)")
        print(f"[STOP] Loss: ${trade['premium'] * 0.70:.2f} (-30%)")
        print(f"[EARLY STOP] -20% before 11:00 AM")

        entry_premium = trade['premium']
        symbol = trade['symbol']
        max_profit_pct = 0  # For trailing stop
        last_telegram_update = 0  # Track last notification
        entry_time = datetime.now()

        while True:
            time.sleep(self.monitor_interval)  # Check every 15 minutes (900 seconds)

            # Get updated option chain
            response = self.client.get_option_chain(
                'SPY',
                contract_type=trade['type'],
                strike=trade['strike'],
                include_underlying_quote=True
            )

            if response.status_code == 200:
                chain_data = response.json()
                option_map = chain_data.get('putExpDateMap' if trade['type'] == 'PUT' else 'callExpDateMap', {})

                current_bid = None
                for exp_date, strikes in option_map.items():
                    strike_str = str(trade['strike'])
                    if strike_str in strikes:
                        contracts = strikes[strike_str]
                        for contract in contracts:
                            if contract.get('symbol') == symbol:
                                current_bid = contract.get('bid', 0)
                                break

                if current_bid:
                    profit_pct = ((current_bid - entry_premium) / entry_premium) * 100
                    profit_dollars = (current_bid - entry_premium) * 100

                    now = datetime.now()
                    minutes_held = (now - entry_time).total_seconds() / 60
                    print(f"[CHECK] {now.strftime('%H:%M')} | Bid: ${current_bid:.2f} | P/L: {profit_pct:+.1f}% | Held: {minutes_held:.0f}min")

                    # Update max profit for trailing stop
                    if profit_pct > max_profit_pct:
                        max_profit_pct = profit_pct

                    # TELEGRAM UPDATES at milestones
                    if profit_pct >= 10 and last_telegram_update < 10:
                        self.telegram.send(f"SPY 1DTE Update: *+{profit_pct:.1f}%* profit (${profit_dollars:+.2f})")
                        last_telegram_update = 10
                    elif profit_pct >= 15 and last_telegram_update < 15:
                        self.telegram.send(f"SPY 1DTE Update: *+{profit_pct:.1f}%* profit - Near target!")
                        last_telegram_update = 15

                    # TRAILING STOP - Activates at 15%, protects profits
                    # This allows trades to run to 50%, 100%, or higher!
                    if max_profit_pct >= 15:
                        trailing_stop_trigger = max_profit_pct - (self.trailing_stop * 100)
                        if profit_pct <= trailing_stop_trigger:
                            print(f"[TRAILING STOP] Dropped to {profit_pct:.1f}% from peak {max_profit_pct:.1f}%")
                            print(f"[TRAILING STOP] Peak was {max_profit_pct:.1f}%, protected profits!")
                            self.close_position(trade, current_bid, reason='TRAILING_STOP')
                            return

                    # ALERT: If profit exceeds 50%, send notification
                    if profit_pct >= 50 and last_telegram_update < 50:
                        self.telegram.send(f"🚀 SPY 1DTE Alert: *+{profit_pct:.1f}%* profit! Trailing stop active.")
                        last_telegram_update = 50

                    # EARLY STOP LOSS - If down 20% before 11:00 AM, exit immediately
                    if now.hour < 11 and profit_pct <= (self.early_stop_loss * 100):
                        print(f"[EARLY STOP] {profit_pct:.1f}% loss before 11 AM - Cutting losses early!")
                        self.close_position(trade, current_bid, reason='EARLY_STOP_LOSS')
                        return

                    # REGULAR STOP LOSS - Close at -30%
                    if profit_pct <= (self.stop_loss * 100):
                        print(f"[STOP LOSS] {profit_pct:.1f}% loss - Cutting losses!")
                        self.close_position(trade, current_bid, reason='STOP_LOSS')
                        return

            # Check if market is about to close (2:45 PM ET)
            now = datetime.now()
            if now.hour >= 14 and now.minute >= 45:
                print("[MARKET CLOSE] Closing position before market close...")
                self.close_position(trade, None, reason='MARKET_CLOSE')
                return

    def close_position(self, trade, exit_price, reason='MANUAL'):
        """Close the position with Telegram notification"""
        print(f"\n[CLOSE] Closing position {trade['symbol']}...")
        print(f"[REASON] {reason}")

        from schwab.orders.options import option_sell_to_close_limit

        if not exit_price:
            # Market order if no price specified
            exit_price = trade['premium'] * 0.8  # Conservative estimate

        order = option_sell_to_close_limit(
            trade['symbol'],
            1,  # 1 contract
            str(exit_price)
        )

        account_hash = os.getenv('SCHWAB_ACCOUNT_HASH')
        response = self.client.place_order(account_hash, order)

        if response.status_code == 201:
            order_id = response.headers.get('Location', '').split('/')[-1]
            profit = (exit_price - trade['premium']) * 100
            profit_pct = ((exit_price - trade['premium']) / trade['premium']) * 100

            print(f"[SUCCESS] Position closed!")
            print(f"[PROFIT] ${profit:.2f} ({profit_pct:+.1f}%)")

            # Update trade log
            trade['status'] = 'CLOSED'
            trade['exit_price'] = exit_price
            trade['exit_time'] = datetime.now().isoformat()
            trade['profit'] = profit
            trade['profit_pct'] = profit_pct
            trade['close_order_id'] = order_id
            trade['close_reason'] = reason

            self.log_trade(trade)

            # RL shadow: feed realized outcome back to the agent
            if self.rl_advisor:
                try:
                    self.rl_advisor.record_outcome(trade['order_id'], profit_pct)
                    print(f"[RL] Outcome recorded: {profit_pct:+.1f}%")
                except Exception as e:
                    print(f"[RL] record_outcome failed: {e}")

            # TELEGRAM NOTIFICATION - EXIT
            entry_time = datetime.fromisoformat(trade['entry_time'])
            exit_time = datetime.now()
            hold_duration = (exit_time - entry_time).total_seconds() / 60  # minutes

            emoji = "🎯" if profit_pct >= 20 else "✅" if profit_pct > 0 else "🛑"

            telegram_msg = f"""
{emoji} *SPY 1DTE TRADE CLOSED*

*Reason:* {reason.replace('_', ' ')}

*Entry:* ${trade['premium']:.2f}
*Exit:* ${exit_price:.2f}
*Profit:* ${profit:+.2f} ({profit_pct:+.1f}%)

*Trade Details:*
Type: {trade['type']}
Strike: ${trade['strike']:.2f}
Hold Time: {hold_duration:.0f} minutes

*Order IDs:*
Entry: `{trade['order_id']}`
Exit: `{order_id}`
"""
            self.telegram.send(telegram_msg)

        else:
            print(f"[FAILED] Close order failed: {response.status_code}")
            self.telegram.send(f"ERROR: Failed to close SPY 1DTE position - {response.status_code}")

    def run_daily_strategy(self):
        """
        ENHANCED: Run the complete 1DTE strategy with trade filtering
        - Only trade on high-confidence setups (70%+)
        - Skip days with high VIX or large gaps
        - Monitor every 15 minutes with early stop loss
        """
        print("=" * 60)
        print("SPY 1DTE OPTIONS STRATEGY - ENHANCED")
        print("=" * 60)
        print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

        # Step 1: Analyze market direction (with filtering)
        analysis = self.analyze_market_direction()

        # TRADE FILTERING: Skip if low confidence or filtered conditions
        if not analysis.get('should_trade', True):
            print("\n" + "=" * 60)
            print("[SKIPPED] No trade today - Conditions not favorable")
            if analysis.get('skip_reasons'):
                for reason in analysis['skip_reasons']:
                    print(f"  - {reason}")
            print(f"  - Confidence: {analysis.get('confidence', 0):.0f}% (need {self.min_confidence}%+)")
            print("=" * 60)

            # Send Telegram notification
            self.telegram.send(f"""
*SPY 1DTE - NO TRADE TODAY*

Market conditions not favorable:
{chr(10).join('- ' + r for r in analysis.get('skip_reasons', []))}

Confidence: {analysis.get('confidence', 0):.0f}% (need {self.min_confidence}%+)
SPY: ${analysis.get('spy_price', 0):.2f} ({analysis.get('spy_change', 0):+.2f}%)
VIX: {analysis.get('vix_level', 0):.2f}
""")
            return

        # Sentiment risk filter (fail-open): may block aggressive longs in
        # Extreme Fear, or signal a size reduction. It never crashes the bot.
        if self.sentiment_service:
            try:
                from sentiment import adjust_trade_risk_by_sentiment, summarize_for_log
                sentiment = self.sentiment_service.get_sentiment()
                print(f"[SENTIMENT] {summarize_for_log(sentiment)}")
                decision = adjust_trade_risk_by_sentiment(
                    {'size': 1,
                     'confidence': analysis.get('confidence', 0),
                     'direction': analysis.get('direction')},
                    sentiment,
                )
                print(f"[SENTIMENT] {decision['reason']}")
                if not decision['allowed']:
                    self.telegram.send(
                        "*SPY 1DTE - NO TRADE (SENTIMENT)*\n\n"
                        f"{decision['reason']}"
                    )
                    print("\n[SKIPPED] Trade blocked by sentiment filter")
                    return
            except Exception as e:
                print(f"[SENTIMENT] filter error (ignored): {e}")

        # Step 2: Find best 1DTE option
        option = self.find_1dte_option(analysis['direction'])

        if not option:
            print("\n[ABORT] No suitable option found")
            self.telegram.send(f"*SPY 1DTE - ABORTED*\n\nNo suitable {analysis['direction']} option found meeting criteria")
            return

        # Step 3: Execute trade
        trade = self.execute_trade(option, analysis)

        if not trade:
            print("\n[ABORT] Trade execution failed")
            return

        print(f"\n[SUCCESS] 1DTE trade executed!")
        print(f"[NEXT] Monitoring for 20% profit target (15-min checks)...")

        # Step 4: Monitor and close at profit/stop
        self.monitor_and_close(trade, target_profit=0.20)

        print("\n" + "=" * 60)
        print("STRATEGY COMPLETE")
        print("=" * 60)


def main():
    strategy = SPY1DTEStrategy()
    strategy.run_daily_strategy()


if __name__ == '__main__':
    main()
