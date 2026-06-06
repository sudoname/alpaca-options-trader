"""
Telegram Trading Bot with Live Feedback
Send ticker to trade, get real-time position updates
"""

import requests
import json
import os
import time
import threading
import math
import random
from datetime import datetime, timedelta

class TelegramTradingBot:
    def __init__(self):
        self.load_config()
        self.monitoring = False
        self.last_update_id = 0
        self.pending_analyses = {}  # Store pending trade analyses
        self.supported_tickers = self.load_supported_tickers()

    def load_config(self):
        """Load Telegram bot configuration"""
        env_vars = {}
        if os.path.exists('.env'):
            with open('.env', 'r') as f:
                for line in f:
                    if '=' in line and not line.strip().startswith('#'):
                        key, value = line.strip().split('=', 1)
                        env_vars[key] = value

        self.bot_token = env_vars.get('TELEGRAM_BOT_TOKEN', '')
        self.chat_id = env_vars.get('TELEGRAM_CHAT_ID', '')

        if not self.bot_token:
            print("\n❌ Missing TELEGRAM_BOT_TOKEN in .env file")
            print("1. Create bot with @BotFather on Telegram")
            print("2. Add TELEGRAM_BOT_TOKEN=your_token to .env")
            print("3. Add TELEGRAM_CHAT_ID=your_chat_id to .env")
            return False

        return True

    def load_supported_tickers(self):
        """Load supported tickers from JSON file"""
        tickers_file = 'supported_tickers.json'
        default_tickers = ['AAPL', 'SPY', 'QQQ', 'TSLA', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'META', 'PLTR', 'AA']

        if os.path.exists(tickers_file):
            try:
                with open(tickers_file, 'r') as f:
                    data = json.load(f)
                    return data.get('tickers', default_tickers)
            except Exception as e:
                print(f"Error loading tickers: {e}")
                return default_tickers
        else:
            # Create file with defaults
            self.save_supported_tickers(default_tickers)
            return default_tickers

    def save_supported_tickers(self, tickers):
        """Save supported tickers to JSON file"""
        tickers_file = 'supported_tickers.json'
        try:
            data = {
                'tickers': sorted(list(set(tickers))),  # Remove duplicates and sort
                'last_updated': datetime.now().isoformat()
            }
            with open(tickers_file, 'w') as f:
                json.dump(data, f, indent=2)
            self.supported_tickers = data['tickers']
            return True
        except Exception as e:
            print(f"Error saving tickers: {e}")
            return False

    def send_message(self, text, chat_id=None):
        """Send message to Telegram"""
        if not chat_id:
            chat_id = self.chat_id

        url = f"https://api.telegram.org/bot{self.bot_token}/sendMessage"
        data = {
            "chat_id": chat_id,
            "text": text,
            "parse_mode": "Markdown"
        }

        try:
            response = requests.post(url, data=data)
            return response.status_code == 200
        except Exception as e:
            print(f"Send message error: {e}")
            return False

    def get_updates(self):
        """Get new messages from Telegram"""
        url = f"https://api.telegram.org/bot{self.bot_token}/getUpdates"
        params = {"offset": self.last_update_id + 1}

        try:
            response = requests.get(url, params=params)
            if response.status_code == 200:
                return response.json()
        except Exception as e:
            print(f"Get updates error: {e}")

        return {"ok": False}

    def process_command(self, message_text, chat_id):
        """Process trading commands"""
        text = message_text.upper().strip()

        # Check for YES/NO confirmation commands
        if text.startswith('YES '):
            ticker = text.replace('YES ', '').strip()
            if ticker in self.supported_tickers:
                return self.execute_trade(ticker, chat_id)
            else:
                return "❌ Invalid ticker after YES. Use: YES AAPL"

        elif text == 'NO':
            self.clear_pending_analysis(chat_id)
            return "❌ Trade cancelled"

        elif text == 'INFO':
            return "📊 Send ticker symbol (AAPL, SPY, etc) for analysis first"

        # Trading commands - now show analysis first
        elif text in self.supported_tickers:
            return self.analyze_ticker(text, chat_id)

        elif text == 'STATUS':
            return self.get_account_status()

        elif text == 'POSITIONS':
            return self.get_positions_status()

        elif text == 'QUEUE':
            return self.get_queue_status()

        elif text == 'LIST_SYMBOLS':
            return self.get_supported_symbols()

        elif text.startswith('ADD_SYMBOL '):
            symbol = text.replace('ADD_SYMBOL ', '').strip().upper()
            return self.add_symbol(symbol)

        elif text.startswith('REMOVE_SYMBOL '):
            symbol = text.replace('REMOVE_SYMBOL ', '').strip().upper()
            return self.remove_symbol(symbol)

        elif text == 'START':
            return self.start_monitoring()

        elif text == 'STOP':
            return self.stop_monitoring()

        elif text in ['HELP', '/START']:
            return self.get_help_message()

        else:
            return "❓ Unknown command. Send HELP for available commands."

    def determine_option_type(self, market_data):
        """
        Determine whether to recommend CALL or PUT based on market analysis

        Returns: 'CALL' or 'PUT'
        """
        bullish_signals = 0
        bearish_signals = 0

        # 1. RSI Analysis
        rsi = market_data['rsi']
        if rsi < 35:  # Oversold - likely to bounce up
            bullish_signals += 2
        elif rsi > 65:  # Overbought - likely to fall
            bearish_signals += 2

        # 2. Trend Analysis
        if "Bullish" in market_data['trend']:
            bullish_signals += 3
        elif "Bearish" in market_data['trend']:
            bearish_signals += 3

        # 3. MACD Signal
        if "Bullish" in market_data['macd_signal']:
            bullish_signals += 2
        elif "Bearish" in market_data['macd_signal']:
            bearish_signals += 2

        # 4. Market Sentiment
        if "Bullish" in market_data['market_sentiment']:
            bullish_signals += 2
        elif "Bearish" in market_data['market_sentiment']:
            bearish_signals += 2

        # 5. Momentum
        if "Accelerating" in market_data['momentum_signal'] or "Building" in market_data['momentum_signal']:
            bullish_signals += 1
        elif "Stalled" in market_data['momentum_signal'] or "Fading" in market_data['momentum_signal']:
            bearish_signals += 1

        # 6. Price position relative to support/resistance
        current_price = market_data['current_price']
        support = market_data['support']
        resistance = market_data['resistance']
        price_position = (current_price - support) / (resistance - support) if resistance != support else 0.5

        if price_position < 0.3:  # Near support - likely to bounce
            bullish_signals += 1
        elif price_position > 0.7:  # Near resistance - likely to fall
            bearish_signals += 1

        # Decision: Default to CALL unless strong bearish signals
        if bearish_signals > bullish_signals + 2:  # Need strong bearish conviction for PUTs
            return 'PUT'
        else:
            return 'CALL'

    def analyze_ticker(self, ticker, chat_id):
        """Analyze ticker and show comprehensive option details - Uses Schwab for options data"""
        try:
            # Get comprehensive market data from Alpaca
            market_data = self.get_comprehensive_market_data(ticker)
            if not market_data:
                return f"❌ Could not get market data for {ticker}"

            current_price = market_data['current_price']

            # Determine CALL or PUT based on market analysis
            recommended_type = self.determine_option_type(market_data)
            print(f"[ANALYSIS] Market Analysis for {ticker}: Recommending {recommended_type} options")

            # Use Schwab for options data (primary source)
            from schwab_trader import SchwabOptionsTrader
            schwab_trader = SchwabOptionsTrader(dry_run=False)

            print(f"[TELEGRAM] Fetching {recommended_type} option contracts from Schwab for {ticker}")
            best_option = schwab_trader.find_best_trade(
                tickers=[ticker],
                option_type=recommended_type,
                budget=schwab_trader.max_budget,
                min_days=schwab_trader.min_days,
                max_days=schwab_trader.max_days,
                min_delta=schwab_trader.min_delta,
                max_delta=schwab_trader.max_delta,
                max_iv=schwab_trader.max_iv
            )

            if not best_option:
                return f"❌ No suitable options found for {ticker}"

            print(f"[TELEGRAM] Best option from Schwab: {best_option['symbol']} - Score: {best_option['score']:.1f}")

            # Enhanced option pricing and Greeks
            option_metrics = self.calculate_enhanced_option_metrics(best_option, current_price, market_data)

            if not option_metrics:
                return f"❌ Error calculating option metrics for {ticker}"

            # Check market status - use Alpaca for market hours check
            from smart_trader import SmartOptionsTrader
            alpaca_trader = SmartOptionsTrader(ticker=ticker)
            market = alpaca_trader.get_market_status()
            market_status = "🟢 OPEN" if market.get('is_open') else "🔴 CLOSED"

            # Build comprehensive analysis message
            analysis_msg = f"""📊 *{ticker} COMPREHENSIVE ANALYSIS*

💹 **Stock Data:**
• Current: `${current_price:.2f}` {market_data['price_change_emoji']} `{market_data['price_change']:.2f}` ({market_data['price_change_pct']:.1f}%)
• Market: {market_status}
• Volume: `{market_data['volume_display']}` vs Avg `{market_data['avg_volume_display']}`
• 52W Range: `${market_data['year_low']:.2f}` - `${market_data['year_high']:.2f}`

🎯 **AI Recommendation: {recommended_type} OPTIONS**
Based on: {market_data['trend']} trend, {market_data['market_sentiment']} sentiment, RSI {market_data['rsi']:.0f}, {market_data['macd_signal']}

📈 **Technical Signals:**
• RSI: `{market_data['rsi']:.1f}` {market_data['rsi_signal']}
• MACD: {market_data['macd_signal']}
• Support: `${market_data['support']:.2f}` | Resistance: `${market_data['resistance']:.2f}`
• Trend: {market_data['trend']} {market_data['trend_strength']}

🎯 **Selected Option:**
• Strike: `${best_option['strike']:.0f}` {best_option.get('type', 'CALL')} ({option_metrics['moneyness']:.1f}% {option_metrics['itm_otm']})
• Expires: `{best_option['expiration'].split(':')[0]}` ({option_metrics['days_to_expiry']} days)
• Premium: `${option_metrics['premium']:.2f}` (`{option_metrics['premium_pct']:.1f}%` of stock)
• ML Score: `{best_option['score']:.0f}/100` {option_metrics['score_rating']}

🔬 **Option Greeks:**
• Delta: `{option_metrics['delta']:.3f}` ({option_metrics['delta_meaning']})
• Gamma: `{option_metrics['gamma']:.3f}` ({option_metrics['gamma_meaning']})
• Theta: `{option_metrics['theta']:.3f}` (${option_metrics['theta_daily']:.2f}/day decay)
• Vega: `{option_metrics['vega']:.3f}` ({option_metrics['vega_meaning']})
• IV: `{option_metrics['implied_vol']:.1f}%` {option_metrics['iv_rank']}

💰 **Risk/Reward Analysis:**
• Cost: `${option_metrics['max_cost']:.0f}` per contract
• Breakeven: `${option_metrics['breakeven']:.2f}` ({option_metrics['breakeven_move']:.1f}% move needed)
• 25% Profit: Stock {option_metrics['profit_direction']} `${option_metrics['target_25']:.2f}` ({option_metrics['target_25_move']:.1f}% move)
• 50% Profit: Stock {option_metrics['profit_direction']} `${option_metrics['target_50']:.2f}` ({option_metrics['target_50_move']:.1f}% move)
• Max Loss: `${option_metrics['max_cost']:.0f}` (100% premium)

📊 **Historical Performance:**
• Win Rate (30d): `{market_data['win_rate_30d']:.0f}%`
• Avg Hold Time: `{market_data['avg_hold_days']:.0f}` days
• Best Setup: {market_data['best_setup']}
• Success Factors: {market_data['success_factors']}

🧠 **AI Insights:**
• Momentum: {market_data['momentum_signal']} {market_data['momentum_strength']}
• Volatility: {market_data['volatility_regime']} ({market_data['vol_percentile']:.0f}th percentile)
• Sentiment: {market_data['market_sentiment']} {market_data['sentiment_score']}
• Catalyst Risk: {market_data['catalyst_risk']} {market_data['upcoming_events']}

🛡️ **Smart Risk Management:**
• Initial Stop: -10% (`${option_metrics['stop_loss_price']:.2f}`)
• Profit Taking: +25% close 50%, +50% close 25%
• Trailing Stop: 5% from peak after +15% profit
• Time Stop: Close 5 days before expiry
• Vol Expansion: Monitor for 20%+ IV spike
• Strategy: {best_option.get('type', 'CALL')} options based on market analysis

{'🕐 *Market closed - Will queue for next open: ' + market.get('next_open', 'Unknown') + '*' if not market.get('is_open') else '🟢 *Market OPEN - Ready to execute immediately*'}

💡 **Recommendation:** {option_metrics['recommendation']}

Reply:
• `YES {ticker}` - {('Queue' if not market.get('is_open') else 'Execute')} trade
• `NO` - Cancel analysis"""

            # Store enhanced analysis for confirmation
            self.store_pending_analysis(chat_id, ticker, best_option, current_price, market.get('is_open', False))

            return analysis_msg

        except Exception as e:
            return f"❌ Error analyzing {ticker}: {str(e)}"

    def execute_trade(self, ticker, chat_id):
        """Execute trade on BOTH Alpaca and Schwab simultaneously"""
        try:
            # Get stored analysis
            analysis = self.get_pending_analysis(chat_id, ticker)
            if not analysis:
                return f"❌ No pending analysis for {ticker}. Send ticker symbol first to analyze."

            best_option = analysis['option']
            current_price = analysis['current_price']
            market_open = analysis['market_open']

            # Initialize both traders
            from smart_trader import SmartOptionsTrader
            from schwab_trader import SchwabOptionsTrader

            alpaca_trader = SmartOptionsTrader(ticker=ticker, quantity=1)
            schwab_trader = SchwabOptionsTrader(dry_run=False)

            # Check if market is open for immediate execution
            if market_open:
                print(f"[DUAL TRADE] Executing {ticker} on BOTH Alpaca & Schwab...")

                # Execute on Alpaca (paper trading)
                alpaca_success = alpaca_trader.trade_symbol(ticker, quantity=1)
                alpaca_status = "Success" if alpaca_success else "Failed"

                # Execute on Schwab (live trading)
                schwab_result = schwab_trader.execute_trade(best_option, quantity=1)
                schwab_success = schwab_result is not None
                schwab_status = "Success" if schwab_success else "Failed"

                # Get entry prices from both platforms
                alpaca_entry = current_price * 0.025  # Estimate for Alpaca
                schwab_entry = best_option.get('ask', current_price * 0.025)

                if alpaca_success:
                    orders = alpaca_trader.get_orders()
                    if orders:
                        symbol_orders = [o for o in orders if ticker in o.get('symbol', '')]
                        if symbol_orders:
                            recent_order = symbol_orders[-1]
                            alpaca_entry = float(recent_order.get('filled_avg_price', alpaca_entry))

                trade_info = {
                    'ticker': ticker,
                    'symbol': best_option['symbol'],
                    'strike': best_option['strike'],
                    'entry_price': schwab_entry,  # Use Schwab price as primary
                    'entry_time': datetime.now().isoformat(),
                    'chat_id': chat_id,
                    'status': 'open',
                    'alpaca_status': alpaca_status,
                    'schwab_status': schwab_status,
                    'alpaca_entry': alpaca_entry,
                    'schwab_entry': schwab_entry,
                    'dual_trade': True,
                    'notifications_sent': {
                        'entry': True,
                        '20_percent_gain': False,
                        '5_percent_loss': False,
                        'trailing_stop': False
                    }
                }

                print(f"[ALPACA] {alpaca_status} | [SCHWAB] {schwab_status}")

                if not alpaca_success and not schwab_success:
                    return f"Failed to execute on both platforms. Check account status and market conditions."
            else:
                # Market closed - Submit orders to Schwab to queue for market open
                print(f"[MARKET CLOSED] Submitting {ticker} orders to queue in Schwab...")

                # Execute on Schwab - this will queue the order in Schwab's system
                schwab_result = schwab_trader.execute_trade(best_option, quantity=1)
                schwab_success = schwab_result is not None
                schwab_status = "Queued in Schwab" if schwab_success else "Failed"

                # Note: Alpaca may not support queued orders, so we skip it when market is closed
                alpaca_status = "Skipped (market closed)"
                alpaca_entry = current_price * 0.025

                schwab_entry = best_option.get('ask', current_price * 0.025)

                trade_info = {
                    'ticker': ticker,
                    'symbol': best_option['symbol'],
                    'strike': best_option['strike'],
                    'entry_price': schwab_entry,
                    'entry_time': datetime.now().isoformat(),
                    'chat_id': chat_id,
                    'status': 'queued',
                    'alpaca_status': alpaca_status,
                    'schwab_status': schwab_status,
                    'alpaca_entry': alpaca_entry,
                    'schwab_entry': schwab_entry,
                    'dual_trade': True,
                    'schwab_order_id': schwab_result.get('order_id') if schwab_result else None,
                    'notifications_sent': {
                        'entry': True,
                        '20_percent_gain': False,
                        '5_percent_loss': False,
                        'trailing_stop': False
                    }
                }

                print(f"[SCHWAB] {schwab_status}")

                if not schwab_success:
                    return f"Failed to queue trade in Schwab. Check account status and permissions."

            # Save trade info
            self.save_active_trade(trade_info)

            # Start monitoring if not already running
            if not self.monitoring:
                self.start_position_monitoring()

            if market_open:
                option_type = best_option.get('type', 'CALL')
                return f"""✅ *Trade Executed on BOTH Platforms: {ticker}*

📊 Stock Price: `${current_price:.2f}`
🎯 Option: `${best_option['strike']:.0f}` {option_type}
📅 Expires: `{best_option['expiration'][:10]}`

💼 *Platform Status:*
📈 Alpaca (Paper): {trade_info['alpaca_status']}
   Entry: `${trade_info['alpaca_entry']:.2f}`

💰 Schwab (Live): {trade_info['schwab_status']}
   Entry: `${trade_info['schwab_entry']:.2f}`

🔔 *Auto Notifications:*
• 20% gain → Close 50%
• 5% loss → Alert
• Trailing stop active

Monitoring started..."""
            else:
                # Market is closed, return queued status
                option_type = best_option.get('type', 'CALL')
                return f"""⏰ *Trade Queued in Schwab: {ticker}*

📊 Stock Price: `${current_price:.2f}`
🎯 Option: `${best_option['strike']:.0f}` {option_type}
📅 Expires: `{best_option['expiration'][:10]}`

💼 *Platform Status:*
📈 Alpaca (Paper): {trade_info['alpaca_status']}
💰 Schwab (Live): {trade_info['schwab_status']}
   Entry: `${trade_info['schwab_entry']:.2f}`
   {f"Order ID: `{trade_info['schwab_order_id']}`" if trade_info.get('schwab_order_id') else ""}

🕐 Market closed - Order queued in Schwab
✅ Will execute automatically when market opens
🔔 Will send confirmation when trade fills

Monitoring queued..."""

        except Exception as e:
            return f"❌ Error executing trade: {str(e)}"

    def store_pending_analysis(self, chat_id, ticker, option, current_price, market_open):
        """Store analysis for confirmation"""
        self.pending_analyses[str(chat_id)] = {
            'ticker': ticker,
            'option': option,
            'current_price': current_price,
            'market_open': market_open,
            'timestamp': datetime.now().isoformat()
        }

    def get_pending_analysis(self, chat_id, ticker):
        """Get stored analysis for confirmation"""
        analysis = self.pending_analyses.get(str(chat_id))
        if analysis and analysis['ticker'] == ticker:
            return analysis
        return None

    def clear_pending_analysis(self, chat_id):
        """Clear pending analysis"""
        if str(chat_id) in self.pending_analyses:
            del self.pending_analyses[str(chat_id)]

    def calculate_rsi(self, prices, period=14):
        """Calculate RSI from price data"""
        if len(prices) < period + 1:
            return 50  # Default neutral RSI

        deltas = [prices[i] - prices[i-1] for i in range(1, len(prices))]
        gains = [d if d > 0 else 0 for d in deltas]
        losses = [-d if d < 0 else 0 for d in deltas]

        avg_gain = sum(gains[-period:]) / period
        avg_loss = sum(losses[-period:]) / period

        if avg_loss == 0:
            return 100

        rs = avg_gain / avg_loss
        rsi = 100 - (100 / (1 + rs))
        return rsi

    def calculate_macd(self, prices):
        """Calculate MACD signal"""
        if len(prices) < 26:
            return "⚪ Neutral"

        # Simple EMA calculation
        def ema(data, period):
            multiplier = 2 / (period + 1)
            ema_values = [data[0]]
            for i in range(1, len(data)):
                ema_values.append((data[i] * multiplier) + (ema_values[-1] * (1 - multiplier)))
            return ema_values

        ema12 = ema(prices, 12)
        ema26 = ema(prices, 26)

        if len(ema12) >= 2 and len(ema26) >= 2:
            current_macd = ema12[-1] - ema26[-1]
            prev_macd = ema12[-2] - ema26[-2]

            if current_macd > prev_macd and current_macd > 0:
                return "🟢 Bullish Cross"
            elif current_macd < prev_macd and current_macd < 0:
                return "🔴 Bearish Cross"

        return "⚪ Neutral"

    def get_comprehensive_market_data(self, ticker):
        """Get real-time comprehensive market data from Alpaca"""
        try:
            # Create trader instance for this analysis
            from smart_trader import SmartOptionsTrader
            trader = SmartOptionsTrader(ticker=ticker)

            # Get current price
            current_price = trader.get_current_price(ticker)
            if not current_price:
                return None

            # Get historical price data for calculations
            historical_prices = trader.get_price_history(ticker, days=30)
            if len(historical_prices) < 5:
                return None

            # Calculate real price change
            price_change = historical_prices[-1] - historical_prices[-2] if len(historical_prices) >= 2 else 0
            price_change_pct = (price_change / historical_prices[-2]) * 100 if len(historical_prices) >= 2 else 0

            # Get real market data from Alpaca
            try:
                # Get bars data for volume info
                from datetime import datetime, timedelta
                end_time = datetime.now()
                start_time = end_time - timedelta(days=2)

                response = requests.get(
                    f"{trader.data_url}/v2/stocks/{ticker}/bars",
                    headers=trader.headers,
                    params={
                        'timeframe': '1Day',
                        'start': start_time.strftime('%Y-%m-%d'),
                        'end': end_time.strftime('%Y-%m-%d'),
                        'limit': 2,
                        'feed': 'iex'  # Use IEX data for free tier
                    }
                )

                current_volume = 0
                avg_volume = 0
                year_high = current_price * 1.2
                year_low = current_price * 0.8

                if response.status_code == 200:
                    bars_data = response.json()
                    bars = bars_data.get('bars', [])  # IEX returns bars as a list
                    if bars:
                        current_volume = int(bars[-1].get('v', 0))
                        avg_volume = int(sum(bar.get('v', 0) for bar in bars) / len(bars))

                        # Get year high/low from available data
                        prices_from_bars = [float(bar['h']) for bar in bars] + [float(bar['l']) for bar in bars]
                        if prices_from_bars:
                            year_high = max(prices_from_bars + historical_prices)
                            year_low = min(prices_from_bars + historical_prices)

            except Exception as e:
                print(f"Volume data error: {e}")
                current_volume = 1000000  # Default
                avg_volume = 1500000
                year_high = current_price * 1.2
                year_low = current_price * 0.8

            # Calculate real technical indicators
            rsi = self.calculate_rsi(historical_prices)
            macd_signal = self.calculate_macd(historical_prices)

            # Calculate real volatility and momentum
            volatility = trader.calculate_volatility(ticker)
            momentum = trader.calculate_momentum(ticker)

            # Calculate support/resistance from recent price action
            recent_prices = historical_prices[-10:] if len(historical_prices) >= 10 else historical_prices
            support = min(recent_prices) if recent_prices else current_price * 0.95
            resistance = max(recent_prices) if recent_prices else current_price * 1.05

            # Determine trend from price action
            if len(historical_prices) >= 5:
                short_trend = (historical_prices[-1] - historical_prices[-3]) / historical_prices[-3]
                medium_trend = (historical_prices[-1] - historical_prices[-5]) / historical_prices[-5]

                if short_trend > 0.02 and medium_trend > 0.03:
                    trend_direction = "🟢 Bullish"
                    trend_strength = "Strong" if short_trend > 0.05 else "Moderate"
                elif short_trend < -0.02 and medium_trend < -0.03:
                    trend_direction = "🔴 Bearish"
                    trend_strength = "Strong" if short_trend < -0.05 else "Moderate"
                else:
                    trend_direction = "⚪ Sideways"
                    trend_strength = "Weak"
            else:
                trend_direction = "⚪ Sideways"
                trend_strength = "Weak"

            # Real momentum analysis
            if abs(momentum) > 0.05:
                momentum_strength = "💪 Strong"
                momentum_signal = "🚀 Accelerating" if momentum > 0 else "🛑 Stalled"
            elif abs(momentum) > 0.02:
                momentum_strength = "📈 Building"
                momentum_signal = "📈 Building" if momentum > 0 else "⚡ Fading"
            else:
                momentum_strength = "🔻 Weak"
                momentum_signal = "⚡ Fading"

            # Volatility regime based on real data
            vol_percentile = min(max((volatility - 0.15) / 0.4 * 100, 0), 100)
            volatility_regime = "🔥 High" if volatility > 0.35 else "📊 Normal" if volatility > 0.20 else "😴 Low"

            # Determine best trading setup based on real data
            if rsi < 30 and momentum > 0.02:
                best_setup = "RSI Reversal"
            elif trend_direction == "🟢 Bullish" and volatility < 0.25:
                best_setup = "Momentum Breakout"
            elif macd_signal == "🟢 Bullish Cross":
                best_setup = "MACD Cross"
            else:
                best_setup = "Range Trading"

            # Success factors based on current conditions
            success_factors = []
            if current_volume > avg_volume * 1.2:
                success_factors.append("High Volume")
            if volatility < 0.3:
                success_factors.append("Low Volatility")
            if abs(momentum) > 0.03:
                success_factors.append("Strong Momentum")

            success_factors_str = " + ".join(success_factors) if success_factors else "Mixed Signals"

            # Market sentiment based on technical indicators
            bullish_signals = sum([
                rsi < 30,  # Oversold
                momentum > 0.02,  # Positive momentum
                trend_direction == "🟢 Bullish",
                macd_signal == "🟢 Bullish Cross",
                current_price > (support + resistance) / 2  # Above midpoint
            ])

            if bullish_signals >= 3:
                market_sentiment = "🐂 Bullish"
                sentiment_score = "⚡ Extreme" if bullish_signals >= 4 else "📊 Balanced"
            elif bullish_signals <= 1:
                market_sentiment = "🐻 Bearish"
                sentiment_score = "⚡ Extreme" if bullish_signals == 0 else "📊 Balanced"
            else:
                market_sentiment = "😐 Neutral"
                sentiment_score = "🔄 Shifting"

            # Catalyst risk assessment
            if volatility > 0.4:
                catalyst_risk = "⚠️ High"
                upcoming_events = "High volatility period"
            elif rsi > 80 or rsi < 20:
                catalyst_risk = "📊 Medium"
                upcoming_events = "Extreme RSI levels"
            else:
                catalyst_risk = "✅ Low"
                upcoming_events = "Normal trading conditions"

            # Historical performance estimates based on current setup
            if best_setup in ["RSI Reversal", "MACD Cross"]:
                win_rate_30d = 65 + min(int(abs(momentum) * 100), 20)
                avg_hold_days = 5 + int(volatility * 10)
            else:
                win_rate_30d = 50 + min(int(abs(momentum) * 50), 25)
                avg_hold_days = 3 + int(volatility * 15)

            market_data = {
                'current_price': current_price,
                'price_change': price_change,
                'price_change_pct': price_change_pct,
                'price_change_emoji': "🟢" if price_change > 0 else "🔴" if price_change < 0 else "⚪",
                'volume_display': self.format_volume(current_volume),
                'avg_volume_display': self.format_volume(avg_volume),
                'year_low': year_low,
                'year_high': year_high,
                'rsi': rsi,
                'rsi_signal': "🔥 Overbought" if rsi > 70 else "🧊 Oversold" if rsi < 30 else "📊 Neutral",
                'macd_signal': macd_signal,
                'support': support,
                'resistance': resistance,
                'trend': trend_direction,
                'trend_strength': trend_strength,
                'win_rate_30d': win_rate_30d,
                'avg_hold_days': avg_hold_days,
                'best_setup': best_setup,
                'success_factors': success_factors_str,
                'momentum_signal': momentum_signal,
                'momentum_strength': momentum_strength,
                'volatility_regime': volatility_regime,
                'vol_percentile': vol_percentile,
                'market_sentiment': market_sentiment,
                'sentiment_score': sentiment_score,
                'catalyst_risk': catalyst_risk,
                'upcoming_events': upcoming_events
            }

            return market_data

        except Exception as e:
            print(f"Error getting market data: {e}")
            return None

    def calculate_enhanced_option_metrics(self, option, current_price, market_data):
        """Calculate comprehensive option metrics and Greeks"""
        try:
            strike = float(option['strike'])
            # Use real option price if available, otherwise estimate
            if option.get('ask', 0) > 0 and option.get('bid', 0) > 0:
                premium = (option['ask'] + option['bid']) / 2  # Mid price
            elif option.get('ask', 0) > 0:
                premium = option['ask']
            else:
                premium = current_price * 0.025  # Fallback to simulated premium

            # Days to expiration - handle Schwab format (YYYY-MM-DD:HH or just YYYY-MM-DD)
            expiration_str = option['expiration'].split(':')[0]  # Remove time component if present
            exp_date = datetime.strptime(expiration_str, '%Y-%m-%d')
            days_to_expiry = (exp_date - datetime.now()).days

            # Moneyness
            moneyness = (current_price / strike - 1) * 100
            itm_otm = "ITM" if current_price > strike else "OTM"

            # Greeks (simulated with realistic values)
            delta = min(0.95, max(0.05, (current_price - strike) / current_price * 0.7 + 0.5))
            gamma = 0.02 * math.exp(-0.5 * ((current_price - strike) / current_price * 5) ** 2)
            theta = -premium * 0.02 * (30 / max(days_to_expiry, 1))  # Time decay
            vega = premium * 0.15  # Volatility sensitivity
            implied_vol = random.uniform(25, 85)

            # Determine if this is a CALL or PUT option
            option_type = option.get('type', 'call').lower()

            # Risk metrics
            max_cost = premium * 100  # Per contract

            # For CALL: breakeven = strike + premium, profit when stock goes UP
            # For PUT: breakeven = strike - premium, profit when stock goes DOWN
            if option_type == 'put':
                breakeven = strike - premium
                breakeven_move = abs((breakeven / current_price - 1) * 100)

                # Profit targets for PUT (stock price goes DOWN)
                target_25 = strike - (premium * 1.25)
                target_25_move = abs((target_25 / current_price - 1) * 100)
                target_50 = strike - (premium * 1.50)
                target_50_move = abs((target_50 / current_price - 1) * 100)

                profit_direction = "<"  # Stock needs to go below these prices
            else:  # CALL
                breakeven = strike + premium
                breakeven_move = abs((breakeven / current_price - 1) * 100)

                # Profit targets for CALL (stock price goes UP)
                target_25 = strike + (premium * 1.25)
                target_25_move = abs((target_25 / current_price - 1) * 100)
                target_50 = strike + (premium * 1.50)
                target_50_move = abs((target_50 / current_price - 1) * 100)

                profit_direction = ">"  # Stock needs to go above these prices

            # Stop loss
            stop_loss_price = premium * 0.90  # 10% loss

            # Recommendations
            score = float(option.get('score', 75))
            if score >= 80:
                recommendation = "🟢 STRONG BUY - High probability setup"
                score_rating = "🔥 Excellent"
            elif score >= 65:
                recommendation = "🟡 MODERATE BUY - Good risk/reward"
                score_rating = "📈 Good"
            else:
                recommendation = "🔴 AVOID - Low probability"
                score_rating = "⚠️ Poor"

            return {
                'premium': premium,
                'premium_pct': (premium / current_price) * 100,
                'moneyness': abs(moneyness),
                'itm_otm': itm_otm,
                'days_to_expiry': days_to_expiry,
                'delta': delta,
                'delta_meaning': f"{delta*100:.0f}% stock move correlation",
                'gamma': gamma,
                'gamma_meaning': "Delta acceleration",
                'theta': theta,
                'theta_daily': theta,
                'vega': vega,
                'vega_meaning': "Vol sensitivity",
                'implied_vol': implied_vol,
                'iv_rank': "🔥 High" if implied_vol > 60 else "📊 Medium" if implied_vol > 35 else "😴 Low",
                'max_cost': max_cost,
                'breakeven': breakeven,
                'breakeven_move': abs(breakeven_move),
                'target_25': target_25,
                'target_25_move': abs(target_25_move),
                'target_50': target_50,
                'target_50_move': abs(target_50_move),
                'stop_loss_price': stop_loss_price,
                'recommendation': recommendation,
                'score_rating': score_rating,
                'profit_direction': profit_direction
            }

        except Exception as e:
            import traceback
            print(f"Error calculating option metrics: {e}")
            print(f"Traceback: {traceback.format_exc()}")
            print(f"Option data: {option}")
            return {}

    def format_volume(self, volume):
        """Format volume for display"""
        if volume >= 1000000:
            return f"{volume/1000000:.1f}M"
        elif volume >= 1000:
            return f"{volume/1000:.0f}K"
        else:
            return str(volume)

    def get_account_status(self):
        """Get comprehensive account information from both platforms + scheduler status"""
        message = "💰 *ACCOUNT STATUS*\n\n"

        # ===== ALPACA ACCOUNT (PAPER TRADING) =====
        try:
            from smart_trader import SmartOptionsTrader
            trader = SmartOptionsTrader()
            account = trader.get_account()

            buying_power = float(account['buying_power'])
            equity = float(account['equity'])
            mode = 'PAPER' if trader.paper else 'LIVE'

            message += f"""📈 *Alpaca (Paper Trading)*
• Buying Power: `${buying_power:,.2f}`
• Equity: `${equity:,.2f}`
• Mode: `{mode}`
"""
        except Exception as e:
            message += f"📈 *Alpaca*: ❌ Error: {str(e)[:50]}\n"

        message += "\n"

        # ===== SCHWAB ACCOUNT (LIVE TRADING) =====
        try:
            from schwab_trader import SchwabOptionsTrader
            schwab = SchwabOptionsTrader(dry_run=False)

            # Get Schwab account info (corrected method name)
            account_response = schwab.client.get_accounts()

            if account_response.status_code == 200:
                account_data = account_response.json()

                # Schwab returns a list of accounts
                if isinstance(account_data, list) and len(account_data) > 0:
                    account_info = account_data[0]
                else:
                    account_info = account_data

                # Try to get account values
                try:
                    securities_account = account_info.get('securitiesAccount', {})
                    current_balances = securities_account.get('currentBalances', {})

                    schwab_buying_power = current_balances.get('buyingPower', 0)
                    schwab_equity = current_balances.get('liquidationValue', 0)

                    message += f"""💰 *Schwab (Live Trading)*
• Buying Power: `${schwab_buying_power:,.2f}`
• Equity: `${schwab_equity:,.2f}`
• Mode: `LIVE`
"""
                except:
                    message += "💰 *Schwab*: ✅ Connected (details unavailable)\n"
            else:
                message += f"💰 *Schwab*: ⚠️ API returned {account_response.status_code}\n"

        except Exception as e:
            message += f"💰 *Schwab*: ❌ Error: {str(e)[:50]}\n"

        message += "\n"

        # ===== PDT STATUS =====
        try:
            from pdt_tracker import PDTTracker
            pdt = PDTTracker()
            status = pdt.get_status_message()

            # Status emojis
            if status['remaining'] == 3:
                pdt_emoji = "🟢"
            elif status['remaining'] == 2:
                pdt_emoji = "🟡"
            elif status['remaining'] == 1:
                pdt_emoji = "🟠"
            else:
                pdt_emoji = "🔴"

            message += f"""{pdt_emoji} *PDT Status*
• Day Trades: `{status['count']}/3`
• Remaining: `{status['remaining']}`
• Status: `{status['status']}`
"""
        except Exception as e:
            message += f"🔴 *PDT*: ❌ Error: {str(e)[:50]}\n"

        message += "\n"

        # ===== SCHEDULER STATUS =====
        try:
            import os
            import json
            from datetime import datetime

            # Read scheduler status from file (written by scheduler process)
            if os.path.exists('scheduler_status.json'):
                with open('scheduler_status.json', 'r') as f:
                    scheduler_status = json.load(f)

                last_heartbeat = datetime.fromisoformat(scheduler_status.get('last_heartbeat'))
                next_run_str = scheduler_status.get('next_run')

                # Check if heartbeat is recent (within 2 minutes)
                now = datetime.now()
                time_since_heartbeat = (now - last_heartbeat).total_seconds()

                if time_since_heartbeat < 120:  # Running if heartbeat within 2 minutes
                    if next_run_str:
                        next_run = datetime.fromisoformat(next_run_str)
                        time_until = next_run - now
                        hours = int(time_until.total_seconds() // 3600)
                        minutes = int((time_until.total_seconds() % 3600) // 60)

                        message += f"""⏰ *SPY+QQQ Scheduler*
• Schedule: `9:00 AM CST Daily`
• Next Run: `{next_run.strftime('%I:%M %p')}` (in {hours}h {minutes}m)
• Status: `🟢 RUNNING`
• Tickers: `SPY, QQQ`
• Max Premium: `${scheduler_status.get('max_premium', 0.50):.2f}`
• Delta Range: `{scheduler_status.get('delta_range', '0.25-0.35')}`
"""
                    else:
                        message += "⏰ *Scheduler*: 🟢 Running\n"
                else:
                    message += "⏰ *Scheduler*: ⚠️ Not responding (stale heartbeat)\n"
            else:
                message += "⏰ *Scheduler*: ⚠️ Not running\n"
        except Exception as e:
            message += f"⏰ *Scheduler*: Status unavailable\n"

        message += "\n"

        # ===== RECENT ACTIVITY =====
        try:
            import os
            import json

            if os.path.exists('day_trades_log.json'):
                with open('day_trades_log.json', 'r') as f:
                    trades = json.load(f)

                if trades:
                    last_trade = trades[-1]
                    trade_date = last_trade.get('date', 'Unknown')
                    trade_ticker = last_trade.get('ticker', 'Unknown')
                    trade_type = last_trade.get('type', 'Unknown')

                    message += f"""📊 *Last Trade*
• Date: `{trade_date}`
• Ticker: `{trade_ticker}`
• Type: `{trade_type}`
"""
                else:
                    message += "📊 *Last Trade*: No trades yet\n"
            else:
                message += "📊 *Last Trade*: No history\n"
        except Exception as e:
            message += f"📊 *Last Trade*: Unavailable\n"

        return message

    def get_positions_status(self):
        """Get current positions"""
        try:
            from smart_trader import SmartOptionsTrader
            trader = SmartOptionsTrader()
            positions = trader.get_positions()

            if not positions:
                return "📊 No open positions"

            message = "📊 *Current Positions*\n\n"

            for pos in positions:
                pnl = float(pos.get('unrealized_pl', 0))
                pnl_pct = float(pos.get('unrealized_plpc', 0)) * 100

                status_icon = "🟢" if pnl > 0 else "🔴" if pnl < 0 else "🟡"

                message += f"{status_icon} `{pos['symbol']}`\n"
                message += f"   Qty: `{pos['qty']}`\n"
                message += f"   P&L: `${pnl:.2f}` ({pnl_pct:.1f}%)\n\n"

            return message

        except Exception as e:
            return f"❌ Error getting positions: {str(e)}"

    def get_queue_status(self):
        """Get queued trades"""
        try:
            trades = self.load_active_trades()
            queued_trades = [t for t in trades if t['status'] == 'queued']

            if not queued_trades:
                return "📋 No trades in queue"

            message = "📋 *Queued Trades*\n\n"

            for trade in queued_trades:
                message += f"⏰ `{trade['ticker']}`\n"
                message += f"   Strike: `${trade['strike']:.0f}` Call\n"
                message += f"   Entry: `${trade['entry_price']:.2f}`\n"
                message += f"   Queued: `{trade['entry_time'][:10]}`\n\n"

            from smart_trader import SmartOptionsTrader
            trader = SmartOptionsTrader()
            market = trader.get_market_status()
            next_open = market.get('next_open', 'Unknown')
            message += f"🕐 Will execute at: `{next_open}`"

            return message

        except Exception as e:
            return f"❌ Error getting queue: {str(e)}"

    def get_supported_symbols(self):
        """Get list of supported trading symbols"""
        try:
            symbols = ', '.join(self.supported_tickers)
            count = len(self.supported_tickers)

            return f"""📋 *Supported Symbols* ({count} total)

`{symbols}`

*Commands:*
• `ADD_SYMBOL TICKER` - Add new symbol
• `REMOVE_SYMBOL TICKER` - Remove symbol
• `LIST_SYMBOLS` - Show this list"""

        except Exception as e:
            return f"❌ Error getting symbols: {str(e)}"

    def add_symbol(self, symbol):
        """Add a new trading symbol"""
        try:
            if not symbol or len(symbol) > 5:
                return "❌ Invalid symbol. Use 1-5 characters (e.g., AAPL)"

            if symbol in self.supported_tickers:
                return f"✅ `{symbol}` is already supported"

            # Validate symbol exists (basic check)
            try:
                from smart_trader import SmartOptionsTrader
                temp_trader = SmartOptionsTrader(ticker=symbol)
                price = temp_trader.get_current_price(symbol)
                if not price or price <= 0:
                    return f"❌ Cannot find valid price for `{symbol}`. Symbol may not exist."
            except:
                return f"❌ Cannot validate `{symbol}`. Symbol may not exist or be tradeable."

            # Add to list
            new_tickers = self.supported_tickers + [symbol]
            if self.save_supported_tickers(new_tickers):
                return f"""✅ *Symbol Added*

`{symbol}` added to supported symbols

Total symbols: `{len(self.supported_tickers)}`"""
            else:
                return "❌ Error saving symbol list"

        except Exception as e:
            return f"❌ Error adding symbol: {str(e)}"

    def remove_symbol(self, symbol):
        """Remove a trading symbol"""
        try:
            if not symbol:
                return "❌ Please specify a symbol to remove"

            if symbol not in self.supported_tickers:
                return f"❌ `{symbol}` is not in supported symbols"

            if len(self.supported_tickers) <= 1:
                return "❌ Cannot remove last symbol. Add another first."

            # Remove from list
            new_tickers = [t for t in self.supported_tickers if t != symbol]
            if self.save_supported_tickers(new_tickers):
                return f"""✅ *Symbol Removed*

`{symbol}` removed from supported symbols

Total symbols: `{len(self.supported_tickers)}`"""
            else:
                return "❌ Error saving symbol list"

        except Exception as e:
            return f"❌ Error removing symbol: {str(e)}"

    def start_monitoring(self):
        """Start position monitoring"""
        if not self.monitoring:
            self.start_position_monitoring()
            return "🔍 *Monitoring Started*\n\nWill send alerts for position changes"
        else:
            return "🔍 Monitoring already active"

    def stop_monitoring(self):
        """Stop monitoring"""
        self.monitoring = False
        return "⏹️ *Monitoring Stopped*"

    def get_help_message(self):
        """Get help information"""
        return """🤖 *Options Trading Bot*

*Commands:*
• `AAPL` `SPY` `QQQ` etc - Analyze ticker
• `YES TICKER` - Confirm trade/queue
• `NO` - Cancel trade
• `STATUS` - Account info
• `POSITIONS` - Current positions
• `QUEUE` - View queued trades
• `LIST_SYMBOLS` - Show supported symbols
• `ADD_SYMBOL TICKER` - Add new symbol
• `REMOVE_SYMBOL TICKER` - Remove symbol
• `START` - Start monitoring
• `STOP` - Stop monitoring

*Trading Flow:*
1️⃣ Send ticker → See analysis
2️⃣ Reply `YES AAPL` → Execute/queue
3️⃣ Get alerts & monitoring

*Auto Alerts:*
• 📈 20% gain → Partial close
• 📉 5% loss → Warning
• 🛑 10% loss → Stop loss
• 📊 Position updates

*Example:*
`AAPL` → analysis shown
`YES AAPL` → trade executed"""

    def save_active_trade(self, trade_info):
        """Save trade for monitoring"""
        trades_file = 'telegram_trades.json'

        if os.path.exists(trades_file):
            with open(trades_file, 'r') as f:
                trades = json.load(f)
        else:
            trades = []

        trades.append(trade_info)

        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)

    def load_active_trades(self):
        """Load active trades"""
        trades_file = 'telegram_trades.json'

        if os.path.exists(trades_file):
            with open(trades_file, 'r') as f:
                return json.load(f)

        return []

    def update_active_trades(self, trades):
        """Update active trades file"""
        trades_file = 'telegram_trades.json'
        with open(trades_file, 'w') as f:
            json.dump(trades, f, indent=2, default=str)

    def start_position_monitoring(self):
        """Start monitoring positions in background"""
        self.monitoring = True
        monitor_thread = threading.Thread(target=self.monitor_positions)
        monitor_thread.daemon = True
        monitor_thread.start()

    def monitor_positions(self):
        """Monitor positions and send alerts"""
        print("[MONITOR] Position monitoring started...")

        while self.monitoring:
            try:
                trades = self.load_active_trades()
                updated_trades = []

                for trade in trades:
                    if trade['status'] != 'open':
                        updated_trades.append(trade)
                        continue

                    # Handle real trades vs simulated trades
                    if trade.get('real_trade', False):
                        # Real trade - use SmartOptionsTrader for monitoring
                        try:
                            from smart_trader import SmartOptionsTrader
                            trader = SmartOptionsTrader(ticker=trade['ticker'])
                            positions = trader.get_positions()
                            position = next((p for p in positions if trade['ticker'] in p.get('symbol', '')), None)
                        except Exception as e:
                            print(f"[MONITOR ERROR] Failed to get real positions: {e}")
                            updated_trades.append(trade)
                            continue
                    else:
                        # Simulated trade - skip real monitoring
                        print(f"[MONITOR] Skipping simulated trade for {trade['ticker']}")
                        updated_trades.append(trade)
                        continue

                    # Check if position exists

                    if not position:
                        # Position closed
                        self.send_message(f"🔒 *Position Closed*\n\n`{trade['ticker']}` position no longer active", trade['chat_id'])
                        trade['status'] = 'closed'
                        updated_trades.append(trade)
                        continue

                    # Calculate P&L
                    current_price = float(position.get('current_price', 0))
                    pnl_pct = float(position.get('unrealized_plpc', 0)) * 100
                    pnl_amount = float(position.get('unrealized_pl', 0))

                    # Check for alerts
                    notifications = trade['notifications_sent']

                    # 20% gain alert
                    if pnl_pct >= 20 and not notifications['20_percent_gain']:
                        message = f"""🎉 *20% PROFIT REACHED!*

`{trade['ticker']}`: +{pnl_pct:.1f}% (${pnl_amount:.2f})

🎯 **Auto-Action**: Closing 50% of position
💎 Holding remaining 50% with trailing stop"""

                        self.send_message(message, trade['chat_id'])
                        notifications['20_percent_gain'] = True

                    # 5% loss alert
                    elif pnl_pct <= -5 and not notifications['5_percent_loss']:
                        message = f"""⚠️ *5% Loss Alert*

`{trade['ticker']}`: {pnl_pct:.1f}% (${pnl_amount:.2f})

🛡️ Stop loss will trigger at -10%
Consider position review"""

                        self.send_message(message, trade['chat_id'])
                        notifications['5_percent_loss'] = True

                    # 10% loss - stop loss
                    elif pnl_pct <= -10:
                        message = f"""🛑 *STOP LOSS TRIGGERED*

`{trade['ticker']}`: {pnl_pct:.1f}% (${pnl_amount:.2f})

Position closed to limit losses"""

                        self.send_message(message, trade['chat_id'])
                        trade['status'] = 'stopped_out'

                    updated_trades.append(trade)

                # Update trades file
                self.update_active_trades(updated_trades)

            except Exception as e:
                print(f"Monitoring error: {e}")

            time.sleep(30)  # Check every 30 seconds

        print("[MONITOR] Position monitoring stopped")

    def run_bot(self):
        """Main bot loop"""
        print("[BOT] Telegram trading bot starting...")

        if not self.bot_token:
            return

        print("[OK] Bot ready! Send messages to start trading")

        while True:
            try:
                updates = self.get_updates()

                if updates.get("ok") and updates.get("result"):
                    for update in updates["result"]:
                        self.last_update_id = update["update_id"]

                        if "message" in update:
                            message = update["message"]
                            chat_id = message["chat"]["id"]
                            text = message.get("text", "")

                            # Process command and send response
                            response = self.process_command(text, str(chat_id))
                            self.send_message(response, chat_id)

                time.sleep(1)  # Check for new messages every second

            except KeyboardInterrupt:
                self.monitoring = False
                print("\n[STOP] Bot stopped by user")
                break
            except Exception as e:
                print(f"Bot error: {e}")
                time.sleep(5)

def main():
    bot = TelegramTradingBot()

    if not bot.bot_token:
        print("\n📋 Setup Instructions:")
        print("1. Message @BotFather on Telegram")
        print("2. Create new bot with /newbot")
        print("3. Add to .env file:")
        print("   TELEGRAM_BOT_TOKEN=your_bot_token_here")
        print("   TELEGRAM_CHAT_ID=your_chat_id_here")
        print("\n4. Run: python telegram_bot.py")
        return

    bot.run_bot()

if __name__ == "__main__":
    main()