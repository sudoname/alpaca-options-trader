# Smart Options Trading System

An advanced, AI-powered options trading system with intelligent position management, machine learning optimization, and automated risk controls.

## 🚀 Key Features

### Intelligent Position Management
- **Partial Profit Taking**: Automatically closes 50% of position at +20% profit
- **Stop Loss Protection**: Auto-closes entire position at -10% loss
- **Trailing Stops**: 5% trailing stop from highest price reached
- **Dynamic Exit Strategy**: Market condition-based exit signals

### Machine Learning Optimization
- **Adaptive Scoring**: ML model learns from each trade to improve selection
- **Pattern Recognition**: Identifies successful and failed trade patterns
- **Weight Adjustment**: Automatically optimizes Greek importance based on performance
- **Performance Tracking**: Comprehensive trade history analysis

### Advanced Risk Management
- **Budget Control**: Maximum $500 per trade with automatic position sizing
- **Time-Based Exits**: Auto-close positions near expiration
- **Volatility Protection**: Exit on extreme price movements
- **Momentum Reversal**: Detect and exit on trend reversals

## 📁 System Components

### Core Files
- `smart_trader.py` - Main trading engine with ML capabilities
- `smart_trade_runner.py` - Command-line interface
- `position_monitor.py` - Real-time position monitoring
- `trade_now.py` - Simple trading interface (fallback)

### Data Files (Auto-Generated)
- `trading_history.json` - Complete trade history for learning
- `trade_optimizer.pkl` - ML model weights and patterns
- `active_trades.json` - Currently monitored positions

## 🎯 Usage Guide

### 1. Analyze & Trade
```bash
# Analyze AAPL with ML optimization
python smart_trade_runner.py --ticker AAPL

# Place live trade (market hours only)
python smart_trade_runner.py --ticker AAPL --live
```

### 2. Monitor Positions
```bash
# Check positions once
python smart_trade_runner.py --monitor

# Continuous monitoring (every 60 seconds)
python position_monitor.py

# Custom interval monitoring
python position_monitor.py --interval 30
```

### 3. Performance Analysis
```bash
# Generate detailed performance report
python smart_trade_runner.py --report
```

## 🧠 Machine Learning Features

### Adaptive Scoring System
The system learns from each trade outcome and adjusts its selection criteria:

- **Delta Weight**: Optimizes directional exposure preference
- **Gamma Weight**: Adjusts acceleration sensitivity
- **Theta Weight**: Balances time decay concerns
- **Vega Weight**: Manages volatility sensitivity
- **IV Weight**: Optimizes implied volatility preferences

### Pattern Recognition
- **Success Patterns**: Remembers characteristics of profitable trades
- **Failure Patterns**: Avoids repeating losing trade setups
- **Continuous Learning**: Updates after every position close

## 📊 Risk Management Matrix

| Condition | Action | Trigger |
|-----------|--------|---------|
| Profit Target | Close 50% | +20% gain |
| Stop Loss | Close 100% | -10% loss |
| Trailing Stop | Close 100% | 5% from high |
| Near Expiration | Close 100% | <2 days |
| High Volatility | Close 100% | >50% price move |
| Momentum Reversal | Close 100% | 15% pullback |

## 🔧 Configuration

### Position Limits
- **Max Budget**: $500 per trade
- **Max Positions**: Unlimited (budget-limited)
- **Min Days to Expiry**: 30 days
- **Preferred Delta**: 0.5-0.7

### ML Parameters
- **Learning Rate**: 1% per trade
- **Pattern Threshold**: 70% similarity
- **Weight Normalization**: Automatic
- **Success Threshold**: >10% profit
- **Failure Threshold**: <-5% loss

## 📈 Performance Tracking

### Metrics Monitored
- **Win Rate**: Percentage of profitable trades
- **Average Win/Loss**: Mean profit/loss percentages
- **Total P&L**: Cumulative performance
- **Pattern Success**: ML pattern effectiveness
- **Risk-Adjusted Returns**: Sharpe-like ratios

### Learning Evolution
The system tracks how its decision-making improves over time:
- Initial random weights
- Performance-based adjustments
- Pattern recognition development
- Strategy refinement

## ⚡ Quick Start

1. **Setup Environment**
```bash
# Ensure .env file has your Alpaca credentials
python api_test.py  # Verify connection
```

2. **First Trade**
```bash
# Start with analysis
python smart_trade_runner.py --ticker SPY

# Monitor the selection
python smart_trade_runner.py --monitor
```

3. **Live Trading** (Market Hours)
```bash
# Place intelligent trade
python smart_trade_runner.py --ticker AAPL --live

# Start continuous monitoring
python position_monitor.py
```

## 🎨 Advanced Features

### Dynamic Exit Strategies
- **Time Decay Acceleration**: Exit when theta impact increases
- **IV Crush Protection**: Exit before earnings/events
- **Correlation Analysis**: Exit on sector weakness
- **Volume Analysis**: Exit on unusual volume patterns

### Smart Position Sizing
- **Kelly Criterion**: Optimal position sizing based on win rate
- **Volatility Adjustment**: Smaller positions in high-vol environments
- **Correlation Limits**: Avoid over-concentration in similar positions

## 🔒 Safety Features

### Market Hours Protection
- Prevents orders outside trading hours
- Queues orders for market open
- Handles holiday schedules

### Account Protection
- Buying power verification
- Position limit enforcement
- Emergency stop functionality

### Data Integrity
- Automatic backups
- Error recovery
- Transaction logging

## 📊 Sample Performance Report

```
Trading Performance:
  Total Trades: 45
  Win Rate: 67.2%
  Average Win: 18.3%
  Average Loss: -8.7%
  Total P&L: +127.4%

Current ML Weights:
  delta: 0.325
  gamma: 0.089
  theta: 0.156
  vega: 0.098
  iv: 0.142
  moneyness: 0.190

Learned Patterns:
  Success Patterns: 12
  Failure Patterns: 8
```

## 🚨 Important Notes

- **Paper Trading**: System defaults to paper trading for safety
- **Market Hours**: Live trades only during 9:30 AM - 4:00 PM ET
- **Risk Warning**: Options trading involves substantial risk
- **Learning Period**: System improves after 10-20 trades

## 🔄 System Evolution

The Smart Options Trading System continuously evolves:

1. **Trade Execution** → Records outcome
2. **Pattern Analysis** → Identifies success factors
3. **Weight Adjustment** → Optimizes future selections
4. **Strategy Refinement** → Improves decision making
5. **Performance Enhancement** → Better risk-adjusted returns

This creates a self-improving system that gets smarter with every trade!