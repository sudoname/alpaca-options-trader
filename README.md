# Alpaca Options Trading Bot

An automated options trading application that uses the Alpaca API to find and purchase in-the-money (ITM) options based on Greeks analysis.

## Features

- Fetches real-time option chains from Alpaca
- Filters for ITM options with >30 days to expiration
- Analyzes options based on Greeks (Delta, Gamma, Theta, Vega, IV)
- Automatically selects the best option within budget constraints
- Supports both CALL and PUT options
- Dry-run mode for testing without placing real orders
- Comprehensive logging and error handling

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure API credentials:
   - Copy `.env.example` to `.env`
   - Add your Alpaca API credentials:
```
ALPACA_API_KEY=your_api_key_here
ALPACA_SECRET_KEY=your_secret_key_here
ALPACA_PAPER=true  # Set to false for live trading
```

## Usage

### Basic Usage (Dry Run)
```bash
python main.py --ticker AAPL
```

### With Custom Parameters
```bash
python main.py --ticker AAPL --budget 1000 --min-days 45 --max-days 120 --option-type PUT
```

### Live Trading (Use with caution!)
```bash
python main.py --ticker AAPL --live
```

### Command Line Arguments

- `--ticker`: Stock ticker symbol (required)
- `--expiration`: Target expiration date in YYYY-MM-DD format (optional)
- `--strike`: Target strike price (optional)
- `--budget`: Budget for option purchase (default: $500)
- `--option-type`: CALL or PUT (default: CALL)
- `--min-days`: Minimum days to expiration (default: 30)
- `--max-days`: Maximum days to expiration (default: 90)
- `--live`: Execute live trades (default: dry run mode)
- `--verbose`: Enable verbose logging

## How It Works

1. **Authentication**: Connects to Alpaca using provided API credentials
2. **Price Retrieval**: Gets the current stock price
3. **Option Chain Fetch**: Retrieves available options within the specified expiration window
4. **ITM Filtering**: Filters for in-the-money options
5. **Greeks Analysis**: Scores options based on:
   - Delta (directional exposure)
   - Gamma (rate of delta change)
   - Theta (time decay)
   - Vega (volatility sensitivity)
   - Implied Volatility
   - Moneyness (distance from strike)
   - Days to expiration
6. **Selection**: Chooses the highest-scoring option within budget
7. **Execution**: Places the order (or simulates in dry-run mode)

## Option Selection Algorithm

The bot uses a weighted scoring system:
- **Delta**: 30% weight - Prefers higher delta for better directional exposure
- **Gamma**: 10% weight - Considers acceleration of delta
- **Theta**: 15% weight - Minimizes time decay impact
- **IV**: 15% weight - Prefers options near median IV
- **Moneyness**: 20% weight - Favors closer to ATM options
- **Days to Expiry**: 10% weight - Balances time value

## Safety Features

- **Dry Run Mode**: Default mode that simulates orders without execution
- **Budget Limits**: Enforces maximum spending per order
- **Buying Power Check**: Verifies sufficient account balance
- **Confirmation Prompt**: Requires explicit confirmation for live orders
- **Comprehensive Logging**: Tracks all operations for audit trail

## Files

- `main.py`: CLI interface and main application flow
- `alpaca_client.py`: Alpaca API wrapper
- `option_chain.py`: Option chain retrieval and processing
- `option_selector.py`: Greeks-based option selection logic
- `order_manager.py`: Order execution and management
- `requirements.txt`: Python dependencies
- `.env.example`: Template for API credentials

## Important Notes

- **Paper Trading**: Always test with paper trading first
- **Market Hours**: Options can only be traded during market hours
- **API Limits**: Be aware of Alpaca's rate limits
- **Risk**: Options trading involves significant risk. This tool is for educational purposes

## Troubleshooting

1. **No options found**: Ensure the ticker supports options and market is open
2. **Authentication failed**: Verify API credentials in .env file
3. **Insufficient data**: Some options may lack Greeks data
4. **Budget constraints**: Increase budget or adjust search parameters

## Disclaimer

This software is for educational purposes only. Options trading carries substantial risk of loss. Always understand the risks and consult with a financial advisor before trading.