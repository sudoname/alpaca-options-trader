import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
import logging
from dataclasses import dataclass
from alpaca_client import AlpacaOptionsClient
from option_chain import OptionChainAnalyzer
from option_selector import OptionSelector
from alpaca.data.historical import StockHistoricalDataClient
from alpaca.data.requests import StockBarsRequest, OptionBarsRequest
from alpaca.data.timeframe import TimeFrame
import json

@dataclass
class Trade:
    entry_date: datetime
    symbol: str
    underlying: str
    strike: float
    expiration: datetime
    option_type: str
    entry_price: float
    contracts: int
    cost: float
    exit_date: Optional[datetime] = None
    exit_price: Optional[float] = None
    pnl: Optional[float] = None
    pnl_percent: Optional[float] = None
    exit_reason: Optional[str] = None

class OptionBacktester:
    def __init__(self, start_date: str, end_date: str, budget_per_trade: float = 500):
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.budget_per_trade = budget_per_trade
        self.initial_capital = 10000
        self.current_capital = self.initial_capital
        self.trades: List[Trade] = []
        self.client = AlpacaOptionsClient()
        self.analyzer = OptionChainAnalyzer(self.client)
        self.selector = OptionSelector(budget=budget_per_trade)
        self.logger = logging.getLogger(__name__)

    def get_historical_price(self, ticker: str, date: datetime) -> Optional[float]:
        try:
            request = StockBarsRequest(
                symbol_or_symbols=[ticker],
                timeframe=TimeFrame.Day,
                start=date - timedelta(days=1),
                end=date + timedelta(days=1),
                limit=1
            )
            bars = self.client.data_client.get_stock_bars(request)
            if ticker in bars and len(bars[ticker]) > 0:
                return float(bars[ticker][0].close)
            return None
        except Exception as e:
            self.logger.error(f"Error getting historical price: {e}")
            return None

    def get_option_price_at_date(self, option_symbol: str, date: datetime) -> Optional[float]:
        try:
            request = OptionBarsRequest(
                symbol_or_symbols=[option_symbol],
                timeframe=TimeFrame.Day,
                start=date - timedelta(days=1),
                end=date + timedelta(days=1),
                limit=1
            )
            bars = self.client.data_client.get_option_bars(request)
            if option_symbol in bars and len(bars[option_symbol]) > 0:
                return float(bars[option_symbol][0].close)
            return None
        except Exception as e:
            self.logger.debug(f"Could not get option price for {option_symbol} at {date}")
            return None

    def simulate_trade_entry(self, ticker: str, trade_date: datetime) -> Optional[Trade]:
        try:
            stock_price = self.get_historical_price(ticker, trade_date)
            if not stock_price:
                return None

            option_chain = self.analyzer.get_option_chain(
                ticker=ticker,
                min_days_to_expiry=30,
                max_days_to_expiry=90
            )

            if option_chain.empty:
                return None

            itm_options = self.analyzer.filter_itm_options(option_chain, stock_price, 'CALL')

            if itm_options.empty:
                return None

            best_option = self.selector.select_best_option(itm_options, stock_price)

            if not best_option:
                return None

            trade = Trade(
                entry_date=trade_date,
                symbol=best_option['symbol'],
                underlying=ticker,
                strike=best_option['strike'],
                expiration=pd.to_datetime(best_option['expiration']),
                option_type=best_option['type'],
                entry_price=best_option['ask'],
                contracts=best_option['contracts_to_buy'],
                cost=best_option['total_cost']
            )

            return trade

        except Exception as e:
            self.logger.error(f"Error simulating trade entry: {e}")
            return None

    def simulate_trade_exit(self, trade: Trade, current_date: datetime) -> Trade:
        if current_date >= trade.expiration:
            trade.exit_date = trade.expiration
            trade.exit_reason = "Expiration"

            stock_price = self.get_historical_price(trade.underlying, trade.expiration)
            if stock_price:
                if trade.option_type.upper() == 'CALL':
                    intrinsic_value = max(0, stock_price - trade.strike)
                else:
                    intrinsic_value = max(0, trade.strike - stock_price)

                trade.exit_price = intrinsic_value
                trade.pnl = (intrinsic_value * 100 * trade.contracts) - trade.cost
                trade.pnl_percent = (trade.pnl / trade.cost) * 100
        else:
            exit_price = self.get_option_price_at_date(trade.symbol, current_date)
            if exit_price and exit_price > trade.entry_price * 1.5:
                trade.exit_date = current_date
                trade.exit_price = exit_price
                trade.exit_reason = "Take Profit (50%)"
                trade.pnl = (exit_price * 100 * trade.contracts) - trade.cost
                trade.pnl_percent = (trade.pnl / trade.cost) * 100
            elif exit_price and exit_price < trade.entry_price * 0.5:
                trade.exit_date = current_date
                trade.exit_price = exit_price
                trade.exit_reason = "Stop Loss (50%)"
                trade.pnl = (exit_price * 100 * trade.contracts) - trade.cost
                trade.pnl_percent = (trade.pnl / trade.cost) * 100

        return trade

    def run_backtest(self, tickers: List[str], trade_frequency: int = 7) -> Dict:
        current_date = self.start_date
        open_trades: List[Trade] = []
        closed_trades: List[Trade] = []
        equity_curve = []

        while current_date <= self.end_date:
            for i, trade in enumerate(open_trades[:]):
                updated_trade = self.simulate_trade_exit(trade, current_date)
                if updated_trade.exit_date:
                    self.current_capital += updated_trade.pnl if updated_trade.pnl else 0
                    closed_trades.append(updated_trade)
                    open_trades.remove(trade)
                    self.logger.info(f"Closed trade: {updated_trade.symbol} PnL: ${updated_trade.pnl:.2f}")

            if current_date.weekday() == 0 and len(open_trades) < 3:
                for ticker in tickers:
                    if self.current_capital > self.budget_per_trade:
                        trade = self.simulate_trade_entry(ticker, current_date)
                        if trade:
                            self.current_capital -= trade.cost
                            open_trades.append(trade)
                            self.logger.info(f"Opened trade: {trade.symbol} Cost: ${trade.cost:.2f}")
                            break

            equity_curve.append({
                'date': current_date,
                'capital': self.current_capital + sum(self._get_trade_value(t, current_date) for t in open_trades),
                'open_trades': len(open_trades)
            })

            current_date += timedelta(days=1)

        for trade in open_trades:
            updated_trade = self.simulate_trade_exit(trade, self.end_date)
            updated_trade.exit_reason = "Backtest End"
            closed_trades.append(updated_trade)

        self.trades = closed_trades
        return self.generate_report(equity_curve)

    def _get_trade_value(self, trade: Trade, date: datetime) -> float:
        price = self.get_option_price_at_date(trade.symbol, date)
        if price:
            return price * 100 * trade.contracts
        return trade.cost

    def generate_report(self, equity_curve: List[Dict]) -> Dict:
        if not self.trades:
            return {
                'total_trades': 0,
                'message': 'No trades executed during backtest period'
            }

        df_trades = pd.DataFrame([{
            'entry_date': t.entry_date,
            'exit_date': t.exit_date,
            'symbol': t.symbol,
            'underlying': t.underlying,
            'strike': t.strike,
            'type': t.option_type,
            'contracts': t.contracts,
            'entry_price': t.entry_price,
            'exit_price': t.exit_price,
            'cost': t.cost,
            'pnl': t.pnl,
            'pnl_percent': t.pnl_percent,
            'exit_reason': t.exit_reason
        } for t in self.trades])

        df_equity = pd.DataFrame(equity_curve)

        winning_trades = df_trades[df_trades['pnl'] > 0]
        losing_trades = df_trades[df_trades['pnl'] < 0]

        total_pnl = df_trades['pnl'].sum()
        win_rate = len(winning_trades) / len(df_trades) * 100 if len(df_trades) > 0 else 0

        avg_win = winning_trades['pnl'].mean() if len(winning_trades) > 0 else 0
        avg_loss = losing_trades['pnl'].mean() if len(losing_trades) > 0 else 0

        max_drawdown = self.calculate_max_drawdown(df_equity['capital'].values)
        sharpe_ratio = self.calculate_sharpe_ratio(df_equity['capital'].pct_change().dropna())

        report = {
            'period': {
                'start': self.start_date.strftime('%Y-%m-%d'),
                'end': self.end_date.strftime('%Y-%m-%d'),
                'days': (self.end_date - self.start_date).days
            },
            'performance': {
                'initial_capital': self.initial_capital,
                'final_capital': df_equity['capital'].iloc[-1] if not df_equity.empty else self.initial_capital,
                'total_pnl': total_pnl,
                'total_return': (total_pnl / self.initial_capital) * 100,
                'max_drawdown': max_drawdown,
                'sharpe_ratio': sharpe_ratio
            },
            'trades': {
                'total': len(df_trades),
                'winners': len(winning_trades),
                'losers': len(losing_trades),
                'win_rate': win_rate,
                'avg_win': avg_win,
                'avg_loss': avg_loss,
                'profit_factor': abs(winning_trades['pnl'].sum() / losing_trades['pnl'].sum()) if len(losing_trades) > 0 and losing_trades['pnl'].sum() != 0 else 0
            },
            'trade_details': df_trades.to_dict('records'),
            'equity_curve': df_equity.to_dict('records')
        }

        return report

    def calculate_max_drawdown(self, equity_curve: np.ndarray) -> float:
        peak = np.maximum.accumulate(equity_curve)
        drawdown = (equity_curve - peak) / peak * 100
        return abs(drawdown.min())

    def calculate_sharpe_ratio(self, returns: pd.Series) -> float:
        if len(returns) < 2:
            return 0
        return (returns.mean() / returns.std()) * np.sqrt(252) if returns.std() != 0 else 0

    def save_report(self, report: Dict, filename: str = None):
        if not filename:
            filename = f"backtest_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        with open(filename, 'w') as f:
            json.dump(report, f, indent=2, default=str)

        self.logger.info(f"Report saved to {filename}")
        return filename