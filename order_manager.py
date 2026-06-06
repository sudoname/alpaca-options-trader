from typing import Dict, Optional
import logging
from alpaca.trading.enums import OrderSide
from tabulate import tabulate

class OrderManager:
    def __init__(self, alpaca_client, dry_run: bool = True):
        self.client = alpaca_client
        self.dry_run = dry_run
        self.logger = logging.getLogger(__name__)

    def execute_order(self, option_details: Dict) -> Optional[Dict]:
        if not option_details:
            self.logger.error("No option details provided")
            return None

        self._display_order_preview(option_details)

        if self.dry_run:
            self.logger.info("DRY RUN MODE - Order not executed")
            return {
                'status': 'dry_run',
                'symbol': option_details['symbol'],
                'quantity': option_details['contracts_to_buy'],
                'total_cost': option_details['total_cost'],
                'message': 'Order would have been placed in live mode'
            }

        try:
            account = self.client.get_account()
            buying_power = float(account.buying_power)

            if buying_power < option_details['total_cost']:
                self.logger.error(f"Insufficient buying power. Available: ${buying_power:.2f}, Required: ${option_details['total_cost']:.2f}")
                return None

            order = self.client.place_option_order(
                symbol=option_details['symbol'],
                qty=option_details['contracts_to_buy'],
                side=OrderSide.BUY
            )

            result = {
                'status': 'submitted',
                'order_id': order.id,
                'symbol': order.symbol,
                'quantity': order.qty,
                'side': order.side,
                'order_type': order.order_type,
                'time_in_force': order.time_in_force,
                'submitted_at': order.submitted_at
            }

            self.logger.info(f"Order placed successfully: {order.id}")
            return result

        except Exception as e:
            self.logger.error(f"Error placing order: {e}")
            return None

    def _display_order_preview(self, option_details: Dict):
        print("\n" + "="*60)
        print("ORDER PREVIEW")
        print("="*60)

        data = [
            ["Symbol", option_details['symbol']],
            ["Strike Price", f"${option_details['strike']:.2f}"],
            ["Expiration", option_details['expiration']],
            ["Type", option_details['type'].upper()],
            ["Days to Expiry", option_details['days_to_expiry']],
            ["", ""],
            ["Ask Price", f"${option_details['ask']:.2f}"],
            ["Bid Price", f"${option_details['bid']:.2f}"],
            ["Contracts to Buy", option_details['contracts_to_buy']],
            ["Total Cost", f"${option_details['total_cost']:.2f}"],
            ["", ""],
            ["Greeks", ""],
            ["Delta", f"{option_details['delta']:.4f}" if option_details['delta'] else "N/A"],
            ["Gamma", f"{option_details['gamma']:.4f}" if option_details['gamma'] else "N/A"],
            ["Theta", f"{option_details['theta']:.4f}" if option_details['theta'] else "N/A"],
            ["Vega", f"{option_details['vega']:.4f}" if option_details['vega'] else "N/A"],
            ["IV", f"{option_details['iv']:.2%}" if option_details['iv'] else "N/A"],
            ["", ""],
            ["Selection Score", f"{option_details['score']:.2f}"]
        ]

        print(tabulate(data, tablefmt="grid"))
        print("="*60)

    def get_positions(self):
        try:
            positions = self.client.trading_client.get_all_positions()
            return positions
        except Exception as e:
            self.logger.error(f"Error getting positions: {e}")
            return []

    def get_orders(self, status='all'):
        try:
            if status == 'all':
                orders = self.client.trading_client.get_orders()
            else:
                orders = self.client.trading_client.get_orders(status=status)
            return orders
        except Exception as e:
            self.logger.error(f"Error getting orders: {e}")
            return []