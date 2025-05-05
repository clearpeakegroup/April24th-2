from backend.execution.etrade_broker import ETradeBroker
from backend.execution.ibkr_broker import IBKRBroker
from backend.models import SessionLocal, Order, OrderStatus
import logging

class ExecutionService:
    def __init__(self):
        self.brokers = {
            "etrade": ETradeBroker(),
            "ibkr": IBKRBroker(),
        }

    def select_broker(self, order):
        # For demo: alternate by symbol or config
        if order.symbol.startswith("A"):  # e.g., AAPL
            return self.brokers["etrade"]
        else:
            return self.brokers["ibkr"]

    def execute_order(self, order_id: int):
        session = SessionLocal()
        try:
            order = session.query(Order).filter_by(id=order_id).first()
            if not order:
                logging.error(f"Order {order_id} not found for execution.")
                return
            if order.status != OrderStatus.pending:
                logging.info(f"Order {order_id} not in pending state.")
                return
            broker = self.select_broker(order)
            response = broker.place_order(order)
            if response.get("status") == "filled":
                order.status = OrderStatus.filled
                order.filled_at = func.now()
                order.broker_order_id = response.get("broker_order_id")
            elif response.get("status") == "rejected":
                order.status = OrderStatus.rejected
                order.rejected_at = func.now()
                order.meta = response.get("reason")
            session.commit()
        except Exception as e:
            logging.error(f"ExecutionService error: {e}")
        finally:
            session.close()

    def flatten_positions(self, user_id):
        """Close all open positions for the user by generating market orders in the opposite direction for each symbol."""
        from backend.models import SessionLocal, Order, OrderStatus
        session = SessionLocal()
        try:
            # Get all open positions from the risk manager
            if not hasattr(self, 'risk_manager'):
                from backend.execution.risk_manager import RiskManager
                self.risk_manager = RiskManager()
            positions = self.risk_manager.positions.get(user_id, {})
            for symbol, pos in positions.items():
                if pos == 0:
                    continue
                # Determine side to flatten
                side = 'sell' if pos > 0 else 'buy'
                qty = abs(pos)
                # Create a market order to flatten
                order = Order(
                    user_id=user_id,
                    agent_id=None,
                    symbol=symbol,
                    qty=qty,
                    side=side,
                    order_type='market',
                    status=OrderStatus.pending
                )
                session.add(order)
                session.commit()
                session.refresh(order)
                # Execute the order
                self.execute_order(order.id)
                # Update risk manager
                self.risk_manager.update_position(user_id, symbol, -pos)
        except Exception as e:
            import logging
            logging.error(f"Error flattening positions for user {user_id}: {e}")
        finally:
            session.close() 