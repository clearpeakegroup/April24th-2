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