import logging
import random
import time
from backend.execution.broker_base import BrokerBase

class IBKRBroker(BrokerBase):
    def place_order(self, order):
        try:
            time.sleep(0.5)
            if random.random() < 0.85:
                return {"broker_order_id": f"ib_{random.randint(1000,9999)}", "status": "filled"}
            else:
                return {"broker_order_id": None, "status": "rejected", "reason": "Simulated rejection"}
        except Exception as e:
            logging.error(f"IBKRBroker place_order error: {e}")
            raise

    def cancel_order(self, order_id):
        try:
            time.sleep(0.2)
            return {"order_id": order_id, "status": "cancelled"}
        except Exception as e:
            logging.error(f"IBKRBroker cancel_order error: {e}")
            raise

    def get_order_status(self, order_id):
        try:
            return {"order_id": order_id, "status": random.choice(["pending", "filled", "cancelled"]) }
        except Exception as e:
            logging.error(f"IBKRBroker get_order_status error: {e}")
            raise 