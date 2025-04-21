from typing import Any

class BrokerBase:
    def place_order(self, order: Any) -> Any:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> Any:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> Any:
        raise NotImplementedError 