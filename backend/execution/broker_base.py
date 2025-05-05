from typing import Any

class BrokerBase:
    def place_order(self, *args, **kwargs):
        raise RuntimeError("place_order() must be implemented by a subclass of BrokerBase.")

    def cancel_order(self, *args, **kwargs):
        raise RuntimeError("cancel_order() must be implemented by a subclass of BrokerBase.")

    def get_order_status(self, *args, **kwargs):
        raise RuntimeError("get_order_status() must be implemented by a subclass of BrokerBase.")

    def place_order(self, order: Any) -> Any:
        raise NotImplementedError

    def cancel_order(self, order_id: str) -> Any:
        raise NotImplementedError

    def get_order_status(self, order_id: str) -> Any:
        raise NotImplementedError 