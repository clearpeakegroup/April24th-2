from typing import Any, Dict, List, Callable
import logging

class Middleware:
    """
    Base class for middleware components that can process data events and orders.
    """
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process or modify a data event before it reaches the strategy."""
        return event

    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        """Pre-process or modify an order before it is executed."""
        return order

class LoggingMiddleware(Middleware):
    """
    Middleware that logs events and orders for monitoring and debugging.
    """
    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"[Middleware] Event: {event}")
        return event

    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        logging.debug(f"[Middleware] Order: {order}")
        return order

class RiskManagementMiddleware(Middleware):
    """
    Middleware that enforces a maximum position size for risk management.
    """
    def __init__(self, max_position: float = 10.0):
        self.max_position = max_position
        self.current_position = 0.0

    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        qty = order.get('qty', 0)
        action = order.get('action', '')
        # Enforce max position size
        if action == 'buy':
            if self.current_position + qty > self.max_position:
                logging.warning(f"Order exceeds max position. Capping to {self.max_position - self.current_position}")
                order['qty'] = max(0, self.max_position - self.current_position)
            self.current_position += order['qty']
        elif action == 'sell':
            if self.current_position - qty < -self.max_position:
                logging.warning(f"Order exceeds min position. Capping to {self.current_position + self.max_position}")
                order['qty'] = max(0, self.current_position + self.max_position)
            self.current_position -= order['qty']
        return order

class PnLMiddleware(Middleware):
    """
    Middleware that tracks running PnL for executed orders (very basic, assumes fill at price).
    """
    def __init__(self):
        self.pnl = 0.0
        self.last_price = None
        self.position = 0.0

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        if 'price' in event:
            self.last_price = event['price']
        return event

    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        qty = order.get('qty', 0)
        action = order.get('action', '')
        if self.last_price is not None:
            if action == 'buy':
                self.pnl -= qty * self.last_price
                self.position += qty
            elif action == 'sell':
                self.pnl += qty * self.last_price
                self.position -= qty
            logging.info(f"[PnL] Current PnL: {self.pnl}, Position: {self.position}")
        return order

class MiddlewareManager:
    """
    Chains multiple middleware components for event and order processing.
    """
    def __init__(self, middlewares: List[Middleware]):
        self.middlewares = middlewares

    def process_event(self, event: Dict[str, Any]) -> Dict[str, Any]:
        for mw in self.middlewares:
            event = mw.process_event(event)
        return event

    def process_order(self, order: Dict[str, Any]) -> Dict[str, Any]:
        for mw in self.middlewares:
            order = mw.process_order(order)
        return order 