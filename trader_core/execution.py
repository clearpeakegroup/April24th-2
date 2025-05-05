import logging
from typing import Dict, Any

class ExecutionInterface:
    """
    Abstract base class for order execution. Subclasses must implement the execute() method.
    """
    def execute(self, order: Dict[str, Any]):
        """
        Execute an order. Must be implemented by subclasses.
        Args:
            order: Dictionary representing the order to execute.
        """
        raise NotImplementedError

class ExecutionConnector:
    def execute(self, order):
        raise RuntimeError("ExecutionConnector.execute() must be implemented by a subclass.")

class SimulatedExecution(ExecutionConnector):
    """
    Simulates order execution for backtesting and dry runs.
    """
    def execute(self, order: Dict[str, Any]):
        """
        Simulate order execution and log the action.
        Args:
            order: Dictionary representing the order to execute.
        Returns:
            dict: Simulated fill result.
        """
        logging.info(f"[SIMULATED] Executing order: {order}")
        # Simulate fill, slippage, etc. (expand as needed)
        order.status = 'filled'
        return order

class BrokerExecution(ExecutionConnector):
    """
    Stub for real broker integration. Extend with actual API calls.
    """
    def execute(self, order: Dict[str, Any]):
        """
        Stub for sending an order to a real broker.
        Args:
            order: Dictionary representing the order to execute.
        Returns:
            dict: Submission result (stub).
        """
        logging.info(f"[BROKER] Would execute order: {order}")
        # TODO: Integrate with broker API
        raise RuntimeError("BrokerExecution is not yet integrated with broker API. Please use SimulatedExecution or implement broker integration.")
        return {'status': 'submitted', 'order': order} 