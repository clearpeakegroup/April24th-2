import argparse
import importlib
import logging
import logging.config
import yaml
import os
from typing import Any
from trader_core.data_connectors import MarketDataConnector, PCAPReplayConnector, LiveFeedConnector
from trader_core.execution import ExecutionInterface, SimulatedExecution, BrokerExecution
from trader_core.config_loader import ConfigLoader
from trader_core.middleware import MiddlewareManager, LoggingMiddleware, RiskManagementMiddleware, PnLMiddleware

class StrategyRunner:
    """
    Orchestrates the loading, configuration, and execution of trading strategies.
    Handles data ingestion, strategy lifecycle, middleware, and order execution.
    """
    def __init__(self, strategy_cls: Any, data_connector: MarketDataConnector, execution: ExecutionInterface, middleware_manager: MiddlewareManager):
        """
        Args:
            strategy_cls: The strategy class to instantiate and run.
            data_connector: An instance of MarketDataConnector for data ingestion.
            execution: An instance of ExecutionInterface for order execution.
            middleware_manager: MiddlewareManager for event and order processing.
        """
        self.strategy = strategy_cls()
        self.data_connector = data_connector
        self.execution = execution
        self.middleware_manager = middleware_manager

    def run(self):
        """
        Main loop: feeds data events to the strategy and handles execution.
        """
        logging.info("Starting strategy runner...")
        for event in self.data_connector.stream():
            event = self.middleware_manager.process_event(event)
            if hasattr(self.strategy, 'on_tick'):
                self.strategy.on_tick(event['price'], event.get('qty', 1), event['timestamp'])
            elif hasattr(self.strategy, 'act'):
                self.strategy.act(event)
            if hasattr(self.strategy, 'active_signal') and getattr(self.strategy, 'active_signal', None):
                order = self.middleware_manager.process_order(self.strategy.active_signal)
                self.execution.execute(order)

if __name__ == "__main__":
    """
    CLI entry point for running trading strategies with configurable data connectors and execution layers.
    Loads logging configuration, parses CLI arguments, loads config, and starts the runner.
    """
    # Try to load YAML logging config
    log_config_path = 'config/logging.yml'
    if os.path.exists(log_config_path):
        with open(log_config_path, 'r') as f:
            logging_config = yaml.safe_load(f)
        logging.config.dictConfig(logging_config)
    else:
        logging.basicConfig(level=logging.INFO)

    parser = argparse.ArgumentParser(description="Run a trading strategy.")
    parser.add_argument('--config', type=str, default='config/default.yml', help='Path to config YAML')
    parser.add_argument('--strategy', type=str, required=True, help='Strategy class name (e.g., Head8IcebergAbsorptionStrategy)')
    parser.add_argument('--module', type=str, required=True, help='Module path (e.g., trader_core.strategies.options.head8_iceberg_absorption)')
    parser.add_argument('--connector', type=str, choices=['live', 'pcap'], help='Data connector type')
    parser.add_argument('--symbol', type=str, help='Symbol for live feed')
    parser.add_argument('--interval', type=float, help='Interval for live feed (seconds)')
    parser.add_argument('--pcap', type=str, help='Path to PCAP/CSV file for replay')
    parser.add_argument('--sleep', type=float, help='Sleep between replay events (seconds)')
    parser.add_argument('--execution', type=str, choices=['sim', 'broker'], help='Execution type')
    args = parser.parse_args()

    # Load config
    config = ConfigLoader.load_config(args.config)

    # Overlay CLI args (if provided)
    connector_type = args.connector or config.get('connector', 'live')
    symbol = args.symbol or config.get('symbol', 'ES')
    interval = args.interval if args.interval is not None else float(config.get('interval', 0.1))
    pcap_path = args.pcap or config.get('pcap')
    sleep = args.sleep if args.sleep is not None else config.get('sleep')
    execution_type = args.execution or config.get('execution', 'sim')

    # Dynamically import strategy class
    module = importlib.import_module(args.module)
    strategy_cls = getattr(module, args.strategy)

    # Select data connector
    if connector_type == 'pcap':
        if not pcap_path:
            raise ValueError('Must provide --pcap path for PCAPReplayConnector')
        data_connector = PCAPReplayConnector(pcap_path, sleep=sleep)
    else:
        data_connector = LiveFeedConnector(symbol=symbol, interval=interval)

    # Select execution interface
    if execution_type == 'broker':
        execution = BrokerExecution()
    else:
        execution = SimulatedExecution()

    # Set up middleware (add more as needed)
    middleware_manager = MiddlewareManager([
        LoggingMiddleware(),
        RiskManagementMiddleware(max_position=10.0),
        PnLMiddleware()
    ])

    runner = StrategyRunner(strategy_cls, data_connector, execution, middleware_manager)
    runner.run() 