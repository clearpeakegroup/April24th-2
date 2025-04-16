import redis
import json
import logging
from execution.broker_connector import ETradeBrokerConnector
from execution.risk_manager import RiskManager

class Executor:
    def __init__(self, redis_url: str, broker: ETradeBrokerConnector, risk_manager: RiskManager):
        self.redis = redis.Redis.from_url(redis_url)
        self.broker = broker
        self.risk_manager = risk_manager
        self.signal_channel = "agent_signals"
        self.confirm_channel = "trade_confirmations"

    def run(self):
        pubsub = self.redis.pubsub()
        pubsub.subscribe(self.signal_channel)
        logging.info("Executor started, waiting for signals...")
        for message in pubsub.listen():
            if message['type'] != 'message':
                continue
            try:
                signal = json.loads(message['data'])
                symbol = signal['symbol']
                qty = signal['qty']
                side = signal['side']
                self.risk_manager.check()
                order = self.broker.place_order(symbol, qty, side)
                self.risk_manager.update_position(symbol, qty if side == 'buy' else -qty)
                # Simulate PnL update (stub)
                self.risk_manager.update_pnl(0.0)
                self.redis.publish(self.confirm_channel, json.dumps(order))
                logging.info(f"Order executed: {order}")
            except Exception as e:
                logging.error(f"Execution error: {e}") 