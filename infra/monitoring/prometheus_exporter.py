from prometheus_client import start_http_server, Gauge
import random
import time

training_progress = Gauge('training_progress', 'Training progress percentage')
queue_length = Gauge('queue_length', 'Length of task queue')
trade_latency = Gauge('trade_latency_seconds', 'Trade execution latency in seconds')
sharpe_ratio = Gauge('sharpe_ratio', 'Portfolio Sharpe ratio')
sortino_ratio = Gauge('sortino_ratio', 'Portfolio Sortino ratio')
fee_pct = Gauge('fee_pct', 'Fee percentage of notional')
maker_fill_pct = Gauge('maker_fill_pct', 'Maker fill percentage')
realised_tax = Gauge('realised_tax', 'Realised tax liability')

def update_metrics(sharpe, sortino, fee, maker_fill, tax):
    sharpe_ratio.set(sharpe)
    sortino_ratio.set(sortino)
    fee_pct.set(fee)
    maker_fill_pct.set(maker_fill)
    realised_tax.set(tax)

def simulate_metrics():
    while True:
        training_progress.set(random.uniform(0, 100))
        queue_length.set(random.randint(0, 100))
        trade_latency.set(random.uniform(0.01, 1.0))
        update_metrics(
            sharpe=random.uniform(0, 3),
            sortino=random.uniform(0, 4),
            fee=random.uniform(0, 0.01),
            maker_fill=random.uniform(0.5, 1.0),
            tax=random.uniform(0, 10000)
        )
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(9100)
    simulate_metrics() 