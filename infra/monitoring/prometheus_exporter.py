from prometheus_client import start_http_server, Gauge
import random
import time

training_progress = Gauge('training_progress', 'Training progress percentage')
queue_length = Gauge('queue_length', 'Length of task queue')
trade_latency = Gauge('trade_latency_seconds', 'Trade execution latency in seconds')

def simulate_metrics():
    while True:
        training_progress.set(random.uniform(0, 100))
        queue_length.set(random.randint(0, 100))
        trade_latency.set(random.uniform(0.01, 1.0))
        time.sleep(5)

if __name__ == "__main__":
    start_http_server(9100)
    simulate_metrics() 