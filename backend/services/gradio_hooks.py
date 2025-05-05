import time
import random

def stream_equity_curve(agent_id):
    t = int(time.time())
    for i in range(1000):
        yield (t + i, 100000 + random.uniform(-1000, 1000)) 