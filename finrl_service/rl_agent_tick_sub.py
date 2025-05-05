import zmq
from shared.zmq_protocol import ZMQ_TICK_SOCKET, ZMQ_EXEC_SOCKET, get_zmq_context
from backend.model_core.model_runner import ModelRunner
from backend.data_ingestion.feature_joiner import join_features
import os
import time

# Initialize RL model runner (ensure model is loaded as per your logic)
model_runner = ModelRunner()

ctx = get_zmq_context()
sub = ctx.socket(zmq.SUB)
sub.connect(f"ipc://{ZMQ_TICK_SOCKET}")
sub.setsockopt_string(zmq.SUBSCRIBE, "")

req = ctx.socket(zmq.REQ)
req.connect(f"ipc://{ZMQ_EXEC_SOCKET}")

def process_and_dispatch():
    while True:
        tick = sub.recv_pyobj()
        # Feature engineering (replace with your actual pipeline)
        features = join_features(tick)
        # RL inference (replace with your actual logic)
        action = model_runner.run_inference(features)
        # Dispatch action to execution service
        req.send_json(action)
        result = req.recv_json()
        # Optionally log or process result
        print(f"Dispatched action: {action}, Received result: {result}")
        # Throttle if needed (remove for max speed)
        # time.sleep(0.001)

if __name__ == "__main__":
    process_and_dispatch() 