import zmq
from shared.zmq_protocol import ZMQ_EXEC_SOCKET, get_zmq_context
from backend.execution.ironbeam_broker import IronbeamBroker
import json

broker = IronbeamBroker()
ctx = get_zmq_context()
rep = ctx.socket(zmq.REP)
rep.bind(f"ipc://{ZMQ_EXEC_SOCKET}")

def execute_action(action):
    # Example: expects action dict with keys 'symbol', 'side', 'qty', etc.
    # Replace with your full execution logic as needed
    try:
        order_id = broker.place_order(action)
        status = broker.get_order_status(order_id)
        return {"status": "filled", "order_id": order_id, "details": status}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def main():
    while True:
        action = rep.recv_json()
        result = execute_action(action)
        rep.send_json(result)

if __name__ == "__main__":
    main() 