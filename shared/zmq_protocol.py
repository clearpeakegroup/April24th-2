import zmq

ZMQ_TICK_SOCKET = "/tmp/dbn_ticks.sock"
ZMQ_EXEC_SOCKET = "/tmp/exec.sock"

def get_zmq_context() -> zmq.Context:
    """Return singleton ZeroMQ context."""
    return zmq.Context.instance()

def send_json(sock: zmq.Socket, obj: dict) -> None:
    """Non-blocking JSON send; ignores EAGAIN."""
    try:
        sock.send_json(obj, flags=zmq.NOBLOCK)
    except zmq.Again:
        pass

def recv_json(sock: zmq.Socket, model_cls):
    """
    Blocking receive, parse into `model_cls` (a pydantic model) and return it.
    """
    data = sock.recv_json()
    return model_cls(**data) 