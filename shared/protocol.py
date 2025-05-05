ZMQ_TICK_SOCKET = "/tmp/dbn_ticks.sock"
ZMQ_EXEC_SOCKET = "/tmp/exec.sock"

def get_zmq_context():
    import zmq
    return zmq.Context.instance()
