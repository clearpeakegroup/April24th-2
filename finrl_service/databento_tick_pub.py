import zmq
import os
from shared.zmq_protocol import ZMQ_TICK_SOCKET, get_zmq_context
from backend.data_ingestion.decompression import decompress_zstd
import databento as db

DBN_PATH = os.environ.get("DBN_PATH", "test.dbn.zst")


def stream_dbn_ticks_to_zmq(dbn_path=DBN_PATH, zmq_pub_addr=ZMQ_TICK_SOCKET):
    ctx = get_zmq_context()
    pub = ctx.socket(zmq.PUB)
    pub.bind(f"ipc://{zmq_pub_addr}")
    with open(dbn_path, 'rb') as f:
        data = f.read()
        decompressed = decompress_zstd(data)
        for tick in db.DBN(decompressed):
            pub.send_pyobj(tick)

if __name__ == "__main__":
    stream_dbn_ticks_to_zmq() 