import pytest
import zmq
from pydantic import BaseModel
from shared.zmq_protocol import get_zmq_context, send_json, recv_json

@pytest.mark.asyncio
async def test_pub_sub_json_roundtrip():
    ctx = get_zmq_context()
    pub = ctx.socket(zmq.PUB)
    sub = ctx.socket(zmq.SUB)
    try:
        endpoint = "inproc://test_pubsub"
        pub.bind(endpoint)
        sub.connect(endpoint)
        sub.setsockopt_string(zmq.SUBSCRIBE, "")

        # Give ZMQ time to connect (inproc needs a tiny sleep)
        import asyncio
        await asyncio.sleep(0.05)

        class Model(BaseModel):
            foo: str

        obj = {"foo": "bar"}
        send_json(pub, obj)

        # ZMQ PUB/SUB is async, so wait for message
        poller = zmq.Poller()
        poller.register(sub, zmq.POLLIN)
        socks = dict(await asyncio.get_event_loop().run_in_executor(
            None, poller.poll, 1000))
        assert sub in socks

        result = await asyncio.get_event_loop().run_in_executor(
            None, recv_json, sub, Model)
        assert result.foo == obj["foo"]
    finally:
        pub.close()
        sub.close() 