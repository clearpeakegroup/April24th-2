import asyncio
from backend.services.tick_stream_consumer import TickStreamConsumer
from backend.services.agent_output_stream import publish_agent_signal

async def my_tick_handler(tick):
    # Example: Replace with your DRL/model/strategy logic
    print("[DEMO_AGENT] Received tick:", tick)
    # Example: Publish a sample signal
    signal = {
        "symbol": tick["instrument"],
        "action": "buy",  # Replace with real logic
        "confidence": 0.95,
        "ts_event": tick["ts_event"]
    }
    publish_agent_signal(signal)

async def run_agent():
    consumer = TickStreamConsumer(group="agents", consumer="demo_agent")
    await consumer.consume(my_tick_handler)

if __name__ == "__main__":
    asyncio.run(run_agent()) 