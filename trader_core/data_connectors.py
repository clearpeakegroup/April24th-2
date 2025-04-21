import time
import csv
from typing import Iterator, Dict, Any, Optional

class MarketDataConnector:
    """
    Abstract base class for market data connectors.
    Subclasses must implement the stream() method to yield market data events.
    """
    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Yield market data events as dictionaries.
        """
        raise NotImplementedError

class PCAPReplayConnector(MarketDataConnector):
    """
    Replays historical data from a CSV file (PCAP or similar format).
    Each row should represent a market data event.
    """
    def __init__(self, filepath: str, sleep: Optional[float] = None):
        """
        Args:
            filepath: Path to the CSV file.
            sleep: Seconds to sleep between events (None for as-fast-as-possible).
        """
        self.filepath = filepath
        self.sleep = sleep

    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Yield events from the CSV file, optionally throttled by sleep.
        """
        with open(self.filepath, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                event = {k: self._parse_value(v) for k, v in row.items()}
                yield event
                if self.sleep:
                    time.sleep(self.sleep)

    @staticmethod
    def _parse_value(val):
        """
        Attempt to convert a value to float, fallback to string.
        """
        try:
            return float(val)
        except ValueError:
            return val

class LiveFeedConnector(MarketDataConnector):
    """
    Simulates or connects to a live data feed. (Stub: yields synthetic data.)
    """
    def __init__(self, symbol: str = 'ES', interval: float = 0.1):
        """
        Args:
            symbol: Symbol to simulate.
            interval: Seconds between events.
        """
        self.symbol = symbol
        self.interval = interval

    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Yield synthetic market data events for demonstration/testing.
        """
        price = 100.0
        for i in range(100):
            event = {
                'symbol': self.symbol,
                'price': price + i * 0.1,
                'qty': 1 + (i % 5),
                'side': 'bid' if i % 2 == 0 else 'ask',
                'timestamp': time.time()
            }
            yield event
            time.sleep(self.interval)

class WebSocketConnector(MarketDataConnector):
    """
    Stub for a real-time WebSocket data connector.
    """
    def __init__(self, url: str, symbol: str):
        self.url = url
        self.symbol = symbol

    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Placeholder: Connect to a WebSocket and yield real-time events.
        """
        raise NotImplementedError("WebSocketConnector is not yet implemented.")

class BrokerAPIConnector(MarketDataConnector):
    """
    Stub for a broker API data connector (e.g., Interactive Brokers, Alpaca).
    """
    def __init__(self, api_key: str, symbol: str):
        self.api_key = api_key
        self.symbol = symbol

    def stream(self) -> Iterator[Dict[str, Any]]:
        """
        Placeholder: Connect to broker API and yield events.
        """
        raise NotImplementedError("BrokerAPIConnector is not yet implemented.") 