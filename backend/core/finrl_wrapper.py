from finrl import config, data_processor, env, models
from typing import List, Dict

data_source_registry = {
    "databento": lambda: data_processor.DataProcessor(data_source="databento"),
    "yahoo": lambda: data_processor.DataProcessor(data_source="yahoo"),
    "binance": lambda: data_processor.DataProcessor(data_source="binance"),
    # Add more as needed
}

def create_data_processor(source: str):
    """Factory for FinRL data processors."""
    if source not in data_source_registry:
        raise ValueError(f"Unsupported data source: {source}")
    return data_source_registry[source]()

def create_env(tickers: List[str], start: str, end: str, data_source: str = "databento", **kwargs):
    """Create a FinRL StockTradingEnv using the selected data source."""
    dp = create_data_processor(data_source)
    price_array, tech_array, turbulence_array = dp.run(
        tickers, start, end, time_interval=kwargs.get("time_interval", "1Min")
    )
    e = env.StockTradingEnv(
        price_array=price_array,
        tech_array=tech_array,
        turbulence_array=turbulence_array,
    )
    return e 