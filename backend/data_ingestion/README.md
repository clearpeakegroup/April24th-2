# Data Ingestion Layer

## Overview

This module ingests and processes both market data and news sources for the full pipeline (training, backtesting, live trading).

### Market Data (5-Feed)
- **Assets:** MES, MNQ, ZF, ZN, UB
- **Purpose:**
  - MES & MNQ: actively traded
  - ZF, ZN, UB: provide additional market context/features
- **Data:** MBO DBN ZSTD (historical & live)
- **Config:** Asset list is centralized in `config/default.yml` under the `assets` key. Use `ConfigLoader.get_assets()` to retrieve.
- **Ingestion:**
  - `databento_ingest.py` and `live_stream.py` are asset-agnostic and use the centralized asset list.

### News Sources
- **Sources:** Twitter, MarketWatch, Benzinga, MacroEdge, etc.
- **Data:** JSON (zstd-compressed), historical and live
- **Ingestion:** Each source has its own async module, all writing to a unified, time-partitioned structure.

### Decompression
- All ingestion and feature pipelines use `decompress_zstd` (GPU-accelerated if available).

## Extending
- **To add/remove assets:** Edit the `assets` list in `config/default.yml`.
- **To add a news source:** Add a new async ingestion module and import it in `__init__.py`.

## Usage
- Use `ConfigLoader.get_assets('config/default.yml')` to retrieve the current asset list for ingestion modules.
- All data is stored in a time-partitioned structure for easy alignment and retrieval.

## Health Checks & Monitoring

All ingestion and feature modules expose a `health_check()` function:
- Returns True if config is valid and core dependencies (API/WS host, output directory, etc.) are reachable.
- Returns False and logs an error if not healthy.
- Use for orchestration, liveness/readiness probes, or external monitoring.

**Example:**
```python
from backend.data_ingestion.news_benzinga import health_check
if not health_check():
    print("Benzinga ingestion is not healthy!")
```

## Error Handling & Logging
- All modules use robust try/except blocks around I/O, network, and model calls.
- All exceptions are logged with stack traces.
- Warnings are logged for non-critical issues (e.g., empty/malformed data).
- All config is validated at startup; missing/invalid config halts the module with a clear error.

## Production-Grade Guarantees
- All business, model, and strategy logic is modular, validated, and monitored.
- No "loose ends" or unhandled edge cases.
- All critical paths are covered by integration and edge-case tests.
- Platform is ready for live, research, and production deployment.

---

This architecture ensures robust, scalable ingestion for both market and news data, supporting advanced multi-asset, multi-source modeling and trading. 