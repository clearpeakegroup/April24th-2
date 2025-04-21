# FinRL Platform

## Overview
A modular, extensible research and execution platform for advanced trading strategies, with a focus on microstructure, event-driven, and quantitative methods. Supports rapid strategy development, backtesting, and deployment.

## Features
- Modular strategy architecture (plug-and-play)
- Real-time and historical (replay) data ingestion
- Simulated and broker execution layers
- Centralized configuration management
- Structured logging and monitoring
- Pytest-based unit and integration tests
- CLI runner for strategy orchestration

## Directory Structure
```
trader-core/
  strategies/         # Strategy implementations
  data_connectors.py  # Data ingestion modules
  execution.py        # Execution layer
  runner.py           # Orchestration/runner
  config_loader.py    # Config loader
config/
  default.yml         # Default configuration
  logging.yml         # Logging configuration
utils/                # Utilities (e.g., orderbook)
tests/                # Unit and integration tests
README.md             # This file
```

## Setup
1. **Install dependencies:**
   ```bash
   pip install -r backend/requirements.txt
   ```
2. **Configure:**
   - Edit `config/default.yml` for strategy and data parameters.
   - Edit `config/logging.yml` for logging preferences.

## Usage
Run a strategy with the CLI runner:
```bash
python trader-core/runner.py \
  --strategy Head8IcebergAbsorptionStrategy \
  --module trader_core.strategies.options.head8_iceberg_absorption \
  --connector live \
  --symbol ES \
  --execution sim
```
For PCAP/CSV replay:
```bash
python trader-core/runner.py \
  --strategy Head8IcebergAbsorptionStrategy \
  --module trader_core.strategies.options.head8_iceberg_absorption \
  --connector pcap \
  --pcap path/to/sample.csv \
  --execution sim
```

## Testing
Run all tests:
```bash
pytest tests/
```

## Documentation
- All public classes and methods are documented with docstrings.
- (Optional) API docs can be generated with Sphinx or MkDocs.

## Extending
- Add new strategies in `trader-core/strategies/options/` or `fut_only/`.
- Implement new data connectors or execution interfaces as needed.

## License
MIT License

## Job & Ops API

### Launch a back-test job
```bash
curl -X POST http://localhost:8000/jobs/train -H 'Content-Type: application/json' -d '{"type": "train", "params": {"symbol": "AAPL", "epochs": 10}}'
```

### Launch a forward-test job
```bash
curl -X POST http://localhost:8000/jobs/forward -H 'Content-Type: application/json' -d '{"type": "forward", "params": {"symbol": "AAPL", "window": 5}}'
```

### Launch a live trading job
```bash
curl -X POST http://localhost:8000/jobs/live -H 'Content-Type: application/json' -d '{"type": "live", "params": {"symbol": "AAPL", "capital": 10000}}'
```

### Pause, resume, or cancel a job
```bash
curl -X PATCH http://localhost:8000/jobs/<job_id> -H 'Content-Type: application/json' -d '{"action": "pause"}'
curl -X PATCH http://localhost:8000/jobs/<job_id> -H 'Content-Type: application/json' -d '{"action": "resume"}'
curl -X PATCH http://localhost:8000/jobs/<job_id> -H 'Content-Type: application/json' -d '{"action": "cancel"}'
```

### Get job metadata and progress
```bash
curl http://localhost:8000/jobs/<job_id>
```

### Stream job progress (SSE)
```bash
curl http://localhost:8000/jobs/<job_id>/stream
```

### Get mock account info
```bash
curl http://localhost:8000/account
```

### Health check
```bash
curl http://localhost:8000/health
```
