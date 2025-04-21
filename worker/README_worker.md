# Worker Service

## Running the Worker

Build and run the worker with Docker Compose:

```bash
docker compose build worker
docker compose run --rm worker pytest  # Run tests
docker compose run --rm worker python -m worker.runner  # Start worker
```

## Enqueue a Job (example)

You can enqueue a job using Python or redis-cli:

```python
import redis
import json
job = {"id": "testid123", "type": "train", "params": {"foo": "bar"}}
r = redis.Redis.from_url("redis://localhost:6379/0")
r.rpush("jobs", json.dumps(job))
```

## Control a Job (pause/cancel)

Set a control key in Redis:

```bash
redis-cli set jobctl:testid123 pause
redis-cli set jobctl:testid123 cancel
redis-cli del jobctl:testid123  # Resume if paused
```

## Progress Streaming

Listen to progress updates:

```bash
redis-cli subscribe progress:testid123
``` 