#!/bin/bash
# Assume conda env finrl310 is already activated
# Remove .venv activation logic

set -e

export DATABASE_URL=${DATABASE_URL:-postgresql://user:pass@localhost:5432/trading}
export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
export BROKER_URL=${BROKER_URL:-$REDIS_URL}
export PYTHONPATH=.:$PYTHONPATH

poetry run uvicorn backend.api.main:app --port 8000 &
poetry run celery -A backend.jobs.worker worker -Q default -l info &
poetry run python frontend/gradio_app.py 