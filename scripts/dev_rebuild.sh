#!/bin/bash
set -e

# Change to project root (the parent of scripts/)
cd "$(dirname "$0")/.."

if [[ "$1" == "--full" ]]; then
  echo "Doing a full rebuild (no cache)..."
  docker compose -f infra/docker-compose.yml down
  docker compose -f infra/docker-compose.yml rm -f
  docker image prune -f
  docker compose -f infra/docker-compose.yml build --no-cache
else
  echo "Doing a fast rebuild (using cache)..."
  docker compose -f infra/docker-compose.yml build
fi

docker compose -f infra/docker-compose.yml up -d gradio

echo "Gradio is rebuilding and starting. Visit http://localhost:7860" 