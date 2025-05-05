#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"

trap 'echo "🛑  Shutting down April24th containers…"; docker compose down' EXIT

echo "🚀  Bringing April24th 3.0 online…"
docker compose up       # foreground mode → logs visible
