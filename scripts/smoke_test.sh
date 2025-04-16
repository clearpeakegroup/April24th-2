#!/bin/bash
set -e

# Bring up Docker Compose
cd "$(dirname "$0")/.."
echo "[SMOKE TEST] Starting Docker Compose..."
docker-compose -f infra/docker-compose.yml up -d
sleep 10

echo "[SMOKE TEST] Running minimal training job..."
curl -X POST "http://localhost:8000/train" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"instruments": ["MES"], "timesteps": 1000, "algo": "ppo"}'

echo "[SMOKE TEST] Running backtest..."
curl -X POST "http://localhost:8000/backtest" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"model": "ppo_final"}'

echo "[SMOKE TEST] Running forward test..."
curl -X POST "http://localhost:8000/forwardtest" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{}'

echo "[SMOKE TEST] Running live mode dry-run..."
curl -X POST "http://localhost:8000/live" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"dry_run": true}'

echo "[SMOKE TEST] Verifying Gradio UI..."
if curl -s "http://localhost:7860" | grep -q "FinRL Platform Control"; then
    echo "[SMOKE TEST] Gradio UI is responsive."
else
    echo "[SMOKE TEST] Gradio UI is NOT responsive!"; exit 1
fi

echo "[SMOKE TEST] Verifying API health..."
if curl -s "http://localhost:8000/docs" | grep -q "FinRL API Gateway"; then
    echo "[SMOKE TEST] API is responsive."
else
    echo "[SMOKE TEST] API is NOT responsive!"; exit 1
fi

echo "[SMOKE TEST] All checks passed." 