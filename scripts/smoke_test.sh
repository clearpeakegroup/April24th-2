#!/bin/bash
set -e

# Bring up Docker Compose
cd "$(dirname "$0")/.."
echo "[SMOKE TEST] Starting Docker Compose..."
docker-compose -f infra/docker-compose.yml up -d
sleep 10

# Helper: poll for task status
poll_task() {
  local TASK_ID=$1
  local DESC=$2
  local TIMEOUT=60
  local INTERVAL=2
  local ELAPSED=0
  while [ $ELAPSED -lt $TIMEOUT ]; do
    STATUS_JSON=$(curl -s -H "Authorization: Bearer testtoken" "http://localhost:8000/task_status/$TASK_ID")
    STATUS=$(echo "$STATUS_JSON" | grep -o '"status": *"[^"]*"' | cut -d '"' -f4)
    if [ "$STATUS" == "SUCCESS" ]; then
      echo "[$DESC] SUCCESS: $STATUS_JSON"
      return 0
    elif [ "$STATUS" == "FAILURE" ]; then
      echo "[$DESC] FAILURE: $STATUS_JSON"; exit 1
    fi
    sleep $INTERVAL
    ELAPSED=$((ELAPSED+INTERVAL))
  done
  echo "[$DESC] TIMEOUT waiting for task $TASK_ID"; exit 1
}

# 1. Upload test .zst file
echo "[SMOKE TEST] Uploading test .zst file..."
echo "ts_event,instrument,side,price,size,exchange,type
1,ABC,buy,100,10,NYSE,trade" > test.csv
zstd test.csv -o test.dbn.zst --quiet
UPLOAD_RESP=$(curl -s -F "file=@test.dbn.zst" -H "Authorization: Bearer testtoken" http://localhost:8000/upload)
echo "[SMOKE TEST] Upload response: $UPLOAD_RESP"
if ! echo "$UPLOAD_RESP" | grep -q '"status": *"uploaded'; then
  echo "[SMOKE TEST] Upload failed!"; exit 1
fi

# 2. Training job
echo "[SMOKE TEST] Running minimal training job..."
TRAIN_RESP=$(curl -s -X POST "http://localhost:8000/train" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"instruments": ["MES"], "timesteps": 1000, "algo": "ppo"}')
TRAIN_TASK=$(echo "$TRAIN_RESP" | grep -o '"task_id": *"[^"]*"' | cut -d '"' -f4)
poll_task "$TRAIN_TASK" "TRAIN"

# 3. Backtest job
echo "[SMOKE TEST] Running backtest..."
BACKTEST_RESP=$(curl -s -X POST "http://localhost:8000/backtest" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"model": "ppo_final"}')
BACKTEST_TASK=$(echo "$BACKTEST_RESP" | grep -o '"task_id": *"[^"]*"' | cut -d '"' -f4)
poll_task "$BACKTEST_TASK" "BACKTEST"

# 4. Forward test job
echo "[SMOKE TEST] Running forward test..."
FORWARD_RESP=$(curl -s -X POST "http://localhost:8000/forwardtest" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{}')
FORWARD_TASK=$(echo "$FORWARD_RESP" | grep -o '"task_id": *"[^"]*"' | cut -d '"' -f4)
poll_task "$FORWARD_TASK" "FORWARDTEST"

# 5. Live job
echo "[SMOKE TEST] Running live mode dry-run..."
LIVE_RESP=$(curl -s -X POST "http://localhost:8000/live" \
    -H "accept: application/json" \
    -H "Authorization: Bearer testtoken" \
    -H "Content-Type: application/json" \
    -d '{"dry_run": true}')
LIVE_TASK=$(echo "$LIVE_RESP" | grep -o '"task_id": *"[^"]*"' | cut -d '"' -f4)
poll_task "$LIVE_TASK" "LIVE"

# 6. Gradio UI check
echo "[SMOKE TEST] Verifying Gradio UI..."
if curl -s "http://localhost:7860" | grep -q "FinRL Platform Control"; then
    echo "[SMOKE TEST] Gradio UI is responsive."
else
    echo "[SMOKE TEST] Gradio UI is NOT responsive!"; exit 1
fi

# 7. API docs check
echo "[SMOKE TEST] Verifying API health..."
if curl -s "http://localhost:8000/docs" | grep -q "FinRL API Gateway"; then
    echo "[SMOKE TEST] API is responsive."
else
    echo "[SMOKE TEST] API is NOT responsive!"; exit 1
fi

echo "[SMOKE TEST] All E2E checks passed." 