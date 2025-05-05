#!/bin/bash
set -e

# --- Assume conda env finrl310 is already activated ---

# Remove all venv search/creation logic
# Proceed with the rest of the script as normal

# --- Pull latest code from git ---
# echo "🔄 Pulling latest code from git..."
# git pull --rebase

# --- Install dependencies in current environment ---
echo "🔄 Installing/updating dependencies from local wheelhouse in conda environment..."
pip install --no-index --find-links=wheelhouse -r requirements.txt

# --- Clear Python Cache ---
echo "🧹 Clearing Python cache..."
find backend -type d -name "__pycache__" -exec rm -rf {} + || true
find backend -type f -name "*.pyc" -delete || true

# --- Launch services as before ---
# (Insert the rest of your original script logic here, minus any venv activation)

IMAGE_NAME="april24th:prod"
CONTAINER_NAME="april24th_container"
HOST_PORT_API=8000
HOST_PORT_GRADIO=7860

# --- ENVIRONMENT ---
export DATABASE_URL=${DATABASE_URL:-postgresql://user:pass@localhost:5432/trading}
export REDIS_URL=${REDIS_URL:-redis://localhost:6379}
export BROKER_URL=${BROKER_URL:-$REDIS_URL}
export PYTHONPATH=.:$PYTHONPATH

# --- PROCESS MANAGEMENT ---
PIDS=()
function cleanup {
    echo "\n🛑 Shutting down all April24th services..."
    for pid in "${PIDS[@]}"; do
        if kill -0 $pid 2>/dev/null; then
            kill $pid
        fi
    done
    wait
    echo "✅ All services stopped."
}
trap cleanup SIGINT SIGTERM EXIT

# --- LAUNCH BACKEND API (FastAPI) ---
echo "🚀 Launching FastAPI backend (port $HOST_PORT_API)..."
fuser -k $HOST_PORT_API/tcp || true
sleep 1
# Force no reload and simplify output
uvicorn backend.api:app --host 0.0.0.0 --port $HOST_PORT_API --no-use-colors --reload-dir /tmp &
PIDS+=("$!")
sleep 2

# --- LAUNCH CELERY WORKER ---
echo "🚀 Launching Celery worker..."
# Add -n worker1@%h to provide a unique name
celery -A backend.jobs.tasks worker -Q default -l info -n worker1@%h &
PIDS+=("$!")
sleep 2

# --- LAUNCH GRADIO FRONTEND ---
echo "🚀 Launching Gradio frontend (port $HOST_PORT_GRADIO)..."
python3 frontend/gradio_app.py &
PIDS+=("$!")
sleep 2

# --- TAIL LOGS ---
echo "📋 All services launched. Tailing logs. Press Ctrl+C to stop."
wait 