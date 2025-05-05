#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="/home/clearpeakegroup/finrl-platform"
COMPOSE_FILE="$PROJECT_DIR/infra/docker-compose.yml"
GRADIO_URL="http://localhost:7860"
CHROME_CMD="/usr/bin/google-chrome"
LOGFILE="$PROJECT_DIR/launcher.log"
STARTUP_WAIT_SECONDS=15 # Allow more time for containers to start

# --- Logging setup ---
echo "\n==== $(date) ====" >> "$LOGFILE"

# --- Ensure Docker is running ---
echo "Checking Docker daemon..." | tee -a "$LOGFILE"
if ! docker info > /dev/null 2>&1; then
    echo "ERROR: Docker is not running. Please start Docker Desktop or the Docker daemon." | tee -a "$LOGFILE"
    read -p "Press Enter to close this window..."
    exit 1
fi

# --- Try to stop and remove containers, handle permissions ---
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd into $PROJECT_DIR" | tee -a "$LOGFILE"; exit 1; }
STOP_CMD="docker compose -f $COMPOSE_FILE down"
UP_CMD="docker compose -f $COMPOSE_FILE up -d --remove-orphans --wait"
BUILD_CMD="docker compose -f $COMPOSE_FILE build --no-cache"

try_docker_cmd() {
    CMD="$1"
    echo "Running: $CMD" | tee -a "$LOGFILE"
    if ! eval $CMD; then
        echo "Permission denied or error. Retrying with sudo..." | tee -a "$LOGFILE"
        if ! sudo $CMD; then
            echo "Still failed with sudo. Attempting to fix docker group..." | tee -a "$LOGFILE"
            sudo usermod -aG docker $USER
            echo "Added $USER to docker group. Please log out and log back in, then re-run this script." | tee -a "$LOGFILE"
            read -p "Press Enter to close this window..."
            exit 1
        fi
    fi
}

try_docker_cmd "$STOP_CMD"
try_docker_cmd "$BUILD_CMD"
try_docker_cmd "$UP_CMD"

# --- Clean up stuck containers if needed ---
if docker ps -a | grep -q 'Exited'; then
    echo "Cleaning up exited containers..." | tee -a "$LOGFILE"
    docker ps -a | grep 'Exited' | awk '{print $1}' | xargs -r sudo docker rm -f
fi

# --- Wait for services to become healthy ---
echo "Waiting $STARTUP_WAIT_SECONDS seconds for services to become healthy..." | tee -a "$LOGFILE"
sleep $STARTUP_WAIT_SECONDS

echo "Checking container health..." | tee -a "$LOGFILE"
docker compose -f "$COMPOSE_FILE" ps | tee -a "$LOGFILE"

# --- Open Gradio UI in browser ---
echo "Opening $GRADIO_URL in $CHROME_CMD..." | tee -a "$LOGFILE"
$CHROME_CMD "$GRADIO_URL" &

echo "All done. FinRL pipeline platform is running. Press Enter to close this window..." | tee -a "$LOGFILE"
read -p "Press Enter to close this window..." 