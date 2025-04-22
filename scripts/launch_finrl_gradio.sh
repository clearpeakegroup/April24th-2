#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="/home/clearpeakegroup/finrl-platform"
GRADIO_URL="http://localhost:7860"
PYTHON_CMD="/home/clearpeakegroup/miniconda3/bin/python"
CHROME_CMD="/usr/bin/google-chrome"
LOGFILE="$PROJECT_DIR/launcher.log"
STARTUP_WAIT_SECONDS=8 # Adjust if your app takes longer/shorter to start

# --- Check for lsof ---
if ! command -v lsof &> /dev/null; then
    echo "ERROR: lsof is not installed. Please run: sudo apt install lsof" | tee -a "$LOGFILE"
    read -p "Press Enter to close this window..."
    exit 1
fi
echo "lsof check passed" | tee -a "$LOGFILE"

# --- Logging setup ---
echo "\n==== $(date) ====" >> "$LOGFILE"

# --- Kill any previous Gradio process on port 7860 ---
echo "Checking for existing process on port 7860..." | tee -a "$LOGFILE"
PID=$(lsof -ti:7860 || true)
if [ -n "$PID" ]; then
    echo "Killing process $PID using port 7860..." | tee -a "$LOGFILE"
    kill -9 $PID
    sleep 1
fi
echo "Port kill block passed" | tee -a "$LOGFILE"
# Check again
PID2=$(lsof -ti:7860 || true)
if [ -n "$PID2" ]; then
    echo "ERROR: Port 7860 is still in use by process $PID2 after kill attempt!" | tee -a "$LOGFILE"
    echo "Process list for port 7860:" | tee -a "$LOGFILE"
    lsof -i :7860 | tee -a "$LOGFILE"
    echo "\nFull python process list:" | tee -a "$LOGFILE"
    ps aux | grep python | tee -a "$LOGFILE"
    read -p "Port 7860 still busy. Press Enter to close this window..."
    exit 1
fi
echo "Port check complete, proceeding..." | tee -a "$LOGFILE"

# --- Logic ---
echo "Navigating to project directory: $PROJECT_DIR" | tee -a "$LOGFILE"
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd into $PROJECT_DIR" | tee -a "$LOGFILE"; exit 1; }
echo "cd passed" | tee -a "$LOGFILE"

echo "About to run Gradio in the background" | tee -a "$LOGFILE"
$PYTHON_CMD -m frontend.gradio_app --headless 2>&1 | tee -a "$LOGFILE" &
GRADIO_PID=$!
echo "Gradio started with PID $GRADIO_PID. Waiting $STARTUP_WAIT_SECONDS seconds..." | tee -a "$LOGFILE"
sleep $STARTUP_WAIT_SECONDS

echo "Opening $GRADIO_URL in $CHROME_CMD..." | tee -a "$LOGFILE"
$CHROME_CMD "$GRADIO_URL" &

echo "All done. Gradio is running in the background. Press Enter to close this window..." | tee -a "$LOGFILE"
read -p "Press Enter to close this window..." 