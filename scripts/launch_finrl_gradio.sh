#!/bin/bash
set -e

# --- Configuration ---
PROJECT_DIR="/home/clearpeakegroup/finrl-platform"
GRADIO_URL="http://localhost:7860"
PYTHON_CMD="/home/clearpeakegroup/miniconda3/bin/python"
CHROME_CMD="/usr/bin/google-chrome"
STARTUP_WAIT_SECONDS=8 # Adjust if your app takes longer/shorter to start

# --- Kill any previous Gradio process on port 7860 ---
echo "Checking for existing process on port 7860..."
PID=$(lsof -ti:7860)
if [ -n "$PID" ]; then
    echo "Killing process $PID using port 7860..."
    kill -9 $PID
fi

# --- Logic ---
echo "Navigating to project directory: $PROJECT_DIR"
cd "$PROJECT_DIR" || { echo "ERROR: Failed to cd into $PROJECT_DIR"; exit 1; }

echo "Starting Gradio app in the foreground..."
$PYTHON_CMD -m frontend.gradio_app --headless

# If the app starts successfully, open Chrome
if [ $? -eq 0 ]; then
    echo "Opening $GRADIO_URL in $CHROME_CMD..."
    $CHROME_CMD "$GRADIO_URL"
else
    echo "Gradio app exited with an error."
fi

read -p "Press Enter to close this window..." 