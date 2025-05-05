#!/bin/bash
# Assume conda env finrl310 is already activated
# Remove venv-related logic or references
set -e

# --- CONFIG ---
IMAGE_NAME="april24th:prod"
DOCKERFILE_PATH="Dockerfile"
WHEELHOUSE="wheelhouse"
REQ_FILE="requirements.txt"
MAX_ATTEMPTS=1000

# --- 1. Overwrite Dockerfile with correct build instructions ---
cat > "$DOCKERFILE_PATH" <<'DOCK'
FROM nvidia/cuda:11.8.0-runtime-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

# --- Python 3.10, pip, header files, and build tools -------------
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.10 python3.10-venv python3-pip python3.10-dev \
        build-essential swig git libffi-dev && \
    ln -s /usr/bin/python3 /usr/bin/python && \
    python3 -m pip install --upgrade "pip==23.3.1" && \
    rm -rf /var/lib/apt/lists/*

# --- locked dependency set ---------------------------------------------
COPY requirements.txt /tmp/
RUN python3 -m pip install -r /tmp/requirements.txt \
        -f https://download.pytorch.org/whl/torch_stable.html

# (optional) COPY your strategy code here
# COPY . /app
# WORKDIR /app
# CMD ["python3","run_avalanche.py"]
DOCK

echo "✅ Dockerfile written."

# 1. Batch repair loop: find all missing wheels before building
attempt=1
while (( attempt <= MAX_ATTEMPTS )); do
    echo "\n==============================="
    echo "🔎 Checking for missing wheels (dry-run pip install)..."
    echo "pip install --no-index --find-links=$WHEELHOUSE -r $REQ_FILE --dry-run"
    echo "===============================\n"
    set +e
    DRYRUN_OUTPUT=$(pip install --no-index --find-links=$WHEELHOUSE -r $REQ_FILE --dry-run 2>&1)
    set -e

    # Parse all missing wheels in one pass
    MISSING_PKGS=$(echo "$DRYRUN_OUTPUT" | grep -Eo 'No matching distribution found for ([^ ]+)==([0-9\.]+)' | awk '{print $6"=="$7}')
    if [[ -z "$MISSING_PKGS" ]]; then
        echo "✅ All wheels present in $WHEELHOUSE. Proceeding to Docker build."
        break
    fi

    for PKGVER in $MISSING_PKGS; do
        PKG=$(echo $PKGVER | cut -d'=' -f1)
        VER=$(echo $PKGVER | cut -d'=' -f3)
        echo "🔎 Detected missing wheel: $PKG==$VER"
        echo "➡️  Downloading: pip download $PKG==$VER -d $WHEELHOUSE/"
        set +e
        pip download "$PKG==$VER" -d "$WHEELHOUSE/"
        DL_RESULT=$?
        set -e
        if [ $DL_RESULT -ne 0 ]; then
            echo "❌ Failed to download $PKG==$VER. Exiting."
            exit 1
        fi
        echo "✅ Downloaded $PKG==$VER."
    done
    ((attempt++))
done

if (( attempt > MAX_ATTEMPTS )); then
    echo "❌ Reached max attempts ($MAX_ATTEMPTS) for wheel repair. Exiting."
    exit 1
fi

# 2. Build Docker image (should be fast and fully cached)
echo "\n==============================="
echo "🚀 Building Docker image..."
echo "docker build -t $IMAGE_NAME ."
echo "===============================\n"
set +e
BUILD_OUTPUT=$(docker build -t "$IMAGE_NAME" . 2>&1)
RESULT=$?
set -e

if [ $RESULT -eq 0 ]; then
    echo "$BUILD_OUTPUT"
    echo "🎉 Build succeeded!"
    exit 0
else
    echo "$BUILD_OUTPUT" | tee last_build.log
    echo "❌ Build failed. Check last_build.log for details."
    exit 1
fi
