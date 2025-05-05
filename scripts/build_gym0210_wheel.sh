#!/bin/bash
set -e

GYM_VERSION=0.21.0
WHEELHOUSE="$(dirname "$0")/../wheelhouse"
TMPDIR="/tmp/gym_build_$$"

mkdir -p "$WHEELHOUSE"
rm -rf "$TMPDIR"
mkdir "$TMPDIR"
cd "$TMPDIR"

# Download gym source
pip download gym==${GYM_VERSION} --no-binary=:all:
tar -xzf gym-${GYM_VERSION}.tar.gz
cd gym-${GYM_VERSION}

# Patch setup.py
sed -i 's/opencv-python>=3\./opencv-python>=3.0.0/' setup.py

# Build wheel
pip wheel . -w "$WHEELHOUSE"

cd /
rm -rf "$TMPDIR"
echo "Patched gym==${GYM_VERSION} wheel built and placed in $WHEELHOUSE." 