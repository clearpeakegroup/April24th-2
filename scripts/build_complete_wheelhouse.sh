#!/bin/bash
# Assume conda env finrl310 is already activated
# Remove venv creation and activation logic
set -e

REQ_FILE="requirements.txt"
WHEELHOUSE="wheelhouse"
PYTHON_BIN="python3.10"
VENV_DIR=".wheelhouse_venv"

# 1. Upgrade pip, setuptools, wheel
pip install --upgrade pip==23.3.1 setuptools==65.5.1 wheel==0.38.4

# 2. Install all requirements (torch with extra index)
pip install torch==2.0.1+cu118 --extra-index-url https://download.pytorch.org/whl/cu118
pip install -r $REQ_FILE

# 3. Freeze the full set of installed packages
pip freeze > requirements.frozen.txt

# 4. Download all wheels for the frozen set
mkdir -p "$WHEELHOUSE"
pip download -r requirements.frozen.txt -d "$WHEELHOUSE" --only-binary=:all:

# 5. Clean up
rm -rf $VENV_DIR requirements.frozen.txt

echo "Complete wheelhouse built in $WHEELHOUSE/. Ready for fully offline, reproducible builds." 