#!/bin/bash

PY_VER=$(python -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
if [ "$PY_VER" != "3.10" ]; then
    echo "ERROR: Python 3.10 is required. Current version: $PY_VER"
    exit 1
fi

#!/usr/bin/env bash
set -euo pipefail
python3 -m pytest --maxfail=5 --disable-warnings -q "$@" 