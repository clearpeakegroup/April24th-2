#!/bin/bash
set -e

REQ_FILE="requirements.txt"
WHEELHOUSE="wheelhouse"

mkdir -p "$WHEELHOUSE"

# Download all available wheels for requirements and dependencies
pip download -r "$REQ_FILE" -d "$WHEELHOUSE" --only-binary=:all:

# Find any .tar.gz or .zip (source) files and build wheels for them
cd "$WHEELHOUSE"
for src in *.tar.gz *.zip 2>/dev/null; do
    if [ -f "$src" ]; then
        echo "Building wheel for $src..."
        pip wheel "$src" -w .
        rm "$src"
    fi
done
cd -

echo "All requirements and dependencies are now cached in $WHEELHOUSE/." 