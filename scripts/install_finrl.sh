#!/bin/bash
set -e
pip install --no-cache-dir -r /tmp/finrl_repair_requirements.txt
python -c 'import finrl; import ccxt; print("✅ FinRL installation validated")'
