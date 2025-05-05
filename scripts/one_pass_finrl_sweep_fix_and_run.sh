#!/bin/bash
set -euo pipefail

# 1. Stop any running sweep processes
pkill -f one_pass_finrl_sweep.py || true

# 2. Ensure probe image is python:3.10 (not slim)
sed -i 's/CPU_IMG   = "python:3.10-slim"/CPU_IMG   = "python:3.10"/' scripts/one_pass_finrl_sweep.py || true

# 3. Remove --no-cache-dir from the script
sed -i 's/--no-cache-dir //g' scripts/one_pass_finrl_sweep.py || true

# 4. Ensure pip cache mount and network host in docker run command
CACHE=/mnt/nvme/pip_cache
sed -i '\#/docker","run# s|docker","run|docker","run","-e","PIP_CACHE_DIR=/root/.cache/pip","-v","'$CACHE':/root/.cache/pip","--network","host"|' scripts/one_pass_finrl_sweep.py || true

# 5. Pre-download all torch wheels in the matrix to cache and set permissions
for v in 2.2.0 2.2.1 2.2.2 2.3.0 2.3.1 2.4.0 2.4.1 2.5.0 2.5.1 2.6.0 2.7.0; do
  pip download torch=="$v" -d $CACHE || true
done
chmod -R 777 $CACHE || true

# 6. Wipe sweep state
echo "Wiping sweep state..."
rm -f scripts/attempt_hashes.db logs/*.log logs/full_sweep.out scripts/working_requirements.txt || true

# 7. Set environment variables for cache and workers
export PIP_CACHE_DIR=$CACHE
export TMPDIR=/mnt/nvme/finrl_tmp
export WORKERS=4

# 8. Launch the sweep in the foreground
python3 scripts/one_pass_finrl_sweep.py &
SWEEP_PID=$!

# 9. Tail the progress log
sleep 5
echo "Tailing logs/full_sweep.out (Ctrl-C to stop tailing, sweep will continue in background if you close this terminal)"
tail -f logs/full_sweep.out &

wait $SWEEP_PID 