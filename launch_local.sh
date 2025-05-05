#!/bin/bash
# Activate conda and environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finrl310
export PATH="/home/clearpeakegroup/miniconda3/envs/finrl310/bin:$PATH"
cd /home/clearpeakegroup/finrl-platform
bash scripts/update_and_launch.sh 2>&1 | tee launch_local.log

echo
echo "Press ENTER to close this window..."
read 