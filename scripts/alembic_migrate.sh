#!/bin/bash
# Activate the conda environment
source ~/miniconda3/etc/profile.d/conda.sh
conda activate finrl-backend
cd /home/clearpeakegroup/finrl-platform
export PYTHONPATH=.
alembic revision --autogenerate -m "Initial schema" 