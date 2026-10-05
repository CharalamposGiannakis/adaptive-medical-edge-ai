#!/bin/bash
# Submit from the repository root after preparing the weight cache; see README.
#SBATCH --job-name=f002c
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/F002c/logs/%x_%j.out

set -euo pipefail
if ! command -v module >/dev/null; then
  export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles:/cm/shared/modulefiles}
  set +u; source /opt/ohpc/admin/lmod/lmod/init/bash; set -u
fi
module purge
module load cuda12.6/toolkit/12.6
# Reuse an existing environment. This script does not install or create one.
F002C_PYTHON=${F002C_PYTHON:-/var/scratch/$USER/miniconda3/envs/myenv/bin/python}
test -x "$F002C_PYTHON"
export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME=/var/scratch/$USER/cache/f002c/torch
export HF_HOME=/var/scratch/$USER/cache/f002c/huggingface
export HF_HUB_OFFLINE=1
export OMP_NUM_THREADS=2
export MKL_NUM_THREADS=2
export PYTHONUNBUFFERED=1

hostname
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
"$F002C_PYTHON" -m unittest discover -s tests -p 'test_f002*.py'
"$F002C_PYTHON" experiments/F002c/run_f002c.py "${@:-train}"
