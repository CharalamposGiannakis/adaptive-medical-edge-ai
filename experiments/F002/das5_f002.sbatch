#!/bin/bash
# One F002 step on a DAS-5 GTX 1080 Ti node. Arguments go to run_f002.py, e.g.
#   sbatch experiments/F002/das5_f002.sbatch train --candidate resnet18 --seed 42
# Submit from the repository root on DAS-5 (see experiments/F002/README.md).
#SBATCH --job-name=f002
#SBATCH --partition=defq
#SBATCH --constraint=GTX1080Ti
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=artifacts/F002/logs/%x_%j.out

set -euo pipefail
if ! command -v module >/dev/null; then  # profile.d/lmod.sh is a no-op inside SLURM
  export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles:/cm/shared/modulefiles}
  set +u; source /opt/ohpc/admin/lmod/lmod/init/bash; set -u
fi
module purge
module load cuda12.6/toolkit/12.6
source /var/scratch/$USER/miniconda3/bin/activate /var/scratch/$USER/miniconda3/envs/myenv

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export TORCH_HOME=/var/scratch/$USER/cache/torch
export HF_HOME=/var/scratch/$USER/cache/huggingface
export PYTHONUNBUFFERED=1

hostname
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-}"
python experiments/F002/run_f002.py "$@"
