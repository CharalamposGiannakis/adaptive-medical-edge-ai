#!/bin/bash
# One F003 (T009) step on DAS-5; arguments go to run_f003.py. Submit from the
# repository root. Timing jobs add `-w node201 --exclusive` (see README).
#SBATCH --job-name=f003
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/F003/logs/%x_%j.out

set -euo pipefail
if ! command -v module >/dev/null; then  # profile.d/lmod.sh is a no-op inside SLURM
  export MODULEPATH=${MODULEPATH:-/opt/ohpc/pub/modulefiles:/cm/shared/modulefiles}
  set +u; source /opt/ohpc/admin/lmod/lmod/init/bash; set -u
fi
module purge
module load cuda12.6/toolkit/12.6
source /var/scratch/$USER/miniconda3/bin/activate /var/scratch/$USER/miniconda3/envs/myenv

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export HF_HUB_OFFLINE=1
export PYTHONUNBUFFERED=1

hostname
if [ -n "${CUDA_VISIBLE_DEVICES:-}" ]; then
  nvidia-smi --query-gpu=name,driver_version --format=csv,noheader -i "${CUDA_VISIBLE_DEVICES%%,*}"
else
  echo "no GPU allocated (CPU-only step)"
fi
python experiments/F003/run_f003.py "$@"
