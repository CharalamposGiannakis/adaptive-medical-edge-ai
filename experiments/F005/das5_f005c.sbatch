#!/bin/bash
# One F005c (T011) step on DAS-5; arguments go to run_f005c.py. Submit from the
# repository root. CPU-only: submit with --gres=gpu:0 --exclusive. Create
# artifacts/F005/logs before sbatch: Slurm opens the output first.
#SBATCH --job-name=f005c
#SBATCH --partition=defq
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=02:00:00
#SBATCH --output=artifacts/F005/logs/%x_%j.out

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
python experiments/F005/run_f005c.py "$@"
