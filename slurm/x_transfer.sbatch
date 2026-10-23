#!/bin/bash
#SBATCH --job-name=tg_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-63%64 -J tg_x_pascal slurm/x_transfer.sbatch pascal_part [config]   (array size = transfer.num_shards)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/x_transfer.py --config "${2:-configs/x_transfer.yaml}" --dataset "$1" --shard "$SLURM_ARRAY_TASK_ID"
