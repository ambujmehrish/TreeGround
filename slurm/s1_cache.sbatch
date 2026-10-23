#!/bin/bash
#SBATCH --job-name=tg_s1_cache
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-255%128 -J tg_s1_paco slurm/s1_cache.sbatch paco_train   (array size = s1.num_shards[source])
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/s1_cache.py --config configs/s1.yaml --source "$1" --shard "$SLURM_ARRAY_TASK_ID"
