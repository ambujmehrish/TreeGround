#!/bin/bash
#SBATCH --job-name=tg_s0_queries
#SBATCH --array=0-63%16
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# Submit from the repo root after: source $TG_ROOT/tg_env.sh. Array size must equal queries.num_shards in configs/s0.yaml
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/s0_queries.py --config configs/s0.yaml --shard "$SLURM_ARRAY_TASK_ID"
