#!/bin/bash
#SBATCH --job-name=tg_baselines
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-31 -J tg_b_sam3_text slurm/baselines.sbatch sam3_text [config]   (array size = baselines.num_shards)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/b_baselines.py --config "${2:-configs/baselines.yaml}" --method "$1" --shard "$SLURM_ARRAY_TASK_ID"
