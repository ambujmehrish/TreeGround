#!/bin/bash
#SBATCH --job-name=tg_b_transfer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-63 -J tg_bt_sam3_pascal slurm/b_transfer.sbatch sam3_text pascal_part [config] [--smoke]
#        (array size = b_transfer.num_shards; smoke: --array=0)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/b_transfer.py --config "${3:-configs/b_transfer.yaml}" --method "$1" --dataset "$2" \
    --shard "$SLURM_ARRAY_TASK_ID" "${@:4}"
