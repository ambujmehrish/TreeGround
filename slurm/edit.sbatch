#!/bin/bash
#SBATCH --job-name=tg_edit
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-15 -J tg_e_qwen_paco slurm/edit.sbatch qwen_image_edit paco_edit [config] [--smoke]
#        (array size = edit.num_shards[dataset]; --smoke runs a single shard: --array=0)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 DIFFUSERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/e_edit.py --config "${3:-configs/edit.yaml}" --editor "$1" --dataset "$2" --shard "$SLURM_ARRAY_TASK_ID" "${@:4}"
