#!/bin/bash
#SBATCH --job-name=tg_judge
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-3 -J tg_j_q_paco slurm/e_judge.sbatch qwen_image_edit paco_edit [config] [--judge KEY] [--smoke]
#        (array size = judge.num_shards[dataset]; the 72B default judge needs all 4 GPUs of a node,
#         an 8B fast pass: sbatch --gres=gpu:1 --cpus-per-task=8 --mem=120G ... --judge judge_qwen3vl_8b)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/e_judge.py --config "${3:-configs/e_judge.yaml}" --editor "$1" --dataset "$2" --shard "$SLURM_ARRAY_TASK_ID" "${@:4}"
