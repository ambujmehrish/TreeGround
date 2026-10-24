#!/bin/bash
#SBATCH --job-name=tg_s2_nodes
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00
#SBATCH --output=logs/%x-%A_%a.out
# usage: sbatch --array=0-31 slurm/s2_nodes.sbatch pascal_part_train [--smoke]   (array size = s2_nodes.num_shards[source]; config: TG_S2_CONFIG, default configs/s2_long.yaml)
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python scripts/s2_nodes.py --config "${TG_S2_CONFIG:-configs/s2_long.yaml}" --source "$1" --shard "${SLURM_ARRAY_TASK_ID:-0}" "${@:2}"
