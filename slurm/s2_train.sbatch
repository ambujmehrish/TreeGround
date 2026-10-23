#!/bin/bash
#SBATCH --job-name=tg_s2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x-%j.out
# usage: sbatch slurm/s2_train.sbatch [scripts/s2_train.py|scripts/s2_eval.py] [--smoke]
# one process per allocated GPU (torchrun); configs/s2.yaml batch_images must be a multiple of GPUs x micro_batch
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$TG_REPO"
NGPU=$(nvidia-smi -L | wc -l)
torchrun --standalone --nproc_per_node="$NGPU" "${1:-scripts/s2_train.py}" --config configs/s2.yaml "${@:2}"
