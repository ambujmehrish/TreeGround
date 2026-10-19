#!/bin/bash
#SBATCH --job-name=tg_stage
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=480G
#SBATCH --output=%x_%j.out
# usage: sbatch slurm/gpu_stage.sbatch scripts/v3_layer.py configs/v3.yaml [--smoke]
set -euo pipefail
: "${TG_ROOT:?}" "${TG_CACHE:?}" "${TG_RUNS:?}" "${TG_VENV:?}"
source "$TG_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$TG_ROOT/hf"
cd "$SLURM_SUBMIT_DIR"
python "$1" --config "$2" "${@:3}"
