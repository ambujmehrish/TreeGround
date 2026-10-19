#!/bin/bash
#SBATCH --job-name=tg_s0_proposals
#SBATCH --array=0-255%32
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=%x_%A_%a.out
# Array size must equal proposals.num_shards in configs/s0.yaml. Submit after: source slurm/env.sh
set -euo pipefail
: "${TG_ROOT:?}" "${TG_CACHE:?}" "${TG_RUNS:?}" "${TG_VENV:?}"
source "$TG_VENV/bin/activate"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 HF_HOME="$TG_ROOT/hf"
cd "$SLURM_SUBMIT_DIR"
python scripts/s0_proposals.py --config configs/s0.yaml --shard "$SLURM_ARRAY_TASK_ID"
