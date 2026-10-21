#!/bin/bash
#SBATCH --job-name=tg_cpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=logs/%x-%j.out
# usage: sbatch -p lrd_all_serial --cpus-per-task=4 --mem=30G slurm/cpu_stage.sbatch scripts/v1_coverage.py configs/v1.yaml
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1
cd "$TG_REPO"
python "$1" --config "$2" "${@:3}"
