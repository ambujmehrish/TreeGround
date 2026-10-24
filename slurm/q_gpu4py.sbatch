#!/bin/bash
#SBATCH --job-name=tg_q4p
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:4
#SBATCH --cpus-per-task=32
#SBATCH --mem=240G
#SBATCH --output=logs/%x-%j.out
# queue wrapper (ops/queue): runs TG_PY (scripts/<name>.py) with TG_ARGS in ONE process on 4 GPUs (device_map models such as the 72B judge); array tasks add --shard <i>
set -euo pipefail
if [ -n "${TG_MODULES:-}" ]; then module purge; module load $TG_MODULES; fi
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
export HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONUNBUFFERED=1 PYTHONUTF8=1 TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=4
cd "$TG_REPO"
[[ "${TG_PY:-}" =~ ^scripts/[a-z0-9_]+\.py$ && -f "$TG_PY" ]] || { echo "TG_PY must be scripts/<name>.py" >&2; exit 2; }
set -f; ARGS=(${TG_ARGS:-}); set +f
[ -n "${SLURM_ARRAY_TASK_ID:-}" ] && ARGS+=(--shard "$SLURM_ARRAY_TASK_ID")
python "$TG_PY" "${ARGS[@]}"
