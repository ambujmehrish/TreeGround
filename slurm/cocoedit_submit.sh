#!/bin/bash
# Submits the CoCoEdit-protocol run (D25) of one config's extra conditions on both benchmarks and both editors:
# mask selection -> edits -> merge -> judge -> judge merge -> evaluation, each step waiting on the one before.
# Usage: slurm/cocoedit_submit.sh <config> <account>     e.g. slurm/cocoedit_submit.sh configs/edit_cocoedit_pool.yaml IscrC_RHYME_0
set -euo pipefail
CFG=${1:?config}; ACC=${2:?account}
cd "${TG_REPO:?}"
[ -f "$CFG" ] || { echo "$CFG missing" >&2; exit 2; }
mkdir -p logs
TAG=$(basename "$CFG" .yaml)
sub() {   # sub <script> <name> <time> <array or ''> <TG_PY> <TG_ARGS> [dependency job ids...]
  local script=$1 name=$2 time=$3 array=$4 py=$5 args=$6; shift 6
  local extra=(); [ -n "$array" ] && extra+=(--array=$array)
  [ $# -gt 0 ] && extra+=(--dependency=afterok:$(IFS=:; echo "$*"))
  TG_PY=$py TG_ARGS="$args" sbatch --parsable --account="$ACC" --job-name="$name" --time="$time" "${extra[@]}" \
    "slurm/$script" | cut -d';' -f1
}
for D in gedit_full imgedit_full; do
  S=$(sub q_gpu.sbatch ${TAG}_sel_$D 04:00:00 0-7 scripts/e_cocoedit.py "--config $CFG --editor qwen_image_edit --dataset $D --select-only")
  for E in qwen_image_edit flux_fill; do
    e=${E%%_*}
    X=$(sub q_gpu.sbatch ${TAG}_edit_${e}_$D 08:00:00 0-7 scripts/e_cocoedit.py "--config $CFG --editor $E --dataset $D" $S)
    M=$(sub q_gpu.sbatch ${TAG}_merge_${e}_$D 01:00:00 "" scripts/e_cocoedit.py "--config $CFG --editor $E --dataset $D --merge" $X)
    J=$(sub q_gpu4py.sbatch ${TAG}_judge_${e}_$D 10:00:00 0-3 scripts/e_cocoedit_judge.py "--config $CFG --editor $E --dataset $D" $M)
    JM=$(sub q_gpu.sbatch ${TAG}_jmerge_${e}_$D 01:00:00 "" scripts/e_cocoedit_judge.py "--config $CFG --editor $E --dataset $D --merge" $J)
    sub q_gpu.sbatch ${TAG}_eval_${e}_$D 04:00:00 "" scripts/e_cocoedit_eval.py "--config $CFG --editor $E --dataset $D" $JM >/dev/null
  done
done
squeue --me -h -o "%j %T %E" | grep "^$TAG" | sort
