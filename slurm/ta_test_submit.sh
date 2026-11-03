#!/bin/bash
# Submits the complete test-split evaluation of one ta_* checkpoint (scripts/ta_test_eval.py): PACO-LVIS test,
# gRefCOCO val/testA/testB, PARTONOMY, OP-HRG x3, each as a shard array followed by its merge. Shards may wait on
# another job (e.g. the training run) with the optional dependency id.
# Usage: slurm/ta_test_submit.sh <variant> <account> [afterok job id]
# e.g.   slurm/ta_test_submit.sh pool_soft_fam IscrC_RHYME_0 63409684
set -euo pipefail
V=${1:?variant}; ACC=${2:?account}; DEP=${3:-}
CFG=configs/ta_test_$V.yaml
[ -f "$CFG" ] || { echo "$CFG missing" >&2; exit 2; }
cd "${TG_REPO:?}"
mkdir -p logs
dep=(); [ -n "$DEP" ] && dep=(--dependency=afterok:$DEP)

sub() {   # sub <name> <time> <array or ''> <TG_ARGS> [extra sbatch args]
  local name=$1 time=$2 array=$3 args=$4; shift 4
  local arr=(); [ -n "$array" ] && arr=(--array=$array)
  TG_PY=scripts/ta_test_eval.py TG_ARGS="$args" \
    sbatch --parsable --account="$ACC" --job-name="$name" --time="$time" "${arr[@]}" "$@" slurm/q_gpu.sbatch | cut -d';' -f1
}

# PACO-LVIS test (8 shards)
J=$(sub tg_tt_paco_$V 12:00:00 0-7 "--bench paco --config $CFG --method ours" "${dep[@]}")
sub tg_tt_paco_${V}_m 01:00:00 "" "--bench paco --config $CFG --method ours --merge" --dependency=afterok:$J >/dev/null
# gRefCOCO: val (threshold), testA, testB (4 shards each), one merge over the three
G=()
for S in val testA testB; do
  G+=($(sub tg_tt_gres_${S}_$V 08:00:00 0-3 "--bench gres --config $CFG --method ours --split $S" "${dep[@]}"))
done
sub tg_tt_gres_${V}_m 02:00:00 "" "--bench gres --config $CFG --method ours --merge" \
  --dependency=afterok:$(IFS=:; echo "${G[*]}") >/dev/null
# PARTONOMY (4 shards)
J=$(sub tg_tt_part_$V 06:00:00 0-3 "--bench partonomy --config $CFG --method ours" "${dep[@]}")
sub tg_tt_part_${V}_m 01:00:00 "" "--bench partonomy --config $CFG --method ours --merge" --dependency=afterok:$J >/dev/null
# OP-HRG: InstructPart, PascalPart-116, PartImageNet (4 shards each, own merges)
for B in instructpart pascalpart partimagenet; do
  J=$(sub tg_tt_oph_${B}_$V 10:00:00 0-3 "--bench ophrg --config $CFG --method ours --benchmark $B" "${dep[@]}")
  sub tg_tt_oph_${B}_${V}_m 01:00:00 "" "--bench ophrg --config $CFG --method ours --benchmark $B --merge" \
    --dependency=afterok:$J >/dev/null
done
squeue --me -h -o "%j %T %E" | grep "_$V" | sort
