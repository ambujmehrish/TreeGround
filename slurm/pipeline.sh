#!/bin/bash
# Submits the full pilot as one SLURM dependency graph. Run from the repo root after: source $TG_ROOT/tg_env.sh
#   wave 1: proposals (256) || queries (64)           wave 2: proposals merge -> V1 gate (CPU) || queries merge
#   wave 3: features check -> features (128) -> merge  wave 4: V3 -> V2 -> V4
# V1 exits non-zero when the gate fails, so every later stage stays pending (DependencyNeverSatisfied).
# Cached shards are skipped, so re-running this script after a failure only redoes missing work.
# Optional: TG_MAX_PARALLEL (array throttle, default 64), TG_CPU_PARTITION (default lrd_all_serial),
#           TG_PROPOSALS_JOB / TG_QUERIES_JOB (reuse a queued or running array),
#           TG_QUERIES_MERGE_JOB (reuse a queued queries merge; skips the queries array and merge, or "done" if merged)
set -euo pipefail
cd "${TG_REPO:?source tg_env.sh first}"
mkdir -p logs
N="${TG_MAX_PARALLEL:-64}"
CPU=(-p "${TG_CPU_PARTITION:-lrd_all_serial}" --cpus-per-task=4 --mem=30G)
sub() { sbatch --parsable "$@" | cut -d';' -f1; }
shards() { python -c "from treeground.run import load_config; print(load_config('configs/s0.yaml')['$1']['num_shards'])"; }
NP=$(shards proposals); NF=$(shards features); NQ=$(shards queries)

P="${TG_PROPOSALS_JOB:-$(sub --array=0-$((NP - 1))%$N slurm/s0_proposals.sbatch)}"
PM=$(sub "${CPU[@]}" -J tg_proposals_merge -d afterok:$P slurm/cpu_stage.sbatch scripts/s0_proposals.py configs/s0.yaml --merge)
if [ -n "${TG_QUERIES_MERGE_JOB:-}" ]; then
    Q=reused; QM="$TG_QUERIES_MERGE_JOB"
else
    Q="${TG_QUERIES_JOB:-$(sub --array=0-$((NQ - 1))%$N slurm/s0_queries.sbatch)}"
    QM=$(sub "${CPU[@]}" -J tg_queries_merge -d afterok:$Q slurm/cpu_stage.sbatch scripts/s0_queries.py configs/s0.yaml --merge)
fi
V2DEP="afterok:$V3"; [ "$QM" = done ] || V2DEP="$V2DEP:$QM"
V1=$(sub "${CPU[@]}" -J tg_v1 -d afterok:$PM slurm/cpu_stage.sbatch scripts/v1_coverage.py configs/v1.yaml)
FC=$(sub -J tg_features_check -d afterok:$V1 slurm/gpu_stage.sbatch scripts/s0_features.py configs/s0.yaml --check)
F=$(sub --array=0-$((NF - 1))%$N -d afterok:$FC slurm/s0_features.sbatch)
FM=$(sub "${CPU[@]}" -J tg_features_merge -d afterok:$F slurm/cpu_stage.sbatch scripts/s0_features.py configs/s0.yaml --merge)
V3=$(sub -J tg_v3 -d afterok:$FM slurm/gpu_stage.sbatch scripts/v3_layer.py configs/v3.yaml)
V2=$(sub -J tg_v2 --time=08:00:00 -d $V2DEP slurm/gpu_stage.sbatch scripts/v2_train_eval.py configs/v2.yaml)
V4=$(sub -J tg_v4 --time=08:00:00 -d afterok:$V2 slurm/gpu_stage.sbatch scripts/v4_tree_value.py configs/v4.yaml)

printf '%-18s %s\n' proposals "$P" queries "$Q" proposals_merge "$PM" queries_merge "$QM" v1_gate "$V1" \
    features_check "$FC" features "$F" features_merge "$FM" v3 "$V3" v2 "$V2" v4 "$V4" | tee logs/pipeline_jobs.txt
