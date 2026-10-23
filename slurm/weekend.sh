#!/bin/bash
# Submits every remaining track with dependencies: baselines (D13), S2 from V2 and from S1 (D10), editing with the
# V2 and the S1 selector (D14). S1 caches/training, V4 and transfer are expected to be queued already.
# usage: bash slurm/weekend.sh [--no-edit] [--no-baselines]
set -euo pipefail
cd "$TG_REPO"
EDIT=1; BASE=1
for a in "$@"; do
  case "$a" in --no-edit) EDIT=0 ;; --no-baselines) BASE=0 ;; *) echo "unknown option $a"; exit 1 ;; esac
done
need=(D10 D11)
[ $BASE = 1 ] && need+=(D13)
[ $EDIT = 1 ] && need+=(D14)
for d in "${need[@]}"; do
  grep -q "\"id\": \"$d\".*\"status\": \"approved\"" "$TG_RUNS/deviations.jsonl" || { echo "$d is not approved"; exit 1; }
done
mkdir -p logs
sub() { sbatch --parsable "$@"; }
S1T=$(squeue -u "$USER" -h -n tg_s1_train -o %i | head -1)
dep_s1() { [ -n "$S1T" ] && echo "afterok:$S1T" || echo "singleton"; }

if [ $BASE = 1 ]; then
  for m in sam3_text qwen_box_sam sa2va; do
    S=$(sub -J tg_bs_$m --time=01:00:00 slurm/gpu_stage.sbatch scripts/b_baselines.py configs/baselines.yaml --method $m --smoke)
    SM=$(sub -J tg_bsm_$m --time=00:30:00 -d afterok:$S slurm/gpu_stage.sbatch scripts/b_baselines.py configs/baselines.yaml --method $m --smoke --merge)
    A=$(sub --array=0-31 -J tg_b_$m -d afterok:$SM slurm/baselines.sbatch $m)
    M=$(sub -J tg_b_${m}_merge -d afterok:$A slurm/gpu_stage.sbatch scripts/b_baselines.py configs/baselines.yaml --method $m --merge)
    echo "baseline $m: smoke $S/$SM array $A merge $M"
  done
fi

S2S=$(sub -J tg_s2_smoke --time=01:00:00 slurm/s2_train.sbatch scripts/s2_train.py --smoke)
S2F=$(sub -J tg_s2_train -d afterok:$S2S slurm/s2_train.sbatch scripts/s2_train.py)
S2E=$(sub -J tg_s2_eval -d afterok:$S2F slurm/s2_train.sbatch scripts/s2_eval.py)
echo "S2 from V2: smoke $S2S train $S2F eval $S2E"
S2G=$(TG_S2_CONFIG=configs/s2_s1.yaml sub -J tg_s2s1_train -d "afterok:$S2S,$(dep_s1)" slurm/s2_train.sbatch scripts/s2_train.py)
S2H=$(TG_S2_CONFIG=configs/s2_s1.yaml sub -J tg_s2s1_eval -d afterok:$S2G slurm/s2_train.sbatch scripts/s2_eval.py)
echo "S2 from S1: train $S2G eval $S2H (after S1 training ${S1T:-already finished})"

if [ $EDIT = 1 ]; then
  declare -A N=([partedit]=3 [gedit]=7 [imgedit]=7 [paco_edit]=15)
  SQ=$(sub --array=0 -J tg_e_smoke_q slurm/edit.sbatch qwen_image_edit partedit configs/edit.yaml --smoke)
  SF=$(sub --array=0 -J tg_e_smoke_f -d afterok:$SQ slurm/edit.sbatch flux_fill partedit configs/edit.yaml --smoke)
  echo "edit smoke: qwen $SQ fill $SF"
  for cfgname in edit edit_s1; do
    dep="afterok:$SF"
    tag=""
    if [ $cfgname = edit_s1 ]; then dep="afterok:$SF,$(dep_s1)"; tag="1"; fi
    for ds in partedit gedit imgedit paco_edit; do
      Q=$(sub --array=0-${N[$ds]} -J tg_e${tag}_q_$ds -d "$dep" slurm/edit.sbatch qwen_image_edit $ds configs/$cfgname.yaml)
      F=$(sub --array=0-${N[$ds]} -J tg_e${tag}_f_$ds -d afterany:$Q slurm/edit.sbatch flux_fill $ds configs/$cfgname.yaml)
      sub -J tg_e${tag}_qm_$ds -d afterany:$Q slurm/gpu_stage.sbatch scripts/e_edit.py configs/$cfgname.yaml --editor qwen_image_edit --dataset $ds --merge >/dev/null
      sub -J tg_e${tag}_fm_$ds -d afterany:$F slurm/gpu_stage.sbatch scripts/e_edit.py configs/$cfgname.yaml --editor flux_fill --dataset $ds --merge >/dev/null
      echo "edit $cfgname $ds: qwen $Q fill $F"
    done
  done
fi
