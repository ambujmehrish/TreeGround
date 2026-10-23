#!/bin/bash
# Copies the small result files (metrics.json of the latest completed run per stage, the deviation log, the chosen
# layer and node budget, and a text summary) into results/ and pushes them on the current branch.
# Machine paths are replaced by $TG_RUNS / $TG_CACHE / $TG_DATA. Large per-chain CSVs and caches are not copied.
# usage (repo root, after sourcing tg_env.sh): bash scripts/publish_results.sh [message]
set -euo pipefail
cd "${TG_REPO:?source tg_env.sh first}"
: "${TG_RUNS:?}" "${TG_CACHE:?}" "${TG_DATA:?}"
out=results
mkdir -p "$out/metrics"
scrub() { sed -e "s#${TG_RUNS}#\$TG_RUNS#g" -e "s#${TG_CACHE}#\$TG_CACHE#g" -e "s#${TG_DATA}#\$TG_DATA#g"; }
for stage_dir in "$TG_RUNS"/*/; do
  stage=$(basename "$stage_dir")
  case "$stage" in smoke|logs) continue ;; esac
  last=$(ls -d "$stage_dir"2*/ 2>/dev/null | while read -r d; do [ -f "$d/metrics.json" ] && echo "$d"; done | tail -1)
  [ -n "$last" ] || continue
  scrub < "$last/metrics.json" > "$out/metrics/$stage.json"
  echo "$(basename "$last")" > "$out/metrics/$stage.run"
done
scrub < "$TG_RUNS/deviations.jsonl" > "$out/deviations.jsonl"
for f in configs/chosen_layer.yaml configs/node_cap.yaml; do [ -f "$f" ] && cp "$f" "$out/"; done
python scripts/summarize_results.py 2>&1 | scrub > "$out/summary.txt"
squeue --me -o "%.14i %.28j %.8T %.10M %R" 2>/dev/null | grep -E "JOBID|tg_" | scrub > "$out/queue.txt" || true
git add -f "$out"
if git diff --cached --quiet; then echo "nothing new to publish"; exit 0; fi
git commit -q -m "results: ${1:-update $(date -u +%Y-%m-%dT%H:%MZ)}"
branch=$(git rev-parse --abbrev-ref HEAD)
for i in 1 2 3 4; do
  git pull -q --no-rebase --no-edit origin "$branch" && git push -q origin HEAD && break || sleep $((2 ** i))
done
git log --oneline -1
