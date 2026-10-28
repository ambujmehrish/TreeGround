#!/bin/bash
# Collector for the ops queue: stages the latest metrics per run stage (publish_results.sh --stage-only), rsyncs
# $TG_ROOT/results/ (json/txt/md/csv < 20 MB only) into results/, copies the last 400 lines of every log in
# $TG_LOGS changed since the previous collect to results/logs/<name>.txt, commits "Results: <message>" if anything changed,
# then rebases on origin and pushes.
# usage (repo root, after sourcing tg_env.sh): bash scripts/collect_results.sh "<message>"
set -euo pipefail
msg=${1:?usage: collect_results.sh "<message>"}
cd "${TG_REPO:?source tg_env.sh first}"
: "${TG_ROOT:?}"
logs=${TG_LOGS:-$TG_REPO/logs}
branch=$(git rev-parse --abbrev-ref HEAD)
bash scripts/publish_results.sh --stage-only
if [ -d "$TG_ROOT/results" ]; then
  rsync -a --prune-empty-dirs --max-size=20m --include='*/' --include='*.json' --include='*.txt' --include='*.md' \
    --include='*.csv' --exclude='*' "$TG_ROOT/results/" results/
fi
mkdir -p results/logs
# only logs changed since the previous collect (first run: last day); thousands of shard logs made every collect slow
stamp=$TG_ROOT/ops_state/collect.stamp
mkdir -p "$(dirname "$stamp")"
newer=(-mtime -1); [ -f "$stamp" ] && newer=(-newer "$stamp")
touch "$stamp.next"
if [ -d "$logs" ]; then
  find "$logs" -maxdepth 1 -type f "${newer[@]}" -print0 | while IFS= read -r -d '' f; do
    n=$(basename "$f"); tail -n 400 "$f" > "results/logs/${n%.*}.txt"
  done
fi
mv -f "$stamp.next" "$stamp"
git add results/
if git diff --cached --quiet; then echo "nothing new to collect"; exit 0; fi
git commit -q -m "Results: $msg"
git fetch -q origin "$branch"
if ! git rebase -q FETCH_HEAD; then git rebase --abort; echo "collect: rebase failed" >&2; exit 1; fi
git push -q origin "HEAD:$branch"
git log --oneline -1
