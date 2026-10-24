#!/bin/bash
# Ops queue runner for a Leonardo login node (inside tmux): every INTERVAL seconds it updates the checkout from
# origin (rebase only, never merge) and runs one queue tick (treeground.ops.queue). State lives outside the repo in
# $TG_ROOT/ops_state/state.json, so restarts never re-submit. Stop with Ctrl-c.
# usage: cd $TG_REPO && bash scripts/runner.sh           (INTERVAL=300 by default)
set -uo pipefail
source "${TG_ENV:-$WORK/treeground/tg_env.sh}"
cd "${TG_REPO:?tg_env.sh did not set TG_REPO}"
BRANCH=claude/lucid-bardeen-k2k41b
INTERVAL=${INTERVAL:-300}
export TG_LOGS=${TG_LOGS:-$TG_REPO/logs}
export SBATCH_EXCLUDE=${SBATCH_EXCLUDE:-lrdn0010}
STATE=$TG_ROOT/ops_state/state.json
LOG=$TG_LOGS/ops_runner.log
mkdir -p "$TG_LOGS" "$(dirname "$STATE")"
cur=$(git rev-parse --abbrev-ref HEAD)
[ "$cur" = "$BRANCH" ] || { echo "checkout is on $cur, expected $BRANCH" >&2; exit 1; }
echo "runner: repo $TG_REPO branch $BRANCH state $STATE every ${INTERVAL}s (log $LOG)"
while true; do
  if git fetch -q origin "$BRANCH" && git rebase -q FETCH_HEAD; then
    python -m treeground.ops.queue tick --repo "$TG_REPO" --state "$STATE" >> "$LOG" 2>&1
    tail -n 3 "$LOG"
  else
    git rebase --abort 2>/dev/null
    echo "$(date '+%F %T') git update failed; tick skipped" | tee -a "$LOG"
  fi
  sleep "$INTERVAL"
done
