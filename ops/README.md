# Ops queue

A git-driven Slurm queue: requests are committed to `ops/queue/<id>.json`; a runner on a Leonardo login node
(`scripts/runner.sh`, inside tmux) pulls the branch every 5 minutes, submits each valid request once, watches the
jobs, runs the request's post step when they finish, and pushes results back through `scripts/collect_results.sh`.
Status of the queue and of all your jobs: `results/ops/status.json` (pushed at least hourly).

## Start / stop (Leonardo login node)

    tmux new -s ops
    source $WORK/treeground/tg_env.sh && cd $TG_REPO && bash scripts/runner.sh
    # Ctrl-b d to detach; `tmux attach -t ops` to look; Ctrl-c inside it to stop

State: `$TG_ROOT/ops_state/state.json` (outside the repo, so restarts never re-submit). Log:
`$TG_REPO/logs/ops_runner.log`. `INTERVAL=600 bash scripts/runner.sh` changes the period. One runner at a time
(a second one finds the lock and idles).

## Request format

Submit (`id` = file stem):

    {"id": "2026-10-24-rerank-s2", "what": "top-k rerank of S2", "script": "q_gpu.sbatch",
     "env": {"TG_PY": "scripts/rerank.py", "TG_ARGS": "--config configs/rerank.yaml --stage s2"},
     "sbatch": {"time": "03:00:00", "array": "0-7%4"}, "after": ["<other request id>"],
     "post": {"module": "treeground.analysis.tables", "args": []}, "collect": true}

Watch-only (jobs submitted by hand): `{"id", "what", "after_jobs": ["62101091"], "post": {...}, "collect": true}`.

- `script`: a file in `slurm/` (`*.sbatch`). The generic wrappers run `TG_PY` (a `scripts/*.py`) with `TG_ARGS`:
  `q_gpu.sbatch` (1 GPU; array tasks append `--shard <i>`) and `q_gpu4.sbatch` (4 GPUs, torchrun).
- `env` values and `post.args` may use `${TG_ROOT}`, `${TG_CACHE}`, `${TG_DATA}`, `${TG_LOGS}`, `${TG_RUNS}`; no
  shell metacharacters. `sbatch` takes only `time`, `qos`, `array`; the account is fixed in code.
- `after`: other request ids (afterok dependency). `post.module`: `treeground.analysis.*` or `treeground.ops.*`.

Check before committing: `python -m treeground.ops.queue validate --repo .`
