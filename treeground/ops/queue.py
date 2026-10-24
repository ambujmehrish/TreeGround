import argparse
import fcntl
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

# Git-driven Slurm queue: requests are committed as ops/queue/<id>.json, a runner on a Leonardo login node calls
# tick() every few minutes, which submits each valid request at most once, polls sacct, runs the post step and
# the collector for finished requests, and writes $TG_ROOT/results/ops/status.json (with a snapshot of ALL the
# user's jobs). The collector pushes results/ to git, so the queue is readable without cluster access.

ACCOUNT = "AIFAC_S07_041"
BRANCH = "claude/lucid-bardeen-k2k41b"
SLURM_DIR = "slurm"
QUEUE_DIR = "ops/queue"
COLLECTOR = "scripts/collect_results.sh"
MAX_NEW = 40
HEARTBEAT_S = 3600
POST_TIMEOUT_S = 1800
CMD_TIMEOUT_S = 300
TERMINAL = {"COMPLETED", "FAILED", "CANCELLED", "TIMEOUT", "OUT_OF_MEMORY", "NODE_FAIL", "PREEMPTED",
            "BOOT_FAIL", "DEADLINE"}

# placeholder directories (tg_env.sh names; TG_LOGS is set by the runner to $TG_REPO/logs)
PLACEHOLDERS = ("TG_ROOT", "TG_CACHE", "TG_DATA", "TG_LOGS", "TG_RUNS")
KEYS = {"id", "what", "script", "env", "sbatch", "after", "after_jobs", "post", "collect"}
ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
SCRIPT_RE = re.compile(r"^[a-z0-9_]+\.sbatch$")
ENV_NAME_RE = re.compile(r"^[A-Z][A-Z0-9_]{0,40}$")
DENIED_ENV = {"PATH", "HOME", "SHELL", "BASH_ENV", "ENV", "IFS", "PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP",
              "PROMPT_COMMAND", "SBATCH_ACCOUNT", "SLURM_CONF", "USER", "LOGNAME"}
DENIED_PREFIX = ("LD_", "DYLD_", "BASH_FUNC", "SLURM_", "PYTHON")
VALUE_RE = re.compile(r"^[A-Za-z0-9_./:+=, %@{}$-]{0,400}$")
PLACEHOLDER_RE = re.compile(r"\$\{(%s)\}|\$(%s)(?![A-Za-z0-9_])" % ("|".join(PLACEHOLDERS), "|".join(PLACEHOLDERS)))
SBATCH_RE = {"time": re.compile(r"^\d{1,2}:\d{2}:\d{2}$"), "qos": re.compile(r"^[a-z_]{1,40}$"),
             "array": re.compile(r"^[0-9,\-]{1,40}(%[0-9]{1,3})?$")}
JOB_RE = re.compile(r"^\d{1,12}$")
POST_RE = re.compile(r"^treeground\.(analysis|ops)\.[a-z0-9_]+$")
# the generic wrappers (slurm/q_*.sbatch) run TG_PY with TG_ARGS; TG_PY must be a script of this repo
TG_PY_RE = re.compile(r"^scripts/[a-z0-9_]+\.py$")


def check_value(field: str, v) -> list[str]:
    if not isinstance(v, str):
        return [f"{field}: must be a string"]
    if not VALUE_RE.match(v):
        return [f"{field}: {v!r} has characters outside [A-Za-z0-9_./:+=, %@{{}}$-] or is longer than 400"]
    rest = PLACEHOLDER_RE.sub("", v)
    if any(c in rest for c in "${}"):
        return [f"{field}: '$', '{{' and '}}' are only allowed as ${{NAME}} or $NAME with NAME in {PLACEHOLDERS}"]
    return []


def expand(v: str, environ) -> str:
    def sub(m):
        name = m.group(1) or m.group(2)
        if not environ.get(name):
            raise ValueError(f"placeholder {name} is not set in the runner environment")
        return environ[name]

    return PLACEHOLDER_RE.sub(sub, v)


def validate(req, stem: str, repo: Path) -> list[str]:
    if not isinstance(req, dict):
        return ["request: must be a JSON object"]
    errs = [f"{k}: unknown key (allowed: {sorted(KEYS)})" for k in req if k not in KEYS]
    rid = req.get("id")
    if not isinstance(rid, str) or not ID_RE.match(rid):
        errs.append("id: must match ^[A-Za-z0-9][A-Za-z0-9_.-]{0,80}$")
    elif rid != stem:
        errs.append(f"id: {rid!r} must equal the file stem {stem!r}")
    if not isinstance(req.get("what"), str) or len(req["what"]) > 200:
        errs.append("what: required string of at most 200 characters")
    if "script" in req:
        s = req["script"]
        if not isinstance(s, str) or not SCRIPT_RE.match(s):
            errs.append(f"script: {s!r} must match ^[a-z0-9_]+\\.sbatch$ (a file name in {SLURM_DIR}/, no paths)")
        elif not (repo / SLURM_DIR / s).is_file():
            errs.append(f"script: {SLURM_DIR}/{s} does not exist")
        if "after_jobs" in req:
            errs.append("after_jobs: only for watch-only requests (no script)")
    else:
        if not req.get("after_jobs"):
            errs.append("script: required unless the request is watch-only (after_jobs)")
        errs += [f"{k}: not allowed in a watch-only request" for k in ("env", "sbatch", "after") if k in req]
    env = req.get("env", {})
    if not isinstance(env, dict):
        errs.append("env: must be an object")
        env = {}
    for k, v in env.items():
        if not ENV_NAME_RE.match(k):
            errs.append(f"env: name {k!r} must match ^[A-Z][A-Z0-9_]{{0,40}}$")
        elif k in DENIED_ENV or k.startswith(DENIED_PREFIX):
            errs.append(f"env: name {k} is denied")
        errs += check_value(f"env.{k}", v)
    if "TG_PY" in env and isinstance(env["TG_PY"], str):
        if not TG_PY_RE.match(env["TG_PY"]):
            errs.append("env.TG_PY: must match ^scripts/[a-z0-9_]+\\.py$")
        elif not (repo / env["TG_PY"]).is_file():
            errs.append(f"env.TG_PY: {env['TG_PY']} does not exist")
    sb = req.get("sbatch", {})
    if not isinstance(sb, dict):
        errs.append("sbatch: must be an object")
        sb = {}
    for k, v in sb.items():
        if k not in SBATCH_RE:
            errs.append(f"sbatch: key {k!r} not allowed (allowed: time, qos, array)")
        elif not isinstance(v, str) or not SBATCH_RE[k].match(v):
            errs.append(f"sbatch.{k}: {v!r} must match {SBATCH_RE[k].pattern}")
    after = req.get("after", [])
    if not isinstance(after, list) or not all(isinstance(a, str) and ID_RE.match(a) for a in after):
        errs.append("after: must be a list of request ids")
    elif rid in after:
        errs.append("after: a request cannot depend on itself")
    aj = req.get("after_jobs", [])
    if not isinstance(aj, list) or not all(isinstance(j, str) and JOB_RE.match(j) for j in aj):
        errs.append("after_jobs: must be a list of numeric job id strings (^\\d{1,12}$)")
    if "post" in req:
        p = req["post"]
        if not isinstance(p, dict) or set(p) - {"module", "args"}:
            errs.append("post: must be an object with keys module and args")
        else:
            if not isinstance(p.get("module"), str) or not POST_RE.match(p["module"]):
                errs.append(f"post.module: must match {POST_RE.pattern}")
            args = p.get("args", [])
            if not isinstance(args, list):
                errs.append("post.args: must be a list")
            else:
                for i, a in enumerate(args):
                    errs += check_value(f"post.args[{i}]", a)
    if "collect" in req and not isinstance(req["collect"], bool):
        errs.append("collect: must be a boolean")
    return errs


def load_requests(repo: Path) -> tuple[dict, dict]:
    valid, invalid = {}, {}
    for f in sorted((repo / QUEUE_DIR).glob("*.json")):
        try:
            req = json.loads(f.read_text())
        except (OSError, ValueError) as e:
            invalid[f.stem] = [f"file: not valid JSON ({str(e)[:200]})"]
            continue
        errs = validate(req, f.stem, repo)
        if errs:
            invalid[f.stem] = errs
        else:
            valid[f.stem] = req
    for rid, req in list(valid.items()):
        bad = [a for a in req.get("after", []) if a not in valid and a not in invalid]
        if bad:
            invalid[rid] = [f"after: unknown request ids {bad}"]
            del valid[rid]
    return valid, invalid


def default_run(cmd, env=None, cwd=None, timeout=CMD_TIMEOUT_S):
    try:
        p = subprocess.run(cmd, env=env, cwd=cwd, capture_output=True, text=True, timeout=timeout)
        return p.returncode, (p.stdout or "") + (p.stderr or "")
    except (OSError, subprocess.TimeoutExpired) as e:
        return 127, f"{type(e).__name__}: {e}"


def save_json(path: Path, obj) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(f"{path}.{os.getpid()}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, sort_keys=True, default=str))
    os.replace(tmp, path)


def parse_jobid(out: str) -> str | None:
    first = out.strip().splitlines()[0].strip() if out.strip() else ""
    jid = first.split(";")[0]
    return jid if JOB_RE.match(jid) else None


def job_ids(entry: dict) -> list[str]:
    return [j for j in str(entry.get("job_id") or "").split(",") if j]


def sacct_states(run, ids: list[str], env, cwd) -> dict[str, list[str]]:
    if not ids:
        return {}
    rc, out = run(["sacct", "-n", "-P", "-X", "-j", ",".join(ids), "--format=JobID,State"], env=env, cwd=cwd)
    if rc != 0:
        raise RuntimeError(out[:300])
    states: dict[str, list[str]] = {}
    for line in out.splitlines():
        parts = line.strip().split("|")
        if len(parts) < 2 or not parts[0] or "." in parts[0]:
            continue
        base = parts[0].split("_")[0]
        state = parts[1].split()[0] if parts[1].split() else ""
        states.setdefault(base, []).append(state)
    return states


def cluster_snapshot(run, env=None, cwd=None) -> dict:
    snap = {"squeue": [], "counts": {}, "ended_24h": []}
    rc, out = run(["squeue", "--me", "-h", "-o", "%i|%j|%T|%M|%l|%D|%R|%S"], env=env, cwd=cwd)
    if rc != 0:
        snap["squeue_error"] = out[:300]
    else:
        for line in out.splitlines():
            f = line.split("|", 7)
            if len(f) < 8:
                continue
            snap["squeue"].append({"jobid": f[0], "name": f[1], "state": f[2], "elapsed": f[3], "time_limit": f[4],
                                   "nodes": f[5], "reason_or_nodelist": f[6], "start": f[7]})
            snap["counts"][f[2]] = snap["counts"].get(f[2], 0) + 1
    rc, out = run(["sacct", "-n", "-P", "-X", "-S", "now-24hours",
                   "--format=JobID,JobName,State,Elapsed,ExitCode,End"], env=env, cwd=cwd)
    if rc != 0:
        snap["sacct_error"] = out[:300]
    else:
        for line in out.splitlines():
            f = line.split("|", 5)
            if len(f) < 6 or "." in f[0]:
                continue
            state = f[2].split()[0] if f[2].split() else ""
            if state in TERMINAL:
                snap["ended_24h"].append({"jobid": f[0], "name": f[1], "state": state, "elapsed": f[3], "exit": f[4],
                                          "end": f[5]})
    return snap


def collect(run, message: str, env, repo: Path) -> str | None:
    rc, out = run(["bash", COLLECTOR, message], env=env, cwd=str(repo))
    return None if rc == 0 else out[-300:]


def status_path(environ) -> Path:
    return Path(environ["TG_ROOT"]) / "results" / "ops" / "status.json"


def tick(repo, state_path, run=default_run, environ=None, now=time.time) -> dict:
    repo, state_path = Path(repo).resolve(), Path(state_path)
    environ = dict(os.environ if environ is None else environ)
    state_path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(f"{state_path}.lock", "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("another runner holds the lock; this tick does nothing")
        return {"locked": True}
    try:
        return _tick(repo, state_path, run, environ, now)
    finally:
        fcntl.flock(lock, fcntl.LOCK_UN)
        lock.close()


def _tick(repo: Path, state_path: Path, run, environ: dict, now) -> dict:
    state = json.loads(state_path.read_text()) if state_path.exists() else {"requests": {}}
    reqs = state.setdefault("requests", {})
    valid, invalid = load_requests(repo)
    for rid, errs in invalid.items():
        reqs.setdefault(rid, {})["invalid"] = "; ".join(errs)
    for rid in valid:
        reqs.setdefault(rid, {}).pop("invalid", None)
    submitted = 0
    for rid, req in valid.items():
        e = reqs[rid]
        if e.get("job_id"):
            continue
        if "script" not in req:
            e.update(job_id=",".join(req["after_jobs"]), submitted=now(), watch_only=True)
            save_json(state_path, state)
            continue
        if submitted >= MAX_NEW:
            continue
        deps = [reqs.get(a, {}).get("job_id") for a in req.get("after", [])]
        missing = [a for a, j in zip(req.get("after", []), deps) if not j]
        if missing:
            e["waiting_for"] = missing
            continue
        e.pop("waiting_for", None)
        # a parent that already finished is resolved here: Slurm forgets finished jobs after MinJobAge and then
        # rejects afterok on them ("Job dependency problem")
        done = {a: reqs[a].get("states") or [] for a in req.get("after", []) if reqs[a].get("finished")}
        failed = [a for a, sts in done.items() if any(x != "COMPLETED" for x in sts)]
        if failed:
            e["dependency_failed"] = failed
            continue
        deps = [j for a, j in zip(req.get("after", []), deps) if a not in done]
        try:
            env = environ | {k: expand(v, environ) for k, v in req.get("env", {}).items()}
        except ValueError as err:
            e["submit_error"] = str(err)[:300]
            continue
        sb = req.get("sbatch", {})
        cmd = ["sbatch", "--parsable", f"--account={ACCOUNT}", f"--job-name=q_{rid}"]
        cmd += [f"--{k}={sb[k]}" for k in ("time", "qos", "array") if k in sb]
        dep_ids = [j for d in deps for j in str(d).split(",") if j]
        if dep_ids:
            cmd.append("--dependency=afterok:" + ":".join(dep_ids))
        cmd.append(f"{SLURM_DIR}/{req['script']}")
        submitted += 1
        rc, out = run(cmd, env=env, cwd=str(repo))
        jid = parse_jobid(out) if rc == 0 else None
        if jid is None:
            e["submit_error"] = out[:300] or f"sbatch exit {rc}"
        else:
            e.pop("submit_error", None)
            e.update(job_id=jid, submitted=now())
        save_json(state_path, state)
    pending = {rid: e for rid, e in reqs.items() if e.get("job_id") and not e.get("collected") and rid in valid}
    ids = sorted({j for e in pending.values() for j in job_ids(e)})
    try:
        states = sacct_states(run, ids, environ, str(repo))
        state.pop("sacct_error", None)
    except RuntimeError as err:
        states, state["sacct_error"] = {}, str(err)
    finished = collected = 0
    for rid, e in pending.items():
        req = valid[rid]
        sts = [s for j in job_ids(e) for s in states.get(j, [])]
        e["states"] = sorted(set(sts))
        known = all(states.get(j) for j in job_ids(e))
        if not (known and sts and all(s in TERMINAL for s in sts)):
            continue
        finished += 1
        e.setdefault("finished", now())
        if "post" in req:
            try:
                args = [expand(a, environ) for a in req["post"].get("args", [])]
                rc, out = run([sys.executable, "-m", req["post"]["module"], *args], env=environ, cwd=str(repo),
                              timeout=POST_TIMEOUT_S)
                e["post_output_tail"] = out[-3000:]
                if rc != 0:
                    e["post_error"] = (f"exit {rc}: " + out[-480:])[:500]
            except Exception as err:  # noqa: BLE001
                e["post_error"] = f"{type(err).__name__}: {err}"[:500]
        e["collected"] = now()
        collected += 1
        save_json(state_path, state)
        if req.get("collect"):
            err = collect(run, f"ops {rid}: {req['what']} ({','.join(e['states'])})", environ, repo)
            if err:
                e["collect_error"] = err
        save_json(state_path, state)
    snap = cluster_snapshot(run, environ, str(repo))
    status = {"updated": now(), "queue": sorted(valid), "cluster": snap, "requests": {
        rid: {"jobid": e.get("job_id"), "states": e.get("states"), "submitted": e.get("submitted"),
              "finished": e.get("finished"), "collected": e.get("collected"), "submit_error": e.get("submit_error"),
              "post_error": e.get("post_error"), "invalid": e.get("invalid"), "waiting_for": e.get("waiting_for"),
              "dependency_failed": e.get("dependency_failed"),
              "post_output_tail": (e.get("post_output_tail") or "")[-800:] or None}
        for rid, e in reqs.items()}}
    if "TG_ROOT" in environ:
        save_json(status_path(environ), status)
    if now() - state.get("last_heartbeat", 0) >= HEARTBEAT_S:
        err = collect(run, "ops heartbeat (status only)", environ, repo)
        if err:
            print(f"heartbeat push failed (retry next hour): {err}")
        state["last_heartbeat"] = now()
    save_json(state_path, state)
    summary = {"submitted": submitted, "finished": finished, "collected": collected, "invalid": len(invalid),
               "cluster": snap["counts"]}
    print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} submitted {submitted} finished {finished} collected {collected} "
          f"invalid {len(invalid)} cluster {snap['counts']}", flush=True)
    return summary


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("tick")
    t.add_argument("--repo", required=True)
    t.add_argument("--state", required=True)
    v = sub.add_parser("validate")
    v.add_argument("--repo", required=True)
    args = ap.parse_args()
    if args.cmd == "tick":
        tick(args.repo, args.state)
        return
    valid, invalid = load_requests(Path(args.repo).resolve())
    for rid, errs in invalid.items():
        for err in errs:
            print(f"INVALID {rid}: {err}")
    print(f"{len(valid)} valid, {len(invalid)} invalid")
    sys.exit(1 if invalid else 0)


if __name__ == "__main__":
    main()
