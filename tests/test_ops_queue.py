import json
import sys
from pathlib import Path

import pytest

from treeground.ops import queue as Q


class Fake:
    def __init__(self, sacct="", squeue="", ended="", squeue_rc=0, sbatch_rc=0, sbatch_out=None):
        self.calls, self.next_id = [], 100
        self.sacct, self.squeue, self.ended = sacct, squeue, ended
        self.squeue_rc, self.sbatch_rc, self.sbatch_out = squeue_rc, sbatch_rc, sbatch_out

    def __call__(self, cmd, env=None, cwd=None, timeout=None):
        self.calls.append({"cmd": cmd, "env": env, "cwd": cwd})
        if cmd[0] == "sbatch":
            if self.sbatch_rc:
                return self.sbatch_rc, self.sbatch_out or "sbatch: error: QOSMaxSubmitJobPerUserLimit"
            self.next_id += 1
            return 0, self.sbatch_out or f"{self.next_id};leonardo\n"
        if cmd[0] == "sacct" and "-S" in cmd:
            return 0, self.ended
        if cmd[0] == "sacct":
            return 0, self.sacct
        if cmd[0] == "squeue":
            return self.squeue_rc, self.squeue if self.squeue_rc == 0 else "squeue: error: timeout"
        return 0, "post ok " + " ".join(cmd)

    def of(self, name):
        return [c for c in self.calls if c["cmd"][0] == name]

    def posts(self):
        return [c for c in self.calls if c["cmd"][0] == sys.executable]

    def collects(self):
        return [c for c in self.calls if c["cmd"][:2] == ["bash", Q.COLLECTOR]]


@pytest.fixture
def repo(tmp_path):
    (tmp_path / "slurm").mkdir()
    (tmp_path / "slurm" / "q_gpu.sbatch").write_text("#!/bin/bash\n")
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "refine.py").write_text("")
    (tmp_path / Q.QUEUE_DIR).mkdir(parents=True)
    return tmp_path


ENV = {"TG_ROOT": "/w/tg", "TG_CACHE": "/s/cache", "TG_DATA": "/s/data", "TG_LOGS": "/w/repo/logs",
       "TG_RUNS": "/w/tg/runs", "PATH": "/usr/bin"}


def put(repo, rid, **req):
    req = {"id": rid, "what": f"job {rid}", **req}
    (repo / Q.QUEUE_DIR / f"{rid}.json").write_text(json.dumps(req))
    return req


def sub(**kw):
    return {"script": "q_gpu.sbatch", "env": {"TG_PY": "scripts/refine.py"}, **kw}


def tick(repo, tmp_path, fake, t=1000.0, env=None):
    e = dict(ENV, TG_ROOT=str(tmp_path / "root")) if env is None else env
    return Q.tick(repo, tmp_path / "state" / "state.json", run=fake, environ=e, now=lambda: t)


@pytest.mark.parametrize("req, msg", [
    ({"script": "../x.sbatch"}, "script:"),
    ({"script": "slurm/q_gpu.sbatch"}, "script:"),
    ({"script": "missing.sbatch"}, "does not exist"),
    ({"script": "q_gpu.sbatch", "env": {"V": "a;b"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "a|b"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "`id`"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "a\"b"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "${HOME}/x"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "$HOME"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "${TG_ROOT"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"V": "$TG_ROOTX"}}, "env.V"),
    ({"script": "q_gpu.sbatch", "env": {"lower": "x"}}, "env: name"),
    ({"script": "q_gpu.sbatch", "env": {"TG_PY": "/etc/passwd"}}, "env.TG_PY"),
    ({"script": "q_gpu.sbatch", "sbatch": {"partition": "x"}}, "sbatch: key"),
    ({"script": "q_gpu.sbatch", "sbatch": {"time": "1 day"}}, "sbatch.time"),
    ({"script": "q_gpu.sbatch", "post": {"module": "os.system"}}, "post.module"),
    ({"script": "q_gpu.sbatch", "post": {"module": "treeground.run"}}, "post.module"),
    ({"script": "q_gpu.sbatch", "post": {"module": "treeground.analysis.tables", "args": ["a&b"]}}, "post.args[0]"),
    ({"script": "q_gpu.sbatch", "after": ["r"]}, "itself"),
    ({"script": "q_gpu.sbatch", "account": "x"}, "account: unknown key"),
    ({"after_jobs": ["12a"]}, "after_jobs"),
    ({"after_jobs": ["123"], "env": {"A": "b"}}, "env: not allowed in a watch-only"),
    ({}, "script: required"),
    ({"script": "q_gpu.sbatch", "collect": "yes"}, "collect"),
])
def test_validation_refuses(repo, req, msg):
    errs = Q.validate({"id": "r", "what": "w", **req}, "r", repo)
    assert any(msg in e for e in errs), errs


@pytest.mark.parametrize("name", sorted(Q.DENIED_ENV) + ["LD_PRELOAD", "DYLD_X", "BASH_FUNC_X", "SLURM_JOB",
                                                           "PYTHONWARNINGS"])
def test_denied_env_names(repo, name):
    errs = Q.validate({"id": "r", "what": "w", "script": "q_gpu.sbatch", "env": {name: "x"}}, "r", repo)
    assert any(f"env: name {name}" in e for e in errs), errs


def test_id_must_equal_stem(repo):
    assert any("file stem" in e for e in Q.validate({"id": "a", "what": "w", "script": "q_gpu.sbatch"}, "b", repo))


def test_valid_request_and_placeholders(repo):
    req = {"id": "r", "what": "w", "script": "q_gpu.sbatch",
           "env": {"TG_PY": "scripts/refine.py", "TG_ARGS": "--config configs/refine.yaml --out ${TG_ROOT}/x,$TG_RUNS"},
           "sbatch": {"time": "02:00:00", "qos": "normal", "array": "0-7%4"}, "collect": True,
           "post": {"module": "treeground.analysis.tables", "args": ["--out", "${TG_ROOT}/t"]}}
    assert Q.validate(req, "r", repo) == []
    assert Q.expand("${TG_ROOT}/x,$TG_RUNS", ENV) == "/w/tg/x,/w/tg/runs"
    with pytest.raises(ValueError):
        Q.expand("${TG_ROOT}", {})


def test_one_tick_submits_once_env_deps_invalid_and_waiting(repo, tmp_path):
    put(repo, "a", **sub(env={"TG_PY": "scripts/refine.py", "TG_ARGS": "--out ${TG_ROOT}/a"},
                         sbatch={"time": "01:00:00", "array": "0-3%2"}))
    put(repo, "b", **sub(after=["a"]))
    put(repo, "bad", script="x/y.sbatch")
    put(repo, "c", **sub(after=["bad"]))
    fake = Fake(sacct="101|PENDING\n")
    s = tick(repo, tmp_path, fake)
    sb = fake.of("sbatch")
    assert len(sb) == 2 and s["invalid"] == 1
    a, b = sb[0], sb[1]
    assert f"--account={Q.ACCOUNT}" in a["cmd"] and "--time=01:00:00" in a["cmd"] and "--array=0-3%2" in a["cmd"]
    assert a["cmd"][-1] == "slurm/q_gpu.sbatch" and not any(x.startswith("--dependency") for x in a["cmd"])
    assert a["env"]["TG_ARGS"] == f"--out {tmp_path / 'root'}/a" and a["env"]["PATH"] == "/usr/bin"
    assert "--dependency=afterok:101" in b["cmd"]
    st = json.loads((tmp_path / "state" / "state.json").read_text())["requests"]
    assert st["a"]["job_id"] == "101" and st["b"]["job_id"] == "102"
    assert "script" in st["bad"]["invalid"] and st["c"]["waiting_for"] == ["bad"]
    status = json.loads((tmp_path / "root" / "results" / "ops" / "status.json").read_text())
    assert status["requests"]["a"]["jobid"] == "101" and status["requests"]["bad"]["invalid"]


def test_running_then_completed_post_collect_once_and_restart_safe(repo, tmp_path):
    put(repo, "a", **sub(collect=True, post={"module": "treeground.analysis.tables",
                                             "args": ["--out", "${TG_ROOT}/t"]}))
    fake = Fake(sacct="101|RUNNING\n101.batch|RUNNING\n")
    tick(repo, tmp_path, fake)
    assert not fake.posts() and len(fake.of("sbatch")) == 1
    collects_before = len(fake.collects())
    fake.sacct = "101|COMPLETED\n101.batch|COMPLETED\n101.extern|COMPLETED\n"
    tick(repo, tmp_path, fake, t=1100.0)
    posts = fake.posts()
    assert len(posts) == 1 and posts[0]["cmd"][1:] == ["-m", "treeground.analysis.tables", "--out",
                                                       f"{tmp_path / 'root'}/t"]
    new = fake.collects()[collects_before:]
    assert len([c for c in new if "ops a:" in c["cmd"][2]]) == 1 and "(COMPLETED)" in new[0]["cmd"][2]
    n_calls = len(fake.calls)
    tick(repo, tmp_path, fake, t=1200.0)
    later = fake.calls[n_calls:]
    assert not [c for c in later if c["cmd"][0] == "sbatch" or c["cmd"][0] == sys.executable]
    assert not [c for c in later if c["cmd"][:2] == ["bash", Q.COLLECTOR] and "ops a:" in c["cmd"][2]]


def test_step_lines_ignored_and_arrays_map_to_base(repo, tmp_path):
    put(repo, "a", **sub(sbatch={"array": "0-1"}))
    fake = Fake(sacct="101_0|COMPLETED\n101_1|RUNNING\n101_0.batch|COMPLETED\n")
    tick(repo, tmp_path, fake)
    fake.sacct = "101_0|COMPLETED\n101_1|COMPLETED\n101_1.batch|FAILED\n"
    tick(repo, tmp_path, fake, t=1100.0)
    st = json.loads((tmp_path / "state" / "state.json").read_text())["requests"]["a"]
    assert st["states"] == ["COMPLETED"] and st["collected"] == 1100.0


def test_submit_error_recorded_and_retried(repo, tmp_path):
    put(repo, "a", **sub())
    fake = Fake(sbatch_rc=1)
    tick(repo, tmp_path, fake)
    st = json.loads((tmp_path / "state" / "state.json").read_text())["requests"]["a"]
    assert "QOSMaxSubmitJobPerUserLimit" in st["submit_error"] and "job_id" not in st
    fake.sbatch_rc = 0
    tick(repo, tmp_path, fake, t=1100.0)
    st = json.loads((tmp_path / "state" / "state.json").read_text())["requests"]["a"]
    assert st["job_id"] == "101" and "submit_error" not in st


def test_non_numeric_job_id_rejected(repo, tmp_path):
    put(repo, "a", **sub())
    tick(repo, tmp_path, Fake(sbatch_out="Submitted batch job 5"))
    st = json.loads((tmp_path / "state" / "state.json").read_text())["requests"]["a"]
    assert "job_id" not in st and st["submit_error"]


def test_max_new_bounds_a_tick(repo, tmp_path):
    for i in range(Q.MAX_NEW + 5):
        put(repo, f"r{i:03d}", **sub())
    fake = Fake()
    tick(repo, tmp_path, fake)
    assert len(fake.of("sbatch")) == Q.MAX_NEW
    tick(repo, tmp_path, fake, t=1100.0)
    assert len(fake.of("sbatch")) == Q.MAX_NEW + 5


def test_watch_only(repo, tmp_path):
    put(repo, "w", after_jobs=["555", "556"], collect=True, post={"module": "treeground.analysis.tables"})
    fake = Fake(sacct="555|COMPLETED\n")
    tick(repo, tmp_path, fake)
    assert not fake.of("sbatch") and not fake.posts()  # 556 unknown to sacct -> not finished
    fake.sacct = "555|COMPLETED\n556_3|CANCELLED by 1\n"
    tick(repo, tmp_path, fake, t=1100.0)
    assert len(fake.posts()) == 1 and any("ops w:" in c["cmd"][2] and "CANCELLED,COMPLETED" in c["cmd"][2]
                                          for c in fake.collects())
    assert not fake.of("sbatch")


def test_heartbeat_hourly(repo, tmp_path):
    fake = Fake()
    tick(repo, tmp_path, fake, t=10_000.0)
    tick(repo, tmp_path, fake, t=10_000.0 + 1800)
    tick(repo, tmp_path, fake, t=10_000.0 + 3700)
    hb = [c for c in fake.collects() if "heartbeat" in c["cmd"][2]]
    assert len(hb) == 2


def test_cluster_snapshot(repo, tmp_path):
    fake = Fake(squeue="900|fact-x|PENDING|0:00|1-00:00:00|2|(Priority)|N/A\n"
                       "901|tg_e_q|RUNNING|1:00|4:00:00|3|lrdn[0001-0002,0004]|2026-10-24T09:00:00\n",
                ended="902|a|COMPLETED|00:10:00|0:0|2026-10-24T08:00:00\n"
                      "903|b|RUNNING|00:10:00|0:0|Unknown\n902.batch|batch|COMPLETED|0|0:0|x\n")
    snap = Q.cluster_snapshot(fake)
    assert snap["squeue"][0]["reason_or_nodelist"] == "(Priority)" and snap["squeue"][1]["start"].startswith("2026")
    assert snap["squeue"][1]["reason_or_nodelist"] == "lrdn[0001-0002,0004]"
    assert snap["counts"] == {"PENDING": 1, "RUNNING": 1}
    assert [r["jobid"] for r in snap["ended_24h"]] == ["902"]
    put(repo, "a", **sub())
    bad = Fake(squeue_rc=1)
    s = tick(repo, tmp_path, bad)
    status = json.loads((tmp_path / "root" / "results" / "ops" / "status.json").read_text())
    assert "squeue_error" in status["cluster"] and s["submitted"] == 1


def test_lock_blocks_second_runner(repo, tmp_path):
    import fcntl
    (tmp_path / "state").mkdir()
    with open(tmp_path / "state" / "state.json.lock", "w") as f:
        fcntl.flock(f, fcntl.LOCK_EX | fcntl.LOCK_NB)
        put(repo, "a", **sub())
        fake = Fake()
        assert tick(repo, tmp_path, fake) == {"locked": True} and not fake.calls


def test_cli_validate(repo, capsys):
    put(repo, "bad", script="../x.sbatch")
    sys.argv = ["queue", "validate", "--repo", str(repo)]
    with pytest.raises(SystemExit) as e:
        Q.main()
    assert e.value.code == 1 and "INVALID bad" in capsys.readouterr().out


def test_repo_queue_is_valid():
    root = Path(__file__).resolve().parents[1]
    valid, invalid = Q.load_requests(root)
    assert not invalid, invalid
