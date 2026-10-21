import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PACKAGES = [
    "torch", "torchvision", "transformers", "accelerate", "huggingface_hub", "tokenizers", "pycocotools",
    "opencv-python-headless", "safetensors", "numpy", "scipy", "pyyaml", "pyarrow", "pandas", "pillow",
]
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProtocolError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ProtocolError(f"environment variable {name} is not set (needed by config value {value!r})")
            return os.environ[name]
        return _ENV.sub(sub, value)
    return value


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    for inc in reversed(_as_list(raw.pop("include", []))):
        raw = _merge(load_config(path.parent / inc), raw)
    return _expand(raw)


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else [x]


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"rev": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def runs_root(cfg: dict) -> Path:
    return Path(cfg["runs_root"])


def start_run(stage: str, cfg: dict, smoke: bool, extra: dict | None = None) -> Path:
    root = runs_root(cfg) / "smoke" / stage if smoke else runs_root(cfg) / stage
    git = git_state()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / f"{stamp}_{(git['rev'] or 'nogit')[:8]}"
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    meta = {
        "stage": stage,
        "smoke": smoke,
        "git": git,
        "packages": package_versions(),
        "python": sys.version,
        "host": platform.node(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "models": cfg.get("models"),
        "seed": cfg.get("seed"),
        "started_utc": stamp,
    }
    meta.update(extra or {})
    write_json(run_dir / "meta.json", meta)
    ensure_deviation_log(cfg)
    return run_dir


def update_meta(run_dir: Path, **fields: Any) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(fields)
    write_json(run_dir / "meta.json", meta)


def write_json(path: str | Path, obj: Any) -> None:
    # atomic: concurrent array tasks may read the file while another writes it
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{platform.node()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def resolve_run(cfg: dict, stage: str, ref: str, smoke: bool) -> Path:
    root = runs_root(cfg) / "smoke" / stage if smoke else runs_root(cfg) / stage
    if ref == "latest":
        done = sorted(p for p in root.glob("*") if (p / "metrics.json").exists())
        if not done:
            raise ProtocolError(f"no completed {stage} run under {root}")
        run = done[-1]
    else:
        run = root / ref
        if not (run / "metrics.json").exists():
            raise ProtocolError(f"{stage} run {run} does not exist or did not complete")
    if read_json(run / "meta.json").get("smoke", False) != smoke:
        raise ProtocolError(f"{run} smoke flag does not match current mode (smoke={smoke})")
    return run


def assert_real_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if (REPO / "tests") in p.parents or any(part in {"fixtures", "synthetic", "mock"} for part in p.parts):
        raise ProtocolError(f"experimental entry points refuse synthetic or fixture inputs: {p}")
    if not p.exists():
        raise ProtocolError(f"missing required input: {p}")
    return p


APPROVED_INTERPRETATIONS = [
    ("I1", "2", "V2/V4 use only (image, object category) pairs with exactly one instance whose (object, part) has exactly one part mask"),
    ("I2", "2.4", "fixed query templates; default chat template; image before text; truncate after last <SEL>"),
    ("I3", "5 (V3)", "deterministic evaluation; l = highest hidden mAP, ties to smaller l; image-level bootstrap CIs descriptive only"),
    ("I4", "5 (V3)", "raw features L2-normalised only"),
    ("I5", "5 (V2)", "10k seeded PACO-LVIS train images + up to 2k gRefCOCO train no-target examples; 2k steps S1 recipe"),
    ("I6", "3, 7", "tau_rej calibrated on PACO val + gRefCOCO val no-target; V2/V4 PACO val numbers labelled dev"),
    ("I7", "2", "MLP Linear(d,1024)-GELU-LayerNorm-Linear(1024,256) for g and f"),
    ("I8", "2.4", "AdamW lr 3e-4 wd 0.01 cosine 100 warm-up, batch 32 images, focal gamma 2 no alpha, tau 0.07"),
    ("I9", "5 (V4)", "5 seeds; fresh randomised tree and pre-scoring samples per seed; common subset over flat, true, all 5 randomised"),
    ("I10", "5 (V1)", "SceneParser matched by box IoU >= 0.5 and flagged box-matched; PARTONOMY/SceneParser optional"),
    ("I11", "5 (V4)", "V2 checkpoint trained with true-subtree chain negatives; confound reported, no extra variant"),
    ("I12", "4.1", "selector success = selected node IoU >= 0.5 with requested GT; best-proposal identification diagnostic only"),
    ("I13", "2.2", "negative sampling: 32 hard + 32 uniform in-image, chain subtree first with top-up, up to 64 cross-image"),
]

PROPOSED_DEVIATIONS = [
    {
        "id": "D1",
        "spec_section": "1 (Proposals)",
        "change": "SAM 3 automatic mask generation implemented as an explicit 32x32 point-grid loop over Sam3TrackerModel "
                  "+ Sam3TrackerProcessor instead of transformers' MaskGenerationPipeline",
        "reason": "the pipeline (transformers 5.0.0) normalises grid points by the longest edge while Sam3 resizes each "
                  "axis to 1008, so on non-square images part of the image receives no points; the pipeline also applies "
                  "box NMS and a predicted-IoU filter that the protocol disables",
        "expected_impact": "grid covers the whole image as the protocol intends; model, prompts, stability filter and "
                           "mask post-processing are those of the transformers SAM 3 tracker route",
    },
]


def deviation_log_path(cfg: dict) -> Path:
    return runs_root(cfg) / "deviations.jsonl"


def read_deviations(cfg: dict) -> dict[str, dict]:
    path = deviation_log_path(cfg)
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in records}


def ensure_deviation_log(cfg: dict) -> None:
    path = deviation_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_deviations(cfg)
    today = dt.date.today().isoformat()
    new = [
        {"id": i, "date": today, "spec_section": sec, "change": change, "reason": "user decision (implementation prompt §2)",
         "expected_impact": "fixes an open point; no silent choice", "status": "approved"}
        for i, sec, change in APPROVED_INTERPRETATIONS if i not in existing
    ]
    new += [dict(d, date=today, status="proposed") for d in PROPOSED_DEVIATIONS if d["id"] not in existing]
    if new:
        with open(path, "a") as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True) + "\n")


def require_approved(cfg: dict, ids: list[str]) -> None:
    records = read_deviations(cfg)
    pending = [i for i in ids if records.get(i, {}).get("status") != "approved"]
    if pending:
        raise ProtocolError(
            f"deviation(s) {pending} are not approved in {deviation_log_path(cfg)}; set status to 'approved' to proceed")
