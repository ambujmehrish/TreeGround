import argparse
import json
import os
import shutil
import sys
import tomllib
from importlib import metadata
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

RESULTS: list[tuple[str, str, str]] = []
ENV_DIRS = ["TG_ROOT", "TG_DATA", "TG_CACHE", "TG_RUNS", "TG_VENV", "HF_HOME", "XDG_CACHE_HOME", "PIP_CACHE_DIR",
            "TORCH_HOME", "TRITON_CACHE_DIR"]


def record(ok: bool, name: str, detail: str = "", warn: bool = False) -> None:
    level = "PASS" if ok else ("WARN" if warn else "FAIL")
    RESULTS.append((level, name, detail))
    print(f"  [{level}] {name}" + (f" - {detail}" if detail else ""))


def inside(p: Path, root: Path) -> bool:
    return p == root or root in p.parents


def check_paths() -> None:
    home = Path.home().resolve()
    scratch = Path(os.environ["SCRATCH"]).resolve() if os.environ.get("SCRATCH") else None
    for var in ENV_DIRS:
        val = os.environ.get(var)
        if not val:
            record(False, f"env {var}", "not set - source tg_env.sh")
            continue
        p = Path(val).resolve()
        if inside(p, home):
            record(False, f"env {var}", f"{p} is inside $HOME")
            continue
        if var != "TG_VENV":
            p.mkdir(parents=True, exist_ok=True)
        record(p.exists() and os.access(p, os.W_OK), f"env {var}", str(p))
    for var in ("TG_RUNS", "TG_VENV", "HF_HOME"):
        if scratch and os.environ.get(var) and inside(Path(os.environ[var]).resolve(), scratch):
            record(False, f"{var} on $SCRATCH", "purged periodically: keep runs, venv and weights on $WORK", warn=True)
    cache = os.environ.get("TG_CACHE")
    if cache and Path(cache).exists():
        free = shutil.disk_usage(cache).free / 1e9
        record(free >= 300, "free space at TG_CACHE", f"{free:.0f} GB (features need ~300 GB)", warn=True)


def check_packages() -> None:
    pins = tomllib.loads((REPO / "pyproject.toml").read_text())["project"]
    for dep in pins["dependencies"] + pins["optional-dependencies"]["dev"]:
        name, want = dep.split("==")
        try:
            have = metadata.version(name)
        except metadata.PackageNotFoundError:
            record(False, f"package {name}", "not installed")
            continue
        record(have.split("+")[0] == want, f"package {name}", f"{have} (pinned {want})")


def check_config_paths(cfg: dict, skip_data: bool) -> None:
    for key in ("cache_root", "runs_root", "hf_home"):
        record(Path(cfg[key]).resolve().exists(), f"config {key}", cfg[key])
    if skip_data:
        return
    from treeground.data import manifest as MF
    d = cfg["data"]
    for source in ("paco_lvis", "grefcoco", "partonomy", "sceneparser"):
        ok = MF.source_available(cfg, source)
        record(ok, f"dataset {source}", "present" if ok else "missing", warn=source in ("partonomy", "sceneparser"))
    for sub in ("train2017", "val2017", "train2014"):
        p = Path(d["coco_images"]) / sub
        n = sum(1 for _ in p.glob("*.jpg")) if p.exists() else 0
        record(n > 0, f"COCO {sub}", f"{n} images at {p}")


def check_models(cfg: dict) -> None:
    from huggingface_hub import snapshot_download
    for name, m in cfg["models"].items():
        try:
            path = Path(snapshot_download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"],
                                          local_files_only=True))
            # a refused gated download can leave an empty snapshot folder behind
            ok = (path / "config.json").exists() and any(path.glob("*.safetensors"))
            record(ok, f"model {name}", f"{m['repo']}@{m['revision'][:8]} {path}" + ("" if ok else " (incomplete)"))
        except Exception as e:  # noqa: BLE001
            record(False, f"model {name}", f"{m['repo']}@{m['revision'][:8]} not cached ({type(e).__name__})")


def check_gpu() -> None:
    import torch
    ok = torch.cuda.is_available()
    record(ok, "cuda available", f"torch {torch.__version__} cuda {torch.version.cuda}")
    if ok:
        x = torch.randn(256, 256, device="cuda")
        record(bool(torch.isfinite(x @ x).all()), "gpu matmul", torch.cuda.get_device_name(0))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-models", action="store_true")
    ap.add_argument("--skip-data", action="store_true")
    ap.add_argument("--expect-gpu", action="store_true")
    ap.add_argument("--out")
    args = ap.parse_args()
    from treeground.run import load_config
    print("paths"); check_paths()
    print("packages"); check_packages()
    cfg = load_config(REPO / "configs" / "paths.yaml")
    print("config"); check_config_paths(cfg, args.skip_data)
    if not args.skip_models:
        print("models"); check_models(cfg)
    if args.expect_gpu:
        print("gpu"); check_gpu()
    fails = [r for r in RESULTS if r[0] == "FAIL"]
    if args.out:
        Path(args.out).write_text(json.dumps([dict(zip(("level", "check", "detail"), r)) for r in RESULTS], indent=2))
    print(f"\n{'ENVIRONMENT READY' if not fails else f'{len(fails)} FAILED CHECK(S)'}")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main()
