import argparse
import sys
import threading
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config  # noqa: E402


def dir_bytes(path: Path) -> int:
    return sum(f.stat().st_size for f in path.rglob("*") if f.is_file()) if path.exists() else 0


def report_progress(path: Path, total: int | None, stop: threading.Event, every: float) -> None:
    start, t0 = dir_bytes(path), time.time()
    while not stop.wait(every):
        done = dir_bytes(path) - start
        rate = done / max(time.time() - t0, 1e-6) / 1e6
        of = f" of {total / 1e9:.1f} GB" if total else ""
        print(f"    ... {done / 1e9:.2f} GB downloaded{of} ({rate:.0f} MB/s)", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--every", type=float, default=20.0, help="seconds between progress lines")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from huggingface_hub import HfApi, snapshot_download
    api = HfApi()
    print(f"HF cache: {cfg['hf_home']}", flush=True)
    failed = []
    for name, m in cfg["models"].items():
        if args.only and name not in args.only:
            continue
        try:
            info = api.model_info(m["repo"], revision=m["revision"], files_metadata=True)
            total = sum(s.size or 0 for s in info.siblings)
            print(f"[{name}] {m['repo']}@{m['revision'][:8]}: {len(info.siblings)} files, {total / 1e9:.1f} GB", flush=True)
        except Exception as e:  # noqa: BLE001
            # facebook/sam3 is gated: request access on the Hub, then `hf auth login`
            print(f"[{name}] FAILED to read {m['repo']} metadata: {type(e).__name__}: {str(e)[:200]}", flush=True)
            failed.append(name)
            continue
        stop = threading.Event()
        watcher = threading.Thread(target=report_progress, args=(Path(cfg["hf_home"]), total, stop, args.every),
                                   daemon=True)
        watcher.start()
        t0 = time.time()
        try:
            path = snapshot_download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"])
            print(f"[{name}] done in {time.time() - t0:.0f}s -> {path}", flush=True)
        except Exception as e:  # noqa: BLE001
            print(f"[{name}] FAILED: {type(e).__name__}: {str(e)[:200]}", flush=True)
            failed.append(name)
        finally:
            stop.set()
            watcher.join()
    print("all models cached" if not failed else f"failed: {failed}", flush=True)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
