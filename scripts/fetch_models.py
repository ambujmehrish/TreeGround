import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs" / "paths.yaml"))
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from huggingface_hub import snapshot_download
    failed = []
    for name, m in cfg["models"].items():
        if args.only and name not in args.only:
            continue
        try:
            path = snapshot_download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"])
            print(f"ok {name} {m['repo']}@{m['revision']} -> {path}")
        except Exception as e:  # noqa: BLE001
            # facebook/sam3 is gated: request access on the Hub, then `hf auth login`
            print(f"FAILED {name} {m['repo']}: {type(e).__name__}: {str(e)[:200]}")
            failed.append(name)
    sys.exit(1 if failed else 0)


if __name__ == "__main__":
    main()
