import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/paths.yaml")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from huggingface_hub import snapshot_download
    for name, m in cfg["models"].items():
        # facebook/sam3 is gated: request access on the Hub and run `hf auth login` on the login node first
        path = snapshot_download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"])
        print(name, m["revision"], path)


if __name__ == "__main__":
    main()
