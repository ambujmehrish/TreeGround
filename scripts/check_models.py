import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.run import load_config  # noqa: E402

# Verifies offline that every model of a config is fully cached at its pinned revision (diffusers pipelines with
# their own download check, everything else via huggingface_hub). Run on the login node after fetch_models.py.


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--only", nargs="*")
    args = ap.parse_args()
    cfg = load_config(args.config)
    from huggingface_hub import snapshot_download
    bad = []
    for name, m in cfg["models"].items():
        if args.only and name not in args.only:
            continue
        try:
            if name in ("flux_fill", "qwen_image_edit"):
                from diffusers import DiffusionPipeline
                DiffusionPipeline.download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"],
                                           local_files_only=True)
            else:
                snapshot_download(m["repo"], revision=m["revision"], cache_dir=cfg["hf_home"], local_files_only=True,
                                  allow_patterns=m.get("allow_patterns"), ignore_patterns=m.get("ignore_patterns"))
            print(f"complete: {name} {m['repo']}@{m['revision'][:8]}")
        except Exception as e:  # noqa: BLE001
            print(f"INCOMPLETE: {name} {m['repo']}: {type(e).__name__}: {str(e)[:300]}")
            bad.append(name)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
