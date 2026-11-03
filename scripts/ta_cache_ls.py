import hashlib
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.run import load_config  # noqa: E402

# Lists the ta_* test-evaluation caches (scripts/ta_test_eval.py): shards present vs. expected and the git commit
# whose code produced each cache's identity hash (the merge of a cache must run on that commit's code).
# Usage: python scripts/ta_cache_ls.py --config configs/ta_test_full.yaml [--commits 40]

KINDS = {"paco_test_eval": "paco", "gres_eval": "gres", "partonomy_eval": "partonomy", "ophrg_eval": "ophrg"}


def files_of(bench: str) -> tuple[str, ...]:
    import ta_test_eval as TTE
    module = __import__(TTE.MODULES[bench])
    return tuple(module.FILES) + TTE.TA_FILES


def code_hash_at(commit: str, files: tuple[str, ...]) -> str | None:
    h = hashlib.sha256()
    for rel in files:
        r = subprocess.run(["git", "show", f"{commit}:{rel}"], cwd=ROOT, capture_output=True)
        if r.returncode:
            return None
        h.update(r.stdout)
    return h.hexdigest()


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--commits", type=int, default=40)
    args = ap.parse_args()
    cfg = load_config(args.config)
    log = subprocess.run(["git", "log", f"-{args.commits}", "--format=%h %ad %s", "--date=short"], cwd=ROOT,
                         capture_output=True, text=True).stdout.splitlines()
    commits = [line.split()[0] for line in log]
    by_hash = {}
    for bench in KINDS.values():
        files = files_of(bench)
        for c in commits:
            h = code_hash_at(c, files)
            if h and h not in by_hash:
                by_hash[h] = c
    root = Path(cfg["cache_root"]) / "full"
    for kind in KINDS:
        for d in sorted((root / kind).glob("ta_*/**/identity.json")):
            d = d.parent
            ident = json.loads((d / "identity.json").read_text())["identity"]
            got = len(list((d / "shards").glob("*.parquet")))
            n = ident.get("num_shards")
            sub = ident.get("split") or ident.get("benchmark") or ""
            print(f"{kind:15s} {ident['stage']:12s} {sub:13s} shards {got}/{n} {'OK ' if got == n else '...'} "
                  f"code@{by_hash.get(ident['code'], 'unknown commit')} step {ident.get('step')} layer "
                  f"{ident.get('layer')} {d}")


if __name__ == "__main__":
    main()
