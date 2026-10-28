import argparse
import importlib
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s5_common as S5C  # noqa: E402
from treeground.run import ProtocolError, load_config, require_approved  # noqa: E402

# Deviation D36 evaluation: runs an existing evaluator (paco_test_eval, gres_eval, partonomy_eval, ophrg_eval) unchanged except
# that its candidate pool is the D36 pool (s5_common.install) and its cache identity also hashes s5_common.py.
# The S5 checkpoint gives a new stage name (stage_suffix _s5), so its caches never mix with S2 long / S3 / S4.
# Usage: python scripts/s5_eval.py <evaluator> --config <cfg> [evaluator arguments]

EVALUATORS = ("paco_test_eval", "gres_eval", "partonomy_eval", "ophrg_eval")


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in EVALUATORS:
        raise ProtocolError(f"first argument must be one of {EVALUATORS}")
    name = sys.argv.pop(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    known, _ = ap.parse_known_args()
    cfg = load_config(known.config)
    if not cfg["s2"].get("stage_suffix", "").startswith("_s5"):
        raise ProtocolError("s5_eval needs an S5 config (s2.stage_suffix _s5...)")
    require_approved(cfg, ["D36"] + (["D37"] if cfg["s2"].get("spec_sources") else []))
    S5C.install()
    mod = importlib.import_module(name)
    mod.FILES = tuple(mod.FILES) + ("scripts/s5_common.py", "scripts/s5_eval.py")
    mod.main()


if __name__ == "__main__":
    main()
