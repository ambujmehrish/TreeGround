import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import mllm  # noqa: E402
from treeground.run import ProtocolError, load_config  # noqa: E402

# D46 launcher: runs a ta_* script on Qwen3-VL-8B. mllm.load_qwen pins the 4B width (hidden 2560); the 8B has hidden
# 4096 with the same 8 x 128 key/value heads. treeground/mllm.py is part of the query-cache identity, so it stays
# unchanged and only this launcher relaxes the width check, only for the pinned 8B revision.
# usage: python scripts/qwen8.py scripts/ta_train.py --config configs/ta_pool_full_8b.yaml [...]

QWEN8 = {"repo": "Qwen/Qwen3-VL-8B-Instruct", "revision": "0c351dd01ed87e9c1b53cbc748cba10e6187ff3b"}
HIDDEN8 = 4096
TARGETS = ("scripts/ta_train.py", "scripts/ta_test_eval.py")


def config_arg(args: list[str]) -> str:
    for i, a in enumerate(args):
        if a == "--config" and i + 1 < len(args):
            return args[i + 1]
        if a.startswith("--config="):
            return a.split("=", 1)[1]
    raise ProtocolError("qwen8.py needs --config")


def prepare(argv: list[str]) -> tuple[str, list[str]]:
    if not argv or argv[0] not in TARGETS:
        raise ProtocolError(f"qwen8.py runs one of {TARGETS}, got {argv[:1]}")
    target, args = argv[0], argv[1:]
    qwen = load_config(ROOT / config_arg(args))["models"]["qwen"]
    if {k: qwen.get(k) for k in QWEN8} != QWEN8:
        raise ProtocolError(f"qwen8.py is only for the pinned Qwen3-VL-8B, the config has {qwen}")
    mllm.EXPECTED_HIDDEN = HIDDEN8
    return target, args


def main() -> None:
    target, args = prepare(sys.argv[1:])
    sys.argv = [str(ROOT / target)] + args
    runpy.run_path(str(ROOT / target), run_name="__main__")


if __name__ == "__main__":
    main()
