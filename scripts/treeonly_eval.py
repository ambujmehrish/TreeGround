import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import paco_test_eval as PT  # noqa: E402
import s3_pool as S3P  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# PACO-LVIS test (D35) for ours with the tree only: the capped SAM 3 proposal nodes, no SAM 3 text instances in the
# candidate pool (the pool S2 long was trained on). paco_test_eval is unchanged except that the text-instance source
# returns nothing and the results live under their own stage '<stage>_treeonly'.


class NoText:
    def __init__(self, cfg: dict):
        pass

    def __call__(self, image, phrase_list, k):
        return [[] for _ in phrase_list]


S3P.TextInstances = NoText
PT.FILES = tuple(PT.FILES) + ("scripts/treeonly_eval.py",)
_stage, _identity = PT.stage_name, PT.identity
PT.stage_name = lambda cfg, method: _stage(cfg, method) + "_treeonly" if method == "ours" else _stage(cfg, method)
PT.identity = lambda cfg, method: {**_identity(cfg, method), "pool_source": "tree_nodes_only"}

if __name__ == "__main__":
    if "ours" not in sys.argv:
        raise ProtocolError("treeonly_eval.py is only for --method ours")
    PT.main()
