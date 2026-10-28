import json
from pathlib import Path

from treeground.run import ProtocolError

# The OP-HRG (arXiv 2607.15374) test sets that S5 / D37 training must not see: the PascalPart-116 image ids of its
# pascalpart_val.txt and the PartImageNet_OOD test images. Kept apart from scripts/ophrg_eval.py so the training-pool
# identity hashes only these lists, not the evaluator.

PASCAL_IDS = Path(__file__).resolve().parent / "ophrg_pascalpart_val.txt"


def pascal_ids() -> list[str]:
    return sorted(line.strip() for line in PASCAL_IDS.read_text().splitlines() if line.strip())


def partimagenet_ood_test(root: str | Path) -> tuple[dict, Path]:
    hits = [p for p in Path(root).rglob("test.json") if "__MACOSX" not in p.parts]
    if len(hits) != 1:
        raise ProtocolError(f"expected one PartImageNet_OOD test.json under {root}, found {hits}")
    return json.loads(hits[0].read_text()), hits[0].parent / "test"
