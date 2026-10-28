import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_baselines as B  # noqa: E402
import s3_pool as S3P  # noqa: E402

# Deviation D36 candidate pool: the D30 pool (capped nodes + top-3 SAM 3 text instances per phrase) plus, per
# phrase, the D13 Qwen3-VL-4B box + SAM 3 mask of that phrase (b_baselines.QwenBoxSam, fixed settings), put first
# among the phrase's instances (score 2.0 marks it; s3_pool.augment takes rank 0 of every phrase before the cap, and
# s5_pool counts proposed vs kept boxes). Installing it replaces s3_pool.TextInstances, so every pool builder and evaluator that calls s3_pool.TextInstances (s5_pool, s3_eval.image_rows, gres_eval, partonomy_eval,
# paco_test_eval) uses the same D36 pool at training and at test time.

BOX_SCORE = 2.0
_TextInstances = S3P.TextInstances


class TextPlusBox:
    def __init__(self, cfg: dict):
        self.text = _TextInstances(cfg)
        self.box = B.QwenBoxSam(cfg)

    def __call__(self, image, phrase_list: list[str], k: int) -> list[list[tuple]]:
        inst = self.text(image, phrase_list, k)
        self.last_boxes = 0
        if not phrase_list:
            return inst
        for lst, (m, _, _, acc) in zip(inst, self.box(image, phrase_list)):
            if acc and m is not None and m.any():
                lst.insert(0, (m, BOX_SCORE))   # rank 0: taken before any rank-1 instance
                self.last_boxes += 1
        return inst


def install() -> None:
    S3P.TextInstances = TextPlusBox
