import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json  # noqa: E402

# Deviation D29: a D25 (CoCoEdit protocol) mask source for the S2-family selector: the D23 edit selection
# (e_select.select_sample: capped nodes + SAM 3 text instances of the referent, the S2 stage of spec['config'],
# pick 'ours2' = flat or 'ours2tree' = tree), so that the CoCoEdit-protocol rows use the current selector.

CODE = ("scripts/cocoedit_s2_source.py", "scripts/e_select.py", "scripts/x_transfer_s2.py", "scripts/s2_train.py",
        "scripts/b_baselines.py")


class S2Select:
    @staticmethod
    def sub_cfg(spec: dict) -> dict:
        return load_config(ROOT / spec["config"])

    @staticmethod
    def requires(spec: dict) -> list[str]:
        return ["D3", "D4", "D10", "D14", "D21", "D23", "D29"] + (["D30"] if spec.get("d30") else [])

    @classmethod
    def identity(cls, cfg: dict, spec: dict) -> dict:
        import e_select as ES
        c = cls.sub_cfg(spec)
        stage, s2_run, ev, _ = ES.X2.s2_inputs(c)
        h = hashlib.sha256()
        for rel in CODE:
            h.update((ROOT / rel).read_bytes())
        return {"spec": spec, "s2_stage": stage, "s2_run": s2_run.name, "s2_eval_run": ev.name,
                "checkpoint": c["s2_eval"]["checkpoint"], "select": ES.settings(c), "sam3": c["models"]["sam3"],
                "sam3_text": c["baselines"]["sam3_text"], "code": h.hexdigest()}

    def __init__(self, cfg: dict, spec: dict):
        import e_select as ES
        self.ES, self.pick = ES, spec.get("pick", "ours2")
        if self.pick not in ("ours2", "ours2tree"):
            raise ProtocolError(f"pick must be ours2 or ours2tree, not {self.pick}")
        c = self.sub_cfg(spec)
        stage, s2_run, ev, _ = ES.X2.s2_inputs(c)
        ckpt, ck_run = ES.X2.S2E.load_checkpoint(c, False)
        if ck_run != s2_run or read_json(ev / "metrics.json")["checkpoint_step"] != ckpt["meta"]["step"]:
            raise ProtocolError(f"checkpoint {ck_run.name} is not the one {ev.name} evaluated")
        self.cfg, self.models = c, ES.X2.load_models(c, ckpt)
        del ckpt
        self.st = ES.B.Sam3Text(c)
        self.cap, self.rr = ES.X.node_cap(), ES.rerank_weight(c, stage)

    def __call__(self, image: np.ndarray, referent: str) -> tuple:
        blank = np.zeros(image.shape[:2], bool)
        s = SimpleNamespace(sample_id="cocoedit", image_key="cocoedit", subset="", referent=referent,
                            load=lambda: (image, blank))
        r = self.ES.select_sample(self.models, self.st, self.cfg, s, self.cap, self.rr)
        rle = json.loads(r[f"{self.pick}_rle"])
        mask = M.decode(rle).astype(bool) if rle is not None else None
        return mask, float(r[f"{self.pick}_sigma"]), {"kind": r[f"{self.pick}_kind"], "nodes": r["nodes"],
                                                       "instances": r["instances"]}
