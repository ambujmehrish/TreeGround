import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.mllm import SEL  # noqa: E402
from treeground.run import load_config  # noqa: E402

# Deviation D47: a D25 (CoCoEdit protocol) mask source for a trained ta_* checkpoint (D41-D46), selected exactly as
# on gRefCOCO by ta_test_eval.GresOurs: the referent is the query ("{referent}" -> answer "{referent} <SEL>"), the
# candidates are the checkpoint's own (grid nodes, plus the referent's top-k SAM 3 text instances for a pool
# checkpoint), and the edit mask is the single highest-scoring candidate (one mask per sample, as every D25 source).
# spec['config'] names the checkpoint's ta_test_<variant>.yaml.

CODE = ("scripts/cocoedit_ta_source.py", "scripts/ta_test_eval.py", "scripts/ta_eval.py", "scripts/ta_train.py",
        "treeground/tree_attn.py", "scripts/s3_pool.py", "scripts/b_baselines.py")


class TASelect:
    @staticmethod
    def sub_cfg(spec: dict) -> dict:
        return load_config(ROOT / spec["config"])

    @classmethod
    def requires(cls, spec: dict) -> list[str]:
        import ta_train as TAT
        return ["D3", "D4", "D47"] + TAT.run_deviations(cls.sub_cfg(spec)["ta"])

    @classmethod
    def identity(cls, cfg: dict, spec: dict) -> dict:
        import ta_eval as TAE
        c = cls.sub_cfg(spec)
        ckpt, run = TAE.load_checkpoint(c, False)
        meta = ckpt["meta"]
        h = hashlib.sha256()
        for rel in CODE:
            h.update((ROOT / rel).read_bytes())
        return {"spec": spec, "stage": f"ta_{c['ta']['variant']}{c['ta'].get('stage_suffix', '')}", "run": run.name,
                "checkpoint": c["ta_eval"]["checkpoint"], "step": meta["step"], "layer": meta["layer"],
                "pool": bool(meta.get("pool")), "pool_settings": meta.get("pool_settings"),
                "qwen": c["models"]["qwen"], "sam3": c["models"]["sam3"], "proposals": c["proposals"],
                "sam3_text": c["baselines"]["sam3_text"], "code": h.hexdigest()}

    def __init__(self, cfg: dict, spec: dict):
        import ta_test_eval as TTE
        self.TTE = TTE
        self.s = TTE.Scorer(self.sub_cfg(spec))

    def __call__(self, image: np.ndarray, referent: str) -> tuple:
        from treeground import s2 as S2
        q = self.TTE.S2T.Query("cocoedit", "object", S2.user_text(f"{referent} {SEL}"),
                               f"{S2.user_text(referent)} {SEL}", [])
        pools, lgs = self.s.score(image, "cocoedit", [f"{referent} {SEL}"], [q])
        pool, lg = pools[0], lgs[0]
        if not pool.n:
            return None, float("-inf"), {"kind": "none", "pool": 0, "nodes": 0}
        a = int(np.argmax(lg[0]))
        return (M.decode(pool.rles[a]).astype(bool), float(self.TTE.sigmoid(lg[0, a])),
                {"kind": "node" if a < pool.n_grid else "inst", "pool": pool.n, "nodes": pool.n_grid})
