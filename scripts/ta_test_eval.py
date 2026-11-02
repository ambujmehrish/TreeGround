import argparse
import hashlib
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_train as S2T  # noqa: E402
import ta_eval as TAE  # noqa: E402
import ta_train as TAT  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground import tree_attn as TA  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# Test-split evaluation of a trained D41 / D42 checkpoint (configs/ta_test_<variant>.yaml) under the existing
# protocol scripts, which run unchanged: paco_test_eval (D35), gres_eval (D33), partonomy_eval (D32), ophrg_eval
# (D38). Only their 'ours' predictor, stage name, cache identity and approvals are replaced. Candidates are the
# capped D3/D4 SAM 3 nodes alone (no SAM 3 text instances), the tree the checkpoint was trained on; queries and
# answers are built exactly as in each script's S2 predictor; scores come from the checkpoint at its read-out
# layer, chosen with the checkpoint on PACO val. gRefCOCO thresholds are chosen on gRefCOCO val by gres_eval's own
# rule and frozen for testA / testB. Nothing is tuned on a test split.

BENCHES = ("paco", "gres", "partonomy", "ophrg")
MODULES = {"paco": "paco_test_eval", "gres": "gres_eval", "partonomy": "partonomy_eval", "ophrg": "ophrg_eval"}
PROTOCOL = {"paco": "D35", "gres": "D33", "partonomy": "D32", "ophrg": "D38"}
TA_FILES = ("scripts/ta_test_eval.py", "scripts/ta_eval.py", "scripts/ta_train.py", "treeground/tree_attn.py")


class MemorySource:
    # ta_train.Source over one in-memory image and its node masks (query_logits interface)
    def __init__(self, processor, sel_id: int, image: np.ndarray, rles: list, parents: np.ndarray, mode: str,
                 seed: int, key: str):
        self.processor, self.sel_id, self.rles, self.parents = processor, sel_id, rles, parents
        self.image = Image.fromarray(image)
        self.mode, self.seed, self.key = mode, seed, key
        self.merge = processor.image_processor.merge_size
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ProtocolError("tokenizer has no pad token")

    def image_item(self, m, queries: list[S2T.Query]) -> tuple[list[S2.Seq], TA.Nodes]:
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, self.image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        return seqs, TA.build_nodes(self.rles, self.parents, grid, self.mode, V4.tree_seed(self.seed, self.key))


class Scorer:
    # the trained checkpoint + SAM 3 node proposals; node logits per query, one row per <SEL>
    def __init__(self, cfg: dict):
        import s2_nodes as S2N
        from treeground import proposals as P
        from treeground.identity import node_cap, qwen_static
        self.cfg, self.S2N, self.device = cfg, S2N, cfg["device"]
        c = TAT.stage_cfg(cfg, False)
        self.model, self.proc, self.extra, self.readout, self.meta, self.run = TAE.load_trained(
            cfg, c, self.device, qwen_static(cfg), False)
        self.model.eval()
        self.readout.eval()
        self.layer = int(self.meta["layer"])
        m, p = cfg["models"], cfg["proposals"]
        self.sam, self.sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], self.device, p["dtype"],
                                           cfg.get("hf_home"))
        self.cap = node_cap(cfg, False)

    def nodes(self, image: np.ndarray) -> tuple[list, np.ndarray | None]:
        rles = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        return rles, (T.build_tree(rles, self.device).parent if rles else None)

    @torch.no_grad()
    def logits(self, image: np.ndarray, key: str, rles: list, parents, queries: list[S2T.Query]) -> list[np.ndarray]:
        if not rles:
            return [np.zeros((q.answer.count(mllm.SEL), 0)) for q in queries]
        src = MemorySource(self.proc, self.extra["sel_id"], image, rles, parents, TAT.node_mode(self.meta["variant"]),
                           self.meta["seed"], key)
        rows = TAT.query_logits(self.model, self.readout, src, None, queries, self.device,
                                self.cfg["ta_eval"]["batch_size"], [self.layer])
        return [r[self.layer].numpy().astype(np.float64) for r in rows]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def single_rows(g: pd.DataFrame, rles: list, parents, sigs: list[np.ndarray], gts: dict) -> list[dict]:
    # s3_eval.image_rows with the pool = the nodes (same columns, same picks and chain rule)
    n = len(rles)
    rows = []
    for q, sig in zip(g.itertuples(), sigs):
        base = {"qid": q.qid, "kind": q.kind, "image": q.image_key, "present": q.kind != "absent", "pool": n,
                "nodes": n}
        if q.kind == "absent":
            rows.append({**base, "max_sigma": float(sig[0].max()) if n else float("-inf")})
            continue
        gt = [gts[a] for a in json.loads(q.gt)]
        ious = M.iou(rles, gt) if n else np.zeros((0, len(gt)))
        if q.kind == "chain":
            a1, af = (int(np.argmax(sig[0])), int(np.argmax(sig[1]))) if n else (-1, -1)
            desc = T.from_parents(parents).strict_descendants(a1) if n else np.zeros(0, np.int64)
            a2 = int(desc[int(np.argmax(sig[1][desc]))]) if len(desc) else a1
            rows.append({**base, "object_iou": float(ious[a1, 0]) if n else 0.0,
                         "part_flat_iou": float(ious[af, 1]) if n else 0.0,
                         "part_tree_iou": float(ious[a2, 1]) if n else 0.0,
                         "max_sigma": float(sig[1].max()) if n else float("-inf"),
                         "part_flat_kind": "node" if n else "none",
                         "best_part_iou": float(ious[:, 1].max()) if n else 0.0})
            continue
        a = int(np.argmax(sig[0])) if n else -1
        inter = int(M.intersections([rles[a]], gt)[0, 0]) if n else 0
        area_p = int(M.areas([rles[a]])[0]) if n else 0
        area_g = int(M.areas(gt)[0])
        rows.append({**base, "sel_iou": float(ious[a, 0]) if n else 0.0,
                     "max_sigma": float(sig[0, a]) if n else float("-inf"), "pick_kind": "node" if n else "none",
                     "best_node_iou": float(ious[:, 0].max()) if n else 0.0,
                     "best_pool_iou": float(ious[:, 0].max()) if n else 0.0,
                     "sel_inter": inter, "sel_union": area_p + area_g - inter, "gt_area": area_g})
    return rows


def queries_of(g: pd.DataFrame) -> list[S2T.Query]:
    return [S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [])
            for q in g.itertuples()]


class PacoOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, g: pd.DataFrame, gts: dict) -> list[dict]:
        rles, parents = self.s.nodes(image)
        sigs = [sigmoid(x) for x in self.s.logits(image, g.image_key.iloc[0], rles, parents, queries_of(g))]
        return single_rows(g, rles, parents, sigs, gts)


class GresOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, texts: list[str], key: str = "gres") -> list[dict]:
        # key only seeds the shuffled tree of a random_tree checkpoint (gres_eval passes no image key)
        import gres_eval as GE
        rles, parents = self.s.nodes(image)
        qs = [S2T.Query(f"q{i}", "object", S2.user_text(f"{t} {mllm.SEL}"), f"{S2.user_text(t)} {mllm.SEL}", [])
              for i, t in enumerate(texts)]
        if not rles:
            return [{"nodes": []} for _ in texts]
        out = []
        for lg in self.s.logits(image, key, rles, parents, qs):
            sig = sigmoid(lg[0])
            order = np.argsort(-sig, kind="stable")[:GE.KEEP_MAX]
            out.append({"nodes": [(float(sig[i]), rles[i]) for i in order if sig[i] >= GE.KEEP_MIN or i == order[0]]})
        return out


class PartonomyOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, label: str, parts: list[str]) -> list[dict]:
        import partonomy_eval as PE
        rles, parents = self.s.nodes(image)
        qs = [S2T.Query(f"partonomy|part|{label}:{p}", "part",
                        S2.user_text(f"Locate {PE.phrase(p, label)}. {mllm.SEL}"), f"{p} of the {label} {mllm.SEL}", [])
              for p in parts]
        if not rles:
            return [{"nodes": None} for _ in parts]
        return [{"nodes": M.decode(rles[int(np.argmax(lg[0]))]).astype(bool)}
                for lg in self.s.logits(image, label, rles, parents, qs)]


class OphrgOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, key: str, qs: list) -> list[dict]:
        import ophrg_eval as OE
        gts = {i: M.encode(q[4])[0] for i, q in enumerate(qs)}
        g = pd.DataFrame([{"qid": q[0], "kind": q[1], "text": OE.text(q[1], q[2], q[3]), "n_sel": 1,
                           "gt": json.dumps([i]), "image_key": key} for i, q in enumerate(qs)])
        rles, parents = self.s.nodes(image)
        sigs = [sigmoid(x) for x in self.s.logits(image, key, rles, parents, queries_of(g))]
        return [{"qid": r["qid"], "kind": r["kind"], "image": key, "inter": r["sel_inter"], "union": r["sel_union"],
                 "pick_kind": r["pick_kind"]} for r in single_rows(g, rles, parents, sigs, gts)]


OURS = {"paco": PacoOurs, "gres": GresOurs, "partonomy": PartonomyOurs, "ophrg": OphrgOurs}


def code_hash(module) -> str:
    h = hashlib.sha256()
    for rel in tuple(module.FILES) + TA_FILES:
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def install(bench: str):
    # swap the 'ours' predictor and bookkeeping of one protocol script; everything else in it runs unchanged
    module = __import__(MODULES[bench])
    stage0, ident0, appr0 = module.stage_name, module.identity, module.approvals
    shards_key = {"paco": "paco_test", "gres": "gres_eval", "partonomy": "partonomy_eval", "ophrg": "ophrg_eval"}[bench]

    def stage_name(cfg, method):
        return TAT.stage(cfg) if method == "ours" else stage0(cfg, method)

    def identity(cfg, method, *extra):
        if method != "ours":
            return ident0(cfg, method, *extra)
        ckpt, run = TAE.load_checkpoint(cfg, False)
        ident = {"kind": {"paco": "paco_test_eval", "gres": "gres_eval", "partonomy": "partonomy_eval",
                          "ophrg": "ophrg_eval"}[bench],
                 "method": method, "stage": stage_name(cfg, method), "num_shards": cfg[shards_key]["num_shards"],
                 "code": code_hash(module), "run": run.name, "checkpoint": cfg["ta_eval"]["checkpoint"],
                 "step": ckpt["meta"]["step"], "layer": ckpt["meta"]["layer"], "pool": "tree_nodes_only"}
        if bench == "gres":
            ident["split"] = extra[0]
        if bench == "ophrg":
            ident["benchmark"] = extra[0]
        return ident

    def approvals(*args):
        method = args[-1]
        if method != "ours":
            return appr0(*args)
        cfg = args[0] if len(args) == 2 else CONFIG["cfg"]
        return [PROTOCOL[bench]] + TAT.deviations(cfg["ta"]["variant"])

    module.Ours = OURS[bench]
    module.stage_name, module.identity, module.approvals = stage_name, identity, approvals
    return module


CONFIG = {}   # gres / partonomy approvals take only the method; the launcher records the config here


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, choices=BENCHES)
    known, rest = ap.parse_known_args()
    if "--method" not in rest or rest[rest.index("--method") + 1] != "ours":
        raise ProtocolError("ta_test_eval.py runs only --method ours (baselines: the protocol scripts themselves)")
    from treeground.run import load_config
    cfg = load_config(rest[rest.index("--config") + 1])
    if cfg["ta"]["variant"] not in TAT.VARIANTS:
        raise ProtocolError(f"unknown variant {cfg['ta']['variant']!r}")
    CONFIG["cfg"] = cfg
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    module = install(known.bench)
    sys.argv = [sys.argv[0]] + rest
    module.main()


if __name__ == "__main__":
    main()
