import argparse
import hashlib
import json
import os
import sys
from dataclasses import dataclass
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
from treeground.data import paco  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# Test-split evaluation of a trained D41 / D42 / D43 / D44 checkpoint (configs/ta_test_<variant>.yaml) under the
# existing protocol scripts, which run unchanged: paco_test_eval (D35), gres_eval (D33), partonomy_eval (D32),
# ophrg_eval (D38), gseval_eval (D45; the gRefCOCO predictor, its 'set' threshold read from the gres run). Only their 'ours' predictor, stage name, cache identity and approvals are replaced. Queries
# and answers are built exactly as in each script's S2 predictor; scores come from the checkpoint at its read-out
# layer, chosen with the checkpoint on PACO val. gRefCOCO thresholds are chosen on gRefCOCO val by gres_eval's own
# rule and frozen for testA / testB. Nothing is tuned on a test split.
# Candidates: grid checkpoints (D41-D43) score the capped D3/D4 SAM 3 nodes alone, the tree they were trained on.
# Pool checkpoints (D44) score, per query, the grid nodes + that query's phrases' top-k SAM 3 text instances
# (s3_pool.augment, as in training), one tree per query; a part query is asked as the object -> part chain and
# decided object-first with the anchor chosen on PACO val and stored in the checkpoint (TA.anchor).

BENCHES = ("paco", "gres", "partonomy", "ophrg", "gseval")
MODULES = {"paco": "paco_test_eval", "gres": "gres_eval", "partonomy": "partonomy_eval", "ophrg": "ophrg_eval",
           "gseval": "gseval_eval"}
PROTOCOL = {"paco": "D35", "gres": "D33", "partonomy": "D32", "ophrg": "D38", "gseval": "D45"}
TA_FILES = ("scripts/ta_test_eval.py", "scripts/ta_eval.py", "scripts/ta_train.py", "treeground/tree_attn.py")


class MemorySource:
    # ta_train.Source over one in-memory image and its node masks (query_logits interface)
    def __init__(self, processor, sel_id: int, image: np.ndarray, rles: list, parents: np.ndarray, mode: str,
                 seed: int, key: str, inst: np.ndarray | None = None):
        self.processor, self.sel_id, self.rles, self.parents, self.inst = processor, sel_id, rles, parents, inst
        self.image = Image.fromarray(image)
        self.mode, self.seed, self.key = mode, seed, key
        self.merge = processor.image_processor.merge_size
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ProtocolError("tokenizer has no pad token")

    def image_item(self, m, queries: list[S2T.Query]) -> tuple[list[S2.Seq], TA.Nodes]:
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, self.image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        return seqs, TA.build_nodes(self.rles, self.parents, grid, self.mode, V4.tree_seed(self.seed, self.key),
                                    self.inst)


@dataclass
class Pool:
    # the candidates of one query: masks, tree, how many lead the grid, pool provenance (None for grid checkpoints)
    rles: list
    parents: np.ndarray | None
    n_grid: int
    inst: np.ndarray | None = None

    @property
    def n(self) -> int:
        return len(self.rles)


def anchor_setting(cfg: dict, meta: dict) -> str | None:
    # the D44 part decision: the checkpoint's val choice, unless ta_eval.anchor names one (none | hard | soft<x>)
    if not meta.get("pool"):
        return None
    name = cfg["ta_eval"].get("anchor", "auto")
    name = meta["anchor"] if name == "auto" else name
    if name is None:
        raise ProtocolError("pool checkpoint without a val-chosen anchor: set ta_eval.anchor")
    TA.anchor_mode(name)
    return name


class Scorer:
    # the trained checkpoint + SAM 3 node proposals (+ SAM 3 text instances for a pool checkpoint); node logits per
    # query, one row per <SEL>
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
        self.pool = bool(self.meta.get("pool"))
        self.anchor = anchor_setting(cfg, self.meta)          # the part decision's name (None: grid checkpoint)
        if self.pool:
            import s3_pool as S3P
            self.S3P = S3P
            self.ps = {k: S3P.settings(cfg)[k] for k in ("instances_per_phrase", "max_instances", "dup_iou")}
            if self.ps != self.meta["pool_settings"]:
                raise ProtocolError(f"s3_pool settings {self.ps} differ from the checkpoint's {self.meta['pool_settings']}")
            self.text = S3P.TextInstances(cfg)

    def nodes(self, image: np.ndarray) -> tuple[list, np.ndarray | None]:
        rles = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        return rles, (T.build_tree(rles, self.device).parent if rles else None)

    def pools(self, image: np.ndarray, texts: list[str]) -> list[Pool]:
        # one Pool per query text: a grid checkpoint shares the image's nodes; a pool checkpoint adds the text's
        # phrases' instances (SAM 3 text run once per image over the distinct phrases) and rebuilds the tree
        rles, parents = self.nodes(image)
        if not self.pool:
            return [Pool(rles, parents, len(rles))] * len(texts)
        plist = list(dict.fromkeys(p for t in texts for p in self.S3P.phrases(t)))
        inst = dict(zip(plist, self.text(image, plist, self.ps["instances_per_phrase"])))
        out = []
        for t in texts:
            pool, imeta = self.S3P.augment(rles, [inst[p] for p in self.S3P.phrases(t)], self.ps["max_instances"],
                                           self.ps["dup_iou"])
            out.append(Pool(pool, T.build_tree(pool, self.device).parent if pool else None, len(rles),
                            TA.pool_features(len(rles), imeta)))
        return out

    @torch.no_grad()
    def logits(self, image: np.ndarray, key: str, pool: Pool, queries: list[S2T.Query]) -> list[np.ndarray]:
        if not pool.n:
            return [np.zeros((q.answer.count(mllm.SEL), 0)) for q in queries]
        src = MemorySource(self.proc, self.extra["sel_id"], image, pool.rles, pool.parents,
                           TAT.node_mode(self.meta["variant"]), self.meta["seed"], key, pool.inst)
        rows = TAT.query_logits(self.model, self.readout, src, None, queries, self.device,
                                self.cfg["ta_eval"]["batch_size"], [self.layer])
        return [r[self.layer].numpy().astype(np.float64) for r in rows]

    def score(self, image: np.ndarray, key: str, texts: list[str], queries: list[S2T.Query]
              ) -> tuple[list[Pool], list[np.ndarray]]:
        # per query its pool and logits (grid checkpoints: every query of the image in one batched pass)
        pools = self.pools(image, texts)
        if not self.pool:
            return pools, self.logits(image, key, pools[0], queries)
        return pools, [self.logits(image, key, p, [q])[0] for p, q in zip(pools, queries)]


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def chain_query(qid: str, obj: str, part: str) -> S2T.Query:
    # a part query asked object-first (the I1 chain form; the answer carries the object and the part <SEL>)
    text = paco.CHAIN_TEMPLATE.format(object=obj, part=part)
    return S2T.Query(qid, "chain", S2.user_text(text), f"{obj} {mllm.SEL} {part} {mllm.SEL}", [])


def part_as_chain(q: S2T.Query) -> S2T.Query:
    # the names of a PACO / PARTONOMY / OP-HRG part query live in its id: <key>|part|<object>:<part>
    obj, part = (paco.display(x) for x in q.qid.split("|")[2].split(":", 1))
    return chain_query(q.qid, obj, part)


def single_rows(g: pd.DataFrame, pools: list[Pool], lgs: list[np.ndarray], gts: dict, anchor=None) -> list[dict]:
    # s3_eval.image_rows over each query's pool (same columns, same picks and chain rule); `anchor` (a pool
    # checkpoint) decides part queries object-first from their two-row chain logits
    rows = []
    for q, pool, lg in zip(g.itertuples(), pools, lgs):
        rles, n, sig = pool.rles, pool.n, sigmoid(lg)
        base = {"qid": q.qid, "kind": q.kind, "image": q.image_key, "present": q.kind != "absent", "pool": n,
                "nodes": pool.n_grid}
        if q.kind == "absent":
            rows.append({**base, "max_sigma": float(sig[0].max()) if n else float("-inf")})
            continue
        gt = [gts[a] for a in json.loads(q.gt)]
        ious = M.iou(rles, gt) if n else np.zeros((0, len(gt)))
        kind_of = lambda i: "none" if i < 0 else "node" if i < pool.n_grid else "inst"
        if q.kind == "chain":
            # the S2 chain rule (s3_eval.image_rows): the part inside the object's subtree, the object itself when
            # that subtree is empty
            a1, af = TA.anchor(lg[0], lg[1], pool.parents, None)
            desc = T.from_parents(pool.parents).strict_descendants(a1) if n else np.zeros(0, np.int64)
            a2 = int(desc[int(np.argmax(lg[1][desc]))]) if len(desc) else a1
            rows.append({**base, "object_iou": float(ious[a1, 0]) if n else 0.0,
                         "part_flat_iou": float(ious[af, 1]) if n else 0.0,
                         "part_tree_iou": float(ious[a2, 1]) if n else 0.0,
                         "max_sigma": float(sig[1].max()) if n else float("-inf"),
                         "part_flat_kind": kind_of(af), "best_part_iou": float(ious[:, 1].max()) if n else 0.0})
            continue
        if q.kind == "part" and anchor is not None:
            if lg.shape[0] != 2:
                raise ProtocolError(f"{q.qid}: an anchored part query needs object and part logits")
            a1, a = TA.anchor(lg[0], lg[1], pool.parents, TA.anchor_mode(anchor))
            _, flat = TA.anchor(lg[0], lg[1], pool.parents, None)
            extra = {"anchor": anchor, "anchor_object": kind_of(a1),
                     "part_none_iou": float(ious[flat, 0]) if n else 0.0, "score_row": 1}
        else:
            a, extra = (int(np.argmax(lg[0])) if n else -1), {"score_row": 0}
        inter = int(M.intersections([rles[a]], gt)[0, 0]) if n else 0
        area_p = int(M.areas([rles[a]])[0]) if n else 0
        area_g = int(M.areas(gt)[0])
        rows.append({**base, **extra, "sel_iou": float(ious[a, 0]) if n else 0.0,
                     "max_sigma": float(sig[extra["score_row"], a]) if n else float("-inf"), "pick_kind": kind_of(a),
                     "best_node_iou": float(ious[:pool.n_grid, 0].max()) if pool.n_grid else 0.0,
                     "best_pool_iou": float(ious[:, 0].max()) if n else 0.0,
                     "sel_inter": inter, "sel_union": area_p + area_g - inter, "gt_area": area_g})
    return rows


def queries_of(g: pd.DataFrame, anchored: bool = False) -> list[S2T.Query]:
    qs = [S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [])
          for q in g.itertuples()]
    return [part_as_chain(q) if anchored and q.kind == "part" else q for q in qs]


class PacoOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, g: pd.DataFrame, gts: dict) -> list[dict]:
        anchor = self.s.anchor
        pools, lgs = self.s.score(image, g.image_key.iloc[0], g.text.tolist(), queries_of(g, anchor is not None))
        return single_rows(g, pools, lgs, gts, anchor)


class GresOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, texts: list[str], key: str = "gres") -> list[dict]:
        # key only seeds the shuffled tree of a random_tree checkpoint (gres_eval passes no image key)
        import gres_eval as GE
        qs = [S2T.Query(f"q{i}", "object", S2.user_text(f"{t} {mllm.SEL}"), f"{S2.user_text(t)} {mllm.SEL}", [])
              for i, t in enumerate(texts)]
        pools, lgs = self.s.score(image, key, [f"{t} {mllm.SEL}" for t in texts], qs)
        out = []
        for pool, lg in zip(pools, lgs):
            if not pool.n:
                out.append({"nodes": []})
                continue
            sig = sigmoid(lg[0])
            order = np.argsort(-sig, kind="stable")[:GE.KEEP_MAX]
            out.append({"nodes": [(float(sig[i]), pool.rles[i]) for i in order if sig[i] >= GE.KEEP_MIN or i == order[0]]})
        return out


class PartonomyOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, label: str, parts: list[str]) -> list[dict]:
        import partonomy_eval as PE
        anchor = self.s.anchor
        texts = [f"Locate {PE.phrase(p, label)}. {mllm.SEL}" for p in parts]
        qs = [S2T.Query(f"partonomy|part|{label}:{p}", "part", S2.user_text(t), f"{p} of the {label} {mllm.SEL}", [])
              for p, t in zip(parts, texts)]
        if anchor is not None:
            qs = [chain_query(q.qid, label, p) for q, p in zip(qs, parts)]
        pools, lgs = self.s.score(image, label, texts, qs)
        out = []
        for pool, lg in zip(pools, lgs):
            if not pool.n:
                out.append({"nodes": None})
                continue
            a = TA.anchor(lg[0], lg[1], pool.parents, TA.anchor_mode(anchor))[1] if anchor else int(np.argmax(lg[0]))
            out.append({"nodes": M.decode(pool.rles[a]).astype(bool)})
        return out


class OphrgOurs:
    def __init__(self, cfg: dict):
        self.s = Scorer(cfg)

    def __call__(self, image: np.ndarray, key: str, qs: list) -> list[dict]:
        import ophrg_eval as OE
        anchor = self.s.anchor
        gts = {i: M.encode(q[4])[0] for i, q in enumerate(qs)}
        g = pd.DataFrame([{"qid": q[0], "kind": q[1], "text": OE.text(q[1], q[2], q[3]), "n_sel": 1,
                           "gt": json.dumps([i]), "image_key": key} for i, q in enumerate(qs)])
        queries = queries_of(g)
        if anchor is not None:   # object-first with the benchmark's own names, as OE.text words the part query
            queries = [chain_query(q[0], q[2], q[3]) if q[1] == "part" else qq for q, qq in zip(qs, queries)]
        pools, lgs = self.s.score(image, key, g.text.tolist(), queries)
        return [{"qid": r["qid"], "kind": r["kind"], "image": key, "inter": r["sel_inter"], "union": r["sel_union"],
                 "pick_kind": r["pick_kind"]} for r in single_rows(g, pools, lgs, gts, anchor)]


OURS = {"paco": PacoOurs, "gres": GresOurs, "partonomy": PartonomyOurs, "ophrg": OphrgOurs, "gseval": GresOurs}


def code_hash(module) -> str:
    h = hashlib.sha256()
    for rel in tuple(module.FILES) + TA_FILES:
        h.update((ROOT / rel).read_bytes())
    return h.hexdigest()


def install(bench: str):
    # swap the 'ours' predictor and bookkeeping of one protocol script; everything else in it runs unchanged
    module = __import__(MODULES[bench])
    stage0, ident0, appr0 = module.stage_name, module.identity, module.approvals
    shards_key = {"paco": "paco_test", "gres": "gres_eval", "partonomy": "partonomy_eval", "ophrg": "ophrg_eval",
                  "gseval": "gseval_eval"}[bench]

    def stage_name(cfg, method):
        return TAT.stage(cfg) if method == "ours" else stage0(cfg, method)

    def identity(cfg, method, *extra):
        if method != "ours":
            return ident0(cfg, method, *extra)
        ckpt, run = TAE.load_checkpoint(cfg, False)
        meta = ckpt["meta"]
        ident = {"kind": {"paco": "paco_test_eval", "gres": "gres_eval", "partonomy": "partonomy_eval",
                          "ophrg": "ophrg_eval", "gseval": "gseval_eval"}[bench],
                 "method": method, "stage": stage_name(cfg, method), "num_shards": cfg[shards_key]["num_shards"],
                 "code": code_hash(module), "run": run.name, "checkpoint": cfg["ta_eval"]["checkpoint"],
                 "step": meta["step"], "layer": meta["layer"], "pool": "tree_nodes_only"}
        if meta.get("pool"):
            ident.update(pool="grid+text per query", pool_settings=meta["pool_settings"],
                         sam3_text=cfg["baselines"]["sam3_text"], anchor=anchor_setting(cfg, meta))
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
        return [PROTOCOL[bench]] + TAT.run_deviations(cfg["ta"])

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
