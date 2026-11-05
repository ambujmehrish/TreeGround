import argparse
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import torch.nn as nn
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_train as S2T  # noqa: E402
import s3_pool as S3P  # noqa: E402
import ta_paco_nodes as TPN  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground import tree_attn as TA  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import (manifest_inputs, node_cap, open_proposals, open_queries,  # noqa: E402
                                 qwen_static)
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, start_run, v1_gate, write_json)

# D41: TreeGround with tree attention, trained from scratch (base Qwen3-VL-4B, fresh LoRA, randomly initialised
# heads and mask-token embedding; nothing from S1 or S2). Variants share data, targets, negatives, losses and
# schedule and differ only in how node features are formed:
#   flat         S2 node features (coverage-weighted layer states of the image tokens), no mask tokens
#   full         mask tokens, every mask token attends to every other
#   tree         mask tokens, attention to self, parent, children and siblings
#   random_tree  as tree, over a tree with parents shuffled within each depth (fixed per image)
# D42 adds the hierarchical decision on top of the tree variant (and its control):
#   hier              tree attention + stop / descend head: P(v) = P(target) x prod_ancestors P(descend) x P(stop v)
#   hier_random_tree  the same over the shuffled tree (attention and decision)
# Heads are trained jointly at every read-out layer; the checkpoint and its layer are chosen on the fixed PACO val
# subset by single-step selector success (S2 rule), best over steps and layers.

# D43 adds soft tree attention (learned per-layer bias by tree relation; every node token sees every other) and a
# family loss (a positive must outscore its own parent and children by a margin), alone and combined, each with
# its shuffled-tree control:
#   soft, soft_random_tree          soft tree attention
#   fam, fam_random_tree            full attention + family loss (tree for geometry and the loss)
#   soft_fam, soft_fam_random_tree  both
# D44 (pool_*; pool_flat = the S2 feature selector over the pool, no mask tokens): the same selectors over the query-aware pool of D30 (grid nodes + SAM 3 text instances of the query
# phrases, one tree); node tokens carry pool provenance; part queries are decided object-first (TA.anchor), the
# anchor chosen on the val chains and stored in the checkpoint.
HIER = {"hier": "tree", "hier_random_tree": "random_tree"}     # variant -> node relations and decision tree
D43 = {"soft": ("soft", False), "soft_random_tree": ("soft_random_tree", False),
       "fam": ("full", True), "fam_random_tree": ("full_random_tree", True),
       "soft_fam": ("soft", True), "soft_fam_random_tree": ("soft_random_tree", True)}   # -> (node mode, family)
# D44: the same selectors over the query-aware pool (D30 s3_pool caches: grid nodes + top-k SAM 3 text instances of
# the image's query phrases, one tree over the pool). Node tokens carry pool provenance (TA.INST_DIM); part queries
# are decided object-first at evaluation (TA.anchor). Trained on the pool caches alone (no PACO rest cache).
# pool_flat: the S2 feature selector (no mask tokens, coverage-pooled layer states) over the same pool: the
# in-context mask tokens' ablation
D44 = {"pool_flat": ("tree", False), "pool_full": ("full", False), "pool_soft": ("soft", False),
       "pool_fam": ("full", True), "pool_soft_fam": ("soft", True), "pool_full_random_tree": ("full_random_tree", False),
       "pool_soft_random_tree": ("soft_random_tree", False), "pool_soft_fam_random_tree": ("soft_random_tree", True)}
FLAT = ("flat", "pool_flat")
VARIANTS = ("flat",) + TA.MODES + tuple(HIER) + tuple(D43) + tuple(D44)
DEVIATIONS = ["D10", "D21", "D41"]


def is_pool(variant: str) -> bool:
    return variant in D44


def query_cfg(cfg: dict) -> dict:
    # the query table is model-independent text; a run on another backbone (D46) reads the cache built for the
    # backbone named in ta.query_cache_qwen instead of rebuilding it
    q = cfg["ta"].get("query_cache_qwen")
    return cfg if not q else {**cfg, "models": {**cfg["models"], "qwen": q}}


def run_deviations(c: dict) -> list[str]:
    # the variant's deviations, plus D46 for a run on another backbone than the default Qwen3-VL-4B
    return deviations(c["variant"]) + (["D46"] if c.get("query_cache_qwen") else [])


def deviations(variant: str) -> list[str]:
    out = DEVIATIONS + (["D42"] if variant in HIER else [])
    mode, fam = D43.get(variant) or D44.get(variant) or ("", False)
    if variant in D43 or (variant in D44 and (mode.startswith("soft") or fam)):
        out.append("D43")
    return out + (["D44"] if variant in D44 else [])


@dataclass
class PoolMeta(S2T.ImageMeta):
    # a training / validation item over a query-aware pool: provenance per pool entry and, for validation items
    # built in memory, the pool masks themselves (training items read their node cache file as every D21 source)
    inst: np.ndarray | None = None
    rles: list | None = None


class Readout(nn.Module):
    def __init__(self, d_model: int, layers: list[int], n_nodes: int, embed: TA.NodeEmbed | None,
                 hier: bool = False, soft: TA.SoftTreeBias | None = None,
                 family: tuple[float, float] | None = None):
        super().__init__()
        self.soft = soft                 # D43 soft tree attention bias (hooks installed on the model by build)
        self.family = family             # D43 family loss (margin, weight)
        self.layers = list(layers)
        self.hier = hier
        make = (lambda: TA.HierHead(d_model, n_nodes)) if hier else (lambda: TA.ScaledSelector(d_model, d_model,
                                                                                                n_nodes))
        self.heads = nn.ModuleList(make() for _ in layers)
        self.embed = embed
        self.decision = "product"   # hier only; ta_eval --decision: "stop" or "stop_root" read parts of the rule

    def head(self, layer: int) -> TA.ScaledSelector:
        # the stop scorer of a hierarchical head is trained exactly like the flat selector
        h = self.heads[self.layers.index(layer)]
        return h.stop if self.hier else h

    def node_logits(self, layer: int, probe: torch.Tensor, states: torch.Tensor, nb: TA.Nodes) -> torch.Tensor:
        # (n_sel, n) decision logits of the node tokens for the <SEL> rows in probe
        if self.hier:
            h = self.heads[self.layers.index(layer)]
            if self.decision == "stop":
                return h.stop.scores(h.stop.queries(probe), h.stop.nodes(states))
            anc = torch.as_tensor(nb.anc, device=states.device)
            if self.decision == "stop_root":                             # no ancestor factors
                anc = torch.zeros_like(anc)
            elif self.decision != "product":
                raise ProtocolError(f"unknown decision {self.decision!r}")
            return h.logits(probe, states, anc)
        h = self.head(layer)
        return h.scores(h.queries(probe), h.nodes(states))


def stage_cfg(cfg: dict, smoke: bool) -> dict:
    c = dict(cfg["ta"])
    if smoke:
        c.update(c.pop("smoke", {}))
    c.pop("smoke", None)
    if c["variant"] not in VARIANTS:
        raise ProtocolError(f"unknown variant {c['variant']!r}")
    return c


def setup_device(cfg: dict) -> str:
    # S2T.setup_device with a 3 h process-group timeout: rank 0 alone loads caches, saves checkpoints and (in
    # ta_eval) runs the evaluators while the other ranks wait at a collective
    rank, world, local = S2T.dist_env()
    cuda = str(cfg["device"]).startswith("cuda")
    if cuda:
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo", timeout=timedelta(hours=3))
    return f"cuda:{local}" if cuda else cfg["device"]


def stage(cfg: dict) -> str:
    # a non-empty ta.stage_suffix (short pipeline checks) keeps those runs out of the real runs/ta_<variant>
    return f"ta_{cfg['ta']['variant']}{cfg['ta'].get('stage_suffix', '')}"


def node_mode(variant: str) -> str:
    # the flat variants only use the coverage maps, which are the same for every mode
    if variant in FLAT:
        return "tree"
    if variant in HIER:
        return HIER[variant]
    return (D43.get(variant) or D44.get(variant) or (variant,))[0]


class Source(S2T.Source):
    def __init__(self, cfg, smoke, manifest, mhash, processor, sel_id, text_labels, variant: str, seed: int,
                 relations: str | None = None):
        # relations: evaluation-time override of which node tokens see each other (TA.RELATIONS); None = as trained
        super().__init__(cfg, smoke, manifest, mhash, processor, sel_id, text_labels)
        if relations is not None and (variant in FLAT or variant in D43 or variant in D44
                                      or relations not in TA.RELATIONS):
            raise ProtocolError(f"relations {relations!r} cannot override variant {variant!r}")
        self.mode, self.seed = relations or node_mode(variant), seed

    def image_item(self, m: S2T.ImageMeta, queries: list[S2T.Query]) -> tuple[list[S2.Seq], TA.Nodes]:
        image = Image.fromarray(load_rgb(assert_real_path(m.file)))
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        mem = getattr(m, "rles", None)
        if mem is not None:
            h, w = mem[0]["size"] if mem else (image.height, image.width)
            nodes = {"h": h, "w": w, "rles": mem}
        else:
            nodes = S2T.read_nodes(m.nodes) if m.nodes else self.props.nodes(m.sha1, self.tau)
        if len(nodes["rles"]) != m.n or (nodes["h"], nodes["w"]) != (image.height, image.width):
            raise ProtocolError(f"{m.key}: proposal nodes do not match the image or its metadata")
        if m.parents is None or len(m.parents) != m.n + 1:
            raise ProtocolError(f"{m.key}: no tree over its {m.n} nodes")
        return seqs, TA.build_nodes(nodes["rles"], m.parents, grid, self.mode, V4.tree_seed(self.seed, m.sha1),
                                    getattr(m, "inst", None))


class MicroBatches(torch.utils.data.Dataset):
    def __init__(self, entries: list[list[tuple]], items: list[S2T.ImageMeta], src: Source):
        self.entries, self.items, self.src = entries, items, src

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, j: int) -> dict:
        samples, seqs, text_seqs = [], [], []
        for i, as_text, seed in self.entries[j]:
            m = self.items[i]
            rng = np.random.default_rng(seed)
            qi = int(rng.integers(len(m.queries)))
            (seq,), nb = self.src.image_item(m, [m.queries[qi]])
            label = None
            if as_text:
                label = str(rng.choice(sorted({lab for lab in m.labels if lab and lab in self.src.text_labels})))
                text_seqs.append(self.src.text_item(label))
            samples.append({"item": i, "as_text": as_text, "query": qi, "nodes": nb, "label": label})
            seqs.append(seq)
        seqs += text_seqs
        return {"samples": samples, "seqs": seqs, "batch": S2.collate(seqs, self.src.pad_id)}


def run_forward(model, readout: Readout, batch: dict, nodes: list[TA.Nodes | None]) -> TA.Output:
    if readout.embed is None:
        return TA.flat_forward(model, batch, nodes, readout.layers)
    return TA.forward(model, batch, nodes, readout.embed, readout.layers, readout.soft)


def micro_step(model, extra, readout: Readout, mb, items, device, rng, gamma, stats):
    batch = {k: v.to(device) for k, v in mb["batch"].items()}
    seqs, samples = mb["seqs"], mb["samples"]
    nodes = [s["nodes"] for s in samples] + [None] * (len(seqs) - len(samples))
    with S2.autocast(device):
        out = run_forward(model, readout, batch, nodes)
        l_lm = S2.lm_loss(model, extra["sel_emb"], extra["tied"], out.last, batch, seqs)
    trees = [T.from_parents(items[smp["item"]].parents) for smp in samples]   # target tree (chain negatives)
    l_sel = 0.0
    for li, k in enumerate(readout.layers):
        recs, text_probes = [], {}
        t = len(samples)
        for b, smp in enumerate(samples):
            m = items[smp["item"]]
            q = m.queries[smp["query"]]
            rec = V2.ImageRec(m.sha1, m.key, out.nodes[k][b].float(), trees[b], m.labels, m.parts, m.high)
            rec.absent = m.absent
            rec.queries = [V2.QRec(q.qid, q.kind, out.hidden[k][b, seqs[b].sel.to(device)].float(), q.steps, q.path)]
            recs.append((rec, smp["as_text"]))
            if smp["label"] is not None:
                text_probes[smp["label"]] = out.hidden[k][t, seqs[t].sel[0].to(device)].float()
                t += 1
        terms = V2.batch_terms(readout.head(k), recs, text_probes, rng, device, gamma,
                               stats if li == 0 else {key: 0 for key in stats})
        l_sel = l_sel + torch.stack(terms).mean()
        if readout.family is not None:
            margin, weight = readout.family
            fterms = []
            for b, smp in enumerate(samples):
                if smp["as_text"]:
                    continue
                q = items[smp["item"]].queries[smp["query"]]
                fterms += TA.family_terms(readout.head(k), out.hidden[k][b, seqs[b].sel.to(device)].float(),
                                          out.nodes[k][b].float(), smp["nodes"], q.steps, margin)
            if fterms:
                l_sel = l_sel + weight * torch.stack(fterms).mean()
        if readout.hier:
            # descend + root terms for every image query (text-only items train only their text probe, as in S2)
            hterms = []
            for b, smp in enumerate(samples):
                if smp["as_text"]:
                    continue
                m = items[smp["item"]]
                q = m.queries[smp["query"]]
                hterms += TA.hier_terms(readout.heads[li], out.hidden[k][b, seqs[b].sel.to(device)].float(),
                                        out.nodes[k][b].float(), smp["nodes"].anc, q.steps, gamma)
            if hterms:
                l_sel = l_sel + torch.stack(hterms).mean()
    return l_sel, l_lm


@torch.no_grad()
def query_logits(model, readout: Readout, src: Source, m: S2T.ImageMeta, queries: list[S2T.Query], device: str,
                 batch_size: int, layers: list[int] | None = None) -> list[dict[int, torch.Tensor]]:
    # per query and read-out layer the node logits, one row per <SEL>: shape (n_sel, n)
    layers = readout.layers if layers is None else layers
    out = []
    for start in range(0, len(queries), batch_size):
        chunk = queries[start:start + batch_size]
        seqs, nb = src.image_item(m, chunk)
        batch = {k: v.to(device) for k, v in S2.collate(seqs, src.pad_id).items()}
        with S2.autocast(device):
            res = run_forward(model, readout, batch, [nb] * len(seqs))
        for b, s in enumerate(seqs):
            out.append({k: readout.node_logits(k, res.hidden[k][b, s.sel.to(device)].float(),
                                               res.nodes[k][b].float(), nb).float().cpu() for k in layers})
    return out


SINGLE_KINDS = ("object", "part")           # the checkpoint-selection kinds (S2 rule)


def chain_rows(q: S2T.Query, lg: torch.Tensor, parents: np.ndarray, layer: int) -> list[dict]:
    # val rows of a supplied object -> part chain (D44): the object pick, the flat part pick and the part pick
    # under every anchor of TA.ANCHORS; q.ious is (2, n)
    obj, part = lg[0].numpy(), lg[1].numpy()
    a1, flat = TA.anchor(obj, part, parents, None)
    base = {"qid": q.qid, "layer": layer}
    rows = [{**base, "kind": "chain_object", "sel_iou": float(q.ious[0, a1]), "best_iou": float(q.ious[0].max())},
            {**base, "kind": "chain_part_none", "sel_iou": float(q.ious[1, flat]), "best_iou": float(q.ious[1].max())}]
    for mode in TA.ANCHORS:
        _, a2 = TA.anchor(obj, part, parents, mode)
        rows.append({**base, "kind": f"chain_part_{TA.anchor_name(mode)}", "sel_iou": float(q.ious[1, a2]),
                     "best_iou": float(q.ious[1].max())})
    return rows


@torch.no_grad()
def val_success(model, readout, src, metas, device, world, batch_size) -> dict:
    model.eval()
    readout.eval()
    rows = []
    for m in metas:
        if m.n == 0:
            for q in m.queries:
                for k in readout.layers:
                    if q.kind == "chain":
                        rows += [{"qid": q.qid, "kind": kind, "layer": k, "sel_iou": 0.0, "best_iou": 0.0}
                                 for kind in ["chain_object", "chain_part_none"]
                                 + [f"chain_part_{TA.anchor_name(a)}" for a in TA.ANCHORS]]
                    else:
                        rows.append({"qid": q.qid, "kind": q.kind, "layer": k, "sel_iou": 0.0, "best_iou": 0.0})
            continue
        for q, lg in zip(m.queries, query_logits(model, readout, src, m, m.queries, device, batch_size)):
            for k in readout.layers:
                if q.kind == "chain":
                    rows += chain_rows(q, lg[k], m.parents, k)
                    continue
                a = int(torch.argmax(lg[k][0]))
                rows.append({"qid": q.qid, "kind": q.kind, "layer": k, "sel_iou": float(q.ious[a]),
                             "best_iou": float(q.ious.max())})
    model.train()
    readout.train()
    df = pd.DataFrame([r for part in S2T.gather(rows, world) for r in part])
    by_layer = {}
    for k, g in df.groupby("layer"):
        single = g[g.kind.isin(SINGLE_KINDS)]
        cov = single[single.best_iou >= 0.5]
        covered = g[g.best_iou >= 0.5]
        by_kind = {kk: float((gg.sel_iou >= 0.5).mean()) for kk, gg in covered.groupby("kind")}
        parts = {a: by_kind.get(f"chain_part_{TA.anchor_name(a)}") for a in (None,) + TA.ANCHORS}
        anchor = None
        if all(v is not None for v in parts.values()):
            # the part decision for evaluation: best covered chain-part success; ties go to the earlier rule
            # (none, hard, then the smaller bonuses)
            anchor = TA.anchor_name(max(parts, key=lambda a: (parts[a], -((None,) + TA.ANCHORS).index(a))))
        by_layer[int(k)] = {"selector_success": float((cov.sel_iou >= 0.5).mean()) if len(cov) else 0.0,
                            "by_kind": by_kind, "covered_queries": int(len(cov)), "queries": int(len(single)),
                            "chains": int((g.kind == "chain_object").sum()), "anchor": anchor}
    best = max(by_layer, key=lambda k: (by_layer[k]["selector_success"], -k))   # ties: earlier layer
    return {"by_layer": by_layer, "best_layer": best, "selector_success": by_layer[best]["selector_success"],
            "anchor": by_layer[best]["anchor"],
            "conditioned_on": "a proposal with IoU >= 0.5 exists; argmax over C_single"}


def pool_cache(cfg: dict, source: str, smoke: bool) -> tuple[Path, list[Path]]:
    d = S3P.out_dir(cfg, S3P.identity(cfg, source, smoke))
    files = sorted((d / "shards").glob("*.pt"))
    n = S3P.num_shards(cfg, source, smoke)
    if len(files) != n:
        raise ProtocolError(f"{source}: {len(files)} of {n} s3_pool shards present in {d} (run scripts/s3_pool.py)")
    return d, files


def pool_meta(cfg: dict, source: str, smoke: bool) -> tuple[list[PoolMeta], dict]:
    # training items over the D30 pool of `source` (s3_pool records: pool node file, tree, step targets over the
    # pool, PACO part labels over the pool, instance provenance); the item keeps the plain source name
    d, files = pool_cache(cfg, source, smoke)
    metas, counts = [], {}
    empty_step = S.StepTarget(np.zeros(0, np.int64), np.zeros(0, np.int64))
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in blob["counts"].items():
            counts[f"{source}_pool:{k}"] = counts.get(f"{source}_pool:{k}", 0) + v
        for r in blob["records"]:
            parents = np.asarray(r["parent"], np.int64)
            n = len(parents) - 1
            if r["n_nodes"] + len(r["instances"]) != n:
                raise ProtocolError(f"{r['key']}: {r['n_nodes']} nodes + {len(r['instances'])} instances for a "
                                    f"pool of {n}")
            empty = np.full(n, "", dtype=object)
            labels = np.asarray(r["labels"], dtype=object) if "labels" in r else empty
            parts = np.asarray(r["parts"], dtype=object) if "parts" in r else empty
            high = np.asarray(r["high"], dtype=bool) if "high" in r else np.zeros(n, dtype=bool)
            if len(labels) != n:
                raise ProtocolError(f"{r['key']}: {len(labels)} labels for {n} pool entries")
            base = dict(sha1=r["key"], key=r["key"], file=r["path"], n=n, parents=parents, labels=labels, parts=parts,
                        high=high, source=source, nodes=str(d / r["nodes"]),
                        inst=TA.pool_features(r["n_nodes"], r["instances"]))
            if r["queries"]:
                metas.append(PoolMeta(queries=[S2T.query_from_cache(q) for q in r["queries"]], **base))
            for a in r["absent"]:
                q = S2T.Query(a["qid"], "absent", S2.user_text(a["text"]),
                              S2.answer_text("absent", a["qid"], a["text"]), [empty_step])
                metas.append(PoolMeta(queries=[q], absent=True, **base))
    counts[f"{source}_pool:training_items"] = len(metas)
    return metas, counts


def pool_val_meta(cfg, wl, props, queries, device, n_images: int, seed: int, rank: int, world: int, text
                  ) -> list[PoolMeta]:
    # the S2T.val_meta image subset (same seed, same images), each over its query-aware pool built as s3_pool does
    # (grid nodes + the image's val query phrases' top-k SAM 3 text instances); object / part queries carry the
    # pool IoUs of their target, chains the (2, n) IoUs of object and part
    s = S3P.settings(cfg)
    tau = cfg["proposals"]["node_dedup"]
    qval = queries[(queries.split == "val") & queries.kind.isin(SINGLE_KINDS)]
    keys = wl[(wl.role == "paco_val") & wl.sha1.isin(set(qval.sha1))].sort_values("image_key")
    pick = np.sort(np.random.default_rng(seed).choice(len(keys), size=min(n_images, len(keys)), replace=False))
    rows = list(keys.iloc[pick].itertuples())[rank::world]
    images = paco.load(cfg["data"]["paco_lvis"], "val", {int(r.image_key.split(":")[1]) for r in rows})
    qall = queries[(queries.split == "val") & queries.kind.isin(SINGLE_KINDS + ("chain",))]
    by_sha = {k: g for k, g in qall.groupby("sha1")}
    out = []
    for r in rows:
        img = images[int(r.image_key.split(":")[1])]
        g = by_sha[r.sha1].sort_values("qid")
        image = load_rgb(assert_real_path(r.path))
        if image.shape[:2] != (img.h, img.w):
            raise ProtocolError(f"{r.image_key}: image {image.shape[:2]} differs from the annotation")
        nodes = props.nodes(r.sha1, tau)["rles"]
        plist = list(dict.fromkeys(p for t in g.text for p in S3P.phrases(t)))
        pool, imeta = S3P.augment(nodes, text(image, plist, s["instances_per_phrase"]), s["max_instances"],
                                  s["dup_iou"])
        parents = T.build_tree(pool, device).parent if pool else None
        gts = paco.gt_rles(img)
        ids = sorted({a for q in g.itertuples() for a in json.loads(q.gt)})
        col = {a: i for i, a in enumerate(ids)}
        all_ious = M.iou(pool, [gts[a] for a in ids]).T if pool and ids else np.zeros((len(ids), len(pool)))
        qs = []
        for q in g.itertuples():
            ious = all_ious[[col[a] for a in json.loads(q.gt)]]
            qs.append(S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [],
                                None, ious if q.kind == "chain" else ious[0]))
        out.append(PoolMeta(r.sha1, r.image_key, r.path, len(pool), parents, None, None, None, qs,
                            inst=TA.pool_features(len(nodes), imeta), rles=pool))
    return out


def rest_meta(cfg: dict, smoke: bool) -> tuple[list[S2T.ImageMeta], dict]:
    # PACO-LVIS train images outside the worklist (ta_paco_nodes cache); they join the paco_train source
    d = TPN.out_dir(cfg, smoke)
    files = sorted((d / "shards").glob("*.pt"))
    n = TPN.num_shards(cfg, smoke)
    if len(files) != n:
        raise ProtocolError(f"{len(files)} of {n} PACO train node shards present in {d} (run scripts/ta_paco_nodes.py)")
    metas, counts = [], {}
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        for k, v in blob["counts"].items():
            counts[f"paco_train_rest:{k}"] = counts.get(f"paco_train_rest:{k}", 0) + v
        for r in blob["records"]:
            if not r["queries"]:
                continue
            parents = np.asarray(r["parent"], np.int64)
            metas.append(S2T.ImageMeta(r["key"], r["key"], r["path"], len(parents) - 1, parents,
                                       np.asarray(r["labels"], dtype=object), np.asarray(r["parts"], dtype=object),
                                       np.asarray(r["high"], dtype=bool),
                                       [S2T.query_from_cache(q) for q in r["queries"]],
                                       source="paco_train", nodes=str(d / r["nodes"])))
    counts["paco_train_rest:training_items"] = len(metas)
    return metas, counts


def add_missing_trees(items: list[S2T.ImageMeta], props, tau: float, device: str) -> int:
    # no-target items of the worklist carry no tree (S2 did not need one); build it from the same nodes
    n = 0
    for m in items:
        if m.parents is None:
            rles = S2T.read_nodes(m.nodes)["rles"] if m.nodes else props.nodes(m.sha1, tau)["rles"]
            m.parents = T.build_tree(rles, device).parent
            n += 1
    return n


def checkpoint_state(model, extra, readout: Readout, meta: dict) -> dict:
    return {**LR.trainable_state(model, extra["sel_emb"], readout), "meta": meta}


def build(cfg: dict, c: dict, device: str, static: dict | None, n_nodes: int) -> tuple:
    model, processor, extra = S2T.build_model(cfg, c, device, static)
    d_model = model.config.text_config.hidden_size
    mllm.check_layers(model, c["readout_layers"])
    v = c["variant"]
    d_geom = TA.GEOM_DIM + (TA.INST_DIM if is_pool(v) else 0)
    embed = None if v in FLAT else TA.NodeEmbed(d_model, TA.embedding_rms(model), d_geom)
    mode, fam = D43.get(v) or D44.get(v) or ("", False)
    soft = TA.SoftTreeBias(model.config.text_config.num_hidden_layers) if mode.startswith("soft") else None
    family = (float(c["fam_margin"]), float(c["fam_weight"])) if fam else None
    readout = Readout(d_model, c["readout_layers"], n_nodes, embed, v in HIER, soft, family).to(device)
    if soft is not None:
        readout.soft.install(model)
    return model, processor, extra, readout


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    c = stage_cfg(cfg, args.smoke)
    rank, world, _ = S2T.dist_env()
    device = setup_device(cfg)
    if rank == 0:
        ensure_deviation_log(cfg)
    if world > 1:
        dist.barrier()
    require_approved(cfg, run_deviations(c))
    TA.check_version()
    gate = v1_gate(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qcfg = query_cfg(cfg)
    qdir, queries = open_queries(qcfg, args.smoke, mhash, qwen_static(qcfg) if qcfg is not cfg else static)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(c["seed"])
    cap = node_cap(cfg, args.smoke)
    n_nodes = int(cap["k"]) if cap else 256   # only sets the initial bias, -log K
    model, processor, extra, readout = build(cfg, c, device, static, n_nodes)
    readout.train()
    model.train()
    lora_params = LR.trainable_parameters(model, extra["sel_emb"])
    soft_params = list(readout.soft.parameters()) if readout.soft is not None else []
    head_params = [p for p in readout.parameters() if all(p is not q for q in soft_params)]
    params = lora_params + head_params + soft_params
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    torch.manual_seed(c["seed"] * 1000 + rank)  # LoRA dropout masks differ across ranks

    text_labels = set(queries[queries.kind == "text"].label)
    weights = S2T.source_weights(c)
    pool = is_pool(c["variant"])
    if pool:
        if c["paco_rest"]:
            raise ProtocolError("pool variants train on the D30 pool caches alone: set ta.paco_rest to false")
        if int(S3P.settings(cfg)["instances_per_phrase"]) > TA.MAX_RANK:
            raise ProtocolError(f"pool provenance encodes ranks up to {TA.MAX_RANK}; s3_pool.instances_per_phrase "
                                f"is {S3P.settings(cfg)['instances_per_phrase']}")
        items, counts = [], {}
        for source in weights:
            metas, pc = pool_meta(cfg, source, args.smoke)
            items += metas
            counts.update(pc)
    else:
        items, counts = S2T.all_train_meta(cfg, wl, props, queries, device, rank, world, list(weights), args.smoke)
        if c["paco_rest"]:
            rest, rest_counts = rest_meta(cfg, args.smoke)
            items += rest
            counts.update(rest_counts)
    counts["trees_built_for_no_target_items"] = add_missing_trees(items, props, cfg["proposals"]["node_dedup"], device)
    if not items:
        raise ProtocolError("no training items")
    if pool:
        text = S3P.TextInstances(cfg)      # SAM 3 text, only to build the val pools once
        vmetas = pool_val_meta(cfg, wl, props, queries, device, c["eval_images"], c["seed"], rank, world, text)
        del text
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    else:
        vmetas = S2T.val_meta(cfg, wl, props, queries, device, c["eval_images"], c["seed"], rank, world)
    src = Source(cfg, args.smoke, manifest, mhash, processor, extra["sel_id"], text_labels, c["variant"], c["seed"])
    entries = S2T.plan(items, c, text_labels, world, rank)
    accum = c["batch_images"] // (world * c["micro_batch"])
    loader = torch.utils.data.DataLoader(MicroBatches(entries, items, src), batch_size=None, shuffle=False,
                                         num_workers=c["workers"], collate_fn=lambda x: x,
                                         prefetch_factor=4 if c["workers"] else None)

    groups = [{"params": head_params, "lr": c["head_lr"], "weight_decay": c["head_weight_decay"]},
              {"params": lora_params, "lr": c["lora_lr"], "weight_decay": c["lora_weight_decay"]}]
    if soft_params:   # D43 relation biases: few parameters, larger steps, not decayed towards full attention
        groups.append({"params": soft_params, "lr": c["soft_lr"], "weight_decay": 0.0})
    opt = torch.optim.AdamW(groups)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, V2.lr_lambda(c["warmup_steps"], c["steps"]))
    run = None
    if rank == 0:
        run = start_run(stage(cfg), cfg, args.smoke, {
            "manifest_run": str(mrun), "manifest_sha256": mhash, "queries": str(qdir), "variant": c["variant"],
            "readout_layers": c["readout_layers"], "world_size": world, "accumulation": accum,
            "sel_init": extra["sel_init"], "lora_modules": len(extra["wrapped"]), "tied_embeddings": extra["tied"],
            "init": "from scratch (base Qwen3-VL, fresh LoRA, random heads)"})
    run = Path(S2T.broadcast(str(run) if run else None, world))
    ckpt_meta = {"variant": c["variant"], "readout_layers": c["readout_layers"], "n_nodes": n_nodes,
                 "d_model": model.config.text_config.hidden_size, "lora": c["lora"], "sel_id": extra["sel_id"],
                 "sel_init": extra["sel_init"], "qwen": cfg["models"]["qwen"], "smoke": args.smoke,
                 "seed": c["seed"], "config_hash": canonical_hash(c), "pool": pool,
                 "pool_settings": {k: S3P.settings(cfg)[k] for k in ("instances_per_phrase", "max_instances",
                                                                     "dup_iou")} if pool else None}

    rng = np.random.default_rng([c["seed"], rank])
    it = iter(loader)
    log, evals = [], []
    t0 = time.time()

    def evaluate_and_save(step: int, best: tuple) -> tuple:
        res = val_success(model, readout, src, vmetas, device, world, c["eval_batch"])
        if rank == 0:
            state = checkpoint_state(model, extra, readout, {**ckpt_meta, "step": step, "val": res,
                                                             "layer": res["best_layer"], "anchor": res["anchor"]})
            torch.save(state, run / f"step{step:06d}.pt")
            if res["selector_success"] > best[0]:
                best = (res["selector_success"], step, res["best_layer"])
                torch.save(state, run / "best.pt")
            evals.append({"step": step, **res})
            print(f"eval step {step}: selector success {res['selector_success']:.4f} at layer {res['best_layer']} "
                  + " ".join(f"L{k} {v['selector_success']:.4f}" for k, v in res["by_layer"].items())
                  + (f" anchor {res['anchor']}" if res["anchor"] else ""), flush=True)
            pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
            write_json(run / "eval_curve.json", evals)
        return best

    best = evaluate_and_save(0, (-1.0, None, None))
    for step in range(c["steps"]):
        stats = {k: 0 for k in S2T.STAT_KEYS}
        sums = torch.zeros(2, device=device)
        for _ in range(accum):
            l_sel, l_lm = micro_step(model, extra, readout, next(it), items, device, rng, c["gamma"], stats)
            ((l_sel + l_lm) / accum).backward()
            sums += torch.stack([l_sel.detach(), l_lm.detach()]) / accum
        S2T.sync_grads(params, world)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, c["max_grad_norm"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(sums)
            sums /= world
        if rank == 0 and (step % c["log_every"] == 0 or step == c["steps"] - 1):
            print(f"step {step}/{c['steps']} L_sel {sums[0].item():.4f} L_LM {sums[1].item():.4f} "
                  f"{time.time() - t0:.0f}s", flush=True)
            log.append({"step": step, "l_sel": sums[0].item(), "l_lm": sums[1].item(), "grad_norm": float(grad_norm),
                        "lr_heads": sched.get_last_lr()[0], "lr_lora": sched.get_last_lr()[1],
                        "seconds": time.time() - t0, **stats})
        if (step + 1) % c["eval_every"] == 0 or step + 1 == c["steps"]:
            best = evaluate_and_save(step + 1, best)
    if rank == 0:
        write_json(run / "metrics.json", {
            "label": "dev", "smoke": args.smoke, "variant": c["variant"], "best_step": best[1],
            "best_layer": best[2], "best_checkpoint": "best.pt", "best_val_selector_success": best[0],
            "pool": pool, "anchor": torch.load(run / "best.pt", map_location="cpu")["meta"]["anchor"],
            "selection": "PACO val fixed subset, single-step selector success, best over steps and read-out layers",
            "eval_images": c["eval_images"], "eval_curve": evals, "training_exclusions": counts,
            "training_items": len(items), "world_size": world, "accumulation": accum, "v1_gate": gate,
            "sources": weights, "deviations": run_deviations(c)})
        print(run)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
