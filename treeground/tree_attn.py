import math
from dataclasses import dataclass
from functools import partial

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import transformers

from treeground import masks as M
from treeground import mllm
from treeground import tree as T
from treeground.selector import TEMPERATURE, Selector, focal, selection_loss

# Tree attention (D41): one input token per candidate mask, appended after the answer. A mask token attends to the
# text, to the image tokens of its own region and to its tree relatives; no earlier token attends to mask tokens.
# The forward replicates Qwen3VLModel.forward of the pinned transformers release for images (no video, no cache)
# so that the language model can be called with the extended sequence and a custom 4-D mask.

PINNED_TRANSFORMERS = "5.0.0"
MODES = ("tree", "full", "random_tree")        # training variants with node tokens
RELATIONS = MODES + ("self",)                  # + evaluation-time swap: node tokens see only themselves
# D43 modes: every node token sees every other (as full); 'soft' adds learned per-layer biases by tree relation
# (SoftTreeBias); 'full_random_tree' / 'soft_random_tree' build geometry, relations and family targets on the
# shuffled tree (controls)
D43_MODES = ("soft", "soft_random_tree", "full_random_tree")
REL_SELF, REL_PARENT, REL_CHILD, REL_SIBLING, REL_ANC, REL_DESC, REL_OTHER = range(7)
N_REL = 7
GEOM_DIM = 9
REGION_DILATE = 1   # image-token cells of margin around a mask's own cells
# D44 query-aware pool: per-node provenance next to the geometry: [from SAM 3 text, rank 1, rank 2, rank 3 among the
# phrase's instances, SAM 3 instance score]; all zero for grid nodes
INST_DIM = 5
MAX_RANK = 3
HARD = "hard"                              # anchor: the part is chosen inside the selected object's subtree
ANCHORS = (HARD, 0.5, 1.0, 2.0, 4.0)       # val grid of the D44 part decision (soft: logit bonus on that subtree)


@dataclass
class Nodes:
    cov: tuple[np.ndarray, np.ndarray, np.ndarray]   # sparse coverage on the merged grid (mllm.sparse_coverages)
    region: list[np.ndarray]                         # grid cells each node attends to
    rel: list[np.ndarray]                            # node indices each node attends to, itself included
    geom: np.ndarray                                 # (n, GEOM_DIM) float32
    cell: np.ndarray                                 # (n, 2) int64 (row, col) of the coverage centroid
    anc: np.ndarray                                  # (n, n) bool, anc[v, u]: u is a strict ancestor of v in the
                                                     # tree behind `rel` (the true tree, or the shuffled one)
    par: np.ndarray | None = None                    # (n,) parent in that tree, -1 for top-level nodes
    relk: np.ndarray | None = None                   # (n, n) int8 tree relation of u to v (D43 soft modes)
    inst: np.ndarray | None = None                   # (n, INST_DIM) float32 pool provenance (D44), else None

    @property
    def n(self) -> int:
        return len(self.rel)

    @property
    def features(self) -> np.ndarray:
        # the node-embedding input: geometry, plus provenance for a query-aware pool
        return self.geom if self.inst is None else np.concatenate([self.geom, self.inst], 1)


def pool_features(n_grid: int, instances: list[dict]) -> np.ndarray:
    # (n_grid + len(instances), INST_DIM): s3_pool.augment's meta (phrase, rank, score) in pool order after the grid
    out = np.zeros((n_grid + len(instances), INST_DIM), np.float32)
    for i, m in enumerate(instances):
        rank = int(m["rank"])
        if not 0 <= rank < MAX_RANK:
            raise ValueError(f"instance rank {rank} outside 0..{MAX_RANK - 1}")
        out[n_grid + i, 0] = 1.0
        out[n_grid + i, 1 + rank] = 1.0
        out[n_grid + i, 1 + MAX_RANK] = float(m["score"])
    return out


def anchor_name(mode) -> str:
    return "none" if mode is None else HARD if mode == HARD else f"soft{float(mode):g}"


def anchor_mode(name: str):
    if name == "none":
        return None
    if name == HARD:
        return HARD
    if name.startswith("soft"):
        return float(name[4:])
    raise ValueError(f"unknown anchor {name!r}")


def anchor(obj_logits: np.ndarray, part_logits: np.ndarray, parent: np.ndarray, mode) -> tuple[int, int]:
    # D44 part decision: the object token picks a1 = argmax; the part token picks a2 inside a1's subtree (mode HARD),
    # or over every node with `mode` added to the logits of that subtree (a float), or flat (mode None); an empty
    # subtree falls back to the flat pick. Returns (a1, a2); (-1, -1) without nodes.
    n = len(part_logits)
    if n == 0:
        return -1, -1
    a1 = int(np.argmax(obj_logits))
    if mode is None:
        return a1, int(np.argmax(part_logits))
    desc = T.from_parents(parent).strict_descendants(a1)
    if len(desc) == 0:
        return a1, int(np.argmax(part_logits))
    if mode == HARD:
        return a1, int(desc[int(np.argmax(part_logits[desc]))])
    bonus = np.zeros(n)
    bonus[desc] = float(mode)
    return a1, int(np.argmax(part_logits + bonus))


def relatives(tree: T.Tree, mode: str) -> list[np.ndarray]:
    n = tree.n
    if mode == "full":
        return [np.arange(n, dtype=np.int64) for _ in range(n)]
    if mode == "self":
        return [np.array([v], dtype=np.int64) for v in range(n)]
    out = []
    for v in range(n):
        p = int(tree.parent[v])
        rel = [v, *tree.children[v].tolist(), *tree.children[p].tolist()]   # self, children, siblings (with v)
        if p != tree.root:
            rel.append(p)
        out.append(np.unique(np.asarray(rel, dtype=np.int64)))
    return out


def ancestors(tree: T.Tree) -> np.ndarray:
    # (n, n) boolean, [v, u] = u is a strict ancestor of v (the virtual root excluded)
    n = tree.n
    out = np.zeros((n, n), bool)
    for v in range(n):
        u = int(tree.parent[v])
        while u != tree.root:
            out[v, u] = True
            u = int(tree.parent[u])
    return out


def relation_matrix(tree: T.Tree) -> np.ndarray:
    # (n, n) int8: [v, u] = relation of u to v. Direct links override further ones; siblings share a real parent
    # (top-level nodes are 'other' to each other: the virtual root is the whole image, not a part-whole link)
    n = tree.n
    anc = ancestors(tree)
    par = tree.parent[:n]
    out = np.full((n, n), REL_OTHER, np.int8)
    out[(par[:, None] == par[None, :]) & (par[:, None] != tree.root)] = REL_SIBLING
    out[anc] = REL_ANC
    out[anc.T] = REL_DESC
    for v in range(n):
        if par[v] != tree.root:
            out[v, par[v]] = REL_PARENT
            out[par[v], v] = REL_CHILD
    out[np.arange(n), np.arange(n)] = REL_SELF
    return out


def geometry(rles: list[M.Rle], tree: T.Tree, h: int, w: int) -> np.ndarray:
    # label-free per-node inputs: normalised box, log area fraction, log(1 + depth / #children / #siblings) and
    # the log area ratio to the parent (the whole image for top-level nodes)
    n = len(rles)
    if n == 0:
        return np.zeros((0, GEOM_DIM), np.float32)
    area = M.areas(rles).astype(np.float64)
    box = M.boxes_xyxy(rles) / np.array([w, h, w, h], np.float64)
    parent_area = np.array([h * w if tree.parent[v] == tree.root else area[tree.parent[v]] for v in range(n)])
    n_children = np.array([len(tree.children[v]) for v in range(n)])
    n_siblings = np.array([len(tree.children[tree.parent[v]]) - 1 for v in range(n)])
    g = np.column_stack([box, np.log(area / (h * w)), np.log1p(tree.depth[:n]), np.log1p(n_children),
                         np.log1p(n_siblings), np.log(area / parent_area)])
    return g.astype(np.float32)


def build_nodes(rles: list[M.Rle], parent: np.ndarray, grid: tuple[int, int], mode: str,
                tree_seed: int | None = None, inst: np.ndarray | None = None) -> Nodes:
    if mode not in RELATIONS + D43_MODES:
        raise ValueError(f"unknown tree-attention mode {mode!r}")
    gh, gw = grid
    h, w = rles[0]["size"] if rles else (1, 1)
    tree = T.from_parents(parent)
    if tree.n != len(rles):
        raise ValueError(f"tree has {tree.n} nodes for {len(rles)} masks")
    if inst is not None:
        inst = np.asarray(inst, np.float32)
        if inst.shape != (len(rles), INST_DIM):
            raise ValueError(f"pool features {inst.shape} for {len(rles)} masks")
    if mode.endswith("random_tree"):
        if tree_seed is None:
            raise ValueError("random_tree needs a per-image seed")
        tree = T.randomised_tree(tree, tree_seed)   # same depth profile, parents shuffled within each depth
    maps = [mllm.coverage(M.decode(r), gh, gw) for r in rles]
    kernel = np.ones((2 * REGION_DILATE + 1,) * 2, np.uint8)
    region, cell = [], np.zeros((len(rles), 2), np.int64)
    rows, cols = np.divmod(np.arange(gh * gw), gw)
    for v, c in enumerate(maps):
        if not (c > 0).any():
            raise ValueError("node mask has zero coverage on the token grid")
        grown = cv2.dilate((c > 0).reshape(gh, gw).astype(np.uint8), kernel).reshape(-1)
        region.append(np.flatnonzero(grown))
        cell[v] = [int(round(float(c @ rows) / c.sum())), int(round(float(c @ cols) / c.sum()))]
    ptr, idx, wts = [0], [], []
    for c in maps:
        nz = np.flatnonzero(c > 0)
        idx.append(nz.astype(np.int32))
        wts.append(c[nz].astype(np.float32))
        ptr.append(ptr[-1] + len(nz))
    cov = (np.asarray(ptr, np.int64), np.concatenate(idx) if idx else np.zeros(0, np.int32),
           np.concatenate(wts) if wts else np.zeros(0, np.float32))
    seen = "self" if mode == "self" else ("tree" if mode in ("tree", "random_tree") else "full")
    par = np.where(tree.parent[:tree.n] == tree.root, -1, tree.parent[:tree.n]).astype(np.int64)
    return Nodes(cov, region, relatives(tree, seen), geometry(rles, tree, h, w), cell, ancestors(tree), par,
                 relation_matrix(tree) if mode.startswith("soft") else None, inst)


class NodeEmbed(nn.Module):
    # x_v = MLP([mean image-token input embedding under v ; geometry MLP(g_v)]) + kind, rescaled to RMS `gain`
    def __init__(self, d_model: int, gain: float, d_geom: int = GEOM_DIM, hidden: int = 256):
        super().__init__()
        self.geom = nn.Sequential(nn.Linear(d_geom, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.mix = nn.Sequential(nn.Linear(d_model + hidden, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.kind = nn.Parameter(torch.zeros(d_model))
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    @property
    def d_geom(self) -> int:
        return self.geom[0].in_features

    def forward(self, pooled: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
        if geom.shape[-1] != self.d_geom:
            raise ValueError(f"node features of width {geom.shape[-1]} for an embedding of width {self.d_geom} "
                             "(a pool checkpoint needs pool features and vice versa)")
        x = self.mix(torch.cat([pooled, self.geom(geom)], -1)) + self.kind
        return x * (self.gain * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6))


@torch.no_grad()
def embedding_rms(model, chunk: int = 8192) -> float:
    # mean per-row RMS of the token embedding matrix, in chunks (the full fp32 copy would be 1.5 GB for Qwen3-VL)
    w = model.model.get_input_embeddings().weight
    total = sum(float(w[s:s + chunk].float().pow(2).mean(-1).sqrt().sum()) for s in range(0, w.shape[0], chunk))
    return total / w.shape[0]


class ScaledSelector(Selector):
    # cosine score with a learned scale (initialised to 1 / 0.07, at most 100) and bias (initialised to -log K)
    def __init__(self, d_node: int, d_query: int, n_nodes: int):
        super().__init__(d_node, d_query)
        self.log_scale = nn.Parameter(torch.tensor(math.log(1.0 / TEMPERATURE)))
        self.bias = nn.Parameter(torch.tensor(-math.log(n_nodes)))

    def scores(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ e.T * self.log_scale.exp().clamp(max=100.0) + self.bias


def hier_logits(stop: torch.Tensor, desc: torch.Tensor, root: torch.Tensor, anc: torch.Tensor) -> torch.Tensor:
    # D42 decision: P(answer = v) = sigma(root) * prod_{u strict ancestor of v} sigma(desc_u) * sigma(stop_v).
    # stop, desc: (S, n) logits, root: (S,), anc: (n, n) bool. Returns the logit of that probability, so that
    # argmax, sigmoid thresholds and the S2 evaluators read it like any other node logit.
    score = (F.logsigmoid(stop) + F.logsigmoid(desc) @ anc.to(desc.dtype).T + F.logsigmoid(root)[:, None])
    score = score.clamp(max=-1e-6)                     # log P < 0; keeps the logit finite
    return score - torch.log(-torch.expm1(score))


class HierHead(nn.Module):
    # stop / descend scorers (cosine with learned scale and bias, as ScaledSelector) and a root "a target exists"
    # logit from the <SEL> state; trained with S2 stop targets plus descend targets on the target's ancestors
    def __init__(self, d_model: int, n_nodes: int, hidden: int = 256):
        super().__init__()
        self.stop = ScaledSelector(d_model, d_model, n_nodes)
        self.desc = ScaledSelector(d_model, d_model, n_nodes)
        self.root = nn.Sequential(nn.Linear(d_model, hidden), nn.GELU(), nn.Linear(hidden, 1))

    def descend(self, probe: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        return self.desc.scores(self.desc.queries(probe), self.desc.nodes(states))

    def root_logit(self, probe: torch.Tensor) -> torch.Tensor:
        return self.root(probe.float())[:, 0]

    def logits(self, probe: torch.Tensor, states: torch.Tensor, anc: torch.Tensor) -> torch.Tensor:
        stop = self.stop.scores(self.stop.queries(probe), self.stop.nodes(states))
        return hier_logits(stop, self.descend(probe, states), self.root_logit(probe), anc)


def descend_targets(anc: np.ndarray, pos: np.ndarray, ignore: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # descend positives: every strict ancestor of a positive; ignored: ancestors of ignored near-copies only;
    # negatives: every other node (the targets themselves, their descendants, all other branches)
    n = anc.shape[0]
    up = anc[pos].any(0) if len(pos) else np.zeros(n, bool)
    up_ignored = anc[ignore].any(0) if len(ignore) else np.zeros(n, bool)
    return np.flatnonzero(up), np.flatnonzero(~up & ~up_ignored)


def hier_terms(head: HierHead, probe: torch.Tensor, states: torch.Tensor, anc: np.ndarray, steps: list,
               gamma: float) -> list[torch.Tensor]:
    # one descend term and one root term per <SEL> step (probe row t, step targets steps[t])
    device = states.device
    e = head.desc.nodes(states)
    out = []
    for t, st in enumerate(steps):
        s = head.desc.scores(head.desc.queries(probe[t:t + 1]), e)[0]
        pos, neg = descend_targets(anc, st.pos, st.ignore)
        loss, _, _ = selection_loss(s[torch.as_tensor(pos, device=device, dtype=torch.long)],
                                    s[torch.as_tensor(neg, device=device, dtype=torch.long)], gamma)
        out.append(loss + focal(head.root_logit(probe[t:t + 1]), int(len(st.pos) > 0), gamma).mean())
    return out


class SoftTreeBias(nn.Module):
    # D43: a learned additive attention bias between node tokens per decoder layer and tree relation (7 relations,
    # initialised to 0 = full attention). Pre-forward hooks on the decoder layers add bias[layer, relation(v, u)] to
    # the node-token block of the 4-D mask; the context of a forward stays set until the next one so that gradient
    # checkpointing recomputes the same masks in backward.
    def __init__(self, n_layers: int):
        super().__init__()
        self.bias = nn.Parameter(torch.zeros(n_layers, N_REL))
        self._ctx = None
        self._hooks = []

    def install(self, model) -> None:
        if self._hooks:
            raise RuntimeError("SoftTreeBias is already installed")
        layers = model.model.language_model.layers
        if len(layers) != self.bias.shape[0]:
            raise ValueError(f"{len(layers)} decoder layers for {self.bias.shape[0]} bias rows")
        for i, layer in enumerate(layers):
            self._hooks.append(layer.register_forward_pre_hook(partial(self._pre, i), with_kwargs=True))

    def clear(self) -> None:
        self._ctx = None

    def set(self, nodes: list, L: int, K: int, device) -> None:
        if K == 0:
            self._ctx = None
            return
        B = len(nodes)
        rel = torch.zeros((B, K, K), dtype=torch.long)
        valid = torch.zeros((B, K, K), dtype=torch.bool)
        for b, nb in enumerate(nodes):
            if nb is not None and nb.n:
                if nb.relk is None:
                    raise ValueError("soft tree attention needs node relations (a soft mode)")
                rel[b, :nb.n, :nb.n] = torch.from_numpy(nb.relk.astype(np.int64))
                valid[b, :nb.n, :nb.n] = True
        self._ctx = (rel.to(device), valid.to(device), L, K)

    def _pre(self, i: int, module, args, kwargs):
        if self._ctx is None:
            return None
        rel, valid, L, K = self._ctx
        m = kwargs.get("attention_mask")
        if m is None or m.dim() != 4 or m.shape[-1] != L + K or m.shape[0] != rel.shape[0]:
            raise RuntimeError("soft tree bias context does not match this forward")
        add = torch.where(valid, self.bias[i][rel], torch.zeros((), device=rel.device)).to(m.dtype)
        out = m.clone()
        out[:, 0, L:, L:] = m[:, 0, L:, L:] + add
        kwargs["attention_mask"] = out
        return args, kwargs


def family_terms(sel, probe: torch.Tensor, states: torch.Tensor, nb: "Nodes", steps: list, margin: float
                 ) -> list[torch.Tensor]:
    # D43 family loss: every positive's cosine with the query must exceed that of its own parent and children (in
    # the tree behind nb) by `margin`; cosines, not scores, so the learned scale cannot satisfy it without changing
    # the ranking; family members that are positives or ignored near-copies are left out
    e = sel.nodes(states)
    out = []
    for t, st in enumerate(steps):
        if not len(st.pos):
            continue
        s = (sel.queries(probe[t:t + 1]) @ e.T)[0]
        excl = set(int(x) for x in st.pos) | set(int(x) for x in st.ignore)
        for p in (int(x) for x in st.pos):
            fam = ([int(nb.par[p])] if nb.par[p] >= 0 else []) + [int(c) for c in np.flatnonzero(nb.par == p)]
            fam = [f for f in fam if f not in excl]
            if fam:
                idx = torch.as_tensor(fam, device=s.device, dtype=torch.long)
                out.append(F.relu(margin - (s[p] - s[idx])).mean())
    return out


@dataclass
class Output:
    hidden: dict[int, torch.Tensor]   # layer -> (B, L, D) states of the text and image positions
    nodes: dict[int, list[torch.Tensor]]   # layer -> per sample (n_b, D) mask-token states
    last: torch.Tensor                # (B, L, D) final normed states (LM loss)


def check_version() -> None:
    if transformers.__version__ != PINNED_TRANSFORMERS:
        raise RuntimeError(f"tree attention replicates Qwen3VLModel.forward of transformers {PINNED_TRANSFORMERS}, "
                           f"found {transformers.__version__}")


def attention_mask(am: torch.Tensor, img: torch.Tensor, nodes: list[Nodes | None], K: int, dtype) -> torch.Tensor:
    # additive (B, 1, T, T) mask, T = L + K: causal text/image rows that never see mask tokens; mask-token rows see
    # every non-image text token, the image tokens of their region and their relatives; padding rows see themselves
    B, L = am.shape
    T_ = L + K
    allow = torch.zeros((B, T_, T_), dtype=torch.bool)
    causal = torch.tril(torch.ones((L, L), dtype=torch.bool))
    for b in range(B):
        n_tok = int(am[b].sum())
        if not bool(am[b, :n_tok].all()):
            raise ValueError("sequences must be right-padded")
        allow[b, :n_tok, :n_tok] = causal[:n_tok, :n_tok]
        allow[b, torch.arange(n_tok, T_), torch.arange(n_tok, T_)] = True
        nb = nodes[b]
        if nb is None or nb.n == 0:
            continue
        img_pos = torch.nonzero(img[b]).reshape(-1)
        text = torch.ones(n_tok, dtype=torch.bool)
        text[img_pos] = False
        for v in range(nb.n):
            row = allow[b, L + v]
            row[:n_tok] = text
            row[img_pos[torch.from_numpy(nb.region[v])]] = True
            row[L + torch.from_numpy(nb.rel[v])] = True
    mask = torch.zeros((B, 1, T_, T_), dtype=dtype)
    return mask.masked_fill_(~allow[:, None], torch.finfo(dtype).min)


def positions(pos: torch.Tensor, am: torch.Tensor, img: torch.Tensor, nodes: list[Nodes | None], K: int
              ) -> torch.Tensor:
    # (3, B, L + K): mask token v of sample b at (t after the last token, image row offset + centroid row, image
    # column offset + centroid column); Qwen3-VL image tokens sit at (off, off + row, off + col)
    B, L = am.shape
    out = torch.ones((3, B, L + K), dtype=pos.dtype)
    out[:, :, :L] = pos
    for b in range(B):
        nb = nodes[b]
        if nb is None or nb.n == 0:
            continue
        n_tok = int(am[b].sum())
        off = int(pos[0, b, torch.nonzero(img[b]).reshape(-1)[0]])
        cell = torch.from_numpy(nb.cell)
        out[0, b, L:L + nb.n] = int(pos[0, b, n_tok - 1]) + 1
        out[1, b, L:L + nb.n] = off + cell[:, 0]
        out[2, b, L:L + nb.n] = off + cell[:, 1]
    return out


def forward(model, batch: dict[str, torch.Tensor], nodes: list[Nodes | None], embed: NodeEmbed,
            layers: list[int], soft: SoftTreeBias | None = None) -> Output:
    check_version()
    inner = model.model
    ids, am = batch["input_ids"], batch["attention_mask"]
    if len(nodes) != ids.shape[0]:
        raise ValueError("one Nodes entry (or None) per sequence")
    emb = inner.get_input_embeddings()(ids)
    img = torch.zeros_like(ids, dtype=torch.bool)
    deepstack = None
    if "pixel_values" in batch:
        feats = inner.get_image_features(batch["pixel_values"], batch["image_grid_thw"], return_dict=True)
        image_embeds = torch.cat(feats.pooler_output, 0).to(emb.device, emb.dtype)
        image_mask, _ = inner.get_placeholder_mask(ids, inputs_embeds=emb, image_features=image_embeds)
        emb = emb.masked_scatter(image_mask, image_embeds)
        img = image_mask[..., 0]
        deepstack = feats.deepstack_features
    pos, _ = inner.get_rope_index(ids, batch.get("image_grid_thw"), None, attention_mask=am)
    B, L, D = emb.shape
    K = max((nb.n for nb in nodes if nb is not None), default=0)
    blocks = []
    for b in range(B):
        block = emb.new_zeros((K, D))
        nb = nodes[b]
        if nb is not None and nb.n:
            img_pos = torch.nonzero(img[b]).reshape(-1)
            if len(img_pos) == 0:
                raise ValueError("a sequence with mask tokens has no image tokens")
            with torch.autocast(device_type=emb.device.type, enabled=False):
                pooled = mllm.pool(emb[b, img_pos], *nb.cov)
            x = embed(pooled, torch.from_numpy(nb.features).to(emb.device))
            block = torch.cat([x.to(emb.dtype), emb.new_zeros((K - nb.n, D))])
        blocks.append(block)
    x = torch.cat([emb, torch.stack(blocks)], 1) if K else emb
    mask = attention_mask(am.cpu(), img.cpu(), nodes, K, emb.dtype).to(emb.device)
    pos_full = positions(pos.cpu(), am.cpu(), img.cpu(), nodes, K).to(emb.device)
    vis = torch.cat([img, img.new_zeros((B, K))], 1)
    if soft is not None:
        soft.set(nodes, L, K, emb.device)
    out = inner.language_model(input_ids=None, inputs_embeds=x, attention_mask=mask, position_ids=pos_full,
                               visual_pos_masks=vis if deepstack is not None else None,
                               deepstack_visual_embeds=deepstack, use_cache=False, output_hidden_states=True)
    if soft is not None and not torch.is_grad_enabled():
        soft.clear()                       # with grad the context must survive until checkpoint recomputation
    hs = out.hidden_states
    if len(hs) != model.config.text_config.num_hidden_layers + 1:
        raise RuntimeError("unexpected number of hidden states")
    node_states = {k: [hs[k][b, L:L + (nodes[b].n if nodes[b] is not None else 0)] for b in range(B)] for k in layers}
    return Output({k: hs[k][:, :L] for k in layers}, node_states, out.last_hidden_state[:, :L])


def pooled_nodes(states: torch.Tensor, img_row: torch.Tensor, nb: Nodes) -> torch.Tensor:
    # flat variant: coverage-weighted mean of the layer states of the image tokens (the S2 node feature)
    pos = torch.nonzero(img_row).reshape(-1)
    if nb.n == 0:
        return states.new_zeros((0, states.shape[-1]), dtype=torch.float32)
    with torch.autocast(device_type=states.device.type, enabled=False):
        return mllm.pool(states[pos], *nb.cov)


def dense_rel(nb: Nodes) -> np.ndarray:
    # (n, n) boolean relative matrix, for tests and diagnostics
    out = np.zeros((nb.n, nb.n), bool)
    for v, r in enumerate(nb.rel):
        out[v, r] = True
    return out


def flat_forward(model, batch: dict[str, torch.Tensor], nodes: list[Nodes | None], layers: list[int]) -> Output:
    # the S2 forward: no mask tokens, node features pooled from the layer states of the image tokens
    out = model.model(**batch, output_hidden_states=True, use_cache=False)
    hs = out.hidden_states
    if len(hs) != model.config.text_config.num_hidden_layers + 1:
        raise RuntimeError("unexpected number of hidden states")
    img = batch["input_ids"] == model.config.image_token_id
    empty = hs[0].new_zeros((0, hs[0].shape[-1]), dtype=torch.float32)
    node_states = {k: [pooled_nodes(hs[k][b], img[b], nb) if nb is not None else empty for b, nb in enumerate(nodes)]
                   for k in layers}
    return Output({k: hs[k] for k in layers}, node_states, out.last_hidden_state)
