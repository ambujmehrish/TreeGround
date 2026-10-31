import math
from dataclasses import dataclass

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
GEOM_DIM = 9
REGION_DILATE = 1   # image-token cells of margin around a mask's own cells


@dataclass
class Nodes:
    cov: tuple[np.ndarray, np.ndarray, np.ndarray]   # sparse coverage on the merged grid (mllm.sparse_coverages)
    region: list[np.ndarray]                         # grid cells each node attends to
    rel: list[np.ndarray]                            # node indices each node attends to, itself included
    geom: np.ndarray                                 # (n, GEOM_DIM) float32
    cell: np.ndarray                                 # (n, 2) int64 (row, col) of the coverage centroid
    anc: np.ndarray                                  # (n, n) bool, anc[v, u]: u is a strict ancestor of v in the
                                                     # tree behind `rel` (the true tree, or the shuffled one)

    @property
    def n(self) -> int:
        return len(self.rel)


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
                tree_seed: int | None = None) -> Nodes:
    if mode not in RELATIONS:
        raise ValueError(f"unknown tree-attention mode {mode!r}")
    gh, gw = grid
    h, w = rles[0]["size"] if rles else (1, 1)
    tree = T.from_parents(parent)
    if tree.n != len(rles):
        raise ValueError(f"tree has {tree.n} nodes for {len(rles)} masks")
    if mode == "random_tree":
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
    rel = relatives(tree, mode if mode in ("full", "self") else "tree")
    return Nodes(cov, region, rel, geometry(rles, tree, h, w), cell, ancestors(tree))


class NodeEmbed(nn.Module):
    # x_v = MLP([mean image-token input embedding under v ; geometry MLP(g_v)]) + kind, rescaled to RMS `gain`
    def __init__(self, d_model: int, gain: float, d_geom: int = GEOM_DIM, hidden: int = 256):
        super().__init__()
        self.geom = nn.Sequential(nn.Linear(d_geom, hidden), nn.GELU(), nn.Linear(hidden, hidden))
        self.mix = nn.Sequential(nn.Linear(d_model + hidden, d_model), nn.GELU(), nn.Linear(d_model, d_model))
        self.kind = nn.Parameter(torch.zeros(d_model))
        self.gain = nn.Parameter(torch.tensor(float(gain)))

    def forward(self, pooled: torch.Tensor, geom: torch.Tensor) -> torch.Tensor:
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
            layers: list[int]) -> Output:
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
            x = embed(pooled, torch.from_numpy(nb.geom).to(emb.device))
            block = torch.cat([x.to(emb.dtype), emb.new_zeros((K - nb.n, D))])
        blocks.append(block)
    x = torch.cat([emb, torch.stack(blocks)], 1) if K else emb
    mask = attention_mask(am.cpu(), img.cpu(), nodes, K, emb.dtype).to(emb.device)
    pos_full = positions(pos.cpu(), am.cpu(), img.cpu(), nodes, K).to(emb.device)
    vis = torch.cat([img, img.new_zeros((B, K))], 1)
    out = inner.language_model(input_ids=None, inputs_embeds=x, attention_mask=mask, position_ids=pos_full,
                               visual_pos_masks=vis if deepstack is not None else None,
                               deepstack_visual_embeds=deepstack, use_cache=False, output_hidden_states=True)
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
