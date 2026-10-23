import argparse
import json
import os
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import v2_train_eval as V2  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, resolve_run, start_run, v1_gate, write_json)
from treeground.selector import Selector  # noqa: E402

STAT_KEYS = ["terms_object", "terms_part", "terms_chain", "terms_text", "terms_absent", "empty_negatives",
             "shortfall_in_image", "shortfall_cross", "chain_topup"]


@dataclass
class Query:
    qid: str
    kind: str
    user: str
    answer: str
    steps: list[S.StepTarget]
    path: tuple[int, ...] | None = None
    ious: np.ndarray | None = None     # evaluation only: IoU of every node with the requested GT


@dataclass
class ImageMeta:
    sha1: str
    key: str
    file: str
    n: int
    parents: np.ndarray | None
    labels: np.ndarray
    parts: np.ndarray
    high: np.ndarray
    queries: list[Query]
    absent: bool = False


def dist_env() -> tuple[int, int, int]:
    return int(os.environ.get("RANK", 0)), int(os.environ.get("WORLD_SIZE", 1)), int(os.environ.get("LOCAL_RANK", 0))


def setup_device(cfg: dict) -> str:
    rank, world, local = dist_env()
    cuda = str(cfg["device"]).startswith("cuda")
    if cuda:
        torch.cuda.set_device(local)
    if world > 1:
        dist.init_process_group("nccl" if cuda else "gloo")
    return f"cuda:{local}" if cuda else cfg["device"]


def gather(obj, world: int) -> list:
    if world == 1:
        return [obj]
    out = [None] * world
    dist.all_gather_object(out, obj)
    return out


def broadcast(obj, world: int):
    if world == 1:
        return obj
    box = [obj]
    dist.broadcast_object_list(box, 0)
    return box[0]


def require_d10(cfg: dict, rank: int, world: int) -> None:
    if rank == 0:
        ensure_deviation_log(cfg)
    if world > 1:
        dist.barrier()
    require_approved(cfg, ["D10"])


def stage_cfg(cfg: dict, smoke: bool) -> dict:
    c = dict(cfg["s2"])
    if smoke:
        c.update(c.pop("smoke", {}))
    c.pop("smoke", None)
    return c


def build_model(cfg: dict, c: dict, device: str, static: dict | None):
    q = cfg["models"]["qwen"]
    model, processor = mllm.load_qwen(q["repo"], q["revision"], device, cfg["features"]["dtype"], cfg.get("hf_home"))
    return wrap_model(model, processor, c, static)


def wrap_model(model, processor, c: dict, static: dict | None):
    sel_id, sel_hash = mllm.add_sel_token(model, processor)
    if static is not None and mllm.chat_template_hash(processor) != static["chat_template"]:
        raise ProtocolError("chat template differs from the pinned processor")
    lc = c["lora"]
    wrapped = LR.apply_lora(model, lc["r"], lc["alpha"], lc["dropout"], tuple(lc["targets"]))
    LR.decoder(model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    sel_emb, tied = LR.install_sel(model, sel_id)
    return model, processor, {"sel_id": sel_id, "sel_init": sel_hash, "sel_emb": sel_emb, "tied": tied,
                              "wrapped": wrapped}


def warm_start(cfg: dict, smoke: bool, layer: int, d_model: int) -> tuple[Selector, str]:
    run = resolve_run(cfg, "v2", cfg["inputs"].get("v2", "latest"), smoke)
    ckpt = torch.load(run / "selector.pt", map_location="cpu")
    if ckpt["smoke"] != smoke:
        raise ProtocolError("V2 checkpoint smoke flag does not match the current mode")
    if int(ckpt["layer"]) != layer or ckpt["d_node"] != d_model or ckpt["d_query"] != d_model:
        raise ProtocolError(f"V2 checkpoint (layer {ckpt['layer']}, dims {ckpt['d_node']}/{ckpt['d_query']}) does not "
                            f"match layer {layer} / hidden {d_model}")
    sel = Selector(ckpt["d_node"], ckpt["d_query"], ckpt["temperature"])
    sel.load_state_dict(ckpt["state_dict"])
    return sel, run.name


def paco_meta(row, img: paco.PacoImage, qrows: pd.DataFrame | None, props, tau: float, device: str | None,
              with_ious: bool = False) -> tuple[ImageMeta, Counter]:
    nodes = props.nodes(row.sha1, tau)
    rles = nodes["rles"]
    if (nodes["h"], nodes["w"]) != (img.h, img.w):
        raise ProtocolError(f"{row.image_key}: annotation size differs from proposal masks")
    tree = T.build_tree(rles, device)
    lab = paco_part_labels(rles, img)
    gts = paco.gt_rles(img)
    counts, out = Counter(), []
    for q in (qrows.sort_values("qid").itertuples() if qrows is not None else []):
        gt = json.loads(q.gt)
        step_ious = M.iou(rles, [gts[a] for a in gt]).T if rles else np.zeros((len(gt), 0))
        user, answer = S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text)
        if with_ious:
            out.append(Query(q.qid, q.kind, user, answer, [], None, step_ious[0]))
            continue
        if q.kind == "chain":
            path = T.max_iou_path(tree, step_ious) if rles else None
            if path is None:
                counts["unreachable_chains"] += 1
                continue
            steps = S.chain_step_targets(path, step_ious)
        else:
            tgt = S.single_step_target(step_ious[0])
            if tgt is None:
                counts["proposal_failures"] += 1
                continue
            steps, path = [tgt], None
        out.append(Query(q.qid, q.kind, user, answer, steps, path))
        counts[f"queries_{q.kind}"] += 1
    return ImageMeta(row.sha1, row.image_key, row.path, len(rles), tree.parent, lab.label, lab.part,
                     lab.high_overlap, out), counts


def absent_meta(q, file: str, props, tau: float) -> ImageMeta:
    n = len(props.nodes(q.sha1, tau)["rles"])
    empty = np.full(n, "", dtype=object)
    step = S.StepTarget(np.zeros(0, np.int64), np.zeros(0, np.int64))
    query = Query(q.qid, "absent", S2.user_text(q.text), S2.answer_text("absent", q.qid, q.text), [step])
    return ImageMeta(q.sha1, q.image_key, file, n, None, empty, empty, np.zeros(n, dtype=bool), [query], True)


def train_meta(cfg, wl, props, queries, device, rank, world) -> tuple[list[ImageMeta], Counter]:
    tau = cfg["proposals"]["node_dedup"]
    rows = list(wl[wl.role == "paco_train"].sort_values("image_key").itertuples())[rank::world]
    images = paco.load(cfg["data"]["paco_lvis"], "train", {int(r.image_key.split(":")[1]) for r in rows})
    qtrain = queries[(queries.split == "train") & queries.kind.isin(["object", "part", "chain"])]
    by_sha = {k: g for k, g in qtrain.groupby("sha1")}
    counts = Counter()
    metas = []
    t0 = time.time()

    def one(row):
        return paco_meta(row, images[int(row.image_key.split(":")[1])], by_sha.get(row.sha1), props, tau, device)

    with ThreadPoolExecutor(max_workers=V2.LOAD_WORKERS) as pool:
        for i, (m, c) in enumerate(pool.map(one, rows), 1):
            metas.append(m)
            counts.update(c)
            counts["images_without_queries"] += int(not m.queries)
            if i % 1000 == 0:
                print(f"rank {rank}: training images {i}/{len(rows)} {time.time() - t0:.0f}s", flush=True)
    files = dict(zip(wl.sha1, wl.path))
    absent = [absent_meta(q, files[q.sha1], props, tau) for q in
              list(queries[(queries.kind == "absent") & (queries.split == "train")].sort_values("qid").itertuples())
              [rank::world]]
    counts["absent_queries"] += len(absent)
    return metas + absent, counts


def all_train_meta(cfg, wl, props, queries, device, rank, world) -> tuple[list[ImageMeta], dict]:
    metas, counts = train_meta(cfg, wl, props, queries, device, rank, world)
    parts = gather((metas, counts), world)
    allm = [m for p, _ in parts for m in p]
    total = Counter()
    for _, c in parts:
        total.update(c)
    paco_items = sorted((m for m in allm if not m.absent and m.queries), key=lambda m: m.key)
    absent = sorted((m for m in allm if m.absent), key=lambda m: m.queries[0].qid)
    total["training_images"] = sum(1 for m in allm if not m.absent)
    return paco_items + absent, dict(total)


def val_meta(cfg, wl, props, queries, device, n_images: int, seed: int, rank: int, world: int) -> list[ImageMeta]:
    # fixed PACO val subset for checkpoint selection (spec 7: checkpoints on PACO val)
    tau = cfg["proposals"]["node_dedup"]
    qval = queries[(queries.split == "val") & queries.kind.isin(["object", "part"])]
    keys = wl[(wl.role == "paco_val") & wl.sha1.isin(set(qval.sha1))].sort_values("image_key")
    pick = np.sort(np.random.default_rng(seed).choice(len(keys), size=min(n_images, len(keys)), replace=False))
    rows = list(keys.iloc[pick].itertuples())[rank::world]
    images = paco.load(cfg["data"]["paco_lvis"], "val", {int(r.image_key.split(":")[1]) for r in rows})
    by_sha = {k: g for k, g in qval.groupby("sha1")}
    return [paco_meta(r, images[int(r.image_key.split(":")[1])], by_sha[r.sha1], props, tau, device, True)[0]
            for r in rows]


def plan(items: list[ImageMeta], c: dict, text_labels: set[str], world: int, rank: int) -> list[list[tuple]]:
    # the S1 batch construction over the global batch; this rank takes its share, cut into micro-batches
    B, micro = c["batch_images"], c["micro_batch"]
    if B % (world * micro):
        raise ProtocolError(f"batch_images {B} is not a multiple of world size {world} x micro_batch {micro}")
    if len(items) < B:
        raise ProtocolError(f"{len(items)} training items for a batch of {B}")
    rng = np.random.default_rng(c["seed"])
    n_text = int(round(c["text_only_fraction"] * B))
    order, ptr = rng.permutation(len(items)), 0
    out = []
    share = B // world
    for step in range(c["steps"]):
        if ptr + B > len(order):
            order, ptr = rng.permutation(len(items)), 0
        idx = order[ptr:ptr + B]
        ptr += B
        batch, left = [], n_text
        for j, i in enumerate(idx):
            r = items[i]
            as_text = left > 0 and not r.absent and any(lab and lab in text_labels for lab in r.labels)
            left -= int(as_text)
            seed = int(np.random.SeedSequence([c["seed"], step, j]).generate_state(1)[0])
            batch.append((int(i), bool(as_text), seed))
        mine = batch[rank * share:(rank + 1) * share]
        out.extend(mine[k:k + micro] for k in range(0, share, micro))
    return out


class Source:
    # opened lazily so that every data-loader worker holds its own proposal store
    def __init__(self, cfg, smoke, manifest, mhash, processor, sel_id, text_labels):
        self.cfg, self.smoke, self.manifest, self.mhash = cfg, smoke, manifest, mhash
        self.processor, self.sel_id, self.text_labels = processor, sel_id, text_labels
        self.tau = cfg["proposals"]["node_dedup"]
        self.merge = processor.image_processor.merge_size
        self.pad_id = processor.tokenizer.pad_token_id
        if self.pad_id is None:
            raise ProtocolError("tokenizer has no pad token")
        self._props = None

    @property
    def props(self):
        if self._props is None:
            self._props = open_proposals(self.cfg, self.smoke, self.manifest, self.mhash, capped=True)
        return self._props

    def image_item(self, m: ImageMeta, queries: list[Query]) -> tuple[list[S2.Seq], tuple[int, int], tuple]:
        image = Image.fromarray(load_rgb(assert_real_path(m.file)))
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        nodes = self.props.nodes(m.sha1, self.tau)
        if len(nodes["rles"]) != m.n or (nodes["h"], nodes["w"]) != (image.height, image.width):
            raise ProtocolError(f"{m.key}: proposal nodes do not match the image or its metadata")
        cov = mllm.sparse_coverages((M.decode(r) for r in nodes["rles"]), *grid)
        return seqs, grid, cov

    def text_item(self, label: str) -> S2.Seq:
        q = paco.text_query(label)
        return S2.encode(self.processor, self.sel_id, S2.user_text(q.text), S2.answer_text("text", q.qid, q.text, label),
                         None)


class MicroBatches(torch.utils.data.Dataset):
    def __init__(self, entries: list[list[tuple]], items: list[ImageMeta], src: Source):
        self.entries, self.items, self.src = entries, items, src

    def __len__(self) -> int:
        return len(self.entries)

    def __getitem__(self, j: int) -> dict:
        samples, seqs, text_seqs = [], [], []
        for i, as_text, seed in self.entries[j]:
            m = self.items[i]
            rng = np.random.default_rng(seed)
            qi = int(rng.integers(len(m.queries)))
            q = m.queries[qi]
            (seq,), grid, cov = self.src.image_item(m, [q])
            label = None
            if as_text:
                label = str(rng.choice(sorted({lab for lab in m.labels if lab and lab in self.src.text_labels})))
                text_seqs.append(self.src.text_item(label))
            samples.append({"item": i, "as_text": as_text, "query": qi, "grid": grid, "cov": cov, "label": label})
            seqs.append(seq)
        seqs += text_seqs
        return {"samples": samples, "seqs": seqs, "batch": S2.collate(seqs, self.src.pad_id)}


def micro_step(model, extra, selector, mb, items, layer, device, rng, gamma, stats):
    batch = {k: v.to(device) for k, v in mb["batch"].items()}
    seqs = mb["seqs"]
    with S2.autocast(device):
        hl, last = S2.forward(model, batch, layer)
        l_lm = S2.lm_loss(model, extra["sel_emb"], extra["tied"], last, batch, seqs)
    image_token_id = model.config.image_token_id
    recs, text_probes = [], {}
    t = len(mb["samples"])
    for b, smp in enumerate(mb["samples"]):
        m = items[smp["item"]]
        z = S2.node_features(hl[b], batch["input_ids"][b], image_token_id, smp["grid"], smp["cov"])
        q = m.queries[smp["query"]]
        tree = T.from_parents(m.parents) if m.parents is not None else None
        rec = V2.ImageRec(m.sha1, m.key, z, tree, m.labels, m.parts, m.high)
        rec.absent = m.absent
        rec.queries = [V2.QRec(q.qid, q.kind, hl[b, seqs[b].sel.to(device)].float(), q.steps, q.path)]
        recs.append((rec, smp["as_text"]))
        if smp["label"] is not None:
            text_probes[smp["label"]] = hl[t, seqs[t].sel[0].to(device)].float()
            t += 1
    terms = V2.batch_terms(selector, recs, text_probes, rng, device, gamma, stats)
    return torch.stack(terms).mean(), l_lm


def sync_grads(params: list[torch.nn.Parameter], world: int) -> None:
    for p in params:
        if p.grad is None:
            p.grad = torch.zeros_like(p)
    if world == 1:
        return
    flat = torch.cat([p.grad.reshape(-1) for p in params])
    dist.all_reduce(flat)
    flat /= world
    o = 0
    for p in params:
        p.grad.copy_(flat[o:o + p.numel()].view_as(p))
        o += p.numel()


@torch.no_grad()
def recompute(model, src: Source, m: ImageMeta, queries: list[Query], layer: int, device: str, batch_size: int
              ) -> tuple[torch.Tensor, list[torch.Tensor]]:
    # node features from the first sequence (image tokens precede the query, so all sequences agree) and the
    # layer-l states at the <SEL> positions of every query of the image
    seqs, grid, cov = src.image_item(m, queries)
    feats, probes = None, []
    for start in range(0, len(seqs), batch_size):
        chunk = seqs[start:start + batch_size]
        batch = {k: v.to(device) for k, v in S2.collate(chunk, src.pad_id).items()}
        with S2.autocast(device):
            hl, _ = S2.forward(model, batch, layer)
        if feats is None:
            feats = S2.node_features(hl[0], batch["input_ids"][0], model.config.image_token_id, grid, cov)
        probes += [hl[b, s.sel.to(device)].float() for b, s in enumerate(chunk)]
    return feats, probes


@torch.no_grad()
def val_success(model, selector, src, metas, layer, device, world, batch_size) -> dict:
    model.eval()
    selector.eval()
    rows = []
    for m in metas:
        if m.n == 0:
            rows += [{"qid": q.qid, "kind": q.kind, "sel_iou": 0.0, "best_iou": 0.0} for q in m.queries]
            continue
        z, probes = recompute(model, src, m, m.queries, layer, device, batch_size)
        e = selector.nodes(z)
        for q, p in zip(m.queries, probes):
            sig = torch.sigmoid(selector.scores(selector.queries(p), e)[0]).cpu().numpy()
            a = int(np.argmax(sig))
            rows.append({"qid": q.qid, "kind": q.kind, "sel_iou": float(q.ious[a]), "best_iou": float(q.ious.max())})
    model.train()
    selector.train()
    df = pd.DataFrame([r for part in gather(rows, world) for r in part])
    cov = df[df.best_iou >= 0.5]
    return {"selector_success": float((cov.sel_iou >= 0.5).mean()) if len(cov) else 0.0,
            "by_kind": {k: float((g.sel_iou >= 0.5).mean()) for k, g in cov.groupby("kind")},
            "covered_queries": int(len(cov)), "queries": int(len(df)),
            "conditioned_on": "a proposal with IoU >= 0.5 exists; argmax over C_single"}


def checkpoint(model, extra, selector, meta: dict) -> dict:
    return {**LR.trainable_state(model, extra["sel_emb"], selector), "meta": meta}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    c = stage_cfg(cfg, args.smoke)
    rank, world, _ = dist_env()
    device = setup_device(cfg)
    require_d10(cfg, rank, world)
    gate = v1_gate(cfg, args.smoke)
    layer, v3_run = V2.chosen_layer(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(c["seed"])
    model, processor, extra = build_model(cfg, c, device, static)
    mllm.check_layers(model, [layer])
    d_model = model.config.text_config.hidden_size
    selector, v2_run = warm_start(cfg, args.smoke, layer, d_model)
    selector = selector.to(device).train()
    model.train()
    lora_params = LR.trainable_parameters(model, extra["sel_emb"])
    params = lora_params + list(selector.parameters())
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    torch.manual_seed(c["seed"] * 1000 + rank)  # LoRA dropout masks differ across ranks

    text_labels = set(queries[queries.kind == "text"].label)
    items, counts = all_train_meta(cfg, wl, props, queries, device, rank, world)
    if not items:
        raise ProtocolError("no training items")
    vmetas = val_meta(cfg, wl, props, queries, device, c["eval_images"], c["seed"], rank, world)
    src = Source(cfg, args.smoke, manifest, mhash, processor, extra["sel_id"], text_labels)
    entries = plan(items, c, text_labels, world, rank)
    accum = c["batch_images"] // (world * c["micro_batch"])
    loader = torch.utils.data.DataLoader(MicroBatches(entries, items, src), batch_size=None, shuffle=False,
                                         num_workers=c["workers"], collate_fn=lambda x: x,
                                         prefetch_factor=4 if c["workers"] else None)

    opt = torch.optim.AdamW([
        {"params": list(selector.parameters()), "lr": c["selector_lr"], "weight_decay": c["selector_weight_decay"]},
        {"params": lora_params, "lr": c["lora_lr"], "weight_decay": c["lora_weight_decay"]}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, V2.lr_lambda(c["warmup_steps"], c["steps"]))
    run = None
    if rank == 0:
        run = start_run("s2", cfg, args.smoke, {
            "manifest_run": str(mrun), "manifest_sha256": mhash, "queries": str(qdir), "layer": layer,
            "v3_run": v3_run, "v2_run": v2_run, "world_size": world, "accumulation": accum,
            "sel_init": extra["sel_init"], "lora_modules": len(extra["wrapped"]), "tied_embeddings": extra["tied"]})
    run = Path(broadcast(str(run) if run else None, world))
    ckpt_meta = {"layer": layer, "d_node": d_model, "d_query": d_model, "temperature": selector.tau,
                 "lora": c["lora"], "sel_id": extra["sel_id"], "sel_init": extra["sel_init"],
                 "qwen": cfg["models"]["qwen"], "smoke": args.smoke, "v2_run": v2_run,
                 "config_hash": canonical_hash(c)}

    rng = np.random.default_rng([c["seed"], rank])
    it = iter(loader)
    log, evals = [], []
    t0 = time.time()

    def evaluate_and_save(step: int, best: tuple) -> tuple:
        # step 0 is the warm start (LoRA B = 0, trained <SEL> row = vocabulary mean); it is a candidate as well
        res = val_success(model, selector, src, vmetas, layer, device, world, c["eval_batch"])
        if rank == 0:
            state = checkpoint(model, extra, selector, {**ckpt_meta, "step": step, "val": res})
            torch.save(state, run / f"step{step:06d}.pt")
            if res["selector_success"] > best[0]:
                best = (res["selector_success"], step)
                torch.save(state, run / "best.pt")
            evals.append({"step": step, **res})
            print(f"eval step {step}: selector success {res['selector_success']:.4f}", flush=True)
            pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
            write_json(run / "eval_curve.json", evals)
        return best

    best = evaluate_and_save(0, (-1.0, None))
    for step in range(c["steps"]):
        stats = {k: 0 for k in STAT_KEYS}
        sums = torch.zeros(2, device=device)
        for _ in range(accum):
            l_sel, l_lm = micro_step(model, extra, selector, next(it), items, layer, device, rng, c["gamma"], stats)
            ((l_sel + l_lm) / accum).backward()
            sums += torch.stack([l_sel.detach(), l_lm.detach()]) / accum
        sync_grads(params, world)
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
                        "lr_selector": sched.get_last_lr()[0], "lr_lora": sched.get_last_lr()[1],
                        "seconds": time.time() - t0, **stats})
        if (step + 1) % c["eval_every"] == 0 or step + 1 == c["steps"]:
            best = evaluate_and_save(step + 1, best)
    if rank == 0:
        write_json(run / "metrics.json", {
            "label": "dev", "smoke": args.smoke, "layer": layer, "best_step": best[1], "best_checkpoint": "best.pt",
            "best_val_selector_success": best[0], "selection": "PACO val fixed subset, single-step selector success",
            "eval_images": c["eval_images"], "eval_curve": evals, "training_exclusions": counts,
            "training_items": len(items), "world_size": world, "accumulation": accum, "v1_gate": gate,
            "deviations": ["D10"]})
        print(run)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
