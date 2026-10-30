import argparse
import os
import sys
import time
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
import ta_paco_nodes as TPN  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground import tree_attn as TA  # noqa: E402
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
# Heads are trained jointly at every read-out layer; the checkpoint and its layer are chosen on the fixed PACO val
# subset by single-step selector success (S2 rule), best over steps and layers.

VARIANTS = ("flat",) + TA.MODES
DEVIATIONS = ["D10", "D21", "D41"]


class Readout(nn.Module):
    def __init__(self, d_model: int, layers: list[int], n_nodes: int, embed: TA.NodeEmbed | None):
        super().__init__()
        self.layers = list(layers)
        self.heads = nn.ModuleList(TA.ScaledSelector(d_model, d_model, n_nodes) for _ in layers)
        self.embed = embed

    def head(self, layer: int) -> TA.ScaledSelector:
        return self.heads[self.layers.index(layer)]


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
    return f"ta_{cfg['ta']['variant']}"


def node_mode(variant: str) -> str:
    # the flat variant only uses the coverage maps, which are the same for every mode
    return "tree" if variant == "flat" else variant


class Source(S2T.Source):
    def __init__(self, cfg, smoke, manifest, mhash, processor, sel_id, text_labels, variant: str, seed: int):
        super().__init__(cfg, smoke, manifest, mhash, processor, sel_id, text_labels)
        self.mode, self.seed = node_mode(variant), seed

    def image_item(self, m: S2T.ImageMeta, queries: list[S2T.Query]) -> tuple[list[S2.Seq], TA.Nodes]:
        image = Image.fromarray(load_rgb(assert_real_path(m.file)))
        seqs = [S2.encode(self.processor, self.sel_id, q.user, q.answer, image) for q in queries]
        grid = mllm.merged_grid(seqs[0].image_grid_thw, self.merge)
        nodes = S2T.read_nodes(m.nodes) if m.nodes else self.props.nodes(m.sha1, self.tau)
        if len(nodes["rles"]) != m.n or (nodes["h"], nodes["w"]) != (image.height, image.width):
            raise ProtocolError(f"{m.key}: proposal nodes do not match the image or its metadata")
        if m.parents is None or len(m.parents) != m.n + 1:
            raise ProtocolError(f"{m.key}: no tree over its {m.n} nodes")
        return seqs, TA.build_nodes(nodes["rles"], m.parents, grid, self.mode, V4.tree_seed(self.seed, m.sha1))


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
    return TA.forward(model, batch, nodes, readout.embed, readout.layers)


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
            row = {}
            for k in layers:
                h = readout.head(k)
                row[k] = h.scores(h.queries(res.hidden[k][b, s.sel.to(device)].float()),
                                  h.nodes(res.nodes[k][b].float())).float().cpu()
            out.append(row)
    return out


@torch.no_grad()
def val_success(model, readout, src, metas, device, world, batch_size) -> dict:
    model.eval()
    readout.eval()
    rows = []
    for m in metas:
        if m.n == 0:
            rows += [{"qid": q.qid, "kind": q.kind, "layer": k, "sel_iou": 0.0, "best_iou": 0.0}
                     for q in m.queries for k in readout.layers]
            continue
        for q, lg in zip(m.queries, query_logits(model, readout, src, m, m.queries, device, batch_size)):
            for k in readout.layers:
                a = int(torch.argmax(lg[k][0]))
                rows.append({"qid": q.qid, "kind": q.kind, "layer": k, "sel_iou": float(q.ious[a]),
                             "best_iou": float(q.ious.max())})
    model.train()
    readout.train()
    df = pd.DataFrame([r for part in S2T.gather(rows, world) for r in part])
    by_layer = {}
    for k, g in df.groupby("layer"):
        cov = g[g.best_iou >= 0.5]
        by_layer[int(k)] = {"selector_success": float((cov.sel_iou >= 0.5).mean()) if len(cov) else 0.0,
                            "by_kind": {kk: float((gg.sel_iou >= 0.5).mean()) for kk, gg in cov.groupby("kind")},
                            "covered_queries": int(len(cov)), "queries": int(len(g))}
    best = max(by_layer, key=lambda k: (by_layer[k]["selector_success"], -k))   # ties: earlier layer
    return {"by_layer": by_layer, "best_layer": best, "selector_success": by_layer[best]["selector_success"],
            "conditioned_on": "a proposal with IoU >= 0.5 exists; argmax over C_single"}


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
    embed = None if c["variant"] == "flat" else TA.NodeEmbed(d_model, TA.embedding_rms(model))
    readout = Readout(d_model, c["readout_layers"], n_nodes, embed).to(device)
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
    require_approved(cfg, DEVIATIONS)
    TA.check_version()
    gate = v1_gate(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    torch.manual_seed(c["seed"])
    cap = node_cap(cfg, args.smoke)
    n_nodes = int(cap["k"]) if cap else 256   # only sets the initial bias, -log K
    model, processor, extra, readout = build(cfg, c, device, static, n_nodes)
    readout.train()
    model.train()
    lora_params = LR.trainable_parameters(model, extra["sel_emb"])
    head_params = list(readout.parameters())
    params = lora_params + head_params
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    torch.manual_seed(c["seed"] * 1000 + rank)  # LoRA dropout masks differ across ranks

    text_labels = set(queries[queries.kind == "text"].label)
    weights = S2T.source_weights(c)
    items, counts = S2T.all_train_meta(cfg, wl, props, queries, device, rank, world, list(weights), args.smoke)
    if c["paco_rest"]:
        rest, rest_counts = rest_meta(cfg, args.smoke)
        items += rest
        counts.update(rest_counts)
    counts["trees_built_for_no_target_items"] = add_missing_trees(items, props, cfg["proposals"]["node_dedup"], device)
    if not items:
        raise ProtocolError("no training items")
    vmetas = S2T.val_meta(cfg, wl, props, queries, device, c["eval_images"], c["seed"], rank, world)
    src = Source(cfg, args.smoke, manifest, mhash, processor, extra["sel_id"], text_labels, c["variant"], c["seed"])
    entries = S2T.plan(items, c, text_labels, world, rank)
    accum = c["batch_images"] // (world * c["micro_batch"])
    loader = torch.utils.data.DataLoader(MicroBatches(entries, items, src), batch_size=None, shuffle=False,
                                         num_workers=c["workers"], collate_fn=lambda x: x,
                                         prefetch_factor=4 if c["workers"] else None)

    opt = torch.optim.AdamW([
        {"params": head_params, "lr": c["head_lr"], "weight_decay": c["head_weight_decay"]},
        {"params": lora_params, "lr": c["lora_lr"], "weight_decay": c["lora_weight_decay"]}])
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
                 "seed": c["seed"], "config_hash": canonical_hash(c)}

    rng = np.random.default_rng([c["seed"], rank])
    it = iter(loader)
    log, evals = [], []
    t0 = time.time()

    def evaluate_and_save(step: int, best: tuple) -> tuple:
        res = val_success(model, readout, src, vmetas, device, world, c["eval_batch"])
        if rank == 0:
            state = checkpoint_state(model, extra, readout, {**ckpt_meta, "step": step, "val": res,
                                                             "layer": res["best_layer"]})
            torch.save(state, run / f"step{step:06d}.pt")
            if res["selector_success"] > best[0]:
                best = (res["selector_success"], step, res["best_layer"])
                torch.save(state, run / "best.pt")
            evals.append({"step": step, **res})
            print(f"eval step {step}: selector success {res['selector_success']:.4f} at layer {res['best_layer']} "
                  + " ".join(f"L{k} {v['selector_success']:.4f}" for k, v in res["by_layer"].items()), flush=True)
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
            "selection": "PACO val fixed subset, single-step selector success, best over steps and read-out layers",
            "eval_images": c["eval_images"], "eval_curve": evals, "training_exclusions": counts,
            "training_items": len(items), "world_size": world, "accumulation": accum, "v1_gate": gate,
            "sources": weights, "deviations": DEVIATIONS})
        print(run)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
