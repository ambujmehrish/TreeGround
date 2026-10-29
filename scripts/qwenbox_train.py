import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
from PIL import Image
from pycocotools import mask as mu

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s5_train as S5T  # noqa: E402,F401  (installs the S5 sources and node_meta into s2_train)
import s5_pool as S5P  # noqa: E402
import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import b_baselines as B  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import mllm  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.data.paco import PART_TEMPLATE  # noqa: E402
from treeground.identity import manifest_inputs, open_queries, qwen_static  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, read_json, require_approved, start_run, write_json)

# Deviation D40, Group A: Qwen3-VL-4B box + SAM 3 fine-tuned on exactly the S5 training data.
#   --prepare  once: for every S5 training item (s2_train.all_train_meta order with the S5 sources, so the same
#              images, qids and query texts) the GT boxes of its queries from the same GT RLEs (s5_pool.items), and
#              the PACO val selection subset (s2_train.val_meta's images); written to the cache
#   training   torchrun, S2 schedule from cfg['s2'] (steps, batch_images, micro_batch, LoRA r / alpha / dropout /
#              modules, lora_lr, lora_weight_decay, warmup, max_grad_norm, seed, eval_every) and s2_train.plan, so
#              each step draws the same images and the same query per image as S5; the <SEL> token and the
#              selector are not used. Target = the Qwen3-VL grounding answer: a JSON list of every target's
#              bbox_2d on the 0-1000 grid with the phrase as label, '[]' for no-target; chain queries are trained
#              as their part ('the P of the X', the part's box). Loss = LM loss on the answer tokens.
#   selection  every eval_every steps (and step 0): mean box IoU of the first predicted box on the PACO val subset
#              (val only, as S2 selects on PACO val); best.pt is the best step
# Inference: b_parity 'qwenbox_ft' (qwen_box_all with this LoRA), D13 prompt, every box -> SAM 3 mask.

STAGE = "qwenbox_s5"
GRID = 1000


def settings(cfg: dict) -> dict:
    return {"stage": STAGE, "prompt": cfg["baselines"]["qwen_box_sam"]["prompt"], **cfg["qwenbox"]}


def to_grid(rle: dict) -> list[int]:
    h, w = rle["size"]
    x, y, bw, bh = (float(v) for v in mu.toBbox(M._coco(rle)))
    box = [round(x / w * GRID), round(y / h * GRID), round((x + bw) / w * GRID), round((y + bh) / h * GRID)]
    return [min(max(v, 0), GRID) for v in box]


def answer(phrase: str, boxes: list[list[int]]) -> str:
    if not boxes:
        return "```json\n[]\n```"
    rows = ",\n".join(f'\t{{"bbox_2d": [{b[0]}, {b[1]}, {b[2]}, {b[3]}], "label": {json.dumps(phrase)}}}' for b in boxes)
    return f"```json\n[\n{rows}\n]\n```"


def data_dir(cfg: dict) -> Path:
    ident = {"kind": "qwenbox_data", "sources": S2T.source_weights(cfg["s2"]),
             "pools": {s: S5P.identity(cfg, S5T.base_source(s), False) for s in S2T.source_weights(cfg["s2"])},
             "eval_images": cfg["s2"]["eval_images"], "seed": cfg["s2"]["seed"],
             "code": (ROOT / "scripts" / "qwenbox_train.py").read_text()}
    return Path(cfg["cache_root"]) / "full" / "qwenbox_data" / canonical_hash(ident)[:16]


def gt_table(cfg: dict, bases: list[str]) -> dict[str, tuple]:
    # qid -> (kind, text, GT boxes of its RLEs or None), from the same item generators the S5 pools were built from
    out = {}
    for base in bases:
        counts = Counter()
        for _, _, qs, _ in S5P.items(cfg, base, 0, 1, counts):
            for qid, kind, text, gts in qs:
                out[qid] = (kind, text, None if gts is None else [to_grid(r) for r in gts])
        print(f"{base}: {len(out)} GT queries so far {dict(counts)}", flush=True)
    return out


def target(q, m, gt: dict) -> tuple[str, list[list[int]]]:
    if m.absent:
        return q.user, []
    if q.qid not in gt:
        raise ProtocolError(f"{q.qid}: no GT in the item generators")
    kind, text, boxes = gt[q.qid]
    if kind == "referring":
        return q.user, boxes
    if kind == "chain":
        obj, part = q.qid.split("|")[2].split(":", 1)
        part_qid = q.qid.replace("|chain|", "|part|")
        ptext = gt[part_qid][1] if part_qid in gt else PART_TEMPLATE.format(object=obj, part=part)
        return B.phrase(ptext), [boxes[-1]]
    if kind in ("object", "part"):
        return B.phrase(text), [boxes[0]]
    raise ProtocolError(f"{q.qid}: unexpected kind {kind}")


def prepare(cfg: dict) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D36", "D40"])
    c = S2T.stage_cfg(cfg, False)
    weights = S2T.source_weights(c)
    d = data_dir(cfg)
    d.mkdir(parents=True, exist_ok=True)
    run = start_run(f"{STAGE}_prepare", cfg, False, {"cache": str(d)})
    items, counts = [], Counter()
    for source in weights:
        metas, cnt = S2T.node_meta(cfg, source, False)
        items += metas
        counts.update(cnt)
    gt = gt_table(cfg, sorted({S5T.base_source(s) for s in weights}))
    recs, kinds = [], Counter()
    for m in items:
        qs = []
        for q in m.queries:
            phrase, boxes = target(q, m, gt)
            kinds["absent" if m.absent else gt[q.qid][0]] += 1
            qs.append({"qid": q.qid, "phrase": phrase, "answer": answer(phrase, boxes), "n_boxes": len(boxes)})
        recs.append({"key": m.key, "file": m.file, "source": m.source, "absent": bool(m.absent), "queries": qs})
    # PACO val selection subset: s2_train.val_meta's images, its object / part queries, single GT
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    _, queries = open_queries(cfg, False, mhash, qwen_static(cfg))
    qval = queries[(queries.split == "val") & queries.kind.isin(["object", "part"])]
    keys = wl[(wl.role == "paco_val") & wl.sha1.isin(set(qval.sha1))].sort_values("image_key")
    pick = np.sort(np.random.default_rng(c["seed"]).choice(len(keys), size=min(c["eval_images"], len(keys)),
                                                          replace=False))
    rows = list(keys.iloc[pick].itertuples())
    images = paco.load(cfg["data"]["paco_lvis"], "val", {int(r.image_key.split(":")[1]) for r in rows})
    wanted = set(qval.qid) if "qid" in qval else None
    val = []
    for r in rows:
        img = images[int(r.image_key.split(":")[1])]
        gts = paco.gt_rles(img)
        for q in paco.i1_queries(img)[0]:
            if q.kind in ("object", "part") and (wanted is None or q.qid in wanted):
                val.append({"qid": q.qid, "file": r.path, "phrase": B.phrase(q.text), "box": to_grid(gts[q.gt[0]])})
    torch.save({"items": recs, "val": val}, d / "data.pt")
    meta = {"items": len(recs), "queries": sum(len(r["queries"]) for r in recs), "kinds": dict(kinds),
            "val_images": len(rows), "val_queries": len(val), "sources": weights, "pool_counts": dict(counts)}
    write_json(d / "summary.json", meta)
    write_json(run / "metrics.json", {**meta, "cache": str(d)})
    print(json.dumps(meta, indent=1))


# ---- training ----

def chat(processor, prompt: str, phrase: str, ans: str | None) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": prompt.format(phrase=phrase)}]}]
    if ans is None:
        return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    msgs.append({"role": "assistant", "content": [{"type": "text", "text": ans}]})
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


class Micro(torch.utils.data.Dataset):
    def __init__(self, entries, items, processor, prompt):
        self.entries, self.items, self.proc, self.prompt = entries, items, processor, prompt

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, j):
        texts, prompts, images = [], [], []
        for i, _, seed in self.entries[j]:
            m = self.items[i]
            q = m["queries"][int(np.random.default_rng(seed).integers(len(m["queries"])))]   # as s2_train.MicroBatches
            images.append(Image.fromarray(load_rgb(assert_real_path(m["file"]))))
            texts.append(chat(self.proc, self.prompt, q["phrase"], q["answer"]))
            prompts.append(chat(self.proc, self.prompt, q["phrase"], None))
            if not texts[-1].startswith(prompts[-1] + q["answer"]):
                # the trained answer must follow exactly the generation prompt used at inference
                raise ProtocolError(f"chat template: answer is not a continuation of the generation prompt "
                                    f"({texts[-1][len(prompts[-1]):][:40]!r})")
        self.proc.tokenizer.padding_side = "right"
        batch = self.proc(text=texts, images=images, padding=True, return_tensors="pt")
        labels = batch["input_ids"].clone()
        for b, (p, im) in enumerate(zip(prompts, images)):
            n = self.proc(text=[p], images=[im], return_tensors="pt")["input_ids"].shape[1]
            labels[b, :n] = -100
        labels[batch["attention_mask"] == 0] = -100
        batch["labels"] = labels
        return dict(batch)


def box_iou(a: list[float] | None, b: list[float]) -> float:
    if a is None:
        return 0.0
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def val_box(model, processor, prompt, val, device, rank, world, batch_size, max_new) -> dict:
    model.eval()
    processor.tokenizer.padding_side = "left"
    mine, rows = val[rank::world], []
    for k in range(0, len(mine), batch_size):
        chunk = mine[k:k + batch_size]
        ims = [Image.fromarray(load_rgb(assert_real_path(v["file"]))) for v in chunk]
        inp = processor(text=[chat(processor, prompt, v["phrase"], None) for v in chunk], images=ims, padding=True,
                        return_tensors="pt").to(device)
        with S2.autocast(device):
            gen = model.generate(**inp, max_new_tokens=max_new, do_sample=False)
        for v, r in zip(chunk, processor.batch_decode(gen[:, inp["input_ids"].shape[1]:], skip_special_tokens=True)):
            rows.append(box_iou(B.parse_box(r), v["box"]))
    model.train()
    allr = [x for part in S2T.gather(rows, world) for x in part]
    return {"box_iou": float(np.mean(allr)) if allr else 0.0, "queries": len(allr)}


def train(cfg: dict) -> None:
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    c, s = S2T.stage_cfg(cfg, False), settings(cfg)
    rank, world, _ = S2T.dist_env()
    device = S2T.setup_device(cfg)
    if rank == 0:
        ensure_deviation_log(cfg)
    require_approved(cfg, ["D36", "D40"])
    d = data_dir(cfg)
    if not (d / "data.pt").exists():
        raise ProtocolError(f"{d / 'data.pt'} missing (run qwenbox_train.py --prepare)")
    data = torch.load(d / "data.pt", weights_only=False)
    items, val = data["items"], data["val"]
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    torch.manual_seed(c["seed"])
    q = cfg["models"]["qwen"]
    model, processor = mllm.load_qwen(q["repo"], q["revision"], device, cfg["features"]["dtype"], cfg.get("hf_home"))
    lc = c["lora"]
    wrapped = LR.apply_lora(model, lc["r"], lc["alpha"], lc["dropout"], tuple(lc["targets"]))
    LR.decoder(model).gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    params = [p for p in model.parameters() if p.requires_grad]
    if world > 1:
        for p in params:
            dist.broadcast(p.data, 0)
    torch.manual_seed(c["seed"] * 1000 + rank)

    # the S5 batch plan: same images per step and (by the same seeds) the same query per image
    plan_items = [S2T.ImageMeta(sha1=r["key"], key=r["key"], file=r["file"], n=0, parents=None,
                                labels=np.array([], dtype=object), parts=np.array([], dtype=object),
                                high=np.array([], bool), queries=r["queries"], absent=r["absent"], source=r["source"])
                  for r in items]
    entries = S2T.plan(plan_items, c, set(), world, rank)
    accum = c["batch_images"] // (world * c["micro_batch"])
    loader = torch.utils.data.DataLoader(Micro(entries, items, processor, s["prompt"]), batch_size=None,
                                         shuffle=False, num_workers=c["workers"], collate_fn=lambda x: x,
                                         prefetch_factor=4 if c["workers"] else None)
    opt = torch.optim.AdamW([{"params": params, "lr": c["lora_lr"], "weight_decay": c["lora_weight_decay"]}])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, V2.lr_lambda(c["warmup_steps"], c["steps"]))
    run = None
    if rank == 0:
        run = start_run(STAGE, cfg, False, {"data": str(d), "world_size": world, "accumulation": accum,
                                             "lora_modules": len(wrapped)})
    run = Path(S2T.broadcast(str(run) if run else None, world))
    meta = {"lora": lc, "qwen": q, "prompt": s["prompt"], "data": str(d), "config_hash": canonical_hash(c)}
    log, evals, best = [], [], (-1.0, None)
    t0 = time.time()

    def evaluate_and_save(step: int, best: tuple) -> tuple:
        res = val_box(model, processor, s["prompt"], val, device, rank, world, s["eval_batch"], s["max_new_tokens"])
        if rank == 0:
            state = {"lora": LR.lora_state(model), "meta": {**meta, "step": step, "val": res}}
            torch.save(state, run / f"step{step:06d}.pt")
            if res["box_iou"] > best[0]:
                best = (res["box_iou"], step)
                torch.save(state, run / "best.pt")
            evals.append({"step": step, **res})
            print(f"eval step {step}: PACO val box IoU {res['box_iou']:.4f}", flush=True)
            pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
            write_json(run / "eval_curve.json", evals)
        return S2T.broadcast(best, world)

    best = evaluate_and_save(0, best)
    it = iter(loader)
    for step in range(c["steps"]):
        total = torch.zeros((), device=device)
        for _ in range(accum):
            batch = {k: v.to(device) for k, v in next(it).items()}
            with S2.autocast(device):
                loss = model(**batch).loss
            (loss / accum).backward()
            total += loss.detach() / accum
        S2T.sync_grads(params, world)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, c["max_grad_norm"])
        opt.step()
        sched.step()
        opt.zero_grad(set_to_none=True)
        if world > 1:
            dist.all_reduce(total)
            total /= world
        if rank == 0 and (step % c["log_every"] == 0 or step == c["steps"] - 1):
            print(f"step {step}/{c['steps']} L_LM {total.item():.4f} {time.time() - t0:.0f}s", flush=True)
            log.append({"step": step, "l_lm": total.item(), "grad_norm": float(grad_norm),
                        "lr": sched.get_last_lr()[0], "seconds": time.time() - t0})
        if (step + 1) % c["eval_every"] == 0 or step + 1 == c["steps"]:
            best = evaluate_and_save(step + 1, best)
    if rank == 0:
        write_json(run / "metrics.json", {
            "label": "dev", "best_step": best[1], "best_checkpoint": "best.pt", "best_val_box_iou": best[0],
            "selection": "PACO val fixed subset (s2_train.val_meta images), first-box IoU", "eval_curve": evals,
            "training_items": len(items), "world_size": world, "accumulation": accum,
            "sources": S2T.source_weights(c), "deviations": ["D36", "D40"]})
        print(run)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


def checkpoint(cfg: dict) -> tuple[Path, dict]:
    from treeground.run import resolve_run
    s = cfg["parity"]["qwenbox_ft"]
    run = resolve_run(cfg, STAGE, s["run"], False)
    return run / s["checkpoint"], read_json(run / "metrics.json")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--prepare", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    if args.prepare:
        prepare(cfg)
    else:
        train(cfg)


if __name__ == "__main__":
    main()
