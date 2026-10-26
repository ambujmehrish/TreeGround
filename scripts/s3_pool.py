import argparse
import hashlib
import itertools
import json
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_baselines as B  # noqa: E402
import s1_cache as S1C  # noqa: E402
import s2_nodes as S2N  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, node_cap  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, ensure_deviation_log,  # noqa: E402
                            load_config, require_approved, start_run, write_json)

# Deviation D30: augmented candidate pool. Per image, the D21 node set (s2_nodes.node_set: D3 proposals, tau_d dedup,
# D4 cap) followed by SAM 3 text instances: for every distinct SAM 3 phrase of the image's queries (object 'the X';
# part / chain 'the P of the X' and 'the X', as e_select builds edit pools; other texts verbatim) its top-k instances
# by SAM 3 text score (b_baselines.Sam3Text scoring and mask threshold). Instances are taken rank by rank over the
# phrases up to max_instances, an instance with IoU >= dup_iou to an earlier one is dropped. The containment tree,
# the step targets (s2_nodes.targets) and, for PACO, the part labels are rebuilt over the whole pool, so the S2
# selector is trained to rank its own nodes against SAM 3's outputs, the pool it is evaluated on.

SOURCES = ("paco_train", "pascal_part_train", "grefcoco_train")
CHAIN_RE = re.compile(r"^Locate the (.+?) <SEL> and then its (.+?) <SEL>\.$")
PART_RE = re.compile(r"^Locate the (.+?) of the (.+?)\. <SEL>$")
OBJECT_RE = re.compile(r"^Locate the (.+?)\. <SEL>$")
FILES = ("scripts/s3_pool.py", "scripts/b_baselines.py")


def phrases(text: str) -> list[str]:
    if m := CHAIN_RE.match(text):
        return [f"the {m[2]} of the {m[1]}", f"the {m[1]}"]
    if m := PART_RE.match(text):
        return [f"the {m[1]} of the {m[2]}", f"the {m[2]}"]
    if m := OBJECT_RE.match(text):
        return [f"the {m[1]}"]
    return [re.sub(r"\s*<SEL>\s*", " ", text).strip().rstrip(".")]


def settings(cfg: dict) -> dict:
    return cfg["s3_pool"]


def num_shards(cfg: dict, source: str, smoke: bool) -> int:
    return 1 if smoke else int(settings(cfg)["num_shards"][source])


def identity(cfg: dict, source: str, smoke: bool) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    base = S2N.identity(cfg, source, smoke)
    base.pop("smoke_images", None)
    s = settings(cfg)
    return {"kind": "s3_pool", "source": source, "smoke": smoke, "nodes": base,
            "smoke_images": s["smoke_images"] if smoke else None, "sam3_text": cfg["baselines"]["sam3_text"],
            "pool": {k: s[k] for k in ("instances_per_phrase", "max_instances", "dup_iou")},
            "num_shards": num_shards(cfg, source, smoke), "code": h.hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    sub = "smoke" if ident["smoke"] else "full"
    return Path(cfg["cache_root"]) / sub / "s3_pool" / ident["source"] / canonical_hash(ident)[:16]


class TextInstances:
    # b_baselines.Sam3Text scoring and mask post-processing, vision features once per image, top-k per phrase
    def __init__(self, cfg: dict):
        self.st = B.Sam3Text(cfg)
        self.tok = self.st.proc.tokenizer
        self.limit = min(self.tok.model_max_length, 32)

    def clip(self, p: str) -> str:
        # SAM 3's CLIP text encoder takes 32 positions; longer phrases keep their leading words
        words = B.concept(p).split()
        while len(words) > 1 and len(self.tok(" ".join(words)).input_ids) > self.limit:
            words.pop()
        return " ".join(words)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrase_list: list[str], k: int) -> list[list[tuple[np.ndarray, float]]]:
        st = self.st
        h, w = image.shape[:2]
        pix = st.proc(images=Image.fromarray(image), return_tensors="pt").pixel_values
        ve = st.model.get_vision_features(pixel_values=pix.to(st.device, st.dtype))
        out = []
        for p in phrase_list:
            t = st.proc(text=self.clip(p), return_tensors="pt").to(st.device)
            r = st.model(vision_embeds=ve, input_ids=t.input_ids, attention_mask=t.attention_mask)
            s = r.pred_logits[0].float().sigmoid()
            if r.presence_logits is not None:
                s = s * r.presence_logits[0].float().sigmoid()
            got = []
            for i in torch.argsort(s, descending=True, stable=True)[:k].tolist():
                m = F.interpolate(r.pred_masks[0, i][None, None].float().sigmoid(), size=(h, w), mode="bilinear",
                                  align_corners=False)[0, 0] > st.s["mask_threshold"]
                got.append((m.cpu().numpy(), float(s[i])))
            out.append(got)
        return out


def augment(nodes: list, inst: list[list[tuple[np.ndarray, float]]], max_n: int, dup: float
            ) -> tuple[list, list[dict]]:
    # rank by rank over the phrases (every phrase's top-1 first), empty and duplicate instances dropped
    added, meta = [], []
    depth = max((len(x) for x in inst), default=0)
    for rank, (pi, lst) in itertools.product(range(depth), enumerate(inst)):
        if len(added) >= max_n:
            break
        if rank >= len(lst) or not lst[rank][0].any():
            continue
        r = M.encode(lst[rank][0])[0]
        if added and M.iou([r], added)[0].max() >= dup:
            continue
        added.append(r)
        meta.append({"phrase": pi, "rank": rank, "score": lst[rank][1]})
    return nodes + added, meta


def run_shard(cfg: dict, source: str, shard: int, smoke: bool) -> None:
    ensure_deviation_log(cfg)
    require_approved(cfg, ["D3", "D4", "D21", "D30"])
    n = num_shards(cfg, source, smoke)
    if not 0 <= shard < n:
        raise ProtocolError(f"shard {shard} outside 0..{n - 1}")
    ident = identity(cfg, source, smoke)
    d = out_dir(cfg, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    (d / "nodes").mkdir(exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.pt"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"s3_pool_{source}", cfg, smoke, {"shard": shard, "num_shards": n, "cache": str(d)})
    device = cfg["device"]
    m, p, s = cfg["models"], cfg["proposals"], settings(cfg)
    sam, sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], device, p["dtype"], cfg.get("hf_home"))
    text = TextInstances(cfg)
    cap = node_cap(cfg, smoke)
    rows = S1C.items(cfg, source, shard, n)
    if source == "paco_train":
        # S2's PACO training images (the worklist's paco_train role), not every PACO train image
        wl = manifest_inputs(cfg, False)[3]
        keep = set(wl[wl.role == "paco_train"].image_key)
        rows = (r for r in rows if r[0] in keep)
    if smoke:
        rows = itertools.islice(rows, s["smoke_images"])
    recs, totals, t0 = [], Counter(), time.time()
    for i, (key, path, queries, img) in enumerate(rows, 1):
        t1 = time.time()
        image = load_rgb(assert_real_path(path))
        nodes = S2N.node_set(sam, sproc, cfg, image, cap)
        plist = list(dict.fromkeys(ph for _, _, t, _ in queries for ph in phrases(t)))
        pool, imeta = augment(nodes, text(image, plist, s["instances_per_phrase"]), s["max_instances"], s["dup_iou"])
        rec, counts = S2N.record(d, key, path, image, pool, queries, device)
        totals.update(counts)
        totals["instances"] += len(imeta)
        if rec is not None:
            rec.update(n_nodes=len(nodes), phrases=plist, instances=imeta)
            if img is not None:
                lab = paco_part_labels(pool, img)
                rec.update(labels=list(lab.label), parts=list(lab.part), high=lab.high_overlap.tolist())
            recs.append(rec)
        print(f"[{i}] {key} nodes {len(nodes)} instances {len(imeta)} queries {counts['queries']} "
              f"absent {counts['absent']} {time.time() - t1:.1f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    torch.save({"records": recs, "counts": dict(totals)}, tmp)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(recs), "counts": dict(totals), "seconds": time.time() - t0,
                                      "deviations": ["D21", "D30"]})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--source", required=True, choices=SOURCES)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    run_shard(load_config(args.config), args.source, args.shard, args.smoke)


if __name__ == "__main__":
    main()
