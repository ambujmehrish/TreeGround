import argparse
import hashlib
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.data.paco import OBJECT_TEMPLATE, PART_TEMPLATE  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, load_config, require_approved, start_run,  # noqa: E402
                            write_json)

# Deviation D38: the part-grounding protocol of OP-HRG (Mehrab et al., ECCV 2026, arXiv 2607.15374; benchmark loaders
# of github.com/sajeedmehrab/op-hrg, MIT licence, ported here line for line) so its Table 1 can be cited unchanged:
#   InstructPart test (600 images, one query '<object>'s <part>' per image from the file name, GT = mask > 0);
#   PascalPart (the 924 image ids of their pascalpart_val.txt, Pascal-Part .mat annotations, VOC JPEGs; per object
#     class with parts one object query (GT = union of the class's instance masks) and one query per readable part
#     name (GT = union of that part over the class's instances));
#   PartImageNet (PartImageNet_OOD test.json: annotations merged by (image, category), GT = PIL-filled polygons,
#     query '<supercategory>'s <part>').
# gIoU = mean per-query IoU (0 when the union is empty), cIoU = sum intersection / sum union; parts and objects apart.
# Ours answers each query top-1 over its candidate pool (s3_eval.image_rows; with s5_eval the D36 pool), worded with
# our templates from the same object / part names; nothing is tuned on these sets. Baselines: D13 / D31 predictors.

BENCHMARKS = ("instructpart", "pascalpart", "partimagenet")
FILES = ("scripts/ophrg_eval.py", "scripts/s3_eval.py", "scripts/s3_pool.py", "scripts/s2_train.py",
         "scripts/s2_nodes.py", "scripts/b_baselines.py", "scripts/b_baselines2.py", "treeground/data/ophrg_pascalpart_val.txt")
PASCAL_IDS = ROOT / "treeground" / "data" / "ophrg_pascalpart_val.txt"
READABLE = {
    "lwing": "left wing", "rwing": "right wing", "lleg": "left leg", "rleg": "right leg", "lfoot": "left foot",
    "rfoot": "right foot", "leye": "left eye", "reye": "right eye", "lear": "left ear", "rear": "right ear",
    "llleg": "left lower leg", "luleg": "left upper leg", "rlleg": "right lower leg", "ruleg": "right upper leg",
    "llarm": "left lower arm", "luarm": "left upper arm", "rlarm": "right lower arm", "ruarm": "right upper arm",
    "lhand": "left hand", "rhand": "right hand", "lebrow": "left eyebrow", "rebrow": "right eyebrow",
    "lfleg": "left front leg", "rfleg": "right front leg", "lfpa": "left front paw", "rfpa": "right front paw",
    "lbleg": "left back leg", "rbleg": "right back leg", "lbpa": "left back paw", "rbpa": "right back paw",
    "lfuleg": "left front upper leg", "lflleg": "left front lower leg", "rfuleg": "right front upper leg",
    "rflleg": "right front lower leg", "lbuleg": "left back upper leg", "lblleg": "left back lower leg",
    "rbuleg": "right back upper leg", "rblleg": "right back lower leg", "lfho": "left front hoof",
    "rfho": "right front hoof", "lbho": "left back hoof", "rbho": "right back hoof",
    "head": "head", "torso": "torso", "neck": "neck", "tail": "tail", "beak": "beak", "nose": "nose",
    "mouth": "mouth", "muzzle": "muzzle", "hair": "hair", "ear": "ear", "eye": "eye", "horn": "horn",
    "screen": "screen", "roofside": "roof side", "frontside": "front side", "backside": "back side",
    "leftside": "left side", "rightside": "right side", "cap": "cap", "body": "body", "pot": "pot", "plant": "plant",
    "saddle": "saddle", "handlebar": "handle bar", "chainwheel": "chain wheel", "fwheel": "front wheel",
    "bwheel": "back wheel", "wheel": "wheel", "headlight": "headlight", "window": "window", "door": "door",
    "fliplate": "front license plate", "bliplate": "back license plate", "mirror": "mirror",
    "leftmirror": "left mirror", "rightmirror": "right mirror", "lhorn": "left horn", "rhorn": "right horn",
    "hfrontside": "head front side", "hleftside": "head left side", "hrightside": "head right side",
    "hbackside": "head back side", "hroofside": "head roof side", "coach": "coach", "cfrontside": "coach front side",
    "cleftside": "coach left side", "crightside": "coach right side", "cbackside": "coach back side",
    "croofside": "coach roof side", "engine": "engine", "stern": "stern"}


def pascal_ids() -> list[str]:
    return sorted(line.strip() for line in PASCAL_IDS.read_text().splitlines() if line.strip())


def partimagenet_ood_test(root: str | Path) -> tuple[dict, Path]:
    hits = [p for p in Path(root).rglob("test.json") if "__MACOSX" not in p.parts]
    if len(hits) != 1:
        raise ProtocolError(f"expected one PartImageNet_OOD test.json under {root}, found {hits}")
    return json.loads(hits[0].read_text()), hits[0].parent / "test"


# ---- benchmark records: (image key, image path, list of (qid, kind, object, part, gt bool mask)) ----

def instructpart_images(cfg: dict) -> list[tuple]:
    d = Path(cfg["data"]["instructpart"]) / "test"
    out = []
    for name in sorted(os.listdir(d / "images")):
        base = os.path.splitext(name)[0]
        obj, part = base.split("-")[-2], base.split("-")[-1]
        out.append((f"instructpart:{base}", str(d / "images" / name),
                    [("instructpart|" + base, "part", obj, part, ("png", str(d / "masks" / f"{base}.png")))]))
    return out


def pascalpart_images(cfg: dict) -> list[tuple]:
    out = []
    for i in pascal_ids():
        out.append((f"pascalpart116:{i}", str(Path(cfg["data"]["pascal_voc"]) / "JPEGImages" / f"{i}.jpg"),
                    [("mat", str(Path(cfg["data"]["pascal_part_ann"]) / f"{i}.mat"))]))
    return out


def pascal_queries(key: str, mat: str) -> list[tuple]:
    import scipy.io
    anno = scipy.io.loadmat(mat)["anno"]
    objs: dict[str, dict] = {}
    for obj in anno[0][0]["objects"][0]:
        name = str(obj["class"][0])
        try:
            parts = obj["parts"][0]
        except IndexError:
            continue
        o = objs.setdefault(name, {"maps": [], "parts": defaultdict(list)})
        o["maps"].append(obj["mask"] > 0)
        for p in parts:
            o["parts"][READABLE[str(p["part_name"][0]).split("_")[0]]].append(p["mask"] > 0)
    qs = []
    for name, o in objs.items():
        qs.append((f"{key}|object|{name}", "object", name, None, np.logical_or.reduce(o["maps"])))
        for part, ms in o["parts"].items():
            qs.append((f"{key}|part|{name}:{part}", "part", name, part, np.logical_or.reduce(ms)))
    return qs


def partimagenet_images(cfg: dict) -> list[tuple]:
    data, img_dir = partimagenet_ood_test(cfg["data"]["partimagenet_ood"])
    files = {im["id"]: im["file_name"] for im in data["images"]}
    cats = {c["id"]: c["name"] for c in data["categories"]}
    grouped = defaultdict(list)
    for a in data["annotations"]:
        grouped[(a["image_id"], a["category_id"])].append(a)
    per_image = defaultdict(list)
    for n, ((iid, cid), anns) in enumerate(grouped.items()):
        words = cats[cid].split()
        obj, part = words[0].lower().strip("."), " ".join(words[1:]).lower().strip(".")
        segs = [a["segmentation"] for a in anns if a["segmentation"]]
        if not segs or not part:
            continue
        per_image[iid].append((f"partimagenet|{n}", "part", obj, part, ("poly", segs)))
    out = []
    for iid, qs in sorted(per_image.items()):
        fn = files[iid]
        out.append((f"partimagenet_ood:{iid}", str(img_dir / fn.split("_")[0] / fn), qs))
    return out


def poly_mask(segs: list, h: int, w: int) -> np.ndarray:
    # their PartImageNetDataset._create_masks: every polygon filled with PIL, union over annotations
    out = np.zeros((h, w), bool)
    for seg in segs:
        img = Image.new("L", (w, h), 0)
        draw = ImageDraw.Draw(img)
        for poly in seg:
            draw.polygon([(poly[i], poly[i + 1]) for i in range(0, len(poly), 2)], fill=1)
        out |= np.asarray(img) > 0
    return out


def resolve(key: str, qs: list, h: int, w: int) -> list[tuple]:
    if len(qs) == 1 and qs[0][0] == "mat":
        return pascal_queries(key, qs[0][1])
    out = []
    for qid, kind, obj, part, g in qs:
        if g[0] == "png":
            gt = np.asarray(Image.open(g[1])) > 0
            gt = gt if gt.ndim == 2 else gt.any(-1)
        else:
            gt = poly_mask(g[1], h, w)
        out.append((qid, kind, obj, part, gt))
    return out


LOADERS = {"instructpart": instructpart_images, "pascalpart": pascalpart_images, "partimagenet": partimagenet_images}


def text(kind: str, obj: str, part: str | None) -> str:
    return OBJECT_TEMPLATE.format(object=obj) if kind == "object" else PART_TEMPLATE.format(part=part, object=obj)


# ---- predictors ----

class Ours:
    def __init__(self, cfg: dict):
        import s2_nodes as S2N
        import s3_eval as S3E
        import s3_pool as S3P
        from treeground import proposals as P
        from treeground.identity import node_cap, qwen_static
        self.cfg, self.S2N, self.S3E = cfg, S2N, S3E
        dev = cfg["device"]
        self.model, self.proc, self.extra, self.sel, self.meta, _ = S3E.load_model(cfg, qwen_static(cfg), dev)
        m, p = cfg["models"], cfg["proposals"]
        self.sam, self.sproc = P.load_sam3(m["sam3"]["repo"], m["sam3"]["revision"], dev, p["dtype"], cfg.get("hf_home"))
        self.text = S3P.TextInstances(cfg)
        self.cap = node_cap(cfg, False)

    def __call__(self, image: np.ndarray, key: str, qs: list) -> list[dict]:
        gts = {i: M.encode(q[4])[0] for i, q in enumerate(qs)}
        g = pd.DataFrame([{"qid": q[0], "kind": q[1], "text": text(q[1], q[2], q[3]), "n_sel": 1,
                           "gt": json.dumps([i]), "image_key": key} for i, q in enumerate(qs)])
        nodes = self.S2N.node_set(self.sam, self.sproc, self.cfg, image, self.cap)
        rows = self.S3E.image_rows(self.model, self.proc, self.extra, self.sel, self.meta["layer"], self.cfg["device"],
                                   self.text, self.cfg, g, image, nodes, gts)
        return [{"qid": r["qid"], "kind": r["kind"], "image": key, "inter": r["sel_inter"],
                 "union": r["sel_union"], "pick_kind": r["pick_kind"]} for r in rows]


class Baseline:
    def __init__(self, cfg: dict, method: str):
        import b_baselines as B
        if method not in B.PREDICTORS:
            import b_baselines2  # noqa: F401
        self.B, self.pred = B, B.PREDICTORS[method](cfg)

    def __call__(self, image: np.ndarray, key: str, qs: list) -> list[dict]:
        phrases = [self.B.phrase(text(q[1], q[2], q[3])) for q in qs]
        rows = []
        for q, (mask, _, _, acc) in zip(qs, self.pred(image, phrases)):
            pred = mask.astype(bool) if (acc and mask is not None) else np.zeros(q[4].shape, bool)
            rows.append({"qid": q[0], "kind": q[1], "image": key, "inter": int((pred & q[4]).sum()),
                         "union": int((pred | q[4]).sum())})
        return rows


# ---- run ----

def stage_name(cfg: dict, method: str) -> str:
    if method != "ours":
        return method
    import s2_train as S2T
    return S2T.s2_stage(cfg)


def identity(cfg: dict, method: str, bench: str) -> dict:
    h = hashlib.sha256()
    for rel in FILES:
        h.update((ROOT / rel).read_bytes())
    ident = {"kind": "ophrg_eval", "benchmark": bench, "method": method, "stage": stage_name(cfg, method),
             "num_shards": cfg["ophrg_eval"]["num_shards"], "code": h.hexdigest()}
    if method == "ours":
        import s2_eval as S2E
        ident.update(run=S2E.load_checkpoint(cfg, False)[1].name, checkpoint=cfg["s2_eval"]["checkpoint"],
                     pool=cfg["s3_pool"]["instances_per_phrase"], sam3_text=cfg["baselines"]["sam3_text"])
    else:
        ident.update(models=cfg["models"], baselines=cfg["baselines"])
    return ident


def out_dir(cfg: dict, method: str, bench: str, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / "full" / "ophrg_eval" / stage_name(cfg, method) / bench /
            canonical_hash(ident)[:16])


def approvals(cfg: dict, method: str) -> list[str]:
    if method != "ours":
        return ["D38", "D13", "D31"]
    import s2_train as S2T
    return ["D38", "D30"] + S2T.deviations(cfg)


def run_shard(cfg: dict, method: str, bench: str, shard: int) -> None:
    require_approved(cfg, approvals(cfg, method))
    n = cfg["ophrg_eval"]["num_shards"]
    ident = identity(cfg, method, bench)
    d = out_dir(cfg, method, bench, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"ophrg_{bench}_{stage_name(cfg, method)}_shard", cfg, False, {"shard": shard, "cache": str(d)})
    images = LOADERS[bench](cfg)[shard::n]
    predict = Ours(cfg) if method == "ours" else Baseline(cfg, method)
    rows, t0 = [], time.time()
    for i, (key, path, qs) in enumerate(images, 1):
        if not Path(path).exists():
            raise ProtocolError(f"{bench}: image {path} is missing (every listed image is evaluated)")
        image = np.asarray(Image.open(path).convert("RGB"))
        qs = resolve(key, qs, *image.shape[:2])
        for q in qs:
            if q[4].shape != image.shape[:2]:
                raise ProtocolError(f"{q[0]}: GT {q[4].shape} differs from the image {image.shape[:2]}")
        if qs:
            rows += predict(image, key, qs)
        if i % 25 == 0:
            print(f"[{i}/{len(images)}] queries {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(images), "queries": len(rows),
                                      "seconds": time.time() - t0})


def merge(cfg: dict, method: str, bench: str) -> None:
    require_approved(cfg, approvals(cfg, method))
    n = cfg["ophrg_eval"]["num_shards"]
    d = out_dir(cfg, method, bench, identity(cfg, method, bench))
    miss = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if miss:
        raise ProtocolError(f"{len(miss)} of {n} shards missing in {d}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate query rows")
    bs = cfg["v2"]["bootstrap"]
    run = start_run(f"ophrg_{bench}_{stage_name(cfg, method)}", cfg, False, {"cache": str(d)})
    df.to_csv(run / "eval_queries.csv", index=False)
    res = {}
    if bench == "pascalpart":
        # PascalPart-116 images that are also in the VOC2010 'train' list our Pascal-Part training source reads
        from treeground.data import pascal_part
        res["train_overlap_images"] = len(set(pascal_ids()) & set(pascal_part.split_ids(cfg["data"]["pascal_voc"], "train")))
    for kind, g in df.groupby("kind"):
        iou = np.where(g.union > 0, g.inter / g.union.clip(lower=1), 0.0)
        res["parts" if kind == "part" else "objects"] = {
            "queries": int(len(g)), "gIoU": mean_ci(iou, g.image.to_numpy(), bs["n"], bs["seed"]),
            "cIoU": float(g.inter.sum() / max(1, g.union.sum()))}
    write_json(run / "metrics.json", {"label": "test", "deviation": "D38", "protocol": "OP-HRG (arXiv 2607.15374)",
                                      "benchmark": bench, "method": method, "stage": stage_name(cfg, method),
                                      "images": int(df.image.nunique()), "results": res})
    print(json.dumps({k: (round(v["gIoU"]["mean"] * 100, 2), round(v["cIoU"] * 100, 2)) if isinstance(v, dict) else v
                      for k, v in res.items()}))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    ap.add_argument("--benchmark", required=True, choices=BENCHMARKS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg, args.method, args.benchmark)
    else:
        with torch.no_grad():
            run_shard(cfg, args.method, args.benchmark, args.shard)


if __name__ == "__main__":
    main()
