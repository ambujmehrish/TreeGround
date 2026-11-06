import argparse
import base64
import io
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.run import load_config  # noqa: E402

# Read-only figure assets (no protocol stage, nothing tuned): runs the trained checkpoint of a ta_test config on PACO
# test part queries of one label (default mug:handle) exactly as ta_test_eval scores a single part query, and stores
# per image the JPEG, the query's candidate pool (grid nodes + SAM 3 text instances with phrase / rank / score), the
# containment parents, our score per candidate, our pick, SAM 3's own rank-1 instance of the part phrase, the best
# mask in the pool and the ground truth, as JSON under $TG_ROOT/results/figures/ (the collector syncs JSON only).


def jpeg_b64(image: np.ndarray, quality: int = 92) -> str:
    buf = io.BytesIO()
    Image.fromarray(image).save(buf, format="JPEG", quality=quality)
    return base64.b64encode(buf.getvalue()).decode()


def rle_json(r: M.Rle) -> dict:
    c = r["counts"]
    return {"size": list(r["size"]), "counts": c.decode() if isinstance(c, bytes) else c}


def candidate_images(root: str, label: str) -> list[int]:
    data = json.loads(paco.ann_path(root, "test").read_text())
    cat = next(c["id"] for c in data["categories"] if c["name"] == label)
    return sorted({a["image_id"] for a in data["annotations"] if a["category_id"] == cat})


def sam3_rank1(meta: list[dict], n_grid: int) -> int:
    # pool index of SAM 3's own top instance of the first phrase (the part phrase), -1 if it was dropped as empty
    return next((n_grid + i for i, m in enumerate(meta) if m["phrase"] == 0 and m["rank"] == 0), -1)


def record(scorer, image: np.ndarray, img: paco.PacoImage, q: paco.Query) -> dict:
    import ta_test_eval as TTE
    from treeground import tree as T
    from treeground import tree_attn as TA
    s = scorer
    rles, _ = s.nodes(image)
    phr = s.S3P.phrases(q.text)
    inst = s.text(image, phr, s.ps["instances_per_phrase"])
    pool, meta = s.S3P.augment(rles, inst, s.ps["max_instances"], s.ps["dup_iou"])
    parents = T.build_tree(pool, s.device).parent
    P = TTE.Pool(pool, parents, len(rles), TA.pool_features(len(rles), meta))
    sq = TTE.queries_of(pd.DataFrame([{"qid": q.qid, "kind": q.kind, "text": q.text}]))[0]
    lg = s.logits(image, img.key, P, [sq])[0][0]
    gt = paco.gt_rles(img)[q.gt[0]]
    ious = M.iou(pool, [gt])[:, 0]
    pick, top1 = int(np.argmax(lg)), sam3_rank1(meta, len(rles))
    part = next(p for p in img.parts if p.ann_id == q.gt[0])
    obj = next(o for o in img.objects if o.ann_id == part.obj_ann_id)
    return {"image_id": img.image_id, "query": q.text, "phrases": phr, "h": img.h, "w": img.w,
            "image_jpeg_b64": jpeg_b64(image), "gt_part": rle_json(gt), "gt_object": rle_json(obj.rle),
            "pool": [rle_json(r) for r in pool], "n_grid": len(rles), "instances": meta,
            "parents": [int(x) for x in parents[:len(pool)]], "logits": [float(x) for x in lg],
            "pick": pick, "pick_iou": float(ious[pick]), "sam3_top1": top1,
            "sam3_top1_iou": float(ious[top1]) if top1 >= 0 else None,
            "best": int(np.argmax(ious)), "best_iou": float(ious.max())}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ta_test_pool_soft_fam.yaml")
    ap.add_argument("--label", default="mug:handle")
    ap.add_argument("--max_images", type=int, default=40)
    ap.add_argument("--keep", type=int, default=8)
    args = ap.parse_args()
    import ta_test_eval as TTE
    cfg = load_config(ROOT / args.config)
    ids = candidate_images(cfg["data"]["paco_lvis"], args.label)[:args.max_images]
    imgs = paco.load(cfg["data"]["paco_lvis"], "test", set(ids))
    scorer = TTE.Scorer(cfg)
    out = []
    for i in ids:
        img = imgs[i]
        qs = [q for q in paco.i1_queries(img)[0] if q.kind == "part" and q.qid.endswith(f"|part|{args.label}")]
        if not qs:
            continue
        image = load_rgb(paco.image_path(cfg["data"]["coco_images"], img))
        out.append(record(scorer, image, img, qs[0]))
        print(f"{img.image_id}: pick IoU {out[-1]['pick_iou']:.2f} SAM 3 top-1 IoU {out[-1]['sam3_top1_iou']} "
              f"best {out[-1]['best_iou']:.2f} pool {len(out[-1]['pool'])}", flush=True)
    # examples where the pool holds the part and our pick finds it; ordered by our IoU minus SAM 3's own top-1 IoU
    out.sort(key=lambda r: -(r["pick_iou"] - (r["sam3_top1_iou"] or 0.0)))
    dest = Path(os.environ["TG_ROOT"]) / "results" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / f"overview_assets_{args.label.replace(':', '_')}.json"
    path.write_text(json.dumps({"config": args.config, "label": args.label, "examined": len(out),
                                "examples": out[:args.keep]}))
    print(path)


if __name__ == "__main__":
    main()
