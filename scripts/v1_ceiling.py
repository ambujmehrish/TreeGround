import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import ProtocolError, assert_real_path, load_config, start_run, write_json  # noqa: E402

# Diagnostics after a failed V1 gate. Nothing here is a gate quantity and nothing is tuned:
#   objects: how often the cached proposals recover whole PACO objects (pipeline sanity check)
#   oracle:  SAM 3 prompted at one interior point of each ground-truth mask, alone and with the parent
#            object's box, plus the mask's own box; the upper bound for any re-proposal inside parents
THR = 0.5
PROMPTS = ("point", "point+parent_box", "own_box")


def interior_point(mask: np.ndarray) -> tuple[float, float]:
    # pixel farthest from the mask boundary; ties go to the one nearest the centroid, then raster order
    dist = cv2.distanceTransform(np.pad(mask.astype(np.uint8), 1), cv2.DIST_L2, 5)[1:-1, 1:-1]
    ys, xs = np.nonzero(dist >= dist.max())
    cy, cx = np.argwhere(mask).mean(0)
    k = int(np.argmin((ys - cy) ** 2 + (xs - cx) ** 2))
    return xs[k] + 0.5, ys[k] + 0.5


def summarize(df: pd.DataFrame, col: str) -> dict:
    if not len(df):
        return {"n": 0}
    return {"recall": float(df[col].mean()), "n": len(df),
            "by_area_bin": {k: {"recall": float(g[col].mean()), "n": len(g)} for k, g in df.groupby("area_bin")}}


def val_images(cfg: dict, wl: pd.DataFrame, max_images: int | None):
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key")
    if max_images:
        rows = rows.iloc[:max_images]
    images = paco.load(cfg["data"]["paco_lvis"], "val", {int(k.split(":")[1]) for k in rows.image_key})
    return [(r, images[int(r.image_key.split(":")[1])]) for r in rows.itertuples()]


def object_recall(props, items, taus) -> tuple[dict, pd.DataFrame]:
    recs = []
    for row, img in items:
        if not img.objects:
            continue
        gts = [o.rle for o in img.objects]
        gt_area = M.areas(gts)
        for tau in [*taus, None]:
            nodes = props.nodes(row.sha1, tau)
            best = M.iou(nodes["rles"], gts).max(0) if nodes["rles"] else np.zeros(len(gts))
            for k, o in enumerate(img.objects):
                recs.append({"image_key": row.image_key, "ann_id": o.ann_id, "label": o.name,
                             "tau": "pre_dedup" if tau is None else str(tau), "area_bin": area_bin(gt_area[k]),
                             "best_iou": float(best[k]), "hit": bool(best[k] >= THR)})
    df = pd.DataFrame(recs)
    return {t: summarize(g, "hit") for t, g in df.groupby("tau")}, df


@torch.no_grad()
def oracle_image(model, processor, image: np.ndarray, img: paco.PacoImage, pcfg: P.ProposalConfig,
                 batch: int) -> list[dict]:
    h, w = image.shape[:2]
    device = next(model.parameters()).device
    dtype = getattr(torch, pcfg.dtype)
    inputs = processor(images=image, return_tensors="pt")
    emb = model.get_image_embeddings(inputs["pixel_values"].to(device, dtype))
    scale = np.array([processor.target_size / w, processor.target_size / h])
    obj_box = {o.ann_id: M.boxes_xyxy([o.rle])[0] for o in img.objects}
    gts = [("object", o.ann_id, o.name, o.rle, None) for o in img.objects]
    gts += [("part", p.ann_id, p.label, p.rle, obj_box[p.obj_ann_id]) for p in img.parts]
    dense = np.stack([M.decode(g[3]) for g in gts]) if gts else np.zeros((0, h, w), bool)
    areas = dense.reshape(len(gts), -1).sum(1)
    points = np.array([interior_point(m) for m in dense]).reshape(-1, 2)
    own_box = M.boxes_xyxy([g[3] for g in gts]).reshape(-1, 4)
    out = []
    for prompt in PROMPTS:
        idx = [i for i, g in enumerate(gts) if prompt != "point+parent_box" or g[4] is not None]
        for s in range(0, len(idx), batch):
            b = idx[s:s + batch]
            kw = {}
            if prompt in ("point", "point+parent_box"):
                kw["input_points"] = torch.tensor(points[b] * scale, dtype=dtype, device=device)[None, :, None, :]
                kw["input_labels"] = torch.ones(1, len(b), 1, dtype=torch.int64, device=device)
            boxes = np.stack([gts[i][4] for i in b]) if prompt == "point+parent_box" else own_box[b]
            if prompt != "point":
                kw["input_boxes"] = torch.tensor(boxes * np.tile(scale, 2), dtype=dtype, device=device)[None]
            res = model(image_embeddings=emb, multimask_output=True, **kw)
            logits = processor.post_process_masks([res.pred_masks[0].float()], [[h, w]], binarize=False)[0]
            binary = logits > pcfg.mask_threshold
            stab = P.stability_scores(logits, pcfg.mask_threshold, pcfg.stability_offset)
            gt = torch.from_numpy(dense[b]).to(binary.device)[:, None]
            inter = (binary & gt).flatten(-2).sum(-1).double()
            m_area = binary.flatten(-2).sum(-1)
            union = m_area.double() + gt.flatten(-2).sum(-1).double() - inter
            iou = (inter / union.clamp(min=1)).cpu().numpy()
            kept = ((stab >= pcfg.stability_thresh) & (m_area >= pcfg.min_area)).cpu().numpy()
            top = res.iou_scores[0].float().argmax(-1).cpu().numpy()
            for j, i in enumerate(b):
                kind, ann_id, label, _, _ = gts[i]
                out.append({"image_key": img.key, "kind": kind, "ann_id": ann_id, "label": label, "prompt": prompt,
                            "area_bin": area_bin(areas[i]), "best_iou": float(iou[j].max()),
                            "best_iou_kept": float(iou[j][kept[j]].max()) if kept[j].any() else 0.0,
                            "top_scored_iou": float(iou[j, top[j]])})
    return out


def oracle_summary(df: pd.DataFrame) -> dict:
    res = {}
    for (kind, prompt), g in df.groupby(["kind", "prompt"]):
        g = g.assign(any_mask=g.best_iou >= THR, kept_mask=g.best_iou_kept >= THR, top_mask=g.top_scored_iou >= THR)
        res[f"{kind}/{prompt}"] = {
            "any_of_3_masks": summarize(g, "any_mask"),
            "stability_and_area_filtered": summarize(g, "kept_mask"),
            "top_scored_mask": summarize(g, "top_mask"),
        }
    return res


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--no-oracle", action="store_true")
    ap.add_argument("--max-images", type=int)
    ap.add_argument("--batch", type=int, default=64)
    args = ap.parse_args()
    cfg = load_config(args.config)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    props = open_proposals(cfg, args.smoke, manifest, mhash)
    run = start_run("v1_ceiling", cfg, args.smoke, {"manifest_run": str(mrun), "proposals": str(props.path),
                                                     "max_images": args.max_images, "oracle": not args.no_oracle})
    items = val_images(cfg, wl, args.max_images)
    print(f"{len(items)} PACO val images", flush=True)
    out = {"split": "PACO-LVIS val", "label": "dev", "diagnostic": True, "iou_threshold": THR}
    out["objects"], obj_df = object_recall(props, items, cfg["v1"]["taus"])
    obj_df.to_csv(run / "object_recall.csv", index=False)
    print("object recall", {k: round(v["recall"], 3) for k, v in out["objects"].items()}, flush=True)
    write_json(run / "metrics.json", out)
    if not args.no_oracle:
        p, sam = cfg["proposals"], cfg["models"]["sam3"]
        pcfg = P.ProposalConfig(p["points_per_side"], p["masks_per_point"], p["stability_thresh"],
                                p["stability_offset"], p["mask_threshold"], p["min_area"], p["points_per_batch"],
                                p["dtype"])
        model, processor = P.load_sam3(sam["repo"], sam["revision"], cfg["device"], p["dtype"], cfg.get("hf_home"))
        recs, t0 = [], time.time()
        for i, (row, img) in enumerate(items, 1):
            image = load_rgb(assert_real_path(row.path))
            if image.shape[:2] != (img.h, img.w):
                raise ProtocolError(f"{row.image_key}: decoded size differs from the annotation")
            recs += oracle_image(model, processor, image, img, pcfg, args.batch)
            if i % 100 == 0 or i == len(items):
                print(f"[{i}/{len(items)}] oracle prompts {len(recs)} {time.time() - t0:.0f}s", flush=True)
        df = pd.DataFrame(recs)
        df.to_csv(run / "oracle_prompts.csv", index=False)
        out["oracle"] = oracle_summary(df)
        parts = df[(df.kind == "part") & (df.prompt == "point+parent_box")]
        cats = parts.assign(hit=parts.best_iou >= THR).groupby("label").hit.agg(["mean", "size"])
        out["oracle_part_categories_point+parent_box"] = {
            k: {"recall": float(r["mean"]), "n": int(r["size"])}
            for k, r in cats.sort_values("size", ascending=False).head(30).iterrows()}
        for k, v in out["oracle"].items():
            print(k, "any-of-3", round(v["any_of_3_masks"]["recall"], 3),
                  "filtered", round(v["stability_and_area_filtered"]["recall"], 3),
                  "top", round(v["top_scored_mask"]["recall"], 3), flush=True)
    write_json(run / "metrics.json", out)
    print(run)


if __name__ == "__main__":
    main()
