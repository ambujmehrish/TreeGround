import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import fig_assets as FA  # noqa: E402
from pycocotools import mask as mu  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground.data import grefcoco, paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.run import load_config  # noqa: E402

# Read-only teaser assets (no protocol stage, nothing tuned): the trained checkpoint of a ta_test config on a seeded
# random sample of test queries, scored exactly as ta_test_eval scores them, stored with image, pool, our logits and
# decision as JSON under $TG_ROOT/results/figures/teaser_assets.json (the collector syncs JSON only).
#   part:      PACO test part queries (one per sampled image); decision = argmax (pool_soft_fam's part rule: none)
#   multi:     gRefCOCO testA/testB expressions with >= 2 targets; decision = every node with sigmoid >= t
#   no_target: gRefCOCO testA/testB no-target expressions; same set rule (empty set = "no target")
# t is the nodes/set threshold the gres merge chose on gRefCOCO val (results/metrics/gres_<stage>.json).
# Kept per kind: examples our decision gets right (part / multi IoU >= 0.6, no_target empty), visible ones first.


def set_threshold(stage: str) -> float:
    return float(json.loads((ROOT / "results" / "metrics" / f"gres_{stage}.json").read_text())
                 ["results"]["nodes/set"]["t_val"])


def pool_record(image, pool, logits, query: str) -> dict:
    return {"query": query, "h": int(image.shape[0]), "w": int(image.shape[1]), "image_jpeg_b64": FA.jpeg_b64(image),
            "pool": [FA.rle_json(r) for r in pool.rles], "n_grid": int(pool.n_grid),
            "parents": [int(x) for x in pool.parents[:pool.n]], "logits": [float(x) for x in logits]}


def part_examples(cfg: dict, scorer, n: int, rng) -> list[dict]:
    root = cfg["data"]["paco_lvis"]
    ids = [im["id"] for im in paco.image_list(root, "test")]
    ids = [int(i) for i in rng.choice(ids, size=min(n, len(ids)), replace=False)]
    imgs = paco.load(root, "test", set(ids))
    out = []
    for i in ids:
        img = imgs[i]
        qs = [q for q in paco.i1_queries(img)[0] if q.kind == "part"]
        if not qs:
            continue
        q = qs[int(rng.integers(len(qs)))]
        r = FA.record(scorer, load_rgb(paco.image_path(cfg["data"]["coco_images"], img)), img, q)
        r["area"] = float(M.areas([r["gt_part"]])[0]) / (img.h * img.w)
        out.append(r)
        print(f"part {img.image_id} '{q.text}': ours {r['pick_iou']:.2f} SAM 3 top-1 {r['sam3_top1_iou']}", flush=True)
    return out


def gres_examples(cfg: dict, scorer, n: int, t: float, rng) -> dict[str, list[dict]]:
    import gres_eval as GE
    import s2_train as S2T
    from treeground import mllm
    from treeground import s2 as S2
    out = {"multi": [], "no_target": []}
    for kind in out:
        rows = []
        for split in ("testA", "testB"):
            df, anns = GE.expressions(cfg, split)
            sel = df[df.no_target] if kind == "no_target" else df[df.ann_ids.map(lambda s: len(json.loads(s)) >= 2)]
            rows += [(split, r, anns) for r in sel.itertuples()]
        for k in rng.choice(len(rows), size=min(n, len(rows)), replace=False):
            split, q, anns = rows[int(k)]
            image = load_rgb(grefcoco.image_path(cfg["data"]["coco_images"], q.file))
            text = f"{q.text} {mllm.SEL}"
            sq = S2T.Query("q0", "object", S2.user_text(text), f"{S2.user_text(q.text)} {mllm.SEL}", [])
            pools, lgs = scorer.score(image, "gres", [text], [sq])
            pool, lg = pools[0], (lgs[0][0] if pools[0].n else np.zeros(0))
            sig = 1.0 / (1.0 + np.exp(-lg))
            order = np.argsort(-sig, kind="stable")[:GE.KEEP_MAX]
            chosen = [int(i) for i in order if sig[i] >= t]
            gt = GE.gt_rle(anns, json.loads(q.ann_ids), q.h, q.w)
            if kind == "no_target":
                iou = 1.0 if not chosen else 0.0
            else:
                g = mu.decode(gt).astype(bool)
                pred = np.zeros_like(g)
                for i in chosen:
                    pred |= M.decode(pool.rles[i])
                iou = float((pred & g).sum() / max(1, (pred | g).sum()))
            r = pool_record(image, pool, lg, q.text) if pool.n else {"query": q.text}
            r.update({"qid": q.qid, "split": split, "t": t, "chosen": chosen, "iou": iou,
                      "gt": None if gt is None else FA.rle_json(gt), "n_targets": len(json.loads(q.ann_ids)),
                      "max_sigmoid": float(sig.max()) if len(sig) else None})
            out[kind].append(r)
            print(f"{kind} {q.qid} '{q.text}': chosen {len(chosen)} IoU {iou:.2f}", flush=True)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ta_test_pool_soft_fam.yaml")
    ap.add_argument("--n_part", type=int, default=120)
    ap.add_argument("--n_gres", type=int, default=80)
    ap.add_argument("--keep", type=int, default=8)   # 3 x 8 examples stay under the collector's 20MB
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import ta_test_eval as TTE
    cfg = load_config(ROOT / args.config)
    stage = Path(args.config).stem.removeprefix("ta_test_")
    t = set_threshold(f"ta_{stage}")
    rng = np.random.default_rng(args.seed)
    scorer = TTE.Scorer(cfg)
    parts = part_examples(cfg, scorer, args.n_part, rng)
    gres = gres_examples(cfg, scorer, args.n_gres, t, rng)
    examined = {"part": len(parts), "multi": len(gres["multi"]), "no_target": len(gres["no_target"])}
    # correct decisions only; parts that are visible (>= 1% of the image) and where SAM 3's own top-1 misses first
    parts = sorted([r for r in parts if r["pick_iou"] >= 0.6 and r["area"] >= 0.01],
                   key=lambda r: ((r["sam3_top1_iou"] or 0.0) >= 0.5, -r["pick_iou"]))
    multi = sorted([r for r in gres["multi"] if r["iou"] >= 0.6], key=lambda r: -r["iou"])
    # near misses first: the closest candidate scored high but below t
    nt = sorted([r for r in gres["no_target"] if r["iou"] == 1.0 and "pool" in r], key=lambda r: -(r["max_sigmoid"] or 0.0))
    dest = Path(os.environ["TG_ROOT"]) / "results" / "figures"
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "teaser_assets.json"
    path.write_text(json.dumps({
        "config": args.config, "seed": args.seed, "t": t,
        "examined": examined,
        "correct": {"part": len(parts), "multi": len(multi), "no_target": len(nt)},
        "part": parts[:args.keep], "multi": multi[:args.keep], "no_target": nt[:args.keep]}))
    print(path)


if __name__ == "__main__":
    main()
