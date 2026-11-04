import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import ta_test_eval as TTE  # noqa: E402
import ta_train as TAT  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground import tree_attn as TA  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.run import ProtocolError, assert_real_path, load_config, start_run, write_json  # noqa: E402

# Read-only diagnostic of a D44 pool checkpoint on part queries (InstructPart through the OP-HRG loader; PARTONOMY
# identification parts with the question's own GT): where the pick comes from (grid node / part-phrase instance of
# rank 1-3 / object-phrase instance), how it compares with SAM 3 text's own rank-1 instance inside the same pool and
# with the best mask the pool holds, and whether the flat part pick lies inside the predicted object (the anchored
# pick does by construction). Not a protocol stage and nothing is tuned: it explains the test rows, it does not
# replace them. Queries are asked object-first (the chain form), as ta_test_eval does for a pool checkpoint.

BENCHES = ("instructpart", "partonomy")
SOURCES = ("grid", "part_r1", "part_r2", "part_r3", "object_r1", "object_r2", "object_r3", "none")
IN_THR = 0.8


def source_of(n_grid: int, imeta: list[dict], i: int) -> str:
    if i < 0:
        return "none"
    if i < n_grid:
        return "grid"
    m = imeta[i - n_grid]
    return f"{'part' if m['phrase'] == 0 else 'object'}_r{int(m['rank']) + 1}"


def containment(inner: np.ndarray, outer: np.ndarray) -> float:
    a = int(inner.sum())
    return float((inner & outer).sum() / a) if a else 0.0


def query_pool(scorer, grid: list, inst: dict, text: str) -> tuple[TTE.Pool, list[dict]]:
    # the query's pool exactly as ta_test_eval.Scorer.pools builds it, plus augment's provenance meta
    S3P = scorer.S3P
    phrases = S3P.phrases(text)
    pool, imeta = S3P.augment(grid, [inst[p] for p in phrases], scorer.ps["max_instances"], scorer.ps["dup_iou"])
    parents = T.build_tree(pool, scorer.device).parent if pool else None
    return TTE.Pool(pool, parents, len(grid), TA.pool_features(len(grid), imeta)), imeta


def diagnose(scorer, image: np.ndarray, key: str, qid: str, text: str, obj: str, part: str, gt: np.ndarray,
             grid: list, inst: dict) -> dict:
    pool, imeta = query_pool(scorer, grid, inst, text)
    n = pool.n
    row = {"qid": qid, "image": key, "n_grid": pool.n_grid, "n_inst": n - pool.n_grid,
           "n_part_inst": sum(m["phrase"] == 0 for m in imeta)}
    if n == 0:
        row.update({f"{k}_iou": 0.0 for k in ("flat", "hard", "anchor", "rank1", "best")})
        row.update(flat_src="none", hard_src="none", best_src="none", rank1_present=False,
                   flat_in_obj=0.0, hard_in_obj=0.0, hard_fallback=True, obj_src="none")
        return row
    lg = scorer.logits(image, key, pool, [TTE.chain_query(qid, obj, part)])[0]
    gt_rle = M.encode(gt[None])[0]
    ious = M.iou(pool.rles, [gt_rle])[:, 0]
    a1, flat = TA.anchor(lg[0], lg[1], pool.parents, None)
    _, hard = TA.anchor(lg[0], lg[1], pool.parents, TA.HARD)
    _, anch = TA.anchor(lg[0], lg[1], pool.parents, TA.anchor_mode(scorer.anchor))
    desc = T.from_parents(pool.parents).strict_descendants(a1)
    rank1 = next((pool.n_grid + j for j, m in enumerate(imeta) if m["phrase"] == 0 and m["rank"] == 0), -1)
    best = int(np.argmax(ious))
    obj_mask = M.decode(pool.rles[a1]).astype(bool)
    row.update(flat_iou=float(ious[flat]), flat_src=source_of(pool.n_grid, imeta, flat),
               hard_iou=float(ious[hard]), hard_src=source_of(pool.n_grid, imeta, hard),
               hard_fallback=bool(len(desc) == 0), anchor_iou=float(ious[anch]),
               rank1_present=bool(rank1 >= 0), rank1_iou=float(ious[rank1]) if rank1 >= 0 else 0.0,
               best_iou=float(ious[best]), best_src=source_of(pool.n_grid, imeta, best),
               obj_src=source_of(pool.n_grid, imeta, a1),
               flat_in_obj=containment(M.decode(pool.rles[flat]).astype(bool), obj_mask),
               hard_in_obj=containment(M.decode(pool.rles[hard]).astype(bool), obj_mask))
    return row


def summarize(df: pd.DataFrame) -> dict:
    def share(col):
        return {s: float((df[col] == s).mean()) for s in SOURCES if (df[col] == s).any()}

    def by_src(col, iou):
        return {s: {"share": float((df[col] == s).mean()), "gIoU": float(df[iou][df[col] == s].mean())}
                for s in SOURCES if (df[col] == s).any()}
    beats = df.flat_iou - df.rank1_iou
    return {
        "queries": int(len(df)), "pool_size_mean": float((df.n_grid + df.n_inst).mean()),
        "gIoU": {"flat": float(df.flat_iou.mean()), "hard": float(df.hard_iou.mean()),
                 "anchor_as_evaluated": float(df.anchor_iou.mean()),
                 "rank1_part_instance": float(df.rank1_iou.mean()), "best_in_pool": float(df.best_iou.mean())},
        "rank1_present": float(df.rank1_present.mean()),
        "flat_pick": by_src("flat_src", "flat_iou"), "hard_pick": by_src("hard_src", "hard_iou"),
        "best_in_pool_source": share("best_src"), "object_pick_source": share("obj_src"),
        "flat_vs_rank1": {"flat_better_by_0.1": float((beats >= 0.1).mean()),
                          "rank1_better_by_0.1": float((beats <= -0.1).mean()),
                          "flat_picks_other_than_rank1_and_wins_0.1": float(((df.flat_src != "part_r1") & (beats >= 0.1)).mean())},
        "containment": {"flat_in_object_mean": float(df.flat_in_obj.mean()),
                        f"flat_in_object_ge_{IN_THR}": float((df.flat_in_obj >= IN_THR).mean()),
                        "hard_in_object_mean": float(df.hard_in_obj.mean()),
                        "hard_fallback_rate": float(df.hard_fallback.mean()),
                        "hard_minus_flat_gIoU": float((df.hard_iou - df.flat_iou).mean())},
    }


def instructpart_queries(cfg: dict):
    import ophrg_eval as OE
    for key, path, qs in OE.instructpart_images(cfg):
        image = load_rgb(assert_real_path(path))
        for qid, kind, obj, part, gt in OE.resolve(key, qs, *image.shape[:2]):
            yield image, key, qid, OE.text(kind, obj, part), obj, part, gt


def partonomy_queries(cfg: dict):
    import partonomy_eval as PE
    root = Path(cfg["data"]["partonomy"])
    qs = PE.questions(root)
    segs = {}
    for r in qs:
        for p in r["parts"]:
            segs.setdefault((r["image_path"], p), r["segmentations"][p])     # the first question's own GT
    df = PE.items(qs)
    for img, g in df.groupby("image", sort=True):
        image = load_rgb(assert_real_path(str(root / img)))
        label = g.label.iloc[0]
        for p in g.part:
            gt = PE.gt_mask(segs[(img, p)], image.shape[:2])
            yield image, img, f"partonomy|part|{label}:{p}", f"Locate {PE.phrase(p, label)}. <SEL>", label, p, gt


QUERIES = {"instructpart": instructpart_queries, "partonomy": partonomy_queries}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="a ta_test_pool_<variant>.yaml")
    ap.add_argument("--bench", required=True, choices=BENCHES)
    ap.add_argument("--max_images", type=int, default=0)
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if not TAT.is_pool(cfg["ta"]["variant"]):
        raise ProtocolError("the pool diagnostic needs a pool_* checkpoint")
    scorer = TTE.Scorer(cfg)
    run = start_run(f"ta_pool_diag_{TAT.stage(cfg)}_{args.bench}", cfg, False,
                    {"bench": args.bench, "anchor": scorer.anchor, "max_images": args.max_images})
    rows, cache, t0 = [], {}, time.time()
    with torch.no_grad():
        for image, key, qid, text, obj, part, gt in QUERIES[args.bench](cfg):
            if key not in cache:
                if args.max_images and len(cache) >= args.max_images:
                    break
                if len(cache) >= 1:
                    cache.clear()                      # one image at a time
                cache[key] = (scorer.nodes(image)[0], {})
            grid, inst = cache[key]
            for p in scorer.S3P.phrases(text):
                if p not in inst:
                    inst[p] = scorer.text(image, [p], scorer.ps["instances_per_phrase"])[0]
            rows.append(diagnose(scorer, image, key, qid, text, obj, part, gt, grid, inst))
            if len(rows) % 50 == 0:
                print(f"[{len(rows)}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    df.to_csv(run / "rows.csv", index=False)
    res = summarize(df)
    write_json(run / "metrics.json", {"label": "test (read-only diagnostic)", "bench": args.bench,
                                      "stage": TAT.stage(cfg), "anchor": scorer.anchor, **res})
    print(json.dumps(res, indent=1))
    print(run)


if __name__ == "__main__":
    main()
