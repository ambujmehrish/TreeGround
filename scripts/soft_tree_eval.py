import argparse
import hashlib
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

import s3_eval as S3E  # noqa: E402  (installs the D30 hooks)
import s2_train as S2T  # noqa: E402
import s3_pool as S3P  # noqa: E402
import x_transfer_s2 as X2  # noqa: E402
from treeground import masks as M  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.data.images import load_rgb  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, require_approved,  # noqa: E402
                            start_run, write_json)

# Deviation D34 (test-time): soft tree scoring for object->part chains on PACO-LVIS val. Per chain and pool (capped
# nodes; nodes + the chain's own D30 SAM 3 instances) the step-1 (object) and step-2 (part) sigmas of every
# candidate, the containment tree of that pool and every candidate's IoU with the GT object and part are cached.
# Picks: flat = argmax sigma_part; hard tree = argmax sigma_part among the strict descendants of argmax sigma_obj
# (the object pick when it has none); soft tree = argmax over candidates p of
#   log sigma_part(p) + lambda * max_{a in anc(p) + {p}} log sigma_obj(a)
# (lambda = 0 is flat). lambda on a fixed grid by 2-fold image cross-validation of part gIoU (the all-val lambda is
# also reported); the object of a soft pick is its best-scoring ancestor.

LAMBDAS = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 32.0)
EPS = 1e-6


def num_shards(cfg: dict) -> int:
    return int(cfg["soft_tree"]["num_shards"])


def chain_rows(model, processor, extra, sel, layer, device, text, cfg, g, image, nodes, gts) -> list[dict]:
    k = cfg["s3_pool"]["instances_per_phrase"]
    plist = list(dict.fromkeys(p for t in g.text for p in S3P.phrases(t)))
    pool, owner = list(nodes), {}
    for p, lst in zip(plist, text(image, plist, k)):
        owner[p] = []
        for m, _ in lst:
            if m.any():
                owner[p].append(len(pool))
                pool.append(M.encode(m)[0])
    n = len(nodes)
    qs = [S2T.Query(q.qid, "chain", S2.user_text(q.text), S2.answer_text("chain", q.qid, q.text), [])
          for q in g.itertuples()]
    if not pool:
        return []
    z, probes = S2T.recompute(model, X2.ImageSource(processor, extra["sel_id"], image, pool), None, qs, layer,
                              device, cfg["s2_eval"]["batch_size"])
    e = sel.nodes(z.half().to(device))
    rows = []
    for q, pr in zip(g.itertuples(), probes):
        sig = torch.sigmoid(sel.scores(sel.queries(pr.half()), e)).float().cpu().numpy()
        gt = [gts[a] for a in json.loads(q.gt)]
        for mode, idx in (("nodes", np.arange(n)), ("aug", np.array(list(range(n)) + [i for p in S3P.phrases(q.text)
                                                                                       for i in owner.get(p, [])],
                                                                    np.int64))):
            if not len(idx):
                continue
            sub = [pool[i] for i in idx]
            ious = M.iou(sub, gt)
            tree = T.build_tree(sub, device)
            rows.append({"qid": q.qid, "image": q.image_key, "mode": mode, "n": int(len(idx)),
                         "sig_obj": sig[0, idx].astype(np.float32).tobytes(),
                         "sig_part": sig[1, idx].astype(np.float32).tobytes(),
                         "parent": np.asarray(tree.parent, np.int32).tobytes(),
                         "iou_obj": ious[:, 0].astype(np.float32).tobytes(),
                         "iou_part": ious[:, 1].astype(np.float32).tobytes()})
    return rows


def identity(cfg: dict, ck_run: Path, step: int) -> dict:
    h = hashlib.sha256()
    for rel in ("scripts/soft_tree_eval.py", "scripts/s3_eval.py", "scripts/s3_pool.py", "scripts/s2_train.py"):
        h.update((ROOT / rel).read_bytes())
    return {"kind": "soft_tree", "run": ck_run.name, "checkpoint": cfg["s2_eval"]["checkpoint"], "step": step,
            "pool": cfg["s3_pool"]["instances_per_phrase"], "num_shards": num_shards(cfg), "code": h.hexdigest()}


def cache_dir(cfg: dict, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "soft_tree" / S2T.s2_stage(cfg) / canonical_hash(ident)[:16]


def approvals() -> list[str]:
    return ["D10", "D21", "D30", "D34"]


def run_shard(cfg: dict, shard: int) -> None:
    require_approved(cfg, approvals())
    device = cfg["device"]
    _, manifest, mhash, wl = manifest_inputs(cfg, False)
    static = qwen_static(cfg)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    _, queries = open_queries(cfg, False, mhash, static)
    model, processor, extra, sel, meta, ck_run = S3E.load_model(cfg, static, device)
    d = cache_dir(cfg, ident := identity(cfg, ck_run, meta["step"]))
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"{S2T.s2_stage(cfg)}_softtree_shard", cfg, False, {"shard": shard, "cache": str(d)})
    text = S3P.TextInstances(cfg)
    files = dict(zip(wl.sha1, wl.path))
    val = queries[(queries.split == "val") & (queries.kind == "chain")]
    groups = sorted(val.sha1.unique())[shard::num_shards(cfg)]
    pimgs = paco.load(cfg["data"]["paco_lvis"], "val", {int(k.split(":")[1]) for k in val[val.sha1.isin(groups)].image_key})
    rows, t0, tau = [], time.time(), cfg["proposals"]["node_dedup"]
    for i, sha1 in enumerate(groups, 1):
        g = val[val.sha1 == sha1].sort_values("offset")
        image = load_rgb(assert_real_path(files[sha1]))
        nd = props.nodes(sha1, tau)
        gts = paco.gt_rles(pimgs[int(g.image_key.iloc[0].split(":")[1])])
        rows += chain_rows(model, processor, extra, sel, meta["layer"], device, text, cfg, g, image, nd["rles"], gts)
        if i % 25 == 0:
            print(f"[{i}/{len(groups)}] chains {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(groups), "rows": len(rows), "seconds": time.time() - t0})


def ancestors_max(parent: np.ndarray, logs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # for every node p: max of logs over p and its strict ancestors (root excluded), and the arg of that max
    n = len(logs)
    best, arg = logs.copy(), np.arange(n)
    order = np.argsort(_depth(parent, n), kind="stable")
    for v in order:
        u = parent[v]
        if 0 <= u < n and best[u] > best[v]:
            best[v], arg[v] = best[u], arg[u]
    return best, arg


def _depth(parent: np.ndarray, n: int) -> np.ndarray:
    d = np.zeros(n, np.int64)
    for v in range(n):
        u, k = parent[v], 0
        while 0 <= u < n:
            u, k = parent[u], k + 1
        d[v] = k
    return d


def picks(r) -> dict:
    so, sp = np.frombuffer(r.sig_obj, np.float32), np.frombuffer(r.sig_part, np.float32)
    io, ip = np.frombuffer(r.iou_obj, np.float32), np.frombuffer(r.iou_part, np.float32)
    parent = np.frombuffer(r.parent, np.int32).astype(np.int64)
    tree = T.from_parents(parent)
    a1 = int(np.argmax(so))
    desc = tree.strict_descendants(a1)
    a2 = int(desc[np.argmax(sp[desc])]) if len(desc) else a1
    lo, lp = np.log(np.clip(so, EPS, 1)), np.log(np.clip(sp, EPS, 1))
    amax, aarg = ancestors_max(parent, lo)
    out = {"flat_part": float(ip[int(np.argmax(sp))]), "hard_part": float(ip[a2]), "hard_obj": float(io[a1])}
    for lam in LAMBDAS:
        p = int(np.argmax(lp + lam * amax))
        out[f"soft{lam}_part"], out[f"soft{lam}_obj"] = float(ip[p]), float(io[aarg[p]])
    return out


def summarize(df: pd.DataFrame, bs: dict) -> dict:
    res = {}
    for mode, g in df.groupby("mode"):
        p = pd.DataFrame([{"image": r.image, **picks(r)} for r in g.itertuples()])
        images = np.array(sorted(p.image.unique()))
        fold = dict(zip(images, np.random.default_rng(0).permutation(len(images)) % 2))
        f = p.image.map(fold).to_numpy()
        cv, lams = np.zeros(len(p)), []
        for k in (0, 1):
            lam = max(LAMBDAS, key=lambda x: p[f"soft{x}_part"][f != k].mean())
            lams.append(lam)
            cv[f == k] = p[f"soft{lam}_part"][f == k]
        lam_all = max(LAMBDAS, key=lambda x: p[f"soft{x}_part"].mean())

        def ci(v):
            return mean_ci(np.asarray(v, float), p.image.to_numpy(), bs["n"], bs["seed"])

        res[mode] = {"chains": int(len(p)), "part_gIoU_flat": ci(p.flat_part), "part_gIoU_hard_tree": ci(p.hard_part),
                     "part_gIoU_soft_cv": ci(cv), "lambda_folds": lams, "lambda_all": lam_all,
                     "part_success_flat": ci(p.flat_part >= 0.5), "part_success_hard_tree": ci(p.hard_part >= 0.5),
                     "part_success_soft_cv": ci(cv >= 0.5),
                     "chain_success_soft_all": ci((p[f"soft{lam_all}_part"] >= 0.5) & (p[f"soft{lam_all}_obj"] >= 0.5)),
                     "chain_success_hard_tree": ci((p.hard_part >= 0.5) & (p.hard_obj >= 0.5)),
                     "curve": {str(x): float(p[f"soft{x}_part"].mean()) for x in LAMBDAS}}
    return res


def merge(cfg: dict) -> None:
    require_approved(cfg, approvals())
    ck_run = S3E.S2E.load_checkpoint(cfg, False)[1]
    step = torch.load(ck_run / cfg["s2_eval"]["checkpoint"], map_location="cpu")["meta"]["step"]
    d = cache_dir(cfg, identity(cfg, ck_run, step))
    n = num_shards(cfg)
    missing = [k for k in range(n) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} of {n} shards missing in {d}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(n)], ignore_index=True)
    run = start_run(f"{S2T.s2_stage(cfg)}_softtree", cfg, False, {"cache": str(d)})
    out = {"deviation": "D34", "label": "dev", "split": "PACO-LVIS val chains", "run": ck_run.name,
           "checkpoint_step": step, "results": summarize(df, cfg["v2"]["bootstrap"])}
    write_json(run / "metrics.json", out)
    print(json.dumps({m: {k: (v["mean"] if isinstance(v, dict) and "mean" in v else v) for k, v in r.items()
                          if k != "curve"} for m, r in out["results"].items()}, indent=1))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    if args.merge:
        merge(cfg)
    else:
        with torch.no_grad():
            run_shard(cfg, args.shard)


if __name__ == "__main__":
    main()
