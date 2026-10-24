import argparse
import hashlib
import importlib.util
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, canonical_hash, load_config, read_json, require_approved,  # noqa: E402
                            start_run, write_json)

# Deviation D22: per-query IoU of our picks on the D7/D8 transfer sets, scored like the D16 baselines. The transfer
# pipeline of x_transfer.py (V2/S1 selector) or x_transfer_s2.py (S2) is reused unchanged per image; for each chain
# the object pick (argmax over all capped nodes), the part pick over all nodes (flat) and the part pick inside the
# object pick's subtree (true tree, V4 natural search) are recorded with their IoU (empty node set = 0), once without
# rejection and once at the stage's tau_rej (rejected = empty = 0).


def _load(name: str):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def is_s2(cfg: dict) -> bool:
    return "s2_eval" in cfg


def label(cfg: dict) -> str:
    stage = cfg.get("inputs", {}).get("selector_stage", "v2")
    if is_s2(cfg):
        return "s2" if stage == "v2" else f"s2_from_{stage}"
    return stage


def picks(item: dict, tau: float, decide) -> dict:
    ious, sig, tree = item["ious"], item["sig"], item["tree"]
    n = ious.shape[1]
    out = {}
    for mode, t in (("norej", -np.inf), ("calib", tau)):
        if n == 0:
            out.update({f"object_{mode}": 0.0, f"part_flat_{mode}": 0.0, f"part_tree_{mode}": 0.0})
            continue
        a1 = decide(sig[0], np.arange(n), t)
        af = decide(sig[1], np.arange(n), t)
        u1 = tree.root if a1 is None else a1
        a2 = decide(sig[1], tree.strict_descendants(u1), t)
        out[f"object_{mode}"] = 0.0 if a1 is None else float(ious[0, a1])
        out[f"part_flat_{mode}"] = 0.0 if af is None else float(ious[1, af])
        out[f"part_tree_{mode}"] = 0.0 if a2 is None else float(ious[1, a2])
    return out


def rows_for_image(key: str, chains: list, items: list, tau: float, decide) -> list[dict]:
    rows = []
    for (qid, obj, part, _, _), it in zip(chains, items):
        if it["qid"] != qid:
            raise ProtocolError(f"chain order mismatch: {qid} vs {it['qid']}")
        rows.append({"image": key, "qid": qid, "object": obj, "part": part,
                     "single_level": it["part_ratio"] >= 0.5, **picks(it, tau, decide)})
    return rows


def summarize(df: pd.DataFrame, bs: dict) -> dict:
    def ci(frame, col):
        return mean_ci(frame[col].to_numpy(float), frame.image.to_numpy(), bs["n"], bs["seed"])

    objects = df.drop_duplicates(["image", "object"])
    out = {"chains": int(len(df)), "objects": int(len(objects)), "images": int(df.image.nunique())}
    for mode in ("norej", "calib"):
        out[mode] = {"object_gIoU": ci(objects, f"object_{mode}"),
                     "part_gIoU_flat": ci(df, f"part_flat_{mode}"),
                     "part_gIoU_tree": ci(df, f"part_tree_{mode}"),
                     "part_success_flat": ci(df.assign(s=df[f"part_flat_{mode}"] >= 0.5), "s"),
                     "part_success_tree": ci(df.assign(s=df[f"part_tree_{mode}"] >= 0.5), "s")}
    return out


def setup(cfg: dict, dataset: str):
    X = _load("x_transfer")
    V4 = X.load_v4()
    code = hashlib.sha256((ROOT / "scripts" / "x_giou.py").read_bytes()).hexdigest()
    if is_s2(cfg):
        XS2 = _load("x_transfer_s2")
        stage, s2_run, ev, tau = XS2.s2_inputs(cfg)
        base = XS2.identity(cfg, dataset, stage, s2_run, ev)
        return X, V4, XS2, {"kind": "x_giou", "base": base, "code": code}, tau, (s2_run, ev)
    v2 = X.selector_run(cfg)
    return X, V4, None, {"kind": "x_giou", "base": X.identity(cfg, dataset, v2), "code": code}, None, v2


def out_dir(cfg: dict, dataset: str, ident: dict) -> Path:
    return Path(cfg["cache_root"]) / "full" / "x_giou" / label(cfg) / dataset / canonical_hash(ident)[:16]


def approvals(cfg: dict, dataset: str) -> list[str]:
    need = ["D3", "D4", "D7", "D22"] + (["D8"] if dataset in ("lvis_hier", "openimages_hier") else [])
    need += ["D9"] if dataset == "openimages_hier" else []
    stage = cfg.get("inputs", {}).get("selector_stage", "v2")
    need += ["D11"] if stage == "s1" else []
    need += ["D10", "D17"] if is_s2(cfg) else []
    return need


def run_shard(cfg: dict, dataset: str, shard: int, num_shards: int) -> None:
    require_approved(cfg, approvals(cfg, dataset))
    X, V4, XS2, ident, tau, ref = setup(cfg, dataset)
    d = out_dir(cfg, dataset, ident)
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    run = start_run(f"x_giou_{label(cfg)}_{dataset}", cfg, False, {"shard": shard, "num_shards": num_shards,
                                                                    "cache": str(d)})
    if XS2 is None:
        models, tau, _ = X.load_models(cfg, ref)
        process = lambda key, image, hw, chains, parts, cap: X.process_image(  # noqa: E731
            models, cfg, key, image, hw, chains, parts, cap, V4)
    else:
        S2E = sys.modules["s2_eval"]
        ckpt, ck_run = S2E.load_checkpoint(cfg, False)
        if ck_run != ref[0] or read_json(ref[1] / "metrics.json")["checkpoint_step"] != ckpt["meta"]["step"]:
            raise ProtocolError(f"checkpoint {ck_run.name} is not the one {ref[1].name} evaluated")
        models = XS2.load_models(cfg, ckpt)
        del ckpt
        process = lambda key, image, hw, chains, parts, cap: XS2.process_image(  # noqa: E731
            models, cfg, key, image, hw, chains, parts, cap)
    cap = X.node_cap()
    rows, t0, n_img = [], time.time(), 0
    for key, image, hw, chains, parts in X.records(cfg, dataset, shard, num_shards):
        _, items, _, _ = process(key, image, hw, chains, parts, cap)
        rows += rows_for_image(key, chains, items, tau, V4.decide) if chains else []
        n_img += 1
        if n_img % 25 == 0:
            print(f"[{n_img}] chains {len(rows)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": n_img, "chains": len(rows), "seconds": time.time() - t0})


def merge(cfg: dict, dataset: str, num_shards: int) -> None:
    require_approved(cfg, ["D22"])
    _, _, _, ident, _, _ = setup(cfg, dataset)
    d = out_dir(cfg, dataset, ident)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} shards missing for {dataset}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    run = start_run(f"x_giou_{label(cfg)}_{dataset}_merge", cfg, False, {"cache": str(d)})
    bs = cfg["v4"]["bootstrap"]
    out = {"label": f"transfer:{dataset}", "deviation": "D22", "selector": label(cfg), "all": summarize(df, bs),
           "two_level": summarize(df[~df.single_level], bs) if (~df.single_level).any() else None,
           "single_level": summarize(df[df.single_level], bs) if df.single_level.any() else None}
    df.to_csv(run / "chains.csv", index=False)
    write_json(run / "metrics.json", out)
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--shard", type=int)
    ap.add_argument("--merge", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    num_shards = cfg["transfer"]["num_shards"]
    if args.merge:
        merge(cfg, args.dataset, num_shards)
    else:
        if args.shard is None or not 0 <= args.shard < num_shards:
            raise ProtocolError(f"--shard must be in 0..{num_shards - 1}")
        run_shard(cfg, args.dataset, args.shard, num_shards)


if __name__ == "__main__":
    main()
