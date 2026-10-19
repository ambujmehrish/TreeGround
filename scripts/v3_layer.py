import argparse
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_features, open_proposals, qwen_static  # noqa: E402
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.metrics import mean_ci, ranked_ap  # noqa: E402
from treeground.run import REPO, ProtocolError, load_config, start_run, write_json  # noqa: E402


def gallery_meta(cfg, smoke, manifest, mhash, wl):
    props = open_proposals(cfg, smoke, manifest, mhash)
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    rows = wl[wl.role == "paco_val"].drop_duplicates("image_key").sort_values("image_key")
    sha1s, img_idx, labels, parts, high = [], [], [], [], []
    for i, row in enumerate(rows.itertuples()):
        img = images[int(row.image_key.split(":")[1])]
        nodes = props.nodes(row.sha1, cfg["proposals"]["node_dedup"])
        if (nodes["h"], nodes["w"]) != (img.h, img.w):
            raise ProtocolError(f"{row.image_key}: annotation size differs from proposal masks")
        lab = paco_part_labels(nodes["rles"], img)
        sha1s.append(row.sha1)
        img_idx.append(np.full(len(nodes["rles"]), i))
        labels.append(lab.label)
        parts.append(lab.part)
        high.append(lab.high_overlap)
    labels = np.concatenate(labels)
    parts = np.concatenate(parts)
    vocab = {s: k for k, s in enumerate(sorted(set(labels) - {""}))}
    pvocab = {s: k for k, s in enumerate(sorted(set(parts) - {""}))}
    return {
        "sha1s": sha1s,
        "img": np.concatenate(img_idx),
        "label": np.array([vocab.get(s, -1) for s in labels]),
        "part": np.array([pvocab.get(s, -1) for s in parts]),
        "high": np.concatenate(high),
        "keys": rows["image_key"].tolist(),
    }


def load_matrix(store, sha1s: list[str], key: str, device: str) -> torch.Tensor:
    mats = [store.load(s, [key])[key] for s in sha1s]
    x = torch.cat(mats).to(device).float()
    return F.normalize(x, dim=-1).half()


@torch.no_grad()
def evaluate(x: torch.Tensor, meta: dict, batch: int, device: str) -> dict:
    img = torch.as_tensor(meta["img"], device=device)
    lab = torch.as_tensor(meta["label"], device=device)
    part = torch.as_tensor(meta["part"], device=device)
    high = torch.as_tensor(meta["high"], device=device)
    queries = torch.nonzero(lab >= 0).reshape(-1)
    aps, r1s, qimg = [], [], []
    no_pos = 0
    for start in range(0, len(queries), batch):
        q = queries[start:start + batch]
        s = (x[q] @ x.T).float()
        same_label = lab[None, :] == lab[q][:, None]
        other_obj = (part[None, :] == part[q][:, None]) & ~same_label & (lab[None, :] >= 0)
        valid = (img[None, :] != img[q][:, None]) & ~high[None, :] & ~other_obj
        positive = same_label & valid
        has = positive.any(1)
        no_pos += int((~has).sum())
        if has.any():
            ap, r1 = ranked_ap(s[has], positive[has], valid[has])
            aps.append(ap.cpu())
            r1s.append(r1.cpu())
            qimg.append(img[q][has].cpu())
    ap = torch.cat(aps).numpy() if aps else np.zeros(0)
    r1 = torch.cat(r1s).numpy() if r1s else np.zeros(0)
    groups = torch.cat(qimg).numpy() if qimg else np.zeros(0)
    return {"ap": ap, "r1": r1, "groups": groups, "queries_without_positives": no_pos}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    store = open_features(cfg, args.smoke, manifest, mhash, static)
    run = start_run("v3", cfg, args.smoke, {"manifest_run": str(mrun), "manifest_sha256": mhash,
                                             "features": str(store.path), "features_identity": store.identity_hash})
    device = cfg["device"]
    meta = gallery_meta(cfg, args.smoke, manifest, mhash, wl)
    layers = static["layers"]
    configs = []
    for ft in cfg["v3"]["feature_types"]:
        if ft in ("hidden", "value"):
            configs += [(f"{ft}_L{k}", f"{ft}_L{k}", ft, k) for k in layers]
        else:
            configs.append((ft, ft, ft, None))
    bs = cfg["v3"]["bootstrap"]
    results = {}
    for name, key, ft, layer in configs:
        x = load_matrix(store, meta["sha1s"], key, device)
        if x.shape[0] != len(meta["img"]):
            raise ProtocolError(f"{key}: {x.shape[0]} feature rows but {len(meta['img'])} nodes")
        r = evaluate(x, meta, cfg["v3"]["query_batch"], device)
        results[name] = {"feature_type": ft, "layer": layer, "mAP": mean_ci(r["ap"], r["groups"], bs["n"], bs["seed"]),
                         "R@1": mean_ci(r["r1"], r["groups"], bs["n"], bs["seed"]),
                         "queries_without_positives": r["queries_without_positives"]}
        del x
        if device.startswith("cuda"):
            torch.cuda.empty_cache()
    hidden = sorted(((results[f"hidden_L{k}"]["mAP"]["mean"], -k) for k in layers), reverse=True)
    chosen = -hidden[0][1]
    at_chosen = {n: r for n, r in results.items() if r["layer"] in (chosen, None)}
    metrics = {
        "split": "PACO-LVIS val", "label": "dev", "smoke": args.smoke,
        "images": len(meta["sha1s"]), "nodes": int(len(meta["img"])), "labelled_nodes": int((meta["label"] >= 0).sum()),
        "results": results, "chosen_layer": chosen, "rule": "I3: highest deterministic hidden-state mAP, ties -> smaller l",
        "feature_type_comparison_at_chosen_layer": at_chosen,
        "note": "feature-type comparison is descriptive; the representation stays decoder hidden states (spec 1)",
    }
    write_json(run / "metrics.json", metrics)
    chosen_doc = {"layer": chosen, "v3_run": run.name, "smoke": args.smoke,
                  "hidden_mAP": results[f"hidden_L{chosen}"]["mAP"]["mean"]}
    (run / "chosen_layer.yaml").write_text(yaml.safe_dump(chosen_doc))
    if not args.smoke:
        (REPO / "configs" / "chosen_layer.yaml").write_text(yaml.safe_dump(chosen_doc))
    print(run, "chosen layer", chosen)


if __name__ == "__main__":
    main()
