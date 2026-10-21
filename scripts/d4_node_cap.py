import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground.cache import keep_column  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals  # noqa: E402
from treeground.metrics import area_bin  # noqa: E402
from treeground.run import REPO, ProtocolError, load_config, require_approved, start_run, write_json  # noqa: E402

# Deviation D4: choose the label-free node ranking on PACO *train* coverage (val is only reported), keep K nodes.
# Rule fixed before seeing results: the ranking with the higher train part recall at K wins; ties -> stability.
THR = 0.5
ORDERS = ("stability", "sam_iou")


def best_ious(df: pd.DataFrame, order: str | None, k: int, gts: list) -> np.ndarray:
    if order and len(df) > k:
        ranked = df.assign(_s=-df[order].astype(np.float64)).sort_values(["_s", "src_index"], kind="stable")
        df = df.loc[ranked.index[:k]]
    h, w = gts[0]["size"]
    rles = [{"size": [h, w], "counts": c} for c in df["rle"]]
    return M.iou(rles, gts).max(0) if rles else np.zeros(len(gts))


def coverage(cfg, props, wl, role, split, k, n_images, tau):
    rows = wl[wl.role == role].drop_duplicates("image_key").sort_values("image_key")
    if n_images:
        rows = rows.iloc[:n_images]
    images = paco.load(cfg["data"]["paco_lvis"], split, {int(x.split(":")[1]) for x in rows.image_key})
    recs = []
    for r in rows.itertuples():
        img = images[int(r.image_key.split(":")[1])]
        if not img.parts:
            continue
        df = props.rows(r.sha1)
        if len(df):
            df = df[df[keep_column(tau)].astype(bool)]
        gts = [p.rle for p in img.parts]
        areas = M.areas(gts)
        res = {name: best_ious(df, o, k, gts) for name, o in [("uncapped", None)] + [(o, o) for o in ORDERS]}
        for j in range(len(gts)):
            recs.append({"area_bin": area_bin(areas[j]), "n_nodes": len(df),
                         **{f"hit_{n}": bool(v[j] >= THR) for n, v in res.items()}})
    df = pd.DataFrame(recs)
    out = {"parts": len(df), "nodes_median": float(df.n_nodes.median()) if len(df) else 0.0}
    for n in ("uncapped",) + ORDERS:
        out[n] = {"part_recall": float(df[f"hit_{n}"].mean()),
                  "by_area_bin": {b: float(g[f"hit_{n}"].mean()) for b, g in df.groupby("area_bin")}}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--k", type=int, default=256)
    ap.add_argument("--train-images", type=int, default=2000)
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D4"])
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    props = open_proposals(cfg, False, manifest, mhash)
    run = start_run("d4_node_cap", cfg, False, {"manifest_run": str(mrun), "proposals": str(props.path),
                                                 "k": args.k, "train_images": args.train_images})
    tau = cfg["proposals"]["node_dedup"]
    train = coverage(cfg, props, wl, "paco_train", "train", args.k, args.train_images, tau)
    print("train", {n: round(train[n]["part_recall"], 3) for n in ("uncapped",) + ORDERS}, flush=True)
    order = max(ORDERS, key=lambda o: (train[o]["part_recall"], o == "stability"))
    val = coverage(cfg, props, wl, "paco_val", "val", args.k, None, tau)
    print("val", {n: round(val[n]["part_recall"], 3) for n in ("uncapped",) + ORDERS}, flush=True)
    if not train["parts"]:
        raise ProtocolError("no PACO train parts found for the D4 choice")
    doc = {"k": args.k, "order": order, "tau_d": tau, "d4_run": run.name, "smoke": False,
           "rule": "ranking with the higher PACO train part recall at k (ties -> stability)"}
    write_json(run / "metrics.json", {"label": "dev", "k": args.k, "chosen_order": order, "tau_d": tau,
                                       "paco_train_subset": train, "paco_val": val})
    (run / "node_cap.yaml").write_text(yaml.safe_dump(doc))
    (REPO / "configs" / "node_cap.yaml").write_text(yaml.safe_dump(doc))
    print(run, "order", order, "k", args.k)


if __name__ == "__main__":
    main()
