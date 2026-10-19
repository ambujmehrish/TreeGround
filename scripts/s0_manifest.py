import argparse
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.data import grefcoco, manifest as MF  # noqa: E402
from treeground.run import ProtocolError, load_config, start_run, write_json  # noqa: E402


def collect(cfg: dict, smoke: bool) -> tuple[pd.DataFrame, dict, list[grefcoco.GrefQuery]]:
    wl = cfg["worklist"]["smoke"] if smoke else cfg["worklist"]
    seed = cfg["seed"]
    status = {}
    rows = []
    gref_queries: list[grefcoco.GrefQuery] = []
    for source in cfg["manifest"]["sources"]:
        ok = MF.source_available(cfg, source)
        status[source] = "present" if ok else "missing"
        if not ok:
            if source in cfg["manifest"]["required"]:
                raise ProtocolError(f"required dataset {source} is missing under {cfg['data']}")
            continue
        if source == "grefcoco":
            refs, images = grefcoco.load(cfg["data"]["grefcoco"])
            entries = MF.grefcoco_entries(cfg)
            if smoke:
                tr = grefcoco.no_target_queries(refs, images, "train")
                va = grefcoco.no_target_queries(refs, images, "val")
                tr = [tr[i] for i in MF.seeded_choice(list(range(len(tr))), wl["grefcoco_train_no_target"], seed)]
                va = [va[i] for i in MF.seeded_choice(list(range(len(va))), wl["grefcoco_val_no_target"], seed)]
                keep = {(q.image_id, q.split) for q in tr + va}
                entries = [e for e in entries if (e["coco_id"], e["split"]) in keep]
                gref_queries = tr + va
            else:
                gref_queries = grefcoco.no_target_queries(refs, images, "train") + \
                    grefcoco.no_target_queries(refs, images, "val")
        else:
            entries = MF.ENTRY_BUILDERS[source](cfg)
            if smoke:
                entries = smoke_subset(entries, source, wl, seed)
        rows += entries
    df = pd.DataFrame(rows)
    df["split_class"] = [MF.SPLIT_CLASS[(s, sp)] for s, sp in zip(df["source"], df["split"])]
    return df, status, gref_queries


def smoke_subset(entries: list[dict], source: str, wl: dict, seed: int) -> list[dict]:
    limits = {("paco_lvis", "train"): wl["paco_train_images"], ("paco_lvis", "val"): wl["paco_val_images"],
              ("paco_lvis", "test"): 0}
    out = []
    for split in sorted({e["split"] for e in entries}):
        sub = sorted((e for e in entries if e["split"] == split), key=lambda e: e["image_key"])
        k = limits.get((source, split), 5)
        out += [sub[i] for i in MF.seeded_choice(list(range(len(sub))), k, seed)] if k else []
    return out


def worklist(df: pd.DataFrame, gref: list[grefcoco.GrefQuery], cfg: dict, smoke: bool
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    wl = cfg["worklist"]["smoke"] if smoke else cfg["worklist"]
    seed = cfg["seed"]
    by_key = df.set_index("image_key")
    roles = []

    def add(keys, role):
        roles.extend({"image_key": k, "role": role} for k in keys)

    paco_train = sorted(df[(df.source == "paco_lvis") & (df.split == "train")].image_key)
    add(MF.seeded_choice(paco_train, wl["paco_train_images"], seed), "paco_train")
    add(sorted(df[(df.source == "paco_lvis") & (df.split == "val")].image_key), "paco_val")
    add(sorted(df[df.source == "partonomy"].image_key), "partonomy")
    add(sorted(df[(df.source == "sceneparser") & (df.split == "val")].image_key), "sceneparser_val")

    present = set(zip(df.image_key, df.split))
    tr = [q for q in gref if q.split == "train" and (q.image_key, "train") in present]
    va = [q for q in gref if q.split == "val" and (q.image_key, "val") in present]
    tr = [tr[i] for i in MF.seeded_choice(list(range(len(tr))), wl["grefcoco_train_no_target"], seed)]
    queries = pd.DataFrame([{"qid": q.qid, "sent_id": q.sent_id, "ref_id": q.ref_id, "image_key": q.image_key,
                             "sha1": by_key.loc[q.image_key, "sha1"] if q.image_key in by_key.index else None,
                             "text": q.text, "split": q.split, "provenance": f"grefcoco/{q.split}/no_target"}
                            for q in tr + va])
    add(sorted({q.image_key for q in tr}), "grefcoco_train_nt")
    add(sorted({q.image_key for q in va}), "grefcoco_val_nt")
    wl_df = pd.DataFrame(roles).merge(df.drop_duplicates("image_key")[["image_key", "sha1", "path", "w", "h"]],
                                      on="image_key", how="left")
    return wl_df, queries


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    run = start_run("s0_manifest", cfg, args.smoke)
    df, status, gref = collect(cfg, args.smoke)
    hashes = MF.hash_images(df["path"].tolist(), cfg["manifest"]["workers"])
    df["sha1"] = [h[0] for h in hashes]
    df["w"] = [h[1] for h in hashes]
    df["h"] = [h[2] for h in hashes]
    df["coco_id"] = pd.array(df["coco_id"], dtype="Int64")
    before = df.groupby(["source", "split"]).size()
    kept, removed = MF.apply_precedence(df)
    MF.check_disjoint(kept)
    kept = kept[MF.COLUMNS]
    kept.to_parquet(run / "manifest.parquet", index=False)
    mhash = MF.manifest_hash(kept)
    (run / "manifest.sha256").write_text(mhash + "\n")
    removed.to_csv(run / "removed.csv", index=False)
    wl, queries = worklist(kept, gref, cfg, args.smoke)
    wl.to_parquet(run / "worklist.parquet", index=False)
    queries.to_parquet(run / "grefcoco_queries.parquet", index=False)
    after = kept.groupby(["source", "split"]).size()
    write_json(run / "metrics.json", {
        "smoke": args.smoke,
        "manifest_sha256": mhash,
        "sources": status,
        "images_before": {f"{a}/{b}": int(v) for (a, b), v in before.items()},
        "images_after": {f"{a}/{b}": int(v) for (a, b), v in after.items()},
        "removed": {f"{a}/{b}->{c}": int(v) for (a, b, c), v in
                    removed.groupby(["source", "split", "kept_class"]).size().items()} if len(removed) else {},
        "unique_images": int(kept["sha1"].nunique()),
        "worklist_roles": wl.groupby("role").size().to_dict(),
        "grefcoco_no_target_queries": queries.groupby("split").size().to_dict() if len(queries) else {},
    })
    print(run)


if __name__ == "__main__":
    main()
