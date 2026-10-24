import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

from treeground import tree as T

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
spec = importlib.util.spec_from_file_location("x_transfer_clean", ROOT / "scripts" / "x_transfer_clean.py")
XC = importlib.util.module_from_spec(spec)
spec.loader.exec_module(XC)
V4, G = XC.V4, XC.G
BS = {"n": 50, "seed": 0}


def train_rows():
    return pd.DataFrame({"coco_id": [11, 12, np.nan], "sha1": ["a", "b", "c"],
                         "tag": ["grefcoco/train", "paco_lvis/train", "pascal_part/train"]})


def test_overlap_by_coco_id_or_pixel_sha1():
    ov = XC.overlap(["lvis_val:11", "lvis_val:20", "lvis_val:30", "ade20k:x"],
                    {"lvis_val:11": "z", "lvis_val:20": "c", "lvis_val:30": "y", "ade20k:x": "q"}, train_rows())
    got = ov.set_index("image")
    cols = ["coco_id_match", "sha1_match", "sources"]
    assert got.loc["lvis_val:11", cols].tolist() == [True, False, "grefcoco/train"]
    assert got.loc["lvis_val:20", cols].tolist() == [False, True, "pascal_part/train"]
    assert not got.loc["lvis_val:30"].overlap and not got.loc["ade20k:x"].overlap
    s = XC.overlap_summary(ov)
    assert (s["eval_images"], s["overlap"], s["overlap_coco_id"], s["overlap_sha1"]) == (4, 2, 1, 1)
    assert s["by_training_source"] == {"grefcoco/train": 1, "pascal_part/train": 1}


def items():
    rng = np.random.default_rng(0)
    parents = np.array([-1, 0, 0, 1, -1])
    tree = T.from_parents(parents)
    out = []
    for i in range(12):
        ious = rng.uniform(0, 1, (2, 5))
        out.append({"qid": f"img{i % 4}|chain|o:p{i}", "image": f"lvis_val:{i % 4}", "tree": tree,
                    "rand": [T.from_parents(parents)] * 2, "box": tree, "ious": ious,
                    "sig": rng.uniform(0, 1, (2, 5)), "part_ratio": 0.2 if i % 3 else 0.7})
    return out


def published_run(tmp_path):
    its = items()
    cov = pd.DataFrame([{"image": it["image"], "kind": "chain", "label": "p", "area_bin": "small",
                         "best_iou": it["ious"][1].max(), "object_iou": it["ious"][0].max(),
                         "feasible_uncapped": True, "part_ratio": it["part_ratio"]} for it in its]
                       + [{"image": "lvis_val:0", "kind": "part", "label": "p", "area_bin": "large", "best_iou": 0.7}])
    cov.to_csv(tmp_path / "coverage.csv", index=False)
    m = {"label": "transfer:lvis_hier", "images": 5, "coverage_uncapped": XC.coverage_block(cov)}
    for mode, box in XC.MODES[:3]:
        df = V4.natural(its, 0.4, inclusive=mode == "d5", box=box)[0]
        df.to_csv(tmp_path / f"chains_{mode}.csv", index=False)
        m[f"natural_{mode}"] = XC.natural_block(df, 2, BS, box)
    m["natural_no_rejection"] = "not reported: deviation D12 is not approved in deviations.jsonl"
    (tmp_path / "metrics.json").write_text(json.dumps(m, default=float))
    return m


def test_clean_blocks_reproduce_published_and_drop_images(tmp_path):
    m = published_run(tmp_path)
    out, extra = XC.x_transfer_block(tmp_path, set(), 2, BS)
    assert extra["reproduces_published"] and out["images"] == 5 and out["images_dropped"] == 0
    assert XC.same(out["natural_strict"], m["natural_strict"]) and out["natural_no_rejection"].startswith("not")
    out, extra = XC.x_transfer_block(tmp_path, {"lvis_val:0"}, 2, BS)
    assert out["images_dropped"] == 1 and out["images"] == 4
    assert out["natural_strict"]["all"]["chains"] == 9 and set(extra["frames"]["strict"].image) == {
        f"lvis_val:{i}" for i in (1, 2, 3)}
    assert out["coverage_uncapped"]["parts"] == 0 and out["coverage_uncapped"]["chains"] == 9
    df = pd.read_csv(tmp_path / "chains_d6.csv")
    assert XC.same(out["natural_d6"]["all"], V4.natural_summary(df[df.image != "lvis_val:0"], 0, BS, box=True))


def test_giou_block_reproduces_and_filters(tmp_path):
    rows = [{"image": f"lvis_val:{i % 3}", "qid": f"q{i}", "object": f"o{i % 3}", "part": "p",
             "single_level": i % 4 == 0, **{f"{k}_{m}": (i * 0.07 + j * 0.1) % 1 for j, k in enumerate(
                 ("object", "part_flat", "part_tree")) for m in ("norej", "calib")}} for i in range(10)]
    df = pd.DataFrame(rows)
    df.to_csv(tmp_path / "chains.csv", index=False)
    pub = {"selector": "s1", "all": G.summarize(df, BS), "two_level": G.summarize(df[~df.single_level], BS),
           "single_level": G.summarize(df[df.single_level], BS)}
    (tmp_path / "metrics.json").write_text(json.dumps(pub))
    out, ok = XC.giou_block(tmp_path, set(), BS)
    assert ok and XC.same(out["all"], pub["all"])
    out, _ = XC.giou_block(tmp_path, {"lvis_val:1"}, BS)
    assert out["all"]["images"] == 2 and out["all"]["chains"] == 7


def test_baseline_block_reproduces_and_uses_clean_common_subset(tmp_path):
    BT = XC.BT
    q = []
    for i in range(3):
        key = f"lvis_val:{i}"
        q.append({"qid": f"{key}|object|car", "kind": "object", "image": key, "object_qid": f"{key}|object|car",
                  "chain_qid": None, "part_ratio": np.nan, "iou": 0.6, "inter": 6, "union": 10, "pred_area": 8,
                  "iou_any": 0.6})
        q.append({"qid": f"{key}|part|car:wheel", "kind": "part", "image": key, "object_qid": f"{key}|object|car",
                  "chain_qid": f"{key}|chain|car:wheel", "part_ratio": 0.1, "iou": 0.3 * i, "inter": i,
                  "union": 5, "pred_area": 3, "iou_any": 0.3 * i})
    df = pd.DataFrame(q)
    ch = BT.chain_table(df)
    df.to_csv(tmp_path / "queries.csv", index=False)
    ch.to_csv(tmp_path / "chains.csv", index=False)
    (tmp_path / "metrics.json").write_text(json.dumps({"method": "sa2va", **BT.summarize(df, ch, BS)}, default=float))
    strict = pd.DataFrame([{"qid": f"lvis_val:{i}|chain|car:wheel", "image": f"lvis_val:{i}", "structure": s,
                            "feasible": not (i == 2 and s == "rand0")} for i in (0, 2) for s in ("flat", "rand0")])
    out, ok = XC.baseline_block(tmp_path, {"lvis_val:1"}, strict, BS, tmp_path)
    assert ok and out["images"] == 2 and out["chains"]["all"]["chains"] == 2
    c = out["common_eligible_subset"]
    assert (c["size"], c["chains_match_x_transfer"], c["chains"]["all"]["chains"]) == (1, True, 1)


def test_transfer_table_prefers_clean_lvis_blocks(tmp_path):
    spec_mt = importlib.util.spec_from_file_location("make_tables", ROOT / "scripts" / "make_tables.py")
    MT = importlib.util.module_from_spec(spec_mt)
    spec_mt.loader.exec_module(MT)
    m = tmp_path / "metrics"
    m.mkdir()

    def blk(chains):
        return {"coverage_uncapped": {"part_recall": 0.5}, "natural_strict": {"all": {"chains": chains}}}
    (m / "x_transfer_lvis_hier_merge.json").write_text(json.dumps(blk(100)))
    (m / "x_transfer_lvis_hier_s1_merge.json").write_text(json.dumps(blk(100)))
    (m / "x_transfer_ade20k_merge.json").write_text(json.dumps(blk(7)))
    (m / "x_transfer_clean_lvis_hier.json").write_text(json.dumps({"clean": {"x_transfer_lvis_hier_merge": blk(80)}}))
    rows = MT.transfer(m)[1]
    lvis = {r["selector"]: r for r in rows if r["dataset"] == "LVIS hier." and r["search"] == "spec"}
    assert (lvis["V2"]["chains"], lvis["V2"]["clean"]) == (80, "yes")
    assert (lvis["S1"]["chains"], lvis["S1"]["clean"]) == (100, "no")
    assert next(r for r in rows if r["dataset"] == "ADE20K")["clean"] is None


def test_main_finds_overlap_and_writes_clean_blocks(tmp_path, monkeypatch):
    from PIL import Image

    import treeground.run as R
    runs, data = tmp_path / "runs", tmp_path / "data"
    run = runs / "x_transfer_lvis_hier_merge" / "r1"
    run.mkdir(parents=True)
    published_run(run)
    (run / "meta.json").write_text(json.dumps({"smoke": False}))
    (data / "coco" / "val2017").mkdir(parents=True)
    for i in range(4):
        Image.fromarray(np.full((4, 4, 3), 10 * i, np.uint8)).save(data / "coco" / "val2017" / f"{i}.png")
    lvis = {"categories": [], "annotations": [],
            "images": [{"id": i, "coco_url": f"http://x/val2017/{i}.png", "height": 4, "width": 4} for i in range(5)]}
    (data / "lvis.json").write_text(json.dumps(lvis))
    voc = data / "voc"
    (voc / "ImageSets" / "Main").mkdir(parents=True)
    (voc / "JPEGImages").mkdir()
    (voc / "ImageSets" / "Main" / "train.txt").write_text("2008_1\n2008_2\n")
    # a lossless copy of lvis_val:2 under a Pascal train id: matched by pixel sha1 only
    Image.open(data / "coco" / "val2017" / "2.png").save(voc / "JPEGImages" / "2008_1.jpg", format="PNG")
    (data / "ann").mkdir()
    (data / "ann" / "2008_1.mat").write_text("")
    manifest = pd.DataFrame({"image_key": ["grefcoco:0", "paco_lvis:9"], "source": ["grefcoco", "paco_lvis"],
                             "split": ["train", "test"], "split_class": ["train", "eval"], "coco_id": [0, 3],
                             "sha1": ["s0", "s3"]})
    cfg = {"runs_root": str(runs), "v4": {"bootstrap": BS, "seeds": [0, 1]},
           "data": {"lvis_val_json": str(data / "lvis.json"), "coco_images": str(data / "coco"),
                    "pascal_voc": str(voc), "pascal_part_ann": str(data / "ann")}, "transfer": {}}
    monkeypatch.setattr(XC, "load_config", lambda p: cfg)
    monkeypatch.setattr(XC, "require_approved", lambda c, ids: None)
    monkeypatch.setattr(XC, "manifest_inputs", lambda c, s: (tmp_path, manifest, "h", None))
    monkeypatch.setattr(R, "git_state", lambda: {"rev": "test", "dirty": False})
    monkeypatch.setattr(sys, "argv", ["x", "--config", "c", "--workers", "1"])
    XC.main()
    out = json.loads(next((runs / "x_transfer_clean_lvis_hier").glob("*/metrics.json")).read_text())
    ov = out["overlap"]["lvis_hier"]
    assert (ov["eval_images"], ov["overlap"], ov["overlap_coco_id"], ov["overlap_sha1"]) == (4, 2, 1, 1)
    assert ov["by_training_source"] == {"grefcoco/train": 1, "pascal_part/train": 1}
    assert out["training_images"]["by_source"] == {"grefcoco/train": 1, "pascal_part/train": 1}
    assert out["reproduces_published"] == {"x_transfer_lvis_hier_merge": True}
    c = out["clean"]["x_transfer_lvis_hier_merge"]
    assert c["images_dropped"] == 2 and c["natural_strict"]["all"]["chains"] == 6
    assert out["overlap"]["ade20k"].startswith("not available")
