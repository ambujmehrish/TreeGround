import argparse
import importlib.util
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import cocoedit_metrics as CM  # noqa: E402
from treeground.data import cocoedit_bench as CB  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (REPO, ProtocolError, load_config, read_json, require_approved, resolve_run,  # noqa: E402
                            start_run, write_json)

# D25 evaluation of one (editor, dataset) edit merge: CoCoEdit's released PSNR / SSIM exactly (PSNR on the white
# mask region, SSIM full-image because skimage ignores the mask argument), plus masked SSIM, PSNR clipped at 50 dB,
# pooled-MSE PSNR, the share of samples at the 100 dB cap and LPIPS-alex of the edited image with the GT edit region
# replaced by source pixels; the judge merge gives the MLLM Overall (mean over the MLLM categories). Every run also
# rewrites the combined table <tg_root>/results/cocoedit/<dataset>.csv with CoCoEdit's published rows.

EDITORS = ("qwen_image_edit", "flux_fill")
PIXEL = ("psnr", "psnr_capped", "ssim", "ssim_masked", "lpips")


def load_script(name: str):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def judge_overall(cfg: dict, editor: str, dataset: str, smoke: bool, merge_run: Path) -> tuple[dict | None, str]:
    stage = f"cocoedit_judge_{editor}_{dataset}"
    try:
        run = resolve_run(cfg, stage, cfg["inputs"].get("cocoedit_judge", "latest"), smoke)
    except ProtocolError as e:
        return None, f"no judge merge: {e}"
    m = read_json(run / "metrics.json")
    if m["merge_run"] != merge_run.name:
        return None, f"latest {stage} run {run.name} judged edit merge {m['merge_run']}, not {merge_run.name}"
    return m, run.name


def pixel_block(g: pd.DataFrame, bs: dict) -> dict:
    out = {"n": int(len(g)), "at_cap_rate": float(g.at_cap.mean()) if len(g) else float("nan"),
           "psnr_pooled": CM.pooled_psnr(g.sq_err.to_numpy(), g.sq_count.to_numpy())}
    for k in PIXEL:
        v = g[k].to_numpy(np.float64)
        ok = ~np.isnan(v)
        out[k] = mean_ci(v[ok], g.image.to_numpy()[ok], bs["n"], bs["seed"]) if ok.any() else None
    return out


def evaluate(cfg: dict, editor: str, dataset: str, smoke: bool) -> None:
    require_approved(cfg, ["D25"])
    os.environ["TORCH_HOME"] = cfg["torch_home"]
    C = load_script("e_cocoedit")
    merge_run = resolve_run(cfg, f"cocoedit_{editor}_{dataset}_merge", cfg["inputs"].get("cocoedit_merge", "latest"),
                            smoke)
    mm = read_json(merge_run / "metrics.json")
    items = [s for s in C.evaluated(C.samples(cfg, dataset), smoke, cfg["cocoedit"]["smoke_per_category"]) if s.scored]
    if not items:
        raise ProtocolError("no PSNR/SSIM samples")
    lp = CM.Lpips(cfg["cocoedit"]["eval"]["lpips_net"], cfg["device"])
    rows, t0 = [], time.time()
    for n, s in enumerate(items, 1):
        image, edit_region = s.load()
        src, keep = CM.read_float(image), ~edit_region
        for c, d in mm["images_dirs"].items():
            out = CM.read_float(C.load_png(Path(d) / f"{C.safe_name(s.sample_id)}.png"))
            rows.append({"sample_id": s.sample_id, "image": s.image_key, "category": s.category, "condition": c,
                         **CM.sample_metrics(out, src, keep, lp)})
        if n % 25 == 0:
            print(f"[eval {n}/{len(items)}] {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    judge, judge_note = judge_overall(cfg, editor, dataset, smoke, merge_run)
    bs = cfg["cocoedit"]["eval"]["bootstrap"]
    metrics, table, cats = {}, [], []
    for c in mm["images_dirs"]:
        g = df[df.condition == c]
        jb = judge["conditions"][c] if judge else None
        metrics[c] = {"all": pixel_block(g, bs), "by_category": {k: pixel_block(h, bs) for k, h in g.groupby("category")},
                      "judge": jb}
        a = metrics[c]["all"]
        table.append({"editor": editor, "dataset": dataset, "condition": c,
                      "overall": jb["overall"]["overall"] if jb else None,
                      **{k: a[k]["mean"] if a[k] else None for k in PIXEL}, "psnr_pooled": a["psnr_pooled"],
                      "at_cap_rate": a["at_cap_rate"], "n_scored": a["n"],
                      **({"sc": jb["sc"]["overall"], "pq": jb["pq"]["overall"]} if jb and "sc" in jb else {})})
        for k, b in metrics[c]["by_category"].items():
            cats.append({"condition": c, "category": k, "n": b["n"], **{m: b[m]["mean"] if b[m] else None for m in PIXEL},
                         "at_cap_rate": b["at_cap_rate"],
                         "judge": jb["overall"]["by_category"].get(k) if jb else None})
        if jb:
            for k in CB.MLLM_CATEGORIES[dataset]:
                if k not in metrics[c]["by_category"]:
                    cats.append({"condition": c, "category": k, "judge": jb["overall"]["by_category"].get(k)})
    run = start_run(f"cocoedit_eval_{editor}_{dataset}", cfg, smoke, {"merge_run": str(merge_run)})
    df.to_csv(run / "samples.csv", index=False)
    pd.DataFrame(table).to_csv(run / "table.csv", index=False)
    pd.DataFrame(cats).to_csv(run / "categories.csv", index=False)
    write_json(run / "metrics.json", {
        "deviation": "D25", "editor": editor, "dataset": dataset, "smoke": smoke, "merge_run": merge_run.name,
        "judge_run": judge_note if judge else None, "judge_note": None if judge else judge_note,
        "scored_samples": len(items), "conditions": metrics, "table": table, "localization": mm["localization"],
        "modes": mm["modes"], "lpips": f"lpips {cfg['cocoedit']['eval']['lpips_net']} (v0.1), [-1, 1] inputs at "
                                       "source resolution: edited image with the GT edit region set to source pixels "
                                       "vs source",
        "protocol": "released run_psnr_score.py: float32 RGB / 255, white (> 128) mask, ImgEdit background inverted; "
                    "PSNR = 10 log10(1 / (sum(mask * diff^2) / (3 sum(mask)))), 100 if MSE = 0, NaN-dropped mean; "
                    "SSIM = skimage(win 11, gaussian, population covariance, data_range 1) per channel, mask "
                    "ignored by skimage (full image); ssim_masked = the same map averaged over the mask inside the "
                    "5-px border crop; psnr_capped = min(PSNR, 50); psnr_pooled from summed squared errors"})
    print(pd.DataFrame(table).to_string(index=False))
    print(run)
    if not smoke:
        combine(cfg, dataset)


def combine(cfg: dict, dataset: str) -> None:
    rows = [{"method": m, "source": f"CoCoEdit (arXiv 2602.14068) {t}", "condition": None, "overall": o, "psnr": p,
             "ssim": s, "lpips": lp} for m, t, o, p, s, lp in CM.PUBLISHED[dataset]]
    for editor in EDITORS:
        try:
            run = resolve_run(cfg, f"cocoedit_eval_{editor}_{dataset}", "latest", False)
        except ProtocolError:
            continue
        m = read_json(run / "metrics.json")
        for r in m["table"]:
            tag = " (oracle: GT mask)" if r["condition"].startswith(("gt_", "bbox_")) else ""
            rows.append({"method": f"{editor} [{r['condition']}]{tag}", "source": f"ours {run.name}",
                         "condition": r["condition"], **{k: r.get(k) for k in ("overall", "psnr", "ssim", "lpips",
                                                                                 "ssim_masked", "psnr_capped",
                                                                                 "psnr_pooled", "at_cap_rate")}})
    out = Path(cfg["tg_root"]) / "results" / "cocoedit"
    out.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows)
    df.to_csv(out / f"{dataset}.csv", index=False)
    write_json(out / f"{dataset}.json", {"deviation": "D25", "dataset": dataset, "rows": [
        {k: (None if isinstance(v, float) and math.isnan(v) else v) for k, v in r.items()} for r in rows]})
    print(df.to_string(index=False))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--editor", choices=EDITORS, required=True)
    ap.add_argument("--dataset", choices=CB.DATASETS, required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    evaluate(load_config(args.config), args.editor, args.dataset, args.smoke)


if __name__ == "__main__":
    main()
