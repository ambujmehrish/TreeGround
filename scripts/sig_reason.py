import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from sig_table1 import check, latest, paired_bootstrap  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json, start_run, write_json  # noqa: E402

# Paired significance of the D50 / D51 reasoning rows (read-only, CPU; nothing re-predicted or tuned): per test set,
# ours + query rewriting (D51) against the best same-size baseline row and against ours (D50), from the merge runs'
# per-query results (eval_queries.csv), each checked to reproduce its reported gIoU and cIoU. gIoU: the paired
# image-level bootstrap of sig_table1; cIoU: the same resampling of the summed intersections / unions. Images are
# clustered on the photo (InstructPart stores one file per (photo, object, part)).

BENCHES = ("instructpart_human", "instructpart_gpt", "reasonseg")
BASELINES = ("qwen_box_sam", "sa2va", "sam3_text_na")
TOL = 1e-9


def photo(qid: str) -> str:
    # reasonseg:<image_id>:<ann_id> -> reasonseg:<image_id>; instructpart:<base>-<object>-<part> -> instructpart:<base>
    src, rest = qid.split(":", 1)
    return f"{src}:{rest.rsplit('-', 2)[0]}" if src == "instructpart" else f"{src}:{rest.split(':')[0]}"


def rows(cfg: dict, stage: str) -> tuple[pd.DataFrame, dict]:
    run = latest(cfg, stage)
    df = pd.read_csv(run / "eval_queries.csv")
    df = pd.DataFrame({"qid": df.qid, "image": df.qid.map(photo), "inter": df.inter.astype(np.float64),
                       "union": df.union.astype(np.float64)}).set_index("qid")
    res = read_json(run / "metrics.json")["results"]
    iou = np.where(df.union > 0, df.inter / df.union.clip(lower=1), 0.0)
    check(iou, res["gIoU"]["mean"], f"{stage} gIoU")
    if abs(df.inter.sum() / max(1.0, df.union.sum()) - res["cIoU"]) > TOL:
        raise ProtocolError(f"{stage}: rebuilt cIoU differs from the reported {res['cIoU']}")
    return df.assign(iou=iou), {"gIoU": float(res["gIoU"]["mean"]), "cIoU": float(res["cIoU"])}


def paired_ratio_bootstrap(a: pd.DataFrame, b: pd.DataFrame, n: int, seed: int) -> dict:
    # cIoU(a) - cIoU(b) on the same queries, images resampled with replacement
    uniq, inv = np.unique(a.image.to_numpy(), return_inverse=True)
    k = len(uniq)
    s = {f"{name}_{col}": np.bincount(inv, weights=frame[col].to_numpy(), minlength=k)
         for name, frame in (("a", a), ("b", b)) for col in ("inter", "union")}
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        boot[i] = (w * s["a_inter"]).sum() / max((w * s["a_union"]).sum(), 1.0) - \
            (w * s["b_inter"]).sum() / max((w * s["b_union"]).sum(), 1.0)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    delta = a.inter.sum() / max(1.0, a.union.sum()) - b.inter.sum() / max(1.0, b.union.sum())
    return {"delta": float(delta), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "p_two_sided": float(min(1.0, max(p, 1.0 / n)))}


def compare(a: pd.DataFrame, b: pd.DataFrame, n: int, seed: int) -> dict:
    if set(a.index) != set(b.index) or a.index.duplicated().any():
        raise ProtocolError("the two rows were scored on different queries")
    b = b.loc[a.index]
    return {"gIoU": paired_bootstrap(a.iou.to_numpy(), b.iou.to_numpy(), a.image.to_numpy(), n, seed),
            "cIoU": paired_ratio_bootstrap(a, b, n, seed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/reason_ours.yaml")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import ta_test_eval as TTE
    cfg = load_config(ROOT / args.config)
    stage = TTE.TAT.stage(cfg)
    out = {}
    for bench in BENCHES:
        rw, rep_rw = rows(cfg, f"reason_{bench}_{stage}_rw")
        ours, rep_ours = rows(cfg, f"reason_{bench}_{stage}")
        base = {m: rows(cfg, f"reason_{bench}_{m}") for m in BASELINES}
        best = max(BASELINES, key=lambda m: base[m][1]["gIoU"])
        out[bench] = {"ours_rw": rep_rw, "ours": rep_ours, "best_baseline": best, "baseline": base[best][1],
                      "vs_best_baseline": compare(rw, base[best][0], args.n, args.seed),
                      "vs_ours": compare(rw, ours, args.n, args.seed)}
        for name in ("vs_best_baseline", "vs_ours"):
            r = out[bench][name]
            print(f"{bench:20s} {name:16s} " + "  ".join(
                f"{k} {v['delta'] * 100:+5.1f} [{v['ci95'][0] * 100:+.1f}, {v['ci95'][1] * 100:+.1f}] p {v['p_two_sided']:.4f}"
                for k, v in r.items()) + (f"  ({best})" if name == "vs_best_baseline" else ""), flush=True)
    run = start_run("sig_reason", cfg, False, {"stage": stage})
    write_json(run / "metrics.json", {"label": "test", "deviation": "D50/D51", "bootstrap": {"n": args.n, "seed": args.seed},
                                      "results": out})
    print(run)


if __name__ == "__main__":
    main()
