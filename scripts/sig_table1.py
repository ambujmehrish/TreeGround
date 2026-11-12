import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground.run import ProtocolError, load_config, read_json, resolve_run, start_run, write_json  # noqa: E402

# Paired significance of Table 1's margins (read-only; nothing is tuned or re-predicted). Per column, the per-query
# values each protocol script averages are rebuilt from its stored predictions (merge run's eval_queries.csv, or the
# cache its run records), checked to reproduce the run's reported metric exactly, and ours is compared with the best
# baseline row of that column on the same queries: delta = mean(ours) - mean(baseline), image-level paired bootstrap
# (images resampled with replacement, as mean_ci resamples them), 95% CI and a two-sided p-value.

BASELINES = ("sam3_text", "grounded_sam2", "sa2va", "qwen_box_sam")
CLIPPED = {"sam3_text": "sam3_text_clip"}    # gRefCOCO / GSEval ran SAM 3 text with the 32-token clip (D27)
TOL = 1e-9


# ---- statistics ----

def paired_bootstrap(a: np.ndarray, b: np.ndarray, groups: np.ndarray, n: int, seed: int) -> dict:
    # a, b: per-unit values of the two methods on the same units; groups: the unit's image
    uniq, inv = np.unique(groups, return_inverse=True)
    k = len(uniq)
    sd = np.bincount(inv, weights=a - b, minlength=k)
    sc = np.bincount(inv, minlength=k).astype(np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty(n)
    for i in range(n):
        w = np.bincount(rng.integers(0, k, k), minlength=k)
        boot[i] = (w * sd).sum() / max((w * sc).sum(), 1.0)
    p = 2 * min((boot <= 0).mean(), (boot >= 0).mean())
    return {"delta": float((a - b).mean()), "ci95": [float(x) for x in np.quantile(boot, [0.025, 0.975])],
            "p_two_sided": float(min(1.0, max(p, 1.0 / n))), "units": int(len(a)), "images": int(k)}


def check(values: np.ndarray, reported: float, what: str) -> None:
    if abs(float(np.mean(values)) - reported) > TOL:
        raise ProtocolError(f"{what}: rebuilt mean {np.mean(values):.6f} differs from the reported {reported:.6f}")


# ---- run access ----

def latest(cfg: dict, stage: str) -> Path:
    return resolve_run(cfg, stage, "latest", False)


def cache_rows(run: Path) -> pd.DataFrame:
    d = Path(read_json(run / "meta.json")["cache"]) / "shards"
    files = sorted(d.glob("*.parquet"))
    if not files:
        raise ProtocolError(f"no shards in {d}")
    return pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)


def reported(run: Path, *keys) -> float:
    v = read_json(run / "metrics.json")
    for k in keys:
        v = v[k]
    return float(v["mean"] if isinstance(v, dict) else v)


def units(df: pd.DataFrame, uid: str, value: np.ndarray, image: str = "image") -> pd.DataFrame:
    return pd.DataFrame({"uid": df[uid].astype(str).to_numpy(), "image": df[image].astype(str).to_numpy(),
                         "value": np.asarray(value, np.float64)})


# ---- columns: each returns {method: (units, reported value)} for ours and the baselines ----

def iou_of(df: pd.DataFrame, inter: str = "inter", union: str = "union") -> np.ndarray:
    return np.where(df[union] > 0, df[inter] / df[union].clip(lower=1), 0.0)


def ophrg(cfg: dict, stage: str, bench: str, kind: str) -> dict:
    key = "parts" if kind == "part" else "objects"
    out = {}
    for m, st in [("ours", stage)] + [(b, b) for b in BASELINES]:
        run = latest(cfg, f"ophrg_{bench}_{st}")
        df = pd.read_csv(run / "eval_queries.csv")
        df = df[df.kind == kind]
        out[m] = (units(df, "qid", iou_of(df)), reported(run, "results", key, "gIoU"))
    return out


def paco(cfg: dict, stage: str, kind: str) -> dict:
    out = {}
    run = latest(cfg, f"pacotest_{stage}")
    df = pd.read_csv(run / "eval_queries.csv")
    df = df[df.kind == kind]
    out["ours"] = (units(df, "qid", df.sel_iou), reported(run, "by_kind", kind, "gIoU"))
    for b in BASELINES:
        run = latest(cfg, f"pacotest_{b}")
        df = pd.read_csv(run / "eval_queries.csv")
        df = df[(df.kind == kind) & df.present.astype(bool)]
        out[b] = (units(df, "qid", df.iou), reported(run, "by_kind", kind, "gIoU"))
    return out


def partonomy(cfg: dict, stage: str, level: str) -> dict:
    import partonomy_eval as PE
    qs = PE.questions(Path(cfg["data"]["partonomy"]))
    qt = next(k for k, v in PE.QTYPES.items() if v == "identification")
    out = {}
    for m, st, col in [("ours", stage, "nodes")] + [(b, b, "pred") for b in BASELINES]:
        run = latest(cfg, f"partonomy_{st}")
        qi = PE.question_ious(qs, cache_rows(run), f"pred_{col}")
        rows = []
        for i, r in enumerate(qs):
            if r["question_type"] != qt or not r["parts"]:
                continue
            if level == "macro":
                rows.append((str(i), r["image_path"], float(np.mean(qi[i]))))
            else:
                rows += [(f"{i}:{j}", r["image_path"], v) for j, v in enumerate(qi[i])]
        u = pd.DataFrame(rows, columns=["uid", "image", "value"])
        out[m] = (u, reported(run, "results", col, "identification", f"{level}_gIoU"))
    return out


def gseval(cfg: dict, stage: str, stratum: str | None) -> dict:
    import gseval_eval as GS
    cid = None if stratum is None else next(k for k, v in GS.STRATA.items() if v == stratum)
    name = stratum or "all"
    out = {}
    for m, st, col, rule, key in [("ours", stage, "cand_nodes", "top1", "nodes/top1")] + \
            [(b, CLIPPED.get(b, b), "cand_pred", "set", "official") for b in BASELINES]:
        run = latest(cfg, f"gseval_{st}")
        df = cache_rows(run)
        s = GS.iou_rows(df, col, rule, -1.0).assign(qid=df.qid.to_numpy())
        if cid is not None:
            s = s[s.class_id == cid]
        out[m] = (units(s, "qid", s.iou), reported(run, "results", key, name, "gIoU"))
    return out


def gres(cfg: dict, stage: str, split: str, metric: str) -> dict:
    import gres_eval as GE
    out = {}
    for m, st in [("ours", stage)] + [(b, CLIPPED.get(b, b)) for b in BASELINES]:
        run = latest(cfg, f"gres_{st}")
        res = read_json(run / "metrics.json")["results"]
        key, col, rule = ("nodes/top1", "cand_nodes", "top1") if m == "ours" else ("pred/set", "cand_pred", "set")
        t = float(res[key]["t_val"]) if m == "ours" else 0.5
        df = cache_rows(latest(cfg, f"gres_{st}_{split}_shard"))
        s = GE.scores(df, col, rule, t).assign(qid=df.qid.to_numpy())
        if metric == "N-acc":
            s = s[s.no_target.astype(bool)]
            u, rep = units(s, "qid", s.pred_empty.astype(float)), float(res[key][split]["N-acc"])
        else:
            u, rep = units(s, "qid", s.iou), float(res[key][split]["gIoU"]["mean"])
        out[m] = (u, rep)
    return out


def columns(cfg: dict, stage: str) -> dict:
    return {
        "PARTONOMY micro": lambda: partonomy(cfg, stage, "micro"),
        "PARTONOMY macro": lambda: partonomy(cfg, stage, "macro"),
        "GSEval Part": lambda: gseval(cfg, stage, "part"),
        "PACO Part": lambda: paco(cfg, stage, "part"),
        "PPS-116 Part": lambda: ophrg(cfg, stage, "pascalpart", "part"),
        "PartImageNet": lambda: ophrg(cfg, stage, "partimagenet", "part"),
        "InstructPart": lambda: ophrg(cfg, stage, "instructpart", "part"),
        "PACO Object": lambda: paco(cfg, stage, "object"),
        "PPS-116 Object": lambda: ophrg(cfg, stage, "pascalpart", "object"),
        "GSEval All": lambda: gseval(cfg, stage, None),
        "gRefCOCO testA": lambda: gres(cfg, stage, "testA", "gIoU"),
        "gRefCOCO testB": lambda: gres(cfg, stage, "testB", "gIoU"),
        "gRefCOCO N-acc testA": lambda: gres(cfg, stage, "testA", "N-acc"),
        "gRefCOCO N-acc testB": lambda: gres(cfg, stage, "testB", "N-acc"),
    }


def compare(per: dict, n: int, seed: int) -> dict:
    # every method's rebuilt values must reproduce its reported number; the best baseline row is the comparison
    for m, (u, rep) in per.items():
        check(u.value.to_numpy(), rep, m)
    best = max((b for b in per if b != "ours"), key=lambda b: per[b][1])
    a, b = per["ours"][0].set_index("uid"), per[best][0].set_index("uid")
    if set(a.index) != set(b.index) or a.index.duplicated().any():
        raise ProtocolError(f"ours and {best} were scored on different queries")
    b = b.loc[a.index]
    if not (a.image == b.image).all():
        raise ProtocolError(f"ours and {best} disagree on query images")
    return {"ours": per["ours"][1], "best_baseline": best, "baseline": per[best][1],
            **paired_bootstrap(a.value.to_numpy(), b.value.to_numpy(), a.image.to_numpy(), n, seed)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="configs/ta_test_pool_soft_fam.yaml")
    ap.add_argument("--n", type=int, default=10000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    import ta_test_eval as TTE
    cfg = load_config(ROOT / args.config)
    stage = TTE.TAT.stage(cfg)
    res = {}
    for name, get in columns(cfg, stage).items():
        r = compare(get(), args.n, args.seed)
        res[name] = r
        print(f"{name:22s} ours {r['ours'] * 100:5.1f}  {r['best_baseline']:13s} {r['baseline'] * 100:5.1f}  "
              f"delta {r['delta'] * 100:+5.1f} [{r['ci95'][0] * 100:+.1f}, {r['ci95'][1] * 100:+.1f}]  "
              f"p {r['p_two_sided']:.4f}", flush=True)
    run = start_run("sig_table1", cfg, False, {"stage": stage})
    write_json(run / "metrics.json", {"label": "test", "what": "paired image-level bootstrap of Table 1 margins",
                                      "stage": stage, "bootstrap": {"n": args.n, "seed": args.seed}, "results": res})
    print(run)


if __name__ == "__main__":
    main()
