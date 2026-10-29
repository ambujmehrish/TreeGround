import argparse
import importlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.run import ProtocolError, load_config, require_approved, start_run, write_json  # noqa: E402

# Deviation D40 (baseline parity), Group C evaluation: runs an existing evaluator (gres_eval, partonomy_eval,
# ophrg_eval) unchanged except for its baseline predictor, which becomes a b_parity predictor (every instance, the
# method's official setting). Hashed evaluator files are not edited; this wrapper and b_parity.py enter the cache
# identity, and every parity method has its own stage name, so D13 / D31 / D36 caches are untouched.
#   gres_eval      per expression the method's instances with scores; merge reports 'official' (union above the
#                  official threshold) and, exactly as for ours, 'set' and 'top1' with t chosen by 2-fold image CV
#                  on val and the all-val t frozen for testA / testB
#   partonomy_eval the official union, query 'the {part} of the {object}' as for every method in D32
#   paco_test_eval the official union, the D35 phrase (single-target queries; the D13 / D31 rows are unchanged)
#   ophrg_eval     the official union and OP-HRG's own query strings (benchmark_datasets.py): InstructPart and
#                  PascalPart parts "{object}'s {part}", PascalPart objects "{object}", PartImageNet the category
#                  name lower-cased ("{object} {part}")
# Usage: python scripts/parity_eval.py <evaluator> --config configs/parity.yaml --method <parity method> [...]

EVALUATORS = ("gres_eval", "partonomy_eval", "ophrg_eval", "paco_test_eval")


def ophrg_query(key: str, kind: str, obj: str, part: str | None) -> str:
    if key.startswith("partimagenet_ood:"):
        return f"{obj} {part}"
    if kind == "object":
        return obj
    return f"{obj}'s {part}"


def install(mod, name: str, cfg: dict, method: str) -> None:
    import b_parity as BP
    if method not in BP.PARITY:
        raise ProtocolError(f"parity_eval runs only {sorted(BP.PARITY)}")
    mod.FILES = tuple(mod.FILES) + ("scripts/b_parity.py", "scripts/parity_eval.py")
    base_identity = mod.identity

    def identity(cfg, *a, **k):
        ident = {**base_identity(cfg, *a, **k), "parity": cfg["parity"]}
        if method == "qwenbox_ft":
            import qwenbox_train as QT
            ident["qwenbox_checkpoint"] = str(QT.checkpoint(cfg)[0])   # the resolved run, not 'latest'
        return ident
    mod.identity = identity

    if name == "gres_eval":
        class Baseline:
            def __init__(self, cfg, method):
                self.pred = BP.PARITY[method](cfg)

            def __call__(self, image, texts):
                return [{"pred": [(s, M.encode(m)[0]) for s, m in c if m.any()]}
                        for c in self.pred.instances(image, texts)]
        mod.Baseline = Baseline
        mod.merge = lambda cfg, method: gres_merge(mod, cfg, method)

    elif name == "partonomy_eval":
        class Baseline:
            def __init__(self, cfg, method):
                self.cfg, self.method, self.pred = cfg, method, BP.PARITY[method](cfg)

            def __call__(self, image, label, parts):
                res = self.pred.instances(image, [mod.phrase(p, label) for p in parts])
                return [{"pred": BP.official_union(self.cfg, self.method, c, image.shape[:2])} for c in res]
        mod.Baseline = Baseline

    elif name == "paco_test_eval":
        import b_baselines as B

        class Baseline:
            def __init__(self, cfg, method):
                self.cfg, self.method, self.pred = cfg, method, BP.PARITY[method](cfg)

            def __call__(self, image, g, gts):
                g = g[g.kind.isin(["object", "part"])]
                res = self.pred.instances(image, [B.phrase(t) for t in g.text])
                rows = []
                for q, c in zip(g.itertuples(), res):
                    mask = BP.official_union(self.cfg, self.method, c, image.shape[:2])
                    rle = M.encode(mask)[0] if mask is not None and mask.any() else None
                    gt = gts[json.loads(q.gt)[0]]
                    rows.append({"qid": q.qid, "kind": q.kind, "image": q.image_key, "present": True,
                                 "score": float(max((s for s, _ in c), default=float("nan"))),
                                 "accepted": rle is not None, **B.overlap(rle, gt), "iou_any": B.overlap(rle, gt)["iou"]})
                return rows
        mod.Baseline = Baseline

    else:
        class Baseline:
            def __init__(self, cfg, method):
                self.cfg, self.method, self.pred = cfg, method, BP.PARITY[method](cfg)

            def __call__(self, image, key, qs):
                res = self.pred.instances(image, [ophrg_query(key, q[1], q[2], q[3]) for q in qs])
                rows = []
                for q, c in zip(qs, res):
                    pred = BP.official_union(self.cfg, self.method, c, q[4].shape)
                    pred = np.zeros(q[4].shape, bool) if pred is None else pred
                    rows.append({"qid": q[0], "kind": q[1], "image": key, "inter": int((pred & q[4]).sum()),
                                 "union": int((pred | q[4]).sum())})
                return rows
        mod.Baseline = Baseline


def gres_merge(G, cfg: dict, method: str) -> None:
    import b_parity as BP
    require_approved(cfg, G.approvals(method))
    bs = cfg["v2"]["bootstrap"]
    splits = {sp: G.load_split(cfg, method, sp) for sp in G.SPLITS if cfg["gres_eval"]["splits"].count(sp)}
    run = start_run(f"gres_{G.stage_name(cfg, method)}", cfg, False, {})
    val = splits["val"]
    out = {"deviation": "D40", "method": method, "stage": G.stage_name(cfg, method), "protocol": "GRES (gRefCOCO)",
           "results": {}}
    thr = BP.official_threshold(cfg, method)
    res = {sp: G.metrics(G.scores(df, "cand_pred", "set", -1.0 if thr is None else thr + 1e-9), bs)
           for sp, df in splits.items()}
    out["results"]["official"] = dict(res, threshold=thr)
    images = np.array(sorted(val.image.unique()))
    fold = dict(zip(images, np.random.default_rng(0).permutation(len(images)) % 2))
    f = val.image.map(fold).to_numpy()
    for rule in ("set", "top1"):
        parts = []
        for k in (0, 1):
            t, _ = G.best_t(val[f != k], "cand_pred", rule)
            parts.append(G.scores(val[f == k], "cand_pred", rule, t).assign(t=t))
        r = {"val (2-fold t)": G.metrics(pd.concat(parts), bs), "t_folds": [float(p.t.iloc[0]) for p in parts]}
        t_all, curve = G.best_t(val, "cand_pred", rule)
        r["t_val"], r["val_curve"] = t_all, curve
        for sp, df in splits.items():
            if sp != "val":
                r[sp] = G.metrics(G.scores(df, "cand_pred", rule, t_all), bs)
        out["results"][f"tuned/{rule}"] = r
    write_json(run / "metrics.json", out)
    print(json.dumps({k: {s: (v[s]["gIoU"]["mean"], v[s]["cIoU"], v[s]["N-acc"]) for s in v
                          if isinstance(v[s], dict) and "gIoU" in v[s]} for k, v in out["results"].items()}, indent=1))
    print(run)


def main() -> None:
    if len(sys.argv) < 2 or sys.argv[1] not in EVALUATORS:
        raise ProtocolError(f"first argument must be one of {EVALUATORS}")
    name = sys.argv.pop(1)
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True)
    known, _ = ap.parse_known_args()
    cfg = load_config(known.config)
    require_approved(cfg, ["D40"])
    mod = importlib.import_module(name)
    install(mod, name, cfg, known.method)
    mod.main()


if __name__ == "__main__":
    main()
