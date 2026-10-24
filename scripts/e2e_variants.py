import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground.calib import GRID  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import ProtocolError, load_config, read_json, require_approved, resolve_run, start_run, write_json  # noqa: E402

# Deviation D15: single-step end-to-end numbers recomputed from each stage's saved per-query table, once at the
# recorded calibrated tau_rej and once with rejection disabled on present queries (always answer the top node, the
# protocol of present-only benchmarks); no-target accuracy is reported separately, plus the full threshold curve.

STAGES = ("v2", "s1", "s2_eval", "s2_from_s1_eval", "s2_from_s1_long_eval")


def block(pres: pd.DataFrame, iou: np.ndarray, answered: np.ndarray, bs: dict) -> dict:
    def ci(v, frame=pres):
        return mean_ci(np.asarray(v, float), frame.image.to_numpy(), bs["n"], bs["seed"])

    out = {"gIoU": ci(iou), "success@0.5": ci(iou >= 0.5),
           "by_kind": {k: ci(iou[(pres.kind == k).to_numpy()], pres[pres.kind == k]) for k in sorted(pres.kind.unique())}}
    if {"sel_inter", "sel_union", "gt_area"} <= set(pres.columns):
        # an answered query adds its prediction-target union; a rejected one (or one with no nodes) its GT area
        keep = answered & (pres.sel_union.to_numpy() > 0)
        union = np.where(keep, pres.sel_union, pres.gt_area)
        out["cIoU"] = float(np.where(keep, pres.sel_inter, 0).sum() / max(1, union.sum()))
    return out


def variants(df: pd.DataFrame, tau: float, bs: dict) -> tuple[dict, pd.DataFrame]:
    pres, absn = df[df.present], df[~df.present]
    sel = pres.sel_iou.to_numpy(float)
    calibrated = np.where(pres.max_sigma >= tau, sel, 0.0)
    rows = [{"tau": float(t), "present_gIoU": float(np.where(pres.max_sigma >= t, sel, 0.0).mean()),
             "no_target_accuracy": float((absn.max_sigma < t).mean()) if len(absn) else None} for t in GRID]
    out = {"tau_rej": tau, "present_queries": int(len(pres)), "no_target_queries": int(len(absn)),
           "calibrated": {**block(pres, calibrated, (pres.max_sigma >= tau).to_numpy(), bs),
                          "present_accept_rate": float((pres.max_sigma >= tau).mean()),
                          "no_target_accuracy": float((absn.max_sigma < tau).mean()) if len(absn) else None},
           "no_rejection_present": block(pres, sel, np.ones(len(pres), bool), bs)}
    return out, pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--stages", nargs="*", default=list(STAGES))
    args = ap.parse_args()
    cfg = load_config(args.config)
    require_approved(cfg, ["D15"])
    bs = cfg["v2"]["bootstrap"]
    found = {}
    for stage in args.stages:
        try:
            found[stage] = resolve_run(cfg, stage, "latest", False)
        except ProtocolError as e:
            print(f"{stage}: skipped ({e})")
    run = start_run("e2e_variants", cfg, False, {s: str(r) for s, r in found.items()})
    metrics = {"deviation": "D15", "label": "dev", "split": "PACO-LVIS val + gRefCOCO val no-target", "stages": {}}
    for stage, r in found.items():
        q = r / "eval_queries.csv"
        if not q.exists():
            metrics["stages"][stage] = f"no eval_queries.csv in {r.name}"
            continue
        m = read_json(r / "metrics.json")
        tau = float(m["tau_rej"] if "tau_rej" in m else m["single_step"]["tau_rej"])
        out, curve = variants(pd.read_csv(q), tau, bs)
        out["run"] = r.name
        metrics["stages"][stage] = out
        curve.to_csv(run / f"threshold_curve_{stage}.csv", index=False)
        c, n = out["calibrated"]["gIoU"]["mean"], out["no_rejection_present"]["gIoU"]["mean"]
        print(f"{stage}: present gIoU calibrated {c:.3f}  no rejection {n:.3f}  tau {tau:.3f}")
    write_json(run / "metrics.json", metrics)
    print(run)


if __name__ == "__main__":
    main()
