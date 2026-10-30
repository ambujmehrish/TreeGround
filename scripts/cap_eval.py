import argparse
import json
import sys
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_eval as S2E  # noqa: E402
from treeground import identity as I  # noqa: E402
from treeground.run import ProtocolError, write_json  # noqa: E402

# Development check (PACO-LVIS val only): the S2-long checkpoint, unchanged, evaluated with a different node budget
# (top-k by stability, as D4, with k from --cap) instead of the D4 k = 256. Node features and probes are recomputed
# by s2_eval exactly as for the official S2-long evaluation; only the cap differs. Results go to their own stage
# '<stage>_eval_cap<k>' so the official '<stage>_eval' run is never replaced. Adds a no-rejection summary over all
# present queries (gIoU, success@0.5) so caps compare directly; the conditioned selector success depends on coverage.

_node_cap, _start_run, _init_pg = I.node_cap, S2E.start_run, dist.init_process_group
RUN = {}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cap", type=int, required=True)
    known, rest = ap.parse_known_args()
    sys.argv = [sys.argv[0]] + rest
    k = known.cap

    def node_cap(cfg, smoke):
        base = _node_cap(cfg, smoke)
        if base is None:
            return None
        if base["order"] != "stability":
            raise ProtocolError(f"cap_eval assumes the D4 stability order, got {base['order']}")
        return {**base, "k": k, "d4_k": base["k"]}

    def start_run(stage, cfg, smoke, meta):
        if not stage.endswith("_eval"):
            raise ProtocolError(f"unexpected stage {stage}")
        run = _start_run(f"{stage}_cap{k}", cfg, smoke, {**meta, "node_cap": k, "label": "dev"})
        RUN["path"] = run
        return run

    def init_pg(*a, **kw):
        # rank 0 scores alone while the other ranks wait at the final barrier; the default 10 min limit is too short at large caps
        return _init_pg(*a, **{"timeout": timedelta(hours=3), **kw})

    dist.init_process_group = init_pg
    I.node_cap = node_cap
    S2E.start_run = start_run
    S2E.main()
    run = RUN.get("path")
    if run is None or not (Path(run) / "eval_queries.csv").exists():
        return  # non-zero ranks, or the evaluation did not finish
    df = pd.read_csv(Path(run) / "eval_queries.csv")
    pres = df[df.present.astype(bool)]
    out = {"label": "dev", "split": "PACO-LVIS val, present object / part queries", "node_cap": k,
           "queries": int(len(pres)),
           "no_rejection": {"gIoU": float(pres.sel_iou.mean()), "success@0.5": float((pres.sel_iou >= 0.5).mean()),
                            "by_kind": {kk: {"gIoU": float(g.sel_iou.mean()),
                                             "success@0.5": float((g.sel_iou >= 0.5).mean())}
                                        for kk, g in pres.groupby("kind")}},
           "coverage@0.5": float((pres.best_iou >= 0.5).mean()),
           "oracle_gIoU": float(pres.best_iou.mean()),
           "selector_success_if_covered": float((pres[pres.best_iou >= 0.5].sel_iou >= 0.5).mean())}
    write_json(Path(run) / "cap_summary.json", out)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
