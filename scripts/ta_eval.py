import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_train as S2T  # noqa: E402
import ta_train as TAT  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground import tree_attn as TA  # noqa: E402
from treeground.cache import save_tensors  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import (ProtocolError, load_config, require_approved, resolve_run, start_run,  # noqa: E402
                            v1_gate, write_json)

# PACO-LVIS val evaluation of a D41 checkpoint (development only). The node logits of every val query are computed
# with the checkpoint's chosen read-out layer and stored in the layout of the S2 recomputed cache (probes/<sha1>,
# one row per <SEL>, query offsets as in the query cache). The unchanged V2 / V4 evaluators then read them through
# an identity selector (query vector = logit row, node matrix = identity), so metrics, rejection calibration and
# chain search are computed by the same code as for S2.

EVAL_KINDS = ("object", "part", "chain", "absent")


def eval_stage(cfg: dict, relations: str | None) -> str:
    # a relation swap (diagnostic: is the tree used?) gets its own stage next to the regular evaluation
    return f"{TAT.stage(cfg)}_eval" + (f"_rel_{relations}" if relations else "")


class IdentitySelector(torch.nn.Module):
    def nodes(self, z: torch.Tensor) -> torch.Tensor:
        return z

    def queries(self, h: torch.Tensor) -> torch.Tensor:
        return h

    def scores(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ e.T


class IdentityStore:
    # same interface as FeatureStore.load: an n x n identity per image, n = its capped node count
    def __init__(self, props, tau: float):
        self.props, self.tau = props, tau

    def load(self, sha1: str, keys: list[str]) -> dict[str, torch.Tensor]:
        n = len(self.props.nodes(sha1, self.tau)["rles"])
        return {k: torch.eye(n) for k in keys}


def load_checkpoint(cfg: dict, smoke: bool) -> tuple[dict, Path]:
    stage = TAT.stage(cfg)
    run = resolve_run(cfg, stage, cfg["inputs"].get(stage, "latest"), smoke)
    ckpt = torch.load(run / cfg["ta_eval"]["checkpoint"], map_location="cpu")
    meta = ckpt["meta"]
    if meta["smoke"] != smoke or meta["variant"] != cfg["ta"]["variant"]:
        raise ProtocolError("checkpoint smoke flag or variant does not match the config")
    return ckpt, run


def compute_logits(model, readout, src, props, queries, wl, layer, device, out: Path, batch_size, rank, world) -> int:
    # every query of an evaluated image in cache order, so that the stored offsets index the logit rows
    files, keys = dict(zip(wl.sha1, wl.path)), dict(zip(wl.sha1, wl.image_key))
    val = queries[(queries.split == "val") & queries.kind.isin(EVAL_KINDS)]
    groups = sorted(val.sha1.unique())[rank::world]
    (out / "probes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, sha1 in enumerate(groups, 1):
        g = queries[queries.sha1 == sha1].sort_values("offset")
        if g.kind.isin(["text"]).any():
            raise ProtocolError(f"text-only queries share image group {sha1}")
        rles = props.nodes(sha1, src.tau)["rles"]
        parents = T.build_tree(rles, device).parent
        qs = [S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [])
              for q in g.itertuples()]
        m = S2T.ImageMeta(sha1, keys.get(sha1, g.image_key.iloc[0]), files[sha1], len(rles), parents, None, None,
                          None, qs)
        rows = torch.cat([lg[layer] for lg in TAT.query_logits(model, readout, src, m, qs, device, batch_size,
                                                                [layer])])
        if rows.shape != (int(g.n_sel.sum()), len(rles)):
            raise ProtocolError(f"logit rows {tuple(rows.shape)} do not match the queries of {sha1}")
        save_tensors(out / "probes" / f"{sha1}.safetensors", {"probes": rows[:, None].contiguous()},
                     {"image_sha1": sha1})
        if i % 100 == 0:
            print(f"rank {rank}: {i}/{len(groups)} images {time.time() - t0:.0f}s", flush=True)
    return len(groups)


def no_rejection(per_query: pd.DataFrame) -> dict:
    pres = per_query[per_query.present.astype(bool)]
    covered = pres[pres.best_iou >= 0.5]
    return {"queries": int(len(pres)), "gIoU": float(pres.sel_iou.mean()),
            "success@0.5": float((pres.sel_iou >= 0.5).mean()),
            "by_kind": {k: {"gIoU": float(g.sel_iou.mean()), "success@0.5": float((g.sel_iou >= 0.5).mean())}
                        for k, g in pres.groupby("kind")},
            "coverage@0.5": float((pres.best_iou >= 0.5).mean()), "oracle_gIoU": float(pres.best_iou.mean()),
            "selector_success_if_covered": float((covered.sel_iou >= 0.5).mean()) if len(covered) else 0.0}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--relations", choices=TA.RELATIONS, default=None,
                    help="diagnostic: evaluate the checkpoint with other node-token relations than it was trained with")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    c = TAT.stage_cfg(cfg, args.smoke)
    rank, world, _ = S2T.dist_env()
    device = TAT.setup_device(cfg)
    require_approved(cfg, TAT.deviations(c["variant"]))
    TA.check_version()
    gate = v1_gate(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    ckpt, train_run = load_checkpoint(cfg, args.smoke)
    meta = ckpt["meta"]
    layer = int(meta["layer"])
    built = {**c, "lora": meta["lora"], "readout_layers": meta["readout_layers"]}
    model, processor, extra, readout = TAT.build(cfg, built, device, static, meta["n_nodes"])
    if extra["sel_id"] != meta["sel_id"]:
        raise ProtocolError("<SEL> token id differs from the checkpoint")
    LR.load_trainable(model, extra["sel_emb"], readout, ckpt)
    model.eval()
    readout.eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = None
    if rank == 0:
        run = start_run(eval_stage(cfg, args.relations), cfg, args.smoke, {
            "relations": args.relations or "as trained",
            "manifest_run": str(mrun), "train_run": str(train_run), "checkpoint": cfg["ta_eval"]["checkpoint"],
            "checkpoint_step": meta["step"], "layer": layer, "variant": meta["variant"], "world_size": world})
    run = Path(S2T.broadcast(str(run) if run else None, world))
    out = run / "logits"
    src = TAT.Source(cfg, args.smoke, manifest, mhash, processor, extra["sel_id"], set(), meta["variant"],
                     meta["seed"], args.relations)
    n = compute_logits(model, readout, src, props, queries, wl, layer, device, out, cfg["ta_eval"]["batch_size"],
                       rank, world)
    print(f"rank {rank}: logits for {n} images", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        sel, store = IdentitySelector(), IdentityStore(props, src.tau)
        metrics, per_query, tau = V2.evaluate(cfg, sel, store, props, queries, out, 0, layer, wl, device)
        per_query.to_csv(run / "eval_queries.csv", index=False)
        chains = V4.chain_inputs(cfg, sel, store, props, queries, out, 0, layer, device)
        chain_metrics = {"label": "dev", "split": "PACO-LVIS val supplied object->part chains (I1)",
                         "tau_rej_chain": tau, "seeds": cfg["v4"]["seeds"]}
        chain_metrics.update(V4.search_metrics(cfg, chains, tau, run))
        write_json(run / "metrics.json", {
            "label": "dev", "smoke": args.smoke, "variant": meta["variant"], "layer": layer,
            "relations": args.relations or "as trained",
            "train_run": train_run.name, "checkpoint_step": meta["step"], "checkpoint_val": meta["val"],
            "single_step": metrics, "no_rejection": no_rejection(per_query), "chains": chain_metrics,
            "v1_gate": gate, "deviations": TAT.deviations(meta["variant"])})
        print(run, "tau_rej", tau)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
