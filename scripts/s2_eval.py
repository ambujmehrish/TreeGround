import argparse
import os
import sys
import time
from pathlib import Path

import torch
import torch.distributed as dist

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
import v4_tree_value as V4  # noqa: E402
from treeground import lora as LR  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground.cache import load_tensors, save_tensors  # noqa: E402
from treeground.identity import manifest_inputs, open_proposals, open_queries, qwen_static  # noqa: E402
from treeground.run import ProtocolError, load_config, resolve_run, start_run, v1_gate, write_json  # noqa: E402
from treeground.selector import Selector  # noqa: E402

EVAL_KINDS = ("object", "part", "chain", "absent")


class RecomputedStore:
    # same interface as FeatureStore.load, over the node features recomputed with the S2 model
    def __init__(self, path: Path):
        self.path = path

    def load(self, sha1: str, keys: list[str] | None = None) -> dict[str, torch.Tensor]:
        f = self.path / "images" / f"{sha1}.safetensors"
        if not f.exists():
            raise ProtocolError(f"missing recomputed features {f}")
        return load_tensors(f, keys)[0]


def load_checkpoint(cfg: dict, smoke: bool) -> tuple[dict, Path]:
    stage = S2T.s2_stage(cfg)
    run = resolve_run(cfg, stage, cfg["inputs"].get(stage, "latest"), smoke)
    ckpt = torch.load(run / cfg["s2_eval"]["checkpoint"], map_location="cpu")
    if ckpt["meta"]["smoke"] != smoke:
        raise ProtocolError("S2 checkpoint smoke flag does not match the current mode")
    return ckpt, run


def recompute_all(model, src, queries, wl, layer, device, out: Path, batch_size, rank, world) -> int:
    # every query of an evaluated image, in cache order, so that the stored offsets index the probe rows
    files = dict(zip(wl.sha1, wl.path))
    keys = dict(zip(wl.sha1, wl.image_key))
    val = queries[(queries.split == "val") & queries.kind.isin(EVAL_KINDS)]
    groups = sorted(val.sha1.unique())[rank::world]
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "probes").mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    for i, sha1 in enumerate(groups, 1):
        g = queries[queries.sha1 == sha1].sort_values("offset")
        if g.kind.isin(["text"]).any():
            raise ProtocolError(f"text-only queries share image group {sha1}")
        n = len(src.props.nodes(sha1, src.tau)["rles"])
        qs = [S2T.Query(q.qid, q.kind, S2.user_text(q.text), S2.answer_text(q.kind, q.qid, q.text), [])
              for q in g.itertuples()]
        m = S2T.ImageMeta(sha1, keys.get(sha1, g.image_key.iloc[0]), files[sha1], n, None, None, None, None, qs)
        z, probes = S2T.recompute(model, src, m, qs, layer, device, batch_size)
        stacked = torch.cat(probes)
        if stacked.shape[0] != int(g.n_sel.sum()):
            raise ProtocolError(f"probe count mismatch for {sha1}")
        save_tensors(out / "images" / f"{sha1}.safetensors", {f"hidden_L{layer}": z.half()}, {"image_sha1": sha1})
        save_tensors(out / "probes" / f"{sha1}.safetensors", {"probes": stacked[:, None].half()}, {"image_sha1": sha1})
        if i % 100 == 0:
            print(f"rank {rank}: {i}/{len(groups)} images {time.time() - t0:.0f}s", flush=True)
    return len(groups)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = load_config(args.config)
    rank, world, _ = S2T.dist_env()
    device = S2T.setup_device(cfg)
    S2T.require_d10(cfg, rank, world)
    gate = v1_gate(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    ckpt, s2_run = load_checkpoint(cfg, args.smoke)
    meta = ckpt["meta"]
    layer = meta["layer"]
    model, processor, extra = S2T.build_model(cfg, {"lora": meta["lora"]}, device, static)
    if extra["sel_id"] != meta["sel_id"]:
        raise ProtocolError("<SEL> token id differs from the checkpoint")
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    LR.load_trainable(model, extra["sel_emb"], sel, ckpt)
    model.eval()
    sel = sel.to(device).eval()
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    run = None
    if rank == 0:
        run = start_run(f"{S2T.s2_stage(cfg)}_eval", cfg, args.smoke, {
            "manifest_run": str(mrun), "s2_run": str(s2_run), "checkpoint": cfg["s2_eval"]["checkpoint"],
            "checkpoint_step": meta["step"], "layer": layer, "world_size": world})
    run = Path(S2T.broadcast(str(run) if run else None, world))
    out = run / "recomputed"
    src = S2T.Source(cfg, args.smoke, manifest, mhash, processor, extra["sel_id"], set())
    n = recompute_all(model, src, queries, wl, layer, device, out, cfg["s2_eval"]["batch_size"], rank, world)
    print(f"rank {rank}: recomputed {n} images", flush=True)
    if world > 1:
        dist.barrier()
    if rank == 0:
        store = RecomputedStore(out)
        metrics, per_query, tau = V2.evaluate(cfg, sel, store, props, queries, out, 0, layer, wl, device)
        per_query.to_csv(run / "eval_queries.csv", index=False)
        chains = V4.chain_inputs(cfg, sel, store, props, queries, out, 0, layer, device)
        chain_metrics = {"label": "dev", "split": "PACO-LVIS val supplied object->part chains (I1)",
                         "tau_rej_chain": tau, "seeds": cfg["v4"]["seeds"],
                         "probes": "recomputed with the S2 model; answer span teacher-forced with the GT referents"}
        chain_metrics.update(V4.search_metrics(cfg, chains, tau, run))
        write_json(run / "metrics.json", {
            "smoke": args.smoke, "layer": layer, "s2_run": s2_run.name, "checkpoint_step": meta["step"],
            "checkpoint_val": meta["val"], "single_step": metrics, "chains": chain_metrics,
            "features": "node features and probes recomputed with the LoRA model at layer l", "v1_gate": gate,
            "deviations": S2T.deviations(cfg)})
        print(run, "tau_rej", tau)
    if world > 1:
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
