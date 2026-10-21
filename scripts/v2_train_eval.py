import argparse
import json
import math
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from treeground import masks as M  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import tree as T  # noqa: E402
from treeground.cache import load_tensors  # noqa: E402
from treeground.calib import GRID, calibrate_rejection  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.identity import (manifest_inputs, open_features, open_proposals, open_queries,  # noqa: E402
                                 qwen_static)
from treeground.labels import paco_part_labels  # noqa: E402
from treeground.metrics import describe, mean_ci  # noqa: E402
from treeground.run import (REPO, ProtocolError, canonical_hash, load_config, read_json, resolve_run,  # noqa: E402
                            start_run, v1_gate, write_json)
from treeground.selector import Selector, selection_loss  # noqa: E402


TEXT_ONLY = "textonly"


@dataclass
class QRec:
    qid: str
    kind: str
    probes: torch.Tensor
    steps: list[S.StepTarget]
    path: tuple[int, ...] | None = None


@dataclass
class ImageRec:
    sha1: str
    key: str
    z: torch.Tensor
    tree: T.Tree
    labels: np.ndarray
    parts: np.ndarray
    high: np.ndarray
    rles: list = field(default_factory=list)
    queries: list[QRec] = field(default_factory=list)
    absent: bool = False

    @property
    def n(self) -> int:
        return self.z.shape[0]


def chosen_layer(cfg: dict, smoke: bool) -> tuple[int, str]:
    if smoke:
        run = resolve_run(cfg, "v3", cfg["inputs"].get("v3", "latest"), True)
        doc = yaml.safe_load((run / "chosen_layer.yaml").read_text())
    else:
        doc = yaml.safe_load((REPO / "configs" / "chosen_layer.yaml").read_text())
        resolve_run(cfg, "v3", doc["v3_run"], False)
    if doc.get("smoke", False) != smoke:
        raise ProtocolError("chosen_layer.yaml smoke flag does not match the current mode")
    return int(doc["layer"]), doc["v3_run"]


class Probes:
    def __init__(self, qdir: Path, layer_index: int):
        self.qdir, self.li, self._cache = qdir, layer_index, {}

    def get(self, sha1: str, offset: int, n_sel: int) -> torch.Tensor:
        if sha1 not in self._cache:
            self._cache = {sha1: load_tensors(self.qdir / "probes" / f"{sha1}.safetensors", ["probes"])[0]["probes"]}
        return self._cache[sha1][offset:offset + n_sel, self.li].clone()


def image_record(sha1, key, store, props, feat_key, tau, img: paco.PacoImage | None) -> ImageRec:
    nodes = props.nodes(sha1, tau)
    z = store.load(sha1, [feat_key])[feat_key]
    if z.shape[0] != len(nodes["rles"]):
        raise ProtocolError(f"{key}: feature rows do not match proposal nodes")
    tree = T.build_tree(nodes["rles"])
    if img is not None:
        if (nodes["h"], nodes["w"]) != (img.h, img.w):
            raise ProtocolError(f"{key}: annotation size differs from proposal masks")
        lab = paco_part_labels(nodes["rles"], img)
        labels, parts, high = lab.label, lab.part, lab.high_overlap
    else:
        labels = parts = np.full(z.shape[0], "", dtype=object)
        high = np.zeros(z.shape[0], dtype=bool)
    return ImageRec(sha1, key, z, tree, labels, parts, high, nodes["rles"])


def attach_paco_queries(rec: ImageRec, img: paco.PacoImage, qrows: pd.DataFrame, probes: Probes, counts: dict
                        ) -> None:
    gts = paco.gt_rles(img)
    for q in qrows.itertuples():
        gt = json.loads(q.gt)
        step_ious = M.iou(rec.rles, [gts[a] for a in gt]).T if rec.n else np.zeros((len(gt), 0))
        if q.kind == "chain":
            path = T.max_iou_path(rec.tree, step_ious) if rec.n else None
            if path is None:
                counts["unreachable_chains"] += 1
                continue
            steps = S.chain_step_targets(path, step_ious)
        else:
            tgt = S.single_step_target(step_ious[0])
            if tgt is None:
                counts["proposal_failures"] += 1
                continue
            steps, path = [tgt], None
        rec.queries.append(QRec(q.qid, q.kind, probes.get(rec.sha1, q.offset, q.n_sel), steps, path))
        counts[f"queries_{q.kind}"] += 1


def build_train(cfg, smoke, manifest, mhash, wl, store, props, qdir, queries, layer, li):
    feat_key = f"hidden_L{layer}"
    tau = cfg["proposals"]["node_dedup"]
    probes = Probes(qdir, li)
    images = paco.load(cfg["data"]["paco_lvis"], "train",
                       {int(k.split(":")[1]) for k in wl[wl.role == "paco_train"].image_key})
    counts = {k: 0 for k in ["proposal_failures", "unreachable_chains", "queries_object", "queries_part",
                             "queries_chain", "images_without_queries", "absent_queries"]}
    qtrain = queries[(queries.split == "train") & queries.kind.isin(["object", "part", "chain"])]
    by_sha = {k: g for k, g in qtrain.groupby("sha1")}
    recs = []
    for row in wl[wl.role == "paco_train"].sort_values("image_key").itertuples():
        img = images[int(row.image_key.split(":")[1])]
        rec = image_record(row.sha1, row.image_key, store, props, feat_key, tau, img)
        if row.sha1 in by_sha:
            attach_paco_queries(rec, img, by_sha[row.sha1], probes, counts)
        if not rec.queries:
            counts["images_without_queries"] += 1
        recs.append(rec)
    absent = []
    for q in queries[(queries.kind == "absent") & (queries.split == "train")].sort_values("qid").itertuples():
        rec = image_record(q.sha1, q.image_key, store, props, feat_key, tau, None)
        rec.absent = True
        rec.queries = [QRec(q.qid, "absent", probes.get(q.sha1, q.offset, 1), [S.StepTarget(np.zeros(0, np.int64),
                                                                                     np.zeros(0, np.int64))])]
        absent.append(rec)
        counts["absent_queries"] += 1
    text = {r.label: probes.get(TEXT_ONLY, r.offset, 1)[0] for r in queries[queries.kind == "text"].itertuples()}
    return recs, absent, text, counts


def lr_lambda(warmup: int, total: int):
    def f(step: int) -> float:
        if step < warmup:
            return (step + 1) / warmup
        return 0.5 * (1 + math.cos(math.pi * (step - warmup) / max(1, total - warmup)))
    return f


def batch_terms(sel: Selector, batch: list[tuple[ImageRec, bool]], text_probes: dict, rng, device, gamma, stats
                ) -> list[torch.Tensor]:
    offsets = np.cumsum([0] + [r.n for r, _ in batch])
    z = torch.cat([r.z for r, _ in batch]).to(device)
    e = sel.nodes(z)
    n_all = int(offsets[-1])
    labels = np.concatenate([r.labels for r, _ in batch]) if batch else np.zeros(0, object)
    parts = np.concatenate([r.parts for r, _ in batch]) if batch else np.zeros(0, object)
    high = np.concatenate([r.high for r, _ in batch]) if batch else np.zeros(0, bool)
    terms = []

    def term(probe: torch.Tensor, pos: np.ndarray, neg: np.ndarray, kind: str) -> None:
        q = sel.queries(probe.to(device)[None])
        s = sel.scores(q, e)[0]
        loss, p, n = selection_loss(s[torch.as_tensor(pos, device=device, dtype=torch.long)],
                                    s[torch.as_tensor(neg, device=device, dtype=torch.long)], gamma)
        stats["terms_" + kind] += 1
        stats["empty_negatives"] += int(len(neg) == 0)
        terms.append(loss)

    def detached_scores(probe: torch.Tensor) -> np.ndarray:
        with torch.no_grad():
            return sel.scores(sel.queries(probe.to(device)[None]), e)[0].float().cpu().numpy()

    for j, (rec, as_text) in enumerate(batch):
        lo, hi = int(offsets[j]), int(offsets[j + 1])
        others = np.concatenate([np.arange(0, lo), np.arange(hi, n_all)]).astype(np.int64)
        if as_text:
            label = rng.choice(sorted(l for l in set(rec.labels) if l and l in text_probes))
            tgt = S.text_target(labels, parts, high, label)
            pool = S.eligible(np.arange(n_all), tgt.pos, tgt.ignore)
            draw = S.cross_image_negatives(pool, rng)
            stats["shortfall_cross"] += draw.shortfall
            term(text_probes[label], tgt.pos, draw.ids, "text")
            continue
        q = rec.queries[rng.integers(len(rec.queries))]
        for t, st in enumerate(q.steps):
            probe = q.probes[t]
            local = detached_scores(probe)[lo:hi]
            if q.kind == "chain" and t > 0:
                sub = rec.tree.strict_descendants(q.path[t - 1])
                inside = S.eligible(sub, st.pos, st.ignore)
                outside = S.eligible(np.setdiff1d(np.arange(rec.n), sub), st.pos, st.ignore)
                draw = S.chain_negatives(inside, outside, local, rng)
                stats["chain_topup"] += draw.stats.get("topup", 0)
            else:
                draw = S.in_image_negatives(S.eligible(np.arange(rec.n), st.pos, st.ignore), local, rng)
            stats["shortfall_in_image"] += draw.shortfall
            cross = S.cross_image_negatives(others, rng)
            stats["shortfall_cross"] += cross.shortfall
            neg = np.concatenate([draw.ids + lo, cross.ids])
            term(probe, st.pos + lo, neg, q.kind)
    return terms


def train(cfg, recs, absent, text_probes, d_node, d_query, run, device) -> Selector:
    c = cfg["v2"]
    torch.manual_seed(c["seed"])
    rng = np.random.default_rng(c["seed"])
    sel = Selector(d_node, d_query, c["temperature"]).to(device)
    opt = torch.optim.AdamW(sel.parameters(), lr=c["lr"], weight_decay=c["weight_decay"])
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lr_lambda(c["warmup_steps"], c["steps"]))
    items = [r for r in recs if r.queries] + absent
    if not items:
        raise ProtocolError("no training items")
    n_text = int(round(c["text_only_fraction"] * c["batch_images"]))
    order, ptr = rng.permutation(len(items)), 0
    log = []
    t0 = time.time()
    for step in range(c["steps"]):
        if ptr + c["batch_images"] > len(order):
            order, ptr = rng.permutation(len(items)), 0
        idx = order[ptr:ptr + c["batch_images"]]
        ptr += c["batch_images"]
        batch, text_left = [], n_text
        for i in idx:
            r = items[i]
            as_text = (text_left > 0 and not r.absent and any(l and l in text_probes for l in r.labels))
            text_left -= int(as_text)
            batch.append((r, as_text))
        stats = {k: 0 for k in ["terms_object", "terms_part", "terms_chain", "terms_text", "terms_absent",
                                "empty_negatives", "shortfall_in_image", "shortfall_cross", "chain_topup"]}
        terms = batch_terms(sel, batch, text_probes, rng, device, c["gamma"], stats)
        loss = torch.stack(terms).mean()
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        sched.step()
        if step % c["log_every"] == 0 or step == c["steps"] - 1:
            log.append({"step": step, "loss": loss.item(), "lr": sched.get_last_lr()[0], "terms": len(terms),
                        "seconds": time.time() - t0, **stats})
    pd.DataFrame(log).to_csv(run / "train_curve.csv", index=False)
    return sel


@torch.no_grad()
def evaluate(cfg, sel, store, props, queries, qdir, li, layer, wl, device) -> tuple[dict, pd.DataFrame, float]:
    feat_key = f"hidden_L{layer}"
    tau = cfg["proposals"]["node_dedup"]
    probes = Probes(qdir, li)
    images = paco.load(cfg["data"]["paco_lvis"], "val")
    rows = []
    qval = queries[(queries.split == "val") & queries.kind.isin(["object", "part"])]
    keys = dict(zip(wl.sha1, wl.image_key))
    for sha1, g in qval.groupby("sha1"):
        key = g.image_key.iloc[0]
        img = images[int(key.split(":")[1])]
        nodes = props.nodes(sha1, tau)
        z = store.load(sha1, [feat_key])[feat_key]
        e = sel.nodes(z.to(device)) if len(z) else None
        gts = paco.gt_rles(img)
        for q in g.itertuples():
            gt = json.loads(q.gt)
            ious = M.iou(nodes["rles"], [gts[gt[0]]])[:, 0] if nodes["rles"] else np.zeros(0)
            if e is None:
                rows.append({"qid": q.qid, "kind": q.kind, "image": key, "present": True, "max_sigma": -np.inf,
                             "sel_iou": 0.0, "best_iou": 0.0, "is_best": False})
                continue
            sig = torch.sigmoid(sel.scores(sel.queries(probes.get(sha1, q.offset, 1).to(device)), e)[0]).cpu().numpy()
            a = int(np.argmax(sig))
            b = int(np.argmax(ious))
            rows.append({"qid": q.qid, "kind": q.kind, "image": key, "present": True, "max_sigma": float(sig[a]),
                         "sel_iou": float(ious[a]), "best_iou": float(ious[b]), "is_best": a == b})
    for q in queries[(queries.kind == "absent") & (queries.split == "val")].itertuples():
        z = store.load(q.sha1, [feat_key])[feat_key]
        if len(z):
            sig = torch.sigmoid(sel.scores(sel.queries(probes.get(q.sha1, q.offset, 1).to(device)),
                                           sel.nodes(z.to(device)))[0]).cpu().numpy()
            m = float(sig.max())
        else:
            m = -np.inf
        rows.append({"qid": q.qid, "kind": "absent", "image": keys.get(q.sha1, q.image_key), "present": False,
                     "max_sigma": m, "sel_iou": 0.0, "best_iou": 0.0, "is_best": False})
    df = pd.DataFrame(rows)
    pres, absn = df[df.present], df[~df.present]
    tau_rej, curve = calibrate_rejection(pres.max_sigma.to_numpy(), pres.sel_iou.to_numpy(),
                                         absn.max_sigma.to_numpy(), cfg["v2"]["calib_lambda"])
    df["accept"] = df.max_sigma >= tau_rej
    df["e2e_iou"] = np.where(df.present, np.where(df.accept, df.sel_iou, 0.0), np.where(df.accept, 0.0, 1.0))
    bs = cfg["v2"]["bootstrap"]
    pres = df[df.present]
    covered = pres[pres.best_iou >= 0.5]

    def ci(frame, col):
        return mean_ci(frame[col].astype(float).to_numpy(), frame.image.to_numpy(), bs["n"], bs["seed"])

    metrics = {
        "label": "dev", "split": "PACO-LVIS val (+ gRefCOCO val no-target for calibration)",
        "tau_rej": tau_rej, "calibration_grid": [float(GRID[0]), float(GRID[-1]), len(GRID)],
        "calibration_objective_max": float(curve.max()),
        "i_best_achievable_iou": {"all": describe(pres.best_iou), "coverage_at_0.5": float((pres.best_iou >= 0.5).mean()),
                                  **{k: describe(g.best_iou) for k, g in pres.groupby("kind")}},
        "ii_selector_success": {"all": ci(covered.assign(s=covered.sel_iou >= 0.5), "s"),
                                **{k: ci(g.assign(s=g.sel_iou >= 0.5), "s") for k, g in covered.groupby("kind")},
                                "gIoU_selected": ci(covered, "sel_iou"),
                                "conditioned_on": "a proposal with IoU >= 0.5 exists; argmax over C_single"},
        "diagnostic_best_proposal_identification": ci(covered.assign(s=covered.is_best), "s"),
        "iii_end_to_end_gIoU": {"present": ci(pres, "e2e_iou"),
                                "present_and_no_target": ci(df, "e2e_iou"),
                                "no_target_abstention_rate": float((~df[~df.present].accept).mean())
                                if (~df.present).any() else None},
        "proposal_failures_eval": int((pres.best_iou < 0.5).sum()),
        "queries": {"present": int(len(pres)), "no_target": int((~df.present).sum())},
    }
    return metrics, df, tau_rej


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    device = cfg["device"]
    gate = v1_gate(cfg, args.smoke)
    v1_run = gate["v1_run"]
    layer, v3_run = chosen_layer(cfg, args.smoke)
    mrun, manifest, mhash, wl = manifest_inputs(cfg, args.smoke)
    static = qwen_static(cfg)
    if layer not in static["layers"]:
        raise ProtocolError(f"chosen layer {layer} is not a cached candidate layer")
    li = static["layers"].index(layer)
    store = open_features(cfg, args.smoke, manifest, mhash, static)
    props = open_proposals(cfg, args.smoke, manifest, mhash, capped=True)
    qdir, queries = open_queries(cfg, args.smoke, mhash, static)
    run = start_run("v2", cfg, args.smoke, {
        "manifest_run": str(mrun), "manifest_sha256": mhash, "features": str(store.path),
        "features_identity": store.identity_hash, "queries": str(qdir),
        "queries_identity": read_json(qdir / "identity.json")["hash"], "layer": layer, "v3_run": v3_run,
        "v1_run": v1_run})
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    recs, absent, text_probes, counts = build_train(cfg, args.smoke, manifest, mhash, wl, store, props, qdir, queries,
                                                    layer, li)
    d_node = recs[0].z.shape[1]
    d_query = next(iter(text_probes.values())).shape[0]
    sel = train(cfg, recs, absent, text_probes, d_node, d_query, run, device)
    torch.save({"state_dict": sel.state_dict(), "d_node": d_node, "d_query": d_query, "layer": layer,
                "temperature": sel.tau, "smoke": args.smoke, "features_identity": store.identity_hash,
                "config_hash": canonical_hash(cfg["v2"])}, run / "selector.pt")
    sel.eval()
    metrics, per_query, tau_rej = evaluate(cfg, sel, store, props, queries, qdir, li, layer, wl, device)
    per_query.to_csv(run / "eval_queries.csv", index=False)
    metrics.update({"smoke": args.smoke, "layer": layer, "training_exclusions": counts,
                    "training_images": len(recs), "training_absent_queries": len(absent), "v1_gate": gate})
    write_json(run / "metrics.json", metrics)
    print(run, "tau_rej", tau_rej)


if __name__ == "__main__":
    main()
