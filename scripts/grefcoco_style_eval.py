import argparse
import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

from treeground import masks as M  # noqa: E402
from treeground.calib import calibrate_rejection  # noqa: E402
from treeground.data import grefcoco  # noqa: E402
from treeground.metrics import mean_ci  # noqa: E402
from treeground.run import (ProtocolError, assert_real_path, canonical_hash, load_config, read_json,  # noqa: E402
                            require_approved, resolve_run, start_run, write_json)

# Deviation D27: style-matched abstention on gRefCOCO val. Present = gRefCOCO val expressions with targets (GT = union
# of the target masks), no-target = gRefCOCO val no-target expressions, on the manifest's gRefCOCO val images that
# carry D3/D4 nodes (worklist role grefcoco_val_nt); both classes are free-form expressions, so query style cannot
# separate them. Ours: single step over the capped nodes, probe '{expression} <SEL>' (frozen Qwen for V2/S1 with
# the cached node features; S2 stages recompute features and probes with the LoRA model as s2_eval, answer
# '{expression} <SEL>' for both classes). Baselines: the D13 predictors on the expression. Reported at the stage's
# own tau_rej and at a tau calibrated (same objective) on the other half of the images (2-fold by image), plus the
# ROC/AUC of the score separating present from no-target, beside the same AUC on the PACO-template evaluation.

OURS_FROZEN = ("v2", "s1")
OURS_S2 = ("s2", "s2_from_s1", "s2_from_s1_long")
BASELINES = ("sam3_text", "qwen_box_sam", "sa2va")
METHODS = OURS_FROZEN + OURS_S2 + BASELINES
COLUMNS = ["qid", "image", "present", "n_targets", "n_nodes", "score", "cand_iou", "best_iou", "native_accept"]


def method_cfg(cfg: dict, method: str) -> dict:
    stage = "s1" if method in ("s1", "s2_from_s1", "s2_from_s1_long") else "v2"
    out = {**cfg, "inputs": {**cfg.get("inputs", {}), "selector_stage": stage}}
    if method == "s2_from_s1_long":
        out["s2"] = {**cfg.get("s2", {}), "stage_suffix": "_long"}
    return out


def approvals(method: str) -> list[str]:
    need = {"v2": [], "s1": ["D11"], "s2": ["D10"], "s2_from_s1": ["D10", "D11"],
            "s2_from_s1_long": ["D10", "D11", "D21"]}.get(method, ["D13"])
    return ["D27"] + need


def build_queries(cfg: dict, wl: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    split = cfg["grefcoco_style"]["split"]
    refs, images = grefcoco.load(cfg["data"]["grefcoco"])
    rows_wl = wl[wl.role == f"grefcoco_{split}_nt"].drop_duplicates("image_key")
    sha1, path = dict(zip(rows_wl.image_key, rows_wl.sha1)), dict(zip(rows_wl.image_key, rows_wl.path))
    rows, skipped = [], {"queries": 0, "images": set()}
    for ref in refs:
        if ref["split"] != split:
            continue
        key = f"grefcoco:{ref['image_id']}"
        absent = grefcoco.is_no_target(ref)
        ann = [] if absent else (ref["ann_id"] if isinstance(ref["ann_id"], list) else [ref["ann_id"]])
        for s in ref["sentences"]:
            if key not in sha1:
                skipped["queries"] += 1
                skipped["images"].add(key)
                continue
            rows.append({"qid": f"grefcoco|{split}|{s['sent_id']}", "image": key, "sha1": sha1[key], "path": path[key],
                         "present": not absent, "text": s["sent"].strip(), "ann_ids": json.dumps(ann),
                         "n_targets": len(ann), "h": images[ref["image_id"]]["height"],
                         "w": images[ref["image_id"]]["width"]})
    df = pd.DataFrame(rows).sort_values(["image", "qid"]).reset_index(drop=True)
    if df.qid.duplicated().any():
        raise ProtocolError("duplicate gRefCOCO sentence ids")
    return df, {"queries_without_nodes": skipped["queries"], "images_without_nodes": len(skipped["images"])}


def gt_union(by_ann: dict, ann_ids: list[int], h: int, w: int) -> M.Rle:
    m = np.zeros((h, w), bool)
    for a in ann_ids:
        m |= M.decode(M.from_annotation(by_ann[a]["segmentation"], h, w)).astype(bool)
    return M.encode(m[None])[0]


def image_keys(df: pd.DataFrame, shard: int, num_shards: int, smoke: bool, n_smoke: int) -> list[str]:
    keys = sorted(df.image.unique())
    return keys[:n_smoke] if smoke else keys[shard::num_shards]


def run_ref(cfg: dict, method: str) -> tuple[dict, float]:
    # identity of the evaluated model and its own tau_rej
    if method in OURS_FROZEN:
        run = resolve_run(cfg, method, cfg["inputs"].get(method, "latest"), False)
        return {"run": f"{method}/{run.name}"}, float(read_json(run / "metrics.json")["tau_rej"])
    if method in OURS_S2:
        import x_transfer_s2 as XS2
        stage, s2_run, ev, _ = XS2.s2_inputs(cfg)
        if stage != method:
            raise ProtocolError(f"config resolves S2 stage {stage}, expected {method}")
        tau = float(read_json(ev / "metrics.json")["single_step"]["tau_rej"])
        return {"run": f"{stage}/{s2_run.name}", "eval_run": f"{stage}_eval/{ev.name}",
                "checkpoint": cfg["s2_eval"]["checkpoint"]}, tau
    return {"settings": cfg["baselines"][method]}, float("nan")


def identity(cfg: dict, method: str, mhash: str, smoke: bool, ref: dict) -> dict:
    h = hashlib.sha256()
    for rel in ("scripts/grefcoco_style_eval.py", "scripts/s2_train.py", "scripts/b_baselines.py",
                "treeground/data/grefcoco.py"):
        h.update((ROOT / rel).read_bytes())
    used = {"sam3_text": ["sam3"], "qwen_box_sam": ["qwen", "sam3"], "sa2va": ["sa2va"]}.get(method, ["qwen"])
    return {"kind": "grefcoco_style", "method": method, "smoke": smoke, "manifest": mhash, "model": ref,
            "models": {k: cfg["models"][k] for k in used}, "settings": cfg["grefcoco_style"],
            "proposals": {k: cfg["proposals"][k] for k in ("node_dedup", "dtype")}, "code": h.hexdigest()}


def out_dir(cfg: dict, ident: dict) -> Path:
    return (Path(cfg["cache_root"]) / ("smoke" if ident["smoke"] else "full") / "grefcoco_style" / ident["method"]
            / canonical_hash(ident)[:16])


def ours_frozen(cfg: dict, method: str, manifest, mhash):
    import torch
    from PIL import Image
    from treeground import mllm
    from treeground.data.images import load_rgb
    from treeground.identity import open_features, open_proposals, qwen_static
    from treeground.selector import Selector
    device = cfg["device"]
    run = resolve_run(cfg, method, cfg["inputs"].get(method, "latest"), False)
    ckpt = torch.load(run / "selector.pt", map_location="cpu")
    if ckpt["smoke"]:
        raise ProtocolError(f"{method} selector is a smoke checkpoint")
    sel = Selector(ckpt["d_node"], ckpt["d_query"], ckpt["temperature"])
    sel.load_state_dict(ckpt["state_dict"])
    sel = sel.to(device).eval()
    layer = int(ckpt["layer"])
    static = qwen_static(cfg)
    store = open_features(cfg, False, manifest, mhash, static)
    props = open_proposals(cfg, False, manifest, mhash, capped=True)
    q = cfg["models"]["qwen"]
    qwen, qproc = mllm.load_qwen(q["repo"], q["revision"], device, cfg["features"]["dtype"], cfg.get("hf_home"))
    sel_id, _ = mllm.add_sel_token(qwen, qproc)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    @torch.no_grad()
    def score(g: pd.DataFrame):
        sha1 = g.sha1.iloc[0]
        nodes = props.nodes(sha1, cfg["proposals"]["node_dedup"])
        z = store.load(sha1, [f"hidden_L{layer}"])[f"hidden_L{layer}"]
        if z.shape[0] != len(nodes["rles"]):
            raise ProtocolError(f"{sha1}: feature rows do not match proposal nodes")
        if not len(z):
            return nodes, [np.zeros(0)] * len(g)
        e = sel.nodes(z.to(device))
        image = Image.fromarray(load_rgb(assert_real_path(g.path.iloc[0])))
        probes = mllm.probe_states(qwen, qproc, image, [f"{t} {mllm.SEL}" for t in g.text], [layer], sel_id,
                                   cfg["grefcoco_style"]["batch_size"])
        # fp16-rounded like the s0_queries probe cache the selector was evaluated on
        return nodes, [torch.sigmoid(sel.scores(sel.queries(torch.from_numpy(p[:, 0].astype(np.float16)).to(device)),
                                                e)[0]).cpu().numpy() for p in probes]
    return score


def ours_s2(cfg: dict, manifest, mhash):
    import torch
    import s2_eval as S2E
    import s2_train as S2T
    import x_transfer_s2 as XS2
    from treeground import lora as LR
    from treeground import s2 as S2
    from treeground.identity import qwen_static
    from treeground.mllm import SEL
    from treeground.selector import Selector
    device = cfg["device"]
    _, s2_run, ev, _ = XS2.s2_inputs(cfg)
    ckpt, ck_run = S2E.load_checkpoint(cfg, False)
    if ck_run != s2_run or read_json(ev / "metrics.json")["checkpoint_step"] != ckpt["meta"]["step"]:
        raise ProtocolError(f"checkpoint {ck_run.name} step {ckpt['meta']['step']} is not the one {ev.name} evaluated")
    meta = ckpt["meta"]
    model, processor, extra = S2T.build_model(cfg, {"lora": meta["lora"]}, device, qwen_static(cfg))
    if extra["sel_id"] != meta["sel_id"]:
        raise ProtocolError("<SEL> token id differs from the checkpoint")
    sel = Selector(meta["d_node"], meta["d_query"], meta["temperature"])
    LR.load_trainable(model, extra["sel_emb"], sel, ckpt)
    model.eval()
    sel = sel.to(device).eval()
    layer = int(meta["layer"])
    del ckpt
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    src = S2T.Source(cfg, False, manifest, mhash, processor, extra["sel_id"], set())

    @torch.no_grad()
    def score(g: pd.DataFrame):
        sha1 = g.sha1.iloc[0]
        nodes = src.props.nodes(sha1, src.tau)
        if not nodes["rles"]:
            return nodes, [np.zeros(0)] * len(g)
        # both classes answered with the expression itself, as D10/D21 train no-target and referring expressions
        qs = [S2T.Query(q.qid, "absent", S2.user_text(q.text), f"{S2.user_text(q.text)} {SEL}", [])
              for q in g.itertuples()]
        m = S2T.ImageMeta(sha1, g.image.iloc[0], g.path.iloc[0], len(nodes["rles"]), None, None, None, None, qs)
        z, probes = S2T.recompute(model, src, m, qs, layer, device, cfg["grefcoco_style"]["batch_size"])
        e = sel.nodes(z.half().to(device))
        return nodes, [torch.sigmoid(sel.scores(sel.queries(p.half()), e)[0]).cpu().numpy() for p in probes]
    return score


def ours_rows(g: pd.DataFrame, nodes: dict, sigs: list, gts: dict, tau: float) -> list[dict]:
    rows = []
    for q, sig in zip(g.itertuples(), sigs):
        ious = M.iou(nodes["rles"], [gts[q.qid]])[:, 0] if q.present and nodes["rles"] else np.zeros(len(sig))
        a = int(np.argmax(sig)) if len(sig) else None
        s = float(sig[a]) if a is not None else -np.inf
        rows.append({"qid": q.qid, "image": q.image, "present": q.present, "n_targets": q.n_targets,
                     "n_nodes": len(sig), "score": s, "cand_iou": float(ious[a]) if a is not None else 0.0,
                     "best_iou": float(ious.max()) if len(ious) else 0.0, "native_accept": s >= tau})
    return rows


def baseline_rows(g: pd.DataFrame, preds: list, gts: dict, shape: tuple) -> list[dict]:
    rows = []
    for q, (mask, sc, _, accepted) in zip(g.itertuples(), preds):
        if mask is not None and mask.shape != shape:
            raise ProtocolError(f"{q.qid}: mask {mask.shape} differs from the image {shape}")
        rle = M.encode(mask)[0] if mask is not None and mask.any() else None
        iou = 0.0
        if q.present and rle is not None:
            iou = float(M.iou([rle], [gts[q.qid]])[0, 0])
        # a candidate that is empty cannot be accepted, whatever its score
        rows.append({"qid": q.qid, "image": q.image, "present": q.present, "n_targets": q.n_targets,
                     "n_nodes": int(rle is not None), "score": float(sc) if rle is not None else -np.inf,
                     "cand_iou": iou, "best_iou": np.nan, "native_accept": bool(accepted) and rle is not None})
    return rows


def run_shard(cfg: dict, method: str, shard: int, num_shards: int, smoke: bool) -> None:
    from treeground.data.images import load_rgb
    from treeground.identity import manifest_inputs
    require_approved(cfg, approvals(method))
    mrun, manifest, mhash, wl = manifest_inputs(cfg, False)
    ref, tau = run_ref(cfg, method)
    ident = identity(cfg, method, mhash, smoke, ref)
    d = out_dir(cfg, ident)
    target = d / "shards" / f"{shard:05d}.parquet"
    if target.exists():
        print(f"shard {shard} already done: {target}")
        return
    (d / "shards").mkdir(parents=True, exist_ok=True)
    if not (d / "identity.json").exists():
        write_json(d / "identity.json", {"identity": ident, "hash": canonical_hash(ident)})
    df, _ = build_queries(cfg, wl)
    keys = image_keys(df, shard, num_shards, smoke, cfg["grefcoco_style"]["smoke_images"])
    run = start_run(f"grefcoco_style_{method}", cfg, smoke, {"shard": shard, "num_shards": num_shards,
                                                              "cache": str(d), "manifest_run": str(mrun), **ref})
    by_ann = {a["id"]: a for a in json.loads((Path(cfg["data"]["grefcoco"]) / "instances.json")
                                             .read_text())["annotations"]}
    if method in OURS_FROZEN:
        score = ours_frozen(cfg, method, manifest, mhash)
    elif method in OURS_S2:
        score = ours_s2(cfg, manifest, mhash)
    else:
        import b_baselines as B
        predict = B.PREDICTORS[method](cfg)
        if method == "sam3_text":
            predict = fit_text(predict, B.concept)
    rows, t0 = [], time.time()
    for n, key in enumerate(keys, 1):
        g = df[df.image == key]
        gts = {q.qid: gt_union(by_ann, json.loads(q.ann_ids), q.h, q.w) for q in g.itertuples() if q.present}
        if method in BASELINES:
            image = load_rgb(assert_real_path(g.path.iloc[0]))
            if image.shape[:2] != (g.h.iloc[0], g.w.iloc[0]):
                raise ProtocolError(f"{key}: image {image.shape[:2]} differs from the annotation")
            rows += baseline_rows(g, predict(image, g.text.tolist()), gts, image.shape[:2])
        else:
            nodes, sigs = score(g)
            if (nodes["h"], nodes["w"]) != (g.h.iloc[0], g.w.iloc[0]):
                raise ProtocolError(f"{key}: proposal masks differ from the annotation size")
            rows += ours_rows(g, nodes, sigs, gts, tau)
        print(f"[{n}/{len(keys)}] {key} queries {len(g)} {time.time() - t0:.0f}s", flush=True)
    tmp = target.with_suffix(".tmp")
    pd.DataFrame(rows, columns=COLUMNS).to_parquet(tmp, index=False)
    tmp.replace(target)
    write_json(run / "metrics.json", {"images": len(keys), "queries": len(rows), "seconds": time.time() - t0})


def fit_text(predict, concept):
    # SAM 3's CLIP text encoder takes 32 positions; longer style-matched phrases keep their leading words
    tok = predict.proc.tokenizer
    limit = min(tok.model_max_length, 32)

    def clip(p: str) -> str:
        words = p.split()
        while len(words) > 1 and len(tok(concept(" ".join(words))).input_ids) > limit:
            words.pop()
        return " ".join(words)

    return lambda image, phrases: predict(image, [clip(p) for p in phrases])


def auc(pos: np.ndarray, neg: np.ndarray) -> float | None:
    # Mann-Whitney: P(score_present > score_no_target) + 0.5 P(tie)
    pos, neg = np.asarray(pos, float), np.asarray(neg, float)
    if not len(pos) or not len(neg):
        return None
    r = rankdata(np.concatenate([pos, neg]))
    return float((r[:len(pos)].sum() - len(pos) * (len(pos) + 1) / 2) / (len(pos) * len(neg)))


def roc(pos: np.ndarray, neg: np.ndarray) -> pd.DataFrame:
    s = np.concatenate([pos, neg]).astype(float)
    y = np.concatenate([np.ones(len(pos)), np.zeros(len(neg))])
    o = np.argsort(-s, kind="stable")
    s, y = s[o], y[o]
    last = np.r_[np.flatnonzero(np.diff(s) != 0), len(s) - 1]
    tp, fp = np.cumsum(y)[last], np.cumsum(1 - y)[last]
    return pd.DataFrame({"threshold": s[last], "tpr": tp / max(1, len(pos)), "fpr": fp / max(1, len(neg))})


def image_folds(images, k: int, seed: int) -> dict[str, int]:
    uniq = sorted(set(images))
    perm = np.random.default_rng(seed).permutation(len(uniq))
    return {uniq[i]: int(j % k) for j, i in enumerate(perm)}


def block(df: pd.DataFrame, accept: np.ndarray, bs: dict) -> dict:
    accept = np.asarray(accept, bool)
    pres, absn = df.present.to_numpy(bool), ~df.present.to_numpy(bool)
    iou = np.where(accept, df.cand_iou.to_numpy(float), 0.0)
    img = df.image.to_numpy()
    out = {"present": int(pres.sum()), "no_target": int(absn.sum()), "images": int(df.image.nunique()),
           "no_target_abstention_rate": float((~accept[absn]).mean()) if absn.any() else None,
           "present_acceptance_rate": float(accept[pres].mean()) if pres.any() else None,
           "present_gIoU": mean_ci(iou[pres], img[pres], bs["n"], bs["seed"]),
           "present_success@0.5": mean_ci((iou[pres] >= 0.5).astype(float), img[pres], bs["n"], bs["seed"])}
    for name, sel in (("single_target", pres & (df.n_targets.to_numpy() == 1)),
                      ("multi_target", pres & (df.n_targets.to_numpy() > 1))):
        out[f"present_gIoU_{name}"] = mean_ci(iou[sel], img[sel], bs["n"], bs["seed"]) if sel.any() else None
    # end-to-end over both classes (no-target scores 1 when abstaining), as V2's present_and_no_target
    e2e = np.where(pres, iou, (~accept).astype(float))
    out["gIoU_present_and_no_target"] = mean_ci(e2e, img, bs["n"], bs["seed"])
    return out


def two_fold(df: pd.DataFrame, lam: float, bs: dict, k: int, seed: int) -> dict:
    fold = df.image.map(image_folds(df.image, k, seed)).to_numpy()
    s = df.score.to_numpy(float)
    taus, per_fold = np.zeros(len(df)), {}
    for f in range(k):
        cal = df[fold != f]
        tau, _ = calibrate_rejection(cal.score[cal.present].to_numpy(float), cal.cand_iou[cal.present].to_numpy(float),
                                     cal.score[~cal.present].to_numpy(float), lam)
        taus[fold == f] = tau
        test = df[fold == f]
        per_fold[f"fold{f}"] = {"tau_calibrated_on_other_folds": tau, **block(test, s[fold == f] >= tau, bs)}
    return {"folds": per_fold, "pooled_held_out": block(df, s >= taus, bs), "k": k, "seed": seed}


def separation(df: pd.DataFrame) -> dict:
    pres, absn = df[df.present], df[~df.present]
    return {"auc": auc(pres.score, absn.score),
            "auc_single_target": auc(pres.score[pres.n_targets == 1], absn.score),
            "auc_multi_target": auc(pres.score[pres.n_targets > 1], absn.score)}


def summarize(df: pd.DataFrame, tau: float, scored: bool, lam: float, bs: dict, k: int, seed: int) -> dict:
    out = {"stage_tau": tau if np.isfinite(tau) else None, "at_stage_tau": block(df, df.native_accept, bs),
           "no_rejection": block(df, df.n_nodes.to_numpy() > 0, bs) if scored else None}
    out["two_fold_calibrated"] = two_fold(df, lam, bs, k, seed) if scored else None
    out["separation"] = separation(df) if scored else None
    return out


def paco_template(cfg: dict, method: str, tau: float) -> dict | None:
    # the original evaluation's present (PACO I1 templates) vs no-target (gRefCOCO val) separation, for contrast
    try:
        if method in OURS_FROZEN:
            run = resolve_run(cfg, method, cfg["inputs"].get(method, "latest"), False)
        elif method in OURS_S2:
            import x_transfer_s2 as XS2
            run = XS2.s2_inputs(cfg)[2]
        else:
            run = resolve_run(cfg, f"b_{method}_merge", "latest", False)
    except ProtocolError as e:
        return {"not available": str(e)}
    q = pd.read_csv(run / ("queries.csv" if method in BASELINES else "eval_queries.csv"))
    if method in BASELINES:
        p = q.present.to_numpy(bool)
        # the published rows keep no unthresholded mask area, so the raw score is used (empty masks not excluded)
        s = q.score.to_numpy(float) if method == "sam3_text" else None
        return {"run": run.name, "auc": auc(s[p], s[~p]) if s is not None else None,
                "no_target_abstention_rate": float((q[~q.present].pred_area == 0).mean()),
                "present_acceptance_rate": float((q[q.present].pred_area > 0).mean())}
    q = q[q.kind.isin(["object", "part", "absent"])]
    return {"run": run.name, "auc": auc(q.max_sigma[q.present], q.max_sigma[~q.present]),
            "no_target_abstention_rate": float((q.max_sigma[~q.present] < tau).mean()),
            "present_acceptance_rate": float((q.max_sigma[q.present] >= tau).mean()),
            "absent_scores": q[~q.present].set_index("qid").max_sigma}


def merge(cfg: dict, method: str, num_shards: int, smoke: bool) -> None:
    from treeground.identity import manifest_inputs
    require_approved(cfg, approvals(method))
    mrun, _, mhash, wl = manifest_inputs(cfg, False)
    ref, tau = run_ref(cfg, method)
    ident = identity(cfg, method, mhash, smoke, ref)
    d = out_dir(cfg, ident)
    missing = [k for k in range(num_shards) if not (d / "shards" / f"{k:05d}.parquet").exists()]
    if missing:
        raise ProtocolError(f"{len(missing)} {method} shards missing under {d}, e.g. {missing[:5]}")
    df = pd.concat([pd.read_parquet(d / "shards" / f"{k:05d}.parquet") for k in range(num_shards)],
                   ignore_index=True)
    queries, skipped = build_queries(cfg, wl)
    expected = set(queries[queries.image.isin(image_keys(queries, 0, 1, smoke,
                                                         cfg["grefcoco_style"]["smoke_images"]))].qid)
    if set(df.qid) != expected or df.qid.duplicated().any():
        raise ProtocolError(f"{method}: merged queries do not match the evaluation set ({len(df)} vs {len(expected)})")
    run = start_run(f"grefcoco_style_{method}_merge", cfg, smoke, {"cache": str(d), "manifest_run": str(mrun), **ref})
    c, bs = cfg["grefcoco_style"], cfg["v2"]["bootstrap"]
    scored = method not in ("qwen_box_sam", "sa2va")
    out = {"method": method, "deviation": "D27", "label": "dev", "smoke": smoke, "model": ref,
           "split": f"gRefCOCO {c['split']}: present expressions (GT = union of targets) vs no-target expressions",
           "skipped": skipped, **summarize(df, tau, scored, cfg["v2"]["calib_lambda"], bs, c["folds"], c["fold_seed"])}
    pt = paco_template(cfg, method, tau)
    if pt is not None and "absent_scores" in pt:
        a = pt.pop("absent_scores")
        mine = df[~df.present].set_index("qid").score
        shared = mine.index.intersection(a.index)
        diff = (mine[shared] - a[shared]).abs()
        pt["absent_score_check"] = {"shared_no_target": int(len(shared)),
                                    "max_abs_diff": float(diff.max()) if len(diff) else None,
                                    "median_abs_diff": float(diff.median()) if len(diff) else None}
    out["paco_template_reference"] = pt
    if scored:
        roc(df.score[df.present].to_numpy(), df.score[~df.present].to_numpy()).to_csv(run / "roc.csv", index=False)
    df.to_csv(run / "queries.csv", index=False)
    write_json(run / "metrics.json", out)
    print(json.dumps({k: out[k] for k in ("separation", "stage_tau")}, indent=1, default=str))
    print(run)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    cfg = method_cfg(load_config(args.config), args.method)
    num_shards = 1 if args.smoke else cfg["grefcoco_style"]["num_shards"][
        "baseline" if args.method in BASELINES else "ours"]
    if args.merge:
        merge(cfg, args.method, num_shards, args.smoke)
    else:
        if not 0 <= args.shard < num_shards:
            raise ProtocolError(f"shard {args.shard} outside 0..{num_shards - 1}")
        run_shard(cfg, args.method, args.shard, num_shards, args.smoke)


if __name__ == "__main__":
    main()
