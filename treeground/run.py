import datetime as dt
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
from importlib import metadata
from pathlib import Path
from typing import Any

import yaml

REPO = Path(__file__).resolve().parents[1]
PACKAGES = [
    "torch", "torchvision", "transformers", "accelerate", "huggingface_hub", "tokenizers", "pycocotools",
    "opencv-python-headless", "safetensors", "numpy", "scipy", "pyyaml", "pyarrow", "pandas", "pillow",
    "qwen-vl-utils", "diffusers", "lpips",
]
_ENV = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")


class ProtocolError(RuntimeError):
    pass


def _expand(value: Any) -> Any:
    if isinstance(value, dict):
        return {k: _expand(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand(v) for v in value]
    if isinstance(value, str):
        def sub(m: re.Match) -> str:
            name = m.group(1)
            if name not in os.environ:
                raise ProtocolError(f"environment variable {name} is not set (needed by config value {value!r})")
            return os.environ[name]
        return _ENV.sub(sub, value)
    return value


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for k, v in over.items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


def load_config(path: str | Path) -> dict:
    path = Path(path)
    raw = yaml.safe_load(path.read_text()) or {}
    for inc in reversed(_as_list(raw.pop("include", []))):
        raw = _merge(load_config(path.parent / inc), raw)
    return _expand(raw)


def _as_list(x: Any) -> list:
    return x if isinstance(x, list) else [x]


def canonical_hash(obj: Any) -> str:
    return hashlib.sha256(json.dumps(obj, sort_keys=True, default=str).encode()).hexdigest()


def file_hash(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def git_state() -> dict:
    def run(*args: str) -> str:
        try:
            return subprocess.run(["git", *args], cwd=REPO, capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            return ""
    return {"rev": run("rev-parse", "HEAD"), "dirty": bool(run("status", "--porcelain", "--untracked-files=no"))}


def package_versions() -> dict:
    out = {}
    for name in PACKAGES:
        try:
            out[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            out[name] = None
    return out


def runs_root(cfg: dict) -> Path:
    return Path(cfg["runs_root"])


def start_run(stage: str, cfg: dict, smoke: bool, extra: dict | None = None) -> Path:
    root = runs_root(cfg) / "smoke" / stage if smoke else runs_root(cfg) / stage
    git = git_state()
    root.mkdir(parents=True, exist_ok=True)
    while True:
        stamp = dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
        run_dir = root / f"{stamp}_{(git['rev'] or 'nogit')[:8]}"
        try:
            run_dir.mkdir(exist_ok=False)
            break
        except FileExistsError:
            continue
    (run_dir / "config.yaml").write_text(yaml.safe_dump(cfg, sort_keys=True))
    meta = {
        "stage": stage,
        "smoke": smoke,
        "git": git,
        "packages": package_versions(),
        "python": sys.version,
        "host": platform.node(),
        "argv": sys.argv,
        "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
        "slurm_array_task_id": os.environ.get("SLURM_ARRAY_TASK_ID"),
        "models": cfg.get("models"),
        "seed": cfg.get("seed"),
        "started_utc": stamp,
    }
    meta.update(extra or {})
    write_json(run_dir / "meta.json", meta)
    ensure_deviation_log(cfg)
    return run_dir


def update_meta(run_dir: Path, **fields: Any) -> None:
    meta = json.loads((run_dir / "meta.json").read_text())
    meta.update(fields)
    write_json(run_dir / "meta.json", meta)


def write_json(path: str | Path, obj: Any) -> None:
    # atomic: concurrent array tasks may read the file while another writes it
    path = Path(path)
    tmp = path.with_name(f".{path.name}.{os.getpid()}.{platform.node()}.tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, default=_json_default))
    tmp.replace(path)


def _json_default(o: Any) -> Any:
    import numpy as np
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    raise TypeError(type(o))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text())


def resolve_run(cfg: dict, stage: str, ref: str, smoke: bool) -> Path:
    root = runs_root(cfg) / "smoke" / stage if smoke else runs_root(cfg) / stage
    if ref == "latest":
        done = sorted(p for p in root.glob("*") if (p / "metrics.json").exists())
        if not done:
            raise ProtocolError(f"no completed {stage} run under {root}")
        run = done[-1]
    else:
        run = root / ref
        if not (run / "metrics.json").exists():
            raise ProtocolError(f"{stage} run {run} does not exist or did not complete")
    if read_json(run / "meta.json").get("smoke", False) != smoke:
        raise ProtocolError(f"{run} smoke flag does not match current mode (smoke={smoke})")
    return run


def assert_real_path(path: str | Path) -> Path:
    p = Path(path).resolve()
    if (REPO / "tests") in p.parents or any(part in {"fixtures", "synthetic", "mock"} for part in p.parts):
        raise ProtocolError(f"experimental entry points refuse synthetic or fixture inputs: {p}")
    if not p.exists():
        raise ProtocolError(f"missing required input: {p}")
    return p


APPROVED_INTERPRETATIONS = [
    ("I1", "2", "V2/V4 use only (image, object category) pairs with exactly one instance whose (object, part) has exactly one part mask"),
    ("I2", "2.4", "fixed query templates; default chat template; image before text; truncate after last <SEL>"),
    ("I3", "5 (V3)", "deterministic evaluation; l = highest hidden mAP, ties to smaller l; image-level bootstrap CIs descriptive only"),
    ("I4", "5 (V3)", "raw features L2-normalised only"),
    ("I5", "5 (V2)", "10k seeded PACO-LVIS train images + up to 2k gRefCOCO train no-target examples; 2k steps S1 recipe"),
    ("I6", "3, 7", "tau_rej calibrated on PACO val + gRefCOCO val no-target; V2/V4 PACO val numbers labelled dev"),
    ("I7", "2", "MLP Linear(d,1024)-GELU-LayerNorm-Linear(1024,256) for g and f"),
    ("I8", "2.4", "AdamW lr 3e-4 wd 0.01 cosine 100 warm-up, batch 32 images, focal gamma 2 no alpha, tau 0.07"),
    ("I9", "5 (V4)", "5 seeds; fresh randomised tree and pre-scoring samples per seed; common subset over flat, true, all 5 randomised"),
    ("I10", "5 (V1)", "SceneParser matched by box IoU >= 0.5 and flagged box-matched; PARTONOMY/SceneParser optional"),
    ("I11", "5 (V4)", "V2 checkpoint trained with true-subtree chain negatives; confound reported, no extra variant"),
    ("I12", "4.1", "selector success = selected node IoU >= 0.5 with requested GT; best-proposal identification diagnostic only"),
    ("I13", "2.2", "negative sampling: 32 hard + 32 uniform in-image, chain subtree first with top-up, up to 64 cross-image"),
]

PROPOSED_DEVIATIONS = [
    {
        "id": "D1",
        "spec_section": "1 (Proposals)",
        "change": "SAM 3 automatic mask generation implemented as an explicit 32x32 point-grid loop over Sam3TrackerModel "
                  "+ Sam3TrackerProcessor instead of transformers' MaskGenerationPipeline",
        "reason": "the pipeline (transformers 5.0.0) normalises grid points by the longest edge while Sam3 resizes each "
                  "axis to 1008, so on non-square images part of the image receives no points; the pipeline also applies "
                  "box NMS and a predicted-IoU filter that the protocol disables",
        "expected_impact": "grid covers the whole image as the protocol intends; model, prompts, stability filter and "
                           "mask post-processing are those of the transformers SAM 3 tracker route",
    },
    {
        "id": "D3",
        "spec_section": "1 (Proposals) / 5 (V1)",
        "change": "SAM 3 stability filter disabled (stability_thresh 0.0); min_area 64 kept; stability still computed "
                  "and used only for dedup tie-breaking",
        "reason": "full V1 failed (part recall 0.231, feasible coverage 0.146). Ceiling diagnostics on PACO val: one "
                  "interior point recovers 0.451 of parts and 0.831 of objects with any of SAM 3's 3 masks, but only "
                  "0.236 / 0.579 after the stability >= 0.9 filter, matching the grid (0.231 / 0.581)",
        "expected_impact": "more proposals and tree nodes; part recall bounded by ~0.45 (point oracle), still below "
                           "the 0.80 gate; gate thresholds unchanged",
    },
    {
        "id": "D2",
        "spec_section": "5 (V1 gate consequence)",
        "change": "V2, V3 and V4 run although the V1 gate failed; every downstream metrics.json is labelled "
                  "below_v1_gate with the V1 numbers; attention-seeded re-proposal is not run",
        "reason": "ceiling diagnostics show no SAM 3 prompting inside parents reaches the gate on PACO parts "
                  "(point+parent box 0.290, own GT box 0.778), so re-proposal cannot change the gate outcome",
        "expected_impact": "V2/V4 results hold on the coverage-eligible subset only and are reported as conditional "
                           "on the failed coverage gate",
    },
    {
        "id": "D4",
        "spec_section": "1 (Proposals) / 2 (Selector)",
        "change": "node budget for features, V2, V3 and V4: per image keep at most 256 tree nodes (tau_d 0.9 set), "
                  "ranked by a label-free score (SAM 3 stability or predicted IoU), ties by lower proposal index; the "
                  "tree is rebuilt on the kept nodes. The score is chosen on PACO train coverage only "
                  "(scripts/d4_node_cap.py); V1 stays uncapped",
        "reason": "with D3 the tau_d 0.9 trees have ~1,100-1,500 nodes per image (was ~56); uncapped, the feature "
                  "cache is ~700 GB and V2 cannot hold its training set in memory",
        "expected_impact": "coverage of the capped node set is reported next to the uncapped V1 numbers on PACO val; "
                           "V2-V4 run on the capped trees",
    },
    {
        "id": "D5",
        "spec_section": "5 (V4)",
        "change": "additional V4 block: the step-2 candidate set is the step-1 node and its strict descendants "
                  "(descendant-or-self) instead of strict descendants only; feasibility likewise. Also reported: "
                  "the split into two-level chains (|GT part| < 0.5 |GT object|) and single-level chains (>= 0.5). "
                  "The spec-strict V4 block is unchanged and stays the reference",
        "reason": "chain audit on PACO val (no disagreement between tree code and a pixel brute force): 58 of 63 "
                  "chains lost with both masks found (tau_d 0.95) have one proposal matching both object and part; "
                  "their GT part covers a median 0.70 of the object (0.18 over all chains), e.g. phone back cover, "
                  "vase body. Under IoU >= 0.5 the object's own mask is a correct part answer when |part| >= 0.5 "
                  "|object|, so a strictly smaller node cannot exist",
        "expected_impact": "higher chain coverage and success on single-level chains for every structure; the "
                           "two-level split isolates the effect of the tree on genuine object->part chains",
    },
    {
        "id": "D6",
        "spec_section": "1 (Tree) / 5 (V4)",
        "change": "additional V4 block: a box tree built from the same nodes' bounding boxes with the spec's parent "
                  "rule (smallest larger box containing >= 80% of the node's box), compared with the mask tree and "
                  "flat search on their own common eligible subset. Nodes, features and selector are unchanged; the "
                  "spec-strict V4 block stays the reference",
        "reason": "tree visualisations on the spec node set place occluded objects under the object occluding or "
                  "holding them only by box containment (phone under the person holding it: mask tree depth 1, box "
                  "tree depth 4); masks of occluded objects do not nest",
        "expected_impact": "box-tree chains include support and occlusion relations; false links (an object behind a "
                           "person inside the person's box) are possible and show up as lower box-tree success",
    },
    {
        "id": "D7",
        "spec_section": "amendment (additional evaluation data)",
        "change": "transfer evaluation of the PACO-trained V2 selector, without retraining, on Pascal-Part (VOC 2010 "
                  "val, left/right and numbered parts merged into one mask per part type) and ADE20K (2021 release, "
                  "validation, visible instance masks, object->part and part->subpart chains, same-name children "
                  "merged); same proposals (D3), node budget (D4), chain template and V4 search incl. D5/D6. "
                  "Reported per dataset as transfer results; PACO stays the main pilot",
        "reason": "PACO parts are dominated by small and boundary-less parts; recent part-grounding work evaluates on "
                  "Pascal-Part and ADE20K-Part; ADE20K adds deeper scene hierarchies",
        "expected_impact": "coverage and chain success on datasets with coarser parts; no change to any PACO result",
    },
    {
        "id": "D8",
        "spec_section": "amendment (additional evaluation data)",
        "change": "two more transfer sets under D7: box-containment object hierarchies on LVIS v1 val and OpenImages "
                  "validation, rebuilt with the rule of arXiv 2411.17490 (80% box containment; pair frequency / "
                  "proportion 5 / 5% on LVIS and 50 / 10% on OpenImages, computed on the evaluated split; boxes under "
                  "1% of the image and IsGroupOf dropped); additionally pairs whose child box covers >= 90% of the "
                  "parent box (the same entity labelled twice, e.g. person / human body) are dropped. Chains: parent "
                  "class once in the image, child class inside it (LVIS: instances merged, scored with masks; "
                  "OpenImages: exactly one instance, scored box-matched as in I10). LVIS images in the PACO / "
                  "gRefCOCO manifest are excluded",
        "reason": "the hyperbolic-hierarchy paper defines its benchmarks by this rule but no data release was found; "
                  "these chains are mostly support and occlusion relations (LVIS child masks are a median 8% inside "
                  "the parent mask), the case the box tree (D6) targets",
        "expected_impact": "object-in-object chains where box containment holds but mask containment mostly does not",
    },
    {
        "id": "D9",
        "spec_section": "amendment (D8 preprocessing)",
        "change": "OpenImages images are resized to at most 1024 px on the longest side (INTER_AREA) before SAM 3 and "
                  "Qwen; GT boxes are normalised and scale exactly",
        "reason": "the public download bucket serves original-resolution photos (several thousand px); 4 of 64 "
                  "shards ran out of GPU memory in SAM 3 mask post-processing (34 GiB for one point batch); the "
                  "1024-px resize matches the commonly distributed OpenImages image set",
        "expected_impact": "fewer pixels for very large photos; all OpenImages shards rerun with the same setting",
    },
    {
        "id": "D11",
        "spec_section": "2.4 (S1) / 7 (train data)",
        "change": "S1 at full scale on the train sources available: PACO-LVIS train (all manifest images), Pascal-Part "
                  "train (original Pascal-Part on VOC 2010 train, parts merged as in D7, not the OV-PARTS -116 "
                  "release) and gRefCOCO train (present-target referring expressions with every target's best node "
                  "a positive, plus no-target queries as the only all-negative examples). RefCOCO/+/g (subsumed by "
                  "gRefCOCO's expressions on the same images), PanoCaps, SceneParser and PARTONOMY train are not "
                  "used. Same recipe as V2 (I7, I8, I13, 20% text-only) for 20k steps, packed per-shard caches of "
                  "the D3/D4 node sets at layer l, checkpoint selection every 2k steps on PACO val single-step "
                  "selector success, tau_rej calibrated by the V2 rule at the selected checkpoint",
        "reason": "spec S1 trains on PACO, Pascal-Part-116, RefCOCO/+/g, gRefCOCO, PanoCaps, SceneParser and "
                  "PARTONOMY; only the listed sources are available on the cluster",
        "expected_impact": "a selector trained on ~6x more images than V2; results reported as S1 next to V2",
    },
    {
        "id": "D12",
        "spec_section": "5 (V4) / transfer diagnostics",
        "change": "an extra diagnostic block in V4 and transfer: natural search rerun with the rejection threshold "
                  "disabled (tau_rej = -inf, every step takes the top-scoring candidate); the primary numbers keep "
                  "the calibrated tau_rej",
        "reason": "transfer chain success is near zero (OpenImages step 1 0.17%); this separates wrong rejections "
                  "from wrong selections",
        "expected_impact": "none on primary numbers; adds natural_no_rejection blocks",
    },
    {
        "id": "D13",
        "spec_section": "amendment (external baselines)",
        "change": "same-protocol baselines on the V2 evaluation set (PACO-LVIS val I1 object and part queries, "
                  "gRefCOCO val no-target queries, same GT masks and image-level bootstrap): (a) SAM 3 text "
                  "prompting (Sam3Model, same facebook/sam3 checkpoint), concept = the I1 query phrase without "
                  "'Locate', the article and <SEL> ('mug', 'handle of the mug'), highest-scoring instance, empty "
                  "if its score < 0.5; (b) Qwen3-VL-4B (pinned revision) asked for one bbox_2d in JSON (0-1000 "
                  "relative coordinates) for 'the mug' / 'the handle of the mug', box prompted to the SAM 3 "
                  "tracker, unparseable output = empty mask; (c) Sa2VA-Qwen3-VL-4B (pinned revision) with its "
                  "RefCOCO prompt 'Please segment <phrase> in this image.', union of its [SEG] masks. No-target "
                  "queries use the gRefCOCO expression. An empty mask is an abstention (IoU 0 on a present "
                  "query, 1 on a no-target query). Reported: gIoU (mean per-query IoU over all present queries), "
                  "cIoU, by kind, abstention, and chains scored from the object and part predictions of the same "
                  "image (final part IoU >= 0.5; object and part both >= 0.5), without step-1 conditioning",
        "reason": "the spec has no external baselines; published segmentation MLLMs report numbers on other data "
                  "and query sets, so a fair comparison needs them on our exact queries, masks and metric",
        "expected_impact": "baseline numbers directly comparable with V2's end-to-end gIoU on present queries; no "
                           "change to any TreeGround result",
    },
    {
        "id": "D10",
        "spec_section": "2.4 (S2)",
        "change": "S2 chains from PACO-LVIS train I1 queries (object, part, object->part chain) instead of "
                  "SceneParser-Bench / PARTONOMY chains (not available). Sequence: chat template [image; I1 query "
                  "template with <SEL> removed] then an assistant answer span with one <SEL> right after each "
                  "referent: '{object} <SEL>', '{part} of the {object} <SEL>', '{object} <SEL> {part} <SEL>'; "
                  "text-only '{part} of a {object} <SEL>' (no image); gRefCOCO no-target '{expression} <SEL>'. "
                  "L_sel as V2 (S1 recipe, 20% text-only, no-target examples); cross-image negatives come from the "
                  "other images of the same per-GPU micro-batch",
        "reason": "the spec's S2 chain sources are not available to us; PACO I1 chains are the only supplied "
                  "object->part chains with masks; per-GPU micro-batches hold a few images, not the 32 of S1",
        "expected_impact": "S2 is trained and evaluated on the same chain distribution as S1/V4; no "
                           "object->part->affordance or explanatory chains",
    },
    {
        "id": "D14",
        "spec_section": "amendment (task 3: region-grounded editing)",
        "change": "region-grounded editing, outside the frozen spec. Pipeline: (image, edit instruction, referent "
                  "phrase) -> our selector (V2 checkpoint, or S1 under D11) on the D3/D4 node set with the I1 query "
                  "'Locate {referent}. <SEL>', top-1 node taken without rejection (an edit always names a present "
                  "region; sigma >= tau_rej recorded) -> mask -> diffusion editor. Datasets: (1) PartEdit-Bench "
                  "(Aleksandar/PartEdit-Bench, real 13 + synth 60 rows), instruction instructp2p_edit2, referent "
                  "'the {part} of the {subject}', GT = gt_mask; (2) GEdit-Bench and ImgEdit-Bench with the edited-"
                  "region masks of wyh6666/GEditBench_ImgEditBench_with_mask (black = edited): local-edit subsets "
                  "only (GEdit color_alter, material_alter, subject-replace, subject-remove; ImgEdit adjust, replace, "
                  "remove), dropping edited_object naming the whole image or background and masks covering 0 or > 90% "
                  "of the image; referent = 'the ' + the dataset's edited_object (no derivation needed); (3) "
                  "PACO-LVIS val: 500 seeded (object, part) pairs eligible as I1 part queries with part area >= 1024 "
                  "px, at most one per image, seeded recolor ('make the {part} of the {object} {color}') or material "
                  "('make the {part} of the {object} look like {material}') instruction, GT = part mask (labelled "
                  "dev: PACO val calibrated tau_rej). Editors (pinned): Qwen-Image-Edit-2511 (QwenImageEditPlus, bf16, "
                  "model-card sampling) and FLUX.1-Fill-dev (FluxFill, bf16, ~1 MP). Conditions: free (instruction "
                  "only, Qwen), {ours,gt,sam3text}_blend (Qwen output composited into the original with the mask "
                  "dilated 8 px and Gaussian-feathered sigma 4) and {ours,gt,sam3text}_fill (Fill on the dilated mask, "
                  "pasted back the same way); sam3text = SAM 3 text prompt on the referent, D13 route, top instance "
                  "without its score threshold. Same seed per sample across conditions. Metrics w.r.t. the GT edit "
                  "mask at source resolution: outside PSNR / SSIM (Gaussian 11, sigma 1.5) / LPIPS-alex (spatial), "
                  "inside mean |change| and LPIPS, CLIP ViT-L/14 directional similarity and CLIP-T against fixed "
                  "per-dataset captions, mask IoU vs GT (ours top-1, best node, SAM 3 text); image-level bootstrap "
                  "95% CIs; edited PNGs kept for a later MLLM judge",
        "reason": "the spec covers grounding only; editing tests whether tree-node selection gives editors usable "
                  "part-level masks, with the GT mask as the localisation upper bound and SAM 3 text as the "
                  "same-backbone baseline",
        "expected_impact": "a new task-3 result family; no change to any grounding result",
    },
    {
        "id": "D15",
        "spec_section": "2.4 / V2 end-to-end reporting",
        "change": "single-step end-to-end numbers recomputed from each stage's saved per-query table (V2, S1, S2): at "
                  "the recorded calibrated tau_rej, and with rejection disabled on present queries (top node always "
                  "returned, the protocol of present-only benchmarks and of the D13 baselines without abstention); "
                  "no-target accuracy reported separately with the full threshold curve; primary numbers unchanged",
        "reason": "S1, trained on gRefCOCO, scores gRefCOCO no-target expressions high, so Eq. 2 calibration on the "
                  "8275 no-target vs 7225 present queries sets tau_rej 0.88 and S1 abstains on most present queries "
                  "(present gIoU 0.086 although selector success rose 0.48 -> 0.64)",
        "expected_impact": "adds no-rejection and curve blocks; separates selection quality from abstention",
    },
    {
        "id": "D16",
        "spec_section": "amendment (external baselines on the transfer sets)",
        "change": "the three D13 baselines (same models, revisions, prompts and settings) on the D7/D8 transfer sets "
                  "(Pascal-Part val, ADE20K-Part validation, LVIS / OpenImages box-containment hierarchies), on the "
                  "same images, chains and GT as x_transfer (same records, D9 OpenImages resize, manifest LVIS "
                  "images excluded). Per chain, one object query per distinct chain object in the image and one "
                  "part query per chain, phrased with the I1 templates as in D13 on the names x_transfer puts in its "
                  "chain template ('the {object}', 'the {part} of the {object}'; ADE20K part->subpart chains use "
                  "the object '{part} of the {object}'). Masks scored by mask IoU (LVIS included); OpenImages GT "
                  "boxes scored box-matched (I10) by the box IoU of the predicted mask's bounding box, as x_transfer "
                  "scores nodes. Reported per dataset: object and part gIoU (empty = 0), box/mask cIoU, success@0.5, "
                  "chain success (object and part IoU >= 0.5) and part-only success over all chains, two-level and "
                  "single-level splits, and the same on the common eligible subset of the V2 x_transfer merge "
                  "(chains_strict.csv: chains feasible under flat, true and all randomised trees); image-level "
                  "bootstrap 95% CIs (v4 settings). No step-1 conditioning",
        "reason": "transfer chain success of TreeGround needs an external reference on the same chains, GT and "
                  "metric; published numbers use other query sets",
        "expected_impact": "baseline rows in the transfer table; no change to any TreeGround or PACO baseline result",
    },
    {
        "id": "D17",
        "spec_section": "amendment (D7/D8 transfer of S2)",
        "change": "the D7/D8 transfer evaluation (Pascal-Part, ADE20K, LVIS and OpenImages hierarchies; same SAM 3 "
                  "proposals, tau_d dedup, D4 node budget, trees and V4 search incl. D5/D6/D12) run with the S2 "
                  "model (D10) instead of a frozen-Qwen selector: LoRA-adapted Qwen with the trained <SEL> row and "
                  "the S2 selector of the S2 run's best.pt (from V2, and from S1), no retraining. Node features and "
                  "chain probes are computed as s2_eval does on PACO: [image; chain query without <SEL>] followed "
                  "by the teacher-forced answer '{object} <SEL> {part} <SEL>' with the dataset's GT names, "
                  "layer-l states rounded to fp16; tau_rej is the PACO-val tau of the S2 eval run of that "
                  "checkpoint. SAM 3 proposals are recomputed (transfer caches keep only per-shard results)",
        "reason": "S2 is evaluated on PACO only; the transfer sets test whether adapting the MLLM helps or hurts "
                  "selection on other part and object hierarchies, next to the V2 and S1 transfer rows",
        "expected_impact": "new S2 and S2 (from S1) transfer rows (x_transfer_<ds>_s2[_from_s1]_merge); no change "
                           "to any existing result",
    },
    {
        "id": "D18",
        "spec_section": "amendment (task 3 evaluation, extends D14)",
        "change": "MLLM judge for every D14 edit cache (editor x dataset x selector stage), all conditions: GEdit-"
                  "Bench VIEScore protocol (task 'tie', prompts verbatim from stepfun-ai/Step1X-Edit@4cca05df "
                  "GEdit-Bench/viescore; source and edited image resized to ~512x512 px keeping the aspect ratio): "
                  "SC = min(editing success, overediting) on [source, edited] with the dataset instruction, PQ = "
                  "min(naturalness, artifacts) on the edited image, overall = sqrt(SC * PQ), each 0-10. Judge "
                  "Qwen2.5-VL-72B-Instruct (pinned, bf16, greedy, 512 new tokens, batch 8, 4 GPUs) instead of "
                  "GEdit's Qwen2.5-VL-72B-Instruct-AWQ fp16; an optional fast pass with Qwen3-VL-8B-Instruct or "
                  "Qwen2.5-VL-7B-Instruct is reported under its own stage name. Second variant 'inside': the same "
                  "prompts on a crop of source and edited image around the GT edit mask bbox padded by 25% per "
                  "edge (min side 64 px). Unparseable outputs stay missing and are reported as a rate (VIEScore "
                  "substitutes a random score). Image-level bootstrap 95% CIs per condition and variant",
        "reason": "D14's pixel and CLIP metrics do not measure whether the instruction was carried out on the "
                  "named part; GEdit-Bench's VIEScore is the standard instruction-editing judge with an open-"
                  "weight setting, and the crop focuses the judge on the requested region instead of the whole "
                  "image",
        "expected_impact": "adds judge scores next to D14 metrics; MLLM judges are biased and noisy, so differences "
                           "between conditions are read with their CIs; no change to any D14 number",
    },
    {
        "id": "D19",
        "spec_section": "amendment (post-hoc mask refinement of the selected node)",
        "change": "selection + re-prompt refinement, evaluated for V2, S1, S2 and S2 (from S1) without retraining: "
                  "the selector's top-1 node (same selector, features, probes, capped proposals and tau_rej as the "
                  "stage's evaluation; single step as V2.evaluate, chains as V4 natural search on the true tree with "
                  "both the object and the part node refined) is re-prompted to the proposals' SAM 3 tracker "
                  "(same repo/revision/dtype, multimask_output, logits > mask_threshold) on the image, and the "
                  "refined mask replaces the node mask. Variants: r0 node mask (unchanged), r1 node bbox, r2 bbox + "
                  "positive point at the node's interior point (max distance transform, V1-ceiling rule), r3 bbox + "
                  "node mask as dense prompt (binary mask resized to the 288x288 low-res frame, logits 20 m - 10), "
                  "r4 r2 + up to 2 negative points inside the node's tree parent and outside the node (greedy "
                  "interior points, clearance >= 2 px; none when the parent is the root). Output: the multimask "
                  "output with the highest predicted IoU. Accept decisions unchanged. Reported with the V2/D15 "
                  "metrics (present gIoU calibrated and without rejection, by kind, cIoU, success@0.5, no-target "
                  "abstention, image-level bootstrap 95% CIs) and chain success (D12 no-rejection included); "
                  "ceilings: r1/r2 of the highest-IoU node and the GT box prompt on the same queries, next to the "
                  "V1-ceiling own-box recall",
        "reason": "our masks are limited to the fixed proposal set (a node with IoU >= 0.5 exists for ~45-60% of "
                  "queries) while the D13 baselines prompt SAM 3 directly; refinement tests whether the tree "
                  "selection is a good prompt even when no proposal matches the target exactly",
        "expected_impact": "new '+ SAM 3 refine' rows (refine_<stage>_merge); r0 must reproduce each stage's "
                           "saved numbers; no change to any existing result",
    },
    {
        "id": "D22",
        "spec_section": "amendment (transfer reporting)",
        "change": "per-chain IoU of our picks on the D7/D8 transfer sets, scored like the D16 baselines: the transfer "
                  "pipeline (x_transfer.py for V2/S1, x_transfer_s2.py for S2) is rerun unchanged per image and, per "
                  "chain, the object pick (argmax over the capped nodes), the part pick over all nodes (flat) and the "
                  "part pick inside the object pick's subtree (true tree, V4 natural search) are recorded with their "
                  "IoU (mask IoU, OpenImages box-matched as in D8); empty node set or rejection = 0. Reported: object "
                  "gIoU per distinct object per image, part gIoU and part success@0.5 per chain, flat and tree, "
                  "without rejection and at the stage's tau_rej; image-level bootstrap 95% CIs",
        "reason": "transfer runs stored chain success flags only; part gIoU next to the D16 baselines and the "
                  "published part gIoU (Pascal-Part-116) needs the IoU of each pick",
        "expected_impact": "part/object gIoU for our rows in the transfer table; no change to any existing result",
    },
]


def deviation_log_path(cfg: dict) -> Path:
    return runs_root(cfg) / "deviations.jsonl"


def read_deviations(cfg: dict) -> dict[str, dict]:
    path = deviation_log_path(cfg)
    if not path.exists():
        return {}
    records = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return {r["id"]: r for r in records}


def ensure_deviation_log(cfg: dict) -> None:
    path = deviation_log_path(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    existing = read_deviations(cfg)
    today = dt.date.today().isoformat()
    new = [
        {"id": i, "date": today, "spec_section": sec, "change": change, "reason": "user decision (implementation prompt §2)",
         "expected_impact": "fixes an open point; no silent choice", "status": "approved"}
        for i, sec, change in APPROVED_INTERPRETATIONS if i not in existing
    ]
    new += [dict(d, date=today, status="proposed") for d in PROPOSED_DEVIATIONS if d["id"] not in existing]
    if new:
        with open(path, "a") as f:
            for r in new:
                f.write(json.dumps(r, sort_keys=True) + "\n")


def require_approved(cfg: dict, ids: list[str]) -> None:
    records = read_deviations(cfg)
    pending = [i for i in ids if records.get(i, {}).get("status") != "approved"]
    if pending:
        raise ProtocolError(
            f"deviation(s) {pending} are not approved in {deviation_log_path(cfg)}; set status to 'approved' to proceed")


def v1_gate(cfg: dict, smoke: bool) -> dict:
    # stages after V1 run on a passed gate, or on a failed one only while deviation D2 is approved
    run = resolve_run(cfg, "v1", cfg.get("inputs", {}).get("v1", "latest"), smoke)
    gate = read_json(run / "metrics.json")["gate"]
    note = {"v1_run": run.name, "v1_gate_passed": bool(gate["passed"]), "below_v1_gate": not gate["passed"],
            "v1_part_recall": gate["part_recall"], "v1_feasible_path_coverage": gate.get("feasible_path_coverage")}
    if not gate["passed"] and not smoke:
        require_approved(cfg, ["D2"])
        note["deviation"] = "D2"
    return note
