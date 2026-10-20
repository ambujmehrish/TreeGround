import hashlib
import json
from pathlib import Path

import pandas as pd

from treeground.cache import FeatureStore, ProposalStore, cache_dir, check_identity, code_version, require_complete
from treeground.data.manifest import load_manifest
from treeground.mllm import candidate_layers
from treeground.proposals import ROUTE
from treeground.run import ProtocolError, canonical_hash, resolve_run

FEATURE_TYPES = ("hidden", "value", "encoder", "siglip")


def manifest_inputs(cfg: dict, smoke: bool) -> tuple[Path, pd.DataFrame, str, pd.DataFrame]:
    run = resolve_run(cfg, "s0_manifest", cfg.get("inputs", {}).get("manifest", "latest"), smoke)
    df, mhash = load_manifest(run)
    wl = pd.read_parquet(run / "worklist.parquet")
    return run, df, mhash, wl


def proposal_identity(cfg: dict, smoke: bool, manifest_hash: str) -> dict:
    p, m = cfg["proposals"], cfg["models"]["sam3"]
    return {
        "kind": "proposals", "smoke": smoke, "manifest": manifest_hash,
        "sam3_repo": m["repo"], "sam3_revision": m["revision"], "route": ROUTE,
        "points_per_side": p["points_per_side"], "masks_per_point": p["masks_per_point"],
        "stability_thresh": p["stability_thresh"], "stability_offset": p["stability_offset"],
        "mask_threshold": p["mask_threshold"], "min_area": p["min_area"],
        "dedup_thresholds": sorted(p["dedup_thresholds"]), "dtype": p["dtype"], "code": code_version("proposals"),
    }


def qwen_static(cfg: dict) -> dict:
    from transformers import AutoConfig, AutoProcessor
    m = cfg["models"]["qwen"]
    # pinned revisions are fetched once by fetch_models.py; never touch the network here
    kw = {"revision": m["revision"], "cache_dir": cfg.get("hf_home"), "local_files_only": True}
    conf = AutoConfig.from_pretrained(m["repo"], **kw)
    proc = AutoProcessor.from_pretrained(m["repo"], **kw)
    image_proc = json.dumps(proc.image_processor.to_dict(), sort_keys=True, default=str)
    return {
        "layers": candidate_layers(conf.text_config.num_hidden_layers),
        "image_processor": hashlib.sha256(image_proc.encode()).hexdigest(),
        "chat_template": hashlib.sha256((proc.chat_template or "").encode()).hexdigest(),
    }


def feature_identity(cfg: dict, smoke: bool, manifest_hash: str, static: dict) -> dict:
    f, q = cfg["features"], cfg["models"]["qwen"]
    ident = {
        "kind": "features", "smoke": smoke, "manifest": manifest_hash,
        "qwen_repo": q["repo"], "qwen_revision": q["revision"], "image_processor": static["image_processor"],
        "dtype": f["dtype"], "layers": static["layers"], "feature_types": sorted(f["feature_types"]),
        "proposals": canonical_hash(proposal_identity(cfg, smoke, manifest_hash)),
        "node_dedup": cfg["proposals"]["node_dedup"], "code": code_version("features"),
    }
    if "siglip" in f["feature_types"]:
        ident["siglip_repo"] = cfg["models"]["siglip"]["repo"]
        ident["siglip_revision"] = cfg["models"]["siglip"]["revision"]
    return ident


def query_identity(cfg: dict, smoke: bool, manifest_hash: str, static: dict, sel_hash: str, query_set: str) -> dict:
    q = cfg["models"]["qwen"]
    from treeground.data.paco import TEMPLATES
    return {
        "kind": "queries", "smoke": smoke, "manifest": manifest_hash,
        "qwen_repo": q["repo"], "qwen_revision": q["revision"], "tokenizer_revision": q["revision"],
        "chat_template": static["chat_template"], "image_processor": static["image_processor"],
        "sel_init": sel_hash, "dtype": cfg["features"]["dtype"], "layers": static["layers"],
        "templates": TEMPLATES, "query_set": query_set, "code": code_version("queries"),
    }


def open_proposals(cfg: dict, smoke: bool, manifest: pd.DataFrame, mhash: str) -> ProposalStore:
    ident = proposal_identity(cfg, smoke, mhash)
    return ProposalStore(cache_dir(cfg, "proposals", ident), ident, manifest)


def open_features(cfg: dict, smoke: bool, manifest: pd.DataFrame, mhash: str, static: dict) -> FeatureStore:
    ident = feature_identity(cfg, smoke, mhash, static)
    return FeatureStore(cache_dir(cfg, "features", ident), ident, manifest)


def open_queries(cfg: dict, smoke: bool, mhash: str, static: dict) -> tuple[Path, pd.DataFrame]:
    root = Path(cfg["cache_root"]) / ("smoke" if smoke else "full") / "queries"
    found = []
    for d in sorted(root.glob("*")) if root.exists() else []:
        try:
            ident = json.loads((d / "identity.json").read_text())["identity"]
        except FileNotFoundError:
            continue
        expected = query_identity(cfg, smoke, mhash, static, ident.get("sel_init", ""), ident.get("query_set", ""))
        if ident == expected and (d / "complete.json").exists():
            found.append((d, expected))
    if len(found) != 1:
        raise ProtocolError(f"expected exactly one complete query cache for manifest {mhash[:12]} and the current "
                            f"code under {root}, found {len(found)}")
    d, expected = found[0]
    check_identity(d, expected)
    require_complete(d)
    queries = pd.read_parquet(d / "queries.parquet")
    if canonical_hash(queries[["qid", "text", "sha1", "split", "provenance"]].values.tolist()) != ident["query_set"]:
        raise ProtocolError(f"queries.parquet in {d} does not match its recorded query set")
    return d, queries
