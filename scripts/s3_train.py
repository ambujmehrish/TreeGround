import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s2_train as S2T  # noqa: E402
import s3_pool as S3P  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# Deviation D30: S2-long training (D10/D21 recipe, S1 init, same steps and source shares) on the augmented pools of
# scripts/s3_pool.py. A source '<name>_aug' reads the s3_pool cache of <name> through the S2 node-cache loader;
# PACO keeps its part labels (text terms) from the cache. PACO's no-target items are the gRefCOCO train no-target
# expressions, which grefcoco_train_aug already holds with their augmented pool, so they are not added twice.
# Checkpoint selection stays on the S2 PACO val subset (nodes only).

AUG = "_aug"


def is_aug(source: str) -> bool:
    return source.endswith(AUG) and source[:-len(AUG)] in S3P.SOURCES


def source_weights(c: dict) -> dict[str, float]:
    # configs merge dicts, so S2-long's 'sources' stays in the config; S3 reads its own 'aug_sources'
    w = c.get("aug_sources", {})
    if not w or not all(is_aug(k) for k in w) or min(w.values()) <= 0:
        raise ProtocolError(f"invalid S3 sources {w}: every source must be an s3_pool source '<name>{AUG}'")
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def deviations(cfg: dict) -> list[str]:
    return ["D10", "D21", "D30"]


def node_cache(cfg: dict, source: str, smoke: bool) -> tuple[Path, list[Path]]:
    base = source[:-len(AUG)]
    d = S3P.out_dir(cfg, S3P.identity(cfg, base, smoke))
    files = sorted((d / "shards").glob("*.pt"))
    n = S3P.num_shards(cfg, base, smoke)
    if len(files) != n:
        raise ProtocolError(f"{source}: {len(files)} of {n} s3_pool shards present in {d} (run scripts/s3_pool.py)")
    return d, files


def node_meta(cfg: dict, source: str, smoke: bool) -> tuple[list, Counter]:
    d, files = node_cache(cfg, source, smoke)
    metas, counts = [], Counter()
    empty_step = S.StepTarget(np.zeros(0, np.int64), np.zeros(0, np.int64))
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        counts.update({f"{source}:{k}": v for k, v in blob["counts"].items()})
        for r in blob["records"]:
            parents = np.asarray(r["parent"], np.int64)
            n = len(parents) - 1
            empty = np.full(n, "", dtype=object)
            labels = np.asarray(r["labels"], dtype=object) if "labels" in r else empty
            parts = np.asarray(r["parts"], dtype=object) if "parts" in r else empty
            high = np.asarray(r["high"], dtype=bool) if "high" in r else np.zeros(n, dtype=bool)
            if len(labels) != n:
                raise ProtocolError(f"{r['key']}: {len(labels)} labels for {n} pool entries")
            base = dict(sha1=r["key"], key=r["key"], file=r["path"], n=n, parents=parents, labels=labels,
                        parts=parts, high=high, source=source, nodes=str(d / r["nodes"]))
            if r["queries"]:
                metas.append(S2T.ImageMeta(queries=[S2T.query_from_cache(q) for q in r["queries"]], **base))
            for a in r["absent"]:
                q = S2T.Query(a["qid"], "absent", S2.user_text(a["text"]),
                              S2.answer_text("absent", a["qid"], a["text"]), [empty_step])
                metas.append(S2T.ImageMeta(queries=[q], absent=True, **base))
    counts[f"{source}:training_items"] = len(metas)
    return metas, counts


S2T.source_weights, S2T.deviations, S2T.node_cache, S2T.node_meta = source_weights, deviations, node_cache, node_meta

if __name__ == "__main__":
    S2T.main()
