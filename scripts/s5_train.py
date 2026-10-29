import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s4_train as S4  # noqa: E402  (installs the D30 / D34 hooks; overridden below where S5 differs)
import s5_pool as S5P  # noqa: E402
import s3_pool as S3P  # noqa: E402
import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# Deviation D36 (trained): S5 = the S4 recipe (per-query pools, soft-tree chain term, S2-long schedule) on the D36
# pools (scripts/s5_pool.py: + the Qwen3-VL-4B box + SAM 3 mask of every phrase), with
# (1) a listwise IoU term per query step over the query's whole own pool (nodes + its phrases' instances and box
#     masks; other queries' instances excluded): target t_i = softmax(IoU_i / tau) over the entries with IoU >= 0.5
#     (0 elsewhere), loss = -sum_i t_i log softmax(s)_i, weight listwise_weight; it ranks a 0.9-IoU mask above a
#     0.55 one, which the positive / ignore targets do not;
# (2) the benchmarks' training splits: PACO-LVIS train, Pascal-Part train, all of gRefCOCO train (present and
#     no-target) and PartImageNet_Seg train (with PACO and Pascal-Part, PARTONOMY's training sources).
# Shares, tau and the weight live in cfg['s2'] (s5_sources, listwise_tau, listwise_weight).
# D37 specialists: cfg['s2'] spec_sources (one benchmark's training data) and fresh_selector.

SUF, ABSENT = "_s5", "_s5_absent"


@dataclass
class IouStep(S.StepTarget):
    iou: np.ndarray | None = None
    allowed: np.ndarray | None = None


def base_source(source: str) -> str:
    for suf in (ABSENT, SUF):
        if source.endswith(suf) and source[:-len(suf)] in S5P.SOURCES:
            return source[:-len(suf)]
    raise ProtocolError(f"invalid S5 source {source}")


def source_weights(c: dict) -> dict[str, float]:
    w = c.get("spec_sources") or c.get("s5_sources", {})   # D37 specialists use their own key (configs merge dicts)
    if not w or min(w.values()) <= 0:
        raise ProtocolError(f"invalid S5 sources {w}")
    for k in w:
        base_source(k)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def node_cache(cfg: dict, source: str, smoke: bool) -> tuple[Path, list[Path]]:
    base = base_source(source)
    d = S5P.out_dir(cfg, S5P.identity(cfg, base, smoke))
    files = sorted((d / "shards").glob("*.pt"))
    n = S5P.num_shards(cfg, base, smoke)
    if len(files) != n:
        raise ProtocolError(f"{source}: {len(files)} of {n} s5_pool shards present in {d} (run scripts/s5_pool.py)")
    return d, files


def node_meta(cfg: dict, source: str, smoke: bool) -> tuple[list, Counter]:
    d, files = node_cache(cfg, source, smoke)
    metas, counts = [], Counter()
    only_absent = source.endswith(ABSENT)
    for f in files:
        blob = torch.load(f, map_location="cpu", weights_only=False)
        if not only_absent:
            counts.update({f"{source}:{k}": v for k, v in blob["counts"].items()})
        for r in blob["records"]:
            parents = np.asarray(r["parent"], np.int64)
            n = len(parents) - 1
            empty = np.full(n, "", dtype=object)
            labels = np.asarray(r["labels"], dtype=object) if "labels" in r else empty
            parts = np.asarray(r["parts"], dtype=object) if "parts" in r else empty
            high = np.asarray(r["high"], dtype=bool) if "high" in r else np.zeros(n, dtype=bool)
            nn, phr = int(r["n_nodes"]), r["phrases"]
            owner = np.full(n, -1, np.int64)
            for j, im in enumerate(r["instances"]):
                owner[nn + j] = im["phrase"]

            def allowed(text: str) -> np.ndarray:
                own = {phr.index(p) for p in S3P.phrases(text) if p in phr}
                a = np.zeros(n, bool)
                a[:nn] = True
                a[nn:] = np.isin(owner[nn:], list(own))
                return a

            def steps_for(raw, a, ious):
                out = []
                for t, (p, i) in enumerate(raw):
                    pos = np.asarray([v for v in p if a[v]], np.int64)
                    ign = np.unique(np.concatenate([np.asarray(i, np.int64), np.flatnonzero(~a)])).astype(np.int64)
                    out.append(IouStep(pos, ign, None if ious is None else np.asarray(ious[t], np.float32), a))
                return out

            base_kw = dict(sha1=r["key"], key=r["key"], file=r["path"], n=n, parents=parents, labels=labels,
                           parts=parts, high=high, source=source, nodes=str(d / r["nodes"]))
            if r["queries"] and not only_absent:
                qs = []
                for q in r["queries"]:
                    a = allowed(q["text"])
                    steps = steps_for(q["steps"], a, q.get("ious"))
                    if any(len(st.pos) == 0 for st in steps) or (q["path"] and not all(a[v] for v in q["path"])):
                        counts[f"{source}:dropped_outside_own_pool"] += 1
                        continue
                    qq = S2T.query_from_cache(q)
                    qq.steps = steps
                    qs.append(qq)
                if qs:
                    metas.append(S2T.ImageMeta(queries=qs, **base_kw))
            for ab in r["absent"]:
                st = steps_for([([], [])], allowed(ab["text"]), None)
                q = S2T.Query(ab["qid"], "absent", S2.user_text(ab["text"]),
                              S2.answer_text("absent", ab["qid"], ab["text"]), st)
                metas.append(S2T.ImageMeta(queries=[q], absent=True, **base_kw))
    counts[f"{source}:training_items"] = len(metas)
    return metas, counts


def make_batch_terms(inner, tau: float, weight: float):
    def batch_terms(sel, batch, text_probes, rng, device, gamma, stats):
        terms = inner(sel, batch, text_probes, rng, device, gamma, stats)
        if not batch:
            return terms
        e = sel.nodes(torch.cat([r.z for r, _ in batch]).to(device))
        lo = 0
        for rec, as_text in batch:
            hi = lo + rec.n
            for q in ([] if as_text or rec.absent else rec.queries):
                for t, st in enumerate(q.steps):
                    if getattr(st, "iou", None) is None:
                        continue
                    idx = np.flatnonzero(st.allowed)
                    u = st.iou[idx]
                    good = u >= S.IOU_THR
                    if not good.any():
                        continue
                    tgt = np.zeros(len(idx), np.float64)
                    g = u[good] / tau
                    tgt[good] = np.exp(g - g.max()) / np.exp(g - g.max()).sum()
                    s = sel.scores(sel.queries(q.probes[t].to(device)[None]), e[lo:hi])[0]
                    logp = F.log_softmax(s[torch.as_tensor(idx, device=device)].float(), 0)
                    loss = -(torch.as_tensor(tgt, device=device, dtype=torch.float32) * logp).sum()
                    stats["terms_listwise"] += 1
                    terms.append(weight * loss)
            lo = hi
        return terms
    return batch_terms


S2T.deviations = lambda cfg: (["D10", "D21", "D30", "D34", "D36"] + (["D37"] if cfg["s2"].get("spec_sources") else [])
                              + (["D39"] if float(cfg["s2"].get("soft_weight", 1)) == 0 else []))
_warm_start = S2T.warm_start


def warm_start(cfg: dict, smoke: bool, layer: int, d_model: int):
    # D37 specialists: a freshly initialised selector (seeded) instead of the S1 one, which saw every benchmark's
    # training split; only the temperature is read from S1
    sel, run = _warm_start(cfg, smoke, layer, d_model)
    if not cfg["s2"].get("fresh_selector"):
        return sel, run
    from treeground.selector import Selector
    torch.manual_seed(int(cfg["s2"]["seed"]))
    return Selector(d_model, d_model, sel.tau), f"fresh(temperature from {run})"


S2T.warm_start = warm_start
S2T.source_weights, S2T.node_cache, S2T.node_meta = source_weights, node_cache, node_meta
if "terms_listwise" not in S2T.STAT_KEYS:
    S2T.STAT_KEYS.append("terms_listwise")


def main() -> None:
    import argparse
    from treeground.run import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    known, _ = ap.parse_known_args()
    c = load_config(known.config)["s2"]
    # soft_weight 0 (D39 ablation) drops the soft-tree term entirely, rather than adding zero terms to the mean
    soft = (S4.make_batch_terms(float(c["soft_lambda"]), float(c["soft_weight"])) if float(c["soft_weight"]) > 0
            else S4._batch_terms)
    V2.batch_terms = make_batch_terms(soft, float(c["listwise_tau"]), float(c["listwise_weight"]))
    S2T.main()


if __name__ == "__main__":
    main()
