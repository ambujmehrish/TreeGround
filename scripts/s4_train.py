import sys
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_pool as S3P  # noqa: E402
import s3_train as S3T  # noqa: E402  (D30 augmented-pool hooks)
import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
from treeground import s2 as S2  # noqa: E402
from treeground import sampling as S  # noqa: E402
from treeground.run import ProtocolError  # noqa: E402

# Deviation D34 (trained): S4 = the S3 recipe with three changes.
# (1) Per-query pools: every query is trained over the capped nodes + the SAM 3 instances of its OWN phrases (as
#     s3_eval / gres_eval / partonomy_eval evaluate), by moving every other query's instances into the step's ignore
#     set (no positive, no in-image negative); a query whose positive or chain path lies only in other queries'
#     instances is dropped and counted. Cross-image negatives are unchanged.
# (2) No-target mix as S2 long: the PACO group of S2 long carried 2,000 gRefCOCO train no-target items (24% of
#     it); source 'grefcoco_train_aug_absent' (the gRefCOCO cache's no-target items only) restores that share.
# (3) Soft-tree chain term, listwise: for a chain item, c(p) = s_part(p) + lambda * max_{a in A(p)} log sigmoid(
#     s_obj(a)), A(p) = strict ancestors of p (root excluded; p itself only when p has none); loss = -log(sum_pos
#     exp c / sum_{pos + negs} exp c) over the step-2 positives and in-image negatives ranked by c over all allowed
#     candidates. Shared ancestors cancel inside the softmax, so the term does not oppose the step-1 term.
# lambda and the weight live in cfg['s2'] (soft_lambda, soft_weight), so they enter the checkpoint's config hash.

AUG, ABSENT = "_aug", "_aug_absent"
_batch_terms = V2.batch_terms


def base_source(source: str) -> str:
    for suf in (ABSENT, AUG):
        if source.endswith(suf) and source[:-len(suf)] in S3P.SOURCES:
            return source[:-len(suf)]
    raise ProtocolError(f"invalid S4 source {source}")


def source_weights(c: dict) -> dict[str, float]:
    w = c.get("aug_sources", {})
    if not w or min(w.values()) <= 0:
        raise ProtocolError(f"invalid S4 sources {w}")
    for k in w:
        base_source(k)
    total = sum(w.values())
    return {k: v / total for k, v in w.items()}


def restrict(steps: list, allowed: np.ndarray, n: int):
    out = []
    for st in steps:
        pos = np.asarray([p for p in st.pos if allowed[p]], np.int64)
        extra = np.flatnonzero(~allowed)
        ign = np.unique(np.concatenate([np.asarray(st.ignore, np.int64), extra])).astype(np.int64)
        out.append(S.StepTarget(pos, ign))
    return out


def node_meta(cfg: dict, source: str, smoke: bool) -> tuple[list, Counter]:
    base = base_source(source)
    d, files = S3T.node_cache(cfg, base + AUG, smoke)
    metas, counts = [], Counter()
    empty_step = S.StepTarget(np.zeros(0, np.int64), np.zeros(0, np.int64))
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

            base_kw = dict(sha1=r["key"], key=r["key"], file=r["path"], n=n, parents=parents, labels=labels,
                           parts=parts, high=high, source=source, nodes=str(d / r["nodes"]))
            if r["queries"] and not only_absent:
                qs = []
                for q in r["queries"]:
                    a = allowed(q["text"])
                    steps = restrict([S.StepTarget(np.asarray(p, np.int64), np.asarray(i, np.int64))
                                      for p, i in q["steps"]], a, n)
                    if any(len(st.pos) == 0 for st in steps) or (q["path"] and not all(a[v] for v in q["path"])):
                        counts[f"{source}:dropped_outside_own_pool"] += 1
                        continue
                    qq = S2T.query_from_cache(q)
                    qq.steps = steps
                    qs.append(qq)
                if qs:
                    metas.append(S2T.ImageMeta(queries=qs, **base_kw))
            for a in r["absent"]:
                st = restrict([empty_step], allowed(a["text"]), n)
                q = S2T.Query(a["qid"], "absent", S2.user_text(a["text"]),
                              S2.answer_text("absent", a["qid"], a["text"]), st)
                metas.append(S2T.ImageMeta(queries=[q], absent=True, **base_kw))
    counts[f"{source}:training_items"] = len(metas)
    return metas, counts


def strict_ancestor_mask(tree, n: int) -> np.ndarray:
    m = np.zeros((n, n), dtype=bool)
    for v in range(n):
        u = tree.parent[v]
        while 0 <= u < n:
            m[v, u] = True
            u = tree.parent[u]
        if not m[v].any():
            m[v, v] = True   # top-level node: its own object
    return m


def make_batch_terms(lam: float, weight: float):
    def batch_terms(sel, batch, text_probes, rng, device, gamma, stats):
        terms = _batch_terms(sel, batch, text_probes, rng, device, gamma, stats)
        if not batch:
            return terms
        e = sel.nodes(torch.cat([r.z for r, _ in batch]).to(device))
        lo = 0
        for rec, as_text in batch:
            hi = lo + rec.n
            for q in ([] if as_text or rec.tree is None else rec.queries):
                if q.kind != "chain" or len(q.steps) != 2 or not len(q.steps[1].pos):
                    continue
                s1 = sel.scores(sel.queries(q.probes[0].to(device)[None]), e[lo:hi])[0]
                s2 = sel.scores(sel.queries(q.probes[1].to(device)[None]), e[lo:hi])[0]
                anc = torch.as_tensor(strict_ancestor_mask(rec.tree, rec.n), device=device)
                amax = F.logsigmoid(s1)[None, :].expand(rec.n, -1).masked_fill(~anc, float("-inf")).max(-1).values
                c = s2 + lam * amax
                st = q.steps[1]
                with torch.no_grad():
                    local = c.float().cpu().numpy()
                draw = S.in_image_negatives(S.eligible(np.arange(rec.n), st.pos, st.ignore), local, rng)
                pos = torch.as_tensor(st.pos, device=device, dtype=torch.long)
                neg = torch.as_tensor(draw.ids, device=device, dtype=torch.long)
                loss = torch.logsumexp(c[torch.cat([pos, neg])], 0) - torch.logsumexp(c[pos], 0)
                stats["terms_soft_tree"] += 1
                terms.append(weight * loss)
            lo = hi
        return terms
    return batch_terms


_deviations = S2T.deviations
S2T.deviations = lambda cfg: _deviations(cfg) + ["D34"]
S2T.source_weights, S2T.node_meta = source_weights, node_meta
if "terms_soft_tree" not in S2T.STAT_KEYS:
    S2T.STAT_KEYS.append("terms_soft_tree")


def main() -> None:
    import argparse
    from treeground.run import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    known, _ = ap.parse_known_args()
    c = load_config(known.config)["s2"]
    V2.batch_terms = make_batch_terms(float(c["soft_lambda"]), float(c["soft_weight"]))
    S2T.main()


if __name__ == "__main__":
    main()
