import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import s3_train  # noqa: E402,F401  (D30 augmented-pool hooks)
import s2_train as S2T  # noqa: E402
import v2_train_eval as V2  # noqa: E402
from treeground import sampling as S  # noqa: E402

# Deviation D34 (trained): S4 = S3 (D30 pools, S2-long recipe) plus a soft-tree chain term. For every chain item the
# step-2 logits are combined with the best step-1 log-sigmoid over each candidate's ancestors (itself included):
#   c(p) = s_part(p) + lambda * max_{a in anc(p) + {p}} log sigmoid(s_obj(a))
# and the D10 selection loss is applied to c with the step-2 positives against in-image negatives drawn over ALL
# candidates (not only the object's subtree), so a wrong object pick is penalised through the part score.
# The term is added with weight w next to the unchanged S3 terms.

_batch_terms = V2.batch_terms


def ancestor_mask(tree, n: int) -> np.ndarray:
    m = np.eye(n, dtype=bool)
    for v in range(n):
        u = tree.parent[v]
        while 0 <= u < n:
            m[v, u] = True
            u = tree.parent[u]
    return m


def make_batch_terms(lam: float, weight: float):
    def batch_terms(sel, batch, text_probes, rng, device, gamma, stats):
        terms = _batch_terms(sel, batch, text_probes, rng, device, gamma, stats)
        if not batch:
            return terms
        z = torch.cat([r.z for r, _ in batch]).to(device)
        e = sel.nodes(z)
        lo = 0
        for rec, as_text in batch:
            hi = lo + rec.n
            for q in ([] if as_text or rec.tree is None else rec.queries):
                if q.kind != "chain" or len(q.steps) != 2:
                    continue
                s1 = sel.scores(sel.queries(q.probes[0].to(device)[None]), e[lo:hi])[0]
                s2 = sel.scores(sel.queries(q.probes[1].to(device)[None]), e[lo:hi])[0]
                anc = torch.as_tensor(ancestor_mask(rec.tree, rec.n), device=device)
                lo1 = F.logsigmoid(s1)[None, :].expand(rec.n, -1).masked_fill(~anc, float("-inf")).max(-1).values
                c = s2 + lam * lo1
                st = q.steps[1]
                with torch.no_grad():
                    local = c.float().cpu().numpy()
                draw = S.in_image_negatives(S.eligible(np.arange(rec.n), st.pos, st.ignore), local, rng)
                loss, _, _ = V2.selection_loss(c[torch.as_tensor(st.pos, device=device, dtype=torch.long)],
                                               c[torch.as_tensor(draw.ids, device=device, dtype=torch.long)], gamma)
                stats["terms_soft_tree"] = stats.get("terms_soft_tree", 0) + 1
                terms.append(weight * loss)
            lo = hi
        return terms
    return batch_terms


_deviations = S2T.deviations
S2T.deviations = lambda cfg: _deviations(cfg) + ["D34"]
if "terms_soft_tree" not in S2T.STAT_KEYS:
    S2T.STAT_KEYS.append("terms_soft_tree")


def main() -> None:
    import argparse
    from treeground.run import load_config
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    known, _ = ap.parse_known_args()
    c = load_config(known.config)["s4"]
    V2.batch_terms = make_batch_terms(float(c["soft_lambda"]), float(c["soft_weight"]))
    S2T.main()


if __name__ == "__main__":
    main()
