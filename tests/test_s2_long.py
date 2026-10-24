import numpy as np
import pytest
import torch
from PIL import Image

from test_s2 import H, LAYER, W, ROOT, box, build_scene, load_script, make_source, wrapped
from treeground import masks as M
from treeground import tree as T
from treeground.data import paco, pascal_part
from treeground.mllm import SEL
from treeground.selector import Selector


@pytest.fixture(scope="module")
def s2t():
    return load_script("s2_train")


@pytest.fixture(scope="module")
def s2n():
    return load_script("s2_nodes")


def configs(monkeypatch):
    from treeground.run import load_config
    for v in ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT"):
        monkeypatch.setenv(v, "/x")
    return {n: load_config(ROOT / "configs" / f"{n}.yaml") for n in
            ("s2", "s2_s1", "s2_long", "x_transfer_s2", "x_transfer_s2_s1", "x_transfer_s2_long")}


def test_stage_names_and_deviations(s2t, monkeypatch):
    cfgs = configs(monkeypatch)
    assert {n: s2t.s2_stage(c) for n, c in cfgs.items()} == {
        "s2": "s2", "s2_s1": "s2_from_s1", "s2_long": "s2_from_s1_long", "x_transfer_s2": "s2",
        "x_transfer_s2_s1": "s2_from_s1", "x_transfer_s2_long": "s2_from_s1_long"}
    assert s2t.deviations(cfgs["s2"]) == s2t.deviations(cfgs["s2_s1"]) == ["D10"]
    assert s2t.deviations(cfgs["s2_long"]) == ["D10", "D21"]
    assert s2t.source_weights(s2t.stage_cfg(cfgs["s2_s1"], False)) == {"paco_train": 1.0}
    smoke = s2t.stage_cfg(cfgs["s2_long"], True)
    assert smoke["steps"] == 20 and set(smoke["sources"]) == {"paco_train", "pascal_part_train", "grefcoco_train"}
    full = s2t.stage_cfg(cfgs["s2_long"], False)
    assert (full["steps"], full["eval_every"], full["batch_images"]) == (12000, 1000, 32)
    new = ("steps", "eval_every", "stage_suffix", "sources")
    base = s2t.stage_cfg(cfgs["s2_s1"], False)
    assert {k: v for k, v in full.items() if k not in new} == {k: v for k, v in base.items() if k not in new}
    with pytest.raises(Exception):
        s2t.source_weights({"sources": {"paco_train": 1.0, "refcoco": 1.0}})


def pre_d21_plan(items, c, text_labels, world, rank):
    B, micro = c["batch_images"], c["micro_batch"]
    rng = np.random.default_rng(c["seed"])
    n_text = int(round(c["text_only_fraction"] * B))
    order, ptr = rng.permutation(len(items)), 0
    out = []
    share = B // world
    for step in range(c["steps"]):
        if ptr + B > len(order):
            order, ptr = rng.permutation(len(items)), 0
        idx = order[ptr:ptr + B]
        ptr += B
        batch, left = [], n_text
        for j, i in enumerate(idx):
            r = items[i]
            as_text = left > 0 and not r.absent and any(lab and lab in text_labels for lab in r.labels)
            left -= int(as_text)
            seed = int(np.random.SeedSequence([c["seed"], step, j]).generate_state(1)[0])
            batch.append((int(i), bool(as_text), seed))
        mine = batch[rank * share:(rank + 1) * share]
        out.extend(mine[k:k + micro] for k in range(0, share, micro))
    return out


def fake_items(s2t, sizes):
    out = []
    for source, n in sizes.items():
        for i in range(n):
            lab = np.array(["mug:handle" if i % 2 else ""], dtype=object)
            out.append(s2t.ImageMeta(f"{source}{i}", f"{source}:{i}", "", 1, None, lab, lab, np.zeros(1, bool),
                                     [], absent=i % 5 == 0, source=source))
    return out


def test_plan_single_source_matches_pre_d21(s2t):
    items = fake_items(s2t, {"paco_train": 23})
    c = {"batch_images": 8, "micro_batch": 2, "seed": 4, "text_only_fraction": 0.25, "steps": 9}
    for rank in range(2):
        assert s2t.plan(items, c, {"mug:handle"}, 2, rank) == pre_d21_plan(items, c, {"mug:handle"}, 2, rank)
    assert s2t.plan(items, {**c, "sources": {"paco_train": 1.0}}, set(), 1, 0) == pre_d21_plan(items, c, set(), 1, 0)


def test_plan_mixes_sources_by_weight(s2t):
    sizes = {"paco_train": 300, "pascal_part_train": 40, "grefcoco_train": 90}
    items = fake_items(s2t, sizes)
    w = {"paco_train": 0.6, "pascal_part_train": 0.2, "grefcoco_train": 0.2}
    c = {"batch_images": 32, "micro_batch": 4, "seed": 0, "text_only_fraction": 0.2, "steps": 400, "sources": w}
    full = s2t.plan(items, {**c, "micro_batch": 8}, {"mug:handle"}, 1, 0)
    ranks = [s2t.plan(items, c, {"mug:handle"}, 4, r) for r in range(4)]
    steps = [sum((ranks[r][2 * s] + ranks[r][2 * s + 1] for r in range(4)), []) for s in range(c["steps"])]
    assert steps == [sum(full[4 * s:4 * s + 4], []) for s in range(c["steps"])]
    drawn = [items[i].source for s in steps for i, _, _ in s]
    for k, v in w.items():
        assert abs(drawn.count(k) / len(drawn) - v) < 0.01
    pascal = [i for s in steps for i, _, _ in s if items[i].source == "pascal_part_train"]
    first = pascal[:sizes["pascal_part_train"] - 32]
    assert len(set(first)) == len(first)
    assert all(sum(t for _, t, _ in s) == 6 for s in steps)
    assert all(not items[i].absent and items[i].labels[0] for s in steps for i, t, _ in s if t)
    with pytest.raises(Exception):
        s2t.plan(items, {**c, "batch_images": 64, "micro_batch": 16}, set(), 4, 0)


def test_grefcoco_multi_target_and_no_target_targets(s2n):
    a, b, big = box(10, 40, 10, 40), box(50, 90, 70, 110), box(5, 95, 5, 120)
    rles = M.encode(np.stack([big, a, b, box(12, 40, 10, 40), box(0, 5, 0, 5)]))
    tree = T.build_tree(rles)
    qs = [("g|train|1", "referring", f"two cups {SEL}", M.encode(np.stack([a, b]))),
          ("g|train|2", "absent", f"a dog {SEL}", None),
          ("g|train|3", "referring", f"tiny {SEL}", M.encode(box(90, 96, 120, 128)))]
    present, absent, counts = s2n.targets(rles, tree, qs)
    assert counts == {"queries": 1, "absent": 1, "proposal_failures": 1}
    (q,) = present
    assert q["kind"] == "referring" and q["path"] is None and len(q["steps"]) == 1
    pos, ign = q["steps"][0]
    assert pos.tolist() == [1, 2] and ign.tolist() == [3]
    assert absent == [{"qid": "g|train|2", "text": f"a dog {SEL}"}]


def test_pascal_part_object_part_and_chain_targets(s2n):
    obj, part = box(10, 90, 20, 60), box(10, 30, 30, 50)
    rles = M.encode(np.stack([box(0, 96, 0, 128), obj, part, box(50, 90, 80, 120)]))
    tree = T.build_tree(rles)
    img = pascal_part.PPImage("2008_000001", H, W, [
        pascal_part.PPObject(0, "cat", M.encode(obj)[0], {"ear": M.encode(part)[0]})])
    (qid, o, p, orle, prle), = pascal_part.chains(img)
    qs = [(f"{img.key}|object|cat", "object", paco.OBJECT_TEMPLATE.format(object="cat"), [orle]),
          (f"{img.key}|part|cat:ear", "part", paco.PART_TEMPLATE.format(object="cat", part="ear"), [prle]),
          (qid, "chain", paco.CHAIN_TEMPLATE.format(object=o, part=p), [orle, prle])]
    present, absent, counts = s2n.targets(rles, tree, qs)
    assert counts == {"queries": 3} and not absent
    assert [q["steps"][0][0].tolist() for q in present] == [[1], [2], [1]]
    assert present[2]["path"] == (1, 2) and present[2]["steps"][1][0].tolist() == [2]


def test_node_cache_items_train_end_to_end(s2t, s2n, tmp_path, monkeypatch):
    items, props = build_scene(s2t, tmp_path)
    obj, part = box(10, 90, 20, 60), box(10, 30, 30, 50)
    rles = M.encode(np.stack([box(0, 96, 0, 128), obj, part, box(50, 90, 80, 120)]))
    ident = {"source": "grefcoco_train", "smoke": True}
    monkeypatch.setattr(s2t.S2N, "identity", lambda cfg, source, smoke: {**ident, "source": source})
    cfg = {"cache_root": str(tmp_path / "cache"), "s2_nodes": {"num_shards": {}}}
    file = tmp_path / "img_g.png"
    Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8)).save(file)
    d = s2n.out_dir(cfg, ident)
    (d / "nodes").mkdir(parents=True)
    (d / "shards").mkdir()
    qs = [("grefcoco|train|7", "referring", f"the cat {SEL}", M.encode(obj)),
          ("grefcoco|train|8", "absent", f"the dog {SEL}", None),
          ("pascal_part:1|chain|cat:ear", "chain", paco.CHAIN_TEMPLATE.format(object="cat", part="ear"),
           M.encode(np.stack([obj, part])))]
    image = np.zeros((H, W, 3), np.uint8)
    rec, counts = s2n.record(d, "grefcoco:5", str(file), image, rles, qs, None)
    assert counts["queries"] == 2 and counts["absent"] == 1
    assert s2n.record(d, "grefcoco:6", str(file), image, [], qs, None) == (None, {"proposal_failures": 2})
    torch.save({"records": [rec], "counts": dict(counts)}, d / "shards" / "00000.pt")
    metas, c = s2t.node_meta(cfg, "grefcoco_train", True)
    assert c["grefcoco_train:training_items"] == 2 and len(metas) == 2
    present, absent = metas
    assert present.source == absent.source == "grefcoco_train" and absent.absent and not present.absent
    assert [(q.kind, q.user, q.answer) for q in present.queries] == [
        ("object", "the cat", f"the cat {SEL}"), ("chain", "Locate the cat and then its ear.", f"cat {SEL} ear {SEL}")]
    assert present.queries[1].path == (1, 2) and absent.queries[0].answer == f"the dog {SEL}"
    assert absent.queries[0].steps[0].pos.size == 0
    assert present.parents.tolist() == T.build_tree(rles).parent.tolist()
    assert s2t.read_nodes(present.nodes)["rles"] == rles
    with pytest.raises(Exception):
        s2t.node_cache({**cfg, "s2_nodes": {"num_shards": {"grefcoco_train": 2}}}, "grefcoco_train", False)

    model, proc, extra = wrapped(s2t)
    model.train()
    for m in items[:2]:
        m.queries = [q for q in m.queries if q.kind == "chain"]
    present.queries = present.queries[:1]
    allm = items + metas
    src = make_source(s2t, proc, extra, props, set())
    mb = s2t.MicroBatches([[(i, False, k) for k, i in enumerate([0, 3, 4])]], allm, src)[0]
    assert [len(s["cov"][0]) - 1 for s in mb["samples"]] == [5, 4, 4]
    stats = {k: 0 for k in s2t.STAT_KEYS}
    l_sel, l_lm = s2t.micro_step(model, extra, Selector(64, 64), mb, allm, LAYER, "cpu", np.random.default_rng(0),
                                 2.0, stats)
    assert torch.isfinite(l_sel) and torch.isfinite(l_lm)
    assert stats["terms_chain"] == 2 and stats["terms_object"] == 1 and stats["terms_absent"] == 1
