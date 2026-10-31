import numpy as np
import pytest
import torch
from PIL import Image

from test_s2 import H, W, box, load_script, wrapped
from treeground import masks as M
from treeground import mllm
from treeground import s2 as S2
from treeground import tree as T
from treeground import tree_attn as TA
from treeground.mllm import SEL

LAYERS = [2, 4]


@pytest.fixture(scope="module")
def s2t():
    return load_script("s2_train")


def scene_masks():
    # whole object, two parts, a sub-part of the first part, an unrelated region
    return [box(10, 90, 10, 70), box(10, 40, 10, 40), box(50, 90, 40, 70), box(12, 24, 12, 24), box(20, 80, 90, 125)]


def sequences(proc, sel_id, image):
    seqs = [S2.encode(proc, sel_id, "Locate the handle of the mug.", f"handle of the mug {SEL}", image),
            S2.encode(proc, sel_id, "Locate the bottle and then its cap.", f"bottle {SEL} cap {SEL}", image),
            S2.encode(proc, sel_id, "Find the handle of a mug.", f"handle of a mug {SEL}", None)]
    return seqs, S2.collate(seqs, proc.tokenizer.pad_token_id)


def nodes_for(seq, proc, masks, mode="tree", seed=None):
    rles = M.encode(np.stack(masks))
    grid = mllm.merged_grid(seq.image_grid_thw, proc.image_processor.merge_size)
    return TA.build_nodes(rles, T.build_tree(rles).parent, grid, mode, seed), rles, grid


def setup(s2t, seed=0):
    model, proc, extra = wrapped(s2t, seed)
    model.eval()
    image = Image.fromarray((np.random.default_rng(1).random((H, W, 3)) * 255).astype(np.uint8))
    seqs, batch = sequences(proc, extra["sel_id"], image)
    torch.manual_seed(seed)
    embed = TA.NodeEmbed(64, TA.embedding_rms(model))
    return model, proc, extra, seqs, batch, embed


def valid(batch):
    return batch["attention_mask"].bool()


def test_relatives_follow_the_tree():
    rles = M.encode(np.stack(scene_masks()))
    tree = T.build_tree(rles)
    assert tree.parent[:5].tolist() == [5, 0, 0, 1, 5]   # 5 is the root
    rel = TA.relatives(tree, "tree")
    assert rel[0].tolist() == [0, 1, 2, 4]                # self, children 1 2, sibling 4
    assert rel[1].tolist() == [0, 1, 2, 3]                # parent 0, sibling 2, child 3
    assert rel[3].tolist() == [1, 3]                      # parent only
    assert rel[4].tolist() == [0, 4]                      # top-level sibling 0
    assert all(len(r) == 5 for r in TA.relatives(tree, "full"))
    d = np.zeros((5, 5), bool)
    for v, r in enumerate(rel):
        d[v, r] = True
    assert (d == d.T).all()                               # tree relations are symmetric


def test_geometry_and_regions(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    nb, rles, (gh, gw) = nodes_for(seqs[0], proc, scene_masks())
    assert nb.n == 5 and nb.geom.shape == (5, TA.GEOM_DIM) and np.isfinite(nb.geom).all()
    assert np.allclose(nb.geom[0, :4], [10 / W, 10 / H, 70 / W, 90 / H])
    assert nb.geom[1, 8] < 0 and np.isclose(nb.geom[0, 8], nb.geom[0, 4])   # log area ratio to parent / image
    for v, m in enumerate(scene_masks()):
        own = np.flatnonzero(mllm.coverage(m, gh, gw) > 0)
        assert set(own) <= set(nb.region[v])                                # own cells always included
        grown = np.zeros((gh, gw), bool)
        for c in own:
            r, k = divmod(int(c), gw)
            grown[max(0, r - 1):r + 2, max(0, k - 1):k + 2] = True
        assert set(np.flatnonzero(grown)) == set(nb.region[v])              # exactly one cell of margin
        assert 0 <= nb.cell[v, 0] < gh and 0 <= nb.cell[v, 1] < gw


def test_attention_mask_structure(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    nb, _, _ = nodes_for(seqs[0], proc, scene_masks())
    nodes = [nb, None, None]
    am, ids = batch["attention_mask"], batch["input_ids"]
    img = ids == model.config.image_token_id
    L, K = ids.shape[1], nb.n
    allow = TA.attention_mask(am, img, nodes, K, torch.float32)[:, 0] == 0
    for b in range(3):
        n_tok = int(am[b].sum())
        assert torch.equal(allow[b, :n_tok, :n_tok], torch.tril(torch.ones(n_tok, n_tok, dtype=torch.bool)))
        assert not allow[b, :L, L:].any()                                    # nothing before sees mask tokens
        assert not allow[b, :n_tok, n_tok:L].any()                           # nor padding
    img_pos = torch.nonzero(img[0]).reshape(-1)
    n_tok = int(am[0].sum())
    rel = TA.dense_rel(nb)
    for v in range(K):
        row = allow[0, L + v]
        text = torch.ones(n_tok, dtype=torch.bool)
        text[img_pos] = False
        assert torch.equal(row[:n_tok] & ~torch.isin(torch.arange(n_tok), img_pos), text)
        assert set(torch.nonzero(row[img_pos]).reshape(-1).tolist()) == set(nb.region[v].tolist())
        assert np.array_equal(row[L:].numpy(), rel[v])
    assert not allow[1, L:, :].any(1).logical_not().any()                    # every row attends somewhere


def test_no_mask_tokens_matches_the_s2_forward(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    with torch.no_grad():
        ref_l, ref_last = S2.forward(model, batch, LAYERS[0])
        ref_hs = model.model(**batch, output_hidden_states=True, use_cache=False).hidden_states
        out = TA.forward(model, batch, [None, None, None], embed, LAYERS)
    m = valid(batch)
    assert torch.allclose(out.hidden[LAYERS[0]][m], ref_l[m], atol=1e-5)
    assert torch.allclose(out.hidden[LAYERS[1]][m], ref_hs[LAYERS[1]][m], atol=1e-5)
    assert torch.allclose(out.last[m], ref_last[m], atol=1e-5)
    assert all(len(v) == 0 for v in out.nodes[LAYERS[0]])


def test_mask_tokens_leave_text_states_unchanged(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    nb0, _, _ = nodes_for(seqs[0], proc, scene_masks())
    nb1, _, _ = nodes_for(seqs[1], proc, scene_masks()[:3])
    with torch.no_grad():
        base = TA.forward(model, batch, [None, None, None], embed, LAYERS)
        out = TA.forward(model, batch, [nb0, nb1, None], embed, LAYERS)
    m = valid(batch)
    for k in LAYERS:
        assert torch.allclose(out.hidden[k][m], base.hidden[k][m], atol=1e-5)
        assert [x.shape[0] for x in out.nodes[k]] == [5, 3, 0]
        assert torch.isfinite(out.nodes[k][0]).all()
    assert torch.allclose(out.last[m], base.last[m], atol=1e-5)


def test_node_order_does_not_matter(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    masks = scene_masks()
    perm = [3, 0, 4, 2, 1]
    nb, _, _ = nodes_for(seqs[0], proc, masks)
    nbp, _, _ = nodes_for(seqs[0], proc, [masks[i] for i in perm])
    sub = S2.collate(seqs[:1], proc.tokenizer.pad_token_id)
    with torch.no_grad():
        a = TA.forward(model, sub, [nb], embed, LAYERS)
        b = TA.forward(model, sub, [nbp], embed, LAYERS)
    for k in LAYERS:
        assert torch.allclose(a.nodes[k][0][perm], b.nodes[k][0], atol=1e-5)


def test_mask_tokens_only_read_their_region(s2t):
    # eager attention returns the probabilities: a mask token puts no weight outside its region and relatives
    model, proc, extra, seqs, batch, embed = setup(s2t)
    nb, _, _ = nodes_for(seqs[0], proc, scene_masks())
    model.config._attn_implementation = "eager"
    model.model.language_model.config._attn_implementation = "eager"
    captured = []
    hooks = [layer.self_attn.register_forward_hook(lambda mod, i, o: captured.append(o[1].detach()))
             for layer in model.model.language_model.layers]
    try:
        with torch.no_grad():
            TA.forward(model, batch, [nb, None, None], embed, LAYERS)
    finally:
        for h in hooks:
            h.remove()
    allowed = TA.attention_mask(batch["attention_mask"], batch["input_ids"] == model.config.image_token_id,
                                [nb, None, None], nb.n, torch.float32)[0, 0] == 0
    L = batch["input_ids"].shape[1]
    assert len(captured) == 4
    for probs in captured:
        rows = probs[0, :, L:L + nb.n]                                       # heads x nodes x keys
        assert torch.all(rows[:, ~allowed[L:L + nb.n]] == 0)
        assert torch.allclose(rows.sum(-1), torch.ones_like(rows.sum(-1)), atol=1e-5)


def test_random_tree_changes_relatives_not_regions(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    masks = scene_masks() + [box(30, 38, 30, 38), box(60, 70, 50, 60)]
    nb, _, _ = nodes_for(seqs[0], proc, masks)
    nr, _, _ = nodes_for(seqs[0], proc, masks, "random_tree", 7)
    assert all(np.array_equal(a, b) for a, b in zip(nb.region, nr.region))
    assert np.array_equal(nb.geom[:, :5], nr.geom[:, :5])                    # box and area are unchanged
    again, _, _ = nodes_for(seqs[0], proc, masks, "random_tree", 7)
    assert all(np.array_equal(a, b) for a, b in zip(nr.rel, again.rel))      # deterministic per seed
    with pytest.raises(ValueError):
        nodes_for(seqs[0], proc, masks, "random_tree", None)


def test_gradients_reach_lora_embed_and_heads(s2t):
    model, proc, extra, seqs, batch, embed = setup(s2t)
    model.train()
    nb, _, _ = nodes_for(seqs[0], proc, scene_masks())
    sel = TA.ScaledSelector(64, 64, 5)
    with S2.autocast("cpu"):
        out = TA.forward(model, batch, [nb, None, None], embed, LAYERS)
        q = sel.queries(out.hidden[LAYERS[1]][0, seqs[0].sel])
        e = sel.nodes(out.nodes[LAYERS[1]][0].float())
        loss = sel.scores(q, e).logsumexp(-1).sum()
    loss.backward()
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in embed.parameters())
    assert sel.log_scale.grad is not None and sel.bias.grad is not None
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for n, p in model.named_parameters() if "lora_B" in n)


def test_scaled_selector_initialisation():
    sel = TA.ScaledSelector(8, 8, 256)
    assert np.isclose(float(sel.log_scale.exp()), 1 / 0.07)
    assert np.isclose(float(torch.sigmoid(sel.bias)), 1 / 257)
    q = torch.nn.functional.normalize(torch.randn(2, 8), dim=-1)
    e = torch.nn.functional.normalize(torch.randn(3, 8), dim=-1)
    with torch.no_grad():
        sel.log_scale.fill_(10.0)
    assert torch.allclose(sel.scores(q, e), q @ e.T * 100.0 + sel.bias)


def test_pinned_transformers_is_enforced(monkeypatch):
    monkeypatch.setattr(TA.transformers, "__version__", "9.9.9")
    with pytest.raises(RuntimeError):
        TA.check_version()


def test_random_tree_really_changes_the_relatives():
    # the random-tree control must not leave the attention pattern intact
    masks = [box(10, 90, 10, 70), box(10, 40, 10, 40), box(50, 90, 40, 70), box(12, 24, 12, 24),
             box(20, 80, 90, 125), box(30, 38, 30, 38), box(60, 70, 50, 60), box(22, 60, 92, 120),
             box(25, 35, 95, 105), box(70, 78, 100, 118)]
    rles = M.encode(np.stack(masks))
    tree = T.build_tree(rles)
    true = TA.relatives(tree, "tree")
    for seed in (1, 2, 3):
        rand = T.randomised_tree(tree, seed)
        assert np.array_equal(rand.depth, tree.depth)                       # levels kept, links shuffled
        rel = TA.relatives(rand, "tree")
        assert sum(np.array_equal(a, b) for a, b in zip(true, rel)) <= len(masks) // 2
