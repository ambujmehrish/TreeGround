import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from treeground import lora as LR
from treeground import masks as M
from treeground import s2 as S2
from treeground.data import paco
from treeground.mllm import SEL
from treeground.selector import Selector

ROOT = Path(__file__).resolve().parents[1]
SPECIAL = ["<|endoftext|>", "<|im_start|>", "<|im_end|>", "<|vision_start|>", "<|vision_end|>", "<|image_pad|>",
           "<|video_pad|>"]
TEMPLATE = ("{% for message in messages %}<|im_start|>{{ message.role }}\n{% if message.content is string %}"
            "{{ message.content }}{% else %}{% for c in message.content %}{% if c.type == 'image' %}"
            "<|vision_start|><|image_pad|><|vision_end|>{% elif c.type == 'text' %}{{ c.text }}{% endif %}{% endfor %}"
            "{% endif %}<|im_end|>\n{% endfor %}{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")
LORA = {"r": 4, "alpha": 8, "dropout": 0.05, "targets": list(LR.TARGETS)}
LAYER = 2
H, W = 96, 128


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def s2t():
    return load_script("s2_train")


def tiny_processor():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    from transformers import PreTrainedTokenizerFast, Qwen3VLProcessor
    from transformers.models.qwen2_vl.image_processing_qwen2_vl import Qwen2VLImageProcessor
    from transformers.models.qwen3_vl.video_processing_qwen3_vl import Qwen3VLVideoProcessor
    tk = Tokenizer(models.BPE())
    tk.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tk.decoder = decoders.ByteLevel()
    corpus = ["Locate the bottle and then its cap. Find the handle of a mug. red car"] * 4
    tk.train_from_iterator(corpus, trainers.BpeTrainer(vocab_size=300, special_tokens=SPECIAL,
                                                       initial_alphabet=pre_tokenizers.ByteLevel.alphabet()))
    tok = PreTrainedTokenizerFast(tokenizer_object=tk, eos_token="<|im_end|>", pad_token="<|endoftext|>")
    ip = Qwen2VLImageProcessor(patch_size=16, merge_size=2, temporal_patch_size=2, min_pixels=32 * 32,
                               max_pixels=256 * 256)
    return Qwen3VLProcessor(image_processor=ip, tokenizer=tok, video_processor=Qwen3VLVideoProcessor(),
                            chat_template=TEMPLATE)


def tiny_model(processor, seed=0):
    from transformers import Qwen3VLConfig, Qwen3VLForConditionalGeneration
    ids = {t: processor.tokenizer.convert_tokens_to_ids(t) for t in SPECIAL}
    cfg = Qwen3VLConfig(
        text_config=dict(vocab_size=320, hidden_size=64, intermediate_size=96, num_hidden_layers=4,
                         num_attention_heads=4, num_key_value_heads=2, head_dim=16,
                         rope_parameters={"rope_type": "default", "rope_theta": 10000.0, "mrope_section": [2, 3, 3],
                                          "mrope_interleaved": True}),
        vision_config=dict(depth=2, hidden_size=32, intermediate_size=64, num_heads=2, out_hidden_size=64,
                           num_position_embeddings=64, deepstack_visual_indexes=[0]),
        image_token_id=ids["<|image_pad|>"], video_token_id=ids["<|video_pad|>"],
        vision_start_token_id=ids["<|vision_start|>"], vision_end_token_id=ids["<|vision_end|>"],
        tie_word_embeddings=True)
    torch.manual_seed(seed)
    return Qwen3VLForConditionalGeneration(cfg)


def wrapped(s2t, seed=0):
    proc = tiny_processor()
    model = tiny_model(proc, seed)
    torch.manual_seed(seed)
    model, proc, extra = s2t.wrap_model(model, proc, {"lora": LORA}, None)
    return model, proc, extra


def box(y0, y1, x0, x1):
    m = np.zeros((H, W), dtype=bool)
    m[y0:y1, x0:x1] = True
    return m


class Props:
    def __init__(self, masks: dict[str, list[np.ndarray]]):
        self.rles = {k: M.encode(np.stack(v)) for k, v in masks.items()}

    def nodes(self, sha1, tau):
        return {"h": H, "w": W, "rles": self.rles[sha1]}


def paco_image(image_id, obj, part, obj_mask, part_mask):
    img = paco.PacoImage(image_id, f"{image_id}.png", H, W)
    img.objects.append(paco.Obj(10 * image_id, obj, M.encode(obj_mask)[0]))
    img.parts.append(paco.Part(10 * image_id + 1, 10 * image_id, obj, part, M.encode(part_mask)[0]))
    return img


def query_rows(img, sha1):
    qs, _ = paco.i1_queries(img)
    return pd.DataFrame([{"qid": q.qid, "kind": q.kind, "text": q.text, "gt": json.dumps(q.gt), "sha1": sha1}
                         for q in qs])


@pytest.fixture()
def scene(s2t, tmp_path):
    return build_scene(s2t, tmp_path)


def build_scene(s2t, tmp_path):
    rng = np.random.default_rng(0)
    masks, items, files = {}, [], {}
    for i, (obj, part) in enumerate([("bottle", "cap"), ("mug", "handle")]):
        o, p = box(10, 90, 20, 60), box(10, 30, 30, 50)
        masks[f"s{i}"] = [o, p, box(50, 90, 80, 120), box(60, 70, 90, 100), box(0, 96, 0, 128)]
        files[f"s{i}"] = tmp_path / f"img{i}.png"
        Image.fromarray((rng.random((H, W, 3)) * 255).astype(np.uint8)).save(files[f"s{i}"])
    masks["s2"] = [box(5, 40, 5, 40), box(40, 90, 40, 120)]
    files["s2"] = tmp_path / "img2.png"
    Image.fromarray((rng.random((H, W, 3)) * 255).astype(np.uint8)).save(files["s2"])
    props = Props(masks)
    for i, (obj, part) in enumerate([("bottle", "cap"), ("mug", "handle")]):
        img = paco_image(i + 1, obj, part, masks[f"s{i}"][0], masks[f"s{i}"][1])
        row = SimpleNamespace(sha1=f"s{i}", image_key=f"paco_lvis:{i + 1}", path=str(files[f"s{i}"]))
        meta, counts = s2t.paco_meta(row, img, query_rows(img, row.sha1), props, 0.9, None)
        assert counts["queries_chain"] == 1 and counts["queries_object"] == 1 and counts["queries_part"] == 1
        items.append(meta)
    q = SimpleNamespace(qid="gref|n1", sha1="s2", image_key="coco:3", text=f"red car {SEL}")
    items.append(s2t.absent_meta(q, str(files["s2"]), props, 0.9))
    return items, props


def make_source(s2t, proc, extra, props, labels):
    src = s2t.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], labels)
    src._props = props
    return src


def test_answer_span_formats():
    assert S2.user_text(paco.CHAIN_TEMPLATE.format(object="bottle", part="cap")) == "Locate the bottle and then its cap."
    assert S2.user_text(paco.OBJECT_TEMPLATE.format(object="wine glass")) == "Locate the wine glass."
    assert S2.answer_text("object", "paco_lvis:1|object|wine_glass", "") == f"wine glass {SEL}"
    assert S2.answer_text("part", "paco_lvis:1|part|bottle:cap", "") == f"cap of the bottle {SEL}"
    assert S2.answer_text("chain", "paco_lvis:1|chain|bottle:cap", "") == f"bottle {SEL} cap {SEL}"
    assert S2.answer_text("text", "text|mug:handle", "", "mug:handle") == f"handle of a mug {SEL}"
    assert S2.answer_text("absent", "g", f"red car {SEL}") == f"red car {SEL}"


def test_lora_wraps_decoder_projections_only(s2t):
    model, proc, extra = wrapped(s2t)
    assert len(extra["wrapped"]) == 4 * len(LR.TARGETS)
    assert all(n.startswith("layers.") for n in extra["wrapped"])
    assert not any(isinstance(m, LR.LoRALinear) for m in model.model.visual.modules())
    assert extra["tied"]
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable == {n for n in trainable if ".lora_" in n or n.endswith("sel_row")}
    assert len(trainable) == 2 * 4 * len(LR.TARGETS) + 1


def test_sel_positions_and_answer_span(s2t):
    model, proc, extra = wrapped(s2t)
    sel_id = extra["sel_id"]
    image = Image.fromarray(np.zeros((H, W, 3), dtype=np.uint8))
    answer = S2.answer_text("chain", "paco_lvis:1|chain|bottle:cap", "")
    seq = S2.encode(proc, sel_id, "Locate the bottle and then its cap.", answer, image)
    a, e = seq.answer
    assert e == len(seq.input_ids)
    assert proc.tokenizer.decode(seq.input_ids[a:e]) == answer
    assert proc.tokenizer.decode(seq.input_ids[:a]).endswith("<|im_start|>assistant\n")
    assert seq.sel.tolist() == torch.nonzero(seq.input_ids == sel_id).reshape(-1).tolist()
    assert len(seq.sel) == 2 and int(seq.sel[-1]) == e - 1 and int(seq.sel[0]) > a
    assert (seq.input_ids[:a] != sel_id).all()
    assert int((seq.input_ids == model.config.image_token_id).sum()) == 3 * 4
    batch = S2.collate([seq, S2.encode(proc, sel_id, "Find the handle of a mug.", f"handle of a mug {SEL}", None)],
                       proc.tokenizer.pad_token_id)
    assert batch["attention_mask"][1].sum() < batch["attention_mask"][0].sum()
    assert batch["image_grid_thw"].shape == (1, 3)


def test_training_step_end_to_end(s2t, scene):
    items, props = scene
    for m in items[:2]:
        m.queries = [q for q in m.queries if q.kind == "chain"]
        assert m.queries[0].path == (0, 1)
    model, proc, extra = wrapped(s2t)
    model.train()
    assert all(layer.gradient_checkpointing for layer in LR.decoder(model).layers)
    selector = Selector(64, 64)
    labels = {"bottle:cap", "mug:handle"}
    src = make_source(s2t, proc, extra, props, labels)
    c = {"batch_images": 3, "micro_batch": 3, "seed": 0, "text_only_fraction": 0.34, "steps": 1}
    entries = s2t.plan(items, c, labels, 1, 0)
    assert len(entries) == 1 and sorted(i for i, _, _ in entries[0]) == [0, 1, 2]
    assert sum(t for _, t, _ in entries[0]) == 1
    mb = s2t.MicroBatches(entries, items, src)[0]
    assert len(mb["seqs"]) == 4 and "pixel_values" in mb["batch"]
    stats = {k: 0 for k in s2t.STAT_KEYS}
    l_sel, l_lm = s2t.micro_step(model, extra, selector, mb, items, LAYER, "cpu", np.random.default_rng(0), 2.0,
                                 stats)
    assert torch.isfinite(l_sel) and torch.isfinite(l_lm) and l_lm > 0
    assert stats["terms_text"] == 1 and stats["terms_absent"] == 1 and stats["terms_chain"] == 2
    (l_sel + l_lm).backward()
    params = LR.trainable_parameters(model, extra["sel_emb"]) + list(selector.parameters())
    assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
    assert extra["sel_emb"].sel_row.grad.abs().sum() > 0
    assert any(p.grad.abs().sum() > 0 for n, p in model.named_parameters() if n.endswith("lora_B"))
    assert all(p.grad is None for n, p in model.named_parameters() if not p.requires_grad)
    s2t.sync_grads(params, 1)
    opt = torch.optim.AdamW(params, lr=1e-3)
    opt.step()
    assert all(torch.isfinite(p).all() for p in params)
    with torch.no_grad():
        val = [s2t.ImageMeta(m.sha1, m.key, m.file, m.n, None, None, None, None,
                             [s2t.Query(q.qid, q.kind, q.user, q.answer, [], None, np.linspace(0, 1, m.n))
                              for q in m.queries]) for m in items[:2]]
    res = s2t.val_success(model, selector, src, val, LAYER, "cpu", 1, 2)
    assert res["queries"] == 2 and 0.0 <= res["selector_success"] <= 1.0


def test_recompute_matches_single_sequences(s2t, scene):
    items, props = scene
    model, proc, extra = wrapped(s2t)
    model.eval()
    src = make_source(s2t, proc, extra, props, set())
    m = items[0]
    z, probes = s2t.recompute(model, src, m, m.queries, LAYER, "cpu", 2)
    assert z.shape == (m.n, 64) and [p.shape[0] for p in probes] == [q.answer.count(SEL) for q in m.queries]
    z1, p1 = s2t.recompute(model, src, m, m.queries[-1:], LAYER, "cpu", 1)
    assert torch.allclose(z, z1, atol=1e-2) and torch.allclose(probes[-1], p1[0], atol=1e-2)


def test_checkpoint_round_trip(s2t, tmp_path):
    model, proc, extra = wrapped(s2t)
    selector = Selector(64, 64)
    with torch.no_grad():
        for p in LR.trainable_parameters(model, extra["sel_emb"]):
            p.add_(torch.randn_like(p))
    torch.save(LR.trainable_state(model, extra["sel_emb"], selector), tmp_path / "ck.pt")
    other, _, extra2 = wrapped(s2t)
    sel2 = Selector(64, 64)
    LR.load_trainable(other, extra2["sel_emb"], sel2, torch.load(tmp_path / "ck.pt"))
    a = dict(model.named_parameters())
    for n, p in other.named_parameters():
        assert torch.equal(p, a[n]), n
    for k, v in selector.state_dict().items():
        assert torch.equal(v, sel2.state_dict()[k])
    model.eval()
    other.eval()
    seq = S2.encode(proc, extra["sel_id"], "Locate the bottle.", f"bottle {SEL}", None)
    batch = S2.collate([seq], proc.tokenizer.pad_token_id)
    with torch.no_grad():
        assert torch.equal(S2.forward(model, batch, LAYER)[0], S2.forward(other, batch, LAYER)[0])


def test_plan_partitions_the_global_batch(s2t, scene):
    items, _ = scene
    c = {"batch_images": 2, "micro_batch": 1, "seed": 3, "text_only_fraction": 0.0, "steps": 5}
    full = s2t.plan(items, {**c, "micro_batch": 2}, set(), 1, 0)
    r0, r1 = s2t.plan(items, c, set(), 2, 0), s2t.plan(items, c, set(), 2, 1)
    assert [a + b for a, b in zip(r0, r1)] == full
    assert full == s2t.plan(items, {**c, "micro_batch": 2}, set(), 1, 0)
    with pytest.raises(Exception):
        s2t.plan(items, {**c, "batch_images": 3}, set(), 2, 0)


def test_recomputed_cache_follows_query_offsets(s2t, scene, tmp_path):
    s2e = load_script("s2_eval")
    v2 = load_script("v2_train_eval")
    items, props = scene
    model, proc, extra = wrapped(s2t)
    model.eval()
    src = make_source(s2t, proc, extra, props, set())
    rows = []
    for m in items:
        for q in m.queries:
            text = f"{q.user} {SEL}" if q.kind == "absent" else q.user
            rows.append({"qid": q.qid, "kind": q.kind, "text": text, "sha1": m.sha1, "split": "val",
                         "image_key": m.key, "n_sel": q.answer.count(SEL)})
    queries = pd.DataFrame(rows).sort_values(["sha1", "qid"]).reset_index(drop=True)
    queries["offset"] = queries.groupby("sha1")["n_sel"].cumsum() - queries["n_sel"]
    wl = pd.DataFrame({"sha1": [m.sha1 for m in items], "path": [m.file for m in items],
                       "image_key": [m.key for m in items]})
    out = tmp_path / "recomputed"
    assert s2e.recompute_all(model, src, queries, wl, LAYER, "cpu", out, 2, 0, 1) == 3
    store, probes = s2e.RecomputedStore(out), v2.Probes(out, 0)
    for m in items:
        assert store.load(m.sha1, [f"hidden_L{LAYER}"])[f"hidden_L{LAYER}"].shape == (m.n, 64)
        byq = {q.qid: q for q in m.queries}
        for q in queries[queries.sha1 == m.sha1].itertuples():
            _, ref = s2t.recompute(model, src, m, [byq[q.qid]], LAYER, "cpu", 1)
            got = probes.get(m.sha1, q.offset, q.n_sel)
            assert got.shape == (q.n_sel, 64)
            assert torch.allclose(got.float(), ref[0], atol=2e-2)
