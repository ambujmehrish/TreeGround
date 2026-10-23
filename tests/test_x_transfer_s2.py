import numpy as np
import pandas as pd
import pytest
import torch
from PIL import Image

from test_s2 import H, LAYER, W, Props, box, load_script, wrapped
from treeground import s2 as S2
from treeground.data import paco
from treeground.run import ProtocolError, read_json, write_json
from treeground.selector import Selector


@pytest.fixture(scope="module")
def xs2():
    return load_script("x_transfer_s2")


def test_chain_query_has_the_s2_eval_layout(xs2):
    q = xs2.chain_query("ade:1|chain|car:wheel", "car", "wheel")
    text = paco.CHAIN_TEMPLATE.format(object="car", part="wheel")
    assert (q.kind, q.user) == ("chain", S2.user_text(text))
    assert q.answer == S2.answer_text("chain", "paco_lvis:1|chain|car:wheel", text)


def test_image_source_matches_s2_source_and_scores_every_chain(xs2, tmp_path):
    s2t = xs2.S2T
    model, proc, extra = wrapped(s2t)
    model.eval()
    rgb = (np.random.default_rng(0).random((H, W, 3)) * 255).astype(np.uint8)
    Image.fromarray(rgb).save(tmp_path / "img.png")
    masks = [box(10, 90, 20, 60), box(10, 30, 30, 50), box(0, 96, 0, 128)]
    props = Props({"s": masks})
    ref = s2t.Source({"proposals": {"node_dedup": 0.9}}, False, None, None, proc, extra["sel_id"], set())
    ref._props = props
    qs = [xs2.chain_query("x|chain|bottle:cap", "bottle", "cap"),
          xs2.chain_query("x|chain|mug:handle", "mug", "handle")]
    m = s2t.ImageMeta("s", "x", str(tmp_path / "img.png"), 3, None, None, None, None, qs)
    src = xs2.ImageSource(proc, extra["sel_id"], rgb, props.rles["s"])
    (seqs, grid, cov), (rseqs, rgrid, rcov) = src.image_item(None, qs), ref.image_item(m, qs)
    assert grid == rgrid and all(np.array_equal(a, b) for a, b in zip(cov, rcov))
    assert all(torch.equal(a.input_ids, b.input_ids) and torch.equal(a.pixel_values, b.pixel_values)
               for a, b in zip(seqs, rseqs))
    sel = Selector(64, 64).eval()
    sigs = xs2.chain_scores(model, src, sel, qs, LAYER, "cpu", 1)
    z, probes = s2t.recompute(model, ref, m, qs, LAYER, "cpu", 2)
    e = sel.nodes(z.half())
    for sig, p in zip(sigs, probes):
        assert sig.shape == (2, 3)
        np.testing.assert_allclose(sig, torch.sigmoid(sel.scores(sel.queries(p.half()), e)).detach().numpy(),
                                   atol=1e-5)


def make_eval(root, stage, name, s2_run, smoke=False, checkpoint="best.pt", step=10):
    d = root / f"{stage}_eval" / name
    d.mkdir(parents=True)
    write_json(d / "meta.json", {"smoke": smoke, "checkpoint": checkpoint})
    write_json(d / "metrics.json", {"smoke": smoke, "s2_run": s2_run, "checkpoint_step": step,
                                    "chains": {"tau_rej_chain": 0.4}})
    return d


def test_eval_run_takes_the_latest_eval_of_the_selected_checkpoint(xs2, tmp_path):
    cfg = {"runs_root": str(tmp_path), "inputs": {"selector_stage": "s1"}, "s2_eval": {"checkpoint": "best.pt"}}
    s2 = tmp_path / "s2_from_s1" / "r1"
    good = make_eval(tmp_path, "s2_from_s1", "a", "r1")
    make_eval(tmp_path, "s2_from_s1", "b", "r0")
    make_eval(tmp_path, "s2_from_s1", "c", "r1", checkpoint="step000250.pt")
    assert xs2.eval_run(cfg, "s2_from_s1", s2) == good
    with pytest.raises(ProtocolError):
        xs2.eval_run(cfg, "s2_from_s1", tmp_path / "s2_from_s1" / "r9")


def test_merge_writes_the_s2_stage_with_the_transfer_schema(xs2, tmp_path, monkeypatch):
    X = xs2.X
    for name in ("selector_run", "identity", "start_run"):
        monkeypatch.setattr(X, name, getattr(X, name))
    monkeypatch.setattr(X, "node_cap", lambda: {"k": 256, "order": "stability"})
    runs = tmp_path / "runs"
    s2 = runs / "s2" / "r1"
    s2.mkdir(parents=True)
    write_json(s2 / "meta.json", {"smoke": False})
    write_json(s2 / "metrics.json", {})
    ev = make_eval(runs, "s2", "e1", "r1")
    cfg = {"runs_root": str(runs), "cache_root": str(tmp_path / "cache"), "inputs": {"selector_stage": "v2"},
           "s2_eval": {"checkpoint": "best.pt"}, "transfer": {"num_shards": 1, "pascal_part": {"split": "val"}},
           "v4": {"seeds": [0], "bootstrap": {"n": 10, "seed": 0}},
           "models": {"sam3": {"repo": "s", "revision": "1"}, "qwen": {"repo": "q", "revision": "2"}},
           "proposals": {k: 0 for k in ("points_per_side", "masks_per_point", "stability_thresh", "stability_offset",
                                        "mask_threshold", "min_area", "node_dedup", "dtype")}}
    d = X.out_dir(cfg, "pascal_part", xs2.identity(cfg, "pascal_part", "s2", s2, ev))
    assert d != X.out_dir(cfg, "pascal_part", X.identity(cfg, "pascal_part", s2))
    (d / "shards").mkdir(parents=True)
    cov = pd.DataFrame([{"image": "a", "kind": "part", "label": "cap", "area_bin": "small", "best_iou": 0.7},
                        {"image": "a", "kind": "chain", "label": "bottle:cap", "area_bin": "small", "best_iou": 0.6,
                         "object_iou": 0.2, "feasible_uncapped": True, "part_ratio": 0.1}])
    pd.to_pickle({"coverage": cov, "strict": pd.DataFrame(), "d5": pd.DataFrame(), "d6": pd.DataFrame(),
                  "no_rejection": pd.DataFrame(), "tau": 0.4, "layer": LAYER, "images": 1}, d / "shards" / "00000.pkl")
    xs2.merge(cfg, "pascal_part", 1)
    (out,) = (runs / "x_transfer_pascal_part_s2_merge").glob("*/metrics.json")
    m = read_json(out)
    assert m["deviation"] == ["D7", "D17"] and (m["s2_run"], m["s2_eval_run"]) == ("r1", "e1")
    assert m["coverage_uncapped"]["part_recall"] == 1.0 and m["coverage_uncapped"]["chain_part_found"] == 1.0
    assert m["tau_rej_chain"] == 0.4 and m["natural_strict"] is None
    assert m["natural_no_rejection"].startswith("not reported")

