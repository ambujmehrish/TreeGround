from types import SimpleNamespace

import numpy as np
import pytest

from test_s2 import load_script
from treeground import mllm
from treeground.run import ProtocolError

ENV = ("TG_CACHE", "TG_DATA", "TG_ROOT", "TG_RUNS", "TG_SLURM_ACCOUNT")


@pytest.fixture
def env(monkeypatch):
    for v in ENV:
        monkeypatch.setenv(v, "/x")


def test_qwen8_launcher_relaxes_only_the_width_for_the_pinned_8b(env, monkeypatch):
    q8 = load_script("qwen8")
    monkeypatch.setattr(mllm, "EXPECTED_HIDDEN", mllm.EXPECTED_HIDDEN)
    for bad in ([], ["scripts/s2_train.py", "--config", "configs/ta_pool_full_8b.yaml"],
                ["scripts/ta_train.py", "--config", "configs/ta_pool_full.yaml"], ["scripts/ta_train.py"]):
        with pytest.raises(ProtocolError):
            q8.prepare(bad)
    assert mllm.EXPECTED_HIDDEN == 2560
    for cfg in ("configs/ta_pool_full_8b.yaml", "configs/ta_pool_full_8b_check.yaml", "configs/ta_test_pool_full_8b.yaml"):
        target, args = q8.prepare(["scripts/ta_test_eval.py", "--config", cfg, "--bench", "gres"])
        assert target == "scripts/ta_test_eval.py" and args == ["--config", cfg, "--bench", "gres"]
    assert mllm.EXPECTED_HIDDEN == 4096 and mllm.EXPECTED_VALUE_DIM == 1024
    assert q8.prepare(["scripts/ta_train.py", "--config=configs/ta_pool_full_8b.yaml"])[1] == [
        "--config=configs/ta_pool_full_8b.yaml"]


def test_gseval_sam3_clip_keeps_leading_words(env, monkeypatch):
    B = load_script("b_baselines")
    seen = []

    class Tok:
        model_max_length = 77

        def __call__(self, text):
            return SimpleNamespace(input_ids=[0] * (len(text.split()) + 2))   # one token per word + bos/eos

    class Fake:
        def __init__(self, cfg):
            self.proc = SimpleNamespace(tokenizer=Tok())

        def __call__(self, image, phrases):
            seen.extend(phrases)
            return [(None, 0.0, "", False) for _ in phrases]

    import sys
    import gres_eval as GE
    import gseval_eval
    monkeypatch.setitem(sys.modules, "b_baselines", B)
    monkeypatch.setitem(B.PREDICTORS, "sam3_text", Fake)
    for mod, names in ((GE, ("Baseline", "identity", "stage_name")), (gseval_eval, ("Baseline", "identity"))):
        for n in names:                                  # the wrapper patches these modules; restore them after
            monkeypatch.setattr(mod, n, getattr(mod, n))
    clip = load_script("gseval_sam3clip")
    gs = sys.modules["gseval_eval"]
    long = " ".join(f"w{i}" for i in range(40))
    out = gs.Baseline({}, "sam3_text")(np.zeros((2, 2, 3), np.uint8), ["a red mug", long])
    assert out == [{"pred": []}, {"pred": []}]
    assert seen[0] == "a red mug" and seen[1] == " ".join(f"w{i}" for i in range(30))
    assert gs.stage_name({}, "sam3_text") == "sam3_text_clip" and clip.METHOD == "sam3_text"
