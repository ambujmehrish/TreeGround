import argparse
import copy
import hashlib
import os
import sys
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import b_baselines as B  # noqa: E402
from treeground import proposals as P  # noqa: E402
from treeground.data import paco  # noqa: E402
from treeground.run import REPO, ProtocolError, load_config  # noqa: E402

# Deviation D31: stronger same-family and standard training-free baselines on the D13 protocol (same queries,
# same scoring, same caches layout; b_baselines.run_shard / merge are reused unchanged):
#   qwen8_box_sam / qwen32_box_sam  Qwen3-VL-8B / -32B-Instruct box (D13 prompt and parsing) + SAM 3 box prompt
#   sa2va_sam3                      Sa2VA-Qwen3-VL-4B-SAM3 (Qwen3-VL processor, D13 prompt, union of [SEG] masks)
#   sa2va_internvl3_14b             Sa2VA-InternVL3-14B (tokenizer interface of its model card)
#   grounded_sam2                   Grounding DINO base, top-1 box above its box/text thresholds (else abstain)
#                                   + SAM 2.1 hiera-large box prompt

USED = {"qwen8_box_sam": ["qwen8", "sam3"], "qwen32_box_sam": ["qwen32", "sam3"], "sa2va_sam3": ["sa2va_sam3"],
        "sa2va_internvl3_14b": ["sa2va_internvl3_14b"], "grounded_sam2": ["grounding_dino", "sam2"]}
METHODS = tuple(USED)


def load_qwen_any(m: dict, dtype: str, cache_dir: str | None):
    # b_baselines' loader pins the 4B dimensions; larger Qwen3-VL models are split over the visible GPUs
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model, info = Qwen3VLForConditionalGeneration.from_pretrained(
        m["repo"], revision=m["revision"], dtype=getattr(torch, dtype), attn_implementation="sdpa",
        cache_dir=cache_dir, device_map="auto", output_loading_info=True)
    if info.get("missing_keys") or info.get("mismatched_keys"):
        raise ProtocolError(f"{m['repo']} did not load cleanly: {info}")
    return model.eval(), AutoProcessor.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cache_dir)


class QwenBigBoxSam(B.QwenBoxSam):
    key = None

    def __init__(self, cfg: dict):
        sm, self.s = cfg["models"]["sam3"], cfg["baselines"]["qwen_box_sam"]
        self.device, self.dtype = cfg["device"], getattr(torch, cfg["proposals"]["dtype"])
        self.qwen, self.qproc = load_qwen_any(cfg["models"][self.key], cfg["features"]["dtype"], cfg.get("hf_home"))
        self.sam, self.sproc = P.load_sam3(sm["repo"], sm["revision"], self.device, cfg["proposals"]["dtype"],
                                           cfg.get("hf_home"))


class Qwen8BoxSam(QwenBigBoxSam):
    key = "qwen8"


class Qwen32BoxSam(QwenBigBoxSam):
    key = "qwen32"


class Sa2VASam3(B.Sa2VA):
    def __init__(self, cfg: dict):
        c = copy.deepcopy(cfg)
        c["models"]["sa2va"] = cfg["models"]["sa2va_sam3"]
        super().__init__(c)


class Sa2VAIntern:
    # Sa2VA model card usage for the InternVL3 variants: predict_forward(image, text, tokenizer)
    def __init__(self, cfg: dict):
        from transformers import AutoModel, AutoTokenizer
        m, self.s = cfg["models"]["sa2va_internvl3_14b"], cfg["baselines"]["sa2va"]
        with B.real_device_init():
            self.model = B.checked_load(AutoModel, m["repo"], m["revision"], cfg.get("hf_home"),
                                        dtype=getattr(torch, self.s["dtype"]), trust_remote_code=True)
        self.model = self.model.to(cfg["device"]).eval()
        self.tok = AutoTokenizer.from_pretrained(m["repo"], revision=m["revision"], cache_dir=cfg.get("hf_home"),
                                                 trust_remote_code=True, use_fast=False)

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrases: list[str]) -> list[tuple]:
        pil = Image.fromarray(image)
        out = []
        for p in phrases:
            r = self.model.predict_forward(image=pil, text=self.s["prompt"].format(phrase=p), past_text="",
                                           mask_prompts=None, tokenizer=self.tok)
            ms = [np.asarray(x)[0] for x in r["prediction_masks"]]
            m = np.logical_or.reduce(ms) if ms else None
            out.append((m, float("nan"), r["prediction"], m is not None))
        return out


class GroundedSam2:
    def __init__(self, cfg: dict):
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor, Sam2Model, Sam2Processor
        g, s2, self.s = cfg["models"]["grounding_dino"], cfg["models"]["sam2"], cfg["baselines"]["grounded_sam2"]
        self.device, hf = cfg["device"], cfg.get("hf_home")
        self.gproc = AutoProcessor.from_pretrained(g["repo"], revision=g["revision"], cache_dir=hf)
        self.gdino = B.checked_load(AutoModelForZeroShotObjectDetection, g["repo"], g["revision"], hf)
        self.gdino = self.gdino.to(self.device).eval()
        self.sproc = Sam2Processor.from_pretrained(s2["repo"], revision=s2["revision"], cache_dir=hf)
        self.sam = B.checked_load(Sam2Model, s2["repo"], s2["revision"], hf).to(self.device).eval()

    @staticmethod
    def caption(p: str) -> str:
        return p.lower().strip().rstrip(".") + "."

    @torch.no_grad()
    def __call__(self, image: np.ndarray, phrases: list[str]) -> list[tuple]:
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        out = []
        for p in phrases:
            inp = self.gproc(images=pil, text=self.caption(p), return_tensors="pt").to(self.device)
            det = self.gproc.post_process_grounded_object_detection(
                self.gdino(**inp), inp.input_ids, threshold=self.s["box_threshold"],
                text_threshold=self.s["text_threshold"], target_sizes=[(h, w)])[0]
            if len(det["scores"]) == 0:
                out.append((None, float("nan"), "", False))
                continue
            k = int(torch.argmax(det["scores"]))
            box = [float(v) for v in det["boxes"][k].tolist()]
            sin = self.sproc(images=pil, input_boxes=[[box]], return_tensors="pt").to(self.device)
            r = self.sam(**sin, multimask_output=False)
            m = self.sproc.post_process_masks(r.pred_masks.cpu(), sin["original_sizes"])[0][0, 0]
            m = np.asarray(m.numpy() if hasattr(m, "numpy") else m, bool)
            out.append((m, float(det["scores"][k]), str(det.get("text_labels", det.get("labels", ""))), True))
        return out


PREDICTORS = {"qwen8_box_sam": Qwen8BoxSam, "qwen32_box_sam": Qwen32BoxSam, "sa2va_sam3": Sa2VASam3,
              "sa2va_internvl3_14b": Sa2VAIntern, "grounded_sam2": GroundedSam2}
_identity = B.identity


def identity(cfg: dict, method: str, mhash: str, smoke: bool) -> dict:
    if method not in USED:
        return _identity(cfg, method, mhash, smoke)
    h = hashlib.sha256()
    for f in ("b_baselines.py", "b_baselines2.py"):
        h.update((REPO / "scripts" / f).read_bytes())
    settings = cfg["baselines"]["grounded_sam2" if method == "grounded_sam2" else
                                "sa2va" if method.startswith("sa2va") else "qwen_box_sam"]
    return {"kind": "baseline", "method": method, "smoke": smoke, "manifest": mhash, "settings": settings,
            "models": {k: cfg["models"][k] for k in USED[method]}, "dtype": cfg["proposals"]["dtype"],
            "templates": paco.TEMPLATES, "code": h.hexdigest()}


B.PREDICTORS.update(PREDICTORS)
B.identity = identity
_require = B.require_approved


def require(cfg: dict, ids: list[str]) -> None:
    _require(cfg, list(ids) + ["D31"])


B.require_approved = require


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--method", required=True, choices=METHODS)
    ap.add_argument("--shard", type=int, default=int(os.environ.get("SLURM_ARRAY_TASK_ID", 0)))
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config)
    n = 1 if args.smoke else cfg["baselines"]["num_shards"]
    if args.merge:
        B.merge(cfg, args.method, n, args.smoke)
    else:
        if not 0 <= args.shard < n:
            raise ProtocolError(f"--shard must be in 0..{n - 1}")
        B.run_shard(cfg, args.method, args.shard, n, args.smoke)


if __name__ == "__main__":
    main()
