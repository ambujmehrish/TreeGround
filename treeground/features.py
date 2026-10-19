import math

import cv2
import numpy as np
import torch
from PIL import Image

from treeground import masks as M
from treeground import mllm


def load_siglip(repo: str, revision: str, device: str, dtype: str, cache_dir: str | None = None):
    from transformers import SiglipImageProcessor, SiglipModel
    model = SiglipModel.from_pretrained(repo, revision=revision, dtype=getattr(torch, dtype), cache_dir=cache_dir)
    # the checkpoint ships the slow processor; pinned explicitly so outputs do not depend on the fast default
    processor = SiglipImageProcessor.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    return model.vision_model.to(device).eval(), processor, model.config.vision_config


def crop_box(box: np.ndarray, w: int, h: int) -> tuple[int, int, int, int]:
    x0, y0 = max(0, math.floor(box[0])), max(0, math.floor(box[1]))
    x1, y1 = min(w, max(x0 + 1, math.ceil(box[2]))), min(h, max(y0 + 1, math.ceil(box[3])))
    return x0, y0, x1, y1


@torch.no_grad()
def siglip_features(vision, image_processor, vcfg, image: np.ndarray, rles: list[M.Rle], boxes: np.ndarray,
                    batch: int) -> torch.Tensor:
    device = next(vision.parameters()).device
    dtype = next(vision.parameters()).dtype
    side = vcfg.image_size // vcfg.patch_size
    h, w = image.shape[:2]
    out = []
    for start in range(0, len(rles), batch):
        crops, weights = [], []
        for rle, box in zip(rles[start:start + batch], boxes[start:start + batch]):
            x0, y0, x1, y1 = crop_box(box, w, h)
            crops.append(Image.fromarray(image[y0:y1, x0:x1]))
            m = M.decode(rle)[y0:y1, x0:x1].astype(np.float32)
            weights.append(cv2.resize(m, (side, side), interpolation=cv2.INTER_AREA).reshape(-1))
        pixel = image_processor(images=crops, return_tensors="pt")["pixel_values"].to(device, dtype)
        feats = vision(pixel_values=pixel).last_hidden_state.float()
        if feats.shape[1] != side * side:
            raise RuntimeError(f"SigLIP returned {feats.shape[1]} patches, expected {side * side}")
        wt = torch.as_tensor(np.stack(weights), device=device)
        out.append((wt[..., None] * feats).sum(1) / wt.sum(1, keepdim=True).clamp(min=1e-12))
    return torch.cat(out) if out else torch.zeros(0, vcfg.hidden_size)


def image_features(model, processor, image: np.ndarray, rles: list[M.Rle], boxes: np.ndarray, layers: list[int],
                   feature_types: list[str], siglip=None, siglip_batch: int = 64) -> dict[str, torch.Tensor]:
    res = mllm.image_pass(model, processor, Image.fromarray(image), layers, "value" in feature_types)
    gh, gw = res["grid"]
    ptr, idx, wts = mllm.sparse_coverages((M.decode(r) for r in rles), gh, gw)
    out: dict[str, torch.Tensor] = {
        "cov_ptr": torch.from_numpy(ptr), "cov_idx": torch.from_numpy(idx), "cov_w": torch.from_numpy(wts),
        "grid": torch.tensor([gh, gw], dtype=torch.int64),
    }
    n = len(rles)
    for k in layers:
        if "hidden" in feature_types:
            out[f"hidden_L{k}"] = _pool(res["hidden"][k], ptr, idx, wts, n)
        if "value" in feature_types:
            out[f"value_L{k}"] = _pool(res["value"][k], ptr, idx, wts, n)
    if "encoder" in feature_types:
        out["encoder"] = _pool(res["encoder"], ptr, idx, wts, n)
    if "siglip" in feature_types:
        vision, image_processor, vcfg = siglip
        out["siglip"] = siglip_features(vision, image_processor, vcfg, image, rles, boxes, siglip_batch).half().cpu()
    return out


def _pool(states: torch.Tensor, ptr, idx, wts, n: int) -> torch.Tensor:
    if n == 0:
        return torch.zeros(0, states.shape[1], dtype=torch.float16)
    return mllm.pool(states, ptr, idx, wts).half().cpu()
