from dataclasses import dataclass

import numpy as np
import torch

from treeground import masks as M

ROUTE = {
    "model_class": "transformers.Sam3TrackerModel",
    "processor_class": "transformers.Sam3TrackerProcessor",
    "entry_point": "Sam3TrackerModel.get_image_embeddings + Sam3TrackerModel.forward(multimask_output=True) "
                   "+ Sam3TrackerProcessor.post_process_masks(binarize=False); explicit point grid, no NMS, "
                   "no predicted-IoU filter, no crop-edge filter, no hole/sprinkle filling",
}


@dataclass
class ProposalConfig:
    points_per_side: int = 32
    masks_per_point: int = 3
    stability_thresh: float = 0.9
    stability_offset: float = 1.0
    mask_threshold: float = 0.0
    min_area: int = 64
    points_per_batch: int = 64
    dtype: str = "float32"


def load_sam3(repo: str, revision: str, device: str, dtype: str, cache_dir: str | None = None):
    from transformers import Sam3TrackerModel, Sam3TrackerProcessor
    model = Sam3TrackerModel.from_pretrained(repo, revision=revision, dtype=getattr(torch, dtype), cache_dir=cache_dir)
    processor = Sam3TrackerProcessor.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    return model.to(device).eval(), processor


def point_grid(n_per_side: int, w: int, h: int) -> np.ndarray:
    # Same lattice as the SAM automatic mask generator (cell centres of an n x n grid), in pixels.
    offset = 1.0 / (2 * n_per_side)
    side = np.linspace(offset, 1 - offset, n_per_side)
    xs, ys = np.meshgrid(side, side)
    return np.stack([xs.ravel() * w, ys.ravel() * h], axis=-1)


def stability_scores(logits: torch.Tensor, thr: float, offset: float) -> torch.Tensor:
    hi = (logits > thr + offset).flatten(-2).sum(-1, dtype=torch.int64)
    lo = (logits > thr - offset).flatten(-2).sum(-1, dtype=torch.int64)
    return hi.double() / lo.clamp(min=1).double()


@torch.no_grad()
def generate(model, processor, image: np.ndarray, cfg: ProposalConfig) -> dict:
    h, w = image.shape[:2]
    device = next(model.parameters()).device
    dtype = getattr(torch, cfg.dtype)
    inputs = processor(images=image, return_tensors="pt")
    embeddings = model.get_image_embeddings(inputs["pixel_values"].to(device, dtype))
    target = processor.target_size
    grid = point_grid(cfg.points_per_side, w, h)
    norm = grid * np.array([target / w, target / h])
    rles, sam_iou, stab, src = [], [], [], []
    for start in range(0, len(norm), cfg.points_per_batch):
        pts = torch.tensor(norm[start:start + cfg.points_per_batch], dtype=dtype, device=device)[None, :, None, :]
        labels = torch.ones(pts.shape[:3], dtype=torch.int64, device=device)
        out = model(image_embeddings=embeddings, input_points=pts, input_labels=labels, multimask_output=True)
        if out.pred_masks.shape[2] != cfg.masks_per_point:
            raise RuntimeError(f"expected {cfg.masks_per_point} masks per point, got {out.pred_masks.shape[2]}")
        logits = processor.post_process_masks([out.pred_masks[0].float()], [[h, w]], binarize=False)[0]
        s = stability_scores(logits, cfg.mask_threshold, cfg.stability_offset)
        binary = logits > cfg.mask_threshold
        area = binary.flatten(-2).sum(-1)
        keep = (s >= cfg.stability_thresh) & (area >= cfg.min_area)
        p_idx, m_idx = torch.nonzero(keep, as_tuple=True)
        if len(p_idx):
            rles += M.encode(binary[p_idx, m_idx].cpu().numpy())
            sam_iou += out.iou_scores[0][p_idx, m_idx].float().cpu().tolist()
            stab += s[p_idx, m_idx].cpu().tolist()
            src += ((p_idx + start) * cfg.masks_per_point + m_idx).cpu().tolist()
    return {
        "rles": rles,
        "areas": M.areas(rles),
        "boxes": M.boxes_xyxy(rles),
        "sam_iou": np.asarray(sam_iou, dtype=np.float64),
        "stability": np.asarray(stab, dtype=np.float64),
        "src_index": np.asarray(src, dtype=np.int64),
        "n_points": len(grid),
    }


def dedup(rles: list[M.Rle], areas: np.ndarray, stability: np.ndarray, src_index: np.ndarray,
          thresh: float) -> np.ndarray:
    # Greedy merge of pairs with IoU >= thresh keeping the larger mask: visit in descending area
    # (ties: higher stability, lower proposal index); a mask is dropped if it duplicates a kept one.
    order = np.lexsort((src_index, -stability, -areas))
    kept: list[int] = []
    for i in order:
        # IoU >= t with a mask at least as large requires area_i >= t * area_j
        cand = [j for j in kept if areas[i] >= thresh * areas[j]]
        if cand and (M.iou([rles[i]], [rles[j] for j in cand]) >= thresh).any():
            continue
        kept.append(int(i))
    mask = np.zeros(len(rles), dtype=bool)
    mask[kept] = True
    return mask
