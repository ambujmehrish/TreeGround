import math

import numpy as np
from PIL import Image

# CoCoEdit masked-benchmark protocol (deviation D25), as in wyh6666/GEditBench_ImgEditBench_with_mask
# {geditbench,imgedit}/run_psnr_score.py: RGB float32 in [0, 1], white mask (> 128) = preserved region

PSNR_CAP = 100.0
PSNR_CLIP = 50.0

# CoCoEdit (arXiv 2602.14068) published rows: (method, table, Overall, PSNR, SSIM, LPIPS)
PUBLISHED = {
    "gedit_full": [
        ("BAGEL", "Table 1", 6.940, 20.783, 0.745, 0.203),
        ("OmniGen2", "Table 1", 6.410, 21.116, 0.736, 0.197),
        ("Step1X-Edit", "Table 1", 7.379, 21.588, 0.750, 0.189),
        ("FLUX.1 Kontext", "Table 1", 6.286, 24.168, 0.825, 0.150),
        ("FLUX.1 Kontext + Edit-R1", "Table 1", 7.113, 19.013, 0.716, 0.214),
        ("FLUX.1 Kontext + CoCoEdit", "Table 1", 6.939, 25.331, 0.874, 0.139),
        ("Qwen-Image-Edit (2509)", "Table 1", 7.560, 19.488, 0.662, 0.185),
        ("Qwen-Image-Edit + Edit-R1", "Table 1", 7.746, 18.441, 0.639, 0.214),
        ("Qwen-Image-Edit + MotionNFT", "Table 1", 7.711, 18.709, 0.642, 0.201),
        ("Qwen-Image-Edit + CoCoEdit", "Table 1", 7.754, 22.283, 0.774, 0.162),
        ("GPT-Image-1", "Table 5", 7.861, 12.285, 0.406, None),
        ("Nano-Banana", "Table 5", 7.062, 19.540, 0.640, None),
        ("FLUX.1-Fill-dev (GT mask)", "Table 7", 3.258, 26.793, 0.786, None),
        ("ICEdit", "Table 7", 5.898, 21.698, 0.730, None),
    ],
    "imgedit_full": [
        ("BAGEL", "Table 2", 3.12, 20.062, 0.611, 0.318),
        ("OmniGen2", "Table 2", 3.24, 17.482, 0.578, 0.364),
        ("Step1X-Edit", "Table 2", 3.57, 16.471, 0.468, 0.369),
        ("FLUX.1 Kontext", "Table 2", 3.43, 19.591, 0.592, 0.303),
        ("FLUX.1 Kontext + Edit-R1", "Table 2", 3.51, 15.722, 0.506, 0.356),
        ("FLUX.1 Kontext + CoCoEdit", "Table 2", 3.52, 20.747, 0.635, 0.297),
        ("Qwen-Image-Edit (2509)", "Table 2", 3.70, 17.635, 0.526, 0.359),
        ("Qwen-Image-Edit + Edit-R1", "Table 2", 3.72, 16.130, 0.482, 0.394),
        ("Qwen-Image-Edit + MotionNFT", "Table 2", 3.71, 17.072, 0.514, 0.369),
        ("Qwen-Image-Edit + CoCoEdit", "Table 2", 3.79, 19.125, 0.555, 0.331),
        ("GPT-Image-1", "Table 5", 3.80, 12.209, 0.308, None),
        ("Nano-Banana", "Table 5", 3.76, 16.926, 0.449, None),
        ("FLUX.1-Fill-dev (GT mask)", "Table 7", 2.04, 26.463, 0.701, None),
        ("ICEdit", "Table 7", 2.88, 21.561, 0.614, None),
    ],
}


def read_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img).astype(np.float32) / 255.0


def to_source_size(edited: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    # released scripts: (x * 255).astype(uint8) -> PIL BILINEAR to the source size -> / 255
    if edited.shape[:2] == shape:
        return edited
    u8 = (edited * 255).astype(np.uint8)
    return np.asarray(Image.fromarray(u8).resize(shape[::-1], Image.BILINEAR)) / 255.0


def sq_error(edited: np.ndarray, source: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    m = mask.astype(np.float32)
    return float(((edited - source) ** 2 * m[..., None]).sum()), float(m.sum() * 3)


def psnr(edited: np.ndarray, source: np.ndarray, mask: np.ndarray) -> float:
    m = mask.astype(np.float32)
    if m.sum() == 0:
        return float("nan")
    mse = ((edited - source) ** 2 * m[..., None]).sum() / (m.sum() * 3)  # float32 throughout, as released
    return PSNR_CAP if mse == 0 else float(10 * np.log10(1.0 / mse))


def pooled_psnr(num: np.ndarray, den: np.ndarray) -> float:
    n, d = float(np.sum(num)), float(np.sum(den))
    if d == 0:
        return float("nan")
    return PSNR_CAP if n == 0 else 10 * math.log10(d / n)


def _ssim_kw() -> dict:
    return {"data_range": 1.0, "win_size": 11, "gaussian_weights": True, "use_sample_covariance": False}


def ssim(edited: np.ndarray, source: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    from skimage.metrics import structural_similarity
    # released: the scripts pass mask=..., which structural_similarity swallows in **kwargs, so it is the full-image
    # SSIM; masked: the same SSIM maps averaged over the mask pixels inside the border crop skimage's own mean uses
    pad = (11 - 1) // 2
    m = np.zeros(mask.shape, bool)
    m[pad:-pad, pad:-pad] = mask[pad:-pad, pad:-pad] > 0
    released, masked = 0, []
    for c in range(3):
        mssim, smap = structural_similarity(edited[..., c], source[..., c], mask=mask, full=True, **_ssim_kw())
        released += mssim
        masked.append(smap[m].mean(dtype=np.float64) if m.any() else float("nan"))
    return float(released / 3), float(np.mean(masked))


def preserved_composite(edited: np.ndarray, source: np.ndarray, mask: np.ndarray) -> np.ndarray:
    # LPIPS input: the GT edit region (mask == 0) replaced by source pixels
    m = mask.astype(bool)[..., None]
    return np.where(m, edited, source)


class Lpips:
    def __init__(self, net: str, device: str):
        import lpips
        import torch
        self.torch, self.device = torch, device
        self.model = lpips.LPIPS(net=net, verbose=False).to(device).eval()

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        def t(x):
            return self.torch.from_numpy(np.ascontiguousarray(x)).permute(2, 0, 1)[None].float().to(self.device) * 2 - 1
        with self.torch.no_grad():
            return float(self.model(t(a), t(b)).item())


def sample_metrics(edited: np.ndarray, source: np.ndarray, mask: np.ndarray, lp: Lpips | None) -> dict:
    edited = to_source_size(edited, source.shape[:2])
    num, den = sq_error(edited, source, mask)
    p = psnr(edited, source, mask)
    s_full, s_mask = ssim(edited, source, mask)
    return {"psnr": p, "psnr_capped": min(p, PSNR_CLIP) if not math.isnan(p) else p, "at_cap": p == PSNR_CAP,
            "ssim": s_full, "ssim_masked": s_mask,
            "lpips": lp(preserved_composite(edited, source, mask), source) if lp is not None else float("nan"),
            "sq_err": num, "sq_count": den, "mask_frac": float(mask.mean())}


def nanmean(v) -> float:
    v = np.asarray(v, np.float64)
    v = v[~np.isnan(v)]
    return float(v.mean()) if len(v) else float("nan")


def category_overall(scores: dict[str, list[float]], categories: tuple[str, ...], ndigits: int | None = None
                     ) -> tuple[float, dict[str, float]]:
    # per-category mean (unparseable samples dropped), then the unweighted mean over the listed categories
    per = {}
    for c in categories:
        m = nanmean(scores.get(c, []))
        per[c] = round(m, ndigits) if ndigits is not None and not math.isnan(m) else m
    vals = [per[c] for c in categories]
    return (float("nan") if any(math.isnan(v) for v in vals) else float(np.mean(vals))), per


def imgedit_score(text: str) -> float:
    # ImgEdit Benchmark/Basic/step1_get_avgscore.py: every "<name>: <integer>" line, averaged, rounded to 2
    scores = []
    for line in text.splitlines():
        parts = line.strip().split(": ")
        if len(parts) == 2 and parts[1].isdigit():
            scores.append(int(parts[1]))
    return round(sum(scores) / len(scores), 2) if scores else float("nan")
