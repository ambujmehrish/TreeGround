import math

import cv2
import numpy as np
import torch
import torch.nn.functional as F

PSNR_CAP = 100.0  # identical pixels; same cap as the GEdit/ImgEdit masked-PSNR scripts
SSIM_WIN, SSIM_SIGMA = 11, 1.5
SSIM_C1, SSIM_C2 = 0.01 ** 2, 0.03 ** 2


def to_float(img: np.ndarray) -> np.ndarray:
    return np.asarray(img, np.float64) / 255.0


def dilate(mask: np.ndarray, px: int) -> np.ndarray:
    if px <= 0:
        return mask.astype(bool)
    k = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * px + 1, 2 * px + 1))
    return cv2.dilate(mask.astype(np.uint8), k) > 0


def blend_alpha(mask: np.ndarray, dilate_px: int, feather_px: float) -> np.ndarray:
    d = dilate(mask, dilate_px).astype(np.float32)
    a = cv2.GaussianBlur(d, (0, 0), sigmaX=feather_px) if feather_px > 0 else d
    return np.maximum(a, mask.astype(np.float32))  # the selected region itself is always fully edited


def composite(orig: np.ndarray, edited: np.ndarray, alpha: np.ndarray) -> np.ndarray:
    a = alpha[..., None].astype(np.float64)
    return np.clip(np.rint(a * edited + (1 - a) * orig), 0, 255).astype(np.uint8)


def masked_psnr(a: np.ndarray, b: np.ndarray, region: np.ndarray) -> float:
    if not region.any():
        return float("nan")
    mse = float(((to_float(a) - to_float(b)) ** 2)[region].mean())
    return PSNR_CAP if mse == 0 else min(PSNR_CAP, 10 * math.log10(1.0 / mse))


def ssim_map(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    x = torch.from_numpy(to_float(a)).permute(2, 0, 1)[None]
    y = torch.from_numpy(to_float(b)).permute(2, 0, 1)[None]
    r = torch.arange(SSIM_WIN, dtype=torch.float64) - SSIM_WIN // 2
    g = torch.exp(-r ** 2 / (2 * SSIM_SIGMA ** 2))
    g = g / g.sum()
    k = (g[:, None] * g[None, :]).expand(3, 1, SSIM_WIN, SSIM_WIN).contiguous()
    p = SSIM_WIN // 2

    def f(t):
        return F.conv2d(F.pad(t, (p, p, p, p), mode="reflect"), k, groups=3)
    mx, my = f(x), f(y)
    sxx, syy, sxy = f(x * x) - mx ** 2, f(y * y) - my ** 2, f(x * y) - mx * my
    s = ((2 * mx * my + SSIM_C1) * (2 * sxy + SSIM_C2)) / ((mx ** 2 + my ** 2 + SSIM_C1) * (sxx + syy + SSIM_C2))
    return s[0].mean(0).numpy()


def masked_mean(values: np.ndarray, region: np.ndarray) -> float:
    return float(values[region].mean()) if region.any() else float("nan")


def abs_change(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return np.abs(to_float(a) - to_float(b)).mean(-1)


def directional(img_src: np.ndarray, img_edit: np.ndarray, txt_src: np.ndarray, txt_tgt: np.ndarray) -> float:
    # CLIP directional similarity (StyleGAN-NADA / InstructPix2Pix): cosine of image and caption deltas of
    # L2-normalised embeddings; an unchanged image has no direction and scores 0
    def unit(v):
        v = np.asarray(v, np.float64)
        return v / max(np.linalg.norm(v), 1e-12)
    di, dt = unit(img_edit) - unit(img_src), unit(txt_tgt) - unit(txt_src)
    n = np.linalg.norm(di) * np.linalg.norm(dt)
    return float(di @ dt / n) if n > 1e-12 else 0.0


def cosine(a: np.ndarray, b: np.ndarray) -> float:
    a, b = np.asarray(a, np.float64), np.asarray(b, np.float64)
    return float(a @ b / max(np.linalg.norm(a) * np.linalg.norm(b), 1e-12))


def mask_iou(a: np.ndarray, b: np.ndarray) -> float:
    u = np.logical_or(a, b).sum()
    return float(np.logical_and(a, b).sum() / u) if u else 0.0


class Lpips:
    def __init__(self, net: str, device: str):
        import lpips
        self.device = device
        self.model = lpips.LPIPS(net=net, spatial=True, verbose=False).to(device).eval()

    @torch.no_grad()
    def __call__(self, a: np.ndarray, b: np.ndarray) -> np.ndarray:
        def t(img):
            return (torch.from_numpy(to_float(img)).permute(2, 0, 1)[None].float() * 2 - 1).to(self.device)
        return self.model(t(a), t(b))[0, 0].float().cpu().numpy()
