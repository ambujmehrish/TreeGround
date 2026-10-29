import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))

import b_baselines as B  # noqa: E402
import b_baselines2 as B2  # noqa: E402

# Deviation D40 (baseline parity), Group C predictors: each baseline at its official inference setting, returning
# every instance it predicts with its score (highest first) instead of the D13 / D31 single mask:
#   sam3_union      SAM 3 text, query verbatim, every instance; official output = union of instances with score > 0.5
#                   (Sam3Processor.post_process_instance_segmentation threshold 0.5, as OP-HRG's SAM3 baseline)
#   qwen_box_all    Qwen3-VL-4B, D13 prompt, every "bbox_2d" of the answer, each box -> SAM 3 mask; no score (1.0)
#   gsam2_all       Grounding DINO, every box at text 0.25, each box -> SAM 2.1 mask; official output = boxes with
#                   score >= 0.35 (Grounded-SAM 2 demo box threshold)
#   sa2va_p         Sa2VA-Qwen3-VL-4B unchanged (already the model card's union of [SEG] masks); a separate stage so
#                   the OP-HRG run with the official query wording does not mix with D13 / D31 caches
#   sam3_union_na, gsam2_all_na   the same predictions; when no instance passes the official threshold the
#                   highest-scoring one is the output (no abstention, as ours on target-only benchmarks)
#   qwenbox_ft      Group A: qwen_box_all with the LoRA trained on the S5 data (scripts/qwenbox_train.py)
# Candidates below `floor` are dropped and at most `max_instances` (highest scores) are kept; the official threshold is applied by the evaluator (parity_eval.py), and
# on gRefCOCO the threshold is also tuned on val exactly as for ours.


class Sam3Union(B.Sam3Text):
    official = "official_threshold"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.p = cfg["parity"]["sam3_union"]   # also the _na variant
        self.tok = self.proc.tokenizer
        self.limit = min(self.tok.model_max_length, 32)

    def clip(self, p: str) -> str:
        # SAM 3's CLIP text encoder takes 32 tokens; longer gRefCOCO expressions keep their leading words (as D27)
        words = p.split()
        while len(words) > 1 and len(self.tok(" ".join(words)).input_ids) > self.limit:
            words.pop()
        return " ".join(words)

    @torch.no_grad()
    def instances(self, image: np.ndarray, phrases: list[str]) -> list[list[tuple]]:
        h, w = image.shape[:2]
        pix = self.proc(images=Image.fromarray(image), return_tensors="pt").pixel_values
        ve = self.model.get_vision_features(pixel_values=pix.to(self.device, self.dtype))
        out = []
        for p in phrases:
            t = self.proc(text=self.clip(p), return_tensors="pt").to(self.device)
            r = self.model(vision_embeds=ve, input_ids=t.input_ids, attention_mask=t.attention_mask)
            s = r.pred_logits[0].float().sigmoid()
            if r.presence_logits is not None:
                s = s * r.presence_logits[0].float().sigmoid()
            keep = torch.nonzero(s >= self.p["floor"]).flatten()
            keep = keep[torch.argsort(s[keep], descending=True)][:self.p["max_instances"]]
            cands = []
            for c in keep.split(8):
                ms = F.interpolate(r.pred_masks[0, c][:, None].float().sigmoid(), size=(h, w), mode="bilinear",
                                   align_corners=False)[:, 0] > self.s["mask_threshold"]
                cands += [(float(s[i]), m) for i, m in zip(c.tolist(), ms.cpu().numpy())]
            out.append(cands)
        return out


class QwenBoxAll(B.QwenBoxSam):
    official = None

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.s = dict(self.s, max_new_tokens=cfg["parity"]["qwen_box_all"]["max_new_tokens"])

    @staticmethod
    def boxes(text: str) -> list[list[float]]:
        out = []
        for m in B._BOX.finditer(text):
            vals = [float(v) for v in B._NUM.findall(m.group(1))]
            if len(vals) != 4:
                continue
            x0, y0, x1, y1 = (min(max(v, 0.0), B.QWEN_GRID) for v in vals)
            if x1 > x0 and y1 > y0:
                out.append([x0, y0, x1, y1])
        if not out and (b := B.parse_box(text)) is not None:
            out.append(b)   # the D13 fallback: a bare [x0, y0, x1, y1] list
        return out

    @torch.no_grad()
    def instances(self, image: np.ndarray, phrases: list[str]) -> list[list[tuple]]:
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        self.qproc.tokenizer.padding_side = "left"
        raws = []
        for i in range(0, len(phrases), self.s["batch_size"]):
            chunk = phrases[i:i + self.s["batch_size"]]
            inp = self.qproc(text=[self.chat(p) for p in chunk], images=[pil] * len(chunk), padding=True,
                             return_tensors="pt").to(self.device)
            gen = self.qwen.generate(**inp, max_new_tokens=self.s["max_new_tokens"], do_sample=False)
            raws += self.qproc.batch_decode(gen[:, inp.input_ids.shape[1]:], skip_special_tokens=True)
        flat = [(k, b) for k, r in enumerate(raws) for b in self.boxes(r)]
        out: list[list[tuple]] = [[] for _ in phrases]
        if flat:
            pix = self.sproc(images=image, return_tensors="pt").pixel_values.to(self.device, self.dtype)
            emb = self.sam.get_image_embeddings(pix)
            t = self.sproc.target_size
            scale = np.array([t / w, t / h, t / w, t / h])
            for j in range(0, len(flat), self.s["sam_batch"]):
                part = flat[j:j + self.s["sam_batch"]]
                px = np.stack([B.box_to_pixels(b, w, h) for _, b in part]) * scale
                r = self.sam(image_embeddings=emb, input_boxes=torch.tensor(px, dtype=self.dtype,
                                                                             device=self.device)[None],
                             multimask_output=False)
                logits = self.sproc.post_process_masks([r.pred_masks[0].float()], [[h, w]], binarize=False)[0]
                for (k, _), m in zip(part, (logits[:, 0] > self.s["mask_threshold"]).cpu().numpy()):
                    out[k].append((1.0, m))
        return out


class QwenBoxFT(QwenBoxAll):
    # D40 Group A: qwen_box_all with the LoRA of scripts/qwenbox_train.py (best.pt, selected on PACO val)
    def __init__(self, cfg: dict):
        super().__init__(cfg)
        import qwenbox_train as QT
        from treeground import lora as LR
        path, _ = QT.checkpoint(cfg)
        state = torch.load(path, map_location="cpu", weights_only=False)
        lc = state["meta"]["lora"]
        if state["meta"]["prompt"] != self.s["prompt"]:
            raise B.ProtocolError("qwenbox_ft: checkpoint prompt differs from the evaluation prompt")
        LR.apply_lora(self.qwen, lc["r"], lc["alpha"], lc["dropout"], tuple(lc["targets"]))
        missing, unexpected = LR.decoder(self.qwen).load_state_dict(state["lora"], strict=False)
        if unexpected or any(".lora_" in k for k in missing):
            raise B.ProtocolError(f"qwenbox_ft LoRA mismatch: missing {missing[:3]}, unexpected {unexpected[:3]}")
        if state["meta"]["qwen"] != cfg["models"]["qwen"]:
            raise B.ProtocolError("qwenbox_ft: checkpoint Qwen revision differs from the evaluation model")
        self.qwen.to(self.device).eval()
        for mod in self.qwen.modules():
            # LoRA weights in the base dtype (training ran them under bf16 autocast); SAM 3 stays as in qwen_box_all
            if isinstance(mod, LR.LoRALinear):
                dt = mod.base.weight.dtype
                mod.lora_A.data, mod.lora_B.data = mod.lora_A.data.to(dt), mod.lora_B.data.to(dt)
        self.checkpoint = str(path)


class GSam2All(B2.GroundedSam2):
    official = "official_box_threshold"

    def __init__(self, cfg: dict):
        super().__init__(cfg)
        self.p = cfg["parity"]["gsam2_all"]

    @torch.no_grad()
    def instances(self, image: np.ndarray, phrases: list[str]) -> list[list[tuple]]:
        h, w = image.shape[:2]
        pil = Image.fromarray(image)
        out = []
        for p in phrases:
            inp = self.gproc(images=pil, text=self.caption(p), return_tensors="pt").to(self.device)
            det = self.gproc.post_process_grounded_object_detection(
                self.gdino(**inp), inp.input_ids, threshold=self.p["floor"],
                text_threshold=self.p["text_threshold"], target_sizes=[(h, w)])[0]
            if len(det["scores"]) == 0:
                out.append([])
                continue
            order = torch.argsort(det["scores"], descending=True).tolist()[:self.p["max_instances"]]
            boxes = [[float(v) for v in det["boxes"][k].tolist()] for k in order]
            masks = []
            for j in range(0, len(boxes), 8):
                sin = self.sproc(images=pil, input_boxes=[boxes[j:j + 8]], return_tensors="pt").to(self.device)
                r = self.sam(**sin, multimask_output=False)
                ms = self.sproc.post_process_masks(r.pred_masks.cpu(), sin["original_sizes"])[0][:, 0]
                masks += list(np.asarray(ms.numpy() if hasattr(ms, "numpy") else ms, bool))
            out.append([(float(det["scores"][k]), m) for k, m in zip(order, masks)])
        return out


class Sa2VAP(B.Sa2VA):
    official = None

    def instances(self, image: np.ndarray, phrases: list[str]) -> list[list[tuple]]:
        return [[(1.0, m)] if acc and m is not None else [] for m, _, _, acc in self(image, phrases)]


class Sam3UnionNA(Sam3Union):
    fallback = True


class GSam2AllNA(GSam2All):
    fallback = True


PARITY = {"sam3_union": Sam3Union, "qwen_box_all": QwenBoxAll, "gsam2_all": GSam2All, "sa2va_p": Sa2VAP,
          "qwenbox_ft": QwenBoxFT, "sam3_union_na": Sam3UnionNA, "gsam2_all_na": GSam2AllNA}
SETTINGS = {"sam3_union_na": "sam3_union", "gsam2_all_na": "gsam2_all"}   # same settings, no-abstain output


def official_threshold(cfg: dict, method: str) -> float | None:
    # None: the method has no score, every instance it returns is its output
    key = PARITY[method].official
    return None if key is None else float(cfg["parity"][SETTINGS.get(method, method)][key])


def official_union(cfg: dict, method: str, cands: list[tuple], shape: tuple) -> np.ndarray | None:
    # the method's official single output: union of its instances with score above its official threshold;
    # the '_na' (no-abstain) variants fall back to their highest-scoring instance when none passes, as ours never
    # abstains on benchmarks where every query has a target (PACO-LVIS test, PARTONOMY, OP-HRG)
    thr = official_threshold(cfg, method)
    ms = [m for s, m in cands if thr is None or s > thr]
    if not ms and getattr(PARITY[method], "fallback", False) and cands:
        ms = [cands[0][1]]
    if not ms:
        return None
    return np.logical_or.reduce([m.astype(bool) for m in ms])
