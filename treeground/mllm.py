import hashlib
from typing import Iterable

import cv2
import numpy as np
import torch
from PIL import Image

SEL = "<SEL>"
EXPECTED_HIDDEN = 2560
EXPECTED_VALUE_DIM = 1024


def load_qwen(repo: str, revision: str, device: str, dtype: str, cache_dir: str | None = None):
    from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        repo, revision=revision, dtype=getattr(torch, dtype), attn_implementation="sdpa", cache_dir=cache_dir)
    processor = AutoProcessor.from_pretrained(repo, revision=revision, cache_dir=cache_dir)
    dims = model_dims(model)
    if dims["hidden"] != EXPECTED_HIDDEN or dims["kv_heads"] * dims["head_dim"] != EXPECTED_VALUE_DIM:
        raise RuntimeError(f"Qwen3-VL config does not match the pinned spec (hidden 2560, 8x128 values): {dims}")
    return model.to(device).eval(), processor


def model_dims(model) -> dict:
    cfg = model.config
    text, vision = cfg.text_config, cfg.vision_config
    return {
        "hidden": text.hidden_size,
        "layers": text.num_hidden_layers,
        "kv_heads": text.num_key_value_heads,
        "head_dim": getattr(text, "head_dim", text.hidden_size // text.num_attention_heads),
        "patch_size": vision.patch_size,
        "merge": vision.spatial_merge_size,
        "image_token_id": cfg.image_token_id,
        "deepstack_visual_indexes": list(vision.deepstack_visual_indexes),
        "encoder_dim": vision.out_hidden_size,
    }


def candidate_layers(num_layers: int) -> list[int]:
    return [int(np.floor(num_layers * f + 0.5)) for f in (1 / 4, 1 / 3, 1 / 2, 2 / 3)]


def deepstack_injection_layers(model) -> list[int]:
    # DeepStack features are added to the residual stream after decoder layers 1..len(indexes) (1-based).
    return list(range(1, len(model.config.vision_config.deepstack_visual_indexes) + 1))


def check_layers(model, layers: list[int]) -> None:
    last = max(deepstack_injection_layers(model))
    bad = [k for k in layers if k <= last]
    if bad:
        raise RuntimeError(f"candidate layers {bad} are not later than the last DeepStack injection layer {last}")


def add_sel_token(model, processor) -> tuple[int, str]:
    tok = processor.tokenizer
    n_vocab = len(tok)
    tok.add_special_tokens({"additional_special_tokens": [SEL]})
    sel_id = tok.convert_tokens_to_ids(SEL)
    emb = model.get_input_embeddings().weight
    if sel_id >= emb.shape[0]:
        model.resize_token_embeddings(len(tok))
        emb = model.get_input_embeddings().weight
    with torch.no_grad():
        emb[sel_id] = emb[:n_vocab].float().mean(0).to(emb.dtype)
    digest = hashlib.sha256(emb[sel_id].detach().float().cpu().numpy().tobytes()).hexdigest()
    return sel_id, digest


def chat_template_hash(processor) -> str:
    return hashlib.sha256((processor.chat_template or "").encode()).hexdigest()


def image_prompt(processor) -> str:
    msgs = [{"role": "user", "content": [{"type": "image"}, {"type": "text", "text": ""}]}]
    return processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=False)


def query_prompt(processor, text: str, with_image: bool) -> str:
    content = ([{"type": "image"}] if with_image else []) + [{"type": "text", "text": text}]
    full = processor.apply_chat_template([{"role": "user", "content": content}], tokenize=False,
                                         add_generation_prompt=False)
    # nothing after the last probe can influence its state under causal attention
    return full[: full.rindex(SEL) + len(SEL)]


def merged_grid(image_grid_thw: torch.Tensor, merge: int) -> tuple[int, int]:
    t, h, w = (int(x) for x in image_grid_thw.reshape(-1, 3)[0])
    if t != 1:
        raise RuntimeError(f"expected a single temporal patch for an image, got t={t}")
    return h // merge, w // merge


def coverage(mask: np.ndarray, gh: int, gw: int) -> np.ndarray:
    return cv2.resize(mask.astype(np.float32), (gw, gh), interpolation=cv2.INTER_AREA).reshape(-1)


def sparse_coverages(node_masks: Iterable[np.ndarray], gh: int, gw: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    ptr, idx, wts = [0], [], []
    for m in node_masks:
        c = coverage(m, gh, gw)
        nz = np.flatnonzero(c > 0)
        if len(nz) == 0:
            raise RuntimeError("node mask has zero coverage on the token grid")
        idx.append(nz.astype(np.int32))
        wts.append(c[nz].astype(np.float32))
        ptr.append(ptr[-1] + len(nz))
    return (np.asarray(ptr, dtype=np.int64), np.concatenate(idx) if idx else np.zeros(0, np.int32),
            np.concatenate(wts) if wts else np.zeros(0, np.float32))


def pool(states: torch.Tensor, ptr: np.ndarray, idx: np.ndarray, wts: np.ndarray) -> torch.Tensor:
    n = len(ptr) - 1
    rows = np.repeat(np.arange(n), np.diff(ptr))
    w = torch.as_tensor(wts, dtype=torch.float32)
    norm = torch.zeros(n, dtype=torch.float32).index_add_(0, torch.as_tensor(rows), w)
    mat = torch.sparse_coo_tensor(np.stack([rows, idx]), w / norm[rows], (n, states.shape[0]),
                                  device=states.device, check_invariants=True)
    return torch.sparse.mm(mat, states.float())


def image_token_positions(input_ids: torch.Tensor, image_token_id: int) -> torch.Tensor:
    return torch.nonzero(input_ids.reshape(-1) == image_token_id).reshape(-1)


@torch.no_grad()
def image_pass(model, processor, image: Image.Image, layers: list[int], want_value: bool, text: str | None = None
               ) -> dict:
    dims = model_dims(model)
    device = next(model.parameters()).device
    prompt = image_prompt(processor) if text is None else text
    inputs = processor(text=[prompt], images=[image], return_tensors="pt").to(device)
    gh, gw = merged_grid(inputs["image_grid_thw"], dims["merge"])
    pos = image_token_positions(inputs["input_ids"], dims["image_token_id"])
    if len(pos) != gh * gw:
        raise RuntimeError(f"{len(pos)} image tokens but merged grid {gh}x{gw}")
    captured: dict[str, torch.Tensor] = {}
    hooks = [model.model.visual.merger.register_forward_hook(
        lambda m, i, o: captured.__setitem__("encoder", o.detach()))]
    if want_value:
        for k in layers:
            proj = model.model.language_model.layers[k - 1].self_attn.v_proj
            hooks.append(proj.register_forward_hook(
                lambda m, i, o, k=k: captured.__setitem__(f"value_{k}", o.detach())))
    try:
        out = model.model(**inputs, output_hidden_states=True, use_cache=False)
    finally:
        for h in hooks:
            h.remove()
    hs = out.hidden_states
    if len(hs) != dims["layers"] + 1:
        raise RuntimeError(f"expected {dims['layers'] + 1} hidden states, got {len(hs)}")
    res = {"grid": (gh, gw), "positions": pos, "input_ids": inputs["input_ids"][0]}
    res["hidden"] = {k: hs[k][0, pos] for k in layers}
    if want_value:
        res["value"] = {k: captured[f"value_{k}"][0, pos] for k in layers}
    res["encoder"] = captured["encoder"]
    if res["encoder"].shape[0] != gh * gw:
        raise RuntimeError("merger output does not match the number of image tokens")
    return res


@torch.no_grad()
def probe_states(model, processor, image: Image.Image | None, texts: list[str], layers: list[int], sel_id: int,
                 batch_size: int) -> list[np.ndarray]:
    device = next(model.parameters()).device
    tok = processor.tokenizer
    out: list[np.ndarray] = []
    for start in range(0, len(texts), batch_size):
        chunk = texts[start:start + batch_size]
        prompts = [query_prompt(processor, t, image is not None) for t in chunk]
        tok.padding_side = "right"
        kwargs = {"images": [image] * len(chunk)} if image is not None else {}
        inputs = processor(text=prompts, padding=True, return_tensors="pt", **kwargs).to(device)
        res = model.model(**inputs, output_hidden_states=True, use_cache=False)
        for b in range(len(chunk)):
            sel_pos = torch.nonzero(inputs["input_ids"][b] == sel_id).reshape(-1)
            if len(sel_pos) != chunk[b].count(SEL):
                raise RuntimeError(f"probe count mismatch in {chunk[b]!r}")
            out.append(torch.stack([res.hidden_states[k][b, sel_pos] for k in layers], 1).float().cpu().numpy())
    return out
