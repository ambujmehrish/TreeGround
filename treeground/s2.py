import re
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from treeground import lora as LR
from treeground import mllm
from treeground.data.paco import display
from treeground.mllm import SEL


@dataclass
class Seq:
    input_ids: torch.Tensor
    answer: tuple[int, int]      # [start, end) of the answer span in input_ids
    sel: torch.Tensor            # positions of <SEL> tokens (all inside the answer span)
    pixel_values: torch.Tensor | None = None
    image_grid_thw: torch.Tensor | None = None


def user_text(template_text: str) -> str:
    # S2 moves the probes from the query into the answer span, so the I1 query keeps its words without <SEL>
    return re.sub(r"\s*" + re.escape(SEL), "", template_text).strip()


def answer_text(kind: str, qid: str, text: str, label: str | None = None) -> str:
    if kind == "object":
        return f"{display(qid.split('|')[2])} {SEL}"
    if kind in ("part", "chain"):
        obj, part = (display(x) for x in qid.split("|")[2].split(":", 1))
        return f"{part} of the {obj} {SEL}" if kind == "part" else f"{obj} {SEL} {part} {SEL}"
    if kind == "text":
        obj, part = (display(x) for x in label.split(":", 1))
        return f"{part} of a {obj} {SEL}"
    if kind == "absent":
        return f"{user_text(text)} {SEL}"
    raise ValueError(f"unknown query kind {kind!r}")


def encode(processor, sel_id: int, user: str, answer: str, image: Image.Image | None) -> Seq:
    content = ([{"type": "image"}] if image is not None else []) + [{"type": "text", "text": user}]
    msgs = [{"role": "user", "content": content}]
    prefix = processor.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    full = processor.apply_chat_template(msgs + [{"role": "assistant", "content": answer}], tokenize=False)
    if not full.startswith(prefix + answer):
        raise RuntimeError("chat template does not render the assistant answer right after the generation prompt")
    kw = {"images": [image]} if image is not None else {}
    x = processor(text=[prefix], return_tensors="pt", **kw)
    ans = torch.tensor(processor.tokenizer(answer, add_special_tokens=False)["input_ids"], dtype=torch.long)
    ids = torch.cat([x["input_ids"][0], ans])
    start = x["input_ids"].shape[1]
    sel = torch.nonzero(ids == sel_id).reshape(-1)
    if len(sel) != answer.count(SEL) or (len(sel) and int(sel[0]) < start):
        raise RuntimeError(f"<SEL> positions do not match the answer span of {answer!r}")
    return Seq(ids, (start, len(ids)), sel, x.get("pixel_values"), x.get("image_grid_thw"))


def collate(seqs: list[Seq], pad_id: int) -> dict[str, torch.Tensor]:
    n = max(len(s.input_ids) for s in seqs)
    ids = torch.full((len(seqs), n), pad_id, dtype=torch.long)
    mask = torch.zeros((len(seqs), n), dtype=torch.long)
    for b, s in enumerate(seqs):
        ids[b, :len(s.input_ids)] = s.input_ids
        mask[b, :len(s.input_ids)] = 1
    out = {"input_ids": ids, "attention_mask": mask}
    imgs = [s for s in seqs if s.pixel_values is not None]
    if imgs:
        out["pixel_values"] = torch.cat([s.pixel_values for s in imgs])
        out["image_grid_thw"] = torch.cat([s.image_grid_thw for s in imgs])
    return out


def autocast(device: str):
    return torch.autocast(device_type=torch.device(device).type, dtype=torch.bfloat16)


def forward(model, batch: dict[str, torch.Tensor], layer: int) -> tuple[torch.Tensor, torch.Tensor]:
    out = model.model(**batch, output_hidden_states=True, use_cache=False)
    if len(out.hidden_states) != model.config.text_config.num_hidden_layers + 1:
        raise RuntimeError("unexpected number of hidden states")
    return out.hidden_states[layer], out.last_hidden_state


def node_features(states: torch.Tensor, input_ids: torch.Tensor, image_token_id: int, grid: tuple[int, int],
                  cov: tuple[np.ndarray, np.ndarray, np.ndarray]) -> torch.Tensor:
    pos = mllm.image_token_positions(input_ids, image_token_id)
    if len(pos) != grid[0] * grid[1]:
        raise RuntimeError(f"{len(pos)} image tokens but merged grid {grid}")
    ptr, idx, wts = cov
    if len(ptr) == 1:
        return states.new_zeros((0, states.shape[-1]), dtype=torch.float32)
    with torch.autocast(device_type=states.device.type, enabled=False):
        return mllm.pool(states[pos], ptr, idx, wts)


def lm_loss(model, sel: LR.SelEmbedding, tied: bool, last: torch.Tensor, batch: dict[str, torch.Tensor],
            seqs: list[Seq]) -> torch.Tensor:
    # token-mean next-token CE over all answer tokens of the micro-batch; <SEL> tokens are targets too
    hs, targets = [], []
    for b, s in enumerate(seqs):
        a, e = s.answer
        hs.append(last[b, a - 1:e - 1])
        targets.append(batch["input_ids"][b, a:e])
    logits = LR.answer_logits(model, sel, tied, torch.cat(hs))
    return F.cross_entropy(logits, torch.cat(targets))
