import math

import torch
import torch.nn as nn
import torch.nn.functional as F

TARGETS = ("q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj")


class LoRALinear(nn.Module):
    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float):
        super().__init__()
        self.base = base
        self.r, self.scale = r, alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features, device=base.weight.device))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r, device=base.weight.device))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.drop = nn.Dropout(dropout)
        base.weight.requires_grad_(False)
        if base.bias is not None:
            base.bias.requires_grad_(False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.base(x) + F.linear(F.linear(self.drop(x), self.lora_A), self.lora_B) * self.scale


class SelEmbedding(nn.Module):
    # the frozen matrix stays untouched; the <SEL> row is a separate fp32 parameter substituted at lookup
    def __init__(self, base: nn.Embedding, sel_id: int):
        super().__init__()
        self.base, self.sel_id = base, sel_id
        self.sel_row = nn.Parameter(base.weight[sel_id].detach().float().clone())

    @property
    def weight(self) -> torch.Tensor:
        return self.base.weight

    def forward(self, ids: torch.Tensor) -> torch.Tensor:
        out = self.base(ids)
        return torch.where((ids == self.sel_id)[..., None], self.sel_row.to(out.dtype), out)


def decoder(model) -> nn.Module:
    return model.model.language_model


def apply_lora(model, r: int, alpha: float, dropout: float, targets=TARGETS) -> list[str]:
    for p in model.parameters():
        p.requires_grad_(False)
    wrapped = []
    for name, module in list(decoder(model).layers.named_modules()):
        for child_name, child in list(module.named_children()):
            if child_name in targets and isinstance(child, nn.Linear):
                setattr(module, child_name, LoRALinear(child, r, alpha, dropout))
                wrapped.append(f"layers.{name}.{child_name}" if name else f"layers.{child_name}")
    if not wrapped:
        raise RuntimeError(f"no decoder linear layers named {targets}")
    return sorted(wrapped)


def install_sel(model, sel_id: int) -> tuple[SelEmbedding, bool]:
    lm = decoder(model)
    base = lm.embed_tokens
    tied = model.lm_head.weight.data_ptr() == base.weight.data_ptr()
    lm.embed_tokens = SelEmbedding(base, sel_id)
    return lm.embed_tokens, tied


def answer_logits(model, sel: SelEmbedding, tied: bool, h: torch.Tensor) -> torch.Tensor:
    logits = model.lm_head(h).float()
    if not tied:
        return logits
    # tied output embedding: the <SEL> logit uses the trainable row as well
    col = h.float() @ sel.sel_row
    k = sel.sel_id
    return torch.cat([logits[:, :k], col[:, None], logits[:, k + 1:]], 1)


def lora_state(model) -> dict[str, torch.Tensor]:
    return {k: v.detach().cpu().clone() for k, v in decoder(model).state_dict().items()
            if ".lora_A" in k or ".lora_B" in k}


def trainable_state(model, sel: SelEmbedding, selector: nn.Module) -> dict:
    return {"lora": lora_state(model), "sel_row": sel.sel_row.detach().cpu().clone(),
            "selector": {k: v.detach().cpu().clone() for k, v in selector.state_dict().items()}}


def load_trainable(model, sel: SelEmbedding, selector: nn.Module, state: dict) -> None:
    expected = set(lora_state(model))
    if set(state["lora"]) != expected:
        raise RuntimeError("checkpoint LoRA modules do not match the wrapped model")
    missing, unexpected = decoder(model).load_state_dict(state["lora"], strict=False)
    if unexpected or any(".lora_" in k for k in missing):
        raise RuntimeError(f"LoRA state mismatch: missing {missing}, unexpected {unexpected}")
    with torch.no_grad():
        sel.sel_row.copy_(state["sel_row"])
    selector.load_state_dict(state["selector"])


def trainable_parameters(model, sel: SelEmbedding) -> list[nn.Parameter]:
    lora = [p for n, p in decoder(model).named_parameters() if ".lora_" in n]
    return lora + [sel.sel_row]
