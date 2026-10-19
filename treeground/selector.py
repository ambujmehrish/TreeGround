import torch
import torch.nn as nn
import torch.nn.functional as F

TEMPERATURE = 0.07


class Head(nn.Module):
    def __init__(self, d_in: int, hidden: int = 1024, d_out: int = 256):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.LayerNorm(hidden), nn.Linear(hidden, d_out))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class Selector(nn.Module):
    def __init__(self, d_node: int, d_query: int, tau: float = TEMPERATURE):
        super().__init__()
        self.g = Head(d_node)
        self.f = Head(d_query)
        self.tau = tau

    def nodes(self, z: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.g(z.float()), dim=-1)

    def queries(self, h: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.f(h.float()), dim=-1)

    def scores(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return q @ e.T / self.tau


def focal(logits: torch.Tensor, target: int, gamma: float = 2.0) -> torch.Tensor:
    log_pt = F.logsigmoid(logits if target == 1 else -logits)
    return -((1 - log_pt.exp()) ** gamma) * log_pt


def selection_loss(pos_logits: torch.Tensor, neg_logits: torch.Tensor, gamma: float = 2.0
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    pos = focal(pos_logits, 1, gamma).sum() / max(1, pos_logits.numel())
    neg = focal(neg_logits, 0, gamma).mean() if neg_logits.numel() else neg_logits.new_zeros(())
    return pos + neg, pos.detach(), neg.detach()
