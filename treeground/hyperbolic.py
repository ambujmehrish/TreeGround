import math

import numpy as np
import torch
import torch.nn as nn

from treeground.selector import TEMPERATURE, Head, Selector

# Lorentz-model selector following MERU (Desai et al. 2023, arXiv 2304.09172): Euclidean heads, learnable
# per-branch input scale alpha, exponential map at the origin onto the hyperboloid of learnable curvature -c, logits
# from the negative geodesic distance with a learnable temperature, entailment cones with K = 0.1. MERU uses softmax
# contrastive logits, which are shift invariant; our focal sigmoid loss is not, so a learnable offset b (in distance
# units) sets where sigma = 0.5: logit = (b - d) / tau. Points are stored by their space components; the hyperbolic
# math runs in float64 because the time component grows like sinh(r) and float32 inner products cancel.

CURV_RANGE = (0.1, 10.0)
MIN_TAU = 0.01
EPS = 1e-12
MAX_RADIUS = math.asinh(2.0 ** 15)


def time_component(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(1 / c + (x * x).sum(-1))


def expmap0(v: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    v = v.double()
    norm = torch.sqrt((v * v).sum(-1, keepdim=True).clamp_min(EPS))
    r = (c.sqrt() * norm).clamp(min=1e-8, max=MAX_RADIUS)
    return torch.sinh(r) * v / r


def pairwise_dist(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    inner = x @ y.T - time_component(x, c)[:, None] * time_component(y, c)[None, :]
    # cosh(sqrt(c) d) = -c <x, y>_L; the asinh form avoids acosh's unbounded slope at 1
    arg = (-c * inner - 1).clamp_min(EPS)
    return 2 * torch.asinh(torch.sqrt(arg / 2)) / c.sqrt()


def dist0(x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    return torch.asinh(c.sqrt() * torch.sqrt((x * x).sum(-1).clamp_min(EPS))) / c.sqrt()


def half_aperture(x: torch.Tensor, c: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    s = 2 * k / (c.sqrt() * torch.sqrt((x * x).sum(-1).clamp_min(EPS)))
    return torch.asin(s.clamp(-1 + 1e-7, 1 - 1e-7))


def exterior_angle(x: torch.Tensor, y: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
    # angle at x between the geodesic to y and the ray from the origin through x (MERU oxy_angle), row-wise
    xt, yt = time_component(x, c), time_component(y, c)
    cxy = c * ((x * y).sum(-1) - xt * yt)
    num = yt + cxy * xt
    den = torch.sqrt((x * x).sum(-1).clamp_min(EPS)) * torch.sqrt((cxy * cxy - 1).clamp_min(EPS))
    return torch.acos((num / den).clamp(-1 + 1e-7, 1 - 1e-7))


def cone_violation(parent: torch.Tensor, child: torch.Tensor, c: torch.Tensor, k: float = 0.1) -> torch.Tensor:
    return (exterior_angle(parent, child, c) - half_aperture(parent, c, k)).clamp_min(0)


class LorentzSelector(nn.Module):
    def __init__(self, d_node: int, d_query: int, tau: float = TEMPERATURE, curvature: float = 1.0,
                 offset: float = 1.0, d_out: int = 256):
        super().__init__()
        self.g = Head(d_node, d_out=d_out)
        self.f = Head(d_query, d_out=d_out)
        self.log_alpha_g = nn.Parameter(torch.tensor(-0.5 * math.log(d_out)))
        self.log_alpha_f = nn.Parameter(torch.tensor(-0.5 * math.log(d_out)))
        self.log_c = nn.Parameter(torch.tensor(math.log(curvature)))
        self.log_tau = nn.Parameter(torch.tensor(math.log(tau)))
        self.offset = nn.Parameter(torch.tensor(float(offset)))

    def scalar_parameters(self) -> list[nn.Parameter]:
        return [self.log_alpha_g, self.log_alpha_f, self.log_c, self.log_tau, self.offset]

    @property
    def curv(self) -> torch.Tensor:
        return self.log_c.double().exp().clamp(*CURV_RANGE)

    @property
    def temperature(self) -> torch.Tensor:
        return self.log_tau.double().exp().clamp_min(MIN_TAU)

    @property
    def tau(self) -> float:
        return float(self.temperature.detach())

    def nodes(self, z: torch.Tensor) -> torch.Tensor:
        return expmap0(self.g(z.float()) * self.log_alpha_g.exp(), self.curv)

    def queries(self, h: torch.Tensor) -> torch.Tensor:
        return expmap0(self.f(h.float()) * self.log_alpha_f.exp(), self.curv)

    def scores(self, q: torch.Tensor, e: torch.Tensor) -> torch.Tensor:
        return ((self.offset.double() - pairwise_dist(q.double(), e.double(), self.curv)) / self.temperature).float()


def make_selector(geometry: dict, d_node: int, d_query: int, tau: float) -> nn.Module:
    if geometry["kind"] == "euclidean":
        return Selector(d_node, d_query, tau)
    if geometry["kind"] == "lorentz":
        return LorentzSelector(d_node, d_query, tau, geometry["curvature_init"], geometry["offset_init"])
    raise ValueError(f"unknown selector geometry {geometry['kind']!r}")


def tree_edges(parent: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    # parent -> child pairs among proposal nodes; the virtual root (index n, no embedding) is skipped
    parent = np.asarray(parent, dtype=np.int64)
    child = np.flatnonzero((parent >= 0) & (parent < len(parent) - 1))
    return parent[child], child
