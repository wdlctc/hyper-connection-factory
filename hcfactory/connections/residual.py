"""Standard residual baselines."""
from __future__ import annotations

import torch.nn as nn

from .base import Connection
from ..model import RMSNorm


class PreNorm(Connection):
    """h <- h + f(norm(h))  (GPT-2 / Llama)."""

    def forward(self, x0, sublayers):
        h = x0
        for f in sublayers:
            h = h + f(h)
        return h


class PostNorm(Connection):
    """h <- norm(h + f(h))  (original Transformer)."""

    sublayer_prenorm = False

    def __init__(self, cfg):
        super().__init__(cfg)
        self.norms = nn.ModuleList(RMSNorm(cfg.d_model) for _ in range(self.n_sub))

    def forward(self, x0, sublayers):
        h = x0
        for f, norm in zip(sublayers, self.norms):
            h = norm(h + f(h))
        return h
