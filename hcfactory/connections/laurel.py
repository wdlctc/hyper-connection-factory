"""LAuReL: Learned Augmented Residual Layer (Menghani et al., arXiv 2411.07501).

Applied per sublayer:
    rw:     x <- a f(x) + b x
    lr:     x <- f(x) + x + x A B              (A: D->r, B: r->D, B = 0 at init)
    rw_lr:  x <- a f(x) + b (x + x A B)

(a, b) = 2 * softmax(w) with w = 0 at init, so a = b = 1 and the model starts
as pre-norm while the weights stay bounded (the paper asks for a bounding
normalisation but does not fix one). A is initialised "column orthogonal":
A[i, j] = 1/sqrt(rD) if i % r == j else 0.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

from .base import Connection


class LAuReL(Connection):
    def __init__(self, cfg, variant: str = "rw_lr", rank: int = 16):
        super().__init__(cfg)
        assert variant in ("rw", "lr", "rw_lr")
        self.variant, self.rank = variant, rank
        D = cfg.d_model
        self.use_rw = "rw" in variant
        self.use_lr = "lr" in variant
        if self.use_rw:
            self.rw = nn.Parameter(torch.zeros(self.n_sub, 2))
        if self.use_lr:
            self.A = nn.Parameter(torch.empty(self.n_sub, D, rank))
            self.B = nn.Parameter(torch.zeros(self.n_sub, rank, D))

    def reset_parameters(self):
        if self.use_lr:
            D, r = self.d_model, self.rank
            with torch.no_grad():
                A = torch.zeros(D, r)
                A[torch.arange(D), torch.arange(D) % r] = 1 / math.sqrt(r * D)
                self.A.copy_(A.expand_as(self.A))
                self.B.zero_()

    def forward(self, x0, sublayers):
        x = x0
        for i, f in enumerate(sublayers):
            fx = f(x)
            skip = x + (x @ self.A[i]) @ self.B[i] if self.use_lr else x
            if self.use_rw:
                a, b = (2 * self.rw[i].softmax(0)).to(x.dtype)
                x = a * fx + b * skip
            else:
                x = fx + skip
        return x

    def extra_repr(self):
        return f"variant={self.variant}, rank={self.rank}"
