"""Hyper-connection family: HC, mHC, Frac-Connections.

All keep a multi-stream residual H and, per sublayer, learn (i) how to read the
sublayer input from the streams, (ii) how to write its output back, and (iii)
how streams mix with each other.

HC   (Zhu et al., ICLR 2025, arXiv 2409.19606)  -- n parallel copies of width d
mHC  (DeepSeek, arXiv 2512.24880)               -- HC with H_res projected onto
                                                  doubly-stochastic matrices
Frac (Zhu et al., arXiv 2503.14125)             -- m fractions of width d/m
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Connection


class _HCSite(nn.Module):
    """One HC / Frac module (static + optional dynamic part).

    Operates on H [B, T, n, e]. alpha has shape n x (k + n): the first k
    columns read the sublayer input, the last n columns are the residual
    stream-mixing matrix A_r. k = 1 for HC, k = n for Frac.
    """

    def __init__(self, n: int, e: int, k: int, read_idx, dynamic: bool):
        super().__init__()
        self.n, self.k, self.dynamic = n, k, dynamic
        static_alpha = torch.zeros(n, k + n)
        if k == 1:
            static_alpha[read_idx, 0] = 1.0  # A_m = e_{layer % n}
        else:
            static_alpha[:, :k] = torch.eye(n)  # Y = I
        static_alpha[:, k:] = torch.eye(n)  # A_r = I
        self.static_alpha = nn.Parameter(static_alpha)
        self.static_beta = nn.Parameter(torch.ones(n))
        if dynamic:
            self.norm = nn.LayerNorm(e)
            self.dyn_alpha = nn.Parameter(torch.zeros(e, k + n))
            self.dyn_beta = nn.Parameter(torch.zeros(e))
            self.alpha_scale = nn.Parameter(torch.tensor(0.01))
            self.beta_scale = nn.Parameter(torch.tensor(0.01))

    def coefficients(self, H):
        if not self.dynamic:
            return self.static_alpha, self.static_beta
        Hn = self.norm(H)
        alpha = self.alpha_scale * torch.tanh(Hn @ self.dyn_alpha) + self.static_alpha
        beta = self.beta_scale * torch.tanh(Hn @ self.dyn_beta) + self.static_beta
        return alpha, beta

    def forward(self, H, f, concat_read: bool):
        alpha, beta = self.coefficients(H)
        # mix[..., c, :] = sum_i alpha[i, c] * H[..., i, :]
        mix = torch.einsum("...nc,...ne->...ce", alpha.to(H.dtype), H)
        read, resid = mix[..., : self.k, :], mix[..., self.k :, :]
        x = read.flatten(-2) if concat_read else read.squeeze(-2)
        y = f(x)
        y = y.unflatten(-1, (self.n, -1)) if concat_read else y.unsqueeze(-2)
        return resid + beta.to(H.dtype).unsqueeze(-1) * y


class HyperConnection(Connection):
    """Hyper-Connections. kwargs: n (expansion rate), dynamic (DHC vs SHC)."""

    def __init__(self, cfg, n: int = 4, dynamic: bool = True):
        super().__init__(cfg)
        self.n, self.dynamic = n, dynamic
        self.sites = nn.ModuleList(
            _HCSite(n, cfg.d_model, 1, i % n, dynamic) for i in range(self.n_sub)
        )

    def forward(self, x0, sublayers):
        H = x0.unsqueeze(-2).expand(*x0.shape[:-1], self.n, x0.shape[-1])
        for site, f in zip(self.sites, sublayers):
            H = site(H, f, concat_read=False)
        return H.sum(-2)

    def extra_repr(self):
        return f"n={self.n}, dynamic={self.dynamic}"


class FracConnection(Connection):
    """Frac-Connections. kwargs: m (number of fractions), dynamic."""

    def __init__(self, cfg, m: int = 2, dynamic: bool = True):
        super().__init__(cfg)
        assert cfg.d_model % m == 0
        self.m, self.dynamic = m, dynamic
        e = cfg.d_model // m
        self.sites = nn.ModuleList(_HCSite(m, e, m, None, dynamic) for _ in range(self.n_sub))

    def forward(self, x0, sublayers):
        H = x0.unflatten(-1, (self.m, -1))
        for site, f in zip(self.sites, sublayers):
            H = site(H, f, concat_read=True)
        return H.flatten(-2)

    def extra_repr(self):
        return f"m={self.m}, dynamic={self.dynamic}"


def _sinkhorn_rows_first(logits, iters: int, eps: float = 1e-6):
    """DeepSeek-V4 reference order: row-softmax, then alternate col/row normalisation."""
    M = logits.float().softmax(-1) + eps
    M = M / (M.sum(-2, keepdim=True) + eps)
    for _ in range(iters - 1):
        M = M / (M.sum(-1, keepdim=True) + eps)
        M = M / (M.sum(-2, keepdim=True) + eps)
    return M


class _MHCSite(nn.Module):
    """One mHC module: pre (n), post (n) and res (n x n) maps from vec(H)."""

    def __init__(self, n: int, d: int, alpha_init: float, res_bias_diag: float, phi_std: float):
        super().__init__()
        self.n = n
        self.phi = nn.Parameter(torch.randn((2 + n) * n, n * d) * phi_std)
        self.alpha = nn.Parameter(torch.full((3,), alpha_init))
        bias = torch.zeros((2 + n) * n)
        bias[2 * n :] = (torch.eye(n) * res_bias_diag).flatten()
        self.bias = nn.Parameter(bias)

    def maps(self, H, sinkhorn_iters: int):
        n = self.n
        x = H.flatten(-2).float()
        # RMSNorm over the flattened n*d vector (no learnable weight), applied
        # after the projection as in the reference kernel.
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        mix = F.linear(x, self.phi.float()) * r
        pre = torch.sigmoid(self.alpha[0] * mix[..., :n] + self.bias[:n]) + 1e-6
        post = 2 * torch.sigmoid(self.alpha[1] * mix[..., n : 2 * n] + self.bias[n : 2 * n])
        res_logits = self.alpha[2] * mix[..., 2 * n :] + self.bias[2 * n :]
        res = _sinkhorn_rows_first(res_logits.unflatten(-1, (n, n)), sinkhorn_iters)
        return pre, post, res

    def forward(self, H, f, sinkhorn_iters: int):
        pre, post, res = self.maps(H, sinkhorn_iters)
        dt = H.dtype
        x = torch.einsum("...n,...ne->...e", pre.to(dt), H)
        y = f(x)
        # out_k = post_k * y + sum_j res[j, k] * H_j
        return torch.einsum("...jk,...je->...ke", res.to(dt), H) + post.to(dt).unsqueeze(-1) * y.unsqueeze(-2)


class ManifoldHyperConnection(Connection):
    """mHC: Manifold-Constrained Hyper-Connections.

    kwargs:
        n:              number of streams (paper: 4)
        sinkhorn_iters: Sinkhorn-Knopp iterations (paper: 20)
        alpha_init:     gating-factor init (paper: 0.01)
        res_bias_diag:  init of diag(b_res); 0 = uniform doubly-stochastic start.
                        Not published -- we default to 0.
        phi_std:        init std of the projection phi. Not published.
    """

    def __init__(self, cfg, n: int = 4, sinkhorn_iters: int = 20, alpha_init: float = 0.01,
                 res_bias_diag: float = 0.0, phi_std: float = 0.02):
        super().__init__(cfg)
        self.n, self.iters = n, sinkhorn_iters
        d = cfg.d_model
        self.sites = nn.ModuleList(
            _MHCSite(n, d, alpha_init, res_bias_diag, phi_std) for _ in range(self.n_sub)
        )
        # "HyperHead": learned sigmoid read-out of the streams before the final norm.
        self.head_phi = nn.Parameter(torch.randn(n, n * d) * phi_std)
        self.head_alpha = nn.Parameter(torch.tensor(alpha_init))
        self.head_bias = nn.Parameter(torch.zeros(n))

    def forward(self, x0, sublayers):
        H = x0.unsqueeze(-2).expand(*x0.shape[:-1], self.n, x0.shape[-1])
        for site, f in zip(self.sites, sublayers):
            H = site(H, f, self.iters)
        x = H.flatten(-2).float()
        r = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + 1e-6)
        w = torch.sigmoid(self.head_alpha * F.linear(x, self.head_phi.float()) * r + self.head_bias)
        return torch.einsum("...n,...ne->...e", (w + 1e-6).to(H.dtype), H)

    def extra_repr(self):
        return f"n={self.n}, sinkhorn_iters={self.iters}"
