"""A small Llama-style decoder whose residual wiring is pluggable.

The model is split into three parts:

* ``embed``     token embedding -> x0            (shared by every variant)
* ``sublayers`` 2L callables, alternating Attention / MLP, each computing
                ``f(norm(x))``                   (shared by every variant)
* ``connection`` decides *what each sublayer reads* and *how its output is
                written back* -- this is the only thing that differs between
                pre-norm, hyper-connections, mHC, attention residuals, ...

Keeping the sublayers identical means that any difference in the results is
due to the connection, not to incidental changes in the transformer body.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class ModelConfig:
    vocab_size: int = 50304
    n_layer: int = 8
    n_head: int = 8
    d_model: int = 512
    mlp_ratio: float = 8 / 3  # SwiGLU hidden = mlp_ratio * d_model (rounded)
    max_seq_len: int = 1024
    rope_theta: float = 10000.0
    tie_embeddings: bool = True
    connection: str = "prenorm"
    connection_kwargs: dict = field(default_factory=dict)


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6, affine: bool = True):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        y = F.rms_norm(x.float(), (x.shape[-1],), eps=self.eps).type_as(x)
        return y * self.weight if self.weight is not None else y


class Rotary(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float):
        super().__init__()
        inv = 1.0 / theta ** (torch.arange(0, head_dim, 2).float() / head_dim)
        t = torch.arange(max_seq_len).float()
        freqs = torch.outer(t, inv)
        self.register_buffer("cos", freqs.cos(), persistent=False)
        self.register_buffer("sin", freqs.sin(), persistent=False)

    def forward(self, x):  # x: [B, H, T, hd]
        T = x.shape[-2]
        cos, sin = self.cos[:T].to(x.dtype), self.sin[:T].to(x.dtype)
        x1, x2 = x[..., ::2], x[..., 1::2]
        out = torch.stack((x1 * cos - x2 * sin, x1 * sin + x2 * cos), dim=-1)
        return out.flatten(-2)


class Attention(nn.Module):
    """Causal self-attention with its own pre-norm.

    ``forward(x)`` reads one stream. ``forward(x, xk, xv)`` lets connections
    such as MUDDFormer feed different streams into Q, K and V.
    """

    is_attention = True

    def __init__(self, cfg: ModelConfig, rotary: Rotary, prenorm: bool = True,
                 qkv_streams: bool = False):
        super().__init__()
        d, h = cfg.d_model, cfg.n_head
        self.n_head, self.head_dim = h, d // h
        norm = (lambda: RMSNorm(d)) if prenorm else nn.Identity
        self.norm_q = norm()
        # separate K/V norms only exist when a connection feeds distinct streams
        self.norm_k = norm() if qkv_streams else None
        self.norm_v = norm() if qkv_streams else None
        self.wq = nn.Linear(d, d, bias=False)
        self.wk = nn.Linear(d, d, bias=False)
        self.wv = nn.Linear(d, d, bias=False)
        self.wo = nn.Linear(d, d, bias=False)
        self.rotary = rotary

    def forward(self, x, xk=None, xv=None):
        B, T, D = x.shape
        xq = self.norm_q(x)
        xk = xq if xk is None else self.norm_k(xk)
        xv = xq if xv is None else self.norm_v(xv)
        q = self.wq(xq).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        k = self.wk(xk).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        v = self.wv(xv).view(B, T, self.n_head, self.head_dim).transpose(1, 2)
        q, k = self.rotary(q), self.rotary(k)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)
        return self.wo(y.transpose(1, 2).reshape(B, T, D))


class MLP(nn.Module):
    is_attention = False

    def __init__(self, cfg: ModelConfig, prenorm: bool = True):
        super().__init__()
        d = cfg.d_model
        hidden = int(cfg.mlp_ratio * d)
        hidden = 64 * ((hidden + 63) // 64)
        self.norm = RMSNorm(d) if prenorm else nn.Identity()
        self.w1 = nn.Linear(d, hidden, bias=False)
        self.w3 = nn.Linear(d, hidden, bias=False)
        self.w2 = nn.Linear(hidden, d, bias=False)

    def forward(self, x):
        x = self.norm(x)
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class GPT(nn.Module):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        from .connections import build_connection

        self.cfg = cfg
        self.connection = build_connection(cfg)
        prenorm = self.connection.sublayer_prenorm
        rotary = Rotary(cfg.d_model // cfg.n_head, cfg.max_seq_len, cfg.rope_theta)
        self.embed = nn.Embedding(cfg.vocab_size, cfg.d_model)
        subs = []
        for _ in range(cfg.n_layer):
            subs.append(Attention(cfg, rotary, prenorm, self.connection.qkv_streams))
            subs.append(MLP(cfg, prenorm))
        self.sublayers = nn.ModuleList(subs)
        self.final_norm = RMSNorm(cfg.d_model)
        self.lm_head = nn.Linear(cfg.d_model, cfg.vocab_size, bias=False)
        self._init_weights()
        if cfg.tie_embeddings:
            self.lm_head.weight = self.embed.weight
        # Connections may need custom init after the body is initialised.
        self.connection.reset_parameters()

    def _init_weights(self):
        std = 0.02
        # GPT-2 style: scale output projections by depth.
        out_std = std / math.sqrt(2 * self.cfg.n_layer)
        for name, p in self.named_parameters():
            if name.startswith("connection."):
                continue
            if p.dim() == 2:
                is_out = name.endswith("wo.weight") or name.endswith("w2.weight")
                nn.init.normal_(p, mean=0.0, std=out_std if is_out else std)

    def forward(self, idx, targets=None):
        x0 = self.embed(idx)
        h = self.connection(x0, self.sublayers)
        logits = self.lm_head(self.final_norm(h))
        if targets is None:
            return logits, None
        loss = F.cross_entropy(logits.float().view(-1, logits.size(-1)), targets.reshape(-1))
        return logits, loss

    def num_params(self, exclude_embedding: bool = True):
        n = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            n -= self.embed.weight.numel()
        return n

    def connection_params(self):
        return sum(p.numel() for p in self.connection.parameters())
