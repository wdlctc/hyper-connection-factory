"""Dense cross-layer connections: DenseFormer (static) and MUDDFormer (dynamic)."""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import Connection, rms


def _run_block(h, attn, mlp):
    h = h + attn(h)
    return h + mlp(h)


class DenseFormer(Connection):
    """DenseFormer: Depth-Weighted Average after every block.

        X_0 = embedding,  X_i = Block_i(Y_{i-1}),  Y_i = sum_{j<=i} a_ij X_j

    a_ii = 1 and a_ij = 0 (j<i) at init, so it starts as a plain pre-norm model.
    ``dilation`` k keeps only X_j with (i - j) % k == 0 (DenseFormer's k x p variants;
    period p is fixed to 1 here).
    """

    def __init__(self, cfg, dilation: int = 1):
        super().__init__(cfg)
        self.dilation = dilation
        L = cfg.n_layer
        self.alphas = nn.ParameterList(nn.Parameter(torch.zeros(i + 2)) for i in range(L))

    def reset_parameters(self):
        with torch.no_grad():
            for a in self.alphas:
                a.zero_()
                a[-1] = 1.0

    def forward(self, x0, sublayers):
        xs = [x0]
        y = x0
        for i in range(self.n_layer):
            xs.append(_run_block(y, sublayers[2 * i], sublayers[2 * i + 1]))
            a = self.alphas[i]
            y = 0
            for j, x in enumerate(xs):
                if (i + 1 - j) % self.dilation == 0:
                    y = y + a[j] * x
        return y


class MUDDFormer(Connection):
    """MUDDFormer: multiway dynamic dense connections (Xiao et al., arXiv 2502.12170).

    After block i, a small MLP on RMSNorm(X_i) emits per-token weights over all
    block outputs X_0..X_i, separately for the four input streams of the next
    block: Q, K, V (fed to the next attention) and R (the residual it adds to).

        dw = GELU(RMSNorm(X_i) W1) W2 + a_i          W2 = 0, a_i = one-hot(i)
        X^c = sum_j dw[c, j] X_j                     c in {Q, K, V, R}

    The final aggregation only produces R (C = 1). At init every stream equals
    X_i, i.e. plain pre-norm. PrePostDANorm and the depth-varying FFN width of
    the paper are not included.

    kwargs (ablations): dynamic=False keeps only the static weights a_i;
    qkv=False uses a single R stream (attention reads R, like DenseFormer but
    dynamic).
    """

    qkv_streams = True

    def __init__(self, cfg, dynamic: bool = True, qkv: bool = True):
        super().__init__(cfg)
        self.dynamic, self.qkv = dynamic, qkv
        self.qkv_streams = qkv
        D, L = cfg.d_model, cfg.n_layer
        self.ways = [4 if qkv else 1] * (L - 1) + [1]
        w1, w2, static = [], [], []
        for i, C in enumerate(self.ways):
            n_src = i + 2
            if not dynamic:
                static.append(nn.Parameter(torch.zeros(C, n_src)))
                continue
            K = C * n_src * (4 if C == 1 else 1)
            K = (K // 64 + 1) * 64  # rounding used by the reference implementation
            w1.append(nn.Linear(D, K, bias=False))
            w2.append(nn.Linear(K, C * n_src, bias=False))
            static.append(nn.Parameter(torch.zeros(C, n_src)))
        if dynamic:
            self.w1, self.w2 = nn.ModuleList(w1), nn.ModuleList(w2)
        self.static = nn.ParameterList(static)

    def reset_parameters(self):
        with torch.no_grad():
            for s in self.static:
                s.zero_()
                s[:, -1] = 1.0
            for w in getattr(self, "w1", []):
                nn.init.normal_(w.weight, std=w.in_features ** -0.5)
            for w in getattr(self, "w2", []):
                nn.init.zeros_(w.weight)

    def forward(self, x0, sublayers):
        xs = [x0]
        q = k = v = r = x0
        for i in range(self.n_layer):
            attn, mlp = sublayers[2 * i], sublayers[2 * i + 1]
            h = r + (attn(q, k, v) if self.qkv else attn(r))
            h = h + mlp(h)
            xs.append(h)
            C = self.ways[i]
            w = self.static[i]
            if self.dynamic:
                dyn = self.w2[i](F.gelu(self.w1[i](rms(h))))
                w = dyn.unflatten(-1, (C, len(xs))) + w
            streams = [sum(w[..., c, j : j + 1] * x for j, x in enumerate(xs)) for c in range(C)]
            if i == self.n_layer - 1:
                return streams[0]
            q, k, v, r = streams if self.qkv else streams * 4
