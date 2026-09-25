from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class Connection(nn.Module):
    """Decides how the 2L sublayers are wired together.

    ``forward(x0, sublayers)`` receives the token embedding ``x0`` [B, T, D] and
    the list of sublayers (alternating Attention, MLP; each computes f(norm(x)))
    and must return the final hidden state [B, T, D] (before the final norm).
    """

    #: whether the sublayers should carry their own input RMSNorm
    sublayer_prenorm: bool = True
    #: whether attention sublayers take separate Q/K/V input streams
    qkv_streams: bool = False

    def __init__(self, cfg):
        super().__init__()
        self.d_model = cfg.d_model
        self.n_layer = cfg.n_layer
        self.n_sub = 2 * cfg.n_layer

    def reset_parameters(self):
        """Called by GPT after the transformer body is initialised."""

    def extra_repr(self):
        return ""


def rms(x, eps: float = 1e-6):
    """Parameter-free RMSNorm over the last dim (computed in fp32)."""
    return F.rms_norm(x.float(), (x.shape[-1],), eps=eps).type_as(x)


def sinkhorn(logits: torch.Tensor, iters: int = 20) -> torch.Tensor:
    """Project exp(logits) onto (approximately) doubly-stochastic matrices.

    Works on the last two dims; done in log space for stability.
    """
    log_p = logits.float()
    for _ in range(iters):
        log_p = log_p - torch.logsumexp(log_p, dim=-1, keepdim=True)  # rows
        log_p = log_p - torch.logsumexp(log_p, dim=-2, keepdim=True)  # cols
    return log_p.exp().type_as(logits)
