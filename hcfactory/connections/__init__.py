"""Registry of residual-connection variants.

Add a new variant by subclassing ``Connection`` and registering it here.
"""
from __future__ import annotations

from .base import Connection
from .dense import DenseFormer, MUDDFormer
from .depth_attention import MHAR, AttnRes, DeltaAttnRes
from .hyper import FracConnection, HyperConnection, ManifoldHyperConnection
from .laurel import LAuReL
from .residual import PostNorm, PreNorm

REGISTRY: dict[str, type[Connection]] = {
    "prenorm": PreNorm,
    "postnorm": PostNorm,
    "hc": HyperConnection,
    "mhc": ManifoldHyperConnection,
    "frac": FracConnection,
    "denseformer": DenseFormer,
    "muddformer": MUDDFormer,
    "laurel": LAuReL,
    "attnres": AttnRes,
    "mhar": MHAR,
    "dar": DeltaAttnRes,
}


def build_connection(cfg) -> Connection:
    if cfg.connection not in REGISTRY:
        raise ValueError(f"unknown connection {cfg.connection!r}; choose from {sorted(REGISTRY)}")
    return REGISTRY[cfg.connection](cfg, **cfg.connection_kwargs)


__all__ = ["Connection", "REGISTRY", "build_connection"]
